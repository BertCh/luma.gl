// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {GPUNetworkReachability} from '../network-reachability/index';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';
import {ACCESSIBILITY_NONE} from './network-accessibility-passes';
import {recommendLaneCount} from './network-accessibility-lanes';

const OPERATION = 'GPUNetworkCostMatrix';
const DEFAULT_LOCAL_ITERATIONS = 16;
/**
 * Elements per 256-byte storage binding alignment. Batch sub-views start at multiples of it, so
 * the element offset `createWGSLKernelNode` bakes into each kernel is the same for every batch and
 * all batches share pipelines (library rule 12).
 */
const ALIGNMENT_ELEMENTS = 64;

/** GPU-written views of {@link GPUNetworkCostMatrix}. */
export type GPUNetworkCostMatrixOutput = {
  /**
   * Row-major `[rowCount x columnCount]` minimum cost from row starts to every node (or, with
   * `targetNodes`, to every target), `+Infinity` when unreached. Its length must be a multiple of
   * the column count (the node count, or the target count with `targetNodes`) and defines
   * `rowCount`.
   */
  costs: GraphDataView<'float32'>;
  /** Optional GPU-written status words. */
  status?: {
    /** Optional one-row flag: 1 when every batch reached a fixpoint within `maximumIterations`. */
    converged?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkCostMatrix}.
 *
 * Compile-time: node, edge, start, target and row counts, `laneCount`, `maximumIterations`,
 * `localIterations`, and which optional views exist. Per-frame: the contents of the topology (live
 * weights, closures), start nodes, start costs, start rows, target nodes, and `costLimit`.
 */
export type GPUNetworkCostMatrixProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-cost-matrix'`. */
  id?: string;
  /**
   * The network. Pass the reverse topology to get the cost from every node to each row's start
   * nodes (opportunity-side accessibility on a directed network); an undirected network that lists
   * both directions uses the same topology either way. Targets `>= nodeCount` are ignored and
   * negative or NaN weights are impassable.
   */
  topology: GPUNetworkTopology;
  /** Start node per start. Out-of-range nodes are ignored. */
  startNodes: GraphDataView<'uint32'>;
  /**
   * Optional initial cost per start, for example the two start costs of a point snapped onto an
   * edge by `GPUNetworkSnapping`. Negative or NaN costs ignore the start. Defaults to 0.
   */
  startCosts?: GraphDataView<'float32'>;
  /**
   * Optional matrix row per start. Starts sharing a row form one multi-start search. Defaults to
   * `floor(start / startsPerRow)`. Rows `>= rowCount` are ignored.
   */
  startRows?: GraphDataView<'uint32'>;
  /** Starts per row when `startRows` is omitted. Defaults to 1; use 2 for snapped starts. */
  startsPerRow?: number;
  /** Optional one-row per-frame cost cutoff. Matrix entries above it stay `+Infinity`. */
  costLimit?: GraphDataView<'float32'>;
  /**
   * Optional target nodes. With targets, `output.costs` is the compact row-major
   * `[rowCount x targetCount]` matrix: column `t` holds the cost to `targetNodes[t]`, or
   * `+Infinity` when that node is unreached or `>= nodeCount`. Duplicate targets repeat their
   * column. The values are bit-identical to the matching columns of the dense matrix, but the
   * output no longer grows with the node count, so accessibility, K-function and OD workloads with
   * few targets avoid the dense `rowCount x nodeCount` allocation. The length (`targetCount`) is
   * compile-time and at least 1; the node IDs are per-frame.
   */
  targetNodes?: GraphDataView<'uint32'>;
  /**
   * Compile-time rows searched together in one lane-expanded reachability pass. Defaults to
   * `recommendLaneCount()`: about 1M expanded nodes per batch (at least 32 lanes), at most
   * `rowCount`, rounded down so that `laneCount * nodeCount` is a multiple of 64 when there are
   * several batches. Lanes share one CSR, so scratch memory is only the frontier state
   * (about `3 * laneCount * nodeCount` words) and the command node count grows with
   * `ceil(rowCount / laneCount) * (maximumIterations + 2)`.
   *
   * Every batch runs at the full lane count (a short last batch masks its unused lanes), so all
   * batches compile to the same pipelines. Full batches write the matrix in place when there is one
   * batch, or when `laneCount * nodeCount` is a multiple of 64 and `costs` starts on a 256-byte
   * boundary; the other batches search into one `laneCount * nodeCount` scratch block and copy it.
   *
   * With `targetNodes` every batch searches into that one `laneCount * nodeCount` scratch block
   * and a gather kernel writes the batch's `targetCount` columns, so memory is the scratch block
   * plus the `rowCount * targetCount` output; lower `laneCount` to shrink the scratch block.
   */
  laneCount?: number;
  /** Rounds of every batch, as `GPUNetworkReachability.maximumIterations`. Defaults to 64. */
  maximumIterations?: number;
  /** Hops chained per workgroup per round, as `GPUNetworkReachability.localIterations`. Defaults to 16, which suits lane-expanded searches. */
  localIterations?: number;
  /** The GPU-written views. */
  output: GPUNetworkCostMatrixOutput;
};

/**
 * Computes a bounded many-to-all cost matrix: one shortest-path search per matrix row, from that
 * row's starts to every node of a directed CSR network.
 *
 * Rows are searched `laneCount` at a time. Each batch runs one `GPUNetworkReachability` over a
 * lane-expanded network of `laneCount` disjoint lanes that all read the same CSR (lane `l`, node `u`
 * is expanded node `l * nodeCount + u`; no per-lane copy is materialized), started only in each row's own lane, so the frontier rounds,
 * workgroup-local hop chaining, cost limit and convergence flag of reachability serve every lane
 * at once and the expanded cost array is exactly the batch's block of matrix rows. Costs are the
 * unique f32 fixpoint of the relaxations, so the matrix is bit-identical for every `laneCount`.
 *
 * Run the search from the smaller side: rows are the number of shortest-path searches, columns are
 * free. For accessibility of every node, put opportunities or facilities on rows and search the
 * reverse CSR. Every encoding recomputes the matrix; keep it in a caller-owned buffer and score it
 * with `GPUNetworkAccessibility` in a separate graph so threshold and decay changes never re-run
 * the searches.
 *
 * With `targetNodes` the matrix is compact, `[rowCount x targetCount]`: each batch is searched
 * into a transient `laneCount * nodeCount` scratch block and only the target columns are gathered
 * into `output.costs`, so the dense `rowCount x nodeCount` allocation disappears.
 */
export class GPUNetworkCostMatrix implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkCostMatrixProps;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of matrix rows. */
  readonly rowCount: number;
  /** Number of matrix columns: the node count, or the target count with `targetNodes`. */
  readonly columnCount: number;
  /** Resolved rows per batch. */
  readonly laneCount: number;
  /** Number of lane-expanded reachability batches. */
  readonly batchCount: number;
  /** Resolved starts per row used when `startRows` is omitted. */
  readonly startsPerRow: number;

  constructor(props: GPUNetworkCostMatrixProps) {
    this.id = props.id ?? 'network-cost-matrix';
    this.props = props;
    const {id} = this;
    const {costs} = props.output;
    const converged = props.output.status?.converged;
    const {nodeCount, edgeCount} = validateGPUNetworkTopology(id, props.topology, {weights: true});
    this.nodeCount = nodeCount;
    for (const [name, view] of [
      ['startNodes', props.startNodes],
      ['startRows', props.startRows],
      ['targetNodes', props.targetNodes],
      ['output.status.converged', converged]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['startCosts', props.startCosts],
      ['costLimit', props.costLimit],
      ['output.costs', costs]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (props.targetNodes && props.targetNodes.length < 1) {
      throw new Error(`${id} targetNodes must contain at least one row`);
    }
    this.columnCount = props.targetNodes?.length ?? this.nodeCount;
    if (costs.length < this.columnCount || costs.length % this.columnCount !== 0) {
      throw new Error(
        `${id} output.costs length must be a positive multiple of the ${
          props.targetNodes ? 'target' : 'node'
        } count`
      );
    }
    this.rowCount = costs.length / this.columnCount;
    for (const [name, view] of [
      ['startCosts', props.startCosts],
      ['startRows', props.startRows]
    ] as const) {
      if (view && view.length !== props.startNodes.length) {
        throw new Error(`${id} ${name} length must equal startNodes length`);
      }
    }
    for (const [name, view] of [
      ['costLimit', props.costLimit],
      ['output.status.converged', converged]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    this.startsPerRow = props.startsPerRow ?? 1;
    validateCapacity(id, 'startsPerRow', this.startsPerRow);
    this.laneCount =
      props.laneCount ??
      alignLaneCount(
        recommendLaneCount({
          rowCount: this.rowCount,
          nodeCount: this.nodeCount,
          edgeCount
        }),
        this.nodeCount,
        this.rowCount
      );
    validateCapacity(id, 'laneCount', this.laneCount, 1, this.rowCount);
    if (this.laneCount * this.nodeCount >= ACCESSIBILITY_NONE) {
      throw new Error(`${id} laneCount * nodeCount must fit in uint32`);
    }
    this.batchCount = Math.ceil(this.rowCount / this.laneCount);
    validateGraphOutputsDisjointFromInputs(id, [costs, converged], getInputs(props));
    if (converged && converged.buffer === costs.buffer) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns the seed expansion node (`expand-seeds`), the
   * nodes of one `GPUNetworkReachability` per batch (IDs `${id}-batch-${batch}-*`), and an optional
   * `converged` reduction.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, rowCount, laneCount, batchCount, startsPerRow} = this;
    const {targetNodes} = props;
    const {offsets, neighbors, weights} = props.topology;
    const {costs} = props.output;
    const converged = props.output.status?.converged;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), costs, converged]);
    const seedCount = props.startNodes.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Batch sub-views start at multiples of ALIGNMENT_ELEMENTS (rule 12): the seeds of batch b
    // at b * seedStride, its converged flag at b * ALIGNMENT_ELEMENTS.
    const seedStride = alignElementCount(seedCount);
    const batchNodeCount = laneCount * nodeCount;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'frontier queues',
      2 * batchNodeCount * 4,
      'lower laneCount'
    );
    const expandedSeeds = createTransientView(
      graph,
      `${id}-expanded-seeds`,
      'uint32',
      Math.max(batchCount * seedStride, 1)
    );
    if (seedCount > 0) {
      const seedBindings: WGSLKernelBinding[] = [
        {
          name: 'seedNodes',
          view: props.startNodes,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'expandedSeeds',
          view: expandedSeeds,
          type: 'u32',
          access: 'read_write'
        }
      ];
      if (props.startRows) {
        seedBindings.push({
          name: 'seedRows',
          view: props.startRows,
          type: 'u32',
          access: 'read'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-expand-seeds`,
          operation: OPERATION,
          variant: 'expand-seeds',
          bindings: seedBindings,
          invocationCount: batchCount * seedCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const ROW_COUNT: u32 = ${rowCount}u;
const LANE_COUNT: u32 = ${laneCount}u;
const SEED_COUNT: u32 = ${seedCount}u;
const SEED_STRIDE: u32 = ${seedStride}u;
const SEEDS_PER_ROW: u32 = ${startsPerRow}u;`,
          body: `let batch = index / SEED_COUNT;
  let seed = index - batch * SEED_COUNT;
  let row = ${props.startRows ? 'seedRows[seedRowsOffset + seed]' : 'seed / SEEDS_PER_ROW'};
  let node = seedNodes[seedNodesOffset + seed];
  let firstRow = batch * LANE_COUNT;
  let isInBatch = row < ROW_COUNT && row >= firstRow && row - firstRow < LANE_COUNT;
  expandedSeeds[expandedSeedsOffset + batch * SEED_STRIDE + seed] =
    select(${ACCESSIBILITY_NONE}u, (row - firstRow) * NODE_COUNT + node, isInBatch && node < NODE_COUNT);`
        })
      );
    }

    const batchConverged = converged
      ? createTransientView(
          graph,
          `${id}-batch-converged`,
          'uint32',
          batchCount * ALIGNMENT_ELEMENTS
        )
      : undefined;
    // A full batch writes its block of `costs` in place when its sub-view offset residue is the
    // same for every batch. Other batches (a short last batch, or every batch of an unaligned
    // layout) search into one scratch block and a copy kernel moves the live rows into `costs`.
    // The copy binds a 256-byte aligned window of `costs` and reads its shift and length from a
    // per-batch info slot, so one copy pipeline serves every batch.
    const costsElementOffset = costs.byteOffset / 4;
    const isAligned =
      batchNodeCount % ALIGNMENT_ELEMENTS === 0 && costsElementOffset % ALIGNMENT_ELEMENTS === 0;
    const isInPlace = (batch: number) =>
      !targetNodes && (batch + 1) * laneCount <= rowCount && (batchCount === 1 || isAligned);
    const copiedBatches = [...Array(batchCount).keys()].filter(batch => !isInPlace(batch));
    const scratchCosts =
      copiedBatches.length > 0
        ? createTransientView(graph, `${id}-scratch-costs`, 'float32', batchNodeCount)
        : undefined;
    const batchInfo = scratchCosts
      ? createTransientView(graph, `${id}-batch-info`, 'uint32', batchCount * ALIGNMENT_ELEMENTS)
      : undefined;
    if (batchInfo && targetNodes) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'output.costs',
        costs.length * 4,
        'lower rowCount or targetNodes length'
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-info`,
          operation: OPERATION,
          variant: 'batch-info-targets',
          bindings: [{name: 'batchInfo', view: batchInfo, type: 'u32', access: 'read_write'}],
          invocationCount: batchCount,
          declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const LANE_COUNT: u32 = ${laneCount}u;
const SLOT_WORDS: u32 = ${ALIGNMENT_ELEMENTS}u;`,
          // Word 0: the batch's first matrix row. Word 1: number of live rows in the batch.
          body: `let firstRow = index * LANE_COUNT;
  batchInfo[batchInfoOffset + index * SLOT_WORDS] = firstRow;
  batchInfo[batchInfoOffset + index * SLOT_WORDS + 1u] = min(LANE_COUNT, ROW_COUNT - firstRow);`
        })
      );
    } else if (batchInfo) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-info`,
          operation: OPERATION,
          variant: 'batch-info',
          bindings: [{name: 'batchInfo', view: batchInfo, type: 'u32', access: 'read_write'}],
          invocationCount: batchCount,
          declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const LANE_COUNT: u32 = ${laneCount}u;
const NODE_COUNT: u32 = ${nodeCount}u;
const COSTS_ELEMENT_RESIDUE: u32 = ${costsElementOffset % ALIGNMENT_ELEMENTS}u;
const SLOT_WORDS: u32 = ${ALIGNMENT_ELEMENTS}u;`,
          // Word 0: position of the batch's first matrix element inside its aligned window.
          // Word 1: number of matrix elements the batch owns.
          body: `let firstRow = index * LANE_COUNT;
  let firstElement = COSTS_ELEMENT_RESIDUE + firstRow * NODE_COUNT;
  batchInfo[batchInfoOffset + index * SLOT_WORDS] = firstElement % SLOT_WORDS;
  batchInfo[batchInfoOffset + index * SLOT_WORDS + 1u] = min(LANE_COUNT, ROW_COUNT - firstRow) * NODE_COUNT;`
        })
      );
    }
    for (let batch = 0; batch < batchCount; batch++) {
      const inPlace = isInPlace(batch);
      const batchCosts = inPlace
        ? getSubView(graph, costs, batch * batchNodeCount, batchNodeCount)
        : scratchCosts!;
      nodes.push(
        ...new GPUNetworkReachability({
          id: `${id}-batch-${batch}`,
          // Lanes share the one CSR; reachability maps expanded node `l * nodeCount + u` onto it.
          topology: {offsets, neighbors, weights},
          laneCount,
          startNodes: getSubView(graph, expandedSeeds, batch * seedStride, seedCount),
          startCosts: props.startCosts,
          costLimit: props.costLimit,
          maximumIterations: props.maximumIterations,
          localIterations: props.localIterations ?? DEFAULT_LOCAL_ITERATIONS,
          output: {
            costs: batchCosts,
            status: {
              converged: batchConverged
                ? getSubView(graph, batchConverged, batch * ALIGNMENT_ELEMENTS, 1)
                : undefined
            }
          }
        }).getCommandNodes(graph)
      );
      if (targetNodes && scratchCosts && batchInfo) {
        // One gather pipeline serves every batch: the first row and live row count come from the
        // batch's info slot, and the binding is the whole `costs` view (rule 12).
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-batch-${batch}-gather`,
            operation: OPERATION,
            variant: 'gather',
            bindings: [
              {
                name: 'batchInfo',
                view: getSubView(graph, batchInfo, batch * ALIGNMENT_ELEMENTS, 2),
                type: 'u32',
                access: 'read'
              },
              {name: 'targetNodes', view: targetNodes, type: 'u32', access: 'read'},
              {name: 'scratchCosts', view: scratchCosts, type: 'f32', access: 'read'},
              {name: 'costs', view: costs, type: 'f32', access: 'read_write'}
            ],
            invocationCount: laneCount * targetNodes.length,
            declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const TARGET_COUNT: u32 = ${targetNodes.length}u;`,
            // The cost travels as bits: Tint rejects a const-expression bitcast to infinity, so
            // the bitcast applies only to the runtime var.
            body: `let lane = index / TARGET_COUNT;
  if (lane >= batchInfo[batchInfoOffset + 1u]) { return; }
  let column = index - lane * TARGET_COUNT;
  let node = targetNodes[targetNodesOffset + column];
  var costBits = 0x7f800000u;
  if (node < NODE_COUNT) {
    costBits = bitcast<u32>(scratchCosts[scratchCostsOffset + lane * NODE_COUNT + node]);
  }
  costs[costsOffset + (batchInfo[batchInfoOffset] + lane) * TARGET_COUNT + column] = bitcast<f32>(costBits);`
          })
        );
      } else if (!inPlace && scratchCosts && batchInfo) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-batch-${batch}-copy`,
            operation: OPERATION,
            variant: 'copy',
            bindings: [
              {
                name: 'batchInfo',
                view: getSubView(graph, batchInfo, batch * ALIGNMENT_ELEMENTS, 2),
                type: 'u32',
                access: 'read'
              },
              {name: 'scratchCosts', view: scratchCosts, type: 'f32', access: 'read'},
              {
                name: 'costs',
                view: getAlignedWindow(graph, costs, batch * batchNodeCount, batchNodeCount),
                type: 'f32',
                access: 'read_write'
              }
            ],
            invocationCount: batchNodeCount,
            body: `if (index >= batchInfo[batchInfoOffset + 1u]) { return; }
  costs[costsOffset + batchInfo[batchInfoOffset] + index] = scratchCosts[scratchCostsOffset + index];`
          })
        );
      }
    }
    if (converged && batchConverged) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-converged`,
          operation: OPERATION,
          variant: 'converged',
          bindings: [
            {
              name: 'batchConverged',
              view: batchConverged,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'converged',
              view: converged,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          declarations: `const BATCH_COUNT: u32 = ${batchCount}u;
const SLOT_WORDS: u32 = ${ALIGNMENT_ELEMENTS}u;`,
          body: `var allConverged = 1u;
  for (var batch = 0u; batch < BATCH_COUNT; batch++) {
    allConverged = min(allConverged, batchConverged[batchConvergedOffset + batch * SLOT_WORDS]);
  }
  converged[convergedOffset] = allConverged;`
        })
      );
    }
    return nodes;
  }
}

/** Returns a packed range of `length` rows starting at row `firstRow` of a packed view. */
function getSubView<Format extends 'uint32' | 'float32', Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<Format>,
  firstRow: number,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(view.buffer, {
    format: view.format as Format,
    length,
    byteOffset: view.byteOffset + firstRow * view.byteStride
  });
}

/**
 * Returns the window of `view` that starts at row `firstRow` rounded down to a 256-byte boundary
 * and covers `length` rows from `firstRow`, clamped to the end of `view`. Its element offset is
 * always 0; the rounding shift is `(view element offset + firstRow) % 64`.
 */
function getAlignedWindow<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<'float32'>,
  firstRow: number,
  length: number
): GraphDataView<'float32'> {
  const firstByte = view.byteOffset + firstRow * 4;
  const windowByteOffset = firstByte - (firstByte % (4 * ALIGNMENT_ELEMENTS));
  const viewEnd = view.byteOffset + view.length * 4;
  const windowEnd = Math.min(firstByte + length * 4, viewEnd);
  return graph.createDataView(view.buffer, {
    format: 'float32',
    length: (windowEnd - windowByteOffset) / 4,
    byteOffset: windowByteOffset
  });
}

/** Rounds an element count up to a multiple of {@link ALIGNMENT_ELEMENTS}. */
function alignElementCount(count: number): number {
  return Math.ceil(count / ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS;
}

/**
 * Rounds a recommended lane count down so `laneCount * nodeCount` is a multiple of
 * {@link ALIGNMENT_ELEMENTS}, which lets every batch write its matrix block in place. A single
 * batch, or a lane count below the rounding unit, is returned unchanged.
 */
function alignLaneCount(laneCount: number, nodeCount: number, rowCount: number): number {
  if (laneCount >= rowCount) {
    return laneCount;
  }
  const unit = ALIGNMENT_ELEMENTS / getGreatestCommonDivisor(nodeCount, ALIGNMENT_ELEMENTS);
  return laneCount >= unit ? Math.floor(laneCount / unit) * unit : laneCount;
}

/** Greatest common divisor of two positive integers. */
function getGreatestCommonDivisor(left: number, right: number): number {
  let a = left;
  let b = right;
  while (b !== 0) {
    [a, b] = [b, a % b];
  }
  return a;
}

/** Returns every read-only view of a cost matrix. */
function getInputs(props: GPUNetworkCostMatrixProps): (GraphDataView | undefined)[] {
  return [
    ...getGPUNetworkTopologyViews(props.topology),
    props.startNodes,
    props.startCosts,
    props.startRows,
    props.targetNodes,
    props.costLimit
  ];
}
