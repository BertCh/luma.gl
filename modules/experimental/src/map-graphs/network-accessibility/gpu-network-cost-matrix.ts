// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {GPUNetworkReachability} from '../network-reachability';
import {ACCESSIBILITY_NONE} from './network-accessibility-passes';

const OPERATION = 'GPUNetworkCostMatrix';
const DEFAULT_LANE_COUNT = 32;

/**
 * Properties for {@link GPUNetworkCostMatrix}.
 *
 * Compile-time: node, edge, seed and row counts, `laneCount`, `maxIterations`, `localIterations`,
 * and which optional views exist. Per-frame: the contents of the CSR (live weights, closures),
 * seeds, seed costs, seed rows, and `costLimit`.
 */
export type GPUNetworkCostMatrixProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-cost-matrix'`. */
  id?: string;
  /**
   * CSR row offsets with `nodeCount + 1` rows, `offsets[0] = 0` and
   * `offsets[nodeCount] = neighbors.length`. Pass the reverse CSR to get the cost from every node
   * to each row's seeds (opportunity-side accessibility on a directed network); an undirected
   * network that lists both directions uses the same CSR either way.
   */
  offsets: GraphDataView<'uint32'>;
  /** CSR destination node per edge. Indices `>= nodeCount` are ignored. */
  neighbors: GraphDataView<'uint32'>;
  /** Non-negative cost per edge. Negative or NaN edges are impassable. */
  weights: GraphDataView<'float32'>;
  /** Seed node per seed row. Out-of-range nodes are ignored. */
  seedNodes: GraphDataView<'uint32'>;
  /**
   * Optional initial cost per seed, for example the two seed costs of a point snapped onto an
   * edge by `GPUNetworkSnapping`. Negative or NaN costs ignore the seed. Defaults to 0.
   */
  seedCosts?: GraphDataView<'float32'>;
  /**
   * Optional matrix row per seed. Seeds sharing a row form one multi-seed search. Defaults to
   * `floor(seed / seedsPerRow)`. Rows `>= rowCount` are ignored.
   */
  seedRows?: GraphDataView<'uint32'>;
  /** Seeds per row when `seedRows` is omitted. Defaults to 1; use 2 for snapped seeds. */
  seedsPerRow?: number;
  /** Optional one-row per-frame cost cutoff. Matrix entries above it stay `+Infinity`. */
  costLimit?: GraphDataView<'float32'>;
  /**
   * Compile-time rows searched together in one lane-expanded reachability pass. Defaults to
   * `min(rowCount, 32)`. Scratch memory grows with `laneCount * (nodeCount + 2 * edgeCount)` words
   * and the command node count with `ceil(rowCount / laneCount) * (maxIterations + 2)`.
   */
  laneCount?: number;
  /** Rounds of every batch, as `GPUNetworkReachability.maxIterations`. Defaults to 64. */
  maxIterations?: number;
  /** Hops chained per workgroup per round, as `GPUNetworkReachability.localIterations`. */
  localIterations?: number;
  /**
   * Row-major `[rowCount x nodeCount]` minimum cost from row seeds to every node, `+Infinity` when
   * unreached. Its length must be a multiple of the node count and defines `rowCount`.
   */
  costs: GraphDataView<'float32'>;
  /** Optional one-row flag: 1 when every batch reached a fixpoint within `maxIterations`. */
  converged?: GraphDataView<'uint32'>;
};

/**
 * Computes a bounded many-to-all cost matrix: one shortest-path search per matrix row, from that
 * row's seeds to every node of a directed CSR network.
 *
 * Rows are searched `laneCount` at a time. Each batch runs one `GPUNetworkReachability` over a
 * lane-expanded network that holds `laneCount` disjoint copies of the CSR (lane `l`, node `u` is
 * expanded node `l * nodeCount + u`), seeded only in each row's own lane, so the frontier rounds,
 * workgroup-local hop chaining, cost limit and convergence flag of reachability serve every lane
 * at once and the expanded cost array is exactly the batch's block of matrix rows. Costs are the
 * unique f32 fixpoint of the relaxations, so the matrix is bit-identical for every `laneCount`.
 *
 * Run the search from the smaller side: rows are the number of shortest-path searches, columns are
 * free. For accessibility of every node, put opportunities or facilities on rows and search the
 * reverse CSR. Every encoding recomputes the matrix; keep it in a caller-owned buffer and score it
 * with `GPUNetworkAccessibility` in a separate graph so threshold and decay changes never re-run
 * the searches.
 */
export class GPUNetworkCostMatrix implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'network-cost-matrix';
  /** Validated properties. */
  readonly props: GPUNetworkCostMatrixProps;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of matrix rows. */
  readonly rowCount: number;
  /** Resolved rows per batch. */
  readonly laneCount: number;
  /** Number of lane-expanded reachability batches. */
  readonly batchCount: number;
  /** Resolved seeds per row used when `seedRows` is omitted. */
  readonly seedsPerRow: number;

  constructor(props: GPUNetworkCostMatrixProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    for (const [name, view] of [
      ['offsets', props.offsets],
      ['neighbors', props.neighbors],
      ['seedNodes', props.seedNodes],
      ['seedRows', props.seedRows],
      ['converged', props.converged]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['weights', props.weights],
      ['seedCosts', props.seedCosts],
      ['costLimit', props.costLimit],
      ['costs', props.costs]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    this.nodeCount = props.offsets.length - 1;
    if (this.nodeCount < 1) {
      throw new Error(`${id} offsets must describe at least one node`);
    }
    if (props.weights.length !== props.neighbors.length) {
      throw new Error(`${id} weights length must equal neighbors length`);
    }
    if (props.costs.length < this.nodeCount || props.costs.length % this.nodeCount !== 0) {
      throw new Error(`${id} costs length must be a positive multiple of the node count`);
    }
    this.rowCount = props.costs.length / this.nodeCount;
    for (const [name, view] of [
      ['seedCosts', props.seedCosts],
      ['seedRows', props.seedRows]
    ] as const) {
      if (view && view.length !== props.seedNodes.length) {
        throw new Error(`${id} ${name} length must equal seedNodes length`);
      }
    }
    for (const [name, view] of [
      ['costLimit', props.costLimit],
      ['converged', props.converged]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    this.seedsPerRow = props.seedsPerRow ?? 1;
    if (!Number.isSafeInteger(this.seedsPerRow) || this.seedsPerRow < 1) {
      throw new Error(`${id} seedsPerRow must be a positive integer`);
    }
    this.laneCount = props.laneCount ?? Math.min(this.rowCount, DEFAULT_LANE_COUNT);
    if (
      !Number.isSafeInteger(this.laneCount) ||
      this.laneCount < 1 ||
      this.laneCount > this.rowCount
    ) {
      throw new Error(`${id} laneCount must be an integer in [1, rowCount]`);
    }
    const expandedEdgeCount = this.laneCount * props.neighbors.length;
    if (this.laneCount * this.nodeCount >= ACCESSIBILITY_NONE || expandedEdgeCount >= 2 ** 32) {
      throw new Error(`${id} laneCount * nodeCount and laneCount * edgeCount must fit in uint32`);
    }
    this.batchCount = Math.ceil(this.rowCount / this.laneCount);
    validateGraphOutputsDisjointFromInputs(id, [props.costs, props.converged], getInputs(props));
    if (props.converged && props.converged.buffer === props.costs.buffer) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns the lane-expansion nodes (`expand-offsets`, `expand-edges`, `expand-seeds`), the
   * nodes of one `GPUNetworkReachability` per batch (IDs `${id}-batch-${batch}-*`), and an optional
   * `converged` reduction.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, rowCount, laneCount, batchCount, seedsPerRow} = this;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), props.costs, props.converged]);
    const edgeCount = props.neighbors.length;
    const seedCount = props.seedNodes.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const expandedOffsets = createTransientView(
      graph,
      `${id}-expanded-offsets`,
      'uint32',
      laneCount * nodeCount + 1
    );
    const expandedNeighbors = createTransientView(
      graph,
      `${id}-expanded-neighbors`,
      'uint32',
      Math.max(laneCount * edgeCount, 1)
    );
    const expandedWeights = createTransientView(
      graph,
      `${id}-expanded-weights`,
      'float32',
      Math.max(laneCount * edgeCount, 1)
    );
    const expandedSeeds = createTransientView(
      graph,
      `${id}-expanded-seeds`,
      'uint32',
      Math.max(batchCount * seedCount, 1)
    );
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-expand-offsets`,
        operation: OPERATION,
        variant: 'expand-offsets',
        bindings: [
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {
            name: 'expandedOffsets',
            view: expandedOffsets,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: laneCount * nodeCount + 1,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;`,
        body: `let lane = index / NODE_COUNT;
  let node = index - lane * NODE_COUNT;
  expandedOffsets[expandedOffsetsOffset + index] = lane * EDGE_COUNT + offsets[offsetsOffset + node];`
      })
    );
    if (edgeCount > 0) {
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-expand-edges`,
          operation: OPERATION,
          variant: 'expand-edges',
          bindings: [
            {
              name: 'neighbors',
              view: props.neighbors,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'weights',
              view: props.weights,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'expandedNeighbors',
              view: expandedNeighbors,
              type: 'u32',
              access: 'read_write'
            },
            {
              name: 'expandedWeights',
              view: expandedWeights,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: laneCount * edgeCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;`,
          body: `let lane = index / EDGE_COUNT;
  let edge = index - lane * EDGE_COUNT;
  let neighbor = neighbors[neighborsOffset + edge];
  expandedNeighbors[expandedNeighborsOffset + index] =
    select(${ACCESSIBILITY_NONE}u, lane * NODE_COUNT + neighbor, neighbor < NODE_COUNT);
  expandedWeights[expandedWeightsOffset + index] = weights[weightsOffset + edge];`
        })
      );
    }
    if (seedCount > 0) {
      const seedBindings: MapGraphKernelBinding[] = [
        {
          name: 'seedNodes',
          view: props.seedNodes,
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
      if (props.seedRows) {
        seedBindings.push({
          name: 'seedRows',
          view: props.seedRows,
          type: 'u32',
          access: 'read'
        });
      }
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-expand-seeds`,
          operation: OPERATION,
          variant: 'expand-seeds',
          bindings: seedBindings,
          invocationCount: batchCount * seedCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const ROW_COUNT: u32 = ${rowCount}u;
const LANE_COUNT: u32 = ${laneCount}u;
const SEED_COUNT: u32 = ${seedCount}u;
const SEEDS_PER_ROW: u32 = ${seedsPerRow}u;`,
          body: `let batch = index / SEED_COUNT;
  let seed = index - batch * SEED_COUNT;
  let row = ${props.seedRows ? 'seedRows[seedRowsOffset + seed]' : 'seed / SEEDS_PER_ROW'};
  let node = seedNodes[seedNodesOffset + seed];
  let firstRow = batch * LANE_COUNT;
  let isInBatch = row < ROW_COUNT && row >= firstRow && row - firstRow < LANE_COUNT;
  expandedSeeds[expandedSeedsOffset + index] =
    select(${ACCESSIBILITY_NONE}u, (row - firstRow) * NODE_COUNT + node, isInBatch && node < NODE_COUNT);`
        })
      );
    }

    const batchConverged = props.converged
      ? createTransientView(graph, `${id}-batch-converged`, 'uint32', batchCount)
      : undefined;
    for (let batch = 0; batch < batchCount; batch++) {
      const batchLaneCount = Math.min(laneCount, rowCount - batch * laneCount);
      const batchNodeCount = batchLaneCount * nodeCount;
      const batchEdgeCount = batchLaneCount * edgeCount;
      nodes.push(
        ...new GPUNetworkReachability({
          id: `${id}-batch-${batch}`,
          offsets: getSubView(graph, expandedOffsets, 0, batchNodeCount + 1),
          neighbors: getSubView(graph, expandedNeighbors, 0, batchEdgeCount),
          weights: getSubView(graph, expandedWeights, 0, batchEdgeCount),
          sources: getSubView(graph, expandedSeeds, batch * seedCount, seedCount),
          sourceCosts: props.seedCosts,
          costLimit: props.costLimit,
          maxIterations: props.maxIterations,
          localIterations: props.localIterations,
          costs: getSubView(graph, props.costs, batch * laneCount * nodeCount, batchNodeCount),
          converged: batchConverged ? getSubView(graph, batchConverged, batch, 1) : undefined
        }).getCommandNodes(graph)
      );
    }
    if (props.converged && batchConverged) {
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
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
              view: props.converged,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          declarations: `const BATCH_COUNT: u32 = ${batchCount}u;`,
          body: `var allConverged = 1u;
  for (var batch = 0u; batch < BATCH_COUNT; batch++) {
    allConverged = min(allConverged, batchConverged[batchConvergedOffset + batch]);
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

/** Returns every read-only view of a cost matrix. */
function getInputs(props: GPUNetworkCostMatrixProps): (GraphDataView | undefined)[] {
  return [
    props.offsets,
    props.neighbors,
    props.weights,
    props.seedNodes,
    props.seedCosts,
    props.seedRows,
    props.costLimit
  ];
}
