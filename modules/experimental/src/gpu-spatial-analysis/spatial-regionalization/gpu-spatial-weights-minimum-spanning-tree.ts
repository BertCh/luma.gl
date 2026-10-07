// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createPublishNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {REGIONALIZATION_CSR_WGSL, REGIONALIZATION_NONE_WGSL} from './regionalization-wgsl';

const OPERATION = 'GPUSpatialWeightsMinimumSpanningTree';

/** Largest supported attribute column count. */
export const GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS = 32;

/** Lanes of the per-column standardisation workgroup. */
const STANDARDIZE_LANES = 256;

/**
 * Properties for {@link GPUSpatialWeightsMinimumSpanningTree}.
 *
 * Compile-time: row count, slot capacity, `columnCount`, `standardize` and which outputs exist.
 * Per-frame: the contents of `values` and `weights` (the whole forest is rebuilt on every
 * encoding, so a changed attribute selection is a buffer write).
 */
export type GPUSpatialWeightsMinimumSpanningTreeProps = {
  /** Prefix for generated node IDs. Defaults to `'minimum-spanning-tree'`. */
  id?: string;
  /**
   * Symmetric contiguity or neighbor weights. Every listed slot is a link; the weight value is
   * ignored (cost is attribute dissimilarity). Every link must be listed in both rows, as
   * `GPUContiguityWeights` and `GPUSpatialWeightsTransform` guarantee.
   */
  weights: GPUSpatialWeights;
  /** Attribute table, row-major with `columnCount` columns: `rows * columnCount` entries. */
  values: GraphDataView<'float32'>;
  /** Number of attribute columns. Defaults to 1; at most 32. */
  columnCount?: number;
  /**
   * When true (the default), each column is standardised to mean 0 and population standard
   * deviation 1 before costs are taken (a constant column becomes all zero). Set false when
   * `values` is already standardised or when exact integer costs matter.
   */
  standardize?: boolean;
  /**
   * Caller-owned flags, one per weights slot (`weights.neighbors.length` entries). Slot `s` is
   * `1` when it is the lower-row slot of an MST edge, otherwise `0`. The slot index is the
   * stable edge ID used by every other output.
   */
  treeEdgeFlags: GraphDataView<'uint32'>;
  /**
   * Caller-owned forest component label per row (`rows` entries): the smallest row index of the
   * row's tree. Isolated rows label themselves.
   */
  componentLabels: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned copy of the standardised attribute table (`rows * columnCount`
   * entries), so {@link GPUSkaterRegions} cuts under the same metric. Requires `standardize`.
   */
  standardizedValues?: GraphDataView<'float32'>;
  /**
   * Optional bounded compact edge list: `ids` hold the edge slots in ascending order.
   * `ids.length` should be at least `rows - 1`; smaller capacities clamp and set `overflow`.
   */
  edges?: GPUCompactOutput;
  /** Optional `[row, neighbor]` pair per compact edge (`2 * edges.ids.length` entries). */
  edgeEndpoints?: GraphDataView<'uint32'>;
  /** Optional squared dissimilarity per compact edge (`edges.ids.length` entries). */
  edgeCosts?: GraphDataView<'float32'>;
};

/**
 * Minimum spanning forest over a {@link GPUSpatialWeights} graph, with edge cost the squared
 * Euclidean distance between the attribute rows of the two neighbors (the metric of SKATER).
 *
 * The solver is Borůvka: each round, every tree selects its cheapest outgoing link, the
 * selections merge trees, and labels are flattened by pointer jumping, for
 * `ceil(log2(rows))` unrolled rounds. A disconnected graph yields a spanning forest with one
 * tree per connected component.
 *
 * **Exactness and ties.** Costs are totally ordered by `(cost, edge ID)` where the edge ID is
 * the lower-row CSR slot, so the minimum spanning forest is unique and equals Kruskal's under
 * the same order. Selection uses integer atomic minima over the cost bits (non-negative f32 bits
 * order like the values), so results are deterministic. Other libraries break cost ties by their
 * own rules (scipy by traversal order), so on data with exact cost ties the edge sets can differ
 * while the total cost is identical.
 *
 * Non-goals: asymmetric weights, and edge costs other than squared Euclidean distance.
 */
export class GPUSpatialWeightsMinimumSpanningTree implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsMinimumSpanningTreeProps;

  constructor(props: GPUSpatialWeightsMinimumSpanningTreeProps) {
    const id = props.id ?? 'minimum-spanning-tree';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    const columnCount = props.columnCount ?? 1;
    if (
      !Number.isSafeInteger(columnCount) ||
      columnCount < 1 ||
      columnCount > GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS
    ) {
      throw new Error(
        `${id} columnCount must be an integer in [1, ${GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS}]`
      );
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length !== rows * columnCount) {
      throw new Error(`${id} values length must equal rows * columnCount`);
    }
    validatePackedUint32View(props.treeEdgeFlags, `${id} treeEdgeFlags`);
    if (props.treeEdgeFlags.length !== props.weights.neighbors.length) {
      throw new Error(`${id} treeEdgeFlags length must equal the weights slot capacity`);
    }
    validatePackedUint32View(props.componentLabels, `${id} componentLabels`);
    if (props.componentLabels.length !== rows) {
      throw new Error(`${id} componentLabels length must equal the weights row count`);
    }
    if (props.standardizedValues) {
      validatePackedView(props.standardizedValues, ['float32'], `${id} standardizedValues`);
      if (props.standardizedValues.length !== rows * columnCount) {
        throw new Error(`${id} standardizedValues length must equal rows * columnCount`);
      }
      if (props.standardize === false) {
        throw new Error(`${id} standardizedValues requires standardize`);
      }
    }
    if (props.edges) {
      validateCompactOutput(id, props.edges);
    } else if (props.edgeEndpoints || props.edgeCosts) {
      throw new Error(`${id} edgeEndpoints and edgeCosts require edges`);
    }
    const capacity = props.edges?.ids.length ?? 0;
    if (props.edgeEndpoints) {
      validatePackedUint32View(props.edgeEndpoints, `${id} edgeEndpoints`);
      if (props.edgeEndpoints.length < 2 * capacity) {
        throw new Error(`${id} edgeEndpoints must hold two entries per edge slot`);
      }
    }
    if (props.edgeCosts) {
      validatePackedView(props.edgeCosts, ['float32'], `${id} edgeCosts`);
      if (props.edgeCosts.length < capacity) {
        throw new Error(`${id} edgeCosts must hold one entry per edge slot`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.treeEdgeFlags,
        props.componentLabels,
        props.standardizedValues,
        props.edges?.ids,
        props.edges?.count,
        props.edges?.overflow,
        props.edgeEndpoints,
        props.edgeCosts
      ],
      [props.values, props.weights.offsets, props.weights.neighbors]
    );
  }

  /** Returns the standardisation, Borůvka round, label and edge-list nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, treeEdgeFlags, componentLabels} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      treeEdgeFlags,
      componentLabels,
      props.standardizedValues,
      props.edges?.ids,
      props.edges?.count,
      props.edges?.overflow,
      props.edges?.totalCount,
      props.edgeEndpoints,
      props.edgeCosts,
      weights.offsets,
      weights.neighbors
    ]);
    const rows = weights.offsets.length - 1;
    const capacity = weights.neighbors.length;
    const columnCount = props.columnCount ?? 1;
    const standardize = props.standardize !== false;
    const csrBindings = [
      {name: 'offsets', view: weights.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'neighbors', view: weights.neighbors, type: 'u32' as const, access: 'read' as const}
    ];
    const rowDeclarations = `const ROWS: u32 = ${rows}u;
const COLUMNS: u32 = ${columnCount}u;
${REGIONALIZATION_NONE_WGSL}
${REGIONALIZATION_CSR_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Attribute table the costs read.
    let table = props.values;
    if (standardize) {
      table =
        props.standardizedValues ??
        createTransientView(graph, `${id}-table`, 'float32', rows * columnCount);
      // One workgroup per column: each lane sums a strided share of the rows and a fixed binary
      // tree merges the lanes, so the reductions are `O(rows / 256 + log 256)` deep instead of one
      // thread walking every row three times, and the sum order is still deterministic.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-standardize`,
          operation: OPERATION,
          variant: 'standardize',
          bindings: [
            {name: 'values', view: props.values, type: 'f32', access: 'read'},
            {name: 'scaled', view: table, type: 'f32', access: 'read_write'}
          ],
          workgroupSize: STANDARDIZE_LANES,
          invocationCount: columnCount * STANDARDIZE_LANES,
          guardIndex: false,
          declarations: `const ROWS: u32 = ${rows}u;
const COLUMNS: u32 = ${columnCount}u;
const LANES: u32 = ${STANDARDIZE_LANES}u;
var<workgroup> partials: array<f32, ${STANDARDIZE_LANES}>;

// Sum of one value per lane over the workgroup, available to every lane.
fn reduceLanes(lane: u32, value: f32) -> f32 {
  partials[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partials[lane] = partials[lane] + partials[lane + stride];
    }
    workgroupBarrier();
  }
  let total = partials[0];
  workgroupBarrier();
  return total;
}`,
          body: `let column = index / LANES;
  let lane = localInvocationIndex;
  var sum = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    sum += values[valuesOffset + row * COLUMNS + column];
  }
  let mean = reduceLanes(lane, sum) / f32(ROWS);
  var squares = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    let delta = values[valuesOffset + row * COLUMNS + column] - mean;
    squares += delta * delta;
  }
  let deviation = sqrt(reduceLanes(lane, squares) / f32(ROWS));
  let inverse = select(0.0, 1.0 / deviation, deviation > 1e-20);
  for (var row = lane; row < ROWS; row += LANES) {
    scaled[scaledOffset + row * COLUMNS + column] =
      (values[valuesOffset + row * COLUMNS + column] - mean) * inverse;
  }`
        })
      );
    }

    const slotCost = createTransientView(graph, `${id}-slot-cost`, 'float32', capacity);
    const slotEdge = createTransientView(graph, `${id}-slot-edge`, 'uint32', capacity);
    const component = createTransientView(graph, `${id}-component`, 'uint32', rows);
    const bestCost = createTransientView(graph, `${id}-best-cost`, 'uint32', rows);
    const bestEdge = createTransientView(graph, `${id}-best-edge`, 'uint32', rows);
    const parentA = createTransientView(graph, `${id}-parent-a`, 'uint32', rows);
    const parentB = createTransientView(graph, `${id}-parent-b`, 'uint32', rows);
    const minimumLabel = createTransientView(graph, `${id}-minimum-label`, 'uint32', rows);

    nodes.push(
      // Zero the flag capacity so unused tail slots are well defined, and seed one tree per row.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-flags`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [{name: 'flags', view: treeEdgeFlags, type: 'u32', access: 'read_write'}],
        invocationCount: capacity,
        body: 'flags[flagsOffset + index] = 0u;'
      }),
      // Seeds one tree per row and clears round 0's selections; later rounds are cleared by the
      // previous round's relabel kernel, so no round needs a separate reset pass.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-seed-components`,
        operation: OPERATION,
        variant: 'seed',
        bindings: [
          {name: 'component', view: component, type: 'u32', access: 'read_write'},
          {name: 'bestCost', view: bestCost, type: 'u32', access: 'read_write'},
          {name: 'bestEdge', view: bestEdge, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: `component[componentOffset + index] = index;
  bestCost[bestCostOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`
      }),
      // Per-slot cost (squared distance of the standardised rows) and canonical edge ID.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slot-costs`,
        operation: OPERATION,
        variant: 'costs',
        bindings: [
          ...csrBindings,
          {name: 'table', view: table, type: 'f32', access: 'read'},
          {name: 'slotCost', view: slotCost, type: 'f32', access: 'read_write'},
          {name: 'slotEdge', view: slotEdge, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: rowDeclarations,
        body: `for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let low = min(index, neighbor);
    let high = max(index, neighbor);
    var cost = 0.0;
    for (var column = 0u; column < COLUMNS; column++) {
      let delta = table[tableOffset + low * COLUMNS + column] - table[tableOffset + high * COLUMNS + column];
      cost += delta * delta;
    }
    slotCost[slotCostOffset + slot] = cost;
    var edge = slot;
    if (neighbor < index) {
      let twin = findSlot(neighbor, index);
      if (twin != NONE) {
        edge = twin;
      }
    }
    slotEdge[slotEdgeOffset + slot] = edge;
  }`
      })
    );

    const roundCount = Math.max(1, Math.ceil(Math.log2(rows)));
    for (let round = 0; round < roundCount; round++) {
      const roundId = `${id}-round-${round}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-cheapest-cost`,
          operation: OPERATION,
          variant: 'cheapest-cost',
          bindings: [
            ...csrBindings,
            {name: 'component', view: component, type: 'u32', access: 'read'},
            {name: 'slotCost', view: slotCost, type: 'f32', access: 'read'},
            {name: 'bestCost', view: bestCost, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const ROWS: u32 = ${rows}u;`,
          body: `let own = component[componentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (component[componentOffset + neighbors[neighborsOffset + slot]] != own) {
      atomicMin(&bestCost[bestCostOffset + own], bitcast<u32>(slotCost[slotCostOffset + slot]));
    }
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-cheapest-edge`,
          operation: OPERATION,
          variant: 'cheapest-edge',
          bindings: [
            ...csrBindings,
            {name: 'component', view: component, type: 'u32', access: 'read'},
            {name: 'slotCost', view: slotCost, type: 'f32', access: 'read'},
            {name: 'slotEdge', view: slotEdge, type: 'u32', access: 'read'},
            {name: 'bestCost', view: bestCost, type: 'u32', access: 'read'},
            {name: 'bestEdge', view: bestEdge, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const ROWS: u32 = ${rows}u;`,
          body: `let own = component[componentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (component[componentOffset + neighbors[neighborsOffset + slot]] != own &&
        bitcast<u32>(slotCost[slotCostOffset + slot]) == bestCost[bestCostOffset + own]) {
      atomicMin(&bestEdge[bestEdgeOffset + own], slotEdge[slotEdgeOffset + slot]);
    }
  }`
        }),
        // Each tree points at the tree across its cheapest link and flags that edge. Two trees
        // that chose the same edge would point at each other; with the total (cost, edge ID) order
        // that is exactly "the far tree's cheapest edge is this edge", and the lower label becomes
        // the root, so no separate pair-breaking pass is needed.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-hook`,
          operation: OPERATION,
          variant: 'hook',
          bindings: [
            ...csrBindings,
            {name: 'component', view: component, type: 'u32', access: 'read'},
            {name: 'bestEdge', view: bestEdge, type: 'u32', access: 'read'},
            {name: 'parentOut', view: parentB, type: 'u32', access: 'read_write'},
            {name: 'flags', view: treeEdgeFlags, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: rowDeclarations,
          body: `var parent = index;
  let edge = bestEdge[bestEdgeOffset + index];
  if (component[componentOffset + index] == index && edge != NONE) {
    let first = component[componentOffset + getRowOfSlot(edge)];
    let second = component[componentOffset + neighbors[neighborsOffset + edge]];
    parent = select(first, second, first == index);
    let isMutual = bestEdge[bestEdgeOffset + parent] == edge;
    parent = select(parent, index, isMutual && index < parent);
    atomicStore(&flags[flagsOffset + edge], 1u);
  }
  parentOut[parentOutOffset + index] = parent;`
        })
      );
      // Active trees at most halve each round, so chains are at most rows / 2^round long.
      const jumpCount =
        Math.max(1, Math.ceil(Math.log2(Math.max(2, Math.ceil(rows / 2 ** round))))) + 1;
      // The last pointer-jumping step is folded into the relabel kernel (a double indirection), so
      // `jumpCount - 1` separate passes remain. The relabel also clears the next round's
      // selections, which no kernel reads after the hook.
      let source = parentB;
      let destination = parentA;
      for (let jump = 0; jump < jumpCount - 1; jump++) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${roundId}-jump-${jump}`,
            operation: OPERATION,
            variant: 'jump',
            bindings: [
              {name: 'parentIn', view: source, type: 'u32', access: 'read'},
              {name: 'parentOut', view: destination, type: 'u32', access: 'read_write'}
            ],
            invocationCount: rows,
            body: `parentOut[parentOutOffset + index] =
    parentIn[parentInOffset + parentIn[parentInOffset + index]];`
          })
        );
        [source, destination] = [destination, source];
      }
      const isLastRound = round === roundCount - 1;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-relabel`,
          operation: OPERATION,
          variant: 'relabel',
          bindings: [
            {name: 'root', view: source, type: 'u32', access: 'read'},
            {name: 'component', view: component, type: 'u32', access: 'read_write'},
            ...(isLastRound
              ? [
                  {
                    name: 'minimumLabel',
                    view: minimumLabel,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [
                  {
                    name: 'bestCost',
                    view: bestCost,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  },
                  {
                    name: 'bestEdge',
                    view: bestEdge,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ])
          ],
          invocationCount: rows,
          body: `let label = component[componentOffset + index];
  component[componentOffset + index] = root[rootOffset + root[rootOffset + label]];
  ${
    isLastRound
      ? 'minimumLabel[minimumLabelOffset + index] = 0xffffffffu;'
      : `bestCost[bestCostOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`
  }`
        })
      );
    }

    // Canonical labels: the smallest row index of each tree.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-label-minimum`,
        operation: OPERATION,
        variant: 'label-minimum',
        bindings: [
          {name: 'component', view: component, type: 'u32', access: 'read'},
          {name: 'minimumLabel', view: minimumLabel, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rows,
        body: 'atomicMin(&minimumLabel[minimumLabelOffset + component[componentOffset + index]], index);'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-label-write`,
        operation: OPERATION,
        variant: 'label-write',
        bindings: [
          {name: 'component', view: component, type: 'u32', access: 'read'},
          {name: 'minimumLabel', view: minimumLabel, type: 'u32', access: 'read'},
          {name: 'labels', view: componentLabels, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: 'labels[labelsOffset + index] = minimumLabel[minimumLabelOffset + component[componentOffset + index]];'
      })
    );

    if (props.edges) {
      const {edges} = props;
      const slotIds = createTransientView(graph, `${id}-slot-ids`, 'uint32', capacity);
      const compactSlots = createTransientView(graph, `${id}-compact-slots`, 'uint32', capacity);
      const totalCount = createTransientView(graph, `${id}-edge-total`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-ids`,
          operation: OPERATION,
          variant: 'slot-ids',
          bindings: [{name: 'slotIds', view: slotIds, type: 'u32', access: 'read_write'}],
          invocationCount: capacity,
          body: 'slotIds[slotIdsOffset + index] = index;'
        }),
        ...new GPUCompaction({
          id: `${id}-compaction`,
          input: slotIds,
          flags: treeEdgeFlags,
          output: compactSlots,
          count: totalCount
        }).getCommandNodes(graph),
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          totalCount,
          compactIds: compactSlots,
          output: edges
        })
      );
      if (props.edgeEndpoints || props.edgeCosts) {
        const bindings = [
          ...csrBindings,
          {name: 'edgeIds', view: edges.ids, type: 'u32' as const, access: 'read' as const},
          {name: 'edgeCount', view: edges.count, type: 'u32' as const, access: 'read' as const},
          {name: 'slotCost', view: slotCost, type: 'f32' as const, access: 'read' as const},
          ...(props.edgeEndpoints
            ? [
                {
                  name: 'endpoints',
                  view: props.edgeEndpoints,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(props.edgeCosts
            ? [
                {
                  name: 'costs',
                  view: props.edgeCosts,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ];
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-edge-details`,
            operation: OPERATION,
            variant: 'edge-details',
            bindings,
            invocationCount: edges.ids.length,
            declarations: rowDeclarations,
            body: `if (index >= edgeCount[edgeCountOffset]) {
    return;
  }
  let slot = edgeIds[edgeIdsOffset + index];
  ${
    props.edgeEndpoints
      ? `endpoints[endpointsOffset + 2u * index] = getRowOfSlot(slot);
  endpoints[endpointsOffset + 2u * index + 1u] = neighbors[neighborsOffset + slot];`
      : ''
  }
  ${props.edgeCosts ? 'costs[costsOffset + index] = slotCost[slotCostOffset + slot];' : ''}`
          })
        );
      }
    }
    return nodes;
  }
}
