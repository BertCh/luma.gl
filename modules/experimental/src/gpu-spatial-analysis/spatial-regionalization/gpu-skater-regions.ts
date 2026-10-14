// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLDispatchGate, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {REGIONALIZATION_CSR_WGSL, REGIONALIZATION_NONE_WGSL} from './regionalization-wgsl';

const OPERATION = 'GPUSkaterRegions';

/** Largest supported attribute column count. */
export const GPU_SKATER_MAXIMUM_COLUMNS = 16;

/**
 * Largest `weights.neighbors` slot count (and row count) times `2 + columnCount` of
 * {@link GPUSkaterRegions} on `device`: the arc and region statistics tables hold `2 + columnCount`
 * float32 values per slot or row in one storage binding, so 33554432 on a default device (128 MiB
 * bindings). At 16 columns that is 1,864,135 slots.
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUSkaterMaximumStatisticCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/** Number of `uint32` words {@link GPUSkaterRegionsProps.parameters} must hold. */
export const GPU_SKATER_REGIONS_PARAMETER_LENGTH = 2;

/** Index of the target region count in the parameter words. */
const PARAMETER_REGION_COUNT = 0;
/** Index of the minimum region size in the parameter words. */
const PARAMETER_MINIMUM_REGION_SIZE = 1;

/**
 * Per-frame settings of {@link GPUSkaterRegions}, packed by
 * {@link getGPUSkaterRegionsParameterValues}.
 */
export type GPUSkaterRegionsSettings = {
  /**
   * Target region count, clamped to `maximumRegionCount` on the GPU. A value at or below the
   * number of forest trees cuts nothing.
   */
  regionCount: number;
  /** Minimum region size in rows. Defaults to 1. */
  minimumRegionSize?: number;
};

/**
 * Packs {@link GPUSkaterRegionsSettings} into the {@link GPU_SKATER_REGIONS_PARAMETER_LENGTH}
 * `uint32` words of {@link GPUSkaterRegionsProps.parameters}.
 */
export function getGPUSkaterRegionsParameterValues(
  settings: GPUSkaterRegionsSettings
): Uint32Array {
  const minimumRegionSize = settings.minimumRegionSize ?? 1;
  validateCapacity('GPUSkaterRegions', 'regionCount', settings.regionCount);
  validateCapacity('GPUSkaterRegions', 'minimumRegionSize', minimumRegionSize);
  const values = new Uint32Array(GPU_SKATER_REGIONS_PARAMETER_LENGTH);
  values[PARAMETER_REGION_COUNT] = settings.regionCount;
  values[PARAMETER_MINIMUM_REGION_SIZE] = minimumRegionSize;
  return values;
}

/** Value of an unused `cutEdges` entry. */
export const GPU_SKATER_NO_CUT = 0xffffffff;

const STATE_BEST_KEY = 0;
const STATE_BEST_EDGE = 1;
const STATE_REGION_COUNT = 2;
const STATE_APPLIED = 3;
const STATE_TARGET = 4;
const STATE_MINIMUM_SIZE = 5;
/** Region label the applied cut was taken from. */
const STATE_OLD_REGION = 6;
/** Row at the child end of the applied cut. */
const STATE_CHILD = 7;
/** Greedy step cursor: the reset-best kernel of step `s` sets it to `s + 1`. */
const STATE_STEP = 8;
const STATE_LOG = 9;

/**
 * Properties for {@link GPUSkaterRegions}.
 *
 * Compile-time: row count, slot capacity, `columnCount` and `maximumRegionCount` (the number of
 * unrolled greedy steps). Per-frame: the contents of `parameters`, `values`, `treeEdgeFlags`
 * and `componentLabels`, so a region-count slider is a buffer write.
 */
export type GPUSkaterRegionsProps = {
  /** Prefix for generated node IDs. Defaults to `'skater-regions'`. */
  id?: string;
  /**
   * The weights the spanning forest was built over. Both the slot count and the row count, times
   * `2 + columnCount`, are at most {@link getGPUSkaterMaximumStatisticCount}.
   */
  weights: GPUSpatialWeights;
  /**
   * `GPUSpatialWeightsMinimumSpanningTree` `treeEdgeFlags`: one flag per slot, set on the
   * lower-row slot of each tree edge.
   */
  treeEdgeFlags: GraphDataView<'uint32'>;
  /** `GPUSpatialWeightsMinimumSpanningTree` `componentLabels` (smallest row of each tree). */
  componentLabels: GraphDataView<'uint32'>;
  /**
   * Attribute table the cuts minimise the within-region sum of squares of, row-major with
   * `columnCount` columns. Pass the tree's `standardizedValues` so cuts and tree share a metric.
   * That table is centered and scaled per column (population standard deviation), which also
   * keeps the f32 subtree-sum subtraction well conditioned: raw attributes with a large mean
   * relative to their spread lose digits when a deep region's sums are subtracted.
   */
  values: GraphDataView<'float32'>;
  /** Number of attribute columns. Defaults to 1; at most 16. */
  columnCount?: number;
  /**
   * Compile-time cap on the region count, and the number of unrolled greedy steps minus the
   * islands: `maximumRegionCount - 1` steps are compiled. Must be at least 2.
   */
  maximumRegionCount: number;
  /**
   * Per-frame `uint32` words ({@link GPU_SKATER_REGIONS_PARAMETER_LENGTH}): the target region
   * count and minimum region size, written with {@link getGPUSkaterRegionsParameterValues}.
   */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUSkaterRegionsOutput;
};

/** Caller-owned outputs of {@link GPUSkaterRegions}. */
export type GPUSkaterRegionsOutput = {
  /**
   * Region label per row (`rows` entries): the smallest-depth row of the region, which is the
   * row nearest its tree's root, so labels are row indices.
   */
  labels: GraphDataView<'uint32'>;
  /** Optional one-row count of regions after the greedy steps (trees plus cuts). */
  regionCount?: GraphDataView<'uint32'>;
  /**
   * Optional cut log with `maximumRegionCount - 1` entries: the edge slot (stable edge ID) of
   * each applied cut in order, or `GPU_SKATER_NO_CUT` for steps that did not apply.
   */
  cutEdges?: GraphDataView<'uint32'>;
  /**
   * Optional per-step reduction of the within-region sum of squares of each applied cut
   * (`maximumRegionCount - 1` entries).
   */
  cutGains?: GraphDataView<'float32'>;
};

/**
 * SKATER regionalization: greedy cuts of a minimum spanning forest into spatially contiguous
 * regions that minimise within-region sum of squares (Assunção et al. 2006; PySAL `spopt` Skater).
 *
 * Each step evaluates every remaining tree edge as a cut in parallel, one thread per edge,
 * computing the SSD reduction `SSD(region) - SSD(side A) - SSD(side B)` from the side's size,
 * attribute sums and sum of squares, and applies the best candidate whose both sides keep at
 * least `minimumRegionSize` rows. Cuts apply while the region count is below the target region
 * count, so a region-count slider re-runs the same compiled steps. Region `k` of the greedy
 * sequence is a prefix of the sequence for `k + 1`, as in SKATER.
 *
 * **Tree statistics.** An Euler tour of the initial forest is list-ranked once; the rank orders the
 * arcs of a tree, so "arc a is an ancestor of arc b" and "row r lies below arc b" are two
 * comparisons. The statistics of the rows below every down arc are computed once by walking the
 * tour (`O(sum of subtree sizes)`, once per encoding). A cut then subtracts the removed side from
 * the arcs on its path to the region root (interval containment) and relabels the rows below it
 * (interval test per row), so a greedy step is `O(edges)` work and `O(1)` depth with no subtree
 * walks, instead of `O(edges * depth)` per step. Subtraction replaces a fresh sum, so the sums of
 * a deep region carry a few more f32 roundings than a direct walk (counts stay exact).
 *
 * **Ties and precision.** The best candidate is the maximum SSD reduction, ties to the lowest
 * edge slot, found with integer atomics so the choice is deterministic. Gains are f32; a CPU
 * oracle in f64 agrees except where two candidates' gains differ by less than f32 rounding.
 * `spopt` breaks ties by its own order and its stopping rule (it can return fewer regions than
 * requested when a floor binds) is reproduced here by leaving the region count below target.
 *
 * **Attributes and adjacency.** Neither SKATER nor the tree standardizes by itself: the MST node
 * does (`standardize`, default true) and exposes the centered table as `standardizedValues`. Feed
 * that table to `values`. The forest is defined over the weights slots, so the weights must list
 * every link in both rows (for directed weights, build the undirected union with
 * `GPUSpatialWeightsTranspose` first); the minimum size is a row count, not an extensive floor.
 *
 * Non-goals: other tree-cut criteria, trace output of every intermediate partition, and weights
 * that are not symmetric.
 */
export class GPUSkaterRegions implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSkaterRegionsProps;

  constructor(props: GPUSkaterRegionsProps) {
    const id = props.id ?? 'skater-regions';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    const columnCount = props.columnCount ?? 1;
    const {output} = props;
    validateCapacity(id, 'columnCount', columnCount, 1, GPU_SKATER_MAXIMUM_COLUMNS);
    validateCapacity(id, 'maximumRegionCount', props.maximumRegionCount, 2);
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length !== rows * columnCount) {
      throw new Error(`${id} values length must equal rows * columnCount`);
    }
    validatePackedView(props.treeEdgeFlags, ['uint32'], `${id} treeEdgeFlags`);
    if (props.treeEdgeFlags.length !== props.weights.neighbors.length) {
      throw new Error(`${id} treeEdgeFlags length must equal the weights slot capacity`);
    }
    validatePackedView(props.componentLabels, ['uint32'], `${id} componentLabels`);
    if (props.componentLabels.length !== rows) {
      throw new Error(`${id} componentLabels length must equal the weights row count`);
    }
    validatePackedView(props.parameters, ['uint32'], `${id} parameters`);
    if (props.parameters.length < GPU_SKATER_REGIONS_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold ${GPU_SKATER_REGIONS_PARAMETER_LENGTH} words`);
    }
    validatePackedView(output.labels, ['uint32'], `${id} output.labels`);
    if (output.labels.length !== rows) {
      throw new Error(`${id} output.labels length must equal the weights row count`);
    }
    if (output.regionCount) {
      validatePackedView(output.regionCount, ['uint32'], `${id} output.regionCount`);
    }
    const stepCount = props.maximumRegionCount - 1;
    if (output.cutEdges) {
      validatePackedView(output.cutEdges, ['uint32'], `${id} output.cutEdges`);
      if (output.cutEdges.length < stepCount) {
        throw new Error(`${id} output.cutEdges must hold maximumRegionCount - 1 entries`);
      }
    }
    if (output.cutGains) {
      validatePackedView(output.cutGains, ['float32'], `${id} output.cutGains`);
      if (output.cutGains.length < stepCount) {
        throw new Error(`${id} output.cutGains must hold maximumRegionCount - 1 entries`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.labels, output.regionCount, output.cutEdges, output.cutGains],
      [
        props.values,
        props.parameters,
        props.treeEdgeFlags,
        props.componentLabels,
        props.weights.offsets,
        props.weights.neighbors
      ]
    );
  }

  /** Returns the orientation, per-step candidate evaluation, selection and cut nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, values, output} = props;
    const {labels} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      props.treeEdgeFlags,
      props.componentLabels,
      props.parameters,
      labels,
      output.regionCount,
      output.cutEdges,
      output.cutGains,
      weights.offsets,
      weights.neighbors
    ]);
    const rows = weights.offsets.length - 1;
    const capacity = weights.neighbors.length;
    const columnCount = props.columnCount ?? 1;
    const maximumStatisticCount = getGPUSkaterMaximumStatisticCount(graph.device);
    validateMaximumCount(
      id,
      'weights.neighbors rows x (2 + columnCount)',
      capacity * (2 + columnCount),
      maximumStatisticCount
    );
    validateMaximumCount(
      id,
      'weights rows x (2 + columnCount)',
      rows * (2 + columnCount),
      maximumStatisticCount
    );
    const stepCount = props.maximumRegionCount - 1;
    const statsStride = 2 + columnCount;
    const stateLength = STATE_LOG + 2 * stepCount;

    const twin = createTransientView(graph, `${id}-twin`, 'uint32', capacity);
    const arcFlags = createTransientView(graph, `${id}-arc-flags`, 'uint32', capacity);
    const arcInfo = createTransientView(graph, `${id}-arc-info`, 'uint32', capacity);
    const gainKey = createTransientView(graph, `${id}-gain-key`, 'uint32', capacity);
    const rankNextA = createTransientView(graph, `${id}-rank-next-a`, 'uint32', capacity);
    const rankNextB = createTransientView(graph, `${id}-rank-next-b`, 'uint32', capacity);
    const rankDistanceA = createTransientView(graph, `${id}-rank-distance-a`, 'uint32', capacity);
    const rankDistanceB = createTransientView(graph, `${id}-rank-distance-b`, 'uint32', capacity);
    const arcStats = createTransientView(
      graph,
      `${id}-arc-stats`,
      'float32',
      capacity * statsStride
    );
    const rowDownArc = createTransientView(graph, `${id}-row-down-arc`, 'uint32', rows);
    const regionStats = createTransientView(
      graph,
      `${id}-region-stats`,
      'float32',
      rows * statsStride
    );
    const state = createTransientView(graph, `${id}-state`, 'uint32', stateLength);
    // Indirect dispatch triples of the gated per-step kernels (rule 13): slot 0 for the per-arc
    // kernels, slot 1 for the per-row relabel. The one-thread apply kernel zeroes them when a step
    // finds no cut and the deactivate kernel when the target region count is reached, so the
    // remaining steps launch only their one-thread kernels.
    const stepGate = createWGSLDispatchGate(graph, {
      id: `${id}-step-dispatch`,
      operation: OPERATION,
      shapes: [[capacity], [rows]]
    });

    const csr = [
      {name: 'offsets', view: weights.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'neighbors', view: weights.neighbors, type: 'u32' as const, access: 'read' as const}
    ];
    const constants = `const ROWS: u32 = ${rows}u;
const COLUMNS: u32 = ${columnCount}u;
const STRIDE: u32 = ${statsStride}u;
const STEP_COUNT: u32 = ${stepCount}u;
const MAXIMUM_REGIONS: u32 = ${props.maximumRegionCount}u;
const STATE_BEST_KEY: u32 = ${STATE_BEST_KEY}u;
const STATE_BEST_EDGE: u32 = ${STATE_BEST_EDGE}u;
const STATE_REGION_COUNT: u32 = ${STATE_REGION_COUNT}u;
const STATE_APPLIED: u32 = ${STATE_APPLIED}u;
const STATE_TARGET: u32 = ${STATE_TARGET}u;
const STATE_MINIMUM_SIZE: u32 = ${STATE_MINIMUM_SIZE}u;
const STATE_OLD_REGION: u32 = ${STATE_OLD_REGION}u;
const STATE_CHILD: u32 = ${STATE_CHILD}u;
const STATE_STEP: u32 = ${STATE_STEP}u;
const STATE_LOG: u32 = ${STATE_LOG}u;
const SLOT_MASK: u32 = 0x3fffffffu;
${REGIONALIZATION_NONE_WGSL}`;
    // Needs the twin and arcFlags bindings plus the CSR bindings.
    const tourFunctions = `${REGIONALIZATION_CSR_WGSL}
fn isLive(flags: u32) -> bool { return (flags & 1u) != 0u; }

// Successor of an arc in the Euler tour of the live arcs: the next live arc after the twin of
// the arc around the arc's head, cyclically.
fn getSuccessor(arc: u32) -> u32 {
  let head = neighbors[neighborsOffset + arc];
  let twinArc = twin[twinOffset + arc];
  let rowStart = offsets[offsetsOffset + head];
  let degree = offsets[offsetsOffset + head + 1u] - rowStart;
  for (var step = 1u; step <= degree; step++) {
    let slot = rowStart + (twinArc - rowStart + step) % degree;
    if (isLive(arcFlags[arcFlagsOffset + slot])) {
      return slot;
    }
  }
  return twinArc;
}

// First live arc of a row, or NONE.
fn getFirstLiveArc(row: u32) -> u32 {
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    if (isLive(arcFlags[arcFlagsOffset + slot])) {
      return slot;
    }
  }
  return NONE;
}`;
    // Accessors for the packed per-step arc table; no bindings needed.
    const infoFunctions = `fn getInfoSuccessor(info: u32) -> u32 { return info & SLOT_MASK; }
fn isInfoLive(info: u32) -> bool { return ((info >> 30u) & 1u) != 0u; }
fn isInfoDown(info: u32) -> bool { return ((info >> 31u) & 1u) != 0u; }
fn isInfoLiveFlag(flags: u32) -> bool { return (flags & 1u) != 0u; }
fn getSsd(count: f32, squares: f32, sumSquares: f32) -> f32 {
  return squares - sumSquares / max(count, 1.0);
}`;
    const declarations = constants;
    const tourDeclarations = `${constants}\n${tourFunctions}`;
    const infoDeclarations = `${constants}\n${infoFunctions}`;

    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Twin slots and live-arc flags from the tree edge flags. Slots past the last row's end
    // (spare capacity) must read as dead arcs, so every flag is cleared first.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-arcs`,
        operation: OPERATION,
        variant: 'clear-arcs',
        bindings: [{name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read_write'}],
        invocationCount: capacity,
        body: 'arcFlags[arcFlagsOffset + index] = 0u;'
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-twins`,
        operation: OPERATION,
        variant: 'twins',
        bindings: [
          ...csr,
          {name: 'treeFlags', view: props.treeEdgeFlags, type: 'u32', access: 'read'},
          {name: 'twin', view: twin, type: 'u32', access: 'read_write'},
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${constants}\n${REGIONALIZATION_CSR_WGSL}`,
        body: `for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let twinSlot = findSlot(neighbor, index);
    twin[twinOffset + slot] = twinSlot;
    var edge = slot;
    if (twinSlot != NONE && twinSlot < slot) {
      edge = twinSlot;
    }
    arcFlags[arcFlagsOffset + slot] =
      select(0u, 1u, twinSlot != NONE && treeFlags[treeFlagsOffset + edge] != 0u);
  }`
      })
    );

    // 2. Tour of the whole forest: successors cut at each tree's head arc, then list ranking.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rank-init`,
        operation: OPERATION,
        variant: 'rank-init',
        bindings: [
          ...csr,
          {name: 'twin', view: twin, type: 'u32', access: 'read'},
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
          {name: 'treeRoots', view: props.componentLabels, type: 'u32', access: 'read'},
          {name: 'nextOut', view: rankNextA, type: 'u32', access: 'read_write'},
          {name: 'distanceOut', view: rankDistanceA, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: tourDeclarations,
        body: `if (!isLive(arcFlags[arcFlagsOffset + index])) {
    nextOut[nextOutOffset + index] = index;
    distanceOut[distanceOutOffset + index] = 0u;
    return;
  }
  let successor = getSuccessor(index);
  let root = treeRoots[treeRootsOffset + neighbors[neighborsOffset + index]];
  let isLast = successor == getFirstLiveArc(root);
  nextOut[nextOutOffset + index] = select(successor, index, isLast);
  distanceOut[distanceOutOffset + index] = select(1u, 0u, isLast);`
      })
    );
    let rankSource = {next: rankNextA, distance: rankDistanceA};
    let rankDestination = {next: rankNextB, distance: rankDistanceB};
    const rankJumpCount = Math.max(1, Math.ceil(Math.log2(Math.max(2, 2 * rows))));
    for (let jump = 0; jump < rankJumpCount; jump++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-rank-jump-${jump}`,
          operation: OPERATION,
          variant: 'rank-jump',
          bindings: [
            {name: 'nextIn', view: rankSource.next, type: 'u32', access: 'read'},
            {name: 'distanceIn', view: rankSource.distance, type: 'u32', access: 'read'},
            {name: 'nextOut', view: rankDestination.next, type: 'u32', access: 'read_write'},
            {name: 'distanceOut', view: rankDestination.distance, type: 'u32', access: 'read_write'}
          ],
          invocationCount: capacity,
          body: `let successor = nextIn[nextInOffset + index];
  distanceOut[distanceOutOffset + index] =
    distanceIn[distanceInOffset + index] + distanceIn[distanceInOffset + successor];
  nextOut[nextOutOffset + index] = nextIn[nextInOffset + successor];`
        })
      );
      [rankSource, rankDestination] = [rankDestination, rankSource];
    }
    // The earlier arc of an edge is its down arc (away from the tree's root).
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-orient`,
        operation: OPERATION,
        variant: 'orient',
        bindings: [
          {name: 'twin', view: twin, type: 'u32', access: 'read'},
          {name: 'distance', view: rankSource.distance, type: 'u32', access: 'read'},
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        body: `let flags = arcFlags[arcFlagsOffset + index];
  if ((flags & 1u) != 0u &&
      distance[distanceOffset + index] > distance[distanceOffset + twin[twinOffset + index]]) {
    arcFlags[arcFlagsOffset + index] = flags | 2u;
  }`
      })
    );

    // 3. State, labels and region count.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-init-state`,
        operation: OPERATION,
        variant: 'init-state',
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          stepGate.binding
        ],
        invocationCount: stateLength,
        declarations,
        body: `if (index == 0u) {
    ${stepGate.armWGSL}
  }
  var word = 0u;
  if (index == STATE_TARGET) {
    word = min(parameters[parametersOffset + ${PARAMETER_REGION_COUNT}u], MAXIMUM_REGIONS);
  } else if (index == STATE_MINIMUM_SIZE) {
    word = max(parameters[parametersOffset + ${PARAMETER_MINIMUM_REGION_SIZE}u], 1u);
  } else if (index >= STATE_LOG && ((index - STATE_LOG) % 2u) == 0u) {
    word = NONE;
  }
  state[stateOffset + index] = word;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-init-labels`,
        operation: OPERATION,
        variant: 'init-labels',
        bindings: [
          {name: 'treeRoots', view: props.componentLabels, type: 'u32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `let root = treeRoots[treeRootsOffset + index];
  labels[labelsOffset + index] = root;
  if (root == index) {
    atomicAdd(&state[stateOffset + STATE_REGION_COUNT], 1u);
  }`
      })
    );

    // 4. One-time tables over the initial forest. `tourDistance` (arcs left in the tour) orders
    // arcs of one tree along its Euler tour, which makes "arc a is an ancestor of arc b" and "row r
    // is below arc b" constant-time interval tests for every later step. Cuts never reorder it.
    const tourDistance = rankSource.distance;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-successors`,
        operation: OPERATION,
        variant: 'successors',
        bindings: [
          ...csr,
          {name: 'twin', view: twin, type: 'u32', access: 'read'},
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
          {name: 'arcInfo', view: arcInfo, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: tourDeclarations,
        body: `let flags = arcFlags[arcFlagsOffset + index];
  var info = (flags & 3u) << 30u;
  if (isLive(flags)) {
    info = info | getSuccessor(index);
  }
  arcInfo[arcInfoOffset + index] = info;`
      }),
      // The down arc into a row is the twin of the row's live up arc (its arc to its parent).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-down-arcs`,
        operation: OPERATION,
        variant: 'row-down-arcs',
        bindings: [
          ...csr,
          {name: 'twin', view: twin, type: 'u32', access: 'read'},
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
          {name: 'rowDownArc', view: rowDownArc, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: infoDeclarations,
        body: `var downArc = NONE;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let flags = arcFlags[arcFlagsOffset + slot];
    if (isInfoLiveFlag(flags) && (flags & 2u) == 0u) {
      downArc = twin[twinOffset + slot];
    }
  }
  rowDownArc[rowDownArcOffset + index] = downArc;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tree-stats`,
        operation: OPERATION,
        variant: 'tree-stats',
        bindings: [
          ...csr,
          {name: 'arcInfo', view: arcInfo, type: 'u32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'regionStats', view: regionStats, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: infoDeclarations,
        body: `if (labels[labelsOffset + index] != index) {
    return;
  }
  var count = 1.0;
  var squares = 0.0;
  var sums: array<f32, ${columnCount}>;
  for (var column = 0u; column < COLUMNS; column++) {
    let value = values[valuesOffset + index * COLUMNS + column];
    sums[column] = value;
    squares += value * value;
  }
  var head = NONE;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (isInfoLive(arcInfo[arcInfoOffset + slot])) {
      head = slot;
      break;
    }
  }
  if (head != NONE) {
    var arc = head;
    for (var guard = 0u; guard < 2u * ROWS; guard++) {
      let info = arcInfo[arcInfoOffset + arc];
      if (isInfoDown(info)) {
        let row = neighbors[neighborsOffset + arc];
        count += 1.0;
        for (var column = 0u; column < COLUMNS; column++) {
          let value = values[valuesOffset + row * COLUMNS + column];
          sums[column] += value;
          squares += value * value;
        }
      }
      arc = getInfoSuccessor(info);
      if (arc == head) {
        break;
      }
    }
  }
  let base = regionStatsOffset + index * STRIDE;
  regionStats[base] = count;
  regionStats[base + 1u] = squares;
  for (var column = 0u; column < COLUMNS; column++) {
    regionStats[base + 2u + column] = sums[column];
  }`
      }),
      // Statistics of the rows below every down arc of the initial forest, by walking the tour
      // between the arc and its twin. This is the only walk: later cuts update these in O(1).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-arc-stats`,
        operation: OPERATION,
        variant: 'arc-stats',
        bindings: [
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'twin', view: twin, type: 'u32', access: 'read'},
          {name: 'arcInfo', view: arcInfo, type: 'u32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'arcStats', view: arcStats, type: 'f32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: infoDeclarations,
        body: `let startInfo = arcInfo[arcInfoOffset + index];
  if (!isInfoLive(startInfo) || !isInfoDown(startInfo)) {
    return;
  }
  let child = neighbors[neighborsOffset + index];
  let end = twin[twinOffset + index];
  var count = 1.0;
  var squares = 0.0;
  var sums: array<f32, ${columnCount}>;
  for (var column = 0u; column < COLUMNS; column++) {
    let value = values[valuesOffset + child * COLUMNS + column];
    sums[column] = value;
    squares += value * value;
  }
  var arc = getInfoSuccessor(startInfo);
  for (var guard = 0u; guard < 2u * ROWS && arc != end; guard++) {
    let info = arcInfo[arcInfoOffset + arc];
    if (isInfoDown(info)) {
      let row = neighbors[neighborsOffset + arc];
      count += 1.0;
      for (var column = 0u; column < COLUMNS; column++) {
        let value = values[valuesOffset + row * COLUMNS + column];
        sums[column] += value;
        squares += value * value;
      }
    }
    arc = getInfoSuccessor(info);
  }
  let base = arcStatsOffset + index * STRIDE;
  arcStats[base] = count;
  arcStats[base + 1u] = squares;
  for (var column = 0u; column < COLUMNS; column++) {
    arcStats[base + 2u + column] = sums[column];
  }`
      })
    );

    // 5. Greedy steps. Each step is O(1) depth: the candidate gains read the maintained arc
    // statistics, the chosen cut relabels the rows below it with an interval test, and only the
    // arcs on the cut's path to the region root (found by interval containment) subtract the
    // removed statistics. No step walks a subtree.
    for (let step = 0; step < stepCount; step++) {
      const stepId = `${id}-step-${step}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-candidates`,
          operation: OPERATION,
          variant: 'candidates',
          ...stepGate.getGate(0),
          bindings: [
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
            {name: 'arcStats', view: arcStats, type: 'f32', access: 'read'},
            {name: 'labels', view: labels, type: 'u32', access: 'read'},
            {name: 'regionStats', view: regionStats, type: 'f32', access: 'read'},
            {name: 'gainKey', view: gainKey, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read'}
          ],
          invocationCount: capacity,
          declarations: infoDeclarations,
          body: `gainKey[gainKeyOffset + index] = 0u;
  let flags = arcFlags[arcFlagsOffset + index];
  if (!isInfoLiveFlag(flags) || (flags & 2u) == 0u ||
      state[stateOffset + STATE_REGION_COUNT] >= state[stateOffset + STATE_TARGET]) {
    return;
  }
  let child = neighbors[neighborsOffset + index];
  let childBase = arcStatsOffset + index * STRIDE;
  let count = arcStats[childBase];
  let squares = arcStats[childBase + 1u];
  let base = regionStatsOffset + labels[labelsOffset + child] * STRIDE;
  let regionCount = regionStats[base];
  let remainder = regionCount - count;
  let minimum = f32(state[stateOffset + STATE_MINIMUM_SIZE]);
  if (count < minimum || remainder < minimum) {
    return;
  }
  var regionSumSquares = 0.0;
  var childSumSquares = 0.0;
  var remainderSumSquares = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    let regionSum = regionStats[base + 2u + column];
    let childSum = arcStats[childBase + 2u + column];
    regionSumSquares += regionSum * regionSum;
    childSumSquares += childSum * childSum;
    let remainderSum = regionSum - childSum;
    remainderSumSquares += remainderSum * remainderSum;
  }
  let gain = getSsd(regionCount, regionStats[base + 1u], regionSumSquares)
    - getSsd(count, squares, childSumSquares)
    - getSsd(remainder, regionStats[base + 1u] - squares, remainderSumSquares);
  gainKey[gainKeyOffset + index] = bitcast<u32>(max(gain, 0.0)) + 1u;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-reset-best`,
          operation: OPERATION,
          variant: 'reset-best',
          bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
          invocationCount: 1,
          declarations,
          body: `state[stateOffset + STATE_BEST_KEY] = 0u;
  state[stateOffset + STATE_BEST_EDGE] = NONE;
  state[stateOffset + STATE_APPLIED] = 0u;
  state[stateOffset + STATE_STEP] = state[stateOffset + STATE_STEP] + 1u;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-best-gain`,
          operation: OPERATION,
          variant: 'best-gain',
          ...stepGate.getGate(0),
          bindings: [
            {name: 'gainKey', view: gainKey, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: capacity,
          declarations,
          body: `let key = gainKey[gainKeyOffset + index];
  if (key != 0u) {
    atomicMax(&state[stateOffset + STATE_BEST_KEY], key);
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-best-edge`,
          operation: OPERATION,
          variant: 'best-edge',
          ...stepGate.getGate(0),
          bindings: [
            {name: 'gainKey', view: gainKey, type: 'u32', access: 'read'},
            {name: 'twin', view: twin, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: capacity,
          declarations,
          body: `let key = gainKey[gainKeyOffset + index];
  if (key != 0u && key == atomicLoad(&state[stateOffset + STATE_BEST_KEY])) {
    atomicMin(&state[stateOffset + STATE_BEST_EDGE], min(index, twin[twinOffset + index]));
  }`
        }),
        // Records the cut and moves the child side's statistics from its region to a new region.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-apply`,
          operation: OPERATION,
          variant: 'apply',
          bindings: [
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'twin', view: twin, type: 'u32', access: 'read'},
            {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
            {name: 'arcStats', view: arcStats, type: 'f32', access: 'read'},
            {name: 'labels', view: labels, type: 'u32', access: 'read'},
            {name: 'regionStats', view: regionStats, type: 'f32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            stepGate.binding
          ],
          invocationCount: 1,
          declarations: infoDeclarations,
          // Ungated (it writes the gate). No cut: this and every later step changes nothing.
          body: `let edge = state[stateOffset + STATE_BEST_EDGE];
  if (state[stateOffset + STATE_BEST_KEY] == 0u || edge == NONE) {
    ${stepGate.disarmWGSL}
    return;
  }
  let other = twin[twinOffset + edge];
  let down = select(other, edge, (arcFlags[arcFlagsOffset + edge] & 2u) != 0u);
  let child = neighbors[neighborsOffset + down];
  let region = labels[labelsOffset + child];
  let regionBase = regionStatsOffset + region * STRIDE;
  let childBase = regionStatsOffset + child * STRIDE;
  let removedBase = arcStatsOffset + down * STRIDE;
  regionStats[childBase] = arcStats[removedBase];
  regionStats[childBase + 1u] = arcStats[removedBase + 1u];
  regionStats[regionBase] = regionStats[regionBase] - arcStats[removedBase];
  regionStats[regionBase + 1u] = regionStats[regionBase + 1u] - arcStats[removedBase + 1u];
  for (var column = 0u; column < COLUMNS; column++) {
    let sum = arcStats[removedBase + 2u + column];
    regionStats[childBase + 2u + column] = sum;
    regionStats[regionBase + 2u + column] = regionStats[regionBase + 2u + column] - sum;
  }
  state[stateOffset + STATE_REGION_COUNT] = state[stateOffset + STATE_REGION_COUNT] + 1u;
  state[stateOffset + STATE_APPLIED] = down + 1u;
  state[stateOffset + STATE_OLD_REGION] = region;
  state[stateOffset + STATE_CHILD] = child;
  // The step index comes from the cursor, so every step shares one pipeline (rule 12).
  let logSlot = STATE_LOG + 2u * (state[stateOffset + STATE_STEP] - 1u);
  state[stateOffset + logSlot] = min(edge, other);
  state[stateOffset + logSlot + 1u] = state[stateOffset + STATE_BEST_KEY];`
        }),
        // Rows of the cut's region that lie below the cut take the child's label. A row is below
        // the cut when its parent arc lies strictly inside the cut arc's tour interval.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-relabel`,
          operation: OPERATION,
          variant: 'relabel',
          ...stepGate.getGate(1),
          bindings: [
            {name: 'rowDownArc', view: rowDownArc, type: 'u32', access: 'read'},
            {name: 'twin', view: twin, type: 'u32', access: 'read'},
            {name: 'tourDistance', view: tourDistance, type: 'u32', access: 'read'},
            {name: 'labels', view: labels, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read'}
          ],
          invocationCount: rows,
          declarations,
          body: `let applied = state[stateOffset + STATE_APPLIED];
  if (applied == 0u || labels[labelsOffset + index] != state[stateOffset + STATE_OLD_REGION]) {
    return;
  }
  let down = applied - 1u;
  let child = state[stateOffset + STATE_CHILD];
  var isBelow = index == child;
  let parentArc = rowDownArc[rowDownArcOffset + index];
  if (parentArc != NONE) {
    let distance = tourDistance[tourDistanceOffset + parentArc];
    isBelow = isBelow || (distance < tourDistance[tourDistanceOffset + down] &&
      distance > tourDistance[tourDistanceOffset + twin[twinOffset + down]]);
  }
  if (isBelow) {
    labels[labelsOffset + index] = child;
  }`
        }),
        // Ancestors of the cut inside its region lose the removed rows. An arc is an ancestor when
        // its tour interval contains the cut arc's interval.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-update-ancestors`,
          operation: OPERATION,
          variant: 'update-ancestors',
          ...stepGate.getGate(0),
          bindings: [
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'twin', view: twin, type: 'u32', access: 'read'},
            {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
            {name: 'tourDistance', view: tourDistance, type: 'u32', access: 'read'},
            {name: 'arcStats', view: arcStats, type: 'f32', access: 'read_write'},
            {name: 'labels', view: labels, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'}
          ],
          invocationCount: capacity,
          declarations: infoDeclarations,
          body: `let applied = state[stateOffset + STATE_APPLIED];
  let flags = arcFlags[arcFlagsOffset + index];
  if (applied == 0u || applied - 1u == index || !isInfoLiveFlag(flags) || (flags & 2u) == 0u) {
    return;
  }
  let down = applied - 1u;
  if (labels[labelsOffset + neighbors[neighborsOffset + index]] != state[stateOffset + STATE_OLD_REGION]) {
    return;
  }
  if (!(tourDistance[tourDistanceOffset + index] > tourDistance[tourDistanceOffset + down] &&
      tourDistance[tourDistanceOffset + twin[twinOffset + index]] < tourDistance[tourDistanceOffset + twin[twinOffset + down]])) {
    return;
  }
  let base = arcStatsOffset + index * STRIDE;
  let removedBase = arcStatsOffset + down * STRIDE;
  for (var word = 0u; word < STRIDE; word++) {
    arcStats[base + word] = arcStats[base + word] - arcStats[removedBase + word];
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-deactivate`,
          operation: OPERATION,
          variant: 'deactivate',
          bindings: [
            {name: 'twin', view: twin, type: 'u32', access: 'read'},
            {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            stepGate.binding
          ],
          invocationCount: 1,
          declarations,
          // Ungated (it writes the gate): the target region count ends the remaining steps.
          body: `if (state[stateOffset + STATE_REGION_COUNT] >= state[stateOffset + STATE_TARGET]) {
    ${stepGate.disarmWGSL}
  }
  let applied = state[stateOffset + STATE_APPLIED];
  if (applied == 0u) {
    return;
  }
  let down = applied - 1u;
  arcFlags[arcFlagsOffset + down] = arcFlags[arcFlagsOffset + down] & ~1u;
  let up = twin[twinOffset + down];
  arcFlags[arcFlagsOffset + up] = arcFlags[arcFlagsOffset + up] & ~1u;`
        })
      );
    }

    // 5. Publish the cut log and region count.
    const publishBindings = [
      {name: 'state', view: state, type: 'u32' as const, access: 'read' as const},
      ...(output.regionCount
        ? [
            {
              name: 'regionCountOut',
              view: output.regionCount,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : []),
      ...(output.cutEdges
        ? [
            {
              name: 'cutEdgesOut',
              view: output.cutEdges,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : []),
      ...(output.cutGains
        ? [
            {
              name: 'cutGainsOut',
              view: output.cutGains,
              type: 'f32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ];
    if (publishBindings.length > 1) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          variant: 'publish',
          bindings: publishBindings,
          invocationCount: Math.max(stepCount, 1),
          declarations,
          body: `${
            output.regionCount
              ? 'if (index == 0u) { regionCountOut[regionCountOutOffset] = state[stateOffset + STATE_REGION_COUNT]; }'
              : ''
          }
  if (index < STEP_COUNT) {
    let edge = state[stateOffset + STATE_LOG + 2u * index];
    let key = state[stateOffset + STATE_LOG + 2u * index + 1u];
    ${output.cutEdges ? 'cutEdgesOut[cutEdgesOutOffset + index] = edge;' : ''}
    ${output.cutGains ? 'cutGainsOut[cutGainsOutOffset + index] = select(0.0, bitcast<f32>(key - 1u), key != 0u);' : ''}
  }`
        })
      );
    }
    return nodes;
  }
}
