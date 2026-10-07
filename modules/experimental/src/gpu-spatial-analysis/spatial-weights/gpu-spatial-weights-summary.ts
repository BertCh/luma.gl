// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getSlotGroupingNodes} from './slot-grouping';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from './spatial-weights';

const OPERATION = 'GPUSpatialWeightsSummary';

/** Rows reduced by one workgroup in the first level of the global reductions. */
const SUMMARY_BLOCK_ROWS = 4096;
/** Threads of the reduction workgroups. */
const SUMMARY_WORKGROUP_SIZE = 256;

/** Element positions of the `statistics` and `counts` outputs of {@link GPUSpatialWeightsSummary}. */
export const GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT = {
  /** Float32 `statistics`: `S0`, `S1`, `S2`. */
  statistics: {s0: 0, s1: 1, s2: 2, length: 3},
  /** Uint32 `counts`: used slots, asymmetric slots, isolates, minimum and maximum cardinality. */
  counts: {
    slots: 0,
    asymmetricSlots: 1,
    isolates: 2,
    minimumCardinality: 3,
    maximumCardinality: 4,
    length: 5
  }
} as const;

/** Properties for {@link GPUSpatialWeightsSummary}. */
export type GPUSpatialWeightsSummaryProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-weights-summary'`. */
  id?: string;
  /** Square weights to summarize. */
  weights: GPUSpatialWeights;
  /** Float32 output of at least 3 values, see {@link GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT}. */
  statistics: GraphDataView<'float32'>;
  /** Uint32 output of at least 5 values, see {@link GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT}. */
  counts: GraphDataView<'uint32'>;
  /** Optional per-row cardinality (neighbor count), one uint32 per row. */
  cardinality?: GraphDataView<'uint32'>;
  /**
   * Absolute tolerance for calling `w_ij` and `w_ji` equal when counting asymmetric slots.
   * Defaults to 0 (exact).
   */
  symmetryTolerance?: number;
};

/**
 * Descriptive statistics of a {@link GPUSpatialWeights} on the GPU, the libpysal `W.s0/s1/s2`,
 * `cardinalities`, `asymmetry()` and `islands` summary:
 *
 * - `S0 = sum_ij w_ij`
 * - `S1 = 1/2 sum_ij (w_ij + w_ji)^2`, a missing `w_ji` counting as 0
 * - `S2 = sum_i (sum_j w_ij + sum_j w_ji)^2`
 * - used slots (`offsets[rows]`), minimum and maximum cardinality, and per-row cardinality
 * - asymmetric slots: slots `(i, j)` whose reverse slot `(j, i)` is missing or differs by more
 *   than `symmetryTolerance` (a pair of different weights counts twice, a one-way link once)
 * - isolates: rows with no slot
 *
 * Sums are fixed-order (row sums in slot order, column sums over a stable sort, a tree
 * reduction), so the results are bitwise reproducible. Slots with a neighbor ID `>= rows` are
 * ignored by the column sums and counted as one-way links. Self slots are included as in libpysal.
 */
export class GPUSpatialWeightsSummary implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsSummaryProps;

  constructor(props: GPUSpatialWeightsSummaryProps) {
    const id = props.id ?? 'spatial-weights-summary';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    validatePackedView(props.statistics, ['float32'], `${id} statistics`);
    if (props.statistics.length < GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.statistics.length) {
      throw new Error(`${id} statistics must hold at least 3 values`);
    }
    validatePackedUint32View(props.counts, `${id} counts`);
    if (props.counts.length < GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.counts.length) {
      throw new Error(`${id} counts must hold at least 5 values`);
    }
    if (props.cardinality) {
      validatePackedUint32View(props.cardinality, `${id} cardinality`);
      if (props.cardinality.length !== rows) {
        throw new Error(`${id} cardinality length must equal the weights row count`);
      }
    }
    if (
      props.symmetryTolerance !== undefined &&
      !(Number.isFinite(props.symmetryTolerance) && props.symmetryTolerance >= 0)
    ) {
      throw new Error(`${id} symmetryTolerance must be a non-negative finite number`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.statistics, props.counts, props.cardinality],
      [props.weights.offsets, props.weights.neighbors, props.weights.weights]
    );
  }

  /** Returns the summary nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, statistics, counts} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      statistics,
      counts,
      props.cardinality
    ]);
    const rows = weights.offsets.length - 1;
    const capacity = weights.neighbors.length;
    const tolerance = getWGSLFloatLiteral(props.symmetryTolerance ?? 0);
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const slotKeys = view('slot-keys', 'uint32', capacity);
    const slotIndices = view('slot-indices', 'uint32', capacity);
    const columnOffsets = view('column-offsets', 'uint32', rows + 1);
    const columnSums = view('column-sums', 'float32', rows);
    const rowSums = view('row-sums', 'float32', rows);
    const s1Partials = view('s1-partials', 'float32', rows);
    const s2Partials = view('s2-partials', 'float32', rows);
    const asymmetric = view('asymmetric', 'uint32', rows);
    const isolates = view('isolates', 'uint32', rows);
    const cardinality = props.cardinality ?? view('cardinality', 'uint32', rows);
    const constants = `const ROWS: u32 = ${rows}u;`;
    const grouping = getSlotGroupingNodes<Parameters>(graph, {
      id: `${id}-columns`,
      operation: OPERATION,
      slotKeys,
      slotIndices,
      columns: rows,
      columnOffsets
    });
    const nodes: GPUCommandNode<Parameters>[] = [
      // Column sums: group the used slots by neighbor, in slot order, so the sums are reproducible.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slot-keys`,
        operation: OPERATION,
        variant: 'slot-keys',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'slotKeys', view: slotKeys, type: 'u32', access: 'read_write'},
          {name: 'slotIndices', view: slotIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: constants,
        body: `let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + ROWS] && neighbor < ROWS;
  slotKeys[slotKeysOffset + index] = select(ROWS, neighbor, used);
  slotIndices[slotIndicesOffset + index] = index;`
      }),
      ...grouping.nodes,
      // One invocation per column: short segments (the degree) do not deserve a workgroup each.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-column-sums`,
        operation: OPERATION,
        variant: 'column-sums',
        bindings: [
          {name: 'columnOffsets', view: columnOffsets, type: 'u32', access: 'read'},
          {name: 'sortedSlots', view: grouping.sortedSlots, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'columnSums', view: columnSums, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        // Slots of one column are in ascending slot order (stable sort), so the sum order is fixed.
        body: `var sum = 0.0;
  for (var position = columnOffsets[columnOffsetsOffset + index];
      position < columnOffsets[columnOffsetsOffset + index + 1u]; position++) {
    sum += weights[weightsOffset + sortedSlots[sortedSlotsOffset + position]];
  }
  columnSums[columnSumsOffset + index] = sum;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rows`,
        operation: OPERATION,
        variant: 'rows',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'rowSums', view: rowSums, type: 'f32', access: 'read_write'},
          {name: 's1Partials', view: s1Partials, type: 'f32', access: 'read_write'},
          {name: 'asymmetric', view: asymmetric, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${constants}
const TOLERANCE: f32 = ${tolerance};
const MISSING: f32 = -1.0;

// Weight w_row,column by binary search of the ascending neighbor IDs of row, or MISSING.
fn findWeight(row: u32, column: u32) -> f32 {
  var low = offsets[offsetsOffset + row];
  let end = offsets[offsetsOffset + row + 1u];
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && neighbors[neighborsOffset + low] == column) {
    return weights[weightsOffset + low];
  }
  return MISSING;
}`,
        body: `var sum = 0.0;
  var s1 = 0.0;
  var unmatched = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    sum += weight;
    var reverse = MISSING;
    if (neighbor < ROWS) {
      reverse = findWeight(neighbor, index);
    }
    if (reverse == MISSING) {
      s1 += 2.0 * weight * weight;
      unmatched++;
    } else {
      s1 += (weight + reverse) * (weight + reverse);
      if (abs(weight - reverse) > TOLERANCE) {
        unmatched++;
      }
    }
  }
  rowSums[rowSumsOffset + index] = sum;
  s1Partials[s1PartialsOffset + index] = s1;
  asymmetric[asymmetricOffset + index] = unmatched;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rows-final`,
        operation: OPERATION,
        variant: 'rows-final',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'rowSums', view: rowSums, type: 'f32', access: 'read'},
          {name: 'columnSums', view: columnSums, type: 'f32', access: 'read'},
          {name: 's2Partials', view: s2Partials, type: 'f32', access: 'read_write'},
          {name: 'cardinality', view: cardinality, type: 'u32', access: 'read_write'},
          {name: 'isolates', view: isolates, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: `let count = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];
  let total = rowSums[rowSumsOffset + index] + columnSums[columnSumsOffset + index];
  s2Partials[s2PartialsOffset + index] = total * total;
  cardinality[cardinalityOffset + index] = count;
  isolates[isolatesOffset + index] = select(0u, 1u, count == 0u);`
      })
    ];
    // The seven global reductions (S0, S1, S2, asymmetric slots, isolates, minimum and maximum
    // cardinality) share one blocked pass over the per-row values and one combining workgroup,
    // instead of seven separate reduction chains. Fixed-shape trees keep the sums reproducible.
    const blockCount = Math.ceil(rows / SUMMARY_BLOCK_ROWS);
    const floatPartials = view('float-partials', 'float32', 3 * blockCount);
    const countPartials = view('count-partials', 'uint32', 4 * blockCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-blocks`,
        operation: OPERATION,
        variant: 'blocks',
        bindings: [
          {name: 'rowSums', view: rowSums, type: 'f32', access: 'read'},
          {name: 's1Partials', view: s1Partials, type: 'f32', access: 'read'},
          {name: 's2Partials', view: s2Partials, type: 'f32', access: 'read'},
          {name: 'asymmetric', view: asymmetric, type: 'u32', access: 'read'},
          {name: 'isolates', view: isolates, type: 'u32', access: 'read'},
          {name: 'cardinality', view: cardinality, type: 'u32', access: 'read'},
          {name: 'floatPartials', view: floatPartials, type: 'f32', access: 'read_write'},
          {name: 'countPartials', view: countPartials, type: 'u32', access: 'read_write'}
        ],
        workgroupSize: SUMMARY_WORKGROUP_SIZE,
        invocationCount: blockCount * SUMMARY_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${constants}
const BLOCK_ROWS: u32 = ${SUMMARY_BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${SUMMARY_WORKGROUP_SIZE}u;
var<workgroup> partialS0: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialS1: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialS2: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialAsymmetric: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialIsolates: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialMinimum: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialMaximum: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `let block = index / WORKGROUP_SIZE;
  let local = localInvocationIndex;
  let isInRange = index < INVOCATION_COUNT;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var asymmetricCount = 0u;
  var isolateCount = 0u;
  var minimum = 0xffffffffu;
  var maximum = 0u;
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROWS);
    for (var row = block * BLOCK_ROWS + local; row < end; row += WORKGROUP_SIZE) {
      s0 += rowSums[rowSumsOffset + row];
      s1 += s1Partials[s1PartialsOffset + row];
      s2 += s2Partials[s2PartialsOffset + row];
      asymmetricCount += asymmetric[asymmetricOffset + row];
      isolateCount += isolates[isolatesOffset + row];
      let degree = cardinality[cardinalityOffset + row];
      minimum = min(minimum, degree);
      maximum = max(maximum, degree);
    }
  }
  partialS0[local] = s0;
  partialS1[local] = s1;
  partialS2[local] = s2;
  partialAsymmetric[local] = asymmetricCount;
  partialIsolates[local] = isolateCount;
  partialMinimum[local] = minimum;
  partialMaximum[local] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      partialS0[local] += partialS0[local + stride];
      partialS1[local] += partialS1[local + stride];
      partialS2[local] += partialS2[local + stride];
      partialAsymmetric[local] += partialAsymmetric[local + stride];
      partialIsolates[local] += partialIsolates[local + stride];
      partialMinimum[local] = min(partialMinimum[local], partialMinimum[local + stride]);
      partialMaximum[local] = max(partialMaximum[local], partialMaximum[local + stride]);
    }
    workgroupBarrier();
  }
  if (isInRange && local == 0u) {
    floatPartials[floatPartialsOffset + block] = partialS0[0];
    floatPartials[floatPartialsOffset + BLOCK_COUNT + block] = partialS1[0];
    floatPartials[floatPartialsOffset + 2u * BLOCK_COUNT + block] = partialS2[0];
    countPartials[countPartialsOffset + block] = partialAsymmetric[0];
    countPartials[countPartialsOffset + BLOCK_COUNT + block] = partialIsolates[0];
    countPartials[countPartialsOffset + 2u * BLOCK_COUNT + block] = partialMinimum[0];
    countPartials[countPartialsOffset + 3u * BLOCK_COUNT + block] = partialMaximum[0];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-totals`,
        operation: OPERATION,
        variant: 'totals',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'floatPartials', view: floatPartials, type: 'f32', access: 'read'},
          {name: 'countPartials', view: countPartials, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        workgroupSize: SUMMARY_WORKGROUP_SIZE,
        invocationCount: SUMMARY_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${constants}
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${SUMMARY_WORKGROUP_SIZE}u;
var<workgroup> partialS0: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialS1: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialS2: array<f32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialAsymmetric: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialIsolates: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialMinimum: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;
var<workgroup> partialMaximum: array<u32, ${SUMMARY_WORKGROUP_SIZE}>;`,
        body: `let local = localInvocationIndex;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var asymmetricCount = 0u;
  var isolateCount = 0u;
  var minimum = 0xffffffffu;
  var maximum = 0u;
  for (var block = local; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    s0 += floatPartials[floatPartialsOffset + block];
    s1 += floatPartials[floatPartialsOffset + BLOCK_COUNT + block];
    s2 += floatPartials[floatPartialsOffset + 2u * BLOCK_COUNT + block];
    asymmetricCount += countPartials[countPartialsOffset + block];
    isolateCount += countPartials[countPartialsOffset + BLOCK_COUNT + block];
    minimum = min(minimum, countPartials[countPartialsOffset + 2u * BLOCK_COUNT + block]);
    maximum = max(maximum, countPartials[countPartialsOffset + 3u * BLOCK_COUNT + block]);
  }
  partialS0[local] = s0;
  partialS1[local] = s1;
  partialS2[local] = s2;
  partialAsymmetric[local] = asymmetricCount;
  partialIsolates[local] = isolateCount;
  partialMinimum[local] = minimum;
  partialMaximum[local] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      partialS0[local] += partialS0[local + stride];
      partialS1[local] += partialS1[local + stride];
      partialS2[local] += partialS2[local + stride];
      partialAsymmetric[local] += partialAsymmetric[local + stride];
      partialIsolates[local] += partialIsolates[local + stride];
      partialMinimum[local] = min(partialMinimum[local], partialMinimum[local + stride]);
      partialMaximum[local] = max(partialMaximum[local], partialMaximum[local + stride]);
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    statistics[statisticsOffset] = partialS0[0];
    statistics[statisticsOffset + 1u] = 0.5 * partialS1[0];
    statistics[statisticsOffset + 2u] = partialS2[0];
    counts[countsOffset] = offsets[offsetsOffset + ROWS];
    counts[countsOffset + 1u] = partialAsymmetric[0];
    counts[countsOffset + 2u] = partialIsolates[0];
    counts[countsOffset + 3u] = partialMinimum[0];
    counts[countsOffset + 4u] = partialMaximum[0];
  }`
      })
    );
    return nodes;
  }
}
