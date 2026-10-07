// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {validateGPUSpatialWeights, type GPUSpatialWeights} from '../spatial-weights/index';
import {
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
} from './spatial-autocorrelation-parameters';

/** Rows reduced by one workgroup in the first level of the deterministic global sums. */
const BLOCK_ROWS = 4096;

/** Threads of the moment-sum workgroups. */
const MOMENT_WORKGROUP_SIZE = 256;

/** Key of a non-finite z-score in the false-discovery-rate sort; sorts after every finite one. */
const INVALID_P_VALUE_KEY = 0x7f800001;

/** Inputs shared by both spatial-autocorrelation contributors. @internal */
export type SpatialAutocorrelationInputProps = {
  id: string;
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  globalStatistics?: GraphDataView<'float32'>;
};

/** Moment views produced by {@link getSpatialAutocorrelationInputNodes}. @internal */
export type SpatialAutocorrelationInputs<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /**
   * `rows + 4` floats: centered values `x - mean` (quiet NaN for excluded rows) followed by the
   * moments `[n, mean, variance, sumOfSquares]`.
   */
  statistics: GraphDataView<'float32'>;
};

/**
 * Validates the props shared by both contributors and returns the row count.
 *
 * @internal
 */
export function validateSpatialAutocorrelationInputs(
  props: SpatialAutocorrelationInputProps
): number {
  const {id} = props;
  const rows = validateGPUSpatialWeights(id, props.weights);
  validatePackedView(props.values, ['float32'], `${id} values`);
  validatePackedView(props.parameters, ['float32'], `${id} parameters`);
  if (rows >= 2 ** 24) {
    // Counts and ranks are compared as f32 inside the kernels.
    throw new Error(`${id} weights must hold fewer than 2^24 rows`);
  }
  if (props.values.length !== rows) {
    throw new Error(`${id} values length must equal the weights row count`);
  }
  if (props.parameters.length < GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH) {
    throw new Error(
      `${id} parameters must hold ${GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} float32 values`
    );
  }
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) {
      throw new Error(`${id} mask length must equal the weights row count`);
    }
  }
  if (props.globalStatistics) {
    validatePackedView(props.globalStatistics, ['float32'], `${id} globalStatistics`);
    if (props.globalStatistics.length < GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH) {
      throw new Error(
        `${id} globalStatistics must hold ${GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH} float32 values`
      );
    }
  }
  return rows;
}

/**
 * Parameter-free WGSL helpers: finiteness test, runtime quiet NaN, and the two-sided normal
 * p-value. Included by {@link getSpatialAutocorrelationSharedWGSL}.
 *
 * @internal
 */
export const SPATIAL_AUTOCORRELATION_FLOAT_WGSL = /* wgsl */ `
fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// A runtime operand keeps the quiet-NaN bit pattern out of constant evaluation.
fn getQuietNaN(seed: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}

// Two-sided normal p-value erfc(|z| / sqrt(2)) with the Numerical Recipes erfcc Chebyshev fit
// (fractional error below 1.2e-7 in exact arithmetic), so tiny p-values keep relative accuracy.
fn getTwoSidedPValue(z: f32) -> f32 {
  let x = abs(z) * 0.70710678118654752;
  let t = 1.0 / (1.0 + 0.5 * x);
  let exponent = -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))));
  return min(t * exp(exponent), 1.0);
}
`;

/**
 * WGSL shared by every kernel that reads the parameters: `readParameter`, quiet NaN, finiteness
 * and the two-sided normal p-value. Kernels that include it must bind `parameters` as
 * `array<f32>`.
 *
 * @internal
 */
export function getSpatialAutocorrelationSharedWGSL(): string {
  return /* wgsl */ `
fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
`;
}

/**
 * WGSL statements that visit, in CSR slot order (ascending neighbor ID), every neighbor of row
 * `index` that is an included row other than `index` itself, and run `action` with `neighbor`
 * (row index) and `weight` (`w_ij`) in scope. Requires bindings `offsets`, `neighbors`,
 * `weights` and `statistics` (whose first `rows` entries are NaN for excluded rows) and the
 * `ROWS` constant.
 *
 * The order is fixed by the weights, so per-row sums are reproducible.
 *
 * @internal
 */
export function getSpatialWeightsNeighborLoopWGSL(action: string): string {
  return /* wgsl */ `
  let slotBegin = offsets[offsetsOffset + index];
  let slotEnd = offsets[offsetsOffset + index + 1u];
  for (var slot = slotBegin; slot < slotEnd; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS || neighbor == index || !isFiniteFloat(statistics[statisticsOffset + neighbor])) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    ${action}
  }`;
}

/**
 * Builds the shared front end of both contributors: per-row validity and the global moments.
 *
 * Moments come from fixed-order two-level workgroup tree sums over the rows (no float atomics):
 * first the count and the mean, then the sum of squared centered values, so the variance never
 * subtracts two large numbers. Fixed moments in the parameters replace both.
 *
 * @internal
 */
export function getSpatialAutocorrelationInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SpatialAutocorrelationInputProps & {operation: string}
): SpatialAutocorrelationInputs<Parameters> {
  const {id, operation, values, parameters, mask} = props;
  const rows = values.length;
  const blockCount = Math.ceil(rows / BLOCK_ROWS);
  const sharedWGSL = getSpatialAutocorrelationSharedWGSL();
  const nodes: GPUCommandNode<Parameters>[] = [];

  const valuePartials = createTransientView(graph, `${id}-value-partials`, 'float32', blockCount);
  const countPartials = createTransientView(graph, `${id}-count-partials`, 'uint32', blockCount);
  const squarePartials = createTransientView(graph, `${id}-square-partials`, 'float32', blockCount);
  const statistics = createTransientView(graph, `${id}-statistics`, 'float32', rows + 4);
  const maskBindings: WGSLKernelBinding[] = mask
    ? [{name: 'mask', view: mask, type: 'u32', access: 'read'}]
    : [];
  const blockDeclarations = `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
const MOMENTS: u32 = ${rows}u;
const ROW_COUNT: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const WORKGROUP_SIZE: u32 = ${MOMENT_WORKGROUP_SIZE}u;
fn isRowValid(row: u32) -> bool {
  return ${mask ? 'mask[maskOffset + row] != 0u && ' : ''}isFiniteFloat(values[valuesOffset + row]);
}`;
  // Each level-1 workgroup reduces one block of rows: thread t sums rows block + t, + 256, ... and a
  // fixed binary tree combines the threads, so the order depends only on the shape and the sums
  // are bitwise reproducible. Validity is evaluated in place (no per-row validity or term
  // buffers), and the level-2 kernels also finish the moments, so the front end is four kernels.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-blocks`,
      operation,
      variant: 'moments-blocks',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...maskBindings,
        {name: 'valuePartials', view: valuePartials, type: 'f32', access: 'read_write'},
        {name: 'countPartials', view: countPartials, type: 'u32', access: 'read_write'}
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${blockDeclarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialCounts: array<u32, ${MOMENT_WORKGROUP_SIZE}>;`,
      // No early return: every invocation of a workgroup must reach the barriers.
      body: `let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  var count = 0u;
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      if (isRowValid(row)) {
        sum += values[valuesOffset + row];
        count++;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    valuePartials[valuePartialsOffset + block] = partialSums[0];
    countPartials[countPartialsOffset + block] = partialCounts[0];
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-mean`,
      operation,
      variant: 'moments-mean',
      bindings: [
        {name: 'valuePartials', view: valuePartials, type: 'f32', access: 'read'},
        {name: 'countPartials', view: countPartials, type: 'u32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'}
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${MOMENT_WORKGROUP_SIZE}u;
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialCounts: array<u32, ${MOMENT_WORKGROUP_SIZE}>;`,
      body: `var sum = 0.0;
  var count = 0u;
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    sum += valuePartials[valuePartialsOffset + block];
    count += countPartials[countPartialsOffset + block];
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    var total = f32(partialCounts[0]);
    var mean = select(getQuietNaN(index), partialSums[0] / total, total >= 1.0);
    if (readParameter(1u) != 0.0) {
      total = readParameter(2u);
      mean = readParameter(3u);
    }
    statistics[statisticsOffset + MOMENTS] = total;
    statistics[statisticsOffset + MOMENTS + 1u] = mean;
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-center`,
      operation,
      variant: 'center',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...maskBindings,
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
        {name: 'squarePartials', view: squarePartials, type: 'f32', access: 'read_write'}
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${blockDeclarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
      // Centers each row and reduces the squared deviations of its block in the same pass.
      body: `let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  if (isInRange) {
    let mean = statistics[statisticsOffset + MOMENTS + 1u];
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      let valid = isRowValid(row);
      let centered = values[valuesOffset + row] - mean;
      // Excluded rows store quiet NaN, which every later kernel reads as "not a focus row".
      statistics[statisticsOffset + row] = select(getQuietNaN(row), centered, valid);
      sum += select(0.0, centered * centered, valid);
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    squarePartials[squarePartialsOffset + block] = partialSums[0];
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-variance`,
      operation,
      variant: 'moments-variance',
      bindings: [
        {name: 'squarePartials', view: squarePartials, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
        ...(props.globalStatistics
          ? [
              {
                name: 'globalStatistics',
                view: props.globalStatistics,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${MOMENT_WORKGROUP_SIZE}u;
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
      body: `var sum = 0.0;
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    sum += squarePartials[squarePartialsOffset + block];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let count = statistics[statisticsOffset + MOMENTS];
    let mean = statistics[statisticsOffset + MOMENTS + 1u];
    var sumOfSquares = partialSums[0];
    var variance = select(getQuietNaN(index), sumOfSquares / count, count >= 1.0);
    if (readParameter(1u) != 0.0) {
      variance = readParameter(4u);
      sumOfSquares = variance * count;
    }
    statistics[statisticsOffset + MOMENTS + 2u] = variance;
    statistics[statisticsOffset + MOMENTS + 3u] = sumOfSquares;
    ${
      props.globalStatistics
        ? `globalStatistics[globalStatisticsOffset] = count;
    globalStatistics[globalStatisticsOffset + 1u] = mean;
    globalStatistics[globalStatisticsOffset + 2u] = variance;
    globalStatistics[globalStatisticsOffset + 3u] = sqrt(variance);`
        : ''
    }
  }`
    })
  );
  return {nodes, statistics};
}

/** Views produced by {@link getFalseDiscoveryRateNodes}. @internal */
export type FalseDiscoveryRateResult<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /** Per row: 1-based rank by ascending p-value, or 0 for a non-finite z-score. */
  ranks: GraphDataView<'uint32'>;
  /**
   * `[m, K_0, K_1, ...]`: the number of tested rows `m` and, per level, the largest rank `k`
   * with `p_(k) <= k * alpha / m` (0 when none). A row is significant at a level when
   * `0 < rank <= K`.
   */
  counters: GraphDataView<'uint32'>;
};

/**
 * Benjamini-Hochberg step-up false discovery rate over the two-sided p-values of `zScores`.
 *
 * Rows are sorted by descending `|z|` (ascending p) with a stable integer sort, then each sorted
 * slot tests its own BH inequality and raises the per-level threshold rank with an integer
 * `atomicMax`, so the result is exact and deterministic. Ties never straddle a threshold, so the
 * tie order does not matter. Rows with non-finite z are not tested and do not count toward `m`.
 *
 * @param levelExpressions WGSL f32 expressions of the significance levels, which may read the
 * parameters through `readParameter(slot)`.
 *
 * @internal
 */
export function getFalseDiscoveryRateNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    zScores: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    levelExpressions: readonly string[];
  }
): FalseDiscoveryRateResult<Parameters> {
  const {id, operation, zScores, parameters, levelExpressions} = props;
  const rows = zScores.length;
  const levelCount = levelExpressions.length;
  const keys = createTransientView(graph, `${id}-keys`, 'uint32', rows);
  const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rows);
  const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
  const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', rows);
  const counters = createTransientView(graph, `${id}-counters`, 'uint32', levelCount + 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-keys`,
      operation,
      variant: 'fdr-keys',
      bindings: [
        {name: 'zScores', view: zScores, type: 'f32', access: 'read'},
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      // Non-negative finite floats order like their bits, so subtracting the bits from the +inf
      // pattern sorts larger |z| (smaller p) first.
      body: `let bits = bitcast<u32>(zScores[zScoresOffset + index]) & 0x7fffffffu;
  keys[keysOffset + index] = select(${INVALID_P_VALUE_KEY}u, 0x7f800000u - bits, bits < 0x7f800000u);
  rowIds[rowIdsOffset + index] = index;`
    }),
    ...new GPUSort({
      id: `${id}-sort`,
      keys,
      values: rowIds,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      keyBits: getSortKeyBits(INVALID_P_VALUE_KEY)
    }).getCommandNodes(graph),
    createFillNode<Parameters>(graph, {
      id: `${id}-clear`,
      operation,
      view: counters,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-ranks`,
      operation,
      variant: 'fdr-ranks',
      bindings: [
        {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'ranks', view: ranks, type: 'u32', access: 'read_write'},
        {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: rows,
      body: `let tested = sortedKeys[sortedKeysOffset + index] != ${INVALID_P_VALUE_KEY}u;
  ranks[ranksOffset + sortedRows[sortedRowsOffset + index]] = select(0u, index + 1u, tested);
  if (tested) {
    atomicMax(&counters[countersOffset], index + 1u);
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-thresholds`,
      operation,
      variant: 'fdr-thresholds',
      bindings: [
        {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: getSpatialAutocorrelationSharedWGSL(),
      body: `let testedCount = atomicLoad(&counters[countersOffset]);
  let rank = index + 1u;
  if (rank <= testedCount) {
    let absoluteZ = bitcast<f32>(0x7f800000u - sortedKeys[sortedKeysOffset + index]);
    let pValue = getTwoSidedPValue(absoluteZ);
    let fraction = f32(rank) / f32(testedCount);
    ${levelExpressions
      .map(
        (expression, level) => `if (pValue <= fraction * (${expression})) {
      atomicMax(&counters[countersOffset + ${level + 1}u], rank);
    }`
      )
      .join('\n    ')}
  }`
    })
  ];
  return {nodes, ranks, counters};
}
