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
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {createSegmentSumNode, getSortKeyBits} from '../../utils/sorted-segment-sums';
import {validateGPUSpatialWeights, type GPUSpatialWeights} from '../spatial-weights/index';
import {
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
} from './spatial-autocorrelation-parameters';

/** Rows reduced by one workgroup in the first level of the deterministic global sums. */
const BLOCK_ROWS = 4096;

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

  const validity = createTransientView(graph, `${id}-validity`, 'uint32', rows);
  const valueContributions = createTransientView(graph, `${id}-value-terms`, 'float32', rows);
  const countContributions = createTransientView(graph, `${id}-count-terms`, 'float32', rows);
  const blockOffsets = createTransientView(graph, `${id}-block-offsets`, 'uint32', blockCount + 1);
  const totalOffsets = createTransientView(graph, `${id}-total-offsets`, 'uint32', 2);
  const valuePartials = createTransientView(graph, `${id}-value-partials`, 'float32', blockCount);
  const valueTotal = createTransientView(graph, `${id}-value-total`, 'float32', 1);
  const countPartials = createTransientView(graph, `${id}-count-partials`, 'float32', blockCount);
  const countTotal = createTransientView(graph, `${id}-count-total`, 'float32', 1);
  const squareContributions = createTransientView(graph, `${id}-square-terms`, 'float32', rows);
  const squarePartials = createTransientView(graph, `${id}-square-partials`, 'float32', blockCount);
  const squareTotal = createTransientView(graph, `${id}-square-total`, 'float32', 1);
  const statistics = createTransientView(graph, `${id}-statistics`, 'float32', rows + 4);

  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-validity`,
      operation,
      variant: 'validity',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'validity', view: validity, type: 'u32', access: 'read_write'},
        {name: 'valueTerms', view: valueContributions, type: 'f32', access: 'read_write'},
        {name: 'countTerms', view: countContributions, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
      body: `let value = values[valuesOffset + index];
  let included = ${mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  let valid = included && isFiniteFloat(value);
  validity[validityOffset + index] = select(0u, 1u, valid);
  valueTerms[valueTermsOffset + index] = select(0.0, value, valid);
  countTerms[countTermsOffset + index] = select(0.0, 1.0, valid);`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-block-offsets`,
      operation,
      variant: 'block-offsets',
      bindings: [
        {name: 'blockOffsets', view: blockOffsets, type: 'u32', access: 'read_write'},
        {name: 'totalOffsets', view: totalOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: blockCount + 1,
      declarations: `const ROW_COUNT: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;`,
      body: `blockOffsets[blockOffsetsOffset + index] = min(index * BLOCK_ROWS, ROW_COUNT);
  if (index == 0u) {
    totalOffsets[totalOffsetsOffset] = 0u;
    totalOffsets[totalOffsetsOffset + 1u] = BLOCK_COUNT;
  }`
    }),
    ...getTotalSumNodes<Parameters>(graph, {
      id: `${id}-value-sum`,
      operation,
      input: valueContributions,
      partials: valuePartials,
      output: valueTotal,
      blockOffsets,
      totalOffsets
    }),
    ...getTotalSumNodes<Parameters>(graph, {
      id: `${id}-count-sum`,
      operation,
      input: countContributions,
      partials: countPartials,
      output: countTotal,
      blockOffsets,
      totalOffsets
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-mean`,
      operation,
      variant: 'moments-mean',
      bindings: [
        {name: 'countTotal', view: countTotal, type: 'f32', access: 'read'},
        {name: 'valueTotal', view: valueTotal, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;`,
      body: `var count = countTotal[countTotalOffset];
  var mean = select(getQuietNaN(index), valueTotal[valueTotalOffset] / count, count >= 1.0);
  if (readParameter(1u) != 0.0) {
    count = readParameter(2u);
    mean = readParameter(3u);
  }
  statistics[statisticsOffset + MOMENTS] = count;
  statistics[statisticsOffset + MOMENTS + 1u] = mean;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-center`,
      operation,
      variant: 'center',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
        {name: 'squareTerms', view: squareContributions, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: `const MOMENTS: u32 = ${rows}u;`,
      body: `let valid = validity[validityOffset + index] != 0u;
  let centered = values[valuesOffset + index] - statistics[statisticsOffset + MOMENTS + 1u];
  // Excluded rows store quiet NaN, which every later kernel reads as "not a focus row".
  statistics[statisticsOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), centered, valid);
  squareTerms[squareTermsOffset + index] = select(0.0, centered * centered, valid);`
    }),
    ...getTotalSumNodes<Parameters>(graph, {
      id: `${id}-square-sum`,
      operation,
      input: squareContributions,
      partials: squarePartials,
      output: squareTotal,
      blockOffsets,
      totalOffsets
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-variance`,
      operation,
      variant: 'moments-variance',
      bindings: [
        {name: 'squareTotal', view: squareTotal, type: 'f32', access: 'read'},
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
      invocationCount: 1,
      declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;`,
      body: `let count = statistics[statisticsOffset + MOMENTS];
  let mean = statistics[statisticsOffset + MOMENTS + 1u];
  var sumOfSquares = squareTotal[squareTotalOffset];
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
  }`
    })
  );
  return {nodes, statistics};
}

/** Two fixed-order tree-sum levels: one workgroup per block of rows, then one over the blocks. */
function getTotalSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    input: GraphDataView<'float32'>;
    partials: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
    blockOffsets: GraphDataView<'uint32'>;
    totalOffsets: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  return [
    createSegmentSumNode<Parameters>(graph, {
      id: `${props.id}-blocks`,
      operation: props.operation,
      segmentCount: props.partials.length,
      input: props.input,
      segmentOffsets: props.blockOffsets,
      output: props.partials
    }),
    createSegmentSumNode<Parameters>(graph, {
      id: `${props.id}-total`,
      operation: props.operation,
      segmentCount: 1,
      input: props.partials,
      segmentOffsets: props.totalOffsets,
      output: props.output
    })
  ];
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
