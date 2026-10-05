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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getPairHistogramNodes, validatePairHistogramShape} from './pair-histogram';
import {
  getPairStatisticsInputNodes,
  getPairStatisticsTotalSumNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs
} from './pair-statistics-grid';
import {
  GPU_SPATIAL_CORRELOGRAM_NO_BAND,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH
} from './spatial-correlogram-parameters';

const OPERATION = 'GPUSpatialCorrelogram';

/** Largest band count: each focus row keeps one private neighbor counter per band. */
const MAXIMUM_BAND_COUNT = 64;

/** Fixed-point full scale of one pair product. */
const FIXED_POINT_SCALE = 16777216;

/** Accumulator channels: ordered pairs, positive and negative products, and `k^2` in 12-bit limbs. */
const CHANNEL_COUNT = 6;

/**
 * A variance below this fraction of the magnitude of its summed terms is reported as undefined:
 * f32 cancellation dominates it.
 */
const VARIANCE_CANCELLATION = 1e-4;

/** Floats per band in the internal result rows: `[I, E[I], Var[I], z]`. */
const BAND_RESULT_LENGTH = 4;

/**
 * Distance bands of {@link GPUSpatialCorrelogram}: `'cumulative'` bands hold every pair up to their
 * upper distance (PySAL `DistanceBand` thresholds, ArcGIS Incremental Spatial Autocorrelation);
 * `'annulus'` bands hold pairs between the previous and their own upper distance.
 */
export type GPUSpatialCorrelogramBandMode = 'cumulative' | 'annulus';

/**
 * Properties for {@link GPUSpatialCorrelogram}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `values`, `mask` and
 * `parameters` (bounds, maximum distance, variance assumption). Compile-time: the row count,
 * `gridSize`, `bandCount`, `bandMode`, and which optional outputs are present.
 */
export type GPUSpatialCorrelogramProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-correlogram'`. */
  id?: string;
  /** Packed planar points, one row per observation. At least one and fewer than 2^24 rows. */
  positions: GraphDataView<'float32x2'>;
  /** Packed analysis values, one per row. Rows with a non-finite value are excluded. */
  values: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: packed float32 view of at least
   * `GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH` elements written with
   * `getGPUSpatialCorrelogramParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the neighbor-search cell lattice. Compile-time. Results never
   * depend on it, only speed.
   */
  gridSize: readonly [number, number];
  /** Number of distance bands, 1 to 64. Compile-time. */
  bandCount: number;
  /** Band layout. Compile-time. Defaults to `'cumulative'`. */
  bandMode?: GPUSpatialCorrelogramBandMode;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned global Moran's I per band (`bandCount` rows). Quiet NaN when undefined. */
  moransI: GraphDataView<'float32'>;
  /** Optional caller-owned z-score per band. Quiet NaN when undefined. */
  zScores?: GraphDataView<'float32'>;
  /** Optional caller-owned two-sided normal p-value per band. Quiet NaN when undefined. */
  pValues?: GraphDataView<'float32'>;
  /** Optional caller-owned expected I, `-1 / (n - 1)`, per band. */
  expectedI?: GraphDataView<'float32'>;
  /** Optional caller-owned analytic variance of I per band. Quiet NaN when undefined. */
  varianceI?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned number of unordered neighbor pairs per band (saturates at
   * `0xffffffff`).
   */
  pairCounts?: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned `[firstPeakBand, maximumBand]` (2 rows): the smallest band whose finite
   * z-score is strictly greater than both neighboring finite z-scores (ArcGIS "first peak"; the
   * first and last band are never peaks), and the band of the largest finite z-score (smallest band
   * on ties). `GPU_SPATIAL_CORRELOGRAM_NO_BAND` when none.
   */
  peakBands?: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned `[n, mean, variance, sumOfSquares, kurtosis]` of the included values
   * (`GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH` rows; population variance, `kurtosis =
   * n * sum c^4 / (sum c^2)^2`).
   */
  statistics?: GraphDataView<'float32'>;
};

/**
 * Global Moran's I at a ladder of distance bands (spatial correlogram), with analytic z-scores, from
 * one deterministic pass over every pair within the per-frame maximum distance.
 *
 * Definition, which the GPU result matches within f32 rounding and the fixed-point bound below:
 * - Included rows have a nonzero mask (when given), a finite value and a finite position inside
 *   the per-frame bounds; `n` is their count, `c = x - mean`, `M2 = sum c^2`.
 * - Band `b` has upper distance `u_b = maximumDistance * (b + 1) / bandCount`. A pair at f32
 *   distance `d` falls in annulus `min(floor(d / maximumDistance * bandCount), bandCount - 1)`;
 *   cumulative bands hold every annulus up to their own. Weights are binary and symmetric,
 *   `w_ii = 0`, and `k_i` is the number of band neighbors of row `i`.
 * - `S0 = sum_{i != j} w_ij`, `I = (n / S0) * sum_{i != j} w_ij c_i c_j / M2`, `E[I] = -1 / (n - 1)`.
 * - With `S1 = 2 S0` and `S2 = sum_i (2 k_i)^2`, the normality variance is
 *   `(n^2 S1 - n S2 + 3 S0^2) / ((n^2 - 1) S0^2) - E[I]^2` and the randomization variance (with
 *   kurtosis `b2 = n sum c^4 / M2^2`) is
 *   `(n ((n^2 - 3n + 3) S1 - n S2 + 3 S0^2) - b2 ((n^2 - n) S1 - 2n S2 + 6 S0^2)) /
 *   ((n - 1)(n - 2)(n - 3) S0^2) - E[I]^2`, as in esda `Moran`. `z = (I - E[I]) / sqrt(Var)`.
 * - Undefined (quiet NaN): bands with `S0 = 0`, `n < 3` (`n < 4` for randomization), or a
 *   variance at most `1e-4` of the summed magnitude of its terms, where f32 cancellation dominates
 *   (for example a band holding every pair, whose exact variance is 0).
 *
 * Determinism and precision: pair products are accumulated as exact 64-bit integer sums of a
 * fixed-point quantization at `2^24 / max|c|^2`, split into positive and negative parts, so the
 * result is bitwise reproducible. Each product carries at most `max|c|^2 * 2^-25` absolute error,
 * so `|I|` errs by at most `n * max|c|^2 * 2^-25 / M2`. The analytic variance is evaluated in f32
 * and loses relative accuracy when a band holds most pairs.
 */
export class GPUSpatialCorrelogram implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialCorrelogramProps;
  /** Band layout. */
  readonly bandMode: GPUSpatialCorrelogramBandMode;

  constructor(props: GPUSpatialCorrelogramProps) {
    this.id = props.id ?? 'spatial-correlogram';
    this.props = props;
    const id = this.id;
    validatePairStatisticsInputs(id, {
      ...props,
      parameterLength: GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH
    });
    const bandMode = props.bandMode ?? 'cumulative';
    if (bandMode !== 'cumulative' && bandMode !== 'annulus') {
      throw new Error(`${id} bandMode must be cumulative or annulus`);
    }
    this.bandMode = bandMode;
    const {bandCount} = props;
    if (!Number.isInteger(bandCount) || bandCount < 1 || bandCount > MAXIMUM_BAND_COUNT) {
      throw new Error(`${id} bandCount must be an integer from 1 to ${MAXIMUM_BAND_COUNT}`);
    }
    validatePairHistogramShape(id, bandCount, CHANNEL_COUNT);
    for (const [name, view] of [
      ['moransI', props.moransI],
      ['zScores', props.zScores],
      ['pValues', props.pValues],
      ['expectedI', props.expectedI],
      ['varianceI', props.varianceI]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length < bandCount) {
          throw new Error(`${id} ${name} must hold bandCount rows`);
        }
      }
    }
    if (props.pairCounts) {
      validatePackedUint32View(props.pairCounts, `${id} pairCounts`);
      if (props.pairCounts.length < bandCount) {
        throw new Error(`${id} pairCounts must hold bandCount rows`);
      }
    }
    if (props.peakBands) {
      validatePackedUint32View(props.peakBands, `${id} peakBands`);
      if (props.peakBands.length < 2) {
        throw new Error(`${id} peakBands must hold 2 rows`);
      }
    }
    if (props.statistics) {
      validatePackedView(props.statistics, ['float32'], `${id} statistics`);
      if (props.statistics.length < GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH) {
        throw new Error(
          `${id} statistics must hold ${GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH} rows`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.moransI,
        props.zScores,
        props.pValues,
        props.expectedI,
        props.varianceI,
        props.pairCounts,
        props.peakBands,
        props.statistics
      ],
      [props.positions, props.values, props.parameters, props.mask]
    );
  }

  /** Returns the correlogram nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, bandMode} = this;
    const {positions, parameters, gridSize, bandCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      props.values,
      parameters,
      props.mask,
      props.moransI,
      props.zScores,
      props.pValues,
      props.expectedI,
      props.varianceI,
      props.pairCounts,
      props.peakBands,
      props.statistics
    ]);
    const rows = positions.length;
    const cumulative = bandMode === 'cumulative';
    const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      values: props.values,
      mask: props.mask
    });
    const statistics = inputs.statistics!;
    const nodes = inputs.nodes;
    const momentsWGSL = `const MOMENTS: u32 = ${rows}u;
const BAND_COUNT: u32 = ${bandCount}u;
const FIXED_POINT_SCALE: f32 = ${FIXED_POINT_SCALE}.0;

fn getProductScale() -> f32 {
  let maximumAbsolute = statistics[statisticsOffset + MOMENTS + 5u];
  let maximumProduct = maximumAbsolute * maximumAbsolute;
  return select(0.0, FIXED_POINT_SCALE / maximumProduct, maximumProduct > 0.0);
}`;

    // Fourth central moment for the randomization variance: a deterministic tree sum of c^4.
    const fourthTerms = createTransientView(graph, `${id}-fourth-terms`, 'float32', rows);
    const fourthTotal = createTransientView(graph, `${id}-fourth-total`, 'float32', 1);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fourth-terms`,
        operation: OPERATION,
        variant: 'fourth-terms',
        bindings: [
          {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
          {name: 'fourthTerms', view: fourthTerms, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: PAIR_STATISTICS_FLOAT_WGSL,
        body: `let centered = statistics[statisticsOffset + index];
  let square = centered * centered;
  fourthTerms[fourthTermsOffset + index] = select(0.0, square * square, isFiniteFloat(centered));`
      }),
      ...getPairStatisticsTotalSumNodes<Parameters>(graph, {
        id: `${id}-fourth-sum`,
        operation: OPERATION,
        input: fourthTerms,
        output: fourthTotal
      })
    );

    const histogram = getPairHistogramNodes<Parameters>(graph, {
      id: `${id}-histogram`,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      sortedRows: inputs.sortedRows,
      cellOffsets: inputs.cellOffsets,
      slotCount: bandCount,
      channelCount: CHANNEL_COUNT,
      pairOrder: 'ordered',
      extraBindings: [{name: 'statistics', view: statistics, type: 'f32', access: 'read'}],
      declarations: momentsWGSL,
      focusPrologue: `let productScale = getProductScale();
    let focusValue = statistics[statisticsOffset + focus];
    var degree: array<u32, ${bandCount}>;`,
      pairAction: `let band = min(u32(pairDistance / lattice.maximumDistance * f32(BAND_COUNT)), BAND_COUNT - 1u);
            degree[band] += 1u;
            let product = focusValue * statistics[statisticsOffset + neighbor];
            accumulate(band, 0u, 1u);
            accumulate(band, select(2u, 1u, product >= 0.0), quantizePairAmount(abs(product) * productScale));`,
      // k^2 can exceed 32 bits: split k (< 2^24) into 12-bit limbs, k = a * 2^12 + c, and
      // accumulate a^2, 2ac and c^2, weighted 2^24, 2^12 and 1 when read.
      focusEpilogue: `var runningDegree = 0u;
    for (var band = 0u; band < BAND_COUNT; band++) {
      ${cumulative ? 'runningDegree += degree[band];' : 'runningDegree = degree[band];'}
      let high = runningDegree >> 12u;
      let low = runningDegree & 0xfffu;
      accumulate(band, 3u, high * high);
      accumulate(band, 4u, 2u * high * low);
      accumulate(band, 5u, low * low);
    }`
    });
    nodes.push(...histogram.nodes);

    // Finish: one invocation per band. Cumulative bands prefix-sum the annulus pair counts and
    // products exactly in 64-bit integers before converting to f32.
    const bandResults = createTransientView(
      graph,
      `${id}-band-results`,
      'float32',
      bandCount * BAND_RESULT_LENGTH
    );
    const bandCounts = createTransientView(graph, `${id}-band-counts`, 'uint32', bandCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: cumulative ? 'finish-cumulative' : 'finish-annulus',
        bindings: [
          {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'fourthTotal', view: fourthTotal, type: 'f32', access: 'read'},
          {name: 'bandResults', view: bandResults, type: 'f32', access: 'read_write'},
          {name: 'bandCounts', view: bandCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: bandCount,
        declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${momentsWGSL}
const CHANNEL_COUNT: u32 = ${CHANNEL_COUNT}u;
const VARIANCE_CANCELLATION: f32 = ${VARIANCE_CANCELLATION};
const FIRST_BAND_OFFSET: u32 = ${cumulative ? 0 : 1}u;

// Exact 64-bit sum of one channel over annuli [first, last], as (low, high) words.
fn sumChannel(first: u32, last: u32, channel: u32) -> vec2<u32> {
  var low = 0u;
  var high = 0u;
  for (var band = first; band <= last; band++) {
    let word = (band * CHANNEL_COUNT + channel) * 2u;
    let addend = accumulators[accumulatorsOffset + word];
    let previous = low;
    low += addend;
    high += accumulators[accumulatorsOffset + word + 1u] + select(0u, 1u, low < previous);
  }
  return vec2<u32>(low, high);
}

fn toFloat(sum: vec2<u32>) -> f32 {
  return f32(sum.y) * 4294967296.0 + f32(sum.x);
}`,
        body: `let firstBand = select(0u, index, FIRST_BAND_OFFSET == 1u);
  let count = f32(statistics[statisticsOffset + MOMENTS]);
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 2u];
  let productScale = getProductScale();
  let orderedPairs = sumChannel(firstBand, index, 0u);
  // Unordered pairs = ordered / 2, saturated to 32 bits.
  bandCounts[bandCountsOffset + index] =
    select(0xffffffffu, (orderedPairs.y << 31u) | (orderedPairs.x >> 1u), orderedPairs.y < 2u);
  let weightSum = toFloat(orderedPairs);
  let productSum = select(
    0.0,
    (toFloat(sumChannel(firstBand, index, 1u)) - toFloat(sumChannel(firstBand, index, 2u))) / productScale,
    productScale > 0.0
  );
  // Squared neighbor counts are per band (already cumulative when the bands are).
  let degreeSquares = toFloat(sumChannel(index, index, 3u)) * 16777216.0 +
    toFloat(sumChannel(index, index, 4u)) * 4096.0 + toFloat(sumChannel(index, index, 5u));
  let nan = getQuietNaN(index);
  var moransI = nan;
  var expected = nan;
  var variance = nan;
  var zScore = nan;
  let randomization = parameters[parametersOffset + 5u] != 0.0;
  if (count >= 2.0) {
    expected = -1.0 / (count - 1.0);
  }
  if (weightSum > 0.0 && count >= 3.0 && sumOfSquares > 0.0) {
    moransI = count / weightSum * productSum / sumOfSquares;
    let s1 = 2.0 * weightSum;
    let s2 = 4.0 * degreeSquares;
    let weightSquare = weightSum * weightSum;
    let countSquare = count * count;
    // Magnitude of the summed terms: a variance below VARIANCE_CANCELLATION of it is f32
    // cancellation noise (for example a band holding every pair, whose true variance is 0).
    var magnitude = 0.0;
    if (!randomization) {
      let firstTerm = countSquare / (countSquare - 1.0) * (s1 / weightSquare);
      let secondTerm = count / (countSquare - 1.0) * (s2 / weightSquare);
      let thirdTerm = 3.0 / (countSquare - 1.0);
      variance = firstTerm - secondTerm + thirdTerm - expected * expected;
      magnitude = firstTerm + secondTerm + thirdTerm + expected * expected;
    } else if (count >= 4.0) {
      let kurtosis = count * fourthTotal[fourthTotalOffset] / (sumOfSquares * sumOfSquares);
      let denominator = (count - 1.0) * (count - 2.0) * (count - 3.0);
      let firstTerm = count * ((countSquare - 3.0 * count + 3.0) * (s1 / weightSquare) -
        count * (s2 / weightSquare) + 3.0);
      let secondTerm = kurtosis * ((countSquare - count) * (s1 / weightSquare) -
        2.0 * count * (s2 / weightSquare) + 6.0);
      variance = (firstTerm - secondTerm) / denominator - expected * expected;
      magnitude = (abs(firstTerm) + abs(secondTerm)) / denominator + expected * expected;
    }
    if (isFiniteFloat(variance) && variance > VARIANCE_CANCELLATION * magnitude) {
      zScore = (moransI - expected) / sqrt(variance);
    } else {
      variance = nan;
    }
  }
  let base = index * ${BAND_RESULT_LENGTH}u;
  bandResults[bandResultsOffset + base] = moransI;
  bandResults[bandResultsOffset + base + 1u] = expected;
  bandResults[bandResultsOffset + base + 2u] = variance;
  bandResults[bandResultsOffset + base + 3u] = zScore;`
      })
    );

    const publishBindings: WGSLKernelBinding[] = [
      {name: 'bandResults', view: bandResults, type: 'f32', access: 'read'},
      {name: 'moransI', view: props.moransI, type: 'f32', access: 'read_write'}
    ];
    const outputLines = [`moransI[moransIOffset + index] = bandResults[bandResultsOffset + base];`];
    for (const [name, view, slot] of [
      ['expectedI', props.expectedI, 1],
      ['varianceI', props.varianceI, 2],
      ['zScores', props.zScores, 3]
    ] as const) {
      if (view) {
        publishBindings.push({name, view, type: 'f32', access: 'read_write'});
        outputLines.push(
          `${name}[${name}Offset + index] = bandResults[bandResultsOffset + base + ${slot}u];`
        );
      }
    }
    if (props.pValues) {
      publishBindings.push({
        name: 'pValues',
        view: props.pValues,
        type: 'f32',
        access: 'read_write'
      });
      outputLines.push(`let zScore = bandResults[bandResultsOffset + base + 3u];
  pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(zScore), isFiniteFloat(zScore));`);
    }
    if (props.pairCounts) {
      publishBindings.push(
        {name: 'bandCounts', view: bandCounts, type: 'u32', access: 'read'},
        {name: 'pairCounts', view: props.pairCounts, type: 'u32', access: 'read_write'}
      );
      outputLines.push(
        'pairCounts[pairCountsOffset + index] = bandCounts[bandCountsOffset + index];'
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        bindings: publishBindings,
        invocationCount: bandCount,
        declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${TWO_SIDED_P_VALUE_WGSL}`,
        body: `let base = index * ${BAND_RESULT_LENGTH}u;
  ${outputLines.join('\n  ')}`
      })
    );

    if (props.peakBands) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-peaks`,
          operation: OPERATION,
          variant: 'peaks',
          bindings: [
            {name: 'bandResults', view: bandResults, type: 'f32', access: 'read'},
            {name: 'peakBands', view: props.peakBands, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
const BAND_COUNT: u32 = ${bandCount}u;
const NO_BAND: u32 = ${GPU_SPATIAL_CORRELOGRAM_NO_BAND}u;

fn readZ(band: u32) -> f32 {
  return bandResults[bandResultsOffset + band * ${BAND_RESULT_LENGTH}u + 3u];
}`,
          body: `var firstPeak = NO_BAND;
  var maximumBand = NO_BAND;
  var maximumZ = 0.0;
  for (var band = 0u; band < BAND_COUNT; band++) {
    let z = readZ(band);
    if (!isFiniteFloat(z)) {
      continue;
    }
    if (maximumBand == NO_BAND || z > maximumZ) {
      maximumBand = band;
      maximumZ = z;
    }
    if (firstPeak == NO_BAND && band > 0u && band + 1u < BAND_COUNT) {
      let previous = readZ(band - 1u);
      let next = readZ(band + 1u);
      if (isFiniteFloat(previous) && isFiniteFloat(next) && z > previous && z > next) {
        firstPeak = band;
      }
    }
  }
  peakBands[peakBandsOffset] = firstPeak;
  peakBands[peakBandsOffset + 1u] = maximumBand;`
        })
      );
    }

    if (props.statistics) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-statistics`,
          operation: OPERATION,
          variant: 'statistics',
          bindings: [
            {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
            {name: 'fourthTotal', view: fourthTotal, type: 'f32', access: 'read'},
            {name: 'statisticsOut', view: props.statistics, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: `const MOMENTS: u32 = ${rows}u;`,
          body: `let count = statistics[statisticsOffset + MOMENTS];
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 2u];
  statisticsOut[statisticsOutOffset] = count;
  statisticsOut[statisticsOutOffset + 1u] = statistics[statisticsOffset + MOMENTS + 1u];
  statisticsOut[statisticsOutOffset + 2u] = sumOfSquares / count;
  statisticsOut[statisticsOutOffset + 3u] = sumOfSquares;
  statisticsOut[statisticsOutOffset + 4u] = count * fourthTotal[fourthTotalOffset] / (sumOfSquares * sumOfSquares);`
        })
      );
    }
    return nodes;
  }
}

/**
 * Two-sided normal p-value `erfc(|z| / sqrt(2))` with the Numerical Recipes `erfcc` Chebyshev fit
 * (fractional error below 1.2e-7 in exact arithmetic), as in the spatial-autocorrelation recipes.
 */
const TWO_SIDED_P_VALUE_WGSL = /* wgsl */ `
fn getTwoSidedPValue(z: f32) -> f32 {
  let x = abs(z) * 0.70710678118654752;
  let t = 1.0 / (1.0 + 0.5 * x);
  let exponent = -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))));
  return min(t * exp(exponent), 1.0);
}
`;
