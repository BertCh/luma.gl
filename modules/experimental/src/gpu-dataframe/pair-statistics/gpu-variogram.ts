// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getPairHistogramNodes,
  getPairHistogramReadWGSL,
  PAIR_HISTOGRAM_LARGE_TERM_SHIFT,
  validatePairHistogramShape
} from './pair-histogram';
import {
  getPairStatisticsInputNodes,
  getPairStatisticsWindowParameterNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs,
  validatePairStatisticsRowCount
} from './pair-statistics-grid';
import {getPairStatisticsUnitVarianceNodes} from './pair-statistics-fixed-point-unit';
import {
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH
} from './variogram-parameters';

import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateGPUStatusPort,
  type GPUOverflowStatusPort
} from '../../analysis-core/contracts/status';

import {
  resolveGPUStudyWindow,
  type GPUStudyWindow
} from '../../analysis-core/contracts/study-window';

const OPERATION = 'GPUVariogram';

/** Pair distances require a distance-compatible planar frame. */
const SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/**
 * Fixed-point full scale of the distance and robust terms: they are scaled to at most `2^24`
 * before rounding (both are bounded by known quantities, `maximumDistance` and `sqrt(range)`).
 */
const FIXED_POINT_SCALE = 16777216;

/**
 * Matheron fixed-point units per unit variance: the typical pair term `(z_i - z_j)^2 / 2` has mean
 * equal to the variance, so a term of one (robust) variance maps to `2^24` units and the rounding
 * unit never depends on the largest value in the data (an outlier or sentinel cannot swamp it).
 */
const SEMIVARIANCE_UNITS_PER_VARIANCE = 16777216;

/** Caller-owned output views of {@link GPUVariogram}. */
export type GPUVariogramOutput = {
  /**
   * Caller-owned Matheron semivariance per sector and lag, `directionCount * lagCount` rows
   * (sector-major). Quiet NaN for empty bins.
   */
  semivariances: GraphDataView<'float32'>;
  /** Optional caller-owned pair count per sector and lag (each unordered pair counted once). */
  pairCounts?: GraphDataView<'uint32'>;
  /** Optional caller-owned mean pair distance per sector and lag. Quiet NaN for empty bins. */
  meanDistances?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned Cressie-Hawkins robust semivariance per sector and lag. Quiet NaN for
   * empty bins.
   */
  robustSemivariances?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned `[n, mean, variance, minimum, maximum]` of the included values
   * (`GPU_VARIOGRAM_STATISTICS_LENGTH` rows; population variance). The variance is the usual
   * sill reference for a stationary field.
   */
  statistics?: GraphDataView<'float32'>;
  /** Optional caller-owned one-row status scalars. */
  status?: {
    /**
     * One row: 1 when the integer sums are incomplete, either because
     * an accumulator carried out of 64 bits (sums beyond `2^40` robust variances for terms below
     * 256, or `2^52` for larger terms, for example an enormous sentinel with very many neighbors), a
     * single term was beyond the clamp (`2^51` robust variances of `(z_i - z_j)^2 / 2`), or a bin pair count reached `2^32 - 1` (`pairCounts` then saturates at
     * `0xffffffff`). 0 otherwise.
     */
    overflow?: GPUOverflowStatusPort['overflow'];
  };
};

/**
 * Properties for {@link GPUVariogram}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `values`, `mask` and
 * `parameters` (bounds, maximum distance, azimuth offset). Compile-time: the row count,
 * `gridSize`, `lagCount`, `directionCount`, and which optional `output` views are present.
 */
export type GPUVariogramProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'variogram'`. */
  id?: string;
  /**
   * Coordinate and metric descriptor of `positions`. Omitted means native planar coordinates.
   * Only planar contexts are supported: pair distances are Euclidean, so a
   * `longitude-latitude` context throws at construction (project positions to a planar frame
   * first).
   * Lambert equal-area frames are rejected because this operator measures distance.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Compile-time study window, default 'rectangle'. Polygon windows are prepared with
   * GPUStudyWindowPreparation from @luma.gl/experimental gpu-spatial-analysis and must be valid
   * (non-self-intersecting, non-overlapping polygons) in the same planar frame. Require
   * bounds: 'study-window' in polygon mode; that setting gives NaN statistics in rectangle mode.
   */
  studyWindow?: GPUStudyWindow;
  /** Packed planar points, one row per observation. At least one and fewer than 2^24 rows. */
  positions: GraphDataView<'float32x2'>;
  /** Packed observed values, one per row. Rows with a non-finite value are excluded. */
  values: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: packed float32 view of at least `GPU_VARIOGRAM_PARAMETER_LENGTH`
   * elements written with `getGPUVariogramParameterValues`. Invalid bounds or a non-positive
   * maximum distance exclude every row.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the neighbor-search cell lattice. Compile-time. Results never
   * depend on it, only speed: each encoding uses cells at least `maximumDistance` wide.
   */
  gridSize: readonly [number, number];
  /** Number of equal-width lag bins over `[0, maximumDistance]`. Compile-time, positive. */
  lagCount: number;
  /**
   * Number of direction sectors over `[0, pi)`. Compile-time. Defaults to 1 (omnidirectional).
   * `lagCount * directionCount` must be at most 512 (409 with `output.robustSemivariances`: the
   * accumulators are limited to 2048 bins x channels).
   */
  directionCount?: number;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned output views. */
  output: GPUVariogramOutput;
};

/**
 * Empirical semivariogram (gstat `variogram`, scikit-gstat `Variogram`) of planar point values,
 * optionally directional, from one deterministic pass over every pair within the per-frame
 * maximum distance.
 *
 * Definition, which the GPU result matches within the fixed-point bound below:
 * - Included rows have a nonzero mask (when given), a finite value and a finite position inside
 *   the per-frame bounds.
 * - Each unordered pair `{i, j}` of included rows with `d_ij <= maximumDistance` (evaluated in
 *   f32 as `sqrt(dx * dx + dy * dy)`) falls in lag `min(floor(d / maximumDistance * lagCount),
 *   lagCount - 1)` and, with `directionCount > 1`, in sector
 *   `min(floor(a / pi * directionCount), directionCount - 1)` where `a` is
 *   `atan2(dy, dx) - azimuthOffset` reduced modulo pi into `[0, pi)`.
 * - Per bin with `N` pairs: `gamma = sum (z_i - z_j)^2 / (2 N)` (Matheron), mean distance
 *   `sum d / N`, and the Cressie-Hawkins estimator
 *   `gamma_CH = (sum |z_i - z_j|^(1/2) / N)^4 / (2 (0.457 + 0.494 / N))`.
 *
 * Determinism and precision: every pair term is accumulated as an exact 64-bit integer sum of a
 * fixed-point quantization (see the shared pair histogram), so results are bitwise reproducible
 * and independent of scheduling.
 * - Matheron `gamma`: the unit is derived from the typical term, never the maximum. A term
 *   `(z_i - z_j)^2 / 2` is multiplied by `2^24 / u` and rounded to the nearest integer, where `u`
 *   is the robust variance `(IQR / 1.349)^2` of the included values (exact quartiles of the
 *   masked finite values), clamped to `[variance * 2^-32, variance]` so that it is never coarser
 *   than the population variance (`variance * 2^-12` when the IQR is 0, i.e. over half of the
 *   values are tied). The absolute error of a
 *   bin mean is at most `u * 2^-25` (at most `variance * 2^-25`), plus f32 rounding of the term
 *   (relative `2^-24`). That is below the former `(max - min)^2 * 2^-26` bound for every data set
 *   (the variance is at most a quarter of the squared range), and unlike the plain variance (which
 *   grows as `outlier^2 / n`) it does not degrade when one outlier or sentinel of any size
 *   inflates the range: the interquartile range ignores a quarter of contaminated rows. A rare
 *   term above `2^32 - 256` units (about 256 variances, a difference beyond 22 standard
 *   deviations) is added exactly as a 64-bit amount in a second channel at `2^12` units, with a
 *   relative rounding error of at most `2^-21` (a bin's contribution from such terms is accurate
 *   to relative `5e-7`). The worst-case relative error of `gamma` in a bin is therefore
 *   `max(u * 2^-25 / gamma, 5e-7 + 2^-24)`.
 * - Capacity (in units of `u`): the first channel holds sums up to `2^40` and the second up to
 *   `2^52`; a term is clamped at `2^51`. With `u` at the variance of typical data that is about
 *   `1e12` pairs of typical terms. A bin beyond that, or a term beyond the clamp, raises the
 *   `overflow` status instead of wrapping.
 * - Distances: unit `maximumDistance * 2^-24`, absolute error at most `maximumDistance * 2^-25`.
 * - Cressie-Hawkins: unit `sqrt(max - min) * 2^-24` for `|z_i - z_j|^(1/2)`, which grows only as
 *   the square root of the range, so a single outlier of 1e6 variances changes it by 1e3.
 *
 * `pairCounts` saturate at `0xffffffff` (`2^32 - 1`) per bin, and `overflow` is raised when they do.
 *
 * Model fitting is a CPU step: `fitVariogramModel` runs on the CPU over the read-back bins
 * (semivariances, pair counts and mean distances); the GPU only produces the empirical variogram.
 * Because saturated counts would distort its pair weights, check `overflow` before fitting.
 */
export class GPUVariogram implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUVariogramProps;
  /** Resolved spatial context (native planar unless the props say otherwise). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved compile-time rectangle or polygon window. */
  readonly studyWindow: GPUStudyWindow;
  /** Number of direction sectors (1 when omnidirectional). */
  readonly directionCount: number;

  constructor(props: GPUVariogramProps) {
    this.id = props.id ?? 'variogram';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
    this.studyWindow = resolveGPUStudyWindow(id, props.studyWindow, this.spatialContext);
    validatePairStatisticsInputs(id, {
      ...props,
      parameterLength: GPU_VARIOGRAM_PARAMETER_LENGTH
    });
    const {output} = props;
    const directionCount = props.directionCount ?? 1;
    this.directionCount = directionCount;
    validateCapacity(id, 'lagCount', props.lagCount);
    validateCapacity(id, 'directionCount', directionCount);
    const binCount = props.lagCount * directionCount;
    if (binCount > 512) {
      throw new Error(`${id} lagCount * directionCount must be at most 512`);
    }
    // Count, Matheron, distance, the large-term Matheron channel and optionally Cressie-Hawkins.
    validatePairHistogramShape(id, binCount, output.robustSemivariances ? 5 : 4);
    if (output.status?.overflow) {
      validateGPUStatusPort(id, {overflow: output.status?.overflow});
    }
    for (const [name, view] of [
      ['output.semivariances', output.semivariances],
      ['output.meanDistances', output.meanDistances],
      ['output.robustSemivariances', output.robustSemivariances]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length < binCount) {
          throw new Error(`${id} ${name} must hold lagCount * directionCount rows`);
        }
      }
    }
    if (output.pairCounts) {
      validatePackedUint32View(output.pairCounts, `${id} output.pairCounts`);
      if (output.pairCounts.length < binCount) {
        throw new Error(`${id} output.pairCounts must hold lagCount * directionCount rows`);
      }
    }
    if (output.statistics) {
      validatePackedView(output.statistics, ['float32'], `${id} output.statistics`);
      if (output.statistics.length < GPU_VARIOGRAM_STATISTICS_LENGTH) {
        throw new Error(
          `${id} output.statistics must hold ${GPU_VARIOGRAM_STATISTICS_LENGTH} rows`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.semivariances,
        output.pairCounts,
        output.meanDistances,
        output.robustSemivariances,
        output.statistics,
        output.status?.overflow
      ],
      [props.positions, props.values, props.parameters, props.mask]
    );
  }

  /** Returns the variogram nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, directionCount} = this;
    const {positions, parameters: callerParameters, gridSize, lagCount, output} = props;
    validatePairStatisticsRowCount(id, graph.device, positions.length);
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      props.values,
      callerParameters,
      props.mask,
      output.semivariances,
      output.pairCounts,
      output.meanDistances,
      output.robustSemivariances,
      output.statistics,
      output.status?.overflow
    ]);
    const rows = positions.length;
    const binCount = lagCount * directionCount;
    const robust = Boolean(output.robustSemivariances);
    // Channels: 0 pair count, 1 Matheron, 2 distance, [3 robust,] then the large Matheron terms.
    const largeChannel = robust ? 4 : 3;
    const channelCount = largeChannel + 1;
    // Window lookup uses graph identity only, independent of the graph's per-frame parameter type.
    const studyWindow =
      this.studyWindow === 'rectangle'
        ? undefined
        : this.studyWindow.getViews(graph as unknown as GPUCommandGraph<unknown>);
    if (studyWindow) {
      validateGraphViewsBelongToGraph(id, graph, [
        studyWindow.edges,
        studyWindow.summary,
        studyWindow.slabs.params,
        studyWindow.slabs.slabOffsets,
        studyWindow.slabs.slabEntries,
        studyWindow.sampleCells,
        studyWindow.sampleCellCount,
        studyWindow.packed
      ]);
    }
    const windowParameters = studyWindow
      ? getPairStatisticsWindowParameterNodes(graph, {
          id,
          operation: OPERATION,
          parameters: callerParameters,
          parameterLength: callerParameters.length,
          window: studyWindow
        })
      : undefined;
    const parameters = windowParameters?.parameters ?? callerParameters;
    const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      values: props.values,
      mask: props.mask,
      studyWindow
    });
    const statistics = inputs.statistics!;
    const nodes = [...(windowParameters?.nodes ?? []), ...inputs.nodes];
    const unit = getPairStatisticsUnitVarianceNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      values: props.values,
      mask: props.mask,
      statistics
    });
    nodes.push(...unit.nodes);
    const unitVariance = unit.unitVariance;
    // Scales derive from the GPU moments and parameters, so the pair and finishing kernels compute
    // bitwise identical values.
    const scaleWGSL = /* wgsl */ `
const MOMENTS: u32 = ${rows}u;
const LAG_COUNT: u32 = ${lagCount}u;
const DIRECTION_COUNT: u32 = ${directionCount}u;
const FIXED_POINT_SCALE: f32 = ${FIXED_POINT_SCALE}.0;
const SEMIVARIANCE_UNITS_PER_VARIANCE: f32 = ${SEMIVARIANCE_UNITS_PER_VARIANCE}.0;
const LARGE_TERM_UNIT: f32 = ${2 ** PAIR_HISTOGRAM_LARGE_TERM_SHIFT}.0;
const PI: f32 = 3.14159265358979;

struct VariogramScales {
  semivariance: f32,
  robust: f32,
  distance: f32
}

fn getVariogramScales() -> VariogramScales {
  let valueRange = statistics[statisticsOffset + MOMENTS + 4u] - statistics[statisticsOffset + MOMENTS + 3u];
  var scales: VariogramScales;
  // The Matheron unit comes from the typical term (the robust variance), not from the value range.
  let unitVariance = unitVarianceView[unitVarianceViewOffset];
  scales.semivariance = select(0.0, SEMIVARIANCE_UNITS_PER_VARIANCE / unitVariance, unitVariance > 0.0);
  scales.robust = select(0.0, FIXED_POINT_SCALE / sqrt(valueRange), valueRange > 0.0);
  scales.distance = FIXED_POINT_SCALE / parameters[parametersOffset + 4u];
  return scales;
}`;
    const histogram = getPairHistogramNodes<Parameters>(graph, {
      id: `${id}-histogram`,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      sortedRows: inputs.sortedRows,
      cellOffsets: inputs.cellOffsets,
      slotCount: binCount,
      channelCount,
      pairOrder: 'unordered',
      extraBindings: [
        {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
        {name: 'unitVarianceView', view: unitVariance, type: 'f32', access: 'read'}
      ],
      declarations: scaleWGSL,
      focusPrologue: `let scales = getVariogramScales();
    let focusValue = statistics[statisticsOffset + focus];
    let azimuthOffset = readParameter(5u);`,
      pairAction: `let lag = min(u32(pairDistance / lattice.maximumDistance * f32(LAG_COUNT)), LAG_COUNT - 1u);
            var sector = 0u;
            if (DIRECTION_COUNT > 1u) {
              let angle = atan2(deltaY, deltaX) - azimuthOffset;
              let reduced = angle - floor(angle / PI) * PI;
              sector = min(u32(max(reduced, 0.0) / PI * f32(DIRECTION_COUNT)), DIRECTION_COUNT - 1u);
            }
            let bin = sector * LAG_COUNT + lag;
            let difference = statistics[statisticsOffset + neighbor] - focusValue;
            accumulate(bin, 0u, 1u);
            accumulateScaled(bin, 1u, ${largeChannel}u, 0.5 * difference * difference * scales.semivariance);
            accumulate(bin, 2u, quantizePairAmount(pairDistance * scales.distance));
            ${robust ? 'accumulate(bin, 3u, quantizePairAmount(sqrt(abs(difference)) * scales.robust));' : ''}`
    });
    nodes.push(...histogram.nodes);

    const finishBindings: WGSLKernelBinding[] = [
      {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
      {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
      {name: 'unitVarianceView', view: unitVariance, type: 'f32', access: 'read'},
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'semivariances', view: output.semivariances, type: 'f32', access: 'read_write'}
    ];
    if (output.pairCounts) {
      finishBindings.push({
        name: 'pairCounts',
        view: output.pairCounts,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (output.meanDistances) {
      finishBindings.push({
        name: 'meanDistances',
        view: output.meanDistances,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.robustSemivariances) {
      finishBindings.push({
        name: 'robustSemivariances',
        view: output.robustSemivariances,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: 'finish',
        bindings: finishBindings,
        invocationCount: binCount,
        declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${scaleWGSL}
${getPairHistogramReadWGSL('accumulators', channelCount, '', binCount)}`,
        body: `let scales = getVariogramScales();
  let pairCount = readPairAccumulator(index, 0u);
  let hasPairs = pairCount > 0.0;
  let nan = getQuietNaN(index);
  // The large channel holds terms in units of LARGE_TERM_UNIT fixed-point units.
  let semivarianceSum = readPairAccumulator(index, 1u) + readPairAccumulator(index, ${largeChannel}u) * LARGE_TERM_UNIT;
  // A zero variance means every difference is 0 (scale 0, sum 0).
  let semivariance = select(0.0, semivarianceSum / scales.semivariance / pairCount, scales.semivariance > 0.0);
  semivariances[semivariancesOffset + index] = select(nan, semivariance, hasPairs);
  ${output.pairCounts ? 'pairCounts[pairCountsOffset + index] = readPairCount(index, 0u);' : ''}
  ${
    output.meanDistances
      ? `meanDistances[meanDistancesOffset + index] =
    select(nan, readPairAccumulator(index, 2u) / scales.distance / pairCount, hasPairs);`
      : ''
  }
  ${
    robust
      ? `let robustMean = select(0.0, readPairAccumulator(index, 3u) / scales.robust / pairCount, scales.robust > 0.0);
  let robustSquare = robustMean * robustMean;
  let robustValue = robustSquare * robustSquare / (2.0 * (0.457 + 0.494 / pairCount));
  robustSemivariances[robustSemivariancesOffset + index] = select(nan, robustValue, hasPairs);`
      : ''
  }`
      })
    );

    if (output.status?.overflow) {
      // One row: the histogram overflow word, or any bin whose pair count left the low word.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-overflow`,
          operation: OPERATION,
          variant: 'overflow',
          bindings: [
            {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
            {name: 'overflow', view: output.status?.overflow, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: getPairHistogramReadWGSL('accumulators', channelCount, '', binCount),
          body: `var flagged = readPairOverflow();
  for (var bin = 0u; bin < ${binCount}u; bin++) {
    flagged = flagged || readPairCount(bin, 0u) == 0xffffffffu;
  }
  overflow[overflowOffset] = select(0u, 1u, flagged);`
        })
      );
    }

    if (output.statistics) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-statistics`,
          operation: OPERATION,
          variant: 'statistics',
          bindings: [
            {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
            {name: 'statisticsOut', view: output.statistics, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: `const MOMENTS: u32 = ${rows}u;`,
          body: `let count = statistics[statisticsOffset + MOMENTS];
  statisticsOut[statisticsOutOffset] = count;
  statisticsOut[statisticsOutOffset + 1u] = statistics[statisticsOffset + MOMENTS + 1u];
  statisticsOut[statisticsOutOffset + 2u] = statistics[statisticsOffset + MOMENTS + 2u] / count;
  statisticsOut[statisticsOutOffset + 3u] = statistics[statisticsOffset + MOMENTS + 3u];
  statisticsOut[statisticsOutOffset + 4u] = statistics[statisticsOffset + MOMENTS + 4u];`
        })
      );
    }
    return nodes;
  }
}
