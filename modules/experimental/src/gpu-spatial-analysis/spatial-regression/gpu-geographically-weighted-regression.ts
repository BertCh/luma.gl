// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {createGridIndexNodes} from './gwr-grid-index';
import {createGoldenSectionSearchNodes} from './gwr-golden-section-nodes';
import {getGWRFitWGSL} from './gwr-fit-wgsl';
import {getStudentTWGSL} from './gwr-student-t';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_MAXIMUM_SEARCH_ITERATIONS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_REFINEMENT_PASS_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_SIGNIFICANCE_LEVEL,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_REFINEMENT_PASS_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  getGPUGeographicallyWeightedRegressionParameterLength
} from './geographically-weighted-regression-parameters';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';

/** Spatial contexts supported by this distance-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUGeographicallyWeightedRegression';
/** Rows summed in fixed order by one thread of a tile pass. */
const TILE_ROWS = 256;
/** Float32 values per row of the packed local statistics scratch. */
const LOCAL_STRIDE = 6;
/** Float32 values per tile and candidate of the tile sums: RSS, trace, singular, invalid, included. */
const TILE_CANDIDATE_STRIDE = 5;
/** Words of the row-state tail (selection results shared with the later kernels). */
const TAIL_WORDS = 8;
/** Default `maxStorageBufferBindingSize`; bounds the per-candidate scratch and the row state. */
const MAXIMUM_SCRATCH_BYTES = 134217728;
/** Values of the tail words of the row-state buffer. */
const TAIL = {
  SELECTED_INDEX: 0,
  SELECTED_VALUE: 1,
  KERNEL: 2,
  MODE: 3,
  SIGMA_SQUARED: 4,
  CRITICAL_T: 5
} as const;

/** Caller-owned outputs of {@link GPUGeographicallyWeightedRegression}. */
export type GPUGeographicallyWeightedRegressionOutput = {
  /**
   * Local coefficients at the selected bandwidth, row-major `row * (predictorCount + 1) + column`;
   * column 0 is the intercept and column `c + 1` is predictor `c`. NaN for excluded or singular rows.
   */
  coefficients: GraphDataView<'float32'>;
  /**
   * Optional local standard errors of the coefficients, laid out like `coefficients`, as mgwr
   * `bse`: `sqrt(diag(C_i C_i') * sigma2)` with `C_i C_i' = (X'W_iX)^-1 X'W_i^2 X (X'W_iX)^-1` and
   * `sigma2 = RSS / (n - tr(S))` (summary `SIGMA_SQUARED`). NaN for excluded or singular rows. The
   * extra work is one more weighted Gram matrix per location in the same neighborhood pass.
   */
  standardErrors?: GraphDataView<'float32'>;
  /**
   * Optional local t values `coefficient / standardError` (mgwr `tvalues`), laid out like
   * `coefficients`; NaN where the coefficient is undefined or the standard error is not positive.
   */
  tValues?: GraphDataView<'float32'>;
  /**
   * Optional significance mask laid out like `coefficients`: 1 where `|t|` exceeds the summary
   * `CRITICAL_T` (mgwr `filter_tvals` with the adjusted-alpha `critical_tval`), else 0. The
   * adjusted alpha corrects for the number of effective parameters (Da Silva and Fotheringham 2016).
   */
  significant?: GraphDataView<'uint32'>;
  /** Optional local weighted R^2 per row (`1 - weighted RSS / weighted TSS`); NaN when undefined. */
  localR2?: GraphDataView<'float32'>;
  /** Optional fitted value `x_i^T beta_i` per row; NaN for excluded or singular rows. */
  fitted?: GraphDataView<'float32'>;
  /** Optional residual `y_i - fitted_i` per row; NaN for excluded or singular rows. */
  residuals?: GraphDataView<'float32'>;
  /** Optional hat-matrix diagonal `S_ii` per row; NaN for excluded or singular rows. */
  hatDiagonal?: GraphDataView<'float32'>;
  /** Optional per-row status, see `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS`. */
  localStatus?: GraphDataView<'uint32'>;
  /**
   * Optional local condition number per row, as mgwr `GWRResults.local_collinearity` (its
   * `local_CN`): the ratio of the largest to the smallest singular value of the design `[1, X]`
   * with every row scaled by its kernel weight and every column scaled to unit length. Large
   * values (mgwr and the GWR literature flag above 30) mean the local design is nearly collinear.
   * NaN for excluded or singular rows. The extra work is one more pass over each neighborhood.
   */
  localConditionNumber?: GraphDataView<'float32'>;
  /**
   * Optional AICc of each caller-ladder candidate (NaN for invalid or unused candidates),
   * `maximumBandwidthCount` values. The AICc of a candidate with singular rows is taken over its
   * non-singular rows, see `bandwidthSingularRowCounts`. Refined candidates are in `candidateTable`.
   * With `bandwidthSearch: 'golden-section'` these are the first `maximumBandwidthCount` rows of the
   * candidate table, in evaluation order (NaN beyond the evaluations that ran).
   */
  bandwidthScores?: GraphDataView<'float32'>;
  /**
   * Optional number of singular included rows of each caller-ladder candidate,
   * `maximumBandwidthCount` values (0 for unused slots). With `'golden-section'` the first
   * `maximumBandwidthCount` rows of the candidate table, in evaluation order.
   */
  bandwidthSingularRowCounts?: GraphDataView<'uint32'>;
  /**
   * Optional full candidate table: for the caller ladder and then every refinement pass,
   * `maximumBandwidthCount` candidates of `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE`
   * float32 fields each (value, AICc, singular rows, RSS, trace, observations, included rows; NaN
   * for unused slots). Length `(1 + refinementPassCount) * maximumBandwidthCount * stride`.
   *
   * With `bandwidthSearch: 'golden-section'` the table has one row per evaluation, in evaluation
   * order: rows 0 and 1 are the two interior points of the first iteration (`b`, then `d`), row
   * `1 + i` is the single point evaluated for iteration `i + 1` (`i >= 1`); rows of iterations that
   * never ran are NaN. Length `(maximumSearchIterations + 1) * stride`.
   */
  candidateTable?: GraphDataView<'float32'>;
  /**
   * Optional `[selected candidate index, selected ladder value]`. The index counts the caller
   * ladder first (`0 .. maximumBandwidthCount - 1`); refined candidates follow, pass by pass
   * (`maximumBandwidthCount + c` is candidate `c` of the first refinement pass). With
   * `bandwidthSearch: 'golden-section'` the index is the `candidateTable` row of the selection.
   */
  selectedBandwidth?: GraphDataView<'float32'>;
  /**
   * Optional state of the golden-section search after the last iteration, see
   * `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE` (12 float32 values). Only with
   * `bandwidthSearch: 'golden-section'`.
   */
  searchState?: GraphDataView<'float32'>;
  /** Optional global summary, see `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY`. */
  summary?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUGeographicallyWeightedRegression}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `predictors`, `response`,
 * `mask` and `parameters` (kernel, bandwidth mode, ladder or search bounds, significance level).
 * Compile-time: the row count, `predictorCount`, `maximumBandwidthCount`, `maximumNeighborCount`,
 * `spatialContext`, `refinement`, `bandwidthSearch`, `maximumSearchIterations`, `gaussianCutoff`,
 * and which optional views are present.
 */
export type GPUGeographicallyWeightedRegressionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geographically-weighted-regression'`. */
  id?: string;
  /**
   * Packed positions, one per row. Planar (the default): coordinates in the units of the
   * bandwidths. Great-circle `spatialContext`: `(longitude, latitude)` in degrees.
   */
  positions: GraphDataView<'float32x2'>;
  /** Packed float32 predictors, row-major `row * predictorCount + column`, without the intercept. */
  predictors: GraphDataView<'float32'>;
  /** Number of predictor columns, 1 to 11 (the intercept is added). Compile-time. */
  predictorCount: number;
  /** Packed float32 response, one per row. */
  response: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: float32 view of at least
   * `getGPUGeographicallyWeightedRegressionParameterLength(maximumBandwidthCount)` elements written
   * with `getGPUGeographicallyWeightedRegressionParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Ladder capacity, 1 to 32. Compile-time. Default 32. */
  maximumBandwidthCount?: number;
  /** Largest adaptive `k`, 1 to 128. Compile-time. Default 128. */
  maximumNeighborCount?: number;
  /**
   * Coordinate space and metric of the distances. Omitted is planar Euclidean (native units).
   * A planar context with a `frame` is accepted. `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` makes `positions` `(longitude, latitude)` degrees and every distance (kernels,
   * fixed bandwidths, adaptive k-th neighbour) the great-circle distance in the context units on a
   * sphere of `sphereRadius` (default the mean Earth radius in meters), so bandwidths are in meters.
   * The grid index is planar-only and is not built for a great-circle context (every location
   * scans every included row, so the 65,536-row scan limit applies); `indexGridSize` must then be
   * omitted or `false`. Rows with a latitude outside [-90, 90] are excluded.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Bandwidth refinement after the caller ladder. `true` or omitted runs
   * `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_REFINEMENT_PASS_COUNT` (2) passes; a number from
   * 1 to 4 sets the pass count; `false` or 0 keeps the fixed caller ladder. Compile-time. Each pass
   * costs up to one more ladder evaluation (`maximumBandwidthCount` candidates), so the default
   * triples the work of a full ladder; shorten the ladder rather than disabling refinement when
   * cost matters. Ignored (no passes) when `maximumBandwidthCount` is 1.
   */
  refinement?: boolean | number;
  /**
   * Bandwidth search strategy. Compile-time. `'ladder'` (default) evaluates the caller ladder and
   * refines around its best candidate (`refinement`), unchanged. `'golden-section'` is mgwr's
   * `Sel_BW(...).search('golden_section')` as a fixed-depth loop on the GPU: the first two
   * `bandwidths` values of the parameters are the search bounds `[minimum, maximum]` (0 or less
   * takes mgwr's default for that bound, so `[0, 0]` is mgwr's default search; a single value is a
   * fixed bandwidth), each of up to `maximumSearchIterations` iterations evaluates one new
   * candidate with the same kernels, and the loop disarms itself when it converges. The ladder
   * `refinement` is ignored. Needs `maximumBandwidthCount >= 2` (the parameter capacity) and
   * `maximumNeighborCount >= 2`.
   */
  bandwidthSearch?: 'ladder' | 'golden-section';
  /**
   * Number of golden-section iterations built into the graph, 1 to 64. Compile-time. Default 30:
   * each iteration shrinks the bracket by 0.618, so 30 iterations reach `0.618^30 = 5e-7` of the
   * initial range, below the f32 resolution of any bandwidth, and an adaptive integer range
   * (at most 128 wide) collapses in about 10. Iterations after convergence launch nothing.
   * Used only with `bandwidthSearch: 'golden-section'`.
   */
  maximumSearchIterations?: number;
  /**
   * Gaussian truncation in bandwidth multiples. Compile-time. Weights of rows farther than
   * `gaussianCutoff * h` are exactly 0 (fixed and adaptive), so the Gaussian kernel can use the grid
   * index like bisquare and costs `O(n * neighbours)` instead of `O(n^2)`. Omitted (default) is the
   * exact unbounded Gaussian. The omitted weight of a row is at most `exp(-gaussianCutoff^2 / 2)`
   * (the Gaussian at the cutoff) of the self weight: 1.1e-2 at 3, 3.4e-4 at 4, 3.7e-6 at 5, 1.5e-8
   * at 6. For a locally uniform density the omitted share of the weight sum is that value and of
   * the weighted second moment `(1 + gaussianCutoff^2 / 2)` times it, which sets the relative effect on
   * the local sums. `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_RECOMMENDED_GAUSSIAN_CUTOFF` (5) keeps
   * that below float32 accumulation error. Ignored by the bisquare kernel.
   */
  gaussianCutoff?: number;
  /**
   * Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the included rows, or `false`
   * for none. Compile-time. Defaults to a square grid of about 8 rows per cell (at most 1024 per
   * axis) when there are at least 1024 rows, otherwise none. The domain is the extent of the
   * included rows, recomputed every encoding. The index is used by bisquare kernels (fixed or
   * adaptive bandwidths), by Gaussian kernels with a `gaussianCutoff`, and by the adaptive k-th
   * neighbour search; exact Gaussian weights scan every row. Aim for a few
   * cells across the typical bandwidth. Planar contexts only.
   */
  indexGridSize?: readonly [number, number] | false;
  /** Caller-owned outputs. */
  output: GPUGeographicallyWeightedRegressionOutput;
};

/**
 * Geographically weighted regression (Brunsdon, Fotheringham and Charlton 1996; Fotheringham,
 * Brunsdon and Charlton 2002) with automatic bandwidth selection by AICc, as in mgwr and ArcGIS
 * "Geographically Weighted Regression", local inference, and optional great-circle distances.
 *
 * For every included location `i` (a row with unmasked, finite position, predictors and response)
 * the contributor fits a weighted least squares `y ~ 1 + x` over all included rows `j` with weights
 * `w_ij = K(d_ij / h_i)`, `d` the planar Euclidean distance (or the great-circle distance of a
 * `spatialContext`). Kernels: `'gaussian'` `exp(-0.5 (d/h)^2)`, `'bisquare'` `(1 - (d/h)^2)^2` for
 * `d < h`. The bandwidth is fixed (`h` is the ladder value) or adaptive (`h` is `1.00001` times the
 * distance to the k-th nearest included row, counting the location itself, as mgwr; ties in
 * distance cannot change the k-th value).
 *
 * Bandwidth selection. The parameter buffer carries a ladder of up to `maximumBandwidthCount`
 * candidates. For every candidate the contributor computes `RSS = sum (y_i - yhat_i)^2`, the hat
 * trace `tr(S) = sum S_ii` with `S_ii = x_i^T (X'WX)^-1 x_i w_ii`, and
 * `AICc = n ln(RSS/n) + n ln(2 pi) + n (n + tr(S)) / (n - 2 - tr(S))` (mgwr `aicc`; the same value
 * as `-2 llf + 2 n (tr(S) + 1) / (n - tr(S) - 2)`). A candidate is invalid (score NaN) when the
 * bandwidth is not positive (fixed), `k` is outside `[2, maximumNeighborCount]` or exceeds the
 * included rows (adaptive), or `n - 2 - tr(S) <= 0`.
 *
 * Singular locations. A location whose weighted local system is singular (for example a dummy
 * predictor that is constant in its neighbourhood) makes only that row NaN (coefficients,
 * standard errors, fitted value, residual, hat, local R^2) with `localStatus` singular; it does not
 * void the candidate as in earlier versions. Candidate statistics (`n`, RSS, `tr(S)`, AICc) are taken
 * over the non-singular included rows, and every candidate reports its singular-row count
 * (`bandwidthSingularRowCounts`). Selection: among the candidates with the smallest singular-row
 * count, take the minimum AICc (ties to the lowest index), so a bandwidth with singular rows wins
 * only when every valid candidate has some. This deviates deliberately from mgwr, which raises on a
 * singular local system. When no candidate is valid, index 0 is used for the outputs and
 * `summary[HAS_VALID_CANDIDATE]` is 0.
 *
 * Refinement. The caller ladder is the coarse pass. Each refinement pass then writes (on the GPU)
 * a new ladder that zooms around the best candidate so far, between its nearest evaluated
 * neighbours (one-sided at an end of the ladder: it never extrapolates beyond the caller range),
 * evaluates it with the same kernels, and the overall best candidate wins. For adaptive bandwidths
 * the ladder holds integers: when the integers inside the bracket number at most
 * `maximumBandwidthCount` every one is evaluated, so the result is the exact integer optimum of
 * the bracket (a global optimum if AICc is unimodal in `k` there); a wider bracket is sampled at
 * equal integer spacing and the next pass narrows it, which reaches the integer optimum of a
 * bracket of up to about `(maximumBandwidthCount + 1)^passes` integers. For fixed bandwidths the
 * ladder is `maximumBandwidthCount` equally spaced distances in the bracket, so each pass narrows
 * the bandwidth interval by a factor `maximumBandwidthCount + 1`. Unlike mgwr's golden-section
 * search, which starts from its own bounds (adaptive `[40 + 2 p, n]`), the search range is the
 * caller's ladder range. A ladder of one value is a fixed, unselected bandwidth (no refinement).
 *
 * Golden-section search (`bandwidthSearch: 'golden-section'`). mgwr's `Sel_BW` golden section as a
 * fixed-depth gated loop, `O(maximumSearchIterations)` candidate evaluations instead of a ladder of
 * `maximumBandwidthCount` per pass. Bounds `[a, c]`: the first two `bandwidths` values of the
 * parameters when positive, else mgwr's defaults computed on the GPU from the included rows:
 * adaptive `a = 40 + 2 (predictorCount + 1)` and `c = n` capped at `maximumNeighborCount` (mgwr's
 * `c = n` cannot be evaluated beyond the neighbour list; for `n <= maximumNeighborCount` it is
 * identical); fixed `a = min_dist / 2` and `c = 2 max_dist`, with `min_dist` the smallest positive
 * distance from a row to another row and `max_dist` the largest pairwise distance. `max_dist` is
 * exact when no grid index is built (one extra scan per row, `O(n^2)` like one candidate); with a
 * grid it is the largest distance from any row to a corner of the grid extent, an upper bound that
 * costs at most one or two extra iterations. Interior points are mgwr's
 * `b = a + 0.38197 |c - a|` and `d = c - 0.38197 |c - a|`, rounded to integers for adaptive
 * bandwidths (round half to even, as numpy). Iteration 1 evaluates `b` and `d`; every later
 * iteration compares the two scores (`score_b <= score_d` keeps the lower interval, ties and
 * invalid scores resolved as documented below), shrinks the bracket, and evaluates the one new
 * interior point, the other being reused. The search stops when `|score_b - score_d|` is not above
 * the tolerance `max(1e-6, 4e-6 max(|score_b|, |score_d|))` (mgwr's `1e-6` with a float32 noise
 * floor, see `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE`) or after
 * `maximumSearchIterations` iterations, and the selected bandwidth is the winner of the last
 * comparison, as mgwr returns it (a candidate with fewer singular rows beats one with more, then
 * the lower AICc; an invalid score loses to a valid one; two invalid scores stop the search). The
 * iteration index lives in the GPU cursor word of the ladder buffer and the loop is gated by an
 * indirect dispatch triple that the iteration kernel zeroes on convergence, so N and N + 1
 * iterations compile the same pipelines and converged iterations cost nothing. No read-back.
 *

 * Local inference. With `standardErrors`, `tValues` or `significant` requested, the fit also
 * accumulates `X'W^2X` and reports mgwr's `bse` and `tvalues`. The summary carries `ENP = tr(S)`,
 * `sigma2 = RSS / (n - tr(S))` (mgwr `sigma2_v1`), the log-likelihood, the Da Silva-Fotheringham
 * adjusted alpha `alpha * k / tr(S)` (`k` design columns, `alpha` 0.05 unless the per-frame
 * `significanceLevel` says otherwise) and the matching critical `|t|` (`t.ppf(1 - adj / 2, n - 1)`,
 * inverted on the GPU from the Student t distribution). The bandwidth is selected on the observed
 * data and the inference is conditional on it, as in mgwr. Use
 * `getGPUGeographicallyWeightedRegressionModelResult` to turn the read-back summary into a
 * `GPURegressionModelResult`.
 *
 * Numerics: the design is centred on the focal row, `z_j = (1, x_j - x_i)`, so the fitted value
 * is the centred intercept and `S_ii` is the `(0, 0)` entry of the inverse; the normal equations
 * are Jacobi-equilibrated (divided by the square root of their diagonal) before the shared
 * Cholesky solve. Original-space intercepts (and their standard errors) are recovered as
 * `b0 - sum b_c x_ic`. A predictor that is constant over a location's neighbourhood makes that
 * location singular.
 *
 * Complexity: one thread per location evaluates every candidate of every pass (and once more to
 * select k-th nearest distances and to evaluate the final fit). Bisquare kernels have bounded
 * support, so with a `GPUGridIndex` (`indexGridSize`, built by default from 1024 rows) they visit
 * only the rows in the cells covering the bandwidth, for fixed and adaptive bandwidths alike: the
 * work is `O(n * neighbours * ladder * p^2)` and the 65,536-row limit of the scan is lifted to
 * 1,048,576 rows (the indexed limit is the `rows * ladder * 8` byte per-candidate scratch and the
 * row-state buffer fitting one 128 MiB storage binding). An adaptive bandwidth finds its k-th
 * nearest distance by walking grid cells in rings around the location and stopping once the k
 * nearest are inside the visited block, `O(k)` per location instead of a scan of every row (any
 * kernel, adaptive). Invocations take the included rows in grid-cell order, so one workgroup fits
 * neighboring locations and shares their cells. Gaussian kernels are unbounded: without
 * `gaussianCutoff` they scan every row per candidate in `O(n^2)` (a grid-enabled instance accepts
 * them but the caller must keep `n` practical); with a cutoff they visit only the cells within
 * `gaussianCutoff * h` like bisquare. The grid path sums rows in cell order (ascending row ID inside
 * each cell) instead of row order, so results differ from the scan only by float rounding (about
 * 1e-5 in coefficients); sums stay in a fixed order and no float atomics are used, so results are
 * bitwise reproducible on one adapter. Candidate RSS and trace are reduced by 256-row tiles merged
 * in tile order. Very clustered data puts many rows in one cell and the in-cell ID ranking costs
 * `O(cell size^2)`.
 *
 * Limits: 11 predictors (12 design columns), see
 * `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT`. No kernel binds more than eight
 * storage buffers: the per-row results share one row-state buffer and the outputs are published
 * by separate kernels.
 */
export class GPUGeographicallyWeightedRegression implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeographicallyWeightedRegressionProps;
  /** Row count, `positions.length`. */
  readonly rowCount: number;
  /** Columns of the design including the intercept, `predictorCount + 1`. */
  readonly coefficientCount: number;
  /** Ladder capacity. */
  readonly maximumBandwidthCount: number;
  /** Largest adaptive `k`. */
  readonly maximumNeighborCount: number;
  /** Grid index dimensions `[columns, rows]`, or null when no index is built. */
  readonly indexGridSize: readonly [number, number] | null;
  /** Resolved coordinate space and metric (planar native when none was given). */
  readonly spatialContext: GPUSpatialContext;
  /** Refinement passes after the caller ladder (0 when refinement is off or searching). */
  readonly refinementPassCount: number;
  /** Resolved `bandwidthSearch`. */
  readonly bandwidthSearch: 'ladder' | 'golden-section';
  /** Golden-section iterations built into the graph (0 for the ladder search). */
  readonly maximumSearchIterations: number;
  /** Resolved `gaussianCutoff`, or null for the exact Gaussian. */
  readonly gaussianCutoff: number | null;
  /**
   * Candidates evaluated together by one pass: `maximumBandwidthCount` for the ladder search, 2 for
   * the golden-section search (its first iteration evaluates both interior points).
   */
  readonly candidateWidth: number;
  /** Rows of the candidate table. */
  readonly candidateTableRowCount: number;

  constructor(props: GPUGeographicallyWeightedRegressionProps) {
    this.id = props.id ?? 'geographically-weighted-regression';
    this.props = props;
    const id = this.id;
    const {predictorCount, output} = props;
    validateCapacity(
      id,
      'predictorCount',
      predictorCount,
      1,
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT
    );
    this.coefficientCount = predictorCount + 1;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.maximumBandwidthCount =
      props.maximumBandwidthCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH;
    validateCapacity(
      id,
      'maximumBandwidthCount',
      this.maximumBandwidthCount,
      1,
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH
    );
    this.maximumNeighborCount =
      props.maximumNeighborCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT;
    validateCapacity(
      id,
      'maximumNeighborCount',
      this.maximumNeighborCount,
      1,
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT
    );
    const refinement = props.refinement ?? true;
    if (typeof refinement === 'number') {
      validateCapacity(
        id,
        'refinement',
        refinement,
        0,
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_REFINEMENT_PASS_COUNT
      );
    }
    const bandwidthSearch = props.bandwidthSearch ?? 'ladder';
    if (bandwidthSearch !== 'ladder' && bandwidthSearch !== 'golden-section') {
      throw new Error(`${id} bandwidthSearch must be 'ladder' or 'golden-section'`);
    }
    this.bandwidthSearch = bandwidthSearch;
    const isGoldenSection = bandwidthSearch === 'golden-section';
    if (isGoldenSection) {
      // The bounds are the first two values of the parameter ladder, and the k-th neighbour list
      // must reach k = 2 for the nearest-neighbour distance of the default fixed bounds.
      if (this.maximumBandwidthCount < 2) {
        throw new Error(`${id} golden-section search needs maximumBandwidthCount >= 2`);
      }
      if (this.maximumNeighborCount < 2) {
        throw new Error(`${id} golden-section search needs maximumNeighborCount >= 2`);
      }
    }
    this.maximumSearchIterations = isGoldenSection
      ? (props.maximumSearchIterations ??
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_MAXIMUM_SEARCH_ITERATIONS)
      : 0;
    if (isGoldenSection) {
      validateCapacity(
        id,
        'maximumSearchIterations',
        this.maximumSearchIterations,
        1,
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS
      );
    }
    const {gaussianCutoff} = props;
    if (gaussianCutoff !== undefined && !(Number.isFinite(gaussianCutoff) && gaussianCutoff > 0)) {
      throw new Error(`${id} gaussianCutoff must be a positive finite number`);
    }
    this.gaussianCutoff = gaussianCutoff ?? null;
    this.refinementPassCount =
      this.maximumBandwidthCount < 2 || isGoldenSection
        ? 0
        : refinement === true
          ? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_REFINEMENT_PASS_COUNT
          : refinement === false
            ? 0
            : refinement;
    this.candidateWidth = isGoldenSection ? 2 : this.maximumBandwidthCount;
    this.candidateTableRowCount = isGoldenSection
      ? this.maximumSearchIterations + 1
      : (1 + this.refinementPassCount) * this.maximumBandwidthCount;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    this.rowCount = props.positions.length;
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    const isGreatCircle = this.spatialContext.metric === 'great-circle';
    if (isGreatCircle) {
      if (props.indexGridSize) {
        throw new Error(`${id} indexGridSize is planar-only; omit it for a great-circle context`);
      }
      this.indexGridSize = null;
    } else if (props.indexGridSize === false) {
      this.indexGridSize = null;
    } else if (props.indexGridSize) {
      const [columns, gridRows] = props.indexGridSize;
      if (
        !Number.isInteger(columns) ||
        !Number.isInteger(gridRows) ||
        columns < 1 ||
        gridRows < 1 ||
        columns * gridRows >= 0xffffffff
      ) {
        throw new Error(
          `${id} indexGridSize must be two positive integers with columns * rows < 2^32 - 1`
        );
      }
      this.indexGridSize = [columns, gridRows];
    } else if (this.rowCount >= GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT) {
      const dimension = Math.min(
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION,
        Math.max(1, Math.ceil(Math.sqrt(this.rowCount / 8)))
      );
      this.indexGridSize = [dimension, dimension];
    } else {
      this.indexGridSize = null;
    }
    if (
      !this.indexGridSize &&
      this.rowCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT
    ) {
      throw new Error(
        `${id} supports at most ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT} rows without a grid index`
      );
    }
    if (
      this.indexGridSize &&
      (this.rowCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT ||
        this.rowCount * this.candidateWidth * 8 > MAXIMUM_SCRATCH_BYTES)
    ) {
      throw new Error(
        `${id} supports at most ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT} rows ` +
          `and rows * ${isGoldenSection ? '2' : 'maximumBandwidthCount'} * 8 <= ${MAXIMUM_SCRATCH_BYTES} with a grid index`
      );
    }
    if (this.getRowStateWordCount() * 4 > MAXIMUM_SCRATCH_BYTES) {
      throw new Error(
        `${id} row state exceeds one storage binding; reduce rows, predictors or inference outputs`
      );
    }
    validatePackedView(props.predictors, ['float32'], `${id} predictors`);
    if (props.predictors.length !== this.rowCount * predictorCount) {
      throw new Error(`${id} predictors length must equal rows * predictorCount`);
    }
    validatePackedView(props.response, ['float32'], `${id} response`);
    if (props.response.length !== this.rowCount) {
      throw new Error(`${id} response length must equal the row count`);
    }
    if (props.mask) {
      validatePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUGeographicallyWeightedRegressionParameterLength(
      this.maximumBandwidthCount
    );
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    const tableLength =
      this.candidateTableRowCount * GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE;
    const lengths: [keyof GPUGeographicallyWeightedRegressionOutput, number][] = [
      ['coefficients', this.rowCount * this.coefficientCount],
      ['standardErrors', this.rowCount * this.coefficientCount],
      ['tValues', this.rowCount * this.coefficientCount],
      ['significant', this.rowCount * this.coefficientCount],
      ['localR2', this.rowCount],
      ['fitted', this.rowCount],
      ['residuals', this.rowCount],
      ['hatDiagonal', this.rowCount],
      ['localStatus', this.rowCount],
      ['localConditionNumber', this.rowCount],
      ['bandwidthScores', this.maximumBandwidthCount],
      ['bandwidthSingularRowCounts', this.maximumBandwidthCount],
      ['candidateTable', tableLength],
      ['selectedBandwidth', 2],
      ['searchState', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH],
      ['summary', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH]
    ];
    if (output.searchState && !isGoldenSection) {
      throw new Error(`${id} output.searchState needs bandwidthSearch 'golden-section'`);
    }
    const uint32Outputs = new Set(['localStatus', 'significant', 'bandwidthSingularRowCounts']);
    for (const [name, length] of lengths) {
      const view = output[name];
      if (!view) {
        if (name === 'coefficients') {
          throw new Error(`${id} output.coefficients is required`);
        }
        continue;
      }
      if (uint32Outputs.has(name)) {
        validatePackedView(view as GraphDataView<'uint32'>, ['uint32'], `${id} output.${name}`);
      } else {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
      }
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold at least ${length} values`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      lengths.map(([name]) => output[name]),
      [props.positions, props.predictors, props.response, props.mask, props.parameters]
    );
  }

  /** True when any of the local inference outputs is requested. */
  private hasInference(): boolean {
    const {output} = this.props;
    return Boolean(output.standardErrors || output.tValues || output.significant);
  }

  /** True when the t values (or the mask derived from them) are requested. */
  private hasTValues(): boolean {
    const {output} = this.props;
    return Boolean(output.tValues || output.significant);
  }

  /**
   * Words of the shared row-state buffer: validity flags, the per-row local results, the standard
   * error and t value blocks when requested, and the tail of selection results.
   */
  private getRowStateWordCount(): number {
    const {rowCount, coefficientCount} = this;
    return (
      rowCount * (1 + LOCAL_STRIDE) +
      (this.hasInference() ? rowCount * coefficientCount : 0) +
      (this.hasTValues() ? rowCount * coefficientCount : 0) +
      TAIL_WORDS
    );
  }

  /**
   * Returns validate, then per candidate pass (the caller ladder, then each refinement) an optional
   * ladder-refinement node, the candidate evaluation, the tile sums and the candidate table;
   * then bandwidth selection, the final fit, optional excluded-row, statistics and publish nodes
   * in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      props,
      id,
      rowCount,
      coefficientCount,
      maximumBandwidthCount,
      maximumNeighborCount,
      indexGridSize,
      refinementPassCount,
      maximumSearchIterations,
      candidateWidth,
      candidateTableRowCount,
      gaussianCutoff
    } = this;
    const isGoldenSection = this.bandwidthSearch === 'golden-section';
    const {output, predictorCount, positions, predictors, response, parameters} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      predictors,
      response,
      props.mask,
      parameters,
      ...Object.values(output)
    ]);
    // Candidate passes: the caller ladder and each refinement, or one per golden-section iteration
    // (iteration i evaluates the candidate its predecessor scheduled; the last one evaluates none).
    const passCount = isGoldenSection ? maximumSearchIterations : 1 + refinementPassCount;
    const tileCount = Math.ceil(rowCount / TILE_ROWS);
    const hasInference = this.hasInference();
    const hasTValues = this.hasTValues();
    const P = coefficientCount;
    const parameterLength =
      getGPUGeographicallyWeightedRegressionParameterLength(maximumBandwidthCount);
    const transient = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    // One u32 buffer holds the row validity flags, then the per-row local results, the standard
    // error and t value blocks, and a tail of selection results. Sharing a binding keeps every
    // kernel within the 8 storage buffers per stage of core WebGPU devices.
    const rowValid = transient('row-state', 'uint32', this.getRowStateWordCount());
    const candidateScratch = transient(
      'candidate-scratch',
      'float32',
      rowCount * candidateWidth * 2
    );
    const tileCandidates = transient(
      'tile-candidates',
      'float32',
      tileCount * candidateWidth * TILE_CANDIDATE_STRIDE
    );
    const tableStride = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE;
    const candidateTable =
      output.candidateTable ??
      transient('candidate-table', 'float32', candidateTableRowCount * tableStride);
    // Rule 12: every candidate pass reads its ladder from one buffer, `[passCursor, ladder 0,
    // ladder 1, ...]`, where ladder 0 is a copy of the caller ladder and the refinement nodes write
    // the others. The pass index lives in the GPU cursor word (an exact small integer in f32), never
    // in WGSL text, so every pass shares one pipeline per kernel.
    const ladders = transient('ladders', 'float32', 1 + passCount * parameterLength);
    const tileStatistics = transient('tile-statistics', 'float32', tileCount * 3);
    const selection = output.selectedBandwidth ?? transient('selected-bandwidth', 'float32', 2);
    const searchState = isGoldenSection
      ? (output.searchState ??
        transient(
          'search-state',
          'float32',
          GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH
        ))
      : null;
    const summary =
      output.summary ??
      transient('summary', 'float32', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH);

    const gridPositions = indexGridSize ? transient('grid-positions', 'float32x2', rowCount) : null;
    const gridIndex =
      indexGridSize && gridPositions
        ? createGridIndexNodes(graph, {
            id,
            operation: OPERATION,
            positions: gridPositions,
            rowCount,
            gridSize: indexGridSize
          })
        : null;
    const cellOffsets = gridIndex?.cellOffsets ?? null;
    const sortedIds = gridIndex?.sortedIds ?? null;

    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read_write'
    });
    const standardErrorBase = rowCount * (1 + LOCAL_STRIDE);
    const tValueBase = standardErrorBase + (hasInference ? rowCount * P : 0);
    const tailBase = tValueBase + (hasTValues ? rowCount * P : 0);
    const isGreatCircle = this.spatialContext.metric === 'great-circle';
    const sphereRadius = isGreatCircle ? getGPUSphereRadius(this.spatialContext) : null;
    const common = `const ROW_COUNT: u32 = ${rowCount}u;
const PREDICTOR_COUNT: u32 = ${predictorCount}u;
const P: u32 = ${P}u;
const LADDER: u32 = ${candidateWidth}u;
const TABLE_ROWS: u32 = ${candidateTableRowCount}u;
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighborCount}u;
const TILE_ROWS: u32 = ${TILE_ROWS}u;
const TILE_COUNT: u32 = ${tileCount}u;
const LOCAL_STRIDE: u32 = ${LOCAL_STRIDE}u;
const TILE_CANDIDATE_STRIDE: u32 = ${TILE_CANDIDATE_STRIDE}u;
const TABLE_STRIDE: u32 = ${tableStride}u;
// Layout of the rowValid binding: validity flags, local results, standard errors, t values, tail.
const LOCALS_BASE: u32 = ${rowCount}u;
const STANDARD_ERROR_BASE: u32 = ${standardErrorBase}u;
const T_VALUE_BASE: u32 = ${tValueBase}u;
const TAIL_BASE: u32 = ${tailBase}u;
const TAIL_SELECTED_INDEX: u32 = ${TAIL.SELECTED_INDEX}u;
const TAIL_SELECTED_VALUE: u32 = ${TAIL.SELECTED_VALUE}u;
const TAIL_KERNEL: u32 = ${TAIL.KERNEL}u;
const TAIL_MODE: u32 = ${TAIL.MODE}u;
const TAIL_SIGMA_SQUARED: u32 = ${TAIL.SIGMA_SQUARED}u;
const TAIL_CRITICAL_T: u32 = ${TAIL.CRITICAL_T}u;
const HEADER: u32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH}u;
const PARAMETER_LENGTH: u32 = ${parameterLength}u;
const SENTINEL: f32 = 3.0e38;
const ADAPTIVE_FACTOR: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR};
const MINIMUM_VARIANCE: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE};
const LN_TWO_PI: f32 = 1.8378770664093453;
${GWR_COMMON_WGSL}`;
    const gridBindings: WGSLKernelBinding[] = indexGridSize
      ? [read('cellOffsets', cellOffsets!, 'u32'), read('sortedIds', sortedIds!, 'u32')]
      : [];
    const mask = props.mask;
    // With a grid, invocations take the included rows in grid-cell order instead of row order, so
    // the invocations of one workgroup fit neighboring locations and read the same neighborhood
    // cells (row order is spatially random). Per-location results do not depend on the order.
    const focalRowWGSL = indexGridSize
      ? `if (index >= getFocalCount()) {
    return;
  }
  let focal = getFocalRow(index);`
      : 'let focal = index;';
    // Settings of a candidate pass come from that pass's parameter block.
    const passDeclarations = `${common}
fn getParameter(slot: u32) -> f32 {
  let passCursor = u32(params[paramsOffset]);
  return params[paramsOffset + 1u + passCursor * PARAMETER_LENGTH + slot];
}
fn getIsBisquare() -> bool {
  return getParameter(0u) > 0.5;
}
fn getIsAdaptive() -> bool {
  return getParameter(1u) > 0.5;
}
fn getCandidateCount() -> u32 {
  return min(u32(max(getParameter(2u), 0.0)), LADDER);
}`;
    const candidateDeclarations = `${passDeclarations}
${getGWRFitWGSL({
  coefficientCount: P,
  indexGridSize,
  hasConditionNumber: false,
  hasVariance: false,
  sphereRadius,
  gaussianCutoff
})}
// Largest requested k among valid adaptive candidates, clamped to [1, MAXIMUM_NEIGHBORS].
fn getMaximumK(candidateCount: u32) -> u32 {
  var maximumK = 1u;
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    let k = getParameter(HEADER + candidate);
    if (k >= 1.0 && k <= f32(MAXIMUM_NEIGHBORS)) {
      maximumK = max(maximumK, u32(floor(k + 0.5)));
    }
  }
  return min(maximumK, MAXIMUM_NEIGHBORS);
}`;
    // The final fit reads the selected bandwidth and settings from the row-state tail.
    const fitDeclarations = `${common}
fn getIsBisquare() -> bool {
  return bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_KERNEL]) > 0.5;
}
fn getIsAdaptive() -> bool {
  return bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_MODE]) > 0.5;
}
${getGWRFitWGSL({
  coefficientCount: P,
  indexGridSize,
  hasConditionNumber: Boolean(output.localConditionNumber),
  hasVariance: hasInference,
  sphereRadius,
  gaussianCutoff
})}`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validate`,
        operation: OPERATION,
        variant: 'validate',
        bindings: [
          read('positions', positions, 'f32'),
          read('predictors', predictors, 'f32'),
          read('response', response, 'f32'),
          ...(mask ? [read('rowMask', mask, 'u32')] : []),
          write('rowValid', rowValid, 'u32')
        ],
        invocationCount: rowCount,
        declarations: common,
        body: `var isValid = true;
  ${mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  if (!isFiniteBits(positions[positionsOffset + 2u * index]) ||
    !isFiniteBits(positions[positionsOffset + 2u * index + 1u]) ||
    !isFiniteBits(response[responseOffset + index])) {
    isValid = false;
  }
  ${isGreatCircle ? 'if (!(abs(positions[positionsOffset + 2u * index + 1u]) <= 90.0)) {\n    isValid = false;\n  }' : ''}
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + index * PREDICTOR_COUNT + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);`
      })
    );

    if (gridIndex && gridPositions) {
      nodes.push(
        // The grid index has no mask, so excluded rows become NaN points, which it ignores.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-grid-positions`,
          operation: OPERATION,
          variant: 'grid-positions',
          bindings: [
            read('positions', positions, 'f32'),
            read('rowValid', rowValid, 'u32'),
            write('gridPositions', gridPositions, 'f32')
          ],
          invocationCount: rowCount,
          declarations: common,
          body: `let included = rowValid[rowValidOffset + index] != 0u;
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridPositions[gridPositionsOffset + index * 2u] =
    select(invalid, positions[positionsOffset + index * 2u], included);
  gridPositions[gridPositionsOffset + index * 2u + 1u] =
    select(invalid, positions[positionsOffset + index * 2u + 1u], included);`
        }),
        ...gridIndex.nodes
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ladder-init`,
        operation: OPERATION,
        variant: 'ladder-init',
        bindings: [read('params', parameters, 'f32'), write('ladders', ladders, 'f32')],
        invocationCount: 1,
        declarations: common,
        body: `ladders[laddersOffset] = 0.0;
  for (var slot = 0u; slot < PARAMETER_LENGTH; slot++) {
    ladders[laddersOffset + 1u + slot] = params[paramsOffset + slot];
  }`
      })
    );
    const candidateBindings: WGSLKernelBinding[] = [
      read('positions', positions, 'f32'),
      read('predictors', predictors, 'f32'),
      read('response', response, 'f32'),
      read('rowValid', rowValid, 'u32'),
      read('params', ladders, 'f32'),
      ...gridBindings
    ];
    const search =
      isGoldenSection && searchState
        ? createGoldenSectionSearchNodes<Parameters>(graph, {
            id,
            operation: OPERATION,
            rowCount,
            tileRows: TILE_ROWS,
            tileCount,
            candidateWidth,
            maximumIterations: maximumSearchIterations,
            tableRowCount: candidateTableRowCount,
            common,
            candidateDeclarations,
            focalRowWGSL,
            hasGrid: Boolean(indexGridSize),
            candidateBindings,
            rowValid,
            ladders,
            parameters,
            candidateScratch,
            tileCandidates,
            candidateTable,
            searchState
          })
        : null;
    if (search) {
      nodes.push(...search.setupNodes);
    }
    for (let pass = 0; pass < passCount; pass++) {
      const suffix = pass === 0 ? '' : isGoldenSection ? `-search-${pass}` : `-refine-${pass}`;
      // Golden-section passes after the first run only while the search is armed (rule 13).
      const gated = search && pass > 0 ? search.gate : null;
      if (search && pass > 0) {
        nodes.push(search.createIterationNode(pass));
      } else if (pass > 0) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-refine-${pass}-ladder`,
            operation: OPERATION,
            variant: 'refine-ladder',
            bindings: [write('ladders', ladders, 'f32'), read('table', candidateTable, 'f32')],
            invocationCount: 1,
            declarations: `${common}
${REFINE_LADDER_WGSL}`,
            body: 'refineLadder();'
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}${suffix}-candidates`,
          operation: OPERATION,
          variant: 'candidates',
          ...gated?.getGate(0),
          bindings: [...candidateBindings, write('candidateScratch', candidateScratch, 'f32')],
          invocationCount: rowCount,
          declarations: candidateDeclarations,
          body: `${focalRowWGSL}
  let candidateCount = getCandidateCount();
  if (candidateCount == 0u || rowValid[rowValidOffset + focal] == 0u) {
    // Excluded rows are skipped by the tile sums, so their scratch is never read.
    return;
  }
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  let isAdaptive = getIsAdaptive();
  if (isAdaptive) {
    selectNeighborDistances(focal, getMaximumK(candidateCount), &neighbors);
  }
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    // Marker in the hat slot: -1 singular local system, -2 invalid bandwidth for this row.
    let slot = candidateScratchOffset + (candidate * ROW_COUNT + focal) * 2u;
    candidateScratch[slot] = 0.0;
    candidateScratch[slot + 1u] = -1.0;
    let bandwidth = getBandwidthForValue(getParameter(HEADER + candidate), isAdaptive, &neighbors);
    if (bandwidth <= 0.0) {
      candidateScratch[slot + 1u] = -2.0;
      continue;
    }
    let fit = fitLocation(focal, bandwidth);
    if (!fit.ok) {
      continue;
    }
    let residual = response[responseOffset + focal] - fit.beta[0];
    candidateScratch[slot] = residual * residual;
    candidateScratch[slot + 1u] = fit.hat;
  }`
        })
      );

      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}${suffix}-tile-candidates`,
          operation: OPERATION,
          variant: 'tile-candidates',
          ...gated?.getGate(1),
          bindings: [
            read('candidateScratch', candidateScratch, 'f32'),
            read('rowValid', rowValid, 'u32'),
            write('tileCandidates', tileCandidates, 'f32')
          ],
          invocationCount: tileCount * candidateWidth,
          declarations: common,
          body: `let tile = index / LADDER;
  let candidate = index % LADDER;
  let firstRow = tile * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var residualSum = 0.0;
  var traceSum = 0.0;
  var singular = 0.0;
  var invalid = 0.0;
  var included = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    included = included + 1.0;
    let slot = candidateScratchOffset + (candidate * ROW_COUNT + row) * 2u;
    let hat = candidateScratch[slot + 1u];
    if (hat < -1.5) {
      invalid = invalid + 1.0;
    } else if (hat < 0.0) {
      singular = singular + 1.0;
    } else {
      residualSum = residualSum + candidateScratch[slot];
      traceSum = traceSum + hat;
    }
  }
  let base = tileCandidatesOffset + index * TILE_CANDIDATE_STRIDE;
  tileCandidates[base] = residualSum;
  tileCandidates[base + 1u] = traceSum;
  tileCandidates[base + 2u] = singular;
  tileCandidates[base + 3u] = invalid;
  tileCandidates[base + 4u] = included;`
        })
      );

      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}${suffix}-select`,
          operation: OPERATION,
          variant: 'select',
          ...gated?.getGate(2),
          bindings: [
            read('tileCandidates', tileCandidates, 'f32'),
            read('params', ladders, 'f32'),
            write('table', candidateTable, 'f32')
          ],
          invocationCount: candidateWidth,
          declarations: passDeclarations,
          body: `let candidate = index;
  let passCursor = u32(params[paramsOffset]);
  let nan = getNaN();
  ${
    isGoldenSection
      ? `// One table row per evaluation, in order: pass 0 fills rows 0 and 1, pass p the row p + 1.
  // Rows of unused candidates keep the NaN fill of the search initialization.
  if (candidate >= getCandidateCount()) {
    return;
  }
  let base = tableOffset + select(passCursor + 1u, candidate, passCursor == 0u) * TABLE_STRIDE;`
      : `let base = tableOffset + (passCursor * LADDER + candidate) * TABLE_STRIDE;
  if (candidate >= getCandidateCount()) {
    for (var field = 0u; field < TABLE_STRIDE; field++) {
      table[base + field] = nan;
    }
    return;
  }`
  }
  var residualSum = 0.0;
  var traceSum = 0.0;
  var singular = 0.0;
  var invalid = 0.0;
  var included = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let tileBase = tileCandidatesOffset + (tile * LADDER + candidate) * TILE_CANDIDATE_STRIDE;
    residualSum = residualSum + tileCandidates[tileBase];
    traceSum = traceSum + tileCandidates[tileBase + 1u];
    singular = singular + tileCandidates[tileBase + 2u];
    invalid = invalid + tileCandidates[tileBase + 3u];
    included = included + tileCandidates[tileBase + 4u];
  }
  // AICc over the non-singular rows; a row with an invalid bandwidth voids the candidate.
  let count = included - singular;
  let denominator = count - 2.0 - traceSum;
  var score = nan;
  if (invalid == 0.0 && count > 0.0 && denominator > 0.0) {
    let variance = max(residualSum / count, MINIMUM_VARIANCE);
    let candidateScore = count * log(variance) + count * LN_TWO_PI + count * (count + traceSum) / denominator;
    if (isFiniteBits(candidateScore)) {
      score = candidateScore;
    }
  }
  table[base] = getParameter(HEADER + candidate);
  table[base + 1u] = score;
  table[base + 2u] = singular;
  table[base + 3u] = residualSum;
  table[base + 4u] = traceSum;
  table[base + 5u] = count;
  table[base + 6u] = included;`
        })
      );
    }
    if (search) {
      // The comparison of the last iteration schedules no candidate.
      nodes.push(search.createIterationNode(passCount));
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-choose`,
        operation: OPERATION,
        variant: 'choose',
        bindings: [
          read('params', parameters, 'f32'),
          read('table', candidateTable, 'f32'),
          write('rowValid', rowValid, 'u32'),
          write('selection', selection, 'f32'),
          write('summary', summary, 'f32'),
          ...(searchState ? [read('state', searchState, 'f32')] : [])
        ],
        invocationCount: 1,
        declarations: `${common}
${getStudentTWGSL()}
${searchState ? GPU_SEARCH_STATE_WGSL : ''}
const DEFAULT_SIGNIFICANCE_LEVEL: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_SIGNIFICANCE_LEVEL};`,
        body: `let nan = getNaN();
  // Fewest singular rows first, then the smallest AICc; ties go to the lowest index.
  var hasBest = false;
  var bestIndex = 0u;
  var bestSingular = 0.0;
  var bestScore = 0.0;
  for (var candidate = 0u; candidate < TABLE_ROWS; candidate++) {
    let base = tableOffset + candidate * TABLE_STRIDE;
    let score = table[base + 1u];
    if (!isFiniteBits(score)) {
      continue;
    }
    let singular = table[base + 2u];
    if (!hasBest || singular < bestSingular || (singular == bestSingular && score < bestScore)) {
      hasBest = true;
      bestIndex = candidate;
      bestSingular = singular;
      bestScore = score;
    }
  }
  ${
    searchState
      ? `// Golden section returns the winner of its last comparison (mgwr), when it is valid.
  let optimumRow = state[stateOffset + STATE_OPTIMUM_ROW];
  if (optimumRow >= 0.0) {
    let optimumBase = tableOffset + u32(optimumRow) * TABLE_STRIDE;
    let optimumScore = table[optimumBase + 1u];
    if (isFiniteBits(optimumScore)) {
      hasBest = true;
      bestIndex = u32(optimumRow);
      bestSingular = table[optimumBase + 2u];
      bestScore = optimumScore;
    }
  }`
      : ''
  }
  let base = tableOffset + bestIndex * TABLE_STRIDE;
  let selectedValue = table[base];
  let residualSum = select(nan, table[base + 3u], hasBest);
  let traceSum = select(nan, table[base + 4u], hasBest);
  let count = table[base + 5u];
  let included = table[tableOffset + 6u];
  let sigmaSquared = select(nan, residualSum / (count - traceSum), hasBest);
  let requestedLevel = params[paramsOffset + 3u];
  let level = select(DEFAULT_SIGNIFICANCE_LEVEL, requestedLevel, requestedLevel > 0.0 && requestedLevel < 1.0);
  let adjustedAlpha = select(nan, level * f32(P) / traceSum, hasBest && traceSum > 0.0);
  let criticalT = select(nan, gwrStudentCriticalValue(adjustedAlpha, count - 1.0), hasBest && traceSum > 0.0);
  let logLikelihood = select(
    nan,
    -0.5 * count * (LN_TWO_PI + log(max(residualSum / count, MINIMUM_VARIANCE)) + 1.0),
    hasBest
  );
  selection[selectionOffset] = f32(bestIndex);
  selection[selectionOffset + 1u] = selectedValue;
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_INDEX] = bestIndex;
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_VALUE] = bitcast<u32>(selectedValue);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_KERNEL] = bitcast<u32>(params[paramsOffset]);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_MODE] = bitcast<u32>(params[paramsOffset + 1u]);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SIGMA_SQUARED] = bitcast<u32>(sigmaSquared);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_CRITICAL_T] = bitcast<u32>(criticalT);
  summary[summaryOffset] = residualSum;
  summary[summaryOffset + 1u] = traceSum;
  summary[summaryOffset + 2u] = select(nan, bestScore, hasBest);
  summary[summaryOffset + 3u] = nan;
  summary[summaryOffset + 4u] = select(included, count, hasBest);
  summary[summaryOffset + 5u] = select(0.0, 1.0, hasBest);
  summary[summaryOffset + 6u] = select(nan, bestSingular, hasBest);
  summary[summaryOffset + 7u] = sigmaSquared;
  summary[summaryOffset + 8u] = adjustedAlpha;
  summary[summaryOffset + 9u] = criticalT;
  summary[summaryOffset + 10u] = logLikelihood;
  summary[summaryOffset + 11u] = included;`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fit`,
        operation: OPERATION,
        variant: 'fit',
        bindings: [
          read('positions', positions, 'f32'),
          read('predictors', predictors, 'f32'),
          read('response', response, 'f32'),
          write('rowValid', rowValid, 'u32'),
          ...gridBindings,
          write('coefficients', output.coefficients, 'f32')
        ],
        invocationCount: rowCount,
        declarations: fitDeclarations,
        body: `${focalRowWGSL}
  let nan = getNaN();
  let coefficientBase = coefficientsOffset + focal * P;
  let localBase = rowValidOffset + LOCALS_BASE + focal * LOCAL_STRIDE;
  var status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.SINGULAR}u;
  var localR2 = nan;
  var conditionNumber = nan;
  var fittedValue = nan;
  var residual = nan;
  var hat = nan;
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientBase + column] = nan;
    ${hasInference ? 'rowValid[rowValidOffset + STANDARD_ERROR_BASE + focal * P + column] = bitcast<u32>(nan);' : ''}
    ${hasTValues ? 'rowValid[rowValidOffset + T_VALUE_BASE + focal * P + column] = bitcast<u32>(nan);' : ''}
  }
  if (rowValid[rowValidOffset + focal] == 0u) {
    status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED}u;
  } else {
    let selectedValue = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_VALUE]);
    var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
    let isAdaptive = getIsAdaptive();
    if (isAdaptive) {
      let maximumK = clamp(u32(max(floor(selectedValue + 0.5), 1.0)), 1u, MAXIMUM_NEIGHBORS);
      selectNeighborDistances(focal, maximumK, &neighbors);
    }
    let bandwidth = getBandwidthForValue(selectedValue, isAdaptive, &neighbors);
    if (bandwidth > 0.0) {
      let fit = fitLocation(focal, bandwidth);
      if (fit.ok) {
        status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK}u;
        var intercept = fit.beta[0];
        for (var column = 1u; column < P; column++) {
          let slope = fit.beta[column];
          coefficients[coefficientBase + column] = slope;
          intercept = intercept - slope * predictors[predictorsOffset + focal * PREDICTOR_COUNT + column - 1u];
        }
        coefficients[coefficientBase] = intercept;
        fittedValue = fit.beta[0];
        residual = response[responseOffset + focal] - fittedValue;
        hat = fit.hat;
        localR2 = getLocalR2(focal, bandwidth, fit);
        ${output.localConditionNumber ? 'conditionNumber = getLocalConditionNumber(focal, bandwidth);' : ''}
        ${
          hasInference
            ? `let sigmaSquared = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_SIGMA_SQUARED]);
        for (var column = 0u; column < P; column++) {
          var standardError = nan;
          let variance = fit.variance[column] * sigmaSquared;
          if (isFiniteBits(variance) && variance >= 0.0) {
            standardError = sqrt(variance);
          }
          rowValid[rowValidOffset + STANDARD_ERROR_BASE + focal * P + column] = bitcast<u32>(standardError);
          ${
            hasTValues
              ? `let coefficient = coefficients[coefficientBase + column];
          let tValue = select(nan, coefficient / standardError, isFiniteBits(standardError) && standardError > 0.0 && isFiniteBits(coefficient));
          rowValid[rowValidOffset + T_VALUE_BASE + focal * P + column] = bitcast<u32>(tValue);`
              : ''
          }
        }`
            : ''
        }
      }
    }
  }
  rowValid[localBase] = bitcast<u32>(localR2);
  rowValid[localBase + 1u] = bitcast<u32>(fittedValue);
  rowValid[localBase + 2u] = bitcast<u32>(residual);
  rowValid[localBase + 3u] = bitcast<u32>(hat);
  rowValid[localBase + 4u] = status;
  rowValid[localBase + 5u] = bitcast<u32>(conditionNumber);`
      })
    );

    if (indexGridSize) {
      // The fit kernel visits only grid (included) rows, so excluded rows are written here.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fit-excluded`,
          operation: OPERATION,
          variant: 'fit-excluded',
          bindings: [
            write('rowValid', rowValid, 'u32'),
            write('coefficients', output.coefficients, 'f32')
          ],
          invocationCount: rowCount,
          declarations: common,
          body: `if (rowValid[rowValidOffset + index] != 0u) {
    return;
  }
  let nan = getNaN();
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientsOffset + index * P + column] = nan;
    ${hasInference ? 'rowValid[rowValidOffset + STANDARD_ERROR_BASE + index * P + column] = bitcast<u32>(nan);' : ''}
    ${hasTValues ? 'rowValid[rowValidOffset + T_VALUE_BASE + index * P + column] = bitcast<u32>(nan);' : ''}
  }
  let localBase = rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE;
  for (var word = 0u; word < LOCAL_STRIDE; word++) {
    rowValid[localBase + word] = bitcast<u32>(nan);
  }
  rowValid[localBase + 4u] = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED}u;`
        })
      );
    }

    // R^2 needs the total sum of squares of the rows that were fitted, known only after the final
    // fit: per-tile count, mean and M2 (two-pass inside a tile), merged in tile order (Chan).
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tile-statistics`,
        operation: OPERATION,
        variant: 'tile-statistics',
        bindings: [
          read('response', response, 'f32'),
          read('rowValid', rowValid, 'u32'),
          write('tileStatistics', tileStatistics, 'f32')
        ],
        invocationCount: tileCount,
        declarations: common,
        body: `let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var count = 0.0;
  var sum = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + LOCALS_BASE + row * LOCAL_STRIDE + 4u] == ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK}u) {
      count = count + 1.0;
      sum = sum + response[responseOffset + row];
    }
  }
  let mean = sum / max(count, 1.0);
  var m2 = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + LOCALS_BASE + row * LOCAL_STRIDE + 4u] == ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK}u) {
      let deviation = response[responseOffset + row] - mean;
      m2 = m2 + deviation * deviation;
    }
  }
  tileStatistics[tileStatisticsOffset + 3u * index] = count;
  tileStatistics[tileStatisticsOffset + 3u * index + 1u] = mean;
  tileStatistics[tileStatisticsOffset + 3u * index + 2u] = m2;`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summarize`,
        operation: OPERATION,
        variant: 'summarize',
        bindings: [read('tileStatistics', tileStatistics, 'f32'), write('summary', summary, 'f32')],
        invocationCount: 1,
        declarations: common,
        body: `var count = 0.0;
  var mean = 0.0;
  var m2 = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let tileCountValue = tileStatistics[tileStatisticsOffset + 3u * tile];
    if (tileCountValue == 0.0) {
      continue;
    }
    let tileMean = tileStatistics[tileStatisticsOffset + 3u * tile + 1u];
    let delta = tileMean - mean;
    let total = count + tileCountValue;
    mean = mean + delta * tileCountValue / total;
    m2 = m2 + tileStatistics[tileStatisticsOffset + 3u * tile + 2u] + delta * delta * count * tileCountValue / total;
    count = total;
  }
  let hasValid = summary[summaryOffset + 5u] > 0.5;
  summary[summaryOffset + 3u] = select(getNaN(), 1.0 - summary[summaryOffset] / m2, hasValid && m2 > 0.0);`
      })
    );

    if (output.bandwidthScores || output.bandwidthSingularRowCounts) {
      const present: WGSLKernelBinding[] = [read('table', candidateTable, 'f32')];
      if (output.bandwidthScores) {
        present.push(write('scores', output.bandwidthScores, 'f32'));
      }
      if (output.bandwidthSingularRowCounts) {
        present.push(write('singularRowCounts', output.bandwidthSingularRowCounts, 'u32'));
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-candidates`,
          operation: OPERATION,
          variant: 'publish-candidates',
          bindings: present,
          invocationCount: maximumBandwidthCount,
          declarations: common,
          body: `${
            isGoldenSection
              ? `// The golden-section table can hold fewer rows than the ladder capacity.
  let isRow = index < TABLE_ROWS;
  let base = tableOffset + min(index, TABLE_ROWS - 1u) * TABLE_STRIDE;`
              : 'let base = tableOffset + index * TABLE_STRIDE;'
          }
  ${output.bandwidthScores ? `scores[scoresOffset + index] = ${isGoldenSection ? 'select(getNaN(), table[base + 1u], isRow)' : 'table[base + 1u]'};` : ''}
  ${output.bandwidthSingularRowCounts ? `let singular = table[base + 2u];\n  singularRowCounts[singularRowCountsOffset + index] = select(0u, u32(singular), ${isGoldenSection ? 'isRow && ' : ''}singular >= 0.0 && singular < 4.0e9);` : ''}`
        })
      );
    }

    const publishTargets: [string, GraphDataView | undefined, 'f32' | 'u32', number][] = [
      ['localR2', output.localR2, 'f32', 0],
      ['fitted', output.fitted, 'f32', 1],
      ['residuals', output.residuals, 'f32', 2],
      ['hatDiagonal', output.hatDiagonal, 'f32', 3],
      ['localStatus', output.localStatus, 'u32', 4],
      ['localConditionNumber', output.localConditionNumber, 'f32', 5]
    ];
    const present = publishTargets.filter(([, view]) => view);
    if (present.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          variant: 'publish',
          bindings: [
            read('rowValid', rowValid, 'u32'),
            ...present.map(([name, view, type]) => write(name, view!, type))
          ],
          invocationCount: rowCount,
          declarations: common,
          body: present
            .map(([name, , type, slot]) =>
              type === 'f32'
                ? `${name}[${name}Offset + index] = bitcast<f32>(rowValid[rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE + ${slot}u]);`
                : `${name}[${name}Offset + index] = rowValid[rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE + ${slot}u];`
            )
            .join('\n  ')
        })
      );
    }

    if (hasInference) {
      const inferenceBindings: WGSLKernelBinding[] = [read('rowValid', rowValid, 'u32')];
      const statements: string[] = [];
      if (output.standardErrors) {
        inferenceBindings.push(write('standardErrors', output.standardErrors, 'f32'));
        statements.push(
          'standardErrors[standardErrorsOffset + index] = bitcast<f32>(rowValid[rowValidOffset + STANDARD_ERROR_BASE + index]);'
        );
      }
      if (output.tValues) {
        inferenceBindings.push(write('tValues', output.tValues, 'f32'));
        statements.push(
          'tValues[tValuesOffset + index] = bitcast<f32>(rowValid[rowValidOffset + T_VALUE_BASE + index]);'
        );
      }
      if (output.significant) {
        inferenceBindings.push(write('significant', output.significant, 'u32'));
        statements.push(`let tValue = bitcast<f32>(rowValid[rowValidOffset + T_VALUE_BASE + index]);
  let criticalT = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_CRITICAL_T]);
  // NaN t values and a NaN critical value are never significant.
  significant[significantOffset + index] =
    select(0u, 1u, isFiniteBits(tValue) && isFiniteBits(criticalT) && abs(tValue) > criticalT);`);
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-inference`,
          operation: OPERATION,
          variant: 'publish-inference',
          bindings: inferenceBindings,
          invocationCount: rowCount * P,
          declarations: common,
          body: statements.join('\n  ')
        })
      );
    }
    return nodes;
  }
}

/** State word indices of the golden-section search, for the kernels that read the state. */
const GPU_SEARCH_STATE_WGSL = Object.entries(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE)
  .map(([name, slot]) => `const STATE_${name}: u32 = ${slot}u;`)
  .join('\n');

const GWR_COMMON_WGSL = /* wgsl */ `
${GET_NAN_WGSL}

${IS_FINITE_BITS_WGSL}
`;

/**
 * WGSL of the ladder-refinement node (one source for every pass): `refineLadder()` advances the
 * pass cursor in word 0 of `ladders`, reads the candidate table of every earlier pass and writes
 * the ladder of the new pass into block `passCursor` of `ladders` (same layout as the caller
 * parameters, which block 0 copies).
 *
 * The best candidate is the one with the fewest singular rows, then the smallest AICc (ties to the
 * first evaluated). Its bracket is the nearest evaluated ladder values below and above it (all
 * evaluated values count, valid or not; at an end of the evaluated range the bracket is one-sided
 * and ends at the best value itself). Adaptive: every integer strictly inside the bracket except
 * the best one when there are at most `LADDER` of them, else `LADDER` integers at equal spacing.
 * Fixed: `LADDER` equally spaced values strictly inside the bracket (none when the bracket is
 * narrower than 1e-5 relative).
 */
const REFINE_LADDER_WGSL = /* wgsl */ `
fn refineLadder() {
  let nan = getNaN();
  // Advance the GPU pass cursor: this node writes the ladder of pass passCursor.
  let passCursor = u32(ladders[laddersOffset]) + 1u;
  ladders[laddersOffset] = f32(passCursor);
  let callerBase = laddersOffset + 1u;
  let isAdaptive = ladders[callerBase + 1u] > 0.5;
  var hasBest = false;
  var bestValue = 0.0;
  var bestSingular = 0.0;
  var bestScore = 0.0;
  for (var candidate = 0u; candidate < passCursor * LADDER; candidate++) {
    let base = tableOffset + candidate * TABLE_STRIDE;
    let score = table[base + 1u];
    if (!isFiniteBits(score)) {
      continue;
    }
    let singular = table[base + 2u];
    if (!hasBest || singular < bestSingular || (singular == bestSingular && score < bestScore)) {
      hasBest = true;
      bestValue = table[base];
      bestSingular = singular;
      bestScore = score;
    }
  }
  // Adaptive ladders hold integer k, so values compare after rounding.
  if (isAdaptive) {
    bestValue = floor(bestValue + 0.5);
  }
  var lower = -SENTINEL;
  var upper = SENTINEL;
  for (var candidate = 0u; candidate < passCursor * LADDER; candidate++) {
    var value = table[tableOffset + candidate * TABLE_STRIDE];
    if (!isFiniteBits(value)) {
      continue;
    }
    if (isAdaptive) {
      value = floor(value + 0.5);
    }
    if (value < bestValue) {
      lower = max(lower, value);
    } else if (value > bestValue) {
      upper = min(upper, value);
    }
  }
  if (lower <= -SENTINEL) {
    lower = bestValue;
  }
  if (upper >= SENTINEL) {
    upper = bestValue;
  }
  let base = callerBase + passCursor * PARAMETER_LENGTH;
  for (var slot = 0u; slot < PARAMETER_LENGTH; slot++) {
    ladders[base + slot] = 0.0;
  }
  ladders[base] = ladders[callerBase];
  ladders[base + 1u] = ladders[callerBase + 1u];
  ladders[base + 3u] = ladders[callerBase + 3u];
  var count = 0u;
  if (hasBest) {
    if (isAdaptive) {
      let first = max(lower + 1.0, 2.0);
      let last = min(upper - 1.0, f32(MAXIMUM_NEIGHBORS));
      let isInside = first <= bestValue && bestValue <= last;
      let integers = max(last - first + 1.0 - select(0.0, 1.0, isInside), 0.0);
      if (integers > 0.0) {
        count = min(u32(integers), LADDER);
        for (var slot = 0u; slot < count; slot++) {
          var position = f32(slot);
          if (integers > f32(LADDER)) {
            position = floor((f32(slot) + 0.5) * integers / f32(LADDER));
          }
          var value = first + position;
          if (isInside && value >= bestValue) {
            value = value + 1.0;
          }
          ladders[base + HEADER + slot] = value;
        }
      }
    } else {
      let low = max(lower, 0.0);
      let span = upper - low;
      if (span > 1.0e-5 * max(abs(upper), 1.0e-30)) {
        count = LADDER;
        for (var slot = 0u; slot < LADDER; slot++) {
          ladders[base + HEADER + slot] = low + span * f32(slot + 1u) / f32(LADDER + 1u);
        }
      }
    }
  }
  ladders[base + 2u] = f32(count);
}
`;
