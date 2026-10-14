// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Largest predictor count (columns of `predictors`, without the intercept) of
 * `GPUGeographicallyWeightedRegression`: 11, a design of 12 columns with the intercept (it was 7).
 *
 * The 8-storage-binding limit does not depend on the predictor count: no kernel binds more than
 * eight storage buffers (the heaviest bind seven), and per-row standard errors share the row-state
 * buffer. The limit is per-thread private memory: the fit holds the `P x P` weighted Gram matrix,
 * the `P x P` squared-weight Gram matrix (standard errors) and `P`-wide solve vectors, about
 * `2 P^2 + 6 P` floats, 360 at `P = 12`, beside the 128-entry neighbour list. That stays within the
 * 16-column limit of the shared Cholesky solver. Not exercised on hardware at the cap: lower this
 * constant if an adapter rejects or slows the shader at 11.
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT = 11;

/** Default two-sided significance level of the adjusted-alpha critical value (mgwr uses 0.05). */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_SIGNIFICANCE_LEVEL = 0.05;

/** Default number of bandwidth refinement passes after the caller ladder (see `refinementPassCount`). */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_REFINEMENT_PASS_COUNT = 2;

/** Largest compile-time number of bandwidth refinement passes. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_REFINEMENT_PASS_COUNT = 4;

/** Default compile-time number of golden-section search iterations (see `maximumSearchIterations`). */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_MAXIMUM_SEARCH_ITERATIONS = 30;

/** Largest compile-time number of golden-section search iterations. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS = 64;

/**
 * Section constant of the golden-section search, mgwr's `delta = 0.38197` (`1 - (sqrt(5) - 1) / 2`
 * to five digits): the interior points are `a + delta (c - a)` and `c - delta (c - a)`.
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA = 0.38197;

/** mgwr's absolute convergence tolerance on the AICc difference of the two interior points. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE = 1e-6;

/**
 * Relative floor of the golden-section convergence tolerance: the search stops when the two interior
 * AICc values differ by no more than `max(1e-6, 4e-6 max(|AICc_b|, |AICc_d|))`. The AICc sums `n` float32
 * terms, so differences below a few times `n 2^-24 |AICc|` are rounding noise and cannot steer
 * the search; mgwr's float64 `1e-6` alone would never fire at AICc values of hundreds.
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE = 4e-6;

/**
 * Recommended `gaussianCutoff`: a Gaussian weight at 5 bandwidths is `exp(-12.5) = 3.7e-6`. For a
 * locally uniform density the omitted share of the weight sum is that same `3.7e-6` and of the
 * second moment `(1 + 12.5) 3.7e-6 = 5e-5`, below float32 accumulation error of the fits.
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_RECOMMENDED_GAUSSIAN_CUTOFF = 5;

/** Largest compile-time bandwidth ladder length. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH = 32;

/** Largest compile-time `k` of adaptive (k-th nearest neighbour) bandwidths. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT = 128;

/**
 * Largest row count without a grid index: every location scans every row, so the cost grows with
 * `rows^2 * ladder`.
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT = 65536;

/**
 * Largest row count with a grid index. Bisquare fits (fixed or adaptive bandwidth) then visit only
 * nearby cells, so the cost grows with `rows * neighbours`; the limit is the per-candidate scratch
 * (`rows * ladder * 8` bytes must fit one 128 MiB storage binding) and `rows^2` cost remains for
 * Gaussian kernels.
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT = 1048576;

/** Smallest row count for which the grid index is built by default. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT = 1024;

/** Largest default grid dimension per axis. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION = 1024;

/** Header float32 slots before the ladder in the parameter view. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH = 4;

/**
 * Multiplier applied to the k-th nearest neighbour distance of adaptive bandwidths so that the
 * k-th neighbour keeps a small positive bisquare weight (mgwr uses 1.0000001; f32 needs more).
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR = 1.00001;

/** Smallest residual variance `RSS / n` used inside the AICc logarithm. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE = 1e-30;

/** Kernel codes stored in parameter slot 0. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_KERNEL = {
  /** `w = exp(-0.5 (d / h)^2)`. */
  gaussian: 0,
  /** `w = (1 - (d / h)^2)^2` for `d < h`, else 0. */
  bisquare: 1
} as const;

/** Bandwidth mode codes stored in parameter slot 1. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_BANDWIDTH_MODE = {
  /** Ladder values are distances `h` in position units. */
  fixed: 0,
  /** Ladder values are neighbour counts `k`; `h` is the distance to the k-th nearest row. */
  adaptive: 1
} as const;

/** Values of the `localStatus` output. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS = {
  /** The local system was solved. */
  OK: 0,
  /**
   * The local weighted system was singular, ill-conditioned, or the bandwidth was invalid. All
   * per-row results of the row (coefficients, standard errors, fitted value, residual, hat) are NaN.
   */
  SINGULAR: 1,
  /** The row is masked out or has a non-finite position, predictor or response. */
  EXCLUDED: 2
} as const;

/**
 * Float32 slots of the `summary` output. Statistics are taken over the rows whose local fit at the
 * selected bandwidth succeeded (all included rows unless some are singular).
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY = {
  /** Residual sum of squares at the selected bandwidth. */
  RESIDUAL_SUM_OF_SQUARES: 0,
  /** Trace of the hat matrix, `sum(S_ii)`: the effective number of parameters (mgwr `ENP`, `tr_S`). */
  TRACE_OF_HAT: 1,
  /** Corrected Akaike information criterion of the selected bandwidth (mgwr `aicc`). */
  AICC: 2,
  /** `1 - RSS / TSS` over the non-singular rows (global, not the mean of local R^2; mgwr `R2`). */
  R_SQUARED: 3,
  /** Number of rows `n` entering the statistics: included rows minus singular rows. */
  OBSERVATION_COUNT: 4,
  /** 1 when at least one ladder candidate had a finite AICc, otherwise 0. */
  HAS_VALID_CANDIDATE: 5,
  /** Included rows whose local system was singular at the selected bandwidth. */
  SINGULAR_ROW_COUNT: 6,
  /** `RSS / (n - tr(S))`, the residual variance the standard errors use (mgwr `sigma2`). */
  SIGMA_SQUARED: 7,
  /**
   * Da Silva-Fotheringham adjusted significance level `alpha * k / tr(S)` with `k` the design
   * columns including the intercept (mgwr `adj_alpha` at the requested level, 0.05 by default).
   */
  ADJUSTED_ALPHA: 8,
  /**
   * Critical absolute t value `t.ppf(1 - adjustedAlpha / 2, n - 1)` (mgwr `critical_tval`); 0 when
   * the adjusted alpha reaches 1.
   */
  CRITICAL_T: 9,
  /** Gaussian log-likelihood `-n / 2 (ln(2 pi RSS / n) + 1)` (mgwr `llf`). */
  LOG_LIKELIHOOD: 10,
  /** Included rows (unmasked, finite), singular or not. */
  INCLUDED_ROW_COUNT: 11
} as const;

/** Number of float32 elements in the `summary` output. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH = 12;

/** Float32 fields of one candidate in the `candidateTable` output, in order. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE = {
  /** Ladder value evaluated (distance for fixed bandwidths, neighbour count for adaptive ones). */
  VALUE: 0,
  /** AICc over the non-singular rows; NaN for an invalid candidate or an unused slot. */
  AICC: 1,
  /** Included rows whose local system was singular. */
  SINGULAR_ROW_COUNT: 2,
  /** Residual sum of squares over the non-singular rows. */
  RESIDUAL_SUM_OF_SQUARES: 3,
  /** Trace of the hat matrix over the non-singular rows. */
  TRACE_OF_HAT: 4,
  /** Rows entering the AICc (included minus singular). */
  OBSERVATION_COUNT: 5,
  /** Included rows. */
  INCLUDED_ROW_COUNT: 6
} as const;

/** Float32 values per candidate of the `candidateTable` output. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE = 7;

/** Float32 slots of the optional `searchState` output of the golden-section search. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE = {
  /** Lower end `a` of the bracket after the last iteration. */
  LOWER: 0,
  /** Lower interior point `b`. */
  LOWER_INTERIOR: 1,
  /** Upper end `c` of the bracket. */
  UPPER: 2,
  /** Upper interior point `d`. */
  UPPER_INTERIOR: 3,
  /** Candidate table row of the evaluation of `b`. */
  LOWER_INTERIOR_ROW: 4,
  /** Candidate table row of the evaluation of `d`. */
  UPPER_INTERIOR_ROW: 5,
  /** Iterations whose two interior scores were compared (mgwr's iteration count). */
  ITERATION_COUNT: 6,
  /** 1 while the search is armed, 0 once it stopped. */
  ACTIVE: 7,
  /** Candidate table row of the optimum of the last iteration, -1 before the first. */
  OPTIMUM_ROW: 8,
  /** Candidate table rows used (evaluations scheduled). */
  EVALUATION_COUNT: 9,
  /** 1 when the search stopped on its tolerance, 0 when it ran out of iterations. */
  CONVERGED: 10,
  /** Unused. */
  RESERVED: 11
} as const;

/** Number of float32 elements in the `searchState` output. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH = 12;

/** Kernel names accepted by {@link getGPUGeographicallyWeightedRegressionParameterValues}. */
export type GPUGeographicallyWeightedRegressionKernel =
  keyof typeof GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_KERNEL;

/** Bandwidth mode names accepted by {@link getGPUGeographicallyWeightedRegressionParameterValues}. */
export type GPUGeographicallyWeightedRegressionBandwidthMode =
  keyof typeof GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_BANDWIDTH_MODE;

/** CPU description of the per-frame parameters of `GPUGeographicallyWeightedRegression`. */
export type GPUGeographicallyWeightedRegressionSettings = {
  /** Weight kernel. Defaults to `'bisquare'`. */
  kernel?: GPUGeographicallyWeightedRegressionKernel;
  /** Meaning of the ladder values. Defaults to `'fixed'`. */
  bandwidthMode?: GPUGeographicallyWeightedRegressionBandwidthMode;
  /**
   * Candidate bandwidths: distances (`'fixed'`) or neighbour counts `k` (`'adaptive'`, rounded to
   * integers). One value is a fixed bandwidth; several are searched by AICc (and refined, see
   * `refinementPassCount`). With `bandwidthSearch: 'golden-section'` the first two values are the
   * search bounds `[minimum, maximum]` (a value of 0 or less takes mgwr's default for that bound);
   * further values are ignored.
   */
  bandwidths: ArrayLike<number>;
  /**
   * Two-sided significance level of the adjusted alpha and critical t value, in (0, 1). Defaults to
   * 0.05. Per frame.
   */
  significanceLevel?: number;
};

/**
 * Returns the float32 length of a parameter view for a ladder capacity.
 *
 * @param maximumBandwidthCount Compile-time ladder length of the contributor.
 */
export function getGPUGeographicallyWeightedRegressionParameterLength(
  maximumBandwidthCount: number = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH
): number {
  return GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH + maximumBandwidthCount;
}

/**
 * Packs per-frame `GPUGeographicallyWeightedRegression` parameters.
 *
 * Layout (float32): `[kernel, bandwidthMode, candidateCount, significanceLevel (0 = default), ladder...]`. Every value can change
 * between encodings without rebuilding the graph.
 *
 * @param settings Kernel, bandwidth mode and ladder.
 * @param maximumBandwidthCount Ladder capacity of the contributor (its `maximumBandwidthCount`).
 * @param target Optional destination of at least the parameter length.
 * @throws If the ladder is empty, longer than the capacity, or has a non-finite value, or a name is
 * not recognized, or `significanceLevel` is outside (0, 1), or `target` is too short.
 */
export function getGPUGeographicallyWeightedRegressionParameterValues(
  settings: GPUGeographicallyWeightedRegressionSettings,
  maximumBandwidthCount: number = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  target: Float32Array = new Float32Array(
    getGPUGeographicallyWeightedRegressionParameterLength(maximumBandwidthCount)
  )
): Float32Array {
  const length = getGPUGeographicallyWeightedRegressionParameterLength(maximumBandwidthCount);
  if (target.length < length) {
    throw new Error(`GWR parameter target must hold ${length} elements`);
  }
  const {bandwidths} = settings;
  if (bandwidths.length < 1 || bandwidths.length > maximumBandwidthCount) {
    throw new Error(`GWR bandwidths must hold 1 to ${maximumBandwidthCount} values`);
  }
  const kernel = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_KERNEL[settings.kernel ?? 'bisquare'];
  const mode =
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_BANDWIDTH_MODE[settings.bandwidthMode ?? 'fixed'];
  if (kernel === undefined || mode === undefined) {
    throw new Error('GWR kernel or bandwidth mode is not recognized');
  }
  target.fill(0, 0, length);
  target[0] = kernel;
  target[1] = mode;
  target[2] = bandwidths.length;
  const level = settings.significanceLevel;
  if (level !== undefined) {
    if (!(level > 0 && level < 1)) {
      throw new Error('GWR significanceLevel must be in (0, 1)');
    }
    target[3] = level;
  }
  for (let index = 0; index < bandwidths.length; index++) {
    if (!Number.isFinite(bandwidths[index])) {
      throw new Error('GWR bandwidths must be finite');
    }
    target[GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH + index] =
      bandwidths[index];
  }
  return target;
}
