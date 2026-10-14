// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Largest predictor count of `GPUSpatialLagMaximumLikelihood`: 14. The variance step factors the
 * `(predictorCount + 2)`-wide information matrix of the intercept, the predictors and `rho` (the
 * `sigma2` row is eliminated by a Schur complement) with the 16-wide Cholesky.
 */
export const GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT = 14;

/**
 * Largest predictor count of `GPUSpatialErrorMaximumLikelihood`: 15, the limit of the 16-wide
 * Cholesky of the filtered normal equations (intercept included).
 */
export const GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT = 15;

/** Default series length `Q` of the log-determinant and trace series. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER = 128;

/** Smallest `maximumPower`: the probe pass needs at least one Horner step. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_MINIMUM_POWER = 2;

/** Grid intervals scanned for stationary points of the concentrated likelihood. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SCAN_STEPS = 1024;

/** Bisection steps that refine each bracketed maximum (below float32 resolution of the bracket). */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BISECTION_STEPS = 32;

/** Fraction of the stationarity bound `1 / ||W||_inf` that `rho` or `lambda` may reach. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION = 0.99;

/**
 * Largest tolerated log-determinant truncation error per observation: the parameter interval is
 * shortened (see {@link getSpatialMaximumLikelihoodSeriesLimit}) until the worst-case series
 * remainder is below `1e-5 n` log-likelihood units.
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SERIES_TOLERANCE = 1e-5;

/** Number of float32 values per `table` row: coefficient, standard error, z statistic, p-value. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE = 4;

/** Number of float32 values `output.summary` must hold. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH = 22;
/** `summary` slot: number of rows `n`, stored as a float32 integer. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT = 0;
/** `summary` slot: the spatial parameter, `rho` (lag) or `lambda` (error). */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER = 1;
/** `summary` slot: `sigma2 = u'u / n` (lag, spreg `sig2`) or `e'e / n` of the filtered residuals (error). */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED = 2;
/**
 * `summary` slot: full Gaussian log-likelihood `-n/2 ln(sigma2) + ln|I - rho W| - n/2 (ln 2 pi + 1)`
 * (spreg `logll`).
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD = 3;
/**
 * `summary` slot: Akaike criterion `2 k - 2 logLikelihood` with spreg's `k`: the number of table
 * rows for the lag model (intercept, predictors, `rho`) and the number of intercept and predictor
 * rows only for the error model (spreg `ML_Error` does not count `lambda`).
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AIC = 4;
/** `summary` slot: Schwarz criterion `k ln(n) - 2 logLikelihood` with the same `k`. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SCHWARZ = 5;
/** `summary` slot: pseudo R squared, the squared correlation of `y` and `y - u` (spreg `pr2`). */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED = 6;
/**
 * `summary` slot: spatial pseudo R squared, the squared correlation of `y` and the reduced-form
 * prediction `(I - rho W)^-1 X b` (spreg `pr2_e`). NaN for the error model.
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL = 7;
/** `summary` slot: residual sum of squares `u'u` (lag: of `u = e0 - rho e1`; error: of `y - X b`). */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES = 8;
/** `summary` slot: Belsley condition number of the design (lag) or of the filtered design (error). */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER = 9;
/** `summary` slot: `lambdaMax / lambdaMin` of the factored (unit-diagonal) predictor matrix. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER = 10;
/** `summary` slot: largest variance inflation factor of the predictors. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR = 11;
/**
 * `summary` slot: 1 when the correlation condition number exceeds
 * `GPU_REGRESSION_PRECISION_CONDITION_LIMIT` (1e4), else 0. Never changes `status`.
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PRECISION_WARNING = 12;
/**
 * `summary` slot: the bound the parameter was searched within, `min(0.99, seriesLimit) / ||W||_inf`
 * with `||W||_inf` the largest absolute row sum; NaN if `W` has no links.
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND = 13;
/**
 * `summary` slot: 1 when the traces of `W` are Hutchinson estimates (more than 4096 rows by
 * default): the log-likelihood, and with it the estimate, then carry stochastic error. 0 when exact.
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION = 14;
/** `summary` slot: 1 when the maximum is an interior stationary point (the optimizer converged). */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED = 15;
/**
 * `summary` slot: 1 when the maximum lies on the search bound, so the estimate is a boundary
 * solution and its standard error is not meaningful.
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY = 16;
/** `summary` slot: 1 when the series limit (not `0.99 / ||W||_inf`) shortened the search interval. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED = 17;
/**
 * `summary` slot: worst-case log-determinant truncation error at the estimate,
 * `getSpatialLogDeterminantTruncationBound(parameter, Q, n, ||W||_inf)`, in log-likelihood units.
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND = 18;
/** `summary` slot: 1 when the information matrix was positive definite (standard errors defined). */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED = 19;
/**
 * `summary` slot: number of local maxima of the concentrated likelihood found on the grid
 * (interior stationary points plus boundary maxima). More than 1 means a multimodal likelihood.
 */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOCAL_MAXIMUM_COUNT = 20;
/** `summary` slot: `ln|I - rho W|` (series) at the estimate. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT = 21;

/** `status` value: fit succeeded. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_OK = 0;
/** `status` value: the design, or the filtered design, is singular or ill-conditioned. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_SINGULAR = 1;
/** `status` value: too few rows, `n <= predictorCount + 2`. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_TOO_FEW_ROWS = 2;
/** `status` value: a non-finite input, weight or moment, or no admissible parameter interval. */
export const GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE = 3;

/**
 * Largest `|rho| * ||W||_inf` at which the truncated log-determinant series of length
 * `maximumPower` keeps its worst-case error below `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SERIES_TOLERANCE`
 * per observation, `s^(Q+1) / ((Q+1) (1 - s)) <= 1e-5` (the geometric tail bound of
 * `getSpatialLogDeterminantTruncationBound`, divided by `n`). Solved by bisection, at most
 * `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION`.
 *
 * About 0.93 for `Q = 128` and 0.96 for `Q = 256`. Truncating the series biases the log-determinant
 * upward for positive `rho` and would pull the maximum toward the end of the interval, so the
 * search interval stops where the remainder can no longer be ignored; beyond it the fit reports a
 * boundary solution.
 */
export function getSpatialMaximumLikelihoodSeriesLimit(maximumPower: number): number {
  const remainder = (s: number) => s ** (maximumPower + 1) / ((maximumPower + 1) * (1 - s));
  if (
    remainder(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION) <=
    GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SERIES_TOLERANCE
  ) {
    return GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION;
  }
  let low = 0;
  let high = GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION;
  for (let iteration = 0; iteration < 60; iteration++) {
    const middle = 0.5 * (low + high);
    if (remainder(middle) <= GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SERIES_TOLERANCE) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

/**
 * Derivative `d/d rho ln|I - rho W| = -s sum_{q=1}^{Q} t^(q-1) tr((W/s)^q)` of the truncated series
 * (Horner), `t = rho * s`. `powerTraces` are the traces of `W / s` and `infinityNorm` is `s`
 * (default 1, for unscaled traces). The negative of the first trace term `tr(W (I - rho W)^-1)`
 * of the information matrix.
 */
export function getSpatialLogDeterminantDerivativeFromTraces(
  rho: number,
  powerTraces: ArrayLike<number>,
  infinityNorm = 1
): number {
  const t = rho * infinityNorm;
  let sum = 0;
  for (let q = powerTraces.length; q >= 1; q--) {
    sum = sum * t + powerTraces[q - 1];
  }
  return -infinityNorm * sum;
}

/**
 * Truncated `tr(W (I - rho W)^-1) = s sum_{q=1}^{Q} t^(q-1) tr((W/s)^q)` (the `tr1` of spreg),
 * `t = rho * s`.
 */
export function getSpatialMaximumLikelihoodTraceOne(
  rho: number,
  powerTraces: ArrayLike<number>,
  infinityNorm = 1
): number {
  return -getSpatialLogDeterminantDerivativeFromTraces(rho, powerTraces, infinityNorm);
}

/**
 * Truncated `tr((W (I - rho W)^-1)^2) = s^2 sum_{m=0}^{Q-2} (m + 1) t^m tr((W/s)^(m+2))` (the
 * `tr2` of spreg), `t = rho * s`, from `(I - rho W)^-2 = sum (m + 1) rho^m W^m`.
 */
export function getSpatialMaximumLikelihoodTraceTwo(
  rho: number,
  powerTraces: ArrayLike<number>,
  infinityNorm = 1
): number {
  const t = rho * infinityNorm;
  let sum = 0;
  for (let m = powerTraces.length - 2; m >= 0; m--) {
    sum = sum * t + (m + 1) * powerTraces[m + 1];
  }
  return infinityNorm * infinityNorm * sum;
}
