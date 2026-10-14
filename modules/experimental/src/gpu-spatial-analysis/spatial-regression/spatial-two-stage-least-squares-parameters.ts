// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Largest predictor count of `GPUSpatialTwoStageLeastSquares` with `instrumentOrder: 1`. The
 * instruments are `[1, X, W X]`; the constant is centered out, so the solve factors the
 * `(order + 1) k` centered instrument columns with the 16-wide Cholesky: `k <= 16 / (order + 1)`,
 * that is 8 for order 1 and 5 for order 2 (see
 * {@link getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount}). No kernel binds more than 8
 * storage buffers.
 */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT = 8;

/** Largest predictor count for an instrument order: `floor(16 / (order + 1))`, 8 or 5. */
export function getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount(
  instrumentOrder: 1 | 2
): number {
  return Math.floor(16 / (instrumentOrder + 1));
}

/** Number of float32 values per `table` row: coefficient, standard error, z statistic, p-value. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE = 4;

/** Number of float32 values `output.summary` must hold. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH = 16;
/** `summary` slot: number of rows `n`, stored as a float32 integer. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ROW_COUNT = 0;
/** `summary` slot: residual variance `u'u / n` (spreg `sig2n`, which `GM_Lag` uses for its covariance by default). */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED = 1;
/** `summary` slot: residual sum of squares `u'u` of the structural residuals `u = y - Z delta`. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES = 2;
/** `summary` slot: pseudo R squared, the squared correlation of `y` and `y - u` (spreg `pr2`). */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PSEUDO_R_SQUARED = 3;

/** `summary` slot: Moran's I of the structural residuals, `(n / S0) u'Wu / u'u` (spreg `AKtest` `mi`). */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MORAN_I = 4;
/** `summary` slot: Anselin-Kelejian statistic for residual spatial autocorrelation (chi-square, 1 df). */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN = 5;
/** `summary` slot: p-value of the Anselin-Kelejian statistic. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN_P_VALUE = 6;

/**
 * `summary` slot: Belsley condition number of the second-stage system, `sqrt(lambdaMax /
 * lambdaMin)` of the raw Gram matrix of `[1, P Z]` (`Z = [X, W y]`, `P` the projection on the
 * instruments) scaled to unit diagonal, equivalently the condition number of that design with
 * unit-length columns. Computed from the correlation Cholesky factor and the column means with
 * `getBelsleyConditionNumberWGSL`, not from the float32 Gram matrix, so it stays accurate into
 * the thousands. Values above about 30 indicate harmful collinearity.
 */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER = 7;
/**
 * `summary` slot: `lambdaMax / lambdaMin` of the unit-diagonal second-stage matrix `Z'PZ` that the
 * float32 Cholesky factors (intercept centered out).
 */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER = 8;
/** `summary` slot: largest variance inflation factor, the largest diagonal of the inverse of the unit-diagonal `Z'PZ`. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR = 9;
/**
 * `summary` slot: 1 when the second-stage correlation condition number or the instrument
 * correlation condition number exceeds `GPU_REGRESSION_PRECISION_CONDITION_LIMIT` (1e4), else 0.
 * Never changes `status`.
 */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PRECISION_WARNING = 10;
/**
 * `summary` slot: `lambdaMax / lambdaMin` of the unit-diagonal Gram matrix of the centered
 * instruments `[X, W X (, W^2 X)]` (the instrument-projection system).
 */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER = 11;
/**
 * `summary` slot: 1 when `|rho| * ||W||_inf >= 1`, that is when `rho` is outside the interval
 * where `I - rho W` is guaranteed invertible and the reduced form `(I - rho W)^-1` exists as a
 * convergent series (the interval is `(-1, 1)` for a row-standardized `W`; for other `W` the
 * check is sufficient, not necessary). 2SLS is an unconstrained estimator, so an estimate outside
 * the interval is reported rather than failed: it is data about the fit (and the model) that
 * impacts and multipliers need, not a numerical error. `status` is unchanged.
 */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RHO_OUTSIDE_ADMISSIBLE_RANGE = 12;
/** `summary` slot: `||W||_inf`, the largest absolute row sum of `W` (1 for a row-standardized `W`). */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM = 13;
/**
 * `summary` slot: first-stage strength, the classical (non-robust) joint F statistic for the
 * excluded instruments `W X` (and `W^2 X`) in the first-stage regression of `W y` on
 * `[1, X, W X, ...]`: `((ESS_full - ESS_restricted) / (m - k)) / (RSS_full / (n - m - 1))` with `m`
 * the instrument columns and `k` the predictors. The Staiger-Stock rule of thumb flags `F < 10` as
 * weak. Homoskedastic errors are assumed (no robust F).
 */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F = 14;
/**
 * `summary` slot: partial R squared of the excluded instruments in the first stage,
 * `(ESS_full - ESS_restricted) / (TSS - ESS_restricted)` (Bound, Jaeger and Baker): the share of
 * the variance of `W y` left by `[1, X]` that `W X` (and `W^2 X`) explains.
 */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_PARTIAL_R_SQUARED = 15;

/** `status` value: fit succeeded. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_OK = 0;
/** `status` value: the instruments or the projected regressors are singular or ill-conditioned. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_SINGULAR = 1;
/** `status` value: too few rows, `n <= predictorCount + 2`. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_TOO_FEW_ROWS = 2;
/** `status` value: a non-finite input or moment. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_NON_FINITE = 3;
