// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Largest predictor count of `GPUSpatialErrorGM`: 15, the limit of the first-stage
 * `GPUOrdinaryLeastSquares` and of the 16-wide Cholesky of the filtered system (intercept
 * included). No kernel binds more than 8 storage buffers, whatever the predictor count.
 */
export const GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT = 15;

/** Estimator choices of `GPUSpatialErrorGM`. */
export type GPUSpatialErrorGMEstimator = 'homoskedastic' | 'kelejian-prucha';

/** Default estimator of `GPUSpatialErrorGM`. */
export const GPU_SPATIAL_ERROR_GM_DEFAULT_ESTIMATOR: GPUSpatialErrorGMEstimator = 'homoskedastic';

/** Fraction of the stationarity bound `1 / ||W||_inf` that `lambda` may reach (spreg's 0.99). */
export const GPU_SPATIAL_ERROR_GM_LAMBDA_BOUND_FRACTION = 0.99;

/** Number of float32 values per `table` row: coefficient, standard error, z statistic, p-value. */
export const GPU_SPATIAL_ERROR_GM_TABLE_STRIDE = 4;

/** Number of float32 values `output.summary` must hold. */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH = 13;
/** `summary` slot: number of rows `n`, stored as a float32 integer. */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_ROW_COUNT = 0;
/**
 * `summary` slot: spatial error parameter `lambda`: the weighted second-step estimate of
 * `homoskedastic` (spreg `GM_Error_Hom` `lambda2`) or the Kelejian-Prucha estimate (spreg
 * `GM_Error` `lambda1`).
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA = 1;
/**
 * `summary` slot: `sigma2 = e'e / n` (spreg `sig2` = `sig2n`) with `e = u - lambda W u` the filtered
 * residuals: of the FGLS fit at `lambda` for `kelejian-prucha`, of the FGLS residuals `u1` filtered at the
 * final `lambda` for `homoskedastic` (`u_s'u_s / n`). It scales the standard errors of the betas.
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_SIGMA_SQUARED = 2;
/** `summary` slot: pseudo R squared, the squared correlation of `y` and `X b` (spreg `pr2`). */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_PSEUDO_R_SQUARED = 3;
/** `summary` slot: residual sum of squares `u'u` of the unfiltered residuals `u = y - X b`. */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_RESIDUAL_SUM_OF_SQUARES = 4;
/**
 * `summary` slot: value of the moment objective at the solution: the scaled sum of squared moment
 * residuals of `kelejian-prucha`, or the weighted criterion `(G p - g)' psi^-1 (G p - g)` of the
 * second step of `homoskedastic`.
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_MOMENT_OBJECTIVE = 5;
/**
 * `summary` slot: Belsley condition number of the final spatially filtered regression, the
 * `sqrt(lambdaMax / lambdaMin)` of its raw filtered `[1, X]` Gram matrix scaled to unit diagonal.
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER = 6;
/**
 * `summary` slot: `lambdaMax / lambdaMin` of the unit-diagonal Gram matrix of the filtered
 * regression (filtered constant first) that the float32 Cholesky factors.
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER = 7;
/** `summary` slot: largest variance inflation factor over the filtered predictor columns. */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR = 8;
/**
 * `summary` slot: 1 when the correlation condition number exceeds
 * `GPU_REGRESSION_PRECISION_CONDITION_LIMIT` (1e4), else 0. Never changes `status`.
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_PRECISION_WARNING = 9;
/**
 * `summary` slot: the bound `0.99 / ||W||_inf` that `lambda` was searched within (`||W||_inf` is
 * the largest absolute row sum), NaN if `W` has no links. For a row-standardized `W` this is 0.99.
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_BOUND = 10;
/**
 * `summary` slot: 1 when the `homoskedastic` variance of `lambda` uses stochastic (Hutchinson)
 * traces of `W` (more than 4096 rows by default), so its standard error is approximate; 0 when the
 * traces are exact or the estimator is `kelejian-prucha`. Betas and `lambda` are not approximated.
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION = 11;
/**
 * `summary` slot: 1 when `|lambda|` reached the search bound (`lambda` is a boundary solution and
 * its standard error is not meaningful), else 0.
 */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_AT_BOUND = 12;

/** `status` value: fit succeeded. */
export const GPU_SPATIAL_ERROR_GM_STATUS_OK = 0;
/** `status` value: the first-stage OLS or the filtered regression is singular or ill-conditioned. */
export const GPU_SPATIAL_ERROR_GM_STATUS_SINGULAR = 1;
/** `status` value: too few rows, `n <= predictorCount + 2`. */
export const GPU_SPATIAL_ERROR_GM_STATUS_TOO_FEW_ROWS = 2;
/** `status` value: a non-finite input or moment, or zero OLS residuals. */
export const GPU_SPATIAL_ERROR_GM_STATUS_NON_FINITE = 3;
