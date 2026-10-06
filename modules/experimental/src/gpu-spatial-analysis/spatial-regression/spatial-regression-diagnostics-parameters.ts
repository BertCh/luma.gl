// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 values `output.tests` of `GPUSpatialRegressionDiagnostics` must hold. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH = 18;

/** Number of float32 values per test row in `output.tests`: statistic, degrees of freedom, p-value. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE = 3;

/** `tests` row: Lagrange multiplier test for a spatially lagged dependent variable (df 1). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_LAG = 0;
/** `tests` row: Lagrange multiplier test for spatially autocorrelated errors (df 1). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_ERROR = 1;
/** `tests` row: robust Lagrange multiplier test for a lag, robust to error dependence (df 1). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_LAG = 2;
/** `tests` row: robust Lagrange multiplier test for error dependence, robust to a lag (df 1). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_ERROR = 3;
/** `tests` row: Lagrange multiplier test for a lag and error dependence together (df 2). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_SARMA = 4;
/**
 * `tests` row: Moran's I of the residuals. The statistic is the standard normal `z`, the degrees
 * of freedom slot is 0 (not a chi-square), and the p-value is two-sided.
 */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_MORAN_RESIDUALS = 5;

/** Number of float32 values `output.summary` of `GPUSpatialRegressionDiagnostics` must hold. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH = 12;

/** `summary` slot: number of rows `n`, stored as a float32 integer. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_ROW_COUNT = 0;
/** `summary` slot: residual variance `e'e / n` (spreg `sig2n`). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_SIGMA_SQUARED = 1;
/** `summary` slot: `T = tr(W'W + WW)`. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_TRACE = 2;
/** `summary` slot: `J = (WXb)' M (WXb) / sigma2 + T`, the information term of the lag test. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_INFORMATION = 3;
/** `summary` slot: weights sum `S0`. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_WEIGHTS_SUM = 4;
/** `summary` slot: Moran's I of the residuals, `(n / S0) e'We / e'e`. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_I = 5;
/** `summary` slot: expectation of Moran's I under the null, `-(n / S0) tr((X'X)^-1 X'WX) / (n - p)`. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_EXPECTATION = 6;
/** `summary` slot: variance of Moran's I under the null (Cliff and Ord, regression residuals). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_VARIANCE = 7;
/** `summary` slot: Moran's I z score, `(I - E[I]) / sqrt(Var[I])`. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_Z = 8;
/** `summary` slot: two-sided normal p-value of the Moran's I z score. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_P_VALUE = 9;
/** `summary` slot: `e'We`, the residual spatial cross-product. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_RESIDUAL_LAG_PRODUCT = 10;
/** `summary` slot: `e'Wy`, the residual cross-product with the lagged response. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_RESPONSE_LAG_PRODUCT = 11;

/** `status` value: diagnostics computed. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_STATUS_OK = 0;
/** `status` value: the (centered) design is singular or ill-conditioned. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_STATUS_SINGULAR = 1;
/** `status` value: too few rows, `n <= predictorCount + 1`. */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_STATUS_TOO_FEW_ROWS = 2;
/** `status` value: degenerate weights (`S0 <= 0` or `T <= 0`, for example no neighbors at all). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_STATUS_DEGENERATE_WEIGHTS = 3;
/** `status` value: a non-finite input, or residuals that are exactly zero (a perfect fit). */
export const GPU_SPATIAL_REGRESSION_DIAGNOSTICS_STATUS_NON_FINITE = 4;
