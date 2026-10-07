// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Largest predictor count of `GPUSpatialTwoStageLeastSquares` (the instrument set has `(order + 1) k` columns). */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT = 8;

/** Number of float32 values per `table` row: coefficient, standard error, z statistic, p-value. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE = 4;

/** Number of float32 values `output.summary` must hold. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH = 7;
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

/** `status` value: fit succeeded. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_OK = 0;
/** `status` value: the instruments or the projected regressors are singular or ill-conditioned. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_SINGULAR = 1;
/** `status` value: too few rows, `n <= predictorCount + 2`. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_TOO_FEW_ROWS = 2;
/** `status` value: a non-finite input or moment. */
export const GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_NON_FINITE = 3;
