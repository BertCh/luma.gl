// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Largest predictor count of `GPUSpatialErrorGM`. */
export const GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT = 8;

/** Number of float32 values per `table` row: coefficient, standard error, z statistic, p-value. */
export const GPU_SPATIAL_ERROR_GM_TABLE_STRIDE = 4;

/** Number of float32 values `output.summary` must hold. */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH = 6;
/** `summary` slot: number of rows `n`, stored as a float32 integer. */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_ROW_COUNT = 0;
/** `summary` slot: spatial error parameter `lambda` of the moment estimator (spreg `lambda1`). */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA = 1;
/** `summary` slot: `e'e / n` of the filtered residuals `e = u - lambda W u` (spreg `sig2` = `sig2n`). */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_SIGMA_SQUARED = 2;
/** `summary` slot: pseudo R squared, the squared correlation of `y` and `X b` (spreg `pr2`). */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_PSEUDO_R_SQUARED = 3;
/** `summary` slot: residual sum of squares `u'u` of the unfiltered residuals `u = y - X b`. */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_RESIDUAL_SUM_OF_SQUARES = 4;
/** `summary` slot: value of the moment objective at the solution (sum of squared moment residuals, scaled). */
export const GPU_SPATIAL_ERROR_GM_SUMMARY_MOMENT_OBJECTIVE = 5;

/** `status` value: fit succeeded. */
export const GPU_SPATIAL_ERROR_GM_STATUS_OK = 0;
/** `status` value: the first-stage OLS or the filtered regression is singular or ill-conditioned. */
export const GPU_SPATIAL_ERROR_GM_STATUS_SINGULAR = 1;
/** `status` value: too few rows, `n <= predictorCount + 2`. */
export const GPU_SPATIAL_ERROR_GM_STATUS_TOO_FEW_ROWS = 2;
/** `status` value: a non-finite input or moment, or zero OLS residuals. */
export const GPU_SPATIAL_ERROR_GM_STATUS_NON_FINITE = 3;
