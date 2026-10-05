// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in the optional per-frame ordinary least squares parameter view. */
export const GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH = 1;

/** `status` value: the fit succeeded. */
export const GPU_ORDINARY_LEAST_SQUARES_STATUS_OK = 0;
/** `status` value: the design is singular or ill-conditioned (collinear, constant, or non-finite column). */
export const GPU_ORDINARY_LEAST_SQUARES_STATUS_SINGULAR = 1;
/** `status` value: too few rows, `n <= predictorCount + 1`. */
export const GPU_ORDINARY_LEAST_SQUARES_STATUS_TOO_FEW_ROWS = 2;

/**
 * Smallest allowed pivot of the Cholesky factor of the predictor correlation matrix. A pivot below
 * it means a predictor is explained by the earlier ones to within `1 - 1e-6` of its variance
 * (variance inflation factor above `1e6`), which float32 cannot resolve, so the fit reports
 * `GPU_ORDINARY_LEAST_SQUARES_STATUS_SINGULAR`.
 */
export const GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE = 1e-3;

/** Number of float32 elements `output.summary` must hold. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH = 16;

/** `summary` slot: number of rows used (finite, unmasked), stored as a float32 integer. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT = 0;
/** `summary` slot: coefficient of determination `1 - RSS / TSS`. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED = 1;
/** `summary` slot: adjusted R squared `1 - (1 - R2) (n - 1) / (n - p)`. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED = 2;
/** `summary` slot: residual variance `RSS / (n - p)`. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED = 3;
/** `summary` slot: Gaussian log-likelihood `-n/2 (ln 2 pi + ln(RSS / n) + 1)`. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD = 4;
/** `summary` slot: Akaike information criterion `2 p - 2 logLikelihood` (p counts the intercept). */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_AIC = 5;
/** `summary` slot: Bayesian information criterion `p ln(n) - 2 logLikelihood`. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BIC = 6;
/** `summary` slot: Jarque-Bera statistic of the residuals. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA = 7;
/** `summary` slot: Jarque-Bera p-value, `exp(-JB / 2)` (chi-square with 2 degrees of freedom). */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE = 8;
/** `summary` slot: Breusch-Pagan statistic, Koenker studentized form `n R2` of `e^2 ~ X`. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN = 9;
/** `summary` slot: Breusch-Pagan p-value (chi-square with `predictorCount` degrees of freedom). */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE = 10;
/** `summary` slot: residual sum of squares. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES = 11;
/** `summary` slot: total (centered) sum of squares of the response. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_TOTAL_SUM_OF_SQUARES = 12;
/** `summary` slot: residual skewness `m3 / m2^1.5`. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SKEWNESS = 13;
/** `summary` slot: residual (non-excess) kurtosis `m4 / m2^2`. */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KURTOSIS = 14;
/** `summary` slot: ridge `lambda` the fit used (0 without a parameter view). */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RIDGE_LAMBDA = 15;

/**
 * Packs the per-frame parameters of `GPUOrdinaryLeastSquares`.
 *
 * Layout: `[ridgeLambda]` as float32. The ridge penalty `lambda * I` is added to the centered
 * normal matrix of the predictors (never to the intercept); a value that is not finite and
 * positive means no penalty.
 *
 * @param ridgeLambda Ridge penalty in squared predictor units (the penalty is added to `X'X`).
 * @param target Optional destination of at least 1 element.
 */
export function getGPUOrdinaryLeastSquaresParameterValues(
  ridgeLambda: number,
  target: Float32Array = new Float32Array(GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH) {
    throw new Error(
      `Ordinary least squares parameter target must hold ${GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH} element`
    );
  }
  target[0] = ridgeLambda;
  return target;
}
