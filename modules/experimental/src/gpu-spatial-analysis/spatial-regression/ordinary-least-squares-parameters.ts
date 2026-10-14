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
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH = 22;

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
/**
 * `summary` slot: Koenker-Bassett (studentized Breusch-Pagan) statistic `n R2` of the auxiliary
 * regression `e^2 ~ [1, X]`. It does not assume normal errors (spreg `koenker_bassett`, with
 * `z = [1, X]`).
 */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT = 9;
/** `summary` slot: Koenker-Bassett p-value (chi-square with `predictorCount` degrees of freedom). */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE = 10;
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
 * `summary` slot: classic Breusch-Pagan statistic (normal errors), `ESS / 2` of `g = e^2 / s2 - 1`
 * regressed on `[1, X]` with `s2 = RSS / n`, which equals `0.5 R2_aux (sum e^4 / s2^2 - n)`
 * (spreg `breusch_pagan` with `z = [1, X]`). It is not robust to non-normal errors: compare with
 * the Koenker-Bassett slot, which is.
 */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN = 16;
/** `summary` slot: Breusch-Pagan p-value (chi-square with `predictorCount` degrees of freedom). */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE = 17;
/**
 * `summary` slot: Belsley multicollinearity condition number of the design with its intercept,
 * `sqrt(lambdaMax / lambdaMin)` of the raw `[1, X]` Gram matrix scaled to unit diagonal (spreg
 * `OLS.mulColli`). Values above about 30 indicate harmful collinearity. Ridge is not included.
 */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER = 18;
/**
 * `summary` slot: `lambdaMax / lambdaMin` of the predictor correlation matrix that the float32
 * Cholesky solve factors (with the ridge penalty when one is set). The relative error of the
 * float32 coefficients is about this value times `6e-8`.
 */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER = 19;
/**
 * `summary` slot: largest variance inflation factor, the largest diagonal entry of the inverse of
 * the factored predictor correlation matrix (`max_j 1 / (1 - R_j^2)`).
 */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR = 20;
/**
 * `summary` slot: 1 when the correlation condition number exceeds
 * `GPU_REGRESSION_PRECISION_CONDITION_LIMIT` (1e4: float32 normal equations may have lost more than
 * about three of seven digits), else 0. A warning never changes `status` and never NaNs the fit.
 */
export const GPU_ORDINARY_LEAST_SQUARES_SUMMARY_PRECISION_WARNING = 21;

/** Per-frame settings of `GPUOrdinaryLeastSquares`, packed by {@link getGPUOrdinaryLeastSquaresParameterValues}. */
export type GPUOrdinaryLeastSquaresSettings = {
  /**
   * Ridge penalty in squared predictor units, added as `lambda * I` to the centered normal matrix
   * of the predictors (never to the intercept). A value that is not finite and positive means no
   * penalty. Defaults to 0.
   */
  ridgeLambda?: number;
};

/**
 * Packs {@link GPUOrdinaryLeastSquaresSettings} into the per-frame `parameters` of
 * `GPUOrdinaryLeastSquares`. Layout: `[ridgeLambda]` as float32.
 *
 * @param settings Ridge penalty.
 * @param target Optional destination of at least `GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH` elements.
 */
export function getGPUOrdinaryLeastSquaresParameterValues(
  settings: GPUOrdinaryLeastSquaresSettings = {},
  target: Float32Array = new Float32Array(GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH) {
    throw new Error(
      `getGPUOrdinaryLeastSquaresParameterValues target must hold ${GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH} element`
    );
  }
  target[0] = settings.ridgeLambda ?? 0;
  return target;
}
