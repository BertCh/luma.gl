// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPURegressionModelResult,
  getGPURegressionPredictorNames,
  type GPURegressionCoefficient,
  type GPURegressionModelResult
} from '../contracts/model-result';
import {
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT
} from './ordinary-least-squares-parameters';

/**
 * Natural log of the gamma function at `degreesOfFreedom / 2` for a positive integer
 * `degreesOfFreedom`, by the exact half-integer recurrence `Gamma(a + 1) = a Gamma(a)` starting
 * from `Gamma(1) = 1` or `Gamma(1/2) = sqrt(pi)`.
 */
export function getLogGammaOfHalfInteger(degreesOfFreedom: number): number {
  if (!Number.isInteger(degreesOfFreedom) || degreesOfFreedom < 1) {
    throw new Error('Chi-square degrees of freedom must be a positive integer');
  }
  const target = degreesOfFreedom / 2;
  let argument = degreesOfFreedom % 2 === 0 ? 1 : 0.5;
  let logGamma = degreesOfFreedom % 2 === 0 ? 0 : 0.5 * Math.log(Math.PI);
  while (argument < target) {
    logGamma += Math.log(argument);
    argument += 1;
  }
  return logGamma;
}

/** Fixed iteration cap shared by the series and the continued fraction, GPU and CPU. */
export const CHI_SQUARE_MAXIMUM_ITERATIONS = 200;

/**
 * Survival function `P(X > value)` of a chi-square distribution with `degreesOfFreedom` degrees of
 * freedom, computed as the regularized upper incomplete gamma `Q(df / 2, value / 2)`.
 *
 * Uses the power series for `x < a + 1` and the modified Lentz continued fraction otherwise, each
 * capped at `CHI_SQUARE_MAXIMUM_ITERATIONS` iterations: the same algorithm as the WGSL in
 * `GPUOrdinaryLeastSquares`, in float64. Returns NaN for a NaN value and 1 for `value <= 0`.
 */
export function getChiSquareSurvival(value: number, degreesOfFreedom: number): number {
  const logGamma = getLogGammaOfHalfInteger(degreesOfFreedom);
  if (Number.isNaN(value)) {
    return NaN;
  }
  if (!(value > 0)) {
    return 1;
  }
  const a = degreesOfFreedom / 2;
  const x = value / 2;
  const logPrefactor = -x + a * Math.log(x) - logGamma;
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let iteration = 1; iteration <= CHI_SQUARE_MAXIMUM_ITERATIONS; iteration++) {
      term *= x / (a + iteration);
      sum += term;
      if (term < sum * 1e-15) {
        break;
      }
    }
    return Math.min(1, Math.max(0, 1 - sum * Math.exp(logPrefactor)));
  }
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let iteration = 1; iteration <= CHI_SQUARE_MAXIMUM_ITERATIONS; iteration++) {
    const an = -iteration * (iteration - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (Math.abs(c) < tiny) {
      c = tiny;
    }
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) {
      break;
    }
  }
  return Math.min(1, Math.max(0, Math.exp(logPrefactor) * h));
}

const BETA_MAXIMUM_ITERATIONS = 400;

/** Continued fraction of the regularized incomplete beta (modified Lentz, Numerical Recipes). */
function getBetaContinuedFraction(x: number, a: number, b: number): number {
  const tiny = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < tiny) {
    d = tiny;
  }
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= BETA_MAXIMUM_ITERATIONS; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) {
      d = tiny;
    }
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) {
      c = tiny;
    }
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) {
      d = tiny;
    }
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) {
      c = tiny;
    }
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-16) {
      break;
    }
  }
  return h;
}

/**
 * Two-sided p-value `P(|T| > |t|)` of a Student t statistic with an integer `degreesOfFreedom`,
 * the regularized incomplete beta `I_x(df / 2, 1 / 2)` with `x = df / (df + t^2)`, in float64
 * (Numerical Recipes continued fraction). Returns NaN for a NaN statistic or non-positive
 * degrees of freedom and 0 for an infinite statistic.
 */
export function getStudentTTwoSidedPValue(statistic: number, degreesOfFreedom: number): number {
  if (Number.isNaN(statistic) || !Number.isInteger(degreesOfFreedom) || degreesOfFreedom < 1) {
    return NaN;
  }
  if (!Number.isFinite(statistic)) {
    return 0;
  }
  const a = degreesOfFreedom / 2;
  const b = 0.5;
  const x = degreesOfFreedom / (degreesOfFreedom + statistic * statistic);
  if (x >= 1) {
    return 1;
  }
  const logPrefactor =
    getLogGammaOfHalfInteger(degreesOfFreedom + 1) -
    getLogGammaOfHalfInteger(degreesOfFreedom) -
    getLogGammaOfHalfInteger(1) +
    a * Math.log(x) +
    b * Math.log1p(-x);
  const prefactor = Math.exp(logPrefactor);
  if (x < (a + 1) / (a + b + 2)) {
    return Math.min(1, (prefactor * getBetaContinuedFraction(x, a, b)) / a);
  }
  return Math.max(0, 1 - (prefactor * getBetaContinuedFraction(1 - x, b, a)) / b);
}

/**
 * Survival function `P(T > t)` of a Student t distribution with an integer `degreesOfFreedom`
 * (scipy `stats.t.sf`), from {@link getStudentTTwoSidedPValue}.
 */
export function getStudentTSurvival(statistic: number, degreesOfFreedom: number): number {
  const twoSided = getStudentTTwoSidedPValue(statistic, degreesOfFreedom);
  if (Number.isNaN(twoSided)) {
    return NaN;
  }
  return statistic >= 0 ? 0.5 * twoSided : 1 - 0.5 * twoSided;
}

/** Read-back arrays of one `GPUOrdinaryLeastSquares` fit. */
export type GPUOrdinaryLeastSquaresReadback = {
  coefficients: Float32Array;
  standardErrors: Float32Array;
  tStatistics: Float32Array;
  /** At least `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH` values. */
  summary: Float32Array;
  status: Uint32Array;
};

/** Options of {@link getGPUOrdinaryLeastSquaresModelResult}. */
export type GPUOrdinaryLeastSquaresModelResultOptions = {
  /** Predictor names after `'CONSTANT'`; missing entries default to `'x0'`, `'x1'`, ... */
  predictorNames?: readonly string[];
};

/**
 * Builds the shared {@link GPURegressionModelResult} from the read-back outputs of
 * `GPUOrdinaryLeastSquares`.
 *
 * The predictor count is `coefficients.length - 1`. Coefficient p-values are two-sided Student t
 * on `n - p` degrees of freedom (float64, {@link getStudentTTwoSidedPValue}) where `p` counts the
 * intercept. `sigmaSquared` is `RSS / (n - p)` (`'degrees-of-freedom'`, the convention of the
 * standard errors) and `sigmaSquaredMaximumLikelihood` is `RSS / n`. The log-likelihood is the
 * Gaussian one with `p` counted parameters, so AIC and the Schwarz criterion equal spreg
 * `diagnostics.akaike` and `diagnostics.schwarz`. Tests are `jarqueBera` (2 df),
 * `breuschPagan` and `koenkerBassett` (`predictorCount` df each). Conditioning is null when the
 * fit failed. A failed fit (`status` not 0) carries NaN estimates.
 */
export function getGPUOrdinaryLeastSquaresModelResult(
  readback: GPUOrdinaryLeastSquaresReadback,
  options: GPUOrdinaryLeastSquaresModelResultOptions = {}
): GPURegressionModelResult {
  const {summary} = readback;
  if (summary.length < GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH) {
    throw new Error(
      `Ordinary least squares summary must hold ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} values`
    );
  }
  const coefficientCount = readback.coefficients.length;
  const predictorCount = coefficientCount - 1;
  const rowCount = summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT];
  const degreesOfFreedom = rowCount - coefficientCount;
  const names = getGPURegressionPredictorNames(predictorCount, options.predictorNames);
  const coefficients: GPURegressionCoefficient[] = names.map((name, index) => ({
    name,
    role: index === 0 ? 'intercept' : 'predictor',
    estimate: readback.coefficients[index],
    standardError: readback.standardErrors[index],
    statistic: readback.tStatistics[index],
    statisticKind: 't',
    pValue: getStudentTTwoSidedPValue(readback.tStatistics[index], degreesOfFreedom)
  }));
  const conditionNumber = summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER];
  const correlationConditionNumber =
    summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER];
  const hasConditioning =
    Number.isFinite(conditionNumber) && Number.isFinite(correlationConditionNumber);
  const chiSquareTest = (statisticSlot: number, pValueSlot: number, freedom: number) => ({
    statistic: summary[statisticSlot],
    degreesOfFreedom: freedom,
    pValue: summary[pValueSlot]
  });
  return createGPURegressionModelResult({
    kind: 'ordinary-least-squares',
    coefficients,
    rowCount,
    residualSumOfSquares: summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES],
    standardErrorSigmaSquared: 'degrees-of-freedom',
    statusCode: readback.status[0],
    rSquared: summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED],
    logLikelihood: Number.isNaN(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD])
      ? null
      : summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD],
    conditioning: hasConditioning
      ? {
          conditionNumber,
          correlationConditionNumber,
          maximumVarianceInflationFactor:
            summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR]
        }
      : null,
    tests: {
      jarqueBera: chiSquareTest(
        GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA,
        GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE,
        2
      ),
      breuschPagan: chiSquareTest(
        GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN,
        GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE,
        predictorCount
      ),
      koenkerBassett: chiSquareTest(
        GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT,
        GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE,
        predictorCount
      )
    }
  });
}
