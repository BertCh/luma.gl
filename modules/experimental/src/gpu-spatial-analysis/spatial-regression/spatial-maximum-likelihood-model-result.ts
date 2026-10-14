// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPURegressionModelResult,
  getGPURegressionPredictorNames,
  type GPURegressionCoefficient,
  type GPURegressionModelResult
} from '../contracts/model-result';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOCAL_MAXIMUM_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
} from './spatial-maximum-likelihood-parameters';

/** Read-back arrays of one `GPUSpatialLagMaximumLikelihood` or `GPUSpatialErrorMaximumLikelihood` fit. */
export type GPUSpatialMaximumLikelihoodReadback = {
  /** `(predictorCount + 2) * 4` values, see the `table` output. */
  table: Float32Array;
  /** At least `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH` values. */
  summary: Float32Array;
  status: Uint32Array;
};

/** Options of {@link getGPUSpatialLagMaximumLikelihoodModelResult}. */
export type GPUSpatialMaximumLikelihoodModelResultOptions = {
  /** Predictor names after `'CONSTANT'`; missing entries default to `'x0'`, `'x1'`, ... */
  predictorNames?: readonly string[];
};

/** Fit evidence of a maximum-likelihood fit that is not part of the shared model result. */
export type GPUSpatialMaximumLikelihoodDiagnostics = {
  /** The search interval `[-bound, bound]` of `rho` or `lambda`. */
  parameterBound: number;
  /** True when the traces were Hutchinson estimates, so the likelihood carries stochastic error. */
  approximation: boolean;
  /** True when the maximum is an interior stationary point. */
  converged: boolean;
  /** True when the maximum is an end of the search interval (a boundary solution). */
  atBoundary: boolean;
  /** True when the log-determinant series limit, not `0.99 / ||W||_inf`, shortened the interval. */
  seriesLimited: boolean;
  /** Worst-case log-determinant truncation error at the estimate, in log-likelihood units. */
  truncationBound: number;
  /** True when the information matrix was positive definite (standard errors are defined). */
  varianceDefined: boolean;
  /** Local maxima of the concentrated likelihood on the scan grid; more than 1 is multimodal. */
  localMaximumCount: number;
  /** `ln|I - rho W|` at the estimate. */
  logDeterminant: number;
  /** Squared correlation of `y` and `(I - rho W)^-1 X b` (spreg `pr2_e`); NaN for the error model. */
  pseudoRSquaredSpatial: number;
};

/** Reads the fit evidence slots of a maximum-likelihood `summary`. */
export function getGPUSpatialMaximumLikelihoodDiagnostics(
  summary: Float32Array
): GPUSpatialMaximumLikelihoodDiagnostics {
  if (summary.length < GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH) {
    throw new Error(
      `Maximum likelihood summary must hold ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH} values`
    );
  }
  return {
    parameterBound: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND],
    approximation: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION] === 1,
    converged: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED] === 1,
    atBoundary: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY] === 1,
    seriesLimited: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED] === 1,
    truncationBound: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND],
    varianceDefined: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED] === 1,
    localMaximumCount: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOCAL_MAXIMUM_COUNT],
    logDeterminant: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT],
    pseudoRSquaredSpatial: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL]
  };
}

function getModelResult(
  kind: 'spatial-lag-maximum-likelihood' | 'spatial-error-maximum-likelihood',
  readback: GPUSpatialMaximumLikelihoodReadback,
  options: GPUSpatialMaximumLikelihoodModelResultOptions
): GPURegressionModelResult {
  const {table, summary} = readback;
  const lag = kind === 'spatial-lag-maximum-likelihood';
  if (summary.length < GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH) {
    throw new Error(
      `Maximum likelihood summary must hold ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH} values`
    );
  }
  const stride = GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE;
  const rowCount = table.length / stride;
  const predictorCount = rowCount - 2;
  if (!Number.isInteger(predictorCount) || predictorCount < 1) {
    throw new Error('Maximum likelihood table must hold (predictorCount + 2) * 4 values');
  }
  const names = getGPURegressionPredictorNames(predictorCount, options.predictorNames);
  names.push(lag ? 'W_y' : 'lambda');
  const coefficients: GPURegressionCoefficient[] = names.map((name, index) => {
    const statistic = table[index * stride + 2];
    return {
      name,
      role:
        index === 0
          ? 'intercept'
          : index === names.length - 1
            ? lag
              ? ('spatial-lag' as const)
              : ('spatial-error' as const)
            : ('predictor' as const),
      estimate: table[index * stride],
      standardError: table[index * stride + 1],
      statistic,
      statisticKind: 'z' as const,
      pValue: Number.isNaN(statistic) ? NaN : getChiSquareSurvival(statistic * statistic, 1)
    };
  });
  const sampleSize = summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT];
  const conditionNumber = summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER];
  const correlationConditionNumber =
    summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER];
  const hasConditioning =
    Number.isFinite(conditionNumber) && Number.isFinite(correlationConditionNumber);
  const sigmaSquared = summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED];
  return createGPURegressionModelResult({
    kind,
    coefficients,
    rowCount: sampleSize,
    // `n sigma2`: the residual sum of squares the standard errors were scaled with. For the lag
    // model it is `u'u`; for the error model it is the FILTERED `e'e` (the unfiltered `u'u` is the
    // summary RESIDUAL_SUM_OF_SQUARES slot).
    residualSumOfSquares: lag
      ? summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES]
      : sigmaSquared * sampleSize,
    standardErrorSigmaSquared: 'maximum-likelihood',
    statusCode: readback.status[0],
    converged:
      readback.status[0] === 0
        ? summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED] === 1
        : false,
    pseudoRSquared: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED],
    logLikelihood: summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD],
    // spreg counts rho for the lag model but not lambda for the error model.
    informationCriterionParameterCount: lag ? coefficients.length : coefficients.length - 1,
    conditioning: hasConditioning
      ? {
          conditionNumber,
          correlationConditionNumber,
          maximumVarianceInflationFactor:
            summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR]
        }
      : null
  });
}

/**
 * Builds the shared {@link GPURegressionModelResult} from the read-back outputs of
 * `GPUSpatialLagMaximumLikelihood`.
 *
 * Rows are the intercept, the predictors and `rho` (role `'spatial-lag'`, name `'W_y'`, as spreg).
 * Statistics are z with p-values recomputed in float64 (standard normal, two-sided).
 * `parameterCount` is `predictorCount + 2` (intercept, predictors, `rho`; `sigma2` is not
 * counted), so `degreesOfFreedom` is `n - k`. `residualSumOfSquares` is `u'u` and
 * `sigmaSquaredMaximumLikelihood = u'u / n` scaled the standard errors
 * (`standardErrorSigmaSquared: 'maximum-likelihood'`; `sigmaSquared` is the `n - k` form).
 * `logLikelihood` is spreg `logll`; `akaikeInformationCriterion = 2 k - 2 logLikelihood` and
 * `schwarzCriterion = k ln(n) - 2 logLikelihood` with spreg's `k = predictorCount + 2`, so they
 * equal `ML_Lag.aic` and `.schwarz`. `pseudoRSquared` is spreg `pr2`; the spatial pseudo R squared
 * is in {@link getGPUSpatialMaximumLikelihoodDiagnostics}. `status.converged` is true for an
 * interior maximum. Conditioning is that of the design `[1, X]`.
 */
export function getGPUSpatialLagMaximumLikelihoodModelResult(
  readback: GPUSpatialMaximumLikelihoodReadback,
  options: GPUSpatialMaximumLikelihoodModelResultOptions = {}
): GPURegressionModelResult {
  return getModelResult('spatial-lag-maximum-likelihood', readback, options);
}

/**
 * Builds the shared {@link GPURegressionModelResult} from the read-back outputs of
 * `GPUSpatialErrorMaximumLikelihood`.
 *
 * Rows are the intercept, the predictors and `lambda` (role `'spatial-error'`). Statistics are z
 * with float64 normal p-values. `parameterCount` is `predictorCount + 2` (intercept, predictors,
 * `lambda`), so `degreesOfFreedom` is `n - k`. `residualSumOfSquares` is the FILTERED `n sigma2`
 * (`e'e`), so `sigmaSquaredMaximumLikelihood` is the `sigma2` that scaled the standard errors.
 * `akaikeInformationCriterion` and `schwarzCriterion` use spreg's `k = predictorCount + 1`
 * (`ML_Error` does not count `lambda`), which equals `ML_Error.aic` and `.schwarz`.
 * `pseudoRSquared` is spreg `pr2` of `y` and `X b`. Conditioning is that of the filtered
 * regression at `lambda`.
 */
export function getGPUSpatialErrorMaximumLikelihoodModelResult(
  readback: GPUSpatialMaximumLikelihoodReadback,
  options: GPUSpatialMaximumLikelihoodModelResultOptions = {}
): GPURegressionModelResult {
  return getModelResult('spatial-error-maximum-likelihood', readback, options);
}
