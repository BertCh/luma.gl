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
  GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_ERROR_GM_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_ERROR_GM_SUMMARY_ROW_COUNT,
  GPU_SPATIAL_ERROR_GM_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_ERROR_GM_TABLE_STRIDE
} from './spatial-error-gm-parameters';

/** Read-back arrays of one `GPUSpatialErrorGM` fit. */
export type GPUSpatialErrorGMReadback = {
  /** `(predictorCount + 2) * 4` values, see `GPUSpatialErrorGMOutput.table`. */
  table: Float32Array;
  /** At least `GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH` values. */
  summary: Float32Array;
  status: Uint32Array;
};

/** Options of {@link getGPUSpatialErrorGMModelResult}. */
export type GPUSpatialErrorGMModelResultOptions = {
  /** Predictor names after `'CONSTANT'`; missing entries default to `'x0'`, `'x1'`, ... */
  predictorNames?: readonly string[];
};

/**
 * Builds the shared {@link GPURegressionModelResult} from the read-back outputs of
 * `GPUSpatialErrorGM`.
 *
 * Rows are the intercept, the predictors and `lambda` (role `'spatial-error'`, name `'lambda'`).
 * Statistics are z with p-values recomputed in float64 from the z statistic (standard normal,
 * two-sided), NaN where the GPU wrote NaN (the `lambda` row of the `kelejian-prucha` estimator).
 * `parameterCount` is `predictorCount + 2` (intercept, predictors, `lambda`; `sigma2` is not
 * counted), so `degreesOfFreedom` is `n - k`. `sigmaSquaredMaximumLikelihood` equals the summary
 * `sigma2 = e'e / n` of the filtered residuals, which scaled the standard errors
 * (`standardErrorSigmaSquared: 'maximum-likelihood'`); for that reason `residualSumOfSquares` is
 * the FILTERED residual sum of squares `n sigma2`, not the `u'u` of the summary. `pseudoRSquared`
 * is spreg `pr2`. No likelihood is defined, so `logLikelihood` and the criteria are null.
 * Conditioning is the final filtered regression's. A failed fit carries NaN estimates.
 */
export function getGPUSpatialErrorGMModelResult(
  readback: GPUSpatialErrorGMReadback,
  options: GPUSpatialErrorGMModelResultOptions = {}
): GPURegressionModelResult {
  const {table, summary} = readback;
  if (summary.length < GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH) {
    throw new Error(
      `Spatial error GM summary must hold ${GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH} values`
    );
  }
  const rowCount = table.length / GPU_SPATIAL_ERROR_GM_TABLE_STRIDE;
  const predictorCount = rowCount - 2;
  if (!Number.isInteger(predictorCount) || predictorCount < 1) {
    throw new Error('Spatial error GM table must hold (predictorCount + 2) * 4 values');
  }
  const names = getGPURegressionPredictorNames(predictorCount, options.predictorNames);
  names.push('lambda');
  const coefficients: GPURegressionCoefficient[] = names.map((name, index) => {
    const statistic = table[index * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE + 2];
    return {
      name,
      role:
        index === 0
          ? 'intercept'
          : index === names.length - 1
            ? ('spatial-error' as const)
            : ('predictor' as const),
      estimate: table[index * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE],
      standardError: table[index * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE + 1],
      statistic,
      statisticKind: 'z' as const,
      pValue: Number.isNaN(statistic) ? NaN : getChiSquareSurvival(statistic * statistic, 1)
    };
  });
  const sampleSize = summary[GPU_SPATIAL_ERROR_GM_SUMMARY_ROW_COUNT];
  const conditionNumber = summary[GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER];
  const correlationConditionNumber =
    summary[GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER];
  const hasConditioning =
    Number.isFinite(conditionNumber) && Number.isFinite(correlationConditionNumber);
  return createGPURegressionModelResult({
    kind: 'spatial-error-generalized-moments',
    coefficients,
    rowCount: sampleSize,
    residualSumOfSquares: summary[GPU_SPATIAL_ERROR_GM_SUMMARY_SIGMA_SQUARED] * sampleSize,
    standardErrorSigmaSquared: 'maximum-likelihood',
    statusCode: readback.status[0],
    pseudoRSquared: summary[GPU_SPATIAL_ERROR_GM_SUMMARY_PSEUDO_R_SQUARED],
    logLikelihood: null,
    conditioning: hasConditioning
      ? {
          conditionNumber,
          correlationConditionNumber,
          maximumVarianceInflationFactor:
            summary[GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR]
        }
      : null
  });
}
