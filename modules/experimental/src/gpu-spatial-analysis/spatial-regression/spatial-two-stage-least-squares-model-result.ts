// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPURegressionModelResult,
  getGPURegressionPredictorNames,
  type GPURegressionCoefficient,
  type GPURegressionModelResult
} from '../../analysis-core/contracts/model-result';
import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN_P_VALUE,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_PARTIAL_R_SQUARED,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RHO_OUTSIDE_ADMISSIBLE_RANGE,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ROW_COUNT,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
} from './spatial-two-stage-least-squares-parameters';

/** Read-back arrays of one `GPUSpatialTwoStageLeastSquares` fit. */
export type GPUSpatialTwoStageLeastSquaresReadback = {
  /** `(predictorCount + 2) * 4` values, see `GPUSpatialTwoStageLeastSquaresOutput.table`. */
  table: Float32Array;
  /** At least `GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH` values. */
  summary: Float32Array;
  status: Uint32Array;
};

/** Options of {@link getGPUSpatialTwoStageLeastSquaresModelResult}. */
export type GPUSpatialTwoStageLeastSquaresModelResultOptions = {
  /** Predictor names after `'CONSTANT'`; missing entries default to `'x0'`, `'x1'`, ... */
  predictorNames?: readonly string[];
};

/** Instrument and spatial-lag diagnostics of the 2SLS summary that the shared model result has no field for. */
export type GPUSpatialTwoStageLeastSquaresDiagnostics = {
  /** Classical joint F statistic of the excluded instruments in the first stage; weak below about 10. */
  firstStageF: number;
  /** Partial R squared of the excluded instruments in the first stage. */
  firstStagePartialRSquared: number;
  /** True when `|rho| ||W||_inf >= 1`: `rho` is outside the interval where `I - rho W` is guaranteed invertible. */
  rhoOutsideAdmissibleRange: boolean;
  /** `||W||_inf`, the largest absolute row sum of `W`. */
  weightsInfinityNorm: number;
  /** Correlation condition number of the instrument system. */
  instrumentCorrelationConditionNumber: number;
};

/** Reads the diagnostics that {@link getGPUSpatialTwoStageLeastSquaresModelResult} does not carry. */
export function getGPUSpatialTwoStageLeastSquaresDiagnostics(
  summary: Float32Array
): GPUSpatialTwoStageLeastSquaresDiagnostics {
  if (summary.length < GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH) {
    throw new Error(
      `Spatial two-stage least squares summary must hold ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH} values`
    );
  }
  return {
    firstStageF: summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F],
    firstStagePartialRSquared:
      summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_PARTIAL_R_SQUARED],
    rhoOutsideAdmissibleRange:
      summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RHO_OUTSIDE_ADMISSIBLE_RANGE] === 1,
    weightsInfinityNorm: summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM],
    instrumentCorrelationConditionNumber:
      summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER]
  };
}

/**
 * Builds the shared {@link GPURegressionModelResult} from the read-back outputs of
 * `GPUSpatialTwoStageLeastSquares`.
 *
 * Rows are the intercept, the predictors and the spatial lag `W y` (role `'spatial-lag'`, name
 * `'W_y'`, the coefficient `rho`). Statistics are z with p-values recomputed in float64 from the z
 * statistic (standard normal, two-sided). `parameterCount` is `predictorCount + 2`, so
 * `degreesOfFreedom` is `n - k`; the standard errors use `sigma2 = u'u / n`
 * (`standardErrorSigmaSquared: 'maximum-likelihood'`, spreg `sig2n`) and `residualSumOfSquares` is
 * the structural `u'u`. `pseudoRSquared` is spreg `pr2`; no likelihood is defined (null). The test
 * `anselinKelejian` is the 1-df chi-square test of residual spatial autocorrelation. Conditioning
 * is the second stage's, except that `correlationConditionNumber` is the larger of the
 * second-stage and instrument-system values, so `illConditioned` matches the summary precision
 * warning. A failed fit carries NaN estimates. First-stage strength and the `rho` admissibility
 * flag are in {@link getGPUSpatialTwoStageLeastSquaresDiagnostics}.
 */
export function getGPUSpatialTwoStageLeastSquaresModelResult(
  readback: GPUSpatialTwoStageLeastSquaresReadback,
  options: GPUSpatialTwoStageLeastSquaresModelResultOptions = {}
): GPURegressionModelResult {
  const {table, summary} = readback;
  if (summary.length < GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH) {
    throw new Error(
      `Spatial two-stage least squares summary must hold ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH} values`
    );
  }
  const rowCount = table.length / GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE;
  const predictorCount = rowCount - 2;
  if (!Number.isInteger(predictorCount) || predictorCount < 1) {
    throw new Error(
      'Spatial two-stage least squares table must hold (predictorCount + 2) * 4 values'
    );
  }
  const names = getGPURegressionPredictorNames(predictorCount, options.predictorNames);
  names.push('W_y');
  const stride = GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE;
  const coefficients: GPURegressionCoefficient[] = names.map((name, index) => {
    const statistic = table[index * stride + 2];
    return {
      name,
      role:
        index === 0
          ? 'intercept'
          : index === names.length - 1
            ? ('spatial-lag' as const)
            : ('predictor' as const),
      estimate: table[index * stride],
      standardError: table[index * stride + 1],
      statistic,
      statisticKind: 'z' as const,
      pValue: Number.isNaN(statistic) ? NaN : getChiSquareSurvival(statistic * statistic, 1)
    };
  });
  const conditionNumber = summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER];
  const secondStage =
    summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER];
  const instruments =
    summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER];
  const hasConditioning = Number.isFinite(conditionNumber) && Number.isFinite(secondStage);
  const anselinKelejian = summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN];
  return createGPURegressionModelResult({
    kind: 'spatial-lag-two-stage-least-squares',
    coefficients,
    rowCount: summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ROW_COUNT],
    residualSumOfSquares:
      summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES],
    standardErrorSigmaSquared: 'maximum-likelihood',
    statusCode: readback.status[0],
    pseudoRSquared: summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PSEUDO_R_SQUARED],
    logLikelihood: null,
    conditioning: hasConditioning
      ? {
          conditionNumber,
          correlationConditionNumber: Number.isFinite(instruments)
            ? Math.max(secondStage, instruments)
            : secondStage,
          maximumVarianceInflationFactor:
            summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR]
        }
      : null,
    tests: {
      anselinKelejian: {
        statistic: anselinKelejian,
        degreesOfFreedom: 1,
        pValue: summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN_P_VALUE]
      }
    }
  });
}
