// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
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
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM,
  getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters';
import {
  getGPUSpatialTwoStageLeastSquaresDiagnostics,
  getGPUSpatialTwoStageLeastSquaresModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-model-result';
import {SPREG_TWO_STAGE_REFERENCE} from './spreg-reference';

/** Packs the spreg `GM_Lag` pins (`createLagScene(12, 21, 0.5, true)`, n = 144) like the GPU finish kernel. */
function createReadback(status = 0) {
  const reference = SPREG_TWO_STAGE_REFERENCE;
  const rowCount = 144;
  const table = new Float32Array(reference.betas.length * 4);
  reference.betas.forEach((beta, row) => {
    table.set(
      [beta, reference.standardErrors[row], reference.z[row], reference.pValues[row]],
      row * 4
    );
  });
  const summary = new Float32Array(GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH).fill(NaN);
  summary[0] = rowCount;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED] = reference.sigma2;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES] =
    reference.sigma2 * rowCount;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PSEUDO_R_SQUARED] = reference.pseudoRSquared;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN] =
    reference.anselinKelejian[0];
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN_P_VALUE] =
    reference.anselinKelejian[1];
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER] = 26.5;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER] = 1.3;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR] = 1.01;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER] =
    2e4;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F] = 591.8;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_PARTIAL_R_SQUARED] = 0.89;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RHO_OUTSIDE_ADMISSIBLE_RANGE] = 1;
  summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM] = 1;
  return {table, summary, status: Uint32Array.of(status)};
}

it('getGPUSpatialTwoStageLeastSquaresModelResult builds the shared result from GM_Lag pins', () => {
  const reference = SPREG_TWO_STAGE_REFERENCE;
  const result = getGPUSpatialTwoStageLeastSquaresModelResult(createReadback(), {
    predictorNames: ['INC', 'HOVAL']
  });
  expect(result.kind).toBe('spatial-lag-two-stage-least-squares');
  expect(result.coefficients.map(coefficient => coefficient.name)).toEqual([
    'CONSTANT',
    'INC',
    'HOVAL',
    'W_y'
  ]);
  expect(result.coefficients.map(coefficient => coefficient.role)).toEqual([
    'intercept',
    'predictor',
    'predictor',
    'spatial-lag'
  ]);
  result.coefficients.forEach((coefficient, index) => {
    expect(coefficient.statisticKind).toBe('z');
    expect(coefficient.estimate).toBeCloseTo(reference.betas[index], 5);
    expect(coefficient.statistic).toBeCloseTo(reference.z[index], 3);
    // Float64 p-value from the float32 z; spreg's p underflows to 0 for the two huge z.
    expect(Math.abs(coefficient.pValue - reference.pValues[index])).toBeLessThan(
      Math.max(1e-5 * reference.pValues[index], 1e-30)
    );
  });
  expect(result.rowCount).toBe(144);
  expect(result.parameterCount).toBe(4);
  expect(result.degreesOfFreedom).toBe(140);
  expect(result.standardErrorSigmaSquared).toBe('maximum-likelihood');
  expect(result.sigmaSquaredMaximumLikelihood).toBeCloseTo(reference.sigma2, 4);
  expect(result.residualSumOfSquares).toBeCloseTo(reference.sigma2 * 144, 3);
  expect(result.pseudoRSquared).toBeCloseTo(reference.pseudoRSquared, 6);
  expect(result.logLikelihood).toBeNull();
  expect(result.tests.anselinKelejian).toEqual({
    statistic: expect.closeTo(reference.anselinKelejian[0], 5),
    degreesOfFreedom: 1,
    pValue: expect.closeTo(reference.anselinKelejian[1], 5)
  });
  // The larger of the second-stage and instrument correlation condition numbers drives the flag.
  expect(result.conditioning).toEqual({
    conditionNumber: 26.5,
    correlationConditionNumber: 2e4,
    maximumVarianceInflationFactor: expect.closeTo(1.01, 5),
    illConditioned: true
  });
  expect(result.status).toMatchObject({code: 0, ok: true, illConditioned: true});
});

it('getGPUSpatialTwoStageLeastSquaresDiagnostics reads first-stage strength and the rho flag', () => {
  const diagnostics = getGPUSpatialTwoStageLeastSquaresDiagnostics(createReadback().summary);
  expect(diagnostics.firstStageF).toBeCloseTo(591.8, 3);
  expect(diagnostics.firstStagePartialRSquared).toBeCloseTo(0.89, 5);
  expect(diagnostics.rhoOutsideAdmissibleRange).toBe(true);
  expect(diagnostics.weightsInfinityNorm).toBe(1);
  expect(diagnostics.instrumentCorrelationConditionNumber).toBe(2e4);
});

it('getGPUSpatialTwoStageLeastSquaresModelResult reports failure and validates its inputs', () => {
  const failed = createReadback(1);
  failed.table.fill(NaN);
  const result = getGPUSpatialTwoStageLeastSquaresModelResult(failed);
  expect(result.status).toMatchObject({code: 1, ok: false});
  expect(Number.isNaN(result.coefficients[0].pValue)).toBe(true);
  expect(() =>
    getGPUSpatialTwoStageLeastSquaresModelResult({
      table: new Float32Array(16),
      summary: new Float32Array(7),
      status: Uint32Array.of(0)
    })
  ).toThrow(/summary/);
  expect(() =>
    getGPUSpatialTwoStageLeastSquaresModelResult({
      table: new Float32Array(7),
      summary: new Float32Array(GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH),
      status: Uint32Array.of(0)
    })
  ).toThrow(/table/);
});

it('the two-stage predictor cap follows the 16-wide instrument Cholesky', () => {
  expect(getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount(1)).toBe(8);
  expect(getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount(2)).toBe(5);
});
