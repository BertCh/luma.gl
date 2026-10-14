// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  fitSpatialErrorMaximumLikelihoodOnCPU,
  fitSpatialLagMaximumLikelihoodOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-cpu';
import {
  getGPUSpatialErrorMaximumLikelihoodModelResult,
  getGPUSpatialLagMaximumLikelihoodModelResult,
  getGPUSpatialMaximumLikelihoodDiagnostics,
  type GPUSpatialMaximumLikelihoodReadback
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-model-result';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AIC,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SCHWARZ,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  SPATIAL_MAXIMUM_LIKELIHOOD_SCENES,
  SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE,
  SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE
} from './spatial-maximum-likelihood-reference';

type Fit = NonNullable<ReturnType<typeof fitSpatialLagMaximumLikelihoodOnCPU>>;

/** Packs a float64 fit the way the GPU finish kernel writes its float32 outputs. */
function createReadback(
  fit: Fit | NonNullable<ReturnType<typeof fitSpatialErrorMaximumLikelihoodOnCPU>>,
  parameter: number,
  rowCount: number,
  lag: boolean
): GPUSpatialMaximumLikelihoodReadback {
  const stride = GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE;
  const table = new Float32Array(fit.coefficients.length * stride);
  fit.coefficients.forEach((value, row) => {
    table[row * stride] = value;
    table[row * stride + 1] = fit.standardErrors[row];
    table[row * stride + 2] = fit.zStatistics[row];
    table[row * stride + 3] = fit.pValues[row];
  });
  const summary = new Float32Array(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH).fill(NaN);
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT] = rowCount;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER] = parameter;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED] = fit.sigmaSquared;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD] = fit.logLikelihood;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AIC] = fit.akaikeInformationCriterion;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SCHWARZ] = fit.schwarzCriterion;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED] = fit.pseudoRSquared;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL] =
    'pseudoRSquaredSpatial' in fit ? fit.pseudoRSquaredSpatial : NaN;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES] =
    fit.residualSumOfSquares;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER] =
    fit.conditioning.conditionNumber;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER] =
    fit.conditioning.correlationConditionNumber;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR] =
    fit.conditioning.maximumVarianceInflationFactor;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND] = fit.bound;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION] = 0;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED] = fit.atBoundary ? 0 : 1;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY] = fit.atBoundary ? 1 : 0;
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED] = 1;
  void lag;
  return {table, summary, status: Uint32Array.of(0)};
}

it('the spatial lag model result carries spreg ML_Lag AIC, Schwarz and the shared conventions', () => {
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  const fixture = SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE.lag_12_21;
  const fit = fitSpatialLagMaximumLikelihoodOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  ) as Fit;
  const rowCount = scene.response.length;
  const result = getGPUSpatialLagMaximumLikelihoodModelResult(
    createReadback(fit, fit.rho, rowCount, true),
    {predictorNames: ['income', 'slope']}
  );
  expect(result.kind).toBe('spatial-lag-maximum-likelihood');
  expect(result.coefficients.map(c => c.name)).toEqual(['CONSTANT', 'income', 'slope', 'W_y']);
  expect(result.coefficients.map(c => c.role)).toEqual([
    'intercept',
    'predictor',
    'predictor',
    'spatial-lag'
  ]);
  expect(result.coefficients.every(c => c.statisticKind === 'z')).toBe(true);
  expect(result.rowCount).toBe(rowCount);
  expect(result.parameterCount).toBe(4);
  expect(result.degreesOfFreedom).toBe(rowCount - 4);
  expect(result.standardErrorSigmaSquared).toBe('maximum-likelihood');
  // float32 summary values: the criteria equal spreg's to float32 rounding of the likelihood.
  expect(result.logLikelihood).toBeCloseTo(fixture.logLikelihood, 3);
  expect(result.akaikeInformationCriterion).toBeCloseTo(fixture.aic, 3);
  expect(result.schwarzCriterion).toBeCloseTo(fixture.schwarz, 3);
  expect(result.sigmaSquaredMaximumLikelihood).toBeCloseTo(fixture.sigma2, 5);
  expect(result.sigmaSquared).toBeCloseTo((fixture.sigma2 * rowCount) / (rowCount - 4), 5);
  expect(result.pseudoRSquared).toBeCloseTo(fixture.pseudoRSquared, 6);
  expect(result.conditioning?.conditionNumber).toBeCloseTo(fixture.conditioning.conditionNumber, 4);
  expect(result.conditioning?.illConditioned).toBe(false);
  expect(result.status).toEqual({code: 0, ok: true, converged: true, illConditioned: false});
  // p-values are recomputed in float64 from z.
  result.coefficients.forEach((c, index) => {
    expect(Math.abs(c.pValue - fixture.p[index])).toBeLessThan(1e-5);
  });
  const diagnostics = getGPUSpatialMaximumLikelihoodDiagnostics(
    createReadback(fit, fit.rho, rowCount, true).summary
  );
  expect(diagnostics.converged).toBe(true);
  expect(diagnostics.atBoundary).toBe(false);
  expect(diagnostics.varianceDefined).toBe(true);
  expect(diagnostics.pseudoRSquaredSpatial).toBeCloseTo(fixture.pseudoRSquaredSpatial, 6);
});

it('the spatial error model result uses spreg ML_Error k (lambda not counted) and the filtered sigma2', () => {
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.err_lattice_400_31();
  const fixture = SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE.err_lattice_400_31;
  const fit = fitSpatialErrorMaximumLikelihoodOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  expect(fit).not.toBeNull();
  if (!fit) {
    return;
  }
  const rowCount = scene.response.length;
  const result = getGPUSpatialErrorMaximumLikelihoodModelResult(
    createReadback(fit, fit.lambda, rowCount, false)
  );
  expect(result.kind).toBe('spatial-error-maximum-likelihood');
  expect(result.coefficients.map(c => c.name)).toEqual(['CONSTANT', 'x0', 'x1', 'lambda']);
  expect(result.coefficients[3].role).toBe('spatial-error');
  expect(result.parameterCount).toBe(4);
  expect(result.degreesOfFreedom).toBe(rowCount - 4);
  // AIC / Schwarz count 3 parameters, as spreg (specK = 3).
  expect(fixture.specK).toBe(3);
  expect(result.logLikelihood).toBeCloseTo(fixture.logLikelihood, 2);
  expect(result.akaikeInformationCriterion).toBeCloseTo(fixture.aic, 2);
  expect(result.schwarzCriterion).toBeCloseTo(fixture.schwarz, 2);
  // The residual sum of squares is the filtered n sigma2, so the ML sigma2 is the pinned one.
  expect(result.sigmaSquaredMaximumLikelihood).toBeCloseTo(fixture.sigma2, 5);
  expect(result.residualSumOfSquares).toBeCloseTo(fixture.sigma2 * rowCount, 3);
  expect(result.pseudoRSquared).toBeCloseTo(fixture.pseudoRSquared, 6);
  expect(
    Number.isNaN(
      getGPUSpatialMaximumLikelihoodDiagnostics(
        createReadback(fit, fit.lambda, rowCount, false).summary
      ).pseudoRSquaredSpatial
    )
  ).toBe(true);
});

it('a failed fit is not ok, not converged, and carries NaN estimates', () => {
  const table = new Float32Array(16).fill(NaN);
  const summary = new Float32Array(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH).fill(NaN);
  summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT] = 10;
  const result = getGPUSpatialLagMaximumLikelihoodModelResult({
    table,
    summary,
    status: Uint32Array.of(3)
  });
  expect(result.status.ok).toBe(false);
  expect(result.status.code).toBe(3);
  expect(result.status.converged).toBe(false);
  expect(result.coefficients.every(c => Number.isNaN(c.estimate) && Number.isNaN(c.pValue))).toBe(
    true
  );
  expect(result.conditioning).toBeNull();
});

it('the model-result readers validate their inputs', () => {
  const summary = new Float32Array(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH);
  expect(() =>
    getGPUSpatialLagMaximumLikelihoodModelResult({
      table: new Float32Array(8),
      summary,
      status: Uint32Array.of(0)
    })
  ).toThrow(/table/);
  expect(() =>
    getGPUSpatialErrorMaximumLikelihoodModelResult({
      table: new Float32Array(16),
      summary: new Float32Array(10),
      status: Uint32Array.of(0)
    })
  ).toThrow(/summary/);
  expect(() => getGPUSpatialMaximumLikelihoodDiagnostics(new Float32Array(3))).toThrow(/summary/);
});
