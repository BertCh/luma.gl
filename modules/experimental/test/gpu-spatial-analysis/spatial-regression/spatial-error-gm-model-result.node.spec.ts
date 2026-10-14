// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_ERROR_GM_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_ERROR_GM_SUMMARY_SIGMA_SQUARED
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-parameters';
import {getGPUSpatialErrorGMModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-model-result';
import {SPREG_ERROR_GM_HOM_REFERENCE} from './spatial-error-gm-reference';

/** Packs a spreg fixture the way the GPU kernels write `table` and `summary`. */
function createReadback(
  fixture: (typeof SPREG_ERROR_GM_HOM_REFERENCE)['lattice'],
  rowCount: number,
  estimator: 'hom' | 'kelejianPrucha',
  status = 0
) {
  const model = estimator === 'hom' ? fixture.hom : fixture.kelejianPrucha;
  const count = model.betas.length;
  const table = new Float32Array((count + 1) * 4).fill(NaN);
  for (let row = 0; row < count; row++) {
    table.set([model.betas[row], model.standardErrors![row], model.z[row], model.p[row]], row * 4);
  }
  table[count * 4] = model.lambda;
  if (estimator === 'hom') {
    table.set([model.standardErrors![count], model.z[count], model.p[count]], count * 4 + 1);
  }
  const summary = new Float32Array(GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH).fill(NaN);
  summary[0] = rowCount;
  summary[GPU_SPATIAL_ERROR_GM_SUMMARY_SIGMA_SQUARED] = model.sigma2;
  summary[GPU_SPATIAL_ERROR_GM_SUMMARY_PSEUDO_R_SQUARED] = model.pseudoRSquared;
  summary[GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER] = 12.5;
  summary[GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER] = 3;
  summary[GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR] = 1.5;
  return {table, summary, status: Uint32Array.of(status)};
}

it('getGPUSpatialErrorGMModelResult builds the shared result from GM_Error_Hom pins', () => {
  const fixture = SPREG_ERROR_GM_HOM_REFERENCE.lattice;
  const result = getGPUSpatialErrorGMModelResult(createReadback(fixture, 100, 'hom'), {
    predictorNames: ['INC', 'HOVAL']
  });
  expect(result.kind).toBe('spatial-error-generalized-moments');
  expect(result.coefficients.map(coefficient => coefficient.name)).toEqual([
    'CONSTANT',
    'INC',
    'HOVAL',
    'lambda'
  ]);
  expect(result.coefficients.map(coefficient => coefficient.role)).toEqual([
    'intercept',
    'predictor',
    'predictor',
    'spatial-error'
  ]);
  expect(result.coefficients.every(coefficient => coefficient.statisticKind === 'z')).toBe(true);
  // p-values are recomputed in float64 from z and agree with spreg's.
  result.coefficients.forEach((coefficient, index) => {
    expect(coefficient.estimate).toBeCloseTo([...fixture.hom.betas, fixture.hom.lambda][index], 5);
    expect(Math.abs(coefficient.pValue - fixture.hom.p[index])).toBeLessThan(
      Math.max(1e-6 * fixture.hom.p[index], 1e-12)
    );
  });
  expect(result.rowCount).toBe(100);
  expect(result.parameterCount).toBe(4);
  expect(result.degreesOfFreedom).toBe(96);
  expect(result.standardErrorSigmaSquared).toBe('maximum-likelihood');
  expect(result.sigmaSquaredMaximumLikelihood).toBeCloseTo(fixture.hom.sigma2, 4);
  expect(result.sigmaSquared).toBeCloseTo((fixture.hom.sigma2 * 100) / 96, 4);
  expect(result.pseudoRSquared).toBeCloseTo(fixture.hom.pseudoRSquared, 5);
  expect(result.logLikelihood).toBeNull();
  expect(result.akaikeInformationCriterion).toBeNull();
  expect(result.conditioning).toEqual({
    conditionNumber: 12.5,
    correlationConditionNumber: 3,
    maximumVarianceInflationFactor: 1.5,
    illConditioned: false
  });
  expect(result.status).toMatchObject({code: 0, ok: true, illConditioned: false});
});

it('getGPUSpatialErrorGMModelResult leaves the lambda statistics NaN for Kelejian-Prucha', () => {
  const fixture = SPREG_ERROR_GM_HOM_REFERENCE.lattice;
  const readback = createReadback(fixture, 100, 'kelejianPrucha');
  readback.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER] = 5e4;
  const result = getGPUSpatialErrorGMModelResult(readback);
  const lambda = result.coefficients[result.coefficients.length - 1];
  expect(lambda.name).toBe('lambda');
  expect(lambda.estimate).toBeCloseTo(fixture.kelejianPrucha.lambda, 5);
  expect(Number.isNaN(lambda.standardError)).toBe(true);
  expect(Number.isNaN(lambda.statistic)).toBe(true);
  expect(Number.isNaN(lambda.pValue)).toBe(true);
  expect(result.coefficients[1].name).toBe('x0');
  // A precision warning is reported without failing the fit.
  expect(result.conditioning?.illConditioned).toBe(true);
  expect(result.status).toMatchObject({code: 0, ok: true, illConditioned: true});
});

it('getGPUSpatialErrorGMModelResult reports a failed fit and validates its inputs', () => {
  const readback = createReadback(SPREG_ERROR_GM_HOM_REFERENCE.knn, 80, 'hom', 1);
  readback.table.fill(NaN);
  readback.summary.fill(NaN);
  readback.summary[0] = 80;
  const failed = getGPUSpatialErrorGMModelResult(readback);
  expect(failed.status).toMatchObject({code: 1, ok: false});
  expect(failed.conditioning).toBeNull();
  expect(Number.isNaN(failed.coefficients[0].estimate)).toBe(true);
  expect(() =>
    getGPUSpatialErrorGMModelResult({
      table: new Float32Array(8),
      summary: new Float32Array(3),
      status: Uint32Array.of(0)
    })
  ).toThrow(/summary/);
  expect(() =>
    getGPUSpatialErrorGMModelResult({
      table: new Float32Array(6),
      summary: new Float32Array(GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH),
      status: Uint32Array.of(0)
    })
  ).toThrow(/table/);
});
