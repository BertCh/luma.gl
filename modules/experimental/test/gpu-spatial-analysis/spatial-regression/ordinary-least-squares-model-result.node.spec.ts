// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_PRECISION_WARNING
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {
  getGPUOrdinaryLeastSquaresModelResult,
  getStudentTSurvival,
  getStudentTTwoSidedPValue
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {
  createCollinearOrdinaryLeastSquaresScene,
  createHeteroskedasticOrdinaryLeastSquaresScene,
  createOrdinaryLeastSquaresScene,
  fitOrdinaryLeastSquaresOnCPU
} from './ordinary-least-squares-oracle';
import {SPREG_ORDINARY_LEAST_SQUARES_REFERENCE as REFERENCE} from './spreg-ols-reference';

// Scenes and oracle values: see spreg-ols-reference.ts (spreg 1.9.1, scipy 1.18.1).
const SCENES = {
  heteroskedastic: createHeteroskedasticOrdinaryLeastSquaresScene(41, 300),
  collinearModerate: createCollinearOrdinaryLeastSquaresScene(17, 200, 0.3),
  collinearSevere: createCollinearOrdinaryLeastSquaresScene(19, 200, 0.03),
  plain: createOrdinaryLeastSquaresScene(2, 400, 2, [3, -1], 0.5)
};

function expectRelative(actual: number, expected: number, tolerance: number, label: string) {
  const scale = Math.max(Math.abs(expected), 1e-300);
  expect(Math.abs(actual - expected) / scale, label).toBeLessThan(tolerance);
}

it('Student t survival and two-sided p-value match scipy.stats.t.sf', () => {
  for (const {t, df, sf, twoSided} of REFERENCE.studentT) {
    expectRelative(getStudentTSurvival(t, df), sf, 1e-10, `sf(${t}, ${df})`);
    expectRelative(getStudentTTwoSidedPValue(t, df), twoSided, 1e-10, `two-sided(${t}, ${df})`);
  }
  expect(getStudentTTwoSidedPValue(NaN, 5)).toBeNaN();
  expect(getStudentTTwoSidedPValue(Infinity, 5)).toBe(0);
  expect(getStudentTTwoSidedPValue(1, 0)).toBeNaN();
});

for (const [name, scene] of Object.entries(SCENES)) {
  it(`oracle summary matches spreg.OLS for the ${name} scene (Koenker, Breusch-Pagan, conditioning)`, () => {
    const reference = REFERENCE[name as keyof typeof SCENES];
    const fit = fitOrdinaryLeastSquaresOnCPU(scene);
    expect(fit.status).toBe(0);
    const s = fit.summary;
    expectRelative(
      s[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN],
      reference.breuschPagan[0],
      1e-8,
      'breuschPagan'
    );
    expectRelative(
      s[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE],
      reference.breuschPagan[1],
      1e-6,
      'breuschPagan p'
    );
    expectRelative(
      s[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT],
      reference.koenkerBassett[0],
      1e-8,
      'koenkerBassett'
    );
    expectRelative(
      s[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KOENKER_BASSETT_P_VALUE],
      reference.koenkerBassett[1],
      1e-6,
      'koenkerBassett p'
    );
    expectRelative(
      s[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER],
      reference.mulColli,
      1e-6,
      'condition number'
    );
    expectRelative(
      s[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER],
      reference.correlationConditionNumber,
      1e-6,
      'correlation condition number'
    );
    expectRelative(
      s[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR],
      reference.maximumVif,
      1e-6,
      'maximum VIF'
    );
    expect(s[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_PRECISION_WARNING]).toBe(
      name === 'collinearSevere' ? 1 : 0
    );
  });
}

it('classic Breusch-Pagan and Koenker-Bassett differ for skewed heteroskedastic errors', () => {
  const [bp] = REFERENCE.heteroskedastic.breuschPagan;
  const [kb] = REFERENCE.heteroskedastic.koenkerBassett;
  // The normal-error statistic is inflated by the excess kurtosis (Koenker 1981).
  expect(bp).toBeGreaterThan(5 * kb);
});

for (const [name, scene] of Object.entries(SCENES)) {
  it(`getGPUOrdinaryLeastSquaresModelResult reproduces spreg.OLS for the ${name} scene`, () => {
    const reference = REFERENCE[name as keyof typeof SCENES];
    const fit = fitOrdinaryLeastSquaresOnCPU(scene);
    const names = scene.predictorCount === 3 ? ['a', 'b', 'c'] : undefined;
    const result = getGPUOrdinaryLeastSquaresModelResult(
      {
        coefficients: Float32Array.from(fit.coefficients),
        standardErrors: Float32Array.from(fit.standardErrors),
        tStatistics: Float32Array.from(fit.tStatistics),
        summary: Float32Array.from(fit.summary),
        status: Uint32Array.from([fit.status])
      },
      {predictorNames: names}
    );
    expect(result.kind).toBe('ordinary-least-squares');
    expect(result.status.ok).toBe(true);
    expect(result.rowCount).toBe(reference.n);
    expect(result.parameterCount).toBe(scene.predictorCount + 1);
    expect(result.degreesOfFreedom).toBe(reference.n - scene.predictorCount - 1);
    expect(result.coefficients.map(row => row.name)).toEqual(
      names ? ['CONSTANT', ...names] : ['CONSTANT', 'x0', 'x1']
    );
    expect(result.coefficients[0].role).toBe('intercept');
    expect(result.coefficients[1].role).toBe('predictor');
    result.coefficients.forEach((row, index) => {
      expect(row.statisticKind).toBe('t');
      expectRelative(row.estimate, reference.betas[index], 5e-5, `beta ${index}`);
      expectRelative(row.standardError, reference.stdErrors[index], 5e-5, `se ${index}`);
      expectRelative(row.statistic, reference.tStats[index], 5e-5, `t ${index}`);
      // p-values from float32 t statistics: compare in log space for the tiny ones.
      const pReference = reference.pValues[index];
      if (pReference > 1e-12) {
        expectRelative(row.pValue, pReference, 1e-3, `p ${index}`);
      } else if (pReference === 0) {
        // scipy underflows to exactly 0 beyond about t = 38 for these degrees of freedom.
        expect(row.pValue).toBeLessThan(1e-250);
      } else {
        expect(Math.abs(Math.log(row.pValue) - Math.log(pReference))).toBeLessThan(0.5);
      }
    });
    expectRelative(result.residualSumOfSquares, reference.utu, 1e-6, 'rss');
    expectRelative(result.sigmaSquared, reference.sig2, 1e-6, 'sigma2');
    expectRelative(result.sigmaSquaredMaximumLikelihood, reference.sig2n, 1e-6, 'sigma2 ml');
    expect(result.standardErrorSigmaSquared).toBe('degrees-of-freedom');
    expectRelative(result.rSquared, reference.r2, 1e-6, 'r2');
    expectRelative(result.logLikelihood as number, reference.logll, 1e-6, 'log-likelihood');
    expectRelative(result.akaikeInformationCriterion as number, reference.aic, 1e-6, 'aic');
    expectRelative(result.schwarzCriterion as number, reference.schwarz, 1e-6, 'bic');
    const conditioning = result.conditioning!;
    expectRelative(conditioning.conditionNumber, reference.mulColli, 1e-5, 'condition number');
    expectRelative(
      conditioning.correlationConditionNumber,
      reference.correlationConditionNumber,
      1e-5,
      'correlation condition number'
    );
    expectRelative(conditioning.maximumVarianceInflationFactor, reference.maximumVif, 1e-5, 'vif');
    expect(conditioning.illConditioned).toBe(name === 'collinearSevere');
    expect(result.status.illConditioned).toBe(name === 'collinearSevere');
    expect(Object.keys(result.tests).sort()).toEqual([
      'breuschPagan',
      'jarqueBera',
      'koenkerBassett'
    ]);
    expectRelative(result.tests.breuschPagan.statistic, reference.breuschPagan[0], 1e-5, 'bp');
    expect(result.tests.breuschPagan.degreesOfFreedom).toBe(reference.breuschPagan[2]);
    expectRelative(result.tests.koenkerBassett.statistic, reference.koenkerBassett[0], 1e-5, 'kb');
    expect(result.tests.jarqueBera.degreesOfFreedom).toBe(2);
    expectRelative(result.tests.jarqueBera.statistic, reference.jarqueBera[0], 1e-4, 'jb');
    expect(result.inference.alternative).toBe('two-sided');
  });
}

it('getGPUOrdinaryLeastSquaresModelResult reports a failed fit without partial numbers', () => {
  const scene = createOrdinaryLeastSquaresScene(4, 3, 1, [2, 1], 0.1);
  const fit = fitOrdinaryLeastSquaresOnCPU(scene);
  expect(fit.status).toBe(2);
  const result = getGPUOrdinaryLeastSquaresModelResult({
    coefficients: Float32Array.from(fit.coefficients),
    standardErrors: Float32Array.from(fit.standardErrors),
    tStatistics: Float32Array.from(fit.tStatistics),
    summary: Float32Array.from(fit.summary),
    status: Uint32Array.from([fit.status])
  });
  expect(result.status.ok).toBe(false);
  expect(result.status.code).toBe(2);
  expect(result.conditioning).toBeNull();
  expect(result.logLikelihood).toBeNull();
  expect(result.coefficients.every(row => Number.isNaN(row.pValue))).toBe(true);
  expect(() =>
    getGPUOrdinaryLeastSquaresModelResult({
      coefficients: new Float32Array(2),
      standardErrors: new Float32Array(2),
      tStatistics: new Float32Array(2),
      summary: new Float32Array(10),
      status: new Uint32Array(1)
    })
  ).toThrow(/summary must hold/);
});
