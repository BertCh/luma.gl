// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createGPURegressionModelResult,
  getGPURegressionPredictorNames,
  GPU_REGRESSION_PRECISION_CONDITION_LIMIT
} from '../../../src/gpu-spatial-analysis/contracts/model-result';

const coefficient = (name: string, estimate: number) => ({
  name,
  role: 'predictor' as const,
  estimate,
  standardError: 1,
  statistic: estimate,
  statisticKind: 't' as const,
  pValue: 0.5
});

it('createGPURegressionModelResult derives both sigma squared conventions and criteria', () => {
  const result = createGPURegressionModelResult({
    kind: 'ordinary-least-squares',
    coefficients: [coefficient('CONSTANT', 1), coefficient('x0', 2)],
    rowCount: 10,
    residualSumOfSquares: 16,
    standardErrorSigmaSquared: 'degrees-of-freedom',
    statusCode: 0,
    logLikelihood: -12,
    conditioning: {
      conditionNumber: 5,
      correlationConditionNumber: GPU_REGRESSION_PRECISION_CONDITION_LIMIT * 2,
      maximumVarianceInflationFactor: 3
    }
  });
  expect(result.parameterCount).toBe(2);
  expect(result.degreesOfFreedom).toBe(8);
  expect(result.sigmaSquared).toBe(2);
  expect(result.sigmaSquaredMaximumLikelihood).toBe(1.6);
  expect(result.akaikeInformationCriterion).toBe(28);
  expect(result.schwarzCriterion).toBeCloseTo(2 * Math.log(10) + 24, 12);
  expect(result.correctedAkaikeInformationCriterion).toBeCloseTo(28 + 12 / 7, 12);
  expect(result.conditioning?.illConditioned).toBe(true);
  expect(result.status).toEqual({code: 0, ok: true, converged: null, illConditioned: true});
  expect(result.inference.alternative).toBe('two-sided');
});

it('createGPURegressionModelResult leaves likelihood criteria null without a likelihood', () => {
  const result = createGPURegressionModelResult({
    kind: 'spatial-lag-two-stage-least-squares',
    coefficients: [coefficient('CONSTANT', 1)],
    rowCount: 4,
    residualSumOfSquares: 3,
    standardErrorSigmaSquared: 'maximum-likelihood',
    statusCode: 1
  });
  expect(result.logLikelihood).toBeNull();
  expect(result.akaikeInformationCriterion).toBeNull();
  expect(result.correctedAkaikeInformationCriterion).toBeNull();
  expect(result.conditioning).toBeNull();
  expect(result.status.ok).toBe(false);
});

it('getGPURegressionPredictorNames uses spreg default names', () => {
  expect(getGPURegressionPredictorNames(2)).toEqual(['CONSTANT', 'x0', 'x1']);
  expect(getGPURegressionPredictorNames(2, ['income'])).toEqual(['CONSTANT', 'income', 'x1']);
});
