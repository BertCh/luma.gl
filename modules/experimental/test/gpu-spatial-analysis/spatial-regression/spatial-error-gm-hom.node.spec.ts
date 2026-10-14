// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {SPREG_ERROR_GM_HOM_REFERENCE, SPREG_ERROR_GM_REFERENCE} from './spatial-error-gm-reference';
import {
  createErrorScene,
  fitSpatialErrorGMHomOnCPU,
  fitSpatialErrorGMOnCPU,
  getErrorGMConditioningOnCPU
} from './spatial-error-gm-oracle';
import type {DiagnosticsOracleWeights} from './spatial-regression-diagnostics-oracle';

type Scene = {
  weights: DiagnosticsOracleWeights;
  predictors: Float32Array;
  response: Float32Array;
  predictorCount: number;
};

function getScene(name: 'lattice' | 'knn' | 'binary'): Scene {
  if (name === 'binary') {
    return createErrorScene(324, 9, 0.5, 'lattice-binary');
  }
  const reference = SPREG_ERROR_GM_REFERENCE[name];
  return {
    weights: {
      offsets: Uint32Array.from(reference.offsets),
      neighbors: Uint32Array.from(reference.neighbors),
      weights: Float32Array.from(reference.weights)
    },
    predictors: Float32Array.from(reference.predictors),
    response: Float32Array.from(reference.response),
    predictorCount: 2
  };
}

function expectRelative(actual: number, expected: number, relative: number, floor = 1e-9) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
    relative * Math.max(Math.abs(expected), floor)
  );
}

// The oracle replaces spreg's n x n matrices A1, A2 by trace and moment-sum closed forms; matching
// spreg's psi, G, lambda, betas and standard errors checks every one of those derivations. Spreg
// minimizes with a local L-BFGS-B, the oracle globally, hence the 1e-4 tolerance on lambda.
for (const name of ['lattice', 'knn', 'binary'] as const) {
  it(`GM_Error_Hom closed-form oracle reproduces spreg GM_Error_Hom on the ${name} scene`, () => {
    const scene = getScene(name);
    const reference = SPREG_ERROR_GM_HOM_REFERENCE[name];
    const result = fitSpatialErrorGMHomOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    expect(result.lambdaBound).toBeCloseTo(0.99 / reference.rowSumNorm, 6);
    expect(result.lambda1).toBeCloseTo(reference.steps.lambda1, 4);
    reference.steps.momentMatrix.forEach((value, index) =>
      expectRelative(result.momentMatrix[index], value, 1e-4)
    );
    reference.steps.momentVector.forEach((value, index) =>
      expectRelative(result.momentVector[index], value, 1e-4)
    );
    reference.steps.psiAtLambda1.forEach((value, index) =>
      expectRelative(result.psiAtLambda1[index], value, 1e-3, 1e-6)
    );
    expect(result.lambda).toBeCloseTo(reference.hom.lambda, 4);
    reference.steps.psiAtLambda.forEach((value, index) =>
      expectRelative(result.psiAtLambda[index], value, 1e-3, 1e-6)
    );
    const count = reference.hom.betas.length;
    for (let index = 0; index < count; index++) {
      expect(result.coefficients[index]).toBeCloseTo(reference.hom.betas[index], 4);
      expect(result.standardErrors[index] / reference.hom.standardErrors[index]).toBeCloseTo(1, 3);
      expect(result.zStatistics[index] / reference.hom.z[index]).toBeCloseTo(1, 3);
    }
    expect(result.lambdaStandardError / reference.hom.standardErrors[count]).toBeCloseTo(1, 3);
    expect(result.lambdaZStatistic / reference.hom.z[count]).toBeCloseTo(1, 3);
    expect(result.lambdaPValue).toBeCloseTo(reference.hom.p[count], 4);
    expect(result.sigmaSquared / reference.hom.sigma2).toBeCloseTo(1, 4);
    expect(result.pseudoRSquared).toBeCloseTo(reference.hom.pseudoRSquared, 4);
  });
}

it('GM_Error (Kelejian-Prucha) oracle reproduces spreg GM_Error on the non-row-standardized scene', () => {
  const scene = getScene('binary');
  const reference = SPREG_ERROR_GM_HOM_REFERENCE.binary.kelejianPrucha;
  const result = fitSpatialErrorGMOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  expect(result.lambda).toBeCloseTo(reference.lambda, 3);
  reference.betas.forEach((beta, index) => {
    expect(result.coefficients[index]).toBeCloseTo(beta, 3);
    expect(result.standardErrors[index] / reference.standardErrors[index]).toBeCloseTo(1, 2);
  });
  expect(result.sigmaSquared / reference.sigma2).toBeCloseTo(1, 3);
});

it('GM_Error_Hom gives lambda a standard error that differs from the Kelejian-Prucha estimate', () => {
  const scene = getScene('lattice');
  const reference = SPREG_ERROR_GM_HOM_REFERENCE.lattice;
  expect(reference.kelejianPrucha.standardErrors.length).toBe(reference.hom.betas.length);
  expect(reference.hom.standardErrors.length).toBe(reference.hom.betas.length + 1);
  const result = fitSpatialErrorGMHomOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  expect(result.lambdaStandardError).toBeGreaterThan(0.01);
  expect(Math.abs(result.lambda - reference.kelejianPrucha.lambda)).toBeGreaterThan(0.01);
});

// The GPU rebuilds the Belsley Gram matrix of the raw filtered [1, X] from the centered-model moments
// and the means; numpy computed the pins directly from the raw filtered design.
for (const name of ['lattice', 'knn', 'binary'] as const) {
  for (const estimator of ['hom', 'kelejianPrucha'] as const) {
    it(`GM conditioning oracle matches numpy on the ${name} scene (${estimator})`, () => {
      const scene = getScene(name);
      const model = SPREG_ERROR_GM_HOM_REFERENCE[name][estimator];
      const result = getErrorGMConditioningOnCPU(
        scene.weights,
        scene.predictors,
        scene.predictorCount,
        model.lambda
      );
      expectRelative(result.conditionNumber, model.conditioning.conditionNumber, 1e-6);
      expectRelative(
        result.correlationConditionNumber,
        model.conditioning.correlationConditionNumber,
        1e-6
      );
      expectRelative(
        result.maximumVarianceInflationFactor,
        model.conditioning.maximumVarianceInflationFactor,
        1e-6
      );
    });
  }
}
