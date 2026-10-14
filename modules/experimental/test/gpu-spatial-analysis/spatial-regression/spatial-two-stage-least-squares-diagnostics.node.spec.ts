// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE,
  TWO_STAGE_ILL_CONDITIONED_REFERENCE
} from './spreg-reference';
import {createNearestNeighborScene} from './spatial-regression-diagnostics-oracle';
import {
  createIllConditionedLagScene,
  createLagScene,
  createWeakInstrumentScene,
  fitSpatialTwoStageLeastSquaresOnCPU
} from './spatial-two-stage-least-squares-oracle';

const SCENES = {
  lag_12_21: () => createLagScene(12, 21, 0.5, true),
  lag_binary_10_5: () => createLagScene(10, 5, 0.1, false),
  lag_binary_10_5_high: () => createLagScene(10, 5, 0.255, false),
  weak_80_7: () => createWeakInstrumentScene(80, 7),
  knn_60_13: () => createNearestNeighborScene(60, 3, 13, 0.6, false),
  knn_90_29: () => createNearestNeighborScene(90, 3, 29, 0.6, false)
} as const;

function expectRelative(label: string, actual: number, expected: number, relative: number) {
  expect(Math.abs(actual - expected), `${label} ${actual} vs ${expected}`).toBeLessThanOrEqual(
    relative * Math.abs(expected)
  );
}

// The oracle computes the diagnostics with dense float64 algebra from the definitions in the
// summary slot docs; numpy computed the pins from the same definitions (rho from spreg GM_Lag).
for (const name of Object.keys(SCENES) as (keyof typeof SCENES)[]) {
  for (const order of [1, 2] as const) {
    it(`two-stage oracle diagnostics match the numpy pins on ${name}, instrument order ${order}`, () => {
      const scene = SCENES[name]();
      const reference = SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE[name];
      const fixture = order === 1 ? reference.order1 : reference.order2;
      const result = fitSpatialTwoStageLeastSquaresOnCPU(
        scene.weights,
        scene.predictors,
        scene.response,
        scene.predictorCount,
        order
      );
      const diagnostics = result.diagnostics;
      expectRelative('norm', diagnostics.weightsInfinityNorm, reference.weightsInfinityNorm, 1e-6);
      expectRelative('F', diagnostics.firstStageF, fixture.firstStageF, 1e-6);
      expectRelative(
        'partial R2',
        diagnostics.firstStagePartialRSquared,
        fixture.firstStagePartialRSquared,
        1e-6
      );
      expectRelative('condition', diagnostics.conditionNumber, fixture.conditionNumber, 1e-6);
      expectRelative(
        'correlation condition',
        diagnostics.correlationConditionNumber,
        fixture.correlationConditionNumber,
        1e-6
      );
      expectRelative(
        'VIF',
        diagnostics.maximumVarianceInflationFactor,
        fixture.maximumVarianceInflationFactor,
        1e-6
      );
      expectRelative(
        'instrument condition',
        diagnostics.instrumentCorrelationConditionNumber,
        fixture.instrumentCorrelationConditionNumber,
        1e-6
      );
      // The oracle rho is the float64 2SLS estimate, spreg's GM_Lag rho the pinned one.
      expect(result.coefficients[result.coefficients.length - 1]).toBeCloseTo(fixture.rho, 6);
      expect(diagnostics.rhoOutsideAdmissibleRange).toBe(fixture.rhoOutsideAdmissibleRange);
    });
  }
}

it('two-stage diagnostics separate a strong first stage, a weak one and an inadmissible rho', () => {
  const strong = SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE.knn_60_13.order1;
  const weak = SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE.weak_80_7.order1;
  expect(strong.firstStageF).toBeGreaterThan(100);
  expect(weak.firstStageF).toBeLessThan(10);
  expect(weak.rhoOutsideAdmissibleRange).toBe(true);
  expect(strong.rhoOutsideAdmissibleRange).toBe(false);
  // rho is inside (-1, 1) yet |rho| ||W||_inf >= 1: the check is conservative for raw weights.
  const raw = SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE.lag_binary_10_5_high;
  expect(Math.abs(raw.order1.rho)).toBeLessThan(1);
  expect(Math.abs(raw.order1.rho) * raw.weightsInfinityNorm).toBeGreaterThan(1);
  expect(raw.order1.rhoOutsideAdmissibleRange).toBe(true);
});

for (const order of [1, 2] as const) {
  it(`two-stage oracle matches np.linalg.cond on the ill-conditioned scene, order ${order}`, () => {
    const scene = createIllConditionedLagScene(12, 21, 0.5);
    const fixture =
      order === 1
        ? TWO_STAGE_ILL_CONDITIONED_REFERENCE.order1
        : TWO_STAGE_ILL_CONDITIONED_REFERENCE.order2;
    const result = fitSpatialTwoStageLeastSquaresOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount,
      order
    );
    expectRelative('condition', result.diagnostics.conditionNumber, fixture.conditionNumber, 1e-6);
    expectRelative(
      'correlation condition',
      result.diagnostics.correlationConditionNumber,
      fixture.correlationConditionNumber,
      1e-6
    );
  });
}
