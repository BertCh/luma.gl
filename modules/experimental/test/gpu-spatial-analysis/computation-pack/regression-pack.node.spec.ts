// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {REGRESSION_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/regression';
import {expectFormulaBindingsExist, expectSpatialEntryWellFormed} from './formula-bindings';

const REGRESSION_DIRECTORY = 'modules/experimental/src/gpu-spatial-analysis/spatial-regression';

/** Generator sources of the WGSL each regression formula binds to. */
const kernelSources: Record<string, string[]> = {
  GPUOrdinaryLeastSquares: [`${REGRESSION_DIRECTORY}/gpu-ordinary-least-squares.ts`],
  GPUGeographicallyWeightedRegression: [
    `${REGRESSION_DIRECTORY}/gpu-geographically-weighted-regression.ts`,
    `${REGRESSION_DIRECTORY}/gwr-fit-wgsl.ts`
  ],
  GPUGeographicallyWeightedRegressionNonstationarityTest: [
    `${REGRESSION_DIRECTORY}/gpu-geographically-weighted-regression-nonstationarity-test.ts`
  ],
  GPUSpatialRegressionDiagnostics: [
    `${REGRESSION_DIRECTORY}/gpu-spatial-regression-diagnostics.ts`
  ],
  GPUSpatialTwoStageLeastSquares: [
    `${REGRESSION_DIRECTORY}/gpu-spatial-two-stage-least-squares.ts`
  ],
  GPUSpatialWeightsTraces: [`${REGRESSION_DIRECTORY}/gpu-spatial-weights-traces.ts`],
  GPUSpatialLagImpacts: [`${REGRESSION_DIRECTORY}/gpu-spatial-lag-impacts.ts`],
  GPUSpatialErrorGM: [`${REGRESSION_DIRECTORY}/gpu-spatial-error-gm.ts`],
  GPUSpatialLagMaximumLikelihood: [`${REGRESSION_DIRECTORY}/gpu-spatial-lag-maximum-likelihood.ts`],
  GPUSpatialErrorMaximumLikelihood: [
    `${REGRESSION_DIRECTORY}/gpu-spatial-error-maximum-likelihood.ts`
  ]
};

test('the regression table declares exactly the ten regression contributors', () => {
  expect(Object.keys(REGRESSION_OPERATORS).sort()).toEqual(Object.keys(kernelSources).sort());
});

for (const [name, entry] of Object.entries(REGRESSION_OPERATORS)) {
  test(`${name}: regression pack and curated kernel bindings`, () => {
    expectSpatialEntryWellFormed(`regression/${name}`, entry);
    if ('formula' in entry.formula) return;
    expect(kernelSources[name], `Add ${name}'s WGSL generator sources`).toBeDefined();
    expectFormulaBindingsExist(entry.formula, kernelSources[name]);
  });
}

test('regression formula binding checks reject stale identifiers', () => {
  const formula = REGRESSION_OPERATORS['GPUOrdinaryLeastSquares'].formula;
  if ('formula' in formula) throw new Error('Ordinary least squares must have a curated formula');
  expect(() =>
    expectFormulaBindingsExist(
      {
        ...formula,
        symbols: formula.symbols.map((symbol, index) =>
          index === 0 ? {...symbol, binding: 'wgsl:missingKernelIdentifier'} : symbol
        )
      },
      kernelSources['GPUOrdinaryLeastSquares']
    )
  ).toThrow();
});

test('the shared regression model keeps the complete-case tracts with finite centroids', async () => {
  const {loadRegressionModel} = await import(
    '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/s1-regression-model'
  );
  const model = await loadRegressionModel();
  // Three of the 791 source tracts miss a value of the response or a predictor.
  expect(model.sourceCount).toBe(791);
  expect(model.droppedCount).toBe(3);
  expect(model.tracts.count).toBe(788);
  expect(model.response.length).toBe(788);
  expect(model.predictors.length).toBe(788 * 3);
  expect(model.predictors.every(Number.isFinite)).toBe(true);
  expect(model.response.every(Number.isFinite)).toBe(true);
  // One ring and one polygon per tract, as the contiguity weights example assumes.
  expect(model.tracts.ringOffsets.length).toBe(789);
  expect(model.tracts.polygonRingOffsets.length).toBe(789);
  expect(model.tracts.vertices.length).toBe(model.tracts.ringOffsets[788] * 2);
  // Centroids are local-frame metres within Chicago's roughly 50 km extent, and distinct.
  expect(model.positions.length).toBe(788 * 2);
  expect(model.positions.every(Number.isFinite)).toBe(true);
  expect(Math.max(...model.positions.map(Math.abs))).toBeLessThan(40000);
  expect(
    new Set(
      Array.from({length: 788}, (_, i) => `${model.positions[2 * i]},${model.positions[2 * i + 1]}`)
    ).size
  ).toBe(788);
});
