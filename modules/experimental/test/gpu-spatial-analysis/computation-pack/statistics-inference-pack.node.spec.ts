// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {STATISTICS_INFERENCE_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics-inference';
import {expectFormulaBindingsExist, expectSpatialEntryWellFormed} from './formula-bindings';

const SOURCE_ROOT = 'modules/experimental/src/gpu-spatial-analysis';

/** Element → the files whose WGSL generators contain the identifiers its curated formula binds. */
const kernelSources: Record<string, string[]> = {
  GPUGlobalSpatialStatistics: [
    `${SOURCE_ROOT}/global-spatial-statistics/gpu-global-spatial-statistics.ts`
  ],
  GPUHotSpotAnalysis: [
    `${SOURCE_ROOT}/spatial-autocorrelation/gpu-hot-spot-analysis.ts`,
    `${SOURCE_ROOT}/spatial-autocorrelation/spatial-autocorrelation-kernels.ts`
  ],
  GPULocalGeary: [`${SOURCE_ROOT}/spatial-autocorrelation/gpu-additional-spatial-statistics.ts`],
  GPUSpatialPearson: [
    `${SOURCE_ROOT}/spatial-autocorrelation/gpu-additional-spatial-statistics.ts`
  ],
  GPUGammaStatistic: [
    `${SOURCE_ROOT}/spatial-autocorrelation/gpu-additional-spatial-statistics.ts`
  ],
  GPULocalPermutationTest: [`${SOURCE_ROOT}/permutation-inference/gpu-local-permutation-test.ts`],
  GPUGlobalPermutationTest: [`${SOURCE_ROOT}/permutation-inference/gpu-global-permutation-test.ts`],
  GPUMultipleTestingAdjustment: [
    `${SOURCE_ROOT}/multiple-testing/gpu-multiple-testing-adjustment.ts`
  ]
};

test('the inference table holds exactly the eight autocorrelation and inference elements', () => {
  expect(Object.keys(STATISTICS_INFERENCE_OPERATORS).sort()).toEqual(
    Object.keys(kernelSources).sort()
  );
});

for (const [name, entry] of Object.entries(STATISTICS_INFERENCE_OPERATORS)) {
  test(`${name}: statistics inference pack and curated kernel bindings`, () => {
    expectSpatialEntryWellFormed(`statistics/${name}`, entry);
    if ('formula' in entry.formula) return;
    expect(kernelSources[name], `Add ${name}'s WGSL generator sources`).toBeDefined();
    expectFormulaBindingsExist(entry.formula, kernelSources[name]);
  });
}

test('inference elements declare the choices their guards read', () => {
  const permutationTests = ['GPULocalPermutationTest', 'GPUGlobalPermutationTest'];
  for (const name of permutationTests) {
    const entry = STATISTICS_INFERENCE_OPERATORS[name];
    expect(
      entry.choices.map(choice => choice.name),
      name
    ).toContain('permutations');
    expect(entry.guards, name).toContain('pseudo-p-grid');
  }
  expect(STATISTICS_INFERENCE_OPERATORS['GPUMultipleTestingAdjustment'].guards).toContain(
    'uncorrected-p'
  );
});
