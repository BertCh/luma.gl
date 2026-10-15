// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {STATISTICS_SUMMARY_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics-summaries';
import {expectFormulaBindingsExist, expectSpatialEntryWellFormed} from './formula-bindings';

const SOURCE_ROOT = 'modules/experimental/src/gpu-spatial-analysis';

/** Element → the files whose WGSL generators contain the identifiers its curated formula binds. */
const kernelSources: Record<string, string[]> = {
  GPUAssuncaoReisStandardization: [
    `${SOURCE_ROOT}/spatial-autocorrelation/gpu-rate-standardization.ts`
  ],
  GPUEmpiricalBayesRates: [`${SOURCE_ROOT}/rate-smoothing/gpu-empirical-bayes-rates.ts`],
  GPUSpatialEmpiricalBayesRates: [
    `${SOURCE_ROOT}/rate-smoothing/gpu-spatial-empirical-bayes-rates.ts`
  ],
  GPUNeighborhoodSummary: [`${SOURCE_ROOT}/neighborhood-summary/gpu-neighborhood-summary.ts`],
  GPUGeographicDistribution: [
    `${SOURCE_ROOT}/geographic-distribution/gpu-geographic-distribution.ts`
  ],
  GPURegionStatistics: [`${SOURCE_ROOT}/region-statistics/gpu-region-statistics.ts`],
  GPURegionMask: [`${SOURCE_ROOT}/region-statistics/gpu-region-mask.ts`],
  GPUPickRegionMask: [`${SOURCE_ROOT}/region-statistics/gpu-pick-region-mask.ts`],
  GPUZonalStatistics: [
    `${SOURCE_ROOT}/zonal-statistics/gpu-zonal-statistics.ts`,
    `${SOURCE_ROOT}/zonal-statistics/zonal-statistics-passes.ts`
  ],
  GPUFocalStatistics: [`${SOURCE_ROOT}/spatial-interpolation/gpu-focal-statistics.ts`]
};

test('the summaries table holds exactly the ten rate, summary, zonal and focal elements', () => {
  expect(Object.keys(STATISTICS_SUMMARY_OPERATORS).sort()).toEqual(
    Object.keys(kernelSources).sort()
  );
});

for (const [name, entry] of Object.entries(STATISTICS_SUMMARY_OPERATORS)) {
  test(`${name}: statistics summaries pack and curated kernel bindings`, () => {
    expectSpatialEntryWellFormed(`statistics/${name}`, entry);
    if ('formula' in entry.formula) return;
    expect(kernelSources[name], `Add ${name}'s WGSL generator sources`).toBeDefined();
    expectFormulaBindingsExist(entry.formula, kernelSources[name]);
  });
}

test('rate elements declare the pooled reference rate their guard reads', () => {
  for (const name of [
    'GPUAssuncaoReisStandardization',
    'GPUEmpiricalBayesRates',
    'GPUSpatialEmpiricalBayesRates'
  ]) {
    expect(STATISTICS_SUMMARY_OPERATORS[name].guards, name).toContain('eb-reference-rate');
  }
  expect(STATISTICS_SUMMARY_OPERATORS['GPUZonalStatistics'].guards).toContain('conservation');
});
