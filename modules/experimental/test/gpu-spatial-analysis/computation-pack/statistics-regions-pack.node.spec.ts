// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {STATISTICS_REGION_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics-regions';
import {expectFormulaBindingsExist, expectSpatialEntryWellFormed} from './formula-bindings';

const SOURCE_ROOT = 'modules/experimental/src/gpu-spatial-analysis';
const REGIONALIZATION_ROOT = `${SOURCE_ROOT}/spatial-regionalization`;

/** Element → the files whose WGSL generators contain the identifiers its curated formula binds. */
const kernelSources: Record<string, string[]> = {
  GPUAZPRegions: [`${REGIONALIZATION_ROOT}/gpu-azp-regions.ts`],
  GPUMaxPRegions: [`${REGIONALIZATION_ROOT}/gpu-max-p-regions.ts`],
  GPUWardRegions: [`${REGIONALIZATION_ROOT}/gpu-ward-regions.ts`],
  GPUSpatialWeightsMinimumSpanningTree: [
    `${REGIONALIZATION_ROOT}/gpu-spatial-weights-minimum-spanning-tree.ts`
  ],
  GPUSkaterRegions: [`${REGIONALIZATION_ROOT}/gpu-skater-regions.ts`],
  GPURegionPartitionEvaluation: [`${REGIONALIZATION_ROOT}/gpu-region-partition-evaluation.ts`],
  GPULocationAllocation: [`${SOURCE_ROOT}/location-allocation/gpu-location-allocation.ts`]
};

test('the regions table holds exactly the regionalization and location-allocation elements', () => {
  expect(Object.keys(STATISTICS_REGION_OPERATORS).sort()).toEqual(
    Object.keys(kernelSources).sort()
  );
});

for (const [name, entry] of Object.entries(STATISTICS_REGION_OPERATORS)) {
  test(`${name}: statistics regions pack and curated kernel bindings`, () => {
    expectSpatialEntryWellFormed(`statistics/${name}`, entry);
    if ('formula' in entry.formula) return;
    expect(kernelSources[name], `Add ${name}'s WGSL generator sources`).toBeDefined();
    expectFormulaBindingsExist(entry.formula, kernelSources[name]);
  });
}

test('regionalization elements declare the choices their lenses record', () => {
  for (const name of Object.keys(STATISTICS_REGION_OPERATORS)) {
    const names = STATISTICS_REGION_OPERATORS[name].choices.map(choice => choice.name);
    expect(new Set(names).size, name).toBe(names.length);
  }
  expect(
    STATISTICS_REGION_OPERATORS['GPUWardRegions'].choices.map(choice => choice.name)
  ).toContain('mergeSchedule');
});
