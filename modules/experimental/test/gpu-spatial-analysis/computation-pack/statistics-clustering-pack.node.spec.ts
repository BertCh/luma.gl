// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {STATISTICS_CLUSTERING_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics-clustering';
import {expectFormulaBindingsExist, expectSpatialEntryWellFormed} from './formula-bindings';

const SOURCE_ROOT = 'modules/experimental/src/gpu-spatial-analysis';
const CLUSTERING_ROOT = `${SOURCE_ROOT}/spatial-clustering`;

/** Element → the files whose WGSL generators contain the identifiers its curated formula binds. */
const kernelSources: Record<string, string[]> = {
  GPURoseStatistic: [`${SOURCE_ROOT}/geographic-distribution/gpu-rose-statistic.ts`],
  GPUSpatialClustering: [`${CLUSTERING_ROOT}/spatial-clustering-kernels.ts`],
  GPUKMeans: [`${CLUSTERING_ROOT}/gpu-kmeans.ts`],
  GPUHierarchicalDensityClustering: [
    `${CLUSTERING_ROOT}/hierarchical-density-clustering-kernels.ts`
  ],
  GPUGroupGeometry: [`${SOURCE_ROOT}/group-geometry/gpu-group-geometry.ts`],
  GPUGroupConvexHull: [`${SOURCE_ROOT}/group-geometry/gpu-group-convex-hull.ts`],
  GPUSimilarLocations: [`${SOURCE_ROOT}/similar-locations/gpu-similar-locations.ts`],
  GPULocalOutlierFactor: [`${SOURCE_ROOT}/outlier-detection/gpu-local-outlier-factor.ts`],
  GPUEmergingHotSpots: [`${SOURCE_ROOT}/emerging-hot-spots/gpu-emerging-hot-spots.ts`],
  addSpaceTimeHotSpotsRecipe: [`${SOURCE_ROOT}/recipes/space-time-hot-spots-recipe.ts`]
};

test('the clustering table holds exactly the directions, clustering, groups, similarity, outlier and space-time elements', () => {
  expect(Object.keys(STATISTICS_CLUSTERING_OPERATORS).sort()).toEqual(
    Object.keys(kernelSources).sort()
  );
});

for (const [name, entry] of Object.entries(STATISTICS_CLUSTERING_OPERATORS)) {
  test(`${name}: statistics clustering pack and curated kernel bindings`, () => {
    expectSpatialEntryWellFormed(`statistics/${name}`, entry);
    if ('formula' in entry.formula) return;
    expect(kernelSources[name], `Add ${name}'s WGSL generator sources`).toBeDefined();
    expectFormulaBindingsExist(entry.formula, kernelSources[name]);
  });
}

test('clustering elements declare unique choices and planar guards carry projected metadata', () => {
  for (const [name, entry] of Object.entries(STATISTICS_CLUSTERING_OPERATORS)) {
    const names = entry.choices.map(choice => choice.name);
    expect(new Set(names).size, name).toBe(names.length);
    if (entry.guards.includes('coordinate-model')) {
      expect(entry.metadata?.['coordinateModel'], name).toBe('projected');
    }
  }
  expect(
    STATISTICS_CLUSTERING_OPERATORS['GPUEmergingHotSpots'].choices.map(choice => choice.name)
  ).toContain('pCorrection');
});
