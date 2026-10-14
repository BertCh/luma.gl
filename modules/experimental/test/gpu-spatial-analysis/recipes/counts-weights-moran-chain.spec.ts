// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import type {
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../../../src/analysis-core/contracts/index';
import {
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPUGlobalSpatialStatistics
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  CHAIN_FEATURE_COUNT,
  createChainPoints,
  createChainPolygons,
  getChainMoranI,
  getChainQueenNeighbors
} from './counts-weights-moran-chain-fixture';
import {RecipeTestFixture} from './recipe-harness';
import {requireTestDevice} from '../../utils/require-test-device';

// A-5: the feature is the unit of support. Points -> uint32 counts per feature (choropleth recipe)
// -> queen contiguity on the SAME four-level polygon port (feature rows, one multipart feature)
// -> global Moran's I on the uint32 counts, with no cast and no part-to-feature collapse.
it('points -> counts -> W -> Moran chains on one feature table, multipart feature included', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const geometry = createChainPolygons();
  const {positions, counts: expectedCounts} = createChainPoints();
  const neighbors = getChainQueenNeighbors();
  // Feature 0 has two disjoint parts; as a feature it neighbors every other feature.
  expect(neighbors[0]).toEqual([1, 2, 3, 4]);
  const capacity = 32;
  const fixture = new RecipeTestFixture(device, 'counts-weights-moran-chain');
  try {
    const polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort> = {
      kind: 'polygons',
      positions: fixture.input(
        'polygon-positions',
        geometry.positions,
        'float32x2',
        geometry.positions.length / 2
      ),
      ringOffsets: fixture.input(
        'ring-offsets',
        geometry.ringOffsets,
        'uint32',
        geometry.ringOffsets.length
      ),
      polygonOffsets: fixture.input(
        'polygon-offsets',
        geometry.polygonOffsets,
        'uint32',
        geometry.polygonOffsets.length
      ),
      featureOffsets: fixture.input(
        'feature-offsets',
        geometry.featureOffsets,
        'uint32',
        geometry.featureOffsets.length
      )
    };
    const counts = fixture.output('counts', 'uint32', CHAIN_FEATURE_COUNT);
    const recipe = addPointsInPolygonsChoroplethRecipe(fixture.graph, {
      points: {
        kind: 'points',
        positions: fixture.input('points', positions, 'float32x2', positions.length / 2)
      },
      polygons: {...polygons, candidateCapacity: 4096},
      outputs: {counts: counts.view}
    });
    const offsets = fixture.output('offsets', 'uint32', CHAIN_FEATURE_COUNT + 1);
    const neighborIds = fixture.output('neighbors', 'uint32', capacity);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const contiguity = new GPUContiguityWeights({
      criterion: 'queen',
      polygons,
      output: {
        offsets: offsets.view,
        neighbors: neighborIds.view,
        weights: fixture.output('weights', 'float32', capacity).view,
        status: {overflow: overflow.view}
      }
    });
    fixture.graph.add(contiguity);
    const summary = fixture.output(
      'summary',
      'float32',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
    );
    // The uint32 counts of the recipe go straight into Moran: no cast kernel in between.
    expect(recipe.counts.format).toBe('uint32');
    fixture.graph.add(
      new GPUGlobalSpatialStatistics({
        weights: contiguity.outputWeights,
        values: recipe.counts,
        statistics: ['moran'],
        output: {summary: summary.view}
      })
    );
    fixture.run();

    expect(await fixture.readUint32(counts, CHAIN_FEATURE_COUNT)).toEqual(expectedCounts);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    const csrOffsets = await fixture.readUint32(offsets, CHAIN_FEATURE_COUNT + 1);
    const csrNeighbors = await fixture.readUint32(neighborIds, csrOffsets[CHAIN_FEATURE_COUNT]);
    const rows = Array.from({length: CHAIN_FEATURE_COUNT}, (_, row) =>
      csrNeighbors.slice(csrOffsets[row], csrOffsets[row + 1])
    );
    expect(rows).toEqual(neighbors);
    const moran = (await fixture.readFloat32(summary, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length))[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.statistic
    ];
    // f32 sums of five integer deviations: a few ulps of |I| <= 1.
    expect(Math.abs(moran - getChainMoranI(expectedCounts, neighbors))).toBeLessThan(1e-5);
  } finally {
    fixture.destroy();
  }
});
