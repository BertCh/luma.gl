// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {CHAIN_FEATURE_COUNT, createChainPolygons} from './counts-weights-moran-chain-fixture';

// A-5 structure: the choropleth counts, the contiguity rows and the Moran rows are one feature
// table (a multipart feature included), and the only conversion is Moran's own uint32 node.
it('points -> counts -> W -> Moran builds on one feature table without a cast', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'counts-weights-moran-chain'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `chain-${serial++}`, format, length);
  const geometry = createChainPolygons();
  const polygons = {
    kind: 'polygons' as const,
    positions: view('float32x2', geometry.positions.length / 2),
    ringOffsets: view('uint32', geometry.ringOffsets.length),
    polygonOffsets: view('uint32', geometry.polygonOffsets.length),
    featureOffsets: view('uint32', geometry.featureOffsets.length)
  };
  // More polygon parts than features: part rows would not line up with the counts.
  expect(geometry.polygonOffsets.length - 1).toBeGreaterThan(CHAIN_FEATURE_COUNT);
  const recipe = addPointsInPolygonsChoroplethRecipe(graph, {
    points: {kind: 'points', positions: view('float32x2', 100)},
    polygons: {...polygons, candidateCapacity: 1024}
  });
  expect(recipe.featureCount).toBe(CHAIN_FEATURE_COUNT);
  expect(recipe.counts.format).toBe('uint32');
  const contiguity = new GPUContiguityWeights({
    criterion: 'queen',
    polygons,
    output: {
      offsets: view('uint32', CHAIN_FEATURE_COUNT + 1),
      neighbors: view('uint32', 32),
      weights: view('float32', 32),
      status: {overflow: view('uint32', 1)}
    }
  });
  expect(contiguity.getFeatureCount()).toBe(CHAIN_FEATURE_COUNT);
  graph.add(contiguity);
  const moran = new GPUGlobalSpatialStatistics({
    weights: contiguity.outputWeights,
    values: recipe.counts,
    statistics: ['moran'],
    output: {summary: view('float32', 64)}
  });
  const nodeIds = moran.getCommandNodes(graph).map(node => node.id);
  // Exactly one uint32 -> f32 conversion, inside Moran.
  expect(nodeIds.filter(id => id.includes('to-float32'))).toHaveLength(1);
});
