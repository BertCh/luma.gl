// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const LONGITUDE_LATITUDE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
const DISCRETE_GRID = {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'} as const;

let serial = 0;

function createGraph() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `cos-recipes-${serial++}`});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const zones = () => ({
    kind: 'polygons' as const,
    positions: view('float32x2', 8),
    featureOffsets: view('uint32', 3),
    polygonOffsets: view('uint32', 3),
    ringOffsets: view('uint32', 3),
    crossingCapacity: 64
  });
  return {graph, view, zones};
}

function createChangeOfSupportProps(context: ReturnType<typeof createGraph>) {
  return {
    grid: makeGPURasterMetadata({width: 8, height: 6}),
    parameters: context.view('float32', 4),
    source: context.zones(),
    target: context.zones(),
    pairCapacity: 16
  };
}

it('the change-of-support recipe resolves a planar or longitude-latitude raster context', () => {
  const planar = createGraph();
  const planarResult = addChangeOfSupportRecipe(planar.graph, createChangeOfSupportProps(planar));
  expect(planarResult.spatialContext.coordinateSpace).toBe('planar');
  // Conservation diagnostics are always produced, and invalidCount joins the stage status.
  expect(planarResult.sourceAreas.length).toBe(planarResult.sourceZoneCount);
  expect(planarResult.sourceTransferredFractions.length).toBe(planarResult.sourceZoneCount);
  expect(planarResult.status.invalidCount.length).toBe(1);
  expect(planarResult.status.stages.at(-1)?.status.invalidCount).toBe(
    planarResult.status.invalidCount
  );

  const geographic = createGraph();
  const geographicResult = addChangeOfSupportRecipe(geographic.graph, {
    ...createChangeOfSupportProps(geographic),
    spatialContext: LONGITUDE_LATITUDE
  });
  expect(geographicResult.spatialContext.coordinateSpace).toBe('longitude-latitude');
  expect(geographicResult.contributors.length).toBe(planarResult.contributors.length);

  const unsupported = createGraph();
  expect(() =>
    addChangeOfSupportRecipe(unsupported.graph, {
      ...createChangeOfSupportProps(unsupported),
      spatialContext: DISCRETE_GRID
    })
  ).toThrow(/does not support discrete-grid/);
});

it('the change-of-support recipe accepts per-kind denominators and source anchors', () => {
  const context = createGraph();
  const result = addChangeOfSupportRecipe(context.graph, {
    ...createChangeOfSupportProps(context),
    denominator: {extensive: 'overlap'},
    sourceAnchorCells: context.view('uint32', 2)
  });
  expect(result.sourceZoneCount).toBe(2);
});

it('the choropleth recipe resolves a planar or longitude-latitude context', () => {
  const create = (
    spatialContext?: Parameters<typeof addPointsInPolygonsChoroplethRecipe>[1]['spatialContext']
  ) => {
    const context = createGraph();
    return addPointsInPolygonsChoroplethRecipe(context.graph, {
      points: {kind: 'points', positions: context.view('float32x2', 5)},
      polygons: {...context.zones(), candidateCapacity: 16},
      spatialContext
    });
  };
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  expect(create(LONGITUDE_LATITUDE).spatialContext.coordinateSpace).toBe('longitude-latitude');
  expect(() => create(DISCRETE_GRID)).toThrow(/does not support discrete-grid/);
});
