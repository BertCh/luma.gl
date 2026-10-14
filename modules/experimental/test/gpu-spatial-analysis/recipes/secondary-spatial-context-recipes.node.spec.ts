// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPULonLatPolygonPreparation} from '../../../src/gpu-spatial-analysis/lon-lat-polygons';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

type PreparationOptions = {
  polygonsSpatialContext?: GPUSpatialContext;
  sourceSpatialContext?: GPUSpatialContext;
  targetSpatialContext?: GPUSpatialContext;
  studyAreaSpatialContext?: GPUSpatialContext;
  prepare?: boolean;
  preparedPositionCapacity?: number;
  preparationParameters?: GraphDataView<'float32'>;
};

function createFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'polygon-recipes'});
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const polygons = () => ({
    kind: 'polygons' as const,
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2),
    candidateCapacity: 64,
    crossingCapacity: 64
  });
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'inference',
    format: 'float32',
    length: 8
  });
  const create = (
    recipe: string,
    spatialContext: GPUSpatialContext,
    options: PreparationOptions = {}
  ) => {
    const common = {id: 'test-recipe', spatialContext, ...options};
    if (recipe === 'change-of-support') {
      return addChangeOfSupportRecipe(graph, {
        ...common,
        grid: makeGPURasterMetadata({width: 4, height: 4}),
        parameters: view('float32', 4),
        source: polygons(),
        target: polygons(),
        pairCapacity: 8
      });
    }
    if (recipe === 'hot-spot') {
      return addHotSpotAnalysisRecipe(graph, {
        ...common,
        parameters,
        source: {
          kind: 'points',
          positions: view('float32x2', 4),
          family: 'quadbin',
          resolution: 5,
          studyArea: polygons(),
          studyAreaSpatialContext: options.studyAreaSpatialContext,
          cellCapacity: 16
        }
      });
    }
    return addPointsInPolygonsChoroplethRecipe(graph, {
      ...common,
      backend: recipe === 'choropleth-group' ? 'group' : 'zonal',
      points: {kind: 'points', positions: view('float32x2', 4)},
      polygons: polygons()
    });
  };
  return {create, view, graph, destroy: () => parameters.destroy()};
}

const PRIMARY: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  datum: 'EPSG:4326'
};
const CASES = [
  ['choropleth-group', 'polygons'],
  ['choropleth-zonal', 'polygons'],
  ['change-of-support', 'source'],
  ['change-of-support', 'target'],
  ['hot-spot', 'studyArea']
] as const;

for (const [recipe, side] of CASES) {
  function createResult(secondary?: GPUSpatialContext, prepare = false, primary = PRIMARY) {
    const fixture = createFixture();
    try {
      const result = fixture.create(recipe, primary, {
        [`${side}SpatialContext`]: secondary,
        ...(prepare
          ? {
              prepare: true,
              preparedPositionCapacity: 32,
              preparationParameters: fixture.view('float32', 4)
            }
          : {})
      });
      const preparations = result.contributors.filter(
        (contributor): contributor is GPULonLatPolygonPreparation =>
          contributor instanceof GPULonLatPolygonPreparation
      );
      return {
        nodeIds: (fixture.graph as unknown as {nodes: {id: string}[]}).nodes.map(node => node.id),
        preparations
      };
    } finally {
      fixture.destroy();
    }
  }

  it(`${recipe} ${side} absent and matching contexts preserve the node list`, () => {
    const baseline = createResult();
    expect(baseline.nodeIds.length).toBeGreaterThan(0);
    expect(createResult({...PRIMARY}).nodeIds).toEqual(baseline.nodeIds);
  });
  it(`${recipe} ${side} rejects a datum mismatch`, () => {
    expect(() => createResult({...PRIMARY, datum: 'EPSG:4267'})).toThrowError(
      new Error(
        `test-recipe ${side}SpatialContext incompatible spatial contexts: datums differ (EPSG:4267 and EPSG:4326)`
      )
    );
  });
  it(`${recipe} ${side} requires linear edges without preparation`, () => {
    expect(() => createResult({...PRIMARY, edgeModel: 'great-circle'})).toThrowError(
      new Error(
        `test-recipe ${side}SpatialContext needs linear edges; prepare polygons with GPULonLatPolygonPreparation`
      )
    );
  });
  for (const edgeModel of ['linear', 'great-circle'] as const) {
    it(`${recipe} ${side} prepares separate great-circle edges with ${edgeModel} primary edges`, () => {
      const secondary = {...PRIMARY, edgeModel: 'great-circle' as const};
      const result = createResult(secondary, true, {...PRIMARY, edgeModel});
      const preparation = result.preparations[side === 'target' ? 1 : 0];
      expect(preparation.props.spatialContext).toBe(secondary);
      expect(preparation.spatialContext.edgeModel).toBe('linear');
      expect(result.nodeIds).toEqual(
        createResult(undefined, true, {...PRIMARY, edgeModel}).nodeIds
      );
      expect(() =>
        createResult({...secondary, datum: 'EPSG:4267'}, true, {...PRIMARY, edgeModel})
      ).toThrowError(
        new Error(
          `test-recipe ${side}SpatialContext incompatible spatial contexts: datums differ (EPSG:4267 and EPSG:4326)`
        )
      );
    });
  }
}
