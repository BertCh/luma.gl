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
import {FIJI_TRANSMERIDIAN} from '../lon-lat-polygons/fixtures';
import {
  pointInLonLatPolygonPrepared,
  prepareLonLatPolygons
} from '../lon-lat-polygons/lon-lat-polygon-oracle';

type PreparationOptions = {
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
  return {create, view, destroy: () => parameters.destroy()};
}

const GEOGRAPHIC = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
for (const recipe of ['choropleth-group', 'choropleth-zonal', 'change-of-support', 'hot-spot']) {
  for (const validation of ['planar', 'capacity', 'parameters', 'invalid-capacity'] as const) {
    it(`${recipe} validates preparation ${validation}`, () => {
      const fixture = createFixture();
      try {
        const spatialContext =
          validation === 'planar'
            ? ({coordinateSpace: 'planar', metric: 'native'} as const)
            : GEOGRAPHIC;
        const options: PreparationOptions = {prepare: true};
        if (validation === 'parameters') options.preparedPositionCapacity = 32;
        if (validation === 'invalid-capacity') options.preparedPositionCapacity = 0;
        const message = {
          planar: 'prepare needs a longitude-latitude spatialContext',
          capacity: 'prepare needs preparedPositionCapacity',
          parameters: 'prepare needs preparationParameters',
          'invalid-capacity': 'preparedPositionCapacity must be a positive integer'
        }[validation];
        expect(() => fixture.create(recipe, spatialContext, options)).toThrow(
          `test-recipe ${message}`
        );
      } finally {
        fixture.destroy();
      }
    });
  }

  it(`${recipe} leaves preparation disabled by default`, () => {
    const fixture = createFixture();
    try {
      const result = fixture.create(recipe, GEOGRAPHIC);
      expect(
        result.contributors.some(contributor => contributor instanceof GPULonLatPolygonPreparation)
      ).toBe(false);
    } finally {
      fixture.destroy();
    }
  });

  it(`${recipe} validates the preparation parameter view length`, () => {
    const fixture = createFixture();
    try {
      expect(() =>
        fixture.create(recipe, GEOGRAPHIC, {
          prepare: true,
          preparedPositionCapacity: 32,
          preparationParameters: fixture.view('float32', 3)
        })
      ).toThrow(
        `test-recipe-${recipe === 'change-of-support' ? 'source-' : ''}prepare parameters must hold 4 float32 values`
      );
    } finally {
      fixture.destroy();
    }
  });

  it(`${recipe} preserves the consumer's great-circle edge error without preparation`, () => {
    const fixture = createFixture();
    const consumer = {
      'choropleth-group': 'join',
      'choropleth-zonal': 'zonal',
      'change-of-support': 'source-raster',
      'hot-spot': 'cover'
    }[recipe];
    try {
      expect(() => fixture.create(recipe, {...GEOGRAPHIC, edgeModel: 'great-circle'})).toThrow(
        `test-recipe-${consumer} needs linear edges; prepare polygons with GPULonLatPolygonPreparation`
      );
    } finally {
      fixture.destroy();
    }
  });

  it(`${recipe} inserts GPU preparation with doubled topology and feature rows preserved`, () => {
    const fixture = createFixture();
    try {
      const parameters = fixture.view('float32', 4);
      const result = fixture.create(
        recipe,
        {...GEOGRAPHIC, edgeModel: 'great-circle'},
        {
          prepare: true,
          preparedPositionCapacity: 32,
          preparationParameters: parameters
        }
      );
      const preparations = result.contributors.filter(
        (contributor): contributor is GPULonLatPolygonPreparation =>
          contributor instanceof GPULonLatPolygonPreparation
      );
      expect(preparations).toHaveLength(recipe === 'change-of-support' ? 2 : 1);
      expect(result.contributors[0]).toBe(preparations[0]);
      for (const preparation of preparations) {
        expect(preparation.props.parameters).toBe(parameters);
        expect(preparation.spatialContext.edgeModel).toBe('linear');
        expect(preparation.props.output.positions.length).toBe(32);
        expect(preparation.props.output.featureOffsets.length).toBe(2);
        expect(preparation.props.output.polygonOffsets.length).toBe(3);
        expect(preparation.props.output.ringOffsets.length).toBe(3);
        expect(
          result.status.stages.some(stage => stage.status === preparation.props.output.status)
        ).toBe(true);
      }
    } finally {
      fixture.destroy();
    }
  });
}

it('the binary64 prepared Fiji oracle counts both sides without counting Greenwich', () => {
  const prepared = prepareLonLatPolygons(FIJI_TRANSMERIDIAN);
  expect(
    [
      [179, -17],
      [-179, -17],
      [0, -17]
    ].map(([longitude, latitude]) => pointInLonLatPolygonPrepared([longitude, latitude], prepared))
  ).toEqual([true, true, false]);
});

it('hot-spot preparation requires a points source', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'lattice-prepare'});
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'inference',
    format: 'float32',
    length: 8
  });
  try {
    expect(() =>
      addHotSpotAnalysisRecipe(graph, {
        id: 'lattice',
        prepare: true,
        preparedPositionCapacity: 32,
        preparationParameters: createTransientView(graph, 'preparation', 'float32', 4),
        parameters,
        source: {
          kind: 'lattice',
          width: 2,
          height: 2,
          neighborCapacity: 12,
          values: createTransientView(graph, 'values', 'float32', 4)
        }
      })
    ).toThrow('lattice prepare needs a points source');
  } finally {
    parameters.destroy();
  }
});
