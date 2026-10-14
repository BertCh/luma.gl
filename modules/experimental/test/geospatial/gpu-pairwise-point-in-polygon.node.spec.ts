// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../src/analysis-core/contracts';
import {GPUPairwisePointInPolygon} from '../../src/geospatial/gpu-pairwise-point-in-polygon';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('GPUPairwisePointInPolygon validates and exposes its declarative spatial context', () => {
  const create = (spatialContext?: GPUSpatialContext) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    return new GPUPairwisePointInPolygon({
      id: 'pairwise',
      spatialContext,
      queryPositions: createTransientView(graph, 'points', 'float32x2', 1),
      positions: createTransientView(graph, 'vertices', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      output: createTransientView(graph, 'output', 'uint32', 1)
    });
  };
  expect(create().spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native'
  });
  const planar: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'none'};
  expect(create(planar).spatialContext).toBe(planar);
  const longitudeLatitude: GPUSpatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'none'
  };
  expect(create(longitudeLatitude).spatialContext).toBe(longitudeLatitude);
  expect(create({...longitudeLatitude, edgeModel: 'linear'}).spatialContext.edgeModel).toBe(
    'linear'
  );
  expect(() => create({...longitudeLatitude, metric: 'great-circle', units: 'meters'})).toThrow(
    'pairwise does not support the great-circle metric'
  );
  expect(() => create({...longitudeLatitude, edgeModel: 'great-circle'})).toThrow(
    'pairwise needs linear edges; prepare polygons with GPULonLatPolygonPreparation'
  );
});
