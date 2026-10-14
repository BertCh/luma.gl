// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts';
import {GPURasterJoin} from '../../../src/gpu-raster/polygon-rasterization';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const PRIMARY: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'none',
  datum: 'EPSG:4326'
};
function createJoin(zonesSpatialContext?: GPUSpatialContext) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  return new GPURasterJoin({
    id: 'join',
    spatialContext: PRIMARY,
    zonesSpatialContext,
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    points: createTransientView(graph, 'points', 'float32x2', 1),
    zones: createTransientView(graph, 'zones', 'uint32', 4),
    zoneCount: 1,
    output: {counts: createTransientView(graph, 'counts', 'uint32', 1)}
  });
}
it('raster join accepts absent and matching zones contexts', () => {
  expect(createJoin().spatialContext).toBe(PRIMARY);
  expect(createJoin({...PRIMARY}).spatialContext).toBe(PRIMARY);
});
it('raster join rejects a zones datum mismatch', () => {
  expect(() => createJoin({...PRIMARY, datum: 'EPSG:4267'})).toThrowError(
    new Error(
      'join zonesSpatialContext incompatible spatial contexts: datums differ (EPSG:4267 and EPSG:4326)'
    )
  );
});
it('raster join requires linear zones edges', () => {
  expect(() => createJoin({...PRIMARY, edgeModel: 'great-circle'})).toThrowError(
    new Error(
      'join zonesSpatialContext needs linear edges; prepare polygons with GPULonLatPolygonPreparation'
    )
  );
});
