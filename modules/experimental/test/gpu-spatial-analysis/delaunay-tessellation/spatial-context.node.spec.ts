// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  GPUDelaunayTessellation,
  GPUVoronoiDiagram
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const LONGITUDE_LATITUDE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
};
const LOCAL_FRAME: GPUSpatialContext = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'native',
  frame: {id: 'test-frame', kind: 'local-azimuthal-equidistant', origin: [10, 50]}
};
const UNSUPPORTED = /does not support longitude-latitude coordinates/;

let serial = 0;
function view<
  Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x3'
>(graph: GPUCommandGraph, format: Format, length: number) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

const status = (graph: GPUCommandGraph) => ({
  requiredCount: view(graph, 'uint32', 1),
  count: view(graph, 'uint32', 1),
  overflow: view(graph, 'uint32', 1)
});

it('GPUDelaunayTessellation and GPUVoronoiDiagram are planar only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'delaunay'});
  const createDelaunay = (spatialContext?: GPUSpatialContext) =>
    new GPUDelaunayTessellation({
      spatialContext,
      positions: view(graph, 'float32x2', 6),
      output: {triangles: view(graph, 'uint32x3', 4), status: status(graph) as never}
    });
  const createVoronoi = (spatialContext?: GPUSpatialContext) =>
    new GPUVoronoiDiagram({
      spatialContext,
      positions: view(graph, 'float32x2', 6),
      triangles: view(graph, 'uint32x3', 4),
      triangleCount: view(graph, 'uint32', 1),
      clipBounds: view(graph, 'float32x4', 1),
      output: {
        segments: view(graph, 'float32x4', 12),
        siteIds: view(graph, 'uint32x2' as never, 12) as never,
        status: status(graph) as never
      }
    });
  expect(() => createDelaunay(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(() => createVoronoi(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(createDelaunay().spatialContext.coordinateSpace).toBe('planar');
  expect(createVoronoi(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
});
