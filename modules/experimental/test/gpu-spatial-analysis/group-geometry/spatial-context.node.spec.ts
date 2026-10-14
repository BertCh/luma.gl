// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/gpu-spatial-analysis/contracts/spatial-context';
import {
  GPUGroupConvexHull,
  GPUGroupGeometry
} from '../../../src/gpu-spatial-analysis/group-geometry';
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

it('GPUGroupGeometry and GPUGroupConvexHull are planar only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'group-geometry'});
  const createGeometry = (spatialContext?: GPUSpatialContext) =>
    new GPUGroupGeometry({
      spatialContext,
      positions: view(graph, 'float32x2', 8),
      labels: view(graph, 'uint32', 8),
      groupCount: 2,
      parameters: view(graph, 'float32', 8),
      output: {counts: view(graph, 'uint32', 2)}
    });
  const createHull = (spatialContext?: GPUSpatialContext) =>
    new GPUGroupConvexHull({
      spatialContext,
      positions: view(graph, 'float32x2', 8),
      labels: view(graph, 'uint32', 8),
      groupCount: 2,
      maximumVerticesPerGroup: 8,
      totalCapacity: 16,
      output: {
        vertexIndices: view(graph, 'uint32', 16),
        offsets: view(graph, 'uint32', 3),
        counts: view(graph, 'uint32', 2)
      }
    });
  expect(() => createGeometry(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(() => createHull(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(createGeometry().spatialContext.coordinateSpace).toBe('planar');
  expect(createHull().spatialContext.coordinateSpace).toBe('planar');
  expect(createGeometry(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
  expect(createHull(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
});
