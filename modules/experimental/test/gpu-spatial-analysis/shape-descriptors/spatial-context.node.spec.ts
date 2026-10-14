// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUShapeDescriptors} from '../../../src/gpu-spatial-analysis/shape-descriptors';
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

it('GPUShapeDescriptors is planar only and keeps the frame identity', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'shape-descriptors'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUShapeDescriptors({
      spatialContext,
      positions: view(graph, 'float32x2', 8),
      ringOffsets: view(graph, 'uint32', 3),
      polygonOffsets: view(graph, 'uint32', 3),
      featureOffsets: view(graph, 'uint32', 3),
      parameters: view(graph, 'float32', 4),
      output: {polsbyPopper: view(graph, 'float32', 2)}
    });
  expect(() => create(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  const framed = create(LOCAL_FRAME);
  expect(framed.spatialContext.frame?.id).toBe('test-frame');
  expect(framed.getCommandNodes(graph).length).toBeGreaterThan(0);
});
