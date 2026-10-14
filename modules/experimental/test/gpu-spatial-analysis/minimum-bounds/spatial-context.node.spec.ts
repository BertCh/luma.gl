// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/gpu-spatial-analysis/contracts/spatial-context';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds';
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

it('GPUMinimumBounds is planar only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'minimum-bounds'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUMinimumBounds({
      spatialContext,
      positions: view(graph, 'float32x2', 8),
      ringOffsets: view(graph, 'uint32', 3),
      output: {bounds: view(graph, 'float32x4', 2)}
    });
  expect(() => create(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  expect(create(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
});
