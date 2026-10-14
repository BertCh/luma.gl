// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUHilbertKeys} from '../../../src/gpu-spatial-analysis/hilbert-keys';
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

let serial = 0;
function view<
  Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x3'
>(graph: GPUCommandGraph, format: Format, length: number) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

it('GPUHilbertKeys accepts planar and plain longitude/latitude only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hilbert-keys'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUHilbertKeys({
      spatialContext,
      positions: view(graph, 'float32x2', 8),
      output: {keys: view(graph, 'uint32', 8)}
    });
  const plain: GPUSpatialContext = {coordinateSpace: 'longitude-latitude', metric: 'none'};
  expect(() => create(LONGITUDE_LATITUDE)).toThrow(/does not support the great-circle metric/);
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  expect(create(plain).spatialContext.coordinateSpace).toBe('longitude-latitude');
  expect(create(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
});
