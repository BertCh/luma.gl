// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/gpu-spatial-analysis/contracts/spatial-context';
import {GPUBoundsFilter} from '../../../src/gpu-spatial-analysis/bounds-filter';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const LONGITUDE_LATITUDE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
};
let serial = 0;
function view<
  Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x3'
>(graph: GPUCommandGraph, format: Format, length: number) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

it('GPUBoundsFilter accepts planar and plain longitude/latitude boxes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'bounds-filter'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUBoundsFilter({
      spatialContext,
      bounds: view(graph, 'float32x4', 4),
      box: view(graph, 'float32', 4),
      output: {
        ids: view(graph, 'uint32', 4),
        count: view(graph, 'uint32', 1),
        overflow: view(graph, 'uint32', 1)
      }
    });
  const plain: GPUSpatialContext = {coordinateSpace: 'longitude-latitude', metric: 'none'};
  expect(() => create(LONGITUDE_LATITUDE)).toThrow(/metric great-circle|does not support the/);
  expect(() =>
    create({coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'})
  ).toThrow(/does not support discrete-grid/);
  expect(create(plain).spatialContext.coordinateSpace).toBe('longitude-latitude');
  expect(create().spatialContext.coordinateSpace).toBe('planar');
});
