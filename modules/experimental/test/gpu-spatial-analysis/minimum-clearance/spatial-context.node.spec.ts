// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUMinimumClearance} from '../../../src/gpu-spatial-analysis/minimum-clearance';
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

it('GPUMinimumClearance is planar only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'minimum-clearance'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUMinimumClearance({
      spatialContext,
      geometry: {
        kind: 'lines',
        positions: view(graph, 'float32x2', 4),
        lineOffsets: view(graph, 'uint32', 2)
      },
      output: {clearances: view(graph, 'float32', 1)}
    });
  expect(() => create(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  expect(() => create(AREA_FRAME)).toThrow(
    /measures distance; project into an azimuthal-equidistant frame/
  );
  expect(create(DISTANCE_FRAME).spatialContext).toEqual(DISTANCE_FRAME);
  expect(create(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
});

const DISTANCE_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-azimuthal-equidistant'}
} as const;
const AREA_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-lambert-equal-area'}
} as const;
