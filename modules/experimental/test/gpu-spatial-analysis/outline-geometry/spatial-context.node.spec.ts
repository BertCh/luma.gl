// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  GPUOffsetCurve,
  GPUOutlineGeometry
} from '../../../src/gpu-spatial-analysis/outline-geometry';
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

it('GPUOffsetCurve is planar only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'offset-curve'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUOffsetCurve({
      spatialContext,
      positions: view(graph, 'float32x2', 4),
      lineOffsets: view(graph, 'uint32', 2),
      parameters: view(graph, 'float32', 4),
      output: {positions: view(graph, 'float32x2', 68)}
    });
  expect(() => create(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  expect(create(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
});

it('GPUOutlineGeometry accepts planar and great-circle longitude/latitude only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'outline'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUOutlineGeometry({
      spatialContext,
      positions: view(graph, 'float32x2', 2),
      lineOffsets: view(graph, 'uint32', 2),
      geometryType: 'lines',
      joinSegments: 4,
      parameters: view(graph, 'float32', 4),
      output: {positions: view(graph, 'float32x2', 2 * 18)}
    });
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  expect(create(LONGITUDE_LATITUDE).spatialContext.metric).toBe('great-circle');
  expect(create(LONGITUDE_LATITUDE).radius).toBeCloseTo(6371008.8, 1);
  expect(() => create({coordinateSpace: 'longitude-latitude', metric: 'none'})).toThrow(/metric/);
  expect(() =>
    create({coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'})
  ).toThrow(/does not support discrete-grid/);
});
