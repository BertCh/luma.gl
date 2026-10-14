// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPULabelPoint} from '../../../src/gpu-spatial-analysis/label-point';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

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

it('GPULabelPoint is planar only', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'label-point'});
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPULabelPoint({
      spatialContext,
      positions: view(graph, 'float32x2', 8),
      ringOffsets: view(graph, 'uint32', 3),
      polygonOffsets: view(graph, 'uint32', 3),
      featureOffsets: view(graph, 'uint32', 3),
      output: {points: view(graph, 'float32x2', 2)}
    });
  expect(() => create(LONGITUDE_LATITUDE)).toThrow(UNSUPPORTED);
  expect(create().spatialContext.coordinateSpace).toBe('planar');
  expect(create(LOCAL_FRAME).spatialContext.frame?.id).toBe('test-frame');
});

it('review S-5: both search kernels fall back to the largest ring before the first vertex', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'label-point'});
  const contributor = new GPULabelPoint({
    positions: view(graph, 'float32x2', 8),
    ringOffsets: view(graph, 'uint32', 3),
    polygonOffsets: view(graph, 'uint32', 3),
    featureOffsets: view(graph, 'uint32', 3),
    output: {points: view(graph, 'float32x2', 2)}
  });
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  expect(sources.length).toBeGreaterThanOrEqual(2);
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
  }
  const serial = sources.find(entry => entry.id.endsWith('-search'))!.source;
  const workgroup = sources.find(entry => entry.id.endsWith('-search-workgroup'))!.source;
  expect(serial).toContain('getLargestRingBounds(ringStart, ringEnd, origin)');
  expect(workgroup).toContain('getLargestRingBoundsCooperative(lane, ringStart, ringEnd, origin)');
});
