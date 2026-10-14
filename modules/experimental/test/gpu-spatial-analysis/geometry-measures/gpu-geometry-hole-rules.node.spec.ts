// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures';
import {GPUShapeDescriptors} from '../../../src/gpu-spatial-analysis/shape-descriptors';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const PLANAR = {coordinateSpace: 'planar', metric: 'native', units: 'native'} as const;

function createGraph() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hole-rules'});
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  return {graph, view};
}

it('review D-13: GPUGeometryMeasures sets ring role by position', () => {
  const {graph, view} = createGraph();
  const measures = new GPUGeometryMeasures({
    spatialContext: PLANAR,
    geometryType: 'polygons',
    positions: view('positions', 'float32x2', 20),
    ringOffsets: view('ring-offsets', 'uint32', 6),
    polygonOffsets: view('polygon-offsets', 'uint32', 4),
    featureOffsets: view('feature-offsets', 'uint32', 3),
    output: {areas: view('areas', 'float32', 2)}
  });
  const sources = getGPUCommandNodeKernelSources(measures.getCommandNodes(graph));
  const serial = sources.find(entry => entry.id.endsWith('-features'))!.source;
  expect(() => new WgslReflect(serial)).not.toThrow();
  expect(serial).toContain('fn isShellRing(ring: u32, isFirstRing: bool) -> bool {\n  if (ring <');
  expect(serial).toContain('isShellRing(ring, isFirstRing)');
  // Feature -> polygon -> ring.
  expect(serial).toContain('polygonOffsets[polygonOffsetsOffset + polygonBegin]');
});

it('review D-13: layout validation and the line and point layouts', () => {
  const {view} = createGraph();
  const base = {
    spatialContext: PLANAR,
    positions: view('positions', 'float32x2', 8),
    output: {lengths: view('lengths', 'float32', 2)}
  } as const;
  expect(
    () =>
      new GPUGeometryMeasures({
        ...base,
        geometryType: 'polygons',
        ringOffsets: view('ring-offsets', 'uint32', 3),
        polygonOffsets: view('polygon-offsets-short', 'uint32', 1),
        featureOffsets: view('feature-offsets', 'uint32', 2)
      })
  ).toThrow(/polygonOffsets must contain at least two rows/);
  // Multi-lines: two lines in the first feature, one in the second.
  const lines = new GPUGeometryMeasures({
    ...base,
    geometryType: 'lines',
    lineOffsets: view('line-offsets', 'uint32', 4),
    featureOffsets: view('feature-line-offsets', 'uint32', 3)
  });
  expect(lines.featureCount).toBe(2);
  // Multi-points: feature f owns the points [featureOffsets[f], featureOffsets[f + 1]).
  const points = new GPUGeometryMeasures({
    ...base,
    geometryType: 'points',
    featureOffsets: view('point-offsets', 'uint32', 3)
  });
  expect(points.featureCount).toBe(2);
});

it('review D-13: GPUShapeDescriptors passes polygonOffsets to the measures and moments kernels', () => {
  const {graph, view} = createGraph();
  const descriptors = new GPUShapeDescriptors({
    positions: view('positions', 'float32x2', 20),
    ringOffsets: view('ring-offsets', 'uint32', 6),
    polygonOffsets: view('polygon-offsets', 'uint32', 4),
    featureOffsets: view('feature-offsets', 'uint32', 3),
    parameters: view('parameters', 'float32', 4),
    output: {orientation: view('orientation', 'float32', 2), areas: view('areas', 'float32', 2)}
  });
  const sources = getGPUCommandNodeKernelSources(descriptors.getCommandNodes(graph));
  const withOffsets = sources.filter(entry => entry.source.includes('polygonOffsets['));
  // The measures feature kernel and the moments kernel both resolve roles by position.
  expect(withOffsets.length).toBeGreaterThanOrEqual(2);
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
  }
  const moments = sources.find(entry => entry.id.endsWith('-moments'))!.source;
  expect(moments).toContain('isShellRing(ring, ring == ringStart)');
});
