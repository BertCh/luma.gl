// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '../../src/analysis-core/contracts/earth';
import {getNetworkMetricWGSL} from '../../src/gpu-network/network-spatial-context';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {interpolateGreatCircle} from '../gpu-spatial-analysis/geometry-measures/geodesic-oracle';
import {createGreatCircleNetworkMetric, wrapLongitudeDelta} from './network-metric-oracle';

it('network snapping keeps seam precision consistent with edge lengths and continuous arcs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Opposite seam quadrants, reversed edges, and already-unwrapped coordinates.
  // Every oracle input is rounded to the same f32 value uploaded to the GPU.
  const fixtures = [
    [179.9998, 60.0001, 179.9999, 60, -179.9995, 60.0002],
    [-179.9998, 60.0001, -179.9999, 60, 179.9995, 60.0002],
    [-179.9997, 60.0001, -179.9995, 60.0002, 179.9999, 60],
    [180.0003, 60.0001, 179.9999, 60, 180.0005, 60.0002],
    [-180.0003, 60.0001, -179.9999, 60, -180.0005, 60.0002],
    [-179.9998, 60.0001, 179.9999, 60, 179.9999, 60]
  ].map(values => values.map(Math.fround));
  const graph = new GPUCommandGraph(device, {id: 'network-seam-precision'});
  const inputBuffer = createInputBuffer(device, Float32Array.from(fixtures.flat()));
  const outputBuffer = createOutputBuffer(device, fixtures.length * 5);
  const input = importGraphBuffer(graph, 'input', inputBuffer, 'float32', fixtures.length * 6);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'float32', fixtures.length * 5);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'network-seam-precision',
      operation: 'GPUMapMatching',
      variant: 'seam-precision',
      invocationCount: fixtures.length,
      bindings: [
        {name: 'coordinates', view: input, type: 'f32', access: 'read'},
        {name: 'results', view: output, type: 'f32', access: 'read_write'}
      ],
      declarations: getNetworkMetricWGSL(GPU_EARTH_MEAN_RADIUS_METERS),
      body: /* wgsl */ `
let inputRow = coordinatesOffset + index * 6u;
let outputRow = resultsOffset + index * 5u;
let point = vec2f(coordinates[inputRow], coordinates[inputRow + 1u]);
let start = vec2f(coordinates[inputRow + 2u], coordinates[inputRow + 3u]);
let end = vec2f(coordinates[inputRow + 4u], coordinates[inputRow + 5u]);
let fraction = networkSegmentFraction(point, start, end);
results[outputRow] = fraction;
results[outputRow + 1u] = networkSegmentDistance(point, start, end, fraction);
results[outputRow + 2u] = networkInterpolate(start, end, fraction).x;
results[outputRow + 3u] = geodesicInterpolate(start, end, geodesicCentralAngle(start, end), 0.5).x;
results[outputRow + 4u] = geodesicLongitudeDelta(start.x, end.x);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(outputBuffer, fixtures.length * 5);
    const metric = createGreatCircleNetworkMetric();
    fixtures.forEach((values, index) => {
      const point = values.slice(0, 2);
      const start = values.slice(2, 4);
      const end = values.slice(4, 6);
      const fraction = metric.getSegmentFraction(point, start, end);
      const snapped = metric.interpolate(start, end, fraction);
      expect(actual[index * 5]).toBeCloseTo(fraction, 5);
      expect(actual[index * 5 + 1]).toBeCloseTo(metric.getDistance(point, snapped), 2);
      expect(actual[index * 5 + 2]).toBeCloseTo(snapped[0], 4);
      // Compare the continuous longitude directly, without hiding a wrong branch by wrapping.
      expect(actual[index * 5 + 3]).toBeCloseTo(interpolateGreatCircle(start, end, 0.5)[0], 4);
      expect(actual[index * 5 + 4]).toBe(wrapLongitudeDelta(end[0] - start[0]));
    });
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  }
});
