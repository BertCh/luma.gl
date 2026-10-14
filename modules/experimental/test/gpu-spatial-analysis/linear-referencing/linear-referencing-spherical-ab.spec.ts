// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A/B timing for the great-circle arc BVH of GPULinearReferencing (algorithms roadmap P2). Prints
// the median wall time of 5 encodings (one-word readback) for 50k arcs and 100k points. To compare
// with the brute-force scan, run it on the current tree and once with
// `src/gpu-spatial-analysis/linear-referencing/*` replaced by the baseline copies the A5-MOVE stream
// saved (see its handoff):
//   npx vitest run --project headless modules/experimental/test/gpu-spatial-analysis/linear-referencing/linear-referencing-spherical-ab.spec.ts

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPULinearReferencing} from '../../../src/gpu-spatial-analysis/linear-referencing';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const LINE_COUNT = 1000;
const VERTICES_PER_LINE = 51;
const POINT_COUNT = 100_000;
const ENCODINGS = 5;

it('GPULinearReferencing great-circle A/B: 50k arcs x 100k points', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let state = 4242;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const positions = new Float32Array(LINE_COUNT * VERTICES_PER_LINE * 2);
  const lineOffsets = new Uint32Array(LINE_COUNT + 1);
  for (let line = 0; line < LINE_COUNT; line++) {
    let longitude = random() * 340 - 170;
    let latitude = random() * 140 - 70;
    for (let vertex = 0; vertex < VERTICES_PER_LINE; vertex++) {
      const row = line * VERTICES_PER_LINE + vertex;
      positions[row * 2] = longitude;
      positions[row * 2 + 1] = latitude;
      longitude = Math.max(-179, Math.min(179, longitude + (random() - 0.5) * 0.2));
      latitude = Math.max(-80, Math.min(80, latitude + (random() - 0.5) * 0.2));
    }
    lineOffsets[line + 1] = (line + 1) * VERTICES_PER_LINE;
  }
  const points = new Float32Array(POINT_COUNT * 2);
  for (let index = 0; index < POINT_COUNT; index++) {
    const row = Math.floor(random() * LINE_COUNT * VERTICES_PER_LINE);
    points[index * 2] = positions[row * 2] + (random() - 0.5) * 0.5;
    points[index * 2 + 1] = positions[row * 2 + 1] + (random() - 0.5) * 0.5;
  }
  const buffers = [
    createInputBuffer(device, positions),
    createInputBuffer(device, lineOffsets),
    createInputBuffer(device, points),
    createOutputBuffer(device, POINT_COUNT),
    createOutputBuffer(device, 1)
  ];
  const [positionsBuffer, offsetsBuffer, pointsBuffer, linesBuffer, overflowBuffer] = buffers;
  const radius = new GPUParameterBuffer(device, {
    id: 'ab-radius',
    format: 'float32',
    length: 1,
    values: new Float32Array([50_000])
  });
  const graph = new GPUCommandGraph(device, {id: 'linear-referencing-spherical-ab'});
  const rowCount = LINE_COUNT * VERTICES_PER_LINE;
  graph.add(
    new GPULinearReferencing({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
      queryPositions: importGraphBuffer(graph, 'points', pointsBuffer, 'float32x2', POINT_COUNT),
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      lineOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', LINE_COUNT + 1),
      radius: radius.importToGraph(graph),
      output: {
        lineIndices: importGraphBuffer(graph, 'lines', linesBuffer, 'uint32', POINT_COUNT),
        status: {overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)}
      }
    })
  );
  const compiled = await graph.compileAsync();
  const times: number[] = [];
  try {
    for (let encoding = 0; encoding <= ENCODINGS; encoding++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      await readUint32(overflowBuffer, 1);
      if (encoding > 0) {
        times.push(performance.now() - start);
      }
    }
  } finally {
    compiled.destroy();
    radius.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
  times.sort((left, right) => left - right);
  const median = times[Math.floor(times.length / 2)];
  // eslint-disable-next-line no-console
  console.log(
    `linear-referencing-spherical-ab arcs=${rowCount - LINE_COUNT} points=${POINT_COUNT} median=${median.toFixed(2)} ms`
  );
  expect(median).toBeGreaterThan(0);
}, 600000);
