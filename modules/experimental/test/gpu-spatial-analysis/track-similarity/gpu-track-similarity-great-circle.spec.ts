// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  GREAT_CIRCLE_RADIUS,
  GREAT_CIRCLE_TRACK_CASES
} from './track-similarity-great-circle-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type Polyline = [number, number][];

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
const CASES = Object.values(GREAT_CIRCLE_TRACK_CASES);

function pack(polylines: readonly Polyline[]): {positions: Float32Array; offsets: Uint32Array} {
  const positions: number[] = [];
  const offsets = [0];
  for (const polyline of polylines) {
    for (const [longitude, latitude] of polyline) {
      positions.push(longitude, latitude);
    }
    offsets.push(positions.length / 2);
  }
  return {positions: Float32Array.from(positions), offsets: Uint32Array.from(offsets)};
}

async function runGreatCircle(
  device: Device,
  options: {sphereRadius?: number; densify?: number; units?: 'meters' | 'kilometers'}
): Promise<{hausdorff: number[]; frechet: number[]; maxDistance: number[]; status: number[]}> {
  const graph = new GPUCommandGraph(device, {id: 'similarity-great-circle'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const packedA = pack(CASES.map(testCase => testCase.a));
  const packedB = pack(CASES.map(testCase => testCase.b));
  const pairCount = CASES.length;
  const indices = Uint32Array.from(CASES, (_, index) => index);
  const out = {
    hausdorff: output(pairCount),
    frechet: output(pairCount),
    maxDistance: output(pairCount),
    status: output(pairCount)
  };
  graph.add(
    new GPUTrackSimilarity({
      id: 'similarity',
      spatialContext: {...GREAT_CIRCLE, sphereRadius: options.sphereRadius, units: options.units},
      linesA: {
        kind: 'lines',
        positions: importGraphBuffer(graph, 'positions-a', input(packedA.positions), 'float32x2'),
        lineOffsets: importGraphBuffer(graph, 'offsets-a', input(packedA.offsets), 'uint32')
      },
      linesB: {
        kind: 'lines',
        positions: importGraphBuffer(graph, 'positions-b', input(packedB.positions), 'float32x2'),
        lineOffsets: importGraphBuffer(graph, 'offsets-b', input(packedB.offsets), 'uint32')
      },
      pairA: importGraphBuffer(graph, 'pair-a', input(indices), 'uint32'),
      pairB: importGraphBuffer(graph, 'pair-b', input(indices), 'uint32'),
      output: {
        hausdorff: importGraphBuffer(graph, 'o-hausdorff', out.hausdorff, 'float32', pairCount),
        frechet: importGraphBuffer(graph, 'o-frechet', out.frechet, 'float32', pairCount),
        maximumDistance: importGraphBuffer(graph, 'o-max', out.maxDistance, 'float32', pairCount),
        pairStatus: importGraphBuffer(graph, 'o-status', out.status, 'uint32', pairCount)
      },
      densify: options.densify
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    hausdorff: await readFloat32(out.hausdorff, pairCount),
    frechet: await readFloat32(out.frechet, pairCount),
    maxDistance: await readFloat32(out.maxDistance, pairCount),
    status: await readUint32(out.status, pairCount)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/** f32 haversine and the segment fraction approximation: a few parts in a thousand of meters. */
function expectMeters(actual: number, expected: number, label: string): void {
  expect(Math.abs(actual - expected), label).toBeLessThanOrEqual(2e-3 * expected + 1);
}

it('GPUTrackSimilarity great-circle matches numpy/pyproj oracles at mid latitude, high latitude and across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runGreatCircle(device, {});
  for (const [index, [name, testCase]] of Object.entries(GREAT_CIRCLE_TRACK_CASES).entries()) {
    expect(result.status[index], `${name} status`).toBe(0);
    expectMeters(result.hausdorff[index], testCase.hausdorff, `${name} hausdorff`);
    expectMeters(result.frechet[index], testCase.frechet, `${name} frechet`);
    expectMeters(result.maxDistance[index], testCase.maxDistance, `${name} maxDistance`);
  }
  device.destroy?.();
});

it('GPUTrackSimilarity great-circle densifies with wrapped longitude deltas', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runGreatCircle(device, {densify: 0.5});
  for (const [index, [name, testCase]] of Object.entries(GREAT_CIRCLE_TRACK_CASES).entries()) {
    expectMeters(result.hausdorff[index], testCase.hausdorffDensify2, `${name} hausdorff`);
    expectMeters(result.frechet[index], testCase.frechetDensify2, `${name} frechet`);
  }
  device.destroy?.();
});

it('GPUTrackSimilarity great-circle distances scale with sphereRadius', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const radius = 1_000_000;
  const result = await runGreatCircle(device, {sphereRadius: radius});
  for (const [index, [name, testCase]] of Object.entries(GREAT_CIRCLE_TRACK_CASES).entries()) {
    const scale = radius / GREAT_CIRCLE_RADIUS;
    expectMeters(result.hausdorff[index], testCase.hausdorff * scale, `${name} hausdorff`);
    expectMeters(result.frechet[index], testCase.frechet * scale, `${name} frechet`);
    expectMeters(result.maxDistance[index], testCase.maxDistance * scale, `${name} maxDistance`);
  }
  device.destroy?.();
});

it('GPUTrackSimilarity reports kilometers against the binary64 great-circle oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runGreatCircle(device, {units: 'kilometers'});
  for (const [index, testCase] of CASES.entries()) {
    expect(result.status[index]).toBe(0);
    for (const [actual, expected] of [
      [result.hausdorff[index], testCase.hausdorff],
      [result.frechet[index], testCase.frechet],
      [result.maxDistance[index], testCase.maxDistance]
    ]) {
      expect(Math.abs(actual - expected / 1000)).toBeLessThanOrEqual(
        (2e-3 * expected) / 1000 + 0.001
      );
    }
  }
  device.destroy?.();
});
