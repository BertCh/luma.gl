// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTrackSimilarity,
  GPU_TRACK_SIMILARITY_STATUS as STATUS
} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {
  computeFrechetOracle,
  computeHausdorffOracle,
  type Polyline
} from './track-similarity-oracle';

/** Shapely 2.1.2 `hausdorff_distance` and `frechet_distance` of these vertex lists (no densify). */
const SHAPELY_CASES: {a: Polyline; b: Polyline; hausdorff: number; frechet: number}[] = [
  {
    a: [
      [0, 0],
      [2.5, 7.94],
      [8.01, 2.44],
      [4.01, 9.91],
      [-5.88, 16.33]
    ],
    b: [
      [5, -3],
      [10.94, -3.64],
      [7.0, -8.07],
      [2.1, -9.17],
      [2.19, -8.1],
      [12.1, -2.25],
      [14.54, 7.53]
    ],
    hausdorff: 22.181598229162837,
    frechet: 22.23547615860744
  },
  {
    a: [
      [0, 0],
      [-5.69, -6.8],
      [-3.44, -15.92],
      [-12.73, -15.62],
      [-13.41, -7.28],
      [-10.83, -7.0],
      [-10.89, -12.05],
      [-20.65, -18.2],
      [-16.81, -24.19],
      [-19.42, -34.12],
      [-12.82, -41.03],
      [-17.47, -33.42]
    ],
    b: [
      [5, -3],
      [5.2, 3.94],
      [7.99, 8.78],
      [-0.18, 9.6],
      [-0.02, 17.03],
      [-2.79, 18.99],
      [-11.6, 16.74],
      [-15.14, 9.74],
      [-8.81, 7.33]
    ],
    hausdorff: 41.99801542930332,
    frechet: 48.52596933601636
  },
  {
    a: [[2, 3]],
    b: [
      [5, -3],
      [1.65, -5.03],
      [-4.29, -14.02],
      [-10.03, -5.71],
      [-3.23, -13.46],
      [-1.15, -13.88]
    ],
    hausdorff: 18.145095756154056,
    frechet: 18.145095756154056
  }
];

function packSet(polylines: readonly Polyline[]): {positions: Float32Array; offsets: Uint32Array} {
  const positions: number[] = [];
  const offsets = [0];
  for (const polyline of polylines) {
    for (const [x, y] of polyline) {
      positions.push(x, y);
    }
    offsets.push(positions.length / 2);
  }
  return {positions: Float32Array.from(positions), offsets: Uint32Array.from(offsets)};
}

async function runSimilarity(
  device: Device,
  setA: readonly Polyline[],
  setB: readonly Polyline[] | undefined,
  pairs: readonly [number, number][],
  options: {maxFrechetVertices?: number; activePairCount?: number} = {}
) {
  const graph = new GPUCommandGraph(device, {id: 'similarity-test'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffer.write(new Uint32Array(Math.max(length, 1)).fill(0x7f7f7f7f));
    buffers.push(buffer);
    return buffer;
  };
  const packedA = packSet(setA);
  const packedB = setB ? packSet(setB) : undefined;
  const pairCount = pairs.length;
  const out = {hausdorff: output(pairCount), frechet: output(pairCount), status: output(pairCount)};
  graph.add(
    new GPUTrackSimilarity({
      id: 'similarity',
      positionsA: importGraphBuffer(graph, 'positions-a', input(packedA.positions), 'float32x2'),
      offsetsA: importGraphBuffer(graph, 'offsets-a', input(packedA.offsets), 'uint32'),
      positionsB: packedB
        ? importGraphBuffer(graph, 'positions-b', input(packedB.positions), 'float32x2')
        : undefined,
      offsetsB: packedB
        ? importGraphBuffer(graph, 'offsets-b', input(packedB.offsets), 'uint32')
        : undefined,
      pairA: importGraphBuffer(
        graph,
        'pair-a',
        input(Uint32Array.from(pairs, pair => pair[0])),
        'uint32'
      ),
      pairB: importGraphBuffer(
        graph,
        'pair-b',
        input(Uint32Array.from(pairs, pair => pair[1])),
        'uint32'
      ),
      activePairCount:
        options.activePairCount === undefined
          ? undefined
          : importGraphBuffer(
              graph,
              'active',
              input(Uint32Array.of(options.activePairCount)),
              'uint32',
              1
            ),
      hausdorff: importGraphBuffer(graph, 'o-hausdorff', out.hausdorff, 'float32', pairCount),
      frechet: importGraphBuffer(graph, 'o-frechet', out.frechet, 'float32', pairCount),
      status: importGraphBuffer(graph, 'o-status', out.status, 'uint32', pairCount),
      maxFrechetVertices: options.maxFrechetVertices
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    hausdorff: await readFloat32(out.hausdorff, pairCount),
    frechet: await readFloat32(out.frechet, pairCount),
    status: await readUint32(out.status, pairCount)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectClose(actual: number, expected: number, label: string): void {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
  } else {
    expect(Math.abs(actual - expected), label).toBeLessThanOrEqual(2e-4 * Math.max(1, expected));
  }
}

it('GPUTrackSimilarity reproduces Shapely Hausdorff and Frechet distances', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const result = await runSimilarity(
    device,
    SHAPELY_CASES.map(testCase => testCase.a),
    SHAPELY_CASES.map(testCase => testCase.b),
    [
      [0, 0],
      [1, 1],
      [2, 2]
    ]
  );
  for (const [index, testCase] of SHAPELY_CASES.entries()) {
    expectClose(result.hausdorff[index], testCase.hausdorff, `hausdorff ${index}`);
    expectClose(result.frechet[index], testCase.frechet, `frechet ${index}`);
    expect(result.status[index]).toBe(0);
    // The oracle agrees with Shapely, so the oracle comparison below is meaningful.
    expectClose(computeHausdorffOracle(testCase.a, testCase.b), testCase.hausdorff, 'oracle h');
    expectClose(computeFrechetOracle(testCase.a, testCase.b), testCase.frechet, 'oracle f');
  }
  expect(result.frechet[1]).toBeGreaterThan(result.hausdorff[1]);
  device.destroy?.();
});

it('GPUTrackSimilarity matches the oracles on random pairs, empty tracks and invalid pairs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(3);
  const makeTrack = (count: number): Polyline => {
    const points: [number, number][] = [];
    let [x, y] = [random() * 100, random() * 100];
    for (let index = 0; index < count; index++) {
      points.push([x, y]);
      x += (random() - 0.5) * 20;
      y += (random() - 0.5) * 20;
    }
    return points;
  };
  // Lengths up to 70 cross the 64-lane Hausdorff stride; 130 exceeds the default Frechet cap.
  const lengths = [1, 2, 3, 17, 33, 64, 65, 70, 0, 130, 40, 9];
  const tracks = lengths.map(makeTrack);
  const pairs: [number, number][] = [];
  for (let a = 0; a < tracks.length; a++) {
    for (let b = a; b < tracks.length; b++) {
      pairs.push([a, b]);
    }
  }
  pairs.push([99, 0]);
  const result = await runSimilarity(device, tracks, undefined, pairs);
  let compared = 0;
  for (const [index, [a, b]] of pairs.entries()) {
    const label = `pair ${a},${b}`;
    if (a >= tracks.length) {
      expect(result.status[index], label).toBe(STATUS.invalidPair);
      expect(result.hausdorff[index]).toBeNaN();
      expect(result.frechet[index]).toBeNaN();
      continue;
    }
    if (tracks[a].length === 0 || tracks[b].length === 0) {
      expect(result.status[index], label).toBe(STATUS.emptyTrack);
      expect(result.hausdorff[index]).toBeNaN();
      continue;
    }
    expectClose(
      result.hausdorff[index],
      computeHausdorffOracle(tracks[a], tracks[b]),
      `${label} hausdorff`
    );
    compared++;
    if (tracks[a].length > 128 || tracks[b].length > 128) {
      expect(result.status[index], label).toBe(STATUS.frechetCapExceeded);
      expect(result.frechet[index]).toBeNaN();
    } else {
      expect(result.status[index], label).toBe(0);
      expectClose(
        result.frechet[index],
        computeFrechetOracle(tracks[a], tracks[b]),
        `${label} frechet`
      );
      expect(result.frechet[index]).toBeGreaterThanOrEqual(result.hausdorff[index] - 1e-3);
    }
  }
  expect(compared).toBeGreaterThan(50);
  device.destroy?.();
});

it('GPUTrackSimilarity honors maxFrechetVertices and activePairCount', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const a: Polyline = Array.from({length: 40}, (_, index) => [index, Math.sin(index)]);
  const b: Polyline = Array.from({length: 40}, (_, index) => [index + 0.5, Math.cos(index)]);
  const capped = await runSimilarity(device, [a], [b], [[0, 0]], {maxFrechetVertices: 32});
  expect(capped.status[0]).toBe(STATUS.frechetCapExceeded);
  expect(capped.frechet[0]).toBeNaN();
  expectClose(capped.hausdorff[0], computeHausdorffOracle(a, b), 'hausdorff beyond the cap');
  const limited = await runSimilarity(
    device,
    [a],
    [b],
    [
      [0, 0],
      [0, 0]
    ],
    {
      maxFrechetVertices: 64,
      activePairCount: 1
    }
  );
  expect(limited.status).toEqual([0, STATUS.invalidPair]);
  expectClose(limited.frechet[0], computeFrechetOracle(a, b), 'frechet');
  expect(limited.frechet[0]).toBeGreaterThan(0.1);
  expect(limited.frechet[1]).toBeNaN();
  device.destroy?.();
});
