// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import type {GPUNeighborSearchParameters} from '../../../src/geospatial/neighbor-search';
import {createNeighborSearchHarness, type NeighborSearchReadback} from './neighbor-search-harness';
import {
  computeNeighborSearchOracle,
  createSeededRandom,
  type NeighborSearchOracleInput
} from './neighbor-search-oracle';

const BOUNDS = [0, 0, 100, 100] as const;

function createRandomPoints(count: number, seed: number, extent = 100): Float32Array {
  const random = createSeededRandom(seed);
  const positions = new Float32Array(count * 2);
  for (let index = 0; index < positions.length; index++) {
    positions[index] = random() * extent;
  }
  return positions;
}

function isClose(actual: number, expected: number, absolute: number, relative: number): boolean {
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected);
}

function expectMatchesOracle(
  result: NeighborSearchReadback,
  input: NeighborSearchOracleInput,
  label: string,
  capacity = Infinity
): void {
  const oracle = computeNeighborSearchOracle(input);
  const rows = oracle.counts.length;
  const total = oracle.offsets[rows];
  expect(result.neighborCounts, `${label} counts`).toEqual(oracle.counts);
  expect(result.totalNeighbors, `${label} total`).toBe(total);
  expect(result.overflow, `${label} overflow`).toBe(total > capacity ? 1 : 0);
  expect(result.offsets, `${label} offsets`).toEqual(
    oracle.offsets.map(o => Math.min(o, capacity))
  );
  if (total <= capacity) {
    expect(result.neighbors, `${label} neighbors`).toEqual(oracle.neighbors);
    for (let slot = 0; slot < total; slot++) {
      if (!isClose(result.distances[slot], oracle.distances[slot], 1e-6, 1e-6)) {
        throw new Error(
          `${label}: slot ${slot} distance ${result.distances[slot]} != ${oracle.distances[slot]}`
        );
      }
      if (!isClose(result.weights[slot], oracle.weights[slot], 1e-6, 2e-5)) {
        throw new Error(
          `${label}: slot ${slot} weight ${result.weights[slot]} != ${oracle.weights[slot]}`
        );
      }
    }
  }
}

it('GPUNeighborSearch kNN self join matches the brute-force oracle for k = 1, 4, 8, 32', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createRandomPoints(1500, 7);
  for (const k of [1, 4, 8, 32]) {
    const parameters: GPUNeighborSearchParameters = {bounds: BOUNDS};
    const harness = createNeighborSearchHarness(device, {
      mode: 'knn',
      k,
      positions,
      capacity: 1500 * k,
      parameters,
      gridSize: [24, 24]
    });
    const result = await harness.run();
    expectMatchesOracle(result, {mode: 'knn', k, positions, parameters}, `k=${k}`);
    harness.destroy();
  }
});

it('GPUNeighborSearch kNN resolves exact ties to the lowest ID and handles tiny and clustered sets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Integer lattice: many exactly equal distances.
  const lattice: number[] = [];
  for (let y = 0; y < 12; y++) {
    for (let x = 0; x < 12; x++) {
      lattice.push(x * 4, y * 4);
    }
  }
  // Coincident duplicates appended at the end.
  lattice.push(8, 8, 8, 8, 20, 20);
  const positions = new Float32Array(lattice);
  const rows = positions.length / 2;
  for (const gridSize of [
    [1, 1],
    [5, 3],
    [40, 40]
  ] as const) {
    for (const k of [3, 5, 9]) {
      const parameters: GPUNeighborSearchParameters = {bounds: [0, 0, 44, 44]};
      const harness = createNeighborSearchHarness(device, {
        mode: 'knn',
        k,
        positions,
        capacity: rows * k,
        parameters,
        gridSize
      });
      expectMatchesOracle(
        await harness.run(),
        {mode: 'knn', k, positions, parameters},
        `grid ${gridSize} k=${k}`
      );
      harness.destroy();
    }
  }
  // Fewer valid targets than k, and a bounded kNN radius.
  const few = new Float32Array([1, 1, 2, 2, 3, 3]);
  const parameters: GPUNeighborSearchParameters = {bounds: [0, 0, 4, 4]};
  const harness = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 8,
    positions: few,
    capacity: 24,
    parameters
  });
  expectMatchesOracle(await harness.run(), {mode: 'knn', k: 8, positions: few, parameters}, 'few');
  const bounded = {bounds: [0, 0, 4, 4], radius: 1.5} as const;
  expectMatchesOracle(
    await harness.run(bounded),
    {mode: 'knn', k: 8, positions: few, parameters: bounded},
    'bounded'
  );
  harness.destroy();
});

it('GPUNeighborSearch kNN and radius cross joins with masks match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createRandomPoints(900, 11);
  const queryPositions = createRandomPoints(400, 12);
  const random = createSeededRandom(13);
  const mask = Uint32Array.from({length: 900}, () => (random() < 0.7 ? 1 : 0));
  const queryMask = Uint32Array.from({length: 400}, () => (random() < 0.8 ? 1 : 0));
  // Some points outside the bounds are excluded.
  positions[0] = -5;
  queryPositions[3] = 150;
  const knnParameters: GPUNeighborSearchParameters = {
    bounds: BOUNDS,
    weightKind: 'inverseDistance',
    power: 2,
    distanceFloor: 0.01
  };
  const knn = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 6,
    positions,
    queryPositions,
    mask,
    queryMask,
    capacity: 2400,
    parameters: knnParameters
  });
  expectMatchesOracle(
    await knn.run(),
    {mode: 'knn', k: 6, positions, queryPositions, mask, queryMask, parameters: knnParameters},
    'knn cross'
  );
  knn.destroy();

  const radiusParameters: GPUNeighborSearchParameters = {bounds: BOUNDS, radius: 6};
  const radius = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    queryPositions,
    mask,
    queryMask,
    capacity: 20000,
    parameters: radiusParameters
  });
  expectMatchesOracle(
    await radius.run(),
    {mode: 'radius', positions, queryPositions, mask, queryMask, parameters: radiusParameters},
    'radius cross'
  );
  radius.destroy();
});

it('GPUNeighborSearch radius self join follows per-frame radius and weights without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createRandomPoints(2500, 21);
  const random = createSeededRandom(22);
  const mask = Uint32Array.from({length: 2500}, () => (random() < 0.9 ? 1 : 0));
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    mask,
    capacity: 200000,
    parameters: {bounds: BOUNDS, radius: 3},
    gridSize: [64, 64]
  });
  const frames: GPUNeighborSearchParameters[] = [
    {bounds: BOUNDS, radius: 3},
    {bounds: BOUNDS, radius: 0.5},
    {bounds: BOUNDS, radius: 8, rowStandardize: true},
    {bounds: BOUNDS, radius: 5, weightKind: 'inverseDistance', power: 1, distanceFloor: 0.001},
    {bounds: BOUNDS, radius: 5, weightKind: 'kernel', kernel: 'gaussian'},
    {bounds: BOUNDS, radius: 5, weightKind: 'kernel', kernel: 'triangular', rowStandardize: true},
    {bounds: BOUNDS, radius: 5, weightKind: 'kernel', kernel: 'epanechnikov'},
    {bounds: BOUNDS, radius: 5, weightKind: 'kernel', kernel: 'bisquare'},
    {bounds: BOUNDS, radius: 5, weightKind: 'kernel', kernel: 'uniform'},
    {bounds: [0, 0, 50, 50], radius: 4},
    {bounds: BOUNDS, radius: 0},
    {bounds: BOUNDS, radius: 250}
  ];
  for (const [frame, parameters] of frames.entries()) {
    expectMatchesOracle(
      await harness.run(parameters),
      {mode: 'radius', positions, mask, parameters},
      `frame ${frame}`,
      200000
    );
  }
  // New positions and mask between submissions.
  const moved = createRandomPoints(2500, 23);
  harness.writePositions(moved);
  harness.writeMask(new Uint32Array(2500).fill(1));
  const parameters = {bounds: BOUNDS, radius: 4} as const;
  expectMatchesOracle(
    await harness.run(parameters),
    {mode: 'radius', positions: moved, parameters},
    'moved'
  );
  expect(harness.buildCount).toBe(1);
  harness.destroy();
});

it('GPUNeighborSearch kNN adaptive kernel weights and row standardization match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createRandomPoints(800, 31);
  const harness = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 10,
    positions,
    capacity: 8000,
    parameters: {bounds: BOUNDS}
  });
  for (const kernel of ['gaussian', 'triangular', 'epanechnikov', 'bisquare', 'uniform'] as const) {
    for (const rowStandardize of [false, true]) {
      const parameters: GPUNeighborSearchParameters = {
        bounds: BOUNDS,
        weightKind: 'kernel',
        kernel,
        rowStandardize
      };
      expectMatchesOracle(
        await harness.run(parameters),
        {mode: 'knn', k: 10, positions, parameters},
        `${kernel} ${rowStandardize}`
      );
    }
  }
  // Coincident points: inverse distance without a floor writes 0, with a floor a finite weight.
  const coincident = new Float32Array([5, 5, 5, 5, 6, 5]);
  const small = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 2,
    positions: coincident,
    capacity: 6,
    parameters: {bounds: BOUNDS}
  });
  for (const distanceFloor of [0, 0.5]) {
    const parameters: GPUNeighborSearchParameters = {
      bounds: BOUNDS,
      weightKind: 'inverseDistance',
      distanceFloor
    };
    const result = await small.run(parameters);
    expectMatchesOracle(
      result,
      {mode: 'knn', k: 2, positions: coincident, parameters},
      `floor ${distanceFloor}`
    );
    expect(result.weights[0]).toBe(distanceFloor === 0 ? 0 : 2);
  }
  harness.destroy();
  small.destroy();
});

it('GPUNeighborSearch clamps offsets and flags overflow when the capacity is exceeded', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createRandomPoints(600, 41);
  const parameters: GPUNeighborSearchParameters = {bounds: BOUNDS, radius: 10};
  const oracle = computeNeighborSearchOracle({mode: 'radius', positions, parameters});
  const capacity = Math.floor(oracle.offsets[600] / 3);
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    capacity,
    parameters
  });
  const result = await harness.run();
  expectMatchesOracle(result, {mode: 'radius', positions, parameters}, 'radius overflow', capacity);
  // Complete rows before the capacity are exact.
  for (let row = 0; row < 600 && oracle.offsets[row + 1] <= capacity; row++) {
    const begin = oracle.offsets[row];
    const end = oracle.offsets[row + 1];
    expect(result.neighbors.slice(begin, end)).toEqual(oracle.neighbors.slice(begin, end));
  }
  // Every truncated row is still strictly ascending.
  for (let row = 0; row < 600; row++) {
    for (let slot = result.offsets[row] + 1; slot < result.offsets[row + 1]; slot++) {
      expect(result.neighbors[slot]).toBeGreaterThan(result.neighbors[slot - 1]);
    }
  }
  harness.destroy();

  const knn = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 4,
    positions,
    capacity: 1000,
    parameters: {bounds: BOUNDS}
  });
  expectMatchesOracle(
    await knn.run(),
    {mode: 'knn', k: 4, positions, parameters: {bounds: BOUNDS}},
    'knn overflow',
    1000
  );
  knn.destroy();
});

it('GPUNeighborSearch is bitwise reproducible across submissions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createRandomPoints(3000, 51);
  const parameters: GPUNeighborSearchParameters = {
    bounds: BOUNDS,
    radius: 4,
    weightKind: 'kernel',
    kernel: 'gaussian',
    rowStandardize: true
  };
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    capacity: 200000,
    parameters,
    gridSize: [8, 8]
  });
  const first = await harness.run();
  const second = await harness.run();
  expect(second.neighbors).toEqual(first.neighbors);
  expect(second.weightBits).toEqual(first.weightBits);
  harness.destroy();
});
