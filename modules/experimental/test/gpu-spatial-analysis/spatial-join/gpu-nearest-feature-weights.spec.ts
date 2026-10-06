// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {assertValidCSR} from '../spatial-weights/spatial-weights-oracle';
import {createRandom} from './spatial-join-oracle';

type Point = [number, number];

/** CPU libpysal-style KNN: k nearest by (distance, row), self excluded, rows sorted by ID. */
function computeKnnOracle(
  queries: Point[],
  features: Point[],
  k: number,
  options: {selfJoin: boolean; maxDistance?: number}
) {
  const rows: {id: number; distance: number}[][] = queries.map((query, row) => {
    const candidates = features
      .map((feature, id) => ({
        id,
        distance: Math.hypot(query[0] - feature[0], query[1] - feature[1])
      }))
      .filter(
        entry =>
          !(options.selfJoin && entry.id === row) &&
          (options.maxDistance === undefined || entry.distance <= options.maxDistance)
      );
    candidates.sort((a, b) => a.distance - b.distance || a.id - b.id);
    return candidates.slice(0, k).sort((a, b) => a.id - b.id);
  });
  return rows;
}

async function runAdapter(
  queries: Point[],
  features: Point[],
  options: {
    k: number;
    selfJoin: boolean;
    weightType?: 'binary' | 'inverse-distance';
    maxDistance?: number;
  }
) {
  const device = await getWebGPUTestDevice();
  if (!device) return undefined;
  const rig = new WeightsRig(device);
  const joinK = options.selfJoin ? options.k + 1 : options.k;
  const slots = queries.length * joinK;
  const ids = rig.output('uint32', slots);
  const distances = rig.output('float32', slots);
  const counts = rig.output('uint32', queries.length);
  const joinOverflow = rig.output('uint32', 1);
  const output = rig.weightsOutput(queries.length, queries.length * options.k, true);
  const overflow = rig.output('uint32', 1);
  const maxDistance =
    options.maxDistance === undefined
      ? undefined
      : rig.input(Float32Array.of(options.maxDistance), 'float32', 1);
  const points = (values: Point[]) =>
    rig.input(Float32Array.from(values.flat()), 'float32x2', values.length);
  rig.run(
    new GPUNearestFeatureJoin({
      points: points(queries),
      features: {kind: 'points', positions: points(features)},
      k: joinK,
      maxDistance,
      neighborIds: ids.view,
      neighborCounts: counts.view,
      neighborDistances: distances.view,
      overflow: joinOverflow.view
    }),
    new GPUNearestFeatureWeights({
      neighborIds: ids.view,
      neighborCounts: counts.view,
      neighborDistances: distances.view,
      slotCapacity: joinK,
      k: options.k,
      excludeSelf: options.selfJoin,
      weightType: options.weightType,
      weights: output.spatialWeights,
      overflow: overflow.view
    })
  );
  const csr = await output.read();
  const flag = (await readUint32(overflow.buffer, 1))[0];
  rig.destroy();
  return {csr, overflow: flag};
}

function randomPoints(count: number, seed: number): Point[] {
  const random = createRandom(seed);
  return Array.from({length: count}, () => [random() * 100, random() * 100] as Point);
}

function expectMatchesOracle(
  csr: NonNullable<Awaited<ReturnType<typeof runAdapter>>>['csr'],
  oracle: ReturnType<typeof computeKnnOracle>,
  inverse: boolean
): void {
  let slot = 0;
  for (const [row, expected] of oracle.entries()) {
    expect(csr.offsets[row + 1] - csr.offsets[row], `row ${row} length`).toBe(expected.length);
    for (const entry of expected) {
      expect(csr.neighbors[slot], `row ${row} neighbor`).toBe(entry.id);
      expect(csr.distances[slot]).toBeCloseTo(entry.distance, 3);
      const weight = inverse ? 1 / Math.max(entry.distance, 1e-6) : 1;
      expect(Math.abs(csr.weights[slot] - weight)).toBeLessThanOrEqual(1e-4 * weight + 1e-6);
      slot++;
    }
  }
  expect(csr.neighbors.length).toBe(slot);
}

it('GPUNearestFeatureWeights builds KNN weights for a self-join (binary and inverse distance)', async () => {
  const points = randomPoints(60, 11);
  for (const weightType of ['binary', 'inverse-distance'] as const) {
    const result = await runAdapter(points, points, {k: 4, selfJoin: true, weightType});
    if (!result) return;
    assertValidCSR(result.csr, points.length, 'knn');
    const oracle = computeKnnOracle(points, points, 4, {selfJoin: true});
    expect(result.csr.neighbors.length).toBe(points.length * 4);
    expect(result.overflow).toBe(0);
    expectMatchesOracle(result.csr, oracle, weightType === 'inverse-distance');
  }
});

it('GPUNearestFeatureWeights shortens rows when fewer than k neighbors exist', async () => {
  const points = randomPoints(40, 5);
  const options = {k: 3, selfJoin: true, maxDistance: 18};
  const result = await runAdapter(points, points, options);
  if (!result) return;
  const oracle = computeKnnOracle(points, points, 3, options);
  // Non-trivial: both full and short rows occur.
  const lengths = oracle.map(row => row.length);
  expect(Math.min(...lengths)).toBeLessThan(3);
  expect(Math.max(...lengths)).toBe(3);
  assertValidCSR(result.csr, points.length, 'knn-short');
  expectMatchesOracle(result.csr, oracle, false);
});

it('GPUNearestFeatureWeights supports cross weights (queries to other features)', async () => {
  const queries = randomPoints(25, 21);
  const features = randomPoints(50, 22);
  const result = await runAdapter(queries, features, {k: 5, selfJoin: false});
  if (!result) return;
  const oracle = computeKnnOracle(queries, features, 5, {selfJoin: false});
  expectMatchesOracle(result.csr, oracle, false);
});

/** libpysal 4.15.0 `KNN.from_array` and `Graph.build_knn` (k = 4): 24 random points, identical neighbor sets. */
const LIBPYSAL_POINTS: Point[] = [
  [62.51, 89.721],
  [77.569, 22.521],
  [30.017, 87.355],
  [0.527, 82.123],
  [79.707, 46.793],
  [30.303, 27.843],
  [25.487, 44.508],
  [50.455, 55.35],
  [99.55, 79.266],
  [62.218, 98.896],
  [21.531, 16.021],
  [61.254, 4.394],
  [3.568, 51.489],
  [46.621, 91.717],
  [62.923, 51.412],
  [49.687, 24.751],
  [1.179, 19.24],
  [69.203, 20.061],
  [36.954, 0.373],
  [83.005, 15.446],
  [26.76, 88.033],
  [50.979, 84.715],
  [63.972, 74.177],
  [9.15, 54.114]
];
const LIBPYSAL_NEIGHBORS: number[][] = [
  [9, 13, 21, 22],
  [4, 11, 17, 19],
  [3, 13, 20, 21],
  [2, 12, 20, 23],
  [1, 7, 14, 17],
  [6, 10, 15, 18],
  [5, 7, 12, 23],
  [6, 14, 21, 22],
  [0, 4, 9, 22],
  [0, 13, 21, 22],
  [5, 6, 16, 18],
  [1, 15, 17, 19],
  [3, 6, 16, 23],
  [0, 2, 9, 21],
  [4, 7, 15, 22],
  [5, 11, 17, 18],
  [5, 6, 10, 12],
  [1, 11, 15, 19],
  [5, 10, 11, 15],
  [1, 4, 11, 17],
  [2, 3, 13, 21],
  [0, 9, 13, 22],
  [0, 7, 14, 21],
  [3, 5, 6, 12]
];

it('GPUNearestFeatureWeights matches libpysal KNN and Graph.build_knn neighbor sets', async () => {
  const result = await runAdapter(LIBPYSAL_POINTS, LIBPYSAL_POINTS, {k: 4, selfJoin: true});
  if (!result) return;
  LIBPYSAL_NEIGHBORS.forEach((expected, row) => {
    const begin = result.csr.offsets[row];
    const end = result.csr.offsets[row + 1];
    expect(result.csr.neighbors.slice(begin, end), `row ${row}`).toEqual(expected);
  });
});

/**
 * libpysal 4.15.0 breaks distance ties arbitrarily (KDTree order, and differently between `KNN` and
 * `Graph.build_knn`); this adapter keeps the lowest feature rows. Rows without a tie at the k-th
 * distance must agree with libpysal, whose k=3 sets on a 4x3 grid are pinned here.
 */
it('GPUNearestFeatureWeights agrees with libpysal on untied rows of a grid and keeps lowest rows on ties', async () => {
  const grid: Point[] = [];
  for (let x = 0; x < 4; x++) for (let y = 0; y < 3; y++) grid.push([x, y]);
  const result = await runAdapter(grid, grid, {k: 3, selfJoin: true});
  if (!result) return;
  const oracle = computeKnnOracle(grid, grid, 3, {selfJoin: true});
  expectMatchesOracle(result.csr, oracle, false);
  // libpysal KNN k=3: rows with a unique third neighbor (0, 2, 9, 11 have the diagonal at distance sqrt 2).
  const libpysalUntied: Record<number, number[]> = {
    0: [1, 3, 4],
    2: [1, 4, 5],
    9: [6, 7, 10],
    11: [7, 8, 10]
  };
  for (const [row, expected] of Object.entries(libpysalUntied)) {
    const begin = result.csr.offsets[Number(row)];
    expect(result.csr.neighbors.slice(begin, begin + 3)).toEqual(expected);
  }
});

it('GPUNearestFeatureWeights validates its props', () => {
  const view = {length: 4, format: 'uint32'} as never;
  expect(
    () =>
      new GPUNearestFeatureWeights({
        neighborIds: view,
        neighborCounts: view,
        slotCapacity: 0,
        weights: {} as never,
        overflow: view
      })
  ).toThrow(/slotCapacity/);
});
