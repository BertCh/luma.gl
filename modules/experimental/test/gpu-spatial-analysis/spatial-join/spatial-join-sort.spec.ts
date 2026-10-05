// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_SPATIAL_JOIN_NO_FEATURE as N} from '../../../src/gpu-spatial-analysis/spatial-join';
import {isSoftwareDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createRandom,
  joinNearestSegments,
  joinPointsInPolygons,
  type OraclePolygonFeature
} from './spatial-join-oracle';
import {
  createNearestJoinRun,
  createPolygonJoinRun,
  createSquareFeature,
  shuffleDeterministic,
  type SpatialJoinRunOptions,
  type SpatialJoinRunResult
} from './spatial-join-sort-utils';

type Point = [number, number];

async function runPolygonJoin(
  device: Device,
  features: OraclePolygonFeature[],
  points: Point[],
  options: Omit<SpatialJoinRunOptions, 'spatialSort'>
): Promise<{unsorted: SpatialJoinRunResult; sorted: SpatialJoinRunResult}> {
  const unsortedRun = createPolygonJoinRun(device, features, points, {
    ...options,
    spatialSort: false
  });
  const sortedRun = createPolygonJoinRun(device, features, points, {...options, spatialSort: true});
  const unsorted = await unsortedRun.readResult();
  const sorted = await sortedRun.readResult();
  // Encoding twice must be stable too: the sort rewrites every transient on every encoding.
  expect(await sortedRun.readResult()).toEqual(sorted);
  unsortedRun.destroy();
  sortedRun.destroy();
  return {unsorted, sorted};
}

async function runNearestJoin(
  device: Device,
  features: {starts: Point[]; ends?: Point[]},
  points: Point[],
  radius: number,
  options: Omit<SpatialJoinRunOptions, 'spatialSort'>
): Promise<{unsorted: SpatialJoinRunResult; sorted: SpatialJoinRunResult}> {
  const unsortedRun = createNearestJoinRun(device, features, points, radius, {
    ...options,
    spatialSort: false
  });
  const sortedRun = createNearestJoinRun(device, features, points, radius, {
    ...options,
    spatialSort: true
  });
  const unsorted = await unsortedRun.readResult();
  const sorted = await sortedRun.readResult();
  unsortedRun.destroy();
  sortedRun.destroy();
  return {unsorted, sorted};
}

/** Overlapping squares (size 1.25 on a pitch of 1) in shuffled row order. */
function createShuffledOverlappingGrid(columns: number, seed: number): OraclePolygonFeature[] {
  const features: OraclePolygonFeature[] = [];
  for (let row = 0; row < columns; row++) {
    for (let column = 0; column < columns; column++) {
      features.push(createSquareFeature(column, row, 1.25));
    }
  }
  return shuffleDeterministic(features, createRandom(seed)).shuffled;
}

function createRandomPoints(count: number, extent: number, seed: number): Point[] {
  const random = createRandom(seed);
  return Array.from({length: count}, () => [
    Math.fround(random() * extent),
    Math.fround(random() * extent)
  ]);
}

it('GPUPointInPolygonJoin spatialSort matches the unsorted join and the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const columns = 45;
  const features = createShuffledOverlappingGrid(columns, 5);
  const points = createRandomPoints(1500, columns + 0.25, 6);
  const {unsorted, sorted} = await runPolygonJoin(device, features, points, {
    candidateCapacity: 20000
  });
  expect(sorted).toEqual(unsorted);
  expect(unsorted.overflow).toBe(0);
  expect(unsorted.candidateCount).toBeGreaterThan(points.length);
  const oracle = joinPointsInPolygons(points, features, true);
  expect(sorted.featureRows).toEqual(oracle.featureRows);
  expect(sorted.counts).toEqual(oracle.counts);
});

it('GPUPointInPolygonJoin spatialSort handles empty features, one feature, and zero extent', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Empty features interleaved with real ones sort last but keep their rows.
  const squares = Array.from({length: 20}, (_, index) =>
    createSquareFeature((index * 7) % 13, (index * 5) % 11, 2)
  );
  const mixed: OraclePolygonFeature[] = [];
  for (const square of squares) {
    mixed.push([], square, []);
  }
  const points = createRandomPoints(400, 14, 7);
  let result = await runPolygonJoin(device, mixed, points, {candidateCapacity: 4096});
  expect(result.sorted).toEqual(result.unsorted);
  const oracle = joinPointsInPolygons(points, mixed, true);
  expect(result.sorted.featureRows).toEqual(oracle.featureRows);
  expect(result.sorted.counts).toEqual(oracle.counts);

  // Every feature has only non-finite vertices, so all bounds are empty.
  const invalidFeature: OraclePolygonFeature = [
    [
      [
        [Number.NaN, 0],
        [Number.NaN, 1],
        [Number.NaN, 2]
      ]
    ]
  ];
  result = await runPolygonJoin(device, [invalidFeature, invalidFeature, invalidFeature], points, {
    candidateCapacity: 16
  });
  expect(result.sorted).toEqual(result.unsorted);
  expect(result.sorted.featureRows.every(row => row === N)).toBe(true);
  expect(result.sorted.candidateCount).toBe(0);

  // One feature skips the sort entirely.
  const single = [createSquareFeature(2, 2, 4)];
  result = await runPolygonJoin(device, single, points, {candidateCapacity: 4096});
  expect(result.sorted).toEqual(result.unsorted);
  expect(result.sorted.featureRows).toEqual(joinPointsInPolygons(points, single, true).featureRows);

  // Identical features share one center, so the scene extent is zero.
  const identical = Array.from({length: 5}, () => createSquareFeature(1, 1, 3));
  result = await runPolygonJoin(device, identical, points, {candidateCapacity: 4096});
  expect(result.sorted).toEqual(result.unsorted);
  expect(result.sorted.counts[0]).toBe(joinPointsInPolygons(points, identical, true).counts[0]);
});

it('GPUPointInPolygonJoin spatialSort still reports leaf overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features = createShuffledOverlappingGrid(8, 9);
  const points = createRandomPoints(200, 8, 10);
  const {unsorted, sorted} = await runPolygonJoin(device, features, points, {
    candidateCapacity: 4096,
    leafCapacity: 16
  });
  expect(unsorted.overflow).toBe(1);
  expect(sorted.overflow).toBe(1);
});

it('GPUNearestFeatureJoin spatialSort matches the unsorted join and the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(21);
  const fround = (value: number) => Math.fround(value);
  const segments: [Point, Point][] = Array.from({length: 1500}, () => {
    const start: Point = [fround(random() * 100), fround(random() * 100)];
    const angle = random() * Math.PI * 2;
    const length = random() * 4;
    return [
      start,
      [fround(start[0] + Math.cos(angle) * length), fround(start[1] + Math.sin(angle) * length)]
    ];
  });
  const points = createRandomPoints(800, 100, 22);
  const {unsorted, sorted} = await runNearestJoin(
    device,
    {starts: segments.map(([start]) => start), ends: segments.map(([, end]) => end)},
    points,
    3,
    {candidateCapacity: 20000}
  );
  expect(sorted).toEqual(unsorted);
  expect(unsorted.overflow).toBe(0);
  const oracle = joinNearestSegments(points, segments, 3);
  for (const [index, row] of sorted.featureRows.entries()) {
    const ambiguous =
      oracle.secondDistances[index] - Math.max(oracle.distances[index], 0) < 1e-3 ||
      Math.abs(oracle.distances[index] - 3) < 1e-3;
    if (!ambiguous) {
      expect(row).toBe(oracle.featureRows[index]);
    }
  }
});

it('GPUNearestFeatureJoin spatialSort keeps the smallest-row tie-break', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Lattice point features in shuffled rows; query points sit at cell centers so four features are
  // exactly equidistant.
  const lattice: Point[] = [];
  for (let y = 0; y < 20; y++) {
    for (let x = 0; x < 20; x++) {
      lattice.push([x, y]);
    }
  }
  const {shuffled} = shuffleDeterministic(lattice, createRandom(31));
  const centers: Point[] = [];
  for (let y = 0; y < 19; y++) {
    for (let x = 0; x < 19; x++) {
      centers.push([x + 0.5, y + 0.5]);
    }
  }
  const {unsorted, sorted} = await runNearestJoin(device, {starts: shuffled}, centers, 1, {
    candidateCapacity: 4096
  });
  expect(sorted).toEqual(unsorted);
  expect(sorted.featureRows.every(row => row !== N)).toBe(true);
  if (!isSoftwareDevice(device)) {
    const oracle = joinNearestSegments(
      centers,
      shuffled.map(point => [point, point] as [Point, Point]),
      1
    );
    expect(sorted.featureRows).toEqual(oracle.featureRows);
    expect(sorted.counts).toEqual(oracle.counts);
  }
});

it('GPUNearestFeatureJoin spatialSort handles one feature, empty features, and leaf overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const points = createRandomPoints(100, 10, 41);
  let result = await runNearestJoin(device, {starts: [[5, 5]]}, points, 3, {
    candidateCapacity: 256
  });
  expect(result.sorted).toEqual(result.unsorted);

  // Non-finite segments have empty bounds and sort last without matching anything.
  const starts: Point[] = [
    [1, 1],
    [Number.NaN, 0],
    [8, 8],
    [Number.NaN, Number.NaN],
    [4, 6]
  ];
  const ends: Point[] = [
    [2, 2],
    [0, 0],
    [9, 8],
    [0, 0],
    [4, 7]
  ];
  result = await runNearestJoin(device, {starts, ends}, points, 3, {candidateCapacity: 512});
  expect(result.sorted).toEqual(result.unsorted);
  expect(result.sorted.counts[1]).toBe(0);
  expect(result.sorted.counts[3]).toBe(0);

  const many = createRandomPoints(64, 10, 42);
  result = await runNearestJoin(device, {starts: many}, points, 3, {
    candidateCapacity: 4096,
    leafCapacity: 16
  });
  expect(result.unsorted.overflow).toBe(1);
  expect(result.sorted.overflow).toBe(1);
});
