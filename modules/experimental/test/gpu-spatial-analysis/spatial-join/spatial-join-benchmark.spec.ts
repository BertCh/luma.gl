// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {getGPUSpatialJoinCapacityPlan} from '../../../src/gpu-spatial-analysis/contracts/index';
import './spatial-relate-benchmark';
import {createRandom, type OraclePolygonFeature} from './spatial-join-oracle';
import {
  createNearestJoinRun,
  createPolygonJoinRun,
  createSquareFeature,
  shuffleDeterministic,
  type SpatialJoinRun,
  type SpatialJoinRunOptions
} from './spatial-join-sort-utils';
import {
  buildModelBVH,
  getLeafCapacity,
  getMortonLeafOrder,
  measureTraversal,
  type FeatureBounds
} from './spatial-join-traversal-model';

type Point = [number, number];

const WARM_UP_ENCODINGS = 3;
const TIMED_ENCODINGS = 10;
const MODEL_POINT_COUNT = 5000;

type BenchmarkRow = {
  scenario: string;
  features: number;
  points: number;
  candidates: number;
  visitedUnsorted: number;
  visitedSorted: number;
  millisecondsUnsorted: number;
  millisecondsSorted: number;
  plannedWorkItems: number;
  plannedPeakTransientBytes: number;
};

const rows: BenchmarkRow[] = [];

async function measureMilliseconds(run: SpatialJoinRun): Promise<number> {
  for (let index = 0; index < WARM_UP_ENCODINGS; index++) {
    await run.encodeAndWait();
  }
  const timings: number[] = [];
  for (let index = 0; index < TIMED_ENCODINGS; index++) {
    const start = performance.now();
    await run.encodeAndWait();
    timings.push(performance.now() - start);
  }
  timings.sort((left, right) => left - right);
  return (timings[TIMED_ENCODINGS / 2 - 1] + timings[TIMED_ENCODINGS / 2]) / 2;
}

/** Runs both configurations, checks identical results, and records one table row. */
async function benchmarkScenario(
  scenario: string,
  featureBounds: FeatureBounds[],
  points: Point[],
  radius: number,
  createRun: (options: SpatialJoinRunOptions) => SpatialJoinRun,
  candidateCapacity: number
): Promise<void> {
  const unsortedRun = createRun({spatialSort: false, candidateCapacity});
  const sortedRun = createRun({spatialSort: true, candidateCapacity});
  const unsortedResult = await unsortedRun.readResult();
  const sortedResult = await sortedRun.readResult();
  expect(unsortedResult.overflow).toBe(0);
  expect(sortedResult).toEqual(unsortedResult);
  const millisecondsUnsorted = await measureMilliseconds(unsortedRun);
  const millisecondsSorted = await measureMilliseconds(sortedRun);
  unsortedRun.destroy();
  sortedRun.destroy();

  const leafCapacity = getLeafCapacity(featureBounds.length);
  const identityOrder = featureBounds.map((_, row) => row);
  const modelPoints = points.slice(0, MODEL_POINT_COUNT);
  const unsortedModel = measureTraversal(
    buildModelBVH(featureBounds, identityOrder, leafCapacity),
    modelPoints,
    radius
  );
  const sortedModel = measureTraversal(
    buildModelBVH(featureBounds, getMortonLeafOrder(featureBounds), leafCapacity),
    modelPoints,
    radius
  );
  // The overlap set is independent of leaf order.
  expect(sortedModel.candidates).toBe(unsortedModel.candidates);
  const capacityPlan = getGPUSpatialJoinCapacityPlan({
    leftCount: points.length,
    rightCount: featureBounds.length,
    candidateCapacity,
    pairCapacity: candidateCapacity,
    observedCandidateCount: sortedResult.candidateCount,
    observedRequiredCount: sortedResult.counts.reduce((sum, count) => sum + count, 0)
  });
  rows.push({
    scenario,
    features: featureBounds.length,
    points: points.length,
    candidates: sortedResult.candidateCount,
    visitedUnsorted: unsortedModel.visitedPerPoint,
    visitedSorted: sortedModel.visitedPerPoint,
    millisecondsUnsorted,
    millisecondsSorted,
    plannedWorkItems: capacityPlan.estimatedWorkItems,
    plannedPeakTransientBytes: capacityPlan.estimatedPeakTransientBytes
  });
}

function getSquareBounds(features: {minX: number; minY: number; size: number}[]): FeatureBounds[] {
  return features.map(({minX, minY, size}) => [minX, minY, minX + size, minY + size]);
}

function createGridSquares(columns: number): {minX: number; minY: number; size: number}[] {
  const squares = [];
  for (let row = 0; row < columns; row++) {
    for (let column = 0; column < columns; column++) {
      squares.push({minX: column, minY: row, size: 0.8});
    }
  }
  return squares;
}

function createRandomPoints(count: number, extent: number, seed: number): Point[] {
  const random = createRandom(seed);
  return Array.from({length: count}, () => [
    Math.fround(random() * extent),
    Math.fround(random() * extent)
  ]);
}

async function benchmarkPolygonJoin(
  device: Device,
  scenario: string,
  squares: {minX: number; minY: number; size: number}[],
  points: Point[]
): Promise<void> {
  const features: OraclePolygonFeature[] = squares.map(({minX, minY, size}) =>
    createSquareFeature(minX, minY, size)
  );
  await benchmarkScenario(
    scenario,
    getSquareBounds(squares),
    points,
    0,
    options => createPolygonJoinRun(device, features, points, options),
    points.length
  );
}

it('spatial join sort benchmark: point-in-polygon, shuffled and coherent features', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const columns = 120;
  const coherent = createGridSquares(columns);
  const points = createRandomPoints(250000, columns, 51);
  const {shuffled} = shuffleDeterministic(coherent, createRandom(52));
  await benchmarkPolygonJoin(device, 'A pip shuffled', shuffled, points);
  await benchmarkPolygonJoin(device, 'B pip coherent', coherent, points);
});

it('spatial join sort benchmark: nearest segments, shuffled', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(61);
  const extent = 1000;
  const segments: [Point, Point][] = Array.from({length: 50000}, () => {
    const start: Point = [Math.fround(random() * extent), Math.fround(random() * extent)];
    const angle = random() * Math.PI * 2;
    const length = random() * 4;
    return [
      start,
      [
        Math.fround(start[0] + Math.cos(angle) * length),
        Math.fround(start[1] + Math.sin(angle) * length)
      ]
    ];
  });
  const {shuffled} = shuffleDeterministic(segments, createRandom(62));
  const points = createRandomPoints(100000, extent, 63);
  const radius = 1;
  const bounds: FeatureBounds[] = shuffled.map(([start, end]) => [
    Math.min(start[0], end[0]),
    Math.min(start[1], end[1]),
    Math.max(start[0], end[0]),
    Math.max(start[1], end[1])
  ]);
  await benchmarkScenario(
    'C nearest shuffled',
    bounds,
    points,
    radius,
    options =>
      createNearestJoinRun(
        device,
        {starts: shuffled.map(([start]) => start), ends: shuffled.map(([, end]) => end)},
        points,
        radius,
        options
      ),
    500000
  );
});

it('spatial join sort benchmark: report', () => {
  if (rows.length === 0) {
    return;
  }
  const format = (value: number, digits: number) => value.toFixed(digits);
  const lines = rows.map(
    row =>
      `${row.scenario.padEnd(20)} ${String(row.features).padStart(6)} ${String(row.points).padStart(7)} ` +
      `${String(row.candidates).padStart(8)} ${format(row.visitedUnsorted, 1).padStart(9)} ` +
      `${format(row.visitedSorted, 1).padStart(9)} ${format(row.millisecondsUnsorted, 2).padStart(9)} ` +
      `${format(row.millisecondsSorted, 2).padStart(9)} ${String(row.plannedWorkItems).padStart(10)} ` +
      `${format(row.plannedPeakTransientBytes / 1048576, 2).padStart(9)}`
  );
  // eslint-disable-next-line no-console
  console.log(
    [
      'scenario              feats  points  candidat  visit/pt  visit/pt   ms/enc    ms/enc       work  peak MiB',
      '                                              unsorted    sorted unsorted    sorted',
      ...lines
    ].join('\n')
  );
});
