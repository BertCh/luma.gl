// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from './spatial-join-oracle';
import {
  createNearestJoinRun,
  createPolygonJoinRun,
  createSquareFeature,
  shuffleDeterministic,
  type SpatialJoinRun,
  type SpatialJoinRunOptions
} from './spatial-join-sort-utils';

type Point = [number, number];
type Curve = 'morton' | 'hilbert';

const REPEATS = 5;
const WARM_UP_ENCODINGS = 3;
const TIMED_ENCODINGS = 10;

function getMedian(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

async function measureOnce(run: SpatialJoinRun): Promise<number> {
  const timings: number[] = [];
  for (let index = 0; index < TIMED_ENCODINGS; index++) {
    const start = performance.now();
    await run.encodeAndWait();
    timings.push(performance.now() - start);
  }
  return getMedian(timings);
}

/** Compiles a buffer selection over segments, exposing mask and nearest ids as the result. */
function createBufferSelectionRun(
  device: Device,
  segments: [Point, Point][],
  points: Point[],
  radius: number,
  options: SpatialJoinRunOptions
): SpatialJoinRun {
  const graph = new GPUCommandGraph(device, {id: 'buffer-selection-ab'});
  const pointCount = points.length;
  const featureCount = segments.length;
  const buffers: Buffer[] = [];
  const input = (name: string, values: Float32Array, length: number) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'float32x2', length);
  };
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const distance = new GPUParameterBuffer(device, {
    id: 'distance',
    format: 'float32',
    length: 1,
    values: Float32Array.of(radius)
  });
  const mask = output('mask', pointCount);
  const nearest = output('nearest', pointCount);
  const overflow = output('overflow', 1);
  graph.add(
    new GPUBufferSelection({
      points: input('points', Float32Array.from(points.flat()), pointCount),
      features: {
        kind: 'segments',
        starts: input(
          'starts',
          Float32Array.from(segments.flatMap(([start]) => start)),
          featureCount
        ),
        ends: input('ends', Float32Array.from(segments.flatMap(([, end]) => end)), featureCount)
      },
      distance: distance.importToGraph(graph),
      candidateCapacity: options.candidateCapacity,
      spatialSort: options.spatialSort,
      spatialSortCurve: options.spatialSortCurve,
      outputMask: mask.view,
      nearestFeatureIds: nearest.view,
      overflow: overflow.view
    })
  );
  const compiled = graph.compile();
  const encode = () => submitGraph(device, compiled, undefined);
  return {
    encode,
    encodeAndWait: async () => {
      encode();
      await readUint32(overflow.buffer, 1);
    },
    readResult: async () => {
      encode();
      return {
        featureRows: [...(await readUint32(nearest.buffer, pointCount))],
        counts: [...(await readUint32(mask.buffer, pointCount))],
        candidateCount: 0,
        overflow: (await readUint32(overflow.buffer, 1))[0]
      };
    },
    destroy: () => {
      compiled.destroy();
      distance.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}

const lines: string[] = [];

/** Paired, interleaved A/B: morton and hilbert runs alternate each repeat. */
async function compareCurves(
  scenario: string,
  createRun: (options: SpatialJoinRunOptions) => SpatialJoinRun,
  candidateCapacity: number
): Promise<void> {
  const runs: Record<Curve, SpatialJoinRun> = {
    morton: createRun({spatialSort: true, spatialSortCurve: 'morton', candidateCapacity}),
    hilbert: createRun({spatialSort: true, spatialSortCurve: 'hilbert', candidateCapacity})
  };
  const mortonResult = await runs.morton.readResult();
  const hilbertResult = await runs.hilbert.readResult();
  expect(mortonResult.overflow).toBe(0);
  expect(hilbertResult).toEqual(mortonResult);
  for (let index = 0; index < WARM_UP_ENCODINGS; index++) {
    await runs.morton.encodeAndWait();
    await runs.hilbert.encodeAndWait();
  }
  const samples: Record<Curve, number[]> = {morton: [], hilbert: []};
  for (let repeat = 0; repeat < REPEATS; repeat++) {
    const order: Curve[] = repeat % 2 === 0 ? ['morton', 'hilbert'] : ['hilbert', 'morton'];
    for (const curve of order) {
      samples[curve].push(await measureOnce(runs[curve]));
    }
  }
  runs.morton.destroy();
  runs.hilbert.destroy();
  const morton = getMedian(samples.morton);
  const hilbert = getMedian(samples.hilbert);
  lines.push(
    `${scenario.padEnd(28)} morton ${morton.toFixed(2)} ms  hilbert ${hilbert.toFixed(2)} ms  ` +
      `hilbert/morton ${(hilbert / morton).toFixed(3)}  ` +
      `[m ${samples.morton.map(value => value.toFixed(2)).join(' ')}] ` +
      `[h ${samples.hilbert.map(value => value.toFixed(2)).join(' ')}]`
  );
}

function createRandomPoints(count: number, extent: number, seed: number): Point[] {
  const random = createRandom(seed);
  return Array.from({length: count}, () => [
    Math.fround(random() * extent),
    Math.fround(random() * extent)
  ]);
}

function createSegments(count: number, extent: number, seed: number): [Point, Point][] {
  const random = createRandom(seed);
  return Array.from({length: count}, () => {
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
}

/** Coherent order: row-major in 50-unit-tall bands. */
function sortRowMajor(segments: [Point, Point][]): [Point, Point][] {
  return [...segments].sort(
    (left, right) =>
      Math.floor(left[0][1] / 20) - Math.floor(right[0][1] / 20) || left[0][0] - right[0][0]
  );
}

/** Identical-result check plus a paired timing table (Morton against Hilbert, `console.log`). */
it('spatial join curve A/B: Morton vs Hilbert', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const columns = 120;
  const squares = [];
  for (let row = 0; row < columns; row++) {
    for (let column = 0; column < columns; column++) {
      squares.push({minX: column, minY: row, size: 0.8});
    }
  }
  const pipPoints = createRandomPoints(250000, columns, 51);
  const shuffledSquares = shuffleDeterministic(squares, createRandom(52)).shuffled;
  for (const [name, set] of [
    ['pip shuffled', shuffledSquares],
    ['pip coherent', squares]
  ] as const) {
    const features = set.map(({minX, minY, size}) => createSquareFeature(minX, minY, size));
    await compareCurves(
      name,
      options => createPolygonJoinRun(device, features, pipPoints, options),
      pipPoints.length
    );
  }

  const segments = createSegments(50000, 1000, 61);
  const shuffled = shuffleDeterministic(segments, createRandom(62)).shuffled;
  const coherent = sortRowMajor(segments);
  const points = createRandomPoints(100000, 1000, 63);
  for (const [name, set] of [
    ['nearest shuffled', shuffled],
    ['nearest coherent', coherent]
  ] as const) {
    await compareCurves(
      name,
      options =>
        createNearestJoinRun(
          device,
          {starts: set.map(([start]) => start), ends: set.map(([, end]) => end)},
          points,
          1,
          options
        ),
      500000
    );
  }
  for (const [name, set] of [
    ['buffer selection shuffled', shuffled],
    ['buffer selection coherent', coherent]
  ] as const) {
    await compareCurves(
      name,
      options => createBufferSelectionRun(device, set, points, 1, options),
      500000
    );
  }
  // eslint-disable-next-line no-console
  console.log(`CURVE-AB\n${lines.join('\n')}`);
}, 600000);
