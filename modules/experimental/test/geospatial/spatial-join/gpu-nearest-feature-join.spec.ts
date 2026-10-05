// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE as N,
  GPUNearestFeatureJoin,
  type GPUNearestFeatureJoinProps
} from '../../../src/geospatial/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readCompactIds,
  readFloat32,
  readUint32,
  sortNumbers
} from '../../utils/gpu-contributor-test-utils';
import {createRandom, joinNearestSegments} from './spatial-join-oracle';

type Point = [number, number];

const SEGMENTS: [Point, Point][] = [
  [
    [0, 0],
    [10, 0]
  ],
  [
    [0, 5],
    [10, 5]
  ],
  [
    [20, 0],
    [20, 0]
  ]
];
const QUERY_POINTS: Point[] = [
  [2, 1],
  [3, 4],
  [5, 2.5],
  [19, 0],
  [15, 0],
  [-2, 0]
];

type NearestFixture = {
  graph: GPUCommandGraph;
  buffers: Buffer[];
  radius: GPUParameterBuffer<'float32'>;
  ids: Buffer;
  distances: Buffer;
  counts: Buffer;
  overflow: Buffer;
  candidateCount: Buffer;
  matchIds: Buffer;
  matchCount: Buffer;
  props: GPUNearestFeatureJoinProps;
};

function createNearestFixture(
  device: Device,
  points: Point[],
  features: {kind: 'segments'; segments: [Point, Point][]} | {kind: 'points'; positions: Point[]},
  radiusValue: number,
  overrides: Partial<GPUNearestFeatureJoinProps> = {}
): NearestFixture {
  const graph = new GPUCommandGraph(device, {id: 'nearest-join'});
  const pointCount = points.length;
  const featureCount =
    features.kind === 'segments' ? features.segments.length : features.positions.length;
  const pointsBuffer = createInputBuffer(device, Float32Array.from(points.flat()));
  const sourceIdsBuffer = createInputBuffer(
    device,
    Uint32Array.from(points.map((_, index) => 500 + index))
  );
  const featureBuffers =
    features.kind === 'segments'
      ? [
          createInputBuffer(
            device,
            Float32Array.from(features.segments.flatMap(([start]) => start))
          ),
          createInputBuffer(device, Float32Array.from(features.segments.flatMap(([, end]) => end)))
        ]
      : [createInputBuffer(device, Float32Array.from(features.positions.flat()))];
  const ids = createOutputBuffer(device, pointCount);
  const distances = createOutputBuffer(device, pointCount);
  const counts = createOutputBuffer(device, featureCount);
  const overflow = createOutputBuffer(device, 1);
  const candidateCount = createOutputBuffer(device, 1);
  const matchIds = createOutputBuffer(device, pointCount);
  const matchCount = createOutputBuffer(device, 1);
  const matchOverflow = createOutputBuffer(device, 1);
  const radius = new GPUParameterBuffer(device, {
    id: 'radius',
    format: 'float32',
    length: 1,
    values: Float32Array.of(radiusValue)
  });
  const props: GPUNearestFeatureJoinProps = {
    points: importGraphBuffer(graph, 'points', pointsBuffer, 'float32x2', pointCount),
    sourceIds: importGraphBuffer(graph, 'source-ids', sourceIdsBuffer, 'uint32', pointCount),
    features:
      features.kind === 'segments'
        ? {
            kind: 'segments',
            starts: importGraphBuffer(
              graph,
              'starts',
              featureBuffers[0],
              'float32x2',
              featureCount
            ),
            ends: importGraphBuffer(graph, 'ends', featureBuffers[1], 'float32x2', featureCount)
          }
        : {
            kind: 'points',
            positions: importGraphBuffer(
              graph,
              'positions',
              featureBuffers[0],
              'float32x2',
              featureCount
            )
          },
    radius: radius.importToGraph(graph),
    candidateCapacity: 64,
    nearestFeatureIds: importGraphBuffer(graph, 'ids', ids, 'uint32', pointCount),
    nearestDistances: importGraphBuffer(graph, 'distances', distances, 'float32', pointCount),
    featureCounts: importGraphBuffer(graph, 'counts', counts, 'uint32', featureCount),
    overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
    candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1),
    matches: {
      ids: importGraphBuffer(graph, 'match-ids', matchIds, 'uint32', pointCount),
      count: importGraphBuffer(graph, 'match-count', matchCount, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'match-overflow', matchOverflow, 'uint32', 1)
    },
    ...overrides
  };
  return {
    graph,
    buffers: [
      pointsBuffer,
      sourceIdsBuffer,
      ...featureBuffers,
      ids,
      distances,
      counts,
      overflow,
      candidateCount,
      matchIds,
      matchCount,
      matchOverflow
    ],
    radius,
    ids,
    distances,
    counts,
    overflow,
    candidateCount,
    matchIds,
    matchCount,
    props
  };
}

function destroyFixture(fixture: NearestFixture): void {
  fixture.radius.destroy();
  for (const buffer of fixture.buffers) buffer.destroy();
}

function expectClose(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    expect(actual[index]).toBeCloseTo(value, 5);
  }
}

it('GPUNearestFeatureJoin snaps points to segments and follows a per-frame radius', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const exactTies = !isSoftwareDevice(device);
  const fixture = createNearestFixture(
    device,
    QUERY_POINTS,
    {kind: 'segments', segments: SEGMENTS},
    3
  );
  fixture.graph.add(new GPUNearestFeatureJoin(fixture.props));
  const compiled = fixture.graph.compile();

  submitGraph(device, compiled, undefined);
  const ids = await readUint32(fixture.ids, 6);
  expect([ids[0], ids[1], ids[3], ids[4], ids[5]]).toEqual([0, 1, 2, N, 0]);
  if (exactTies) {
    expect(ids[2]).toBe(0);
    expect(await readUint32(fixture.counts, 3)).toEqual([3, 1, 1]);
  }
  expectClose(await readFloat32(fixture.distances, 6), [1, 1, 2.5, 1, -1, 2]);
  expect(await readUint32(fixture.overflow, 1)).toEqual([0]);
  expect((await readCompactIds(fixture.matchIds, fixture.matchCount)).sort(sortNumbers)).toEqual([
    500, 501, 502, 503, 505
  ]);

  fixture.radius.write(Float32Array.of(6));
  submitGraph(device, compiled, undefined);
  if (exactTies) {
    expect((await readUint32(fixture.ids, 6))[4]).toBe(0);
    expect(await readUint32(fixture.counts, 3)).toEqual([4, 1, 1]);
  }
  expect((await readFloat32(fixture.distances, 6))[4]).toBeCloseTo(5, 5);

  fixture.radius.write(Float32Array.of(Number.NaN));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.ids, 6)).toEqual(new Array(6).fill(N));
  expect(await readFloat32(fixture.distances, 6)).toEqual(new Array(6).fill(-1));
  expect(await readUint32(fixture.candidateCount, 1)).toEqual([0]);

  fixture.radius.write(Float32Array.of(-1));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.ids, 6)).toEqual(new Array(6).fill(N));

  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNearestFeatureJoin writes feature IDs for point features', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const featureIdsBuffer = createInputBuffer(device, Uint32Array.from([7, 8, 9]));
  const fixture = createNearestFixture(
    device,
    [
      [1, 0],
      [9, 1],
      [5, 4],
      [50, 50]
    ],
    {
      kind: 'points',
      positions: [
        [0, 0],
        [10, 0],
        [5, 5]
      ]
    },
    2
  );
  fixture.graph.add(
    new GPUNearestFeatureJoin({
      ...fixture.props,
      featureIds: importGraphBuffer(fixture.graph, 'feature-ids', featureIdsBuffer, 'uint32', 3)
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.ids, 4)).toEqual([7, 8, 9, N]);
  expectClose(await readFloat32(fixture.distances, 4), [1, Math.SQRT2, 1, -1]);
  compiled.destroy();
  destroyFixture(fixture);
  featureIdsBuffer.destroy();
});

it('GPUNearestFeatureJoin reports candidate overflow and matches a random oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const overflowFixture = createNearestFixture(
    device,
    QUERY_POINTS,
    {kind: 'segments', segments: SEGMENTS},
    6,
    {candidateCapacity: 2}
  );
  overflowFixture.graph.add(new GPUNearestFeatureJoin(overflowFixture.props));
  let compiled = overflowFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(overflowFixture.overflow, 1)).toEqual([1]);
  const distances = await readFloat32(overflowFixture.distances, 6);
  expect(distances.every(distance => distance <= 6)).toBe(true);
  compiled.destroy();
  destroyFixture(overflowFixture);

  const random = createRandom(11);
  const points: Point[] = Array.from({length: 300}, () => [random() * 50, random() * 50]);
  const segments: [Point, Point][] = Array.from({length: 60}, () => {
    const start: Point = [random() * 50, random() * 50];
    const angle = random() * Math.PI * 2;
    const length = random() * 5;
    return [start, [start[0] + Math.cos(angle) * length, start[1] + Math.sin(angle) * length]];
  });
  const fround = (point: Point): Point => [Math.fround(point[0]), Math.fround(point[1])];
  const roundedPoints = points.map(fround);
  const roundedSegments = segments.map(
    ([start, end]) => [fround(start), fround(end)] as [Point, Point]
  );
  const randomFixture = createNearestFixture(
    device,
    roundedPoints,
    {kind: 'segments', segments: roundedSegments},
    2,
    {candidateCapacity: 4096}
  );
  randomFixture.graph.add(new GPUNearestFeatureJoin(randomFixture.props));
  compiled = randomFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const oracle = joinNearestSegments(roundedPoints, roundedSegments, 2);
  const gpuIds = await readUint32(randomFixture.ids, 300);
  const gpuDistances = await readFloat32(randomFixture.distances, 300);
  for (let index = 0; index < 300; index++) {
    const ambiguous =
      oracle.secondDistances[index] - Math.max(oracle.distances[index], 0) < 1e-3 ||
      Math.abs(oracle.distances[index] - 2) < 1e-3;
    if (ambiguous) {
      continue;
    }
    expect(gpuIds[index]).toBe(oracle.featureRows[index]);
    expect(Math.abs(gpuDistances[index] - oracle.distances[index])).toBeLessThan(1e-3);
  }
  expect(await readUint32(randomFixture.overflow, 1)).toEqual([0]);
  compiled.destroy();
  destroyFixture(randomFixture);
});
