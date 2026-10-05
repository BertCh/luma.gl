// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {GPUBufferSelection} from '../../../src/geospatial/spatial-join/gpu-buffer-selection';
import {GPU_SPATIAL_JOIN_NO_FEATURE} from '../../../src/geospatial/spatial-join/spatial-join-types';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from './spatial-join-oracle';

type Point = [number, number];
type Segment = [Point, Point];

/** Boundary guard: randomized points never lie this close to the buffer boundary (f32 rounding). */
const BOUNDARY_GUARD = 1e-3;

function getPointSegmentDistance(point: Point, [start, end]: Segment): number {
  const [px, py] = point;
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  let t = lengthSquared === 0 ? 0 : ((px - start[0]) * dx + (py - start[1]) * dy) / lengthSquared;
  t = Math.min(1, Math.max(0, t));
  return Math.hypot(px - (start[0] + t * dx), py - (start[1] + t * dy));
}

/** Brute-force f64 oracle: nearest distance per point; selection iff distance <= radius. */
function getOracleDistances(points: Point[], segments: Segment[]): number[] {
  return points.map(point =>
    Math.min(...segments.map(segment => getPointSegmentDistance(point, segment)))
  );
}

function getOracleSelection(distances: number[], radius: number): boolean[] {
  return distances.map(distance => Number.isFinite(radius) && radius >= 0 && distance <= radius);
}

function createRandomScene(
  seed: number,
  pointCount: number,
  radiusValues: number[]
): {points: Point[]; segments: Segment[]} {
  const random = createRandom(seed);
  const segments: Segment[] = [];
  for (let index = 0; index < 40; index++) {
    const start: Point = [random() * 100, random() * 100];
    // A polyline walks consecutive segments; every fifth feature is a degenerate point.
    const end: Point =
      index % 5 === 4
        ? start
        : [start[0] + (random() - 0.5) * 20, start[1] + (random() - 0.5) * 20];
    segments.push([start, end]);
  }
  const points: Point[] = [];
  while (points.length < pointCount) {
    const point: Point = [random() * 100, random() * 100];
    const distance = Math.min(...segments.map(segment => getPointSegmentDistance(point, segment)));
    if (radiusValues.every(radius => Math.abs(distance - radius) > BOUNDARY_GUARD)) {
      points.push(point);
    }
  }
  return {points, segments};
}

type Fixture = {
  device: Device;
  graph: GPUCommandGraph;
  buffers: Buffer[];
  distance: GPUParameterBuffer<'float32'>;
  importInput: <Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ) => GraphDataView<Format>;
  importOutput: <Format extends 'uint32' | 'float32' = 'uint32'>(
    name: string,
    length: number,
    format?: Format
  ) => {buffer: Buffer; view: GraphDataView<Format>};
  destroy: () => void;
};

function createFixture(device: Device, distanceValue: number): Fixture {
  const graph = new GPUCommandGraph(device, {id: 'buffer-selection'});
  const buffers: Buffer[] = [];
  const distance = new GPUParameterBuffer(device, {
    id: 'distance',
    format: 'float32',
    length: 1,
    values: Float32Array.of(distanceValue)
  });
  return {
    device,
    graph,
    buffers,
    distance,
    importInput: (name, values, format, length) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, length);
    },
    importOutput: (name, length, format) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(graph, name, buffer, format ?? 'uint32', length)
      } as never;
    },
    destroy: () => {
      distance.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

function getSegmentFeatures(fixture: Fixture, segments: Segment[]) {
  return {
    kind: 'segments' as const,
    starts: fixture.importInput(
      'starts',
      Float32Array.from(segments.flatMap(([start]) => start)),
      'float32x2',
      segments.length
    ),
    ends: fixture.importInput(
      'ends',
      Float32Array.from(segments.flatMap(([, end]) => end)),
      'float32x2',
      segments.length
    )
  };
}

function getIds(selected: boolean[], offset: number = 0): number[] {
  return selected.flatMap((isSelected, index) => (isSelected ? [index + offset] : []));
}

it('GPUBufferSelection selects points near polylines and follows a per-frame distance', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const radii = [3, 8, 0, 25];
  const {points, segments} = createRandomScene(11, 5000, radii);
  const fixture = createFixture(device, 3);
  const pointCount = points.length;
  const outputMask = fixture.importOutput('mask', pointCount);
  const outputIds = fixture.importOutput('ids', pointCount);
  const outputCount = fixture.importOutput('count', 1);
  const outputOverflow = fixture.importOutput('overflow', 1);
  const outputTotal = fixture.importOutput('total', 1);
  const distances = fixture.importOutput('distances', pointCount, 'float32');
  const nearest = fixture.importOutput('nearest', pointCount);
  const joinOverflow = fixture.importOutput('join-overflow', 1);
  fixture.graph.add(
    new GPUBufferSelection({
      points: fixture.importInput(
        'points',
        Float32Array.from(points.flat()),
        'float32x2',
        pointCount
      ),
      sourceIds: fixture.importInput(
        'source-ids',
        Uint32Array.from(points.map((_, index) => 1000 + index)),
        'uint32',
        pointCount
      ),
      features: getSegmentFeatures(fixture, segments),
      distance: fixture.distance.importToGraph(fixture.graph),
      candidateCapacity: 200000,
      outputMask: outputMask.view,
      output: {
        ids: outputIds.view,
        count: outputCount.view,
        overflow: outputOverflow.view,
        totalCount: outputTotal.view
      },
      distances: distances.view,
      nearestFeatureIds: nearest.view,
      overflow: joinOverflow.view
    })
  );
  const compiled = fixture.graph.compile();
  const oracleDistances = getOracleDistances(points, segments);

  const check = async (radius: number, expectEmpty: boolean = false) => {
    fixture.distance.write(Float32Array.of(radius));
    submitGraph(device, compiled, undefined);
    const selected = getOracleSelection(oracleDistances, radius);
    const expectedIds = getIds(selected, 1000);
    if (!expectEmpty && Number.isFinite(radius) && radius > 0) {
      expect(expectedIds.length).toBeGreaterThan(0);
    }
    expect(await readUint32(outputMask.buffer, pointCount)).toEqual(
      selected.map(isSelected => (isSelected ? 1 : 0))
    );
    const [count] = await readUint32(outputCount.buffer, 1);
    expect(count).toBe(expectedIds.length);
    expect(await readUint32(outputIds.buffer, count)).toEqual(expectedIds);
    expect(await readUint32(outputTotal.buffer, 1)).toEqual([expectedIds.length]);
    expect(await readUint32(outputOverflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(joinOverflow.buffer, 1)).toEqual([0]);
    const nearestIds = await readUint32(nearest.buffer, pointCount);
    expect(nearestIds.map(id => id !== GPU_SPATIAL_JOIN_NO_FEATURE)).toEqual(selected);
    const reported = await readFloat32(distances.buffer, pointCount);
    for (const [index, isSelected] of selected.entries()) {
      if (isSelected) {
        expect(reported[index]).toBeCloseTo(oracleDistances[index], 3);
      } else {
        expect(reported[index]).toBe(-1);
      }
    }
  };
  await check(3);
  await check(8);
  await check(0);
  await check(25);
  await check(Number.NaN, true);
  await check(-2, true);
  await check(Number.POSITIVE_INFINITY, true);
  await check(3);

  compiled.destroy();
  fixture.destroy();
});

it('GPUBufferSelection selects points near point features without source IDs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {points, segments} = createRandomScene(23, 3000, [4]);
  const featurePositions: Point[] = segments.map(([start]) => start);
  const pointSegments: Segment[] = featurePositions.map(position => [position, position]);
  const fixture = createFixture(device, 4);
  const pointCount = points.length;
  const outputIds = fixture.importOutput('ids', pointCount);
  const outputCount = fixture.importOutput('count', 1);
  const outputOverflow = fixture.importOutput('overflow', 1);
  fixture.graph.add(
    new GPUBufferSelection({
      points: fixture.importInput(
        'points',
        Float32Array.from(points.flat()),
        'float32x2',
        pointCount
      ),
      features: {
        kind: 'points',
        positions: fixture.importInput(
          'positions',
          Float32Array.from(featurePositions.flat()),
          'float32x2',
          featurePositions.length
        )
      },
      distance: fixture.distance.importToGraph(fixture.graph),
      candidateCapacity: 100000,
      output: {ids: outputIds.view, count: outputCount.view, overflow: outputOverflow.view}
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const expectedIds = getIds(getOracleSelection(getOracleDistances(points, pointSegments), 4));
  expect(expectedIds.length).toBeGreaterThan(0);
  const [count] = await readUint32(outputCount.buffer, 1);
  expect(count).toBe(expectedIds.length);
  expect(await readUint32(outputIds.buffer, count)).toEqual(expectedIds);
  compiled.destroy();
  fixture.destroy();
});

it('GPUBufferSelection treats distance <= radius as inclusive on exact boundaries', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const segments: Segment[] = [
    [
      [0, 0],
      [10, 0]
    ]
  ];
  // Distances 3 (perpendicular), 3 (to the end point), 3 (to the start point), then just outside.
  const points: Point[] = [
    [5, 3],
    [13, 0],
    [-3, 0],
    [5, -3],
    [5, 3.5],
    [14, 0]
  ];
  const fixture = createFixture(device, 3);
  const outputMask = fixture.importOutput('mask', points.length);
  fixture.graph.add(
    new GPUBufferSelection({
      points: fixture.importInput(
        'points',
        Float32Array.from(points.flat()),
        'float32x2',
        points.length
      ),
      features: getSegmentFeatures(fixture, segments),
      distance: fixture.distance.importToGraph(fixture.graph),
      candidateCapacity: 64,
      outputMask: outputMask.view
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(outputMask.buffer, points.length)).toEqual([1, 1, 1, 1, 0, 0]);
  compiled.destroy();
  fixture.destroy();
});

it('GPUBufferSelection preserves chunked points for the mask and source IDs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {points, segments} = createRandomScene(37, 3000, [6]);
  const fixture = createFixture(device, 6);
  const chunkLengths = [1200, 0, 1800];
  const chunkStarts = [0, 1200, 1200];
  const importChunks = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    values: (start: number, length: number) => Float32Array | Uint32Array
  ) =>
    createVectorView(
      name,
      format,
      chunkLengths.map((length, chunkIndex) =>
        fixture.importInput(
          `${name}-${chunkIndex}`,
          values(chunkStarts[chunkIndex], length),
          format,
          length
        )
      )
    );
  const maskChunks = chunkLengths.map((length, chunkIndex) =>
    fixture.importOutput(`mask-${chunkIndex}`, length, 'uint32')
  );
  const outputIds = fixture.importOutput('ids', points.length);
  const outputCount = fixture.importOutput('count', 1);
  const outputOverflow = fixture.importOutput('overflow', 1);
  fixture.graph.add(
    new GPUBufferSelection({
      points: importChunks('points', 'float32x2', (start, length) =>
        Float32Array.from(points.slice(start, start + length).flat())
      ),
      sourceIds: importChunks('source-ids', 'uint32', (start, length) =>
        Uint32Array.from({length}, (_, index) => 7000 + start + index)
      ),
      features: getSegmentFeatures(fixture, segments),
      distance: fixture.distance.importToGraph(fixture.graph),
      candidateCapacity: 100000,
      outputMask: createVectorView(
        'mask',
        'uint32',
        maskChunks.map(chunk => chunk.view)
      ),
      output: {ids: outputIds.view, count: outputCount.view, overflow: outputOverflow.view}
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const selected = getOracleSelection(getOracleDistances(points, segments), 6);
  const expectedMask = selected.map(isSelected => (isSelected ? 1 : 0));
  expect(await readUint32(maskChunks[0].buffer, 1200)).toEqual(expectedMask.slice(0, 1200));
  expect(await readUint32(maskChunks[2].buffer, 1800)).toEqual(expectedMask.slice(1200));
  const expectedIds = getIds(selected, 7000);
  const [count] = await readUint32(outputCount.buffer, 1);
  expect(count).toBe(expectedIds.length);
  expect(await readUint32(outputIds.buffer, count)).toEqual(expectedIds);
  compiled.destroy();
  fixture.destroy();
});

it('GPUBufferSelection clamps output and reports overflow when capacity is too small', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {points, segments} = createRandomScene(41, 2000, [10]);
  const fixture = createFixture(device, 10);
  const capacity = 5;
  const outputIds = fixture.importOutput('ids', capacity);
  const outputCount = fixture.importOutput('count', 1);
  const outputOverflow = fixture.importOutput('overflow', 1);
  const outputTotal = fixture.importOutput('total', 1);
  fixture.graph.add(
    new GPUBufferSelection({
      points: fixture.importInput(
        'points',
        Float32Array.from(points.flat()),
        'float32x2',
        points.length
      ),
      features: getSegmentFeatures(fixture, segments),
      distance: fixture.distance.importToGraph(fixture.graph),
      candidateCapacity: 100000,
      output: {
        ids: outputIds.view,
        count: outputCount.view,
        overflow: outputOverflow.view,
        totalCount: outputTotal.view
      }
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const expectedIds = getIds(getOracleSelection(getOracleDistances(points, segments), 10));
  expect(expectedIds.length).toBeGreaterThan(capacity);
  expect(await readUint32(outputCount.buffer, 1)).toEqual([capacity]);
  expect(await readUint32(outputIds.buffer, capacity)).toEqual(expectedIds.slice(0, capacity));
  expect(await readUint32(outputOverflow.buffer, 1)).toEqual([1]);
  expect(await readUint32(outputTotal.buffer, 1)).toEqual([expectedIds.length]);
  compiled.destroy();
  fixture.destroy();
});

it('GPUBufferSelection reports join overflow when candidateCapacity is too small', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {points, segments} = createRandomScene(53, 2000, [10]);
  const fixture = createFixture(device, 10);
  const outputIds = fixture.importOutput('ids', points.length);
  const outputCount = fixture.importOutput('count', 1);
  const outputOverflow = fixture.importOutput('overflow', 1);
  const joinOverflow = fixture.importOutput('join-overflow', 1);
  fixture.graph.add(
    new GPUBufferSelection({
      points: fixture.importInput(
        'points',
        Float32Array.from(points.flat()),
        'float32x2',
        points.length
      ),
      features: getSegmentFeatures(fixture, segments),
      distance: fixture.distance.importToGraph(fixture.graph),
      candidateCapacity: 2,
      output: {ids: outputIds.view, count: outputCount.view, overflow: outputOverflow.view},
      overflow: joinOverflow.view
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(joinOverflow.buffer, 1)).toEqual([1]);
  expect(await readUint32(outputOverflow.buffer, 1)).toEqual([1]);
  const [count] = await readUint32(outputCount.buffer, 1);
  expect(count).toBeLessThanOrEqual(points.length);
  compiled.destroy();
  fixture.destroy();
});
