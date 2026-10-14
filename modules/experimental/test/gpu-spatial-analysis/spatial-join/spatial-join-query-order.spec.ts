// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {HALF_OPEN_SCENES} from './half-open-fixtures';
import {createRandom} from './spatial-join-oracle';
import {
  createNearestJoinRun,
  createPolygonJoinRun,
  createSquareFeature
} from './spatial-join-sort-utils';

type Point = [number, number];

/** Clustered, uniform, NaN, infinite and duplicate points in one deterministic, shuffled set. */
function createMixedPoints(count: number, extent: number, seed: number): Point[] {
  const random = createRandom(seed);
  const centers = Array.from({length: 12}, () => [random() * extent, random() * extent]);
  const points: Point[] = [];
  for (let index = 0; index < count; index++) {
    const kind = index % 10;
    if (kind < 6) {
      const [cx, cy] = centers[Math.floor(random() * centers.length)];
      points.push([Math.fround(cx + (random() - 0.5) * 2), Math.fround(cy + (random() - 0.5) * 2)]);
    } else if (kind < 8) {
      points.push([Math.fround(random() * extent), Math.fround(random() * extent)]);
    } else if (kind === 8) {
      points.push(points.length > 0 ? [...points[points.length - 1]] : [1, 1]);
    } else {
      const special: Point[] = [
        [Number.NaN, 1],
        [2, Number.NaN],
        [Number.POSITIVE_INFINITY, 3],
        [-1e30, 5],
        [extent * 3, extent * 3]
      ];
      points.push(special[Math.floor(random() * special.length)]);
    }
  }
  return points;
}

function createSegments(count: number, extent: number, seed: number): [Point, Point][] {
  const random = createRandom(seed);
  return Array.from({length: count}, () => {
    const start: Point = [Math.fround(random() * extent), Math.fround(random() * extent)];
    const angle = random() * Math.PI * 2;
    const length = random() * 2;
    return [
      start,
      [
        Math.fround(start[0] + Math.cos(angle) * length),
        Math.fround(start[1] + Math.sin(angle) * length)
      ]
    ];
  });
}

type NeighborRun = {
  encodeAndWait: () => Promise<void>;
  read: () => Promise<{ids: number[]; distanceBits: number[]; counts: number[]; overflow: number}>;
  destroy: () => void;
};

/** Compiles a k-nearest join (segment features, optional `maxDistance`) with `querySort`. */
function createNeighborRun(
  device: Device,
  segments: [Point, Point][],
  points: Point[],
  options: {k: number; maxDistance?: number; querySort: boolean}
): NeighborRun {
  const graph = new GPUCommandGraph(device, {id: 'query-order-neighbors'});
  const {k} = options;
  const pointCount = points.length;
  const featureCount = segments.length;
  const buffers: Buffer[] = [];
  const input = (name: string, values: Float32Array, length: number) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'float32x2', length);
  };
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    length: number,
    format: Format
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const ids = output('ids', pointCount * k, 'uint32');
  const distances = output('distances', pointCount * k, 'float32');
  const counts = output('counts', pointCount, 'uint32');
  const overflow = output('overflow', 1, 'uint32');
  const maxDistance =
    options.maxDistance === undefined
      ? undefined
      : new GPUParameterBuffer(device, {
          id: 'max-distance',
          format: 'float32',
          length: 1,
          values: Float32Array.of(options.maxDistance)
        });
  graph.add(
    new GPUNearestFeatureJoin({
      id: 'query-order-neighbors',
      queries: {
        kind: 'points',
        positions: input('points', Float32Array.from(points.flat()), pointCount)
      },
      features: {
        kind: 'segments',
        starts: input(
          'starts',
          Float32Array.from(segments.flatMap(([start]) => start)),
          featureCount
        ),
        ends: input('ends', Float32Array.from(segments.flatMap(([, end]) => end)), featureCount)
      },
      k,
      radius: maxDistance?.importToGraph(graph),
      spatialSort: true,
      querySort: options.querySort,
      output: {
        neighborIds: ids.view,
        neighborDistances: distances.view,
        neighborCounts: counts.view,
        status: {overflow: overflow.view}
      }
    })
  );
  const compiled = graph.compile();
  const encode = () => submitGraph(device, compiled, undefined);
  return {
    encodeAndWait: async () => {
      encode();
      await readUint32(overflow.buffer, 1);
    },
    read: async () => {
      encode();
      return {
        ids: [...(await readUint32(ids.buffer, pointCount * k))],
        // Distances are compared as bit patterns: the results must be bitwise identical.
        distanceBits: [...(await readUint32(distances.buffer, pointCount * k))],
        counts: [...(await readUint32(counts.buffer, pointCount))],
        overflow: (await readUint32(overflow.buffer, 1))[0]
      };
    },
    destroy: () => {
      compiled.destroy();
      maxDistance?.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}

for (const {name, k, maxDistance} of [
  {name: 'k = 1', k: 1, maxDistance: undefined},
  {name: 'k = 4', k: 4, maxDistance: undefined},
  {name: 'k = 1 with maxDistance', k: 1, maxDistance: 0.5},
  {name: 'k = 4 with maxDistance', k: 4, maxDistance: 1.5}
]) {
  it(`nearest neighbors are bitwise identical with querySort on and off (${name})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const points = createMixedPoints(6000, 40, 101);
    const segments = createSegments(1500, 40, 102);
    const off = createNeighborRun(device, segments, points, {k, maxDistance, querySort: false});
    const on = createNeighborRun(device, segments, points, {k, maxDistance, querySort: true});
    const expected = await off.read();
    const actual = await on.read();
    off.destroy();
    on.destroy();
    expect(actual.overflow).toBe(expected.overflow);
    expect(actual.counts).toEqual(expected.counts);
    expect(actual.ids).toEqual(expected.ids);
    expect(actual.distanceBits).toEqual(expected.distanceBits);
    // Sanity: the scene has real matches, and NaN / infinite points match nothing.
    expect(expected.counts.some(count => count > 0)).toBe(true);
  }, 120000);
}

it('nearest-feature radius mode is bitwise identical with querySort on and off', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = createMixedPoints(6000, 40, 111);
  const segments = createSegments(1500, 40, 112);
  const options = {candidateCapacity: 200000, spatialSort: true};
  const features = {starts: segments.map(([start]) => start), ends: segments.map(([, end]) => end)};
  const off = createNearestJoinRun(device, features, points, 1, {...options, querySort: false});
  const on = createNearestJoinRun(device, features, points, 1, {...options, querySort: true});
  const expected = await off.readResult();
  const actual = await on.readResult();
  off.destroy();
  on.destroy();
  expect(expected.overflow).toBe(0);
  expect(actual).toEqual(expected);
}, 120000);

it('point-in-polygon is identical with querySort on and off, including half-open boundaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 300 unit squares (one cell gap) with mixed points, plus the half-open scene's boundary points
  // appended so shared edges and vertices are probed in a shuffled order.
  const columns = 20;
  const features = Array.from({length: 300}, (_, index) =>
    createSquareFeature(index % columns, Math.floor(index / columns), 0.8)
  );
  const scene = HALF_OPEN_SCENES.squares;
  const squares = scene.features.length;
  const boundaryFeatures = [...scene.features, ...features.map(feature => shift(feature, 10, 10))];
  const points = [...createMixedPoints(6000, 20, 121), ...scene.points, ...scene.points];
  for (const boundaryRule of ['half-open', 'closed', 'open'] as const) {
    const options = {
      spatialSort: true,
      candidateCapacity: points.length * 4,
      boundaryRule
    };
    const off = createPolygonJoinRun(device, boundaryFeatures, points, {
      ...options,
      querySort: false
    });
    const on = createPolygonJoinRun(device, boundaryFeatures, points, {
      ...options,
      querySort: true
    });
    const expected = await off.readResult();
    const actual = await on.readResult();
    off.destroy();
    on.destroy();
    expect(expected.overflow).toBe(0);
    expect(actual).toEqual(expected);
    if (boundaryRule === 'half-open') {
      // The appended scene points keep their documented owners.
      const sceneStart = points.length - 2 * scene.points.length;
      for (const copy of [0, 1]) {
        scene.owners.forEach((owner, index) => {
          const id = actual.featureRows[sceneStart + copy * scene.points.length + index];
          expect(id === 0xffffffff ? null : id, `scene point ${index}`).toBe(owner);
        });
      }
    }
  }
  expect(squares).toBe(4);
}, 120000);

function shift(feature: number[][][][], dx: number, dy: number): number[][][][] {
  return feature.map(polygon => polygon.map(ring => ring.map(([x, y]) => [x + dx, y + dy])));
}

it('point-in-polygon candidate overflow drops the same pairs with querySort on and off', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features = Array.from({length: 300}, (_, index) =>
    createSquareFeature(index % 20, Math.floor(index / 20), 0.8)
  );
  const points = createMixedPoints(6000, 20, 131);
  // A capacity far below the candidate count: slots follow the original row, so the same
  // leading rows keep their candidates in both modes.
  const options = {spatialSort: true, candidateCapacity: 500};
  const off = createPolygonJoinRun(device, features, points, {...options, querySort: false});
  const on = createPolygonJoinRun(device, features, points, {...options, querySort: true});
  const expected = await off.readResult();
  const actual = await on.readResult();
  off.destroy();
  on.destroy();
  expect(expected.overflow).toBe(1);
  expect(actual).toEqual(expected);
}, 120000);

const median = (values: number[]) =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

/** A/B harness: logs paired GPU times of querySort off and on (1e6 clustered points, ~5e4 polygons). */
it('A/B: querySort off vs on, 1e6 clustered points against ~5e4 polygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const columns = 224;
  const squares = Array.from({length: columns * columns}, (_, index) =>
    createSquareFeature(index % columns, Math.floor(index / columns), 0.8)
  );
  const random = createRandom(141);
  const centers = Array.from({length: 200}, () => [random() * columns, random() * columns]);
  const points: Point[] = Array.from({length: 1_000_000}, () => {
    const [cx, cy] = centers[Math.floor(random() * centers.length)];
    return [
      Math.fround(Math.min(columns, Math.max(0, cx + (random() - 0.5) * 8))),
      Math.fround(Math.min(columns, Math.max(0, cy + (random() - 0.5) * 8)))
    ];
  });
  // Input order is a shuffle of cluster membership, i.e. incoherent, the case the sort targets.
  const lines: string[] = [];
  const runs = {
    off: createPolygonJoinRun(device, squares, points, {
      spatialSort: true,
      candidateCapacity: points.length,
      querySort: false
    }),
    on: createPolygonJoinRun(device, squares, points, {
      spatialSort: true,
      candidateCapacity: points.length,
      querySort: true
    })
  };
  const expected = await runs.off.readResult();
  const actual = await runs.on.readResult();
  expect(actual).toEqual(expected);
  for (let index = 0; index < 3; index++) {
    await runs.off.encodeAndWait();
    await runs.on.encodeAndWait();
  }
  const samples: Record<'off' | 'on', number[]> = {off: [], on: []};
  for (let repeat = 0; repeat < 5; repeat++) {
    for (const mode of (repeat % 2 === 0 ? ['off', 'on'] : ['on', 'off']) as ('off' | 'on')[]) {
      const timings: number[] = [];
      for (let index = 0; index < 10; index++) {
        const start = performance.now();
        await runs[mode].encodeAndWait();
        timings.push(performance.now() - start);
      }
      samples[mode].push(median(timings));
    }
  }
  runs.off.destroy();
  runs.on.destroy();
  const off = median(samples.off);
  const on = median(samples.on);
  lines.push(
    `pip 1e6 clustered x ${squares.length} polygons: off ${off.toFixed(2)} ms  on ${on.toFixed(2)} ms  on/off ${(on / off).toFixed(3)}`
  );

  const segments = createSegments(50000, columns, 142);
  for (const k of [1, 4]) {
    const neighborRuns = {
      off: createNeighborRun(device, segments, points, {k, maxDistance: 2, querySort: false}),
      on: createNeighborRun(device, segments, points, {k, maxDistance: 2, querySort: true})
    };
    const neighborExpected = await neighborRuns.off.read();
    const neighborActual = await neighborRuns.on.read();
    expect(neighborActual).toEqual(neighborExpected);
    const neighborSamples: Record<'off' | 'on', number[]> = {off: [], on: []};
    for (let repeat = 0; repeat < 5; repeat++) {
      for (const mode of (repeat % 2 === 0 ? ['off', 'on'] : ['on', 'off']) as ('off' | 'on')[]) {
        await neighborRuns[mode].encodeAndWait();
        const timings: number[] = [];
        for (let index = 0; index < 5; index++) {
          const start = performance.now();
          await neighborRuns[mode].encodeAndWait();
          timings.push(performance.now() - start);
        }
        neighborSamples[mode].push(median(timings));
      }
    }
    neighborRuns.off.destroy();
    neighborRuns.on.destroy();
    const neighborOff = median(neighborSamples.off);
    const neighborOn = median(neighborSamples.on);
    lines.push(
      `nearest k=${k} 1e6 clustered x 5e4 segments: off ${neighborOff.toFixed(2)} ms  on ${neighborOn.toFixed(2)} ms  on/off ${(neighborOn / neighborOff).toFixed(3)}`
    );
  }
  // eslint-disable-next-line no-console
  console.log(`QUERY-SORT-AB\n${lines.join('\n')}`);
}, 900000);
