// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {describe, expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPU_SPATIAL_JOIN_NO_FEATURE as NO_FEATURE} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-types';
import {
  GPU_NEAREST_NO_SEGMENT,
  type GPUNearestFeatureGeometry,
  type GPUNearestTieMode
} from '../../../src/gpu-spatial-analysis/spatial-join/nearest-types';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-prepared';
import {createRandom} from './spatial-join-oracle';
import {
  buildNearestArrays,
  nearestNeighborsOracle,
  type NearestOracleGeometry,
  type NearestPoint,
  type OracleNeighbor
} from './nearest-feature-oracle';

type NeighborCase = {
  queries: NearestOracleGeometry;
  features: NearestOracleGeometry;
  k: number;
  ties?: GPUNearestTieMode;
  capacity?: number;
  maxDistance?: number;
  featureIds?: number[];
  spatialSort?: boolean;
  /** Splits point queries into chunks of these sizes. */
  queryChunks?: number[];
  leafCapacity?: number;
  /** Index the features once through a prepared handle. */
  prepared?: boolean;
};

type NeighborResult = {
  ids: number[][];
  distances: number[][];
  foot: NearestPoint[][];
  segments: number[][];
  counts: number[];
  overflow: number;
};

type NeighborRun = {
  prepared?: GPUSpatialJoinPrepared;
  run: () => Promise<NeighborResult>;
  setMaxDistance: (value: number) => void;
  destroy: () => void;
};

function importGeometry(
  device: Device,
  graph: GPUCommandGraph,
  geometry: NearestOracleGeometry,
  name: string,
  buffers: Buffer[]
): GPUNearestFeatureGeometry & {kind: 'points' | 'lines' | 'polygons'} {
  const arrays = buildNearestArrays(geometry);
  const make = (
    suffix: string,
    values: Float32Array | Uint32Array,
    format: 'float32x2' | 'uint32'
  ) => {
    const buffer = createInputBuffer(device, values.length ? values : new Uint32Array(2));
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `${name}-${suffix}`,
      buffer,
      format,
      format === 'float32x2' ? values.length / 2 : values.length
    );
  };
  const positions = make('positions', arrays.positions, 'float32x2') as GraphDataView<'float32x2'>;
  if (geometry.kind === 'points') {
    return {kind: 'points', positions};
  }
  if (geometry.kind === 'lines') {
    return {
      kind: 'lines',
      positions,
      lineOffsets: make('line-offsets', arrays.lineOffsets!, 'uint32') as GraphDataView<'uint32'>
    };
  }
  return {
    kind: 'polygons',
    positions,
    featureOffsets: make(
      'feature-offsets',
      arrays.featureOffsets!,
      'uint32'
    ) as GraphDataView<'uint32'>,
    polygonOffsets: make(
      'polygon-offsets',
      arrays.polygonOffsets!,
      'uint32'
    ) as GraphDataView<'uint32'>,
    ringOffsets: make('ring-offsets', arrays.ringOffsets!, 'uint32') as GraphDataView<'uint32'>
  };
}

function createNeighborRun(device: Device, input: NeighborCase): NeighborRun {
  const graph = new GPUCommandGraph(device, {id: 'nearest-neighbors'});
  const buffers: Buffer[] = [];
  const features = importGeometry(device, graph, input.features, 'features', buffers);
  const queryCount = input.queries.features.length;
  const featureCount = input.features.features.length;
  const capacity = input.capacity ?? input.k;
  const slotCount = queryCount * capacity;
  let queryProps:
    | {points: NonNullable<ConstructorParameters<typeof GPUNearestFeatureJoin>[0]['points']>}
    | {queries: ReturnType<typeof importGeometry>};
  if (input.queries.kind === 'points') {
    const positions = input.queries.features;
    const sizes = input.queryChunks ?? [positions.length];
    const chunks: GraphDataView<'float32x2'>[] = [];
    let first = 0;
    for (const [index, size] of sizes.entries()) {
      const chunkPoints = positions.slice(first, first + size);
      first += size;
      const buffer = createInputBuffer(device, Float32Array.from(chunkPoints.flat()));
      buffers.push(buffer);
      chunks.push(
        importGraphBuffer(graph, `query-chunk-${index}`, buffer, 'float32x2', chunkPoints.length)
      );
    }
    queryProps = {
      points:
        chunks.length === 1 ? chunks[0] : createVectorView('query-points', 'float32x2', chunks)
    };
  } else {
    queryProps = {queries: importGeometry(device, graph, input.queries, 'queries', buffers)};
  }
  const ids = createOutputBuffer(device, slotCount);
  const distances = createOutputBuffer(device, slotCount);
  const foot = createOutputBuffer(device, slotCount * 2);
  const segments = createOutputBuffer(device, slotCount);
  const counts = createOutputBuffer(device, queryCount);
  const overflow = createOutputBuffer(device, 1);
  buffers.push(ids, distances, foot, segments, counts, overflow);
  let featureIds: GraphDataView<'uint32'> | undefined;
  if (input.featureIds) {
    const buffer = createInputBuffer(device, Uint32Array.from(input.featureIds));
    buffers.push(buffer);
    featureIds = importGraphBuffer(graph, 'feature-ids', buffer, 'uint32', featureCount);
  }
  let maxDistance: GPUParameterBuffer<'float32'> | undefined;
  if (input.maxDistance !== undefined) {
    maxDistance = new GPUParameterBuffer(device, {
      id: 'max-distance',
      format: 'float32',
      length: 1,
      values: Float32Array.of(input.maxDistance)
    });
  }
  const prepared = input.prepared
    ? new GPUSpatialJoinPrepared({geometry: features, leafCapacity: input.leafCapacity})
    : undefined;
  if (prepared) graph.add(prepared);
  graph.add(
    new GPUNearestFeatureJoin({
      ...queryProps,
      features,
      featureIds,
      k: input.k,
      ties: input.ties,
      neighborCapacity: capacity,
      spatialSort: prepared ? undefined : input.spatialSort,
      leafCapacity: input.leafCapacity,
      prepared,
      maxDistance: maxDistance?.importToGraph(graph),
      neighborIds: importGraphBuffer(graph, 'neighbor-ids', ids, 'uint32', slotCount),
      neighborDistances: importGraphBuffer(
        graph,
        'neighbor-distances',
        distances,
        'float32',
        slotCount
      ),
      neighborFootPoints: importGraphBuffer(graph, 'neighbor-foot', foot, 'float32x2', slotCount),
      neighborSegmentIndices: importGraphBuffer(
        graph,
        'neighbor-segments',
        segments,
        'uint32',
        slotCount
      ),
      neighborCounts: importGraphBuffer(graph, 'neighbor-counts', counts, 'uint32', queryCount),
      overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  return {
    prepared,
    setMaxDistance: value => maxDistance?.write(Float32Array.of(value)),
    run: async () => {
      submitGraph(device, compiled, undefined);
      const flatIds = await readUint32(ids, slotCount);
      const flatDistances = await readFloat32(distances, slotCount);
      const flatFoot = await readFloat32(foot, slotCount * 2);
      const flatSegments = await readUint32(segments, slotCount);
      const result: NeighborResult = {
        ids: [],
        distances: [],
        foot: [],
        segments: [],
        counts: await readUint32(counts, queryCount),
        overflow: (await readUint32(overflow, 1))[0]
      };
      for (let query = 0; query < queryCount; query++) {
        const base = query * capacity;
        result.ids.push(flatIds.slice(base, base + capacity));
        result.distances.push(flatDistances.slice(base, base + capacity));
        result.segments.push(flatSegments.slice(base, base + capacity));
        result.foot.push(
          Array.from({length: capacity}, (_, slot) => [
            flatFoot[(base + slot) * 2],
            flatFoot[(base + slot) * 2 + 1]
          ])
        );
      }
      return result;
    },
    destroy: () => {
      compiled.destroy();
      prepared?.destroy();
      maxDistance?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

async function runNeighbors(device: Device, input: NeighborCase): Promise<NeighborResult> {
  const run = createNeighborRun(device, input);
  try {
    return await run.run();
  } finally {
    run.destroy();
  }
}

/**
 * Compares a GPU result to the oracle. IDs are compared only where the oracle's distance gap to
 * the previous and next neighbor is at least `gap`, since f32 distances may reorder near-ties.
 */
function expectMatchesOracle(
  result: NeighborResult,
  oracle: OracleNeighbor[][],
  capacity: number,
  options: {exactIds?: boolean; checkFoot?: boolean; footSlots?: number} = {}
): void {
  expect(result.counts.length).toBe(oracle.length);
  for (const [query, expected] of oracle.entries()) {
    expect(result.counts[query], `count of query ${query}`).toBe(expected.length);
    for (let slot = 0; slot < capacity; slot++) {
      if (slot >= expected.length) {
        expect(result.ids[query][slot]).toBe(NO_FEATURE);
        expect(result.distances[query][slot]).toBe(-1);
        expect(result.segments[query][slot]).toBe(GPU_NEAREST_NO_SEGMENT);
        expect(Number.isNaN(result.foot[query][slot][0])).toBe(true);
        continue;
      }
      const neighbor = expected[slot];
      expect(result.distances[query][slot]).toBeCloseTo(neighbor.distance, 3);
      if (options.exactIds) {
        expect(result.ids[query][slot], `id of query ${query} slot ${slot}`).toBe(neighbor.id);
      }
      if (options.checkFoot && slot < (options.footSlots ?? capacity)) {
        expect(result.foot[query][slot][0]).toBeCloseTo(neighbor.foot[0], 3);
        expect(result.foot[query][slot][1]).toBeCloseTo(neighbor.foot[1], 3);
        expect(result.segments[query][slot]).toBe(
          neighbor.segment < 0 ? GPU_NEAREST_NO_SEGMENT : neighbor.segment
        );
      }
    }
  }
}

function randomIntegerPoints(random: () => number, count: number, extent: number): NearestPoint[] {
  return Array.from({length: count}, () => [
    Math.floor(random() * extent),
    Math.floor(random() * extent)
  ]);
}

function randomIntegerLines(random: () => number, count: number, extent: number): NearestPoint[][] {
  return Array.from({length: count}, () => {
    const start: NearestPoint = [Math.floor(random() * extent), Math.floor(random() * extent)];
    const vertexCount = 2 + Math.floor(random() * 3);
    const line: NearestPoint[] = [start];
    for (let index = 1; index < vertexCount; index++) {
      const previous = line[index - 1];
      line.push([
        Math.min(extent, Math.max(0, previous[0] + Math.floor(random() * 9) - 4)),
        Math.min(extent, Math.max(0, previous[1] + Math.floor(random() * 9) - 4))
      ]);
    }
    return line;
  });
}

/** Axis-aligned squares (some with a hole) and a few two-part multipolygons on integer corners. */
function randomIntegerPolygons(random: () => number, count: number, extent: number) {
  return Array.from({length: count}, (_, index) => {
    const x = Math.floor(random() * extent);
    const y = Math.floor(random() * extent);
    const size = 4 + Math.floor(random() * 5);
    const square = (cx: number, cy: number, s: number): NearestPoint[] => [
      [cx, cy],
      [cx + s, cy],
      [cx + s, cy + s],
      [cx, cy + s]
    ];
    const rings = [square(x, y, size)];
    if (index % 3 === 0) rings.push(square(x + 1, y + 1, size - 2).reverse());
    const feature = [rings];
    if (index % 4 === 0) feature.push([square(x + size + 3, y, 2)]);
    return feature;
  });
}

async function getDevice(): Promise<Device | undefined> {
  const device = await getWebGPUTestDevice();
  return device ?? undefined;
}

describe('GPUNearestFeatureJoin neighbors mode', () => {
  it('returns k nearest points with lowest-ID ties against the oracle', async () => {
    const device = await getDevice();
    if (!device) return;
    const random = createRandom(3);
    // A tight integer grid forces many exact distance ties.
    const input: NeighborCase = {
      queries: {kind: 'points', features: randomIntegerPoints(random, 160, 20)},
      features: {kind: 'points', features: randomIntegerPoints(random, 90, 20)},
      k: 4
    };
    const oracle = nearestNeighborsOracle(input.queries, input.features, {k: 4});
    const result = await runNeighbors(device, input);
    expect(result.counts.every(count => count === 4)).toBe(true);
    expectMatchesOracle(result, oracle, 4, {exactIds: true, checkFoot: true});
    expect(result.overflow).toBe(0);
    // The same answer with spatial sorting, a tiny leaf capacity mismatch avoided, and chunked queries.
    const sorted = await runNeighbors(device, {
      ...input,
      spatialSort: true,
      queryChunks: [100, 0, 60]
    });
    expect(sorted.ids).toEqual(result.ids);
    expect(sorted.distances).toEqual(result.distances);
  });

  it('writes feature IDs, honors maxDistance per frame, and pads missing neighbors', async () => {
    const device = await getDevice();
    if (!device) return;
    const features: NearestPoint[] = [
      [0, 0],
      [10, 0],
      [0, 10],
      [30, 30]
    ];
    const queries: NearestPoint[] = [
      [1, 0],
      [9, 9],
      [100, 100]
    ];
    const featureIds = [70, 71, 72, 73];
    const input: NeighborCase = {
      queries: {kind: 'points', features: queries},
      features: {kind: 'points', features},
      k: 3,
      maxDistance: 12,
      featureIds
    };
    const run = createNeighborRun(device, input);
    let result = await run.run();
    let oracle = nearestNeighborsOracle(input.queries, input.features, {
      k: 3,
      maxDistance: 12,
      featureIds
    });
    expect(result.counts).toEqual([3, 2, 0]);
    expectMatchesOracle(result, oracle, 3, {exactIds: true, checkFoot: true});
    expect(result.ids[0].slice(0, 3)).toEqual([70, 71, 72]);

    run.setMaxDistance(1);
    result = await run.run();
    expect(result.counts).toEqual([1, 0, 0]);
    expect(result.ids[0]).toEqual([70, NO_FEATURE, NO_FEATURE]);

    run.setMaxDistance(Number.NaN);
    result = await run.run();
    expect(result.counts).toEqual([0, 0, 0]);

    run.setMaxDistance(1000);
    result = await run.run();
    oracle = nearestNeighborsOracle(input.queries, input.features, {k: 3, featureIds});
    expect(result.counts).toEqual([3, 3, 3]);
    expectMatchesOracle(result, oracle, 3, {exactIds: true});
    run.destroy();
  });

  it('keeps all ties at the k-th distance and flags truncation', async () => {
    const device = await getDevice();
    if (!device) return;
    // Four features equidistant from the first query, one farther. The second query has no tie.
    const features: NearestPoint[] = [
      [2, 0],
      [-2, 0],
      [0, 2],
      [0, -2],
      [5, 5]
    ];
    const queries: NearestPoint[] = [
      [0, 0],
      [5, 4]
    ];
    const base: NeighborCase = {
      queries: {kind: 'points', features: queries},
      features: {kind: 'points', features},
      k: 1,
      ties: 'all',
      capacity: 6
    };
    const result = await runNeighbors(device, base);
    expect(result.counts).toEqual([4, 1]);
    expect(result.ids[0]).toEqual([0, 1, 2, 3, NO_FEATURE, NO_FEATURE]);
    expect(result.ids[1][0]).toBe(4);
    expect(result.overflow).toBe(0);
    expectMatchesOracle(
      result,
      nearestNeighborsOracle(base.queries, base.features, {k: 1, ties: 'all'}),
      6,
      {exactIds: true, checkFoot: true}
    );

    // Capacity 2 cannot hold four ties: the lowest two rows stay and overflow is raised.
    const truncated = await runNeighbors(device, {...base, capacity: 2});
    expect(truncated.counts).toEqual([2, 1]);
    expect(truncated.ids[0]).toEqual([0, 1]);
    expect(truncated.overflow).toBe(1);

    // lowest-id mode with k = 1 keeps only the lowest row and never overflows.
    const lowest = await runNeighbors(device, {...base, ties: 'lowest-id', capacity: undefined});
    expect(lowest.counts).toEqual([1, 1]);
    expect(lowest.ids.map(row => row[0])).toEqual([0, 4]);
    expect(lowest.overflow).toBe(0);

    // k = 2 with ties: the 2nd distance is shared by rows 1..3, so all of 0..3 are returned.
    const kTwo = await runNeighbors(device, {...base, k: 2, capacity: 8});
    expect(kTwo.counts[0]).toBe(4);
  });

  it('reports exact foot points and segment indices for lines', async () => {
    const device = await getDevice();
    if (!device) return;
    const features: NearestPoint[][] = [
      [
        [0, 0],
        [10, 0],
        [10, 10]
      ],
      [
        [0, 5],
        [6, 5]
      ]
    ];
    const input: NeighborCase = {
      queries: {
        kind: 'points',
        features: [
          [4, 1],
          [12, 6],
          [3, 9]
        ]
      },
      features: {kind: 'lines', features},
      k: 2
    };
    const result = await runNeighbors(device, input);
    // Query 0: line 0 edge 0 at (4, 0), then line 1 edge at vertex 3 (absolute) at (4, 5).
    expect(result.ids[0]).toEqual([0, 1]);
    expect(result.foot[0][0][0]).toBeCloseTo(4, 4);
    expect(result.foot[0][0][1]).toBeCloseTo(0, 4);
    expect(result.segments[0]).toEqual([0, 3]);
    expect(result.distances[0][0]).toBeCloseTo(1, 5);
    // Query 1: nearest is line 0's second edge (absolute vertex 1) at (10, 6).
    expect(result.ids[1][0]).toBe(0);
    expect(result.foot[1][0][0]).toBeCloseTo(10, 4);
    expect(result.foot[1][0][1]).toBeCloseTo(6, 4);
    expect(result.segments[1][0]).toBe(1);
    expectMatchesOracle(result, nearestNeighborsOracle(input.queries, input.features, {k: 2}), 2, {
      exactIds: true,
      checkFoot: true
    });
  });

  it('matches the oracle for point, line and polygon queries against lines and polygons', async () => {
    const device = await getDevice();
    if (!device) return;
    const random = createRandom(21);
    const extent = 60;
    const geometries = {
      points: () => ({kind: 'points', features: randomIntegerPoints(random, 70, extent)}) as const,
      lines: (count = 45) =>
        ({kind: 'lines', features: randomIntegerLines(random, count, extent)}) as const,
      polygons: (count = 30) =>
        ({kind: 'polygons', features: randomIntegerPolygons(random, count, extent)}) as const
    };
    const cases: [string, NearestOracleGeometry, NearestOracleGeometry][] = [
      ['points to lines', geometries.points(), geometries.lines()],
      ['points to polygons', geometries.points(), geometries.polygons()],
      ['lines to points', geometries.lines(), geometries.points()],
      ['lines to lines', geometries.lines(), geometries.lines()],
      ['lines to polygons', geometries.lines(), geometries.polygons()],
      ['polygons to lines', geometries.polygons(20), geometries.lines()],
      ['polygons to polygons', geometries.polygons(20), geometries.polygons()]
    ];
    for (const [name, queries, features] of cases) {
      const input: NeighborCase = {queries, features, k: 3, spatialSort: name.includes('lines')};
      const wide = nearestNeighborsOracle(queries, features, {k: 4});
      const oracle = wide.map(row => row.slice(0, 3));
      const result = await runNeighbors(device, input);
      expect(result.overflow, name).toBe(0);
      // Distances must agree everywhere. IDs are compared where no tie neighbors the slot.
      expectMatchesOracle(result, oracle, 3);
      for (const [query, expected] of oracle.entries()) {
        for (const [slot, neighbor] of expected.entries()) {
          const loose = [wide[query][slot - 1], wide[query][slot + 1]].some(
            other => other && Math.abs(other.distance - neighbor.distance) < 1e-3
          );
          if (!loose) {
            expect(result.ids[query][slot], `${name} query ${query} slot ${slot}`).toBe(
              neighbor.id
            );
          }
        }
      }
      // A nonzero result guards against silent shader failure.
      expect(
        result.distances.flat().some(distance => distance > 0) ||
          result.counts.some(count => count > 0),
        name
      ).toBe(true);
    }
  });

  it('gives distance 0 for polygon containment and reports the contained vertex', async () => {
    const device = await getDevice();
    if (!device) return;
    const outer = (x: number, y: number, s: number): NearestPoint[] => [
      [x, y],
      [x + s, y],
      [x + s, y + s],
      [x, y + s]
    ];
    // Feature 0 is a square with a hole; feature 1 is a distant square.
    const features: NearestOracleGeometry = {
      kind: 'polygons',
      features: [[[outer(0, 0, 20), outer(8, 8, 4).reverse()]], [[outer(40, 40, 5)]]]
    };
    const queries: NearestOracleGeometry = {
      kind: 'points',
      features: [
        [3, 3], // inside the shell: distance 0
        [10, 10], // inside the hole: distance 2 to the hole boundary
        [30, 10] // outside
      ]
    };
    const input: NeighborCase = {queries, features, k: 2};
    const result = await runNeighbors(device, input);
    expect(result.ids[0][0]).toBe(0);
    expect(result.distances[0][0]).toBe(0);
    expect(result.foot[0][0]).toEqual([3, 3]);
    expect(result.segments[0][0]).toBe(GPU_NEAREST_NO_SEGMENT);
    expect(result.distances[1][0]).toBeCloseTo(2, 5);
    expect(result.distances[2][0]).toBeCloseTo(10, 5);
    expectMatchesOracle(result, nearestNeighborsOracle(queries, features, {k: 2}), 2, {
      exactIds: true,
      checkFoot: true,
      footSlots: 1
    });

    // Polygon queries: one contained in feature 0's shell, one covering feature 1, one crossing.
    const polygonQueries: NearestOracleGeometry = {
      kind: 'polygons',
      features: [[[outer(2, 2, 3)]], [[outer(38, 38, 10)]], [[outer(18, 18, 4)]]]
    };
    const polygonResult = await runNeighbors(device, {queries: polygonQueries, features, k: 2});
    expect(polygonResult.ids.map(row => row[0])).toEqual([0, 1, 0]);
    expect(polygonResult.distances.map(row => row[0])).toEqual([0, 0, 0]);
    expect(polygonResult.segments.map(row => row[0])[0]).toBe(GPU_NEAREST_NO_SEGMENT);
    expectMatchesOracle(
      polygonResult,
      nearestNeighborsOracle(polygonQueries, features, {k: 2}),
      2,
      {exactIds: true}
    );
  });

  it('reports crossing line geometries with the crossing point as the foot', async () => {
    const device = await getDevice();
    if (!device) return;
    const queries: NearestOracleGeometry = {
      kind: 'lines',
      features: [
        [
          [0, 0],
          [10, 10]
        ],
        [
          [0, 20],
          [10, 20]
        ]
      ]
    };
    const features: NearestOracleGeometry = {
      kind: 'lines',
      features: [
        [
          [0, 10],
          [10, 0]
        ],
        [
          [0, 14],
          [10, 14]
        ]
      ]
    };
    const result = await runNeighbors(device, {queries, features, k: 2});
    expect(result.ids[0]).toEqual([0, 1]);
    expect(result.distances[0][0]).toBe(0);
    expect(result.foot[0][0][0]).toBeCloseTo(5, 4);
    expect(result.foot[0][0][1]).toBeCloseTo(5, 4);
    expect(result.distances[0][1]).toBeCloseTo(Math.hypot(0, 4), 4);
    // The second query line is 6 above the second feature and 10 above the first.
    expect(result.ids[1]).toEqual([1, 0]);
    expect(result.distances[1][0]).toBeCloseTo(6, 4);
  });

  it('reuses a prepared feature index across encodings until it is invalidated', async () => {
    const device = await getDevice();
    if (!device) return;
    const random = createRandom(5);
    for (const features of [
      {kind: 'lines', features: randomIntegerLines(random, 40, 50)} as const,
      {kind: 'polygons', features: randomIntegerPolygons(random, 25, 50)} as const
    ]) {
      const input: NeighborCase = {
        queries: {kind: 'points', features: randomIntegerPoints(random, 80, 50)},
        features,
        k: 3,
        prepared: true
      };
      const oracle = nearestNeighborsOracle(input.queries, input.features, {k: 3});
      const run = createNeighborRun(device, input);
      const first = await run.run();
      expectMatchesOracle(first, oracle, 3);
      expect(first.counts.every(count => count === 3)).toBe(true);
      const second = await run.run();
      expect(second).toEqual(first);
      expect(run.prepared?.encodedBuildCount).toBe(1);
      run.prepared?.invalidate();
      expect(await run.run()).toEqual(first);
      expect(run.prepared?.encodedBuildCount).toBe(2);
      run.destroy();
    }
  });

  it('accepts a prepared index in nearest-feature mode', async () => {
    const device = await getDevice();
    if (!device) return;
    const graph = new GPUCommandGraph(device, {id: 'nearest-prepared-legacy'});
    const buffers: Buffer[] = [];
    const features = importGeometry(
      device,
      graph,
      {
        kind: 'points',
        features: [
          [0, 0],
          [10, 0],
          [5, 5]
        ]
      },
      'features',
      buffers
    );
    const queries = importGeometry(
      device,
      graph,
      {
        kind: 'points',
        features: [
          [1, 0],
          [9, 1],
          [5, 4],
          [50, 50]
        ]
      },
      'queries',
      buffers
    );
    const ids = createOutputBuffer(device, 4);
    const overflow = createOutputBuffer(device, 1);
    const radius = new GPUParameterBuffer(device, {
      id: 'radius',
      format: 'float32',
      length: 1,
      values: Float32Array.of(2)
    });
    const prepared = new GPUSpatialJoinPrepared({geometry: features});
    graph.add(prepared);
    graph.add(
      new GPUNearestFeatureJoin({
        points: queries.positions,
        features,
        prepared,
        radius: radius.importToGraph(graph),
        candidateCapacity: 32,
        nearestFeatureIds: importGraphBuffer(graph, 'ids', ids, 'uint32', 4),
        overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
      })
    );
    const compiled = graph.compile();
    for (let encoding = 0; encoding < 2; encoding++) {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(ids, 4)).toEqual([0, 1, 2, NO_FEATURE]);
    }
    expect(prepared.encodedBuildCount).toBe(1);
    compiled.destroy();
    prepared.destroy();
    radius.destroy();
    for (const buffer of [...buffers, ids, overflow]) buffer.destroy();
  });

  it('handles empty and degenerate features and an empty feature set', async () => {
    const device = await getDevice();
    if (!device) return;
    const queries: NearestOracleGeometry = {kind: 'points', features: [[1, 1]]};
    const lines: NearestOracleGeometry = {
      kind: 'lines',
      features: [
        [[0, 0]],
        [],
        [
          [5, 5],
          [5, 5]
        ],
        [
          [2, 2],
          [3, 3]
        ]
      ]
    };
    const result = await runNeighbors(device, {queries, features: lines, k: 4});
    // The one-vertex and empty lines are unusable; the zero-length line is a point.
    expect(result.counts).toEqual([2]);
    expect(result.ids[0].slice(0, 2)).toEqual([3, 2]);
    expect(result.distances[0][0]).toBeCloseTo(Math.SQRT2, 5);
    expect(result.distances[0][1]).toBeCloseTo(Math.hypot(4, 4), 5);
  });

  it('matches the oracle on a deep tree with a large k', async () => {
    const device = await getDevice();
    if (!device) return;
    const random = createRandom(77);
    const input: NeighborCase = {
      queries: {
        kind: 'points',
        features: Array.from({length: 250}, () => [random() * 300, random() * 300] as NearestPoint)
      },
      features: {
        kind: 'lines',
        features: Array.from({length: 700}, () => {
          const x = random() * 300;
          const y = random() * 300;
          return [
            [x, y],
            [x + random() * 6 - 3, y + random() * 6 - 3]
          ] as NearestPoint[];
        }).map(line => line.map(([x, y]) => [Math.fround(x), Math.fround(y)] as NearestPoint))
      },
      k: 16,
      spatialSort: true
    };
    input.queries = {
      kind: 'points',
      features: (input.queries as {features: NearestPoint[]}).features.map(
        ([x, y]) => [Math.fround(x), Math.fround(y)] as NearestPoint
      )
    };
    const result = await runNeighbors(device, input);
    const oracle = nearestNeighborsOracle(input.queries, input.features, {k: 16});
    expectMatchesOracle(result, oracle, 16);
    expect(result.counts.every(count => count === 16)).toBe(true);
  });
});

describe('GPUNearestFeatureJoin neighbors mode validation', () => {
  it('rejects inconsistent properties without a GPU device', async () => {
    const device = await getDevice();
    if (!device) return;
    const graph = new GPUCommandGraph(device, {id: 'nearest-validation'});
    const buffers = [0, 1, 2, 3, 4, 5].map(() => createInputBuffer(device, new Float32Array(16)));
    const view = <Format extends 'float32x2' | 'uint32' | 'float32'>(
      index: number,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, `view-${index}`, buffers[index], format, length);
    const base = {
      points: view(0, 'float32x2', 4),
      features: {kind: 'points', positions: view(1, 'float32x2', 4)} as const,
      neighborIds: view(2, 'uint32', 4),
      neighborCounts: view(3, 'uint32', 4),
      overflow: view(4, 'uint32', 1)
    };
    expect(() => new GPUNearestFeatureJoin(base)).not.toThrow();
    expect(() => new GPUNearestFeatureJoin({...base, k: 0})).toThrow(/k must be/);
    expect(() => new GPUNearestFeatureJoin({...base, neighborCapacity: 2})).toThrow(/ties/);
    expect(() => new GPUNearestFeatureJoin({...base, neighborCounts: undefined})).toThrow(
      /neighborCounts/
    );
    expect(() => new GPUNearestFeatureJoin({...base, nearestFeatureIds: base.neighborIds})).toThrow(
      /belong to nearest-feature mode/
    );
    expect(() => new GPUNearestFeatureJoin({...base, neighborIds: view(5, 'uint32', 3)})).toThrow(
      /length must be 4/
    );
    for (const buffer of buffers) buffer.destroy();
  });
});
