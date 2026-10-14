// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join';
import {getNearestNeighborNodes} from '../../../src/gpu-spatial-analysis/spatial-join/nearest-nodes';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  buildNearestArrays,
  nearestNeighborsOracle,
  type NearestOracleGeometry,
  type NearestPoint
} from './nearest-feature-oracle';

type GeometryKind = 'lines' | 'polygons';

function createSquare(x: number, y: number, size: number): NearestPoint[] {
  return [
    [x, y],
    [x + size, y],
    [x + size, y + size],
    [x, y + size]
  ];
}

/** Incoherent rows, duplicate centers, empty/short/non-finite rows, a hole and a multipolygon. */
function createQueries(kind: GeometryKind, count: number): NearestOracleGeometry {
  const rings: NearestPoint[][] = [
    createSquare(41, 1, 1),
    createSquare(1, 1, 1),
    createSquare(21, 1, 1),
    [],
    [[3, 3]],
    [
      [Number.NaN, 0],
      [Number.NaN, 1],
      [Number.NaN, 2]
    ],
    createSquare(21, 1, 1),
    createSquare(200, 200, 2),
    [
      [Number.POSITIVE_INFINITY, 0],
      [Number.POSITIVE_INFINITY, 1],
      [Number.POSITIVE_INFINITY, 2]
    ],
    createSquare(15, -5, 15),
    createSquare(-10, -10, 2)
  ];
  if (kind === 'lines') {
    return {kind, features: Array.from({length: count}, (_, index) => rings[index % rings.length])};
  }
  return {
    kind,
    features: Array.from({length: count}, (_, index) => {
      const row = index % rings.length;
      if (row === 9) return [[rings[row], createSquare(18, -2, 8)]];
      if (row === 10) return [[rings[row]], [createSquare(39, -3, 2)]];
      return [[rings[row]]];
    })
  };
}

function createRun(
  device: Device,
  queryKind: GeometryKind,
  options: {
    querySort: boolean;
    scratchChunkQueryCount?: number;
    prepared?: boolean;
    filters?: boolean;
    sphereRadius?: number;
    queryCount?: number;
    featureCount?: number;
  }
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const queries = createQueries(queryKind, options.queryCount ?? 133);
  const features: NearestOracleGeometry = {
    kind: 'polygons',
    features: Array.from({length: options.featureCount ?? 4}, (_, index) => [
      [createSquare(index === 3 ? 20 : index * 20, 0, 4)]
    ])
  };
  const queryCount = queries.features.length;
  const featureCount = features.features.length;
  const capacity = 4;
  const input = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const importGeometry = (name: string, geometry: NearestOracleGeometry) => {
    const arrays = buildNearestArrays(geometry);
    const positions = input(`${name}-positions`, arrays.positions, 'float32x2');
    if (geometry.kind === 'lines') {
      return {
        kind: 'lines' as const,
        positions,
        lineOffsets: input(`${name}-lines`, arrays.lineOffsets!, 'uint32')
      };
    }
    return {
      kind: 'polygons' as const,
      positions,
      featureOffsets: input(`${name}-features`, arrays.featureOffsets!, 'uint32'),
      polygonOffsets: input(`${name}-polygons`, arrays.polygonOffsets!, 'uint32'),
      ringOffsets: input(`${name}-rings`, arrays.ringOffsets!, 'uint32')
    };
  };
  const queryGeometry = importGeometry('query', queries);
  const featureGeometry = importGeometry('feature', features);
  const output = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const words = length * (format === 'float32x2' ? 2 : 1);
    const buffer = createOutputBuffer(device, words);
    buffers.push(buffer);
    return {buffer, words, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const ids = output('ids', 'uint32', queryCount * capacity);
  const counts = output('counts', 'uint32', queryCount);
  const distances = output('distances', 'float32', queryCount * capacity);
  const footPoints = output('foot', 'float32x2', queryCount * capacity);
  const queryPoints = output('query-foot', 'float32x2', queryCount * capacity);
  const segments = output('segments', 'uint32', queryCount * capacity);
  const overflow = output('overflow', 'uint32', 1);
  const prepared = options.prepared
    ? new GPUSpatialJoinPrepared({geometry: featureGeometry})
    : undefined;
  if (prepared) graph.add(prepared);
  graph.add(
    getNearestNeighborNodes(graph, {
      id: 'sorted-geometry',
      operation: 'GPUNearestFeatureJoin',
      queries: {kind: 'geometry', geometry: queryGeometry},
      queryCount,
      features: featureGeometry,
      featureCount,
      k: 2,
      capacity,
      leafCapacity: 2 ** Math.ceil(Math.log2(featureCount)),
      spatialSort: true,
      prepared,
      neighborIds: ids.view,
      neighborCounts: counts.view,
      neighborDistances: distances.view,
      neighborFootPoints: footPoints.view,
      neighborQueryPoints: queryPoints.view,
      neighborSegmentIndices: segments.view,
      overflow: overflow.view,
      maxDistance: input(
        'max-distance',
        Float32Array.of(options.sphereRadius ? 1500000 : 12),
        'float32'
      ),
      featureIds: input(
        'feature-ids',
        Uint32Array.from({length: featureCount}, (_, index) => 100 + index),
        'uint32'
      ),
      ...(options.filters
        ? {
            exclusive: true,
            queryIds: input(
              'query-ids',
              Uint32Array.from({length: queryCount}, (_, index) => 100 + (index % featureCount)),
              'uint32'
            ),
            onAttribute: {
              left: input(
                'query-keys',
                Uint32Array.from({length: queryCount}, (_, index) => index % 2),
                'uint32'
              ),
              right: input(
                'feature-keys',
                Uint32Array.from({length: featureCount}, (_, index) => index % 2),
                'uint32'
              )
            }
          }
        : {}),
      querySort: options.querySort,
      scratchChunkQueryCount: options.scratchChunkQueryCount,
      sphereRadius: options.sphereRadius
    }).nodes
  );
  const compiled = graph.compile();
  const encodeAndWait = async () => {
    submitGraph(device, compiled, undefined);
    await readUint32(overflow.buffer, 1);
  };
  return {
    queries,
    features,
    capacity,
    encodeAndWait,
    read: async () => {
      await encodeAndWait();
      return {
        ids: await readUint32(ids.buffer, ids.words),
        counts: await readUint32(counts.buffer, counts.words),
        distances: await readUint32(distances.buffer, distances.words),
        footPoints: await readUint32(footPoints.buffer, footPoints.words),
        queryPoints: await readUint32(queryPoints.buffer, queryPoints.words),
        segments: await readUint32(segments.buffer, segments.words),
        overflow: await readUint32(overflow.buffer, overflow.words)
      };
    },
    destroy: () => {
      compiled.destroy();
      prepared?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

for (const queryKind of ['lines', 'polygons'] as const) {
  for (const scratchChunkQueryCount of [undefined, 64]) {
    it(`Hilbert ${queryKind} queries retain every output bit in source order (scratch ${scratchChunkQueryCount})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const mode of [
        {},
        {prepared: true, filters: true},
        {prepared: true, sphereRadius: 6371008.8}
      ]) {
        const options = {scratchChunkQueryCount, ...mode};
        const off = createRun(device, queryKind, {...options, querySort: false});
        const on = createRun(device, queryKind, {...options, querySort: true});
        try {
          const expected = await off.read();
          const actual = await on.read();
          expect(actual).toEqual(expected);
          expect(actual.counts.some(count => count > 0)).toBe(true);
          expect(actual.counts[3]).toBe(0);
          expect(actual.counts[4]).toBe(0);
          expect(actual.counts[5]).toBe(0);
          expect(actual.counts[8]).toBe(0);
          expect(actual.overflow).toEqual([0]);
          if (!mode.filters && !mode.sphereRadius) {
            // Existing independent edge-pair oracle, on the first three finite source rows.
            const finiteQueries = {
              ...off.queries,
              features: off.queries.features.slice(0, 3)
            } as NearestOracleGeometry;
            const oracle = nearestNeighborsOracle(finiteQueries, off.features, {
              k: 2,
              ties: 'all',
              maxDistance: 12,
              featureIds: [100, 101, 102, 103]
            });
            oracle.forEach((neighbors, row) => {
              expect(actual.counts[row]).toBe(neighbors.length);
              expect(
                actual.ids.slice(row * off.capacity, row * off.capacity + neighbors.length)
              ).toEqual(neighbors.map(neighbor => neighbor.id));
            });
          }
          // Re-encoding recomputes the GPU order, including with a prepared feature index.
          expect(await on.read()).toEqual(actual);
        } finally {
          off.destroy();
          on.destroy();
        }
      }
    }, 120000);
  }
}

it('A/B: geometry Hilbert query order, clustered lines and polygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const queryKind of ['lines', 'polygons'] as const) {
    const options = {queryCount: 8192, featureCount: 512, prepared: true};
    const off = createRun(device, queryKind, {...options, querySort: false});
    const on = createRun(device, queryKind, {...options, querySort: true});
    try {
      expect(await on.read()).toEqual(await off.read());
      await off.encodeAndWait();
      await on.encodeAndWait();
      const samples = {off: [] as number[], on: [] as number[]};
      for (let repeat = 0; repeat < 5; repeat++) {
        for (const mode of (repeat % 2 ? ['on', 'off'] : ['off', 'on']) as ('off' | 'on')[]) {
          const start = performance.now();
          await (mode === 'off' ? off : on).encodeAndWait();
          samples[mode].push(performance.now() - start);
        }
      }
      const median = (values: number[]) => values.sort((left, right) => left - right)[2];
      // eslint-disable-next-line no-console
      console.log(
        `GEOMETRY-QUERY-SORT-AB ${queryKind}: off ${median(samples.off).toFixed(2)} ms, on ${median(samples.on).toFixed(2)} ms (encode-to-readback)`
      );
    } finally {
      off.destroy();
      on.destroy();
    }
  }
}, 120000);
