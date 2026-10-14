// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPU_SPATIAL_JOIN_NO_FEATURE} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-types';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GEODESIC_NEAREST_SCENES, type GeodesicNearestScene} from './geodesic-nearest-fixtures';
import {requireTestDevice} from '../../utils/require-test-device';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
/** Meters of absolute slack: float32 longitude/latitude quantization is about 1 m at 100 degrees. */
const ABSOLUTE_METERS = 1.5;
const RELATIVE = 2e-3;

function expectDistance(actual: number, expected: number, label: string) {
  expect(Math.abs(actual - expected), label).toBeLessThanOrEqual(
    ABSOLUTE_METERS + RELATIVE * expected
  );
}

/** Longitude difference wrapped to +-180. */
function wrapDelta(delta: number): number {
  return delta - 360 * Math.round(delta / 360);
}

type NeighborsRun = {
  ids: number[][];
  distances: number[][];
  foot: [number, number][][];
  counts: number[];
  overflow: number;
};

async function runNeighbors(
  device: Device,
  scene: GeodesicNearestScene,
  options: {k: number; maxDistance?: number; units?: 'meters' | 'kilometers'}
): Promise<NeighborsRun> {
  const graph = new GPUCommandGraph(device, {id: 'geodesic-neighbors'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'uint32'>(
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
      format === 'float32x2' ? values.length / 2 : values.length
    );
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const vertices = scene.features.flat();
  const lineOffsets = scene.features.map((_, row) => row * 2).concat(vertices.length);
  const queryCount = scene.queries.length;
  const slots = queryCount * options.k;
  const ids = output(slots);
  const distances = output(slots);
  const foot = output(slots * 2);
  const counts = output(queryCount);
  const overflow = output(1);
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
      spatialContext: {...GREAT_CIRCLE, units: options.units},
      queries: {
        kind: 'points',
        positions: input('queries', Float32Array.from(scene.queries.flat()), 'float32x2')
      },
      features: {
        kind: 'lines',
        positions: input('positions', Float32Array.from(vertices.flat()), 'float32x2'),
        lineOffsets: input('line-offsets', Uint32Array.from(lineOffsets), 'uint32')
      },
      k: options.k,
      radius: maxDistance?.importToGraph(graph),
      output: {
        neighborIds: importGraphBuffer(graph, 'ids', ids, 'uint32', slots),
        neighborDistances: importGraphBuffer(graph, 'distances', distances, 'float32', slots),
        neighborFootPoints: importGraphBuffer(graph, 'foot', foot, 'float32x2', slots),
        neighborCounts: importGraphBuffer(graph, 'counts', counts, 'uint32', queryCount),
        status: {overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const flatIds = await readUint32(ids, slots);
  const flatDistances = await readFloat32(distances, slots);
  const flatFoot = await readFloat32(foot, slots * 2);
  const result: NeighborsRun = {
    ids: [],
    distances: [],
    foot: [],
    counts: await readUint32(counts, queryCount),
    overflow: (await readUint32(overflow, 1))[0]
  };
  for (let query = 0; query < queryCount; query++) {
    const base = query * options.k;
    result.ids.push(flatIds.slice(base, base + options.k));
    result.distances.push(flatDistances.slice(base, base + options.k));
    result.foot.push(
      Array.from({length: options.k}, (_, slot) => [
        flatFoot[(base + slot) * 2],
        flatFoot[(base + slot) * 2 + 1]
      ])
    );
  }
  compiled.destroy();
  maxDistance?.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

for (const [name, scene] of Object.entries(GEODESIC_NEAREST_SCENES)) {
  it(`GPUNearestFeatureJoin great-circle neighbors match the oracle (${name})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const k = 2;
    const result = await runNeighbors(device, scene, {k});
    expect(result.overflow).toBe(0);
    for (const [query, ranked] of scene.ranked.entries()) {
      expect(result.counts[query], `${name} query ${query}`).toBe(k);
      for (let slot = 0; slot < k; slot++) {
        const label = `${name} query ${query} slot ${slot}`;
        expectDistance(result.distances[query][slot], ranked[slot].distance, label);
        // IDs are compared where the oracle's next distance is clearly different (f32 near-ties).
        const gap = Math.min(
          slot > 0 ? ranked[slot].distance - ranked[slot - 1].distance : Infinity,
          ranked[slot + 1].distance - ranked[slot].distance
        );
        if (gap > 10) {
          expect(result.ids[query][slot], label).toBe(ranked[slot].row);
          // The foot point is on the edge: compare in meters through the longitude/latitude gap.
          const [footLongitude, footLatitude] = result.foot[query][slot];
          const [expectedLongitude, expectedLatitude] = ranked[slot].foot;
          const metersPerDegree = (Math.PI / 180) * 6371008.8;
          const north = (footLatitude - expectedLatitude) * metersPerDegree;
          const east =
            wrapDelta(footLongitude - expectedLongitude) *
            metersPerDegree *
            Math.cos((expectedLatitude * Math.PI) / 180);
          expect(Math.hypot(north, east), `${label} foot`).toBeLessThanOrEqual(ABSOLUTE_METERS * 2);
        }
      }
    }
  });
}

it('GPUNearestFeatureJoin great-circle neighbors honor a per-frame maxDistance in meters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = GEODESIC_NEAREST_SCENES['equator'];
  // Between the second (3145 m) and third (3336 m) feature of query 0.
  const result = await runNeighbors(device, scene, {k: 3, maxDistance: 3200});
  expect(result.counts).toEqual([2, 1]);
  expect(result.ids[0].slice(0, 2)).toEqual([0, 2]);
  expect(result.ids[0][2]).toBe(GPU_SPATIAL_JOIN_NO_FEATURE);
  expect(result.ids[1][0]).toBe(1);
  expect(result.distances[0][2]).toBe(-1);
  // Negative or NaN limits match nothing.
  expect((await runNeighbors(device, scene, {k: 1, maxDistance: -1})).counts).toEqual([0, 0]);
  expect((await runNeighbors(device, scene, {k: 1, maxDistance: Number.NaN})).counts).toEqual([
    0, 0
  ]);
});

it('GPUNearestFeatureJoin great-circle nearest-feature mode returns the nearest segment in meters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = GEODESIC_NEAREST_SCENES['antimeridian'];
  const graph = new GPUCommandGraph(device, {id: 'geodesic-nearest-feature'});
  const buffers: Buffer[] = [];
  const input = (name: string, values: Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'float32x2', values.length / 2);
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const queryCount = scene.queries.length;
  const featureCount = scene.features.length;
  const ids = output(queryCount);
  const distances = output(queryCount);
  const featureCounts = output(featureCount);
  const overflow = output(1);
  const radius = new GPUParameterBuffer(device, {
    id: 'radius',
    format: 'float32',
    length: 1,
    values: Float32Array.of(2000)
  });
  graph.add(
    new GPUNearestFeatureJoin({
      spatialContext: GREAT_CIRCLE,
      queries: {
        kind: 'points',
        positions: input('queries', Float32Array.from(scene.queries.flat()))
      },
      features: {
        kind: 'segments',
        starts: input('starts', Float32Array.from(scene.features.flatMap(edge => edge[0]))),
        ends: input('ends', Float32Array.from(scene.features.flatMap(edge => edge[1])))
      },
      radius: radius.importToGraph(graph),
      output: {
        nearestFeatureIds: importGraphBuffer(graph, 'ids', ids, 'uint32', queryCount),
        nearestDistances: importGraphBuffer(graph, 'distances', distances, 'float32', queryCount),
        featureCounts: importGraphBuffer(graph, 'counts', featureCounts, 'uint32', featureCount),
        status: {overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const resultIds = await readUint32(ids, queryCount);
  const resultDistances = await readFloat32(distances, queryCount);
  const resultCounts = await readUint32(featureCounts, featureCount);
  expect((await readUint32(overflow, 1))[0]).toBe(0);
  for (const [query, ranked] of scene.ranked.entries()) {
    const nearest = ranked[0];
    if (nearest.distance <= 2000) {
      // Queries 1 and 2 sit equidistant (to 1e-5 m) from two edges, below f32 resolution, so
      // either tied row is correct; the distance is still checked against the oracle.
      const tiedRows = ranked
        .filter(candidate => candidate.distance - nearest.distance <= ABSOLUTE_METERS)
        .map(candidate => candidate.row);
      expect(tiedRows, `query ${query}`).toContain(resultIds[query]);
      expectDistance(resultDistances[query], nearest.distance, `query ${query}`);
    } else {
      expect(resultIds[query]).toBe(GPU_SPATIAL_JOIN_NO_FEATURE);
      expect(resultDistances[query]).toBe(-1);
    }
  }
  expect(resultCounts.reduce((sum, count) => sum + count, 0)).toBe(
    scene.ranked.filter(ranked => ranked[0].distance <= 2000).length
  );
  compiled.destroy();
  radius.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUNearestFeatureJoin measures and thresholds great-circle neighbors in kilometers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = GEODESIC_NEAREST_SCENES.equator;
  const result = await runNeighbors(device, scene, {k: 3, maxDistance: 3.2, units: 'kilometers'});
  expect(result.overflow).toBe(0);
  expect(result.counts).toEqual([2, 1]);
  expect(result.ids[0]).toEqual([0, 2, GPU_SPATIAL_JOIN_NO_FEATURE]);
  for (const [query, ranked] of scene.ranked.entries()) {
    for (let slot = 0; slot < result.counts[query]; slot++) {
      expectDistance(
        result.distances[query][slot] * 1000,
        ranked[slot].distance,
        `kilometers query ${query} slot ${slot}`
      );
    }
  }
  expect(result.distances[0][2]).toBe(-1);
});
