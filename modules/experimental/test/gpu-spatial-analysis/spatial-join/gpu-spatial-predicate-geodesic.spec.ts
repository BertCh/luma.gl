// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GEODESIC_DWITHIN_POINTS, GEODESIC_DWITHIN_POLYGON} from './geodesic-dwithin-fixtures';
import {requireTestDevice} from '../../utils/require-test-device';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

function createPoints(
  device: Device,
  graph: GPUCommandGraph,
  name: string,
  points: [number, number][],
  buffers: Buffer[]
): GPUSpatialJoinGeometry {
  const buffer = createInputBuffer(device, Float32Array.from(points.flat()));
  buffers.push(buffer);
  return {
    kind: 'points',
    positions: importGraphBuffer(graph, `${name}-positions`, buffer, 'float32x2', points.length)
  };
}

function createPolygon(
  device: Device,
  graph: GPUCommandGraph,
  ring: [number, number][],
  buffers: Buffer[]
): GPUSpatialJoinGeometry {
  const upload = (data: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  return {
    kind: 'polygons',
    positions: importGraphBuffer(
      graph,
      'polygon-positions',
      upload(Float32Array.from(ring.flat())),
      'float32x2',
      ring.length
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'polygon-feature-offsets',
      upload(Uint32Array.from([0, 1])),
      'uint32',
      2
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-polygon-offsets',
      upload(Uint32Array.from([0, 1])),
      'uint32',
      2
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'polygon-ring-offsets',
      upload(Uint32Array.from([0, ring.length])),
      'uint32',
      2
    )
  };
}

/** Runs a great-circle dwithin join and reads the pairs after each distance in `distances`. */
async function runGreatCircleJoin(
  device: Device,
  makeSides: (
    graph: GPUCommandGraph,
    buffers: Buffer[]
  ) => {left: GPUSpatialJoinGeometry; right: GPUSpatialJoinGeometry; leftCount: number},
  distances: number[],
  perFrame: boolean
): Promise<{pairs: [number, number][]; overflow: number}[]> {
  const graph = new GPUCommandGraph(device, {id: 'geodesic-dwithin'});
  const buffers: Buffer[] = [];
  const {left, right, leftCount} = makeSides(graph, buffers);
  const pairCapacity = 64;
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const leftIds = output('left-ids', pairCapacity);
  const rightIds = output('right-ids', pairCapacity);
  const count = output('count', 1);
  const overflow = output('overflow', 1);
  const pairOverflow = output('pair-overflow', 1);
  const distanceBuffer = createInputBuffer(device, Float32Array.from([distances[0]]));
  buffers.push(distanceBuffer);
  graph.add(
    new GPUSpatialPredicateJoin({
      left,
      right,
      predicate: 'dwithin',
      distance: perFrame
        ? importGraphBuffer(graph, 'distance', distanceBuffer, 'float32', 1)
        : distances[0],
      spatialContext: GREAT_CIRCLE,
      candidateCapacity: Math.max(leftCount * 8, 64),
      output: {
        pairs: {
          leftIds: leftIds.view,
          rightIds: rightIds.view,
          status: {
            count: count.view,
            overflow: pairOverflow.view
          }
        },
        status: {overflow: overflow.view}
      }
    })
  );
  const compiled = graph.compile();
  const results: {pairs: [number, number][]; overflow: number}[] = [];
  for (const distance of distances) {
    distanceBuffer.write(Float32Array.from([distance]));
    submitGraph(device, compiled, undefined);
    const [countValue] = await readUint32(count.buffer, 1);
    const lefted = await readUint32(leftIds.buffer, pairCapacity);
    const righted = await readUint32(rightIds.buffer, pairCapacity);
    results.push({
      pairs: lefted.slice(0, countValue).map((row, slot) => [row, righted[slot]]),
      overflow: (await readUint32(overflow.buffer, 1))[0]
    });
    if (!perFrame) {
      break;
    }
  }
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return results;
}

const pointSides =
  (left: [number, number][], right: [number, number][]) =>
  (graph: GPUCommandGraph, buffers: Buffer[]) => ({
    left: createPoints(graph.device, graph, 'left', left, buffers),
    right: createPoints(graph.device, graph, 'right', right, buffers),
    leftCount: left.length
  });

it('GPUSpatialPredicateJoin great-circle dwithin matches haversine across latitude, antimeridian and pole', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {left, right, pairsWithin1500} = GEODESIC_DWITHIN_POINTS;
  const [result] = await runGreatCircleJoin(device, pointSides(left, right), [1500], false);
  expect(result.overflow).toBe(0);
  // The pole pair (2.22 km), the 54.8 km pair and every far pair are outside 1.5 km.
  expect(result.pairs).toEqual(pairsWithin1500);
});

it('GPUSpatialPredicateJoin great-circle dwithin reads a per-frame distance in meters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {left, right, pairsWithin1500, pairsWithin2500} = GEODESIC_DWITHIN_POINTS;
  const results = await runGreatCircleJoin(
    device,
    pointSides(left, right),
    [1500, 2500, 1500, 1000, -1],
    true
  );
  expect(results.map(result => result.pairs)).toEqual([
    pairsWithin1500,
    pairsWithin2500,
    pairsWithin1500,
    // 1000 m is below the 1.11 km pairs; only the 78.6 m pair remains.
    [[4, 4]],
    []
  ]);
});

it('GPUSpatialPredicateJoin great-circle dwithin measures to polygon edges and counts containment as zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {ring, queries, distances} = GEODESIC_DWITHIN_POLYGON;
  const sides = (graph: GPUCommandGraph, buffers: Buffer[]) => ({
    left: createPoints(graph.device, graph, 'queries', queries, buffers),
    right: createPolygon(graph.device, graph, ring, buffers),
    leftCount: queries.length
  });
  for (const limit of [1000, 2000]) {
    const [result] = await runGreatCircleJoin(device, sides, [limit], false);
    const expected = distances
      .map((distance, row) => [row, distance] as const)
      .filter(([, distance]) => distance <= limit)
      .map(([row]) => [row, 0]);
    expect(result.pairs, `within ${limit} m`).toEqual(expected);
  }
});
