// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_POINT_IN_POLYGON_CLASSIFICATION,
  GPUPairwisePointInPolygon
} from '../../../src/geospatial';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom, orientExact} from '../segment-intersection/segment-intersection-oracle';

type Point = [number, number];

function stepFloat32(value: number, steps: number): number {
  const floats = new Float32Array([value]);
  const bits = new Int32Array(floats.buffer);
  bits[0] += value >= 0 ? steps : -steps;
  return floats[0];
}

/** Exact point in simple polygon (one ring) classification using BigInt orientation. */
function classifyExact(point: Point, ring: Point[]): 'inside' | 'boundary' | 'outside' {
  let inside = false;
  for (let index = 0; index < ring.length; index++) {
    const start = ring[index];
    const end = ring[(index + 1) % ring.length];
    const sign = orientExact(start, end, point);
    if (
      sign === 0 &&
      point[0] >= Math.min(start[0], end[0]) &&
      point[0] <= Math.max(start[0], end[0]) &&
      point[1] >= Math.min(start[1], end[1]) &&
      point[1] <= Math.max(start[1], end[1])
    ) {
      return 'boundary';
    }
    if (start[1] > point[1] !== end[1] > point[1]) {
      const upward = end[1] > start[1];
      if (upward ? sign > 0 : sign < 0) inside = !inside;
    }
  }
  return inside ? 'inside' : 'outside';
}

/** Near-degenerate rows: a point 0..3 ulps off the middle of a random triangle edge. */
function makeRows(seed: number, offset: number, scale: number, count: number) {
  const random = createRandom(seed);
  const rows: {point: Point; ring: Point[]}[] = [];
  const f = (value: number) => Math.fround(value);
  while (rows.length < count) {
    const ring: Point[] = [0, 1, 2].map(
      () => [f(offset + random() * scale), f(offset + random() * scale)] as Point
    );
    const edge = Math.floor(random() * 3);
    const [a, b] = [ring[edge], ring[(edge + 1) % 3]];
    const t = random();
    const base: Point = [f(a[0] + (b[0] - a[0]) * t), f(a[1] + (b[1] - a[1]) * t)];
    const point: Point = [
      stepFloat32(base[0], Math.floor(random() * 7) - 3),
      stepFloat32(base[1], Math.floor(random() * 7) - 3)
    ];
    rows.push({point, ring});
  }
  return rows;
}

async function runProbe(offset: number, scale: number, seed: number) {
  const device = await getWebGPUTestDevice();
  if (!device) return null;
  const rows = makeRows(seed, offset, scale, 6000);
  const positions = rows.flatMap(row => row.ring.flat());
  const graph = new GPUCommandGraph(device, {id: 'pip-exact-probe'});
  const count = rows.length;
  const geometryOffsets = Uint32Array.from({length: count + 1}, (_, i) => i);
  const ringOffsets = Uint32Array.from({length: count + 1}, (_, i) => i * 3);
  const buffers = [
    createInputBuffer(device, Float32Array.from(rows.flatMap(row => row.point))),
    createInputBuffer(device, Float32Array.from(positions)),
    createInputBuffer(device, geometryOffsets),
    createInputBuffer(device, geometryOffsets),
    createInputBuffer(device, ringOffsets),
    createOutputBuffer(device, count)
  ];
  new GPUPairwisePointInPolygon({
    points: importGraphBuffer(graph, 'points', buffers[0], 'float32x2', count),
    polygonPositions: importGraphBuffer(graph, 'pos', buffers[1], 'float32x2', count * 3),
    geometryOffsets: importGraphBuffer(graph, 'go', buffers[2], 'uint32', count + 1),
    polygonOffsets: importGraphBuffer(graph, 'po', buffers[3], 'uint32', count + 1),
    ringOffsets: importGraphBuffer(graph, 'ro', buffers[4], 'uint32', count + 1),
    output: importGraphBuffer(graph, 'out', buffers[5], 'uint32', count)
  }).addToGraph(graph);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = await readUint32(buffers[5], count);
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  const {outside, inside, boundary, uncertain} = GPU_POINT_IN_POLYGON_CLASSIFICATION;
  const names = {[outside]: 'outside', [inside]: 'inside', [boundary]: 'boundary'} as const;
  const tally = {wrong: 0, uncertain: 0, exactBoundary: 0, correct: 0};
  const examples: unknown[] = [];
  rows.forEach((row, index) => {
    const expected = classifyExact(row.point, row.ring);
    if (expected === 'boundary') tally.exactBoundary++;
    if (result[index] === uncertain) tally.uncertain++;
    else if (names[result[index] as 0 | 1 | 2] !== expected) {
      tally.wrong++;
      if (examples.length < 3) examples.push({row, expected, got: result[index]});
    } else tally.correct++;
  });
  return {tally, examples};
}

for (const [label, offset, scale] of [
  ['small coordinates', 0, 100],
  ['offset coordinates', 12345, 50],
  ['wide coordinates', -500, 1000]
] as const) {
  it(`GPUPairwisePointInPolygon matches the exact oracle on near-degenerate points (${label})`, async () => {
    const outcome = await runProbe(offset, scale, 17 + offset);
    if (!outcome) return;
    console.log(`PIP probe ${label}`, JSON.stringify(outcome));
    expect(
      outcome.tally.exactBoundary + outcome.tally.correct + outcome.tally.uncertain
    ).toBeGreaterThan(5000);
    expect(outcome.tally.wrong).toBe(0);
  });
}

it('GPUPointInPolygonJoin resolves near-degenerate points exactly with no uncertainty', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const totals = {points: 0, inside: 0, mismatches: 0, uncertain: 0};
  for (const [offset, scale, seed] of [
    [0, 100, 3],
    [12345, 50, 4],
    [-500, 1000, 5]
  ] as const) {
    // One triangle per run, so every near-edge point's answer is decided by that triangle alone.
    const rows = makeRows(seed, offset, scale, 12);
    for (const [triangleIndex, {ring}] of rows.entries()) {
      const random = createRandom(seed * 100 + triangleIndex);
      const points: Point[] = [];
      for (let index = 0; index < 150; index++) {
        const edge = Math.floor(random() * 3);
        const [a, b] = [ring[edge], ring[(edge + 1) % 3]];
        const t = random();
        const base: Point = [
          Math.fround(a[0] + (b[0] - a[0]) * t),
          Math.fround(a[1] + (b[1] - a[1]) * t)
        ];
        points.push([
          stepFloat32(base[0], Math.floor(random() * 7) - 3),
          stepFloat32(base[1], Math.floor(random() * 7) - 3)
        ]);
      }
      const count = points.length;
      const graph = new GPUCommandGraph(device, {id: 'pip-join-exact'});
      const buffers = {
        points: createInputBuffer(device, Float32Array.from(points.flat())),
        positions: createInputBuffer(device, Float32Array.from(ring.flat())),
        featureOffsets: createInputBuffer(device, Uint32Array.from([0, 1])),
        polygonOffsets: createInputBuffer(device, Uint32Array.from([0, 1])),
        ringOffsets: createInputBuffer(device, Uint32Array.from([0, 3])),
        pointFeatureIds: createOutputBuffer(device, count),
        featureCounts: createOutputBuffer(device, 1),
        overflow: createOutputBuffer(device, 1),
        uncertainCount: createOutputBuffer(device, 1)
      };
      graph.add(
        new GPUPointInPolygonJoin({
          points: importGraphBuffer(graph, 'points', buffers.points, 'float32x2', count),
          polygonPositions: importGraphBuffer(graph, 'pos', buffers.positions, 'float32x2', 3),
          featureOffsets: importGraphBuffer(graph, 'fo', buffers.featureOffsets, 'uint32', 2),
          polygonOffsets: importGraphBuffer(graph, 'po', buffers.polygonOffsets, 'uint32', 2),
          ringOffsets: importGraphBuffer(graph, 'ro', buffers.ringOffsets, 'uint32', 2),
          candidateCapacity: count,
          pointFeatureIds: importGraphBuffer(graph, 'pf', buffers.pointFeatureIds, 'uint32', count),
          featureCounts: importGraphBuffer(graph, 'fc', buffers.featureCounts, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'of', buffers.overflow, 'uint32', 1),
          uncertainCount: importGraphBuffer(graph, 'uc', buffers.uncertainCount, 'uint32', 1),
          includeBoundary: true
        })
      );
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const assigned = await readUint32(buffers.pointFeatureIds, count);
      totals.uncertain += (await readUint32(buffers.uncertainCount, 1))[0];
      points.forEach((point, index) => {
        const expectedInside = classifyExact(point, ring) !== 'outside';
        totals.points++;
        if (expectedInside) totals.inside++;
        if ((assigned[index] === 0) !== expectedInside) totals.mismatches++;
      });
      compiled.destroy();
      Object.values(buffers).forEach(buffer => buffer.destroy());
    }
  }
  console.log('PIP join exact', JSON.stringify(totals));
  expect(totals.inside).toBeGreaterThan(totals.points / 5);
  expect(totals.inside).toBeLessThan((totals.points * 4) / 5);
  expect(totals.uncertain).toBe(0);
  expect(totals.mismatches).toBe(0);
});
