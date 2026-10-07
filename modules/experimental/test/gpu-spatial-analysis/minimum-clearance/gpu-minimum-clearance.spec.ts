// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUMinimumClearance} from '../../../src/gpu-spatial-analysis/minimum-clearance/index';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSegmentGeometry} from '../segment-intersection/segment-geometry-harness';
import type {OracleSegmentFeature} from '../segment-intersection/segment-intersection-oracle';
import {SHAPELY_CLEARANCE_LINES, SHAPELY_CLEARANCE_POLYGONS} from './shapely-clearance-fixture';

type Point = [number, number];
type ClearanceResult = {clearances: number[]; lines: number[]; vertexIds: number[]};

const NO_VERTEX = 0xffffffff;

async function runClearance(
  device: Device,
  features: OracleSegmentFeature[],
  options: {spatialSort?: boolean} = {}
): Promise<ClearanceResult> {
  const graph = new GPUCommandGraph(device, {id: 'minimum-clearance'});
  const buffers: Buffer[] = [];
  const geometry = createSegmentGeometry(device, graph, 'geometry', features, buffers);
  const output = (name: string, format: 'float32' | 'float32x4' | 'uint32', words: number) => {
    const buffer = createOutputBuffer(device, features.length * words);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, features.length)};
  };
  const clearances = output('clearances', 'float32', 1);
  const lines = output('lines', 'float32x4', 4);
  const vertexIds = output('vertex-ids', 'uint32', 1);
  graph.add(
    new GPUMinimumClearance({
      geometry,
      clearances: clearances.view,
      lines: lines.view,
      vertexIds: vertexIds.view,
      spatialSort: options.spatialSort
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    clearances: await readFloat32(clearances.buffer, features.length),
    lines: await readFloat32(lines.buffer, features.length * 4),
    vertexIds: await readUint32(vertexIds.buffer, features.length)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectClearance(actual: number, expected: number | null, label: string, tolerance = 2e-5) {
  if (expected === null) {
    expect(actual, label).toBe(Number.POSITIVE_INFINITY);
  } else {
    expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
      tolerance * Math.max(1, expected)
    );
  }
}

const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Shapely entries whose minimizing pair is unique, so the reported line must match as a set. */
const UNIQUE_PAIR_NAMES = new Set([
  'spike_triangle',
  'obtuse_sliver',
  'triangle',
  'hole_near_shell',
  'u_turn_t'
]);

function expectLines(
  result: ClearanceResult,
  entries: readonly {name: string; clearance: number | null; line: [number, number][] | null}[]
) {
  entries.forEach((entry, row) => {
    const line = result.lines.slice(row * 4, row * 4 + 4);
    if (entry.line === null) {
      expect(line.every(Number.isNaN), `${entry.name} empty line`).toBe(true);
      expect(result.vertexIds[row], `${entry.name} vertex id`).toBe(NO_VERTEX);
      return;
    }
    // Same length as the clearance, whichever of the tied pairs was chosen.
    expect(
      distance([line[0], line[1]], [line[2], line[3]]),
      `${entry.name} line length`
    ).toBeCloseTo(entry.clearance as number, 4);
    if (UNIQUE_PAIR_NAMES.has(entry.name)) {
      for (const [index, point] of entry.line.entries()) {
        const other = entry.line[1 - index];
        // Shapely reports (fragile vertex, nearest point); compare as an unordered pair.
        const matchesFirst =
          distance([line[0], line[1]], point) < 1e-4 && distance([line[2], line[3]], other) < 1e-4;
        const matchesSecond =
          distance([line[2], line[3]], point) < 1e-4 && distance([line[0], line[1]], other) < 1e-4;
        expect(matchesFirst || matchesSecond, `${entry.name} line endpoints`).toBe(true);
      }
    }
  });
}

it('GPUMinimumClearance matches Shapely minimum_clearance on polygons', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features: OracleSegmentFeature[] = SHAPELY_CLEARANCE_POLYGONS.map(entry => ({
    kind: 'polygons',
    polygons: entry.shape
  }));
  const result = await runClearance(device, features);
  SHAPELY_CLEARANCE_POLYGONS.forEach((entry, row) => {
    expectClearance(result.clearances[row], entry.clearance, entry.name);
  });
  expectLines(result, SHAPELY_CLEARANCE_POLYGONS);
  // The fragile vertex row points at the first line endpoint.
  const row = SHAPELY_CLEARANCE_POLYGONS.findIndex(entry => entry.name === 'spike_triangle');
  expect(result.vertexIds[row]).toBeLessThan(0xffffffff);
});

it('GPUMinimumClearance matches Shapely minimum_clearance on linestrings', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features: OracleSegmentFeature[] = SHAPELY_CLEARANCE_LINES.map(entry => ({
    kind: 'lines',
    vertices: entry.shape
  }));
  const result = await runClearance(device, features);
  SHAPELY_CLEARANCE_LINES.forEach((entry, row) => {
    expectClearance(result.clearances[row], entry.clearance, entry.name);
  });
  expectLines(result, SHAPELY_CLEARANCE_LINES);
  const names = SHAPELY_CLEARANCE_LINES.map(entry => entry.name);
  // Infinity cases: a degenerate pair of equal points and an empty feature.
  expect(result.clearances[names.indexOf('degenerate_pair')]).toBe(Number.POSITIVE_INFINITY);
  expect(result.clearances[names.indexOf('empty')]).toBe(Number.POSITIVE_INFINITY);
});

it('GPUMinimumClearance is identical with and without the Morton sort on a large scene', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Tile the polygon fixture along x (f32 rounding of the shifted coordinates sets the 4e-4 tolerance) so the scene exceeds 1024 vertices and features interleave.
  const features: OracleSegmentFeature[] = [];
  const expected: (number | null)[] = [];
  for (let tile = 0; tile < 12; tile++) {
    for (const entry of SHAPELY_CLEARANCE_POLYGONS) {
      features.push({
        kind: 'polygons',
        polygons: entry.shape.map(polygon =>
          polygon.map(ring => ring.map(([x, y]): Point => [x + tile * 256, y]))
        )
      });
      expected.push(entry.clearance);
    }
  }
  const start = performance.now();
  const sorted = await runClearance(device, features, {spatialSort: true});
  const sortedMilliseconds = performance.now() - start;
  const unsorted = await runClearance(device, features, {spatialSort: false});
  console.log(
    `GPUMinimumClearance ${features.length} features: ${sortedMilliseconds.toFixed(1)} ms (incl. compile)`
  );
  expect(sorted.clearances).toEqual(unsorted.clearances);
  expect(sorted.vertexIds).toEqual(unsorted.vertexIds);
  expected.forEach((value, row) => {
    expectClearance(sorted.clearances[row], value, `feature ${row}`, 4e-4);
  });
});

it('GPUMinimumClearance validates properties', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'minimum-clearance-validation'});
  const buffers: Buffer[] = [];
  const geometry = createSegmentGeometry(
    device,
    graph,
    'geometry',
    [
      {
        kind: 'lines',
        vertices: [
          [0, 0],
          [1, 1]
        ]
      }
    ],
    buffers
  );
  expect(() => new GPUMinimumClearance({geometry})).toThrow(/at least one of/);
  const wrong = createOutputBuffer(device, 4);
  buffers.push(wrong);
  const clearances = importGraphBuffer(graph, 'wrong', wrong, 'float32', 3);
  expect(() => new GPUMinimumClearance({geometry, clearances})).toThrow(/feature count/);
  expect(
    () =>
      new GPUMinimumClearance({
        geometry,
        clearances: undefined,
        leafCapacity: 3,
        vertexIds: undefined
      })
  ).toThrow();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

it('GPUMinimumClearance builds on a core-limits device', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const features: OracleSegmentFeature[] = SHAPELY_CLEARANCE_POLYGONS.slice(0, 3).map(entry => ({
    kind: 'polygons',
    polygons: entry.shape
  }));
  const result = await runClearance(device, features);
  expectClearance(result.clearances[0], SHAPELY_CLEARANCE_POLYGONS[0].clearance, 'square');
});
