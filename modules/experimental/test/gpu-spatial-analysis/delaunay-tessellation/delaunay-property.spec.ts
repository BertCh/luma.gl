// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {checkTriangulation, toTriangleRows} from './delaunay-reference';
import fixtureJson from './fixtures/delaunay-properties.json';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: fixtures/delaunay-properties.json, written by test/oracles/geometry-properties.py with
// scipy.spatial.Delaunay and ConvexHull on three seeded float32 point sets (uniform 300, clustered
// 300, thin band 200 at aspect 1e3). Properties checked: T = 2n - 2 - h, set equality with scipy
// for sets in strict general position, and the empty-circle property with exact BigInt arithmetic
// on the float32 inputs (checkTriangulation).

type SiteSet = {
  name: string;
  coordinates: number[];
  siteCount: number;
  hullCount: number;
  triangleCount: number;
  generalPosition: boolean;
  triangles: number[][] | null;
};

const fixture: {sets: SiteSet[]} = fixtureJson as unknown as {sets: SiteSet[]};

async function runDelaunay(device: Device, coordinates: Float32Array, triangleCapacity: number) {
  const graph = new GPUCommandGraph(device, {id: 'delaunay-property'});
  const buffers: Buffer[] = [];
  const positionsBuffer = createInputBuffer(device, coordinates);
  const triangleBuffer = createOutputBuffer(device, 3 * triangleCapacity);
  buffers.push(positionsBuffer, triangleBuffer);
  const scalar = (name: string) => {
    const buffer = createOutputBuffer(device, 1);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', 1)};
  };
  const count = scalar('count');
  const requiredCount = scalar('required-count');
  const overflow = scalar('overflow');
  const invalidCount = scalar('invalid-count');
  const duplicateCount = scalar('duplicate-count');
  graph.add(
    new GPUDelaunayTessellation({
      positions: importGraphBuffer(
        graph,
        'positions',
        positionsBuffer,
        'float32x2',
        coordinates.length / 2
      ),
      output: {
        triangles: importGraphBuffer(
          graph,
          'triangles',
          triangleBuffer,
          'uint32x3',
          triangleCapacity
        ),
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          invalidCount: invalidCount.view,
          duplicateCount: duplicateCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const written = (await readUint32(count.buffer, 1))[0];
  const result = {
    written,
    required: (await readUint32(requiredCount.buffer, 1))[0],
    didOverflow: (await readUint32(overflow.buffer, 1))[0],
    invalid: (await readUint32(invalidCount.buffer, 1))[0],
    duplicates: (await readUint32(duplicateCount.buffer, 1))[0],
    indices: await readUint32(triangleBuffer, 3 * written)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUDelaunayTessellation satisfies T = 2n - 2 - h, scipy set equality and the empty circle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expect(fixture.sets.map(set => set.name)).toEqual([
    'uniform-300',
    'clustered-300',
    'thin-band-200-aspect-1e3'
  ]);
  for (const set of fixture.sets) {
    const coordinates = Float32Array.from(set.coordinates);
    // The fixture states Euler's formula for scipy's own answer.
    expect(set.triangleCount, `${set.name} Euler`).toBe(2 * set.siteCount - 2 - set.hullCount);
    const actual = await runDelaunay(device, coordinates, set.triangleCount + 16);
    expect(actual.didOverflow, `${set.name} overflow`).toBe(0);
    expect(actual.invalid, `${set.name} invalid`).toBe(0);
    expect(actual.duplicates, `${set.name} duplicates`).toBe(0);
    expect(actual.required, `${set.name} required`).toBe(set.triangleCount);
    expect(actual.written, `${set.name} triangle count`).toBe(set.triangleCount);
    const triangles = toTriangleRows(actual.indices);
    // Exact empty-circle, orientation, tiling and manifold checks on the float32 inputs.
    const check = checkTriangulation(coordinates, triangles);
    expect(check.degenerate, `${set.name} degenerate`).toBe(0);
    expect(check.circleViolations, `${set.name} empty circle`).toBe(0);
    expect(check.duplicateEdges, `${set.name} duplicate edges`).toBe(0);
    expect(check.areaDifference, `${set.name} tiles the hull`).toBe(0n);
    if (set.generalPosition && set.triangles) {
      const canonical = (rows: number[][]) =>
        rows
          .map(row => [...row].sort((a, b) => a - b).join(','))
          .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      expect(canonical(triangles), `${set.name} triangle set`).toEqual(canonical(set.triangles));
    }
  }
  device.destroy?.();
});
