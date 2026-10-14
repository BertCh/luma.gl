// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUDelaunayTessellation,
  GPUVoronoiDiagram
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  checkTriangulation,
  createUniformFixture,
  digestTriangleSet,
  SCIPY_FIXTURES,
  toTriangleRows
} from './delaunay-reference';

async function runDelaunay(device: Device, coordinates: Float32Array, triangleCapacity: number) {
  const graph = new GPUCommandGraph(device, {id: 'delaunay-bounded-test'});
  const buffers: Buffer[] = [];
  const positionsBuffer = createInputBuffer(device, coordinates);
  const triangleBuffer = createOutputBuffer(device, 3 * triangleCapacity);
  buffers.push(positionsBuffer, triangleBuffer);
  const scalar = (name: string) => {
    const buffer = createOutputBuffer(device, 1);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', 1)};
  };
  const count = scalar('bounded-count');
  const requiredCount = scalar('bounded-required-count');
  const overflow = scalar('bounded-overflow');
  const invalidCount = scalar('bounded-invalid-count');
  const duplicateCount = scalar('bounded-duplicate-count');
  graph.add(
    new GPUDelaunayTessellation({
      positions: importGraphBuffer(
        graph,
        'bounded-positions',
        positionsBuffer,
        'float32x2',
        coordinates.length / 2
      ),
      output: {
        triangles: importGraphBuffer(
          graph,
          'bounded-triangles',
          triangleBuffer,
          'uint32x3',
          triangleCapacity
        ),
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          invalidCount: invalidCount.view
        },
        duplicateCount: duplicateCount.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [written, required, didOverflow, invalid, duplicates] = await Promise.all([
    readUint32(count.buffer, 1).then(values => values[0]),
    readUint32(requiredCount.buffer, 1).then(values => values[0]),
    readUint32(overflow.buffer, 1).then(values => values[0]),
    readUint32(invalidCount.buffer, 1).then(values => values[0]),
    readUint32(duplicateCount.buffer, 1).then(values => values[0])
  ]);
  const indices = await readUint32(triangleBuffer, 3 * written);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {written, required, didOverflow, invalid, duplicates, indices};
}

it('GPUDelaunayTessellation triangulates unique sites and reports bounded cardinality', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const coordinates = Float32Array.from([
    0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5,
    // First-row ownership makes this final row a duplicate rather than another site.
    0, 0
  ]);
  const graph = new GPUCommandGraph(device, {id: 'delaunay-test'});
  const buffers: Buffer[] = [];
  const inputBuffer = createInputBuffer(device, coordinates);
  buffers.push(inputBuffer);
  const positions = importGraphBuffer(graph, 'positions', inputBuffer, 'float32x2', 6);
  const triangleBuffer = createOutputBuffer(device, 3 * 8);
  buffers.push(triangleBuffer);
  const triangles = importGraphBuffer(graph, 'triangles', triangleBuffer, 'uint32x3', 8);
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
      positions,
      output: {
        triangles,
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          invalidCount: invalidCount.view
        },
        duplicateCount: duplicateCount.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [written, required, didOverflow, invalid, duplicates] = await Promise.all([
    readUint32(count.buffer, 1).then(values => values[0]),
    readUint32(requiredCount.buffer, 1).then(values => values[0]),
    readUint32(overflow.buffer, 1).then(values => values[0]),
    readUint32(invalidCount.buffer, 1).then(values => values[0]),
    readUint32(duplicateCount.buffer, 1).then(values => values[0])
  ]);
  expect({written, required, didOverflow, invalid, duplicates}).toEqual({
    written: 4,
    required: 4,
    didOverflow: 0,
    invalid: 0,
    duplicates: 1
  });
  const indices = await readUint32(triangleBuffer, 3 * written);
  expect(Math.max(...indices)).toBeLessThan(5);
  let doubledArea = 0;
  for (let triangle = 0; triangle < written; triangle++) {
    const [a, b, c] = indices.slice(3 * triangle, 3 * triangle + 3);
    const ax = coordinates[2 * a];
    const ay = coordinates[2 * a + 1];
    const bx = coordinates[2 * b];
    const by = coordinates[2 * b + 1];
    const cx = coordinates[2 * c];
    const cy = coordinates[2 * c + 1];
    doubledArea += (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  }
  expect(doubledArea).toBeCloseTo(2);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

it('GPUDelaunayTessellation reports collinear, cocircular, and capacity edge cases deterministically', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const collinear = await runDelaunay(device, Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0), 4);
  expect(collinear).toMatchObject({
    written: 0,
    required: 0,
    didOverflow: 0,
    invalid: 0,
    duplicates: 0
  });

  const cocircularCoordinates = Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1);
  const first = await runDelaunay(device, cocircularCoordinates, 4);
  const second = await runDelaunay(device, cocircularCoordinates, 4);
  expect(first).toMatchObject({written: 2, required: 2, didOverflow: 0, invalid: 0});
  expect(second.indices).toEqual(first.indices);

  // Sort-based duplicate detection: -0 equals +0, non-finite rows are invalid and never owners.
  const duplicates = await runDelaunay(
    device,
    Float32Array.of(0, 0, 1, 0, 1, 1, -0, 0, Number.NaN, 0, 0, 1, 1, 0),
    8
  );
  expect(duplicates).toMatchObject({invalid: 1, duplicates: 2});

  const bounded = await runDelaunay(device, Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5), 2);
  expect(bounded).toMatchObject({written: 2, required: 4, didOverflow: 1, invalid: 0});
});

// Symbolic super vertices (finding D-01): the finite supertriangle lost real hull triangles on
// ordinary input. Expected counts, hull counts and triangle-set digests are scipy.spatial 1.18.1
// constants from scratchpad/fix/work/W2A/delaunay/oracle.py; emptiness is checked in exact
// rational arithmetic on the f32 inputs, so no assertion depends on tie-breaking.
for (const [name, fixture] of Object.entries(SCIPY_FIXTURES)) {
  it(`GPUDelaunayTessellation keeps every hull triangle on ${name} (T = 2n - 2 - h)`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const coordinates = createUniformFixture(
      fixture.seed,
      fixture.count,
      fixture.width,
      fixture.height
    );
    const result = await runDelaunay(device, coordinates, 2 * fixture.count);
    expect(result).toMatchObject({
      written: 2 * fixture.count - 2 - fixture.hull,
      required: fixture.triangles,
      didOverflow: 0,
      invalid: 0,
      duplicates: 0
    });
    const triangles = toTriangleRows(result.indices);
    expect(digestTriangleSet(triangles)).toBe(fixture.digest);
    expect(checkTriangulation(coordinates, triangles)).toEqual({
      degenerate: 0,
      circleViolations: 0,
      duplicateEdges: 0,
      areaDifference: 0n
    });
  });
}

it('GPUDelaunayTessellation triangulates cocircular and hull-collinear sites without losing area', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cases: {name: string; points: number[][]; triangles: number}[] = [
    {
      // Every site is a hull vertex: T = 2n - 2 - n = 10 whatever diagonals the ties select.
      name: 'twelve cocircular sites',
      points: [
        [5, 0],
        [4, 3],
        [3, 4],
        [0, 5],
        [-3, 4],
        [-4, 3],
        [-5, 0],
        [-4, -3],
        [-3, -4],
        [0, -5],
        [3, -4],
        [4, -3]
      ],
      triangles: 10
    },
    {
      // Boundary points b = 8 (corners and midpoints), n = 9: T = 2n - 2 - b = 8; scipy agrees.
      name: 'square with edge midpoints, corners first',
      points: [
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
        [1, 1],
        [1, 0],
        [2, 1],
        [1, 2],
        [0, 1]
      ],
      triangles: 8
    },
    {
      name: 'square with edge midpoints, midpoints first',
      points: [
        [1, 0],
        [2, 1],
        [1, 2],
        [0, 1],
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
        [1, 1]
      ],
      triangles: 8
    },
    {
      // Boundary points 16, n = 25: T = 32; scipy agrees.
      name: '5x5 lattice',
      points: Array.from({length: 25}, (_, index) => [index % 5, Math.floor(index / 5)]),
      triangles: 32
    }
  ];
  for (const {name, points, triangles: expected} of cases) {
    const coordinates = Float32Array.from(points.flat());
    const result = await runDelaunay(device, coordinates, 2 * points.length);
    expect({
      name,
      written: result.written,
      required: result.required,
      overflow: result.didOverflow
    }).toEqual({
      name,
      written: expected,
      required: expected,
      overflow: 0
    });
    expect({name, ...checkTriangulation(coordinates, toTriangleRows(result.indices))}).toEqual({
      name,
      degenerate: 0,
      circleViolations: 0,
      duplicateEdges: 0,
      areaDifference: 0n
    });
  }
});

it('GPUVoronoiDiagram derives compact internal and clipped hull edges', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'voronoi-test'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'float32x4' | 'uint32x3' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const positions = input(
    'positions',
    'float32x2',
    Float32Array.from([0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5]),
    5
  );
  const triangles = input(
    'triangles',
    'uint32x3',
    Uint32Array.from([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4]),
    4
  );
  const triangleCount = input('triangle-count', 'uint32', Uint32Array.of(4), 1);
  const clipBounds = input('clip-bounds', 'float32x4', Float32Array.of(-1, -1, 2, 2), 1);
  const output = <Format extends 'float32x4' | 'uint32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const components = format === 'float32x4' ? 4 : format === 'uint32x2' ? 2 : 1;
    const buffer = createOutputBuffer(device, components * length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const segments = output('segments', 'float32x4', 8);
  const siteIds = output('site-ids', 'uint32x2', 8);
  const count = output('count', 'uint32', 1);
  const requiredCount = output('required-count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const invalidCount = output('invalid-count', 'uint32', 1);
  graph.add(
    new GPUVoronoiDiagram({
      positions,
      triangles,
      triangleCount,
      clipBounds,
      output: {
        segments: segments.view,
        siteIds: siteIds.view,
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          invalidCount: invalidCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(count.buffer, 1)).toEqual([8]);
  expect(await readUint32(requiredCount.buffer, 1)).toEqual([8]);
  expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
  expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
  const pairs = await readUint32(siteIds.buffer, 16);
  expect(
    new Set(Array.from({length: 8}, (_, row) => `${pairs[2 * row]}:${pairs[2 * row + 1]}`))
  ).toEqual(new Set(['0:1', '1:4', '0:4', '1:2', '2:4', '2:3', '3:4', '0:3']));
  // Compact order is candidate (triangle, local edge) order; internal edges belong to the lower row.
  expect(Array.from({length: 8}, (_, row) => `${pairs[2 * row]}:${pairs[2 * row + 1]}`)).toEqual([
    '0:1',
    '1:4',
    '0:4',
    '1:2',
    '2:4',
    '2:3',
    '3:4',
    '0:3'
  ]);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
