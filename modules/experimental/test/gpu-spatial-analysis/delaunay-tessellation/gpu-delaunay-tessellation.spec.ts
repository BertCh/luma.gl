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
  checkLocallyDelaunay,
  checkTriangulation,
  countHullBoundarySites,
  createLatticeFixture,
  createUniformFixture,
  digestTriangleSet,
  SCIPY_FIXTURES,
  toTriangleRows
} from './delaunay-reference';
import {requireTestDevice} from '../../utils/require-test-device';

async function runDelaunay(
  device: Device,
  coordinates: Float32Array,
  triangleCapacity: number,
  maximumParallelSteps?: number
) {
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
  const converged = scalar('converged');
  const iterationCount = scalar('iteration-count');
  const iterationLimitReached = scalar('iteration-limit-reached');
  const contributor = new GPUDelaunayTessellation({
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
        invalidCount: invalidCount.view,
        duplicateCount: duplicateCount.view,
        converged: converged.view,
        iterationCount: iterationCount.view,
        iterationLimitReached: iterationLimitReached.view
      }
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  graph.add(
    maximumParallelSteps === undefined
      ? nodes
      : nodes.filter(node => {
          const step = /-step-(\d+)-/.exec(node.id);
          return !step || Number(step[1]) < maximumParallelSteps;
        })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [
    written,
    required,
    didOverflow,
    invalid,
    duplicates,
    didConverge,
    iterations,
    limitReached
  ] = await Promise.all([
    readUint32(count.buffer, 1).then(values => values[0]),
    readUint32(requiredCount.buffer, 1).then(values => values[0]),
    readUint32(overflow.buffer, 1).then(values => values[0]),
    readUint32(invalidCount.buffer, 1).then(values => values[0]),
    readUint32(duplicateCount.buffer, 1).then(values => values[0]),
    readUint32(converged.buffer, 1).then(values => values[0]),
    readUint32(iterationCount.buffer, 1).then(values => values[0]),
    readUint32(iterationLimitReached.buffer, 1).then(values => values[0])
  ]);
  const indices = await readUint32(triangleBuffer, 3 * written);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {
    written,
    required,
    didOverflow,
    invalid,
    duplicates,
    indices,
    didConverge,
    iterations,
    limitReached
  };
}

it('GPUDelaunayTessellation finishes the exact GPU continuation after seed insertion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of Object.values(SCIPY_FIXTURES)) {
    const coordinates = createUniformFixture(
      fixture.seed,
      fixture.count,
      fixture.width,
      fixture.height
    );
    const result = await runDelaunay(device, coordinates, 2 * fixture.count, 3);
    expect(result).toMatchObject({
      written: fixture.triangles,
      invalid: 0,
      didConverge: 1,
      limitReached: 0
    });
    expect(result.iterations).toBeGreaterThan(3);
    expect(digestTriangleSet(toTriangleRows(result.indices))).toBe(fixture.digest);
  }
}, 600000);

it('GPUDelaunayTessellation triangulates unique sites and reports bounded cardinality', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
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
          invalidCount: invalidCount.view,
          duplicateCount: duplicateCount.view
        }
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
  requireTestDevice(device);
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
    requireTestDevice(device);
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
  requireTestDevice(device);
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
  requireTestDevice(device);
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

// Parallel insertion and flipping at scale (DEL-REF additions). The output order is canonical:
// every row is rotated so its smallest vertex comes first, and rows ascend by (v0, v1).
function expectCanonicalOrder(triangles: number[][]) {
  for (let index = 0; index < triangles.length; index++) {
    const [a, b, c] = triangles[index];
    expect(a < b && a < c, `row ${index} starts with its smallest vertex`).toBe(true);
    if (index > 0) {
      const [previousA, previousB] = triangles[index - 1];
      expect(previousA < a || (previousA === a && previousB < b), `row ${index} ascends`).toBe(
        true
      );
    }
  }
}

it('GPUDelaunayTessellation triangulates 1e4 uniform sites (T = 2n - 2 - b, exact empty circle)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 10000;
  const coordinates = createUniformFixture(20261014, count, 100, 100);
  const expected = 2 * count - 2 - countHullBoundarySites(coordinates);
  const result = await runDelaunay(device, coordinates, 2 * count);
  expect(result).toMatchObject({
    written: expected,
    required: expected,
    didOverflow: 0,
    invalid: 0,
    duplicates: 0
  });
  const triangles = toTriangleRows(result.indices);
  expectCanonicalOrder(triangles);
  expect(checkTriangulation(coordinates, triangles)).toEqual({
    degenerate: 0,
    circleViolations: 0,
    duplicateEdges: 0,
    areaDifference: 0n
  });
}, 600_000);

it('GPUDelaunayTessellation triangulates a 64x64 lattice (cocircular ties by row order)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const coordinates = createLatticeFixture(64 * 64, 64);
  // Boundary sites 4 * 63 = 252: T = 2 * 4096 - 2 - 252.
  const expected = 2 * 64 * 64 - 2 - 252;
  const result = await runDelaunay(device, coordinates, 2 * 64 * 64);
  expect(result).toMatchObject({written: expected, required: expected, didOverflow: 0, invalid: 0});
  const triangles = toTriangleRows(result.indices);
  expectCanonicalOrder(triangles);
  expect(checkLocallyDelaunay(coordinates, triangles)).toEqual({
    degenerate: 0,
    circleViolations: 0,
    duplicateEdges: 0,
    areaDifference: 0n
  });
}, 600_000);

it('GPUDelaunayTessellation triangulates 1e5 uniform sites deterministically', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 100000;
  const coordinates = createUniformFixture(20261015, count, 100, 100);
  const expected = 2 * count - 2 - countHullBoundarySites(coordinates);
  const first = await runDelaunay(device, coordinates, 2 * count);
  expect(first).toMatchObject({written: expected, required: expected, didOverflow: 0, invalid: 0});
  const triangles = toTriangleRows(first.indices);
  expectCanonicalOrder(triangles);
  expect(checkLocallyDelaunay(coordinates, triangles)).toEqual({
    degenerate: 0,
    circleViolations: 0,
    duplicateEdges: 0,
    areaDifference: 0n
  });
  // Slot allocation follows atomic order, but the canonical output must not.
  const second = await runDelaunay(device, coordinates, 2 * count);
  expect(second.indices).toEqual(first.indices);
}, 900_000);

// The CPU WGSL executor runs this same scene; this spec additionally checks real indirect dispatch.
it('GPUDelaunayTessellation converges on 2000 collinear-plus-apex sites', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pointCount = 2000;
  const coordinates = Float32Array.from(
    Array.from({length: pointCount}, (_, row) =>
      row === pointCount - 1 ? [pointCount / 2, 1] : [row, 0]
    ).flat()
  );
  const result = await runDelaunay(device, coordinates, 2 * pointCount);
  expect(result).toMatchObject({
    written: pointCount - 2,
    required: pointCount - 2,
    invalid: 0,
    didConverge: 1,
    limitReached: 0
  });
  expect(result.iterations).toBeGreaterThan(3);
  expect(result.iterations).toBeLessThan(100);
  expect(result.written).toBe(2 * pointCount - 2 - countHullBoundarySites(coordinates));
  expect(checkTriangulation(coordinates, toTriangleRows(result.indices))).toEqual({
    degenerate: 0,
    circleViolations: 0,
    duplicateEdges: 0,
    areaDifference: 0n
  });
}, 600_000);
