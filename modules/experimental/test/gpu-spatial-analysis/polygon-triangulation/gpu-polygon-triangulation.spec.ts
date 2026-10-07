// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/index';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createSegmentGeometry} from '../segment-intersection/segment-geometry-harness';
import type {OracleSegmentFeature} from '../segment-intersection/segment-intersection-oracle';
import {EARCUT_FIXTURE} from './polygon-triangulation-fixture';
import {RANDOM_POLYGON_FIXTURE} from './polygon-triangulation-random-fixture';

type Point = [number, number];

type Triangulation = {
  indices: number[];
  valid: number[];
  triangleCount: number[];
  positions: Point[];
  /** First index of each polygon's slice, with a terminal entry. */
  sliceStarts: number[];
};

async function runTriangulation(
  device: Device,
  polygons: Point[][][],
  options: {maximumWork?: number; useZOrderHash?: boolean} = {}
): Promise<Triangulation> {
  const graph = new GPUCommandGraph(device, {id: 'polygon-triangulation'});
  const buffers: Buffer[] = [];
  const features: OracleSegmentFeature[] = polygons.map(rings => ({
    kind: 'polygons',
    polygons: [rings]
  }));
  const geometry = createSegmentGeometry(device, graph, 'polygons', features, buffers);
  if (geometry.kind !== 'polygons') {
    throw new Error('expected polygons');
  }
  const vertexCount = geometry.positions.length;
  const ringCount = geometry.ringOffsets.length - 1;
  const indexCount = getPolygonTriangulationIndexCount(vertexCount, ringCount, polygons.length);
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const indices = output('indices', indexCount);
  const valid = output('valid', polygons.length);
  const triangleCount = output('triangle-count', polygons.length);
  graph.add(
    new GPUPolygonTriangulation({
      polygons: geometry,
      indices: indices.view,
      valid: valid.view,
      triangleCount: triangleCount.view,
      maximumWork: options.maximumWork,
      useZOrderHash: options.useZOrderHash
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const flatPositions: Point[] = polygons.flatMap(rings => rings.flat());
  const sliceStarts = [0];
  for (const rings of polygons) {
    const vertices = rings.reduce((sum, ring) => sum + ring.length, 0);
    sliceStarts.push(
      sliceStarts[sliceStarts.length - 1] + 3 * (vertices + 2 * (rings.length - 1) - 2)
    );
  }
  const result = {
    indices: await readUint32(indices.buffer, indexCount),
    valid: await readUint32(valid.buffer, polygons.length),
    triangleCount: await readUint32(triangleCount.buffer, polygons.length),
    positions: flatPositions,
    sliceStarts
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

type Triangle = [Point, Point, Point];

function getSlice(result: Triangulation, polygon: number, triangleCount: number): Triangle[] {
  const triangles: Triangle[] = [];
  for (let k = 0; k < triangleCount; k++) {
    const base = result.sliceStarts[polygon] + 3 * k;
    triangles.push([
      result.positions[result.indices[base]],
      result.positions[result.indices[base + 1]],
      result.positions[result.indices[base + 2]]
    ]);
  }
  return triangles;
}

const cross = (a: Point, b: Point, c: Point) =>
  (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);

/** True when the open interiors of two triangles share area (separating-axis test, exact). */
function trianglesOverlap(first: Triangle, second: Triangle): boolean {
  for (const triangle of [first, second]) {
    for (let edge = 0; edge < 3; edge++) {
      const a = triangle[edge];
      const b = triangle[(edge + 1) % 3];
      const axis: Point = [b[1] - a[1], a[0] - b[0]];
      const project = (point: Point) => axis[0] * point[0] + axis[1] * point[1];
      const firstValues = first.map(project);
      const secondValues = second.map(project);
      if (
        Math.max(...firstValues) <= Math.min(...secondValues) ||
        Math.max(...secondValues) <= Math.min(...firstValues)
      ) {
        return false;
      }
    }
  }
  return true;
}

it('GPUPolygonTriangulation covers Shapely polygon areas exactly without overlap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const polygons = EARCUT_FIXTURE.map(entry => entry.rings);
  const start = performance.now();
  const result = await runTriangulation(device, polygons);
  const elapsed = performance.now() - start;
  // eslint-disable-next-line no-console
  console.log(
    `GPUPolygonTriangulation: ${EARCUT_FIXTURE.length} polygons in ${elapsed.toFixed(0)} ms`
  );
  EARCUT_FIXTURE.forEach((entry, polygon) => {
    expect(result.valid[polygon], `${entry.name} valid`).toBe(1);
    const triangleCount = result.triangleCount[polygon];
    const slots = (result.sliceStarts[polygon + 1] - result.sliceStarts[polygon]) / 3;
    expect(triangleCount, `${entry.name} within slice`).toBeLessThanOrEqual(slots);
    const triangles = getSlice(result, polygon, slots);
    // Real triangles come first, padding is degenerate (first, first, first).
    const real = triangles.slice(0, triangleCount);
    for (const padding of triangles.slice(triangleCount)) {
      expect(cross(...padding), `${entry.name} padding`).toBe(0);
      expect(padding[0]).toEqual(padding[1]);
    }
    let area = 0;
    for (const triangle of real) {
      const signed = cross(...triangle);
      expect(signed, `${entry.name} nondegenerate`).not.toBe(0);
      area += Math.abs(signed) / 2;
    }
    expect(area, `${entry.name} area vs Shapely`).toBe(entry.shapelyArea);
    expect(Math.sign(cross(...real[0])), `${entry.name} winding`).not.toBe(0);
    for (const triangle of real) {
      expect(Math.sign(cross(...triangle)), `${entry.name} consistent winding`).toBe(
        Math.sign(cross(...real[0]))
      );
    }
    for (let i = 0; i < real.length; i++) {
      for (let j = i + 1; j < real.length; j++) {
        expect(trianglesOverlap(real[i], real[j]), `${entry.name} ${i}/${j} overlap`).toBe(false);
      }
    }
    // Sanity against the earcut package: same count of triangles for these inputs.
    expect(triangleCount, `${entry.name} earcut count`).toBe(entry.earcutTriangleCount);
  });
});

it('GPUPolygonTriangulation flags broken and oversized polygons and keeps neighbors intact', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const square: Point[][] = [
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ]
  ];
  const twoVertexShell: Point[][] = [
    [
      [0, 0],
      [5, 5]
    ]
  ];
  const bowtie: Point[][] = [
    [
      [0, 0],
      [10, 10],
      [10, 0],
      [0, 10]
    ]
  ];
  const wavy = EARCUT_FIXTURE.find(entry => entry.name === 'wavy_grid_holes')!.rings as Point[][];
  const polygons = [square, twoVertexShell, square, wavy, square];
  // The 2-vertex shell breaks the slice formula's exactness only by one polygon with 0 slots.
  const result = await runTriangulation(device, polygons, {maximumWork: 5000});
  expect(result.valid[0]).toBe(1);
  expect(result.valid[1]).toBe(0);
  expect(result.valid[2]).toBe(1);
  expect(result.valid[3], 'oversized polygon is flagged, not hung').toBe(0);
  expect(result.valid[4]).toBe(1);
  expect(result.triangleCount[3]).toBe(0);
  const wavySlots = (result.sliceStarts[4] - result.sliceStarts[3]) / 3;
  for (const triangle of getSlice(result, 3, wavySlots)) {
    expect(cross(...triangle)).toBe(0);
  }
  for (const polygon of [0, 2, 4]) {
    const triangles = getSlice(result, polygon, result.triangleCount[polygon]);
    expect(triangles.length).toBe(2);
    expect(triangles.reduce((sum, triangle) => sum + Math.abs(cross(...triangle)) / 2, 0)).toBe(
      100
    );
  }
  // A self-intersecting bowtie is not triangulable as a simple polygon; it must not hang or
  // report more triangles than slots.
  const bowtieResult = await runTriangulation(device, [bowtie, square]);
  expect(bowtieResult.triangleCount[0]).toBeLessThanOrEqual(2);
  expect(bowtieResult.valid[1]).toBe(1);
  expect(bowtieResult.triangleCount[1]).toBe(2);
});

it('GPUPolygonTriangulation validates its props', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'polygon-triangulation-props'});
  const buffers: Buffer[] = [];
  const geometry = createSegmentGeometry(
    device,
    graph,
    'p',
    [
      {
        kind: 'polygons',
        polygons: [
          [
            [
              [0, 0],
              [1, 0],
              [1, 1]
            ]
          ]
        ]
      }
    ],
    buffers
  );
  if (geometry.kind !== 'polygons') {
    throw new Error('expected polygons');
  }
  const view = (name: string, length: number) =>
    importGraphBuffer(graph, name, createOutputBuffer(device, length), 'uint32', length);
  const base = {polygons: geometry, indices: view('indices', 3), valid: view('valid', 1)};
  expect(getPolygonTriangulationIndexCount(3, 1, 1)).toBe(3);
  expect(() => new GPUPolygonTriangulation({...base, indices: view('short', 2)})).toThrow(
    /indices needs at least 3/
  );
  expect(() => new GPUPolygonTriangulation({...base, valid: view('wrong', 2)})).toThrow(
    /polygon count/
  );
  expect(() => new GPUPolygonTriangulation({...base, maximumWork: 0})).toThrow(/maximumWork/);
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

it('GPUPolygonTriangulation matches Shapely areas on random valid polygons', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const result = await runTriangulation(
    device,
    RANDOM_POLYGON_FIXTURE.map(([rings]) => rings)
  );
  expect(RANDOM_POLYGON_FIXTURE.length).toBeGreaterThan(40);
  RANDOM_POLYGON_FIXTURE.forEach(([rings, shapelyArea], polygon) => {
    expect(result.valid[polygon], `random ${polygon} valid`).toBe(1);
    const slots = (result.sliceStarts[polygon + 1] - result.sliceStarts[polygon]) / 3;
    const real = getSlice(result, polygon, slots).slice(0, result.triangleCount[polygon]);
    const area = real.reduce((sum, triangle) => sum + Math.abs(cross(...triangle)) / 2, 0);
    expect(area, `random ${polygon} area`).toBe(shapelyArea);
    if (rings.length === 1 || polygon % 4 === 0) {
      for (let i = 0; i < real.length; i++) {
        for (let j = i + 1; j < real.length; j++) {
          expect(trianglesOverlap(real[i], real[j]), `random ${polygon} ${i}/${j}`).toBe(false);
        }
      }
    }
  });
});

it('GPUPolygonTriangulation z-order hash gives the same triangles as the ring scan', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Star-shaped outline with deep spikes (many reflex vertices) and two holes, above the 80
  // vertex threshold, next to a small polygon that stays unhashed.
  const spiky: Point[] = [];
  for (let k = 0; k < 700; k++) {
    const angle = (2 * Math.PI * k) / 700;
    const radius = k % 2 === 0 ? 100 : 60 + (k % 7);
    spiky.push([Math.cos(angle) * radius, Math.sin(angle) * radius]);
  }
  const hole = (cx: number, cy: number): Point[] => [
    [cx - 5, cy - 5],
    [cx - 5, cy + 5],
    [cx + 5, cy + 5],
    [cx + 5, cy - 5]
  ];
  const square: Point[][] = [
    [
      [0, 0],
      [4, 0],
      [4, 4],
      [0, 4]
    ]
  ];
  const polygons = [[spiky, hole(-20, 0), hole(20, 10)], square, [spiky]];
  const hashed = await runTriangulation(device, polygons, {useZOrderHash: true});
  const scanned = await runTriangulation(device, polygons, {useZOrderHash: false});
  expect(hashed.valid).toEqual([1, 1, 1]);
  expect(hashed.valid).toEqual(scanned.valid);
  expect(hashed.triangleCount).toEqual(scanned.triangleCount);
  expect(hashed.indices).toEqual(scanned.indices);
});
