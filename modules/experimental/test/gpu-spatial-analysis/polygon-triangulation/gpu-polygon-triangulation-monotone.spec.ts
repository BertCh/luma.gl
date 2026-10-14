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
  getPolygonTriangulationIndexCount,
  type GPUPolygonTriangulationProps
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/index';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSegmentGeometry} from '../segment-intersection/segment-geometry-harness';
import type {OracleSegmentFeature} from '../segment-intersection/segment-intersection-oracle';
import largeFixture from '../../oracles/fixtures/geometry-triangulation-large.json';
import {EARCUT_FIXTURE} from './polygon-triangulation-fixture';
import {RANDOM_POLYGON_FIXTURE} from './polygon-triangulation-random-fixture';
import {
  checkTriangulation,
  createLargePolygonRings,
  createSkylineRings,
  getRingsChecksum,
  type LargePolygonDescriptor
} from './polygon-triangulation-monotone-cases';
import type {ModelPoint} from './polygon-triangulation-monotone-model';

// GPU spec of the monotone path of GPUPolygonTriangulation (algorithms roadmap P2, decision 2).
// Oracles: Shapely area and `shapely.constrained_delaunay_triangles` area sums of
// `test/oracles/geometry-triangulation-large.py`, exact triangle counts, exact edge cancellation
// (the triangles' directed edges must cancel to the oriented polygon boundary, which proves the
// triangles are non-overlapping and cover the polygon), and the Earcut path against itself.

type Point = ModelPoint;

type Triangulation = {
  indices: number[];
  valid: number[];
  triangleCount: number[];
  positions: Point[];
  /** First index slot of each polygon, with a terminal entry. */
  sliceStarts: number[];
  /** First row of each polygon. */
  polygonRows: number[];
  milliseconds: number;
};

async function runTriangulation(
  device: Device,
  polygons: readonly (readonly (readonly Point[])[])[],
  options: Partial<GPUPolygonTriangulationProps> = {}
): Promise<Triangulation> {
  const graph = new GPUCommandGraph(device, {id: 'polygon-triangulation-monotone'});
  const buffers: Buffer[] = [];
  const features: OracleSegmentFeature[] = polygons.map(rings => ({
    kind: 'polygons',
    polygons: [rings.map(ring => ring.map(point => [point[0], point[1]] as [number, number]))]
  }));
  const geometry = createSegmentGeometry(device, graph, 'polygons', features, buffers);
  if (geometry.kind !== 'polygons') {
    throw new Error('expected polygons');
  }
  const {kind: _kind, ...fields} = geometry;
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
      ...fields,
      output: {indices: indices.view, valid: valid.view, triangleCount: triangleCount.view},
      ...options
    })
  );
  const compiled = graph.compile();
  const start = performance.now();
  submitGraph(device, compiled, undefined);
  const result = {
    indices: await readUint32(indices.buffer, indexCount),
    valid: await readUint32(valid.buffer, polygons.length),
    triangleCount: await readUint32(triangleCount.buffer, polygons.length)
  };
  const milliseconds = performance.now() - start;
  const sliceStarts = [0];
  const polygonRows = [0];
  for (const rings of polygons) {
    const vertices = rings.reduce((sum, ring) => sum + ring.length, 0);
    sliceStarts.push(
      sliceStarts[sliceStarts.length - 1] + 3 * (vertices + 2 * (rings.length - 1) - 2)
    );
    polygonRows.push(polygonRows[polygonRows.length - 1] + vertices);
  }
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {
    ...result,
    positions: polygons.flatMap(rings => rings.flat()),
    sliceStarts,
    polygonRows,
    milliseconds
  };
}

/** The triangles of polygon `polygon`, as local row triples, real ones first. */
function getTriangles(
  result: Triangulation,
  polygon: number,
  count: number
): [number, number, number][] {
  const triangles: [number, number, number][] = [];
  const base = result.polygonRows[polygon];
  for (let k = 0; k < count; k++) {
    const at = result.sliceStarts[polygon] + 3 * k;
    triangles.push([
      result.indices[at] - base,
      result.indices[at + 1] - base,
      result.indices[at + 2] - base
    ]);
  }
  return triangles;
}

function expectDegenerateSlice(result: Triangulation, polygon: number, name: string) {
  const slots = (result.sliceStarts[polygon + 1] - result.sliceStarts[polygon]) / 3;
  for (const triangle of getTriangles(result, polygon, slots)) {
    expect(triangle[0], `${name} padding`).toBe(triangle[1]);
    expect(triangle[1], `${name} padding`).toBe(triangle[2]);
  }
}

const cases = largeFixture.cases as (LargePolygonDescriptor & {
  vertexCount: number;
  holeCount: number;
  triangleCount: number;
  checksum: number;
  area: number;
  valid: boolean;
  cdtArea: number;
  cdtTriangleCount: number;
})[];

it('GPUPolygonTriangulation monotone method triangulates 5k to 200k vertex polygons exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expect(cases.length).toBeGreaterThanOrEqual(6);
  for (const entry of cases) {
    const rings = createLargePolygonRings(entry);
    // The rebuilt polygon is the one Shapely measured.
    expect(rings.flat().length, `${entry.name} vertices`).toBe(entry.vertexCount);
    expect(getRingsChecksum(rings), `${entry.name} checksum`).toBe(entry.checksum);
    expect(entry.valid, `${entry.name} Shapely validity`).toBe(true);
    const result = await runTriangulation(device, [rings]);
    // eslint-disable-next-line no-console
    console.log(
      `GPUPolygonTriangulation monotone ${entry.name}: ${entry.vertexCount} vertices in ${result.milliseconds.toFixed(0)} ms`
    );
    expect(result.valid[0], `${entry.name} valid`).toBe(1);
    expect(result.triangleCount[0], `${entry.name} triangle count`).toBe(entry.triangleCount);
    expect(entry.cdtTriangleCount, `${entry.name} constrained Delaunay count`).toBe(
      entry.triangleCount
    );
    const triangles = getTriangles(result, 0, result.triangleCount[0]);
    const check = checkTriangulation(rings, triangles);
    expect(check.minimumSignedArea, `${entry.name} counter-clockwise`).toBeGreaterThanOrEqual(0);
    expect(
      Math.abs(check.area - entry.area) / entry.area,
      `${entry.name} Shapely area`
    ).toBeLessThan(1e-9);
    expect(
      Math.abs(check.area - entry.cdtArea) / entry.cdtArea,
      `${entry.name} constrained Delaunay area`
    ).toBeLessThan(1e-9);
    expect(check.boundaryMatches, `${entry.name} covers the polygon without overlap`).toBe(true);
  }
}, 900_000);

it('GPUPolygonTriangulation monotone method leaves the Earcut path bitwise unchanged for small polygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons: Point[][][] = [
    ...EARCUT_FIXTURE.map(entry => entry.rings as Point[][]),
    ...RANDOM_POLYGON_FIXTURE.map(([rings]) => rings as Point[][])
  ];
  const monotone = await runTriangulation(device, polygons);
  const none = await runTriangulation(device, polygons, {largePolygonMethod: 'none'});
  expect(none.valid.every(flag => flag === 1)).toBe(true);
  expect(monotone.valid).toEqual(none.valid);
  expect(monotone.triangleCount).toEqual(none.triangleCount);
  expect(monotone.indices).toEqual(none.indices);
});

it("GPUPolygonTriangulation largePolygonMethod 'none' keeps valid = 0 past maximumWork and 'monotone' rescues it", async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const wavy = EARCUT_FIXTURE.find(entry => entry.name === 'wavy_grid_holes')!.rings as Point[][];
  const square: Point[][] = [
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ]
  ];
  const skyline = createSkylineRings(700, 5, 0);
  const polygons = [square, wavy, skyline, square];
  const none = await runTriangulation(device, polygons, {
    largePolygonMethod: 'none',
    maximumWork: 1000
  });
  expect(none.valid[0]).toBe(1);
  expect(none.valid[1], 'oversized polygon is flagged, not hung').toBe(0);
  expect(none.valid[2]).toBe(0);
  expect(none.valid[3]).toBe(1);
  for (const polygon of [1, 2]) {
    expect(none.triangleCount[polygon]).toBe(0);
    expectDegenerateSlice(none, polygon, `polygon ${polygon}`);
  }
  // With the monotone retry the same input completes, and the neighbors are untouched.
  const monotone = await runTriangulation(device, polygons, {maximumWork: 1000});
  expect(monotone.valid).toEqual([1, 1, 1, 1]);
  for (const [polygon, rings] of [
    [1, wavy],
    [2, skyline]
  ] as const) {
    const triangles = getTriangles(monotone, polygon, monotone.triangleCount[polygon]);
    const check = checkTriangulation(rings, triangles);
    expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
    expect(check.boundaryMatches).toBe(true);
  }
  for (const polygon of [0, 3]) {
    expect(
      monotone.indices.slice(monotone.sliceStarts[polygon], monotone.sliceStarts[polygon + 1])
    ).toEqual(none.indices.slice(none.sliceStarts[polygon], none.sliceStarts[polygon + 1]));
  }
});

it('GPUPolygonTriangulation monotone method flags invalid polygons and keeps neighbors intact', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const skyline = createSkylineRings(2100, 3, 0);
  const bowtie: Point[][] = [
    [
      [0, 0],
      [10, 10],
      [10, 0],
      [0, 10]
    ]
  ];
  // A large self-intersecting polygon: the skyline with two swapped vertices.
  const broken = [skyline[0].slice()];
  [broken[0][10], broken[0][400]] = [broken[0][400], broken[0][10]];
  const polygons = [skyline, bowtie, broken, skyline];
  const result = await runTriangulation(device, polygons, {monotoneMinimumVertices: 8});
  const alone = await runTriangulation(device, [skyline], {monotoneMinimumVertices: 8});
  expect(result.valid[0]).toBe(1);
  expect(result.valid[3]).toBe(1);
  // The bowtie and the swapped polygon are not simple. Whatever the cheap checks decide, a slice
  // stays inside its own slots, the count agrees with `valid`, and a flagged slice is degenerate.
  for (const [polygon, name] of [
    [1, 'bowtie'],
    [2, 'broken']
  ] as const) {
    const slots = (result.sliceStarts[polygon + 1] - result.sliceStarts[polygon]) / 3;
    expect(result.triangleCount[polygon], name).toBeLessThanOrEqual(slots);
    if (result.valid[polygon] === 0) {
      expect(result.triangleCount[polygon], name).toBe(0);
      expectDegenerateSlice(result, polygon, name);
    }
  }

  expect(result.triangleCount[0]).toBe(alone.triangleCount[0]);
  // The first polygon's slice is the same set of triangles as when it ran alone.
  const sorted = (run: Triangulation, polygon: number) =>
    getTriangles(run, polygon, run.triangleCount[polygon])
      .map(triangle => triangle.join(','))
      .sort();
  expect(sorted(result, 0)).toEqual(sorted(alone, 0));
  expect(sorted(result, 3)).toEqual(sorted(alone, 0));
});

it('GPUPolygonTriangulation monotone method hands polygons beyond monotoneMaximumVertices to Earcut', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const small = createSkylineRings(600, 1, 0);
  const large = createSkylineRings(2400, 2, 0);
  const options = {monotoneMinimumVertices: 8, monotoneMaximumVertices: small[0].length + 100};
  // With a work budget Earcut cannot meet, the overflowed polygons are flagged and degenerate.
  const flagged = await runTriangulation(device, [small, large, small], {
    ...options,
    maximumWork: 1000
  });
  expect(flagged.valid).toEqual([1, 0, 0]);
  expect(flagged.triangleCount).toEqual([small[0].length - 2, 0, 0]);
  expectDegenerateSlice(flagged, 1, 'overflow');
  expectDegenerateSlice(flagged, 2, 'overflow');
  // With the default budget the Earcut fallback pass triangulates them: no worse than 'none'.
  const rescued = await runTriangulation(device, [small, large, small], options);
  expect(rescued.valid).toEqual([1, 1, 1]);
  const triangles = getTriangles(rescued, 1, rescued.triangleCount[1]);
  const check = checkTriangulation(large, triangles);
  expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
  expect(check.boundaryMatches).toBe(true);
});

it('GPUPolygonTriangulation monotone method keeps large polygons that touch a hole triangulable by Earcut', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A hole sharing a vertex with the shell is valid input that the decomposition flags (a diagonal
  // would join two coincident vertices); the Earcut fallback pass then triangulates it.
  const shell: Point[] = [
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100]
  ];
  const hole: Point[] = [
    [0, 0],
    [30, 20],
    [20, 50]
  ];
  const result = await runTriangulation(device, [[shell, hole]], {monotoneMinimumVertices: 3});
  expect(result.valid).toEqual([1]);
  const check = checkTriangulation([shell, hole], getTriangles(result, 0, result.triangleCount[0]));
  expect(check.boundaryMatches).toBe(true);
  expect(check.area).toBe(9450);
  expect(check.minimumSignedArea).toBeGreaterThan(0);
});
