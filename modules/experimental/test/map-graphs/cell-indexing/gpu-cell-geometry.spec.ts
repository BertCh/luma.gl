// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellToBoundary as a5CellToBoundary, cellToLonLat, lonLatToCell} from 'a5-js';
import {
  cellToBoundary as h3CellToBoundary,
  cellToLatLng,
  getPentagons,
  getRes0Cells,
  latLngToCell
} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {createMapGraphKernelNode} from '../../../src/map-graphs/map-graph-kernels';
import {
  GPU_CELL_GEOMETRY_VERTEX_COUNTS,
  GPUCellGeometry,
  type GPUCellGeometryFamily
} from '../../../src/map-graphs/cell-indexing';
import {GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT} from '../../../src/map-graphs/cell-indexing/gpu-cell-geometry';
import {
  h3ToBigInt,
  quadbinCellToTile,
  quadbinPointToCell
} from '../cell-aggregation/cell-aggregation-oracle';
import {createRandom} from '../cell-aggregation/cell-aggregation-points';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  geohashCellToBounds,
  geohashPointToCell,
  getAngularDistance,
  quadkeyCellToBounds,
  quadkeyCellToTile,
  quadkeyPointToCell,
  s2CellToBoundary,
  s2CellToCenter,
  s2CellToFaceIJ,
  s2CellFromFaceIJ,
  s2PointToCell,
  splitKey,
  webMercatorTileBounds,
  webMercatorTileCenter
} from './cell-indexing-oracle';

type LngLat = [number, number];
type Expected = {center: LngLat; boundary: LngLat[]};

/** Rectangle corners NW, NE, SE, SW from `[west, south, east, north]`. */
function getRectangle([west, south, east, north]: number[]): LngLat[] {
  return [
    [west, north],
    [east, north],
    [east, south],
    [west, south]
  ];
}

/** CPU geometry of one key: the reference the GPU decoders are compared with. */
function getExpected(family: GPUCellGeometryFamily, key: bigint): Expected {
  switch (family) {
    case 'quadbin': {
      const {x, y, z} = quadbinCellToTile(key);
      return {
        center: webMercatorTileCenter(x, y, z),
        boundary: getRectangle(webMercatorTileBounds(x, y, z))
      };
    }
    case 'quadkey': {
      const {x, y, z} = quadkeyCellToTile(key);
      return {
        center: webMercatorTileCenter(x, y, z),
        boundary: getRectangle(quadkeyCellToBounds(key))
      };
    }
    case 'geohash': {
      const bounds = geohashCellToBounds(key);
      return {
        center: [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2],
        boundary: getRectangle(bounds)
      };
    }
    case 's2':
      return {center: s2CellToCenter(key), boundary: s2CellToBoundary(key)};
    case 'h3': {
      const cell = key.toString(16);
      const [latitude, longitude] = cellToLatLng(cell);
      return {
        center: [longitude, latitude],
        boundary: h3CellToBoundary(cell).map(([lat, lng]) => [lng, lat] as LngLat)
      };
    }
    case 'a5': {
      // The dggs decoder starts at the last a5-js vertex: it lists a5-js vertices (n-1, 0, 1, ...).
      const ring = a5CellToBoundary(key, {segments: 1, closedRing: false}) as LngLat[];
      return {
        center: cellToLonLat(key) as LngLat,
        boundary: [ring[ring.length - 1], ...ring.slice(0, -1)]
      };
    }
  }
}

type GeometryResult = {
  centers: number[];
  boundaries: number[];
  vertexCounts: number[];
};

/** One compiled `GPUCellGeometry` graph over a writable key buffer. */
function createGeometry(
  device: Device,
  family: GPUCellGeometryFamily,
  rows: number,
  options: {maximumVertexCount?: number; wordOrder?: 'little-endian' | 'high-low'} = {}
) {
  const maximumVertexCount = options.maximumVertexCount ?? GPU_CELL_GEOMETRY_VERTEX_COUNTS[family];
  const cells = createInputBuffer(device, new Uint32Array(2 * rows));
  const centers = createOutputBuffer(device, 2 * rows);
  const boundaries = createOutputBuffer(device, 2 * rows * maximumVertexCount);
  const vertexCounts = createOutputBuffer(device, rows);
  const graph = new GPUCommandGraph(device, {id: `cell-geometry-${family}`});
  graph.add(
    new GPUCellGeometry({
      family,
      wordOrder: options.wordOrder,
      maximumVertexCount: options.maximumVertexCount,
      cells: importGraphBuffer(graph, 'cells', cells, 'uint32x2', rows),
      output: {
        centers: importGraphBuffer(graph, 'centers', centers, 'float32x2', rows),
        boundaries: importGraphBuffer(
          graph,
          'boundaries',
          boundaries,
          'float32x2',
          rows * maximumVertexCount
        ),
        vertexCounts: importGraphBuffer(graph, 'vertex-counts', vertexCounts, 'uint32', rows)
      }
    })
  );
  const compiled = graph.compile();
  const buffers: Buffer[] = [cells, centers, boundaries, vertexCounts];
  return {
    compiled,
    maximumVertexCount,
    async run(keys: bigint[]): Promise<GeometryResult> {
      const words = new Uint32Array(2 * rows);
      keys.forEach((key, row) => {
        const [low, high] = splitKey(key);
        const first = options.wordOrder === 'high-low' ? high : low;
        const second = options.wordOrder === 'high-low' ? low : high;
        words[2 * row] = first;
        words[2 * row + 1] = second;
      });
      cells.write(words);
      submitGraph(device, compiled, undefined);
      return {
        centers: await readFloat32(centers, 2 * rows),
        boundaries: await readFloat32(boundaries, 2 * rows * maximumVertexCount),
        vertexCounts: await readUint32(vertexCounts, rows)
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function getRandomPoints(seed: number, count: number): LngLat[] {
  const random = createRandom(seed);
  const points: LngLat[] = [];
  for (let index = 0; index < count; index++) {
    points.push([
      Math.fround(random() * 360 - 180),
      Math.fround((Math.asin(random() * 1.9 - 0.95) * 180) / Math.PI)
    ]);
  }
  return points;
}

/** Largest vertex distance from the center in radians: the cell's angular size. */
function getCellSize({center, boundary}: Expected): number {
  return Math.max(...boundary.map(vertex => getAngularDistance(center, vertex)));
}

type Tolerance = {
  /** Absolute angular error floor in radians (f32 coordinates and WGSL trigonometry). */
  absolute: number;
  /** Fraction of the cell size. */
  relative: number;
};

type GeometryReport = {
  rows: number;
  maximumCenterError: number;
  maximumVertexError: number;
};

/** Runs `keys` and compares centers and boundaries with the CPU reference. */
async function expectGeometry(
  device: Device,
  family: GPUCellGeometryFamily,
  keys: bigint[],
  tolerance: Tolerance,
  label: string,
  options: {maximumVertexCount?: number} = {}
): Promise<GeometryReport> {
  const geometry = createGeometry(device, family, keys.length, options);
  const result = await geometry.run(keys);
  const stride = geometry.maximumVertexCount;
  const report: GeometryReport = {
    rows: keys.length,
    maximumCenterError: 0,
    maximumVertexError: 0
  };
  keys.forEach((key, row) => {
    const expected = getExpected(family, key);
    const size = getCellSize(expected);
    const allowed = tolerance.absolute + tolerance.relative * size;
    const center: LngLat = [result.centers[2 * row], result.centers[2 * row + 1]];
    const centerError = getAngularDistance(center, expected.center);
    report.maximumCenterError = Math.max(report.maximumCenterError, centerError / allowed);
    expect(centerError, `${label} center of ${key.toString(16)}`).toBeLessThanOrEqual(allowed);
    const count = result.vertexCounts[row];
    // Every key given here is valid, so every cell must have a boundary.
    expect(count, `${label} vertex count of ${key.toString(16)}`).toBeGreaterThan(0);
    expect(count, `${label} vertex count of ${key.toString(16)}`).toBe(expected.boundary.length);
    for (let vertex = 0; vertex < stride; vertex++) {
      const slot = 2 * (row * stride + vertex);
      if (vertex >= count) {
        expect(result.boundaries[slot]).toBe(0);
        expect(result.boundaries[slot + 1]).toBe(0);
        continue;
      }
      const error = getAngularDistance(
        [result.boundaries[slot], result.boundaries[slot + 1]],
        expected.boundary[vertex]
      );
      report.maximumVertexError = Math.max(report.maximumVertexError, error / allowed);
      expect(
        error,
        `${label} vertex ${vertex} of ${key.toString(16)}: got ${result.boundaries[slot]}, ${result.boundaries[slot + 1]} want ${expected.boundary[vertex]}`
      ).toBeLessThanOrEqual(allowed);
    }
  });
  geometry.destroy();
  return report;
}

it('GPUCellGeometry decodes Quadbin and quadkey tiles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const points = getRandomPoints(1, 200);
  for (const resolution of [1, 4, 9, 14, 20, 24]) {
    const quadbinKeys = points.map(([lng, lat]) => quadbinPointToCell(lng, lat, resolution));
    await expectGeometry(
      device,
      'quadbin',
      quadbinKeys,
      {absolute: 3e-6, relative: 0.02},
      'quadbin'
    );
    const quadkeyKeys = points.map(([lng, lat]) => quadkeyPointToCell(lng, lat, resolution));
    await expectGeometry(
      device,
      'quadkey',
      quadkeyKeys,
      {absolute: 3e-6, relative: 0.02},
      'quadkey'
    );
  }
  // Zoom 26-29 tile columns exceed the f32 mantissa: only the center must stay in the tile.
  for (const zoom of [27, 29]) {
    await expectGeometry(
      device,
      'quadkey',
      points.map(([lng, lat]) => quadkeyPointToCell(lng, lat, zoom)),
      {absolute: 2e-6, relative: 1},
      'quadkey deep'
    );
  }
});

it('GPUCellGeometry decodes geohash bounds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const points = getRandomPoints(2, 200);
  for (const length of [1, 2, 4, 6, 8, 12]) {
    await expectGeometry(
      device,
      'geohash',
      points.map(([lng, lat]) => geohashPointToCell(lng, lat, length)),
      {absolute: 3e-6, relative: 0.02},
      'geohash'
    );
  }
});

it('GPUCellGeometry decodes S2 cells and agrees with the S2 Hilbert curve', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const points = getRandomPoints(3, 200);
  const reports: Record<number, GeometryReport> = {};
  for (const level of [0, 1, 2, 5, 9, 14, 18, 22]) {
    reports[level] = await expectGeometry(
      device,
      's2',
      points.map(([lng, lat]) => s2PointToCell(lng, lat, level)),
      {absolute: 3e-6, relative: 0.02},
      `s2 level ${level}`
    );
  }
  // f32 face coordinates cannot place levels 23-30; the cell must still contain its center
  // within the f32 error.
  for (const level of [26, 30]) {
    await expectGeometry(
      device,
      's2',
      points.map(([lng, lat]) => s2PointToCell(lng, lat, level)),
      {absolute: 3e-6, relative: 1},
      `s2 level ${level}`
    );
  }
  expect(reports[14].rows).toBe(200);
});

it('dggs S2 decode agrees with the S2 oracle on face, level and (i, j) at every level', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Random leaf cells from the integer oracle, truncated to every level 0..30.
  const random = createRandom(77);
  const keys: bigint[] = [];
  for (let index = 0; index < 4000; index++) {
    const level = index % 31;
    keys.push(
      s2CellFromFaceIJ(
        Math.floor(random() * 6),
        Math.floor(random() * 2 ** 30),
        Math.floor(random() * 2 ** 30),
        level
      )
    );
  }
  const rows = keys.length;
  const cells = createInputBuffer(device, new Uint32Array(2 * rows));
  const decoded = createOutputBuffer(device, 4 * rows);
  const graph = new GPUCommandGraph(device, {id: 'dggs-s2-decode'});
  graph.add(
    createMapGraphKernelNode(graph, {
      id: 'dggs-s2-decode-kernel',
      operation: 'DggsS2Decode',
      bindings: [
        {
          name: 'cells',
          view: importGraphBuffer(graph, 'cells', cells, 'uint32x2', rows),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'decoded',
          view: importGraphBuffer(graph, 'decoded', decoded, 'uint32', rows * 4),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: rows,
      declarations: dggs.source,
      body: `let key = vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]);
  let ij = dggs_s2_get_ij(key);
  decoded[decodedOffset + 4u * index] = dggs_s2_get_face(key);
  decoded[decodedOffset + 4u * index + 1u] = dggs_s2_get_level(key);
  decoded[decodedOffset + 4u * index + 2u] = ij.x;
  decoded[decodedOffset + 4u * index + 3u] = ij.y;`
    })
  );
  const compiled = graph.compile();
  cells.write(Uint32Array.from(keys.flatMap(key => splitKey(key))));
  submitGraph(device, compiled, undefined);
  const words = await readUint32(decoded, 4 * rows);
  let disagreements = 0;
  keys.forEach((key, row) => {
    const expected = s2CellToFaceIJ(key);
    const actual = words.slice(4 * row, 4 * row + 4);
    if (
      actual[0] !== expected.face ||
      actual[1] !== expected.level ||
      actual[2] !== expected.i ||
      actual[3] !== expected.j
    ) {
      disagreements++;
    }
  });
  expect(disagreements).toBe(0);
  compiled.destroy();
  cells.destroy();
  decoded.destroy();
});

it('GPUCellGeometry decodes H3 centers and hexagon boundaries like h3-js', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const points = getRandomPoints(4, 200);
  const tolerance = {absolute: 3e-6, relative: 0.02};
  // Stride 10 holds every H3 boundary, distortion vertices included; every cell needs one.
  const options = {maximumVertexCount: GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT};
  for (const resolution of [0, 1, 2, 3, 5, 8, 11, 14, 15]) {
    const keys = points.map(([lng, lat]) => h3ToBigInt(latLngToCell(lat, lng, resolution)));
    await expectGeometry(device, 'h3', keys, tolerance, `h3 ${resolution}`, options);
  }
  // Every pentagon at every resolution (5 or 10 vertices) and every base cell.
  const pentagons: bigint[] = [];
  for (let resolution = 0; resolution <= 15; resolution++) {
    pentagons.push(...getPentagons(resolution).map(h3ToBigInt));
  }
  await expectGeometry(device, 'h3', pentagons, tolerance, 'h3 pentagons', options);
  const base = getRes0Cells().map(h3ToBigInt);
  await expectGeometry(device, 'h3', base, tolerance, 'h3 base cells', options);
  // A Class III pentagon has 10 vertices; the default stride (6) clamps the count to 6.
  const geometry = createGeometry(device, 'h3', 1);
  const clamped = await geometry.run([h3ToBigInt(getPentagons(3)[0])]);
  expect(clamped.vertexCounts[0]).toBe(6);
  geometry.destroy();
});

it('GPUCellGeometry decodes A5 cells like a5-js', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const points = getRandomPoints(5, 100);
  for (const resolution of [2, 5, 10, 16]) {
    await expectGeometry(
      device,
      'a5',
      points.map(point => lonLatToCell(point as never, resolution)),
      {absolute: 3e-6, relative: 0.02},
      `a5 ${resolution}`
    );
  }
});

it('GPUCellGeometry marks invalid keys, pads boundaries and honors word order', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const quadbin = quadbinPointToCell(10, 50, 8);
  const keys = [0n, 1n, quadbin, 0xffffffffffffffffn];
  for (const wordOrder of ['little-endian', 'high-low'] as const) {
    const geometry = createGeometry(device, 'quadbin', keys.length, {
      maximumVertexCount: 6,
      wordOrder
    });
    const result = await geometry.run(keys);
    expect(result.vertexCounts).toEqual([0, 0, 4, 0]);
    for (const row of [0, 1, 3]) {
      expect(Number.isNaN(result.centers[2 * row])).toBe(true);
      expect(Number.isNaN(result.centers[2 * row + 1])).toBe(true);
    }
    const expected = getExpected('quadbin', quadbin);
    expect(result.centers[4]).toBeCloseTo(expected.center[0], 3);
    expect(result.centers[5]).toBeCloseTo(expected.center[1], 3);
    // Slots 4 and 5 of the valid row are padding.
    const base = 2 * 2 * 6;
    expect(Array.from(result.boundaries.slice(base + 8, base + 12))).toEqual([0, 0, 0, 0]);
    geometry.destroy();
  }
});

it('GPUCellGeometry follows per-frame keys without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const geometry = createGeometry(device, 'geohash', 4);
  const compiled = geometry.compiled;
  const seen = new Set<number>();
  for (const seed of [1, 2, 3, 1]) {
    const keys = getRandomPoints(seed, 4).map(([lng, lat]) => geohashPointToCell(lng, lat, 6));
    const result = await geometry.run(keys);
    for (const [row, key] of keys.entries()) {
      const expected = getExpected('geohash', key);
      expect(result.centers[2 * row]).toBeCloseTo(expected.center[0], 3);
      expect(result.centers[2 * row + 1]).toBeCloseTo(expected.center[1], 3);
    }
    seen.add(result.centers[0]);
    expect(geometry.compiled).toBe(compiled);
  }
  expect(seen.size).toBeGreaterThan(2);
  geometry.destroy();
});
