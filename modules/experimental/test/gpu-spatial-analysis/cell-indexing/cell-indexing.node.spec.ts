// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_CELL_INDEX_RESOLUTION_RANGES,
  GPU_CELL_GEOMETRY_VERTEX_COUNTS,
  GPUCellGeometry,
  GPUPointToCell,
  type GPUCellGeometryProps,
  validateCellIndexResolution,
  type GPUCellIndexFamily,
  type GPUPointToCellProps
} from '../../../src/gpu-spatial-analysis/cell-indexing';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getQuadbinTileYFloat64,
  quadbinPointToCell
} from '../cell-aggregation/cell-aggregation-oracle';
import {createPointPositions} from '../cell-aggregation/cell-aggregation-points';
import {
  geohashCellToBounds,
  geohashCellToString,
  geohashPointToCell,
  geohashStringToCell,
  quadkeyCellToString,
  quadkeyCellToTile,
  quadkeyPointToCell,
  quadkeyStringToCell,
  quadkeyTileToCell,
  s2CellFromFaceIJ,
  s2CellToCenter,
  s2CellToFaceIJ,
  s2CellToToken,
  s2PointToCell,
  s2TokenToCell
} from './cell-indexing-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUPointToCellProps> = {}
): GPUPointToCellProps {
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32x2'>(format: Format, length = 8) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    family: 'quadbin',
    resolution: 6,
    positions: view('float32x2'),
    output: {cells: view('uint32x2'), validity: view('uint32')},
    ...overrides
  };
}

it('family resolution ranges are validated', () => {
  expect(GPU_CELL_INDEX_RESOLUTION_RANGES).toEqual({
    quadbin: {minimum: 0, maximum: 26},
    h3: {minimum: 0, maximum: 15},
    quadkey: {minimum: 1, maximum: 29},
    geohash: {minimum: 1, maximum: 12},
    s2: {minimum: 0, maximum: 30}
  });
  for (const [family, {minimum, maximum}] of Object.entries(GPU_CELL_INDEX_RESOLUTION_RANGES)) {
    const typed = family as GPUCellIndexFamily;
    validateCellIndexResolution('x', typed, minimum);
    validateCellIndexResolution('x', typed, maximum);
    expect(() => validateCellIndexResolution('x', typed, minimum - 1)).toThrow(/resolution/);
    expect(() => validateCellIndexResolution('x', typed, maximum + 1)).toThrow(/resolution/);
    expect(() => validateCellIndexResolution('x', typed, 1.5)).toThrow(/resolution/);
  }
  expect(() => validateCellIndexResolution('x', 'a5' as GPUCellIndexFamily, 1)).toThrow(/family/);
});

it('GPUPointToCell validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const expectThrows = (overrides: Partial<GPUPointToCellProps>, message: RegExp) =>
    expect(() => new GPUPointToCell(createProps(graph, overrides))).toThrow(message);
  expect(() => new GPUPointToCell(createProps(graph))).not.toThrow();
  expectThrows({resolution: 27}, /resolution/);
  expectThrows({family: 'geohash', resolution: 0}, /resolution/);
  expectThrows({family: 'bogus' as GPUCellIndexFamily}, /family/);
  expectThrows({mask: createTransientView(graph, 'mask-a', 'uint32', 7)}, /mask length/);
  expectThrows({mask: createTransientView(graph, 'mask-b', 'float32', 8) as never}, /mask/);
  expectThrows(
    {positions: createTransientView(graph, 'positions-a', 'float32', 8) as never},
    /positions/
  );
  expectThrows(
    {output: {cells: createTransientView(graph, 'cells-a', 'uint32x2', 7)}},
    /output.cells length/
  );
  expectThrows(
    {
      output: {
        cells: createTransientView(graph, 'cells-b', 'uint32x2', 8),
        validity: createTransientView(graph, 'validity-a', 'uint32', 9)
      }
    },
    /output.validity length/
  );
  expectThrows({positions: createTransientView(graph, 'positions-b', 'float32x2', 0)}, /row/);
  // Aliasing: the output shares the mask buffer.
  const shared = createTransientView(graph, 'shared', 'uint32', 8);
  expectThrows(
    {
      mask: shared,
      output: {cells: createTransientView(graph, 'cells-c', 'uint32x2', 8), validity: shared}
    },
    /outputs must not share buffers with inputs/
  );
  // Views from another graph are rejected when the nodes are created.
  const other = new GPUCommandGraph(device);
  const foreign = new GPUPointToCell(createProps(other));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/target graph/);
  device.destroy();
});

it('GPUPointToCell emits one deterministic node per family', () => {
  const device = createNullWebGPUDevice();
  for (const [family, {minimum}] of Object.entries(GPU_CELL_INDEX_RESOLUTION_RANGES)) {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, {
      id: `cells-${family}`,
      family: family as GPUCellIndexFamily,
      resolution: minimum + 1
    });
    const contributor = new GPUPointToCell(props);
    const first = contributor.getCommandNodes(graph).map(node => node.id);
    expect(first).toEqual([`cells-${family}-keys`]);
    // Without validity or mask the same kernel is emitted.
    const minimal = new GPUPointToCell({
      ...props,
      id: `min-${family}`,
      output: {cells: props.output.cells}
    });
    expect(minimal.getCommandNodes(graph).length).toBe(1);
  }
  expect(new GPUPointToCell(createProps(new GPUCommandGraph(device))).id).toBe('point-to-cell');
  device.destroy();
});

it('quadkey oracle follows the Bing tile system', () => {
  // Microsoft Bing Maps Tile System documentation: tile (3, 5) at level 3 is quadkey '213'.
  expect(quadkeyCellToString(quadkeyTileToCell(3, 5, 3))).toBe('213');
  expect(quadkeyStringToCell('213')).toBe(quadkeyTileToCell(3, 5, 3));
  expect(quadkeyCellToTile(quadkeyStringToCell('0231'))).toEqual({x: 3, y: 6, z: 4});
  // Level 1 quadrants: NW 0, NE 1, SW 2, SE 3.
  expect(quadkeyCellToString(quadkeyPointToCell(-90, 45, 1))).toBe('0');
  expect(quadkeyCellToString(quadkeyPointToCell(90, 45, 1))).toBe('1');
  expect(quadkeyCellToString(quadkeyPointToCell(-90, -45, 1))).toBe('2');
  expect(quadkeyCellToString(quadkeyPointToCell(90, -45, 1))).toBe('3');
  // Same tile as the Quadbin of equal zoom.
  expect(quadkeyCellToTile(quadkeyPointToCell(-3.7038, 40.4168, 4))).toEqual({x: 7, y: 6, z: 4});
  expect(quadbinPointToCell(-3.7038, 40.4168, 4)).toBe(5207251884775047167n);
});

it('quadkey rows above zoom 26 versus the f64 formula', () => {
  const report: Record<number, string> = {};
  for (const zoom of [20, 26, 27, 28, 29]) {
    const positions = createPointPositions(zoom, 20000, zoom);
    let mismatches = 0;
    let maximum = 0;
    const rows = 20000;
    for (let row = 0; row < rows; row++) {
      const latitude = positions[2 * row + 1];
      const error = Math.abs(
        quadkeyCellToTile(quadkeyPointToCell(0, latitude, zoom)).y -
          getQuadbinTileYFloat64(latitude, zoom)
      );
      mismatches += error > 0 ? 1 : 0;
      maximum = Math.max(maximum, error);
    }
    report[zoom] = `${(mismatches / rows).toExponential(2)} (max ${maximum} rows)`;
    // The fixed-point row is within one row of the f64 formula through zoom 29 for f32 input.
    expect(maximum, `zoom ${zoom}`).toBeLessThanOrEqual(Math.max(1, Math.ceil(2 ** zoom * 1e-7)));
  }
  // eslint-disable-next-line no-console
  console.log('quadkey fixed-point row vs f64 formula, mismatch rate per zoom', report);
});

it('geohash oracle matches known geohashes', () => {
  // Wikipedia: (57.64911, 10.40744) is u4pruydqqvj.
  expect(geohashCellToString(geohashPointToCell(10.40744, 57.64911, 11))).toBe('u4pruydqqvj');
  expect(geohashCellToString(geohashPointToCell(10.40744, 57.64911, 5))).toBe('u4pru');
  // Wikipedia example: 42.605, -5.603 is ezs42.
  expect(geohashCellToString(geohashPointToCell(-5.603, 42.605, 5))).toBe('ezs42');
  expect(geohashStringToCell('ezs42')).toBe(geohashPointToCell(-5.603, 42.605, 5));
  const [west, south, east, north] = geohashCellToBounds(geohashStringToCell('ezs42'));
  expect(-5.603).toBeGreaterThanOrEqual(west);
  expect(-5.603).toBeLessThan(east);
  expect(42.605).toBeGreaterThanOrEqual(south);
  expect(42.605).toBeLessThan(north);
  expect(east - west).toBeCloseTo(0.0439453125, 12);
  // Packed layout: length in bits 60..63, first character most significant.
  expect(geohashStringToCell('s')).toBe((1n << 60n) | 24n);
  // Extremes land in the last bin; the packed key is never zero.
  expect(geohashCellToString(geohashPointToCell(180, 90, 3))).toBe('zzz');
  expect(geohashCellToString(geohashPointToCell(-180, -90, 3))).toBe('000');
});

/** Published Go S2 cell tokens at level 15 from the s2-geometry test corpus (lat, lng, face). */
const S2_PUBLISHED_LEVEL_15 = [
  {latitude: -13.846153846153854, longitude: -13.846153846153854, token: '044520e0c', face: 0},
  {latitude: 13.84615384615384, longitude: 69.23076923076923, token: '3b8ebc30c', face: 1},
  {latitude: 41.53846153846153, longitude: 41.53846153846153, token: '406790fac', face: 2},
  {latitude: 35.67628696987912, longitude: 139.71965789794925, token: '60188c9ac', face: 3},
  {latitude: 13.84615384615384, longitude: -124.61538461538463, token: '823551744', face: 4},
  {latitude: -41.53846153846155, longitude: -41.53846153846155, token: 'bf986f054', face: 5},
  {latitude: -43.525166, longitude: 172.655096, token: '6d318985c', face: 3},
  {latitude: 69.23076923076923, longitude: 124.61538461538461, token: '5b992a2bc', face: 2},
  {latitude: -69.23076923076924, longitude: -69.23076923076924, token: 'bb6ff8fec', face: 5},
  {latitude: 41.53846153846153, longitude: -96.92307692307693, token: '87911aa74', face: 4}
];

it('S2 oracle matches published S2CellId values', () => {
  // Face cells: (face << 61) | (1 << 60).
  for (let face = 0; face < 6; face++) {
    expect(s2CellFromFaceIJ(face, 0, 0, 0)).toBe((BigInt(face) << 61n) | (1n << 60n));
  }
  expect(s2CellToToken(s2CellFromFaceIJ(0, 0, 0, 0))).toBe('1');
  // Provo, UT, level 15 (s2-geometry tests, generated with the Go S2 library).
  expect(s2PointToCell(-111.709298, 40.256312, 15)).toBe(9749618446378729472n);
  expect(s2PointToCell(-111.664582, 40.226063, 15)).toBe(9749615171466166272n);
  for (const {latitude, longitude, token, face} of S2_PUBLISHED_LEVEL_15) {
    const cell = s2PointToCell(longitude, latitude, 15);
    expect(s2CellToToken(cell), `${latitude}, ${longitude}`).toBe(token);
    expect(Number(cell >> 61n)).toBe(face);
    expect(s2TokenToCell(token)).toBe(cell);
  }
  // Leaf (i, j) of the Tokyo entry in the same corpus is face 3, i 23989781, j 63280597.
  expect(s2CellToToken(s2CellFromFaceIJ(3, 23989781, 63280597, 15))).toBe('60188c9ac');
  // (0, 0) is the corner of the first child positions: i = j = 2^29 at face 0.
  expect(s2PointToCell(0, 0, 30).toString(16)).toBe('1000000000000001');
});

it('S2 oracle ids decode to their cell and nest like a Hilbert curve', () => {
  for (const {latitude, longitude} of S2_PUBLISHED_LEVEL_15) {
    const leaf = s2PointToCell(longitude, latitude, 30);
    for (const level of [0, 1, 7, 15, 22, 29, 30]) {
      const cell = s2PointToCell(longitude, latitude, level);
      const parent = (leaf & -(1n << BigInt(2 * (30 - level)))) | (1n << BigInt(2 * (30 - level)));
      expect(cell).toBe(parent);
      const {face, i, j, level: decodedLevel} = s2CellToFaceIJ(cell);
      expect(decodedLevel).toBe(level);
      expect(s2CellFromFaceIJ(face, i * 2 ** (30 - level), j * 2 ** (30 - level), level)).toBe(
        cell
      );
    }
  }
  // Consecutive positions on one level are edge neighbors: the curve is continuous.
  for (const face of [0, 1]) {
    const level = 5;
    let previous: ReturnType<typeof s2CellToFaceIJ> | undefined;
    const first = s2CellFromFaceIJ(face, 0, 0, level);
    const step = 1n << BigInt(2 * (30 - level) + 1);
    for (let index = 0n; index < 4n ** 5n; index++) {
      const decoded = s2CellToFaceIJ(first + index * step - (first & (step - 1n)) * 0n);
      if (previous) {
        expect(Math.abs(decoded.i - previous.i) + Math.abs(decoded.j - previous.j)).toBe(1);
      }
      previous = decoded;
    }
  }
  // The cell center decodes back into the same cell.
  const cell = s2PointToCell(10, 50, 12);
  const [centerLongitude, centerLatitude] = s2CellToCenter(cell);
  expect(s2PointToCell(centerLongitude, centerLatitude, 12)).toBe(cell);
});

it('GPUCellGeometry validates props and emits deterministic nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32x2'>(format: Format, length = 8) =>
    createTransientView(graph, `geometry-view-${serial++}`, format, length);
  const createGeometryProps = (overrides: Partial<GPUCellGeometryProps> = {}) =>
    ({
      family: 'h3',
      cells: view('uint32x2'),
      output: {
        centers: view('float32x2'),
        boundaries: view('float32x2', 48),
        vertexCounts: view('uint32')
      },
      ...overrides
    }) as GPUCellGeometryProps;
  const expectThrows = (overrides: Partial<GPUCellGeometryProps>, message: RegExp) =>
    expect(() => new GPUCellGeometry(createGeometryProps(overrides))).toThrow(message);
  expect(GPU_CELL_GEOMETRY_VERTEX_COUNTS).toEqual({
    quadbin: 4,
    h3: 6,
    quadkey: 4,
    geohash: 4,
    s2: 4,
    a5: 5
  });
  const contributor = new GPUCellGeometry(createGeometryProps({id: 'geometry'}));
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual(['geometry-geometry']);
  expectThrows({family: 'bogus' as never}, /family/);
  expectThrows({wordOrder: 'big' as never}, /wordOrder/);
  expectThrows({maximumVertexCount: 5}, /maximumVertexCount/);
  expectThrows({maximumVertexCount: 17}, /maximumVertexCount/);
  expectThrows({maximumVertexCount: 7}, /boundaries length/);
  expectThrows({cells: view('float32x2') as never}, /cells/);
  expectThrows({output: {}}, /needs output.centers or output.boundaries/);
  expectThrows({output: {vertexCounts: view('uint32')}}, /needs output.boundaries/);
  expectThrows({output: {centers: view('float32x2', 7)}}, /centers length/);
  expectThrows({output: {boundaries: view('float32x2', 47)}}, /boundaries length/);
  expectThrows(
    {output: {boundaries: view('float32x2', 48), vertexCounts: view('uint32', 7)}},
    /vertexCounts length/
  );
  const shared = view('uint32x2');
  expectThrows(
    {cells: shared, output: {centers: shared as never}},
    /outputs must not share buffers with inputs|centers/
  );
  expect(
    () =>
      new GPUCellGeometry({
        family: 'a5',
        cells: view('uint32x2'),
        maximumVertexCount: 8,
        output: {boundaries: view('float32x2', 64)}
      })
  ).not.toThrow();
  device.destroy();
});
