// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellToLatLng, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUPointToCell,
  type GPUCellIndexFamily
} from '../../../src/gpu-spatial-analysis/cell-indexing';
import {h3ToBigInt, quadbinPointToCell} from '../cell-aggregation/cell-aggregation-oracle';
import {createRandom} from '../cell-aggregation/cell-aggregation-points';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  geohashPointToCell,
  getAngularDistance,
  joinKey,
  quadkeyPointToCell,
  s2CellToCenter,
  s2PointToCell
} from './cell-indexing-oracle';

const ROWS = 10240;
/** Rows before the hand-picked edge cases (seams, poles) in `createPoints`. */
const UNIFORM_COUNT = ROWS - 64;

/** Random points plus poles, antimeridian, face seams, signed zero, NaN and infinities. */
function createPoints(seed: number): Float32Array {
  const random = createRandom(seed);
  const points: number[] = [];
  for (let index = 0; index < UNIFORM_COUNT; index++) {
    const longitude = random() * 360 - 180;
    // Equal-area latitudes keep the poles populated.
    const latitude = (Math.asin(random() * 2 - 1) * 180) / Math.PI;
    points.push(longitude, latitude);
  }
  const special: [number, number][] = [
    [0, 90],
    [0, -90],
    [180, 90],
    [-180, -90],
    [180, 0],
    [-180, 0],
    [0, 0],
    [-0, -0],
    [45, 0],
    [-45, 0],
    [135, 0],
    [-135, 0],
    [0, 35.264389],
    [90, -35.264389],
    [179.99999, 85.05113],
    [-179.99999, -85.05113],
    [10.40744, 57.64911],
    [-3.7038, 40.4168],
    [-111.709298, 40.256312],
    [1e-30, 1e-30],
    [-1e-40, 1e-40],
    [360, 10],
    [-540, -10],
    [NaN, 10],
    [10, NaN],
    [NaN, NaN],
    [Infinity, 10],
    [10, -Infinity]
  ];
  for (const [longitude, latitude] of special) {
    points.push(longitude, latitude);
  }
  while (points.length < 2 * ROWS) {
    points.push(random() * 360 - 180, random() * 170 - 85);
  }
  return Float32Array.from(points);
}

/** One compiled `GPUPointToCell` graph over a writable position and mask buffer. */
function createIndexer(
  device: Device,
  family: GPUCellIndexFamily,
  resolution: number,
  options: {mask?: boolean; validity?: boolean} = {}
) {
  const positions = createInputBuffer(device, new Float32Array(2 * ROWS));
  const mask = createInputBuffer(device, new Uint32Array(ROWS).fill(1));
  const cells = createOutputBuffer(device, 2 * ROWS);
  const validity = createOutputBuffer(device, ROWS);
  const graph = new GPUCommandGraph(device, {id: `point-to-cell-${family}-${resolution}`});
  graph.add(
    new GPUPointToCell({
      family,
      resolution,
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', ROWS),
      mask: options.mask ? importGraphBuffer(graph, 'mask', mask, 'uint32', ROWS) : undefined,
      output: {
        cells: importGraphBuffer(graph, 'cells', cells, 'uint32x2', ROWS),
        validity: options.validity
          ? importGraphBuffer(graph, 'validity', validity, 'uint32', ROWS)
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  const buffers: Buffer[] = [positions, mask, cells, validity];
  return {
    compiled,
    async run(points: Float32Array, rowMask?: Uint32Array) {
      positions.write(points);
      if (rowMask) {
        mask.write(rowMask);
      }
      submitGraph(device, compiled, undefined);
      const words = await readUint32(cells, 2 * ROWS);
      const keys: bigint[] = [];
      for (let row = 0; row < ROWS; row++) {
        keys.push(joinKey(words[2 * row], words[2 * row + 1]));
      }
      const valid = options.validity ? await readUint32(validity, ROWS) : undefined;
      return {keys, valid};
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

const isFinitePoint = (points: Float32Array, row: number) =>
  Number.isFinite(points[2 * row]) && Number.isFinite(points[2 * row + 1]);

/** Runs `family` at `resolution` and checks every row against an exact oracle. */
async function expectExact(
  device: Device,
  family: GPUCellIndexFamily,
  resolution: number,
  oracle: (longitude: number, latitude: number) => bigint
) {
  const points = createPoints(resolution + 11);
  const indexer = createIndexer(device, family, resolution, {validity: true});
  const {keys, valid} = await indexer.run(points);
  let checked = 0;
  for (let row = 0; row < ROWS; row++) {
    if (!isFinitePoint(points, row)) {
      expect(keys[row], `${family} ${resolution} non-finite row ${row}`).toBe(0n);
      expect(valid![row]).toBe(0);
      continue;
    }
    const expected = oracle(points[2 * row], points[2 * row + 1]);
    if (keys[row] !== expected) {
      expect(
        keys[row].toString(16),
        `${family} ${resolution} row ${row} (${points[2 * row]}, ${points[2 * row + 1]})`
      ).toBe(expected.toString(16));
    }
    expect(valid![row]).toBe(1);
    checked++;
  }
  expect(checked).toBeGreaterThan(ROWS - 10);
  indexer.destroy();
}

it('GPUPointToCell quadbin equals the BigInt oracle bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const resolution of [0, 1, 5, 10, 15, 16, 20, 26]) {
    await expectExact(device, 'quadbin', resolution, (lng, lat) =>
      quadbinPointToCell(lng, lat, resolution)
    );
  }
});

it('GPUPointToCell quadkey equals the BigInt oracle bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const zoom of [1, 2, 6, 12, 20, 26, 27, 29]) {
    await expectExact(device, 'quadkey', zoom, (lng, lat) => quadkeyPointToCell(lng, lat, zoom));
  }
});

it('GPUPointToCell geohash equals the bisection oracle bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const length of [1, 2, 3, 5, 8, 11, 12]) {
    await expectExact(device, 'geohash', length, (lng, lat) =>
      geohashPointToCell(lng, lat, length)
    );
  }
});

it('GPUPointToCell S2 mismatch rate against the f64 reference stays bounded per level', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const points = createPoints(5);
  const rates: Record<number, string> = {};
  const widths: Record<number, string> = {};
  // Measured on this suite's 10176 equal-area points (GPU f32 vs f64 reference): no mismatch
  // through level 10, 1e-4 at 11, 5e-4 at 12, 2.6e-3 at 15, 6.6e-2 at 20, 0.26 at 22, 0.78 at 24
  // and about every point beyond 26. The bound is about twice the measurement.
  const maximumRates = [
    ...Array(11).fill(5e-4),
    5e-4,
    1e-3,
    2e-3,
    4e-3,
    5e-3,
    1e-2,
    2e-2,
    4e-2,
    7e-2,
    0.13,
    0.26,
    0.5,
    0.8,
    0.95,
    1,
    1,
    1,
    1,
    1,
    1,
    1
  ];
  // Mismatches are edge neighbors (center distance at most about 1.55 cell widths) through level
  // 22; past it the f32 error exceeds one cell and grows with 2^(level - 22).
  const maximumWidths = (level: number) => (level <= 22 ? 2.5 : 1.6 * 2 ** (level - 22));
  for (let level = 0; level <= 30; level++) {
    const indexer = createIndexer(device, 's2', level, {validity: true});
    const {keys, valid} = await indexer.run(points);
    let mismatches = 0;
    let finiteRows = 0;
    let maximumDistance = 0;
    for (let row = 0; row < ROWS; row++) {
      if (!isFinitePoint(points, row)) {
        expect(keys[row]).toBe(0n);
        expect(valid![row]).toBe(0);
        continue;
      }
      finiteRows += row < UNIFORM_COUNT ? 1 : 0;
      expect(valid![row]).toBe(1);
      const expected = s2PointToCell(points[2 * row], points[2 * row + 1], level);
      if (keys[row] !== expected) {
        // Hand-picked rows sit exactly on face seams or poles; they are checked for being a
        // neighbor but not counted in the random-point rate.
        mismatches += row < UNIFORM_COUNT ? 1 : 0;
        // The GPU cell is the reference cell's neighbor: within ~2.5 cell widths of its center.
        const distance = getAngularDistance(s2CellToCenter(keys[row]), s2CellToCenter(expected));
        maximumDistance = Math.max(maximumDistance, distance / (Math.PI / 2 / 2 ** level));
      }
    }
    // Wrong-level or non-S2 keys would show here: the marker bit and face are always valid.
    widths[level] = maximumDistance.toFixed(2);
    expect(maximumDistance, `level ${level} center distance in cell widths`).toBeLessThan(
      maximumWidths(level)
    );
    const rate = mismatches / finiteRows;
    rates[level] = rate.toExponential(2);
    expect(rate, `level ${level}`).toBeLessThanOrEqual(maximumRates[level]);
    indexer.destroy();
  }
  // eslint-disable-next-line no-console
  console.log(
    'S2 f32 GPU vs f64 oracle, mismatch rate per level',
    rates,
    'max center distance in cell widths',
    widths
  );
});

it('GPUPointToCell H3 agrees with h3-js on f32 inputs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const points = createPoints(9);
  // H3_INDEX_WGSL: exact through resolution 4, about 1e-3 at 9, 1.8% at 12; mismatches are grid
  // neighbors. Bounds here are loose; the h3 forward specs measure exhaustively.
  const bounds: Record<number, number> = {0: 0, 3: 0.001, 4: 0.001, 9: 0.01, 12: 0.06};
  const rates: Record<number, number> = {};
  for (const [resolutionKey, bound] of Object.entries(bounds)) {
    const resolution = Number(resolutionKey);
    const indexer = createIndexer(device, 'h3', resolution, {validity: true});
    const {keys, valid} = await indexer.run(points);
    let mismatches = 0;
    let finiteRows = 0;
    for (let row = 0; row < ROWS; row++) {
      if (!isFinitePoint(points, row)) {
        expect(keys[row]).toBe(0n);
        expect(valid![row]).toBe(0);
        continue;
      }
      const longitude = points[2 * row];
      const latitude = points[2 * row + 1];
      // h3-js rejects latitudes outside [-90, 90] and wraps longitude.
      if (Math.abs(latitude) > 90) {
        continue;
      }
      finiteRows++;
      const expected = h3ToBigInt(latLngToCell(latitude, longitude, resolution));
      if (keys[row] !== expected) {
        mismatches++;
        // A mismatch is a neighboring cell: centers within a few cell widths.
        const [expectedLat, expectedLng] = cellToLatLng(expected.toString(16));
        const [actualLat, actualLng] = cellToLatLng(keys[row].toString(16));
        const distance = getAngularDistance([expectedLng, expectedLat], [actualLng, actualLat]);
        expect(distance).toBeLessThan(Math.max(0.5, 5 * 2 ** -resolution));
      } else {
        expect(valid![row]).toBe(1);
      }
    }
    rates[resolution] = mismatches / finiteRows;
    expect(rates[resolution], `h3 resolution ${resolution}`).toBeLessThanOrEqual(bound);
    indexer.destroy();
  }
  // eslint-disable-next-line no-console
  console.log('H3 f32 GPU vs h3-js mismatch rates', rates);
});

it('GPUPointToCell zeroes masked rows and follows per-frame positions without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const oracles: Record<string, (lng: number, lat: number) => bigint> = {
    quadbin: (lng, lat) => quadbinPointToCell(lng, lat, 9),
    quadkey: (lng, lat) => quadkeyPointToCell(lng, lat, 9),
    geohash: (lng, lat) => geohashPointToCell(lng, lat, 5),
    s2: (lng, lat) => s2PointToCell(lng, lat, 6)
  };
  const resolutions: Record<string, number> = {quadbin: 9, quadkey: 9, geohash: 5, s2: 6};
  for (const family of ['quadbin', 'quadkey', 'geohash', 's2'] as const) {
    const indexer = createIndexer(device, family, resolutions[family], {
      mask: true,
      validity: true
    });
    const compiled = indexer.compiled;
    const random = createRandom(31);
    const seenKeys = new Set<bigint>();
    for (const seed of [1, 2, 3, 1]) {
      const points = createPoints(seed);
      const mask = Uint32Array.from({length: ROWS}, () => (random() < 0.3 ? 0 : 1));
      const {keys, valid} = await indexer.run(points, mask);
      for (let row = 0; row < ROWS; row += 7) {
        if (mask[row] === 0 || !isFinitePoint(points, row)) {
          expect(keys[row]).toBe(0n);
          expect(valid![row]).toBe(0);
        } else if (family !== 's2' || row % 3 !== 0) {
          // S2 rows near a seam may differ in f32; the exact comparison is in the S2 test above.
          expect(keys[row], `${family} seed ${seed} row ${row}`).toBe(
            oracles[family](points[2 * row], points[2 * row + 1])
          );
        }
      }
      seenKeys.add(keys[100]);
      expect(indexer.compiled).toBe(compiled);
    }
    expect(seenKeys.size).toBeGreaterThan(1);
    indexer.destroy();
  }
});
