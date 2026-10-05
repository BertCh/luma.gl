// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU reference for GPUCellAggregation, GPUCellRollup and GPUCellPyramid. No quadbin package is
// installed, so Quadbin follows quadbin-py / quadbin-js (CARTO) with BigInt; H3 uses h3-js.

import {cellToParent as h3CellToParent, getResolution as h3GetResolution, isValidCell} from 'h3-js';
import {QUADBIN_FIXED_POINT_CONSTANTS} from '../../../src/map-graphs/cell-aggregation/cell-keys';
import {CELL_MAXIMUM_SCALED_VALUE} from '../../../src/map-graphs/cell-aggregation/cell-table';

const f = Math.fround;
const Q = QUADBIN_FIXED_POINT_CONSTANTS;
const FOOTER = 0xfffffffffffffn;
const QUADBIN_HEADER = 0x4000000000000000n | (1n << 59n);

export type CellFamily = 'quadbin' | 'h3';

// ---------------------------------------------------------------------------------------------
// Quadbin

/** Exact tile column: `floor((lng + 180) * 2^z / 360) mod 2^z` for the f32 longitude. */
export function getQuadbinTileX(longitude: number, resolution: number): number {
  const clipped = Math.min(Math.max(f(longitude), -180), 180);
  const scale = 2n ** 149n;
  const numerator = (BigInt(clipped * 2 ** 149) + 180n * scale) * 2n ** BigInt(resolution);
  const column = numerator / (360n * scale);
  return Number(column % 2n ** BigInt(resolution));
}

function getFloat32Bits(value: number): number {
  const buffer = new DataView(new ArrayBuffer(4));
  buffer.setFloat32(0, value);
  return buffer.getUint32(0);
}

const U32 = 0xffffffffn;

function multiplyHigh(a: bigint, b: bigint): bigint {
  return (a * b) >> 32n;
}

function getLatitudeFixed(magnitudeBits: number): bigint {
  const bits = Math.min(magnitudeBits, Q.maximumLatitudeBits);
  const exponentBits = bits >>> 23;
  let mantissa = bits & 0x7fffff;
  if (exponentBits !== 0) {
    mantissa |= 0x800000;
  }
  const shift = Math.max(exponentBits, 1) - 125;
  if (shift >= 0) {
    return (BigInt(mantissa) << BigInt(shift)) & U32;
  }
  return shift > -32 ? BigInt(mantissa >>> -shift) : 0n;
}

function getSinFixed(t: bigint): bigint {
  const t2 = multiplyHigh(t, t);
  let bracket = BigInt(Q.sinCoefficients[0]);
  for (let index = 1; index < Q.sinCoefficients.length; index++) {
    bracket = (BigInt(Q.sinCoefficients[index]) - multiplyHigh(t2, bracket)) & U32;
  }
  return (t - multiplyHigh(t, multiplyHigh(t2, bracket))) & U32;
}

function getCosFixed(t: bigint): bigint {
  const t2 = multiplyHigh(t, t);
  let bracket = BigInt(Q.cosCoefficients[0]);
  for (let index = 1; index < Q.cosCoefficients.length; index++) {
    bracket = (BigInt(Q.cosCoefficients[index]) - multiplyHigh(t2, bracket)) & U32;
  }
  return (2n ** 32n - multiplyHigh(t2, bracket)) & U32;
}

function getNegativeLnFixed(v: bigint): bigint {
  const leadingZeros = BigInt(Math.clz32(Number(v)));
  const m = (v << leadingZeros) & U32;
  const difference = (2n ** 32n - m) & U32;
  const z = (difference << 31n) / (0x80000000n + (m >> 1n));
  const z2 = multiplyHigh(z, z);
  let bracket = BigInt(Q.atanhCoefficients[0]);
  for (let index = 1; index < Q.atanhCoefficients.length; index++) {
    bracket = (BigInt(Q.atanhCoefficients[index]) + multiplyHigh(z2, bracket)) & U32;
  }
  const atanhZ = (z + multiplyHigh(z, multiplyHigh(z2, bracket))) & U32;
  return 2n * atanhZ + leadingZeros * BigInt(Q.ln2);
}

/** Tile row with the fixed-point integer algorithm of the WGSL kernel (bit exact). */
export function getQuadbinTileY(latitude: number, resolution: number): number {
  if (resolution === 0) {
    return 0;
  }
  const bits = getFloat32Bits(latitude);
  const magnitudeBits = bits & 0x7fffffff;
  const half = 2 ** (resolution - 1);
  let row = half;
  if (magnitudeBits !== 0) {
    const colatitude = BigInt(Q.ninetyDegrees) - getLatitudeFixed(magnitudeBits);
    const t = ((colatitude * BigInt(Q.halfColatitudeScale)) >> 31n) & U32;
    const sinPart = getNegativeLnFixed(getSinFixed(t));
    const cosPart = getNegativeLnFixed(getCosFixed(t));
    const psi = sinPart > cosPart ? sinPart - cosPart : 0n;
    const fraction = multiplyHigh((psi >> 2n) & U32, BigInt(Q.inverseTwoPi));
    const fractionBits = BigInt(32 - resolution);
    const whole = Number(fraction >> fractionBits);
    const hasFraction = (fraction & ((1n << fractionBits) - 1n)) !== 0n;
    row = bits >>> 31 ? half + whole : half - Math.max(whole + (hasFraction ? 1 : 0), 1);
  }
  return Math.min(Math.max(row, 0), 2 ** resolution - 1);
}

/** Tile row with the f64 formula of quadbin-py `point_to_tile` (the accuracy reference). */
export function getQuadbinTileYFloat64(latitude: number, resolution: number): number {
  const clipped = Math.min(Math.max(latitude, -85.051129), 85.051129);
  const sinLatitude = Math.sin((clipped * Math.PI) / 180);
  const tileCount = 2 ** resolution;
  const y = tileCount * (0.5 - (0.25 * Math.log((1 + sinLatitude) / (1 - sinLatitude))) / Math.PI);
  return Math.min(Math.max(Math.floor(y), 0), tileCount - 1);
}

/** quadbin-js `tileToCell`. */
export function quadbinTileToCell(x: number, y: number, z: number): bigint {
  let morton = 0n;
  for (let bit = 0; bit < z; bit++) {
    morton |= BigInt((x >>> bit) & 1) << BigInt(2 * bit);
    morton |= BigInt((y >>> bit) & 1) << BigInt(2 * bit + 1);
  }
  const zz = BigInt(z);
  return QUADBIN_HEADER | (zz << 52n) | (morton << (52n - 2n * zz)) | (FOOTER >> (2n * zz));
}

/** Quadbin of an f32 longitude/latitude with the kernel's algorithm. */
export function quadbinPointToCell(
  longitude: number,
  latitude: number,
  resolution: number
): bigint {
  return quadbinTileToCell(
    getQuadbinTileX(longitude, resolution),
    getQuadbinTileY(latitude, resolution),
    resolution
  );
}

/** Quadbin with the f64 row formula of quadbin-py. */
export function quadbinPointToCellFloat64(
  longitude: number,
  latitude: number,
  resolution: number
): bigint {
  return quadbinTileToCell(
    getQuadbinTileX(longitude, resolution),
    getQuadbinTileYFloat64(latitude, resolution),
    resolution
  );
}

/** quadbin-js `getResolution`. */
export function quadbinGetResolution(cell: bigint): number {
  return Number((cell >> 52n) & 0x1fn);
}

/** quadbin-js `cellToParent`. */
export function quadbinCellToParent(cell: bigint, resolution: number): bigint {
  const z = BigInt(resolution);
  return (cell & ~(0x1fn << 52n)) | (z << 52n) | (FOOTER >> (2n * z));
}

/** quadbin-py `is_valid_cell`. */
export function quadbinIsValidCell(cell: bigint): boolean {
  const header = 0x4000000000000000n;
  const mode = (cell >> 59n) & 7n;
  const resolution = (cell >> 52n) & 0x1fn;
  const mask = FOOTER >> (resolution << 1n);
  return (
    cell >= 0n &&
    cell < 2n ** 63n &&
    (cell & header) === header &&
    mode === 1n &&
    resolution <= 26n &&
    (cell & mask) === mask
  );
}

/** quadbin-js `cellToTile`. */
export function quadbinCellToTile(cell: bigint): {
  x: number;
  y: number;
  z: number;
} {
  const z = quadbinGetResolution(cell);
  const morton = (cell >> BigInt(52 - 2 * z)) & ((1n << BigInt(2 * z)) - 1n);
  let x = 0;
  let y = 0;
  for (let bit = 0; bit < z; bit++) {
    x |= Number((morton >> BigInt(2 * bit)) & 1n) << bit;
    y |= Number((morton >> BigInt(2 * bit + 1)) & 1n) << bit;
  }
  return {x: x >>> 0, y: y >>> 0, z};
}

// ---------------------------------------------------------------------------------------------
// H3 and shared key helpers

export function h3ToBigInt(index: string): bigint {
  return BigInt(`0x${index}`);
}

export function bigIntToH3(cell: bigint): string {
  return cell.toString(16);
}

/** Splits a 64-bit key into little-endian `[low, high]` words. */
export function splitCellKey(cell: bigint): [number, number] {
  return [Number(cell & 0xffffffffn), Number((cell >> 32n) & 0xffffffffn)];
}

export function joinCellKey(low: number, high: number): bigint {
  return (BigInt(high >>> 0) << 32n) | BigInt(low >>> 0);
}

export function isValidCellKey(family: CellFamily, cell: bigint): boolean {
  if (family === 'quadbin') {
    return quadbinIsValidCell(cell);
  }
  return cell >= 0n && cell < 2n ** 64n && isValidCell(bigIntToH3(cell));
}

export function getCellResolution(family: CellFamily, cell: bigint): number {
  return family === 'quadbin' ? quadbinGetResolution(cell) : h3GetResolution(bigIntToH3(cell));
}

export function getCellParent(family: CellFamily, cell: bigint, resolution: number): bigint {
  return family === 'quadbin'
    ? quadbinCellToParent(cell, resolution)
    : h3ToBigInt(h3CellToParent(bigIntToH3(cell), resolution));
}

// ---------------------------------------------------------------------------------------------
// Aggregation

/** One oracle table row. */
export type OracleCell = {
  key: bigint;
  count: number;
  /** Exact fixed-point sum. */
  sum: bigint;
  minimum: number;
  maximum: number;
};

/** `roundHalfEven(fround(value * sumScale))`, saturated like the kernel. */
export function getScaledValue(value: number, sumScale: number): bigint {
  const product = f(value * sumScale);
  const clamped = Math.min(
    Math.max(product, -CELL_MAXIMUM_SCALED_VALUE),
    CELL_MAXIMUM_SCALED_VALUE
  );
  const floor = Math.floor(clamped);
  const difference = clamped - floor;
  let rounded =
    difference > 0.5 ? floor + 1 : difference < 0.5 ? floor : floor % 2 === 0 ? floor : floor + 1;
  if (Object.is(rounded, -0)) {
    rounded = 0;
  }
  return BigInt(rounded);
}

/** Two's-complement 64-bit wrap of a BigInt. */
export function wrapInt64(value: bigint): bigint {
  return BigInt.asIntN(64, value);
}

/** Order-preserving key comparison treating `-0` below `+0`. */
function isLess(left: number, right: number): boolean {
  return (
    left < right || (left === 0 && right === 0 && Object.is(left, -0) && !Object.is(right, -0))
  );
}

export type OracleRowsInput = {
  family: CellFamily;
  resolution: number;
  /** Per-row key at the row's own resolution, or `null` for a skipped row. */
  keys: readonly (bigint | null)[];
  values?: Float32Array;
  mask?: Uint32Array;
  sumScale: number;
};

/** Aggregates keyed rows at `resolution`, ascending by key. */
export function aggregateCellsOnCPU(input: OracleRowsInput): OracleCell[] {
  const cells = new Map<bigint, OracleCell>();
  for (let row = 0; row < input.keys.length; row++) {
    const key = input.keys[row];
    if (key === null || (input.mask && input.mask[row] === 0)) {
      continue;
    }
    const value = input.values ? input.values[row] : 0;
    if (input.values && !Number.isFinite(value)) {
      continue;
    }
    if (!isValidCellKey(input.family, key)) {
      continue;
    }
    if (getCellResolution(input.family, key) < input.resolution) {
      continue;
    }
    const parent = getCellParent(input.family, key, input.resolution);
    let cell = cells.get(parent);
    if (!cell) {
      cell = {key: parent, count: 0, sum: 0n, minimum: value, maximum: value};
      cells.set(parent, cell);
    }
    cell.count++;
    cell.sum = wrapInt64(cell.sum + getScaledValue(value, input.sumScale));
    if (isLess(value, cell.minimum)) {
      cell.minimum = value;
    }
    if (isLess(cell.maximum, value)) {
      cell.maximum = value;
    }
  }
  return [...cells.values()].sort((left, right) => (left.key < right.key ? -1 : 1));
}

/** Quadbin keys of f32 points with the kernel's algorithm (`null` for NaN coordinates). */
export function getQuadbinPointKeys(
  positions: Float32Array,
  resolution: number
): (bigint | null)[] {
  const keys: (bigint | null)[] = [];
  for (let row = 0; row < positions.length / 2; row++) {
    const longitude = positions[2 * row];
    const latitude = positions[2 * row + 1];
    keys.push(
      Number.isNaN(longitude) || Number.isNaN(latitude)
        ? null
        : quadbinPointToCell(longitude, latitude, resolution)
    );
  }
  return keys;
}
