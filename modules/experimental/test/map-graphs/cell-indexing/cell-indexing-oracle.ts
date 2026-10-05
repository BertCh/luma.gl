// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU references for GPUPointToCell and GPUCellGeometry: quadkey (reusing the Quadbin tile
// functions), geohash (f64 bisection) and S2 (S2CellId::FromFaceIJ with BigInt). No S2, geohash or
// quadkey package is installed. Quadbin lives in ../cell-aggregation/cell-aggregation-oracle.

import {getQuadbinTileX, getQuadbinTileY} from '../cell-aggregation/cell-aggregation-oracle';

export {quadbinPointToCell} from '../cell-aggregation/cell-aggregation-oracle';

// ---------------------------------------------------------------------------------------------
// Quadkey

/** Packs a Web Mercator tile as a quadkey key: `zoom << 58 | base-4 digits` (digit = x | y << 1). */
export function quadkeyTileToCell(x: number, y: number, zoom: number): bigint {
  let digits = 0n;
  for (let bit = 0; bit < zoom; bit++) {
    digits |= BigInt((x >>> bit) & 1) << BigInt(2 * bit);
    digits |= BigInt((y >>> bit) & 1) << BigInt(2 * bit + 1);
  }
  return (BigInt(zoom) << 58n) | digits;
}

/** Quadkey of an f32 longitude/latitude with the integer Quadbin tile functions. */
export function quadkeyPointToCell(longitude: number, latitude: number, zoom: number): bigint {
  return quadkeyTileToCell(getQuadbinTileX(longitude, zoom), getQuadbinTileY(latitude, zoom), zoom);
}

/** Tile column, row and zoom of a packed quadkey. */
export function quadkeyCellToTile(cell: bigint): {x: number; y: number; z: number} {
  const z = Number(cell >> 58n);
  let x = 0;
  let y = 0;
  for (let bit = 0; bit < z; bit++) {
    x |= Number((cell >> BigInt(2 * bit)) & 1n) << bit;
    y |= Number((cell >> BigInt(2 * bit + 1)) & 1n) << bit;
  }
  return {x: x >>> 0, y: y >>> 0, z};
}

/** Bing quadkey string such as `'0231'` of a packed quadkey. */
export function quadkeyCellToString(cell: bigint): string {
  const z = Number(cell >> 58n);
  let text = '';
  for (let digit = z - 1; digit >= 0; digit--) {
    text += String(Number((cell >> BigInt(2 * digit)) & 3n));
  }
  return text;
}

/** Packed quadkey of a Bing quadkey string. */
export function quadkeyStringToCell(text: string): bigint {
  let digits = 0n;
  for (const character of text) {
    digits = (digits << 2n) | BigInt(Number(character));
  }
  return (BigInt(text.length) << 58n) | digits;
}

/** Bounds `[west, south, east, north]` of a Web Mercator tile in degrees (f64). */
export function webMercatorTileBounds(
  x: number,
  y: number,
  z: number
): [number, number, number, number] {
  const scale = 2 ** z;
  const latitude = (row: number) =>
    (Math.atan(Math.sinh(Math.PI * (1 - (2 * row) / scale))) * 180) / Math.PI;
  return [(x / scale) * 360 - 180, latitude(y + 1), ((x + 1) / scale) * 360 - 180, latitude(y)];
}

/** Center `[longitude, latitude]` of a Web Mercator tile: the middle of the tile in projected space. */
export function webMercatorTileCenter(x: number, y: number, z: number): [number, number] {
  const scale = 2 ** z;
  return [
    ((x + 0.5) / scale) * 360 - 180,
    (Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + 0.5)) / scale))) * 180) / Math.PI
  ];
}

/** Bounds `[west, south, east, north]` of a packed quadkey in degrees (f64 Web Mercator). */
export function quadkeyCellToBounds(cell: bigint): [number, number, number, number] {
  const {x, y, z} = quadkeyCellToTile(cell);
  return webMercatorTileBounds(x, y, z);
}

// ---------------------------------------------------------------------------------------------
// Geohash

const GEOHASH_ALPHABET = '0123456789bcdefghjkmnpqrstuvwxyz';

/**
 * Geohash of a longitude/latitude by bisection (the reference algorithm), packed as `length << 60`
 * plus the 5-bit character codes. Bisection midpoints are dyadic, so f64 arithmetic is exact for
 * any f32 input. Coordinates beyond the range clamp into the first or last bin.
 */
export function geohashPointToCell(longitude: number, latitude: number, length: number): bigint {
  let west = -180;
  let east = 180;
  let south = -90;
  let north = 90;
  let bits = 0n;
  for (let bit = 0; bit < 5 * length; bit++) {
    let isOne: boolean;
    if (bit % 2 === 0) {
      const midpoint = (west + east) / 2;
      isOne = longitude >= midpoint;
      if (isOne) {
        west = midpoint;
      } else {
        east = midpoint;
      }
    } else {
      const midpoint = (south + north) / 2;
      isOne = latitude >= midpoint;
      if (isOne) {
        south = midpoint;
      } else {
        north = midpoint;
      }
    }
    bits = (bits << 1n) | (isOne ? 1n : 0n);
  }
  return (BigInt(length) << 60n) | bits;
}

/** Geohash string of a packed geohash. */
export function geohashCellToString(cell: bigint): string {
  const length = Number(cell >> 60n);
  let text = '';
  for (let character = length - 1; character >= 0; character--) {
    text += GEOHASH_ALPHABET[Number((cell >> BigInt(5 * character)) & 31n)];
  }
  return text;
}

/** Packed geohash of a geohash string. */
export function geohashStringToCell(text: string): bigint {
  let bits = 0n;
  for (const character of text) {
    bits = (bits << 5n) | BigInt(GEOHASH_ALPHABET.indexOf(character));
  }
  return (BigInt(text.length) << 60n) | bits;
}

/** Bounds `[west, south, east, north]` of a packed geohash. */
export function geohashCellToBounds(cell: bigint): [number, number, number, number] {
  const length = Number(cell >> 60n);
  let west = -180;
  let east = 180;
  let south = -90;
  let north = 90;
  for (let bit = 0; bit < 5 * length; bit++) {
    const isOne = ((cell >> BigInt(5 * length - 1 - bit)) & 1n) === 1n;
    if (bit % 2 === 0) {
      const midpoint = (west + east) / 2;
      if (isOne) {
        west = midpoint;
      } else {
        east = midpoint;
      }
    } else {
      const midpoint = (south + north) / 2;
      if (isOne) {
        south = midpoint;
      } else {
        north = midpoint;
      }
    }
  }
  return [west, south, east, north];
}

// ---------------------------------------------------------------------------------------------
// S2 (S2CellId::FromFaceIJ, S2CellId::ToFaceIJOrientation)

const S2_MAX_LEVEL = 30;
const S2_SWAP_MASK = 1;
const S2_INVERT_MASK = 2;
/** S2 `kPosToIJ[orientation][position]` with `ij = (i << 1) | j`. */
const S2_POS_TO_IJ = [
  [0, 1, 3, 2],
  [0, 2, 3, 1],
  [3, 2, 0, 1],
  [3, 1, 0, 2]
];
/** S2 `kPosToOrientation[position]`. */
const S2_POS_TO_ORIENTATION = [S2_SWAP_MASK, 0, 0, S2_SWAP_MASK | S2_INVERT_MASK];

/** S2CellId of face cell `(face, i, j)` where `i`, `j` are level-30 leaf coordinates, at `level`. */
export function s2CellFromFaceIJ(face: number, i: number, j: number, level: number): bigint {
  let orientation = face & S2_SWAP_MASK;
  let position = 0n;
  for (let step = 0; step < level; step++) {
    const bit = S2_MAX_LEVEL - 1 - step;
    const ij = (((i >>> bit) & 1) << 1) | ((j >>> bit) & 1);
    const digit = S2_POS_TO_IJ[orientation].indexOf(ij);
    position = (position << 2n) | BigInt(digit);
    orientation ^= S2_POS_TO_ORIENTATION[digit];
  }
  const shift = BigInt(2 * (S2_MAX_LEVEL - level));
  return (BigInt(face) << 61n) | (position << (shift + 1n)) | (1n << shift);
}

/** Face, level and level-`level` cell coordinates of an S2CellId. */
export function s2CellToFaceIJ(cell: bigint): {face: number; i: number; j: number; level: number} {
  const face = Number(cell >> 61n);
  let trailingZeros = 0;
  while (((cell >> BigInt(trailingZeros)) & 1n) === 0n) {
    trailingZeros++;
  }
  const level = S2_MAX_LEVEL - trailingZeros / 2;
  let orientation = face & S2_SWAP_MASK;
  let i = 0;
  let j = 0;
  for (let step = 0; step < level; step++) {
    const digit = Number((cell >> BigInt(60 - 2 * step - 1)) & 3n);
    const ij = S2_POS_TO_IJ[orientation][digit];
    i = (i << 1) | (ij >> 1);
    j = (j << 1) | (ij & 1);
    orientation ^= S2_POS_TO_ORIENTATION[digit];
  }
  return {face, i, j, level};
}

function xyzToFace(x: number, y: number, z: number): number {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  const axis = ax > ay ? (ax > az ? 0 : 2) : ay > az ? 1 : 2;
  const component = [x, y, z][axis];
  return axis + (component < 0 ? 3 : 0);
}

function xyzToUV(face: number, x: number, y: number, z: number): [number, number] {
  switch (face) {
    case 0:
      return [y / x, z / x];
    case 1:
      return [-x / y, z / y];
    case 2:
      return [-x / z, -y / z];
    case 3:
      return [z / x, y / x];
    case 4:
      return [z / y, -x / y];
    default:
      return [-y / z, -x / z];
  }
}

function faceUVToXYZ(face: number, u: number, v: number): [number, number, number] {
  switch (face) {
    case 0:
      return [1, u, v];
    case 1:
      return [-u, 1, v];
    case 2:
      return [-u, -v, 1];
    case 3:
      return [-1, -v, -u];
    case 4:
      return [v, -1, -u];
    default:
      return [v, u, -1];
  }
}

function uvToST(uv: number): number {
  return uv >= 0 ? 0.5 * Math.sqrt(1 + 3 * uv) : 1 - 0.5 * Math.sqrt(1 - 3 * uv);
}

function stToUV(st: number): number {
  return st >= 0.5 ? (4 * st * st - 1) / 3 : (1 - 4 * (1 - st) * (1 - st)) / 3;
}

function stToIJ(st: number): number {
  return Math.max(0, Math.min(2 ** S2_MAX_LEVEL - 1, Math.floor(2 ** S2_MAX_LEVEL * st)));
}

/** Face of a longitude/latitude in degrees (f64). */
export function s2PointToFace(longitude: number, latitude: number): number {
  const {x, y, z} = lngLatToXYZ(longitude, latitude);
  return xyzToFace(x, y, z);
}

function lngLatToXYZ(longitude: number, latitude: number) {
  const lng = (longitude * Math.PI) / 180;
  const lat = (latitude * Math.PI) / 180;
  const cosLatitude = Math.cos(lat);
  return {x: cosLatitude * Math.cos(lng), y: cosLatitude * Math.sin(lng), z: Math.sin(lat)};
}

/** Leaf `(face, i, j)` of a longitude/latitude in degrees, in f64. */
export function s2PointToFaceIJ(
  longitude: number,
  latitude: number
): {face: number; i: number; j: number} {
  const {x, y, z} = lngLatToXYZ(longitude, latitude);
  const face = xyzToFace(x, y, z);
  const [u, v] = xyzToUV(face, x, y, z);
  return {face, i: stToIJ(uvToST(u)), j: stToIJ(uvToST(v))};
}

/** S2CellId at `level` of a longitude/latitude in degrees (f64 pipeline, the accuracy reference). */
export function s2PointToCell(longitude: number, latitude: number, level: number): bigint {
  const {face, i, j} = s2PointToFaceIJ(longitude, latitude);
  return s2CellFromFaceIJ(face, i, j, level);
}

/** Center of an S2 cell as `[longitude, latitude]` degrees (f64). */
export function s2CellToCenter(cell: bigint): [number, number] {
  const {face, i, j, level} = s2CellToFaceIJ(cell);
  const scale = 2 ** level;
  const [x, y, z] = faceUVToXYZ(face, stToUV((i + 0.5) / scale), stToUV((j + 0.5) / scale));
  return [(Math.atan2(y, x) * 180) / Math.PI, (Math.atan2(z, Math.hypot(x, y)) * 180) / Math.PI];
}

/** Corner vertices of an S2 cell in the dggs order (i, j), (i, j+1), (i+1, j+1), (i+1, j). */
export function s2CellToBoundary(cell: bigint): [number, number][] {
  const {face, i, j, level} = s2CellToFaceIJ(cell);
  const scale = 2 ** level;
  return [
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
  ].map(([di, dj]) => {
    const [x, y, z] = faceUVToXYZ(face, stToUV((i + di) / scale), stToUV((j + dj) / scale));
    return [(Math.atan2(y, x) * 180) / Math.PI, (Math.atan2(z, Math.hypot(x, y)) * 180) / Math.PI];
  });
}

/** S2 token: lowercase hex of the id with trailing zeros removed. */
export function s2CellToToken(cell: bigint): string {
  return cell === 0n ? 'X' : cell.toString(16).padStart(16, '0').replace(/0+$/, '');
}

/** S2CellId of a token. */
export function s2TokenToCell(token: string): bigint {
  return BigInt(`0x${token.padEnd(16, '0')}`);
}

/** Great-circle angular distance in radians between two `[longitude, latitude]` degree pairs. */
export function getAngularDistance(a: readonly number[], b: readonly number[]): number {
  const toRadians = Math.PI / 180;
  const sinHalfLat = Math.sin(((b[1] - a[1]) * toRadians) / 2);
  const sinHalfLng = Math.sin(((b[0] - a[0]) * toRadians) / 2);
  const h =
    sinHalfLat ** 2 + Math.cos(a[1] * toRadians) * Math.cos(b[1] * toRadians) * sinHalfLng ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** Little-endian `(low, high)` words of a key. */
export function splitKey(cell: bigint): [number, number] {
  return [Number(cell & 0xffffffffn), Number(cell >> 32n)];
}

/** Key of little-endian `(low, high)` words. */
export function joinKey(low: number, high: number): bigint {
  return (BigInt(high >>> 0) << 32n) | BigInt(low >>> 0);
}
