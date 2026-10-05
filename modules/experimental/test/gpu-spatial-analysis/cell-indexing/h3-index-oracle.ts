// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// The forward H3 algorithm (geoToFaceIjk, faceIjkToH3) is derived from Uber's Apache-2.0 H3.

import {dggs} from '@luma.gl/shadertools';
import {cellToBoundary, getHexagonEdgeLengthAvg, getPentagons, latLngToCell} from 'h3-js';
import {
  CELL_INDEX_H3_BASE_CELL_PACKED_TABLE,
  CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES
} from '../../../src/gpu-spatial-analysis/cell-indexing/h3-index-tables';

/** Lookup tables consumed by the oracle; the generator passes candidate tables here. */
export type H3OracleTables = {
  /** `[face][i][j][k]` flattened to `face * 27 + i * 9 + j * 3 + k`; `baseCell | rotation << 8`, or -1. */
  packedBaseCells: readonly number[];
  /** Pentagon base cell -> the two faces whose rotation is clockwise. */
  cwOffsetFaces: ReadonlyMap<number, readonly [number, number]>;
};

/** The tables embedded in the shader, in oracle form. */
export const SHIPPED_H3_ORACLE_TABLES: H3OracleTables = {
  packedBaseCells: CELL_INDEX_H3_BASE_CELL_PACKED_TABLE,
  cwOffsetFaces: new Map(
    CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES.map(([baseCell, faceA, faceB]) => [
      baseCell,
      [faceA, faceB] as const
    ])
  )
};

const RES0_U_GNOMONIC = 0.381966011250105;
const SQRT7 = Math.sqrt(7);
const AP7_ROT_RADS = 0.3334731722518321;
const SIN60 = 0.8660254037844386;
const PENTAGON_BASE_CELLS = new Set([4, 14, 24, 38, 49, 58, 63, 72, 83, 97, 107, 117]);

type Vec3 = [number, number, number];

function parseVec2Table(functionName: string): number[][] {
  const start = dggs.source.indexOf(`fn ${functionName}`);
  const end = dggs.source.indexOf('return values', start);
  const block = dggs.source.slice(start, end);
  return [...block.matchAll(/vec[23]f\(([^)]*)\)/g)].map(match => match[1].split(',').map(Number));
}

const FACE_CENTER_GEO = parseVec2Table('dggs_h3_get_face_center_geo');
const FACE_AXIS_AZIMUTHS = parseVec2Table('dggs_h3_get_face_axis_azimuths');

/** Float64 face center / hex i axis / hex j axis unit vectors, derived from center and azimuth. */
export const H3_FACE_BASES: Vec3[][] = FACE_CENTER_GEO.map(([latitude, longitude], face) => {
  const sinLat = Math.sin(latitude);
  const cosLat = Math.cos(latitude);
  const sinLng = Math.sin(longitude);
  const cosLng = Math.cos(longitude);
  const center: Vec3 = [cosLat * cosLng, cosLat * sinLng, sinLat];
  const north: Vec3 = [-sinLat * cosLng, -sinLat * sinLng, cosLat];
  const east: Vec3 = [-sinLng, cosLng, 0];
  const direction = (azimuth: number): Vec3 => [
    Math.cos(azimuth) * north[0] + Math.sin(azimuth) * east[0],
    Math.cos(azimuth) * north[1] + Math.sin(azimuth) * east[1],
    Math.cos(azimuth) * north[2] + Math.sin(azimuth) * east[2]
  ];
  const axisAzimuth = FACE_AXIS_AZIMUTHS[face][0];
  return [center, direction(axisAzimuth), direction(axisAzimuth - Math.PI / 2)];
});

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Converts degrees to a float64 unit vector. */
export function lngLatToUnitVector(lngDegrees: number, latDegrees: number): Vec3 {
  const lng = (lngDegrees * Math.PI) / 180;
  const lat = (latDegrees * Math.PI) / 180;
  return [Math.cos(lat) * Math.cos(lng), Math.cos(lat) * Math.sin(lng), Math.sin(lat)];
}

/** Hex2d position of a unit vector on its nearest icosahedron face at `resolution`. */
export function unitVectorToFaceHex2d(
  point: Vec3,
  resolution: number
): {face: number; x: number; y: number} {
  let face = 0;
  let best = -2;
  for (let candidate = 0; candidate < 20; candidate++) {
    const alignment = dot3(point, H3_FACE_BASES[candidate][0]);
    if (alignment > best) {
      best = alignment;
      face = candidate;
    }
  }
  const [center, axisI, axisJ] = H3_FACE_BASES[face];
  const scale = 1 / (dot3(point, center) * RES0_U_GNOMONIC);
  let x = dot3(point, axisI) * scale;
  let y = dot3(point, axisJ) * scale;
  if (resolution % 2 === 1) {
    const cosRot = Math.cos(AP7_ROT_RADS);
    const sinRot = Math.sin(AP7_ROT_RADS);
    [x, y] = [cosRot * x + sinRot * y, -sinRot * x + cosRot * y];
  }
  const resolutionScale = SQRT7 ** resolution;
  return {face, x: x * resolutionScale, y: y * resolutionScale};
}

/** Integer IJK coordinates. */
export type Ijk = [number, number, number];

/** Normalizes IJK so all components are non-negative with at least one zero. */
export function ijkNormalize([i, j, k]: Ijk): Ijk {
  if (i < 0) {
    j -= i;
    k -= i;
    i = 0;
  }
  if (j < 0) {
    i -= j;
    k -= j;
    j = 0;
  }
  if (k < 0) {
    i -= k;
    j -= k;
    k = 0;
  }
  const minimum = Math.min(i, j, k);
  return [i - minimum, j - minimum, k - minimum];
}

/** Quantizes a hex2d point to the containing cell's IJK (H3 `_hex2dToCoordIJK`). */
export function hex2dToCoordIjk(x: number, y: number): Ijk {
  const a1 = Math.abs(x);
  const a2 = Math.abs(y);
  const x2 = a2 / SIN60;
  const x1 = a1 + x2 / 2;
  const m1 = Math.trunc(x1);
  const m2 = Math.trunc(x2);
  const r1 = x1 - m1;
  const r2 = x2 - m2;
  let i: number;
  let j: number;
  if (r1 < 0.5) {
    if (r1 < 1 / 3) {
      i = m1;
      j = r2 < (1 + r1) / 2 ? m2 : m2 + 1;
    } else {
      j = r2 < 1 - r1 ? m2 : m2 + 1;
      i = 1 - r1 <= r2 && r2 < 2 * r1 ? m1 + 1 : m1;
    }
  } else if (r1 < 2 / 3) {
    j = r2 < 1 - r1 ? m2 : m2 + 1;
    i = 2 * r1 - 1 < r2 && r2 < 1 - r1 ? m1 : m1 + 1;
  } else {
    i = m1 + 1;
    j = r2 < r1 / 2 ? m2 : m2 + 1;
  }
  if (x < 0) {
    if (j % 2 === 0) {
      const axisI = j / 2;
      i -= 2 * (i - axisI);
    } else {
      const axisI = (j + 1) / 2;
      i -= 2 * (i - axisI) + 1;
    }
  }
  if (y < 0) {
    i -= Math.trunc((2 * j + 1) / 2);
    j = -j;
  }
  return ijkNormalize([i, j, 0]);
}

const roundNearestSeven = (value: number): number =>
  value >= 0 ? Math.trunc((value + 3) / 7) : Math.trunc((value - 3) / 7);

/** Aperture-7 counter-clockwise parent step (Class III child to Class II parent). */
export function upAp7([i0, j0, k0]: Ijk): Ijk {
  const i = i0 - k0;
  const j = j0 - k0;
  return ijkNormalize([roundNearestSeven(3 * i - j), roundNearestSeven(i + 2 * j), 0]);
}

/** Aperture-7 clockwise parent step (Class II child to Class III parent). */
export function upAp7r([i0, j0, k0]: Ijk): Ijk {
  const i = i0 - k0;
  const j = j0 - k0;
  return ijkNormalize([roundNearestSeven(2 * i + j), roundNearestSeven(3 * j - i), 0]);
}

/** Aperture-7 counter-clockwise child step. */
export function downAp7([i, j, k]: Ijk): Ijk {
  return ijkNormalize([3 * i + j, 3 * j + k, i + 3 * k]);
}

/** Aperture-7 clockwise child step. */
export function downAp7r([i, j, k]: Ijk): Ijk {
  return ijkNormalize([3 * i + k, i + 3 * j, j + 3 * k]);
}

const UNIT_IJK_TO_DIGIT: Record<string, number> = {
  '0,0,0': 0,
  '0,0,1': 1,
  '0,1,0': 2,
  '0,1,1': 3,
  '1,0,0': 4,
  '1,0,1': 5,
  '1,1,0': 6
};
const ROTATE_CCW = [0, 5, 3, 1, 6, 4, 2, 7];
const ROTATE_CW = [0, 3, 6, 2, 5, 1, 4, 7];

/** Result of walking an IJK at `resolution` up to the res-0 base cell lattice. */
export type H3UpWalk = {baseIjk: Ijk; digits: number[]};

/** Walks `ijk` up to resolution 0, collecting digits `[1..resolution]` (index 0 unused). */
export function walkUpToBaseIjk(ijk: Ijk, resolution: number): H3UpWalk {
  const digits: number[] = new Array(resolution + 1).fill(7);
  let current = ijk;
  for (let r = resolution - 1; r >= 0; r--) {
    const last = current;
    let lastCenter: Ijk;
    if ((r + 1) % 2 === 1) {
      current = upAp7(current);
      lastCenter = downAp7(current);
    } else {
      current = upAp7r(current);
      lastCenter = downAp7r(current);
    }
    const difference = ijkNormalize([
      last[0] - lastCenter[0],
      last[1] - lastCenter[1],
      last[2] - lastCenter[2]
    ]);
    digits[r + 1] = UNIT_IJK_TO_DIGIT[difference.join(',')] ?? 7;
  }
  return {baseIjk: current, digits};
}

function leadingNonZeroDigit(digits: number[], resolution: number): number {
  for (let r = 1; r <= resolution; r++) {
    if (digits[r] !== 0) {
      return digits[r];
    }
  }
  return 0;
}

function rotateDigits(digits: number[], resolution: number, table: number[]): number[] {
  return digits.map((digit, r) => (r >= 1 && r <= resolution ? table[digit] : digit));
}

function rotatePentagonCcw(digits: number[], resolution: number): number[] {
  let result = digits.slice();
  let foundFirstNonZero = false;
  for (let r = 1; r <= resolution; r++) {
    result[r] = ROTATE_CCW[result[r]];
    if (!foundFirstNonZero && result[r] !== 0) {
      foundFirstNonZero = true;
      if (leadingNonZeroDigit(result, resolution) === 1) {
        result = rotateDigits(result, resolution, ROTATE_CCW);
      }
    }
  }
  return result;
}

/** Packs a mode-1 H3 index from its parts. */
export function packH3Index(baseCell: number, resolution: number, digits: number[]): bigint {
  let index = (1n << 59n) | (BigInt(resolution) << 52n) | (BigInt(baseCell) << 45n);
  for (let r = 1; r <= 15; r++) {
    const digit = r <= resolution ? digits[r] : 7;
    index |= BigInt(digit) << BigInt(3 * (15 - r));
  }
  return index;
}

/** H3 `_faceIjkToH3` for a given face, IJK at `resolution`, and tables. Returns 0n when invalid. */
export function faceIjkToH3(
  face: number,
  ijk: Ijk,
  resolution: number,
  tables: H3OracleTables = SHIPPED_H3_ORACLE_TABLES
): bigint {
  const {baseIjk, digits} = walkUpToBaseIjk(ijk, resolution);
  if (baseIjk.some(component => component > 2)) {
    return 0n;
  }
  const packed = tables.packedBaseCells[face * 27 + baseIjk[0] * 9 + baseIjk[1] * 3 + baseIjk[2]];
  if (packed === undefined || packed < 0) {
    return 0n;
  }
  const baseCell = packed & 0xff;
  const rotations = packed >> 8;
  let result = digits;
  if (PENTAGON_BASE_CELLS.has(baseCell)) {
    if (leadingNonZeroDigit(result, resolution) === 1) {
      const cwFaces = tables.cwOffsetFaces.get(baseCell);
      const isCw = cwFaces !== undefined && (cwFaces[0] === face || cwFaces[1] === face);
      result = rotateDigits(result, resolution, isCw ? ROTATE_CW : ROTATE_CCW);
    }
    for (let rotation = 0; rotation < rotations; rotation++) {
      result = rotatePentagonCcw(result, resolution);
    }
  } else {
    for (let rotation = 0; rotation < rotations; rotation++) {
      result = rotateDigits(result, resolution, ROTATE_CCW);
    }
  }
  return packH3Index(baseCell, resolution, result);
}

/**
 * Float64 CPU port of H3 `latLngToCell` (face selection, gnomonic projection, hex2d quantization,
 * `_faceIjkToH3`). Returns 0n for non-finite input or resolution above 15.
 */
export function latLngToCellOracle(
  lngDegrees: number,
  latDegrees: number,
  resolution: number,
  tables: H3OracleTables = SHIPPED_H3_ORACLE_TABLES
): bigint {
  if (!Number.isFinite(lngDegrees) || !Number.isFinite(latDegrees) || resolution > 15) {
    return 0n;
  }
  const {face, x, y} = unitVectorToFaceHex2d(
    lngLatToUnitVector(lngDegrees, latDegrees),
    resolution
  );
  return faceIjkToH3(face, hex2dToCoordIjk(x, y), resolution, tables);
}

/** Seeded 32-bit generator (mulberry32) for reproducible test points. */
export function createH3TestRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Wraps a longitude into [-180, 180]. */
function wrapLongitude(longitude: number): number {
  let result = longitude;
  while (result > 180) result -= 360;
  while (result < -180) result += 360;
  return result;
}

/** Packs finite lng/lat degrees as interleaved float32 values, clamping latitude to +-90. */
function pushPoint(points: number[], lng: number, lat: number): void {
  points.push(Math.fround(wrapLongitude(lng)), Math.fround(Math.max(-90, Math.min(90, lat))));
}

/** Area-uniform random points on the sphere as interleaved float32 lng/lat degrees. */
export function createUniformSpherePoints(count: number, seed: number): Float32Array {
  const random = createH3TestRandom(seed);
  const points: number[] = [];
  for (let index = 0; index < count; index++) {
    pushPoint(points, 360 * random() - 180, (Math.asin(2 * random() - 1) * 180) / Math.PI);
  }
  return new Float32Array(points);
}

/** Random points within about two edge lengths of each resolution-`resolution` pentagon center. */
export function createPentagonNeighborhoodPoints(
  resolution: number,
  perPentagon: number,
  seed: number
): Float32Array {
  const random = createH3TestRandom(seed);
  const edgeDegrees = getHexagonEdgeLengthAvg(resolution, 'm') / 111320;
  const points: number[] = [];
  for (const pentagon of getPentagons(resolution)) {
    const [centerLat, centerLng] = cellToBoundaryCenter(pentagon);
    pushPoint(points, centerLng, centerLat);
    for (let index = 0; index < perPentagon; index++) {
      const angle = 2 * Math.PI * random();
      const distance = 2 * edgeDegrees * Math.sqrt(random());
      const latitude = centerLat + distance * Math.sin(angle);
      pushPoint(
        points,
        centerLng +
          (distance * Math.cos(angle)) / Math.max(Math.cos((centerLat * Math.PI) / 180), 1e-3),
        latitude
      );
    }
  }
  return new Float32Array(points);
}

function cellToBoundaryCenter(cell: string): [number, number] {
  const boundary = cellToBoundary(cell);
  let latitude = 0;
  let longitude = boundary[0][1];
  for (const [vertexLat, vertexLng] of boundary) {
    latitude += vertexLat / boundary.length;
    longitude += wrapLongitude(vertexLng - boundary[0][1]) / boundary.length;
  }
  return [latitude, longitude];
}

/**
 * Random points on or near cell edges: vertices and edge points of random resolution-`resolution`
 * cells, jittered by up to `jitterFraction` of the average edge length. Returns float32 lng/lat.
 */
export function createCellEdgePoints(
  resolution: number,
  count: number,
  seed: number,
  jitterFraction: number = 0.02
): Float32Array {
  const random = createH3TestRandom(seed);
  const edgeDegrees = getHexagonEdgeLengthAvg(resolution, 'm') / 111320;
  const points: number[] = [];
  for (let index = 0; index < count; index++) {
    const lng = 360 * random() - 180;
    const lat = (Math.asin(2 * random() - 1) * 180) / Math.PI;
    const boundary = cellToBoundary(latLngToCell(lat, lng, resolution));
    const vertex = Math.floor(random() * boundary.length);
    const [latA, lngA] = boundary[vertex];
    const [latB, lngB] = boundary[(vertex + 1) % boundary.length];
    const t = random() < 0.25 ? 0 : random();
    const edgeLat = latA + t * (latB - latA);
    const edgeLng = lngA + t * wrapLongitude(lngB - lngA);
    const angle = 2 * Math.PI * random();
    const jitter = jitterFraction * edgeDegrees * (2 * random() - 1);
    pushPoint(
      points,
      edgeLng + (jitter * Math.cos(angle)) / Math.max(Math.cos((edgeLat * Math.PI) / 180), 1e-3),
      edgeLat + jitter * Math.sin(angle)
    );
  }
  return new Float32Array(points);
}

/** Poles, antimeridian, equator and prime meridian probes as interleaved float32 lng/lat. */
export function createSpecialH3Points(): Float32Array {
  const points: number[] = [];
  for (const lat of [90, -90, 89.9999, -89.9999, 0, 1e-6, -1e-6, 45, -45, 30]) {
    for (const lng of [-180, 180, 179.9999, -179.9999, 0, 1e-6, -1e-6, 90, -90, 120, -120]) {
      pushPoint(points, lng, lat);
    }
  }
  return new Float32Array(points);
}
