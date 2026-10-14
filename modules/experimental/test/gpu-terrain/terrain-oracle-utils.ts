// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** WGS84 semi-major axis in metres. */
const ORACLE_WGS84_SEMI_MAJOR_AXIS = 6378137;
/** WGS84 first eccentricity squared `f (2 - f)` with `f = 1 / 298.257223563`. */
const ORACLE_WGS84_ECCENTRICITY_SQUARED = (2 - 1 / 298.257223563) / 298.257223563;

/**
 * WGS84 meridional `M` and prime-vertical `N` radii of curvature in metres at a latitude:
 * `N = a / sqrt(1 - e^2 sin^2(lat))`, `M = a (1 - e^2) / (1 - e^2 sin^2(lat))^1.5`.
 */
export function getOracleRadiiOfCurvature(latitudeDegrees: number): [number, number] {
  const sinLatitude = Math.sin((latitudeDegrees * Math.PI) / 180);
  const w = 1 - ORACLE_WGS84_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude;
  return [
    (ORACLE_WGS84_SEMI_MAJOR_AXIS * (1 - ORACLE_WGS84_ECCENTRICITY_SQUARED)) / w ** 1.5,
    ORACLE_WGS84_SEMI_MAJOR_AXIS / Math.sqrt(w)
  ];
}

export type OracleCellSizeMode = 'uniform' | 'web-mercator' | 'geographic';

/**
 * Ground cell size of one row in the three cell size models, matching the GPU contributors.
 *
 * @param settings Settings array with the cell size at `[0]`, `[1]` and the north and south
 *   latitude or Mercator edges at `[northEdgeIndex]`, `[northEdgeIndex + 1]`.
 * @param row Raster row.
 * @param height Raster height in rows.
 * @param mode Cell size model.
 * @param northEdgeIndex Index of the north edge in `settings`. Defaults to 3.
 */
export function getOracleGroundCellSize(
  settings: ArrayLike<number>,
  row: number,
  height: number,
  mode: OracleCellSizeMode,
  northEdgeIndex = 3
): [number, number] {
  const cellX = settings[0];
  const cellY = settings[1];
  if (mode === 'uniform') {
    return [cellX, cellY];
  }
  const fraction = (row + 0.5) / height;
  const edge =
    settings[northEdgeIndex] + (settings[northEdgeIndex + 1] - settings[northEdgeIndex]) * fraction;
  if (mode === 'web-mercator') {
    // EPSG:3857: geodetic latitude = atan(sinh(PI (1 - 2y))); ground = radius * cos(lat) / a.
    const latitude = (Math.atan(Math.sinh(Math.PI * (1 - 2 * edge))) * 180) / Math.PI;
    const [meridional, primeVertical] = getOracleRadiiOfCurvature(latitude);
    const cosLatitude = Math.cos((latitude * Math.PI) / 180);
    return [
      (cellX * primeVertical * cosLatitude) / ORACLE_WGS84_SEMI_MAJOR_AXIS,
      (cellY * meridional * cosLatitude) / ORACLE_WGS84_SEMI_MAJOR_AXIS
    ];
  }
  const [meridional, primeVertical] = getOracleRadiiOfCurvature(edge);
  return [
    (cellX * primeVertical * Math.cos((edge * Math.PI) / 180) * Math.PI) / 180,
    (cellY * meridional * Math.PI) / 180
  ];
}

/** Seeded linear congruential generator returning floats in `[0, 1)`. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Seeded mulberry32 generator returning floats in `[0, 1)`. */
export function createMulberryRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic xorshift generator returning floats in `[0, 1)`. A zero seed maps to 1. */
export function createXorshiftRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}
