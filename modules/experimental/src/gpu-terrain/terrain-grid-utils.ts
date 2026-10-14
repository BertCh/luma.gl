// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_WGS84_FLATTENING,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from '../gpu-spatial-analysis/contracts/earth';
import {getWGSLFloatLiteral} from '../utils/wgsl-kernel-nodes';

/**
 * How terrain kernels convert settings cell sizes into ground meters per pixel.
 *
 * - `'uniform'`: cell sizes are already in meters (or any consistent ground unit).
 * - `'web-mercator'`: cell sizes are EPSG:3857 meters. Each row is scaled to WGS84 ground meters at
 *   the latitude of its center (`atan(sinh(PI * (1 - 2y)))` for normalized Web Mercator `y`): x by
 *   `N(lat) cos(lat) / a` and y by `M(lat) cos(lat) / a`.
 * - `'geographic'`: cell sizes are degrees. Each row is scaled by the WGS84 meters per degree at the
 *   latitude of its center: `N(lat) cos(lat) PI / 180` east-west and `M(lat) PI / 180` north-south.
 *
 * `a` is the WGS84 semi-major axis, `M` the meridional and `N` the prime-vertical radius of
 * curvature. Rows are row-constant: the scale is evaluated once at the row center.
 */
export type GPUTerrainCellSizeMode = 'uniform' | 'web-mercator' | 'geographic';

/**
 * Direction in which raster rows advance: `'south'` means row 0 is the northernmost row
 * (the usual image convention), `'north'` means row 0 is the southernmost row.
 */
export type GPUTerrainRowDirection = 'south' | 'north';

/** WGS84 first eccentricity squared `f (2 - f)`. @internal */
const WGS84_ECCENTRICITY_SQUARED = GPU_WGS84_FLATTENING * (2 - GPU_WGS84_FLATTENING);

/** WGS84 semi-major axis times radians per degree: meters per degree of a radius `a` arc. @internal */
const WGS84_SEMI_MAJOR_AXIS_METERS_PER_DEGREE = (GPU_WGS84_SEMI_MAJOR_AXIS_METERS * Math.PI) / 180;

/** Radians per degree, as written into terrain WGSL. @internal */
const TERRAIN_DEGREES_TO_RADIANS = 0.017453292519943295;

/** WGSL `const DEGREES_TO_RADIANS: f32` declaration shared by terrain kernels. @internal */
export const TERRAIN_DEGREES_TO_RADIANS_WGSL = `const DEGREES_TO_RADIANS: f32 = ${TERRAIN_DEGREES_TO_RADIANS};`;

/**
 * Returns `[x, y]` ground meters per settings unit at one row edge value, on the WGS84 ellipsoid:
 * meters per degree for `'geographic'` (edge is latitude in degrees), ground meters per EPSG:3857
 * meter for `'web-mercator'` (edge is normalized Web Mercator y), `[1, 1]` for `'uniform'`.
 *
 * Float64 CPU twin of {@link getTerrainGroundScaleWGSL}.
 *
 * @internal
 */
export function getTerrainGroundScale(
  mode: GPUTerrainCellSizeMode,
  edge: number
): [number, number] {
  if (mode === 'uniform') {
    return [1, 1];
  }
  if (mode === 'web-mercator') {
    // EPSG:3857 maps geodetic latitude with the spherical formula, so the row latitude is exact.
    const mercatorY = Math.PI * (1 - 2 * edge);
    const sinLatitude = Math.tanh(mercatorY);
    const curvature = 1 - WGS84_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude;
    const primeVerticalRatio = 1 / Math.sqrt(curvature);
    const cosLatitude = 1 / Math.cosh(mercatorY);
    return [
      primeVerticalRatio * cosLatitude,
      ((primeVerticalRatio * (1 - WGS84_ECCENTRICITY_SQUARED)) / curvature) * cosLatitude
    ];
  }
  const latitude = (edge * Math.PI) / 180;
  const sinLatitude = Math.sin(latitude);
  const curvature = 1 - WGS84_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude;
  const primeVertical = WGS84_SEMI_MAJOR_AXIS_METERS_PER_DEGREE / Math.sqrt(curvature);
  return [
    primeVertical * Math.cos(latitude),
    (primeVertical * (1 - WGS84_ECCENTRICITY_SQUARED)) / curvature
  ];
}

/**
 * Returns WGSL statements that declare `let groundScale: vec2<f32>`, the WGSL twin of
 * {@link getTerrainGroundScale}, from an `f32` variable named `edge` in scope. Self-contained (all
 * constants are literals); declares the locals `mercatorY`, `latitude`, `sinLatitude`,
 * `curvature`, `primeVertical` and `groundScale`. Not used for `'uniform'`.
 *
 * @internal
 */
export function getTerrainGroundScaleWGSL(mode: 'web-mercator' | 'geographic'): string {
  const eccentricitySquared = getWGSLFloatLiteral(WGS84_ECCENTRICITY_SQUARED);
  const polarRatio = getWGSLFloatLiteral(1 - WGS84_ECCENTRICITY_SQUARED);
  if (mode === 'web-mercator') {
    return `// WGS84 ground meters per EPSG:3857 meter at the row latitude atan(sinh(mercatorY)):
  // x = N cos(lat) / a, y = M cos(lat) / a, with cos(lat) = 1 / cosh(mercatorY).
  let mercatorY = ${getWGSLFloatLiteral(Math.PI)} * (1.0 - 2.0 * edge);
  let sinLatitude = tanh(mercatorY);
  let curvature = 1.0 - ${eccentricitySquared} * sinLatitude * sinLatitude;
  let primeVertical = 1.0 / sqrt(curvature);
  let groundScale = vec2<f32>(primeVertical, primeVertical * ${polarRatio} / curvature) / cosh(mercatorY);`;
  }
  return `// WGS84 meters per degree at the row latitude: x = N cos(lat) PI / 180, y = M PI / 180.
  let latitude = edge * ${getWGSLFloatLiteral(TERRAIN_DEGREES_TO_RADIANS)};
  let sinLatitude = sin(latitude);
  let curvature = 1.0 - ${eccentricitySquared} * sinLatitude * sinLatitude;
  let primeVertical = ${getWGSLFloatLiteral(WGS84_SEMI_MAJOR_AXIS_METERS_PER_DEGREE)} / sqrt(curvature);
  let groundScale = vec2<f32>(primeVertical * cos(latitude), primeVertical * ${polarRatio} / curvature);`;
}

/**
 * Returns the WGSL `const ROW_NORTH_SIGN` declaration: `-1` when rows advance south (a positive
 * row step moves toward the south), `+1` when they advance north.
 *
 * @internal
 */
export function getTerrainRowNorthSignWGSL(
  rowDirection: GPUTerrainRowDirection,
  type: 'f32' | 'i32' = 'f32'
): string {
  const sign = rowDirection === 'south' ? -1 : 1;
  return `const ROW_NORTH_SIGN: ${type} = ${type === 'f32' ? `${sign}.0` : sign};`;
}

/** Settings slots read by {@link getTerrainGroundCellSizeWGSL}. @internal */
export type TerrainGroundCellSizeSlots = {
  /** Name of the `f32` settings binding. Defaults to `'settings'`. */
  settingsName?: string;
  /** Index of cell size x; cell size y is the next slot. */
  cellSizeIndex: number;
  /** Index of the north edge (top of row 0); the south edge is the next slot. */
  northEdgeIndex: number;
};

/**
 * Returns WGSL `fn getGroundCellSize(row: u32) -> vec2<f32>`: ground meters (times the settings
 * units in `'uniform'` mode) per pixel of one raster row.
 *
 * The kernel must declare `HEIGHT`. The generated code is self-contained (it needs no `PI`
 * declaration). The edge of a row is interpolated linearly between the north and south edges at
 * the row center: normalized Web Mercator y in `[0, 1]` for `'web-mercator'`, latitude degrees for
 * `'geographic'`. Both models use WGS84 radii of curvature at the row center, constant per row.
 *
 * @internal
 */
export function getTerrainGroundCellSizeWGSL(
  mode: GPUTerrainCellSizeMode,
  slots: TerrainGroundCellSizeSlots
): string {
  const settings = slots.settingsName ?? 'settings';
  const offset = `${settings}Offset`;
  const cellSize = `vec2<f32>(${settings}[${offset} + ${slots.cellSizeIndex}u], ${settings}[${offset} + ${slots.cellSizeIndex + 1}u])`;
  if (mode === 'uniform') {
    return `fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = ${cellSize};
  return cellSize;
}`;
  }
  const edge = `mix(${settings}[${offset} + ${slots.northEdgeIndex}u], ${settings}[${offset} + ${slots.northEdgeIndex + 1}u], rowFraction)`;
  return `fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = ${cellSize};
  let rowFraction = (f32(row) + 0.5) / f32(HEIGHT);
  let edge = ${edge};
  ${getTerrainGroundScaleWGSL(mode)}
  return cellSize * groundScale;
}`;
}

/**
 * CPU twin of {@link getTerrainGroundCellSizeWGSL} for oracles, in float64.
 *
 * @returns `[x, y]` ground meters per pixel for `row`.
 * @internal
 */
export function getTerrainGroundCellSize(
  mode: GPUTerrainCellSizeMode,
  cellSize: readonly [number, number],
  northEdge: number,
  southEdge: number,
  row: number,
  height: number
): [number, number] {
  if (mode === 'uniform') {
    return [cellSize[0], cellSize[1]];
  }
  const fraction = (row + 0.5) / height;
  const edge = northEdge + (southEdge - northEdge) * fraction;
  const [scaleX, scaleY] = getTerrainGroundScale(mode, edge);
  return [cellSize[0] * scaleX, cellSize[1] * scaleY];
}

/**
 * Returns the unit pixel-space step `(columns, rows)` that travels along a ground-space unit
 * direction `(east, row-ward)` on cells of `groundCell = [x, y]` ground meters:
 * `normalize(groundDirection / groundCell)`. Square cells return `groundDirection` unchanged, so
 * isotropic grids keep bit-identical rays. CPU twin of WGSL {@link TERRAIN_PIXEL_DIRECTION_WGSL}.
 *
 * Without this, a compass azimuth used directly as a pixel direction drifts toward north-south
 * wherever cells are anisotropic: a 45 degree ray on a 60 degrees north geographic grid points at
 * about 26.6 degrees.
 *
 * @internal
 */
export function getTerrainPixelDirection(
  groundDirection: readonly [number, number],
  groundCell: readonly [number, number]
): [number, number] {
  if (groundCell[0] === groundCell[1]) {
    return [groundDirection[0], groundDirection[1]];
  }
  const x = groundDirection[0] / groundCell[0];
  const y = groundDirection[1] / groundCell[1];
  const length = Math.hypot(x, y);
  return [x / length, y / length];
}

/** WGSL twin of {@link getTerrainPixelDirection}. @internal */
export const TERRAIN_PIXEL_DIRECTION_WGSL = /* wgsl */ `
// Unit pixel step along a ground-space unit direction on cells of groundCell meters.
fn getPixelDirection(groundDirection: vec2<f32>, groundCell: vec2<f32>) -> vec2<f32> {
  if (groundCell.x == groundCell.y) { return groundDirection; }
  return normalize(groundDirection / groundCell);
}`;

/** Throws unless `mode` is a known cell size mode; `undefined` means `'uniform'`. @internal */
export function validateTerrainCellSizeMode(id: string, mode: string | undefined): void {
  if (!['uniform', 'web-mercator', 'geographic'].includes(mode ?? 'uniform')) {
    throw new Error(`${id} cellSizeMode must be uniform, web-mercator, or geographic`);
  }
}

/** Throws unless `rowDirection` is `'south'` or `'north'`; `undefined` means `'south'`. @internal */
export function validateTerrainRowDirection(id: string, rowDirection: string | undefined): void {
  if (!['south', 'north'].includes(rowDirection ?? 'south')) {
    throw new Error(`${id} rowDirection must be south or north`);
  }
}
