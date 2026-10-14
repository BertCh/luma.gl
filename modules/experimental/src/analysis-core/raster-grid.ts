// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_WGS84_FLATTENING, GPU_WGS84_SEMI_MAJOR_AXIS_METERS} from './contracts/earth';
import {getWGSLFloatLiteral} from '../utils/wgsl-kernel-nodes';

/**
 * How raster and terrain kernels convert grid cell sizes into ground meters per pixel.
 *
 * Cell sizes and row edges come from the grid (`GPURasterMetadata.affine`), in grid world units:
 *
 * - `'uniform'`: world units are already meters (or any consistent ground unit).
 * - `'web-mercator'`: world units are EPSG:3857 meters. Each row is scaled to WGS84 ground meters
 *   at the latitude of its center (`atan(sinh(y / a))` for EPSG:3857 `y`): x by
 *   `N(lat) cos(lat) / a` and y by `M(lat) cos(lat) / a`.
 * - `'geographic'`: world units are degrees. Each row is scaled by the WGS84 meters per degree at
 *   the latitude of its center: `N(lat) cos(lat) PI / 180` east-west and `M(lat) PI / 180`
 *   north-south.
 *
 * `a` is the WGS84 semi-major axis, `M` the meridional and `N` the prime-vertical radius of
 * curvature. Rows are row-constant: the scale is evaluated once at the row center.
 */
export type GPURasterCellSizeMode = 'uniform' | 'web-mercator' | 'geographic';

/**
 * Direction in which raster rows advance: `'south'` means row 0 is the northernmost row
 * (the usual image convention), `'north'` means row 0 is the southernmost row.
 */
export type GPURasterRowDirection = 'south' | 'north';

/**
 * Per-frame georeference words of a grid, in grid world units (see {@link GPURasterCellSizeMode}).
 *
 * @internal
 */
export type RasterGridGeoreference = {
  /** `[x, y]` cell size. */
  cellSize: readonly [number, number];
  /** World y of the top edge of row 0 (EPSG:3857 meters or latitude degrees). Defaults to 0. */
  northEdge?: number;
  /** World y of the bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
};

/**
 * The part of `GPURasterMetadata` that georeferences a grid. Declared structurally so that
 * analysis-core does not import the raster entry.
 *
 * @internal
 */
export type RasterGridAffine = {
  width: number;
  height: number;
  affine: readonly [number, number, number, number, number, number];
  pixelInterpretation: 'area' | 'point';
};

/** WGS84 first eccentricity squared `f (2 - f)`. @internal */
const WGS84_ECCENTRICITY_SQUARED = GPU_WGS84_FLATTENING * (2 - GPU_WGS84_FLATTENING);

/** WGS84 semi-major axis times radians per degree: meters per degree of a radius `a` arc. @internal */
const WGS84_SEMI_MAJOR_AXIS_METERS_PER_DEGREE = (GPU_WGS84_SEMI_MAJOR_AXIS_METERS * Math.PI) / 180;

/** Radians per degree, as written into raster WGSL. @internal */
const RASTER_DEGREES_TO_RADIANS = 0.017453292519943295;

/** WGSL `const DEGREES_TO_RADIANS: f32` declaration shared by raster and terrain kernels. @internal */
export const RASTER_DEGREES_TO_RADIANS_WGSL = `const DEGREES_TO_RADIANS: f32 = ${RASTER_DEGREES_TO_RADIANS};`;

/**
 * Returns the georeference words of a grid: `cellSize = [hypot(a, d), hypot(b, e)]` and the world
 * y of the top edge of row 0 and of the bottom edge of the last row (column 0). Area pixels are
 * addressed at their corners, point pixels at their centers, so point grids move the edges by half
 * a row.
 *
 * @internal
 */
export function getRasterGridGeoreference(
  grid: RasterGridAffine
): Required<RasterGridGeoreference> {
  const [a, b, , d, e, f] = grid.affine;
  const rowShift = grid.pixelInterpretation === 'point' ? -0.5 : 0;
  return {
    cellSize: [Math.hypot(a, d), Math.hypot(b, e)],
    northEdge: e * rowShift + f,
    southEdge: e * (grid.height + rowShift) + f
  };
}

/** Packed-word slots of a grid georeference inside one parameter layout. @internal */
export type RasterGroundCellSizeSlots = {
  /** Name of the `f32` parameter binding. Defaults to `'settings'`. */
  settingsName?: string;
  /** Index of cell size x; cell size y is the next slot. */
  cellSizeIndex: number;
  /** Index of the north edge (top of row 0); the south edge is the next slot. */
  northEdgeIndex: number;
};

/**
 * Writes the georeference of `grid` (see {@link getRasterGridGeoreference}) into the cell size
 * and edge slots of a parameter layout. CPU half of {@link getRasterGroundCellSizeWGSL}.
 *
 * @internal
 */
export function writeRasterGridParameterValues(
  target: Float32Array,
  grid: RasterGridAffine,
  slots: RasterGroundCellSizeSlots
): void {
  const {cellSize, northEdge, southEdge} = getRasterGridGeoreference(grid);
  target[slots.cellSizeIndex] = cellSize[0];
  target[slots.cellSizeIndex + 1] = cellSize[1];
  target[slots.northEdgeIndex] = northEdge;
  target[slots.northEdgeIndex + 1] = southEdge;
}

/**
 * Returns `[x, y]` ground meters per world unit at one row edge value, on the WGS84 ellipsoid:
 * meters per degree for `'geographic'` (edge is latitude in degrees), ground meters per EPSG:3857
 * meter for `'web-mercator'` (edge is EPSG:3857 y in meters), `[1, 1]` for `'uniform'`.
 *
 * Float64 CPU twin of {@link getRasterGroundScaleWGSL}.
 *
 * @internal
 */
export function getRasterGroundScale(mode: GPURasterCellSizeMode, edge: number): [number, number] {
  if (mode === 'uniform') {
    return [1, 1];
  }
  if (mode === 'web-mercator') {
    // EPSG:3857 maps geodetic latitude with the spherical formula, so the row latitude is exact.
    const mercatorY = edge / GPU_WGS84_SEMI_MAJOR_AXIS_METERS;
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
 * {@link getRasterGroundScale}, from an `f32` variable named `edge` in scope. Self-contained (all
 * constants are literals); declares the locals `mercatorY`, `latitude`, `sinLatitude`,
 * `curvature`, `primeVertical` and `groundScale`. Not used for `'uniform'`.
 *
 * @internal
 */
export function getRasterGroundScaleWGSL(mode: 'web-mercator' | 'geographic'): string {
  const eccentricitySquared = getWGSLFloatLiteral(WGS84_ECCENTRICITY_SQUARED);
  const polarRatio = getWGSLFloatLiteral(1 - WGS84_ECCENTRICITY_SQUARED);
  if (mode === 'web-mercator') {
    return `// WGS84 ground meters per EPSG:3857 meter at the row latitude atan(sinh(y / a)):
  // x = N cos(lat) / a, y = M cos(lat) / a, with cos(lat) = 1 / cosh(y / a).
  let mercatorY = edge * ${getWGSLFloatLiteral(1 / GPU_WGS84_SEMI_MAJOR_AXIS_METERS)};
  let sinLatitude = tanh(mercatorY);
  let curvature = 1.0 - ${eccentricitySquared} * sinLatitude * sinLatitude;
  let primeVertical = 1.0 / sqrt(curvature);
  let groundScale = vec2<f32>(primeVertical, primeVertical * ${polarRatio} / curvature) / cosh(mercatorY);`;
  }
  return `// WGS84 meters per degree at the row latitude: x = N cos(lat) PI / 180, y = M PI / 180.
  let latitude = edge * ${getWGSLFloatLiteral(RASTER_DEGREES_TO_RADIANS)};
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
export function getRasterRowNorthSignWGSL(
  rowDirection: GPURasterRowDirection,
  type: 'f32' | 'i32' = 'f32'
): string {
  const sign = rowDirection === 'south' ? -1 : 1;
  return `const ROW_NORTH_SIGN: ${type} = ${type === 'f32' ? `${sign}.0` : sign};`;
}

/**
 * Returns WGSL `fn getGroundCellSize(row: u32) -> vec2<f32>`: ground meters (times the world
 * units in `'uniform'` mode) per pixel of one raster row.
 *
 * The kernel must declare `HEIGHT`. The generated code is self-contained (it needs no `PI`
 * declaration). The edge of a row is interpolated linearly between the north and south edges at
 * the row center: EPSG:3857 y for `'web-mercator'`, latitude degrees for `'geographic'`. Both
 * models use WGS84 radii of curvature at the row center, constant per row.
 *
 * @internal
 */
export function getRasterGroundCellSizeWGSL(
  mode: GPURasterCellSizeMode,
  slots: RasterGroundCellSizeSlots
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
  ${getRasterGroundScaleWGSL(mode)}
  return cellSize * groundScale;
}`;
}

/**
 * Returns ground `[x, y]` meters per cell at a continuous row coordinate (row index + 0.5 is a
 * cell center), in float64. CPU twin of the WGSL `getGroundCellSize` of
 * {@link getRasterGroundCellSizeWGSL} and of the cost-distance grid helpers.
 *
 * @internal
 */
export function getRasterGroundCellSize(
  mode: GPURasterCellSizeMode,
  georeference: RasterGridGeoreference,
  rowCoordinate: number,
  height: number
): [number, number] {
  const [cellSizeX, cellSizeY] = georeference.cellSize;
  if (mode === 'uniform') {
    return [cellSizeX, cellSizeY];
  }
  const northEdge = georeference.northEdge ?? 0;
  const southEdge = georeference.southEdge ?? 0;
  const edge = northEdge + (southEdge - northEdge) * (rowCoordinate / height);
  const [scaleX, scaleY] = getRasterGroundScale(mode, edge);
  return [cellSizeX * scaleX, cellSizeY * scaleY];
}

/**
 * Returns the unit pixel-space step `(columns, rows)` that travels along a ground-space unit
 * direction `(east, row-ward)` on cells of `groundCell = [x, y]` ground meters:
 * `normalize(groundDirection / groundCell)`. Square cells return `groundDirection` unchanged, so
 * isotropic grids keep bit-identical rays. CPU twin of WGSL {@link RASTER_PIXEL_DIRECTION_WGSL}.
 *
 * Without this, a compass azimuth used directly as a pixel direction drifts toward north-south
 * wherever cells are anisotropic: a 45 degree ray on a 60 degrees north geographic grid points at
 * about 26.6 degrees.
 *
 * @internal
 */
export function getRasterPixelDirection(
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

/** WGSL twin of {@link getRasterPixelDirection}. @internal */
export const RASTER_PIXEL_DIRECTION_WGSL = /* wgsl */ `
// Unit pixel step along a ground-space unit direction on cells of groundCell meters.
fn getPixelDirection(groundDirection: vec2<f32>, groundCell: vec2<f32>) -> vec2<f32> {
  if (groundCell.x == groundCell.y) { return groundDirection; }
  return normalize(groundDirection / groundCell);
}`;

/** Throws unless `mode` is a known cell size mode; `undefined` means `'uniform'`. @internal */
export function validateRasterCellSizeMode(id: string, mode: string | undefined): void {
  if (!['uniform', 'web-mercator', 'geographic'].includes(mode ?? 'uniform')) {
    throw new Error(`${id} cellSizeMode must be uniform, web-mercator, or geographic`);
  }
}

/** Throws unless `rowDirection` is `'south'` or `'north'`; `undefined` means `'south'`. @internal */
export function validateRasterRowDirection(id: string, rowDirection: string | undefined): void {
  if (!['south', 'north'].includes(rowDirection ?? 'south')) {
    throw new Error(`${id} rowDirection must be south or north`);
  }
}
