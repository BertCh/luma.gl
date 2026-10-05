// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * How terrain kernels convert settings cell sizes into ground meters per pixel.
 *
 * - `'uniform'`: cell sizes are already in meters (or any consistent ground unit).
 * - `'web-mercator'`: cell sizes are equatorial Web Mercator meters, scaled per row by
 *   `1 / cosh(PI * (1 - 2y))` where `y` is the normalized Web Mercator y of the row center.
 * - `'geographic'`: cell sizes are degrees, scaled by meters per degree, and the x size by
 *   `cos(latitude)` at the row center.
 */
export type GPUTerrainCellSizeMode = 'uniform' | 'web-mercator' | 'geographic';

/**
 * Direction in which raster rows advance: `'south'` means row 0 is the northernmost row
 * (the usual image convention), `'north'` means row 0 is the southernmost row.
 */
export type GPUTerrainRowDirection = 'south' | 'north';

/** Meters per degree of latitude on the WGS84 equatorial sphere used by terrain kernels. @internal */
const TERRAIN_METERS_PER_DEGREE = 111319.49079327357;

/** Radians per degree, as written into terrain WGSL. @internal */
const TERRAIN_DEGREES_TO_RADIANS = 0.017453292519943295;

/** WGSL `const DEGREES_TO_RADIANS: f32` declaration shared by terrain kernels. @internal */
export const TERRAIN_DEGREES_TO_RADIANS_WGSL = `const DEGREES_TO_RADIANS: f32 = ${TERRAIN_DEGREES_TO_RADIANS};`;

/** WGSL `const METERS_PER_DEGREE: f32` declaration shared by terrain kernels. @internal */
export const TERRAIN_METERS_PER_DEGREE_WGSL = `const METERS_PER_DEGREE: f32 = ${TERRAIN_METERS_PER_DEGREE};`;

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
 * The kernel must declare `HEIGHT`. The generated code is self-contained (it needs no `PI` or
 * `METERS_PER_DEGREE` declarations). The edge of a row is interpolated linearly between the north
 * and south edges at the row center: normalized Web Mercator y in `[0, 1]` for `'web-mercator'`,
 * latitude degrees for `'geographic'`. Both models are a spherical, row-constant approximation.
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
  const scaled =
    mode === 'web-mercator'
      ? `// cos(latitude) = 1 / cosh(PI * (1 - 2y)) for normalized Web Mercator y.
  return cellSize / cosh(${Math.PI} * (1.0 - 2.0 * edge));`
      : `return cellSize * ${TERRAIN_METERS_PER_DEGREE} * vec2<f32>(cos(edge * ${TERRAIN_DEGREES_TO_RADIANS}), 1.0);`;
  return `fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = ${cellSize};
  let rowFraction = (f32(row) + 0.5) / f32(HEIGHT);
  let edge = ${edge};
  ${scaled}
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
  if (mode === 'web-mercator') {
    const scale = 1 / Math.cosh(Math.PI * (1 - 2 * edge));
    return [cellSize[0] * scale, cellSize[1] * scale];
  }
  return [
    cellSize[0] * TERRAIN_METERS_PER_DEGREE * Math.cos((edge * Math.PI) / 180),
    cellSize[1] * TERRAIN_METERS_PER_DEGREE
  ];
}

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
