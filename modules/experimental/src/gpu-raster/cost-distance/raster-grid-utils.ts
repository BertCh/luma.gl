// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUTerrainCellSizeMode} from '../../gpu-terrain/terrain-analysis/index';
import {
  getTerrainGroundScale,
  getTerrainGroundScaleWGSL
} from '../../gpu-terrain/terrain-grid-utils';
import {GET_QUIET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

/**
 * Creates a contributor-owned transient view and attributes an ID collision to the contributor.
 *
 * Contributors derive transient IDs as `${id}-<suffix>`. When the caller already uses such an ID the
 * graph reports only that the resource ID is taken; this rethrows an error naming the contributor class
 * and ID, keeping the graph's error as `cause`.
 *
 * @param operation Contributor class name, such as `'GPUCostDistance'`.
 * @param contributorId The contributor's `id` prop.
 * @param transientId The generated resource ID to create.
 * @internal
 */
export function createContributorTransientView<
  Format extends 'uint32' | 'float32',
  Parameters = unknown
>(
  graph: GPUCommandGraph<Parameters>,
  operation: string,
  contributorId: string,
  transientId: string,
  format: Format,
  length: number,
  usage?: number
): GraphDataView<Format> {
  try {
    return createTransientView(graph, transientId, format, length, usage);
  } catch (error) {
    if (error instanceof Error && /resource id ".*" is already in use/.test(error.message)) {
      throw new Error(
        `${operation} "${contributorId}": internal transient resource "${transientId}" (generated from the contributor id) collides with an existing graph resource; choose a different contributor id or rename the resource`,
        {cause: error}
      );
    }
    throw error;
  }
}

/**
 * One of the eight D8 grid moves, ordered clockwise from east.
 *
 * Columns increase east and rows increase in the raster's row direction (south for north-up
 * rasters). `code` is the conventional ESRI power-of-two D8 code.
 */
export type GPURasterD8Direction = {
  /** ESRI D8 code: 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32 north-west, 64 north, 128 north-east. */
  code: number;
  /** Column delta of the move. */
  columnOffset: number;
  /** Row delta of the move. */
  rowOffset: number;
};

/**
 * The eight D8 moves in direction-index order `0..7` (east, south-east, south, south-west, west,
 * north-west, north, north-east). Raster flow-direction and back-link outputs store the `code`.
 */
export const GPU_RASTER_D8_DIRECTIONS: readonly GPURasterD8Direction[] = [
  {code: 1, columnOffset: 1, rowOffset: 0},
  {code: 2, columnOffset: 1, rowOffset: 1},
  {code: 4, columnOffset: 0, rowOffset: 1},
  {code: 8, columnOffset: -1, rowOffset: 1},
  {code: 16, columnOffset: -1, rowOffset: 0},
  {code: 32, columnOffset: -1, rowOffset: -1},
  {code: 64, columnOffset: 0, rowOffset: -1},
  {code: 128, columnOffset: 1, rowOffset: -1}
];

/** Settings prefix shared by raster grid contributors: `[cellSizeX, cellSizeY, northEdge, southEdge]`. @internal */
export const RASTER_GRID_SETTINGS_LENGTH = 4;

/** CPU description of the shared settings prefix. @internal */
export type RasterGridSettings = {
  /** `[x, y]` cell size: meters (uniform), EPSG:3857 meters (web-mercator), or degrees (geographic). */
  cellSize: readonly [number, number];
  /** Top edge of row 0: normalized Web Mercator y in `[0, 1]` or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
};

/** Writes the shared grid settings prefix into `target[0..3]`. @internal */
export function writeRasterGridSettings(target: Float32Array, settings: RasterGridSettings): void {
  target[0] = settings.cellSize[0];
  target[1] = settings.cellSize[1];
  target[2] = settings.northEdge ?? 0;
  target[3] = settings.southEdge ?? 0;
}

/**
 * Returns ground `[x, y]` meters per cell at a continuous row coordinate (row index + 0.5 is a
 * cell center), in float64, with WGS84 radii of curvature at that row's latitude for the
 * `'web-mercator'` and `'geographic'` modes (see `GPUTerrainCellSizeMode`). CPU twin of the WGSL
 * `getGroundCellSize`.
 *
 * @internal
 */
export function getRasterGroundCellSize(
  cellSizeMode: GPUTerrainCellSizeMode,
  rowCoordinate: number,
  height: number,
  settings: RasterGridSettings
): [number, number] {
  const [cellSizeX, cellSizeY] = settings.cellSize;
  if (cellSizeMode === 'uniform') {
    return [cellSizeX, cellSizeY];
  }
  const rowFraction = rowCoordinate / height;
  const northEdge = settings.northEdge ?? 0;
  const southEdge = settings.southEdge ?? 0;
  const edge = northEdge + (southEdge - northEdge) * rowFraction;
  const [scaleX, scaleY] = getTerrainGroundScale(cellSizeMode, edge);
  return [cellSizeX * scaleX, cellSizeY * scaleY];
}

/**
 * Returns the ground distance of one D8 move from a cell in `centerRow`, in float64.
 *
 * The cell size is evaluated at the midpoint row of the two cells, so the distance is symmetric.
 * CPU twin of the WGSL `getD8Distance`.
 *
 * @internal
 */
export function getRasterD8Distance(
  cellSizeMode: GPUTerrainCellSizeMode,
  direction: number,
  centerRow: number,
  height: number,
  settings: RasterGridSettings
): number {
  const {columnOffset, rowOffset} = GPU_RASTER_D8_DIRECTIONS[direction];
  const neighborRow = centerRow + rowOffset;
  const [groundX, groundY] = getRasterGroundCellSize(
    cellSizeMode,
    (centerRow + neighborRow + 1) * 0.5,
    height,
    settings
  );
  if (rowOffset === 0) {
    return groundX;
  }
  if (columnOffset === 0) {
    return groundY;
  }
  return Math.sqrt(groundX * groundX + groundY * groundY);
}

/** Throws unless `cellSizeMode` is a supported mode. @internal */
export function validateRasterCellSizeMode(id: string, cellSizeMode: string): void {
  if (!['uniform', 'web-mercator', 'geographic'].includes(cellSizeMode)) {
    throw new Error(`${id} cellSizeMode must be uniform, web-mercator, or geographic`);
  }
}

/**
 * Returns WGSL grid constants and D8 helpers for one compile-time grid.
 *
 * The generated `getGroundCellSize` and `getD8Distance` read the shared settings prefix from a
 * float32 storage binding named `settings` (with the generated `settingsOffset` constant), so the
 * kernel must bind it. Declares `GRID_WIDTH`, `GRID_HEIGHT`, `getD8ColumnOffset`,
 * `getD8RowOffset`, `getD8Code`, `getD8Neighbor` (returns `0xffffffff` outside the grid),
 * `getInfinity`, `getQuietNaN`, `isFiniteValue`, and `isNaNValue` (bit tests, immune to NaN folding).
 *
 * @internal
 */
export function getRasterGridWGSL(props: {
  width: number;
  height: number;
  cellSizeMode: GPUTerrainCellSizeMode;
}): string {
  const groundCellSource =
    props.cellSizeMode === 'uniform'
      ? 'return cellSize;'
      : `let rowFraction = rowCoordinate / f32(GRID_HEIGHT);
  let edge = mix(settings[settingsOffset + 2u], settings[settingsOffset + 3u], rowFraction);
  ${getTerrainGroundScaleWGSL(props.cellSizeMode)}
  return cellSize * groundScale;`;
  return /* wgsl */ `
const GRID_WIDTH: u32 = ${props.width}u;
const GRID_HEIGHT: u32 = ${props.height}u;
const GRID_NONE: u32 = 0xffffffffu;

// Bit tests: shader compilers may fold 'value != value' away under fast-math assumptions.
${IS_FINITE_VALUE_WGSL}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn getInfinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }
${GET_QUIET_NAN_WGSL}

fn getD8ColumnOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  return offsets[direction];
}
fn getD8RowOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  return offsets[direction];
}
fn getD8Code(direction: u32) -> u32 { return 1u << direction; }

// Returns the neighboring cell index in one D8 direction, or GRID_NONE outside the grid.
fn getD8Neighbor(cell: u32, direction: u32) -> u32 {
  let column = i32(cell % GRID_WIDTH) + getD8ColumnOffset(direction);
  let row = i32(cell / GRID_WIDTH) + getD8RowOffset(direction);
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) {
    return GRID_NONE;
  }
  return u32(row) * GRID_WIDTH + u32(column);
}

// Ground meters per cell at a continuous row coordinate (row index + 0.5 is a cell center).
fn getGroundCellSize(rowCoordinate: f32) -> vec2<f32> {
  let cellSize = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  ${groundCellSource}
}

// Ground length of one D8 move from centerRow, measured at the midpoint row (symmetric).
fn getD8Distance(direction: u32, centerRow: u32) -> f32 {
  let rowOffset = getD8RowOffset(direction);
  let neighborRow = i32(centerRow) + rowOffset;
  let ground = getGroundCellSize((f32(centerRow) + f32(neighborRow) + 1.0) * 0.5);
  if (rowOffset == 0) { return ground.x; }
  if (getD8ColumnOffset(direction) == 0) { return ground.y; }
  return sqrt(ground.x * ground.x + ground.y * ground.y);
}`;
}
