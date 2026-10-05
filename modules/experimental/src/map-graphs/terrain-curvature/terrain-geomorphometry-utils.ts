// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUTerrainCellSizeMode} from '../terrain-analysis/gpu-terrain-derivatives';

/**
 * Number of leading float32 values shared by every geomorphometry settings layout:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge]`.
 *
 * Recipe-specific values follow from index 5.
 *
 * @internal
 */
export const TERRAIN_GEOMORPHOMETRY_SHARED_PARAMETER_LENGTH = 5;

/** Shared cell-size part of every geomorphometry settings object. @internal */
export type TerrainGeomorphometryCellSettings = {
  /** `[x, y]` cell size: metres (uniform), equatorial Web Mercator metres, or degrees (geographic). */
  cellSize: readonly [number, number];
  /** Elevation multiplier converting elevation units into ground units. Defaults to 1. */
  zFactor?: number;
  /** Top edge of row 0: normalized Web Mercator y in `[0, 1]` or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
};

/**
 * Writes the shared `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge]` prefix.
 *
 * @internal
 */
export function writeTerrainGeomorphometryCellSettings(
  settings: TerrainGeomorphometryCellSettings,
  target: Float32Array
): void {
  target[0] = settings.cellSize[0];
  target[1] = settings.cellSize[1];
  target[2] = settings.zFactor ?? 1;
  target[3] = settings.northEdge ?? 0;
  target[4] = settings.southEdge ?? 0;
}

/** Throws unless `mode` is a supported cell size mode. @internal */
export function validateTerrainCellSizeMode(id: string, mode: string | undefined): void {
  if (!['uniform', 'web-mercator', 'geographic'].includes(mode ?? 'uniform')) {
    throw new Error(`${id} cellSizeMode must be uniform, web-mercator, or geographic`);
  }
}

/** Throws unless `rowDirection` is `'south'` or `'north'`. @internal */
export function validateTerrainRowDirection(id: string, rowDirection: string | undefined): void {
  if (!['south', 'north'].includes(rowDirection ?? 'south')) {
    throw new Error(`${id} rowDirection must be south or north`);
  }
}

/**
 * Returns WGSL declaring `getGroundCellSize(row: u32) -> vec2<f32>`, the ground cell size in
 * metres (times the settings units in `'uniform'` mode) of one raster row.
 *
 * The generated function reads the shared settings prefix from the storage binding named
 * `settings` and requires a `HEIGHT` constant. `'web-mercator'` divides by `cosh(PI * (1 - 2y))`
 * (the reciprocal of cos(latitude)) at the row centre; `'geographic'` multiplies degrees by
 * 111319.49 m and the x size by cos(latitude) at the row centre. Both models are a spherical,
 * row-constant approximation, exactly like `GPUTerrainDerivatives`.
 *
 * @internal
 */
export function getTerrainGroundCellSizeWGSL(mode: GPUTerrainCellSizeMode): string {
  const body =
    mode === 'uniform'
      ? 'return cellSize;'
      : `let rowFraction = (f32(row) + 0.5) / f32(HEIGHT);
  let edge = mix(settings[settingsOffset + 3u], settings[settingsOffset + 4u], rowFraction);
  ${
    mode === 'web-mercator'
      ? '// cos(latitude) = 1 / cosh(PI * (1 - 2y)) for normalized Web Mercator y.\n  return cellSize / cosh(3.141592653589793 * (1.0 - 2.0 * edge));'
      : 'return cellSize * 111319.49079327357 * vec2<f32>(cos(edge * 0.017453292519943295), 1.0);'
  }`;
  return /* wgsl */ `
fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  ${body}
}`;
}
