// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Number of leading float32 values shared by every geomorphometry settings layout:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge]`.
 *
 * Contributor-specific values follow from index 5.
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

/**
 * Settings slots of the shared prefix, for
 * {@link getTerrainGroundCellSizeWGSL}: cell size at index 0 and the row edges at 3 and 4.
 *
 * @internal
 */
export const TERRAIN_GEOMORPHOMETRY_CELL_SLOTS = {cellSizeIndex: 0, northEdgeIndex: 3} as const;
