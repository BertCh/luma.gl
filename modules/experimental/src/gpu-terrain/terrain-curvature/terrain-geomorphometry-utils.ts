// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '../../gpu-raster/index';
import {writeRasterGridParameterValues} from '../../analysis-core/raster-grid';

/**
 * Number of leading float32 values shared by every geomorphometry settings layout:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge]`.
 *
 * Contributor-specific values follow from index 5.
 *
 * @internal
 */
export const TERRAIN_GEOMORPHOMETRY_SHARED_PARAMETER_LENGTH = 5;

/** Shared elevation-scale part of every geomorphometry settings object. @internal */
export type TerrainGeomorphometryCellSettings = {
  /** Elevation multiplier converting elevation units into ground units. Defaults to 1. */
  zFactor?: number;
};

/**
 * Writes the shared `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge]` prefix. The cell size
 * and row edges come from `grid`.
 *
 * @internal
 */
export function writeTerrainGeomorphometryCellSettings(
  settings: TerrainGeomorphometryCellSettings,
  grid: GPURasterMetadata,
  target: Float32Array
): void {
  target[2] = settings.zFactor ?? 1;
  writeRasterGridParameterValues(target, grid, TERRAIN_GEOMORPHOMETRY_CELL_SLOTS);
}

/**
 * Settings slots of the shared prefix, for
 * {@link getRasterGroundCellSizeWGSL}: cell size at index 0 and the row edges at 3 and 4.
 *
 * @internal
 */
export const TERRAIN_GEOMORPHOMETRY_CELL_SLOTS = {cellSizeIndex: 0, northEdgeIndex: 3} as const;
