// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_RASTER_D8_DIRECTIONS,
  getRasterD8Distance,
  getRasterGroundCellSize
} from '../../../src/map-graphs/cost-distance/raster-grid-utils';
import {
  getGPUTerrainHydrologicIndicesParameterValues,
  type GPUTerrainHydrologicIndicesSettings
} from '../../../src/map-graphs/hydrology/gpu-terrain-hydrologic-indices';
import type {GPUTerrainCellSizeMode} from '../../../src/map-graphs/terrain-analysis';

/** Inputs of {@link computeHydrologicIndices}. NaN elevation marks an invalid cell. */
export type HydrologicIndicesOracleInput = {
  elevation: Float32Array;
  accumulation: Float32Array;
  width: number;
  height: number;
  settings: GPUTerrainHydrologicIndicesSettings;
  cellSizeMode?: GPUTerrainCellSizeMode;
};

/** Outputs of {@link computeHydrologicIndices}, float64. */
export type HydrologicIndicesOracleResult = {
  slope: Float64Array;
  specificCatchmentArea: Float64Array;
  wetnessIndex: Float64Array;
  streamPowerIndex: Float64Array;
};

/**
 * Float64 reference of `GPUTerrainHydrologicIndices`: steepest D8 descent clamped to the minimum
 * slope, contour width `sqrt(groundX * groundY)`, `a = A / b`, `ln(a / tan(beta))`, `a * tan(beta)`.
 */
export function computeHydrologicIndices(
  input: HydrologicIndicesOracleInput
): HydrologicIndicesOracleResult {
  const {elevation, accumulation, width, height} = input;
  const cellSizeMode = input.cellSizeMode ?? 'uniform';
  const packed = getGPUTerrainHydrologicIndicesParameterValues(input.settings);
  const gridSettings = {
    cellSize: [packed[0], packed[1]] as [number, number],
    northEdge: packed[2],
    southEdge: packed[3]
  };
  let minimumSlope = packed[4];
  if (!(minimumSlope >= 0) || !Number.isFinite(minimumSlope)) {
    minimumSlope = 0;
  }
  const cellCount = width * height;
  const result: HydrologicIndicesOracleResult = {
    slope: new Float64Array(cellCount).fill(NaN),
    specificCatchmentArea: new Float64Array(cellCount).fill(NaN),
    wetnessIndex: new Float64Array(cellCount).fill(NaN),
    streamPowerIndex: new Float64Array(cellCount).fill(NaN)
  };
  for (let cell = 0; cell < cellCount; cell++) {
    const center = elevation[cell];
    const area = accumulation[cell];
    if (!Number.isFinite(center) || !Number.isFinite(area)) {
      continue;
    }
    const column = cell % width;
    const row = Math.floor(cell / width);
    let steepest = 0;
    for (let direction = 0; direction < 8; direction++) {
      const {columnOffset, rowOffset} = GPU_RASTER_D8_DIRECTIONS[direction];
      const neighborColumn = column + columnOffset;
      const neighborRow = row + rowOffset;
      if (
        neighborColumn < 0 ||
        neighborRow < 0 ||
        neighborColumn >= width ||
        neighborRow >= height
      ) {
        continue;
      }
      const neighborValue = elevation[neighborRow * width + neighborColumn];
      if (!Number.isFinite(neighborValue)) {
        continue;
      }
      const distance = getRasterD8Distance(cellSizeMode, direction, row, height, gridSettings);
      steepest = Math.max(steepest, (center - neighborValue) / distance);
    }
    const tangent = Math.max(steepest, minimumSlope);
    const [groundX, groundY] = getRasterGroundCellSize(
      cellSizeMode,
      row + 0.5,
      height,
      gridSettings
    );
    const catchment = area / Math.sqrt(groundX * groundY);
    result.slope[cell] = tangent;
    result.specificCatchmentArea[cell] = catchment;
    result.wetnessIndex[cell] =
      catchment <= 0 ? -Infinity : tangent > 0 ? Math.log(catchment / tangent) : Infinity;
    result.streamPowerIndex[cell] = catchment * tangent;
  }
  return result;
}
