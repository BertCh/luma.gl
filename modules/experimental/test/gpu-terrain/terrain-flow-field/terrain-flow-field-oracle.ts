// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getRasterD8Distance,
  getRasterGroundCellSize
} from '../../../src/gpu-raster/cost-distance/raster-grid-utils';
import type {GPUTerrainCellSizeMode} from '../../../src/gpu-terrain/terrain-analysis';
import {
  getGPUTerrainFlowFieldParameterValues,
  type GPUTerrainFlowFieldSettings
} from '../../../src/gpu-terrain/terrain-flow-field/gpu-terrain-flow-field';

/** Inputs of {@link computeTerrainFlowField}. NaN elevation marks an invalid cell. */
export type TerrainFlowFieldOracleInput = {
  elevation: Float32Array;
  width: number;
  height: number;
  settings: GPUTerrainFlowFieldSettings;
  cellSizeMode?: GPUTerrainCellSizeMode;
};

/**
 * Float64 reference of `GPUTerrainFlowField`: central-difference gradient in ground meters
 * (one-sided at the border), `v = w - (w . g) g / (1 + |g|^2)`, NaN where a used sample is invalid.
 * Settings are read back from the packed float32 values the GPU sees.
 */
export function computeTerrainFlowField(input: TerrainFlowFieldOracleInput): Float64Array {
  const {elevation, width, height} = input;
  const cellSizeMode = input.cellSizeMode ?? 'uniform';
  const packed = getGPUTerrainFlowFieldParameterValues(input.settings);
  const gridSettings = {
    cellSize: [packed[0], packed[1]] as [number, number],
    northEdge: packed[2],
    southEdge: packed[3]
  };
  const windX = packed[4];
  const windY = packed[5];
  const exaggeration = packed[6];
  const velocities = new Float64Array(width * height * 2);
  const at = (column: number, row: number) => elevation[row * width + column];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const output = 2 * (row * width + column);
      const column0 = Math.max(column - 1, 0);
      const column1 = Math.min(column + 1, width - 1);
      const row0 = Math.max(row - 1, 0);
      const row1 = Math.min(row + 1, height - 1);
      const samples = [
        at(column, row),
        at(column0, row),
        at(column1, row),
        at(column, row0),
        at(column, row1)
      ];
      if (samples.some(value => !Number.isFinite(value))) {
        velocities[output] = NaN;
        velocities[output + 1] = NaN;
        continue;
      }
      let gradientX = 0;
      if (column1 > column0) {
        const [groundX] = getRasterGroundCellSize(cellSizeMode, row + 0.5, height, gridSettings);
        gradientX =
          (exaggeration * (at(column1, row) - at(column0, row))) / ((column1 - column0) * groundX);
      }
      let gradientY = 0;
      if (row1 > row0) {
        let spanY = 0;
        if (row1 > row) {
          spanY += getRasterD8Distance(cellSizeMode, 2, row, height, gridSettings);
        }
        if (row0 < row) {
          spanY += getRasterD8Distance(cellSizeMode, 6, row, height, gridSettings);
        }
        gradientY = (exaggeration * (at(column, row1) - at(column, row0))) / spanY;
      }
      const deflection =
        (windX * gradientX + windY * gradientY) /
        (1 + gradientX * gradientX + gradientY * gradientY);
      velocities[output] = windX - deflection * gradientX;
      velocities[output + 1] = windY - deflection * gradientY;
    }
  }
  return velocities;
}
