// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainHorizon,
  getGPUTerrainHorizonParameterValues,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainHorizonExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  skyViewFactor: GraphDataView<'float32'>;
  positiveOpenness: GraphDataView<'float32'>;
};

/** Adds the 16-sector horizon march of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainHorizonExampleInput) {
  const {graph, grid, dem, skyViewFactor, positiveOpenness} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'horizon-parameters',
    format: 'float32',
    length: GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
    values: getGPUTerrainHorizonParameterValues({}, grid)
  });
  graph.add(
    new GPUTerrainHorizon({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      directionCount: 16,
      maximumRadius: 32, // cells, 3.84 km
      output: {skyViewFactor, positiveOpenness} // 0–1, degrees
    })
  );
  // #endregion example
  return {parameters};
}
