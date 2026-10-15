// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainWeissLandforms,
  getGPUTerrainWeissLandformsParameterValues,
  GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainWeissLandformsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  weissLandforms: GraphDataView<'uint32'>;
  /** Two planes of one value per cell: small scale, then large scale. */
  standardizedPosition: GraphDataView<'float32'>;
};

/** Adds the Weiss landform classes of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainWeissLandformsExampleInput) {
  const {graph, grid, dem, weissLandforms, standardizedPosition} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'weiss-parameters',
    format: 'float32',
    length: GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH,
    values: getGPUTerrainWeissLandformsParameterValues(
      {standardThreshold: 1, slopeThresholdDegrees: 5},
      grid
    )
  });
  graph.add(
    new GPUTerrainWeissLandforms({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      smallScale: {radius: 3}, // cells
      largeScale: {radius: 15},
      cellSizeMode: 'uniform',
      output: {landforms: weissLandforms, standardizedPosition} // classes 1–10; z-scores
    })
  );
  // #endregion example
  return {parameters};
}
