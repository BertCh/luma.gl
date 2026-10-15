// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUTerrainCastShadowParameterValues,
  getSolarPosition,
  GPUParameterBuffer,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPUTerrainCastShadow
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainCastShadowExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  sunVisibility: GraphDataView<'float32'>;
  horizonAngle: GraphDataView<'float32'>;
};

/** Adds the single-sun cast shadow of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainCastShadowExampleInput) {
  const {graph, grid, dem, sunVisibility, horizonAngle} = input;
  // #region example
  const sun = getSolarPosition(new Date('2021-08-05T13:45:00Z'), -120.95, 40.15); // 06:45 PDT
  const parameters = new GPUParameterBuffer(device, {
    id: 'cast-shadow-parameters',
    format: 'float32',
    length: GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
    values: getGPUTerrainCastShadowParameterValues(sun, grid)
  });
  graph.add(
    new GPUTerrainCastShadow({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      output: {sunVisibility, horizonAngle} // 0–1, degrees toward the sun
    })
  );
  // #endregion example
  return {parameters};
}
