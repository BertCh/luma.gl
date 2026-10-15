// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUTerrainFlowFieldParameterValues,
  GPUParameterBuffer,
  GPUTerrainFlowField,
  GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainFlowFieldExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  velocities: GraphDataView<'float32x2'>;
};

/** Adds the terrain-following wind of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainFlowFieldExampleInput) {
  const {graph, grid, dem, velocities} = input;
  // #region example
  const speed = 8 * Math.SQRT1_2; // 8 m/s from the south-west, toward the north-east
  const parameters = new GPUParameterBuffer(device, {
    id: 'flow-field-parameters',
    format: 'float32',
    length: GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH,
    values: getGPUTerrainFlowFieldParameterValues({wind: [speed, -speed]}, grid) // x east, y south
  });
  graph.add(
    new GPUTerrainFlowField({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      output: {velocities} // (u, v) in m/s, raster frame
    })
  );
  // #endregion example
  return {parameters};
}
