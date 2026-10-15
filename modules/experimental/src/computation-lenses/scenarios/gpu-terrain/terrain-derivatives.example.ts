// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainDerivativesExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  slope: GraphDataView<'float32'>;
  aspect: GraphDataView<'float32'>;
  hillshade: GraphDataView<'float32'>;
};

/** Adds Horn slope, aspect and hillshade of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainDerivativesExampleInput) {
  const {graph, grid, dem, slope, aspect, hillshade} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'derivatives-parameters',
    format: 'float32',
    length: GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
    values: getGPUTerrainDerivativesParameterValues(
      {azimuthDegrees: 315, altitudeDegrees: 45},
      grid
    )
  });
  graph.add(
    new GPUTerrainDerivatives({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      output: {slope, aspect, hillshade} // degrees, degrees from north, 0–1
    })
  );
  // #endregion example
  return {parameters};
}
