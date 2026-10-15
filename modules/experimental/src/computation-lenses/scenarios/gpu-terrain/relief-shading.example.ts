// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUReliefShadingParameterValues,
  GPUParameterBuffer,
  GPUReliefShading,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type ReliefShadingExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  hillshade: GraphDataView<'float32'>;
  relief: GraphDataView<'float32'>;
};

/** Adds multidirectional relief shading of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: ReliefShadingExampleInput) {
  const {graph, grid, dem, hillshade, relief} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'relief-parameters',
    format: 'float32',
    length: GPU_RELIEF_SHADING_PARAMETER_LENGTH,
    values: getGPUReliefShadingParameterValues(
      {hillshadeStrength: 0.7, exposure: 1.1}, // USGS multidirectional lights by default
      grid
    )
  });
  graph.add(
    new GPUReliefShading({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      output: {hillshade, relief} // both 0–1
    })
  );
  // #endregion example
  return {parameters};
}
