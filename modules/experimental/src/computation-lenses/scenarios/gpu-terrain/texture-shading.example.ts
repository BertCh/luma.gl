// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUTextureShadingParameterValues,
  GPUParameterBuffer,
  GPUTextureShading,
  GPU_TEXTURE_SHADING_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TextureShadingExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  textureShade: GraphDataView<'float32'>;
};

/** Adds Brown texture shading of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TextureShadingExampleInput) {
  const {graph, grid, dem, textureShade} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'texture-parameters',
    format: 'float32',
    length: GPU_TEXTURE_SHADING_PARAMETER_LENGTH,
    values: getGPUTextureShadingParameterValues({detail: 0.5})
  });
  graph.add(
    new GPUTextureShading({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      levelCount: 5, // Gaussian sigmas of 1, 2, 4, 8 and 16 cells
      output: {textureShade} // metres: positive on convex relief
    })
  );
  // #endregion example
  return {parameters};
}
