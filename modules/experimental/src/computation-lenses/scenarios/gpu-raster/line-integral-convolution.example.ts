// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPULineIntegralConvolution,
  GPUParameterBuffer,
  GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH,
  GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH,
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionUint32ParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grids and caller-owned views for line-integral-convolution. */
export type LineIntegralConvolutionExampleInput = {
  graph: GPUCommandGraph;
  /** Wind grid: 64 by 48 cells of 0.25 degrees. */
  field: GPURasterMetadata;
  /** Output pixels: 128 by 96 pixels of 0.125 degrees, rows counted from the south. */
  grid: GPURasterMetadata;
  /** Wind in degrees per hour, rows from the south edge as the contributor expects. */
  velocities: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  speeds: GraphDataView<'float32'>;
};

/** Adds the recorded line-integral-convolution computation; returns its owned parameter buffers. */
export async function runExample(device: Device, input: LineIntegralConvolutionExampleInput) {
  const {graph, field, grid, velocities, values, speeds} = input;
  const parameters = new GPUParameterBuffer(device, {
    id: 'lic-parameters',
    format: 'float32',
    length: GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH
  });
  const words = new GPUParameterBuffer(device, {
    id: 'lic-words',
    format: 'uint32',
    length: GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH
  });
  // #region example
  const [west, cell, pixel] = [field.affine[2], field.affine[0], grid.affine[0]];
  const south = field.affine[5] - field.height * cell; // field rows start at the south edge
  parameters.write(
    getGPULineIntegralConvolutionParameterValues({
      fieldExtent: [west, south, cell, cell], // degrees
      outputExtent: [west, south, pixel, pixel],
      stepLength: 0.5 // output pixels per step
    })
  );
  words.write(getGPULineIntegralConvolutionUint32ParameterValues({seed: 7})); // noise seed
  graph.add(
    new GPULineIntegralConvolution({
      velocities, // degrees per hour
      fieldWidth: field.width,
      fieldHeight: field.height,
      grid,
      stepCount: 24, // 24 steps each way: 12 pixels
      parameters: parameters.importToGraph(graph),
      uint32Parameters: words.importToGraph(graph),
      output: {values, speeds} // values in 0 to 1
    })
  );
  // #endregion example
  return {parameters: [parameters, words]};
}
