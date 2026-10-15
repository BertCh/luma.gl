// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPURasterArithmetic,
  GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH,
  getGPURasterArithmeticParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-arithmetic. */
export type RasterArithmeticExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  nirBefore: GraphDataView<'float32'>;
  nirAfter: GraphDataView<'float32'>;
  nirDifference: GraphDataView<'float32'>;
};

/** Adds the recorded raster-arithmetic computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: RasterArithmeticExampleInput) {
  const {graph, grid, nirBefore, nirAfter, nirDifference} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'arithmetic-parameters',
    format: 'float32',
    length: GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH,
    // Sentinel-2 digital numbers to reflectance, then after minus before.
    values: getGPURasterArithmeticParameterValues({
      operation: 'subtract',
      scaleA: 1e-4,
      scaleB: 1e-4
    })
  });
  graph.add(
    new GPURasterArithmetic({
      cellCount: grid.width * grid.height,
      a: nirAfter,
      b: nirBefore,
      noDataValue: 0, // digital number 0 means no data
      parameters: parameters.importToGraph(graph),
      output: {values: nirDifference} // reflectance, 0 = unchanged
    })
  );
  // #endregion example
  return {parameters};
}
