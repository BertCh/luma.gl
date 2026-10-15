// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPURasterConditional,
  GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH,
  getGPURasterConditionalParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-conditional. */
export type RasterConditionalExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  landCover: GraphDataView<'float32'>;
  friction: GraphDataView<'float32'>;
};

/** Adds the recorded raster-conditional computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: RasterConditionalExampleInput) {
  const {graph, grid, landCover, friction} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'conditional-parameters',
    format: 'float32',
    length: GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH,
    // Built-up land (class 50) costs 0.4 per metre, everything else 1.
    values: getGPURasterConditionalParameterValues({
      comparison: '==',
      threshold: 50,
      constantA: 0.4,
      constantB: 1
    })
  });
  graph.add(
    new GPURasterConditional({
      cellCount: grid.width * grid.height,
      conditionValues: landCover,
      noDataValue: 0,
      parameters: parameters.importToGraph(graph),
      output: {values: friction} // cost per metre
    })
  );
  // #endregion example
  return {parameters};
}
