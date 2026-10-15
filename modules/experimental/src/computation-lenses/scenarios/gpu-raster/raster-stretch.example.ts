// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPURasterStretch,
  GPU_RASTER_STRETCH_PARAMETER_LENGTH,
  getGPURasterStretchParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-stretch. */
export type RasterStretchExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  nirAfter: GraphDataView<'float32'>;
  stretchedNir: GraphDataView<'float32'>;
};

/** Adds the recorded raster-stretch computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: RasterStretchExampleInput) {
  const {graph, grid, nirAfter, stretchedNir} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'stretch-parameters',
    format: 'float32',
    length: GPU_RASTER_STRETCH_PARAMETER_LENGTH,
    values: getGPURasterStretchParameterValues({mode: 'linear', domain: 'auto', gamma: 1})
  });
  graph.add(
    new GPURasterStretch({
      grid,
      values: nirAfter, // reflectance; NaN means missing
      binCount: 256,
      parameters: parameters.importToGraph(graph),
      output: {stretched: stretchedNir} // 0 at the smallest valid value, 1 at the largest
    })
  );
  // #endregion example
  return {parameters};
}
