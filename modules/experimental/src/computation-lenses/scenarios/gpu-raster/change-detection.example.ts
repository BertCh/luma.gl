// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPUChangeDetection,
  GPU_CHANGE_DETECTION_PARAMETER_LENGTH,
  getGPUChangeDetectionParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for change-detection. */
export type ChangeDetectionExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  bands: {name: string; slices: GraphDataView<'float32'>; difference: GraphDataView<'float32'>}[];
};

/** Adds the recorded change-detection computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: ChangeDetectionExampleInput) {
  const {graph, grid, bands} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'change-parameters',
    format: 'float32',
    length: GPU_CHANGE_DETECTION_PARAMETER_LENGTH,
    values: getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 1})
  });
  const parameterView = parameters.importToGraph(graph);
  for (const {name, slices, difference} of bands) {
    graph.add(
      new GPUChangeDetection({
        id: `change-${name}`,
        cellCount: grid.width * grid.height,
        sliceCount: 2, // before, after; cell-major, reflectance with NaN for missing
        slices,
        parameters: parameterView,
        output: {difference} // after minus before, 0 = unchanged
      })
    );
  }
  // #endregion example
  return {parameters};
}
