// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUGeomorphons,
  GPUParameterBuffer,
  getGPUGeomorphonsParameterValues,
  GPU_GEOMORPHONS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type GeomorphonsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  geomorphonForms: GraphDataView<'uint32'>;
};

/** Adds the landform classification of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: GeomorphonsExampleInput) {
  const {graph, grid, dem, geomorphonForms} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'geomorphons-parameters',
    format: 'float32',
    length: GPU_GEOMORPHONS_PARAMETER_LENGTH,
    values: getGPUGeomorphonsParameterValues({flatThresholdDegrees: 1}, grid)
  });
  graph.add(
    new GPUGeomorphons({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      searchRadius: 8, // cells, so lines of sight reach 960 m
      comparison: 'anglev1',
      cellSizeMode: 'uniform',
      output: {forms: geomorphonForms} // 1 flat … 10 pit, 0 invalid
    })
  );
  // #endregion example
  return {parameters};
}
