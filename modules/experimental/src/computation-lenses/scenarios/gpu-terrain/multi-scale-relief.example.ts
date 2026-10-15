// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUMultiScaleRelief,
  GPUParameterBuffer,
  getGPUMultiScaleReliefParameterValues,
  GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type MultiScaleReliefExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  multiScaleRelief: GraphDataView<'float32'>;
};

/** Adds the multi-scale relief model of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: MultiScaleReliefExampleInput) {
  const {graph, grid, dem, multiScaleRelief} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'msrm-parameters',
    format: 'float32',
    length: GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH,
    values: getGPUMultiScaleReliefParameterValues({verticalExaggeration: 1})
  });
  graph.add(
    new GPUMultiScaleRelief({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      resolution: 120, // metres per cell
      featureMinimum: 360, // metres: filter radii 1 to 5 cells
      featureMaximum: 1200,
      scalingFactor: 1,
      output: {relief: multiScaleRelief} // metres above (+) or below (−) the surrounding scales
    })
  );
  // #endregion example
  return {parameters};
}
