// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUSimpleLocalRelief,
  getGPUSimpleLocalReliefParameterValues,
  GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type SimpleLocalReliefExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  simpleLocalRelief: GraphDataView<'float32'>;
};

/** Adds the simple local relief model of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: SimpleLocalReliefExampleInput) {
  const {graph, grid, dem, simpleLocalRelief} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'slrm-parameters',
    format: 'float32',
    length: GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH,
    values: getGPUSimpleLocalReliefParameterValues({verticalExaggeration: 1, radius: 8})
  });
  graph.add(
    new GPUSimpleLocalRelief({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      maximumRadius: 8, // cells, a 17 × 17 window
      output: {relief: simpleLocalRelief} // metres above (+) or below (−) the local mean
    })
  );
  // #endregion example
  return {parameters};
}
