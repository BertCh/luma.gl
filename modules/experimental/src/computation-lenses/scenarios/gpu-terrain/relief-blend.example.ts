// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUReliefBlend,
  getGPUReliefBlendParameterValues,
  GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and the two layer views (computed by earlier nodes of the graph) plus the output. */
export type ReliefBlendExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  hillshade: GraphDataView<'float32'>;
  slope: GraphDataView<'float32'>;
  reliefBlend: GraphDataView<'float32'>;
};

/** Adds a two-layer hillshade and slope blend to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: ReliefBlendExampleInput) {
  const {graph, grid, hillshade, slope, reliefBlend} = input;
  // #region example
  // Bottom to top: hillshade 0–1, then slope 0–50° inverted at 50 % luminosity (VAT recipe).
  const layers = GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL.slice(0, 2);
  const parameters = new GPUParameterBuffer(device, {
    id: 'relief-blend-parameters',
    format: 'float32',
    length: layers.length * GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH,
    values: getGPUReliefBlendParameterValues(layers)
  });
  graph.add(
    new GPUReliefBlend({
      grid,
      layers: [hillshade, slope],
      parameters: parameters.importToGraph(graph),
      output: {blend: reliefBlend} // 0–1 grey
    })
  );
  // #endregion example
  return {parameters};
}
