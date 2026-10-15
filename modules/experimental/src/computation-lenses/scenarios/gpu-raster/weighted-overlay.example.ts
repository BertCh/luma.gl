// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPUWeightedOverlay,
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for weighted-overlay. */
export type WeightedOverlayExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  landscapeStack: GraphDataView<'float32'>;
  remapBreaks: GraphDataView<'float32'>;
  remapValues: GraphDataView<'float32'>;
  suitabilityScore: GraphDataView<'float32'>;
};

/** Adds the recorded weighted-overlay computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: WeightedOverlayExampleInput) {
  const {graph, grid, landscapeStack, remapBreaks, remapValues, suitabilityScore} = input;
  // #region example
  const layers = [
    {weight: 0.7, mode: 'table', breakCount: 4}, // land cover
    {weight: 0.3, mode: 'table', breakCount: 3} // elevation
  ] as const;
  const parameters = new GPUParameterBuffer(device, {
    id: 'overlay-parameters',
    format: 'float32',
    length: getGPUWeightedOverlayParameterLength(layers.length),
    values: getGPUWeightedOverlayParameterValues({layers}, undefined, {hasTables: true})
  });
  graph.add(
    new GPUWeightedOverlay({
      cellCount: grid.width * grid.height,
      layerCount: layers.length,
      stack: landscapeStack, // layer after layer: land cover codes, then elevation in metres
      remapBreaks, // per layer, ascending
      remapValues, // per layer, one more than the breaks
      maximumBreakCount: 4,
      parameters: parameters.importToGraph(graph),
      output: {score: suitabilityScore} // weighted sum of the remapped layers, 0–1
    })
  );
  // #endregion example
  return {parameters};
}
