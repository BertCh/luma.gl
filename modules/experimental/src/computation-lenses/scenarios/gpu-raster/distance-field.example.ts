// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPUDistanceField,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  getGPUDistanceFieldParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for distance-field. */
export type DistanceFieldExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  waterSeeds: GraphDataView<'uint32'>;
  waterDistance: GraphDataView<'float32'>;
};

/** Adds the recorded distance-field computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: DistanceFieldExampleInput) {
  const {graph, grid, waterSeeds, waterDistance} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'distance-field-parameters',
    format: 'float32',
    length: GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
    values: getGPUDistanceFieldParameterValues({}, grid)
  });
  graph.add(
    new GPUDistanceField({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      mode: 'exact', // Felzenszwalb-Huttenlocher transform, not jump flooding
      seedMask: waterSeeds, // nonzero marks a seed cell
      parameters: parameters.importToGraph(graph),
      output: {distances: waterDistance} // metres between cell centres
    })
  );
  // #endregion example
  return {parameters};
}
