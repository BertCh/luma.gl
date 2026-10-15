// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPULocalDominance,
  GPUParameterBuffer,
  getGPULocalDominanceParameterValues,
  GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type LocalDominanceExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  localDominance: GraphDataView<'float32'>;
};

/** Adds the local dominance of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: LocalDominanceExampleInput) {
  const {graph, grid, dem, localDominance} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'dominance-parameters',
    format: 'float32',
    length: GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH,
    values: getGPULocalDominanceParameterValues({observerHeight: 1.7, verticalExaggeration: 1})
  });
  graph.add(
    new GPULocalDominance({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      minimumRadius: 5, // cells: rings of 5–10 cells, every 15°
      maximumRadius: 10,
      angularResolution: 15,
      output: {dominance: localDominance} // 1 on a flat surface, larger on summits
    })
  );
  // #endregion example
  return {parameters};
}
