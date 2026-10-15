// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainVectorRuggedness,
  getGPUTerrainVectorRuggednessParameterValues,
  GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainVectorRuggednessExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  vectorRuggednessMeasure: GraphDataView<'float32'>;
};

/** Adds the vector ruggedness measure of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainVectorRuggednessExampleInput) {
  const {graph, grid, dem, vectorRuggednessMeasure} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'vector-ruggedness-parameters',
    format: 'float32',
    length: GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH,
    values: getGPUTerrainVectorRuggednessParameterValues({radius: 2}, grid) // 5 × 5 window
  });
  graph.add(
    new GPUTerrainVectorRuggedness({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      maximumRadius: 2,
      cellSizeMode: 'uniform',
      output: {vectorRuggedness: vectorRuggednessMeasure} // 0 smooth … 1 rugged
    })
  );
  // #endregion example
  return {parameters};
}
