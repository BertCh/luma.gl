// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainViewshed,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  getGPUTerrainViewshedParameterValues
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views used by this example. */
export type TerrainViewshedExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  viewshedVisibility: GraphDataView<'uint32'>;
};

/** Adds the terrain computation and returns its parameter buffer for disposal. */
export async function runExample(device: Device, input: TerrainViewshedExampleInput) {
  const {graph, grid, dem, viewshedVisibility} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'terrain-viewshed-parameters',
    format: 'float32',
    length: GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
    values: getGPUTerrainViewshedParameterValues(
      {observer: [45, 49], observerHeight: 1.7, curvatureCoefficient: 6.8278187e-8},
      grid
    )
  });
  graph.add(
    new GPUTerrainViewshed({
      grid,
      elevation: makeGPURasterBand(dem, grid),
      cellSizeMode: 'uniform',
      parameters: parameters.importToGraph(graph),
      output: {visibility: viewshedVisibility}
    })
  );
  // #endregion example
  return {parameters};
}
