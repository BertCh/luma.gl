// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainLineOfSight,
  GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH,
  getGPUTerrainLineOfSightParameterValues
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views used by this example. */
export type TerrainLineOfSightExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  pairs: GraphDataView<'float32x4'>;
  visibility: GraphDataView<'uint32'>;
  clearance: GraphDataView<'float32'>;
};

/** Adds the terrain computation and returns its parameter buffer for disposal. */
export async function runExample(device: Device, input: TerrainLineOfSightExampleInput) {
  const {graph, grid, dem, pairs, visibility, clearance} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'terrain-line-of-sight-parameters',
    format: 'float32',
    length: GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH,
    values: getGPUTerrainLineOfSightParameterValues(
      {observerHeight: 1.7, curvatureCoefficient: 6.8278187e-8},
      grid
    )
  });
  graph.add(
    new GPUTerrainLineOfSight({
      grid,
      elevation: makeGPURasterBand(dem, grid),
      cellSizeMode: 'uniform',
      pairs,
      parameters: parameters.importToGraph(graph),
      output: {visibility, clearance}
    })
  );
  // #endregion example
  return {parameters};
}
