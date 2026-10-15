// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainFlow,
  GPUTerrainWatersheds,
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainWatershedsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  flowDirections: GraphDataView<'uint32'>;
  labels: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  iterationCount: GraphDataView<'uint32'>;
};

/** Adds drainage basins of `dem` to `graph` on top of a GPUTerrainFlow; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainWatershedsExampleInput) {
  const {graph, grid, dem, flowDirections, labels, converged, iterationCount} = input;
  // Prerequisite: filled D8 routing of the DEM (see GPUTerrainFlow).
  const flowParameters = new GPUParameterBuffer(device, {
    id: 'watersheds-flow-parameters',
    format: 'float32',
    length: GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
    values: getGPUTerrainFlowParameterValues({fillEpsilon: 0.01, streamThreshold: 40}, grid)
  });
  graph.add(
    new GPUTerrainFlow({
      id: 'watersheds-flow',
      grid,
      elevation: makeGPURasterBand(dem, grid),
      parameters: flowParameters.importToGraph(graph),
      fillDepressions: true,
      maximumFillIterations: 64,
      output: {flowDirections}
    })
  );
  // #region example
  graph.add(
    new GPUTerrainWatersheds({
      grid,
      flowDirections, // ESRI D8 codes; no pour points, so one basin per outlet or pit
      output: {labels, status: {converged, iterationCount}} // terminal cell index per cell
    })
  );
  // #endregion example
  return {parameters: flowParameters};
}
