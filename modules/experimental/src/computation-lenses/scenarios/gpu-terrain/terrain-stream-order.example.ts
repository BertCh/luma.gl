// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainFlow,
  GPUTerrainStreamOrder,
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainStreamOrderExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  flowDirections: GraphDataView<'uint32'>;
  streams: GraphDataView<'uint32'>;
  streamOrder: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  iterationCount: GraphDataView<'uint32'>;
};

/** Adds Strahler stream order of `dem` to `graph` on top of a GPUTerrainFlow; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainStreamOrderExampleInput) {
  const {graph, grid, dem, flowDirections, streams, streamOrder, converged, iterationCount} = input;
  // Prerequisite: filled D8 routing of the DEM (see GPUTerrainFlow).
  const flowParameters = new GPUParameterBuffer(device, {
    id: 'stream-order-flow-parameters',
    format: 'float32',
    length: GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
    values: getGPUTerrainFlowParameterValues({fillEpsilon: 0.01, streamThreshold: 40}, grid)
  });
  graph.add(
    new GPUTerrainFlow({
      id: 'stream-order-flow',
      grid,
      elevation: makeGPURasterBand(dem, grid),
      parameters: flowParameters.importToGraph(graph),
      fillDepressions: true,
      maximumFillIterations: 64,
      output: {flowDirections, streams}
    })
  );
  // #region example
  graph.add(
    new GPUTerrainStreamOrder({
      grid,
      flowDirections, // ESRI D8 codes
      streams, // 1 on stream cells (40 or more upstream cells)
      output: {streamOrder, status: {converged, iterationCount}} // Strahler order, 0 off streams
    })
  );
  // #endregion example
  return {parameters: flowParameters};
}
