// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainFlowExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  rainfall: GraphDataView<'float32'>;
  filledElevation: GraphDataView<'float32'>;
  flowDirections: GraphDataView<'uint32'>;
  cellClasses: GraphDataView<'uint32'>;
  accumulation: GraphDataView<'float32'>;
  streams: GraphDataView<'uint32'>;
  fillConverged: GraphDataView<'uint32'>;
  fillIterationCount: GraphDataView<'uint32'>;
  accumulationConverged: GraphDataView<'uint32'>;
  accumulationIterationCount: GraphDataView<'uint32'>;
};

/** Adds D8 hydrology of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainFlowExampleInput) {
  const {graph, grid, dem, rainfall, filledElevation, flowDirections, cellClasses} = input;
  const {accumulation, streams, fillConverged, fillIterationCount} = input;
  const {accumulationConverged, accumulationIterationCount} = input;
  // One-row convergence and iteration words of the two gated loops.
  const status = {
    fillConverged,
    fillIterationCount,
    accumulationConverged,
    accumulationIterationCount
  };
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'flow-parameters',
    format: 'float32',
    length: GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
    values: getGPUTerrainFlowParameterValues({fillEpsilon: 0.01, streamThreshold: 40}, grid)
  });
  graph.add(
    new GPUTerrainFlow({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      fillDepressions: true,
      maximumFillIterations: 64,
      runoff: rainfall, // 1 per cell: accumulation counts upstream cells
      accumulationUnits: 'cells',
      maximumAccumulationIterations: 16,
      output: {filledElevation, flowDirections, cellClasses, accumulation, streams, status}
    })
  );
  // #endregion example
  return {parameters};
}
