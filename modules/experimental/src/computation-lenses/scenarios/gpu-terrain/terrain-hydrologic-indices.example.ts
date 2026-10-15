// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainFlow,
  GPUTerrainHydrologicIndices,
  getGPUTerrainFlowParameterValues,
  getGPUTerrainHydrologicIndicesParameterValues,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainHydrologicIndicesExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  filledElevation: GraphDataView<'float32'>;
  accumulation: GraphDataView<'float32'>;
  slope: GraphDataView<'float32'>;
  specificCatchmentArea: GraphDataView<'float32'>;
  wetnessIndex: GraphDataView<'float32'>;
  streamPowerIndex: GraphDataView<'float32'>;
};

/** Adds TWI and SPI of `dem` to `graph` on top of a GPUTerrainFlow; returns the parameter buffers to destroy. */
export async function runExample(device: Device, input: TerrainHydrologicIndicesExampleInput) {
  const {graph, grid, dem, filledElevation, accumulation} = input;
  const {slope, specificCatchmentArea, wetnessIndex, streamPowerIndex} = input;
  // Prerequisite: filled D8 routing with contributing area in m² (see GPUTerrainFlow).
  const flowParameters = new GPUParameterBuffer(device, {
    id: 'indices-flow-parameters',
    format: 'float32',
    length: GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
    values: getGPUTerrainFlowParameterValues({fillEpsilon: 0.01}, grid)
  });
  graph.add(
    new GPUTerrainFlow({
      id: 'indices-flow',
      grid,
      elevation: makeGPURasterBand(dem, grid),
      parameters: flowParameters.importToGraph(graph),
      fillDepressions: true,
      maximumFillIterations: 64,
      accumulationUnits: 'area',
      maximumAccumulationIterations: 16,
      output: {filledElevation, accumulation}
    })
  );
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'indices-parameters',
    format: 'float32',
    length: GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
    values: getGPUTerrainHydrologicIndicesParameterValues({minimumSlope: 0.001}, grid)
  });
  graph.add(
    new GPUTerrainHydrologicIndices({
      grid,
      elevation: makeGPURasterBand(filledElevation, grid, {id: 'filled'}), // m
      accumulation, // upstream area, m²
      parameters: parameters.importToGraph(graph),
      output: {slope, specificCatchmentArea, wetnessIndex, streamPowerIndex}
    })
  );
  // #endregion example
  return {parameters, flowParameters};
}
