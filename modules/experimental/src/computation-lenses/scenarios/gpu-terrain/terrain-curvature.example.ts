// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainCurvature,
  getGPUTerrainCurvatureParameterValues,
  GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainCurvatureExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  curvatureProfile: GraphDataView<'float32'>;
  curvaturePlan: GraphDataView<'float32'>;
};

/** Adds profile and plan curvature of `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainCurvatureExampleInput) {
  const {graph, grid, dem, curvatureProfile, curvaturePlan} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'curvature-parameters',
    format: 'float32',
    length: GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH,
    values: getGPUTerrainCurvatureParameterValues({zFactor: 1}, grid)
  });
  graph.add(
    new GPUTerrainCurvature({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      method: 'evans-young',
      cellSizeMode: 'uniform',
      output: {curvatures: {profile: curvatureProfile, plan: curvaturePlan}} // 1/m, convex positive
    })
  );
  // #endregion example
  return {parameters};
}
