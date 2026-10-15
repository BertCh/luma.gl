// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUPointHorizonProfile,
  GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH,
  getGPUPointHorizonProfileParameterValues
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views used by this example. */
export type PointHorizonProfileExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  observers: GraphDataView<'float32x4'>;
  skylineAngle: GraphDataView<'float32'>;
  tangent: GraphDataView<'float32'>;
};

/** Adds the terrain computation and returns its parameter buffer for disposal. */
export async function runExample(device: Device, input: PointHorizonProfileExampleInput) {
  const {graph, grid, dem, observers, skylineAngle, tangent} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'point-horizon-profile-parameters',
    format: 'float32',
    length: GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH,
    values: getGPUPointHorizonProfileParameterValues(
      {projection: 'planar', curvatureCoefficient: 6.8278187e-8},
      grid
    )
  });
  graph.add(
    new GPUPointHorizonProfile({
      grid,
      elevation: makeGPURasterBand(dem, grid),
      projection: 'planar',
      traversal: 'march',
      azimuthCount: 36,
      maximumDistance: 10000,
      cellSize: 120,
      observers,
      parameters: parameters.importToGraph(graph),
      output: {skylineAngle, tangent}
    })
  );
  // #endregion example
  return {parameters};
}
