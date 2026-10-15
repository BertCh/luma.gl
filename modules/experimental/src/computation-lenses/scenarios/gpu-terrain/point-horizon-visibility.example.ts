// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUPointHorizonVisibility,
  GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH,
  getGPUPointHorizonVisibilityParameterValues
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views used by this example. */
export type PointHorizonVisibilityExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  observers: GraphDataView<'float32x4'>;
  targets: GraphDataView<'float32x4'>;
  visibility: GraphDataView<'uint32'>;
  details: GraphDataView<'float32x4'>;
};

/** Adds the terrain computation and returns its parameter buffer for disposal. */
export async function runExample(device: Device, input: PointHorizonVisibilityExampleInput) {
  const {graph, grid, dem, observers, targets, visibility, details} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'point-horizon-visibility-parameters',
    format: 'float32',
    length: GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH,
    values: getGPUPointHorizonVisibilityParameterValues(
      {projection: 'planar', curvatureCoefficient: 6.8278187e-8},
      grid
    )
  });
  graph.add(
    new GPUPointHorizonVisibility({
      grid,
      elevation: makeGPURasterBand(dem, grid),
      projection: 'planar',
      traversal: 'march',
      maximumDistance: 10000,
      cellSize: 120,
      observers,
      targets,
      parameters: parameters.importToGraph(graph),
      output: {visibility, details}
    })
  );
  // #endregion example
  return {parameters};
}
