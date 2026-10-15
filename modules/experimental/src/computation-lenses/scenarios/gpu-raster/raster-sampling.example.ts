// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPURasterSampling,
  GPU_RASTER_SAMPLING_PARAMETER_LENGTH,
  getGPURasterSamplingParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-sampling. */
export type RasterSamplingExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  /** Elevation in metres, rows as in the grid (north to south). */
  dem: GraphDataView<'float32'>;
  /** Sample points in metres from the south-west corner of the window. */
  points: GraphDataView<'float32x2'>;
  elevations: GraphDataView<'float32'>;
};

/** Adds the recorded raster-sampling computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: RasterSamplingExampleInput) {
  const {graph, grid, dem, points, elevations} = input;
  // #region example
  const [west, north, cell] = [grid.affine[2], grid.affine[5], grid.affine[0]];
  const south = north - grid.height * cell;
  const parameters = new GPUParameterBuffer(device, {
    id: 'sampling-parameters',
    format: 'float32',
    length: GPU_RASTER_SAMPLING_PARAMETER_LENGTH,
    values: getGPURasterSamplingParameterValues(
      {method: 'bilinear', localOrigin: [west, south]}, // points relative to the corner
      grid
    )
  });
  graph.add(
    new GPURasterSampling({
      grid,
      values: dem, // elevation in metres, UTM 10N grid
      positions: points, // 25 points, metres from the south-west corner
      parameters: parameters.importToGraph(graph),
      output: {values: elevations} // metres
    })
  );
  // #endregion example
  return {parameters};
}
