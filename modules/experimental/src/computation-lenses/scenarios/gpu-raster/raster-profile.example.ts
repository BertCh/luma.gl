// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPURasterProfile,
  type GPURasterProfileOutput,
  GPU_RASTER_PROFILE_PARAMETER_LENGTH,
  getGPURasterProfileParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-profile. */
export type RasterProfileExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  /** Elevation in metres, rows as in the grid (north to south). */
  dem: GraphDataView<'float32'>;
  /** Line vertices in metres from the south-west corner of the window. */
  vertices: GraphDataView<'float32x2'>;
  lineOffsets: GraphDataView<'uint32'>;
  sampleCount: number;
  output: GPURasterProfileOutput;
};

/** Adds the recorded raster-profile computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: RasterProfileExampleInput) {
  const {graph, grid, dem, vertices, lineOffsets, sampleCount, output} = input;
  const parameters = new GPUParameterBuffer(device, {
    id: 'profile-parameters',
    format: 'float32',
    length: GPU_RASTER_PROFILE_PARAMETER_LENGTH
  });
  // #region example
  const [west, north, cell] = [grid.affine[2], grid.affine[5], grid.affine[0]];
  const south = north - grid.height * cell;
  parameters.write(
    getGPURasterProfileParameterValues(
      {spacing: 250, method: 'bilinear', localOrigin: [west, south]}, // a sample every 250 m
      grid
    )
  );
  graph.add(
    new GPURasterProfile({
      grid,
      values: dem, // elevation in metres, UTM 10N grid
      positions: vertices, // one straight line, metres from the corner
      lineOffsets, // [0, 2]: the line owns both vertices
      sampleCapacity: sampleCount,
      parameters: parameters.importToGraph(graph),
      output // status, sample positions, distances and values, total ascent
    })
  );
  // #endregion example
  return {parameters};
}
