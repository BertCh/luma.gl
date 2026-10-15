// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPUIsolinesPolylineOutput,
  type GPURasterMetadata,
  GPUIsolines,
  GPUParameterBuffer,
  GPU_ISOLINES_PARAMETER_LENGTH,
  getGPUIsolinesParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for isolines. */
export type IsolinesExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  /** Elevation with row 0 at the southern edge, as GPUIsolines expects. */
  dem: GraphDataView<'float32'>;
  levels: GraphDataView<'float32'>;
  segments: GraphDataView<'float32x4'>;
  segmentLevels: GraphDataView<'uint32'>;
  polylines: GPUIsolinesPolylineOutput;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  polylineOverflow: GraphDataView<'uint32'>;
};

/** Adds the recorded isolines computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: IsolinesExampleInput) {
  const {graph, grid, dem, levels, segments, segmentLevels, polylines} = input;
  const {count, overflow, polylineOverflow} = input;
  // #region example
  const [west, north, cell] = [grid.affine[2], grid.affine[5], grid.affine[0]];
  const south = north - grid.height * cell;
  const extent = [west, south, west + grid.width * cell, north] as const; // UTM 10N metres
  const parameters = new GPUParameterBuffer(device, {
    id: 'isolines-parameters',
    format: 'float32',
    length: GPU_ISOLINES_PARAMETER_LENGTH,
    values: getGPUIsolinesParameterValues(
      {levelCount: 7, extent, localOrigin: [west, south]}, // vertices relative to the corner
      grid
    )
  });
  graph.add(
    new GPUIsolines({
      grid,
      values: dem, // elevation in metres, row 0 at the southern edge
      levels, // 1100, 1200, ... 1700 m
      parameters: parameters.importToGraph(graph),
      output: {segments, segmentLevels, polylines, status: {count, overflow, polylineOverflow}}
    })
  );
  // #endregion example
  return {parameters};
}
