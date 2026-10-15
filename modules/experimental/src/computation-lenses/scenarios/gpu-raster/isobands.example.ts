// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUIsobands,
  GPUParameterBuffer,
  GPU_ISOBANDS_PARAMETER_LENGTH,
  getGPUIsobandsParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for isobands. */
export type IsobandsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  /** Elevation with row 0 at the southern edge, as GPUIsobands expects. */
  dem: GraphDataView<'float32'>;
  breaks: GraphDataView<'float32'>;
  triangles: GraphDataView<'float32x2'>;
  triangleBands: GraphDataView<'uint32'>;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

/** Adds the recorded isobands computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: IsobandsExampleInput) {
  const {graph, grid, dem, breaks, triangles, triangleBands, count, overflow} = input;
  // #region example
  const [west, north, cell] = [grid.affine[2], grid.affine[5], grid.affine[0]];
  const south = north - grid.height * cell;
  const extent = [west, south, west + grid.width * cell, north] as const; // UTM 10N metres
  const parameters = new GPUParameterBuffer(device, {
    id: 'isobands-parameters',
    format: 'float32',
    length: GPU_ISOBANDS_PARAMETER_LENGTH,
    values: getGPUIsobandsParameterValues(
      {breakCount: 7, firstBand: 3, lastBand: 5, extent, localOrigin: [west, south]},
      grid
    )
  });
  graph.add(
    new GPUIsobands({
      grid,
      values: dem, // elevation in metres, row 0 at the southern edge
      breaks, // 1100, 1200, ... 1700 m: eight bands, three of them emitted
      parameters: parameters.importToGraph(graph),
      output: {triangles, triangleBands, status: {count, overflow}} // vertices relative to the corner
    })
  );
  // #endregion example
  return {parameters};
}
