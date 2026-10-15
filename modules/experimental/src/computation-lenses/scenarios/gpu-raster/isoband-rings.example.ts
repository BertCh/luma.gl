// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUIsobandRings,
  GPUParameterBuffer,
  GPU_ISOBANDS_PARAMETER_LENGTH,
  getGPUIsobandsParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for isoband-rings. */
export type IsobandRingsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  /** Elevation with row 0 at the southern edge, as GPUIsobandRings expects. */
  dem: GraphDataView<'float32'>;
  breaks: GraphDataView<'float32'>;
  ringOffsets: GraphDataView<'uint32'>;
  positions: GraphDataView<'float32x2'>;
  ringGroups: GraphDataView<'uint32'>;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

/** Adds the recorded isoband-rings computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: IsobandRingsExampleInput) {
  const {graph, grid, dem, breaks, ringOffsets, positions, ringGroups, count, overflow} = input;
  // #region example
  const [west, north, cell] = [grid.affine[2], grid.affine[5], grid.affine[0]];
  const south = north - grid.height * cell;
  const extent = [west, south, west + grid.width * cell, north] as const; // UTM 10N metres
  const parameters = new GPUParameterBuffer(device, {
    id: 'isoband-rings-parameters',
    format: 'float32',
    length: GPU_ISOBANDS_PARAMETER_LENGTH,
    values: getGPUIsobandsParameterValues(
      {breakCount: 7, extent, localOrigin: [west, south]}, // vertices relative to the corner
      grid
    )
  });
  graph.add(
    new GPUIsobandRings({
      grid,
      values: dem, // elevation in metres, row 0 at the southern edge
      breaks, // 1100, 1200, ... 1700 m: eight bands
      parameters: parameters.importToGraph(graph),
      edgeCapacity: 3600, // band boundary edges before chaining
      output: {ringOffsets, positions, ringGroups, status: {count, overflow}} // ring band = group
    })
  );
  // #endregion example
  return {parameters};
}
