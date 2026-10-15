// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {GPUTerrainContours, type GPUTerrainContourLevel} from '@luma.gl/experimental/gpu-terrain';

/** Caller-owned level vertices and status words on the elevation grid. */
export type TerrainContoursExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  levels: readonly GPUTerrainContourLevel[];
  overflow: GraphDataView<'uint32'>;
};

/** Adds all contour levels; the contributor owns no resources requiring disposal. */
export async function runExample(device: Device, input: TerrainContoursExampleInput) {
  const {graph, grid, dem, levels, overflow} = input;
  // #region example
  graph.add(
    new GPUTerrainContours({
      grid,
      elevation: makeGPURasterBand(dem, grid),
      levels, // 1100–1700 m every 100 m; local pixel-center vertices
      output: {status: {overflow}}
    })
  );
  // #endregion example
  return {};
}
