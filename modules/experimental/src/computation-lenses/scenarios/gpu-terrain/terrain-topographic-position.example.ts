// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {GPUTerrainTopographicPosition} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainTopographicPositionExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  topographicPositionIndex: GraphDataView<'float32'>;
};

/** Adds the topographic position index of `dem` to `graph`; owns no resources to destroy. */
export async function runExample(_device: Device, input: TerrainTopographicPositionExampleInput) {
  const {graph, grid, dem, topographicPositionIndex} = input;
  // #region example
  graph.add(
    new GPUTerrainTopographicPosition({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      scales: [{radius: 5}], // 11 × 11 cells, centre excluded
      output: {topographicPositionIndex} // metres above (+) or below (−) the neighbourhood mean
    })
  );
  // #endregion example
  return {};
}
