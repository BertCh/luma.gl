// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {GPUTerrainRuggedness} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainRuggednessExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  terrainRuggednessIndex: GraphDataView<'float32'>;
};

/** Adds the Riley terrain ruggedness index of `dem` to `graph`; owns no resources to destroy. */
export async function runExample(_device: Device, input: TerrainRuggednessExampleInput) {
  const {graph, grid, dem, terrainRuggednessIndex} = input;
  // #region example
  graph.add(
    new GPUTerrainRuggedness({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      terrainRuggednessAlgorithm: 'riley',
      edgeMode: 'nodata', // the outer ring stays NaN
      output: {terrainRuggednessIndex} // metres of elevation
    })
  );
  // #endregion example
  return {};
}
