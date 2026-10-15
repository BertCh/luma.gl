// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {GPUTerrainCriticalPoints} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainCriticalPointsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  classes: GraphDataView<'uint32'>;
  signChanges: GraphDataView<'uint32'>;
  counts: GraphDataView<'uint32'>;
};

/** Adds critical-point classification of `dem` to `graph`; nothing needs destroying. */
export async function runExample(_device: Device, input: TerrainCriticalPointsExampleInput) {
  const {graph, grid, dem, classes, signChanges, counts} = input;
  // #region example
  graph.add(
    new GPUTerrainCriticalPoints({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      connectivity: 8,
      output: {classes, signChanges, counts} // class code, ring sign changes, cells per class
    })
  );
  // #endregion example
  return {};
}
