// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {GPUTerrainPeakProminence} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainPeakProminenceExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  prominence: GraphDataView<'float32'>;
  keySaddle: GraphDataView<'float32'>;
  parentPeak: GraphDataView<'uint32'>;
};

/** Adds peak prominence of `dem` to `graph`; nothing needs destroying. */
export async function runExample(_device: Device, input: TerrainPeakProminenceExampleInput) {
  const {graph, grid, dem, prominence, keySaddle, parentPeak} = input;
  // #region example
  graph.add(
    new GPUTerrainPeakProminence({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      output: {prominence, keySaddle, parentPeak} // m, key saddle m, elder peak cell
    })
  );
  // #endregion example
  return {};
}
