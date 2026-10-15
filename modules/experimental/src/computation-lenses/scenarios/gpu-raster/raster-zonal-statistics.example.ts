// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPURasterZonalStatistics,
  makeGPURasterBand
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-zonal-statistics. */
export type RasterZonalStatisticsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  zones: GraphDataView<'uint32'>;
  cellCounts: GraphDataView<'uint32'>;
  means: GraphDataView<'float32'>;
  minimums: GraphDataView<'float32'>;
  maximums: GraphDataView<'float32'>;
};

/** Adds the recorded raster-zonal-statistics computation; it owns no extra resources. */
export async function runExample(device: Device, input: RasterZonalStatisticsExampleInput) {
  const {graph, grid, dem, zones, cellCounts, means, minimums, maximums} = input;
  // #region example
  graph.add(
    new GPURasterZonalStatistics({
      grid, // 64 × 64 cells of 120 m
      zones, // dense zone ID per cell: 0 tree, 1 grass, 2 crop, 3 built-up, 4 water
      values: makeGPURasterBand(dem, grid), // elevation in metres
      zoneCapacity: 5, // one output row per zone
      output: {cellCounts, means, minimums, maximums} // elevation in metres per zone
    })
  );
  // #endregion example
  return {};
}
