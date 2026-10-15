// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPURasterExtremaPyramid,
  makeGPURasterBand
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-extrema-pyramid. */
export type RasterExtremaPyramidExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  /** Level 0 (16 × 16 cells) followed by level 1 (8 × 8 cells). */
  maximum: GraphDataView<'float32'>;
  minimum: GraphDataView<'float32'>;
};

/** Adds the recorded raster-extrema-pyramid computation; it owns no extra resources. */
export async function runExample(device: Device, input: RasterExtremaPyramidExampleInput) {
  const {graph, grid, dem, maximum, minimum} = input;
  // #region example
  graph.add(
    new GPURasterExtremaPyramid({
      grid, // 64 × 64 cells of 120 m
      input: makeGPURasterBand(dem, grid), // elevation in metres
      firstBlockSize: 4, // level 0 cells cover 4 × 4 pixels, level 1 cells 8 × 8
      levelCount: 2,
      footprint: 'cell', // plain block extrema, no extra pixel
      output: {maximum, minimum} // levels packed one after the other, level 0 first
    })
  );
  // #endregion example
  return {};
}
