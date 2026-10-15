// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {GPUTerrainRGBDecode} from '@luma.gl/experimental/gpu-terrain';

/** Packed Terrarium words and caller-owned decoded heights and validity. */
export type TerrainRGBDecodeExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  terrariumRGBA: GraphDataView<'uint32'>;
  decodedElevation: GraphDataView<'float32'>;
  decodedValidity: GraphDataView<'uint32'>;
};

/** Decodes exact byte channels without filtering; no owned resources need disposal. */
export async function runExample(device: Device, input: TerrainRGBDecodeExampleInput) {
  const {graph, grid, terrariumRGBA, decodedElevation, decodedValidity} = input;
  // #region example
  graph.add(
    new GPUTerrainRGBDecode({
      grid, // 32 × 32 cells of 20 m
      encoding: 'terrarium',
      input: {buffer: terrariumRGBA},
      output: {values: decodedElevation, outputValidity: decodedValidity}
    })
  );
  // #endregion example
  return {};
}
