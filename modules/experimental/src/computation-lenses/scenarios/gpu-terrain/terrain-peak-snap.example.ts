// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterBand, type GPURasterMetadata} from '@luma.gl/experimental/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainPeakSnap,
  getGPUTerrainPeakSnapParameterValues,
  GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainPeakSnapExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  /** Three candidates as `column, row` pairs: six float32 values. */
  candidateValues: GraphDataView<'float32'>;
  /** Snapped positions as `column, row` pairs: six float32 values. */
  positionValues: GraphDataView<'float32'>;
  heights: GraphDataView<'float32'>;
  snapStatus: GraphDataView<'uint32'>;
  snapDistance: GraphDataView<'float32'>;
};

/** Adds peak snapping of three candidates on `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainPeakSnapExampleInput) {
  const {graph, grid, dem, heights, snapStatus, snapDistance} = input;
  // The coordinate pairs are read as float32x2 rows of the same buffers.
  const candidates = graph.createDataView(input.candidateValues.buffer, {
    format: 'float32x2',
    length: 3
  });
  const positions = graph.createDataView(input.positionValues.buffer, {
    format: 'float32x2',
    length: 3
  });
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'peak-snap-parameters',
    format: 'float32',
    length: GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH,
    values: getGPUTerrainPeakSnapParameterValues({radius: 600, maximumMove: 300}, grid) // m
  });
  graph.add(
    new GPUTerrainPeakSnap({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      candidates, // column, row in cell-centre index space
      parameters: parameters.importToGraph(graph),
      maximumRadiusPixels: 8,
      output: {positions, heights, snapStatus, snapDistance}
    })
  );
  // #endregion example
  return {parameters};
}
