// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPUPolygonRasterization,
  GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH,
  getGPUPolygonRasterizationParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for polygon-rasterization. */
export type PolygonRasterizationExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  perimeterPositions: GraphDataView<'float32x2'>;
  featureOffsets: GraphDataView<'uint32'>;
  polygonOffsets: GraphDataView<'uint32'>;
  ringOffsets: GraphDataView<'uint32'>;
  zones: GraphDataView<'uint32'>;
  boundary: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  crossingCount: GraphDataView<'uint32'>;
};

/** Adds the recorded polygon-rasterization computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: PolygonRasterizationExampleInput) {
  const {graph, grid, perimeterPositions, featureOffsets, polygonOffsets, ringOffsets} = input;
  const {zones, boundary, overflow, crossingCount} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'rasterization-parameters',
    format: 'float32',
    length: GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH,
    values: getGPUPolygonRasterizationParameterValues(grid)
  });
  graph.add(
    new GPUPolygonRasterization({
      grid, // 64 × 64 cells of 120 m, north-up
      parameters: parameters.importToGraph(graph),
      positions: perimeterPositions, // UTM 10N metres, exact in float32 at 1 m
      featureOffsets, // one feature: the whole perimeter
      polygonOffsets, // two polygons: a small island and the main fire with its hole
      ringOffsets, // three rings
      crossingCapacity: 256,
      output: {zones, boundary, status: {overflow, crossingCount}} // zone 0 or no zone
    })
  );
  // #endregion example
  return {parameters};
}
