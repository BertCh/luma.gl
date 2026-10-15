// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  type GPURasterMetadata,
  GPUParameterBuffer,
  GPURasterCellStatistics,
  GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH,
  getGPURasterCellStatisticsParameterValues
} from '@luma.gl/experimental/gpu-raster';

/** Graph, grid and caller-owned views for raster-cell-statistics. */
export type RasterCellStatisticsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  nirStack: GraphDataView<'float32'>;
  nirMean: GraphDataView<'float32'>;
};

/** Adds the recorded raster-cell-statistics computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: RasterCellStatisticsExampleInput) {
  const {graph, grid, nirStack, nirMean} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'cell-statistics-parameters',
    format: 'float32',
    length: GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH,
    // A missing layer makes the cell missing.
    values: getGPURasterCellStatisticsParameterValues({
      noDataPolicy: 'propagate',
      minimumValidCount: 2
    })
  });
  graph.add(
    new GPURasterCellStatistics({
      cellCount: grid.width * grid.height,
      layerCount: 2, // NIR before, then NIR after, one layer after the other
      stack: nirStack, // reflectance; NaN means missing
      parameters: parameters.importToGraph(graph),
      output: {mean: nirMean}
    })
  );
  // #endregion example
  return {parameters};
}
