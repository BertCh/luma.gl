// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGridAggregation,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station positions and rainfall with the caller-owned per-cell total handle. */
export type GridAggregationExampleInput = {
  stationPositions: GraphDataView<'float32x2'>;
  rainfall: GraphDataView<'float32'>;
  rainfallPerCellBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {stationPositions, rainfall, rainfallPerCellBuffer}: GridAggregationExampleInput
) {
  // #region example
  const rainfallPerCell = graph.createDataView(rainfallPerCellBuffer, {
    format: 'float32',
    length: 18 * 16
  });
  graph.add(
    new GPUGridAggregation({
      id: 'station-grid-rainfall',
      positions: stationPositions, // interleaved lon, lat
      weights: rainfall,
      output: rainfallPerCell,
      operation: 'sum',
      gridSize: [18, 16],
      bounds: [-92, 24, -74, 40]
    })
  );
  // rainfallPerCell[row * 18 + column] totals the rainfall in millimetres of that cell's stations.
  return {rainfallPerCell};
  // #endregion example
}
