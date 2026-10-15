// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGridBinning,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station positions and the caller-owned cell-count handle. */
export type GridBinningExampleInput = {
  stationPositions: GraphDataView<'float32x2'>;
  stationsPerCellBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {stationPositions, stationsPerCellBuffer}: GridBinningExampleInput
) {
  // #region example
  const stationsPerCell = graph.createDataView(stationsPerCellBuffer, {
    format: 'uint32',
    length: 18 * 16
  });
  graph.add(
    new GPUGridBinning({
      id: 'station-grid-counts',
      positions: stationPositions, // interleaved lon, lat
      output: stationsPerCell,
      gridSize: [18, 16],
      bounds: [-92, 24, -74, 40]
    })
  );
  // stationsPerCell[row * 18 + column] counts the stations in that 1° cell.
  return {stationsPerCell};
  // #endregion example
}
