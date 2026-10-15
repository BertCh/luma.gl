// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGridIndex,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station positions and the caller-owned index destinations. */
export type GridIndexExampleInput = {
  stationPositions: GraphDataView<'float32x2'>;
  cellOffsetsBuffer: GraphBufferHandle;
  stationIdsBuffer: GraphBufferHandle;
  indexedCount: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    stationPositions,
    cellOffsetsBuffer,
    stationIdsBuffer,
    indexedCount,
    overflow
  }: GridIndexExampleInput
) {
  // #region example
  const cellOffsets = graph.createDataView(cellOffsetsBuffer, {
    format: 'uint32',
    length: 18 * 16 + 1
  });
  const stationIds = graph.createDataView(stationIdsBuffer, {
    format: 'uint32',
    length: stationPositions.length
  });
  graph.add(
    new GPUGridIndex({
      id: 'station-grid',
      positions: stationPositions, // interleaved lon, lat
      gridSize: [18, 16],
      bounds: [-92, 24, -74, 40],
      cellOffsets,
      objectIds: stationIds,
      count: indexedCount,
      overflow
    })
  );
  // stationIds[cellOffsets[c] .. cellOffsets[c + 1]) lists the stations in 1° cell c.
  return {cellOffsets, stationIds};
  // #endregion example
}
