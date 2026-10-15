// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUHashIndex,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station rows, the caller-owned table handles and the statistics block. */
export type HashIndexExampleInput = {
  stationRows: GraphDataView<'uint32'>;
  tableKeysBuffer: GraphBufferHandle;
  tableValuesBuffer: GraphBufferHandle;
  statistics: GraphDataView<'uint32'>;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {stationRows, tableKeysBuffer, tableValuesBuffer, statistics}: HashIndexExampleInput
) {
  // #region example
  const tableKeys = graph.createDataView(tableKeysBuffer, {format: 'uint32', length: 2048});
  const tableValues = graph.createDataView(tableValuesBuffer, {format: 'uint32', length: 2048});
  graph.add(
    new GPUHashIndex({
      id: 'station-hash',
      keys: stationRows, // sparse original row numbers
      tableKeys,
      tableValues,
      statistics // [unique, duplicates, overflow, invalid, total probes, max probes]
    })
  );
  // Slot s holds tableKeys[s] = a station row and tableValues[s] = its station index.
  return {tableKeys, tableValues};
  // #endregion example
}
