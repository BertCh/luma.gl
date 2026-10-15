// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCompaction,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station inputs and caller-owned destination handles. */
export type CompactionExampleInput = {
  stationRows: GraphDataView<'uint32'>;
  wet: GraphDataView<'uint32'>;
  wetStationsBuffer: GraphBufferHandle;
  wetCountBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {stationRows, wet, wetStationsBuffer, wetCountBuffer}: CompactionExampleInput
) {
  // #region example
  const wetStations = graph.createDataView(wetStationsBuffer, {
    format: 'uint32',
    length: stationRows.length
  });
  const wetCount = graph.createDataView(wetCountBuffer, {format: 'uint32', length: 1});
  graph.add(
    new GPUCompaction({
      input: stationRows,
      flags: wet,
      output: wetStations,
      count: wetCount
    })
  );
  // wetStations[0..wetCount) contains original station row identifiers, preserving source order.
  return {wetStations, wetCount};
  // #endregion example
}
