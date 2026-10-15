// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station keys, station indices and caller-owned destination handles. */
export type SortExampleInput = {
  rainfallTenthsMm: GraphDataView<'uint32'>;
  stationIndex: GraphDataView<'uint32'>;
  rankedRainfallBuffer: GraphBufferHandle;
  rankedStationsBuffer: GraphBufferHandle;
  keyBits: number;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    rainfallTenthsMm,
    stationIndex,
    rankedRainfallBuffer,
    rankedStationsBuffer,
    keyBits
  }: SortExampleInput
) {
  // #region example
  const length = rainfallTenthsMm.length;
  const rankedRainfall = graph.createDataView(rankedRainfallBuffer, {format: 'uint32', length});
  const rankedStations = graph.createDataView(rankedStationsBuffer, {format: 'uint32', length});
  graph.add(
    new GPUSort({
      id: 'rainfall-sort',
      keys: rainfallTenthsMm,
      values: stationIndex,
      outputKeys: rankedRainfall,
      outputValues: rankedStations,
      algorithm: 'radix',
      direction: 'descending',
      keyBits
    })
  );
  // rankedStations[0] is the wettest station; ties keep station order.
  return {rankedRainfall, rankedStations};
  // #endregion example
}
