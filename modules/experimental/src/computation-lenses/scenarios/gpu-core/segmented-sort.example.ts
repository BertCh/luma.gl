// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUSegmentedSort,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station keys, station indices, state starts and caller-owned destinations. */
export type SegmentedSortExampleInput = {
  rainfallTenthsMm: GraphDataView<'uint32'>;
  stationIndex: GraphDataView<'uint32'>;
  stateOffsets: readonly number[];
  sortedRainfallBuffer: GraphBufferHandle;
  sortedStationsBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    rainfallTenthsMm,
    stationIndex,
    stateOffsets,
    sortedRainfallBuffer,
    sortedStationsBuffer
  }: SegmentedSortExampleInput
) {
  // #region example
  const length = rainfallTenthsMm.length;
  const sortedRainfall = graph.createDataView(sortedRainfallBuffer, {format: 'uint32', length});
  const sortedStations = graph.createDataView(sortedStationsBuffer, {format: 'uint32', length});
  const stateSegments = stateOffsets.slice(0, -1).map((offset, state) => ({
    keysOffset: offset,
    valuesOffset: offset,
    outputKeysOffset: offset,
    outputValuesOffset: offset,
    length: stateOffsets[state + 1] - offset
  }));
  graph.add(
    new GPUSegmentedSort({
      id: 'state-rainfall-sort',
      keys: rainfallTenthsMm,
      values: stationIndex,
      outputKeys: sortedRainfall,
      outputValues: sortedStations,
      segments: stateSegments,
      direction: 'descending'
    })
  );
  // Within each state's stations, sortedRainfall runs from wettest to driest.
  return {sortedRainfall, sortedStations};
  // #endregion example
}
