// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUHashIndex,
  GPUHashJoin,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station and wet-station rows with the caller-owned table, pair and count views. */
export type HashJoinExampleInput = {
  stationRows: GraphDataView<'uint32'>;
  wetStationRows: GraphDataView<'uint32'>;
  tableKeys: GraphDataView<'uint32'>;
  tableValues: GraphDataView<'uint32'>;
  indexStatistics: GraphDataView<'uint32'>;
  lookupStatistics: GraphDataView<'uint32'>;
  wetPositions: GraphDataView<'uint32'>;
  wetStationIndices: GraphDataView<'uint32'>;
  pairCount: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    stationRows,
    wetStationRows,
    tableKeys,
    tableValues,
    indexStatistics,
    lookupStatistics,
    wetPositions,
    wetStationIndices,
    pairCount,
    overflow
  }: HashJoinExampleInput
) {
  // #region example
  const stationIndex = new GPUHashIndex({
    id: 'station-hash',
    keys: stationRows,
    tableKeys,
    tableValues,
    statistics: indexStatistics
  });
  graph.add([
    stationIndex,
    new GPUHashJoin({
      id: 'wet-join',
      index: stationIndex,
      keys: wetStationRows,
      outputLeftRows: wetPositions,
      outputRightRows: wetStationIndices,
      count: pairCount,
      overflow,
      statistics: lookupStatistics
    })
  ]);
  // Pair i joins wetStationRows[wetPositions[i]] to station wetStationIndices[i].
  return {wetPositions, wetStationIndices};
  // #endregion example
}
