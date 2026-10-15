// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGroupAggregation,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station states and rainfall with the caller-owned per-state destinations. */
export type GroupAggregationExampleInput = {
  stationState: GraphDataView<'uint32'>;
  rainfall: GraphDataView<'float32'>;
  stateCount: number;
  stationCountsBuffer: GraphBufferHandle;
  rainTotalsBuffer: GraphBufferHandle;
  rainMeansBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    stationState,
    rainfall,
    stateCount: length,
    stationCountsBuffer,
    rainTotalsBuffer,
    rainMeansBuffer
  }: GroupAggregationExampleInput
) {
  // #region example
  const stationCounts = graph.createDataView(stationCountsBuffer, {format: 'uint32', length});
  const rainTotals = graph.createDataView(rainTotalsBuffer, {format: 'float32', length});
  const rainMeans = graph.createDataView(rainMeansBuffer, {format: 'float32', length});
  graph.add([
    new GPUGroupAggregation({id: 'counts', keys: stationState, output: stationCounts}),
    new GPUGroupAggregation({
      id: 'totals',
      keys: stationState,
      values: rainfall,
      output: rainTotals,
      operation: 'sum'
    }),
    new GPUGroupAggregation({
      id: 'means',
      keys: stationState,
      values: rainfall,
      output: rainMeans,
      operation: 'mean'
    })
  ]);
  // Row g holds the station count, total and mean rainfall (mm) of state g.
  return {stationCounts, rainTotals, rainMeans};
  // #endregion example
}
