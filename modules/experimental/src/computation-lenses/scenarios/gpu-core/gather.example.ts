// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGather,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded previous-day rainfall, one state's station indices and the destination handle. */
export type GatherExampleInput = {
  prevDayRainfall: GraphDataView<'float32'>;
  stateStations: GraphDataView<'uint32'>;
  statePrevDayRainfallBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {prevDayRainfall, stateStations, statePrevDayRainfallBuffer}: GatherExampleInput
) {
  // #region example
  const statePrevDayRainfall = graph.createDataView(statePrevDayRainfallBuffer, {
    format: 'float32',
    length: stateStations.length
  });
  graph.add(
    new GPUGather({
      id: 'state-prev-day-rainfall',
      source: prevDayRainfall,
      indices: stateStations,
      output: statePrevDayRainfall
    })
  );
  // statePrevDayRainfall[i] = prevDayRainfall[stateStations[i]], in millimetres.
  return {statePrevDayRainfall};
  // #endregion example
}
