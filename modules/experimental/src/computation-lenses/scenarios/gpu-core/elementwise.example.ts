// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUElementwise,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station rainfall and the caller-owned destination handle. */
export type ElementwiseExampleInput = {
  rainfall: GraphDataView<'float32'>;
  prevDayRainfall: GraphDataView<'float32'>;
  twoDayRainfallBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {rainfall, prevDayRainfall, twoDayRainfallBuffer}: ElementwiseExampleInput
) {
  // #region example
  const twoDayRainfall = graph.createDataView(twoDayRainfallBuffer, {
    format: 'float32',
    length: rainfall.length
  });
  graph.add(
    new GPUElementwise({
      id: 'two-day-rainfall',
      input: rainfall,
      inputB: prevDayRainfall,
      output: twoDayRainfall,
      operation: 'add'
    })
  );
  // twoDayRainfall[i] = rainfall[i] + prevDayRainfall[i], in millimetres.
  return {twoDayRainfall};
  // #endregion example
}
