// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station rainfall and the caller-owned destination handle. */
export type ReductionExampleInput = {
  rainfall: GraphDataView<'float32'>;
  maxRainfallBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {rainfall, maxRainfallBuffer}: ReductionExampleInput
) {
  // #region example
  const maxRainfall = graph.createDataView(maxRainfallBuffer, {format: 'float32', length: 1});
  graph.add(
    new GPUReduction({
      id: 'rainfall-max',
      input: rainfall,
      output: maxRainfall,
      operation: 'max'
    })
  );
  // maxRainfall[0] is the largest finite rainfall in millimetres over all stations.
  return {maxRainfall};
  // #endregion example
}
