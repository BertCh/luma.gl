// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUHistogram,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station rainfall and the caller-owned bin-count handle. */
export type HistogramExampleInput = {
  rainfall: GraphDataView<'float32'>;
  stationsPerBinBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {rainfall, stationsPerBinBuffer}: HistogramExampleInput
) {
  // #region example
  const stationsPerBin = graph.createDataView(stationsPerBinBuffer, {
    format: 'uint32',
    length: 16
  });
  graph.add(
    new GPUHistogram({
      id: 'rainfall-histogram',
      input: rainfall,
      output: stationsPerBin,
      domain: [0, 320]
    })
  );
  // stationsPerBin[b] counts stations with rainfall in the b-th 20 mm bin of [0, 320] mm.
  return {stationsPerBin};
  // #endregion example
}
