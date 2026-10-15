// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUMask,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station inputs and caller-owned destination handles. */
export type MaskExampleInput = {
  wet: GraphDataView<'uint32'>;
  high: GraphDataView<'uint32'>;
  heavyRainHighStationsBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {wet, high, heavyRainHighStationsBuffer}: MaskExampleInput
) {
  // #region example
  const heavyRainHighStations = graph.createDataView(heavyRainHighStationsBuffer, {
    format: 'uint32',
    length: wet.length
  });
  graph.add(new GPUMask({inputs: [wet, high], output: heavyRainHighStations, operation: 'and'}));
  // heavyRainHighStations marks rainfall >= 100 mm AND elevation >= 500 m.
  return {heavyRainHighStations};
  // #endregion example
}
