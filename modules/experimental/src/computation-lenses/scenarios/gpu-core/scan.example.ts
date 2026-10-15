// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUScan,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station inputs and caller-owned destination handles. */
export type ScanExampleInput = {
  rainfall: GraphDataView<'uint32'>;
  rainTotalsBuffer: GraphBufferHandle;
  segmentFlags?: GraphDataView<'uint32'>;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {rainfall, rainTotalsBuffer, segmentFlags}: ScanExampleInput
) {
  // #region example
  const rainTotals = graph.createDataView(rainTotalsBuffer, {
    format: 'uint32',
    length: rainfall.length
  });
  graph.add(
    new GPUScan({
      id: 'rainfall-scan',
      input: rainfall,
      output: rainTotals,
      mode: 'inclusive',
      segmentFlags
    })
  );
  // rainTotals[k] sums integer rainfall through station k, restarting at state flags when supplied.
  return {rainTotals};
  // #endregion example
}
