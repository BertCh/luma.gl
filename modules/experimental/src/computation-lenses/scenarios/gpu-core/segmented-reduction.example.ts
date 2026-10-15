// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUSegmentedReduction,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station rainfall, state starts and the caller-owned destination handle. */
export type SegmentedReductionExampleInput = {
  rainfall: GraphDataView<'float32'>;
  stateOffsets: GraphDataView<'uint32'>;
  stateTotalsBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {rainfall, stateOffsets, stateTotalsBuffer}: SegmentedReductionExampleInput
) {
  // #region example
  const stateTotals = graph.createDataView(stateTotalsBuffer, {
    format: 'float32',
    length: stateOffsets.length - 1
  });
  graph.add(
    new GPUSegmentedReduction({
      id: 'state-rainfall-total',
      input: rainfall,
      segmentOffsets: stateOffsets,
      output: stateTotals,
      operation: 'sum'
    })
  );
  // stateTotals[s] sums rainfall over stations stateOffsets[s] up to stateOffsets[s + 1].
  return {stateTotals};
  // #endregion example
}
