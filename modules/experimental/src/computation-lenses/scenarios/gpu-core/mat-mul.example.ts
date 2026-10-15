// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUMatMul,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded elevation block, forward-difference matrix, and the caller-owned product handle. */
export type MatMulExampleInput = {
  elevation: GraphDataView<'float32'>;
  forwardDifference: GraphDataView<'float32'>;
  eastWestStepsBuffer: GraphBufferHandle;
};

/** Adds the real tiled product; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {elevation, forwardDifference, eastWestStepsBuffer}: MatMulExampleInput
) {
  // #region example
  const eastWestSteps = graph.createDataView(eastWestStepsBuffer, {
    format: 'float32',
    length: 64
  });
  graph.add(
    new GPUMatMul({
      id: 'east-west-steps',
      left: elevation,
      right: forwardDifference,
      output: eastWestSteps,
      m: 8,
      k: 8,
      n: 8
    })
  );
  // eastWestSteps[r][c] = elevation[r][c + 1] - elevation[r][c] in m per cell; the last column is 0.
  return {eastWestSteps};
  // #endregion example
}
