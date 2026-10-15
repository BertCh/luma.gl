// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUMatVec,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded elevation block, eight equal weights, and the caller-owned row-mean handle. */
export type MatVecExampleInput = {
  elevation: GraphDataView<'float32'>;
  weights: GraphDataView<'float32'>;
  rowMeansBuffer: GraphBufferHandle;
};

/** Adds the real row-mean product; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {elevation, weights, rowMeansBuffer}: MatVecExampleInput
) {
  // #region example
  const rowMeans = graph.createDataView(rowMeansBuffer, {format: 'float32', length: 8});
  graph.add(
    new GPUMatVec({
      id: 'row-means',
      matrix: elevation,
      vector: weights,
      output: rowMeans,
      rows: 8,
      columns: 8
    })
  );
  // rowMeans[r] = the mean elevation of block row r: sum over columns of elevation[r][c] / 8, in m.
  return {rowMeans};
  // #endregion example
}
