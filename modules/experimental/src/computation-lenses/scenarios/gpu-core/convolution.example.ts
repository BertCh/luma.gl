// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUConvolution,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded elevation tile, Laplacian stencil, and the caller-owned curvature handle. */
export type ConvolutionExampleInput = {
  elevation: GraphDataView<'float32'>;
  laplacian: GraphDataView<'float32'>;
  curvatureBuffer: GraphBufferHandle;
};

/** Adds the real 3 × 3 convolution; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {elevation, laplacian, curvatureBuffer}: ConvolutionExampleInput
) {
  // #region example
  const curvature = graph.createDataView(curvatureBuffer, {format: 'float32', length: 64 * 64});
  graph.add(
    new GPUConvolution({
      id: 'terrain-curvature',
      input: elevation,
      kernel: laplacian,
      output: curvature,
      width: 64,
      height: 64,
      kernelWidth: 3,
      kernelHeight: 3,
      strategy: 'direct',
      boundary: 'zero'
    })
  );
  // curvature[r][c] = sum of the four edge neighbours of elevation[r][c] - 4 * elevation[r][c], in m;
  // neighbours outside the tile count as 0 m, so the outer ring of cells is not valid curvature.
  return {curvature};
  // #endregion example
}
