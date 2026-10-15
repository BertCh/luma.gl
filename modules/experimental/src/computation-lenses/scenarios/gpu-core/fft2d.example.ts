// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUFFT2D,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded complex elevation tile and the caller-owned spectrum handle. */
export type FFT2DExampleInput = {
  elevationTile: GraphDataView<'float32x2'>;
  spectrumBuffer: GraphBufferHandle;
};

/** Adds the real 64 × 64 transform; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {elevationTile, spectrumBuffer}: FFT2DExampleInput
) {
  // #region example
  const spectrum = graph.createDataView(spectrumBuffer, {format: 'float32x2', length: 64 * 64});
  graph.add(
    new GPUFFT2D({
      id: 'tile-spectrum',
      input: elevationTile,
      output: spectrum,
      width: 64,
      height: 64,
      direction: 'forward'
    })
  );
  // spectrum[v][u] = sum over r, c of elevationTile[r][c] * exp(-2πi·(u·c + v·r)/64), in m;
  // (u, v) = (0, 0) is the sum of all elevations and sits in the top-left corner (no fftshift).
  return {spectrum};
  // #endregion example
}
