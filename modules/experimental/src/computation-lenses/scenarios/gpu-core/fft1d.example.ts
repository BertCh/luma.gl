// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUFFT1D,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded complex elevation profile and the caller-owned spectrum handle. */
export type FFT1DExampleInput = {
  elevationProfile: GraphDataView<'float32x2'>;
  spectrumBuffer: GraphBufferHandle;
};

/** Adds the real 64-point transform; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {elevationProfile, spectrumBuffer}: FFT1DExampleInput
) {
  // #region example
  const spectrum = graph.createDataView(spectrumBuffer, {format: 'float32x2', length: 64});
  graph.add(
    new GPUFFT1D({
      id: 'profile-spectrum',
      input: elevationProfile,
      output: spectrum,
      length: 64,
      direction: 'forward'
    })
  );
  // spectrum[k] = sum over j of elevationProfile[j] * exp(-2πi·j·k/64), stored as (re, im) in m.
  return {spectrum};
  // #endregion example
}
