// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUFlagOffsets,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station inputs and caller-owned destination handles. */
export type FlagOffsetsExampleInput = {
  wet: GraphDataView<'uint32'>;
  wetOffsetsBuffer: GraphBufferHandle;
  wetCountBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {wet, wetOffsetsBuffer, wetCountBuffer}: FlagOffsetsExampleInput
) {
  // #region example
  const wetOffsets = graph.createDataView(wetOffsetsBuffer, {format: 'uint32', length: wet.length});
  const wetCount = graph.createDataView(wetCountBuffer, {format: 'uint32', length: 1});
  graph.add(new GPUFlagOffsets({flags: wet, offsets: wetOffsets, count: wetCount}));
  // wetOffsets assigns dense destinations; wetCount counts stations with at least 100 mm rainfall.
  return {wetOffsets, wetCount};
  // #endregion example
}
