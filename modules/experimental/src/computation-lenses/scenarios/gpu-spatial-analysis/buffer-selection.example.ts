// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUBufferSelection,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type BufferSelectionExampleInput = {
  observations: GraphDataView<'float32x2'>;
  libraries: GraphDataView<'float32x2'>;
  spatialContext: GPUSpatialContext;
  distance: GraphDataView<'float32'>;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: BufferSelectionExampleInput
) {
  const selected = createTransientView(
    graph,
    'selected',
    'uint32',
    input.observations.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const distances = createTransientView(
    graph,
    'distances',
    'float32',
    input.observations.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const overflow = createTransientView(
    graph,
    'overflow',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  // #region example
  graph.add(
    new GPUBufferSelection({
      points: {kind: 'points', positions: input.observations},
      features: {kind: 'points', positions: input.libraries},
      spatialContext: input.spatialContext,
      distance: input.distance,
      candidateCapacity: input.observations.length * input.libraries.length,
      output: {outputMask: selected, distances, status: {overflow}}
    })
  );
  // selected colours observations within 300 m; distances is -1 outside.
  // #endregion example
  return {selected, distances, overflow};
}
