// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUBoundsFilter, type GPUSpatialContext} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type BoundsFilterExampleInput = {
  bounds: GraphDataView<'float32x4'>;
  parameters: GraphDataView<'float32'>;
  spatialContext: GPUSpatialContext;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: BoundsFilterExampleInput
) {
  const selected = createTransientView(
    graph,
    'selected',
    'uint32',
    input.bounds.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const ids = createTransientView(
    graph,
    'ids',
    'uint32',
    input.bounds.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const count = createTransientView(graph, 'count', 'uint32', 1, Buffer.STORAGE | Buffer.COPY_SRC);
  const overflow = createTransientView(
    graph,
    'overflow',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  // #region example
  graph.add(
    new GPUBoundsFilter({
      bounds: input.bounds,
      parameters: input.parameters,
      spatialContext: input.spatialContext,
      mode: 'within',
      output: {ids, outputMask: selected, status: {count, overflow}}
    })
  );
  // selected colours observations inside the closed Lincoln Park bounding box.
  // #endregion example
  return {selected, ids, count, overflow};
}
