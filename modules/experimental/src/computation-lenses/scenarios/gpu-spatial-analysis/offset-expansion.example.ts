// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUOffsetExpansion} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type OffsetExpansionExampleInput = {
  offsets: GraphDataView<'uint32'>;
  observationCount: number;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: OffsetExpansionExampleInput
) {
  const owners = createTransientView(
    graph,
    'owners',
    'uint32',
    input.observationCount,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const localIndex = createTransientView(
    graph,
    'localIndex',
    'uint32',
    input.observationCount,
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
    new GPUOffsetExpansion({
      offsets: input.offsets,
      output: {owners, localIndex, status: {count, overflow}}
    })
  );
  // owners colours tract-grouped observations; localIndex is each observation's index within its tract.
  // #endregion example
  return {owners, localIndex, count, overflow};
}
