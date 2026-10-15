// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUSegregationLayout, GPUSegregation} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-tracts.json`. */
export type SegregationExampleInput = {
  /** Residents per tract and group, `tracts * 5` values: White, Black, Hispanic, Asian, Other. */
  groupCounts: GraphDataView<'float32'>;
};

/** Aspatial segregation indices of five population groups across census tracts. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {groupCounts}: SegregationExampleInput
) {
  const groupCount = 5;
  const tractCount = groupCounts.length / groupCount;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const indices = createTransientView(
    graph,
    'indices',
    'float32',
    getGPUSegregationLayout(groupCount).stride,
    usage
  );
  const localTheil = createTransientView(graph, 'localTheil', 'float32', tractCount, usage);
  const localEntropy = createTransientView(graph, 'localEntropy', 'float32', tractCount, usage);
  // #region example
  graph.add(
    new GPUSegregation({
      groupCount,
      groupCounts,
      scales: [null], // aspatial: each tract is its own environment
      output: {indices, localTheil, localEntropy}
    })
  );
  // indices: entropy H, dissimilarity D and per-group indices; localTheil sums to H
  // #endregion example
  return {indices, localTheil, localEntropy, groupCount};
}
