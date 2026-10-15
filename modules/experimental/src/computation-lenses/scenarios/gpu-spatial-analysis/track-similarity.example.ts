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
  GPUTrackSimilarity,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded AIS tracks and track pairs for GPUTrackSimilarity. */
export type TrackSimilarityExampleInput = {
  tracks: GPULineGeometryFields;
  pairA: GraphDataView<'uint32'>;
  pairB: GraphDataView<'uint32'>;
};

/** Runs GPUTrackSimilarity and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: TrackSimilarityExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {tracks, pairA, pairB} = input;
  const pairCount = pairA.length;
  const output = {
    hausdorff: createTransientView(graph, 'hausdorff', 'float32', pairCount, usage),
    frechet: createTransientView(graph, 'frechet', 'float32', pairCount, usage),
    maximumDistance: createTransientView(graph, 'maximumDistance', 'float32', pairCount, usage),
    pairStatus: createTransientView(graph, 'pairStatus', 'uint32', pairCount, usage)
  };
  // #region example
  // UTM 18N metres: all 66 pairs of 12 tracks, each segment split into 4 parts.
  graph.add(
    new GPUTrackSimilarity({
      linesA: {kind: 'lines', ...tracks},
      pairA,
      pairB,
      densify: 0.25,
      maximumFrechetVertexCount: 512,
      output
    })
  );
  const {hausdorff, frechet} = output;
  // #endregion example
  return {...output, hausdorff, frechet};
}
