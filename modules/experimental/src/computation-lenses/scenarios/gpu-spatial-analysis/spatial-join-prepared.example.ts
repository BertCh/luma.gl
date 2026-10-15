// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUHistogram, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUSpatialJoinPrepared,
  type GPUSpatialContext,
  type GPUSpatialPredicateJoinProps,
  type GPUPointInPolygonJoinProps,
  GPUSpatialPredicateJoin,
  GPUSpatialJoinCandidates
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type SpatialJoinPreparedExampleInput = {
  parks: GPUPointInPolygonJoinProps['polygons'];
  tracts: GPUPointInPolygonJoinProps['polygons'];
  pairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
  spatialContext: GPUSpatialContext;
  candidatePairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: SpatialJoinPreparedExampleInput
) {
  const tractMatches = createTransientView(
    graph,
    'tractMatches',
    'uint32',
    input.tracts.featureOffsets.length - 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const tractCounts = new GPUHistogram({
    input: input.pairs.rightIds,
    output: tractMatches,
    domain: [0, tractMatches.length - 1]
  });
  // #region example
  const prepared = new GPUSpatialJoinPrepared({geometry: input.tracts, spatialSort: false});
  graph.add(prepared);
  graph.add(
    new GPUSpatialJoinCandidates({
      left: input.parks,
      right: input.tracts,
      prepared,
      spatialContext: input.spatialContext,
      output: input.candidatePairs
    })
  );
  graph.add(
    new GPUSpatialPredicateJoin({
      left: input.parks,
      right: input.tracts,
      spatialContext: input.spatialContext,
      predicate: 'intersects',
      engine: 'relate',
      candidateCapacity: input.pairs.leftIds.length,
      prepared,
      output: {pairs: input.pairs}
    })
  );
  graph.add(tractCounts);
  // tractMatches colours tracts; pairs identify parks and tracts.
  // #endregion example
  return {tractMatches, prepared};
}
