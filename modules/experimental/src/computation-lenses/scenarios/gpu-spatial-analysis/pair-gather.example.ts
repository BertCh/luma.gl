// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUHistogram,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUPairGather,
  type GPUSpatialContext,
  type GPUSpatialPredicateJoinProps,
  type GPUPointInPolygonJoinProps,
  GPUSpatialPredicateJoin
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type PairGatherExampleInput = {
  parks: GPUPointInPolygonJoinProps['polygons'];
  tracts: GPUPointInPolygonJoinProps['polygons'];
  pairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
  spatialContext: GPUSpatialContext;
  parkAreas: GraphDataView<'float32'>;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(graph: GPUCommandGraph<undefined>, input: PairGatherExampleInput) {
  const tractMatches = createTransientView(
    graph,
    'tractMatches',
    'uint32',
    input.tracts.featureOffsets.length - 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const parkRows = createTransientView(
    graph,
    'parkRows',
    'uint32',
    input.pairs.leftIds.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const tractRows = createTransientView(
    graph,
    'tractRows',
    'uint32',
    input.pairs.leftIds.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const joinedAreas = createTransientView(
    graph,
    'joinedAreas',
    'float32',
    input.pairs.leftIds.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const rowCount = createTransientView(
    graph,
    'rowCount',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const overflow = createTransientView(
    graph,
    'overflow',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const tractCounts = new GPUHistogram({
    input: input.pairs.rightIds,
    output: tractMatches,
    domain: [0, tractMatches.length - 1]
  });
  // #region example
  graph.add(
    new GPUSpatialPredicateJoin({
      left: input.parks,
      right: input.tracts,
      spatialContext: input.spatialContext,
      predicate: 'intersects',
      engine: 'relate',
      candidateCapacity: input.pairs.leftIds.length,
      output: {pairs: input.pairs}
    })
  );
  graph.add(tractCounts);
  graph.add(
    new GPUPairGather({
      pairs: input.pairs,
      leftColumns: [{source: input.parkAreas, output: joinedAreas}],
      output: {leftRows: parkRows, rightRows: tractRows, rowCount, status: {overflow}}
    })
  );
  // tractMatches colours tracts; pairs identify parks and tracts; joinedAreas carries source park area per pair.
  // #endregion example
  return {tractMatches, parkRows, tractRows, joinedAreas, rowCount, overflow};
}
