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
  GPUNearestFeatureJoin,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type NearestFeatureJoinExampleInput = {
  observations: GraphDataView<'float32x2'>;
  libraries: GraphDataView<'float32x2'>;
  spatialContext: GPUSpatialContext;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: NearestFeatureJoinExampleInput
) {
  const neighborIds = createTransientView(
    graph,
    'neighborIds',
    'uint32',
    input.observations.length * 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const neighborDistances = createTransientView(
    graph,
    'neighborDistances',
    'float32',
    input.observations.length * 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const neighborCounts = createTransientView(
    graph,
    'neighborCounts',
    'uint32',
    input.observations.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const joinOverflow = createTransientView(
    graph,
    'joinOverflow',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  // #region example
  graph.add(
    new GPUNearestFeatureJoin({
      queries: {kind: 'points', positions: input.observations},
      features: {kind: 'points', positions: input.libraries},
      spatialContext: input.spatialContext,
      k: 1,
      output: {neighborIds, neighborDistances, neighborCounts, status: {overflow: joinOverflow}}
    })
  );
  // neighborIds and neighborDistances colour observations by their nearest library.
  // #endregion example
  return {neighborIds, neighborDistances, neighborCounts, joinOverflow};
}
