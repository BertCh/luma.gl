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
  GPUNearestFeatureWeights,
  type GPUSpatialContext,
  GPUNearestFeatureJoin
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type NearestFeatureWeightsExampleInput = {
  observations: GraphDataView<'float32x2'>;
  libraries: GraphDataView<'float32x2'>;
  spatialContext: GPUSpatialContext;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: NearestFeatureWeightsExampleInput
) {
  const neighborIds = createTransientView(
    graph,
    'neighborIds',
    'uint32',
    input.observations.length * 3,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const neighborDistances = createTransientView(
    graph,
    'neighborDistances',
    'float32',
    input.observations.length * 3,
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
  const offsets = createTransientView(
    graph,
    'offsets',
    'uint32',
    input.observations.length + 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const neighbors = createTransientView(
    graph,
    'neighbors',
    'uint32',
    input.observations.length * 3,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const weights = createTransientView(
    graph,
    'weights',
    'float32',
    input.observations.length * 3,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const distances = createTransientView(
    graph,
    'distances',
    'float32',
    input.observations.length * 3,
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
    new GPUNearestFeatureJoin({
      queries: {kind: 'points', positions: input.observations},
      features: {kind: 'points', positions: input.libraries},
      spatialContext: input.spatialContext,
      k: 3,
      output: {neighborIds, neighborDistances, neighborCounts, status: {overflow: joinOverflow}}
    })
  );
  graph.add(
    new GPUNearestFeatureWeights({
      neighborIds,
      neighborCounts,
      neighborDistances,
      slotCapacity: 3,
      k: 3,
      weightType: 'inverse-distance',
      power: 1,
      distanceFloor: 1,
      spatialContext: input.spatialContext,
      output: {offsets, neighbors, weights, distances, status: {overflow}}
    })
  );
  // offsets, neighbors, weights and distances describe three-library cross weights.
  // #endregion example
  return {
    neighborIds,
    neighborDistances,
    neighborCounts,
    joinOverflow,
    offsets,
    neighbors,
    weights,
    distances,
    overflow
  };
}
