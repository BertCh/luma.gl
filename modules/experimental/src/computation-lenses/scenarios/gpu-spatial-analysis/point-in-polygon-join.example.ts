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
  GPUPointInPolygonJoin,
  type GPUSpatialContext,
  type GPUPointInPolygonJoinProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded inputs and caller-owned outputs for this recorded example. */
export type PointInPolygonJoinExampleInput = {
  observations: GraphDataView<'float32x2'>;
  tracts: GPUPointInPolygonJoinProps['polygons'];
  spatialContext: GPUSpatialContext;
};

/** Adds the public contributor and returns the views captured by the figure. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: PointInPolygonJoinExampleInput
) {
  const tractIds = createTransientView(
    graph,
    'tractIds',
    'uint32',
    input.observations.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const featureCounts = createTransientView(
    graph,
    'featureCounts',
    'uint32',
    input.tracts.featureOffsets.length - 1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const overflow = createTransientView(
    graph,
    'overflow',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const uncertainCount = createTransientView(
    graph,
    'uncertainCount',
    'uint32',
    1,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  // #region example
  graph.add(
    new GPUPointInPolygonJoin({
      points: {kind: 'points', positions: input.observations},
      polygons: input.tracts,
      spatialContext: input.spatialContext,
      boundaryRule: 'half-open',
      edgeSlabs: false,
      candidateCapacity: input.observations.length * featureCounts.length,
      output: {pointFeatureIds: tractIds, featureCounts, status: {overflow, uncertainCount}}
    })
  );
  // tractIds colours observations; featureCounts counts assignments per tract.
  // #endregion example
  return {tractIds, featureCounts, overflow, uncertainCount};
}
