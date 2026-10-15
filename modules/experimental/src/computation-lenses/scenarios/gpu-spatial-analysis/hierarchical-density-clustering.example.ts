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
  GPUHierarchicalDensityClustering,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type HierarchicalDensityClusteringExampleInput = {
  /** Every third iNaturalist observation, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** HDBSCAN of observations: clusters of at least 12 points, core distance from 5 neighbours. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, spatialContext}: HierarchicalDensityClusteringExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const labels = createTransientView(graph, 'labels', 'uint32', observations.length, usage);
  const coreDistances = createTransientView(
    graph,
    'coreDistances',
    'float32',
    observations.length,
    usage
  );
  const clusterCount = createTransientView(graph, 'clusterCount', 'uint32', 1, usage);
  // #region example
  graph.add(
    new GPUHierarchicalDensityClustering({
      spatialContext,
      positions: observations,
      minimumClusterSize: 12,
      minimumSamples: 5,
      output: {labels, coreDistances, status: {count: clusterCount}}
    })
  );
  // labels: cluster id per observation, 0xffffffff for noise; no epsilon to choose
  // #endregion example
  return {labels, coreDistances, clusterCount};
}
