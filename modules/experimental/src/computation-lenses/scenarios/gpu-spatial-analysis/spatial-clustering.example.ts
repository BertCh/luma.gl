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
  getGPUSpatialClusteringParameterValues,
  GPUParameterBuffer,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type SpatialClusteringExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Observation extent `[minX, minY, maxX, maxY]` in frame metres. */
  bounds: readonly [number, number, number, number];
};

/** DBSCAN of observations with a 250 m radius and a six point core. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, spatialContext, bounds}: SpatialClusteringExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'clustering-parameters',
    format: 'float32',
    length: GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
    values: getGPUSpatialClusteringParameterValues({bounds, epsilon: 250, minimumPoints: 6})
  });
  const labels = createTransientView(graph, 'labels', 'uint32', observations.length, usage);
  const coreFlags = createTransientView(graph, 'coreFlags', 'uint32', observations.length, usage);
  const clusterCount = createTransientView(graph, 'clusterCount', 'uint32', 1, usage);
  // #region example
  graph.add(
    new GPUSpatialClustering({
      spatialContext,
      positions: observations,
      parameters: parameters.importToGraph(graph),
      gridSize: [64, 64],
      output: {labels, coreFlags, clusterCount}
    })
  );
  // labels: cluster id per observation, 0xffffffff for noise; coreFlags: 1 for core points
  // #endregion example
  return {labels, coreFlags, clusterCount, parameters};
}
