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
  addClusterAndOutlineRecipe,
  getGPUGeographicDistributionParameterValues,
  getGPUSpatialClusteringParameterValues,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type ClusterAndOutlineRecipeExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Observation extent `[minX, minY, maxX, maxY]` in frame metres. */
  bounds: readonly [number, number, number, number];
};

/** DBSCAN clusters of observations (250 m, six points) with a convex hull per cluster. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, spatialContext, bounds}: ClusterAndOutlineRecipeExampleInput
) {
  const maximumClusterCount = 128;
  const hullCapacity = 8192;
  const clusteringParameters = new GPUParameterBuffer(graph.device, {
    id: 'clustering-parameters',
    format: 'float32',
    length: GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
    values: getGPUSpatialClusteringParameterValues({bounds, epsilon: 250, minimumPoints: 6})
  });
  const geometryParameters = new GPUParameterBuffer(graph.device, {
    id: 'geometry-parameters',
    format: 'float32',
    length: GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
    values: getGPUGeographicDistributionParameterValues()
  });
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const labels = output('labels', 'uint32', observations.length);
  const hullPositions = output('hullPositions', 'float32x2', hullCapacity);
  const hullOffsets = output('hullOffsets', 'uint32', maximumClusterCount + 1);
  const areas = output('areas', 'float32', maximumClusterCount);
  const perimeters = output('perimeters', 'float32', maximumClusterCount);
  const clusterCount = output('clusterCount', 'uint32', 1);
  // #region example
  const clusters = addClusterAndOutlineRecipe(graph, {
    spatialContext,
    points: {kind: 'points', positions: observations},
    clusteringParameters: clusteringParameters.importToGraph(graph),
    gridSize: [64, 64],
    geometryParameters: geometryParameters.importToGraph(graph),
    maximumClusterCount,
    maximumVerticesPerHull: 128,
    hullCapacity,
    outputs: {labels, clusterCount, hullPositions, hullOffsets, areas, perimeters}
  });
  // labels colour the observations; hullPositions with hullOffsets outline each cluster
  // #endregion example
  return {
    clusters,
    labels,
    clusterCount,
    hullPositions,
    hullOffsets,
    areas,
    perimeters,
    parameters: [clusteringParameters, geometryParameters]
  };
}
