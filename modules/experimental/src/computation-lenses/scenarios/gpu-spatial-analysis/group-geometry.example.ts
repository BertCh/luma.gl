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
  getGPUGeographicDistributionParameterValues,
  getGPUSpatialClusteringParameterValues,
  GPUGroupGeometry,
  GPUParameterBuffer,
  GPUSpatialClustering,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Label slots of the example; DBSCAN labels at or above it are left out of every group. */
export const GROUP_GEOMETRY_GROUP_COUNT = 64;

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type GroupGeometryExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Observation extent `[minX, minY, maxX, maxY]` in frame metres. */
  bounds: readonly [number, number, number, number];
};

/** Bounds, centres, medoids and spread of the DBSCAN clusters of observations. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, spatialContext, bounds}: GroupGeometryExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const count = GROUP_GEOMETRY_GROUP_COUNT;
  const create = <Format extends 'float32' | 'uint32' | 'float32x2' | 'float32x4'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
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
  const labels = create('labels', 'uint32', observations.length);
  graph.add(
    new GPUSpatialClustering({
      spatialContext,
      positions: observations,
      parameters: clusteringParameters.importToGraph(graph),
      gridSize: [64, 64],
      output: {labels}
    })
  );
  const counts = create('counts', 'uint32', count);
  const bounds4 = create('groupBounds', 'float32x4', count);
  const meanCenters = create('meanCenters', 'float32x2', count);
  const medoidIndices = create('medoidIndices', 'uint32', count);
  const standardDistances = create('standardDistances', 'float32', count);
  // #region example
  // labels: DBSCAN cluster ids from GPUSpatialClustering; noise (0xffffffff) is excluded
  graph.add(
    new GPUGroupGeometry({
      spatialContext,
      positions: observations,
      labels,
      groupCount: 64,
      parameters: geometryParameters.importToGraph(graph),
      output: {counts, bounds: bounds4, meanCenters, medoidIndices, standardDistances}
    })
  );
  // meanCenters[g] and medoidIndices[g] are the centre and the most central observation of group g
  // #endregion example
  return {
    labels,
    counts,
    bounds: bounds4,
    meanCenters,
    medoidIndices,
    standardDistances,
    parameters: [clusteringParameters, geometryParameters]
  };
}
