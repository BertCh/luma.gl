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
  GPUGroupConvexHull,
  GPUParameterBuffer,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Label slots of the example; DBSCAN labels at or above it are left out of every hull. */
export const GROUP_CONVEX_HULL_GROUP_COUNT = 64;
/** Capacity of the hull vertex list shared by all groups. */
export const GROUP_CONVEX_HULL_VERTEX_CAPACITY = 1024;

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type GroupConvexHullExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Observation extent `[minX, minY, maxX, maxY]` in frame metres. */
  bounds: readonly [number, number, number, number];
};

/** Convex outlines of the DBSCAN clusters of observations. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, spatialContext, bounds}: GroupConvexHullExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const groupCount = GROUP_CONVEX_HULL_GROUP_COUNT;
  const vertexCapacity = GROUP_CONVEX_HULL_VERTEX_CAPACITY;
  const create = <Format extends 'float32' | 'uint32' | 'float32x2'>(
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
  const vertexIndices = create('vertexIndices', 'uint32', vertexCapacity);
  const vertexPositions = create('vertexPositions', 'float32x2', vertexCapacity);
  const offsets = create('offsets', 'uint32', groupCount + 1);
  const counts = create('counts', 'uint32', groupCount);
  const sizes = create('sizes', 'uint32', groupCount);
  const overflow = create('overflow', 'uint32', 1);
  // #region example
  // labels: DBSCAN cluster ids from GPUSpatialClustering; noise (0xffffffff) is excluded
  graph.add(
    new GPUGroupConvexHull({
      spatialContext,
      positions: observations,
      labels,
      groupCount: 64,
      maximumVerticesPerGroup: 64,
      vertexCapacity: 1024,
      output: {vertexIndices, vertexPositions, offsets, counts, sizes, status: {overflow}}
    })
  );
  // hull g is vertexPositions[offsets[g] .. offsets[g] + counts[g]), counter-clockwise
  // #endregion example
  return {
    labels,
    vertexIndices,
    vertexPositions,
    offsets,
    counts,
    sizes,
    overflow,
    parameters: clusteringParameters
  };
}
