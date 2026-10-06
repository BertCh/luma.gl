// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUGeometryMeasures} from '../geometry-measures/index';
import type {GPUGeometryCoordinateSystem} from '../geometry-measures/index';
import {GPUGroupConvexHull, GPUGroupGeometry} from '../group-geometry/index';
import {GPUSpatialClustering} from '../spatial-clustering/index';
import {assertRecipe, getOrCreateView, RecipeBuilder, type GPURecipeResult} from './recipe-utils';

const ID = 'GPUClusterAndOutlineRecipe';

/** Properties for {@link addClusterAndOutlineRecipe}. */
export type GPUClusterAndOutlineRecipeProps = {
  /** Prefix for every node and transient ID. Defaults to `'cluster-outline-recipe'`. */
  id?: string;
  /** Planar points (or longitude/latitude degrees with a geographic `coordinateSystem`). */
  positions: GraphDataView<'float32x2'>;
  /** `getGPUSpatialClusteringParameterValues` view: bounds, epsilon, minimum points. */
  clusteringParameters: GraphDataView<'float32'>;
  /** Maximum neighbor-search lattice of the clustering. */
  gridSize: readonly [number, number];
  /** `getGPUGeographicDistributionParameterValues` view for the per-cluster summaries. */
  geometryParameters: GraphDataView<'float32'>;
  /**
   * Compile-time cluster capacity: clusters with a label at or above it are dropped from the
   * summaries and hulls (their labels are still written).
   */
  maximumClusterCount: number;
  /** Largest hull emitted per cluster (at least 3); larger hulls are dropped and flagged. */
  maximumVerticesPerHull: number;
  /** Capacity of the shared hull vertex list. */
  hullCapacity: number;
  /** Area unit of the measures. Defaults to `'planar'`. */
  coordinateSystem?: GPUGeometryCoordinateSystem;
  /** Caller-owned cluster label per point (`0xffffffff` is noise). */
  labels?: GraphDataView<'uint32'>;
  /** Caller-owned one-row cluster count (unclamped). */
  clusterCount?: GraphDataView<'uint32'>;
  /** Caller-owned member count per cluster (`maximumClusterCount` rows). */
  counts?: GraphDataView<'uint32'>;
  /** Caller-owned `[minX, minY, maxX, maxY]` per cluster. */
  bounds?: GraphDataView<'float32x4'>;
  /** Caller-owned mean center per cluster. */
  meanCenters?: GraphDataView<'float32x2'>;
  /** Caller-owned hull vertex row per slot of the shared list. */
  hullVertexIndices?: GraphDataView<'uint32'>;
  /** Caller-owned hull vertex position per slot of the shared list. */
  hullPositions?: GraphDataView<'float32x2'>;
  /** Caller-owned hull start per cluster (`maximumClusterCount + 1` rows). */
  hullOffsets?: GraphDataView<'uint32'>;
  /** Caller-owned hull vertex count per cluster. */
  hullCounts?: GraphDataView<'uint32'>;
  /** Caller-owned hull overflow bits (one row). */
  hullOverflow?: GraphDataView<'uint32'>;
  /** Caller-owned hull area per cluster (0 for clusters with fewer than 3 hull vertices). */
  areas?: GraphDataView<'float32'>;
  /** Caller-owned hull perimeter per cluster (a 2-vertex hull counts both ways). */
  perimeters?: GraphDataView<'float32'>;
  /** Caller-owned hull centroid per cluster. */
  centroids?: GraphDataView<'float32x2'>;
};

/** Named outputs of {@link addClusterAndOutlineRecipe}. */
export type GPUClusterAndOutlineRecipeResult = GPURecipeResult & {
  /** Cluster capacity: the row count of every per-cluster output. */
  maximumClusterCount: number;
  labels: GraphDataView<'uint32'>;
  clusterCount: GraphDataView<'uint32'>;
  counts: GraphDataView<'uint32'>;
  bounds: GraphDataView<'float32x4'>;
  meanCenters: GraphDataView<'float32x2'>;
  hullVertexIndices: GraphDataView<'uint32'>;
  hullPositions: GraphDataView<'float32x2'>;
  hullOffsets: GraphDataView<'uint32'>;
  hullCounts: GraphDataView<'uint32'>;
  hullOverflow: GraphDataView<'uint32'>;
  areas: GraphDataView<'float32'>;
  perimeters: GraphDataView<'float32'>;
  centroids: GraphDataView<'float32x2'>;
};

/**
 * Cluster and outline recipe: DBSCAN clusters, per-cluster summaries, convex hulls and their
 * area, perimeter and centroid.
 *
 * Chain: `GPUSpatialClustering` -> `GPUGroupGeometry` (counts, bounds, mean centers) and
 * `GPUGroupConvexHull` -> `GPUGeometryMeasures` with each hull as one polygon ring (the hull's
 * `offsets` is the ring offset list, so no adapter is needed). With `minimumPoints = 1` the
 * clusters are PostGIS `ST_ClusterWithin` groups.
 */
export function addClusterAndOutlineRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUClusterAndOutlineRecipeProps
): GPUClusterAndOutlineRecipeResult {
  const id = props.id ?? 'cluster-outline-recipe';
  const pointCount = props.positions.length;
  const groupCount = props.maximumClusterCount;
  assertRecipe(ID, groupCount >= 1, 'maximumClusterCount must be positive');
  const builder = new RecipeBuilder(graph);
  const labels = getOrCreateView(graph, `${id}-labels`, 'uint32', pointCount, props.labels);
  const clusterCount = getOrCreateView(
    graph,
    `${id}-cluster-count`,
    'uint32',
    1,
    props.clusterCount
  );
  const clusterIds = getOrCreateView(graph, `${id}-cluster-ids`, 'uint32', groupCount);
  const clusterBoundCount = getOrCreateView(graph, `${id}-cluster-bound-count`, 'uint32', 1);
  const clusterOverflow = getOrCreateView(graph, `${id}-cluster-overflow`, 'uint32', 1);
  builder.add(
    new GPUSpatialClustering({
      id: `${id}-clustering`,
      positions: props.positions,
      parameters: props.clusteringParameters,
      gridSize: props.gridSize,
      labels,
      clusterCount,
      clusters: {ids: clusterIds, count: clusterBoundCount, overflow: clusterOverflow}
    })
  );

  const counts = getOrCreateView(graph, `${id}-counts`, 'uint32', groupCount, props.counts);
  const bounds = getOrCreateView(graph, `${id}-bounds`, 'float32x4', groupCount, props.bounds);
  const meanCenters = getOrCreateView(
    graph,
    `${id}-mean-centers`,
    'float32x2',
    groupCount,
    props.meanCenters
  );
  builder.add(
    new GPUGroupGeometry({
      id: `${id}-group-geometry`,
      positions: props.positions,
      labels,
      groupCount,
      parameters: props.geometryParameters,
      output: {counts, bounds, meanCenters}
    })
  );

  const hullVertexIndices = getOrCreateView(
    graph,
    `${id}-hull-vertex-indices`,
    'uint32',
    props.hullCapacity,
    props.hullVertexIndices
  );
  const hullPositions = getOrCreateView(
    graph,
    `${id}-hull-positions`,
    'float32x2',
    props.hullCapacity,
    props.hullPositions
  );
  const hullOffsets = getOrCreateView(
    graph,
    `${id}-hull-offsets`,
    'uint32',
    groupCount + 1,
    props.hullOffsets
  );
  const hullCounts = getOrCreateView(
    graph,
    `${id}-hull-counts`,
    'uint32',
    groupCount,
    props.hullCounts
  );
  const hullOverflow = getOrCreateView(
    graph,
    `${id}-hull-overflow`,
    'uint32',
    1,
    props.hullOverflow
  );
  builder.add(
    new GPUGroupConvexHull({
      id: `${id}-convex-hull`,
      positions: props.positions,
      labels,
      groupCount,
      maximumVerticesPerGroup: props.maximumVerticesPerHull,
      totalCapacity: props.hullCapacity,
      output: {
        vertexIndices: hullVertexIndices,
        vertexPositions: hullPositions,
        offsets: hullOffsets,
        counts: hullCounts,
        overflow: hullOverflow
      }
    })
  );

  const areas = getOrCreateView(graph, `${id}-areas`, 'float32', groupCount, props.areas);
  const perimeters = getOrCreateView(
    graph,
    `${id}-perimeters`,
    'float32',
    groupCount,
    props.perimeters
  );
  const centroids = getOrCreateView(
    graph,
    `${id}-centroids`,
    'float32x2',
    groupCount,
    props.centroids
  );
  builder.add(
    new GPUGeometryMeasures({
      id: `${id}-hull-measures`,
      positions: hullPositions,
      geometryType: 'polygons',
      ringOffsets: hullOffsets,
      coordinateSystem: props.coordinateSystem,
      output: {areas, lengths: perimeters, centroids}
    })
  );

  return {
    contributors: builder.contributors,
    maximumClusterCount: groupCount,
    labels,
    clusterCount,
    counts,
    bounds,
    meanCenters,
    hullVertexIndices,
    hullPositions,
    hullOffsets,
    hullCounts,
    hullOverflow,
    areas,
    perimeters,
    centroids
  };
}
