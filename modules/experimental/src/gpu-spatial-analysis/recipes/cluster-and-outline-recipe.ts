// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPURecipeStatusPort,
  GPUStatusScalar,
  GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {GPUGeometryMeasures} from '../geometry-measures/index';
import {GPUGroupConvexHull, GPUGroupGeometry} from '../group-geometry/index';
import {GPUSpatialClustering} from '../spatial-clustering/index';
import {
  assertRecipe,
  createIdentityOffsetsNode,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUClusterAndOutlineRecipe';

/** Properties for {@link addClusterAndOutlineRecipe}. */
export type GPUClusterAndOutlineRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    labels?: GraphDataView<'uint32'>;
    clusterCount?: GraphDataView<'uint32'>;
    hullPositions?: GraphDataView<'float32x2'>;
    hullOffsets?: GraphDataView<'uint32'>;
    areas?: GraphDataView<'float32'>;
    perimeters?: GraphDataView<'float32'>;
    centroids?: GraphDataView<'float32x2'>;
    /** Mirrors `result.status`. */
    status?: {
      clusterOverflow?: GPUStatusScalar;
      hullOverflow?: GPUStatusScalar;
      /** One-row count of hulls rejected by `GPUGeometryMeasures` (stage `hull-measures`). */
      invalidCount?: GPUStatusScalar;
    };
  },
  {
    clusterIds?: GraphDataView<'uint32'>;
    clusterBoundCount?: GraphDataView<'uint32'>;
    counts?: GraphDataView<'uint32'>;
    bounds?: GraphDataView<'float32x4'>;
    meanCenters?: GraphDataView<'float32x2'>;
    hullVertexIndices?: GraphDataView<'uint32'>;
    hullCounts?: GraphDataView<'uint32'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'cluster-outline-recipe'`. */
  id?: string;
  /**
   * Points in the coordinates of `spatialContext`; optional `sourceIds` become the cluster
   * provenance IDs of `GPUSpatialClustering`.
   */
  points: GPUPackedGeometryPort<GPUPointGeometryPort>;
  /**
   * Canonical coordinate and metric contract of `points`, used by the clustering and by hull
   * measurement. Planar contexts cluster with a Euclidean `epsilon`; a longitude/latitude
   * great-circle context clusters with `epsilon` in meters (see `GPUSpatialClustering`, including
   * its antimeridian rule). Other metrics throw at graph construction.
   */
  spatialContext: GPUSpatialContext;
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
  areas: GraphDataView<'float32'>;
  perimeters: GraphDataView<'float32'>;
  centroids: GraphDataView<'float32x2'>;
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row flag: 1 when the cluster summary list was incomplete. Stage `clustering`. */
    clusterOverflow: GPUStatusScalar;
    /** One-row flag: 1 when the shared hull vertex list overflowed. Stage `convex-hull`. */
    hullOverflow: GPUStatusScalar;
    /**
     * One-row count of hull polygons the measurement rejected (geographic hulls enclosing a pole
     * or spanning 180 degrees of longitude); their area and centroid are NaN. Stage
     * `hull-measures`.
     */
    invalidCount: GPUStatusScalar;
  } & GPURecipeStatusPort;
};

/**
 * Cluster and outline recipe: DBSCAN clusters, per-cluster summaries, convex hulls and their
 * area, perimeter and centroid.
 *
 * Chain: `GPUSpatialClustering` -> `GPUGroupGeometry` (counts, bounds, mean centers) and
 * `GPUGroupConvexHull` -> `GPUGeometryMeasures` with each hull as one single-ring polygon feature
 * (the hull's `offsets` is the ring offset list; feature and polygon offsets are the identity). With `minimumPoints = 1` the
 * clusters are PostGIS `ST_ClusterWithin` groups.
 *
 * Status: `status.stages` reports `clustering`, `convex-hull` and `hull-measures` (`status.invalidCount`
 * of rejected hull polygons).
 */
export function addClusterAndOutlineRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUClusterAndOutlineRecipeProps
): GPUClusterAndOutlineRecipeResult {
  const id = props.id ?? 'cluster-outline-recipe';
  const positions = props.points.positions;
  const pointCount = positions.length;
  const groupCount = props.maximumClusterCount;
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  assertRecipe(ID, groupCount >= 1, 'maximumClusterCount must be positive');
  const builder = new RecipeBuilder(graph);
  const labels = getOrCreateView(graph, `${id}-labels`, 'uint32', pointCount, outputs.labels);
  const clusterCount = getOrCreateView(
    graph,
    `${id}-cluster-count`,
    'uint32',
    1,
    outputs.clusterCount
  );
  const clusterIds = getOrCreateView(
    graph,
    `${id}-cluster-ids`,
    'uint32',
    groupCount,
    scratch.clusterIds
  );
  const clusterBoundCount = getOrCreateView(
    graph,
    `${id}-cluster-bound-count`,
    'uint32',
    1,
    scratch.clusterBoundCount
  );
  const clusterOverflow = getOrCreateView(
    graph,
    `${id}-cluster-overflow`,
    'uint32',
    1,
    outputs.status?.clusterOverflow
  );
  builder.add(
    new GPUSpatialClustering({
      id: `${id}-clustering`,
      spatialContext: props.spatialContext,
      positions,
      sourceIds: props.points.sourceIds,
      parameters: props.clusteringParameters,
      gridSize: props.gridSize,
      output: {
        labels,
        clusterCount,
        clusters: {
          ids: clusterIds,
          status: {count: clusterBoundCount, overflow: clusterOverflow}
        }
      }
    })
  );

  const counts = getOrCreateView(graph, `${id}-counts`, 'uint32', groupCount, scratch.counts);
  const bounds = getOrCreateView(graph, `${id}-bounds`, 'float32x4', groupCount, scratch.bounds);
  const meanCenters = getOrCreateView(
    graph,
    `${id}-mean-centers`,
    'float32x2',
    groupCount,
    scratch.meanCenters
  );
  builder.add(
    new GPUGroupGeometry({
      id: `${id}-group-geometry`,
      positions,
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
    scratch.hullVertexIndices
  );
  const hullPositions = getOrCreateView(
    graph,
    `${id}-hull-positions`,
    'float32x2',
    props.hullCapacity,
    outputs.hullPositions
  );
  const hullOffsets = getOrCreateView(
    graph,
    `${id}-hull-offsets`,
    'uint32',
    groupCount + 1,
    outputs.hullOffsets
  );
  const hullCounts = getOrCreateView(
    graph,
    `${id}-hull-counts`,
    'uint32',
    groupCount,
    scratch.hullCounts
  );
  const hullOverflow = getOrCreateView(
    graph,
    `${id}-hull-overflow`,
    'uint32',
    1,
    outputs.status?.hullOverflow
  );
  builder.add(
    new GPUGroupConvexHull({
      id: `${id}-convex-hull`,
      positions,
      labels,
      groupCount,
      maximumVerticesPerGroup: props.maximumVerticesPerHull,
      vertexCapacity: props.hullCapacity,
      output: {
        vertexIndices: hullVertexIndices,
        vertexPositions: hullPositions,
        offsets: hullOffsets,
        counts: hullCounts,
        status: {overflow: hullOverflow}
      }
    })
  );

  const areas = getOrCreateView(graph, `${id}-areas`, 'float32', groupCount, outputs.areas);
  const perimeters = getOrCreateView(
    graph,
    `${id}-perimeters`,
    'float32',
    groupCount,
    outputs.perimeters
  );
  const centroids = getOrCreateView(
    graph,
    `${id}-centroids`,
    'float32x2',
    groupCount,
    outputs.centroids
  );
  const invalidCount = getOrCreateView(
    graph,
    `${id}-invalid-count`,
    'uint32',
    1,
    outputs.status?.invalidCount
  );
  // Each hull is one feature with one polygon of one ring: identity feature and polygon offsets.
  const identityOffsets = createIdentityOffsetsNode(graph, `${id}-hull-identity`, groupCount);
  builder.add(
    new GPUGeometryMeasures({
      spatialContext: props.spatialContext,
      id: `${id}-hull-measures`,
      geometryType: 'polygons',
      positions: hullPositions,
      featureOffsets: identityOffsets,
      polygonOffsets: identityOffsets,
      ringOffsets: hullOffsets,
      output: {areas, lengths: perimeters, centroids, status: {invalidCount}}
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
    areas,
    perimeters,
    centroids,
    outputs: {labels, clusterCount, hullPositions, hullOffsets, areas, perimeters, centroids},
    intermediates: {counts, bounds, meanCenters, hullVertexIndices, hullCounts},
    status: {
      clusterOverflow,
      hullOverflow,
      invalidCount,
      stages: [
        {
          stage: 'clustering',
          status: {count: clusterBoundCount, requiredCount: clusterCount, overflow: clusterOverflow}
        },
        {stage: 'convex-hull', status: {overflow: hullOverflow}},
        {stage: 'hull-measures', status: {invalidCount}}
      ]
    }
  };
}
