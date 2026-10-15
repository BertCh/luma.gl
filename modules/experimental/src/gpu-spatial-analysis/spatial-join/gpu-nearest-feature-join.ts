// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPULonLatPolygonStatusPort} from '../../analysis-core/contracts/index';
import {
  createLonLatPolygonCheckNode,
  hasLonLatPolygonStatus
} from '../../analysis-core/lon-lat-ring-wgsl';
import {createFillNode} from '../../utils/wgsl-kernel-nodes';

import {
  createTransientView,
  type GPUBVH,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUPairwisePointSegmentDistance} from '../../geospatial/gpu-pairwise-point-segment-distance';
import {
  createWGSLActiveCountDispatch,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {
  GPUCompactOutput,
  GPUFloat32Positions,
  GPUUint32Rows
} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getGraphViewChunks,
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validatePowerOfTwo,
  validateCapacity,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createSpatialJoinClearNode,
  createSpatialJoinFinalizeNode,
  getChunkNodeId,
  getDefaultSpatialSort,
  type SpatialSortCurve,
  getNextPowerOfTwo,
  getSpatialJoinAssignNodes,
  getSpatialJoinCollectNodes,
  getDefaultQuerySort,
  getSpatialJoinProbeNodes,
  validateDisjointOutputs,
  validateMatchingChunks
} from './spatial-join-passes';
import {NEAREST_NODE_WORDS} from './nearest-wgsl';
import type {GPUNearestFeatureSource, GPUSpatialJoinOnAttribute} from './spatial-join-types';
import {validateStatusScalars} from './spatial-join-geometry';
import type {
  GPULineGeometryPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';
import type {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';
import {
  getNearestBVHNodes,
  getNearestGeometryCount,
  getNearestGeometryViews,
  getNearestNeighborNodes
} from './nearest-nodes';
import {
  type GPUNearestFeatureGeometry,
  GPUNearestQueryGeometry,
  GPUNearestTieMode
} from './nearest-types';

import {
  GPU_LINEAR_EDGE_MODELS,
  getGPUGreatCircleRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {assert, type Device} from '@luma.gl/core';

/** Planar or great-circle contexts suitable for distance measurements in their length unit. */
const NEAREST_FEATURE_JOIN_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  edgeModels: GPU_LINEAR_EDGE_MODELS,
  measure: 'distance'
} as const);

const OPERATION = 'GPUNearestFeatureJoin';

/** Caller-owned outputs of {@link GPUNearestFeatureJoin}. */
export type GPUNearestFeatureJoinOutput = {
  /**
   * Nearest-feature mode: per-point nearest feature ID or row, or `GPU_SPATIAL_JOIN_NO_FEATURE`.
   * Chunked like the query points.
   */
  nearestFeatureIds?: GPUUint32Rows;
  /**
   * Neighbors mode: feature ID (or row, without `features.sourceIds`) per `(query, slot)` at index
   * `query * neighborCapacity + slot`, ordered by distance then feature row, padded with
   * `GPU_SPATIAL_JOIN_NO_FEATURE`. Length `queryCount * neighborCapacity`.
   */
  neighborIds?: GraphDataView<'uint32'>;
  /** Neighbors mode, required with `neighborIds`: number of filled slots per query. Length `queryCount`. */
  neighborCounts?: GraphDataView<'uint32'>;
  /** Neighbors mode: distance per slot (in the spatial context length unit), `-1` in unused slots. Same layout as `neighborIds`. */
  neighborDistances?: GraphDataView<'float32'>;
  /**
   * Neighbors mode: the point of the feature nearest to the query per slot (the query point
   * projected onto the feature, the crossing point when geometries cross, or the contained vertex
   * for polygon containment); NaN in unused slots. Same layout as `neighborIds`.
   */
  neighborFootPoints?: GraphDataView<'float32x2'>;
  /**
   * Neighbors mode: absolute index in the feature's position buffer of the first vertex of the
   * edge attaining the distance (`GPU_NEAREST_NO_SEGMENT` for point features, containment and
   * unused slots; always 0 for `segments` features). Same layout as `neighborIds`.
   */
  neighborSegmentIndices?: GraphDataView<'uint32'>;
  /**
   * Neighbors mode: the point of the query geometry nearest to the feature per slot (the query
   * point itself for point queries); NaN in unused slots. Same layout as `neighborIds`.
   * `(neighborQueryPoints, neighborFootPoints)` is `shapely.shortest_line` and
   * `shapely.ops.nearest_points` for every kind pair (line and polygon queries included). For
   * crossing geometries both are the crossing point and for polygon containment both are the
   * contained vertex. Where several point pairs attain the distance (parallel edges) the first in
   * vertex order is returned, which may differ from GEOS.
   */
  neighborQueryPoints?: GraphDataView<'float32x2'>;
  /** Nearest-feature mode: optional per-point distance to the nearest feature (in the spatial context length unit), or -1. Chunked like the query points. */
  nearestDistances?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Nearest-feature mode: optional per-feature count of points whose nearest feature it is. */
  featureCounts?: GraphDataView<'uint32'>;
  /** Optional compact stable IDs of points with a feature within the radius. */
  matches?: GPUCompactOutput;
  /**
   * Completeness scalars. Optional `antimeridianUncertainCount` counts unprepared crossing rings;
   * optional `poleUncertainCount` counts unprepared pole-enclosing rings. Prepare these polygons
   * with `GPULonLatPolygonPreparation`. Both fields are left untouched in planar contexts.
   */
  status: GPULonLatPolygonStatusPort & {
    /**
     * One-row flag: 1 when BVH leaf, candidate, `matches` or (neighbors mode) tie capacity
     * overflowed.
     */
    overflow: GraphDataView<'uint32'>;
    /** Optional one-row unclamped candidate count. */
    candidateCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Largest packed node table of {@link GPUNearestFeatureJoin} on `device`, in uint32 words, in
 * neighbors mode and on a great-circle context: `5 * (2 * leafCapacity - 1) + 2 * queryCount`, plus
 * `2 * featureCount` with `exclusive` or `onAttribute`. The table is read at random by the BVH
 * traversal and must fit one storage binding, so 2^25 words on a default device (128 MiB
 * bindings). Without queries or filters that allows `leafCapacity` up to 2^21 (2097152 features).
 * The per-query traversal scratch is already chunked and does not add a limit.
 */
export function getGPUNearestFeatureJoinMaximumNodeTableWordCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Largest `leafCapacity` of {@link GPUNearestFeatureJoin} on `device`: the BVH holds
 * `2 * leafCapacity - 1` float32x2 node bounds per corner table, 8 bytes per node, in one storage
 * binding, so 2^23 leaves (features) on a default device.
 */
export function getGPUNearestFeatureJoinMaximumLeafCapacity(device: Device): number {
  // Largest power of two L with (2 * L - 1) nodes of 8 bytes in one binding.
  return 2 ** Math.floor(Math.log2(Math.floor((getStorageBindingElementLimit(device, 8) + 1) / 2)));
}

/**
 * Properties for {@link GPUNearestFeatureJoin}.
 *
 * Two output modes share one class:
 *
 * - **Nearest-feature mode**: point `queries`, `radius`, `candidateCapacity` and
 *   `output.nearestFeatureIds`. One nearest point or segment feature per point within a radius.
 * - **Neighbors mode**: set `output.neighborIds` and `output.neighborCounts`. Each query (a chunked
 *   point set, or any line or polygon set) gets its `k` nearest features from a
 *   branch-and-bound BVH traversal, in dense slots of `neighborCapacity` per query, ordered by
 *   `(distance, feature row)`. Features may be points, segments, linestrings or polygons, so this
 *   mode covers point, line/line, line/polygon and polygon/polygon minimum distance. It is
 *   `sjoin_nearest` and PostGIS `ORDER BY a <-> b LIMIT k`.
 *
 * Per-frame: `radius` and the contents of every input buffer. Topology: view
 * lengths and chunking, `k`, `neighborCapacity`, `ties`, `candidateCapacity`, `leafCapacity`,
 * `spatialSort`, and which optional views exist.
 */
export type GPUNearestFeatureJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'nearest-feature-join'`. */
  id?: string;
  /**
   * Coordinate space and metric of the points and features.
   *
   * - Omitted, or planar: native planar coordinates; distances (`radius`, `nearestDistances`,
   *   `neighborDistances`) are planar in coordinate units.
   * - `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`: positions are longitude and
   *   latitude in degrees and every distance is a great-circle distance in the context length unit on the mean Earth
   *   sphere (meters by default, or kilometers with `units: 'kilometers'`; `sphereRadius`
   *   overrides the radius in that unit). Both modes use the branch-and-bound BVH traversal with a
   *   conservative great-circle lower bound on node boxes (latitude gap, longitude gap through the
   *   antimeridian when shorter, and the cosine of the latitude farthest from the equator), so
   *   `candidateCapacity` is not used (it is accepted but ignored, `candidateCount` reads 0) and a
   *   pair across the antimeridian or near a pole is measured the short way round.
   *   Edges are straight in longitude/latitude (longitude differences wrapped at +-180), the same
   *   edge model as the planar-in-degrees containment and crossing tests, so densify long edges;
   *   foot points are on the edge and longitudes are wrapped to +-180. Prepare polygons that
   *   cross ±180 or enclose a pole with `GPULonLatPolygonPreparation`; optional status counters
   *   report unprepared rings on both polygon sides. Longitude/latitude requires `edgeModel: 'linear'`
   *   (the default); great-circle edges require preparation. Positions are float32, so distances
   *   carry the coordinate quantization (about 1 m at longitude 100).
   * - Any other longitude/latitude metric throws at construction.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `features` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Longitude/latitude must use linear edges.
   * Omitted means `spatialContext` describes both inputs.
   */
  featuresSpatialContext?: GPUSpatialContext;
  /**
   * Query geometry, one query per row. Points may be chunked (their optional `sourceIds`, with the
   * same chunk topology, are the IDs written to `output.matches`); line and polygon queries
   * (neighbors mode only) are one packed view each.
   */
  queries: GPUNearestQueryGeometry;
  /**
   * Features. Nearest-feature mode accepts points and segments; neighbors mode also accepts
   * linestrings and polygons. Polygon containment gives distance 0. Optional `sourceIds` are
   * stable feature IDs written instead of feature rows.
   */
  features: GPUNearestFeatureGeometry;
  /**
   * Per-frame inclusive search radius (in the spatial context length unit), one float32
   * row; features farther than this are never returned. NaN, negative, or infinite matches
   * nothing. Required in nearest-feature mode; omitted in neighbors mode means unbounded.
   */
  radius?: GraphDataView<'float32'>;
  /**
   * Skip features whose ID equals the query's ID (`STRtree.query_nearest(exclusive=True)`, the
   * self-neighbor rule of a nearest self-join). A feature's ID is `features.sourceIds[row]`, or its row
   * without them; a query's ID is `queryIds[row]`, or its row without `queryIds`. Skipped
   * features are never counted, so `k` neighbors are the nearest *other* features. Compile-time.
   * Works in both modes.
   */
  exclusive?: boolean;
  /**
   * Optional `uint32` ID per query row (length is the query count), compared with feature IDs by
   * `exclusive`. Requires `exclusive`. Unlike point `sourceIds` it is never chunked.
   */
  queryIds?: GraphDataView<'uint32'>;
  /**
   * Attribute-equality condition (GeoPandas `sjoin_nearest` after `on_attribute`): only features
   * whose `right` key equals the query's `left` key compete, so the nearest *key-equal* feature is
   * returned. `left` has one key per query row and `right` one per feature row. Compile-time
   * presence, per-frame contents. Combines with `exclusive`. Works in both modes.
   */
  onAttribute?: GPUSpatialJoinOnAttribute;
  /** Nearest-feature mode only: maximum `(point, feature)` bounding-box candidates per encoding. */
  candidateCapacity?: number;
  /** Neighbors mode: neighbors wanted per query, an integer in `[1, 32]`. Default 1. */
  k?: number;
  /**
   * Neighbors mode: output slots per query, at least `k` and at most 64. Defaults to `k`. It only
   * matters with `ties: 'all'`: ties beyond the capacity are dropped and `overflow` is set.
   */
  neighborCapacity?: number;
  /** Neighbors mode: tie rule at the k-th distance. Default `'lowest-id'`. */
  ties?: GPUNearestTieMode;
  /**
   * Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. In
   * neighbors mode and on a great-circle context the node table of `leafCapacity` and the query
   * count is at most {@link getGPUNearestFeatureJoinMaximumNodeTableWordCount} words (2^25 on a default device).
   */
  leafCapacity?: number;
  /**
   * Compile-time. When true, features are reordered along a Hilbert curve (see `spatialSortCurve`) of their bound
   * centers before the BVH build, so leaves that are close in space are close in the tree and
   * internal node bounds stay tight. Empty or invalid features sort last.
   *
   * Results are identical either way (the nearest feature wins and ties keep the smallest feature row); only traversal cost changes. Default false.
   *
   * Measured on an Apple M3 Pro with 50,000 shuffled short
   * segments, 100,000 points and radius 1: 309 ms drops to 4.1 ms per encoding (about 33,000 to 76
   * BVH nodes visited per point). Candidate counts are identical.
   *
   * Defaults to on from 256 features and off below, where the BVH is too shallow for the order to
   * matter. On coherent data the sort costs about 1 ms; pass `false` to skip it.
   */
  spatialSort?: boolean;
  /**
   * Experimental, compile-time. Curve used by `spatialSort`: `'hilbert'` (default; 10 to 15% faster joins
   * than Morton in paired A/B runs) or `'morton'` (Z-order). Results are identical; only BVH
   * traversal cost changes.
   */
  spatialSortCurve?: SpatialSortCurve;
  /**
   * Compile-time. When true, queries are processed in Hilbert order of points (each chunk) or
   * line/polygon bounds centers (quantized over the feature BVH root bounds, non-finite points
   * and empty bounds last) instead
   * of input order, so neighboring invocations walk neighboring BVH subtrees. Every query still
   * reads and writes its own row, so all outputs are identical either way; only traversal cost
   * changes. Costs one key pass and a radix sort per query chunk, plus geometry bounds and centers
   * for line/polygon queries. Polygon query topology arrays are packed to stay within eight bindings.
   *
   * In nearest-feature mode with `radius` (planar), the second candidate pass keeps input order
   * to stay within eight storage bindings.
   *
   * Defaults off for line and polygon queries: bounds, centers and radix sorting cost more than
   * traversal saves in the measured 8192-line and polygon workloads. Explicit `true` still enables
   * it. Point queries default to on from 4096 queries and 256 features (see
   * `spatial-join-query-order.spec.ts` for the A/B harness), and off below, where the sort cannot
   * pay for itself.
   */
  querySort?: boolean;
  /** Caller-owned outputs; `neighborIds` selects neighbors mode. */
  output: GPUNearestFeatureJoinOutput;
  /**
   * Prepared (static) feature index from {@link GPUSpatialJoinPrepared}: reuses its bounds and BVH
   * across encodings instead of rebuilding them, and takes `leafCapacity` and `spatialSort` from
   * the handle. The handle must be added to the graph before this join, be built over the same
   * `features` (points, lines or polygons), and be invalidated when they change. It reuses a
   * build; results are never cached.
   */
  prepared?: GPUSpatialJoinPrepared;
};

/** Flat internal view of the props that the node builders read. */
type NearestFeatureJoinViews = Omit<GPUNearestFeatureJoinProps, 'queries' | 'output'> &
  Omit<GPUNearestFeatureJoinOutput, 'status'> &
  GPULonLatPolygonStatusPort & {
    /** Point queries, possibly chunked. */
    points?: GPUFloat32Positions;
    /** Line or polygon queries. */
    queries?: GPUPackedGeometryPort<GPULineGeometryPort | GPUPolygonGeometryPort>;
    /** Point query IDs with the chunk topology of `points`. */
    sourceIds?: GPUUint32Rows;
    /** Feature IDs written instead of feature rows. */
    featureIds?: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    candidateCount?: GraphDataView<'uint32'>;
  };

function getNearestFeatureJoinViews(props: GPUNearestFeatureJoinProps): NearestFeatureJoinViews {
  const {queries, output, ...scalars} = props;
  const {status, ...outputs} = output;
  return {
    ...scalars,
    ...outputs,
    points: queries.kind === 'points' ? queries.positions : undefined,
    queries: queries.kind === 'points' ? undefined : queries,
    sourceIds: queries.kind === 'points' ? queries.sourceIds : undefined,
    featureIds: props.features.sourceIds,
    overflow: status.overflow,
    candidateCount: status.candidateCount,
    antimeridianUncertainCount: status.antimeridianUncertainCount,
    poleUncertainCount: status.poleUncertainCount
  };
}

/**
 * Joins queries to their nearest features by planar or great-circle distance.
 *
 * Nearest-feature mode: a `GPUBVH` over feature bounds produces candidates inside the radius box,
 * `GPUPairwisePointSegmentDistance` measures them, and deterministic `atomicMin` passes keep the
 * nearest feature (ties go to the smallest feature row).
 *
 * Neighbors mode (`neighborIds` set): every query walks the BVH with a stack, nearer child first,
 * pruning nodes farther than its current k-th distance, and keeps the best rows ordered by
 * `(distance, feature row)`. The result is exact and deterministic: ties at the k-th distance keep
 * the lowest feature rows, or all of them with `ties: 'all'`. Distances are measured between
 * the true geometries (point, linestring or polygon on either side), so no radius guess or
 * candidate capacity is needed; the only bounded output is the per-query slot count, and tie
 * truncation raises `overflow`.
 */
export class GPUNearestFeatureJoin implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNearestFeatureJoinProps;
  /** Flat view of `props` read by the node builders. */
  private readonly views: NearestFeatureJoinViews;
  /** `'neighbors'` when `neighborIds` is set, otherwise the original `'nearest-feature'` mode. */
  readonly mode: 'nearest-feature' | 'neighbors';
  /** Number of features. */
  readonly featureCount: number;
  /** Number of queries (points or query geometries). */
  readonly queryCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Candidate pair capacity (nearest-feature mode; 0 in neighbors mode). */
  readonly candidateCapacity: number;
  /** Whether features are spatially sorted before the BVH build. */
  readonly spatialSort: boolean;
  /** Whether queries are processed in Hilbert order. */
  readonly querySort: boolean;
  /** Resolved neighbors per query (1 in nearest-feature mode). */
  readonly k: number;
  /** Resolved output slots per query (1 in nearest-feature mode). */
  readonly neighborCapacity: number;
  /** Resolved tie rule. */
  readonly ties: GPUNearestTieMode;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius in the context length unit of a great-circle search, or undefined for planar distances. */
  readonly sphereRadius: number | undefined;

  constructor(props: GPUNearestFeatureJoinProps) {
    this.id = props.id ?? 'nearest-feature-join';
    this.props = props;
    const views = getNearestFeatureJoinViews(props);
    this.views = views;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      views.spatialContext,
      NEAREST_FEATURE_JOIN_SPATIAL_CONTEXT_SUPPORT
    );
    resolveGPUSecondarySpatialContext(
      id,
      'features',
      views.featuresSpatialContext,
      this.spatialContext,
      NEAREST_FEATURE_JOIN_SPATIAL_CONTEXT_SUPPORT.edgeModels
    );
    this.sphereRadius = getGPUGreatCircleRadius(this.spatialContext);
    this.mode = views.neighborIds ? 'neighbors' : 'nearest-feature';
    this.featureCount = getNearestGeometryCount(views.features);
    this.k = views.k ?? 1;
    this.ties = views.ties ?? 'lowest-id';
    const {prepared} = views;
    if (prepared) {
      if (views.features.kind === 'segments' || prepared.geometry.kind !== views.features.kind) {
        throw new Error(`${id} prepared must index the same point, line or polygon features`);
      }
      if (prepared.featureCount !== this.featureCount) {
        throw new Error(`${id} prepared feature count must equal the feature count`);
      }
      if (
        (views.leafCapacity !== undefined && views.leafCapacity !== prepared.leafCapacity) ||
        (views.spatialSort !== undefined && views.spatialSort !== prepared.spatialSort)
      ) {
        throw new Error(`${id} leafCapacity and spatialSort come from the prepared handle`);
      }
    }
    this.spatialSort = prepared
      ? prepared.spatialSort
      : (views.spatialSort ?? getDefaultSpatialSort(this.featureCount));
    this.leafCapacity =
      prepared?.leafCapacity ??
      views.leafCapacity ??
      getNextPowerOfTwo(Math.max(this.featureCount, 1));
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    if (views.radius) {
      validatePackedView(views.radius, ['float32'], `${id} radius`);
      if (views.radius.length !== 1) {
        throw new Error(`${id} radius needs one float32 row`);
      }
    }
    for (const view of getNearestGeometryViews(views.features)) {
      validatePackedView(
        view,
        view.format === 'uint32' ? ['uint32'] : ['float32x2'],
        `${id} features`
      );
    }
    if (views.features.kind === 'segments') {
      if (views.features.starts.length !== views.features.ends.length) {
        throw new Error(`${id} features.starts and ends lengths differ`);
      }
    }
    for (const [name, view] of [
      ['features.sourceIds', views.featureIds],
      ['output.featureCounts', views.featureCounts]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} ${name} needs one row per feature`);
        }
      }
    }
    validateStatusScalars(id, props.output.status);
    if (this.mode === 'neighbors') {
      this.neighborCapacity = views.neighborCapacity ?? this.k;
      this.candidateCapacity = 0;
      this.queryCount = this.validateNeighbors();
    } else {
      this.neighborCapacity = 1;
      this.candidateCapacity = this.sphereRadius === undefined ? (views.candidateCapacity ?? 0) : 0;
      this.queryCount = this.validateNearestFeature();
    }
    this.querySort =
      views.querySort ??
      (views.queries === undefined && getDefaultQuerySort(this.queryCount, this.featureCount));
    this.validateFilters();
    validateDisjointOutputs(id, getInputs(views), getOutputs(views));
  }

  /** Validates `exclusive`, `queryIds` and `onAttribute` against the query and feature counts. */
  private validateFilters(): void {
    const {id, views, queryCount, featureCount} = this;
    if (views.queryIds && !views.exclusive) {
      throw new Error(`${id} queryIds requires exclusive`);
    }
    if (views.queryIds) {
      validatePackedView(views.queryIds, ['uint32'], `${id} queryIds`);
      if (views.queryIds.length !== queryCount) {
        throw new Error(`${id} queryIds needs one row per query`);
      }
    }
    if (views.onAttribute) {
      const {left, right} = views.onAttribute;
      validatePackedView(left, ['uint32'], `${id} onAttribute.left`);
      validatePackedView(right, ['uint32'], `${id} onAttribute.right`);
      if (left.length !== queryCount || right.length !== featureCount) {
        throw new Error(`${id} onAttribute needs one key per query and feature`);
      }
    }
  }

  /** Validates the neighbors-mode views and returns the query count. */
  private validateNeighbors(): number {
    const {id, views, k, neighborCapacity} = this;
    validateCapacity(id, 'k', k, 1, 32);
    validateCapacity(id, 'neighborCapacity', neighborCapacity, k, 64);
    if (neighborCapacity > k && this.ties !== 'all') {
      throw new Error(`${id} neighborCapacity above k needs ties: 'all'`);
    }
    if (!['lowest-id', 'all'].includes(this.ties)) {
      throw new Error(`${id} ties must be 'lowest-id' or 'all'`);
    }
    if (views.nearestFeatureIds || views.nearestDistances || views.featureCounts || views.matches) {
      throw new Error(`${id} nearest-feature outputs need nearestFeatureIds`);
    }
    let queryCount: number;
    if (views.points) {
      for (const chunk of getGraphViewChunks(views.points)) {
        validatePackedView(chunk, ['float32x2'], `${id} queries.positions`);
      }
      queryCount = views.points.length;
    } else {
      for (const view of getNearestGeometryViews(views.queries!)) {
        validatePackedView(
          view,
          view.format === 'uint32' ? ['uint32'] : ['float32x2'],
          `${id} queries`
        );
      }
      queryCount = getNearestGeometryCount(views.queries!);
    }
    const slotCount = queryCount * neighborCapacity;
    for (const [name, view, length] of [
      ['neighborIds', views.neighborIds, slotCount],
      ['neighborCounts', views.neighborCounts, queryCount],
      ['neighborDistances', views.neighborDistances, slotCount],
      ['neighborFootPoints', views.neighborFootPoints, slotCount],
      ['neighborSegmentIndices', views.neighborSegmentIndices, slotCount],
      ['neighborQueryPoints', views.neighborQueryPoints, slotCount]
    ] as const) {
      if (name === 'neighborCounts' && !view) {
        throw new Error(`${id} neighborIds needs neighborCounts`);
      }
      if (view) {
        validatePackedView(
          view,
          name === 'neighborDistances'
            ? ['float32']
            : name === 'neighborFootPoints' || name === 'neighborQueryPoints'
              ? ['float32x2']
              : ['uint32'],
          `${id} output.${name}`
        );
        if (view.length !== length) {
          throw new Error(`${id} output.${name} needs ${length} rows`);
        }
      }
    }
    return queryCount;
  }

  /** Validates the nearest-feature-mode views and returns the point count. */
  private validateNearestFeature(): number {
    const {id, views} = this;
    const {features} = views;
    if (
      views.queries ||
      views.k !== undefined ||
      views.neighborCapacity ||
      views.ties ||
      views.neighborQueryPoints
    ) {
      throw new Error(`${id} line/polygon queries, k and ties need neighborIds`);
    }
    if (!views.points || !views.nearestFeatureIds) {
      throw new Error(`${id} needs point queries and nearestFeatureIds`);
    }
    if (features.kind !== 'points' && features.kind !== 'segments') {
      throw new Error(`${id} nearest-feature mode needs point or segment features`);
    }
    if (!views.radius) {
      throw new Error(`${id} needs radius`);
    }
    for (const chunk of getGraphViewChunks(views.points)) {
      validatePackedView(chunk, ['float32x2'], `${id} queries.positions`);
    }
    for (const [name, view] of [
      ['output.nearestFeatureIds', views.nearestFeatureIds],
      ['queries.sourceIds', views.sourceIds]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validatePackedView(chunk, ['uint32'], `${id} ${name}`);
      }
      validateMatchingChunks(id, name, views.points, view);
    }
    for (const chunk of views.nearestDistances ? getGraphViewChunks(views.nearestDistances) : []) {
      validatePackedView(chunk, ['float32'], `${id} output.nearestDistances`);
    }
    validateMatchingChunks(id, 'output.nearestDistances', views.points, views.nearestDistances);
    if (views.matches) {
      validateCompactOutput(id, views.matches);
    }
    if (this.sphereRadius === undefined) {
      validateCapacity(id, 'candidateCapacity', this.candidateCapacity);
    }
    return views.points.length;
  }

  /**
   * Returns the command nodes: bounds, BVH, then either the candidate/distance/reduction passes of
   * nearest-feature mode or the traversal and column passes of neighbors mode.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {views} = this;
    validateGraphViewsBelongToGraph(this.id, graph, [...getInputs(views), ...getOutputs(views)]);
    validatePowerOfTwo(
      this.id,
      'leafCapacity',
      this.leafCapacity,
      1,
      getGPUNearestFeatureJoinMaximumLeafCapacity(graph.device)
    );
    if (this.mode === 'neighbors' || this.sphereRadius !== undefined) {
      // The traversal reads one packed node table: 5 words per BVH node (2 * leafCapacity - 1
      // nodes), 2 per query and, with a candidate filter, 2 per feature.
      validateMaximumCount(
        this.id,
        'node table words (leafCapacity, queries)',
        (2 * this.leafCapacity - 1) * NEAREST_NODE_WORDS +
          2 * this.queryCount +
          (views.exclusive || views.onAttribute ? 2 * this.featureCount : 0),
        getGPUNearestFeatureJoinMaximumNodeTableWordCount(graph.device)
      );
    }
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (
      this.spatialContext.coordinateSpace === 'longitude-latitude' &&
      hasLonLatPolygonStatus(views)
    ) {
      for (const name of ['antimeridianUncertainCount', 'poleUncertainCount'] as const) {
        const view = views[name];
        if (view) {
          nodes.push(
            createFillNode(graph, {
              id: `${this.id}-clear-${name}`,
              operation: OPERATION,
              view,
              type: 'u32',
              value: '0u',
              componentCount: 1
            })
          );
        }
      }
      for (const [polygonIndex, polygons] of [views.queries, views.features]
        .filter(geometry => geometry?.kind === 'polygons')
        .entries()) {
        nodes.push(
          createLonLatPolygonCheckNode(graph, {
            id: `${this.id}-lon-lat-polygon-check-${polygonIndex}`,
            operation: OPERATION,
            positions: polygons.positions,
            ringOffsets: polygons.ringOffsets,
            status: {
              antimeridianUncertainCount: views.antimeridianUncertainCount,
              poleUncertainCount: views.poleUncertainCount
            }
          })
        );
      }
    }
    if (this.mode === 'neighbors') {
      nodes.push(
        ...getNearestNeighborNodes<Parameters>(graph, {
          id: this.id,
          operation: OPERATION,
          queries: views.points
            ? {kind: 'points', points: views.points}
            : {kind: 'geometry', geometry: views.queries!},
          queryCount: this.queryCount,
          features: views.features,
          featureCount: this.featureCount,
          k: this.k,
          capacity: this.neighborCapacity,
          ties: this.ties,
          leafCapacity: this.leafCapacity,
          spatialSort: this.spatialSort,
          spatialSortCurve: views.spatialSortCurve,
          prepared: views.prepared,
          maxDistance: views.radius,
          featureIds: views.featureIds,
          neighborIds: views.neighborIds!,
          neighborCounts: views.neighborCounts!,
          neighborDistances: views.neighborDistances,
          neighborFootPoints: views.neighborFootPoints,
          neighborSegmentIndices: views.neighborSegmentIndices,
          neighborQueryPoints: views.neighborQueryPoints,
          exclusive: views.exclusive,
          queryIds: views.queryIds,
          onAttribute: views.onAttribute,
          overflow: views.overflow,
          sphereRadius: this.sphereRadius,
          querySort: this.querySort
        }).nodes
      );
    } else {
      nodes.push(
        ...(this.sphereRadius === undefined
          ? this.getNearestFeatureNodes(graph)
          : this.getGreatCircleNearestFeatureNodes(graph))
      );
    }
    return nodes;
  }

  /**
   * Nearest-feature mode on longitude/latitude: the neighbors traversal with `k = 1` (ties to the
   * smallest feature row) followed by the nearest-feature outputs (IDs, distances, counts,
   * matches, overflow).
   */
  private getGreatCircleNearestFeatureNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, views, featureCount, queryCount} = this;
    const points = views.points!;
    const {matches} = views;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const featureRows = createTransientView(graph, `${id}-feature-rows`, 'uint32', queryCount);
    const rowCounts = createTransientView(graph, `${id}-row-counts`, 'uint32', queryCount);
    const rowDistances = createTransientView(graph, `${id}-row-distances`, 'float32', queryCount);
    const traversalOverflow = createTransientView(graph, `${id}-traversal-overflow`, 'uint32', 1);
    // Clears the state and the per-feature counts before the traversal fills the rows.
    nodes.push(
      createSpatialJoinClearNode<Parameters>(graph, {
        id: `${id}-clear-state`,
        operation: OPERATION,
        state,
        assignment: featureRows,
        pointCount: queryCount,
        featureCounts: views.featureCounts
      })
    );
    const {nodes: traversalNodes} = getNearestNeighborNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      queries: {kind: 'points', points},
      queryCount,
      features: views.features,
      featureCount,
      k: 1,
      capacity: 1,
      leafCapacity: this.leafCapacity,
      spatialSort: this.spatialSort,
      spatialSortCurve: views.spatialSortCurve,
      prepared: views.prepared,
      maxDistance: views.radius!,
      featureIds: views.featureIds,
      neighborIds: featureRows,
      neighborCounts: rowCounts,
      neighborDistances: rowDistances,
      exclusive: views.exclusive,
      queryIds: views.queryIds,
      onAttribute: views.onAttribute,
      overflow: traversalOverflow,
      sphereRadius: this.sphereRadius,
      writeFeatureRows: true,
      querySort: this.querySort
    });
    nodes.push(...traversalNodes);

    const distanceChunks = views.nearestDistances ? getGraphViewChunks(views.nearestDistances) : [];
    let chunkFirstRow = 0;
    for (const [chunkIndex, chunk] of getGraphViewChunks(views.nearestFeatureIds!).entries()) {
      if (chunk.length > 0) {
        const bindings: WGSLKernelBinding[] = [
          {name: 'featureRows', view: featureRows, type: 'u32', access: 'read'},
          {name: 'pointFeatureIds', view: chunk, type: 'u32', access: 'read_write'}
        ];
        if (views.featureIds) {
          bindings.push({name: 'featureIds', view: views.featureIds, type: 'u32', access: 'read'});
        }
        if (views.featureCounts) {
          bindings.push({
            name: 'featureCounts',
            view: views.featureCounts,
            type: 'atomic<u32>',
            access: 'read_write'
          });
        }
        const distanceChunk = distanceChunks[chunkIndex];
        if (distanceChunk) {
          bindings.push(
            {name: 'rowDistances', view: rowDistances, type: 'f32', access: 'read'},
            {name: 'distances', view: distanceChunk, type: 'f32', access: 'read_write'}
          );
        }
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: getChunkNodeId(`${id}-assign`, views.nearestFeatureIds!, chunkIndex),
            operation: OPERATION,
            variant: 'assign-great-circle',
            bindings,
            invocationCount: chunk.length,
            declarations: `const NO_FEATURE: u32 = 0xffffffffu;
const CHUNK_FIRST_ROW: u32 = ${chunkFirstRow}u;`,
            body: `let row = CHUNK_FIRST_ROW + index;
  let featureRow = featureRows[featureRowsOffset + row];
  let matched = featureRow != NO_FEATURE;
  var featureId = featureRow;
  ${views.featureIds ? 'if (matched) { featureId = featureIds[featureIdsOffset + featureRow]; }' : ''}
  pointFeatureIds[pointFeatureIdsOffset + index] = featureId;
  ${views.featureCounts ? 'if (matched) { atomicAdd(&featureCounts[featureCountsOffset + featureRow], 1u); }' : ''}
  ${distanceChunk ? 'distances[distancesOffset + index] = select(-1.0, rowDistances[rowDistancesOffset + row], matched);' : ''}`
          })
        );
      }
      chunkFirstRow += chunk.length;
    }
    if (matches) {
      nodes.push(
        ...getSpatialJoinCollectNodes<Parameters>(graph, {
          id: `${id}-collect-matches`,
          operation: OPERATION,
          points,
          assignment: featureRows,
          state,
          sourceIds: views.sourceIds,
          matches
        })
      );
    }
    nodes.push(
      createSpatialJoinFinalizeNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        state,
        // The traversal clear pass already folds the BVH overflow into its overflow flag.
        bvhOverflow: traversalOverflow,
        candidateCapacity: 0,
        overflow: views.overflow,
        candidateCount: views.candidateCount,
        matches
      })
    );
    return nodes;
  }

  /** Nearest-feature mode: bounds, BVH, candidate, distance, reduction, assignment, finalize. */
  private getNearestFeatureNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, views, featureCount, leafCapacity, candidateCapacity} = this;
    const points = views.points!;
    const features = views.features as GPUNearestFeatureSource;
    const radius = views.radius!;
    const {matches} = views;
    const pointCount = points.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const {bvh: featureBVH, nodes: bvhNodes} = getNearestBVHNodes(graph, {
      id,
      operation: OPERATION,
      features,
      featureCount,
      leafCapacity,
      spatialSort: this.spatialSort,
      spatialSortCurve: views.spatialSortCurve,
      prepared: views.prepared
    });
    // The probe pass only reads node bounds, leaf IDs and the internal node count.
    const bvh = featureBVH as GPUBVH;
    nodes.push(...bvhNodes);

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const assignment = createTransientView(graph, `${id}-assignment`, 'uint32', pointCount);
    const bestDistanceBits = createTransientView(
      graph,
      `${id}-best-distance-bits`,
      'uint32',
      pointCount
    );
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      candidateCapacity
    );
    const candidatePoints = createTransientView(
      graph,
      `${id}-candidate-points`,
      'float32x2',
      candidateCapacity
    );
    const candidateStarts = createTransientView(
      graph,
      `${id}-candidate-starts`,
      'float32x2',
      candidateCapacity
    );
    const candidateEnds = createTransientView(
      graph,
      `${id}-candidate-ends`,
      'float32x2',
      candidateCapacity
    );
    const candidateDistances = createTransientView(
      graph,
      `${id}-candidate-distances`,
      'float32',
      candidateCapacity
    );
    nodes.push(
      createSpatialJoinClearNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        state,
        assignment,
        pointCount,
        featureCounts: views.featureCounts,
        bestDistanceBits
      })
    );
    nodes.push(
      ...getSpatialJoinProbeNodes<Parameters>(graph, {
        id: `${id}-probe`,
        operation: OPERATION,
        points,
        bvh,
        featureCount,
        candidateCapacity,
        state,
        candidatePairs,
        candidatePoints,
        pointRowScale: 1,
        pointRowOffset: 0,
        radius,
        querySort: this.querySort
      })
    );
    const candidateDispatch = createWGSLActiveCountDispatch(graph, {
      id: `${id}-candidate-dispatch`,
      operation: OPERATION,
      count: state,
      maximumItemCount: candidateCapacity
    });
    nodes.push(candidateDispatch.updateNode);

    const expandBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'},
      {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'}
    ];
    if (features.kind === 'points') {
      expandBindings.push({
        name: 'featureStarts',
        view: features.positions,
        type: 'f32',
        access: 'read'
      });
    } else {
      expandBindings.push(
        {name: 'featureStarts', view: features.starts, type: 'f32', access: 'read'},
        {name: 'featureEnds', view: features.ends, type: 'f32', access: 'read'}
      );
    }
    expandBindings.push(
      {name: 'candidateStarts', view: candidateStarts, type: 'f32', access: 'read_write'},
      {name: 'candidateEnds', view: candidateEnds, type: 'f32', access: 'read_write'}
    );
    const endSource = features.kind === 'points' ? 'featureStarts' : 'featureEnds';
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-expand`,
        operation: OPERATION,
        variant: `expand-${features.kind}`,
        bindings: expandBindings,
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  candidateStarts[candidateStartsOffset + index * 2u] = featureStarts[featureStartsOffset + featureRow * 2u];
  candidateStarts[candidateStartsOffset + index * 2u + 1u] = featureStarts[featureStartsOffset + featureRow * 2u + 1u];
  candidateEnds[candidateEndsOffset + index * 2u] = ${endSource}[${endSource}Offset + featureRow * 2u];
  candidateEnds[candidateEndsOffset + index * 2u + 1u] = ${endSource}[${endSource}Offset + featureRow * 2u + 1u];`
      })
    );
    nodes.push(
      ...new GPUPairwisePointSegmentDistance({
        id: `${id}-distance`,
        positions: candidatePoints,
        segmentStarts: candidateStarts,
        segmentEnds: candidateEnds,
        output: candidateDistances
      })
        .getCommandNodes(graph)
        .map(node => {
          // GPUPairwisePointSegmentDistance declares compute passes only.
          assert(node.type === 'compute');
          return {...node, condition: candidateDispatch.condition};
        })
    );
    // Candidates failing `exclusive` or `onAttribute` never compete for the nearest feature.
    const filterBindings: WGSLKernelBinding[] = [];
    const filterTests: string[] = [];
    if (views.exclusive) {
      if (views.queryIds) {
        filterBindings.push({name: 'queryIds', view: views.queryIds, type: 'u32', access: 'read'});
      }
      if (views.featureIds) {
        filterBindings.push({
          name: 'featureIds',
          view: views.featureIds,
          type: 'u32',
          access: 'read'
        });
      }
      const queryId = views.queryIds ? 'queryIds[queryIdsOffset + pointRow]' : 'pointRow';
      const featureId = views.featureIds
        ? 'featureIds[featureIdsOffset + candidateFeature]'
        : 'candidateFeature';
      filterTests.push(`if (${queryId} == ${featureId}) { rejectCandidate(index); return; }`);
    }
    if (views.onAttribute) {
      filterBindings.push(
        {name: 'queryKeys', view: views.onAttribute.left, type: 'u32', access: 'read'},
        {name: 'featureKeys', view: views.onAttribute.right, type: 'u32', access: 'read'}
      );
      filterTests.push(
        'if (queryKeys[queryKeysOffset + pointRow] != featureKeys[featureKeysOffset + candidateFeature]) { rejectCandidate(index); return; }'
      );
    }
    if (filterTests.length) {
      // A separate pass marks filtered candidates with distance -1, which the reductions reject.
      // Folding the filter views into the reductions bound up to 10 storage buffers, above the
      // default limit of 8.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-filter`,
          operation: OPERATION,
          variant: 'filter',
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'candidateDistances',
              view: candidateDistances,
              type: 'f32',
              access: 'read_write'
            },
            ...filterBindings
          ],
          invocationCount: candidateCapacity,
          condition: candidateDispatch.condition,
          declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
fn rejectCandidate(index: u32) { candidateDistances[candidateDistancesOffset + index] = -1.0; }`,
          body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * 2u];
  let candidateFeature = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  ${filterTests.join('\n  ')}`
        })
      );
    }
    const reducePrologue = `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * 2u];
  let distance = candidateDistances[candidateDistancesOffset + index];
  let searchRadius = radius[radiusOffset];
  if (!(distance >= 0.0 && distance <= searchRadius)) { return; }
  // Fold -0.0 so the bit pattern of non-negative distances is monotone.
  let distanceBits = select(bitcast<u32>(distance), 0u, distance == 0.0);`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce-distance`,
        operation: OPERATION,
        variant: 'reduce-distance',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'candidateDistances', view: candidateDistances, type: 'f32', access: 'read'},
          {name: 'radius', view: radius, type: 'f32', access: 'read'},
          {
            name: 'bestDistanceBits',
            view: bestDistanceBits,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `${reducePrologue}
  atomicMin(&bestDistanceBits[bestDistanceBitsOffset + pointRow], distanceBits);`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce-feature`,
        operation: OPERATION,
        variant: 'reduce-feature',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'candidateDistances', view: candidateDistances, type: 'f32', access: 'read'},
          {name: 'radius', view: radius, type: 'f32', access: 'read'},
          {name: 'bestDistanceBits', view: bestDistanceBits, type: 'u32', access: 'read'},
          {name: 'assignment', view: assignment, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `${reducePrologue}
  if (distanceBits == bestDistanceBits[bestDistanceBitsOffset + pointRow]) {
    let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    atomicMin(&assignment[assignmentOffset + pointRow], featureRow);
  }`
      })
    );
    nodes.push(
      ...getSpatialJoinAssignNodes<Parameters>(graph, {
        id: `${id}-assign`,
        operation: OPERATION,
        assignment,
        pointFeatureIds: views.nearestFeatureIds!,
        featureIds: views.featureIds,
        featureCounts: views.featureCounts,
        bestDistanceBits,
        distances: views.nearestDistances
      })
    );
    if (matches) {
      nodes.push(
        ...getSpatialJoinCollectNodes<Parameters>(graph, {
          id: `${id}-collect-matches`,
          operation: OPERATION,
          points,
          assignment,
          state,
          sourceIds: views.sourceIds,
          matches
        })
      );
    }
    nodes.push(
      createSpatialJoinFinalizeNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        state,
        bvhOverflow: bvh.overflow,
        candidateCapacity,
        overflow: views.overflow,
        candidateCount: views.candidateCount,
        matches
      })
    );
    return nodes;
  }
}

/** Returns every read-only view of a nearest-feature join. */
function getInputs(
  props: NearestFeatureJoinViews
): (GraphDataView | GraphVectorView | undefined)[] {
  return [
    props.points,
    ...(props.queries ? getNearestGeometryViews(props.queries) : []),
    props.sourceIds,
    props.featureIds,
    props.queryIds,
    props.onAttribute?.left,
    props.onAttribute?.right,
    props.radius,
    ...getNearestGeometryViews(props.features)
  ];
}

/** Returns every writable view of a nearest-feature join. */
function getOutputs(
  props: NearestFeatureJoinViews
): (GraphDataView | GraphVectorView | undefined)[] {
  return [
    props.nearestFeatureIds,
    props.nearestDistances,
    props.featureCounts,
    props.overflow,
    props.candidateCount,
    props.antimeridianUncertainCount,
    props.poleUncertainCount,
    props.neighborIds,
    props.neighborCounts,
    props.neighborDistances,
    props.neighborFootPoints,
    props.neighborSegmentIndices,
    props.neighborQueryPoints,
    props.matches?.ids,
    props.matches?.status.count,
    props.matches?.status.overflow,
    props.matches?.status.requiredCount
  ];
}
