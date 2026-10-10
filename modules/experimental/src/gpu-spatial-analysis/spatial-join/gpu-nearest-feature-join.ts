// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUBVH,
  validatePackedUint32View,
  validatePackedView,
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
  captureGraphCommandNodes,
  getGraphViewChunks,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  createSpatialJoinClearNode,
  createSpatialJoinFinalizeNode,
  getDefaultSpatialSort,
  type SpatialSortCurve,
  getNextPowerOfTwo,
  getSpatialJoinAssignNodes,
  getSpatialJoinCollectNodes,
  getSpatialJoinProbeNodes,
  isPowerOfTwo,
  validateDisjointOutputs,
  validateMatchingChunks
} from './spatial-join-passes';
import type {GPUNearestFeatureSource, GPUSpatialJoinOnAttribute} from './spatial-join-types';
import type {GPUSpatialJoinPrepared} from './spatial-join-prepared';
import {
  getNearestBVHNodes,
  getNearestGeometryCount,
  getNearestGeometryViews,
  getNearestNeighborNodes
} from './nearest-nodes';
import type {
  GPUNearestFeatureGeometry,
  GPUNearestQueryGeometry,
  GPUNearestTieMode
} from './nearest-types';

const OPERATION = 'GPUNearestFeatureJoin';

/**
 * Properties for {@link GPUNearestFeatureJoin}.
 *
 * Two output modes share one class:
 *
 * - **Nearest-feature mode** (the original API): `points`, `radius`, `candidateCapacity` and
 *   `nearestFeatureIds`. One nearest point or segment feature per point within a radius.
 * - **Neighbors mode**: set `neighborIds` and `neighborCounts`. Each query (a chunked point set via
 *   `points`, or any line or polygon set via `queries`) gets its `k` nearest features from a
 *   branch-and-bound BVH traversal, in dense slots of `neighborCapacity` per query, ordered by
 *   `(distance, feature row)`. Features may be points, segments, linestrings or polygons, so this
 *   mode covers point, line/line, line/polygon and polygon/polygon minimum distance. It is
 *   `sjoin_nearest` and PostGIS `ORDER BY a <-> b LIMIT k`.
 *
 * Per-frame: `radius` or `maxDistance` and the contents of every input buffer. Topology: view
 * lengths and chunking, `k`, `neighborCapacity`, `ties`, `candidateCapacity`, `leafCapacity`,
 * `spatialSort`, and which optional views exist.
 */
export type GPUNearestFeatureJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'nearest-feature-join'`. */
  id?: string;
  /**
   * Packed planar query points, possibly chunked. In neighbors mode set either `points` or
   * `queries`.
   */
  points?: GPUFloat32Positions;
  /**
   * Neighbors mode only: line or polygon query geometry (or unchunked points), one query per
   * row. Mutually exclusive with `points`.
   */
  queries?: GPUNearestQueryGeometry;
  /** Optional stable point IDs with the chunk topology of `points`, used by `matches`. */
  sourceIds?: GPUUint32Rows;
  /**
   * Features. Nearest-feature mode accepts points and segments; neighbors mode also accepts
   * linestrings and polygons. Polygon containment gives distance 0.
   */
  features: GPUNearestFeatureGeometry;
  /** Optional stable feature IDs written instead of feature rows. */
  featureIds?: GraphDataView<'uint32'>;
  /**
   * Per-frame search radius, one float32 row. NaN, negative, or infinite matches nothing.
   * Required in nearest-feature mode unless `maxDistance` is given.
   */
  radius?: GraphDataView<'float32'>;
  /**
   * Per-frame inclusive distance limit, one float32 row; features farther than this are never
   * returned. Alias of `radius` (use one of them). In neighbors mode, omitting both is
   * unbounded. NaN, negative, or infinite matches nothing.
   */
  maxDistance?: GraphDataView<'float32'>;
  /**
   * Skip features whose ID equals the query's ID (`STRtree.query_nearest(exclusive=True)`, the
   * self-neighbor rule of a nearest self-join). A feature's ID is `featureIds[row]`, or its row
   * without `featureIds`; a query's ID is `queryIds[row]`, or its row without `queryIds`. Skipped
   * features are never counted, so `k` neighbors are the nearest *other* features. Compile-time.
   * Works in both modes.
   */
  exclusive?: boolean;
  /**
   * Optional `uint32` ID per query row (length is the query count), compared with feature IDs by
   * `exclusive`. Requires `exclusive`. Unlike `sourceIds` it is never chunked.
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
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
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
   * Nearest-feature mode: per-point nearest feature ID or row, or `GPU_SPATIAL_JOIN_NO_FEATURE`.
   * Chunked like `points`.
   */
  nearestFeatureIds?: GPUUint32Rows;
  /**
   * Neighbors mode: feature ID (or row, without `featureIds`) per `(query, slot)` at index
   * `query * neighborCapacity + slot`, ordered by distance then feature row, padded with
   * `GPU_SPATIAL_JOIN_NO_FEATURE`. Length `queryCount * neighborCapacity`.
   */
  neighborIds?: GraphDataView<'uint32'>;
  /** Neighbors mode, required with `neighborIds`: number of filled slots per query. Length `queryCount`. */
  neighborCounts?: GraphDataView<'uint32'>;
  /** Neighbors mode: planar distance per slot, `-1` in unused slots. Same layout as `neighborIds`. */
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
  /** Nearest-feature mode: optional per-point planar distance to the nearest feature, or -1. Chunked like `points`. */
  nearestDistances?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Nearest-feature mode: optional per-feature count of points whose nearest feature it is. */
  featureCounts?: GraphDataView<'uint32'>;
  /**
   * One-row flag: 1 when BVH leaf, candidate, `matches` or (neighbors mode) tie capacity
   * overflowed.
   */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row unclamped candidate count. */
  candidateCount?: GraphDataView<'uint32'>;
  /** Optional compact stable IDs of points with a feature within the radius. */
  matches?: GPUCompactOutput;
  /**
   * Prepared (static) feature index from {@link GPUSpatialJoinPrepared}: reuses its bounds and BVH
   * across encodings instead of rebuilding them, and takes `leafCapacity` and `spatialSort` from
   * the handle. The handle must be added to the graph before this join, be built over the same
   * `features` (points, lines or polygons), and be invalidated when they change. It reuses a
   * build; results are never cached.
   */
  prepared?: GPUSpatialJoinPrepared;
};

/**
 * Joins planar queries to their nearest features.
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
  /** Resolved neighbors per query (1 in nearest-feature mode). */
  readonly k: number;
  /** Resolved output slots per query (1 in nearest-feature mode). */
  readonly neighborCapacity: number;
  /** Resolved tie rule. */
  readonly ties: GPUNearestTieMode;

  constructor(props: GPUNearestFeatureJoinProps) {
    this.id = props.id ?? 'nearest-feature-join';
    this.props = props;
    const {id} = this;
    this.mode = props.neighborIds ? 'neighbors' : 'nearest-feature';
    this.featureCount = getNearestGeometryCount(props.features);
    this.k = props.k ?? 1;
    this.ties = props.ties ?? 'lowest-id';
    const {prepared} = props;
    if (prepared) {
      if (props.features.kind === 'segments' || prepared.geometry.kind !== props.features.kind) {
        throw new Error(`${id} prepared must index the same point, line or polygon features`);
      }
      if (prepared.featureCount !== this.featureCount) {
        throw new Error(`${id} prepared feature count must equal the feature count`);
      }
      if (
        (props.leafCapacity !== undefined && props.leafCapacity !== prepared.leafCapacity) ||
        (props.spatialSort !== undefined && props.spatialSort !== prepared.spatialSort)
      ) {
        throw new Error(`${id} leafCapacity and spatialSort come from the prepared handle`);
      }
    }
    this.spatialSort = prepared
      ? prepared.spatialSort
      : (props.spatialSort ?? getDefaultSpatialSort(this.featureCount));
    this.leafCapacity =
      prepared?.leafCapacity ??
      props.leafCapacity ??
      getNextPowerOfTwo(Math.max(this.featureCount, 1));
    if (!isPowerOfTwo(this.leafCapacity)) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
    const limit = props.maxDistance ?? props.radius;
    if (props.maxDistance && props.radius) {
      throw new Error(`${id} accepts radius or maxDistance, not both`);
    }
    if (limit) {
      validatePackedView(
        limit,
        ['float32'],
        `${id} ${props.maxDistance ? 'maxDistance' : 'radius'}`
      );
      if (limit.length !== 1) {
        throw new Error(`${id} radius and maxDistance must contain one float32 row`);
      }
    }
    for (const view of getNearestGeometryViews(props.features)) {
      validatePackedView(
        view,
        view.format === 'uint32' ? ['uint32'] : ['float32x2'],
        `${id} features`
      );
    }
    if (props.features.kind === 'segments') {
      if (props.features.starts.length !== props.features.ends.length) {
        throw new Error(`${id} features.starts and features.ends must have equal lengths`);
      }
    }
    for (const [name, view] of [
      ['featureIds', props.featureIds],
      ['featureCounts', props.featureCounts]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} ${name} length must equal the feature count`);
        }
      }
    }
    for (const [name, view] of [
      ['overflow', props.overflow],
      ['candidateCount', props.candidateCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    if (this.mode === 'neighbors') {
      this.neighborCapacity = props.neighborCapacity ?? this.k;
      this.candidateCapacity = 0;
      this.queryCount = this.validateNeighbors();
    } else {
      this.neighborCapacity = 1;
      this.candidateCapacity = props.candidateCapacity ?? 0;
      this.queryCount = this.validateNearestFeature();
    }
    this.validateFilters();
    validateDisjointOutputs(id, getInputs(props), getOutputs(props));
  }

  /** Validates `exclusive`, `queryIds` and `onAttribute` against the query and feature counts. */
  private validateFilters(): void {
    const {id, props, queryCount, featureCount} = this;
    if (props.queryIds && !props.exclusive) {
      throw new Error(`${id} queryIds requires exclusive`);
    }
    if (props.queryIds) {
      validatePackedUint32View(props.queryIds, `${id} queryIds`);
      if (props.queryIds.length !== queryCount) {
        throw new Error(`${id} queryIds length must equal the query count`);
      }
    }
    if (props.onAttribute) {
      const {left, right} = props.onAttribute;
      validatePackedUint32View(left, `${id} onAttribute.left`);
      validatePackedUint32View(right, `${id} onAttribute.right`);
      if (left.length !== queryCount || right.length !== featureCount) {
        throw new Error(`${id} onAttribute keys must have one row per query and feature`);
      }
    }
  }

  /** Validates the neighbors-mode props and returns the query count. */
  private validateNeighbors(): number {
    const {id, props, k, neighborCapacity} = this;
    if (!Number.isInteger(k) || k < 1 || k > 32) {
      throw new Error(`${id} k must be an integer in [1, 32]`);
    }
    if (!Number.isInteger(neighborCapacity) || neighborCapacity < k || neighborCapacity > 64) {
      throw new Error(`${id} neighborCapacity must be an integer in [k, 64]`);
    }
    if (neighborCapacity > k && this.ties !== 'all') {
      throw new Error(`${id} neighborCapacity above k needs ties: 'all'`);
    }
    if (!['lowest-id', 'all'].includes(this.ties)) {
      throw new Error(`${id} ties must be 'lowest-id' or 'all'`);
    }
    if (props.nearestFeatureIds || props.nearestDistances || props.featureCounts || props.matches) {
      throw new Error(
        `${id} nearestFeatureIds, nearestDistances, featureCounts and matches belong to nearest-feature mode`
      );
    }
    if (Boolean(props.points) === Boolean(props.queries)) {
      throw new Error(`${id} neighbors mode needs exactly one of points and queries`);
    }
    let queryCount: number;
    if (props.points) {
      for (const chunk of getGraphViewChunks(props.points)) {
        validatePackedView(chunk, ['float32x2'], `${id} points`);
      }
      queryCount = props.points.length;
    } else {
      for (const view of getNearestGeometryViews(props.queries!)) {
        validatePackedView(
          view,
          view.format === 'uint32' ? ['uint32'] : ['float32x2'],
          `${id} queries`
        );
      }
      queryCount = getNearestGeometryCount(props.queries!);
    }
    const slotCount = queryCount * neighborCapacity;
    for (const [name, view, length] of [
      ['neighborIds', props.neighborIds, slotCount],
      ['neighborCounts', props.neighborCounts, queryCount],
      ['neighborDistances', props.neighborDistances, slotCount],
      ['neighborFootPoints', props.neighborFootPoints, slotCount],
      ['neighborSegmentIndices', props.neighborSegmentIndices, slotCount],
      ['neighborQueryPoints', props.neighborQueryPoints, slotCount]
    ] as const) {
      if (name === 'neighborCounts' && !view) {
        throw new Error(`${id} neighborCounts is required with neighborIds`);
      }
      if (view) {
        validatePackedView(
          view,
          name === 'neighborDistances'
            ? ['float32']
            : name === 'neighborFootPoints' || name === 'neighborQueryPoints'
              ? ['float32x2']
              : ['uint32'],
          `${id} ${name}`
        );
        if (view.length !== length) {
          throw new Error(`${id} ${name} length must be ${length}`);
        }
      }
    }
    return queryCount;
  }

  /** Validates the nearest-feature-mode props and returns the point count. */
  private validateNearestFeature(): number {
    const {id, props} = this;
    const {features} = props;
    if (
      props.queries ||
      props.k !== undefined ||
      props.neighborCapacity ||
      props.ties ||
      props.neighborQueryPoints
    ) {
      throw new Error(
        `${id} queries, k, neighborCapacity, ties and neighborQueryPoints need neighborIds`
      );
    }
    if (!props.points || !props.nearestFeatureIds) {
      throw new Error(`${id} needs points and nearestFeatureIds, or neighborIds`);
    }
    if (features.kind !== 'points' && features.kind !== 'segments') {
      throw new Error(`${id} nearest-feature mode accepts point and segment features`);
    }
    if (!(props.maxDistance ?? props.radius)) {
      throw new Error(`${id} needs radius or maxDistance`);
    }
    for (const chunk of getGraphViewChunks(props.points)) {
      validatePackedView(chunk, ['float32x2'], `${id} points`);
    }
    for (const [name, view] of [
      ['nearestFeatureIds', props.nearestFeatureIds],
      ['sourceIds', props.sourceIds]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validatePackedUint32View(chunk, `${id} ${name}`);
      }
      validateMatchingChunks(id, name, props.points, view);
    }
    for (const chunk of props.nearestDistances ? getGraphViewChunks(props.nearestDistances) : []) {
      validatePackedView(chunk, ['float32'], `${id} nearestDistances`);
    }
    validateMatchingChunks(id, 'nearestDistances', props.points, props.nearestDistances);
    if (props.matches) {
      validateCompactOutput(id, props.matches);
    }
    if (!Number.isSafeInteger(this.candidateCapacity) || this.candidateCapacity < 1) {
      throw new Error(`${id} candidateCapacity must be a positive integer`);
    }
    return props.points.length;
  }

  /**
   * Returns the command nodes: bounds, BVH, then either the candidate/distance/reduction passes of
   * nearest-feature mode or the traversal and column passes of neighbors mode.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props} = this;
    validateGraphViewsBelongToGraph(this.id, graph, [...getInputs(props), ...getOutputs(props)]);
    if (this.mode === 'neighbors') {
      return getNearestNeighborNodes<Parameters>(graph, {
        id: this.id,
        operation: OPERATION,
        queries: props.points
          ? {kind: 'points', points: props.points}
          : {kind: 'geometry', geometry: props.queries!},
        queryCount: this.queryCount,
        features: props.features,
        featureCount: this.featureCount,
        k: this.k,
        capacity: this.neighborCapacity,
        leafCapacity: this.leafCapacity,
        spatialSort: this.spatialSort,
        spatialSortCurve: props.spatialSortCurve,
        prepared: props.prepared,
        maxDistance: props.maxDistance ?? props.radius,
        featureIds: props.featureIds,
        neighborIds: props.neighborIds!,
        neighborCounts: props.neighborCounts!,
        neighborDistances: props.neighborDistances,
        neighborFootPoints: props.neighborFootPoints,
        neighborSegmentIndices: props.neighborSegmentIndices,
        neighborQueryPoints: props.neighborQueryPoints,
        exclusive: props.exclusive,
        queryIds: props.queryIds,
        onAttribute: props.onAttribute,
        overflow: props.overflow
      }).nodes;
    }
    return this.getNearestFeatureNodes(graph);
  }

  /** Nearest-feature mode: bounds, BVH, candidate, distance, reduction, assignment, finalize. */
  private getNearestFeatureNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, leafCapacity, candidateCapacity} = this;
    const points = props.points!;
    const features = props.features as GPUNearestFeatureSource;
    const radius = (props.maxDistance ?? props.radius)!;
    const {matches} = props;
    const pointCount = points.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const {bvh: featureBVH, nodes: bvhNodes} = getNearestBVHNodes(graph, {
      id,
      operation: OPERATION,
      features,
      featureCount,
      leafCapacity,
      spatialSort: this.spatialSort,
      spatialSortCurve: props.spatialSortCurve,
      prepared: props.prepared
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
        featureCounts: props.featureCounts,
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
        radius
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
      ...captureGraphCommandNodes(graph, () =>
        new GPUPairwisePointSegmentDistance({
          id: `${id}-distance`,
          points: candidatePoints,
          segmentStarts: candidateStarts,
          segmentEnds: candidateEnds,
          output: candidateDistances
        }).addToGraph(graph)
      ).map(node => {
        if (node.type !== 'compute') {
          throw new Error(`${id}-distance expected a compute node`);
        }
        return {...node, condition: candidateDispatch.condition};
      })
    );
    // Candidates failing `exclusive` or `onAttribute` never compete for the nearest feature.
    const filterBindings: WGSLKernelBinding[] = [];
    const filterTests: string[] = [];
    if (props.exclusive) {
      if (props.queryIds) {
        filterBindings.push({name: 'queryIds', view: props.queryIds, type: 'u32', access: 'read'});
      }
      if (props.featureIds) {
        filterBindings.push({
          name: 'featureIds',
          view: props.featureIds,
          type: 'u32',
          access: 'read'
        });
      }
      const queryId = props.queryIds ? 'queryIds[queryIdsOffset + pointRow]' : 'pointRow';
      const featureId = props.featureIds
        ? 'featureIds[featureIdsOffset + candidateFeature]'
        : 'candidateFeature';
      filterTests.push(`if (${queryId} == ${featureId}) { return; }`);
    }
    if (props.onAttribute) {
      filterBindings.push(
        {name: 'queryKeys', view: props.onAttribute.left, type: 'u32', access: 'read'},
        {name: 'featureKeys', view: props.onAttribute.right, type: 'u32', access: 'read'}
      );
      filterTests.push(
        'if (queryKeys[queryKeysOffset + pointRow] != featureKeys[featureKeysOffset + candidateFeature]) { return; }'
      );
    }
    const filterTest = filterTests.length
      ? `let candidateFeature = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  ${filterTests.join('\n  ')}`
      : '';
    const reducePrologue = `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * 2u];
  let distance = candidateDistances[candidateDistancesOffset + index];
  let searchRadius = radius[radiusOffset];
  if (!(distance >= 0.0 && distance <= searchRadius)) { return; }
  ${filterTest}
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
          },
          ...filterBindings
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
          {name: 'assignment', view: assignment, type: 'atomic<u32>', access: 'read_write'},
          ...filterBindings
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
        pointFeatureIds: props.nearestFeatureIds!,
        featureIds: props.featureIds,
        featureCounts: props.featureCounts,
        bestDistanceBits,
        distances: props.nearestDistances
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
          sourceIds: props.sourceIds,
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
        overflow: props.overflow,
        candidateCount: props.candidateCount,
        matches
      })
    );
    return nodes;
  }
}

/** Returns every read-only view of a nearest-feature join. */
function getInputs(
  props: GPUNearestFeatureJoinProps
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
    props.maxDistance,
    ...getNearestGeometryViews(props.features)
  ];
}

/** Returns every writable view of a nearest-feature join. */
function getOutputs(
  props: GPUNearestFeatureJoinProps
): (GraphDataView | GraphVectorView | undefined)[] {
  return [
    props.nearestFeatureIds,
    props.nearestDistances,
    props.featureCounts,
    props.overflow,
    props.candidateCount,
    props.neighborIds,
    props.neighborCounts,
    props.neighborDistances,
    props.neighborFootPoints,
    props.neighborSegmentIndices,
    props.neighborQueryPoints,
    props.matches?.ids,
    props.matches?.count,
    props.matches?.overflow,
    props.matches?.requiredCount
  ];
}
