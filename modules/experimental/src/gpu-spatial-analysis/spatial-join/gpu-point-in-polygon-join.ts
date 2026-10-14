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
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPU_POINT_IN_POLYGON_CLASSIFICATION} from '../../geospatial/gpu-pairwise-point-in-polygon';
import {EXACT_ORIENTATION_WGSL} from '../segment-intersection/exact-orientation-wgsl';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../contracts/spatial-context';
import {createWGSLActiveCountDispatch, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {
  GPUCompactOutput,
  GPUFloat32Positions,
  GPUUint32Rows
} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getGraphViewChunks,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  createSpatialJoinBoundsNode,
  createSpatialJoinClearNode,
  createSpatialJoinFinalizeNode,
  getSortedFeatureBVHNodes,
  type SpatialSortCurve,
  getDefaultSpatialSort,
  getNextPowerOfTwo,
  getSpatialJoinAssignNodes,
  getSpatialJoinCollectNodes,
  getSpatialJoinProbeNodes,
  isPowerOfTwo,
  validateDisjointOutputs,
  validateMatchingChunks
} from './spatial-join-passes';
import {isSameSpatialJoinGeometry} from './spatial-join-geometry';
import type {GPUSpatialJoinPrepared} from './spatial-join-prepared';

const OPERATION = 'GPUPointInPolygonJoin';

/**
 * The spatial context to give a `GPUPointInPolygonJoin` for an operation resolved with `context`:
 * a longitude/latitude context becomes containment-only (`metric: 'none'`, polygon edges straight
 * in degrees); any other context is passed through unchanged.
 */
export function getPointInPolygonContainmentContext(context: GPUSpatialContext): GPUSpatialContext {
  return context.coordinateSpace === 'longitude-latitude'
    ? {coordinateSpace: 'longitude-latitude', metric: 'none'}
    : context;
}

/** Planar coordinates, or longitude/latitude with no metric (topological containment). */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metrics: Object.freeze(['none', 'native'] as const)
});

/**
 * How {@link GPUPointInPolygonJoin} treats a point exactly on a polygon boundary: `'half-open'`
 * (each boundary point goes to exactly one feature of a valid coverage), `'closed'` (boundary counts
 * as contained, the smallest row wins) or `'open'` (boundary belongs to no feature).
 */
export type GPUPointInPolygonBoundaryRule = 'half-open' | 'closed' | 'open';

/** Bit of a classification word set when a boundary point is owned under the half-open rule. */
const HALF_OPEN_OWNER_BIT = 0x100;

/**
 * Properties for {@link GPUPointInPolygonJoin}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths and chunking,
 * `candidateCapacity`, `leafCapacity`, `includeBoundary`, `spatialSort`, and which optional views exist.
 */
export type GPUPointInPolygonJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'point-in-polygon-join'`. */
  id?: string;
  /** Packed planar points. */
  points: GPUFloat32Positions;
  /** Optional stable point IDs with the chunk topology of `points`, used by `matches`. */
  sourceIds?: GPUUint32Rows;
  /** Flattened polygon vertices (GeoArrow layout, as `GPUPairwisePointInPolygon`). */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets with `featureCount + 1` entries, first 0. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry. Ring 0 is the shell; others are holes. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly. */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Optional prepared (static) polygon set over the same `polygonPositions`, `featureOffsets`,
   * `polygonOffsets` and `ringOffsets` views. Its bounds and BVH are reused across encodings until
   * the handle is invalidated, so animated points query a fixed polygon tree without rebuilding
   * it. The handle must be added to the graph before the join. Its `leafCapacity` and
   * `spatialSort` replace this join's. Reuses a build only; assignments are recomputed every encoding.
   */
  prepared?: GPUSpatialJoinPrepared;
  /** Optional stable feature IDs written instead of feature rows. */
  featureIds?: GraphDataView<'uint32'>;
  /** Maximum `(point, feature)` bounding-box candidates per encoding. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
  leafCapacity?: number;
  /**
   * Compile-time. How a point exactly on a polygon boundary is treated. The decision uses the exact
   * orientation predicate, so "exactly on" is exact for the f32 vertices.
   *
   * - `'half-open'` (default): the boundary is split by the crossing-number convention of a
   *   half-open region, so a point on a boundary shared by several features of a valid coverage
   *   belongs to exactly one of them, whatever the feature rows are. The owner is the feature that
   *   contains the point after an infinitesimal move toward +x and +y (+y dominates), which means
   *   an edge or vertex belongs to the feature whose interior is to its right/above: left and
   *   bottom boundaries are included, right and top boundaries are excluded. A horizontal edge
   *   belongs to the feature above it, a vertical edge to the feature on its right, a vertex to the
   *   feature whose interior touches the upper right of the vertex. Points on the outer right or
   *   top boundary of a coverage belong to no feature (they are not lost by a bug: the rule that
   *   makes interior boundaries unique makes those open).
   * - `'closed'`: every boundary point counts as contained by every feature whose boundary it
   *   touches, then the smallest feature row wins (the assignment keeps one feature per point).
   *   Nothing is lost on the outer boundary, but the owner of a shared-boundary point depends on
   *   the feature row order. This was the behavior before `boundaryRule` existed.
   * - `'open'`: boundary points belong to no feature (only strict interior counts).
   *
   * A point strictly inside is unaffected by the rule. Overlapping features (an invalid coverage)
   * still resolve to the smallest containing row. `matches` lists exactly the points that get a
   * feature under the same rule.
   */
  boundaryRule?: GPUPointInPolygonBoundaryRule;
  /**
   * Compile-time shorthand kept for existing callers: `true` is `boundaryRule: 'closed'` and `false`
   * is `boundaryRule: 'open'`. It is not a switch for the half-open rule: with neither prop set the
   * join uses `'half-open'`. Passing it together with a `boundaryRule` that disagrees throws.
   */
  includeBoundary?: boolean;
  /**
   * Coordinate space and metric of the points and polygons. Omitted means native planar
   * coordinates. Containment is a topological test evaluated in the plane of the given
   * coordinates: polygon edges are straight lines in the coordinates (as in GeoJSON), not
   * geodesics. `{coordinateSpace: 'longitude-latitude', metric: 'none'}` is therefore accepted and
   * behaves exactly like planar. A longitude/latitude context with any metric (for example
   * `'great-circle'`) throws at construction because there are no geodesic edges; project to a
   * planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Compile-time. When true, features are reordered along a Hilbert curve (see `spatialSortCurve`) of their bound
   * centers before the BVH build, so leaves that are close in space are close in the tree and
   * internal node bounds stay tight. Empty or invalid features sort last.
   *
   * Results are identical either way (assignment still keeps the smallest containing feature row); only traversal cost changes. Default false.
   *
   * Measured on an Apple M3 Pro with 14,400 small square
   * features and 250,000 points: shuffled rows drop from 218 ms to 6.7 ms per encoding (about 9,500
   * to 61 BVH nodes visited per point); row-major rows are 6.0 ms unsorted and 5.6 ms sorted.
   *
   * Defaults to on from 256 features and off below, where the BVH is too shallow for the order to
   * matter. On coherent data the sort costs about 1 ms; pass `false` to skip it. Candidate order is
   * then unspecified (the point-in-polygon result is not affected).
   */
  spatialSort?: boolean;
  /**
   * Experimental, compile-time. Curve used by `spatialSort`: `'hilbert'` (default; 10 to 15% faster joins
   * than Morton in paired A/B runs) or `'morton'` (Z-order). Results are identical; only BVH
   * traversal cost changes.
   */
  spatialSortCurve?: SpatialSortCurve;
  /** Per-point containing feature ID or row, or `GPU_SPATIAL_JOIN_NO_FEATURE`. Chunked like `points`. */
  pointFeatureIds: GPUUint32Rows;
  /** Optional per-feature count of assigned points. */
  featureCounts?: GraphDataView<'uint32'>;
  /** One-row flag: 1 when BVH leaf, candidate, or `matches` capacity overflowed. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row unclamped candidate count, for sizing `candidateCapacity`. */
  candidateCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row count of refined (point, feature) candidates classified `uncertain`. Uncertain
   * pairs are never assigned: a point whose only candidates were uncertain gets no feature, so a
   * nonzero value means the assignments and `featureCounts` may be incomplete.
   */
  uncertainCount?: GraphDataView<'uint32'>;
  /** Optional compact stable IDs of points that joined a feature (unspecified order). */
  matches?: GPUCompactOutput;
};

/**
 * Joins planar points to the polygon features that contain them.
 *
 * A `GPUBVH` over feature bounds produces bounding-box candidates, an exact even/odd classifier
 * (exact orientation signs, see `boundaryRule`) refines them, and each point keeps the containing
 * feature with the smallest row.
 *
 * **Boundary points.** By default (`boundaryRule: 'half-open'`) a point on an edge or vertex shared
 * by several features of a valid coverage is assigned to exactly one of them, independent of the
 * feature row order, so counts over a coverage partition the points (only points on the outer right
 * and top boundary are left out). `'closed'` and `'open'` select the previous behaviors; see
 * {@link GPUPointInPolygonJoinProps.boundaryRule}. Zonal statistics rely on the half-open default. Per-feature counts and an optional compact list of joined points are produced on
 * the GPU. Every capacity reports overflow on the GPU.
 *
 * Candidates whose containment cannot be proven (`uncertain`) are not assigned and are not
 * silently ignored: they are counted in `uncertainCount`. Read it to know whether assignments may
 * be incomplete.
 */
export class GPUPointInPolygonJoin implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointInPolygonJoinProps;
  /** Number of features, `featureOffsets.length - 1`. */
  readonly featureCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Candidate pair capacity. */
  readonly candidateCapacity: number;
  /** Resolved boundary rule. */
  readonly boundaryRule: GPUPointInPolygonBoundaryRule;
  /** Whether boundary points can count as contained (`boundaryRule` is not `'open'`). */
  readonly includeBoundary: boolean;
  /** Resolved spatial context: native planar coordinates. */
  readonly spatialContext: GPUSpatialContext;
  /** Whether features are spatially sorted before the BVH build. */
  readonly spatialSort: boolean;

  constructor(props: GPUPointInPolygonJoinProps) {
    this.id = props.id ?? 'point-in-polygon-join';
    this.props = props;
    const {id} = this;
    for (const chunk of getGraphViewChunks(props.points)) {
      validatePackedView(chunk, ['float32x2'], `${id} points`);
    }
    validatePackedView(props.polygonPositions, ['float32x2'], `${id} polygonPositions`);
    for (const [name, view] of [
      ['featureOffsets', props.featureOffsets],
      ['polygonOffsets', props.polygonOffsets],
      ['ringOffsets', props.ringOffsets]
    ] as const) {
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name} requires a terminal entry`);
      }
    }
    this.featureCount = props.featureOffsets.length - 1;
    for (const [name, view] of [
      ['pointFeatureIds', props.pointFeatureIds],
      ['sourceIds', props.sourceIds]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validatePackedUint32View(chunk, `${id} ${name}`);
      }
      validateMatchingChunks(id, name, props.points, view);
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
      ['candidateCount', props.candidateCount],
      ['uncertainCount', props.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    if (props.matches) {
      validateCompactOutput(id, props.matches);
    }
    this.candidateCapacity = props.candidateCapacity;
    if (
      !Number.isSafeInteger(this.candidateCapacity) ||
      this.candidateCapacity < 1 ||
      2 * this.candidateCapacity + 2 > 0xffffffff
    ) {
      throw new Error(`${id} candidateCapacity must be a positive integer`);
    }
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.featureCount, 1));
    if (!isPowerOfTwo(this.leafCapacity)) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
    if (
      props.spatialContext?.coordinateSpace === 'longitude-latitude' &&
      props.spatialContext.metric !== 'none'
    ) {
      throw new Error(
        `${id} does not support the ${props.spatialContext.metric} metric on longitude-latitude coordinates; ` +
          `use metric 'none' for containment or project positions to a planar frame first`
      );
    }
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
    const legacyRule: GPUPointInPolygonBoundaryRule | undefined =
      props.includeBoundary === undefined ? undefined : props.includeBoundary ? 'closed' : 'open';
    if (
      props.boundaryRule !== undefined &&
      !['half-open', 'closed', 'open'].includes(props.boundaryRule)
    ) {
      throw new Error(`${id} boundaryRule must be 'half-open', 'closed' or 'open'`);
    }
    if (legacyRule && props.boundaryRule && legacyRule !== props.boundaryRule) {
      throw new Error(`${id} includeBoundary ${props.includeBoundary} conflicts with boundaryRule`);
    }
    this.boundaryRule = props.boundaryRule ?? legacyRule ?? 'half-open';
    this.includeBoundary = this.boundaryRule !== 'open';
    this.spatialSort =
      props.prepared?.spatialSort ?? props.spatialSort ?? getDefaultSpatialSort(this.featureCount);
    if (props.prepared) {
      this.leafCapacity = props.prepared.leafCapacity;
      if (
        !isSameSpatialJoinGeometry(props.prepared.geometry, {
          kind: 'polygons',
          positions: props.polygonPositions,
          featureOffsets: props.featureOffsets,
          polygonOffsets: props.polygonOffsets,
          ringOffsets: props.ringOffsets
        })
      ) {
        throw new Error(`${id} prepared must index the same polygon views`);
      }
    }
    validateDisjointOutputs(
      id,
      [
        props.points,
        props.sourceIds,
        props.polygonPositions,
        props.featureOffsets,
        props.polygonOffsets,
        props.ringOffsets,
        props.featureIds
      ],
      [
        props.pointFeatureIds,
        props.featureCounts,
        props.overflow,
        props.candidateCount,
        props.uncertainCount,
        props.matches?.ids,
        props.matches?.count,
        props.matches?.overflow,
        props.matches?.requiredCount
      ]
    );
  }

  /** Returns bounds, BVH, candidate, refinement, assignment, and finalize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, leafCapacity, candidateCapacity} = this;
    const {points, matches} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      points,
      props.sourceIds,
      props.polygonPositions,
      props.featureOffsets,
      props.polygonOffsets,
      props.ringOffsets,
      props.featureIds,
      props.pointFeatureIds,
      props.featureCounts,
      props.overflow,
      props.candidateCount,
      props.uncertainCount,
      matches?.ids,
      matches?.count,
      matches?.overflow,
      matches?.requiredCount
    ]);
    const pointCount = points.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    if (props.prepared && !props.prepared.isDeclaredIn(graph)) {
      throw new Error(`${id} requires prepared to be added to the graph first`);
    }
    let bvh: Pick<
      GPUBVH,
      'nodeMinima' | 'nodeMaxima' | 'leafIds' | 'overflow' | 'internalNodeCount'
    >;
    if (props.prepared) {
      const {storage} = props.prepared;
      bvh = {...storage, internalNodeCount: props.prepared.bvhInternalNodeCount};
    } else {
      const minima = createTransientView(graph, `${id}-feature-minima`, 'float32x2', featureCount);
      const maxima = createTransientView(graph, `${id}-feature-maxima`, 'float32x2', featureCount);
      if (featureCount > 0) {
        nodes.push(
          createSpatialJoinBoundsNode<Parameters>(graph, {
            id: `${id}-bounds`,
            operation: OPERATION,
            featureCount,
            source: {
              kind: 'polygons',
              polygonPositions: props.polygonPositions,
              featureOffsets: props.featureOffsets,
              polygonOffsets: props.polygonOffsets,
              ringOffsets: props.ringOffsets
            },
            minima,
            maxima
          })
        );
      }
      const built = getSortedFeatureBVHNodes(
        graph,
        id,
        OPERATION,
        minima,
        maxima,
        leafCapacity,
        this.spatialSort,
        undefined,
        this.props.spatialSortCurve
      );
      nodes.push(...built.nodes);
      bvh = built.bvh;
    }

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const assignment = createTransientView(graph, `${id}-assignment`, 'uint32', pointCount);
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
    const pairClassifications = createTransientView(
      graph,
      `${id}-pair-classifications`,
      'uint32',
      candidateCapacity
    );
    nodes.push(
      createSpatialJoinClearNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        state,
        assignment,
        pointCount,
        featureCounts: props.featureCounts
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
        pointRowOffset: 0
      })
    );
    const candidateDispatch = createWGSLActiveCountDispatch(graph, {
      id: `${id}-candidate-dispatch`,
      operation: OPERATION,
      count: state,
      maximumItemCount: candidateCapacity
    });
    nodes.push(candidateDispatch.updateNode);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: 'classify-exact',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'candidatePoints', view: candidatePoints, type: 'f32', access: 'read'},
          {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
          {name: 'polygonPositions', view: props.polygonPositions, type: 'f32', access: 'read'},
          {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {
            name: 'pairClassifications',
            view: pairClassifications,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const POLYGON_COUNT: u32 = ${props.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const VERTEX_COUNT: u32 = ${props.polygonPositions.length}u;
const OUTSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.outside}u;
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${HALF_OPEN_OWNER_BIT}u;
${EXACT_ORIENTATION_WGSL}
fn readVertex(vertexIndex: u32) -> vec2f {
  return vec2f(
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
  );
}
// Exact even/odd classification, or UNCERTAIN when some orientation has no sign. A BOUNDARY result
// also carries HALF_OPEN_OWNER_BIT when the point belongs to the feature under the half-open rule.
//
// Half-open rule: classify the point moved by (epsilon^2, epsilon) with epsilon infinitesimal. The
// crossing test against an edge only changes when the point is on the edge's supporting line, where
// the sign of the moved point is the sign of dx (the epsilon term), or of -dy for a vertical edge
// (the epsilon^2 term). The y comparisons are unchanged because a vertex y is either equal to the
// point's y or at least one ulp away.
fn classifyExactly(point: vec2f, geometryStart: u32, geometryEnd: u32) -> u32 {
  var geometryInside = false;
  var geometryOwned = false;
  var onBoundary = false;
  for (var polygonIndex = geometryStart; polygonIndex < geometryEnd; polygonIndex++) {
    let polygonStart = polygonOffsets[polygonOffsetsOffset + polygonIndex];
    let polygonEnd = polygonOffsets[polygonOffsetsOffset + polygonIndex + 1u];
    // Malformed offsets cannot be classified; an empty part contains nothing (as in shapely) and
    // must not hide the other parts of the feature.
    if (polygonStart > polygonEnd || polygonEnd > RING_COUNT) { return UNCERTAIN; }
    var polygonInside = false;
    var polygonOwned = false;
    for (var ringIndex = polygonStart; ringIndex < polygonEnd; ringIndex++) {
      let ringStart = ringOffsets[ringOffsetsOffset + ringIndex];
      let ringEnd = ringOffsets[ringOffsetsOffset + ringIndex + 1u];
      if (ringEnd < ringStart || ringEnd > VERTEX_COUNT) { return UNCERTAIN; }
      // A ring with fewer than 3 vertices bounds no area: skip it, as shapely treats empty parts.
      if (ringEnd < ringStart + 3u) { continue; }
      var previous = readVertex(ringEnd - 1u);
      for (var vertexIndex = ringStart; vertexIndex < ringEnd; vertexIndex++) {
        let current = readVertex(vertexIndex);
        let sign = orientSign(previous, current, point);
        if (sign == 2) { return UNCERTAIN; }
        var side = sign;
        if (sign == 0) {
          if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
              point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
            onBoundary = true;
          }
          side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
          if (side == 0) {
            side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
          }
        }
        if ((previous.y > point.y) != (current.y > point.y)) {
          if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
          if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
          if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
            polygonOwned = !polygonOwned;
          }
        }
        previous = current;
      }
    }
    geometryInside = geometryInside || polygonInside;
    geometryOwned = geometryOwned || polygonOwned;
  }
  if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
  return select(OUTSIDE, INSIDE, geometryInside);
}`,
        body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let geometryStart = featureOffsets[featureOffsetsOffset + featureRow];
  let geometryEnd = featureOffsets[featureOffsetsOffset + featureRow + 1u];
  var classification = UNCERTAIN;
  if (geometryStart <= geometryEnd && geometryEnd <= POLYGON_COUNT) {
    let point = vec2f(candidatePoints[candidatePointsOffset + index * 2u], candidatePoints[candidatePointsOffset + index * 2u + 1u]);
    classification = classifyExactly(point, geometryStart, geometryEnd);
  }
  pairClassifications[pairClassificationsOffset + index] = classification;`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-resolve`,
        operation: OPERATION,
        variant: 'resolve',
        bindings: [
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'pairClassifications', view: pairClassifications, type: 'u32', access: 'read'},
          {name: 'assignment', view: assignment, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${HALF_OPEN_OWNER_BIT}u;
const BOUNDARY_CLOSED: bool = ${this.boundaryRule === 'closed'};
const BOUNDARY_HALF_OPEN: bool = ${this.boundaryRule === 'half-open'};`,
        body: `let activeCount = min(atomicLoad(&state[stateOffset]), CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * 2u];
  let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let word = pairClassifications[pairClassificationsOffset + index];
  let classification = word & ~HALF_OPEN_OWNER_BIT;
  let owned = BOUNDARY_CLOSED || (BOUNDARY_HALF_OPEN && (word & HALF_OPEN_OWNER_BIT) != 0u);
  if (classification == INSIDE || (owned && classification == BOUNDARY)) {
    atomicMin(&assignment[assignmentOffset + pointRow], featureRow);
  } else if (classification == UNCERTAIN) {
    atomicAdd(&state[stateOffset + 1u], 1u);
  }`
      })
    );
    nodes.push(
      ...getSpatialJoinAssignNodes<Parameters>(graph, {
        id: `${id}-assign`,
        operation: OPERATION,
        assignment,
        pointFeatureIds: props.pointFeatureIds,
        featureIds: props.featureIds,
        featureCounts: props.featureCounts
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
        uncertainCount: props.uncertainCount,
        matches
      })
    );
    return nodes;
  }
}
