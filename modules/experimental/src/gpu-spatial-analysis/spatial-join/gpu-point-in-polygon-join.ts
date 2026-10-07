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
import {
  GPU_POINT_IN_POLYGON_CLASSIFICATION,
  GPUPairwisePointInPolygon
} from '../../geospatial/gpu-pairwise-point-in-polygon';
import {EXACT_ORIENTATION_WGSL} from '../segment-intersection/exact-orientation-wgsl';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
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
  /** Count points on a ring boundary as contained. Defaults to `true`. */
  includeBoundary?: boolean;
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
 * A `GPUBVH` over feature bounds produces bounding-box candidates, `GPUPairwisePointInPolygon`
 * refines them with robust predicates, and each point keeps the containing feature with the
 * smallest row. Per-feature counts and an optional compact list of joined points are produced on
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
  /** Whether boundary points count as contained. */
  readonly includeBoundary: boolean;
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
    this.includeBoundary = props.includeBoundary ?? true;
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
        props.matches?.totalCount
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
      matches?.totalCount
    ]);
    const pointCount = points.length;
    const pairRowCount = 2 * candidateCapacity + 1;
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
    const pairPoints = createTransientView(graph, `${id}-pair-points`, 'float32x2', pairRowCount);
    const pairGeometryOffsets = createTransientView(
      graph,
      `${id}-pair-geometry-offsets`,
      'uint32',
      pairRowCount + 1
    );
    const pairClassifications = createTransientView(
      graph,
      `${id}-pair-classifications`,
      'uint32',
      pairRowCount
    );
    nodes.push(
      createSpatialJoinClearNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        state,
        assignment,
        pointCount,
        featureCounts: props.featureCounts,
        pairs: {
          points: pairPoints,
          geometryOffsets: pairGeometryOffsets,
          polygonCount: props.polygonOffsets.length - 1
        }
      })
    );
    // Candidate slot `s` uses pair row `2s + 1`; the NaN rows between slots let each candidate own
    // its own geometry offsets pair in GPUPairwisePointInPolygon.
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
        candidatePoints: pairPoints,
        pointRowScale: 2,
        pointRowOffset: 1
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-expand`,
        operation: OPERATION,
        variant: 'expand',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
          {
            name: 'pairGeometryOffsets',
            view: pairGeometryOffsets,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: candidateCapacity,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  pairGeometryOffsets[pairGeometryOffsetsOffset + index * 2u + 1u] = featureOffsets[featureOffsetsOffset + featureRow];
  pairGeometryOffsets[pairGeometryOffsetsOffset + index * 2u + 2u] = featureOffsets[featureOffsetsOffset + featureRow + 1u];`
      })
    );
    nodes.push(
      ...captureGraphCommandNodes(graph, () =>
        new GPUPairwisePointInPolygon({
          id: `${id}-classify`,
          points: pairPoints,
          polygonPositions: props.polygonPositions,
          geometryOffsets: pairGeometryOffsets,
          polygonOffsets: props.polygonOffsets,
          ringOffsets: props.ringOffsets,
          output: pairClassifications
        }).addToGraph(graph)
      )
    );
    // The double-single classifier answers `uncertain` whenever a determinant is within 2^-20 of
    // its product magnitudes. Re-decide those candidates with the exact orientation predicate, so
    // `uncertain` is left only for non-finite input, malformed offsets and degenerate rings.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-exact`,
        operation: OPERATION,
        variant: 'exact',
        bindings: [
          {name: 'pairPoints', view: pairPoints, type: 'f32', access: 'read'},
          {name: 'pairGeometryOffsets', view: pairGeometryOffsets, type: 'u32', access: 'read'},
          {name: 'polygonPositions', view: props.polygonPositions, type: 'f32', access: 'read'},
          {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {
            name: 'pairClassifications',
            view: pairClassifications,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: candidateCapacity,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const POLYGON_COUNT: u32 = ${props.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const OUTSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.outside}u;
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
${EXACT_ORIENTATION_WGSL}
fn readVertex(vertexIndex: u32) -> vec2f {
  return vec2f(
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
  );
}
// Exact even/odd classification, or UNCERTAIN when some orientation has no sign.
fn classifyExactly(point: vec2f, geometryStart: u32, geometryEnd: u32) -> u32 {
  var geometryInside = false;
  for (var polygonIndex = geometryStart; polygonIndex < geometryEnd; polygonIndex++) {
    let polygonStart = polygonOffsets[polygonOffsetsOffset + polygonIndex];
    let polygonEnd = polygonOffsets[polygonOffsetsOffset + polygonIndex + 1u];
    if (polygonStart >= polygonEnd || polygonEnd > RING_COUNT) { return UNCERTAIN; }
    var polygonInside = false;
    for (var ringIndex = polygonStart; ringIndex < polygonEnd; ringIndex++) {
      let ringStart = ringOffsets[ringOffsetsOffset + ringIndex];
      let ringEnd = ringOffsets[ringOffsetsOffset + ringIndex + 1u];
      if (ringEnd < ringStart + 3u) { return UNCERTAIN; }
      var previous = readVertex(ringEnd - 1u);
      for (var vertexIndex = ringStart; vertexIndex < ringEnd; vertexIndex++) {
        let current = readVertex(vertexIndex);
        let sign = orientSign(previous, current, point);
        if (sign == 2) { return UNCERTAIN; }
        if (sign == 0 &&
            point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
            point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
          return BOUNDARY;
        }
        if ((previous.y > point.y) != (current.y > point.y)) {
          if ((current.y > previous.y && sign > 0) || (current.y < previous.y && sign < 0)) {
            polygonInside = !polygonInside;
          }
        }
        previous = current;
      }
    }
    geometryInside = geometryInside || polygonInside;
  }
  return select(OUTSIDE, INSIDE, geometryInside);
}`,
        body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let row = index * 2u + 1u;
  if (pairClassifications[pairClassificationsOffset + row] != UNCERTAIN) { return; }
  let geometryStart = pairGeometryOffsets[pairGeometryOffsetsOffset + row];
  let geometryEnd = pairGeometryOffsets[pairGeometryOffsetsOffset + row + 1u];
  if (geometryStart > geometryEnd || geometryEnd > POLYGON_COUNT) { return; }
  let point = vec2f(pairPoints[pairPointsOffset + row * 2u], pairPoints[pairPointsOffset + row * 2u + 1u]);
  pairClassifications[pairClassificationsOffset + row] = classifyExactly(point, geometryStart, geometryEnd);`
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
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
const INCLUDE_BOUNDARY: bool = ${this.includeBoundary};`,
        body: `let activeCount = min(atomicLoad(&state[stateOffset]), CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * 2u];
  let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
  if (classification == INSIDE || (INCLUDE_BOUNDARY && classification == BOUNDARY)) {
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
