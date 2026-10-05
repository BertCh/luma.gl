// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_POINT_IN_POLYGON_CLASSIFICATION,
  GPUPairwisePointInPolygon
} from '../gpu-pairwise-point-in-polygon';
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
  getNextPowerOfTwo,
  getSpatialJoinAssignNodes,
  getSpatialJoinCollectNodes,
  getSpatialJoinProbeNodes,
  isPowerOfTwo,
  validateDisjointOutputs,
  validateMatchingChunks
} from './spatial-join-passes';

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
  /** Optional stable feature IDs written instead of feature rows. */
  featureIds?: GraphDataView<'uint32'>;
  /** Maximum `(point, feature)` bounding-box candidates per encoding. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
  leafCapacity?: number;
  /** Count points on a ring boundary as contained. Defaults to `true`. */
  includeBoundary?: boolean;
  /**
   * Compile-time. When true, features are reordered along a Morton (Z-order) curve of their bound
   * centers before the BVH build, so leaves that are close in space are close in the tree and
   * internal node bounds stay tight. Empty or invalid features sort last.
   *
   * Results are identical either way (assignment still keeps the smallest containing feature row); only traversal cost changes. Default false.
   *
   * Measured on an Apple M3 Pro with 14,400 small square
   * features and 250,000 points: shuffled rows drop from 218 ms to 6.7 ms per encoding (about 9,500
   * to 61 BVH nodes visited per point); row-major rows are 6.0 ms unsorted and 5.6 ms sorted.
   *
   * Enable it when features are not already spatially coherent in row order and the feature count
   * is large; leave it off for coherent data or small feature sets, where the extra sort passes
   * cost more than the traversal they save.
   */
  spatialSort?: boolean;
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
  /** Whether features are Morton sorted before the BVH build. */
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
    this.spatialSort = props.spatialSort ?? false;
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
    const {bvh, nodes: bvhNodes} = getSortedFeatureBVHNodes(
      graph,
      id,
      OPERATION,
      minima,
      maxima,
      leafCapacity,
      this.spatialSort
    );
    nodes.push(...bvhNodes);

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
