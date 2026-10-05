// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUPairwisePointSegmentDistance} from '../../geospatial/index';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {
  GPUMapGraphCompactOutput,
  GPUMapGraphPositions2D,
  GPUMapGraphRecipe,
  GPUMapGraphUint32Rows
} from '../map-graph-types';
import {
  captureGraphCommandNodes,
  getGraphViewChunks,
  validateGraphViewsBelongToGraph,
  validateMapGraphCompactOutput
} from '../map-graph-utils';
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
import type {GPUNearestFeatureSource} from './spatial-join-types';

const OPERATION = 'GPUNearestFeatureJoin';

/**
 * Properties for {@link GPUNearestFeatureJoin}.
 *
 * Per-frame: `radius` and the contents of every input buffer. Topology: view lengths and
 * chunking, `candidateCapacity`, `leafCapacity`, `spatialSort`, and which optional views exist.
 */
export type GPUNearestFeatureJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'nearest-feature-join'`. */
  id?: string;
  /** Packed planar points. */
  points: GPUMapGraphPositions2D;
  /** Optional stable point IDs with the chunk topology of `points`, used by `matches`. */
  sourceIds?: GPUMapGraphUint32Rows;
  /** Point or segment features. */
  features: GPUNearestFeatureSource;
  /** Optional stable feature IDs written instead of feature rows. */
  featureIds?: GraphDataView<'uint32'>;
  /** Per-frame search radius, one float32 row. NaN, negative, or infinite matches nothing. */
  radius: GraphDataView<'float32'>;
  /** Maximum `(point, feature)` bounding-box candidates per encoding. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
  leafCapacity?: number;
  /**
   * Compile-time. When true, features are reordered along a Morton (Z-order) curve of their bound
   * centers before the BVH build, so leaves that are close in space are close in the tree and
   * internal node bounds stay tight. Empty or invalid features sort last.
   *
   * Results are identical either way (the nearest feature wins and ties keep the smallest feature row); only traversal cost changes. Default false.
   *
   * Measured on an Apple M3 Pro with 50,000 shuffled short
   * segments, 100,000 points and radius 1: 309 ms drops to 4.1 ms per encoding (about 33,000 to 76
   * BVH nodes visited per point). Candidate counts are identical.
   *
   * Enable it when features are not already spatially coherent in row order and the feature count
   * is large; leave it off for coherent data or small feature sets, where the extra sort passes
   * cost more than the traversal they save.
   */
  spatialSort?: boolean;
  /** Per-point nearest feature ID or row, or `GPU_SPATIAL_JOIN_NO_FEATURE`. Chunked like `points`. */
  nearestFeatureIds: GPUMapGraphUint32Rows;
  /** Optional per-point planar distance to the nearest feature, or -1. Chunked like `points`. */
  nearestDistances?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional per-feature count of points whose nearest feature it is. */
  featureCounts?: GraphDataView<'uint32'>;
  /** One-row flag: 1 when BVH leaf, candidate, or `matches` capacity overflowed. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row unclamped candidate count. */
  candidateCount?: GraphDataView<'uint32'>;
  /** Optional compact stable IDs of points with a feature within the radius. */
  matches?: GPUMapGraphCompactOutput;
};

/**
 * Joins each planar point to its nearest point or segment feature within a per-frame radius.
 *
 * A `GPUBVH` over feature bounds produces candidates inside the radius box,
 * `GPUPairwisePointSegmentDistance` measures them, and deterministic `atomicMin` passes keep the
 * nearest feature (ties go to the smallest feature row).
 */
export class GPUNearestFeatureJoin implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'nearest-feature-join';
  /** Validated properties. */
  readonly props: GPUNearestFeatureJoinProps;
  /** Number of features. */
  readonly featureCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Candidate pair capacity. */
  readonly candidateCapacity: number;
  /** Whether features are Morton sorted before the BVH build. */
  readonly spatialSort: boolean;

  constructor(props: GPUNearestFeatureJoinProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const {features} = props;
    for (const chunk of getGraphViewChunks(props.points)) {
      validatePackedView(chunk, ['float32x2'], `${id} points`);
    }
    if (features.kind === 'points') {
      validatePackedView(features.positions, ['float32x2'], `${id} features.positions`);
      this.featureCount = features.positions.length;
    } else {
      validatePackedView(features.starts, ['float32x2'], `${id} features.starts`);
      validatePackedView(features.ends, ['float32x2'], `${id} features.ends`);
      if (features.starts.length !== features.ends.length) {
        throw new Error(`${id} features.starts and features.ends must have equal lengths`);
      }
      this.featureCount = features.starts.length;
    }
    validatePackedView(props.radius, ['float32'], `${id} radius`);
    if (props.radius.length !== 1) {
      throw new Error(`${id} radius must contain one float32 row`);
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
    if (props.matches) {
      validateMapGraphCompactOutput(id, props.matches);
    }
    this.candidateCapacity = props.candidateCapacity;
    if (!Number.isSafeInteger(this.candidateCapacity) || this.candidateCapacity < 1) {
      throw new Error(`${id} candidateCapacity must be a positive integer`);
    }
    this.spatialSort = props.spatialSort ?? false;
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.featureCount, 1));
    if (!isPowerOfTwo(this.leafCapacity)) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
    validateDisjointOutputs(id, getInputs(props), [
      props.nearestFeatureIds,
      props.nearestDistances,
      props.featureCounts,
      props.overflow,
      props.candidateCount,
      props.matches?.ids,
      props.matches?.count,
      props.matches?.overflow,
      props.matches?.totalCount
    ]);
  }

  /** Returns bounds, BVH, candidate, distance, reduction, assignment, and finalize nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, leafCapacity, candidateCapacity} = this;
    const {points, features, radius, matches} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getInputs(props),
      props.nearestFeatureIds,
      props.nearestDistances,
      props.featureCounts,
      props.overflow,
      props.candidateCount,
      matches?.ids,
      matches?.count,
      matches?.overflow,
      matches?.totalCount
    ]);
    const pointCount = points.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const minima = createTransientView(graph, `${id}-feature-minima`, 'float32x2', featureCount);
    const maxima = createTransientView(graph, `${id}-feature-maxima`, 'float32x2', featureCount);
    if (featureCount > 0) {
      nodes.push(
        createSpatialJoinBoundsNode<Parameters>(graph, {
          id: `${id}-bounds`,
          operation: OPERATION,
          featureCount,
          source: features,
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

    const expandBindings: MapGraphKernelBinding[] = [
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
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-expand`,
        operation: OPERATION,
        variant: `expand-${features.kind}`,
        bindings: expandBindings,
        invocationCount: candidateCapacity,
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
      )
    );
    const reducePrologue = `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * 2u];
  let distance = candidateDistances[candidateDistancesOffset + index];
  let searchRadius = radius[radiusOffset];
  if (!(distance >= 0.0 && distance <= searchRadius)) { return; }
  // Fold -0.0 so the bit pattern of non-negative distances is monotone.
  let distanceBits = select(bitcast<u32>(distance), 0u, distance == 0.0);`;
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
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
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `${reducePrologue}
  atomicMin(&bestDistanceBits[bestDistanceBitsOffset + pointRow], distanceBits);`
      })
    );
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
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
        pointFeatureIds: props.nearestFeatureIds,
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
  const {features} = props;
  return [
    props.points,
    props.sourceIds,
    props.featureIds,
    props.radius,
    ...(features.kind === 'points' ? [features.positions] : [features.starts, features.ends])
  ];
}
