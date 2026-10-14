// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validatePowerOfTwo,
  validateCapacity,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createBoundsNode,
  createFeatureRingsNode,
  getSpatialJoinFeatureCount,
  getSpatialJoinGeometryViews,
  isSameSpatialJoinGeometry,
  validateSpatialJoinGeometry
} from './spatial-join-geometry';
import type {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';
import {
  getFeatureBVHNodes,
  getNextPowerOfTwo,
  SPATIAL_JOIN_WGSL_HELPERS,
  validateDisjointOutputs
} from './spatial-join-passes';
import {type GPUSpatialJoinGeometry} from './spatial-join-types';

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUCompactPairPort} from '../../analysis-core/contracts/index';

import {
  getSpatialJoinCandidateNodes,
  type SpatialJoinCandidateTree
} from './spatial-join-candidates';
import {validateGPUCompactPairPort} from '../../analysis-core/contracts/index';

const OPERATION = 'GPUSpatialJoinCandidates';

/**
 * Largest candidate pair capacity (`output.leftIds.length`) of {@link GPUSpatialJoinCandidates} on
 * `device`: the scratch table of pairs holds one `uint32x2` (8 bytes) per pair in one storage
 * binding, so 2^24 on a default device (128 MiB bindings).
 */
export function getGPUSpatialJoinCandidatesMaximumPairCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Properties for {@link GPUSpatialJoinCandidates}.
 *
 * Per-frame: the contents of every input buffer, including a view-backed `distance`. Topology:
 * view lengths, whether `distance` is a number or view, `leafCapacity`, the pair capacity and the
 * presence of `prepared`.
 */
export type GPUSpatialJoinCandidatesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-join-candidates'`. */
  id?: string;
  /**
   * Coordinate space and metric of both sides. Omitted means native planar coordinates, with
   * `distance` in coordinate units. `longitude-latitude` throws at construction: the box expansion
   * here is planar. For great-circle candidates use `GPUSpatialPredicateJoin` with `dwithin`.
   */
  spatialContext?: GPUSpatialContext;
  /** Left features. Their bounding boxes probe the right-hand tree. */
  left: GPUSpatialJoinGeometry;
  /** Right features, indexed by a BVH unless `prepared` supplies one. */
  right: GPUSpatialJoinGeometry;
  /**
   * Optional prepared right-hand side over the same views as `right`. It must be added to the
   * graph before this contributor, and must not use `spatialSort` (that breaks the sorted order).
   */
  prepared?: GPUSpatialJoinPrepared;
  /**
   * Expand each left box by this planar distance before probing, which makes the candidates
   * of a `dwithin` join. A one-row float32 view is read every encoding; negative, NaN and infinite
   * values act as `0`. Defaults to `0` (boxes that overlap or touch).
   */
  distance?: number | GraphDataView<'float32'>;
  /** Power-of-two BVH leaf slots when no `prepared` handle is given. Defaults to the next power of two. */
  leafCapacity?: number;
  /**
   * Caller-owned candidate pairs sorted by `(left, right)`. `output.leftIds.length` is the
   * candidate capacity; `count` is clamped to it, `requiredCount` is the unclamped number and
   * `overflow` is set when candidates were dropped or the BVH overflowed. The capacity is at most
   * {@link getGPUSpatialJoinCandidatesMaximumPairCapacity} (2^24 on a default device).
   */
  output: GPUCompactPairPort;
};

/**
 * The bounding-box candidate stage of a spatial join on its own: every `(left, right)` feature pair
 * whose bounding boxes overlap (after expanding the left box by `distance`), sorted by left row and
 * then right row.
 *
 * This is the pair table that `GPUSpatialPredicateJoin` refines with an exact predicate. Expose it
 * to share one BVH probe between relate, distance, intersection and clip kernels: the output has the
 * layout of {@link GPUCompactPairPort}, so downstream contributors read it like any join result.
 * It is conservative: a candidate pair may not intersect, and no intersecting pair is missing.
 * Features that are empty, or have non-finite coordinates, have inverted bounds and never match.
 */
export class GPUSpatialJoinCandidates implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialJoinCandidatesProps;
  /** Number of left features. */
  readonly leftCount: number;
  /** Number of right features. */
  readonly rightCount: number;
  /** Candidate pair capacity, `output.leftIds.length`. */
  readonly candidateCapacity: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Resolved spatial context: native planar coordinates. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUSpatialJoinCandidatesProps) {
    this.id = props.id ?? 'spatial-join-candidates';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    validateSpatialJoinGeometry(id, 'left', props.left);
    validateSpatialJoinGeometry(id, 'right', props.right);
    this.leftCount = getSpatialJoinFeatureCount(props.left);
    this.rightCount = getSpatialJoinFeatureCount(props.right);
    if (
      typeof props.distance === 'number' &&
      (!Number.isFinite(props.distance) || props.distance < 0)
    ) {
      throw new Error(`${id} distance must be finite and non-negative`);
    }
    if (props.distance !== undefined && typeof props.distance !== 'number') {
      validatePackedView(props.distance, ['float32'], `${id} distance`);
      if (props.distance.length < 1) {
        throw new Error(`${id} distance must contain one float32 row`);
      }
    }
    const {output: pairs, prepared} = props;
    validateGPUCompactPairPort(id, pairs);
    for (const view of [
      pairs.leftIds,
      pairs.rightIds,
      pairs.status.count,
      pairs.status.overflow,
      pairs.status.requiredCount
    ]) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output`);
      }
    }
    if (pairs.leftIds.length < 1) {
      throw new Error(`${id} output.leftIds needs at least one row`);
    }
    this.candidateCapacity = pairs.leftIds.length;
    this.leafCapacity = prepared
      ? prepared.leafCapacity
      : (props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.rightCount, 1)));
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    if (prepared) {
      if (!isSameSpatialJoinGeometry(prepared.geometry, props.right)) {
        throw new Error(`${id} prepared must index the same geometry views as right`);
      }
      if (prepared.spatialSort) {
        throw new Error(`${id} prepared.spatialSort leaves candidates unsorted; disable it`);
      }
    }
    validateDisjointOutputs(
      id,
      [
        ...getSpatialJoinGeometryViews(props.left),
        ...getSpatialJoinGeometryViews(props.right),
        typeof props.distance === 'number' ? undefined : props.distance
      ],
      [
        pairs.leftIds,
        pairs.rightIds,
        pairs.status.count,
        pairs.status.overflow,
        pairs.status.requiredCount
      ]
    );
  }

  /** Returns bounds, optional BVH build, candidate probe and finalize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, leftCount, rightCount, candidateCapacity} = this;
    const {left, right, prepared, output: pairs} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getSpatialJoinGeometryViews(left),
      ...getSpatialJoinGeometryViews(right),
      typeof props.distance === 'number' ? undefined : props.distance,
      pairs.leftIds,
      pairs.rightIds,
      pairs.status.count,
      pairs.status.overflow,
      pairs.status.requiredCount
    ]);
    // Candidate pairs are stored as two words each in one scratch binding.
    validateCapacity(
      id,
      'output.leftIds.length',
      candidateCapacity,
      1,
      getGPUSpatialJoinCandidatesMaximumPairCapacity(graph.device)
    );
    if (prepared && !prepared.isDeclaredIn(graph)) {
      throw new Error(`${id} requires prepared to be added to the graph first`);
    }
    const nodes: GPUCommandNode<Parameters>[] = [];

    const leftMinima = createTransientView(graph, `${id}-left-minima`, 'float32x2', leftCount);
    const leftMaxima = createTransientView(graph, `${id}-left-maxima`, 'float32x2', leftCount);
    let leftFeatureRings: GraphDataView<'uint32x2'> | undefined;
    if (left.kind === 'polygons') {
      leftFeatureRings = createTransientView(
        graph,
        `${id}-left-feature-rings`,
        'uint32x2',
        leftCount
      );
      nodes.push(
        createFeatureRingsNode<Parameters>(graph, `${id}-left-rings`, {
          featureCount: leftCount,
          geometry: left,
          featureRings: leftFeatureRings
        })
      );
    }
    nodes.push(
      createBoundsNode<Parameters>(graph, `${id}-left-bounds`, {
        featureCount: leftCount,
        geometry: left,
        featureRings: leftFeatureRings,
        minima: leftMinima,
        maxima: leftMaxima
      })
    );

    let tree: SpatialJoinCandidateTree;
    let bvhOverflow: GraphDataView<'uint32'>;
    if (prepared) {
      tree = {
        nodeMinima: prepared.storage.nodeMinima,
        nodeMaxima: prepared.storage.nodeMaxima,
        leafIds: prepared.storage.leafIds,
        internalNodeCount: prepared.bvhInternalNodeCount
      };
      bvhOverflow = prepared.storage.overflow;
    } else {
      const rightMinima = createTransientView(graph, `${id}-right-minima`, 'float32x2', rightCount);
      const rightMaxima = createTransientView(graph, `${id}-right-maxima`, 'float32x2', rightCount);
      let rightFeatureRings: GraphDataView<'uint32x2'> | undefined;
      if (right.kind === 'polygons') {
        rightFeatureRings = createTransientView(
          graph,
          `${id}-right-feature-rings`,
          'uint32x2',
          rightCount
        );
        nodes.push(
          createFeatureRingsNode<Parameters>(graph, `${id}-right-rings`, {
            featureCount: rightCount,
            geometry: right,
            featureRings: rightFeatureRings
          })
        );
      }
      nodes.push(
        createBoundsNode<Parameters>(graph, `${id}-right-bounds`, {
          featureCount: rightCount,
          geometry: right,
          featureRings: rightFeatureRings,
          minima: rightMinima,
          maxima: rightMaxima
        })
      );
      const built = getFeatureBVHNodes(graph, id, rightMinima, rightMaxima, this.leafCapacity);
      nodes.push(...built.nodes);
      tree = {
        nodeMinima: built.bvh.nodeMinima as GraphDataView<'float32x2'>,
        nodeMaxima: built.bvh.nodeMaxima as GraphDataView<'float32x2'>,
        leafIds: built.bvh.leafIds,
        internalNodeCount: built.bvh.internalNodeCount
      };
      bvhOverflow = built.bvh.overflow;
    }

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      candidateCapacity
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(4, candidateCapacity),
        declarations: SPATIAL_JOIN_WGSL_HELPERS,
        body: `if (index < 4u) { state[stateOffset + index] = 0u; }
  if (index < ${candidateCapacity}u) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
  }`
      })
    );
    nodes.push(
      ...getSpatialJoinCandidateNodes<Parameters>(graph, {
        id,
        leftCount,
        rightCount,
        leftMinima,
        leftMaxima,
        tree,
        candidateCapacity,
        margin: props.distance ?? 0,
        state,
        candidatePairs
      })
    );
    const scalars: [string, GraphDataView<'uint32'> | undefined][] = [
      ['pairsCount', pairs.status.count],
      ['pairsOverflow', pairs.status.overflow],
      ['pairsTotal', pairs.status.requiredCount]
    ];
    const bindings: WGSLKernelBinding[] = [
      {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
      {name: 'leftIds', view: pairs.leftIds, type: 'u32', access: 'read_write'},
      {name: 'rightIds', view: pairs.rightIds, type: 'u32', access: 'read_write'},
      {name: 'state', view: state, type: 'u32', access: 'read'},
      {name: 'bvhOverflow', view: bvhOverflow, type: 'u32', access: 'read'}
    ];
    for (const [name, view] of scalars) {
      if (view) {
        bindings.push({name, view, type: 'u32', access: 'read_write'});
      }
    }
    const has = (name: string) => scalars.some(([key, view]) => key === name && view);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-pairs`,
        operation: OPERATION,
        variant: 'write-pairs',
        bindings,
        invocationCount: candidateCapacity,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `leftIds[leftIdsOffset + index] = candidatePairs[candidatePairsOffset + index * 2u];
  rightIds[rightIdsOffset + index] = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (index == 0u) {
    let total = state[stateOffset];
    let overflowValue = select(0u, 1u, bvhOverflow[bvhOverflowOffset] != 0u || total > CANDIDATE_CAPACITY);
    ${has('pairsCount') ? 'pairsCount[pairsCountOffset] = min(total, CANDIDATE_CAPACITY);' : ''}
    ${has('pairsOverflow') ? 'pairsOverflow[pairsOverflowOffset] = overflowValue;' : ''}
    ${has('pairsTotal') ? 'pairsTotal[pairsTotalOffset] = total;' : ''}
  }`
      })
    );
    return nodes;
  }
}
