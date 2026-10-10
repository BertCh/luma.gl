// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  createBoundsNode,
  createFeatureRingsNode,
  getSpatialJoinFeatureCount,
  getSpatialJoinGeometryViews,
  isSameSpatialJoinGeometry,
  validateSpatialJoinGeometry
} from './spatial-join-geometry';
import type {GPUSpatialJoinPrepared} from './spatial-join-prepared';
import {
  getFeatureBVHNodes,
  getNextPowerOfTwo,
  isPowerOfTwo,
  SPATIAL_JOIN_WGSL_HELPERS,
  validateDisjointOutputs
} from './spatial-join-passes';
import {type GPUSpatialJoinGeometry, type GPUSpatialJoinPairs} from './spatial-join-types';

const OPERATION = 'GPUSpatialJoinCandidates';

/** BVH facts the candidate probe needs. @internal */
export type SpatialJoinCandidateTree = {
  /** Node minima of the right-hand BVH. */
  nodeMinima: GraphDataView<'float32x2'>;
  /** Node maxima of the right-hand BVH. */
  nodeMaxima: GraphDataView<'float32x2'>;
  /** Leaf slot to right feature row. */
  leafIds: GraphDataView<'uint32'>;
  /** Number of internal nodes. */
  internalNodeCount: number;
};

/** Formats a float as a WGSL literal. @internal */
export function formatSpatialJoinFloat(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

/**
 * Returns the shared bounding-box candidate stage: a counted and scanned BVH probe from every left
 * box (expanded by `margin`), writing `(left, right)` rows ordered by left row and then right row.
 *
 * `state[0]` receives the unclamped candidate total. Slots past the total keep whatever the caller
 * cleared them to; slots past `candidateCapacity` are dropped. Nodes: count, scan, total, write.
 *
 * @internal
 */
export function getSpatialJoinCandidateNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    leftCount: number;
    rightCount: number;
    leftMinima: GraphDataView<'float32x2'>;
    leftMaxima: GraphDataView<'float32x2'>;
    tree: SpatialJoinCandidateTree;
    candidateCapacity: number;
    /**
     * Box expansion in coordinate units: a compile-time number, or a one-row float32 view read at
     * every encoding (a negative, NaN or infinite value counts as 0).
     */
    margin: number | GraphDataView<'float32'>;
    state: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, leftCount, rightCount, candidateCapacity, tree, state, candidatePairs} = props;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const leftCounts = createTransientView(graph, `${id}-left-counts`, 'uint32', leftCount);
  const leftOffsets = createTransientView(graph, `${id}-left-offsets`, 'uint32', leftCount);
  const marginView = typeof props.margin === 'number' ? undefined : props.margin;
  const probeBindings = (extra: WGSLKernelBinding[]): WGSLKernelBinding[] => [
    {name: 'leftMinima', view: props.leftMinima, type: 'f32', access: 'read'},
    {name: 'leftMaxima', view: props.leftMaxima, type: 'f32', access: 'read'},
    {name: 'nodeMinima', view: tree.nodeMinima, type: 'f32', access: 'read'},
    {name: 'nodeMaxima', view: tree.nodeMaxima, type: 'f32', access: 'read'},
    {name: 'leafIds', view: tree.leafIds, type: 'u32', access: 'read'},
    ...(marginView
      ? [{name: 'marginRow', view: marginView, type: 'f32', access: 'read'} as const]
      : []),
    ...extra
  ];
  const marginWGSL = marginView
    ? `fn getMargin() -> f32 {
  let margin = marginRow[marginRowOffset];
  return select(0.0, margin, margin >= 0.0 && margin <= 3.4028234e38);
}`
    : `fn getMargin() -> f32 { return ${formatSpatialJoinFloat(Math.fround(props.margin as number))}; }`;
  const probeDeclarations = `${SPATIAL_JOIN_WGSL_HELPERS}
const INTERNAL_NODE_COUNT: u32 = ${tree.internalNodeCount}u;
const RIGHT_COUNT: u32 = ${rightCount}u;
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
${marginWGSL}
fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}`;
  const probeBody = (
    leaf: string,
    prologue: string,
    epilogue: string
  ) => `let boxMinimum = vec2f(leftMinima[leftMinimaOffset + index * 2u], leftMinima[leftMinimaOffset + index * 2u + 1u]);
  let boxMaximum = vec2f(leftMaxima[leftMaximaOffset + index * 2u], leftMaxima[leftMaximaOffset + index * 2u + 1u]);
  ${prologue}
  if (boxMinimum.x <= boxMaximum.x && boxMinimum.y <= boxMaximum.y) {
    let margin = getMargin();
    let queryMinimum = boxMinimum - vec2f(margin);
    let queryMaximum = boxMaximum + vec2f(margin);
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT) {
          ${leaf}
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${epilogue}`;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-count-candidates`,
      operation: OPERATION,
      variant: 'count-candidates',
      bindings: probeBindings([
        {name: 'leftCounts', view: leftCounts, type: 'u32', access: 'read_write'}
      ]),
      invocationCount: leftCount,
      declarations: probeDeclarations,
      body: probeBody(
        'found = found + 1u;',
        'var found = 0u;',
        'leftCounts[leftCountsOffset + index] = found;'
      )
    })
  );
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan-candidates`,
      input: leftCounts,
      output: leftOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-total-candidates`,
      operation: OPERATION,
      variant: 'total-candidates',
      bindings: [
        {name: 'leftCounts', view: leftCounts, type: 'u32', access: 'read'},
        {name: 'leftOffsets', view: leftOffsets, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `state[stateOffset] = leftOffsets[leftOffsetsOffset + ${leftCount - 1}u] + leftCounts[leftCountsOffset + ${leftCount - 1}u];`
    })
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-write-candidates`,
      operation: OPERATION,
      variant: 'write-candidates',
      bindings: probeBindings([
        {name: 'leftOffsets', view: leftOffsets, type: 'u32', access: 'read'},
        {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'}
      ]),
      invocationCount: leftCount,
      declarations: probeDeclarations,
      body: probeBody(
        `let slot = leftOffsets[leftOffsetsOffset + index] + found;
          if (slot < CANDIDATE_CAPACITY) {
            candidatePairs[candidatePairsOffset + slot * 2u] = index;
            candidatePairs[candidatePairsOffset + slot * 2u + 1u] = rightRow;
          }
          found = found + 1u;`,
        'var found = 0u;',
        ''
      )
    })
  );
  return nodes;
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
   * Candidate pairs sorted by `(left, right)`. `pairs.leftIds.length` is the candidate capacity;
   * `count` is clamped to it, `requiredCount` is the unclamped number and `overflow` is set when
   * candidates were dropped or the BVH overflowed.
   */
  pairs: GPUSpatialJoinPairs;
};

/**
 * The bounding-box candidate stage of a spatial join on its own: every `(left, right)` feature pair
 * whose bounding boxes overlap (after expanding the left box by `distance`), sorted by left row and
 * then right row.
 *
 * This is the pair table that `GPUSpatialPredicateJoin` refines with an exact predicate. Expose it
 * to share one BVH probe between relate, distance, intersection and clip kernels: the output has the
 * layout of {@link GPUSpatialJoinPairs}, so downstream contributors read it like any join result.
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
  /** Candidate pair capacity, `pairs.leftIds.length`. */
  readonly candidateCapacity: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;

  constructor(props: GPUSpatialJoinCandidatesProps) {
    this.id = props.id ?? 'spatial-join-candidates';
    this.props = props;
    const {id} = this;
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
    const {pairs, prepared} = props;
    validatePackedUint32View(pairs.leftIds, `${id} pairs.leftIds`);
    validatePackedUint32View(pairs.rightIds, `${id} pairs.rightIds`);
    if (pairs.leftIds.length < 1 || pairs.rightIds.length !== pairs.leftIds.length) {
      throw new Error(`${id} pairs.leftIds and pairs.rightIds must have equal nonzero length`);
    }
    for (const [name, view] of [
      ['count', pairs.count],
      ['overflow', pairs.overflow],
      ['requiredCount', pairs.requiredCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} pairs.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} pairs.${name} must contain one uint32 row`);
        }
      }
    }
    this.candidateCapacity = pairs.leftIds.length;
    if (2 * this.candidateCapacity + 2 > 0xffffffff) {
      throw new Error(`${id} pair capacity is too large`);
    }
    this.leafCapacity = prepared
      ? prepared.leafCapacity
      : (props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.rightCount, 1)));
    if (!isPowerOfTwo(this.leafCapacity)) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
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
      [pairs.leftIds, pairs.rightIds, pairs.count, pairs.overflow, pairs.requiredCount]
    );
  }

  /** Returns bounds, optional BVH build, candidate probe and finalize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, leftCount, rightCount, candidateCapacity} = this;
    const {left, right, prepared, pairs} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getSpatialJoinGeometryViews(left),
      ...getSpatialJoinGeometryViews(right),
      typeof props.distance === 'number' ? undefined : props.distance,
      pairs.leftIds,
      pairs.rightIds,
      pairs.count,
      pairs.overflow,
      pairs.requiredCount
    ]);
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
      ['pairsCount', pairs.count],
      ['pairsOverflow', pairs.overflow],
      ['pairsTotal', pairs.requiredCount]
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
