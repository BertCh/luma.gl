// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createContributorTransientView} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {createRasterGatedLoopNodes} from '../../gpu-raster/cost-distance/raster-relaxation';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createSpillLoopState,
  createSpillResetNode,
  getSpillRoundCount,
  SPILL_WORKGROUP_SIZE,
  type SpillGateUse
} from './terrain-flow-spill-utils';

/*
 * Rooting of an explicit tree and bottleneck paths to its root, for the spill fill.
 *
 * The tree has `n + 1` nodes (the ocean is node `n`) and at most `n` edges, `mstFirst[k] -
 * mstSecond[k]` with weight key `mstKey[k]`. Edge `k` owns the half-edges `2 k` (first to second)
 * and `2 k + 1` (second to first).
 *
 * Euler tour: every node keeps a singly linked list of its outgoing half-edges, built with one
 * `atomicExchange` per half-edge (arbitrary order, no CSR). The tour successor of the half-edge
 * `u -> v` is the half-edge that follows `v -> u` in the cyclic list of `v`. The tour is one cycle
 * over all half-edges of a connected tree; it is cut at the ocean, where the half-edge whose
 * successor is the head of the ocean list becomes the terminal. Wyllie pointer jumping ranks every
 * half-edge by its distance to the terminal. A half-edge is traversed away from the root exactly
 * when it comes before its reverse, that is when its distance is larger. The child endpoint of
 * every edge then points to its parent with the edge key, and doubling with `max` turns that into
 * the maximum key on the path to the root. The result does not depend on the order of the lists.
 */

/** Result of {@link createTerrainFlowSpillTreeNodes}. @internal */
export type TerrainFlowSpillTree<Parameters> = {
  /** Lists, tour link, ranking rounds, rooting, and path-maximum rounds. */
  nodes: GPUCommandNode<Parameters>[];
};

/**
 * Returns the nodes that root the tree at the ocean node and compute the maximum edge key on each
 * node's path to it. On completion `parent[v]` is the ocean for every node of the ocean's tree and
 * `parentKey[v]` is the bottleneck key of `v` (0 for the ocean).
 *
 * `parent` and `parentKey` must hold `parent[v] = v` and `parentKey[v] = 0` for every node before
 * the first node runs, and `head` must be all `0xffffffff`. The edge count is word 4 of
 * `mstStatus`. Every node is gated: the lists, link and rooting on the supplied gates, the loops on
 * their own states, which {@link createSpillResetNode} arms only when `enabledExpression` holds.
 *
 * @internal
 */
export function createTerrainFlowSpillTreeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    contributorId: string;
    cellCount: number;
    mstFirst: GraphDataView<'uint32'>;
    mstSecond: GraphDataView<'uint32'>;
    mstKey: GraphDataView<'uint32'>;
    head: GraphDataView<'uint32'>;
    /** Status words of the spanning-tree loop; word 4 holds the number of tree edges. */
    mstStatus: GraphDataView<'uint32'>;
    parent: GraphDataView<'uint32'>;
    parentKey: GraphDataView<'uint32'>;
    enabledBindings: readonly WGSLKernelBinding[];
    enabledExpression: string;
    /** Gate of kernels over `2 * cellCount` half-edges. */
    getHalfEdgeGate: (nodeId: string) => SpillGateUse<Parameters>;
    /** Gate of kernels over `cellCount` tree edges. */
    getEdgeGate: (nodeId: string) => SpillGateUse<Parameters>;
  }
): TerrainFlowSpillTree<Parameters> {
  const {cellCount} = props;
  const nodeCount = cellCount + 1;
  const halfEdgeCount = 2 * cellCount;
  const create = (name: string, length: number) =>
    createContributorTransientView(
      graph,
      props.operation,
      props.contributorId,
      `${props.id}-spill-tree-${name}`,
      'uint32',
      length
    );
  const listNext = create('list-next', halfEdgeCount);
  const successor = create('successor', halfEdgeCount);
  const distance = create('distance', halfEdgeCount);
  const scratchSuccessor = create('successor-scratch', halfEdgeCount);
  const scratchDistance = create('distance-scratch', halfEdgeCount);
  const scratchParent = create('parent-scratch', nodeCount);
  const scratchParentKey = create('parent-key-scratch', nodeCount);
  const kernel = (
    name: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string,
    gate: SpillGateUse<Parameters>
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-spill-tree-${name}`,
      operation: props.operation,
      variant: `spill-tree-${name}`,
      bindings,
      invocationCount,
      workgroupSize: SPILL_WORKGROUP_SIZE,
      declarations: 'const TREE_NONE: u32 = 0xffffffffu;',
      body,
      ...gate
    });
  const edgeBindings: WGSLKernelBinding[] = [
    {name: 'mstFirst', view: props.mstFirst, type: 'u32', access: 'read'},
    {name: 'mstSecond', view: props.mstSecond, type: 'u32', access: 'read'}
  ];
  const statusBinding: WGSLKernelBinding = {
    name: 'status',
    view: props.mstStatus,
    type: 'u32',
    access: 'read'
  };

  const lists = kernel(
    'lists',
    halfEdgeCount,
    [
      ...edgeBindings,
      {name: 'head', view: props.head, type: 'atomic<u32>', access: 'read_write'},
      {name: 'listNext', view: listNext, type: 'u32', access: 'read_write'},
      statusBinding
    ],
    `if (index >= 2u * status[statusOffset + 4u]) { return; }
  let edge = index >> 1u;
  let origin = select(mstFirst[mstFirstOffset + edge], mstSecond[mstSecondOffset + edge], (index & 1u) == 1u);
  // Push this half-edge on the list of its origin; the previous head becomes its successor.
  listNext[listNextOffset + index] = atomicExchange(&head[headOffset + origin], index);`,
    props.getHalfEdgeGate(`${props.id}-spill-tree-lists`)
  );
  const link = kernel(
    'link',
    halfEdgeCount,
    [
      ...edgeBindings,
      {name: 'head', view: props.head, type: 'atomic<u32>', access: 'read_write'},
      {name: 'listNext', view: listNext, type: 'u32', access: 'read'},
      {name: 'tourSucc', view: successor, type: 'u32', access: 'read_write'},
      {name: 'tourDist', view: distance, type: 'u32', access: 'read_write'},
      statusBinding
    ],
    `if (index >= 2u * status[statusOffset + 4u]) {
    // Unused slots are their own terminals, so the ranking rounds never read out of range.
    tourSucc[tourSuccOffset + index] = index;
    tourDist[tourDistOffset + index] = 0u;
    return;
  }
  let edge = index >> 1u;
  // The half-edge arrives at the other endpoint than its origin.
  let arrival = select(mstSecond[mstSecondOffset + edge], mstFirst[mstFirstOffset + edge], (index & 1u) == 1u);
  var following = listNext[listNextOffset + (index ^ 1u)];
  if (following == TREE_NONE) { following = atomicLoad(&head[headOffset + arrival]); }
  let rootStart = atomicLoad(&head[headOffset + ${cellCount}u]);
  // The tour is cut where it would return to the first half-edge of the ocean.
  if (following == rootStart) {
    tourSucc[tourSuccOffset + index] = index;
    tourDist[tourDistOffset + index] = 0u;
  } else {
    tourSucc[tourSuccOffset + index] = following;
    tourDist[tourDistOffset + index] = 1u;
  }`,
    props.getHalfEdgeGate(`${props.id}-spill-tree-link`)
  );

  const rankState = createSpillLoopState<Parameters>(graph, {
    id: `${props.id}-spill-tree-rank`,
    operation: props.operation,
    contributorId: props.contributorId,
    shapes: [[halfEdgeCount, SPILL_WORKGROUP_SIZE]]
  });
  const rankReset = createSpillResetNode<Parameters>(graph, {
    id: `${props.id}-spill-tree-rank-reset`,
    operation: props.operation,
    state: rankState,
    enabledBindings: props.enabledBindings,
    enabledExpression: props.enabledExpression
  });
  const rankRounds = createRasterGatedLoopNodes<Parameters>(graph, {
    id: `${props.id}-spill-tree-rank-round`,
    gateId: `${props.id}-spill-tree-rank-gate`,
    operation: props.operation,
    state: rankState,
    maxIterations: getSpillRoundCount(halfEdgeCount),
    createRound: ({nodeId, iteration, slotBinding, markChangedWGSL, condition, extraResources}) => {
      // Even rounds read the primary pair and write the scratch pair, odd rounds the reverse. The
      // kernel text is the same, so both directions share one pipeline.
      const [fromSuccessor, fromDistance, toSuccessor, toDistance] =
        iteration % 2 === 0
          ? [successor, distance, scratchSuccessor, scratchDistance]
          : [scratchSuccessor, scratchDistance, successor, distance];
      return createWGSLKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: props.operation,
        variant: 'spill-tree-rank',
        bindings: [
          {name: 'succIn', view: fromSuccessor, type: 'u32', access: 'read'},
          {name: 'distIn', view: fromDistance, type: 'u32', access: 'read'},
          {name: 'succOut', view: toSuccessor, type: 'u32', access: 'read_write'},
          {name: 'distOut', view: toDistance, type: 'u32', access: 'read_write'},
          {name: 'status', view: rankState.status, type: 'atomic<u32>', access: 'read_write'},
          slotBinding
        ],
        invocationCount: halfEdgeCount,
        workgroupSize: SPILL_WORKGROUP_SIZE,
        body: `let next = succIn[succInOffset + index];
  let jumped = succIn[succInOffset + next];
  var rankedDistance = distIn[distInOffset + index];
  if (next != jumped) {
    rankedDistance = rankedDistance + distIn[distInOffset + next];
    ${markChangedWGSL}
  }
  succOut[succOutOffset + index] = jumped;
  distOut[distOutOffset + index] = rankedDistance;`,
        condition,
        extraResources
      });
    }
  });

  const root = kernel(
    'root',
    cellCount,
    [
      ...edgeBindings,
      {name: 'mstKey', view: props.mstKey, type: 'u32', access: 'read'},
      {name: 'tourDist', view: distance, type: 'u32', access: 'read'},
      {name: 'parent', view: props.parent, type: 'u32', access: 'read_write'},
      {name: 'parentKey', view: props.parentKey, type: 'u32', access: 'read_write'},
      statusBinding
    ],
    `if (index >= status[statusOffset + 4u]) { return; }
  // The half-edge that comes first in the tour leads away from the root (larger distance).
  let forward = tourDist[tourDistOffset + 2u * index] > tourDist[tourDistOffset + 2u * index + 1u];
  let child = select(mstFirst[mstFirstOffset + index], mstSecond[mstSecondOffset + index], forward);
  let owner = select(mstSecond[mstSecondOffset + index], mstFirst[mstFirstOffset + index], forward);
  parent[parentOffset + child] = owner;
  parentKey[parentKeyOffset + child] = mstKey[mstKeyOffset + index];`,
    props.getEdgeGate(`${props.id}-spill-tree-root`)
  );

  const pathState = createSpillLoopState<Parameters>(graph, {
    id: `${props.id}-spill-tree-path`,
    operation: props.operation,
    contributorId: props.contributorId,
    shapes: [[nodeCount, SPILL_WORKGROUP_SIZE]]
  });
  const pathReset = createSpillResetNode<Parameters>(graph, {
    id: `${props.id}-spill-tree-path-reset`,
    operation: props.operation,
    state: pathState,
    enabledBindings: props.enabledBindings,
    enabledExpression: props.enabledExpression
  });
  const pathRounds = createRasterGatedLoopNodes<Parameters>(graph, {
    id: `${props.id}-spill-tree-path-round`,
    gateId: `${props.id}-spill-tree-path-gate`,
    operation: props.operation,
    state: pathState,
    maxIterations: getSpillRoundCount(nodeCount),
    createRound: ({nodeId, iteration, slotBinding, markChangedWGSL, condition, extraResources}) => {
      const [fromParent, fromKey, toParent, toKey] =
        iteration % 2 === 0
          ? [props.parent, props.parentKey, scratchParent, scratchParentKey]
          : [scratchParent, scratchParentKey, props.parent, props.parentKey];
      return createWGSLKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: props.operation,
        variant: 'spill-tree-path',
        bindings: [
          {name: 'parentIn', view: fromParent, type: 'u32', access: 'read'},
          {name: 'keyIn', view: fromKey, type: 'u32', access: 'read'},
          {name: 'parentOut', view: toParent, type: 'u32', access: 'read_write'},
          {name: 'keyOut', view: toKey, type: 'u32', access: 'read_write'},
          {name: 'status', view: pathState.status, type: 'atomic<u32>', access: 'read_write'},
          slotBinding
        ],
        invocationCount: nodeCount,
        workgroupSize: SPILL_WORKGROUP_SIZE,
        body: `let ancestor = parentIn[parentInOffset + index];
  let jumped = parentIn[parentInOffset + ancestor];
  // The path from here to 'jumped' is the path to 'ancestor' followed by the path from 'ancestor'.
  keyOut[keyOutOffset + index] = max(keyIn[keyInOffset + index], keyIn[keyInOffset + ancestor]);
  parentOut[parentOutOffset + index] = jumped;
  if (jumped != ancestor) {
    ${markChangedWGSL}
  }`,
        condition,
        extraResources
      });
    }
  });

  return {
    nodes: [lists, link, rankReset, ...rankRounds, root, pathReset, ...pathRounds]
  };
}
