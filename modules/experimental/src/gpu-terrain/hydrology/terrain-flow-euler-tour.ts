// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createContributorTransientView} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {
  createRasterGatedLoopNodes,
  createRasterIterationResetNode,
  createRasterIterationState,
  RASTER_GATE_INTERVAL,
  type RasterIterationState
} from '../../gpu-raster/cost-distance/raster-relaxation';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

/*
 * Euler tour of a D8 receiver forest, ranked by Wyllie pointer jumping.
 *
 * Every cell `v` owns two tour edges: enter `E(v) = 2 v` and exit `X(v) = 2 v + 1`. A cell
 * enumerates its children itself (neighbors whose parent is `v`, in fixed direction order), so
 * `firstChild` and `nextSibling` need no CSR:
 *
 * - `succ(E(v)) = E(firstChild(v))`, or `X(v)` for a leaf;
 * - `succ(X(v)) = E(nextSibling(v))`, or `X(parent(v))`, or `X(v)` itself for a root.
 *
 * The exit edge of a root is the terminal of its tour. Pointer jumping with ping-pong buffers then
 * leaves `distance[e]` equal to the number of edges from `e` to the terminal, and `successor[e]`
 * equal to the terminal. The edges `E(v)..X(v)` are exactly the tour of the subtree of `v`, so
 * `2 * subtreeSize(v) = distance[E(v)] - distance[X(v)] + 1`, an exact integer.
 *
 * A forest has no cycles, so every edge reaches a terminal. Parent cycles (possible only in
 * caller-supplied directions) form a closed list: its edges end at no terminal, which
 * TERRAIN_FLOW_TOUR_RESOLVED_WGSL reports per edge.
 */

/** Number of unrolled ranking rounds for `cellCount` cells: `ceil(log2(2 n)) + 1`, rounded up to the gate group size. @internal */
export function getTerrainFlowTourRoundCount(cellCount: number): number {
  const edgeCount = Math.max(2 * cellCount, 2);
  // A list of `edgeCount` edges needs ceil(log2(edgeCount)) doublings, plus one that confirms
  // that nothing changed. Groups of four rounds keep the result in the first buffer pair.
  const doublings = 32 - Math.clz32(edgeCount - 1);
  return Math.ceil((doublings + 1) / RASTER_GATE_INTERVAL) * RASTER_GATE_INTERVAL;
}

/**
 * WGSL helpers of the tour: `TOUR_NONE`, `getTourNeighbor`, `getTourFirstChild`,
 * `getTourNextSibling`. The including kernel must define `fn getTourParent(cell: u32) -> u32`
 * returning the parent cell or `TOUR_NONE`, and bind whatever that function reads.
 *
 * @internal
 */
export function getTerrainFlowEulerTourWGSL(width: number, height: number): string {
  return /* wgsl */ `
const TOUR_WIDTH: u32 = ${width}u;
const TOUR_HEIGHT: u32 = ${height}u;
const TOUR_NONE: u32 = 0xffffffffu;

fn getTourNeighbor(cell: u32, direction: u32) -> u32 {
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let column = i32(cell % TOUR_WIDTH) + columnOffsets[direction];
  let row = i32(cell / TOUR_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(TOUR_WIDTH) || row >= i32(TOUR_HEIGHT)) {
    return TOUR_NONE;
  }
  return u32(row) * TOUR_WIDTH + u32(column);
}

fn getTourFirstChild(cell: u32) -> u32 {
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getTourNeighbor(cell, direction);
    if (neighbor != TOUR_NONE && getTourParent(neighbor) == cell) { return neighbor; }
  }
  return TOUR_NONE;
}

// The child of 'parent' that follows 'cell' in the direction order of 'parent'.
fn getTourNextSibling(cell: u32, parent: u32) -> u32 {
  var passed = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getTourNeighbor(parent, direction);
    if (neighbor == TOUR_NONE) { continue; }
    if (neighbor == cell) { passed = true; continue; }
    if (passed && getTourParent(neighbor) == parent) { return neighbor; }
  }
  return TOUR_NONE;
}`;
}

/**
 * `isTourEdgeResolved(edge)`: true when the ranked list of `edge` ends at the exit edge of a root.
 * Needs a read binding named `tourSucc` (the ranked successors) next to the
 * {@link getTerrainFlowEulerTourWGSL} helpers. @internal
 */
export const TERRAIN_FLOW_TOUR_RESOLVED_WGSL = /* wgsl */ `
fn isTourEdgeResolved(edge: u32) -> bool {
  let terminal = tourSucc[tourSuccOffset + edge];
  // A terminal is the exit edge (odd) of a cell without a parent. A cycle edge that jumped onto
  // itself has a parent, so it is rejected.
  return (terminal & 1u) == 1u && getTourParent(terminal >> 1u) == TOUR_NONE;
}`;

/** Result of {@link createTerrainFlowEulerTourNodes}. @internal */
export type TerrainFlowEulerTour<Parameters> = {
  /** Link kernel, loop reset and gated ranking rounds. */
  nodes: GPUCommandNode<Parameters>[];
  /** Per edge: the terminal edge of its list once ranked (`2 * cellCount` rows). */
  successor: GraphDataView<'uint32'>;
  /** Per edge: number of edges to the terminal (`2 * cellCount` rows). */
  distance: GraphDataView<'uint32'>;
  /** Ranking-loop state: `[changed, enabled, rounds executed, converged]`. */
  state: RasterIterationState<Parameters>;
  /**
   * Two scratch word arrays of `2 * cellCount` rows. They are free once the ranking loop has run,
   * so later passes may reuse them instead of allocating.
   */
  scratch: [GraphDataView<'uint32'>, GraphDataView<'uint32'>];
};

/**
 * Returns the nodes that build and rank the Euler tour of the forest described by `getTourParent`.
 *
 * The link kernel and the ranking rounds are one pipeline each whatever the grid size; only the
 * number of unrolled rounds grows, with `log2(2 n)`. A converged list stops the remaining rounds on
 * the GPU, so a round count of `ceil(log2(2 n)) + 1` is a bound, not a cost. Rounds ping-pong
 * between two buffer pairs in groups of four, so the ranked result always sits in `successor` and
 * `distance`.
 *
 * @internal
 */
export function createTerrainFlowEulerTourNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    contributorId: string;
    width: number;
    height: number;
    /** WGSL defining `getTourParent`, plus anything it needs. */
    declarations: string;
    /** Bindings read by `getTourParent`. */
    parentBindings: readonly WGSLKernelBinding[];
  }
): TerrainFlowEulerTour<Parameters> {
  const cellCount = props.width * props.height;
  const edgeCount = 2 * cellCount;
  const create = (suffix: string) =>
    createContributorTransientView(
      graph,
      props.operation,
      props.contributorId,
      `${props.id}-tour-${suffix}`,
      'uint32',
      edgeCount
    );
  const successor = create('successor');
  const distance = create('distance');
  const scratchSuccessor = create('successor-scratch');
  const scratchDistance = create('distance-scratch');
  const state = createRasterIterationState(
    graph,
    `${props.id}-tour`,
    props.operation,
    edgeCount,
    props.contributorId
  );
  const link = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-tour-link`,
    operation: props.operation,
    variant: 'tour-link',
    bindings: [
      ...props.parentBindings,
      {name: 'tourSucc', view: successor, type: 'u32', access: 'read_write'},
      {name: 'tourDist', view: distance, type: 'u32', access: 'read_write'}
    ],
    invocationCount: cellCount,
    declarations: `${getTerrainFlowEulerTourWGSL(props.width, props.height)}\n${props.declarations}`,
    body: `let parent = getTourParent(index);
  let firstChild = getTourFirstChild(index);
  let exitEdge = 2u * index + 1u;
  var enterSuccessor = exitEdge;
  if (firstChild != TOUR_NONE) { enterSuccessor = 2u * firstChild; }
  // A root exits to itself: its exit edge is the terminal, at distance 0.
  var exitSuccessor = exitEdge;
  var exitDistance = 0u;
  if (parent != TOUR_NONE) {
    let sibling = getTourNextSibling(index, parent);
    exitSuccessor = select(2u * parent + 1u, 2u * sibling, sibling != TOUR_NONE);
    exitDistance = 1u;
  }
  tourSucc[tourSuccOffset + 2u * index] = enterSuccessor;
  tourSucc[tourSuccOffset + exitEdge] = exitSuccessor;
  tourDist[tourDistOffset + 2u * index] = 1u;
  tourDist[tourDistOffset + exitEdge] = exitDistance;`
  });
  const reset = createRasterIterationResetNode<Parameters>(graph, {
    id: `${props.id}-tour-reset`,
    operation: props.operation,
    state
  });
  const rounds = createRasterGatedLoopNodes<Parameters>(graph, {
    id: `${props.id}-tour-round`,
    gateId: `${props.id}-tour-gate`,
    operation: props.operation,
    state,
    maxIterations: getTerrainFlowTourRoundCount(cellCount),
    groupSize: RASTER_GATE_INTERVAL,
    createRound: ({nodeId, iteration, slotBinding, markChangedWGSL, condition, extraResources}) => {
      // Even rounds read the primary pair and write the scratch pair; odd rounds the reverse. The
      // kernel text is the same, so both directions share one pipeline.
      const [fromSuccessor, fromDistance, toSuccessor, toDistance] =
        iteration % 2 === 0
          ? [successor, distance, scratchSuccessor, scratchDistance]
          : [scratchSuccessor, scratchDistance, successor, distance];
      return createWGSLKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: props.operation,
        variant: 'tour-round',
        bindings: [
          {name: 'succIn', view: fromSuccessor, type: 'u32', access: 'read'},
          {name: 'distIn', view: fromDistance, type: 'u32', access: 'read'},
          {name: 'succOut', view: toSuccessor, type: 'u32', access: 'read_write'},
          {name: 'distOut', view: toDistance, type: 'u32', access: 'read_write'},
          {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'},
          slotBinding
        ],
        invocationCount: edgeCount,
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
  return {
    nodes: [link, reset, ...rounds],
    successor,
    distance,
    state,
    scratch: [scratchSuccessor, scratchDistance]
  };
}
