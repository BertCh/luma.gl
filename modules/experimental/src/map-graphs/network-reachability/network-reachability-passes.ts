// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {
  createFrontierRoundNode,
  getFrontierBindings,
  getFrontierFinalizeSource,
  getFrontierPushSource,
  getFrontierResetInvocationCount,
  getFrontierResetSource,
  FRONTIER_CONTROL_PARAMETER_0,
  FRONTIER_CONTROL_TRUNCATED,
  type FrontierPhase,
  type FrontierState
} from './network-frontier';

const OPERATION = 'GPUNetworkReachability';

/** WGSL constants shared by reachability kernels. @internal */
const REACHABILITY_WGSL_CONSTANTS = /* wgsl */ `
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`;

/** CSR inputs read by relaxation and predecessor passes. @internal */
export type NetworkReachabilityGraphViews = {
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  weights: GraphDataView<'float32'>;
};

/** Returns the WGSL expression for the per-frame round limit. */
function getRoundLimitSource(maxIterations: number, hasActiveIterations: boolean): string {
  return hasActiveIterations
    ? `min(activeIterations[activeIterationsOffset], ${maxIterations}u)`
    : `${maxIterations}u`;
}

/**
 * Resets costs, the stamp set, the cost phase control words and dispatch slots, and predecessors.
 * Copies `costLimit` bits and the resolved round limit into the control region. @internal
 */
export function createReachabilityInitializeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    state: FrontierState;
    phase: FrontierPhase;
    costs: GraphDataView<'float32'>;
    predecessors?: GraphDataView<'uint32'>;
    activeIterations?: GraphDataView<'uint32'>;
    costLimit?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const {state, phase} = props;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'costBits', view: props.costs, type: 'u32', access: 'read_write'},
    {
      name: 'queuedRound',
      view: state.queuedRound,
      type: 'u32',
      access: 'read_write'
    },
    {name: 'control', view: state.control, type: 'u32', access: 'read_write'},
    {
      name: 'dispatchEven',
      view: state.dispatchEven,
      type: 'u32',
      access: 'read_write'
    },
    {
      name: 'dispatchOdd',
      view: state.dispatchOdd,
      type: 'u32',
      access: 'read_write'
    }
  ];
  if (props.predecessors) {
    bindings.push({
      name: 'predecessors',
      view: props.predecessors,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.activeIterations) {
    bindings.push({
      name: 'activeIterations',
      view: props.activeIterations,
      type: 'u32',
      access: 'read'
    });
  }
  if (props.costLimit) {
    bindings.push({
      name: 'costLimit',
      view: props.costLimit,
      type: 'f32',
      access: 'read'
    });
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'initialize',
    bindings,
    invocationCount: Math.max(props.nodeCount, getFrontierResetInvocationCount(phase)),
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;`,
    body: `${getFrontierResetSource(phase, {
      limit: getRoundLimitSource(phase.maxRounds, Boolean(props.activeIterations)),
      parameter0: props.costLimit ? 'bitcast<u32>(costLimit[costLimitOffset])' : undefined
    })}
  if (index < NODE_COUNT) {
    costBits[costBitsOffset + index] = INFINITY_BITS;
    // Stamps never need clearing within one encoding, but a previous encoding left them dirty.
    queuedRound[queuedRoundOffset + index] = 0u;
    ${props.predecessors ? 'predecessors[predecessorsOffset + index] = NONE;' : ''}
  }`
  });
}

/** Seeds source costs and pushes the sources into round 0. @internal */
export function createReachabilitySeedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    state: FrontierState;
    phase: FrontierPhase;
    sources: GraphDataView<'uint32'>;
    sourceCosts?: GraphDataView<'float32'>;
    sourceCount?: GraphDataView<'uint32'>;
    hasCostLimit: boolean;
    costs: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: MapGraphKernelBinding[] = [
    {name: 'sources', view: props.sources, type: 'u32', access: 'read'}
  ];
  if (props.sourceCosts) {
    bindings.push({
      name: 'sourceCosts',
      view: props.sourceCosts,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.sourceCount) {
    bindings.push({
      name: 'sourceCount',
      view: props.sourceCount,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push(
    {
      name: 'costBits',
      view: props.costs,
      type: 'atomic<u32>',
      access: 'read_write'
    },
    ...getFrontierBindings(props.state, false)
  );
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'seed',
    bindings,
    invocationCount: props.sources.length,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
${getFrontierPushSource(
  props.phase,
  props.nodeCount,
  0,
  graph.device.limits.maxComputeWorkgroupsPerDimension
)}`,
    body: `${props.sourceCount ? 'if (index >= sourceCount[sourceCountOffset]) { return; }' : ''}
  let node = sources[sourcesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${props.sourceCosts ? 'sourceCosts[sourceCostsOffset + index]' : '0.0'};
  if (!(cost >= 0.0)) { return; }
  ${props.hasCostLimit ? 'if (cost > bitcast<f32>(atomicLoad(&control[FRONTIER_PARAMETER_0_WORD]))) { return; }' : ''}
  // Fold -0.0 so the u32 bit order of non-negative floats matches numeric order.
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  atomicMin(&costBits[costBitsOffset + node], bits);
  frontierPush(node);`
  });
}

/**
 * One relaxation round of the cost phase: chaotic Bellman-Ford with atomic float minima over the
 * round's compact queue, chaining up to `localIterations` hops per workgroup. @internal
 */
export function createReachabilityRelaxNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    state: FrontierState;
    phase: FrontierPhase;
    round: number;
    localIterations: number;
    csr: NetworkReachabilityGraphViews;
    hasCostLimit: boolean;
    costs: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const {csr} = props;
  return createFrontierRoundNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'relax',
    state: props.state,
    phase: props.phase,
    round: props.round,
    localIterations: props.localIterations,
    bindings: [
      {name: 'offsets', view: csr.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: csr.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: csr.weights, type: 'f32', access: 'read'},
      {
        name: 'costBits',
        view: props.costs,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${csr.neighbors.length}u;`,
    visitSource: `fn visit(node: u32) {
  let sourceCost = bitcast<f32>(atomicLoad(&costBits[costBitsOffset + node]));
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  ${props.hasCostLimit ? 'let costLimitValue = bitcast<f32>(atomicLoad(&control[FRONTIER_PARAMETER_0_WORD]));' : ''}
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    let weight = weights[weightsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT || !(weight >= 0.0)) { continue; }
    let candidate = sourceCost + weight;
    ${props.hasCostLimit ? 'if (candidate > costLimitValue) { continue; }' : ''}
    let candidateBits = select(bitcast<u32>(candidate), 0u, candidate == 0.0);
    if (candidateBits >= INFINITY_BITS) { continue; }
    if (candidateBits < atomicLoad(&costBits[costBitsOffset + targetNode])) {
      if (atomicMin(&costBits[costBitsOffset + targetNode], candidateBits) > candidateBits) {
        push(targetNode);
      }
    }
  }
}`
  });
}

/** Publishes the converged flag and executed round count. @internal */
export function createReachabilityFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    state: FrontierState;
    phase: FrontierPhase;
    /** Tie-level phase whose truncation also clears `converged`. */
    tiePhase?: FrontierPhase;
    converged?: GraphDataView<'uint32'>;
    iterationCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: MapGraphKernelBinding[] = [
    {name: 'control', view: props.state.control, type: 'u32', access: 'read'}
  ];
  if (props.converged) {
    bindings.push({
      name: 'converged',
      view: props.converged,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.iterationCount) {
    bindings.push({
      name: 'iterationCount',
      view: props.iterationCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'finalize',
    bindings,
    invocationCount: 1,
    body: `${getFrontierFinalizeSource(props.phase)}
  ${
    props.converged
      ? `converged[convergedOffset] = frontierConverged${
          props.tiePhase
            ? ` & select(0u, 1u, control[controlOffset + ${props.tiePhase.controlWordOffset + FRONTIER_CONTROL_TRUNCATED}u] == 0u)`
            : ''
        };`
      : ''
  }
  ${props.iterationCount ? 'iterationCount[iterationCountOffset] = frontierRoundCount;' : ''}`
  });
}

/**
 * One-thread gate after relaxation `iteration`: clears the change flag, counts iterations, and
 * zeroes the relax dispatch once converged or once the per-frame iteration limit is reached.
 *
 * Not used by {@link GPUNetworkReachability} any more; kept for the `GPUNetworkServiceAreas` label
 * phase, which still runs the gated per-node relaxation.
 *
 * @internal
 */
export function createReachabilityGateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    iteration: number;
    maxIterations: number;
    status: GraphDataView<'uint32'>;
    dispatch: GraphDataView<'uint32'>;
    activeIterations?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: MapGraphKernelBinding[] = [
    {name: 'status', view: props.status, type: 'u32', access: 'read_write'},
    {
      name: 'dispatch',
      view: props.dispatch,
      type: 'u32',
      access: 'read_write'
    }
  ];
  if (props.activeIterations) {
    bindings.push({
      name: 'activeIterations',
      view: props.activeIterations,
      type: 'u32',
      access: 'read'
    });
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'gate',
    bindings,
    invocationCount: 1,
    body: `if (status[statusOffset + 1u] == 0u) { return; }
  let changed = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  status[statusOffset + 2u] = status[statusOffset + 2u] + 1u;
  if (changed == 0u) {
    status[statusOffset + 1u] = 0u;
    status[statusOffset + 3u] = 1u;
    dispatch[dispatchOffset] = 0u;
    return;
  }
  if (${props.iteration}u + 1u >= ${getRoundLimitSource(props.maxIterations, Boolean(props.activeIterations))}) {
    status[statusOffset + 1u] = 0u;
    dispatch[dispatchOffset] = 0u;
  }`
  });
}

/**
 * Writes the smallest strictly-cheaper predecessor on a shortest path. With `tieBits`, also sets
 * the bit of every tie edge (a tight edge between equal costs) for the tie-level phase. @internal
 */
export function createReachabilityPredecessorsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    csr: NetworkReachabilityGraphViews;
    costs: GraphDataView<'float32'>;
    predecessors: GraphDataView<'uint32'>;
    tieBits?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {csr} = props;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'offsets', view: csr.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: csr.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: csr.weights, type: 'f32', access: 'read'},
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {
      name: 'predecessors',
      view: props.predecessors,
      type: 'atomic<u32>',
      access: 'read_write'
    }
  ];
  if (props.tieBits) {
    bindings.push({
      name: 'tieBits',
      view: props.tieBits,
      type: 'atomic<u32>',
      access: 'read_write'
    });
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'predecessors',
    bindings,
    invocationCount: props.nodeCount,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${csr.neighbors.length}u;`,
    body: `let sourceCost = costs[costsOffset + index];
  if (bitcast<u32>(sourceCost) >= INFINITY_BITS) { return; }
  let firstEdge = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    let weight = weights[weightsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT || !(weight >= 0.0)) { continue; }
    let targetCost = costs[costsOffset + targetNode];
    if (bitcast<u32>(targetCost) >= INFINITY_BITS || sourceCost + weight != targetCost) { continue; }
    if (sourceCost < targetCost) {
      atomicMin(&predecessors[predecessorsOffset + targetNode], index);
    }${
      props.tieBits
        ? ` else if (sourceCost == targetCost) {
      atomicOr(&tieBits[tieBitsOffset + (edgeIndex >> 5u)], 1u << (edgeIndex & 31u));
    }`
        : ''
    }
  }`
  });
}

/** Writes per-node isochrone band IDs. @internal */
export function createReachabilityBandsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    costs: GraphDataView<'float32'>;
    thresholds: GraphDataView<'float32'>;
    bands: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'bands',
    bindings: [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {
        name: 'thresholds',
        view: props.thresholds,
        type: 'f32',
        access: 'read'
      },
      {name: 'bands', view: props.bands, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.nodeCount,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const THRESHOLD_COUNT: u32 = ${props.thresholds.length}u;`,
    body: `let cost = costs[costsOffset + index];
  var band = NONE;
  if (bitcast<u32>(cost) < INFINITY_BITS) {
    var count = 0u;
    for (var thresholdIndex = 0u; thresholdIndex < THRESHOLD_COUNT; thresholdIndex++) {
      if (thresholds[thresholdsOffset + thresholdIndex] < cost) { count++; }
    }
    if (count < THRESHOLD_COUNT) { band = count; }
  }
  bands[bandsOffset + index] = band;`
  });
}

/** WGSL helper shared by tie-level kernels: whether `node` has any tie out-edge. @internal */
const TIE_EDGE_WGSL = /* wgsl */ `
fn isTieEdge(edgeIndex: u32) -> bool {
  return ((tieBits[tieBitsOffset + (edgeIndex >> 5u)] >> (edgeIndex & 31u)) & 1u) != 0u;
}

fn hasTieOutEdge(node: u32) -> bool {
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (isTieEdge(edgeIndex)) { return true; }
  }
  return false;
}`;

/**
 * Prepares the tie-level phase: clears `levels` to `NONE` and `tieBits` to zero, and resets the
 * tie phase's control words and dispatch slots. Runs before the predecessor pass. @internal
 */
export function createReachabilityTieInitializeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    state: FrontierState;
    tiePhase: FrontierPhase;
    levels: GraphDataView<'uint32'>;
    tieBits: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {state, tiePhase} = props;
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-initialize',
    bindings: [
      {name: 'levels', view: props.levels, type: 'u32', access: 'read_write'},
      {name: 'tieBits', view: props.tieBits, type: 'u32', access: 'read_write'},
      {name: 'control', view: state.control, type: 'u32', access: 'read_write'},
      {name: 'dispatchEven', view: state.dispatchEven, type: 'u32', access: 'read_write'},
      {name: 'dispatchOdd', view: state.dispatchOdd, type: 'u32', access: 'read_write'}
    ],
    invocationCount: Math.max(
      props.nodeCount,
      props.tieBits.length,
      getFrontierResetInvocationCount(tiePhase)
    ),
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const TIE_WORD_COUNT: u32 = ${props.tieBits.length}u;`,
    body: `${getFrontierResetSource(tiePhase, {limit: `${tiePhase.maxRounds}u`})}
  if (index < NODE_COUNT) { levels[levelsOffset + index] = NONE; }
  if (index < TIE_WORD_COUNT) { tieBits[tieBitsOffset + index] = 0u; }`
  });
}

/**
 * Marks root sources: accepted source rows whose seeded cost equals the final cost of their node
 * get tie level 0. Mirrors the seed kernel's acceptance rules. @internal
 */
export function createReachabilityTieRootsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    state: FrontierState;
    phase: FrontierPhase;
    sources: GraphDataView<'uint32'>;
    sourceCosts?: GraphDataView<'float32'>;
    sourceCount?: GraphDataView<'uint32'>;
    hasCostLimit: boolean;
    costs: GraphDataView<'float32'>;
    levels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: MapGraphKernelBinding[] = [
    {name: 'sources', view: props.sources, type: 'u32', access: 'read'}
  ];
  if (props.sourceCosts) {
    bindings.push({name: 'sourceCosts', view: props.sourceCosts, type: 'f32', access: 'read'});
  }
  if (props.sourceCount) {
    bindings.push({name: 'sourceCount', view: props.sourceCount, type: 'u32', access: 'read'});
  }
  bindings.push(
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'levels', view: props.levels, type: 'u32', access: 'read_write'}
  );
  if (props.hasCostLimit) {
    bindings.push({name: 'control', view: props.state.control, type: 'u32', access: 'read'});
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-roots',
    bindings,
    invocationCount: props.sources.length,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;`,
    body: `${props.sourceCount ? 'if (index >= sourceCount[sourceCountOffset]) { return; }' : ''}
  let node = sources[sourcesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${props.sourceCosts ? 'sourceCosts[sourceCostsOffset + index]' : '0.0'};
  if (!(cost >= 0.0)) { return; }
  ${props.hasCostLimit ? `if (cost > bitcast<f32>(control[controlOffset + ${props.phase.controlWordOffset + FRONTIER_CONTROL_PARAMETER_0}u])) { return; }` : ''}
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  if (bits == bitcast<u32>(costs[costsOffset + node])) {
    levels[levelsOffset + node] = 0u;
  }`
  });
}

/**
 * Assigns level 0 to every node with a strict predecessor and pushes each level-0 node that has a
 * tie out-edge into round 0 of the tie-level phase. Roots were marked by the roots pass. @internal
 */
export function createReachabilityTieSeedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    edgeCount: number;
    state: FrontierState;
    tiePhase: FrontierPhase;
    offsets: GraphDataView<'uint32'>;
    tieBits: GraphDataView<'uint32'>;
    predecessors: GraphDataView<'uint32'>;
    levels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-seed',
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'tieBits', view: props.tieBits, type: 'u32', access: 'read'},
      {name: 'predecessors', view: props.predecessors, type: 'u32', access: 'read'},
      {name: 'levels', view: props.levels, type: 'u32', access: 'read_write'},
      ...getFrontierBindings(props.state, false)
    ],
    invocationCount: props.nodeCount,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${props.edgeCount}u;
${TIE_EDGE_WGSL}
${getFrontierPushSource(
  props.tiePhase,
  props.nodeCount,
  0,
  graph.device.limits.maxComputeWorkgroupsPerDimension
)}`,
    body: `if (predecessors[predecessorsOffset + index] != NONE) {
    levels[levelsOffset + index] = 0u;
  }
  if (levels[levelsOffset + index] == 0u && hasTieOutEdge(index)) {
    frontierPush(index);
  }`
  });
}

/**
 * One round of the tie-level phase: relaxes `levels[v] = min(levels[v], levels[u] + 1)` along tie
 * edges, chaining `localIterations` hops per workgroup. @internal
 */
export function createReachabilityTieLevelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    edgeCount: number;
    state: FrontierState;
    tiePhase: FrontierPhase;
    round: number;
    localIterations: number;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    tieBits: GraphDataView<'uint32'>;
    levels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createFrontierRoundNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-level',
    state: props.state,
    phase: props.tiePhase,
    round: props.round,
    localIterations: props.localIterations,
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      {name: 'tieBits', view: props.tieBits, type: 'u32', access: 'read'},
      {name: 'levels', view: props.levels, type: 'atomic<u32>', access: 'read_write'}
    ],
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${props.edgeCount}u;
${TIE_EDGE_WGSL}`,
    // Only nodes with a tie out-edge are pushed, so a plateau that ends exactly at the round budget
    // is not reported as truncated.
    visitSource: `fn visit(node: u32) {
  let level = atomicLoad(&levels[levelsOffset + node]);
  if (level == NONE) { return; }
  let nextLevel = level + 1u;
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (!isTieEdge(edgeIndex)) { continue; }
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT) { continue; }
    if (nextLevel < atomicLoad(&levels[levelsOffset + targetNode])) {
      if (atomicMin(&levels[levelsOffset + targetNode], nextLevel) > nextLevel && hasTieOutEdge(targetNode)) {
        push(targetNode);
      }
    }
  }
}`
  });
}

/**
 * Writes the smallest tie in-neighbor one level below each plateau node that has no strict
 * predecessor. Reads `levels` and the tie edge bits; leaves strict predecessors and roots alone.
 *
 * @internal
 */
export function createReachabilityTiePredecessorsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    edgeCount: number;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    tieBits: GraphDataView<'uint32'>;
    levels: GraphDataView<'uint32'>;
    predecessors: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-predecessors',
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      {name: 'tieBits', view: props.tieBits, type: 'u32', access: 'read'},
      {name: 'levels', view: props.levels, type: 'u32', access: 'read'},
      {name: 'predecessors', view: props.predecessors, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.nodeCount,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${props.edgeCount}u;
${TIE_EDGE_WGSL}`,
    body: `let level = levels[levelsOffset + index];
  if (level == NONE) { return; }
  let firstEdge = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (!isTieEdge(edgeIndex)) { continue; }
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT) { continue; }
    let targetLevel = levels[levelsOffset + targetNode];
    if (targetLevel != 0u && targetLevel != NONE && level + 1u == targetLevel) {
      atomicMin(&predecessors[predecessorsOffset + targetNode], index);
    }
  }`
  });
}
