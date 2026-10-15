// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../utils/wgsl-kernel-nodes';
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

/**
 * Control words of the inflation counters, placed after every phase region of the frontier control
 * buffer: relaxation count (low, high) then reached edge count (low, high). @internal
 */
export const NETWORK_REACHABILITY_COUNTER_WORD_COUNT = 4;

/**
 * WGSL for a 64-bit counter kept as two `u32` words (low, high). Adding with carry is correct for
 * concurrent adders: the adder whose low-word sum wraps adds the carry to the high word. @internal
 */
const COUNTER_WGSL = /* wgsl */ `
var<workgroup> counterLow: atomic<u32>;
var<workgroup> counterHigh: atomic<u32>;

// Adds to this workgroup's 64-bit partial count.
fn addCount(count: u32) {
  if (count == 0u) { return; }
  let previous = atomicAdd(&counterLow, count);
  if (previous + count < previous) { atomicAdd(&counterHigh, 1u); }
}`;

/** WGSL epilogue for thread 0 (after a barrier): adds the workgroup partial to the global words. */
function getCounterFlushSource(wordOffset: number): string {
  return `if (localInvocationIndex == 0u) {
    let partialLow = atomicLoad(&counterLow);
    let partialHigh = atomicLoad(&counterHigh);
    if (partialLow != 0u || partialHigh != 0u) {
      let previous = atomicAdd(&control[controlOffset + ${wordOffset}u], partialLow);
      var carry = partialHigh;
      if (previous + partialLow < previous) { carry += 1u; }
      if (carry != 0u) { atomicAdd(&control[controlOffset + ${wordOffset + 1}u], carry); }
    }
  }`;
}

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
    /** Control word offset of the four inflation counter words to zero, when counting. */
    counterWordOffset?: number;
  }
): GPUCommandNode<Parameters> {
  const {state, phase} = props;
  const bindings: WGSLKernelBinding[] = [
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
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'initialize',
    record: {datum: {keySpace: 'node'}},
    bindings,
    invocationCount: Math.max(
      props.nodeCount,
      getFrontierResetInvocationCount(phase),
      props.counterWordOffset === undefined ? 0 : NETWORK_REACHABILITY_COUNTER_WORD_COUNT
    ),
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;`,
    body: `${getFrontierResetSource(phase, {
      limit: getRoundLimitSource(phase.maxRounds, Boolean(props.activeIterations)),
      parameter0: props.costLimit ? 'bitcast<u32>(costLimit[costLimitOffset])' : undefined
    })}
  ${
    props.counterWordOffset === undefined
      ? ''
      : `if (index < ${NETWORK_REACHABILITY_COUNTER_WORD_COUNT}u) { control[controlOffset + ${props.counterWordOffset}u + index] = 0u; }`
  }
  if (index < NODE_COUNT) {
    costBits[costBitsOffset + index] = INFINITY_BITS;
    // Stamps never need clearing within one encoding, but a previous encoding left them dirty.
    queuedRound[queuedRoundOffset + index] = 0u;
    ${props.predecessors ? 'predecessors[predecessorsOffset + index] = NONE;' : ''}
  }`
  });
}

/**
 * Returns the WGSL early return that skips start rows at or after the active start count. A view
 * is read per frame from the `sourceCount` binding; a number is baked in as a constant and adds no
 * binding. @internal
 */
export function getStartCountGuardSource(
  startCount: number | GraphDataView<'uint32'> | undefined
): string {
  if (startCount === undefined) {
    return '';
  }
  return typeof startCount === 'number'
    ? `if (index >= ${startCount}u) { return; }`
    : 'if (index >= sourceCount[sourceCountOffset]) { return; }';
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
    sourceCount?: number | GraphDataView<'uint32'>;
    hasCostLimit: boolean;
    costs: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
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
  if (typeof props.sourceCount === 'object') {
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
    ...getFrontierBindings(graph, props.state, props.phase, 0)
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'seed',
    record: {datum: {keySpace: 'node'}},
    bindings,
    invocationCount: props.sources.length,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
${getFrontierPushSource(
  props.phase,
  props.nodeCount,
  graph.device.limits.maxComputeWorkgroupsPerDimension
)}`,
    body: `${getStartCountGuardSource(props.sourceCount)}
  let node = sources[sourcesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${props.sourceCosts ? 'sourceCosts[sourceCostsOffset + index]' : '0.0'};
  if (!(cost >= 0.0)) { return; }
  ${props.hasCostLimit ? 'if (cost > bitcast<f32>(atomicLoad(&control[controlOffset + FRONTIER_PARAMETER_0_WORD]))) { return; }' : ''}
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
    record?: WGSLKernelProps<Parameters>['record'];
    nodeCount: number;
    state: FrontierState;
    phase: FrontierPhase;
    round: number;
    localIterations: number;
    csr: NetworkReachabilityGraphViews;
    hasCostLimit: boolean;
    costs: GraphDataView<'float32'>;
    /**
     * Nodes per lane when `costs` holds several lanes of one shared CSR (lane `l`, node `u` is
     * node `l * laneNodeCount + u`). The CSR is then read through `node % laneNodeCount` and the
     * lane base is added to every target, so no per-lane copy of the CSR exists.
     */
    laneNodeCount?: number;
    /**
     * Control word offset of the relaxation counter (low, high). When given, every visit adds the
     * edges it scans. The source differs by variant only, never by round.
     */
    counterWordOffset?: number;
  }
): GPUCommandNode<Parameters> {
  const {csr, laneNodeCount, counterWordOffset} = props;
  const isCounting = counterWordOffset !== undefined;
  return createFrontierRoundNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    record: props.record,
    variant: `${laneNodeCount === undefined ? 'relax' : 'relax-lanes'}${isCounting ? '-counted' : ''}`,
    epilogueSource: isCounting ? getCounterFlushSource(counterWordOffset) : undefined,
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
const NODE_COUNT: u32 = ${laneNodeCount ?? props.nodeCount}u;
const EDGE_COUNT: u32 = ${csr.neighbors.length}u;${isCounting ? COUNTER_WGSL : ''}`,
    visitSource: `fn visit(node: u32) {
  let sourceCost = bitcast<f32>(atomicLoad(&costBits[costBitsOffset + node]));
  ${
    laneNodeCount === undefined
      ? 'let csrNode = node; let laneBase = 0u;'
      : 'let laneBase = (node / NODE_COUNT) * NODE_COUNT; let csrNode = node - laneBase;'
  }
  let firstEdge = min(offsets[offsetsOffset + csrNode], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + csrNode + 1u], EDGE_COUNT);
  ${props.hasCostLimit ? 'let costLimitValue = bitcast<f32>(atomicLoad(&control[controlOffset + FRONTIER_PARAMETER_0_WORD]));' : ''}
  ${isCounting ? 'addCount(select(0u, lastEdge - firstEdge, lastEdge > firstEdge));' : ''}
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    let targetCsrNode = neighbors[neighborsOffset + edgeIndex];
    let weight = weights[weightsOffset + edgeIndex];
    if (targetCsrNode >= NODE_COUNT || !(weight >= 0.0)) { continue; }
    let targetNode = laneBase + targetCsrNode;
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
    /** Two-row (low, high) outputs and the control word offset of the counter words they copy. */
    counters?: {
      wordOffset: number;
      relaxationCount?: GraphDataView<'uint32'>;
      reachedEdgeCount?: GraphDataView<'uint32'>;
    };
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'control', view: props.state.control, type: 'u32', access: 'read'}
  ];
  const counterCopies: string[] = [];
  if (props.counters?.relaxationCount) {
    bindings.push({
      name: 'relaxationCount',
      view: props.counters.relaxationCount,
      type: 'u32',
      access: 'read_write'
    });
    const word = props.counters.wordOffset;
    counterCopies.push(
      `relaxationCount[relaxationCountOffset] = control[controlOffset + ${word}u];`,
      `relaxationCount[relaxationCountOffset + 1u] = control[controlOffset + ${word + 1}u];`
    );
  }
  if (props.counters?.reachedEdgeCount) {
    bindings.push({
      name: 'reachedEdgeCount',
      view: props.counters.reachedEdgeCount,
      type: 'u32',
      access: 'read_write'
    });
    const word = props.counters.wordOffset + 2;
    counterCopies.push(
      `reachedEdgeCount[reachedEdgeCountOffset] = control[controlOffset + ${word}u];`,
      `reachedEdgeCount[reachedEdgeCountOffset + 1u] = control[controlOffset + ${word + 1}u];`
    );
  }
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
  return createWGSLKernelNode<Parameters>(graph, {
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
  ${props.iterationCount ? 'iterationCount[iterationCountOffset] = frontierRoundCount;' : ''}
  ${counterCopies.join('\n  ')}`
  });
}

/**
 * Adds, for every node with a finite final cost, its out-degree (edges with an in-range target,
 * any weight) to the reached edge counter: the edges a label-setting Dijkstra scans exactly once.
 * One thread per expanded node, a workgroup partial sum, then one 64-bit add per workgroup. With
 * lanes the CSR is read through `node % laneNodeCount`, so the count sums over lanes. @internal
 */
export function createReachabilityReachedEdgesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    state: FrontierState;
    csr: NetworkReachabilityGraphViews;
    costs: GraphDataView<'float32'>;
    /** Nodes per lane of one shared CSR; defaults to `nodeCount`. */
    laneNodeCount?: number;
    counterWordOffset: number;
  }
): GPUCommandNode<Parameters> {
  const {csr} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'reached-edges',
    bindings: [
      {name: 'offsets', view: csr.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: csr.neighbors, type: 'u32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'control', view: props.state.control, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.nodeCount,
    // Workgroup barrier: every invocation of a dispatched workgroup must reach it.
    guardIndex: false,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const LANE_NODE_COUNT: u32 = ${props.laneNodeCount ?? props.nodeCount}u;
const EDGE_COUNT: u32 = ${csr.neighbors.length}u;${COUNTER_WGSL}`,
    body: `if (index < NODE_COUNT && bitcast<u32>(costs[costsOffset + index]) < INFINITY_BITS) {
    let csrNode = index % LANE_NODE_COUNT;
    let firstEdge = min(offsets[offsetsOffset + csrNode], EDGE_COUNT);
    let lastEdge = min(offsets[offsetsOffset + csrNode + 1u], EDGE_COUNT);
    var degree = 0u;
    for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
      if (neighbors[neighborsOffset + edgeIndex] < LANE_NODE_COUNT) { degree++; }
    }
    addCount(degree);
  }
  workgroupBarrier();
  ${getCounterFlushSource(props.counterWordOffset + 2)}`
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
    /**
     * Optional per-node label (a service-area facility row). When given, only tight edges between
     * nodes of the same label count, so predecessor links stay inside one label's tree.
     */
    labels?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {csr} = props;
  const bindings: WGSLKernelBinding[] = [
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
  if (props.labels) {
    bindings.push({name: 'labels', view: props.labels, type: 'u32', access: 'read'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: props.labels ? 'predecessors-labelled' : 'predecessors',
    record: {datum: {keySpace: 'node'}},
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
    ${props.labels ? 'if (labels[labelsOffset + index] != labels[labelsOffset + targetNode]) { continue; }' : ''}
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
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'bands',
    record: {datum: {keySpace: 'node'}, formula: 'gpu-network.reachability.bands'},
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
    unresolvedCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {state, tiePhase} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'levels', view: props.levels, type: 'u32', access: 'read_write'},
    {name: 'tieBits', view: props.tieBits, type: 'u32', access: 'read_write'},
    {name: 'control', view: state.control, type: 'u32', access: 'read_write'},
    {name: 'dispatchEven', view: state.dispatchEven, type: 'u32', access: 'read_write'},
    {name: 'dispatchOdd', view: state.dispatchOdd, type: 'u32', access: 'read_write'}
  ];
  if (props.unresolvedCount) {
    bindings.push({
      name: 'unresolvedCount',
      view: props.unresolvedCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-initialize',
    record: {datum: {keySpace: 'node'}},
    bindings,
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
  if (index < TIE_WORD_COUNT) { tieBits[tieBitsOffset + index] = 0u; }${
    props.unresolvedCount
      ? '\n  if (index == 0u) { unresolvedCount[unresolvedCountOffset] = 0u; }'
      : ''
  }`
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
    sourceCount?: number | GraphDataView<'uint32'>;
    hasCostLimit: boolean;
    costs: GraphDataView<'float32'>;
    levels: GraphDataView<'uint32'>;
    /** Optional per-node label; a source row roots only the node whose label is that row. */
    labels?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'sources', view: props.sources, type: 'u32', access: 'read'}
  ];
  if (props.labels) {
    bindings.push({name: 'labels', view: props.labels, type: 'u32', access: 'read'});
  }
  if (props.sourceCosts) {
    bindings.push({name: 'sourceCosts', view: props.sourceCosts, type: 'f32', access: 'read'});
  }
  if (typeof props.sourceCount === 'object') {
    bindings.push({name: 'sourceCount', view: props.sourceCount, type: 'u32', access: 'read'});
  }
  bindings.push(
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'levels', view: props.levels, type: 'u32', access: 'read_write'}
  );
  if (props.hasCostLimit) {
    bindings.push({name: 'control', view: props.state.control, type: 'u32', access: 'read'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-roots',
    record: {datum: {keySpace: 'node'}},
    bindings,
    invocationCount: props.sources.length,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;`,
    body: `${getStartCountGuardSource(props.sourceCount)}
  let node = sources[sourcesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${props.sourceCosts ? 'sourceCosts[sourceCostsOffset + index]' : '0.0'};
  if (!(cost >= 0.0)) { return; }
  ${props.hasCostLimit ? `if (cost > bitcast<f32>(control[controlOffset + ${props.phase.controlWordOffset + FRONTIER_CONTROL_PARAMETER_0}u])) { return; }` : ''}
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  ${props.labels ? 'if (labels[labelsOffset + node] != index) { return; }' : ''}
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
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-seed',
    record: {datum: {keySpace: 'node'}},
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'tieBits', view: props.tieBits, type: 'u32', access: 'read'},
      {name: 'predecessors', view: props.predecessors, type: 'u32', access: 'read'},
      {name: 'levels', view: props.levels, type: 'u32', access: 'read_write'},
      ...getFrontierBindings(graph, props.state, props.tiePhase, 0)
    ],
    invocationCount: props.nodeCount,
    declarations: `${REACHABILITY_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${props.edgeCount}u;
${TIE_EDGE_WGSL}
${getFrontierPushSource(
  props.tiePhase,
  props.nodeCount,
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
    record?: WGSLKernelProps<Parameters>['record'];
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
    record: props.record,
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
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'tie-predecessors',
    record: {datum: {keySpace: 'node'}},
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

/**
 * Counts reached nodes (finite cost) that ended without a predecessor although they are not roots:
 * a node is a root exactly when its tie level is 0 and it has no strict predecessor. Nonzero only
 * when the tie-level phase was truncated. Runs after the tie predecessor pass; `unresolvedCount` is
 * zeroed by the tie-initialize node. @internal
 */
export function createReachabilityUnresolvedCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    costs: GraphDataView<'float32'>;
    levels: GraphDataView<'uint32'>;
    predecessors: GraphDataView<'uint32'>;
    unresolvedCount: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'unresolved-count',
    bindings: [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'levels', view: props.levels, type: 'u32', access: 'read'},
      {name: 'predecessors', view: props.predecessors, type: 'u32', access: 'read'},
      {
        name: 'unresolvedCount',
        view: props.unresolvedCount,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    invocationCount: props.nodeCount,
    declarations: REACHABILITY_WGSL_CONSTANTS,
    body: `if (bitcast<u32>(costs[costsOffset + index]) >= INFINITY_BITS) { return; }
  if (predecessors[predecessorsOffset + index] != NONE) { return; }
  if (levels[levelsOffset + index] == 0u) { return; }
  atomicAdd(&unresolvedCount[unresolvedCountOffset], 1u);`
  });
}
