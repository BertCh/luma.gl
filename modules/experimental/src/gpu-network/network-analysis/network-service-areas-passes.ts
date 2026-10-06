// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createFrontierRoundNode,
  getFrontierBindings,
  getFrontierFinalizeSource,
  getFrontierPushSource,
  getFrontierResetInvocationCount,
  getFrontierResetSource,
  type FrontierPhase,
  type FrontierState
} from '../network-reachability/network-frontier';
import type {NetworkReachabilityGraphViews} from '../network-reachability/network-reachability-passes';

const OPERATION = 'GPUNetworkServiceAreas';

/** WGSL constants shared by service-area kernels. @internal */
const SERVICE_AREAS_WGSL_CONSTANTS = /* wgsl */ `
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`;

/**
 * Resets labels, the frontier state of the label phase (control words, dispatch slots, round
 * stamps), and the tight-edge bit set. @internal
 */
export function createServiceAreasLabelInitializeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    maxIterations: number;
    state: FrontierState;
    phase: FrontierPhase;
    labels: GraphDataView<'uint32'>;
    tightBits: GraphDataView<'uint32'>;
    activeIterations?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {state, phase} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'},
    {name: 'tightBits', view: props.tightBits, type: 'u32', access: 'read_write'},
    {name: 'queuedRound', view: state.queuedRound, type: 'u32', access: 'read_write'},
    {name: 'control', view: state.control, type: 'u32', access: 'read_write'},
    {name: 'dispatchEven', view: state.dispatchEven, type: 'u32', access: 'read_write'},
    {name: 'dispatchOdd', view: state.dispatchOdd, type: 'u32', access: 'read_write'}
  ];
  if (props.activeIterations) {
    bindings.push({
      name: 'activeIterations',
      view: props.activeIterations,
      type: 'u32',
      access: 'read'
    });
  }
  const limit = props.activeIterations
    ? `min(activeIterations[activeIterationsOffset], ${props.maxIterations}u)`
    : `${props.maxIterations}u`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'label-initialize',
    bindings,
    invocationCount: Math.max(
      props.nodeCount,
      props.tightBits.length,
      getFrontierResetInvocationCount(phase)
    ),
    declarations: `${SERVICE_AREAS_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const TIGHT_WORD_COUNT: u32 = ${props.tightBits.length}u;`,
    body: `${getFrontierResetSource(phase, {limit})}
  if (index < NODE_COUNT) {
    labels[labelsOffset + index] = NONE;
    // Stamps never need clearing within one encoding, but a previous encoding left them dirty.
    queuedRound[queuedRoundOffset + index] = 0u;
  }
  if (index < TIGHT_WORD_COUNT) { tightBits[tightBitsOffset + index] = 0u; }`
  });
}

/** Assigns facility rows to the labels of facility nodes whose cost equals the seed cost. @internal */
export function createServiceAreasLabelSeedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    facilities: GraphDataView<'uint32'>;
    facilityCosts?: GraphDataView<'float32'>;
    facilityCount?: GraphDataView<'uint32'>;
    costs: GraphDataView<'float32'>;
    labels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'facilities', view: props.facilities, type: 'u32', access: 'read'}
  ];
  if (props.facilityCosts) {
    bindings.push({name: 'facilityCosts', view: props.facilityCosts, type: 'f32', access: 'read'});
  }
  if (props.facilityCount) {
    bindings.push({name: 'facilityCount', view: props.facilityCount, type: 'u32', access: 'read'});
  }
  bindings.push(
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'labels', view: props.labels, type: 'atomic<u32>', access: 'read_write'}
  );
  // A facility whose cost exceeds the cost limit never equals the (limited) node cost, so the
  // equality test below also rejects it.
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'label-seed',
    bindings,
    invocationCount: props.facilities.length,
    declarations: `${SERVICE_AREAS_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;`,
    body: `${props.facilityCount ? 'if (index >= facilityCount[facilityCountOffset]) { return; }' : ''}
  let node = facilities[facilitiesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${props.facilityCosts ? 'facilityCosts[facilityCostsOffset + index]' : '0.0'};
  if (!(cost >= 0.0)) { return; }
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  if (cost != costs[costsOffset + node]) { return; }
  atomicMin(&labels[labelsOffset + node], index);`
  });
}

/**
 * Marks the tight edges `u -> v` (`costs[u] + weight == costs[v]` in f32, both finite) in a bit set
 * with one bit per edge. Label rounds then need neither the weights nor the costs. @internal
 */
export function createServiceAreasTightEdgesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    csr: NetworkReachabilityGraphViews;
    costs: GraphDataView<'float32'>;
    tightBits: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {csr} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'label-tight-edges',
    bindings: [
      {name: 'offsets', view: csr.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: csr.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: csr.weights, type: 'f32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'tightBits', view: props.tightBits, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.nodeCount,
    declarations: `${SERVICE_AREAS_WGSL_CONSTANTS}
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
    if (bitcast<u32>(targetCost) >= INFINITY_BITS) { continue; }
    if (sourceCost + weight != targetCost) { continue; }
    atomicOr(&tightBits[tightBitsOffset + (edgeIndex >> 5u)], 1u << (edgeIndex & 31u));
  }`
  });
}

/** Pushes every labelled node (the seeded facility nodes) into round 0 of the label phase. @internal */
export function createServiceAreasLabelFrontierSeedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    state: FrontierState;
    phase: FrontierPhase;
    labels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'label-frontier-seed',
    bindings: [
      {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
      ...getFrontierBindings(props.state, false)
    ],
    invocationCount: props.nodeCount,
    declarations: `${SERVICE_AREAS_WGSL_CONSTANTS}
${getFrontierPushSource(
  props.phase,
  props.nodeCount,
  0,
  graph.device.limits.maxComputeWorkgroupsPerDimension
)}`,
    body: `if (labels[labelsOffset + index] != NONE) {
    frontierPush(index);
  }`
  });
}

/**
 * One round of the label phase: min-label propagation over tight edges with an atomic minimum,
 * chaining up to `localIterations` hops per workgroup. @internal
 */
export function createServiceAreasLabelRoundNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    state: FrontierState;
    phase: FrontierPhase;
    round: number;
    localIterations: number;
    csr: Pick<NetworkReachabilityGraphViews, 'offsets' | 'neighbors'>;
    tightBits: GraphDataView<'uint32'>;
    labels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {csr} = props;
  return createFrontierRoundNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'label-round',
    state: props.state,
    phase: props.phase,
    round: props.round,
    localIterations: props.localIterations,
    bindings: [
      {name: 'offsets', view: csr.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: csr.neighbors, type: 'u32', access: 'read'},
      {name: 'tightBits', view: props.tightBits, type: 'u32', access: 'read'},
      {name: 'labels', view: props.labels, type: 'atomic<u32>', access: 'read_write'}
    ],
    declarations: `${SERVICE_AREAS_WGSL_CONSTANTS}
const EDGE_COUNT: u32 = ${csr.neighbors.length}u;`,
    // Tight edges are validated when the bit set is built, so targets are in range here.
    visitSource: `fn visit(node: u32) {
  let sourceLabel = atomicLoad(&labels[labelsOffset + node]);
  if (sourceLabel == NONE) { return; }
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (((tightBits[tightBitsOffset + (edgeIndex >> 5u)] >> (edgeIndex & 31u)) & 1u) == 0u) { continue; }
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    if (sourceLabel < atomicLoad(&labels[labelsOffset + targetNode])) {
      if (atomicMin(&labels[labelsOffset + targetNode], sourceLabel) > sourceLabel) {
        push(targetNode);
      }
    }
  }
}`
  });
}

/** Publishes `converged = reachabilityConverged && labelPhaseConverged`. @internal */
export function createServiceAreasFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    state: FrontierState;
    phase: FrontierPhase;
    reachabilityConverged: GraphDataView<'uint32'>;
    converged: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'finalize',
    bindings: [
      {name: 'control', view: props.state.control, type: 'u32', access: 'read'},
      {
        name: 'reachabilityConverged',
        view: props.reachabilityConverged,
        type: 'u32',
        access: 'read'
      },
      {name: 'converged', view: props.converged, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    body: `${getFrontierFinalizeSource(props.phase)}
  converged[convergedOffset] = select(
    0u,
    1u,
    reachabilityConverged[reachabilityConvergedOffset] != 0u && frontierConverged != 0u
  );`
  });
}
