// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUBoundedDispatchLayout,
  GPUCommandGraph,
  GPUCommandNode,
  GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {NetworkReachabilityGraphViews} from '../network-reachability/network-reachability-passes';

const OPERATION = 'GPUNetworkServiceAreas';

/** WGSL constants shared by service-area kernels. @internal */
const SERVICE_AREAS_WGSL_CONSTANTS = /* wgsl */ `
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`;

/** Resets labels, both label frontiers, the label status words, and the label relax dispatch. @internal */
export function createServiceAreasLabelInitializeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    maxIterations: number;
    relaxLayout: GPUBoundedDispatchLayout;
    labels: GraphDataView<'uint32'>;
    frontierA: GraphDataView<'uint32'>;
    frontierB: GraphDataView<'uint32'>;
    status: GraphDataView<'uint32'>;
    dispatch: GraphDataView<'uint32'>;
    activeIterations?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'},
    {name: 'frontierA', view: props.frontierA, type: 'u32', access: 'read_write'},
    {name: 'frontierB', view: props.frontierB, type: 'u32', access: 'read_write'},
    {name: 'status', view: props.status, type: 'u32', access: 'read_write'},
    {name: 'dispatch', view: props.dispatch, type: 'u32', access: 'read_write'}
  ];
  if (props.activeIterations) {
    bindings.push({
      name: 'activeIterations',
      view: props.activeIterations,
      type: 'u32',
      access: 'read'
    });
  }
  const {x, y, z} = props.relaxLayout;
  const limit = props.activeIterations
    ? `min(activeIterations[activeIterationsOffset], ${props.maxIterations}u)`
    : `${props.maxIterations}u`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'label-initialize',
    bindings,
    invocationCount: props.nodeCount,
    declarations: SERVICE_AREAS_WGSL_CONSTANTS,
    body: `if (index == 0u) {
    let enabled = ${limit} > 0u;
    status[statusOffset + 0u] = 0u;
    status[statusOffset + 1u] = select(0u, 1u, enabled);
    status[statusOffset + 2u] = 0u;
    status[statusOffset + 3u] = 0u;
    dispatch[dispatchOffset + 0u] = select(0u, ${x}u, enabled);
    dispatch[dispatchOffset + 1u] = ${y}u;
    dispatch[dispatchOffset + 2u] = ${z}u;
  }
  labels[labelsOffset + index] = NONE;
  frontierA[frontierAOffset + index] = 0u;
  // Clear both frontiers: an iteration-limited previous encoding may leave the next one dirty.
  frontierB[frontierBOffset + index] = 0u;`
  });
}

/** Seeds facility labels at nodes whose cost equals the facility seed cost. @internal */
export function createServiceAreasLabelSeedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    facilities: GraphDataView<'uint32'>;
    facilityCosts?: GraphDataView<'float32'>;
    facilityCount?: GraphDataView<'uint32'>;
    costLimit?: GraphDataView<'float32'>;
    costs: GraphDataView<'float32'>;
    labels: GraphDataView<'uint32'>;
    frontier: GraphDataView<'uint32'>;
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
  if (props.costLimit) {
    bindings.push({name: 'costLimit', view: props.costLimit, type: 'f32', access: 'read'});
  }
  bindings.push(
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'labels', view: props.labels, type: 'atomic<u32>', access: 'read_write'},
    {name: 'frontier', view: props.frontier, type: 'atomic<u32>', access: 'read_write'}
  );
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
  ${props.costLimit ? 'if (cost > costLimit[costLimitOffset]) { return; }' : ''}
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  if (cost != costs[costsOffset + node]) { return; }
  atomicMin(&labels[labelsOffset + node], index);
  atomicStore(&frontier[frontierOffset + node], 1u);`
  });
}

/** One GPU-gated min-label propagation over tight edges of the current frontier. @internal */
export function createServiceAreasLabelRelaxNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    csr: NetworkReachabilityGraphViews;
    costs: GraphDataView<'float32'>;
    current: GraphDataView<'uint32'>;
    next: GraphDataView<'uint32'>;
    labels: GraphDataView<'uint32'>;
    status: GraphDataView<'uint32'>;
    dispatch: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {csr} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'label-relax',
    bindings: [
      {name: 'offsets', view: csr.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: csr.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: csr.weights, type: 'f32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'current', view: props.current, type: 'u32', access: 'read_write'},
      {name: 'next', view: props.next, type: 'atomic<u32>', access: 'read_write'},
      {name: 'labels', view: props.labels, type: 'atomic<u32>', access: 'read_write'},
      {name: 'status', view: props.status, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.nodeCount,
    condition: {
      id: `${props.id}-gate`,
      source: 'gpu',
      mode: 'indirect',
      buffer: props.dispatch.buffer,
      byteOffset: props.dispatch.byteOffset
    },
    extraResources: [{buffer: props.dispatch, usage: 'indirect'}],
    declarations: `${SERVICE_AREAS_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${csr.neighbors.length}u;`,
    body: `if (current[currentOffset + index] == 0u) { return; }
  // Only this invocation touches current[index] in this pass.
  current[currentOffset + index] = 0u;
  let sourceCost = costs[costsOffset + index];
  if (bitcast<u32>(sourceCost) >= INFINITY_BITS) { return; }
  let sourceLabel = atomicLoad(&labels[labelsOffset + index]);
  if (sourceLabel == NONE) { return; }
  let firstEdge = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  var improved = false;
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    let weight = weights[weightsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT || !(weight >= 0.0)) { continue; }
    let targetCost = costs[costsOffset + targetNode];
    if (bitcast<u32>(targetCost) >= INFINITY_BITS) { continue; }
    if (sourceCost + weight != targetCost) { continue; }
    if (sourceLabel < atomicLoad(&labels[labelsOffset + targetNode])) {
      if (atomicMin(&labels[labelsOffset + targetNode], sourceLabel) > sourceLabel) {
        atomicStore(&next[nextOffset + targetNode], 1u);
        improved = true;
      }
    }
  }
  if (improved) { atomicStore(&status[statusOffset], 1u); }`
  });
}

/** Publishes `converged = reachabilityConverged && labelConverged`. @internal */
export function createServiceAreasFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    status: GraphDataView<'uint32'>;
    reachabilityConverged: GraphDataView<'uint32'>;
    converged: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'finalize',
    bindings: [
      {name: 'status', view: props.status, type: 'u32', access: 'read'},
      {
        name: 'reachabilityConverged',
        view: props.reachabilityConverged,
        type: 'u32',
        access: 'read'
      },
      {name: 'converged', view: props.converged, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    body: `converged[convergedOffset] = select(
    0u,
    1u,
    reachabilityConverged[reachabilityConvergedOffset] != 0u && status[statusOffset + 3u] != 0u
  );`
  });
}
