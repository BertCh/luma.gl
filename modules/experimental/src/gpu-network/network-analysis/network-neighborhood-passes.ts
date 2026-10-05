// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUNetworkNeighborhood';

/**
 * Converts relaxed unit-weight costs into hop distances and an optional 0/1 node mask.
 *
 * A node is inside the ego network when its cost is finite and at most `min(hops, maximumHops)`.
 * Every output word is rewritten on every encoding.
 *
 * @internal
 */
export function createNeighborhoodHopsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    maximumHops: number;
    costs: GraphDataView<'float32'>;
    hops: GraphDataView<'uint32'>;
    hopDistances: GraphDataView<'uint32'>;
    nodeMask?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'hopsIn', view: props.hops, type: 'u32', access: 'read'},
    {name: 'hopDistances', view: props.hopDistances, type: 'u32', access: 'read_write'}
  ];
  if (props.nodeMask) {
    bindings.push({name: 'nodeMask', view: props.nodeMask, type: 'u32', access: 'read_write'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'hops',
    bindings,
    invocationCount: props.nodeCount,
    declarations: `const NONE: u32 = 0xffffffffu;
const MAXIMUM_HOPS: u32 = ${props.maximumHops}u;`,
    body: `let limit = min(hopsIn[hopsInOffset], MAXIMUM_HOPS);
  let cost = costs[costsOffset + index];
  // NaN and +Infinity compare false and stay outside the ego network.
  let inside = cost <= f32(limit);
  hopDistances[hopDistancesOffset + index] = select(NONE, u32(select(0.0, cost, inside)), inside);
  ${props.nodeMask ? 'nodeMask[nodeMaskOffset + index] = select(0u, 1u, inside);' : ''}`
  });
}

/**
 * Marks every edge whose source and destination are both inside the ego network.
 *
 * Each edge belongs to exactly one CSR row, so rows written by different invocations never
 * overlap when offsets are monotonic. The caller clears the edge mask first.
 *
 * @internal
 */
export function createNeighborhoodEdgeMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    edgeCount: number;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    nodeMask: GraphDataView<'uint32'>;
    edgeMask: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'edge-mask',
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      {name: 'nodeMask', view: props.nodeMask, type: 'u32', access: 'read'},
      {name: 'edgeMask', view: props.edgeMask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.nodeCount,
    declarations: `const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${props.edgeCount}u;`,
    body: `if (nodeMask[nodeMaskOffset + index] == 0u) {
    return;
  }
  let rowStart = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let rowEnd = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = rowStart; edge < rowEnd; edge++) {
    let neighbor = neighbors[neighborsOffset + edge];
    var inside = 0u;
    if (neighbor < NODE_COUNT && nodeMask[nodeMaskOffset + neighbor] != 0u) {
      inside = 1u;
    }
    edgeMask[edgeMaskOffset + edge] = inside;
  }`
  });
}
