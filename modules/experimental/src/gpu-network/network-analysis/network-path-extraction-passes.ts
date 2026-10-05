// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUNetworkPathExtraction';

/** WGSL constants shared by path extraction kernels. @internal */
const PATH_WGSL_CONSTANTS = /* wgsl */ `
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`;

/** Measures every target path: length, found flag, cost, and the truncation flag. @internal */
export function createPathMeasureNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    maxPathLength: number;
    predecessors: GraphDataView<'uint32'>;
    costs: GraphDataView<'float32'>;
    targets: GraphDataView<'uint32'>;
    targetCount?: GraphDataView<'uint32'>;
    nodeLengths: GraphDataView<'uint32'>;
    status: GraphDataView<'uint32'>;
    pathFound?: GraphDataView<'uint32'>;
    pathCosts?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'predecessors', view: props.predecessors, type: 'u32', access: 'read'},
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'targets', view: props.targets, type: 'u32', access: 'read'},
    {name: 'nodeLengths', view: props.nodeLengths, type: 'u32', access: 'read_write'},
    {name: 'status', view: props.status, type: 'atomic<u32>', access: 'read_write'}
  ];
  if (props.targetCount) {
    bindings.push({name: 'targetCount', view: props.targetCount, type: 'u32', access: 'read'});
  }
  if (props.pathFound) {
    bindings.push({name: 'pathFound', view: props.pathFound, type: 'u32', access: 'read_write'});
  }
  if (props.pathCosts) {
    bindings.push({name: 'pathCosts', view: props.pathCosts, type: 'u32', access: 'read_write'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'measure',
    bindings,
    invocationCount: props.targets.length,
    declarations: `${PATH_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const MAXIMUM_PATH_LENGTH: u32 = ${props.maxPathLength}u;`,
    body: `var length = 0u;
  var found = false;
  var costBits = INFINITY_BITS;
  let isActive = ${props.targetCount ? 'index < targetCount[targetCountOffset]' : 'true'};
  let targetNode = targets[targetsOffset + index];
  if (isActive && targetNode < NODE_COUNT) {
    let targetCost = costs[costsOffset + targetNode];
    if ((bitcast<u32>(targetCost) & 0x7fffffffu) < INFINITY_BITS) {
      var node = targetNode;
      var walked = 1u;
      // 0 walking, 1 root reached, 2 broken chain, 3 too long.
      var state = 0u;
      for (var step = 0u; step <= MAXIMUM_PATH_LENGTH; step++) {
        let predecessor = predecessors[predecessorsOffset + node];
        if (predecessor == NONE) {
          state = 1u;
          break;
        }
        if (predecessor >= NODE_COUNT) {
          state = 2u;
          break;
        }
        if (walked >= MAXIMUM_PATH_LENGTH) {
          state = 3u;
          break;
        }
        node = predecessor;
        walked++;
      }
      if (state == 1u) {
        found = true;
        length = walked;
        costBits = bitcast<u32>(targetCost);
      }
      if (state == 3u) {
        atomicOr(&status[statusOffset], 1u);
      }
    }
  }
  nodeLengths[nodeLengthsOffset + index] = length;
  ${props.pathFound ? 'pathFound[pathFoundOffset + index] = select(0u, 1u, found);' : ''}
  ${props.pathCosts ? 'pathCosts[pathCostsOffset + index] = costBits;' : ''}`
  });
}

/** Derives per-target edge counts `max(nodes - 1, 0)` from node counts. @internal */
export function createPathEdgeLengthsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeLengths: GraphDataView<'uint32'>;
    edgeLengths: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'edge-lengths',
    bindings: [
      {name: 'nodeLengths', view: props.nodeLengths, type: 'u32', access: 'read'},
      {name: 'edgeLengths', view: props.edgeLengths, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.nodeLengths.length,
    body: `let length = nodeLengths[nodeLengthsOffset + index];
  edgeLengths[edgeLengthsOffset + index] = select(0u, length - 1u, length > 0u);`
  });
}

/** Computes totals and optional unclamped path offsets for nodes and edges. @internal */
export function createPathTotalsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    targetCapacity: number;
    nodes: {
      lengths?: GraphDataView<'uint32'>;
      starts?: GraphDataView<'uint32'>;
      total: GraphDataView<'uint32'>;
      pathOffsets?: GraphDataView<'uint32'>;
    };
    edges?: {
      lengths?: GraphDataView<'uint32'>;
      starts?: GraphDataView<'uint32'>;
      total: GraphDataView<'uint32'>;
      pathOffsets?: GraphDataView<'uint32'>;
    };
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [];
  const statements: string[] = [];
  const capacity = props.targetCapacity;
  for (const [prefix, side] of [
    ['node', props.nodes],
    ['edge', props.edges]
  ] as const) {
    if (!side) {
      continue;
    }
    if (side.lengths && side.starts) {
      bindings.push(
        {name: `${prefix}Lengths`, view: side.lengths, type: 'u32', access: 'read'},
        {name: `${prefix}Starts`, view: side.starts, type: 'u32', access: 'read'}
      );
    }
    bindings.push({name: `${prefix}Total`, view: side.total, type: 'u32', access: 'read_write'});
    if (side.pathOffsets) {
      bindings.push({
        name: `${prefix}PathOffsets`,
        view: side.pathOffsets,
        type: 'u32',
        access: 'read_write'
      });
    }
    const totalExpression =
      capacity > 0
        ? `${prefix}Starts[${prefix}StartsOffset + ${capacity - 1}u] + ${prefix}Lengths[${prefix}LengthsOffset + ${capacity - 1}u]`
        : '0u';
    statements.push(`if (index == ${capacity}u) {
    let total = ${totalExpression};
    ${prefix}Total[${prefix}TotalOffset] = total;
    ${side.pathOffsets ? `${prefix}PathOffsets[${prefix}PathOffsetsOffset + ${capacity}u] = total;` : ''}
  }
  ${side.pathOffsets && capacity > 0 ? `if (index < ${capacity}u) { ${prefix}PathOffsets[${prefix}PathOffsetsOffset + index] = ${prefix}Starts[${prefix}StartsOffset + index]; }` : ''}`);
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'totals',
    bindings,
    invocationCount: capacity + 1,
    body: statements.join('\n  ')
  });
}

/** Writes node IDs of every found path in place, ordered source to target. @internal */
export function createPathWriteNodesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    predecessors: GraphDataView<'uint32'>;
    targets: GraphDataView<'uint32'>;
    nodeLengths: GraphDataView<'uint32'>;
    nodeStarts: GraphDataView<'uint32'>;
    nodeIds?: GraphDataView<'uint32'>;
    ids: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'predecessors', view: props.predecessors, type: 'u32', access: 'read'},
    {name: 'targets', view: props.targets, type: 'u32', access: 'read'},
    {name: 'nodeLengths', view: props.nodeLengths, type: 'u32', access: 'read'},
    {name: 'nodeStarts', view: props.nodeStarts, type: 'u32', access: 'read'},
    {name: 'ids', view: props.ids, type: 'u32', access: 'read_write'}
  ];
  if (props.nodeIds) {
    bindings.push({name: 'nodeIds', view: props.nodeIds, type: 'u32', access: 'read'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'write-nodes',
    bindings,
    invocationCount: props.targets.length,
    declarations: `const CAPACITY: u32 = ${props.capacity}u;`,
    body: `let length = nodeLengths[nodeLengthsOffset + index];
  if (length == 0u) { return; }
  let start = nodeStarts[nodeStartsOffset + index];
  var node = targets[targetsOffset + index];
  for (var step = 0u; step < length; step++) {
    let position = start + length - 1u - step;
    if (position < CAPACITY) {
      ids[idsOffset + position] = ${props.nodeIds ? 'nodeIds[nodeIdsOffset + node]' : 'node'};
    }
    node = predecessors[predecessorsOffset + node];
  }`
  });
}

/** Records the tail and head node of every path link at its output slot. @internal */
export function createPathWriteLinksNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    predecessors: GraphDataView<'uint32'>;
    targets: GraphDataView<'uint32'>;
    nodeLengths: GraphDataView<'uint32'>;
    edgeStarts: GraphDataView<'uint32'>;
    linkTails: GraphDataView<'uint32'>;
    linkHeads: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'write-links',
    bindings: [
      {name: 'predecessors', view: props.predecessors, type: 'u32', access: 'read'},
      {name: 'targets', view: props.targets, type: 'u32', access: 'read'},
      {name: 'nodeLengths', view: props.nodeLengths, type: 'u32', access: 'read'},
      {name: 'edgeStarts', view: props.edgeStarts, type: 'u32', access: 'read'},
      {name: 'linkTails', view: props.linkTails, type: 'u32', access: 'read_write'},
      {name: 'linkHeads', view: props.linkHeads, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.targets.length,
    declarations: `const CAPACITY: u32 = ${props.capacity}u;`,
    body: `let length = nodeLengths[nodeLengthsOffset + index];
  if (length < 2u) { return; }
  let start = edgeStarts[edgeStartsOffset + index];
  var head = targets[targetsOffset + index];
  for (var step = 0u; step + 1u < length; step++) {
    let tail = predecessors[predecessorsOffset + head];
    let position = start + length - 2u - step;
    if (position < CAPACITY) {
      linkTails[linkTailsOffset + position] = tail;
      linkHeads[linkHeadsOffset + position] = head;
    }
    head = tail;
  }`
  });
}

/** Resolves every recorded link to a CSR edge index or stable edge ID. @internal */
export function createPathResolveEdgesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    edgeCount: number;
    linkTails: GraphDataView<'uint32'>;
    linkHeads: GraphDataView<'uint32'>;
    costs: GraphDataView<'float32'>;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    weights: GraphDataView<'float32'>;
    edgeIds?: GraphDataView<'uint32'>;
    ids: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'linkTails', view: props.linkTails, type: 'u32', access: 'read'},
    {name: 'linkHeads', view: props.linkHeads, type: 'u32', access: 'read'},
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
    {name: 'ids', view: props.ids, type: 'u32', access: 'read_write'}
  ];
  if (props.edgeIds) {
    bindings.push({name: 'edgeIds', view: props.edgeIds, type: 'u32', access: 'read'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'resolve-edges',
    bindings,
    invocationCount: props.ids.length,
    declarations: `${PATH_WGSL_CONSTANTS}
const NODE_COUNT: u32 = ${props.nodeCount}u;
const EDGE_COUNT: u32 = ${props.edgeCount}u;`,
    body: `let tail = linkTails[linkTailsOffset + index];
  var result = NONE;
  if (tail < NODE_COUNT) {
    let head = linkHeads[linkHeadsOffset + index];
    let tailCost = costs[costsOffset + tail];
    let headCost = costs[costsOffset + head];
    let rangeStart = min(offsets[offsetsOffset + tail], EDGE_COUNT);
    let rangeEnd = min(offsets[offsetsOffset + tail + 1u], EDGE_COUNT);
    var exact = NONE;
    var fallback = NONE;
    for (var edge = rangeStart; edge < rangeEnd; edge++) {
      if (neighbors[neighborsOffset + edge] != head) { continue; }
      if (fallback == NONE) { fallback = edge; }
      let weight = weights[weightsOffset + edge];
      if (weight >= 0.0 && tailCost + weight == headCost) {
        exact = edge;
        break;
      }
    }
    result = select(exact, fallback, exact == NONE);
    ${props.edgeIds ? 'if (result != NONE) { result = edgeIds[edgeIdsOffset + result]; }' : ''}
  }
  ids[idsOffset + index] = result;`
  });
}
