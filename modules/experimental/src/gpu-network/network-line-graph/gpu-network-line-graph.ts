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
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH} from './network-line-graph-parameters';

const OPERATION = 'GPUNetworkLineGraph';

/**
 * Properties for {@link GPUNetworkLineGraph}.
 *
 * Compile-time: node, edge and arc-capacity counts and whether `bannedTurns` exists. Per-frame:
 * the CSR contents (weights, closures), node positions, `parameters` and the banned turns.
 */
export type GPUNetworkLineGraphProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-line-graph'`. */
  id?: string;
  /** Directed CSR row offsets with `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** CSR destination node per edge. Edges with an out-of-range destination have no turns. */
  neighbors: GraphDataView<'uint32'>;
  /** Non-negative travel cost per edge. Negative or NaN edges stay impassable in the line graph. */
  weights: GraphDataView<'float32'>;
  /** Planar node positions, one per node, used for the turn angles. */
  nodePositions: GraphDataView<'float32x2'>;
  /**
   * Per-frame `GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH` float32 elements written with
   * `getGPUNetworkLineGraphParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Optional banned turns as interleaved edge-row pairs `[fromEdge0, toEdge0, fromEdge1, ...]`.
   * `parameters.bannedTurnCount` pairs are active. Every arc checks every active pair, so this is
   * intended for tens to a few thousand bans.
   */
  bannedTurns?: GraphDataView<'uint32'>;
  /** Line graph CSR row offsets, `edgeCount + 1` rows: line node `e` is base edge row `e`. */
  lineOffsets: GraphDataView<'uint32'>;
  /**
   * Line graph CSR destinations: the following base edge row of every allowed turn. Its length is
   * the compile-time arc capacity; arcs past it are dropped and raise `overflow`.
   */
  lineNeighbors: GraphDataView<'uint32'>;
  /**
   * Line graph arc costs, `weights[following edge] + turn cost`, or -1 when the following edge is
   * impassable. Same length as `lineNeighbors`.
   */
  lineWeights: GraphDataView<'float32'>;
  /** Optional one-row arc count, `min(total, capacity)`. */
  arcCount?: GraphDataView<'uint32'>;
  /** Optional one-row flag: 1 when the allowed turns exceeded the arc capacity. */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Builds the turn-restriction line graph (edge-based graph) of a directed road CSR.
 *
 * Every base directed edge becomes a line node; edge `e = (u, v)` gets one arc to every edge
 * `f = (v, w)` whose turn is allowed. The arc costs `weights[f]` plus a turn cost from the turn
 * angle (`atan2` of the cross and dot product of the two unit directions, positive to the left):
 * `angleCost * |angle|`, plus the left or right extra beyond `straightAngle`, with an angle of
 * `uTurnAngle` or more costing `uTurnCost` instead (a negative `uTurnCost` bans U-turns). Listed
 * banned turns are removed. Rows keep their arcs in ascending following-edge order.
 *
 * The output is an ordinary CSR, so `GPUNetworkReachability`, `GPUNetworkCostMatrix` and the other
 * routing contributors run on it unchanged. Seed a search from an origin node `s` with the line
 * nodes `offsets[s] .. offsets[s + 1]` at cost `weights[e]`. The cost of line node `e` is then the
 * cost of arriving at the head of `e` along `e`, with all turns paid.
 *
 * Everything is recomputed on every encoding, so live weights, closures, turn settings and bans
 * need no recompile. The CSR must have `edgeCount` and `arcCapacity` fixed at compile time; size
 * the capacity from the node degrees, for example `sum over e of outDegree(head(e))`.
 */
export class GPUNetworkLineGraph implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkLineGraphProps;
  /** Number of base nodes. */
  readonly nodeCount: number;
  /** Number of base edges, the line graph node count. */
  readonly edgeCount: number;
  /** Compile-time arc capacity. */
  readonly arcCapacity: number;

  constructor(props: GPUNetworkLineGraphProps) {
    this.id = props.id ?? 'network-line-graph';
    this.props = props;
    const {id} = this;
    for (const [name, view] of [
      ['offsets', props.offsets],
      ['neighbors', props.neighbors],
      ['bannedTurns', props.bannedTurns],
      ['lineOffsets', props.lineOffsets],
      ['lineNeighbors', props.lineNeighbors],
      ['arcCount', props.arcCount],
      ['overflow', props.overflow]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['weights', props.weights],
      ['parameters', props.parameters],
      ['lineWeights', props.lineWeights]
    ] as const) {
      validatePackedView(view, ['float32'], `${id} ${name}`);
    }
    validatePackedView(props.nodePositions, ['float32x2'], `${id} nodePositions`);
    this.nodeCount = props.nodePositions.length;
    this.edgeCount = props.neighbors.length;
    this.arcCapacity = props.lineNeighbors.length;
    if (this.nodeCount < 1 || this.edgeCount < 1) {
      throw new Error(`${id} requires at least one node and one edge`);
    }
    if (props.offsets.length !== this.nodeCount + 1) {
      throw new Error(`${id} offsets must contain one more row than nodePositions`);
    }
    if (props.weights.length !== this.edgeCount) {
      throw new Error(`${id} weights length must equal neighbors length`);
    }
    if (props.parameters.length < GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH} elements`
      );
    }
    if (props.bannedTurns && props.bannedTurns.length % 2 !== 0) {
      throw new Error(`${id} bannedTurns length must be even`);
    }
    if (props.lineOffsets.length !== this.edgeCount + 1) {
      throw new Error(`${id} lineOffsets must contain edgeCount + 1 rows`);
    }
    if (props.lineWeights.length !== this.arcCapacity) {
      throw new Error(`${id} lineWeights length must equal lineNeighbors length`);
    }
    if (this.arcCapacity < 1) {
      throw new Error(`${id} lineNeighbors must hold at least one arc`);
    }
    for (const [name, view] of [
      ['arcCount', props.arcCount],
      ['overflow', props.overflow]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns `edge-vectors`, `count`, the `GPUScan` nodes, optional `publish`, `fill-neighbors`,
   * `fill-weights` and `clamp` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, edgeCount, arcCapacity} = this;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const hasBans = Boolean(props.bannedTurns);
    const edgeVectors = createTransientView(graph, `${id}-edge-vectors`, 'float32', 2 * edgeCount);
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', edgeCount + 1);
    const declarations = `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const ARC_CAPACITY: u32 = ${arcCapacity}u;
${TURN_WGSL}
${
  hasBans
    ? `fn isBannedTurn(fromEdge: u32, toEdge: u32) -> bool {
  let bannedCount = u32(parameters[parametersOffset + 5u]);
  for (var ban = 0u; ban < bannedCount; ban++) {
    if (bannedTurns[bannedTurnsOffset + 2u * ban] == fromEdge &&
        bannedTurns[bannedTurnsOffset + 2u * ban + 1u] == toEdge) {
      return true;
    }
  }
  return false;
}`
    : `fn isBannedTurn(fromEdge: u32, toEdge: u32) -> bool {
  return false;
}`
}
// Only a negative U-turn cost makes getTurnCost negative (every other term is non-negative), so
// with U-turns priced the angle is never needed, and a turn whose edges point forward (positive dot
// product) is below any U-turn threshold clearly above a right angle (1.6 rad leaves a margin for
// backend atan2 error). Both skip the atan2 that the cost would need; the remaining arcs evaluate
// the exact cost as before.
fn isAllowedTurn(fromEdge: u32, toEdge: u32) -> bool {
  if (isBannedTurn(fromEdge, toEdge)) {
    return false;
  }
  if (parameters[parametersOffset + 3u] >= 0.0) {
    return true;
  }
  if (parameters[parametersOffset + 6u] > 1.6) {
    let a = vec2<f32>(
      edgeVectors[edgeVectorsOffset + 2u * fromEdge],
      edgeVectors[edgeVectorsOffset + 2u * fromEdge + 1u]
    );
    let b = vec2<f32>(
      edgeVectors[edgeVectorsOffset + 2u * toEdge],
      edgeVectors[edgeVectorsOffset + 2u * toEdge + 1u]
    );
    if (dot(a, a) >= 0.5 && dot(b, b) >= 0.5 && dot(a, b) > 0.0) {
      return true;
    }
  }
  return getTurnCost(fromEdge, toEdge) >= 0.0;
}`;
    const turnBindings: WGSLKernelBinding[] = [
      {name: 'edgeVectors', view: edgeVectors, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (props.bannedTurns) {
      turnBindings.push({
        name: 'bannedTurns',
        view: props.bannedTurns,
        type: 'u32',
        access: 'read'
      });
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-vectors`,
        operation: OPERATION,
        variant: 'edge-vectors',
        bindings: [
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
          {name: 'nodePositions', view: props.nodePositions, type: 'f32', access: 'read'},
          {name: 'edgeVectors', view: edgeVectors, type: 'f32', access: 'read_write'}
        ],
        invocationCount: nodeCount,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;`,
        body: `let start = nodePositions[nodePositionsOffset + 2u * index];
  let startY = nodePositions[nodePositionsOffset + 2u * index + 1u];
  let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    let headNode = neighbors[neighborsOffset + edge];
    var direction = vec2<f32>(0.0);
    if (headNode < NODE_COUNT) {
      let delta = vec2<f32>(
        nodePositions[nodePositionsOffset + 2u * headNode] - start,
        nodePositions[nodePositionsOffset + 2u * headNode + 1u] - startY
      );
      let edgeLength = sqrt(dot(delta, delta));
      if (edgeLength > 0.0) {
        direction = delta / edgeLength;
      }
    }
    edgeVectors[edgeVectorsOffset + 2u * edge] = direction.x;
    edgeVectors[edgeVectorsOffset + 2u * edge + 1u] = direction.y;
  }`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'count',
        bindings: [
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
          ...turnBindings,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: edgeCount + 1,
        declarations,
        body: `var count = 0u;
  if (index < EDGE_COUNT) {
    let head = neighbors[neighborsOffset + index];
    if (head < NODE_COUNT) {
      let end = min(offsets[offsetsOffset + head + 1u], EDGE_COUNT);
      for (var next = offsets[offsetsOffset + head]; next < end; next++) {
        if (isAllowedTurn(index, next)) {
          count++;
        }
      }
    }
  }
  counts[countsOffset + index] = count;`
      })
    );

    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: props.lineOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    if (props.arcCount || props.overflow) {
      const publishBindings: WGSLKernelBinding[] = [
        {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'}
      ];
      if (props.arcCount) {
        publishBindings.push({
          name: 'arcCount',
          view: props.arcCount,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (props.overflow) {
        publishBindings.push({
          name: 'overflow',
          view: props.overflow,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          variant: 'publish',
          bindings: publishBindings,
          invocationCount: 1,
          declarations: `const EDGE_COUNT: u32 = ${edgeCount}u;
const ARC_CAPACITY: u32 = ${arcCapacity}u;`,
          body: `let total = lineOffsets[lineOffsetsOffset + EDGE_COUNT];
  ${props.arcCount ? 'arcCount[arcCountOffset] = min(total, ARC_CAPACITY);' : ''}
  ${props.overflow ? 'overflow[overflowOffset] = select(0u, 1u, total > ARC_CAPACITY);' : ''}`
        })
      );
    }

    // Fusing the weight fill into the neighbor fill needs one more storage binding than the
    // separate kernels. The fused kernel evaluates each arc's turn cost once and already knows the
    // arc's source edge, so it drops the second angle evaluation and the per-arc binary search over
    // `lineOffsets` that the standalone weight kernel needs to recover the source edge.
    const fillBindingCount = 8 + (props.bannedTurns ? 1 : 0);
    const fuseWeights = fillBindingCount <= graph.device.limits.maxStorageBuffersPerShaderStage;
    const fillWeightStatement = `let baseCost = weights[weightsOffset + next];
        lineWeights[lineWeightsOffset + slot] =
          select(-1.0, baseCost + max(turnCost, 0.0), baseCost >= 0.0);`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fill-neighbors`,
        operation: OPERATION,
        variant: fuseWeights ? 'fill' : 'fill-neighbors',
        bindings: [
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
          ...turnBindings,
          {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {name: 'lineNeighbors', view: props.lineNeighbors, type: 'u32', access: 'read_write'},
          ...(fuseWeights
            ? [
                {name: 'weights', view: props.weights, type: 'f32', access: 'read'} as const,
                {
                  name: 'lineWeights',
                  view: props.lineWeights,
                  type: 'f32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: edgeCount,
        declarations,
        body: `let head = neighbors[neighborsOffset + index];
  if (head >= NODE_COUNT) {
    return;
  }
  var slot = lineOffsets[lineOffsetsOffset + index];
  let end = min(offsets[offsetsOffset + head + 1u], EDGE_COUNT);
  for (var next = offsets[offsetsOffset + head]; next < end; next++) {
    ${
      fuseWeights
        ? `if (isBannedTurn(index, next)) {
      continue;
    }
    let turnCost = getTurnCost(index, next);
    if (turnCost >= 0.0) {
      if (slot < ARC_CAPACITY) {
        lineNeighbors[lineNeighborsOffset + slot] = next;
        ${fillWeightStatement}
      }
      slot++;
    }`
        : `if (isAllowedTurn(index, next)) {
      if (slot < ARC_CAPACITY) {
        lineNeighbors[lineNeighborsOffset + slot] = next;
      }
      slot++;
    }`
    }
  }`
      })
    );

    if (!fuseWeights) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fill-weights`,
          operation: OPERATION,
          variant: 'fill-weights',
          bindings: [
            {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
            {name: 'lineNeighbors', view: props.lineNeighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
            ...turnBindings,
            {name: 'lineWeights', view: props.lineWeights, type: 'f32', access: 'read_write'}
          ],
          invocationCount: arcCapacity,
          declarations,
          body: `if (index >= lineOffsets[lineOffsetsOffset + EDGE_COUNT]) {
    return;
  }
  var low = 0u;
  var high = EDGE_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle + 1u] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  let next = lineNeighbors[lineNeighborsOffset + index];
  let baseCost = weights[weightsOffset + next];
  lineWeights[lineWeightsOffset + index] =
    select(-1.0, baseCost + max(getTurnCost(low, next), 0.0), baseCost >= 0.0);`
        })
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clamp`,
        operation: OPERATION,
        variant: 'clamp',
        bindings: [
          {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: edgeCount + 1,
        declarations: `const ARC_CAPACITY: u32 = ${arcCapacity}u;`,
        body: `lineOffsets[lineOffsetsOffset + index] =
    min(lineOffsets[lineOffsetsOffset + index], ARC_CAPACITY);`
      })
    );
    return nodes;
  }
}

/** Turn geometry and cost helpers, reading `edgeVectors` and `parameters`. */
const TURN_WGSL = /* wgsl */ `
fn getTurnAngle(fromEdge: u32, toEdge: u32) -> f32 {
  let a = vec2<f32>(
    edgeVectors[edgeVectorsOffset + 2u * fromEdge],
    edgeVectors[edgeVectorsOffset + 2u * fromEdge + 1u]
  );
  let b = vec2<f32>(
    edgeVectors[edgeVectorsOffset + 2u * toEdge],
    edgeVectors[edgeVectorsOffset + 2u * toEdge + 1u]
  );
  if (dot(a, a) < 0.5 || dot(b, b) < 0.5) {
    return 0.0;
  }
  let cross = a.x * b.y - a.y * b.x;
  let projection = dot(a, b);
  // atan2 with a signed-zero x is unreliable on some backends; right angles are common on grids.
  if (projection == 0.0) {
    return select(-1.5707963, 1.5707963, cross > 0.0);
  }
  return atan2(cross, projection);
}
// Turn cost of the turn fromEdge -> toEdge, or a negative value when U-turns are banned.
fn getTurnCost(fromEdge: u32, toEdge: u32) -> f32 {
  let angle = getTurnAngle(fromEdge, toEdge);
  let magnitude = abs(angle);
  if (magnitude >= parameters[parametersOffset + 6u]) {
    return parameters[parametersOffset + 3u];
  }
  var cost = parameters[parametersOffset] * magnitude;
  if (magnitude > parameters[parametersOffset + 4u]) {
    cost += select(
      parameters[parametersOffset + 2u],
      parameters[parametersOffset + 1u],
      angle > 0.0
    );
  }
  return cost;
}`;

/** Returns every read-only view. */
function getInputs(props: GPUNetworkLineGraphProps): (GraphDataView | undefined)[] {
  return [
    props.offsets,
    props.neighbors,
    props.weights,
    props.nodePositions,
    props.parameters,
    props.bannedTurns
  ];
}

/** Returns every writable view. */
function getOutputs(props: GPUNetworkLineGraphProps): (GraphDataView | undefined)[] {
  return [
    props.lineOffsets,
    props.lineNeighbors,
    props.lineWeights,
    props.arcCount,
    props.overflow
  ];
}
