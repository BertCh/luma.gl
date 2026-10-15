// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
} from '../../analysis-core/contracts/spatial-context';
import {createSaturatedScanTotalNodes} from '../../utils/scan-total';
import {
  createTransientView,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateMaximumCount,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUNetworkTopology} from '../../analysis-core/network-topology';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology
} from '../../analysis-core/network-topology';
import type {
  GPUSpatialContext,
  GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {GEODESIC_WGSL} from '../../gpu-spatial-analysis/geometry-measures/geodesic-wgsl';
import {GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH} from './network-line-graph-parameters';

const LINE_GRAPH_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
});

const OPERATION = 'GPUNetworkLineGraph';

/**
 * Largest `topology` edge count of {@link GPUNetworkLineGraph} on planar coordinates on `device`:
 * the edge direction vectors hold 8 bytes per base edge in one storage binding, so 16777216 on a
 * default device (128 MiB bindings).
 */
export function getGPUNetworkLineGraphMaximumEdgeCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Largest `topology` edge count of {@link GPUNetworkLineGraph} with a great-circle
 * `spatialContext` on `device`: the departure and arrival direction vectors hold 16 bytes per base
 * edge in one storage binding, so 8388608 on a default device (128 MiB bindings).
 */
export function getGPUNetworkLineGraphMaximumGeodesicEdgeCount(device: Device): number {
  return getStorageBindingElementLimit(device, 16);
}

/**
 * Properties for {@link GPUNetworkLineGraph}.
 *
 * Compile-time: node, edge and arc-capacity counts and whether `bannedTurns` exists. Per-frame:
 * the CSR contents (weights, closures), node positions, `parameters` and the banned turns.
 */
export type GPUNetworkLineGraphProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-line-graph'`. */
  id?: string;
  /**
   * Directed road network. Edges with an out-of-range destination have no turns. Weights are the
   * non-negative travel cost per edge; negative or NaN edges stay impassable in the line graph.
   * `nodePositions` (one per node) give the turn angles. At most
   * {@link getGPUNetworkLineGraphMaximumEdgeCount} edges (16777216 on a default device), or
   * {@link getGPUNetworkLineGraphMaximumGeodesicEdgeCount} (8388608) with a great-circle
   * `spatialContext`.
   */
  topology: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  /**
   * Coordinates of `topology.nodePositions`. Omitted means native planar coordinates (x east, y north).
   * `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` takes degrees and measures
   * turn angles between great-circle bearings at the turn node. Meter and kilometer contexts
   * produce the same angles; travel costs keep the caller's units. Local planar frames must
   * be azimuthal-equidistant, not Lambert equal-area.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Per-frame `GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH` float32 elements written with
   * `getGPUNetworkLineGraphParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Optional banned turns as interleaved edge-row pairs `[fromEdge0, toEdge0, fromEdge1, ...]`.
   * `parameters.bannedTurnCount` pairs are active, in any order. The graph sorts the active pairs
   * by `fromEdge` on the GPU every frame and each candidate turn binary-searches them, so a turn
   * test costs `O(log bans + bans per from-edge)`, which suits city-scale restriction sets.
   */
  bannedTurns?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUNetworkLineGraphOutput;
};

/** GPU-written output of {@link GPUNetworkLineGraph}. */
export type GPUNetworkLineGraphOutput = {
  /**
   * The line graph as a CSR network, so it feeds `GPUNetworkReachability.topology` and the other
   * routing contributors directly.
   */
  topology: {
    /** Line graph CSR row offsets, `edgeCount + 1` rows: line node `e` is base edge row `e`. */
    offsets: GraphDataView<'uint32'>;
    /**
     * Line graph CSR destinations: the following base edge row of every allowed turn. Its length is
     * the compile-time arc capacity; arcs past it are dropped and raise `output.status.overflow`. Rows past the
     * arc total hold `0xffffffff`.
     */
    neighbors: GraphDataView<'uint32'>;
    /**
     * Line graph arc costs, `weights[following edge] + turn cost`, or -1 when the following edge is
     * impassable. Same length as `neighbors`; rows past the arc total hold -1.
     */
    weights: GraphDataView<'float32'>;
  };
  /** Optional one-row arc count, `min(total, capacity)`. */
  arcCount?: GraphDataView<'uint32'>;
  /** GPU-written status words. */
  status: {
    /**
     * One-row flag: 1 when the allowed turns exceeded the arc capacity, so arcs were dropped.
     * Required (library rule 9): the arc total is per-frame data, so any capacity can be exceeded.
     */
    overflow: GraphDataView<'uint32'>;
  };
};

/**
 * Builds the turn-restriction line graph (edge-based graph) of a directed road CSR.
 *
 * Edge rows past `offsets[nodeCount]` (a CSR shorter than its compile-time edge count) are not
 * edges: their line nodes get no arcs.
 *
 * Every base directed edge becomes a line node; edge `e = (u, v)` gets one arc to every edge
 * `f = (v, w)` whose turn is allowed. The arc costs `weights[f]` plus a turn cost from the turn
 * angle (`atan2` of the cross and dot product of the two unit directions, positive to the left):
 * `angleCost * |angle|`, plus the left or right extra beyond `straightAngle`, with an angle of
 * `uTurnAngle` or more costing `uTurnCost` instead (a negative `uTurnCost` bans U-turns). Listed
 * banned turns are removed. Rows keep their arcs in ascending following-edge order.
 *
 * Turn directions are planar by default. With a great-circle `spatialContext` positions are
 * longitude and latitude degrees, and the angle is between the arrival bearing of `e` and the
 * departure bearing of `f` at `v` (great-circle bearings), so turns are not skewed by the
 * east-west stretch of degrees away from the equator.
 *
 * The output is an ordinary CSR, so `GPUNetworkReachability`, `GPUNetworkCostMatrix` and the other
 * routing contributors run on it unchanged. Start a search from an origin node `s` with the line
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
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUNetworkLineGraphProps) {
    this.id = props.id ?? 'network-line-graph';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      LINE_GRAPH_SPATIAL_CONTEXT_SUPPORT
    );
    const {output} = props;
    const {nodeCount, edgeCount} = validateGPUNetworkTopology(id, props.topology, {
      weights: true,
      nodePositions: true
    });
    for (const [name, view] of [
      ['bannedTurns', props.bannedTurns],
      ['output.topology.offsets', output.topology.offsets],
      ['output.topology.neighbors', output.topology.neighbors],
      ['output.arcCount', output.arcCount],
      ['output.status.overflow', output.status?.overflow]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['parameters', props.parameters],
      ['output.topology.weights', output.topology.weights]
    ] as const) {
      validatePackedView(view, ['float32'], `${id} ${name}`);
    }
    this.nodeCount = nodeCount;
    this.edgeCount = edgeCount;
    this.arcCapacity = output.topology.neighbors.length;
    if (this.arcCapacity < 1) {
      throw new Error(`${id} output.topology.neighbors must hold at least one arc`);
    }
    if (!output.status?.overflow) {
      throw new Error(`${id} output.status.overflow is required`);
    }
    if (props.parameters.length < GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH} elements`
      );
    }
    if (props.bannedTurns && props.bannedTurns.length % 2 !== 0) {
      throw new Error(`${id} bannedTurns length must be even`);
    }
    if (output.topology.offsets.length !== this.edgeCount + 1) {
      throw new Error(`${id} output.topology.offsets must contain edgeCount + 1 rows`);
    }
    if (output.topology.weights.length !== this.arcCapacity) {
      throw new Error(`${id} output.topology.weights length must equal neighbors length`);
    }
    for (const [name, view] of [
      ['output.arcCount', output.arcCount],
      ['output.status.overflow', output.status?.overflow]
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
   * Returns the ban `ban-keys`, `GPUSort` and `ban-pairs` nodes when `bannedTurns` is given, then
   * `edge-vectors`, `count`, the `GPUScan` nodes, optional `publish`, the tail clears,
   * `fill-neighbors`, optional `fill-weights` and `clamp` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, edgeCount, arcCapacity} = this;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    if (this.spatialContext.coordinateSpace === 'longitude-latitude') {
      validateMaximumCount(
        id,
        'topology edges with a great-circle spatialContext',
        edgeCount,
        getGPUNetworkLineGraphMaximumGeodesicEdgeCount(graph.device)
      );
    } else {
      validateMaximumCount(
        id,
        'topology edges',
        edgeCount,
        getGPUNetworkLineGraphMaximumEdgeCount(graph.device)
      );
    }
    const nodes: GPUCommandNode<Parameters>[] = [];
    const bannedTurns = props.bannedTurns?.length ? props.bannedTurns : undefined;
    const hasBans = Boolean(bannedTurns);
    const banCapacity = bannedTurns ? bannedTurns.length / 2 : 0;
    const sortedBans = bannedTurns ? this.addBanSortNodes(graph, nodes, bannedTurns) : undefined;
    // Planar edges keep one direction; great-circle edges keep the departure direction at the
    // tail and the arrival direction at the head, which differ along a great circle.
    const isGeodesic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const edgeVectorStride = isGeodesic ? 4 : 2;
    const edgeVectors = createTransientView(
      graph,
      `${id}-edge-vectors`,
      'float32',
      edgeVectorStride * edgeCount
    );
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', edgeCount + 1);
    const declarations = `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const ARC_CAPACITY: u32 = ${arcCapacity}u;
const EDGE_VECTOR_STRIDE: u32 = ${edgeVectorStride}u;
const ARRIVAL_OFFSET: u32 = ${edgeVectorStride - 2}u;
${TURN_WGSL}
${
  hasBans
    ? `const BAN_CAPACITY: u32 = ${banCapacity}u;
// sortedBans holds (fromEdge, toEdge) pairs in ascending fromEdge order; inactive rows have
// fromEdge = EDGE_COUNT and sort last. Find the first pair of fromEdge, then scan its run.
fn isBannedTurn(fromEdge: u32, toEdge: u32) -> bool {
  var low = 0u;
  var high = BAN_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedBans[sortedBansOffset + 2u * middle] < fromEdge) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  for (var ban = low; ban < BAN_CAPACITY; ban++) {
    if (sortedBans[sortedBansOffset + 2u * ban] != fromEdge) {
      break;
    }
    if (sortedBans[sortedBansOffset + 2u * ban + 1u] == toEdge) {
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
    let a = getArrivalDirection(fromEdge);
    let b = getDepartureDirection(toEdge);
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
    if (sortedBans) {
      turnBindings.push({
        name: 'sortedBans',
        view: sortedBans,
        type: 'u32',
        access: 'read'
      });
    }
    // A line node is a base edge row inside the live CSR range offsets[0]..offsets[NODE_COUNT].
    // Rows past it hold stale words (transients alias), so they get no arcs (review B-09).
    const liveEdgeSource = `fn isLiveEdge(edge: u32) -> bool {
  return edge >= offsets[offsetsOffset] && edge < min(offsets[offsetsOffset + NODE_COUNT], EDGE_COUNT);
}`;

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-vectors`,
        operation: OPERATION,
        variant: 'edge-vectors',
        bindings: [
          {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
          {name: 'nodePositions', view: props.topology.nodePositions, type: 'f32', access: 'read'},
          {name: 'edgeVectors', view: edgeVectors, type: 'f32', access: 'read_write'}
        ],
        invocationCount: nodeCount,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
${isGeodesic ? GEODESIC_WGSL : ''}`,
        body: isGeodesic
          ? GEODESIC_EDGE_VECTORS_WGSL
          : `let start = nodePositions[nodePositionsOffset + 2u * index];
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
          {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
          ...turnBindings,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: edgeCount + 1,
        declarations: `${declarations}
${liveEdgeSource}`,
        body: `var count = 0u;
  if (index < EDGE_COUNT && isLiveEdge(index)) {
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
        output: props.output.topology.offsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    const scanTotal = createTransientView(graph, `${id}-scan-total`, 'uint32', 1);
    nodes.push(
      ...createSaturatedScanTotalNodes<Parameters>(graph, {
        id: id,
        operation: OPERATION,
        counts: counts,
        offsets: props.output.topology.offsets,
        rowCount: edgeCount,
        total: scanTotal
      })
    );

    if (props.output.arcCount || props.output.status.overflow) {
      const publishBindings: WGSLKernelBinding[] = [
        {name: 'scanTotal', view: scanTotal, type: 'u32', access: 'read'}
      ];
      if (props.output.arcCount) {
        publishBindings.push({
          name: 'arcCount',
          view: props.output.arcCount,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (props.output.status.overflow) {
        publishBindings.push({
          name: 'overflow',
          view: props.output.status.overflow,
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
          body: `let total = scanTotal[scanTotalOffset];
  ${props.output.arcCount ? 'arcCount[arcCountOffset] = min(total, ARC_CAPACITY);' : ''}
  ${props.output.status.overflow ? 'overflow[overflowOffset] = select(0u, 1u, total > ARC_CAPACITY);' : ''}`
        })
      );
    }

    // Fusing the weight fill into the neighbor fill needs one more storage binding than the
    // separate kernels. The fused kernel evaluates each arc's turn cost once and already knows the
    // arc's source edge, so it drops the second angle evaluation and the per-arc binary search over
    // `lineOffsets` that the standalone weight kernel needs to recover the source edge.
    const fillBindingCount = 8 + (sortedBans ? 1 : 0);
    // Arc rows past the total are never written by the fill; clear them first (rule 11).
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-line-neighbors`,
        operation: OPERATION,
        view: props.output.topology.neighbors,
        type: 'u32',
        value: '0xffffffffu'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-line-weights`,
        operation: OPERATION,
        view: props.output.topology.weights,
        type: 'f32',
        value: '-1.0'
      })
    );
    const fuseWeights = fillBindingCount <= graph.device.limits.maxStorageBuffersPerShaderStage;
    const fillWeightStatement = `let baseCost = weights[weightsOffset + next];
        lineWeights[lineWeightsOffset + slot] =
          select(-1.0, baseCost + max(turnCost, 0.0), baseCost >= 0.0);`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fill-neighbors`,
        operation: OPERATION,
        variant: fuseWeights ? 'fill' : 'fill-neighbors',
        record: {
          datum: {keySpace: 'edge'},
          formula: fuseWeights ? 'gpu-network.line-graph.arc-weight' : undefined
        },
        bindings: [
          {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
          ...turnBindings,
          {name: 'lineOffsets', view: props.output.topology.offsets, type: 'u32', access: 'read'},
          {
            name: 'lineNeighbors',
            view: props.output.topology.neighbors,
            type: 'u32',
            access: 'read_write'
          },
          ...(fuseWeights
            ? [
                {
                  name: 'weights',
                  view: props.topology.weights,
                  type: 'f32',
                  access: 'read'
                } as const,
                {
                  name: 'lineWeights',
                  view: props.output.topology.weights,
                  type: 'f32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: edgeCount,
        declarations: `${declarations}
${liveEdgeSource}`,
        body: `if (!isLiveEdge(index)) {
    return;
  }
  let head = neighbors[neighborsOffset + index];
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
          record: {datum: {keySpace: 'arc'}, formula: 'gpu-network.line-graph.arc-weight'},
          bindings: [
            {name: 'lineOffsets', view: props.output.topology.offsets, type: 'u32', access: 'read'},
            {
              name: 'lineNeighbors',
              view: props.output.topology.neighbors,
              type: 'u32',
              access: 'read'
            },
            {name: 'weights', view: props.topology.weights, type: 'f32', access: 'read'},
            ...turnBindings,
            {
              name: 'lineWeights',
              view: props.output.topology.weights,
              type: 'f32',
              access: 'read_write'
            }
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
          {
            name: 'lineOffsets',
            view: props.output.topology.offsets,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: edgeCount + 1,
        declarations: `const ARC_CAPACITY: u32 = ${arcCapacity}u;`,
        body: `lineOffsets[lineOffsetsOffset + index] =
    min(lineOffsets[lineOffsetsOffset + index], ARC_CAPACITY);`
      })
    );
    return nodes;
  }

  /**
   * Sorts the active banned turns by `fromEdge` into interleaved `(fromEdge, toEdge)` pairs.
   * Inactive rows (at or past `bannedTurnCount`) and out-of-range `fromEdge` values get key
   * `EDGE_COUNT`, which no line node matches.
   */
  private addBanSortNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    nodes: GPUCommandNode<Parameters>[],
    bannedTurns: GraphDataView<'uint32'>
  ): GraphDataView<'uint32'> {
    const {id, edgeCount} = this;
    const banCapacity = bannedTurns.length / 2;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const keys = u32('ban-keys', banCapacity);
    const values = u32('ban-values', banCapacity);
    const sortedKeys = u32('ban-sorted-keys', banCapacity);
    const sortedValues = u32('ban-sorted-values', banCapacity);
    const sortedBans = u32('sorted-bans', 2 * banCapacity);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ban-keys`,
        operation: OPERATION,
        variant: 'ban-keys',
        bindings: [
          {name: 'parameters', view: this.props.parameters, type: 'f32', access: 'read'},
          {name: 'bannedTurns', view: bannedTurns, type: 'u32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'values', view: values, type: 'u32', access: 'read_write'}
        ],
        invocationCount: banCapacity,
        declarations: `const EDGE_COUNT: u32 = ${edgeCount}u;`,
        body: `let bannedCount = u32(max(parameters[parametersOffset + 5u], 0.0));
  let fromEdge = bannedTurns[bannedTurnsOffset + 2u * index];
  let isActive = index < bannedCount && fromEdge < EDGE_COUNT;
  keys[keysOffset + index] = select(EDGE_COUNT, fromEdge, isActive);
  values[valuesOffset + index] = bannedTurns[bannedTurnsOffset + 2u * index + 1u];`
      }),
      ...new GPUSort({
        id: `${id}-ban-sort`,
        keys,
        values,
        outputKeys: sortedKeys,
        outputValues: sortedValues,
        keyBits: Math.max(1, Math.ceil(Math.log2(edgeCount + 1)))
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ban-pairs`,
        operation: OPERATION,
        variant: 'ban-pairs',
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'sortedValues', view: sortedValues, type: 'u32', access: 'read'},
          {name: 'sortedBans', view: sortedBans, type: 'u32', access: 'read_write'}
        ],
        invocationCount: banCapacity,
        body: `sortedBans[sortedBansOffset + 2u * index] = sortedKeys[sortedKeysOffset + index];
  sortedBans[sortedBansOffset + 2u * index + 1u] = sortedValues[sortedValuesOffset + index];`
      })
    );
    return sortedBans;
  }
}

/**
 * Edge-vectors body for longitude/latitude: per edge the unit departure direction at the tail and
 * the unit arrival direction at the head, `(east, north)` from great-circle bearings. The arrival
 * direction is the reverse of the bearing from the head back to the tail. Degenerate edges keep 0.
 */
const GEODESIC_EDGE_VECTORS_WGSL = /* wgsl */ `let start = vec2<f32>(
    nodePositions[nodePositionsOffset + 2u * index],
    nodePositions[nodePositionsOffset + 2u * index + 1u]
  );
  let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    let headNode = neighbors[neighborsOffset + edge];
    var departure = vec2<f32>(0.0);
    var arrival = vec2<f32>(0.0);
    if (headNode < NODE_COUNT) {
      let head = vec2<f32>(
        nodePositions[nodePositionsOffset + 2u * headNode],
        nodePositions[nodePositionsOffset + 2u * headNode + 1u]
      );
      if (geodesicCentralAngle(start, head) > 0.0) {
        let departureBearing = geodesicInitialBearingDegrees(start, head) * GEODESIC_DEGREES_TO_RADIANS;
        let returnBearing = geodesicInitialBearingDegrees(head, start) * GEODESIC_DEGREES_TO_RADIANS;
        departure = vec2<f32>(sin(departureBearing), cos(departureBearing));
        arrival = -vec2<f32>(sin(returnBearing), cos(returnBearing));
      }
    }
    edgeVectors[edgeVectorsOffset + 4u * edge] = departure.x;
    edgeVectors[edgeVectorsOffset + 4u * edge + 1u] = departure.y;
    edgeVectors[edgeVectorsOffset + 4u * edge + 2u] = arrival.x;
    edgeVectors[edgeVectorsOffset + 4u * edge + 3u] = arrival.y;
  }`;

/**
 * Turn geometry and cost helpers, reading `edgeVectors` (`EDGE_VECTOR_STRIDE` floats per edge, the
 * arrival direction at `ARRIVAL_OFFSET`) and `parameters`.
 */
const TURN_WGSL = /* wgsl */ `
fn getDepartureDirection(edge: u32) -> vec2<f32> {
  let base = edgeVectorsOffset + EDGE_VECTOR_STRIDE * edge;
  return vec2<f32>(edgeVectors[base], edgeVectors[base + 1u]);
}
fn getArrivalDirection(edge: u32) -> vec2<f32> {
  let base = edgeVectorsOffset + EDGE_VECTOR_STRIDE * edge + ARRIVAL_OFFSET;
  return vec2<f32>(edgeVectors[base], edgeVectors[base + 1u]);
}
fn getTurnAngle(fromEdge: u32, toEdge: u32) -> f32 {
  let a = getArrivalDirection(fromEdge);
  let b = getDepartureDirection(toEdge);
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
  return [...getGPUNetworkTopologyViews(props.topology), props.parameters, props.bannedTurns];
}

/** Returns every writable view. */
function getOutputs(props: GPUNetworkLineGraphProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.topology.offsets,
    output.topology.neighbors,
    output.topology.weights,
    output.arcCount,
    output.status.overflow
  ];
}
