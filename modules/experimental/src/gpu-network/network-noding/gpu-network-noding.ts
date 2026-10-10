// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {createSortedSegmentOffsetsNode} from '../../utils/sorted-segment-offsets';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPULineSplit} from '../../gpu-spatial-analysis/line-split/index';
import type {GPULineSplitPieces} from '../../gpu-spatial-analysis/line-split/index';
import type {GPUSpatialJoinLines} from '../../gpu-spatial-analysis/spatial-join/index';

const OPERATION = 'GPUNetworkNoding';

/** Value written to node and neighbor slots that hold no node. */
export const GPU_NETWORK_NODING_NONE = 0xffffffff;

/** Nodes of a noded network. */
export type GPUNetworkNodingNodes = {
  /** Node coordinates. Capacity (`nodeCount` rows of the CSR) is the length. */
  positions: GraphDataView<'float32x2'>;
  /** One-row scalar receiving `min(requiredCount, positions.length)`. */
  count: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of distinct nodes. */
  requiredCount?: GraphDataView<'uint32'>;
};

/** Edge list of a noded network: edge `e` is piece `e` of {@link GPUNetworkNodingProps.pieces}. */
export type GPUNetworkNodingEdges = {
  /** Node ID at the start of each piece. `GPU_NETWORK_NODING_NONE` beyond `pieces.status.count`. */
  fromNodes: GraphDataView<'uint32'>;
  /** Node ID at the end of each piece. */
  toNodes: GraphDataView<'uint32'>;
  /** Length of each piece along its vertices, in input coordinate units. */
  lengths: GraphDataView<'float32'>;
};

/**
 * Undirected CSR of the noded network: every edge appears twice, once in the row of each end node
 * (a loop appears twice in one row). Rows are ordered by edge, then start before end. The layout
 * is the one `GPUNetworkReachability` and `GPUNetworkServiceAreas` take as `offsets`,
 * `neighbors` and `weights`.
 */
export type GPUNetworkNodingCSR = {
  /** Row offsets with `nodes.positions.length + 1` entries. Rows of unused nodes are empty. */
  offsets: GraphDataView<'uint32'>;
  /** Neighbor node per entry. Capacity `2 * pieces.sourceIds.length`. */
  neighbors: GraphDataView<'uint32'>;
  /** Edge length per entry. 3.4028234e38 beyond the last valid entry, which no traversal reaches. */
  weights: GraphDataView<'float32'>;
  /** Optional edge row per entry. */
  edgeIds?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUNetworkNoding}.
 *
 * Per-frame: the contents of every input buffer and `tolerance`. Topology: view lengths,
 * capacities, `leafCapacity`, `spatialSort`.
 */
export type GPUNetworkNodingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-noding'`. */
  id?: string;
  /** Linestrings, in input coordinate units (planar meters for a routable network). */
  lines: GPUSpatialJoinLines;
  /** Capacity of the internal intersection pair list; see `GPULineSplitProps`. */
  intersectionCapacity: number;
  /**
   * One-row per-frame snap tolerance, at least 0. Piece end points whose tolerance cells
   * (`floor(coordinate / tolerance)`) coincide share one node, whose position is the first such end
   * point. 0 merges only bit-identical end points. Two end points closer than the tolerance but on
   * opposite sides of a cell boundary stay separate nodes.
   */
  tolerance: GraphDataView<'float32'>;
  /** Output pieces, the geometry of the edges. */
  pieces: GPULineSplitPieces;
  /** Output nodes. */
  nodes: GPUNetworkNodingNodes;
  /** Output edge list. */
  edges: GPUNetworkNodingEdges;
  /** Output CSR. */
  csr: GPUNetworkNodingCSR;
  /** Passed to `GPULineSplit`. */
  spatialSort?: boolean;
  /** Passed to `GPULineSplit`. */
  leafCapacity?: number;
  /** Optional one-row count of uncertified segment pairs; see `GPUSegmentIntersection`. */
  uncertainCount?: GraphDataView<'uint32'>;
  /** Optional one-row flag: 1 when pieces, vertices, intersections or nodes exceeded a capacity. */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Builds a routable planar network from linestrings.
 *
 * `GPULineSplit` cuts the lines at every intersection, so each piece runs between two nodes. Piece
 * end points are then snapped within `tolerance` and numbered: a stable two-pass `GPUSort` by
 * snapped (x, y) cell groups equal cells, and the group rank is the node ID. Node IDs are
 * deterministic (ordered by snapped x, then y) and dense; edges keep piece order. The edge list
 * and an undirected CSR (both directions, edge-length weights) come out ready for
 * `GPUNetworkReachability`, `GPUNetworkServiceAreas` and other CSR contributors.
 *
 * Nodes appear at line ends and at proper crossings, T junctions, shared vertices and overlap ends;
 * a line passing through another line's vertex without a vertex of its own is split there too.
 * Lines meeting only in the interior of both without a crossing do not exist in the plane, but
 * lines that stop within `tolerance` of another line's interior are not connected: tolerance merges
 * end points only. Bridges and tunnels that cross without connecting are not modelled.
 *
 * Capacity: `pieces` bounds edges and vertices, `nodes.positions` bounds nodes. When any capacity
 * overflows, edges incident to a node beyond the capacity are left out of the CSR and `overflow`
 * is set. Nothing is read back.
 */
export class GPUNetworkNoding implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkNodingProps;
  /** Node capacity. */
  readonly nodeCapacity: number;
  /** Edge capacity. */
  readonly edgeCapacity: number;

  constructor(props: GPUNetworkNodingProps) {
    this.id = props.id ?? 'network-noding';
    this.props = props;
    const {id} = this;
    const {nodes, edges, csr, pieces} = props;
    validatePackedView(props.tolerance, ['float32'], `${id} tolerance`);
    if (props.tolerance.length < 1) {
      throw new Error(`${id} tolerance must contain one float32 row`);
    }
    validatePackedView(nodes.positions, ['float32x2'], `${id} nodes.positions`);
    for (const [name, view] of [
      ['nodes.count', nodes.count],
      ['nodes.requiredCount', nodes.requiredCount],
      ['overflow', props.overflow],
      ['uncertainCount', props.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    this.nodeCapacity = nodes.positions.length;
    this.edgeCapacity = pieces.sourceIds.length;
    if (this.nodeCapacity < 1) {
      throw new Error(`${id} nodes.positions must be non-empty`);
    }
    if (csr.offsets.length !== this.nodeCapacity + 1) {
      throw new Error(`${id} csr.offsets length must be nodes.positions.length + 1`);
    }
    validatePackedUint32View(csr.offsets, `${id} csr.offsets`);
    validatePackedUint32View(csr.neighbors, `${id} csr.neighbors`);
    validatePackedView(csr.weights, ['float32'], `${id} csr.weights`);
    for (const [name, view] of [
      ['csr.neighbors', csr.neighbors],
      ['csr.weights', csr.weights],
      ['csr.edgeIds', csr.edgeIds]
    ] as const) {
      if (view && view.length !== this.edgeCapacity * 2) {
        throw new Error(`${id} ${name} length must be twice the edge capacity`);
      }
    }
    validatePackedUint32View(edges.fromNodes, `${id} edges.fromNodes`);
    validatePackedUint32View(edges.toNodes, `${id} edges.toNodes`);
    validatePackedView(edges.lengths, ['float32'], `${id} edges.lengths`);
    for (const [name, view] of [
      ['edges.fromNodes', edges.fromNodes],
      ['edges.toNodes', edges.toNodes],
      ['edges.lengths', edges.lengths]
    ] as const) {
      if (view.length !== this.edgeCapacity) {
        throw new Error(`${id} ${name} length must equal pieces.sourceIds.length`);
      }
    }
  }

  /** Returns line split, endpoint snapping, edge and CSR nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCapacity, edgeCapacity} = this;
    const {pieces, nodes: nodeOutputs, edges, csr, tolerance} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      tolerance,
      nodeOutputs.positions,
      nodeOutputs.count,
      nodeOutputs.requiredCount,
      edges.fromNodes,
      edges.toNodes,
      edges.lengths,
      csr.offsets,
      csr.neighbors,
      csr.weights,
      csr.edgeIds,
      props.overflow
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const endpointCount = edgeCapacity * 2;
    const T = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);

    const splitOverflow = pieces.status.overflow;
    nodes.push(
      ...new GPULineSplit({
        id: `${id}-split`,
        lines: props.lines,
        intersectionCapacity: props.intersectionCapacity,
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity,
        uncertainCount: props.uncertainCount,
        pieces
      }).getCommandNodes(graph)
    );

    // Snap keys per piece end point (index = 2 * edge + side).
    const keysX = T('keys-x', 'uint32', endpointCount);
    const keysY = T('keys-y', 'uint32', endpointCount);
    const endpointPoints = T('endpoint-points', 'float32x2', endpointCount);
    const endpointIndices = T('endpoint-indices', 'uint32', endpointCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-endpoint-keys`,
        operation: OPERATION,
        variant: 'endpoint-keys',
        bindings: [
          {name: 'pieceCount', view: pieces.status.count, type: 'u32', access: 'read'},
          {name: 'pieceOffsets', view: pieces.geometry.lineOffsets, type: 'u32', access: 'read'},
          {name: 'piecePositions', view: pieces.geometry.positions, type: 'f32', access: 'read'},
          {name: 'tolerance', view: tolerance, type: 'f32', access: 'read'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read_write'},
          {name: 'keysY', view: keysY, type: 'u32', access: 'read_write'},
          {name: 'endpointPoints', view: endpointPoints, type: 'f32', access: 'read_write'},
          {name: 'endpointIndices', view: endpointIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: endpointCount,
        declarations: `fn snapKey(value: f32, size: f32) -> u32 {
  if (size > 0.0) {
    let cell = clamp(floor(value / size), -1.0e9, 1.0e9);
    return min(u32(i32(cell) + 1073741824), 0xfffffffeu);
  }
  return min(bitcast<u32>(select(value, 0.0, value == 0.0)), 0xfffffffeu);
}`,
        body: `let edge = index / 2u;
  endpointIndices[endpointIndicesOffset + index] = index;
  var point = vec2f(0.0);
  var keyX = 0xffffffffu;
  var keyY = 0xffffffffu;
  if (edge < pieceCount[pieceCountOffset]) {
    var vertex = pieceOffsets[pieceOffsetsOffset + edge];
    if ((index & 1u) == 1u) { vertex = pieceOffsets[pieceOffsetsOffset + edge + 1u] - 1u; }
    point = vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
    let size = tolerance[toleranceOffset];
    keyX = snapKey(point.x, size);
    keyY = snapKey(point.y, size);
  }
  keysX[keysXOffset + index] = keyX;
  keysY[keysYOffset + index] = keyY;
  endpointPoints[endpointPointsOffset + index * 2u] = point.x;
  endpointPoints[endpointPointsOffset + index * 2u + 1u] = point.y;`
      })
    );

    // Stable two-pass sort by (x cell, y cell): sort by y first, then by x.
    const sortedKeysY = T('sorted-keys-y', 'uint32', endpointCount);
    const orderY = T('order-y', 'uint32', endpointCount);
    const gatheredKeysX = T('gathered-keys-x', 'uint32', endpointCount);
    const sortedKeysX = T('sorted-keys-x', 'uint32', endpointCount);
    const order = T('order', 'uint32', endpointCount);
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-y`,
        keys: keysY,
        values: endpointIndices,
        outputKeys: sortedKeysY,
        outputValues: orderY
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather-x`,
        operation: OPERATION,
        variant: 'gather-x',
        bindings: [
          {name: 'orderY', view: orderY, type: 'u32', access: 'read'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read'},
          {name: 'gatheredKeysX', view: gatheredKeysX, type: 'u32', access: 'read_write'}
        ],
        invocationCount: endpointCount,
        body: 'gatheredKeysX[gatheredKeysXOffset + index] = keysX[keysXOffset + orderY[orderYOffset + index]];'
      }),
      ...new GPUSort({
        id: `${id}-sort-x`,
        keys: gatheredKeysX,
        values: orderY,
        outputKeys: sortedKeysX,
        outputValues: order
      }).getCommandNodes(graph)
    );

    // Group heads, ranks (node IDs) and node positions.
    const headFlags = T('head-flags', 'uint32', endpointCount);
    const headRanks = T('head-ranks', 'uint32', endpointCount);
    const endpointNodes = T('endpoint-nodes', 'uint32', endpointCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-head-flags`,
        operation: OPERATION,
        variant: 'head-flags',
        bindings: [
          {name: 'order', view: order, type: 'u32', access: 'read'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read'},
          {name: 'keysY', view: keysY, type: 'u32', access: 'read'},
          {name: 'headFlags', view: headFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: endpointCount,
        body: `let current = order[orderOffset + index];
  var flag = select(0u, 1u, keysX[keysXOffset + current] != 0xffffffffu);
  if (flag == 1u && index > 0u) {
    let previous = order[orderOffset + index - 1u];
    if (keysX[keysXOffset + previous] == keysX[keysXOffset + current] &&
        keysY[keysYOffset + previous] == keysY[keysYOffset + current]) { flag = 0u; }
  }
  headFlags[headFlagsOffset + index] = flag;`
      }),
      ...new GPUScan({
        id: `${id}-head-scan`,
        input: headFlags,
        output: headRanks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-assign-nodes`,
        operation: OPERATION,
        variant: 'assign-nodes',
        bindings: [
          {name: 'order', view: order, type: 'u32', access: 'read'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read'},
          {name: 'headFlags', view: headFlags, type: 'u32', access: 'read'},
          {name: 'headRanks', view: headRanks, type: 'u32', access: 'read'},
          {name: 'endpointPoints', view: endpointPoints, type: 'f32', access: 'read'},
          {name: 'endpointNodes', view: endpointNodes, type: 'u32', access: 'read_write'},
          {name: 'nodePositions', view: nodeOutputs.positions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: endpointCount,
        declarations: `const NODE_CAPACITY: u32 = ${nodeCapacity}u;`,
        body: `let endpoint = order[orderOffset + index];
  if (keysX[keysXOffset + endpoint] == 0xffffffffu) {
    endpointNodes[endpointNodesOffset + endpoint] = 0xffffffffu;
    return;
  }
  let group = headRanks[headRanksOffset + index] + headFlags[headFlagsOffset + index] - 1u;
  endpointNodes[endpointNodesOffset + endpoint] = group;
  if (headFlags[headFlagsOffset + index] == 1u && group < NODE_CAPACITY) {
    nodePositions[nodePositionsOffset + group * 2u] = endpointPoints[endpointPointsOffset + endpoint * 2u];
    nodePositions[nodePositionsOffset + group * 2u + 1u] = endpointPoints[endpointPointsOffset + endpoint * 2u + 1u];
  }`
      })
    );

    // Edge list.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edges`,
        operation: OPERATION,
        variant: 'edges',
        bindings: [
          {name: 'pieceCount', view: pieces.status.count, type: 'u32', access: 'read'},
          {name: 'pieceOffsets', view: pieces.geometry.lineOffsets, type: 'u32', access: 'read'},
          {name: 'piecePositions', view: pieces.geometry.positions, type: 'f32', access: 'read'},
          {name: 'endpointNodes', view: endpointNodes, type: 'u32', access: 'read'},
          {name: 'fromNodes', view: edges.fromNodes, type: 'u32', access: 'read_write'},
          {name: 'toNodes', view: edges.toNodes, type: 'u32', access: 'read_write'},
          {name: 'lengths', view: edges.lengths, type: 'f32', access: 'read_write'}
        ],
        invocationCount: edgeCapacity,
        body: `var length = 0.0;
  var fromNode = 0xffffffffu;
  var toNode = 0xffffffffu;
  if (index < pieceCount[pieceCountOffset]) {
    fromNode = endpointNodes[endpointNodesOffset + index * 2u];
    toNode = endpointNodes[endpointNodesOffset + index * 2u + 1u];
    let first = pieceOffsets[pieceOffsetsOffset + index];
    let last = pieceOffsets[pieceOffsetsOffset + index + 1u];
    for (var vertex = first + 1u; vertex < last; vertex++) {
      let a = vec2f(piecePositions[piecePositionsOffset + (vertex - 1u) * 2u], piecePositions[piecePositionsOffset + (vertex - 1u) * 2u + 1u]);
      let b = vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
      length = length + distance(a, b);
    }
  }
  fromNodes[fromNodesOffset + index] = fromNode;
  toNodes[toNodesOffset + index] = toNode;
  lengths[lengthsOffset + index] = length;`
      })
    );

    // Undirected CSR: two directed entries per edge, grouped by start node. The endpoint sort above
    // already groups endpoints by node (ascending node ID) and keeps endpoints of one node in
    // ascending endpoint index (`2 * edge + side`, both radix sorts are stable), which is exactly
    // the CSR entry order. The CSR therefore needs no second sort: a scan compacts out the
    // endpoints of edges that lost a node to overflow and the remaining entries stay grouped.
    const csrFlags = T('csr-flags', 'uint32', endpointCount);
    const csrPositions = T('csr-positions', 'uint32', endpointCount);
    const sortedStarts = T('sorted-starts', 'uint32', endpointCount);
    const sortedEntries = T('sorted-entries', 'uint32', endpointCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-csr-flags`,
        operation: OPERATION,
        variant: 'csr-flags',
        bindings: [
          {name: 'order', view: order, type: 'u32', access: 'read'},
          {name: 'fromNodes', view: edges.fromNodes, type: 'u32', access: 'read'},
          {name: 'toNodes', view: edges.toNodes, type: 'u32', access: 'read'},
          {name: 'csrFlags', view: csrFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: endpointCount,
        declarations: `const NODE_CAPACITY: u32 = ${nodeCapacity}u;`,
        body: `let edge = order[orderOffset + index] / 2u;
  let isValid = fromNodes[fromNodesOffset + edge] < NODE_CAPACITY &&
    toNodes[toNodesOffset + edge] < NODE_CAPACITY;
  csrFlags[csrFlagsOffset + index] = select(0u, 1u, isValid);`
      }),
      ...new GPUScan({
        id: `${id}-csr-scan`,
        input: csrFlags,
        output: csrPositions,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-csr-compact`,
        operation: OPERATION,
        variant: 'csr-compact',
        bindings: [
          {name: 'order', view: order, type: 'u32', access: 'read'},
          {name: 'fromNodes', view: edges.fromNodes, type: 'u32', access: 'read'},
          {name: 'toNodes', view: edges.toNodes, type: 'u32', access: 'read'},
          {name: 'csrFlags', view: csrFlags, type: 'u32', access: 'read'},
          {name: 'csrPositions', view: csrPositions, type: 'u32', access: 'read'},
          {name: 'sortedStarts', view: sortedStarts, type: 'u32', access: 'read_write'},
          {name: 'sortedEntries', view: sortedEntries, type: 'u32', access: 'read_write'}
        ],
        invocationCount: endpointCount,
        declarations: `const NODE_CAPACITY: u32 = ${nodeCapacity}u;
const LAST: u32 = ${endpointCount - 1}u;`,
        // Valid entries scatter to their compacted position; every position past the valid total
        // gets the sentinel start, written by the invocation with that index.
        body: `let total = csrPositions[csrPositionsOffset + LAST] + csrFlags[csrFlagsOffset + LAST];
  if (csrFlags[csrFlagsOffset + index] != 0u) {
    let endpoint = order[orderOffset + index];
    let edge = endpoint / 2u;
    let position = csrPositions[csrPositionsOffset + index];
    sortedStarts[sortedStartsOffset + position] =
      select(fromNodes[fromNodesOffset + edge], toNodes[toNodesOffset + edge], (endpoint & 1u) == 1u);
    sortedEntries[sortedEntriesOffset + position] = endpoint;
  }
  if (index >= total) {
    sortedStarts[sortedStartsOffset + index] = NODE_CAPACITY;
    sortedEntries[sortedEntriesOffset + index] = 0xffffffffu;
  }`
      }),
      createSortedSegmentOffsetsNode<Parameters>(graph, {
        id: `${id}-csr-offsets`,
        operation: OPERATION,
        variant: 'csr-offsets',
        segmentCount: nodeCapacity,
        sortedKeys: sortedStarts,
        segmentOffsets: csr.offsets
      })
    );
    const writeBindings: WGSLKernelBinding[] = [
      {name: 'sortedStarts', view: sortedStarts, type: 'u32', access: 'read'},
      {name: 'sortedEntries', view: sortedEntries, type: 'u32', access: 'read'},
      {name: 'fromNodes', view: edges.fromNodes, type: 'u32', access: 'read'},
      {name: 'toNodes', view: edges.toNodes, type: 'u32', access: 'read'},
      {name: 'lengths', view: edges.lengths, type: 'f32', access: 'read'},
      {name: 'neighbors', view: csr.neighbors, type: 'u32', access: 'read_write'},
      {name: 'weights', view: csr.weights, type: 'f32', access: 'read_write'}
    ];
    if (csr.edgeIds) {
      writeBindings.push({name: 'edgeIds', view: csr.edgeIds, type: 'u32', access: 'read_write'});
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-csr-entries`,
        operation: OPERATION,
        variant: 'csr-entries',
        bindings: writeBindings,
        invocationCount: endpointCount,
        declarations: `const NODE_CAPACITY: u32 = ${nodeCapacity}u;`,
        body: `var neighbor = 0xffffffffu;
  var weight = 3.4028234e38;
  var edgeId = 0xffffffffu;
  if (sortedStarts[sortedStartsOffset + index] < NODE_CAPACITY) {
    let entry = sortedEntries[sortedEntriesOffset + index];
    let edge = entry / 2u;
    neighbor = select(toNodes[toNodesOffset + edge], fromNodes[fromNodesOffset + edge], (entry & 1u) == 1u);
    weight = lengths[lengthsOffset + edge];
    edgeId = edge;
  }
  neighbors[neighborsOffset + index] = neighbor;
  weights[weightsOffset + index] = weight;
  ${csr.edgeIds ? 'edgeIds[edgeIdsOffset + index] = edgeId;' : ''}`
      })
    );

    // Scalars.
    const scalarBindings: WGSLKernelBinding[] = [
      {name: 'headFlags', view: headFlags, type: 'u32', access: 'read'},
      {name: 'headRanks', view: headRanks, type: 'u32', access: 'read'},
      {name: 'splitOverflow', view: splitOverflow, type: 'u32', access: 'read'},
      {name: 'nodeCount', view: nodeOutputs.count, type: 'u32', access: 'read_write'}
    ];
    if (nodeOutputs.requiredCount) {
      scalarBindings.push({
        name: 'requiredCount',
        view: nodeOutputs.requiredCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (props.overflow) {
      scalarBindings.push({
        name: 'overflow',
        view: props.overflow,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scalars`,
        operation: OPERATION,
        variant: 'scalars',
        bindings: scalarBindings,
        invocationCount: 1,
        declarations: `const NODE_CAPACITY: u32 = ${nodeCapacity}u;
const LAST: u32 = ${endpointCount - 1}u;`,
        body: `let total = headRanks[headRanksOffset + LAST] + headFlags[headFlagsOffset + LAST];
  nodeCount[nodeCountOffset] = min(total, NODE_CAPACITY);
  ${nodeOutputs.requiredCount ? 'requiredCount[requiredCountOffset] = total;' : ''}
  ${props.overflow ? 'overflow[overflowOffset] = select(0u, 1u, total > NODE_CAPACITY || splitOverflow[splitOverflowOffset] != 0u);' : ''}`
      })
    );
    return nodes;
  }
}
