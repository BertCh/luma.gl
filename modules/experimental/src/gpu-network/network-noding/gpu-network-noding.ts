// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {createSortedSegmentOffsetsNode} from '../../utils/sorted-segment-offsets';
import {
  validateCapacity,
  validatePackedView,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getGPULineSplitMaximumPieceBound,
  GPULineSplit
} from '../../gpu-spatial-analysis/line-split/index';
import {validateLineNodingBounds} from '../../gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import type {
  GPULineSplitPieces,
  GPULineSplitProps
} from '../../gpu-spatial-analysis/line-split/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getWorkgroupCounterKernelSource,
  WORKGROUP_COUNTER_WORKGROUP_SIZE
} from '../network-workgroup-counters';

const OPERATION = 'GPUNetworkNoding';

/**
 * Largest `intersectionCapacity` of {@link GPUNetworkNoding} on `device`: the internal line split
 * bounds its pieces by `lines + 4 * intersectionCapacity` (see
 * {@link getGPULineSplitMaximumPieceBound}), so a quarter of that bound, 1048576 on a default
 * device (128 MiB bindings). `lines + 4 * intersectionCapacity` is validated too.
 */
export function getGPUNetworkNodingMaximumIntersectionCapacity(device: Device): number {
  return Math.floor(getGPULineSplitMaximumPieceBound(device) / 4);
}

/** Order-preserving u32 keys of f32 values (for the atomic minimum origin) and the finite test. */
const NODING_SNAP_WGSL = /* wgsl */ `
fn isFiniteCoordinate(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
fn getOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getOrderedValue(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}`;

/** Value written to node and neighbor slots that hold no node. */
export const GPU_NETWORK_NODING_NONE = 0xffffffff;

/** Nodes of a noded network. */
export type GPUNetworkNodingNodes = {
  /** Node coordinates. Capacity (`nodeCount` rows of the CSR) is the length. */
  positions: GraphDataView<'float32x2'>;
  /** GPU-written status words. */
  status: {
    /** One-row scalar receiving `min(requiredCount, positions.length)`. */
    count: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving the unclamped number of distinct nodes. */
    requiredCount?: GraphDataView<'uint32'>;
  };
};

/** Edge list of a noded network: edge `e` is piece `e` of {@link GPUNetworkNodingOutput.pieces}. */
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
 * (a loop appears twice in one row). Rows are ordered by edge, then start before end. It has the
 * shape of `GPUNetworkTopology` (without node positions, which are `nodes.positions`), so it feeds
 * the `topology` of `GPUNetworkReachability` and `GPUNetworkServiceAreas`.
 */
export type GPUNetworkNodingTopology = {
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
  /**
   * Coordinates of `lines`. Planar only: snapping cells, `tolerance` and edge lengths are
   * Euclidean, so longitude/latitude throws; project to a planar frame first (for example
   * `createGPULocalFrame`). Omitted means native planar coordinates.
   */
  spatialContext?: GPUSpatialContext;
  /** Linestrings, in input coordinate units (planar meters for a routable network). */
  lines: GPULineSplitProps['lines'];
  /**
   * Capacity of the internal intersection pair list; see `GPULineSplitProps`. At most
   * {@link getGPUNetworkNodingMaximumIntersectionCapacity} (1048576 on a default device).
   */
  intersectionCapacity: number;
  /**
   * One-row per-frame snap tolerance, at least 0. Piece end points whose integer tolerance cells
   * (`floor((coordinate - origin) / tolerance)`, with `origin` the minimum corner of the live end
   * points, computed on the GPU every encoding) coincide share one node, whose position is the
   * first such end point. 0 merges only bit-identical end points. Two end points closer than the
   * tolerance but on opposite sides of a cell boundary stay separate nodes. Cells are exact up to
   * 2^24 per axis (for example 8.3 km at 0.5 mm, or 8,388 km at 0.5 m); a wider extent raises
   * `output.status.overflow`.
   */
  tolerance: GraphDataView<'float32'>;
  /** Passed to `GPULineSplit`. */
  spatialSort?: boolean;
  /** Passed to `GPULineSplit`. */
  leafCapacity?: number;
  /** Caller-owned outputs. */
  output: GPUNetworkNodingOutput;
};

/** GPU-written outputs of {@link GPUNetworkNoding}. */
export type GPUNetworkNodingOutput = {
  /** Pieces, the geometry of the edges. */
  pieces: GPULineSplitPieces;
  /** Nodes. */
  nodes: GPUNetworkNodingNodes;
  /** Edge list. */
  edges: GPUNetworkNodingEdges;
  /** Undirected CSR network. */
  topology: GPUNetworkNodingTopology;
  /** Optional GPU-written status words. */
  status?: {
    /** Optional one-row count of uncertified segment pairs; see `GPUSegmentIntersection`. */
    uncertainCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row flag: 1 when pieces, vertices, intersections or nodes exceeded a capacity, or
     * the end points span more than 2^24 tolerance cells on an axis.
     */
    overflow?: GraphDataView<'uint32'>;
  };
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
 * Planar only: the snapping cells, `tolerance` and edge lengths are Euclidean in the input units,
 * so longitude/latitude lines must be projected first (a `spatialContext` with longitude/latitude
 * throws).
 *
 * Capacity: `output.pieces` bounds edges and vertices, `output.nodes.positions` bounds nodes. When any capacity
 * overflows, edges incident to a node beyond the capacity are left out of the CSR and `output.status.overflow`
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
  /** Resolved spatial context, always planar. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUNetworkNodingProps) {
    this.id = props.id ?? 'network-noding';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    const {nodes, edges, topology: csr, pieces} = props.output;
    validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity);
    validatePackedView(props.tolerance, ['float32'], `${id} tolerance`);
    if (props.tolerance.length < 1) {
      throw new Error(`${id} tolerance must contain one float32 row`);
    }
    validatePackedView(nodes.positions, ['float32x2'], `${id} output.nodes.positions`);
    for (const [name, view] of [
      ['output.nodes.status.count', nodes.status.count],
      ['output.nodes.status.requiredCount', nodes.status.requiredCount],
      ['output.status.overflow', props.output.status?.overflow],
      ['output.status.uncertainCount', props.output.status?.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    this.nodeCapacity = nodes.positions.length;
    this.edgeCapacity = pieces.sourceIds.length;
    if (this.nodeCapacity < 1) {
      throw new Error(`${id} output.nodes.positions must be non-empty`);
    }
    // The endpoint kernels index the last endpoint, so an empty piece capacity would compile to
    // an out-of-range constant (review B-25).
    if (this.edgeCapacity < 1) {
      throw new Error(`${id} output.pieces.sourceIds must be non-empty`);
    }
    if (csr.offsets.length !== this.nodeCapacity + 1) {
      throw new Error(
        `${id} output.topology.offsets length must be output.nodes.positions.length + 1`
      );
    }
    validatePackedView(csr.offsets, ['uint32'], `${id} output.topology.offsets`);
    validatePackedView(csr.neighbors, ['uint32'], `${id} output.topology.neighbors`);
    validatePackedView(csr.weights, ['float32'], `${id} output.topology.weights`);
    for (const [name, view] of [
      ['topology.neighbors', csr.neighbors],
      ['topology.weights', csr.weights],
      ['topology.edgeIds', csr.edgeIds]
    ] as const) {
      if (view && view.length !== this.edgeCapacity * 2) {
        throw new Error(`${id} output.${name} length must be twice the edge capacity`);
      }
    }
    validatePackedView(edges.fromNodes, ['uint32'], `${id} output.edges.fromNodes`);
    validatePackedView(edges.toNodes, ['uint32'], `${id} output.edges.toNodes`);
    validatePackedView(edges.lengths, ['float32'], `${id} output.edges.lengths`);
    for (const [name, view] of [
      ['output.edges.fromNodes', edges.fromNodes],
      ['output.edges.toNodes', edges.toNodes],
      ['output.edges.lengths', edges.lengths]
    ] as const) {
      if (view.length !== this.edgeCapacity) {
        throw new Error(`${id} ${name} length must equal output.pieces.sourceIds.length`);
      }
    }
  }

  /** Returns line split, endpoint snapping, edge and CSR nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCapacity, edgeCapacity} = this;
    const {tolerance, output} = props;
    const {pieces, nodes: nodeOutputs, edges, topology: csr} = output;
    validateCapacity(
      id,
      'intersectionCapacity',
      props.intersectionCapacity,
      1,
      getGPUNetworkNodingMaximumIntersectionCapacity(graph.device)
    );
    validateLineNodingBounds(
      id,
      graph.device,
      props.lines.positions.length,
      Math.max(props.lines.lineOffsets.length - 1, 0),
      props.intersectionCapacity
    );
    validateGraphViewsBelongToGraph(id, graph, [
      tolerance,
      nodeOutputs.positions,
      nodeOutputs.status.count,
      nodeOutputs.status.requiredCount,
      edges.fromNodes,
      edges.toNodes,
      edges.lengths,
      csr.offsets,
      csr.neighbors,
      csr.weights,
      csr.edgeIds,
      props.output.status?.overflow
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
        output: {
          pieces,
          status: output.status?.uncertainCount
            ? {uncertainCount: output.status.uncertainCount}
            : undefined
        }
      }).getCommandNodes(graph)
    );

    // Snap keys per piece end point (index = 2 * edge + side). Cells are counted from a per-frame
    // origin, the minimum corner of the live end points, so projected coordinates (UTM northings
    // of millions of meters) keep exact integer cells: an end point minus the origin is exact in
    // f32, and the quotient resolves unit cells up to 2^24 of them per axis. A larger extent in
    // tolerance cells raises `overflow` instead of silently merging cells.
    const keysX = T('keys-x', 'uint32', endpointCount);
    const keysY = T('keys-y', 'uint32', endpointCount);
    const endpointPoints = T('endpoint-points', 'float32x2', endpointCount);
    const endpointIndices = T('endpoint-indices', 'uint32', endpointCount);
    const snapOrigin = T('snap-origin', 'uint32', 2);
    const snapOverflow = T('snap-overflow', 'uint32', 1);
    // The snap origin is the minimum endpoint, folded per workgroup into one atomic per axis
    // (rule 14) instead of two atomics per endpoint.
    const endpointOrigin = getWorkgroupCounterKernelSource({
      counters: [
        {target: 'snapOrigin', word: 0, operation: 'min'},
        {target: 'snapOrigin', word: 1, operation: 'min'}
      ],
      rowCount: 'ENDPOINT_COUNT',
      rowSource: `let edge = index / 2u;
  endpointIndices[endpointIndicesOffset + index] = index;
  var point = vec2f(0.0);
  if (edge < pieceCount[pieceCountOffset]) {
    var vertex = pieceOffsets[pieceOffsetsOffset + edge];
    if ((index & 1u) == 1u) { vertex = pieceOffsets[pieceOffsetsOffset + edge + 1u] - 1u; }
    point = vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
    if (isFiniteCoordinate(point.x) && isFiniteCoordinate(point.y)) {
      minCounter(0u, getOrderedKey(point.x));
      minCounter(1u, getOrderedKey(point.y));
    }
  }
  endpointPoints[endpointPointsOffset + index * 2u] = point.x;
  endpointPoints[endpointPointsOffset + index * 2u + 1u] = point.y;`
    });
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-snap-origin`,
        operation: OPERATION,
        view: snapOrigin,
        type: 'u32',
        value: '0xffffffffu'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-snap-overflow`,
        operation: OPERATION,
        view: snapOverflow,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-endpoints`,
        operation: OPERATION,
        variant: 'endpoints',
        bindings: [
          {name: 'pieceCount', view: pieces.status.count, type: 'u32', access: 'read'},
          {name: 'pieceOffsets', view: pieces.geometry.lineOffsets, type: 'u32', access: 'read'},
          {name: 'piecePositions', view: pieces.geometry.positions, type: 'f32', access: 'read'},
          {name: 'endpointPoints', view: endpointPoints, type: 'f32', access: 'read_write'},
          {name: 'endpointIndices', view: endpointIndices, type: 'u32', access: 'read_write'},
          {name: 'snapOrigin', view: snapOrigin, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: endpointCount,
        workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${NODING_SNAP_WGSL}
const ENDPOINT_COUNT: u32 = ${endpointCount}u;
${endpointOrigin.declarations}`,
        body: endpointOrigin.body
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-endpoint-keys`,
        operation: OPERATION,
        variant: 'endpoint-keys',
        bindings: [
          {name: 'pieceCount', view: pieces.status.count, type: 'u32', access: 'read'},
          {name: 'endpointPoints', view: endpointPoints, type: 'f32', access: 'read'},
          {name: 'tolerance', view: tolerance, type: 'f32', access: 'read'},
          {name: 'snapOrigin', view: snapOrigin, type: 'u32', access: 'read'},
          {name: 'snapOverflow', view: snapOverflow, type: 'u32', access: 'read_write'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read_write'},
          {name: 'keysY', view: keysY, type: 'u32', access: 'read_write'}
        ],
        invocationCount: endpointCount,
        declarations: `${NODING_SNAP_WGSL}
// Integer tolerance cell of value counted from origin, or the bit pattern for tolerance 0
// (only bit-identical end points merge). Sets isOverflow past 2^24 cells, where the f32 quotient
// no longer resolves unit cells.
fn snapKey(value: f32, origin: f32, size: f32, isOverflow: ptr<function, bool>) -> u32 {
  if (!isFiniteCoordinate(value)) {
    return 0xfffffffeu;
  }
  if (size > 0.0) {
    let cell = floor(max(value - origin, 0.0) / size);
    if (!(cell < 16777216.0)) {
      *isOverflow = true;
      return 16777215u;
    }
    return u32(cell);
  }
  return min(bitcast<u32>(select(value, 0.0, value == 0.0)), 0xfffffffeu);
}`,
        body: `var keyX = 0xffffffffu;
  var keyY = 0xffffffffu;
  if (index / 2u < pieceCount[pieceCountOffset]) {
    let point = vec2f(endpointPoints[endpointPointsOffset + index * 2u], endpointPoints[endpointPointsOffset + index * 2u + 1u]);
    let origin = vec2f(
      getOrderedValue(snapOrigin[snapOriginOffset]),
      getOrderedValue(snapOrigin[snapOriginOffset + 1u])
    );
    let size = tolerance[toleranceOffset];
    var isOverflow = false;
    keyX = snapKey(point.x, origin.x, size, &isOverflow);
    keyY = snapKey(point.y, origin.y, size, &isOverflow);
    if (isOverflow) {
      snapOverflow[snapOverflowOffset] = 1u;
    }
  }
  keysX[keysXOffset + index] = keyX;
  keysY[keysYOffset + index] = keyY;`
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
      {name: 'snapOverflow', view: snapOverflow, type: 'u32', access: 'read'},
      {name: 'nodeCount', view: nodeOutputs.status.count, type: 'u32', access: 'read_write'}
    ];
    if (nodeOutputs.status.requiredCount) {
      scalarBindings.push({
        name: 'requiredCount',
        view: nodeOutputs.status.requiredCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (props.output.status?.overflow) {
      scalarBindings.push({
        name: 'overflow',
        view: props.output.status.overflow,
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
  ${nodeOutputs.status.requiredCount ? 'requiredCount[requiredCountOffset] = total;' : ''}
  ${props.output.status?.overflow ? 'overflow[overflowOffset] = select(0u, 1u, total > NODE_CAPACITY || splitOverflow[splitOverflowOffset] != 0u || snapOverflow[snapOverflowOffset] != 0u);' : ''}`
      })
    );
    return nodes;
  }
}
