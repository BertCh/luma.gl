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
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getKeyGroupNodes,
  getKeyPairSortNodes,
  KEY_PAIR_INVALID
} from '../spatial-weights/key-pair-grouping';
import {
  GPULineSimplification,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  type GPULineSimplificationProps
} from '../line-simplification/index';

const OPERATION = 'GPUCoverageSimplification';
const INVALID = `${KEY_PAIR_INVALID}u`;
/** Bit of a vertex's `prev` entry that marks the first vertex of its ring. */
const RING_BEGIN_BIT = '0x80000000u';

/** Caller-owned outputs of {@link GPUCoverageSimplification}. */
export type GPUCoverageSimplificationOutput = {
  /**
   * Simplified ring vertices, ring after ring in input order, vertex order preserved. Its length is
   * the capacity; `positions.length` of the input always suffices.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * `ringCount + 1` offsets into `positions`; ring `r` keeps rows `[ringOffsets[r], ringOffsets[r + 1])`.
   * Polygon-to-ring offsets are unchanged, so the input `polygonOffsets` still apply. Offsets are
   * clamped to the capacity so rings stay consistent after an overflow.
   */
  ringOffsets: GraphDataView<'uint32'>;
  /** Optional per-input-vertex `1`/`0` keep mask, `positions.length` rows of the input. */
  keepMask?: GraphDataView<'uint32'>;
  /** One-row flag: 1 when more vertices were kept than `positions` can hold, else 0. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row unclamped kept-vertex total. */
  totalCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCoverageSimplification}.
 *
 * Per-frame (no recompile): the contents of `parameters` (the tolerance) and of the input
 * buffers, as long as lengths stay the same. Compile-time: `snapTolerance`, `maximumRounds`, all
 * view lengths and which optional outputs are present.
 */
export type GPUCoverageSimplificationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'coverage-simplification'`. */
  id?: string;
  /**
   * Flattened ring vertices (GeoArrow layout, as `GPUContiguityWeights`). Planar, finite
   * coordinates; use tile- or view-local units because distances are f32.
   */
  positions: GraphDataView<'float32x2'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry. Every ring (shells, holes) is simplified. */
  polygonOffsets: GraphDataView<'uint32'>;
  /**
   * Vertex snap tolerance, with the same meaning as `GPUContiguityWeights.snapTolerance`. Two
   * vertices that snap to the same point are one point of the coverage. Defaults to `0` (exact f32
   * equality).
   */
  snapTolerance?: number;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH}
   * elements written with `getGPULineSimplificationParameterValues`: the Douglas-Peucker tolerance.
   */
  parameters: GraphDataView<'float32'>;
  /** Compile-time cap on Douglas-Peucker rounds, as `GPULineSimplification.maximumRounds`. */
  maximumRounds?: number;
  /** Optional one-row scalar: 1 when every arc was decided within `maximumRounds`, else 0. */
  converged?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUCoverageSimplificationOutput;
};

/**
 * Gap-free polygon-coverage simplification (the Shapely `simplify_coverage` and PostGIS
 * `ST_CoverageSimplify` idea): every arc shared by two polygons is simplified once, so both
 * neighbours keep identical vertices along it.
 *
 * **Definition.**
 * - Vertices are snapped as in `GPUContiguityWeights` and get dense point IDs; every ring edge gets
 *   the key `(min, max)` of its endpoint IDs. The *partner* of an edge is the lowest polygon ID
 *   (other than its own) that has an edge with the same key, or none for outer boundary.
 * - An *arc* is a maximal run of consecutive ring edges with the same partner. Arcs also break at
 *   the first vertex of every ring, so a shared arc that crosses a ring start keeps one extra
 *   vertex.
 * - An arc is *owned* by the lower polygon ID of its pair (an outer-boundary arc by its own
 *   polygon). Owned arcs are copied into one polyline each (a ring without any break is one
 *   closed polyline) and simplified by `GPULineSimplification`. A vertex is kept when its point is
 *   kept in any owned arc, or is a junction (an arc endpoint); the non-owning neighbour reads that
 *   decision through the shared point ID, so it cannot disagree.
 * - Output rings are the input rings filtered by the keep mask. Ring and vertex order and
 *   direction are unchanged.
 *
 * **Guarantees.** Neighbours stay gap-free and overlap-free along shared arcs: the simplified shared
 * boundary of two polygons is the same polyline in both rings. The kept set along each arc equals
 * Douglas-Peucker on that arc when `converged` is 1. **Not guaranteed:** topology. As in plain
 * Douglas-Peucker, a simplified arc may cross another arc or itself, and a ring can collapse to
 * fewer than three vertices when the tolerance exceeds the feature size; callers drop rings of
 * fewer than three rows with `ringOffsets`. Non-finite vertices are unsupported (never kept).
 *
 * **Cost.** Arcs are copied by one thread each walking the arc, so a single very long arc costs a
 * serial walk of its length. Intermediate buffers are `O(vertices)`; the simplification rows are
 * bounded by `2 * vertices`. The output is bounded by the `output.positions` capacity with a GPU
 * overflow flag; `positions.length` of the input always suffices.
 */
export class GPUCoverageSimplification implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCoverageSimplificationProps;

  constructor(props: GPUCoverageSimplificationProps) {
    const id = props.id ?? 'coverage-simplification';
    this.id = id;
    this.props = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1 || props.positions.length >= 2 ** 30) {
      throw new Error(`${id} positions must hold between 1 and 2^30 - 1 vertices`);
    }
    validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
    validatePackedUint32View(props.polygonOffsets, `${id} polygonOffsets`);
    if (props.ringOffsets.length < 2) {
      throw new Error(`${id} ringOffsets must hold at least two entries`);
    }
    if (props.polygonOffsets.length < 2) {
      throw new Error(`${id} polygonOffsets must hold at least two entries`);
    }
    if (
      props.snapTolerance !== undefined &&
      (!Number.isFinite(props.snapTolerance) || props.snapTolerance < 0)
    ) {
      throw new Error(`${id} snapTolerance must be a finite number >= 0`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH} float32 values`
      );
    }
    const {output} = props;
    validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    if (output.positions.length < 1) {
      throw new Error(`${id} output.positions must hold at least one row`);
    }
    validatePackedUint32View(output.ringOffsets, `${id} output.ringOffsets`);
    if (output.ringOffsets.length !== props.ringOffsets.length) {
      throw new Error(`${id} output.ringOffsets length must equal ringOffsets length`);
    }
    validatePackedUint32View(output.overflow, `${id} output.overflow`);
    if (output.overflow.length < 1) {
      throw new Error(`${id} output.overflow must hold one uint32`);
    }
    if (output.keepMask) {
      validatePackedUint32View(output.keepMask, `${id} output.keepMask`);
      if (output.keepMask.length !== props.positions.length) {
        throw new Error(`${id} output.keepMask length must equal positions length`);
      }
    }
    if (output.totalCount) {
      validatePackedUint32View(output.totalCount, `${id} output.totalCount`);
      if (output.totalCount.length < 1) {
        throw new Error(`${id} output.totalCount must hold one uint32`);
      }
    }
    if (props.converged) {
      validatePackedUint32View(props.converged, `${id} converged`);
      if (props.converged.length < 1) {
        throw new Error(`${id} converged must hold one uint32`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.positions,
        output.ringOffsets,
        output.keepMask,
        output.overflow,
        output.totalCount,
        props.converged
      ],
      [props.positions, props.ringOffsets, props.polygonOffsets, props.parameters]
    );
  }

  /** Returns the coverage-simplification nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, ringOffsets, polygonOffsets, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      ringOffsets,
      polygonOffsets,
      props.parameters,
      props.converged,
      output.positions,
      output.ringOffsets,
      output.keepMask,
      output.overflow,
      output.totalCount
    ]);
    const vertexCount = positions.length;
    const ringCount = ringOffsets.length - 1;
    const polygonCount = polygonOffsets.length - 1;
    const outputCapacity = output.positions.length;
    const snapTolerance = props.snapTolerance ?? 0;
    const lineRowCapacity = 2 * vertexCount;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    const vertexHigh = view('vertex-high', vertexCount);
    const vertexLow = view('vertex-low', vertexCount);
    const vertexPolygon = view('vertex-polygon', vertexCount);
    const vertexNext = view('vertex-next', vertexCount);
    const vertexPrev = view('vertex-prev', vertexCount);

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertex-topology`,
        operation: OPERATION,
        variant: 'vertex-topology',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
          {name: 'vertexHigh', view: vertexHigh, type: 'u32', access: 'read_write'},
          {name: 'vertexLow', view: vertexLow, type: 'u32', access: 'read_write'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read_write'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read_write'},
          {name: 'vertexPrev', view: vertexPrev, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const RING_COUNT: u32 = ${ringCount}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const SNAP_TOLERANCE: f32 = ${getWGSLFloatLiteral(snapTolerance)};

fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn quantise(value: f32) -> u32 {
  if (SNAP_TOLERANCE <= 0.0) {
    return bitcast<u32>(select(value, 0.0, value == 0.0));
  }
  let snapped = clamp(floor(value / SNAP_TOLERANCE + 0.5), -2147483520.0, 2147483520.0);
  return bitcast<u32>(i32(snapped)) ^ 0x80000000u;
}`,
        body: `var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  var polygon = 0u;
  high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let inRing = index >= ringBegin && index < ringEnd &&
    ring >= polygonOffsets[polygonOffsetsOffset] && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
    ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  let valid = inRing && isFiniteFloat(x) && isFiniteFloat(y);
  vertexHigh[vertexHighOffset + index] = select(${INVALID}, quantise(x), valid);
  vertexLow[vertexLowOffset + index] = select(${INVALID}, quantise(y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${INVALID}, polygon, valid);
  vertexNext[vertexNextOffset + index] = select(index + 1u, ringBegin, index + 1u == ringEnd);
  let previous = select(index - 1u, ringEnd - 1u, index == ringBegin);
  vertexPrev[vertexPrevOffset + index] = select(previous, previous | ${RING_BEGIN_BIT}, index == ringBegin);`
      })
    ];

    // Dense point IDs, as the rook branch of GPUContiguityWeights.
    const vertexSort = getKeyPairSortNodes(
      graph,
      `${id}-vertex`,
      OPERATION,
      vertexCount,
      vertexHigh,
      vertexLow
    );
    const vertexGroups = getKeyGroupNodes(
      graph,
      `${id}-vertex`,
      OPERATION,
      vertexCount,
      vertexHigh,
      vertexLow,
      vertexSort.sortedItems
    );
    const pointIds = view('point-ids', vertexCount);
    const edgeHigh = view('edge-high', vertexCount);
    const edgeLow = view('edge-low', vertexCount);
    nodes.push(
      ...vertexSort.nodes,
      ...vertexGroups.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-point-ids`,
        operation: OPERATION,
        variant: 'point-ids',
        bindings: [
          {name: 'sortedItems', view: vertexSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'groupIndex', view: vertexGroups.groupIndex, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let vertex = sortedItems[sortedItemsOffset + index];
  pointIds[pointIdsOffset + vertex] = select(${INVALID},
    groupIndex[groupIndexOffset + index] - 1u, vertexPolygon[vertexPolygonOffset + vertex] != ${INVALID});`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-keys`,
        operation: OPERATION,
        variant: 'edge-keys',
        bindings: [
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
          {name: 'edgeHigh', view: edgeHigh, type: 'u32', access: 'read_write'},
          {name: 'edgeLow', view: edgeLow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + vertexNext[vertexNextOffset + index]];
  let valid = first != ${INVALID} && second != ${INVALID} && first != second &&
    vertexPolygon[vertexPolygonOffset + index] != ${INVALID};
  edgeHigh[edgeHighOffset + index] = select(${INVALID}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${INVALID}, max(first, second), valid);`
      })
    );
    const edgeSort = getKeyPairSortNodes(
      graph,
      `${id}-edge`,
      OPERATION,
      vertexCount,
      edgeHigh,
      edgeLow
    );
    const edgeGroups = getKeyGroupNodes(
      graph,
      `${id}-edge`,
      OPERATION,
      vertexCount,
      edgeHigh,
      edgeLow,
      edgeSort.sortedItems
    );
    nodes.push(...edgeSort.nodes, ...edgeGroups.nodes);

    // Partner polygon of every edge: the lowest other polygon holding the same edge.
    const edgePartner = view('edge-partner', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-partner`,
        operation: OPERATION,
        variant: 'edge-partner',
        bindings: [
          {name: 'sortedItems', view: edgeSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'edgeHigh', view: edgeHigh, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'groupIndex', view: edgeGroups.groupIndex, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: edgeGroups.groupStarts, type: 'u32', access: 'read'},
          {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let item = sortedItems[sortedItemsOffset + index];
  let polygon = vertexPolygon[vertexPolygonOffset + item];
  var partner = ${INVALID};
  if (polygon != ${INVALID} && edgeHigh[edgeHighOffset + item] != ${INVALID}) {
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    for (var other = groupStarts[groupStartsOffset + groupId]; other < end; other++) {
      let otherPolygon = vertexPolygon[vertexPolygonOffset + sortedItems[sortedItemsOffset + other]];
      if (otherPolygon != ${INVALID} && otherPolygon != polygon) {
        partner = min(partner, otherPolygon);
      }
    }
  }
  edgePartner[edgePartnerOffset + item] = partner;`
      })
    );

    // Arc starts, junction flags and ownership. Bit 0: arc start. Bit 1: junction (partner change).
    const arcFlags = view('arc-flags', vertexCount);
    const startFlag = view('start-flag', vertexCount);
    const startRank = view('start-rank', vertexCount);
    // Arc starts in vertex order, plus one sentinel entry: the end of the last ring.
    const startList = view('start-list', vertexCount + 1);
    const ownedLength = view('owned-length', vertexCount);
    const ownedFlag = view('owned-flag', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-arc-flags`,
        operation: OPERATION,
        variant: 'arc-flags',
        bindings: [
          {name: 'vertexPrev', view: vertexPrev, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read'},
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read_write'},
          {name: 'startFlag', view: startFlag, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let prevEntry = vertexPrev[vertexPrevOffset + index];
  let previous = prevEntry & 0x7fffffffu;
  let ringBegin = (prevEntry & ${RING_BEGIN_BIT}) != 0u;
  let valid = vertexPolygon[vertexPolygonOffset + index] != ${INVALID};
  let junction = valid && edgePartner[edgePartnerOffset + previous] != edgePartner[edgePartnerOffset + index];
  arcFlags[arcFlagsOffset + index] = select(0u, 1u, ringBegin || junction) | select(0u, 2u, junction);
  startFlag[startFlagOffset + index] = select(0u, 1u, ringBegin || junction);`
      }),
      ...new GPUScan({
        id: `${id}-start-rank-scan`,
        input: startFlag,
        output: startRank,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-start-list`,
        operation: OPERATION,
        variant: 'start-list',
        bindings: [
          {name: 'startFlag', view: startFlag, type: 'u32', access: 'read'},
          {name: 'startRank', view: startRank, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
          {name: 'startList', view: startList, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
const RING_COUNT: u32 = ${ringCount}u;`,
        body: `let flag = startFlag[startFlagOffset + index];
  let rank = startRank[startRankOffset + index];
  if (flag != 0u) {
    startList[startListOffset + rank] = index;
  }
  if (index == VERTEX_COUNT - 1u) {
    startList[startListOffset + rank + flag] = min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-arc-lengths`,
        operation: OPERATION,
        variant: 'arc-lengths',
        bindings: [
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
          {name: 'startRank', view: startRank, type: 'u32', access: 'read'},
          {name: 'startList', view: startList, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read'},
          {name: 'ownedLength', view: ownedLength, type: 'u32', access: 'read_write'},
          {name: 'ownedFlag', view: ownedFlag, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `var length = 0u;
  var owned = 0u;
  let polygon = vertexPolygon[vertexPolygonOffset + index];
  if ((arcFlags[arcFlagsOffset + index] & 1u) != 0u && polygon != ${INVALID}) {
    let partner = edgePartner[edgePartnerOffset + index];
    if (partner == ${INVALID} || polygon < partner) {
      owned = 1u;
      // The arc runs to the next arc start in vertex order, or to the end of the ring, whose
      // first vertex is itself an arc start. The closing vertex is repeated.
      let arcEnd = startList[startListOffset + startRank[startRankOffset + index] + 1u];
      length = arcEnd - index + 1u;
    }
  }
    ownedLength[ownedLengthOffset + index] = length;
  ownedFlag[ownedFlagOffset + index] = owned;`
      })
    );
    const arcIndex = view('arc-index', vertexCount);
    const lineStart = view('line-start', vertexCount);
    const trackOffsets = view('track-offsets', vertexCount + 1);
    const linePositions = createTransientView(
      graph,
      `${id}-line-positions`,
      'float32x2',
      lineRowCapacity
    );
    const linePoint = view('line-point', lineRowCapacity);
    nodes.push(
      ...new GPUScan({
        id: `${id}-arc-index-scan`,
        input: ownedFlag,
        output: arcIndex,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-line-start-scan`,
        input: ownedLength,
        output: lineStart,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-track-offsets-fill`,
        operation: OPERATION,
        variant: 'track-offsets-fill',
        bindings: [
          {name: 'lineStart', view: lineStart, type: 'u32', access: 'read'},
          {name: 'ownedLength', view: ownedLength, type: 'u32', access: 'read'},
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount + 1,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `trackOffsets[trackOffsetsOffset + index] =
    lineStart[lineStartOffset + VERTEX_COUNT - 1u] + ownedLength[ownedLengthOffset + VERTEX_COUNT - 1u];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-track-offsets`,
        operation: OPERATION,
        variant: 'track-offsets',
        bindings: [
          {name: 'ownedFlag', view: ownedFlag, type: 'u32', access: 'read'},
          {name: 'arcIndex', view: arcIndex, type: 'u32', access: 'read'},
          {name: 'lineStart', view: lineStart, type: 'u32', access: 'read'},
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `if (ownedFlag[ownedFlagOffset + index] != 0u) {
    trackOffsets[trackOffsetsOffset + arcIndex[arcIndexOffset + index]] = lineStart[lineStartOffset + index];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-copy-arcs`,
        operation: OPERATION,
        variant: 'copy-arcs',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read'},
          {name: 'ownedLength', view: ownedLength, type: 'u32', access: 'read'},
          {name: 'lineStart', view: lineStart, type: 'u32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
          {name: 'linePositions', view: linePositions, type: 'f32', access: 'read_write'},
          {name: 'linePoint', view: linePoint, type: 'u32', access: 'read_write'}
        ],
        invocationCount: lineRowCapacity,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        // One invocation per output row, so a long arc is copied in parallel. The arc owning a
        // row is the last vertex whose line start is at or before it; row k of an arc is its
        // start for k = 0 and the successor of vertex start + k - 1 afterwards.
        body: `let total = lineStart[lineStartOffset + VERTEX_COUNT - 1u] + ownedLength[ownedLengthOffset + VERTEX_COUNT - 1u];
  if (index < total) {
    var low = 0u;
    var high = VERTEX_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (lineStart[lineStartOffset + middle] <= index) {
        low = middle;
      } else {
        high = middle;
      }
    }
    let step = index - lineStart[lineStartOffset + low];
    let vertex = select(vertexNext[vertexNextOffset + low + step - 1u], low, step == 0u);
    let pointId = pointIds[pointIdsOffset + vertex];
    let x = positions[positionsOffset + vertex * 2u];
    let y = positions[positionsOffset + vertex * 2u + 1u];
    let finite = pointId != ${INVALID};
    linePositions[linePositionsOffset + index * 2u] = select(0.0, x, finite);
    linePositions[linePositionsOffset + index * 2u + 1u] = select(0.0, y, finite);
    linePoint[linePointOffset + index] = pointId;
  }`
      })
    );

    // Simplify every owned arc once.
    const lineImportance = createTransientView(
      graph,
      `${id}-line-importance`,
      'float32',
      lineRowCapacity
    );
    const lineKeep = view('line-keep', lineRowCapacity);
    const selectionIds = view('selection-ids', lineRowCapacity);
    const selectionCount = view('selection-count', 1);
    const selectionOverflow = view('selection-overflow', 1);
    const lineProps: GPULineSimplificationProps = {
      id: `${id}-arcs`,
      positions: linePositions,
      trackOffsets,
      importance: lineImportance,
      maximumRounds: props.maximumRounds,
      status: props.converged ? {converged: props.converged} : undefined,
      parameters: props.parameters,
      selection: {
        output: {ids: selectionIds, count: selectionCount, overflow: selectionOverflow},
        keepMask: lineKeep
      }
    };
    nodes.push(...new GPULineSimplification(lineProps).getCommandNodes(graph));

    // Share the decisions through point IDs: junctions and kept arc vertices.
    const keepPoint = view('keep-point', vertexCount);
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-keep-point-clear`,
        operation: OPERATION,
        view: keepPoint,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mark-junctions`,
        operation: OPERATION,
        variant: 'mark-junctions',
        bindings: [
          {name: 'arcFlags', view: arcFlags, type: 'u32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
          {name: 'keepPoint', view: keepPoint, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let pointId = pointIds[pointIdsOffset + index];
  if ((arcFlags[arcFlagsOffset + index] & 2u) != 0u && pointId != ${INVALID}) {
    atomicMax(&keepPoint[keepPointOffset + pointId], 1u);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mark-arcs`,
        operation: OPERATION,
        variant: 'mark-arcs',
        bindings: [
          {name: 'lineKeep', view: lineKeep, type: 'u32', access: 'read'},
          {name: 'linePoint', view: linePoint, type: 'u32', access: 'read'},
          {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
          {name: 'keepPoint', view: keepPoint, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: lineRowCapacity,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `let pointId = linePoint[linePointOffset + index];
  if (index < trackOffsets[trackOffsetsOffset + VERTEX_COUNT] && lineKeep[lineKeepOffset + index] != 0u &&
      pointId != ${INVALID}) {
    atomicMax(&keepPoint[keepPointOffset + pointId], 1u);
  }`
      })
    );

    // Per-vertex mask, rank, and ring-wise compaction.
    const keepMask = output.keepMask ?? view('keep-mask', vertexCount);
    const keepRank = view('keep-rank', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keep-mask`,
        operation: OPERATION,
        variant: 'keep-mask',
        bindings: [
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
          {name: 'keepPoint', view: keepPoint, type: 'u32', access: 'read'},
          {name: 'keepMask', view: keepMask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let pointId = pointIds[pointIdsOffset + index];
  keepMask[keepMaskOffset + index] =
    select(0u, keepPoint[keepPointOffset + pointId], pointId != ${INVALID});`
      }),
      ...new GPUScan({
        id: `${id}-keep-rank-scan`,
        input: keepMask,
        output: keepRank,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit-positions`,
        operation: OPERATION,
        variant: 'emit-positions',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'keepMask', view: keepMask, type: 'u32', access: 'read'},
          {name: 'keepRank', view: keepRank, type: 'u32', access: 'read'},
          {name: 'outputPositions', view: output.positions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const CAPACITY: u32 = ${outputCapacity}u;`,
        body: `let rank = keepRank[keepRankOffset + index];
  if (keepMask[keepMaskOffset + index] != 0u && rank < CAPACITY) {
    outputPositions[outputPositionsOffset + rank * 2u] = positions[positionsOffset + index * 2u];
    outputPositions[outputPositionsOffset + rank * 2u + 1u] = positions[positionsOffset + index * 2u + 1u];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit-offsets`,
        operation: OPERATION,
        variant: 'emit-offsets',
        bindings: [
          {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
          {name: 'keepMask', view: keepMask, type: 'u32', access: 'read'},
          {name: 'keepRank', view: keepRank, type: 'u32', access: 'read'},
          {name: 'outputRingOffsets', view: output.ringOffsets, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.overflow, type: 'u32', access: 'read_write'},
          ...(output.totalCount
            ? [
                {
                  name: 'totalCount',
                  view: output.totalCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: ringCount + 1,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
const RING_COUNT: u32 = ${ringCount}u;
const CAPACITY: u32 = ${outputCapacity}u;`,
        body: `let total = keepRank[keepRankOffset + VERTEX_COUNT - 1u] + keepMask[keepMaskOffset + VERTEX_COUNT - 1u];
  var begin = total;
  if (index < RING_COUNT) {
    let ringBegin = ringOffsets[ringOffsetsOffset + index];
    if (ringBegin < VERTEX_COUNT) {
      begin = keepRank[keepRankOffset + ringBegin];
    }
  }
  outputRingOffsets[outputRingOffsetsOffset + index] = min(begin, CAPACITY);
  if (index == RING_COUNT) {
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${output.totalCount ? 'totalCount[totalCountOffset] = total;' : ''}
  }`
      })
    );
    return nodes;
  }
}
