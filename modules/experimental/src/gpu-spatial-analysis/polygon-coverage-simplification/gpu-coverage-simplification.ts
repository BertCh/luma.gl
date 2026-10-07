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
import {GPUSegmentIntersection} from '../segment-intersection/index';
import {
  GPULineSimplification,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  type GPULineSimplificationProps
} from '../line-simplification/index';

const OPERATION = 'GPUCoverageSimplification';
const INVALID = `${KEY_PAIR_INVALID}u`;
/** Bit of a vertex's `prev` entry that marks the first vertex of its ring. */
const RING_BEGIN_BIT = '0x80000000u';
/** Default number of detect-and-repair rounds of the topology pass. */
const DEFAULT_TOPOLOGY_ROUNDS = 4;
/** Number of rows `topologyStats` must hold. */
export const GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATS_LENGTH = 4;
const TOPOLOGY_STATS_LENGTH = GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATS_LENGTH;

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
  /**
   * Optional {@link GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATS_LENGTH} rows describing the topology
   * pass (all zero when `topologyRounds` is 0):
   * - `[0]` crossings before any repair: pairs of simplified arc segments that intersect (cross,
   *   touch in a T-junction or overlap) without sharing a coverage point.
   * - `[1]` crossings remaining after the last repair round.
   * - `[2]` vertices restored by the repair rounds (the ring-minimum step is not counted).
   * - `[3]` 1 when the candidate pair capacity overflowed in some round, so the counts are lower
   *   bounds.
   * Crossings fixed is `[0] - [1]`.
   */
  topologyStats?: GraphDataView<'uint32'>;
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
  /**
   * Whether the outer boundary of the coverage is simplified. Defaults to `true`. With `false`
   * (Shapely `coverage_simplify(simplify_boundary=False)`), every vertex on an edge that no other
   * polygon shares is kept, so only arcs shared by two polygons are simplified; hole boundaries
   * of the coverage count as outer boundary. Compile-time.
   */
  simplifyBoundary?: boolean;
  /** Compile-time cap on Douglas-Peucker rounds, as `GPULineSimplification.maximumRounds`. */
  maximumRounds?: number;
  /**
   * Compile-time number of detect-and-repair rounds that make the simplified coverage
   * topology-preserving. Each round finds simplified segments that cross, touch or overlap
   * without sharing a coverage point and restores the original vertex farthest from each of them.
   * `0` skips detection and repair (the ring minimum still applies). Defaults to `4`.
   */
  topologyRounds?: number;
  /**
   * Compile-time capacity of the candidate pair list of each detection round. Shared arcs appear
   * in two rings and every node touches several segments, so the list holds those legitimate
   * pairs too. Defaults to `max(256, 4 * vertices)`. On overflow `topologyStats[3]` is set.
   */
  topologyPairCapacity?: number;
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
 * With `simplifyBoundary: false` the keep decision additionally marks every vertex that touches an
 * edge without a partner, so boundary arcs are copied unchanged.
 *
 * **Guarantees.** Neighbours stay gap-free and overlap-free along shared arcs: the simplified shared
 * boundary of two polygons is the same polyline in both rings. The kept set along each arc equals
 * Douglas-Peucker on that arc when `converged` is 1, before the topology steps below add vertices.
 *
 * **Topology preservation** (Shapely `coverage_simplify`, PostGIS `ST_CoverageSimplify`). Arc
 * endpoints (junctions where the partner changes, and ring starts) are always kept, so nodes never
 * move. A ring never keeps fewer than three vertices: if Douglas-Peucker leaves fewer, the
 * farthest original vertices are restored. Then `topologyRounds` rounds run: the simplified rings
 * are compacted, `GPUSegmentIntersection` lists intersecting segment pairs with exact predicates,
 * pairs that share a coverage point (a node) or are the same shared edge are ignored, and for every
 * remaining pair the original vertex farthest from each segment's span is restored (one
 * Douglas-Peucker split step; decisions are shared through point IDs, so neighbours still agree).
 * `output.topologyStats` reports crossings before, crossings remaining after the last round, and
 * restored vertices. Crossings that need more splits than the rounds provide stay in the output
 * and are counted as remaining; an input that is itself self-intersecting cannot be repaired.
 * Non-finite vertices are unsupported (never kept).
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
      props.topologyRounds !== undefined &&
      (!Number.isInteger(props.topologyRounds) ||
        props.topologyRounds < 0 ||
        props.topologyRounds > 16)
    ) {
      throw new Error(`${id} topologyRounds must be an integer from 0 to 16`);
    }
    if (
      props.topologyPairCapacity !== undefined &&
      (!Number.isInteger(props.topologyPairCapacity) || props.topologyPairCapacity < 1)
    ) {
      throw new Error(`${id} topologyPairCapacity must be a positive integer`);
    }
    if (props.simplifyBoundary !== undefined && typeof props.simplifyBoundary !== 'boolean') {
      throw new Error(`${id} simplifyBoundary must be a boolean`);
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
    if (output.topologyStats) {
      validatePackedUint32View(output.topologyStats, `${id} output.topologyStats`);
      if (output.topologyStats.length < TOPOLOGY_STATS_LENGTH) {
        throw new Error(
          `${id} output.topologyStats must hold ${TOPOLOGY_STATS_LENGTH} uint32 rows`
        );
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
        output.topologyStats,
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
      output.totalCount,
      output.topologyStats
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
          {name: 'vertexPrev', view: vertexPrev, type: 'u32', access: 'read'},
          {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read'},
          {name: 'keepPoint', view: keepPoint, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const KEEP_BOUNDARY: bool = ${props.simplifyBoundary === false};`,
        // A vertex is on the outer boundary when its outgoing or incoming edge has no partner.
        body: `let pointId = pointIds[pointIdsOffset + index];
  let previous = vertexPrev[vertexPrevOffset + index] & 0x7fffffffu;
  let isBoundary = KEEP_BOUNDARY && (edgePartner[edgePartnerOffset + index] == ${INVALID} ||
    edgePartner[edgePartnerOffset + previous] == ${INVALID});
  if (((arcFlags[arcFlagsOffset + index] & 2u) != 0u || isBoundary) && pointId != ${INVALID}) {
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

    // Topology repair: keep at least three vertices per ring, then detect and repair crossings.
    const topologyRounds = props.topologyRounds ?? DEFAULT_TOPOLOGY_ROUNDS;
    {
      nodes.push(
        ...getRingCollapseNodes(graph, {
          id,
          positions,
          ringOffsets,
          vertexPolygon,
          pointIds,
          keepPoint,
          vertexCount,
          ringCount
        })
      );
    }
    if (topologyRounds === 0 && output.topologyStats) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-topology-stats-zero`,
          operation: OPERATION,
          view: output.topologyStats,
          type: 'u32',
          value: '0u'
        })
      );
    }
    if (topologyRounds > 0) {
      const workStats = view('topology-work', TOPOLOGY_STATS_LENGTH + topologyRounds + 1);
      const ownedVertex = view('owned-vertex', vertexCount);
      const featureOffsets = view('feature-offsets', polygonCount + 1);
      const pairCapacity = props.topologyPairCapacity ?? Math.max(256, 4 * vertexCount);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-topology-stats-clear`,
          operation: OPERATION,
          view: workStats,
          type: 'u32',
          value: '0u'
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-owned-vertex`,
          operation: OPERATION,
          variant: 'owned-vertex',
          bindings: [
            {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
            {name: 'edgePartner', view: edgePartner, type: 'u32', access: 'read'},
            {name: 'ownedVertex', view: ownedVertex, type: 'u32', access: 'read_write'}
          ],
          invocationCount: vertexCount,
          body: `let polygon = vertexPolygon[vertexPolygonOffset + index];
  let partner = edgePartner[edgePartnerOffset + index];
  ownedVertex[ownedVertexOffset + index] =
    select(0u, select(0u, 1u, partner == ${INVALID} || polygon < partner), polygon != ${INVALID});`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-feature-offsets`,
          operation: OPERATION,
          variant: 'feature-offsets',
          bindings: [
            {name: 'featureOffsets', view: featureOffsets, type: 'u32', access: 'read_write'}
          ],
          invocationCount: polygonCount + 1,
          body: 'featureOffsets[featureOffsetsOffset + index] = index;'
        })
      );
      // The last round only measures the remaining crossings, so without `topologyStats` it would
      // be a full compaction plus segment intersection whose result nobody reads.
      const lastRound = output.topologyStats ? topologyRounds : topologyRounds - 1;
      for (let round = 0; round <= lastRound; round++) {
        const isInitial = round === 0;
        const isFinal = round === topologyRounds;
        nodes.push(
          ...getTopologyRoundNodes(graph, {
            id: `${id}-round-${round}`,
            positions,
            ringOffsets,
            polygonOffsets,
            featureOffsets,
            pointIds,
            vertexNext,
            ownedVertex,
            keepPoint,
            vertexCount,
            ringCount,
            pairCapacity,
            repair: !isFinal,
            statsView: workStats,
            crossingSlot: isInitial ? 0 : isFinal ? 1 : -1,
            roundIndex: round
          })
        );
      }
      if (output.topologyStats) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-topology-stats-copy`,
            operation: OPERATION,
            variant: 'topology-stats-copy',
            bindings: [
              {name: 'workStats', view: workStats, type: 'u32', access: 'read'},
              {name: 'topologyStats', view: output.topologyStats, type: 'u32', access: 'read_write'}
            ],
            invocationCount: TOPOLOGY_STATS_LENGTH,
            body: 'topologyStats[topologyStatsOffset + index] = workStats[workStatsOffset + index];'
          })
        );
      }
    }

    // Per-vertex mask, rank, and ring-wise compaction.
    const keepMask = output.keepMask ?? view('keep-mask', vertexCount);
    const keepRank = view('keep-rank', vertexCount);
    nodes.push(
      createKeepMaskNode<Parameters>(graph, `${id}-keep-mask`, pointIds, keepPoint, keepMask),
      ...new GPUScan({
        id: `${id}-keep-rank-scan`,
        input: keepMask,
        output: keepRank,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createEmitPositionsNode<Parameters>(graph, {
        id: `${id}-emit-positions`,
        positions,
        keepMask,
        keepRank,
        outputPositions: output.positions,
        vertexCount
      }),
      createEmitOffsetsNode<Parameters>(graph, {
        id: `${id}-emit-offsets`,
        ringOffsets,
        keepMask,
        keepRank,
        outputRingOffsets: output.ringOffsets,
        overflow: output.overflow,
        totalCount: output.totalCount,
        vertexCount,
        ringCount,
        capacity: outputCapacity
      })
    );
    return nodes;
  }
}

/** Per-vertex keep flag from the shared per-point keep decisions. */
function createKeepMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  pointIds: GraphDataView<'uint32'>,
  keepPoint: GraphDataView<'uint32'>,
  keepMask: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: 'keep-mask',
    bindings: [
      {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
      {name: 'keepPoint', view: keepPoint, type: 'u32', access: 'read'},
      {name: 'keepMask', view: keepMask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: pointIds.length,
    body: `let pointId = pointIds[pointIdsOffset + index];
  keepMask[keepMaskOffset + index] =
    select(0u, keepPoint[keepPointOffset + pointId], pointId != ${INVALID});`
  });
}

/** Writes the kept vertices, in order, to `outputPositions`, bounded by its capacity. */
function createEmitPositionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    keepMask: GraphDataView<'uint32'>;
    keepRank: GraphDataView<'uint32'>;
    outputPositions: GraphDataView<'float32x2'>;
    vertexCount: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'emit-positions',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'keepMask', view: props.keepMask, type: 'u32', access: 'read'},
      {name: 'keepRank', view: props.keepRank, type: 'u32', access: 'read'},
      {name: 'outputPositions', view: props.outputPositions, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.vertexCount,
    declarations: `const CAPACITY: u32 = ${props.outputPositions.length}u;`,
    body: `let rank = keepRank[keepRankOffset + index];
  if (keepMask[keepMaskOffset + index] != 0u && rank < CAPACITY) {
    outputPositions[outputPositionsOffset + rank * 2u] = positions[positionsOffset + index * 2u];
    outputPositions[outputPositionsOffset + rank * 2u + 1u] = positions[positionsOffset + index * 2u + 1u];
  }`
  });
}

/** Writes the ring offsets of the kept vertices, plus the overflow flag and optional total. */
function createEmitOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    ringOffsets: GraphDataView<'uint32'>;
    keepMask: GraphDataView<'uint32'>;
    keepRank: GraphDataView<'uint32'>;
    outputRingOffsets: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    totalCount?: GraphDataView<'uint32'>;
    vertexCount: number;
    ringCount: number;
    capacity: number;
  }
): GPUCommandNode<Parameters> {
  const {totalCount} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'emit-offsets',
    bindings: [
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'keepMask', view: props.keepMask, type: 'u32', access: 'read'},
      {name: 'keepRank', view: props.keepRank, type: 'u32', access: 'read'},
      {name: 'outputRingOffsets', view: props.outputRingOffsets, type: 'u32', access: 'read_write'},
      {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
      ...(totalCount
        ? [
            {
              name: 'totalCount',
              view: totalCount,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ],
    invocationCount: props.ringCount + 1,
    declarations: `const VERTEX_COUNT: u32 = ${props.vertexCount}u;
const RING_COUNT: u32 = ${props.ringCount}u;
const CAPACITY: u32 = ${props.capacity}u;`,
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
    ${totalCount ? 'totalCount[totalCountOffset] = total;' : ''}
  }`
  });
}

/**
 * Restores vertices of rings that kept fewer than three distinct points: the farthest original
 * vertex from the kept point, then the farthest from the line through the two. Reads the frozen
 * keep decisions and writes a separate buffer that is merged afterwards, so the result does not
 * depend on thread order.
 */
function getRingCollapseNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    vertexPolygon: GraphDataView<'uint32'>;
    pointIds: GraphDataView<'uint32'>;
    keepPoint: GraphDataView<'uint32'>;
    vertexCount: number;
    ringCount: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, vertexCount, ringCount} = props;
  const restorePoint = createTransientView(graph, `${id}-restore-point`, 'uint32', vertexCount);
  return [
    createFillNode<Parameters>(graph, {
      id: `${id}-restore-point-clear`,
      operation: OPERATION,
      view: restorePoint,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-ring-minimum`,
      operation: OPERATION,
      variant: 'ring-minimum',
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
        {name: 'pointIds', view: props.pointIds, type: 'u32', access: 'read'},
        {name: 'keepPoint', view: props.keepPoint, type: 'u32', access: 'read'},
        {name: 'restorePoint', view: restorePoint, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: ringCount,
      declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn isKept(vertex: u32) -> bool {
  let pointId = pointIds[pointIdsOffset + vertex];
  return pointId != ${INVALID} && keepPoint[keepPointOffset + pointId] != 0u;
}`,
      body: `let begin = ringOffsets[ringOffsetsOffset + index];
  var end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  if (end > begin + 1u && pointIds[pointIdsOffset + end - 1u] == pointIds[pointIdsOffset + begin]) {
    end = end - 1u;
  }
  var keptCount = 0u;
  var firstKept = ${INVALID};
  var secondKept = ${INVALID};
  for (var vertex = begin; vertex < end; vertex++) {
    if (isKept(vertex)) {
      if (keptCount == 0u) { firstKept = vertex; } else if (keptCount == 1u) { secondKept = vertex; }
      keptCount++;
    }
  }
  if (keptCount >= 1u && keptCount < 3u && end > begin + 2u) {
    let anchor = vertexAt(firstKept);
    var other = vertexAt(firstKept);
    var otherVertex = secondKept;
    var haveOther = keptCount >= 2u;
    if (haveOther) {
      other = vertexAt(secondKept);
    } else {
      var best = 0.0;
      for (var vertex = begin; vertex < end; vertex++) {
        if (!isKept(vertex) && pointIds[pointIdsOffset + vertex] != ${INVALID}) {
          let distanceToAnchor = distance(vertexAt(vertex), anchor);
          if (distanceToAnchor > best) { best = distanceToAnchor; otherVertex = vertex; }
        }
      }
      if (otherVertex != ${INVALID}) {
        haveOther = true;
        other = vertexAt(otherVertex);
        atomicMax(&restorePoint[restorePointOffset + pointIds[pointIdsOffset + otherVertex]], 1u);
      }
    }
    if (haveOther) {
      let direction = other - anchor;
      let directionLength = max(length(direction), 1e-30);
      var best = 0.0;
      var thirdVertex = ${INVALID};
      for (var vertex = begin; vertex < end; vertex++) {
        let pointId = pointIds[pointIdsOffset + vertex];
        if (!isKept(vertex) && pointId != ${INVALID} && vertex != otherVertex &&
            pointId != pointIds[pointIdsOffset + otherVertex]) {
          let offset = vertexAt(vertex) - anchor;
          let distanceToLine = abs(direction.x * offset.y - direction.y * offset.x) / directionLength;
          if (distanceToLine > best) { best = distanceToLine; thirdVertex = vertex; }
        }
      }
      if (thirdVertex != ${INVALID}) {
        atomicMax(&restorePoint[restorePointOffset + pointIds[pointIdsOffset + thirdVertex]], 1u);
      }
    }
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-merge-restored`,
      operation: OPERATION,
      variant: 'merge-restored',
      bindings: [
        {name: 'restorePoint', view: restorePoint, type: 'u32', access: 'read'},
        {name: 'keepPoint', view: props.keepPoint, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      body: 'keepPoint[keepPointOffset + index] = max(keepPoint[keepPointOffset + index], restorePoint[restorePointOffset + index]);'
    })
  ];
}

/**
 * Span ends and split candidates of the compacted simplification (see the call site). `spanKey`
 * receives, per kept vertex, the walk-order key of its farthest dropped vertex (`INVALID` when none
 * is strictly away from the span): the vertex index, plus `vertexCount` for vertices the walk
 * reaches only after wrapping past the ring end.
 */
function getSpanNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    keepMask: GraphDataView<'uint32'>;
    keepRank: GraphDataView<'uint32'>;
    spanEnd: GraphDataView<'uint32'>;
    spanKey: GraphDataView<'uint32'>;
    vertexCount: number;
    ringCount: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, vertexCount, ringCount, keepMask, keepRank, ringOffsets} = props;
  const keptVertex = createTransientView(graph, `${id}-kept-vertex`, 'uint32', vertexCount);
  const distanceBits = createTransientView(graph, `${id}-distance-bits`, 'uint32', vertexCount);
  const spanBest = createTransientView(graph, `${id}-span-best`, 'uint32', vertexCount);
  const shared = `const VERTEX_COUNT: u32 = ${vertexCount}u;
const RING_COUNT: u32 = ${ringCount}u;
fn getRing(vertex: u32) -> u32 {
  var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= vertex) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  return ring;
}
fn getKeptTotal() -> u32 {
  return keepRank[keepRankOffset + VERTEX_COUNT - 1u] + keepMask[keepMaskOffset + VERTEX_COUNT - 1u];
}
// Number of kept vertices before 'position'.
fn getKeptBefore(position: u32) -> u32 {
  return select(keepRank[keepRankOffset + position], getKeptTotal(), position >= VERTEX_COUNT);
}`;
  const readOnly = (name: string, view: GraphDataView) =>
    ({name, view, type: 'u32', access: 'read'}) as const;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-kept-vertex`,
      operation: OPERATION,
      variant: 'kept-vertex',
      bindings: [
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        {name: 'keptVertex', view: keptVertex, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      body: `if (keepMask[keepMaskOffset + index] != 0u) {
    keptVertex[keptVertexOffset + keepRank[keepRankOffset + index]] = index;
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-span-ends`,
      operation: OPERATION,
      variant: 'span-ends',
      bindings: [
        readOnly('ringOffsets', ringOffsets),
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        readOnly('keptVertex', keptVertex),
        {name: 'spanEnd', view: props.spanEnd, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: shared,
      // The next kept vertex in rank order, unless that one starts the next ring: then the first
      // kept vertex of this ring (a single kept vertex is its own span end).
      body: `var end = ${INVALID};
  if (keepMask[keepMaskOffset + index] != 0u) {
    let ring = getRing(index);
    let ringBegin = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    let rank = keepRank[keepRankOffset + index];
    end = keptVertex[keptVertexOffset + getKeptBefore(ringBegin)];
    if (rank + 1u < getKeptTotal()) {
      let candidate = keptVertex[keptVertexOffset + rank + 1u];
      if (candidate < ringEnd) {
        end = candidate;
      }
    }
  }
  spanEnd[spanEndOffset + index] = end;`
    }),
    createFillNode<Parameters>(graph, {
      id: `${id}-span-best-clear`,
      operation: OPERATION,
      view: spanBest,
      type: 'u32',
      value: '0u'
    }),
    createFillNode<Parameters>(graph, {
      id: `${id}-span-key-clear`,
      operation: OPERATION,
      view: props.spanKey,
      type: 'u32',
      value: INVALID
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-span-distance`,
      operation: OPERATION,
      variant: 'span-distance',
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        readOnly('ringOffsets', ringOffsets),
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        readOnly('keptVertex', keptVertex),
        readOnly('spanEnd', props.spanEnd),
        {name: 'distanceBits', view: distanceBits, type: 'u32', access: 'read_write'},
        {name: 'spanBest', view: spanBest, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: `${shared}
${SPAN_OWNER_WGSL}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`,
      // Distance bits of positive floats order like the floats, so atomicMax finds the farthest.
      body: `var bits = 0u;
  if (keepMask[keepMaskOffset + index] == 0u) {
    let owner = getSpanOwner(index);
    if (owner != ${INVALID}) {
      let a = vertexAt(owner);
      let p = vertexAt(index);
      let direction = vertexAt(spanEnd[spanEndOffset + owner]) - a;
      let segmentLength = length(direction);
      let offset = p - a;
      let separation = select(
        distance(p, a),
        abs(direction.x * offset.y - direction.y * offset.x) / max(segmentLength, 1e-30),
        segmentLength > 0.0);
      if (separation > 0.0) {
        bits = bitcast<u32>(separation);
        atomicMax(&spanBest[spanBestOffset + owner], bits);
      }
    }
  }
  distanceBits[distanceBitsOffset + index] = bits;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-span-split`,
      operation: OPERATION,
      variant: 'span-split',
      bindings: [
        readOnly('ringOffsets', ringOffsets),
        readOnly('keepMask', keepMask),
        readOnly('keepRank', keepRank),
        readOnly('keptVertex', keptVertex),
        readOnly('distanceBits', distanceBits),
        readOnly('spanBest', spanBest),
        {name: 'spanKey', view: props.spanKey, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      declarations: `${shared}
${SPAN_OWNER_WGSL}`,
      // Ties keep the vertex the walk from the span start meets first: later indices than the
      // span start come first, wrapped (smaller) indices after them.
      body: `let bits = distanceBits[distanceBitsOffset + index];
  if (bits != 0u && keepMask[keepMaskOffset + index] == 0u) {
    let owner = getSpanOwner(index);
    if (owner != ${INVALID} && bits == spanBest[spanBestOffset + owner]) {
      atomicMin(&spanKey[spanKeyOffset + owner], select(index + VERTEX_COUNT, index, index > owner));
    }
  }`
    })
  ];
}

/** WGSL: kept vertex whose span contains the dropped vertex, in the same ring, or INVALID. */
const SPAN_OWNER_WGSL = `
fn getSpanOwner(vertex: u32) -> u32 {
  let ring = getRing(vertex);
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let firstRank = getKeptBefore(ringBegin);
  let endRank = getKeptBefore(ringEnd);
  if (vertex < ringBegin || vertex >= ringEnd || endRank <= firstRank) {
    return ${INVALID};
  }
  // The kept vertex before it in the ring, else the ring's last one (the span wraps over the end).
  let before = keepRank[keepRankOffset + vertex];
  return keptVertex[keptVertexOffset + select(endRank - 1u, before - 1u, before > firstRank)];
}`;

/**
 * One topology round: compact the current simplified rings, find intersecting segment pairs,
 * classify them against the coverage points, count the offending ones and (when `repair`) restore
 * the farthest original vertex of every offending segment's span.
 */
function getTopologyRoundNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
    featureOffsets: GraphDataView<'uint32'>;
    pointIds: GraphDataView<'uint32'>;
    vertexNext: GraphDataView<'uint32'>;
    ownedVertex: GraphDataView<'uint32'>;
    keepPoint: GraphDataView<'uint32'>;
    vertexCount: number;
    ringCount: number;
    pairCapacity: number;
    repair: boolean;
    statsView: GraphDataView<'uint32'>;
    crossingSlot: number;
    roundIndex: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, vertexCount, ringCount, pairCapacity, statsView} = props;
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const mask = view('mask', vertexCount);
  const rank = view('rank', vertexCount);
  const compactPositions = createTransientView(
    graph,
    `${id}-compact-positions`,
    'float32x2',
    vertexCount
  );
  const compactRingOffsets = view('compact-ring-offsets', ringCount + 1);
  const compactOverflow = view('compact-overflow', 1);
  const spanEnd = view('span-end', vertexCount);
  const spanSplit = view('span-split', vertexCount);
  const rowInfo = view('row-info', vertexCount * 4);
  const badFlag = view('bad', vertexCount);
  const pairLeft = view('pair-left', pairCapacity);
  const pairRight = view('pair-right', pairCapacity);
  const pairCount = view('pair-count', 1);
  const pairOverflow = view('pair-overflow', 1);
  const pairKinds = view('pair-kinds', pairCapacity);
  const crossingSlot = props.crossingSlot >= 0 ? props.crossingSlot : 4 + props.roundIndex;

  const nodes: GPUCommandNode<Parameters>[] = [
    createKeepMaskNode<Parameters>(graph, `${id}-keep-mask`, props.pointIds, props.keepPoint, mask),
    ...new GPUScan({
      id: `${id}-rank-scan`,
      input: mask,
      output: rank,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createEmitPositionsNode<Parameters>(graph, {
      id: `${id}-emit-positions`,
      positions: props.positions,
      keepMask: mask,
      keepRank: rank,
      outputPositions: compactPositions,
      vertexCount
    }),
    createEmitOffsetsNode<Parameters>(graph, {
      id: `${id}-emit-offsets`,
      ringOffsets: props.ringOffsets,
      keepMask: mask,
      keepRank: rank,
      outputRingOffsets: compactRingOffsets,
      overflow: compactOverflow,
      vertexCount,
      ringCount,
      capacity: vertexCount
    }),
    // For every kept vertex: the next kept vertex of its ring and the farthest dropped vertex
    // between them (the vertex a Douglas-Peucker step would restore). Instead of one thread walking
    // each span twice (a long arc that simplifies to two points is a serial walk of its length),
    // the next kept vertex comes from the rank order and every dropped vertex scores itself
    // against the span that owns it: two atomic passes pick the maximum distance and, among equal
    // distances, the first vertex in walk order (the old walk's strict `>` rule).
    ...getSpanNodes(graph, {
      id,
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      keepMask: mask,
      keepRank: rank,
      spanEnd,
      spanKey: spanSplit,
      vertexCount,
      ringCount
    }),
    // Segment rows of the compacted rings: coverage points of both ends, vertex, ownership.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-row-info`,
      operation: OPERATION,
      variant: 'row-info',
      bindings: [
        {name: 'keepMask', view: mask, type: 'u32', access: 'read'},
        {name: 'keepRank', view: rank, type: 'u32', access: 'read'},
        {name: 'spanEnd', view: spanEnd, type: 'u32', access: 'read'},
        {name: 'pointIds', view: props.pointIds, type: 'u32', access: 'read'},
        {name: 'ownedVertex', view: props.ownedVertex, type: 'u32', access: 'read'},
        {name: 'rowInfo', view: rowInfo, type: 'u32', access: 'read_write'}
      ],
      invocationCount: vertexCount,
      body: `if (keepMask[keepMaskOffset + index] != 0u) {
    let base = rowInfoOffset + keepRank[keepRankOffset + index] * 4u;
    rowInfo[base] = pointIds[pointIdsOffset + index];
    rowInfo[base + 1u] = pointIds[pointIdsOffset + spanEnd[spanEndOffset + index]];
    rowInfo[base + 2u] = index;
    rowInfo[base + 3u] = ownedVertex[ownedVertexOffset + index];
  }`
    }),
    createFillNode<Parameters>(graph, {
      id: `${id}-bad-clear`,
      operation: OPERATION,
      view: badFlag,
      type: 'u32',
      value: '0u'
    }),
    ...new GPUSegmentIntersection({
      id: `${id}-intersect`,
      left: {
        kind: 'polygons',
        positions: compactPositions,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets,
        ringOffsets: compactRingOffsets
      },
      pairs: {
        leftIds: pairLeft,
        rightIds: pairRight,
        count: pairCount,
        overflow: pairOverflow
      },
      kinds: pairKinds
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-classify`,
      operation: OPERATION,
      variant: `classify-${crossingSlot}`,
      bindings: [
        {name: 'pairLeft', view: pairLeft, type: 'u32', access: 'read'},
        {name: 'pairRight', view: pairRight, type: 'u32', access: 'read'},
        {name: 'pairKinds', view: pairKinds, type: 'u32', access: 'read'},
        {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
        {name: 'pairOverflow', view: pairOverflow, type: 'u32', access: 'read'},
        {name: 'rowInfo', view: rowInfo, type: 'u32', access: 'read'},
        {name: 'badFlag', view: badFlag, type: 'atomic<u32>', access: 'read_write'},
        {name: 'stats', view: statsView, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: pairCapacity,
      declarations: `const SLOT: u32 = ${crossingSlot}u;`,
      // Pair kinds: 1 proper, 2 touch, 3 collinear touch, 4 overlap, 5 uncertain. Segments that
      // share a coverage point meet legitimately at it; identical segments are one shared edge.
      body: `if (index == 0u && pairOverflow[pairOverflowOffset] != 0u) {
    atomicMax(&stats[statsOffset + 3u], 1u);
  }
  if (index < pairCount[pairCountOffset]) {
    let kind = pairKinds[pairKindsOffset + index];
    let leftBase = rowInfoOffset + pairLeft[pairLeftOffset + index] * 4u;
    let rightBase = rowInfoOffset + pairRight[pairRightOffset + index] * 4u;
    let a0 = rowInfo[leftBase];
    let a1 = rowInfo[leftBase + 1u];
    let b0 = rowInfo[rightBase];
    let b1 = rowInfo[rightBase + 1u];
    let shares = a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1;
    let identical = (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0);
    let offending = select(!shares, !identical, kind == 4u);
    if (offending) {
      atomicMax(&badFlag[badFlagOffset + rowInfo[leftBase + 2u]], 1u);
      atomicMax(&badFlag[badFlagOffset + rowInfo[rightBase + 2u]], 1u);
      if (rowInfo[leftBase + 3u] != 0u && rowInfo[rightBase + 3u] != 0u) {
        atomicAdd(&stats[statsOffset + SLOT], 1u);
      }
    }
  }`
    })
  ];
  if (props.repair) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-repair`,
        operation: OPERATION,
        variant: 'repair',
        bindings: [
          {name: 'badFlag', view: badFlag, type: 'u32', access: 'read'},
          {name: 'spanSplit', view: spanSplit, type: 'u32', access: 'read'},
          {name: 'pointIds', view: props.pointIds, type: 'u32', access: 'read'},
          {name: 'keepPoint', view: props.keepPoint, type: 'atomic<u32>', access: 'read_write'},
          {name: 'stats', view: statsView, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        // spanSplit holds the first-in-walk-order key: vertex, or vertex + VERTEX_COUNT when the
        // walk wrapped past the ring end.
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `let key = spanSplit[spanSplitOffset + index];
  let split = select(key, key - VERTEX_COUNT, key >= VERTEX_COUNT && key != ${INVALID});
  if (badFlag[badFlagOffset + index] != 0u && key != ${INVALID}) {
    let pointId = pointIds[pointIdsOffset + split];
    if (pointId != ${INVALID}) {
      if (atomicMax(&keepPoint[keepPointOffset + pointId], 1u) == 0u) {
        atomicAdd(&stats[statsOffset + 2u], 1u);
      }
    }
  }`
      })
    );
  }
  return nodes;
}
