// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getStorageBindingElementLimit} from '../../utils/gpu-contributor-utils';
import {createSaturatedScanTotalNodes} from '../../utils/scan-total';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';

/**
 * Polygon edge-slab index (internal building block).
 *
 * An OWNER is a contiguous range of rings: one feature (all rings of all its polygons) or one
 * polygon. Each slabbed owner splits its y-range into `n` equal slabs (about one slab per four
 * vertices, at most {@link POLYGON_EDGE_SLAB_MAXIMUM_SLABS}) and lists every ring edge in each slab
 * between the slab of the edge's lower end and the slab of its upper end. A query at height `y`
 * then visits only the edges of one slab instead of all edges of the owner.
 *
 * INVARIANT (what makes the index exact): `edgeSlabOf` is a monotone f32 expression (subtract,
 * multiply by a positive scale, floor, clamp). For an edge with `min(a.y, b.y) <= y <= max(a.y, b.y)`
 * the slab of `y` therefore lies between the slabs of the two endpoints, so every edge whose CLOSED
 * y-range contains `y` (straddling it, ending on it or lying horizontally on it) is listed in the
 * slab of `y`. Build and query share the same WGSL function, so they round alike. A consumer that
 * visits the edges of `slabOf(y)` sees a superset of the edges a brute-force walk would have
 * accepted for that height, in a different order.
 *
 * The index is built by `getPolygonEdgeSlabNodes` and queried with the WGSL of
 * `getPolygonEdgeSlabQueryWGSL` and `getPolygonEdgeSlabLoopWGSL`.
 */

/** Maximum number of slabs of one owner. @internal */
export const POLYGON_EDGE_SLAB_MAXIMUM_SLABS = 4096;

/** Owners with fewer vertices than this are not slabbed unless `minimumVertexCount` says otherwise. @internal */
export const POLYGON_EDGE_SLAB_DEFAULT_MINIMUM_VERTEX_COUNT = 8;

/**
 * Number of `uint32` words per owner in the params table. @internal
 *
 * Word 0: slab count (0 = not slabbed, the owner's edges are walked). Word 1: index of the owner's
 * first slab in the `slabOffsets` table. Word 2: slab origin (f32 bits, the lowest finite y).
 * Word 3: slab scale (f32 bits, slabs per unit of y). Words 4 and 5: the owner's first vertex and
 * the end of its last ring (`[vertexBegin, vertexEnd)`; the unslabbed walk range).
 */
export const POLYGON_EDGE_SLAB_PARAMS_STRIDE = 6;

/** Granularity of an owner. @internal */
export type PolygonEdgeSlabOwner = 'feature' | 'polygon';

/** Built views of a polygon edge-slab index. @internal */
export type PolygonEdgeSlabs = {
  /** Owner granularity the index was built for. */
  readonly owner: PolygonEdgeSlabOwner;
  /** Number of owners (features or polygons). */
  readonly ownerCount: number;
  /** Number of vertices of the polygon layout (also the length of the unslabbed edge region). */
  readonly vertexCount: number;
  /** Capacity of the slab entries; an owner whose entries exceed it is walked instead. */
  readonly entryCapacity: number;
  /** Slab slots over all owners. */
  readonly slabCapacity: number;
  /** Packed `ownerCount * POLYGON_EDGE_SLAB_PARAMS_STRIDE` params table (see the stride). */
  readonly params: GraphDataView<'uint32'>;
  /**
   * Exclusive scan of the per-slab entry counts, `slabCapacity + 1` words, indexed by
   * `params[owner].base + slab`. The entries of a slab are `[offsets[i], offsets[i + 1])`.
   */
  readonly slabOffsets: GraphDataView<'uint32'>;
  /**
   * `vertexCount + entryCapacity` pairs of vertex indices `(first, second)` of an edge. Pairs
   * `[0, vertexCount)` are the unslabbed edge region: pair `v` is `(v, next vertex of the ring)`, or
   * `(0xffffffff, 0xffffffff)` for a vertex outside every owner. Slab entry `e` is pair
   * `vertexCount + e`. Entries within one slab are in no guaranteed order (the fill claims slots with
   * an atomic cursor), so a consumer must be order independent: parity, any-hit, or an extremum with
   * a tie-break on the vertex indices.
   */
  readonly slabEntries: GraphDataView<'uint32x2'>;
};

/** Properties of {@link getPolygonEdgeSlabNodes}. @internal */
export type PolygonEdgeSlabProps = {
  /** Node and view ID prefix: nodes are `${id}-slab-*` and `${id}-ring-owner`. */
  id: string;
  /** `workload.operation` name of the generated nodes. */
  operation: string;
  /** Owner granularity. `'feature'` needs `featureOffsets`. */
  owner: PolygonEdgeSlabOwner;
  /** Vertex positions (`float32x2`). Rings are implicitly closed (last vertex joins the first). */
  positions: GraphDataView<'float32x2'>;
  /** GeoArrow polygon offsets of features, required for `owner: 'feature'`. */
  featureOffsets?: GraphDataView<'uint32'>;
  /** GeoArrow ring offsets of polygons. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** GeoArrow vertex offsets of rings. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Entry capacity over all owners. Default `max(4 * vertexCount, 1024)`, capped to fit the binding. */
  entryCapacity?: number;
  /** Owners with fewer vertices are not slabbed. Default {@link POLYGON_EDGE_SLAB_DEFAULT_MINIMUM_VERTEX_COUNT}. */
  minimumVertexCount?: number;
  /**
   * Optional per-owner work gate: an owner is slabbed only when `counts[owner] >= minimum`
   * (`counts` has at least `ownerCount` words). Cell cover gates on the candidate count.
   */
  gate?: {counts: GraphDataView<'uint32'>; minimum: number};
};

/** Result of {@link getPolygonEdgeSlabNodes}. @internal */
export type PolygonEdgeSlabNodes<Parameters> = {
  /** Nodes in execution order. */
  nodes: GPUCommandNode<Parameters>[];
  /** Views of the built index. */
  slabs: PolygonEdgeSlabs;
};

/**
 * WGSL shared by build and query: parameter word indices and `edgeSlabOf`, the monotone slab
 * function. Bindings are always named `edgeSlabParams`, `edgeSlabOffsets` and `edgeSlabEntries`.
 */
const EDGE_SLAB_COMMON_WGSL = /* wgsl */ `
const EDGE_SLAB_NO_VERTEX: u32 = 0xffffffffu;
const EDGE_SLAB_PARAMS_STRIDE: u32 = ${POLYGON_EDGE_SLAB_PARAMS_STRIDE}u;
const EDGE_SLAB_COUNT_WORD: u32 = 0u;
const EDGE_SLAB_BASE_WORD: u32 = 1u;
const EDGE_SLAB_ORIGIN_WORD: u32 = 2u;
const EDGE_SLAB_SCALE_WORD: u32 = 3u;
const EDGE_SLAB_VERTEX_BEGIN_WORD: u32 = 4u;
const EDGE_SLAB_VERTEX_END_WORD: u32 = 5u;

// Monotone in y for a positive scale (subtract, multiply, floor, clamp). Do not rewrite it as an
// fma or in any other form: edges are listed by this exact expression.
fn edgeSlabOf(y: f32, slabCount: u32, origin: f32, scale: f32) -> u32 {
  if (slabCount <= 1u) { return 0u; }
  return min(u32(clamp(floor((y - origin) * scale), 0.0, f32(slabCount - 1u))), slabCount - 1u);
}
`;

/** Bindings of the query side, always three views: params, slab offsets and slab entries. @internal */
export function getPolygonEdgeSlabBindings(slabs: PolygonEdgeSlabs): WGSLKernelBinding[] {
  return [
    {name: 'edgeSlabParams', view: slabs.params, type: 'u32', access: 'read'},
    {name: 'edgeSlabOffsets', view: slabs.slabOffsets, type: 'u32', access: 'read'},
    {name: 'edgeSlabEntries', view: slabs.slabEntries, type: 'u32', access: 'read'}
  ];
}

/**
 * Module-scope WGSL of the query side. Needs the three bindings of
 * {@link getPolygonEdgeSlabBindings}. Defines
 *
 * - `edgeSlabIsSlabbed(owner: u32) -> bool`: whether the owner is walked through its slabs (false
 *   for tiny owners, non-finite vertices, zero height, gated-out owners and entry overflow).
 * - `edgeSlabOfHeight(owner: u32, y: f32) -> u32`: slab of `y` within a slabbed owner. A NaN `y`
 *   returns an unspecified slab inside the owner's range; guard it in the consumer.
 *
 * Use {@link getPolygonEdgeSlabLoopWGSL} for the edge loop. @internal
 */
export function getPolygonEdgeSlabQueryWGSL(
  slabs: Pick<PolygonEdgeSlabs, 'vertexCount' | 'entryCapacity'>
): string {
  return /* wgsl */ `${EDGE_SLAB_COMMON_WGSL}
const EDGE_SLAB_ENTRY_CAPACITY: u32 = ${slabs.entryCapacity}u;
const EDGE_SLAB_VERTEX_COUNT: u32 = ${slabs.vertexCount}u;

fn edgeSlabIsSlabbed(owner: u32) -> bool {
  let ownerBase = edgeSlabParamsOffset + owner * EDGE_SLAB_PARAMS_STRIDE;
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  return slabCount != 0u &&
    edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD] + slabCount] <= EDGE_SLAB_ENTRY_CAPACITY;
}

fn edgeSlabOfHeight(owner: u32, y: f32) -> u32 {
  let ownerBase = edgeSlabParamsOffset + owner * EDGE_SLAB_PARAMS_STRIDE;
  return edgeSlabOf(
    y,
    edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD],
    bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD]),
    bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD])
  );
}
`;
}

/** Options of {@link getPolygonEdgeSlabLoopWGSL}. @internal */
export type PolygonEdgeSlabLoopOptions = {
  /** WGSL `u32` expression of the owner (a local or constant: it is evaluated several times). */
  owner: string;
  /** WGSL `f32` expression of the lowest query height. */
  lowY: string;
  /** WGSL `f32` expression of the highest query height (the same as `lowY` for a point query). */
  highY: string;
  /**
   * Loop body, run once per candidate edge from ONE call site. It sees `u32` locals named
   * `edgeStart` and `edgeEnd` (vertex indices; override with `indexNames`) and, when `getVertex` is
   * given, `vec2f` locals `a` and `b` (override with `pointNames`). Use `return` to leave the
   * enclosing function early; `continue` skips the edge; `break` leaves only the current slab.
   */
  body: string;
  /** Name of a consumer WGSL function `(vertex: u32) -> vec2f`; omit to load the vertices yourself. */
  getVertex?: string;
  /** Names of the vertex index locals. Default `['edgeStart', 'edgeEnd']`. */
  indexNames?: readonly [string, string];
  /** Names of the vertex position locals. Default `['a', 'b']`. */
  pointNames?: readonly [string, string];
};

/**
 * WGSL statements (a block) that loop over the candidate edges of `owner` for query heights
 * `[lowY, highY]`: for a slabbed owner, the edges of the slabs from `slabOf(lowY)` to `slabOf(highY)`
 * (a superset of every edge whose closed y-range meets the interval, see the module invariant);
 * for an unslabbed owner, all edges of its rings (the heights are ignored). The loop body is emitted
 * once, so a consumer keeps a single call site for large exact helpers. The edges of a slab come in
 * no guaranteed order and an edge listed in several slabs is visited once per slab in the range, so
 * for height ranges spanning several slabs use idempotent tests (any-hit), and for single heights
 * parity or extremum tests. No loop index reaches the source text (rule 12): all sizes are baked
 * by {@link getPolygonEdgeSlabQueryWGSL}, which must be declared. Locals are prefixed `edgeSlab`.
 * @internal
 */
export function getPolygonEdgeSlabLoopWGSL(options: PolygonEdgeSlabLoopOptions): string {
  const [startName, endName] = options.indexNames ?? ['edgeStart', 'edgeEnd'];
  const [firstPoint, secondPoint] = options.pointNames ?? ['a', 'b'];
  const loadPoints = options.getVertex
    ? `let ${firstPoint} = ${options.getVertex}(${startName});
      let ${secondPoint} = ${options.getVertex}(${endName});`
    : '';
  return `{
    let edgeSlabOwner = ${options.owner};
    let edgeSlabOwnerBase = edgeSlabParamsOffset + edgeSlabOwner * EDGE_SLAB_PARAMS_STRIDE;
    let edgeSlabIsWalked = !edgeSlabIsSlabbed(edgeSlabOwner);
    var edgeSlabFirst = 0u;
    var edgeSlabLast = 0u;
    if (!edgeSlabIsWalked) {
      edgeSlabFirst = edgeSlabOfHeight(edgeSlabOwner, min(${options.lowY}, ${options.highY}));
      edgeSlabLast = edgeSlabOfHeight(edgeSlabOwner, max(${options.lowY}, ${options.highY}));
    }
    let edgeSlabBase = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_BASE_WORD];
    for (var edgeSlabIndex = edgeSlabFirst; edgeSlabIndex <= edgeSlabLast; edgeSlabIndex++) {
      var edgeSlabBegin = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_VERTEX_BEGIN_WORD];
      var edgeSlabEnd = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_VERTEX_END_WORD];
      var edgeSlabShift = 0u;
      if (!edgeSlabIsWalked) {
        edgeSlabBegin = edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabBase + edgeSlabIndex];
        edgeSlabEnd = edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabBase + edgeSlabIndex + 1u];
        edgeSlabShift = EDGE_SLAB_VERTEX_COUNT;
      }
      for (var edgeSlabEntry = edgeSlabBegin; edgeSlabEntry < edgeSlabEnd; edgeSlabEntry++) {
        let ${startName} = edgeSlabEntries[edgeSlabEntriesOffset + 2u * (edgeSlabShift + edgeSlabEntry)];
        if (${startName} == EDGE_SLAB_NO_VERTEX) { continue; }
        let ${endName} = edgeSlabEntries[edgeSlabEntriesOffset + 2u * (edgeSlabShift + edgeSlabEntry) + 1u];
        ${loadPoints}
        ${options.body}
      }
    }
  }`;
}

function getOwnerPolygonRangeWGSL(owner: PolygonEdgeSlabOwner): string {
  return owner === 'feature'
    ? `let firstPolygon = featureOffsets[featureOffsetsOffset + index];
  let endPolygon = featureOffsets[featureOffsetsOffset + index + 1u];`
    : `let firstPolygon = index;
  let endPolygon = index + 1u;`;
}

function getBuildVertexWGSL(): string {
  return /* wgsl */ `
fn edgeSlabGetVertex(vertex: u32) -> vec2f {
  let vertexBase = positionsOffset + 2u * vertex;
  return vec2f(positions[vertexBase], positions[vertexBase + 1u]);
}
`;
}

function validateOwnerLayout(props: PolygonEdgeSlabProps): void {
  if (props.owner === 'feature' && !props.featureOffsets) {
    throw new Error(
      `${props.operation} polygon edge slabs with owner 'feature' need featureOffsets`
    );
  }
  if (props.polygonOffsets.length < 1 || props.ringOffsets.length < 1) {
    throw new Error(`${props.operation} polygon edge slabs need terminal polygon and ring offsets`);
  }
  if (props.entryCapacity !== undefined && !(props.entryCapacity >= 1)) {
    throw new Error(`${props.operation} polygon edge slab entryCapacity must be at least 1`);
  }
}

/**
 * Builds the polygon edge-slab index: per-owner slab parameters (about one slab per four vertices,
 * capped by total vertical edge travel to fit the entry budget; none for tiny owners,
 * non-finite vertices, zero height, gated-out owners), a scan of the
 * slab counts into bases, a ring-to-owner map, then a counting sort of the edges into slabs (count,
 * scan, fill). Every kernel is vertex, ring or owner parallel; there is no host iteration.
 *
 * Entry capacity overflow: the owners whose cumulative entry count exceeds `entryCapacity` are
 * walked (all their edges) by the query helpers. Entries within a slab are unordered (see
 * {@link PolygonEdgeSlabs.slabEntries}); the unslabbed walk is in vertex order.
 *
 * The nodes read `gate.counts` when given, so they must run after it is written. @internal
 */
export function getPolygonEdgeSlabNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PolygonEdgeSlabProps
): PolygonEdgeSlabNodes<Parameters> {
  validateOwnerLayout(props);
  const {id, operation, owner, positions, polygonOffsets, ringOffsets} = props;
  const featureOffsets = props.featureOffsets;
  const vertexCount = positions.length;
  const ownerCount = (owner === 'feature' ? featureOffsets!.length : polygonOffsets.length) - 1;
  const slabCapacity = Math.floor(vertexCount / 4) + 1;
  // The entries share one uint32x2 binding with the unslabbed edges. Owners that do
  // not fit still use the exact walk; the optional index must fit the device limit.
  const entryCapacity = Math.min(
    props.entryCapacity ?? Math.max(4 * vertexCount, 1024),
    Math.max(0, getStorageBindingElementLimit(graph.device, 8, vertexCount))
  );
  const minimumVertexCount =
    props.minimumVertexCount ?? POLYGON_EDGE_SLAB_DEFAULT_MINIMUM_VERTEX_COUNT;
  const ringCount = ringOffsets.length - 1;
  const polygonCount = polygonOffsets.length - 1;

  const params = createTransientView(
    graph,
    `${id}-slab-params`,
    'uint32',
    Math.max(ownerCount, 1) * POLYGON_EDGE_SLAB_PARAMS_STRIDE
  );
  const slabNumbers = createTransientView(graph, `${id}-slab-numbers`, 'uint32', ownerCount + 1);
  const slabStarts = createTransientView(graph, `${id}-slab-starts`, 'uint32', ownerCount + 1);
  const ringOwner = createTransientView(
    graph,
    `${id}-ring-owner`,
    'uint32',
    Math.max(ringCount, 1)
  );
  const vertexOwner = createTransientView(graph, `${id}-vertex-owner`, 'uint32', vertexCount);
  const slabCounters = createTransientView(
    graph,
    `${id}-slab-counters`,
    'uint32',
    slabCapacity + 1
  );
  const slabOffsets = createTransientView(graph, `${id}-slab-offsets`, 'uint32', slabCapacity + 1);
  const slabEntries = createTransientView(
    graph,
    `${id}-slab-entries`,
    'uint32x2',
    vertexCount + entryCapacity
  );
  const slabs: PolygonEdgeSlabs = {
    owner,
    ownerCount,
    vertexCount,
    entryCapacity,
    slabCapacity,
    params,
    slabOffsets,
    slabEntries
  };

  const polygonBindings: WGSLKernelBinding[] = [
    {name: 'positions', view: positions, type: 'f32', access: 'read'},
    ...(owner === 'feature'
      ? [{name: 'featureOffsets', view: featureOffsets!, type: 'u32', access: 'read'} as const]
      : []),
    {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
    {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'}
  ];

  // Per owner: vertex range, finite y extent, slab count and lattice. One thread per owner plus a
  // trailing zero slab number so the scan has a total.
  const parametersNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-slab-params`,
    operation,
    variant: 'slab-params',
    bindings: [
      ...polygonBindings,
      {name: 'edgeSlabParams', view: params, type: 'u32', access: 'read_write'},
      {name: 'slabNumbers', view: slabNumbers, type: 'u32', access: 'read_write'},
      ...(props.gate
        ? [{name: 'gateCounts', view: props.gate.counts, type: 'u32', access: 'read'} as const]
        : [])
    ],
    invocationCount: ownerCount + 1,
    declarations: `${IS_FINITE_BITS_WGSL}${EDGE_SLAB_COMMON_WGSL}${getBuildVertexWGSL()}
const OWNER_COUNT: u32 = ${ownerCount}u;
const MINIMUM_VERTEX_COUNT: u32 = ${minimumVertexCount}u;
const MAXIMUM_SLABS: u32 = ${POLYGON_EDGE_SLAB_MAXIMUM_SLABS}u;
const ENTRY_BUDGET_PER_VERTEX: f32 = ${Math.max(0, entryCapacity / Math.max(vertexCount, 1)).toFixed(8)};
fn edgeSlabIsFinite(p: vec2f) -> bool {
  return isFiniteBits(p.x) && isFiniteBits(p.y) && abs(p.x) < 1e30 && abs(p.y) < 1e30;
}`,
    body: `if (index == OWNER_COUNT) { slabNumbers[slabNumbersOffset + index] = 0u; return; }
  ${getOwnerPolygonRangeWGSL(owner)}
  var vertexBegin = 0u;
  var vertexEnd = 0u;
  var vertexCount = 0u;
  var isFirstRing = true;
  var hasNonFinite = false;
  var verticalTravel = 0.0;
  var minY = 1e30;
  var maxY = -1e30;
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      if (isFirstRing) {
        vertexBegin = firstVertex;
        isFirstRing = false;
      }
      vertexEnd = endVertex;
      if (endVertex > firstVertex) {
        vertexCount += endVertex - firstVertex;
      }
      var previousY = 0.0;
      if (endVertex > firstVertex) { previousY = edgeSlabGetVertex(endVertex - 1u).y; }
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let p = edgeSlabGetVertex(vertex);
        if (edgeSlabIsFinite(p)) {
          minY = min(minY, p.y);
          maxY = max(maxY, p.y);
          verticalTravel += abs(p.y - previousY);
        } else {
          hasNonFinite = true;
        }
        previousY = p.y;
      }
    }
  }
  ${props.gate ? `let isGated = gateCounts[gateCountsOffset + index] >= ${props.gate.minimum}u;` : 'let isGated = true;'}
  var slabCount = 0u;
  var slabScale = 0.0;
  if (isGated && !hasNonFinite && vertexCount >= MINIMUM_VERTEX_COUNT) {
    // An edge contributes at most 2 + abs(dy) * scale entries. Reserve two entries per
    // vertex, and use half the remaining budget for travel (rounding/accumulation headroom).
    // Vertex count alone chose 4096 slabs even for jagged rings, overflowing 4N entries and
    // turning the refine into a full indirect walk. All statistics here are GPU-resident.
    let travelBudget = max(0.0, ENTRY_BUDGET_PER_VERTEX - 2.0) * 0.5 * f32(vertexCount);
    var wanted = min(vertexCount / 4u, MAXIMUM_SLABS);
    if (verticalTravel > 0.0) {
      wanted = min(wanted, max(1u, u32(floor(travelBudget * ((maxY - minY) / verticalTravel)))));
    }
    let scale = f32(wanted) / (maxY - minY);
    if (scale > 0.0 && scale < 1e30) {
      slabCount = wanted;
      slabScale = scale;
    }
  }
  let ownerBase = edgeSlabParamsOffset + index * EDGE_SLAB_PARAMS_STRIDE;
  edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] = slabCount;
  edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD] = bitcast<u32>(minY);
  edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD] = bitcast<u32>(slabScale);
  edgeSlabParams[ownerBase + EDGE_SLAB_VERTEX_BEGIN_WORD] = vertexBegin;
  edgeSlabParams[ownerBase + EDGE_SLAB_VERTEX_END_WORD] = vertexEnd;
  slabNumbers[slabNumbersOffset + index] = slabCount;`
  });

  // First slab index of every owner; drops the slabs of owners beyond the slab capacity.
  const basesNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-slab-bases`,
    operation,
    variant: 'slab-bases',
    bindings: [
      {name: 'edgeSlabParams', view: params, type: 'u32', access: 'read_write'},
      {name: 'slabStarts', view: slabStarts, type: 'u32', access: 'read'}
    ],
    invocationCount: Math.max(ownerCount, 1),
    declarations: `${EDGE_SLAB_COMMON_WGSL}
const OWNER_COUNT: u32 = ${ownerCount}u;
const SLAB_CAPACITY: u32 = ${slabCapacity}u;`,
    body: `if (index >= OWNER_COUNT) { return; }
  let ownerBase = edgeSlabParamsOffset + index * EDGE_SLAB_PARAMS_STRIDE;
  let base = slabStarts[slabStartsOffset + index];
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  if (slabCount != 0u && base + slabCount > SLAB_CAPACITY) {
    edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] = 0u;
  }
  edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD] = base;`
  });

  // One thread per ring: its owner by binary search, or 0xffffffff for rings outside every owner.
  const ringOwnerNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-ring-owner`,
    operation,
    variant: 'ring-owner',
    bindings: [
      ...(owner === 'feature'
        ? [{name: 'featureOffsets', view: featureOffsets!, type: 'u32', access: 'read'} as const]
        : []),
      {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
      {name: 'ringOwner', view: ringOwner, type: 'u32', access: 'read_write'}
    ],
    invocationCount: Math.max(ringCount, 1),
    declarations: `const OWNER_COUNT: u32 = ${ownerCount}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const RING_COUNT: u32 = ${ringCount}u;
const NO_OWNER: u32 = 0xffffffffu;`,
    body: `if (index >= RING_COUNT) { return; }
  var polygon = 0u;
  var high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  var ringOwnerValue = NO_OWNER;
  if (POLYGON_COUNT > 0u && index >= polygonOffsets[polygonOffsetsOffset + polygon] &&
      index < polygonOffsets[polygonOffsetsOffset + polygon + 1u]) {
    ${
      owner === 'feature'
        ? `var low = 0u;
    high = OWNER_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (featureOffsets[featureOffsetsOffset + middle] <= polygon) {
        low = middle;
      } else {
        high = middle;
      }
    }
    if (OWNER_COUNT > 0u && polygon >= featureOffsets[featureOffsetsOffset + low] &&
        polygon < featureOffsets[featureOffsetsOffset + low + 1u]) {
      ringOwnerValue = low;
    }`
        : 'ringOwnerValue = polygon;'
    }
  }
  ringOwner[ringOwnerOffset + index] = ringOwnerValue;`
  });

  const clearCounters = (name: string) =>
    createFillNode<Parameters>(graph, {
      id: `${id}-slab-clear-${name}`,
      operation,
      view: slabCounters,
      type: 'u32',
      value: '0u'
    });

  const edgeDeclarations = `${EDGE_SLAB_COMMON_WGSL}${getBuildVertexWGSL()}
const RING_COUNT: u32 = ${ringCount}u;
const ENTRY_CAPACITY: u32 = ${entryCapacity}u;
const VERTEX_COUNT: u32 = ${vertexCount}u;

/** Slab range [first, last] of the edge a-b within the owner's lattice. */
fn edgeSlabGetEdgeRange(ownerBase: u32, a: vec2f, b: vec2f) -> vec2u {
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  let origin = bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD]);
  let scale = bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD]);
  return vec2u(
    edgeSlabOf(min(a.y, b.y), slabCount, origin, scale),
    edgeSlabOf(max(a.y, b.y), slabCount, origin, scale)
  );
}`;

  // Count phase: next vertex and owner per vertex (the unslabbed edge region of the entries), and
  // one count per slab an edge spans.
  const countNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-slab-edges-count`,
    operation,
    variant: 'slab-edges-count',
    bindings: [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'edgeSlabParams', view: params, type: 'u32', access: 'read'},
      {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
      {name: 'ringOwner', view: ringOwner, type: 'u32', access: 'read'},
      {name: 'vertexOwner', view: vertexOwner, type: 'u32', access: 'read_write'},
      {name: 'edgeSlabEntries', view: slabEntries, type: 'u32', access: 'read_write'},
      {name: 'slabCounters', view: slabCounters, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: Math.max(vertexCount, 1),
    declarations: edgeDeclarations,
    body: `if (index >= VERTEX_COUNT) { return; }
  var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  var ownerIndex = EDGE_SLAB_NO_VERTEX;
  var next = EDGE_SLAB_NO_VERTEX;
  if (RING_COUNT > 0u && index >= ringBegin && index < ringEnd) {
    ownerIndex = ringOwner[ringOwnerOffset + ring];
    if (ownerIndex != EDGE_SLAB_NO_VERTEX) {
      next = select(index + 1u, ringBegin, index + 1u == ringEnd);
    }
  }
  edgeSlabEntries[edgeSlabEntriesOffset + 2u * index] = select(EDGE_SLAB_NO_VERTEX, index, next != EDGE_SLAB_NO_VERTEX);
  edgeSlabEntries[edgeSlabEntriesOffset + 2u * index + 1u] = next;
  vertexOwner[vertexOwnerOffset + index] = select(EDGE_SLAB_NO_VERTEX, ownerIndex, next != EDGE_SLAB_NO_VERTEX);
  if (next == EDGE_SLAB_NO_VERTEX) { return; }
  let ownerBase = edgeSlabParamsOffset + ownerIndex * EDGE_SLAB_PARAMS_STRIDE;
  if (edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] == 0u) { return; }
  let edgeRange = edgeSlabGetEdgeRange(ownerBase, edgeSlabGetVertex(index), edgeSlabGetVertex(next));
  let base = edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD];
  for (var slab = edgeRange.x; slab <= edgeRange.y; slab++) {
    atomicAdd(&slabCounters[slabCountersOffset + base + slab], 1u);
  }`
  });

  // Fill phase: each edge claims a position in every slab it spans through a per-slab cursor.
  const fillNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-slab-edges-fill`,
    operation,
    variant: 'slab-edges-fill',
    bindings: [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'edgeSlabParams', view: params, type: 'u32', access: 'read'},
      {name: 'vertexOwner', view: vertexOwner, type: 'u32', access: 'read'},
      {name: 'edgeSlabOffsets', view: slabOffsets, type: 'u32', access: 'read'},
      {name: 'edgeSlabEntries', view: slabEntries, type: 'u32', access: 'read_write'},
      {name: 'slabCounters', view: slabCounters, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: Math.max(vertexCount, 1),
    declarations: edgeDeclarations,
    body: `if (index >= VERTEX_COUNT) { return; }
  let ownerIndex = vertexOwner[vertexOwnerOffset + index];
  if (ownerIndex == EDGE_SLAB_NO_VERTEX) { return; }
  let ownerBase = edgeSlabParamsOffset + ownerIndex * EDGE_SLAB_PARAMS_STRIDE;
  if (edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] == 0u) { return; }
  let next = edgeSlabEntries[edgeSlabEntriesOffset + 2u * index + 1u];
  let edgeRange = edgeSlabGetEdgeRange(ownerBase, edgeSlabGetVertex(index), edgeSlabGetVertex(next));
  let base = edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD];
  for (var slab = edgeRange.x; slab <= edgeRange.y; slab++) {
    let slabIndex = base + slab;
    let position = edgeSlabOffsets[edgeSlabOffsetsOffset + slabIndex] +
      atomicAdd(&slabCounters[slabCountersOffset + slabIndex], 1u);
    if (position < ENTRY_CAPACITY) {
      edgeSlabEntries[edgeSlabEntriesOffset + 2u * (VERTEX_COUNT + position)] = index;
      edgeSlabEntries[edgeSlabEntriesOffset + 2u * (VERTEX_COUNT + position) + 1u] = next;
    }
  }`
  });

  const scanTotal = createTransientView(graph, `${id}-slab-entry-total`, 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    parametersNode,
    ...new GPUScan({
      id: `${id}-slab-number-scan`,
      input: slabNumbers,
      output: slabStarts,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    basesNode,
    ringOwnerNode,
    clearCounters('count'),
    countNode,
    ...new GPUScan({
      id: `${id}-slab-offset-scan`,
      input: slabCounters,
      output: slabOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    ...createSaturatedScanTotalNodes<Parameters>(graph, {
      id: `${id}-slab-entry`,
      operation,
      counts: slabCounters,
      offsets: slabOffsets,
      rowCount: slabCapacity,
      total: scanTotal
    }),
    // Wrapped offsets can alias earlier owners' slots. Disable the index before filling so every
    // owner uses the existing exact unslabbed walk instead (there is no public slab status).
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-slab-entry-fallback`,
      operation,
      variant: 'slab-entry-fallback',
      bindings: [
        {name: 'scanTotal', view: scanTotal, type: 'u32', access: 'read'},
        {name: 'edgeSlabParams', view: params, type: 'u32', access: 'read_write'}
      ],
      invocationCount: ownerCount,
      body: `if (scanTotal[scanTotalOffset] == 0xffffffffu) {
    edgeSlabParams[edgeSlabParamsOffset + index * ${POLYGON_EDGE_SLAB_PARAMS_STRIDE}u] = 0u;
  }`
    }),
    clearCounters('cursor'),
    fillNode
  ];
  return {nodes, slabs};
}
