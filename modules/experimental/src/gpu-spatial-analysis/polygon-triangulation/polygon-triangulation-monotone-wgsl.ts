// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL of the monotone path of `GPUPolygonTriangulation` (internal).
 *
 * # Algorithm
 *
 * Polygons routed to this path are triangulated by a parallel y-monotone decomposition (de Berg et
 * al., Computational Geometry, chapter 3), reformulated as data-parallel steps:
 *
 * 1. Clean the rings (drop consecutive duplicate vertices and the repeated closing vertex) and
 *    rank every vertex in sweep order: y descending, x ascending, cleaned row last. The exact
 *    comparisons on f32 values make the order total, and it is the order of a sweep line that is
 *    tilted by an infinitesimal angle, so vertices with equal y need no special case.
 * 2. Orient every ring (shell counter-clockwise, holes clockwise) from the exact orientation at
 *    its topmost vertex, then classify every vertex as start, end, split, merge or regular (left
 *    or right boundary) from the ranks of its two neighbors and the exact orientation.
 * 3. A split, merge or right-boundary vertex shoots a ray to the left through the shared polygon
 *    edge-slab index. Among the down edges that straddle its rank and lie strictly left of it, the
 *    closest is its "left edge" `e`.
 * 4. Helper diagonals. The sequential sweep keeps `helper(e)`, the most recent vertex that
 *    touched the left edge `e` in the status structure. Every touch is a member of the group of
 *    `e`: the upper end of `e` (insertion), every vertex whose left edge is `e`, and the lower end
 *    of `e` (deletion), and every touch sets `helper(e)` to the touching vertex, except the
 *    deletion, which only reads it. Hence `helper(e)` at the time of a vertex `b` is the group
 *    member that precedes `b` in sweep order. The sweep adds a diagonal `(b, a)` for the
 *    predecessor `a` of `b` exactly when `b` is a split vertex (always), or when `a` is a merge
 *    vertex (and `b` is a merge, regular or end vertex that touches `e`). Sorting all members by
 *    `(edge, rank)` and testing consecutive pairs therefore reproduces the sequential algorithm
 *    without a status structure. The members are generated in rank order and sorted stably by
 *    edge, so the rank order inside a group comes for free. No diagonal is generated twice: a
 *    vertex has at most one predecessor-carrying role per incident group, and a merge or split
 *    vertex has no second role in the group of its predecessor. A vertex has at most four
 *    diagonals (at most two as the later and two as the earlier end), which is why four slots per
 *    vertex suffice and why the angular order at a vertex is a sort of at most four items.
 * 5. Half-edges: the ring edge of vertex `v` is half-edge `v`, diagonal slot `s` of `v` is
 *    half-edge `CAPACITY + 4 v + s`. The outgoing half-edges of a vertex, counter-clockwise from
 *    the ring edge, are `[ring, d_1 .. d_k]` followed by the direction back to the previous ring
 *    vertex; `next(h)` is the half-edge before `reverse(h)` in that list. The cycles of `next` are
 *    the monotone pieces. They are labeled by double-buffered pointer jumping (minimum half-edge
 *    id of the cycle, `ceil(log2(3 n + 3))` rounds in GPU-gated pairs), and the cycle's minimum
 *    half-edge is its representative.
 * 6. One invocation per piece walks the cycle (`O(piece)`), merges the left and right chains by
 *    rank and runs the stack triangulation of de Berg section 3.3. Triangles go to the polygon's
 *    index slice at an offset from a scan of `(piece vertices - 2)`.
 *
 * Validity: the counts `clean + 2 holes - 2` must equal the sum of `(piece vertices - 2)`, every
 * piece must be strictly monotone in rank along both chains, every walk must close, every
 * diagonal must be symmetric and lie strictly inside the angular wedge of both of its ends, and
 * every predicate must be certified. Not detected: self-intersections or overlapping holes that
 * keep all of these consistent. Not supported (flagged invalid): a diagonal between two coincident
 * vertices (rings that touch at a vertex).
 *
 * @internal
 */

/** Words of the GPU state array `counts`. */
export const MONOTONE_COUNT_WORDS = 8;
/** `counts` words. */
export const MONOTONE_COUNT = {
  /** Vertices of all routed polygons (raw rows), including overflowed polygons. */
  routed: 0,
  /** Cleaned vertices (including those of dead rings). */
  clean: 1,
  /** Monotone pieces. */
  pieces: 2,
  /** Pairs of pointer-jumping rounds still to run. */
  jumpPairs: 3,
  /** Largest cleaned vertex count of one polygon. */
  maximumClean: 4,
  /** Vertices that the current pair of jump rounds covers (0 disarms the pair). */
  jumpActive: 5
} as const;

/** Words per polygon of the statistics array. */
export const MONOTONE_STATS_STRIDE = 8;
/** Failure bits of `polygonStats[8 p]`. */
export const MONOTONE_FAILURE = {
  nonFinite: 1,
  shell: 2,
  orientation: 4,
  ray: 8,
  diagonal: 16,
  piece: 32,
  stack: 64
} as const;

/** Vertex classes stored in the low byte of the type word. */
export const MONOTONE_TYPE = {
  start: 0,
  end: 1,
  split: 2,
  merge: 3,
  regularLeft: 4,
  regularRight: 5
} as const;

/** Words per piece in the piece table. */
export const MONOTONE_PIECE_STRIDE = 8;

/** Compile-time sizes baked into the kernels. */
export type MonotoneSizes = {
  /** Rows of `positions`. */
  vertexCount: number;
  polygonCount: number;
  ringCount: number;
  /** Cleaned-vertex capacity of the monotone path. */
  capacity: number;
  /** Length of `output.indices`. */
  indexLength: number;
  /** Polygons with at least this many vertices skip Earcut. */
  minimumVertices: number;
};

/** Constants shared by every kernel of the monotone path. */
export function getMonotoneConstantsWGSL(sizes: MonotoneSizes): string {
  return /* wgsl */ `
const NONE: u32 = 0xffffffffu;
const OVERFLOW_BIT: u32 = 0x80000000u;
const VERTEX_COUNT: u32 = ${sizes.vertexCount}u;
const POLYGON_COUNT: u32 = ${sizes.polygonCount}u;
const RING_COUNT: u32 = ${sizes.ringCount}u;
const CAPACITY: u32 = ${sizes.capacity}u;
const INDEX_LENGTH: u32 = ${sizes.indexLength}u;
const MINIMUM_VERTICES: u32 = ${sizes.minimumVertices}u;
const COUNT_ROUTED: u32 = ${MONOTONE_COUNT.routed}u;
const COUNT_CLEAN: u32 = ${MONOTONE_COUNT.clean}u;
const COUNT_PIECES: u32 = ${MONOTONE_COUNT.pieces}u;
const COUNT_JUMP_PAIRS: u32 = ${MONOTONE_COUNT.jumpPairs}u;
const COUNT_MAXIMUM_CLEAN: u32 = ${MONOTONE_COUNT.maximumClean}u;
const COUNT_JUMP_ACTIVE: u32 = ${MONOTONE_COUNT.jumpActive}u;
const COUNT_WORDS: u32 = ${MONOTONE_COUNT_WORDS}u;
const STATS_STRIDE: u32 = ${MONOTONE_STATS_STRIDE}u;
const STATS_FAIL: u32 = 0u;
const STATS_CLEAN: u32 = 1u;
const STATS_HOLES: u32 = 2u;
const STATS_TRIANGLES: u32 = 3u;
const STATS_WRITTEN: u32 = 4u;
const FAIL_NON_FINITE: u32 = ${MONOTONE_FAILURE.nonFinite}u;
const FAIL_SHELL: u32 = ${MONOTONE_FAILURE.shell}u;
const FAIL_ORIENTATION: u32 = ${MONOTONE_FAILURE.orientation}u;
const FAIL_RAY: u32 = ${MONOTONE_FAILURE.ray}u;
const FAIL_DIAGONAL: u32 = ${MONOTONE_FAILURE.diagonal}u;
const FAIL_PIECE: u32 = ${MONOTONE_FAILURE.piece}u;
const FAIL_STACK: u32 = ${MONOTONE_FAILURE.stack}u;
const TYPE_START: u32 = ${MONOTONE_TYPE.start}u;
const TYPE_END: u32 = ${MONOTONE_TYPE.end}u;
const TYPE_SPLIT: u32 = ${MONOTONE_TYPE.split}u;
const TYPE_MERGE: u32 = ${MONOTONE_TYPE.merge}u;
const TYPE_REGULAR_LEFT: u32 = ${MONOTONE_TYPE.regularLeft}u;
const TYPE_REGULAR_RIGHT: u32 = ${MONOTONE_TYPE.regularRight}u;
const A_POLYGON: u32 = 0u;
const A_RING: u32 = 1u;
const A_ROW: u32 = 2u;
const A_TYPE: u32 = 3u;
const B_RANK: u32 = 0u;
const B_NEXT: u32 = 1u;
const B_PREVIOUS: u32 = 2u;
const B_LEFT: u32 = 3u;
const PIECE_STRIDE: u32 = ${MONOTONE_PIECE_STRIDE}u;
`;
}

/** Accessors of the compact vertex record `vertexA` (polygon, ring, row, type word). */
export const MONOTONE_VERTEX_A_WGSL = /* wgsl */ `
fn polygonOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_POLYGON]; }
fn ringOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_RING]; }
fn rowOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_ROW]; }
fn typeOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_TYPE] & 255u; }
fn isFlipped(vertex: u32) -> bool { return ((vertexA[vertexAOffset + 4u * vertex + A_TYPE] >> 8u) & 1u) == 1u; }
`;

/** Accessors of the compact vertex record `vertexB` (rank, next, previous, left edge). */
export const MONOTONE_VERTEX_B_WGSL = /* wgsl */ `
fn rankOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_RANK]; }
fn nextOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_NEXT]; }
fn previousOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_PREVIOUS]; }
`;

/** `positionOf(vertex)` over the compact positions (`vertexPosition`, two floats per vertex). */
export const MONOTONE_POSITION_WGSL = /* wgsl */ `
fn positionOf(vertex: u32) -> vec2f {
  return vec2f(vertexPosition[vertexPositionOffset + 2u * vertex], vertexPosition[vertexPositionOffset + 2u * vertex + 1u]);
}
`;

/**
 * `orientOf(a, b, c)`: the exact orientation sign (positive when `c` is left of `a -> b`), with an
 * uncertain predicate counted as zero and recorded in `orientUncertain`. Every orientation of the
 * monotone path goes through this one wrapper, so the large exact helper has a single call site.
 */
export const MONOTONE_ORIENT_WGSL = /* wgsl */ `
var<private> orientUncertain: bool;
fn orientOf(a: vec2f, b: vec2f, c: vec2f) -> i32 {
  let result = orientSign(a, b, c);
  if (result == 2) {
    orientUncertain = true;
    return 0;
  }
  return result;
}
`;

/** Atomic failure flag of a polygon. Needs `polygonStats` bound as `array<atomic<u32>>`. */
export const MONOTONE_FAIL_WGSL = /* wgsl */ `
fn failPolygon(failedPolygon: u32, failureBit: u32) {
  atomicOr(&polygonStats[polygonStatsOffset + STATS_STRIDE * failedPolygon + STATS_FAIL], failureBit);
}
`;

/** `tailOf(halfEdge)`: the vertex a half-edge starts at. */
export const MONOTONE_TAIL_WGSL = /* wgsl */ `
fn tailOf(halfEdge: u32) -> u32 {
  if (halfEdge < CAPACITY) { return halfEdge; }
  return (halfEdge - CAPACITY) / 4u;
}
`;

/** Order-preserving u32 key of an f32 in which -0 and +0 are equal. */
export const MONOTONE_ORDERED_KEY_WGSL = /* wgsl */ `
fn orderedKey(x: f32) -> u32 {
  var bits = bitcast<u32>(x);
  if (x == 0.0) { bits = 0u; }
  return bits ^ select(0x80000000u, 0xffffffffu, (bits & 0x80000000u) != 0u);
}
`;

/** Slice arithmetic shared with the Earcut kernel: first index slot of polygon `p`. */
export const MONOTONE_SLICE_WGSL = /* wgsl */ `
fn baseSlotOf(polygon: u32) -> i32 {
  let ringPrefix = i32(polygonOffsets[polygonOffsetsOffset + polygon]);
  let vertexPrefix = i32(ringOffsets[ringOffsetsOffset + u32(ringPrefix)]);
  return 3 * (vertexPrefix + 2 * ringPrefix - 4 * i32(polygon));
}
`;

/** Polygon vertex range `[first, end)` of the raw positions. */
export const MONOTONE_POLYGON_ROWS_WGSL = /* wgsl */ `
fn polygonFirstRow(polygon: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + polygonOffsets[polygonOffsetsOffset + polygon]], VERTEX_COUNT);
}
fn polygonEndRow(polygon: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + polygonOffsets[polygonOffsetsOffset + polygon + 1u]], VERTEX_COUNT);
}
`;

// ---------------------------------------------------------------------------------------------
// Kernel bodies and declarations. Each entry is `{declarations, body}` of one kernel; the
// bindings are listed next to the node in polygon-triangulation-monotone.ts.
// ---------------------------------------------------------------------------------------------

/** One WGSL kernel: module-scope declarations and the per-invocation body. */
export type MonotoneKernelSource = {declarations: string; body: string};

/**
 * R0, the size routing, is part of the Earcut kernel (the first pass): `sizeRouteOf(polygon)` is 1
 * for polygons with at least `minimumVertices` vertices over all rings, which skip Earcut and take
 * the monotone path, and 0 otherwise. The declarations need `polygonOffsets`, `ringOffsets` and a
 * `VERTEX_COUNT` constant in the module.
 */
export function getSizeRouteWGSL(minimumVertices: number): string {
  return /* wgsl */ `
const MONOTONE_MINIMUM_VERTICES: u32 = ${minimumVertices}u;
fn sizeRouteOf(polygon: u32) -> u32 {
  let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
  let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  if (endRing <= firstRing) { return 0u; }
  let first = min(ringOffsets[ringOffsetsOffset + firstRing], VERTEX_COUNT);
  let end = min(ringOffsets[ringOffsetsOffset + endRing], VERTEX_COUNT);
  return select(0u, 1u, end >= first + MONOTONE_MINIMUM_VERTICES);
}
`;
}

/** R1: routed vertex count per polygon; zero the statistics and the state words. */
export function getRouteCountsSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_POLYGON_ROWS_WGSL}`,
    body: `var count = 0u;
  if (index < POLYGON_COUNT) {
    if (routing[routingOffset + index] != 0u) {
      count = polygonEndRow(index) - min(polygonFirstRow(index), polygonEndRow(index));
    }
    for (var word = 0u; word < STATS_STRIDE; word++) {
      polygonStats[polygonStatsOffset + STATS_STRIDE * index + word] = 0u;
    }
  }
  routedCounts[routedCountsOffset + index] = count;
  if (index == 0u) {
    for (var word = 0u; word < COUNT_WORDS; word++) { counts[countsOffset + word] = 0u; }
  }`
  };
}

/** R2: polygons that do not fit the capacity overflow (route 3); publish the routed total. */
export function getRouteOverflowSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let count = routedCounts[routedCountsOffset + index];
  if (count > 0u && routedOffsets[routedOffsetsOffset + index] + count > CAPACITY) {
    routing[routingOffset + index] = 3u;
    routedCounts[routedCountsOffset + index] = 0u;
  }
  if (index == 0u) {
    counts[countsOffset + COUNT_ROUTED] = routedOffsets[routedOffsetsOffset + POLYGON_COUNT];
  }`
  };
}

/**
 * K1: one invocation per routed vertex `j`. Finds its polygon and ring by binary search (the
 * routed vertices of the polygons are concatenated, so no per-vertex polygon table is needed),
 * flags non-finite coordinates and sets the keep flag: a vertex is kept when it differs from its
 * cyclic predecessor in the ring, which drops consecutive duplicates and the repeated closing
 * vertex (the first of a run of equal vertices is kept).
 */
export function getIndexSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_FAIL_WGSL}
fn vertexAt(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn isFiniteBits(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }`,
    body: `let j = index;
  keepFlags[keepFlagsOffset + j] = 0u;
  // The largest polygon whose routed range starts at or before j.
  var low = 0u;
  var high = POLYGON_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (routedOffsets[routedOffsetsOffset + middle] <= j) { low = middle; } else { high = middle; }
  }
  let polygon = low;
  let polygonBegin = routedOffsets[routedOffsetsOffset + polygon];
  let polygonEnd = routedOffsets[routedOffsetsOffset + polygon + 1u];
  let info = jInfoOffset + 4u * j;
  jInfo[info + 1u] = NONE;
  if (j >= polygonEnd || routing[routingOffset + polygon] == 0u) { return; }
  let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
  let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  let row = ringOffsets[ringOffsetsOffset + firstRing] + (j - polygonBegin);
  if (row >= VERTEX_COUNT) { return; }
  // The ring containing the row: the largest ring of the polygon that starts at or before it.
  var ringLow = firstRing;
  var ringHigh = endRing;
  while (ringLow + 1u < ringHigh) {
    let middle = (ringLow + ringHigh) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= row) { ringLow = middle; } else { ringHigh = middle; }
  }
  let ring = ringLow;
  jInfo[info] = row;
  jInfo[info + 2u] = ring;
  jInfo[info + 3u] = firstRing;
  if (routing[routingOffset + polygon] == 3u) {
    jInfo[info + 1u] = polygon | OVERFLOW_BIT;
    return;
  }
  jInfo[info + 1u] = polygon;
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
  let point = vertexAt(row);
  if (!isFiniteBits(point.x) || !isFiniteBits(point.y)) {
    failPolygon(polygon, FAIL_NON_FINITE);
    return;
  }
  let previousRow = select(row - 1u, ringEnd - 1u, row == ringBegin);
  let previous = vertexAt(previousRow);
  if (point.x != previous.x || point.y != previous.y) { keepFlags[keepFlagsOffset + j] = 1u; }`
  };
}

/**
 * K2: one invocation per routed vertex; the first vertex of each ring records the ring's cleaned
 * range, decides whether it is alive (3 or more cleaned vertices and a live shell), and adds to the
 * polygon statistics. A hole with fewer than 3 cleaned vertices is ignored, as in Earcut.
 */
export function getRingSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_FAIL_WGSL}`,
    body: `let info = jInfoOffset + 4u * index;
  let polygon = jInfo[info + 1u];
  if (polygon >= OVERFLOW_BIT) { return; }
  let ring = jInfo[info + 2u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  if (jInfo[info] != ringBegin) { return; }
  let ringEnd = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
  let begin = keepOffsets[keepOffsetsOffset + index];
  let count = keepOffsets[keepOffsetsOffset + index + (ringEnd - ringBegin)] - begin;
  let shellRing = jInfo[info + 3u];
  let shellBegin = ringOffsets[ringOffsetsOffset + shellRing];
  let shellEnd = min(ringOffsets[ringOffsetsOffset + shellRing + 1u], VERTEX_COUNT);
  let polygonBegin = routedOffsets[routedOffsetsOffset + polygon];
  let shellCount = keepOffsets[keepOffsetsOffset + polygonBegin + (shellEnd - shellBegin)] -
    keepOffsets[keepOffsetsOffset + polygonBegin];
  let alive = count >= 3u && shellCount >= 3u;
  ringInfo[ringInfoOffset + 4u * ring] = begin;
  ringInfo[ringInfoOffset + 4u * ring + 1u] = count;
  ringInfo[ringInfoOffset + 4u * ring + 2u] = select(0u, 1u, alive);
  ringInfo[ringInfoOffset + 4u * ring + 3u] = 0u;
  atomicStore(&ringTop[ringTopOffset + ring], NONE);
  let statsBase = polygonStatsOffset + STATS_STRIDE * polygon;
  if (alive) {
    let before = atomicAdd(&polygonStats[statsBase + STATS_CLEAN], count);
    atomicMax(&counts[countsOffset + COUNT_MAXIMUM_CLEAN], before + count);
    if (ring != shellRing) { atomicAdd(&polygonStats[statsBase + STATS_HOLES], 1u); }
  }
  if (ring == shellRing && shellCount < 3u) { failPolygon(polygon, FAIL_SHELL); }`
  };
}

/** K3: cleaned vertex total and the number of jump-round pairs, from the largest polygon. */
export function getCountsSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let routed = min(counts[countsOffset + COUNT_ROUTED], VERTEX_COUNT);
  counts[countsOffset + COUNT_CLEAN] = min(keepOffsets[keepOffsetsOffset + routed], CAPACITY);
  // A cycle has at most 3 n half-edges (n ring edges and up to 2 n diagonal half-edges); a
  // pointer window of 2^r half-edges closes it once 2^r reaches that bound.
  let bound = 3u * counts[countsOffset + COUNT_MAXIMUM_CLEAN] + 3u;
  var rounds = 0u;
  var reach = 1u;
  while (reach < bound) {
    reach = reach * 2u;
    rounds += 1u;
  }
  counts[countsOffset + COUNT_JUMP_PAIRS] = (rounds + 1u) / 2u;`
  };
}

/** K4: scatter the kept vertices to their compact index. */
export function getCompactSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let info = jInfoOffset + 4u * index;
  let polygon = jInfo[info + 1u];
  if (polygon >= OVERFLOW_BIT || keepFlags[keepFlagsOffset + index] == 0u) { return; }
  let row = jInfo[info];
  let ring = jInfo[info + 2u];
  let vertex = keepOffsets[keepOffsetsOffset + index];
  if (vertex >= CAPACITY) { return; }
  let alive = (ringInfo[ringInfoOffset + 4u * ring + 2u] & 1u) != 0u;
  let record = vertexAOffset + 4u * vertex;
  vertexA[record + A_POLYGON] = select(NONE, polygon, alive);
  vertexA[record + A_RING] = ring;
  vertexA[record + A_ROW] = row;
  vertexA[record + A_TYPE] = 0u;
  vertexPosition[vertexPositionOffset + 2u * vertex] = positions[positionsOffset + 2u * row];
  vertexPosition[vertexPositionOffset + 2u * vertex + 1u] = positions[positionsOffset + 2u * row + 1u];
  rowToClean[rowToCleanOffset + row] = vertex;`
  };
}

/**
 * K5: sort keys of every cleaned vertex (y descending, x ascending), the identity permutation the
 * stable sort permutes, and the neutralisation of the rows beyond the cleaned count: a stale
 * record there would otherwise look alive to the kernels whose dispatch is rounded up to a whole
 * workgroup.
 */
export function getKeysSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_ORDERED_KEY_WGSL}`,
    body: `let vertex = index;
  var high = NONE;
  var low = NONE;
  if (vertex < counts[countsOffset + COUNT_CLEAN]) {
    if (vertexA[vertexAOffset + 4u * vertex + A_POLYGON] != NONE) {
      let x = vertexPosition[vertexPositionOffset + 2u * vertex];
      let y = vertexPosition[vertexPositionOffset + 2u * vertex + 1u];
      high = min(~orderedKey(y), 0xfffffffeu);
      low = min(orderedKey(x), 0xfffffffeu);
    }
  } else {
    vertexA[vertexAOffset + 4u * vertex + A_POLYGON] = NONE;
  }
  sortKeyHigh[sortKeyHighOffset + vertex] = high;
  sortKeyLow[sortKeyLowOffset + vertex] = low;
  identity[identityOffset + vertex] = vertex;`
  };
}

/** K6: rank of every vertex (its position in the sorted order) and the top rank of each ring. */
export function getRankSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}`,
    body: `let vertex = sortedIndices[sortedIndicesOffset + index];
  if (vertex >= CAPACITY) { return; }
  if (polygonOf(vertex) == NONE) { return; }
  vertexB[vertexBOffset + 4u * vertex + B_RANK] = index;
  atomicMin(&ringTop[ringTopOffset + ringOf(vertex)], index);`
  };
}

/**
 * K7: the topmost vertex of each ring is convex, so its exact orientation is the ring's winding.
 * The logical order has the interior on the left: shell counter-clockwise, holes clockwise.
 */
export function getFlipSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_VERTEX_B_WGSL}${MONOTONE_POSITION_WGSL}${MONOTONE_FAIL_WGSL}${MONOTONE_ORIENT_WGSL}`,
    body: `let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let ring = ringOf(vertex);
  if (rankOf(vertex) != atomicLoad(&ringTop[ringTopOffset + ring])) { return; }
  let info = ringInfoOffset + 4u * ring;
  let begin = ringInfo[info];
  let count = ringInfo[info + 1u];
  let local = vertex - begin;
  let nextVertex = begin + (local + 1u) % count;
  let previousVertex = begin + (local + count - 1u) % count;
  let orientation = orientOf(positionOf(previousVertex), positionOf(vertex), positionOf(nextVertex));
  var flip = false;
  if (orientation == 0 || orientUncertain) {
    failPolygon(polygon, FAIL_ORIENTATION);
  } else if (ring == polygonOffsets[polygonOffsetsOffset + polygon]) {
    flip = orientation < 0;
  } else {
    flip = orientation > 0;
  }
  if (flip) { ringInfo[info + 2u] = ringInfo[info + 2u] | 2u; }`
  };
}

/**
 * K8: logical neighbors, vertex class, and the initial (empty) diagonal slots and self
 * next-pointers of the vertex.
 */
export function getLinkSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_VERTEX_B_WGSL}${MONOTONE_POSITION_WGSL}${MONOTONE_FAIL_WGSL}${MONOTONE_ORIENT_WGSL}`,
    body: `let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let info = ringInfoOffset + 4u * ringOf(vertex);
  let begin = ringInfo[info];
  let count = ringInfo[info + 1u];
  let flip = (ringInfo[info + 2u] & 2u) != 0u;
  let local = vertex - begin;
  let rawNext = begin + (local + 1u) % count;
  let rawPrevious = begin + (local + count - 1u) % count;
  let nextVertex = select(rawNext, rawPrevious, flip);
  let previousVertex = select(rawPrevious, rawNext, flip);
  let record = vertexBOffset + 4u * vertex;
  vertexB[record + B_NEXT] = nextVertex;
  vertexB[record + B_PREVIOUS] = previousVertex;
  vertexB[record + B_LEFT] = NONE;
  let rank = rankOf(vertex);
  let nextBelow = rankOf(nextVertex) > rank;
  let previousBelow = rankOf(previousVertex) > rank;
  var kind = TYPE_REGULAR_LEFT;
  if (previousBelow == nextBelow) {
    let orientation = orientOf(positionOf(previousVertex), positionOf(vertex), positionOf(nextVertex));
    if (orientation == 0 || orientUncertain) {
      failPolygon(polygon, FAIL_ORIENTATION);
    } else if (previousBelow) {
      kind = select(TYPE_SPLIT, TYPE_START, orientation > 0);
    } else {
      kind = select(TYPE_MERGE, TYPE_END, orientation > 0);
    }
  } else {
    kind = select(TYPE_REGULAR_LEFT, TYPE_REGULAR_RIGHT, previousBelow);
  }
  vertexA[vertexAOffset + 4u * vertex + A_TYPE] = kind | select(0u, 256u, flip);
  nextHalfEdge[nextHalfEdgeOffset + vertex] = vertex;
  for (var slot = 0u; slot < 4u; slot++) {
    vertexDiagonal[vertexDiagonalOffset + 4u * vertex + slot] = NONE;
    nextHalfEdge[nextHalfEdgeOffset + CAPACITY + 4u * vertex + slot] = CAPACITY + 4u * vertex + slot;
  }`
  };
}

/**
 * K9 declarations: the exact comparison of two candidate left edges. An edge is named by its upper
 * vertex (the logical start of a down edge). Both edges are active at the query rank and left of
 * the query point, so one end of the later-starting edge lies within the span of the other and its
 * side decides the order; touching edges fall back to the other end and then to the edge name.
 */
export const MONOTONE_LEFT_EDGE_WGSL = /* wgsl */ `
// Positive when edge \`second\` lies to the right of edge \`first\` at the common sweep position.
fn isRightOf(second: u32, first: u32) -> i32 {
  let upper1 = first;
  let lower1 = nextOf(first);
  let upper2 = second;
  let lower2 = nextOf(second);
  var a = upper1;
  var b = lower1;
  var c = upper2;
  var d = lower2;
  var multiplier = 1;
  if (upper2 == upper1) {
    c = lower2;
  } else if (rankOf(upper2) < rankOf(upper1)) {
    a = upper2;
    b = lower2;
    c = upper1;
    d = lower1;
    multiplier = -1;
  }
  var side = multiplier * orientOf(positionOf(a), positionOf(b), positionOf(c));
  if (side == 0 && upper2 != upper1) {
    side = multiplier * orientOf(positionOf(a), positionOf(b), positionOf(d));
  }
  if (side != 0) { return side; }
  // Touching edges (invalid input): a deterministic order by name.
  return select(-1, 1, second > first);
}
`;

/** K9 body: the edge directly left of a split, merge or right-boundary vertex, via the edge slabs. */
export function getLeftEdgeBody(slabLoop: (body: string) => string): string {
  const candidate = `let cleanEnd = rowToClean[rowToCleanOffset + edgeEnd];
        if (cleanEnd >= CAPACITY) { continue; }
        if (polygonOf(cleanEnd) != polygon || rowOf(cleanEnd) != edgeEnd) { continue; }
        let flipped = isFlipped(cleanEnd);
        let rawPrevious = select(previousOf(cleanEnd), nextOf(cleanEnd), flipped);
        let upperVertex = select(rawPrevious, cleanEnd, flipped);
        let lowerVertex = select(cleanEnd, rawPrevious, flipped);
        let upperRank = rankOf(upperVertex);
        if (!(upperRank < queryRank && queryRank < rankOf(lowerVertex))) { continue; }
        if (orientOf(positionOf(upperVertex), positionOf(lowerVertex), point) <= 0) { continue; }
        var closer = best == NONE;
        if (!closer) { closer = isRightOf(upperVertex, best) > 0; }
        if (closer) { best = upperVertex; }`;
  return `let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let kind = typeOf(vertex);
  if (kind != TYPE_SPLIT && kind != TYPE_MERGE && kind != TYPE_REGULAR_RIGHT) { return; }
  let point = positionOf(vertex);
  let queryRank = rankOf(vertex);
  var best = NONE;
  ${slabLoop(candidate)}
  if (orientUncertain) { best = NONE; }
  vertexB[vertexBOffset + 4u * vertex + B_LEFT] = best;`;
}

/** K10: three member slots per vertex in rank order (left-edge touch, deletion, insertion). */
export function getMembersSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_VERTEX_B_WGSL}${MONOTONE_FAIL_WGSL}`,
    body: `let rank = index;
  var left = CAPACITY;
  var lower = CAPACITY;
  var upper = CAPACITY;
  if (rank < counts[countsOffset + COUNT_CLEAN]) {
    let vertex = sortedIndices[sortedIndicesOffset + rank];
    var usable = vertex < CAPACITY;
    if (usable) { usable = polygonOf(vertex) != NONE; }
    if (usable) {
      let kind = typeOf(vertex);
      if (kind == TYPE_SPLIT || kind == TYPE_MERGE || kind == TYPE_REGULAR_RIGHT) {
        let edge = vertexB[vertexBOffset + 4u * vertex + B_LEFT];
        if (edge == NONE) { failPolygon(polygonOf(vertex), FAIL_RAY); } else { left = edge; }
      }
      // The edge into the vertex is a down edge when its start is above; the edge out of the
      // vertex is a down edge when its end is below.
      if (rankOf(previousOf(vertex)) < rank) { lower = previousOf(vertex); }
      if (rankOf(nextOf(vertex)) > rank) { upper = vertex; }
    }
  }
  memberGroup[memberGroupOffset + 3u * rank] = left;
  memberGroup[memberGroupOffset + 3u * rank + 1u] = lower;
  memberGroup[memberGroupOffset + 3u * rank + 2u] = upper;
  memberIdentity[memberIdentityOffset + 3u * rank] = 3u * rank;
  memberIdentity[memberIdentityOffset + 3u * rank + 1u] = 3u * rank + 1u;
  memberIdentity[memberIdentityOffset + 3u * rank + 2u] = 3u * rank + 2u;`
  };
}

/** K11: the helper diagonals from consecutive members of a group (see the module comment). */
export function getDiagonalsSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_FAIL_WGSL}`,
    body: `let member = sortedMembers[sortedMembersOffset + index];
  let groupId = memberGroup[memberGroupOffset + member];
  if (groupId == CAPACITY || index == 0u) { return; }
  let previousMember = sortedMembers[sortedMembersOffset + index - 1u];
  if (memberGroup[memberGroupOffset + previousMember] != groupId) { return; }
  let later = sortedIndices[sortedIndicesOffset + member / 3u];
  let earlier = sortedIndices[sortedIndicesOffset + previousMember / 3u];
  if (later >= CAPACITY || earlier >= CAPACITY) { return; }
  // A diagonal (later, earlier) is added when the later vertex is a split vertex or the earlier
  // one (the helper) is a merge vertex.
  if (!(typeOf(later) == TYPE_SPLIT || typeOf(earlier) == TYPE_MERGE)) { return; }
  let laterRole = member % 3u;
  let earlierRole = previousMember % 3u;
  // The insertion (role 2) is always first in its group and the deletion (role 1) always last.
  if (earlierRole == 1u || laterRole == 2u) {
    failPolygon(polygonOf(later), FAIL_DIAGONAL);
    return;
  }
  let laterSlot = select(1u, 0u, laterRole == 0u);
  let earlierSlot = select(3u, 2u, earlierRole == 0u);
  vertexDiagonal[vertexDiagonalOffset + 4u * later + laterSlot] = (earlier << 2u) | earlierSlot;
  vertexDiagonal[vertexDiagonalOffset + 4u * earlier + earlierSlot] = (later << 2u) | laterSlot;`
  };
}

/**
 * K12: the angular order of the diagonals at a vertex and the next-pointers of the half-edges that
 * end at it. Directions are classified counter-clockwise from the ring edge by the exact
 * orientation: category 0 is within (0, pi), 1 is exactly pi, 2 is within (pi, 2 pi), and -1 is
 * the direction of the ring edge itself (invalid); within a category the orientation orders.
 */
export function getNextSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_VERTEX_B_WGSL}${MONOTONE_POSITION_WGSL}${MONOTONE_FAIL_WGSL}${MONOTONE_ORIENT_WGSL}
var<private> sortCenter: vec2f;
var<private> sortBase: vec2f;
var<private> sortInvalid: bool;
fn sameDirection(center: vec2f, first: vec2f, second: vec2f) -> bool {
  let dx1 = select(select(-1i, 0i, first.x == center.x), 1i, first.x > center.x);
  let dy1 = select(select(-1i, 0i, first.y == center.y), 1i, first.y > center.y);
  let dx2 = select(select(-1i, 0i, second.x == center.x), 1i, second.x > center.x);
  let dy2 = select(select(-1i, 0i, second.y == center.y), 1i, second.y > center.y);
  return dx1 == dx2 && dy1 == dy2;
}
fn angleCategory(candidate: vec2f) -> i32 {
  let orientation = orientOf(sortCenter, sortBase, candidate);
  if (orientation > 0) { return 0; }
  if (orientation < 0) { return 2; }
  return select(1i, -1i, sameDirection(sortCenter, sortBase, candidate));
}
fn angleBefore(first: vec2f, second: vec2f) -> bool {
  let firstCategory = angleCategory(first);
  let secondCategory = angleCategory(second);
  if (firstCategory != secondCategory) { return firstCategory < secondCategory; }
  if (firstCategory == 1 || firstCategory == -1) {
    sortInvalid = true;
    return false;
  }
  return orientOf(sortCenter, first, second) > 0;
}`,
    body: `let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let center = positionOf(vertex);
  let nextVertex = nextOf(vertex);
  let previousVertex = previousOf(vertex);
  sortCenter = center;
  sortBase = positionOf(nextVertex);
  let backPoint = positionOf(previousVertex);
  var others = array<u32, 4>(0u, 0u, 0u, 0u);
  var slots = array<u32, 4>(0u, 0u, 0u, 0u);
  var count = 0u;
  var invalid = false;
  for (var slot = 0u; slot < 4u; slot++) {
    let value = vertexDiagonal[vertexDiagonalOffset + 4u * vertex + slot];
    if (value == NONE) { continue; }
    let other = value >> 2u;
    let reverseSlot = value & 3u;
    // The reverse slot must point back, and a diagonal joins two vertices of one polygon that
    // are not neighbors on the ring.
    if (other >= CAPACITY || other == vertex || other == nextVertex || other == previousVertex) {
      invalid = true;
      continue;
    }
    if (polygonOf(other) != polygon ||
        vertexDiagonal[vertexDiagonalOffset + 4u * other + reverseSlot] != ((vertex << 2u) | slot)) {
      invalid = true;
      continue;
    }
    others[count] = other;
    slots[count] = slot;
    count += 1u;
  }
  // Insertion sort of at most four diagonals by counter-clockwise angle from the ring edge.
  for (var item = 1u; item < count; item++) {
    let heldOther = others[item];
    let heldSlot = slots[item];
    let heldPoint = positionOf(heldOther);
    var hole = item;
    loop {
      if (hole == 0u) { break; }
      if (!angleBefore(heldPoint, positionOf(others[hole - 1u]))) { break; }
      others[hole] = others[hole - 1u];
      slots[hole] = slots[hole - 1u];
      hole -= 1u;
    }
    others[hole] = heldOther;
    slots[hole] = heldSlot;
  }
  // Every diagonal lies strictly inside the wedge from the ring edge to the back edge, and the
  // directions are pairwise distinct.
  for (var item = 0u; item < count; item++) {
    let point = positionOf(others[item]);
    if (point.x == center.x && point.y == center.y) { invalid = true; }
    if (angleCategory(point) == -1) { invalid = true; }
    if (!angleBefore(point, backPoint)) { invalid = true; }
    if (item > 0u) {
      if (!angleBefore(positionOf(others[item - 1u]), point)) { invalid = true; }
    }
  }
  if (invalid || sortInvalid || orientUncertain) {
    failPolygon(polygon, FAIL_DIAGONAL);
    return;
  }
  // The list is [ring edge, d_1 .. d_k]; the successor of a half-edge ending at this vertex is the
  // entry before its reverse in the list (the back edge being the end of the list).
  var lastHalfEdge = vertex;
  if (count > 0u) { lastHalfEdge = CAPACITY + 4u * vertex + slots[count - 1u]; }
  nextHalfEdge[nextHalfEdgeOffset + previousVertex] = lastHalfEdge;
  for (var item = 0u; item < count; item++) {
    let value = vertexDiagonal[vertexDiagonalOffset + 4u * vertex + slots[item]];
    let incoming = CAPACITY + 4u * (value >> 2u) + (value & 3u);
    var before = vertex;
    if (item > 0u) { before = CAPACITY + 4u * vertex + slots[item - 1u]; }
    nextHalfEdge[nextHalfEdgeOffset + incoming] = before;
  }`
  };
}

/** K13: initial pointer and label of the five half-edges of a vertex. */
export function getJumpInitSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}`,
    body: `let vertex = index;
  let alive = polygonOf(vertex) != NONE;
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    var valid = false;
    if (alive) {
      if (slot == 0u) {
        valid = true;
      } else {
        valid = vertexDiagonal[vertexDiagonalOffset + 4u * vertex + slot - 1u] != NONE;
      }
    }
    jumpA[jumpAOffset + 2u * halfEdge] = select(halfEdge, nextHalfEdge[nextHalfEdgeOffset + halfEdge], valid);
    jumpA[jumpAOffset + 2u * halfEdge + 1u] = select(NONE, halfEdge, valid);
  }`
  };
}

/** One round of pointer jumping from \`jumpSource\` into \`jumpTarget\` (one invocation per vertex). */
export function getJumpRoundSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let vertex = index;
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    let pointer = min(jumpSource[jumpSourceOffset + 2u * halfEdge], 5u * CAPACITY - 1u);
    let label = jumpSource[jumpSourceOffset + 2u * halfEdge + 1u];
    jumpTarget[jumpTargetOffset + 2u * halfEdge] = jumpSource[jumpSourceOffset + 2u * pointer];
    jumpTarget[jumpTargetOffset + 2u * halfEdge + 1u] = min(label, jumpSource[jumpSourceOffset + 2u * pointer + 1u]);
  }`
  };
}

/** Per pair of rounds: spend one pair and arm (or disarm) the jump dispatch. */
export function getJumpStateSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let remaining = counts[countsOffset + COUNT_JUMP_PAIRS];
  if (remaining > 0u) {
    counts[countsOffset + COUNT_JUMP_PAIRS] = remaining - 1u;
    counts[countsOffset + COUNT_JUMP_ACTIVE] = counts[countsOffset + COUNT_CLEAN];
  } else {
    counts[countsOffset + COUNT_JUMP_ACTIVE] = 0u;
  }`
  };
}

/** K14: number of cycle representatives (label equals half-edge id) among a vertex's half-edges. */
export function getRepresentativeCountSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let vertex = index;
  var count = 0u;
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    if (jumpA[jumpAOffset + 2u * halfEdge + 1u] == halfEdge) { count += 1u; }
  }
  repCounts[repCountsOffset + vertex] = count;`
  };
}

/** K15: write the representative half-edge of every piece at its piece index. */
export function getPiecesSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let vertex = index;
  var piece = repOffsets[repOffsetsOffset + vertex];
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    if (jumpA[jumpAOffset + 2u * halfEdge + 1u] == halfEdge) {
      if (piece < CAPACITY) { pieceTable[pieceTableOffset + PIECE_STRIDE * piece] = halfEdge; }
      piece += 1u;
    }
  }`
  };
}

/** The number of pieces. */
export function getPieceCountSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let clean = counts[countsOffset + COUNT_CLEAN];
  counts[countsOffset + COUNT_PIECES] = min(repOffsets[repOffsetsOffset + clean], CAPACITY);`
  };
}

/** K16: vertices of each piece (a bounded walk of its cycle) and the polygon triangle sums. */
export function getMeasureSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_FAIL_WGSL}${MONOTONE_TAIL_WGSL}`,
    body: `let piece = index;
  if (piece >= counts[countsOffset + COUNT_PIECES]) { return; }
  let table = pieceTableOffset + PIECE_STRIDE * piece;
  let head = pieceTable[table];
  let polygon = polygonOf(tailOf(head));
  var vertexCountOfPiece = 0u;
  if (polygon != NONE) {
    // A cycle holds at most 3 n half-edges; a longer walk is not a cycle.
    let limit = 3u * atomicLoad(&polygonStats[polygonStatsOffset + STATS_STRIDE * polygon + STATS_CLEAN]) + 3u;
    var id = head;
    var steps = 0u;
    loop {
      id = min(nextHalfEdge[nextHalfEdgeOffset + id], 5u * CAPACITY - 1u);
      steps += 1u;
      if (id == head || steps > limit) { break; }
    }
    if (id == head && steps >= 3u) {
      vertexCountOfPiece = steps;
      atomicAdd(&polygonStats[polygonStatsOffset + STATS_STRIDE * polygon + STATS_TRIANGLES], steps - 2u);
    } else {
      failPolygon(polygon, FAIL_PIECE);
    }
  }
  pieceTable[table + 1u] = vertexCountOfPiece;
  pieceTriangles[pieceTrianglesOffset + piece] = select(0u, vertexCountOfPiece - 2u, vertexCountOfPiece >= 3u);`
  };
}

/**
 * Per polygon: first index slot, triangle capacity (the slice formula of the Earcut kernel) and
 * the first piece's triangle offset, so a piece's triangles start at
 * \`slice + 3 * (offset - firstOffset)\`.
 */
export function getPolygonPlanSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_SLICE_WGSL}`,
    body: `let route = routing[routingOffset + index];
  let plan = polygonPlanOffset + 4u * index;
  polygonPlan[plan] = 0u;
  polygonPlan[plan + 1u] = 0u;
  polygonPlan[plan + 2u] = 0u;
  polygonPlan[plan + 3u] = 0u;
  if (route == 0u || route == 3u) { return; }
  let sliceStart = baseSlotOf(index);
  let sliceEnd = min(baseSlotOf(index + 1u), i32(INDEX_LENGTH));
  if (sliceStart < 0 || sliceEnd <= sliceStart) { return; }
  let compact = keepOffsets[keepOffsetsOffset + routedOffsets[routedOffsetsOffset + index]];
  let firstPiece = repOffsets[repOffsetsOffset + min(compact, CAPACITY)];
  polygonPlan[plan] = u32(sliceStart);
  polygonPlan[plan + 1u] = u32(sliceEnd - sliceStart) / 3u;
  polygonPlan[plan + 2u] = pieceTriangleOffsets[pieceTriangleOffsetsOffset + min(firstPiece, CAPACITY)];
  polygonPlan[plan + 3u] = 1u;`
  };
}

/** Per piece: absolute index base and limit, scratch base and polygon. */
export function getPiecePlanSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_FAIL_WGSL}${MONOTONE_TAIL_WGSL}`,
    body: `let piece = index;
  let table = pieceTableOffset + PIECE_STRIDE * piece;
  if (piece >= counts[countsOffset + COUNT_PIECES]) {
    pieceTable[table + 1u] = 0u;
    return;
  }
  let polygon = polygonOf(tailOf(pieceTable[table]));
  if (polygon == NONE) {
    pieceTable[table + 1u] = 0u;
    return;
  }
  let plan = polygonPlanOffset + 4u * polygon;
  let triangleOffset = pieceTriangleOffsets[pieceTriangleOffsetsOffset + piece];
  let firstOffset = polygonPlan[plan + 2u];
  if (polygonPlan[plan + 3u] == 0u || triangleOffset < firstOffset) {
    pieceTable[table + 1u] = 0u;
    failPolygon(polygon, FAIL_PIECE);
    return;
  }
  pieceTable[table + 2u] = polygonPlan[plan] + 3u * (triangleOffset - firstOffset);
  pieceTable[table + 3u] = polygonPlan[plan] + 3u * polygonPlan[plan + 1u];
  pieceTable[table + 4u] = triangleOffset + 2u * piece;
  pieceTable[table + 5u] = polygon;`
  };
}

/**
 * K17: the stack triangulation of one monotone piece (de Berg, section 3.3), one invocation per
 * piece with linear work. The cycle is copied into scratch starting at the top vertex; the left
 * chain is the prefix up to the bottom vertex and the right chain the reversed suffix. The chains
 * are merged by rank while checking that each is strictly monotone; the stack lives in scratch
 * too, so the invocation has no private array. Triangles are counter-clockwise.
 */
export function getTriangulateSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_VERTEX_B_WGSL}${MONOTONE_POSITION_WGSL}${MONOTONE_FAIL_WGSL}${MONOTONE_TAIL_WGSL}${MONOTONE_ORIENT_WGSL}
const SCRATCH_REGION: u32 = ${3 * sizes.capacity}u;
var<private> indexBase: u32;
var<private> indexLimit: u32;
var<private> triangleCursor: u32;
var<private> pieceFailed: bool;
var<private> sequenceBase: u32;
var<private> sequenceCount: u32;
var<private> sequenceTop: u32;
var<private> stackBase: u32;
var<private> stackSize: u32;

fn sequenceAt(position: u32) -> u32 {
  return pieceScratch[pieceScratchOffset + sequenceBase + (sequenceTop + position) % sequenceCount];
}
fn pushStack(pushedVertex: u32, pushedChain: u32) {
  if (stackSize >= sequenceCount) {
    pieceFailed = true;
    return;
  }
  pieceScratch[pieceScratchOffset + stackBase + stackSize] = pushedVertex * 2u + pushedChain;
  stackSize += 1u;
}
fn stackVertexAt(position: u32) -> u32 { return pieceScratch[pieceScratchOffset + stackBase + position] >> 1u; }
fn stackChainAt(position: u32) -> u32 { return pieceScratch[pieceScratchOffset + stackBase + position] & 1u; }
fn emitTriangle(cornerA: u32, cornerB: u32, cornerC: u32) {
  let triangleSlot = indexBase + 3u * triangleCursor;
  if (triangleSlot + 3u > indexLimit) {
    pieceFailed = true;
    return;
  }
  indices[indicesOffset + triangleSlot] = rowOf(cornerA);
  indices[indicesOffset + triangleSlot + 1u] = rowOf(cornerB);
  indices[indicesOffset + triangleSlot + 2u] = rowOf(cornerC);
  triangleCursor += 1u;
}
// Fan from \`vertex\` over the whole stack: counter-clockwise on a left-chain stack as
// (upper, lower, vertex) and on a right-chain stack as (lower, upper, vertex).
fn fanStack(fanVertex: u32, fanChain: u32) {
  for (var fanItem = 0u; fanItem + 1u < stackSize; fanItem++) {
    let fanUpper = stackVertexAt(fanItem);
    let fanLower = stackVertexAt(fanItem + 1u);
    if (fanChain == 0u) { emitTriangle(fanUpper, fanLower, fanVertex); } else { emitTriangle(fanLower, fanUpper, fanVertex); }
  }
}

fn triangulatePiece(piece: u32) {
  let table = pieceTableOffset + PIECE_STRIDE * piece;
  let pieceVertexCount = pieceTable[table + 1u];
  if (pieceVertexCount < 3u) { return; }
  let head = pieceTable[table];
  let polygon = pieceTable[table + 5u];
  indexBase = pieceTable[table + 2u];
  indexLimit = pieceTable[table + 3u];
  sequenceBase = pieceTable[table + 4u];
  sequenceCount = pieceVertexCount;
  stackBase = SCRATCH_REGION + sequenceBase;
  stackSize = 0u;
  triangleCursor = 0u;
  pieceFailed = false;
  if (sequenceBase + pieceVertexCount > SCRATCH_REGION) {
    failPolygon(polygon, FAIL_STACK);
    return;
  }
  // Copy the cycle and find its top and bottom vertices.
  var id = head;
  var topIndex = 0u;
  var bottomIndex = 0u;
  var topRank = 0u;
  var bottomRank = 0u;
  for (var position = 0u; position < pieceVertexCount; position++) {
    let vertex = tailOf(id);
    pieceScratch[pieceScratchOffset + sequenceBase + position] = vertex;
    let rank = rankOf(vertex);
    if (position == 0u || rank < topRank) {
      topRank = rank;
      topIndex = position;
    }
    if (position == 0u || rank > bottomRank) {
      bottomRank = rank;
      bottomIndex = position;
    }
    id = min(nextHalfEdge[nextHalfEdgeOffset + id], 5u * CAPACITY - 1u);
  }
  if (id != head) {
    failPolygon(polygon, FAIL_PIECE);
    return;
  }
  sequenceTop = topIndex;
  let bottom = (bottomIndex + pieceVertexCount - topIndex) % pieceVertexCount;
  // Merge the chains by rank. Positions 1..bottom are the left chain (the bottom included) and
  // pieceVertexCount - 1 down to bottom + 1 the right chain.
  var leftPosition = 1u;
  var rightPosition = pieceVertexCount - 1u;
  var leftRank = topRank;
  var rightRank = topRank;
  var processed = 1u;
  pushStack(sequenceAt(0u), 0u);
  loop {
    let leftAvailable = leftPosition <= bottom;
    let rightAvailable = rightPosition > bottom;
    if (!leftAvailable && !rightAvailable) { break; }
    var takeLeft = leftAvailable;
    var vertex = 0u;
    if (leftAvailable) { vertex = sequenceAt(leftPosition); }
    if (rightAvailable && (!leftAvailable || rankOf(sequenceAt(rightPosition)) < rankOf(vertex))) {
      takeLeft = false;
      vertex = sequenceAt(rightPosition);
    }
    let rank = rankOf(vertex);
    if (takeLeft) {
      if (rank <= leftRank) { pieceFailed = true; break; }
      leftRank = rank;
      leftPosition += 1u;
    } else {
      if (rank <= rightRank) { pieceFailed = true; break; }
      rightRank = rank;
      rightPosition -= 1u;
    }
    processed += 1u;
    let chain = select(1u, 0u, takeLeft);
    if (processed == 2u) {
      pushStack(vertex, chain);
      continue;
    }
    let topChain = stackChainAt(stackSize - 1u);
    if (leftPosition > bottom && rightPosition <= bottom) {
      // The bottom vertex: fan over everything left on the stack.
      fanStack(vertex, topChain);
      break;
    }
    if (chain != topChain) {
      // Opposite chains: fan over the whole stack, then keep its top and the new vertex.
      let previousTop = stackVertexAt(stackSize - 1u);
      fanStack(vertex, topChain);
      stackSize = 0u;
      pushStack(previousTop, topChain);
      pushStack(vertex, chain);
    } else {
      // Same chain: pop while the diagonal to the vertex stays inside the piece.
      var last = stackVertexAt(stackSize - 1u);
      stackSize -= 1u;
      loop {
        if (stackSize == 0u || pieceFailed) { break; }
        let upper = stackVertexAt(stackSize - 1u);
        var first = upper;
        var third = vertex;
        if (chain == 1u) {
          first = vertex;
          third = upper;
        }
        if (orientOf(positionOf(first), positionOf(last), positionOf(third)) < 0) { break; }
        if (chain == 0u) { emitTriangle(upper, last, vertex); } else { emitTriangle(vertex, last, upper); }
        last = upper;
        stackSize -= 1u;
      }
      pushStack(last, chain);
      pushStack(vertex, chain);
    }
    if (pieceFailed) { break; }
  }
  if (pieceFailed || orientUncertain || triangleCursor != pieceVertexCount - 2u) {
    failPolygon(polygon, FAIL_STACK);
  }
}
`,
    body: 'triangulatePiece(index);'
  };
}

/** K18: per-polygon validity and the triangle count the pad kernel keeps. */
export function getFinalSource(
  sizes: MonotoneSizes,
  writeTriangleCount: boolean
): MonotoneKernelSource {
  return {
    declarations: getMonotoneConstantsWGSL(sizes),
    body: `let route = routing[routingOffset + index];
  if (route == 0u) { return; }
  let statsBase = polygonStatsOffset + STATS_STRIDE * index;
  let clean = polygonStats[statsBase + STATS_CLEAN];
  let holes = polygonStats[statsBase + STATS_HOLES];
  let triangles = polygonStats[statsBase + STATS_TRIANGLES];
  var ok = route != 3u && polygonStats[statsBase + STATS_FAIL] == 0u && clean >= 3u;
  // The decomposition must account for every triangle of the polygon, no more and no less.
  if (ok) { ok = triangles == clean + 2u * holes - 2u; }
  // A size-routed or overflowed polygon that did not complete goes to the Earcut fallback pass.
  if (!ok && (route == 1u || route == 3u)) { routing[routingOffset + index] = 4u; }
  polygonStats[statsBase + STATS_WRITTEN] = select(0u, triangles, ok);
  valid[validOffset + index] = select(0u, 1u, ok);
  ${writeTriangleCount ? 'triangleCount[triangleCountOffset + index] = select(0u, triangles, ok);' : ''}`
  };
}

/**
 * K19: one invocation per routed vertex pads the unused triangle slots of its polygon with the
 * degenerate triangle (first, first, first) of the shell. Unit \`u\` of a polygon is slot \`3 u\`;
 * vertex \`u\` pads unit \`u\`, and the first vertex of ring \`k >= 2\` also pads the units
 * \`n + 2 (k - 2)\` and \`n + 2 (k - 2) + 1\`, which together cover \`n + 2 holes - 2\` units.
 */
export function getPadSource(sizes: MonotoneSizes): MonotoneKernelSource {
  return {
    declarations: `${getMonotoneConstantsWGSL(sizes)}${MONOTONE_SLICE_WGSL}${MONOTONE_POLYGON_ROWS_WGSL}
fn padUnit(padIndex: u32, padSliceStart: u32, padCapacity: u32, padWritten: u32, padVertex: u32) {
  if (padIndex >= padCapacity || padIndex < padWritten) { return; }
  let padSlot = indicesOffset + padSliceStart + 3u * padIndex;
  indices[padSlot] = padVertex;
  indices[padSlot + 1u] = padVertex;
  indices[padSlot + 2u] = padVertex;
}`,
    body: `let info = jInfoOffset + 4u * index;
  let rawPolygon = jInfo[info + 1u];
  if (rawPolygon == NONE) { return; }
  let polygon = rawPolygon & ~OVERFLOW_BIT;
  let sliceStart = baseSlotOf(polygon);
  let sliceEnd = min(baseSlotOf(polygon + 1u), i32(INDEX_LENGTH));
  if (sliceStart < 0 || sliceEnd <= sliceStart) { return; }
  let capacity = u32(sliceEnd - sliceStart) / 3u;
  let written = polygonStats[polygonStatsOffset + STATS_STRIDE * polygon + STATS_WRITTEN];
  let padRow = polygonFirstRow(polygon);
  let local = index - routedOffsets[routedOffsetsOffset + polygon];
  padUnit(local, u32(sliceStart), capacity, written, padRow);
  let ring = jInfo[info + 2u];
  let firstRing = jInfo[info + 3u];
  if (jInfo[info] == ringOffsets[ringOffsetsOffset + ring] && ring >= firstRing + 2u) {
    let polygonVertexCount = polygonEndRow(polygon) - padRow;
    let extra = polygonVertexCount + 2u * (ring - firstRing - 2u);
    padUnit(extra, u32(sliceStart), capacity, written, padRow);
    padUnit(extra + 1u, u32(sliceStart), capacity, written, padRow);
  }`
  };
}
