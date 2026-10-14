// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NearestSide} from './nearest-types';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {SPATIAL_GEODESIC_WGSL} from './spatial-geodesic-wgsl';

/** Slots of the per-query header in the traversal result buffer: count, flags, bound bits. @internal */
export const NEAREST_HEADER_WORDS = 3;

/** u32 words per BVH node in the packed node table: min.xy, max.xy (f32 bits), feature row. @internal */
export const NEAREST_NODE_WORDS = 5;

/**
 * Constants and finite checks shared by every nearest kernel, including the bounds kernels that do
 * not evaluate geometry.
 *
 * @internal
 */
export const NEAREST_BASE_WGSL = /* wgsl */ `
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const NO_FEATURE: u32 = 0xffffffffu;
const NO_SEGMENT: u32 = 0xffffffffu;
${IS_FINITE_VALUE_WGSL}
`;

const NEAREST_ORIENTATION_WGSL = /* wgsl */ `${NEAREST_BASE_WGSL}
${EXACT_ORIENTATION_WGSL}
// Exact sign (-1, 0, 1) of cross(b - a, c - a).
fn orientSide(a: vec2f, b: vec2f, c: vec2f) -> i32 {
  if (a.x == b.x && a.y == b.y) { return 0; }
  let sign = orientSign(a, b, c);
  if (sign != 2) { return sign; }
  let determinant = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  return select(select(0, -1, determinant < 0.0), 1, determinant > 0.0);
}
fn inSegmentBox(a: vec2f, b: vec2f, q: vec2f) -> bool {
  return q.x >= min(a.x, b.x) && q.x <= max(a.x, b.x) &&
    q.y >= min(a.y, b.y) && q.y <= max(a.y, b.y);
}
// Each orientation is evaluated once; a zero sign plus the range test is the on-segment test.
fn segmentsIntersect(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  let o1 = orientSide(a, b, c);
  let o2 = orientSide(a, b, d);
  let o3 = orientSide(c, d, a);
  let o4 = orientSide(c, d, b);
  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
      ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) {
    return true;
  }
  return (o1 == 0 && inSegmentBox(a, b, c)) || (o2 == 0 && inSegmentBox(a, b, d)) ||
    (o3 == 0 && inSegmentBox(c, d, a)) || (o4 == 0 && inSegmentBox(c, d, b));
}
fn crossProduct(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }
`;

/**
 * Planar helpers shared by the nearest kernels: segment/segment distance, the foot point on the
 * second segment, and the topological tests (on-segment, segment intersection, containment).
 *
 * Distances and foot points are plain f32. Every orientation sign is exact: `orientSide` wraps the
 * shared `orientSign` (an f32 filter with an exact 256-bit integer fallback, see
 * `exact-orientation-wgsl.ts`), so "distance 0" agrees with the point-in-polygon join and the
 * predicate join for the same vertices, including points exactly on an edge. Only an orientation
 * with no exact sign (non-finite coordinate, product exponents spanning more than 200 bits) falls
 * back to the sign of the f32 determinant.
 *
 * @internal
 */
export const NEAREST_COMMON_WGSL = /* wgsl */ `${NEAREST_ORIENTATION_WGSL}
fn closestPointOnSegment(p: vec2f, a: vec2f, b: vec2f) -> vec2f {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (lengthSq == 0.0) { return a; }
  let t = clamp(dot(p - a, ab) / lengthSq, 0.0, 1.0);
  return a + ab * t;
}
fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let d = p - closestPointOnSegment(p, a, b);
  return dot(d, d);
}
fn segmentPairDistanceSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  if (a.x == b.x && a.y == b.y) { return pointSegmentDistanceSq(a, c, d); }
  if (c.x == d.x && c.y == d.y) { return pointSegmentDistanceSq(c, a, b); }
  if (segmentsIntersect(a, b, c, d)) { return 0.0; }
  return min(
    min(pointSegmentDistanceSq(a, c, d), pointSegmentDistanceSq(b, c, d)),
    min(pointSegmentDistanceSq(c, a, b), pointSegmentDistanceSq(d, a, b))
  );
}
// The point of segment c-d nearest to segment a-b (the crossing point when they cross).
fn segmentFoot(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec2f {
  if (c.x == d.x && c.y == d.y) { return c; }
  if (a.x == b.x && a.y == b.y) { return closestPointOnSegment(a, c, d); }
  let ab = b - a;
  let cd = d - c;
  let denominator = crossProduct(cd, ab);
  if (denominator != 0.0 && segmentsIntersect(a, b, c, d)) {
    return c + cd * clamp(crossProduct(a - c, ab) / denominator, 0.0, 1.0);
  }
  var best = closestPointOnSegment(a, c, d);
  var bestSq = dot(a - best, a - best);
  let second = closestPointOnSegment(b, c, d);
  let secondSq = dot(b - second, b - second);
  if (secondSq < bestSq) { best = second; bestSq = secondSq; }
  let thirdSq = pointSegmentDistanceSq(c, a, b);
  if (thirdSq < bestSq) { best = c; bestSq = thirdSq; }
  if (pointSegmentDistanceSq(d, a, b) < bestSq) { best = d; }
  return best;
}
`;

/**
 * Great-circle counterpart of the distance half of {@link NEAREST_COMMON_WGSL}, with the same
 * function names. Positions are `vec2f(longitude, latitude)` in degrees and every "distance
 * squared" is a central angle in radians (the traversal compares and bounds angles; multiply by the
 * sphere radius for meters).
 *
 * Edges are straight in longitude/latitude with the longitude difference wrapped at +-180 (the same
 * edge model as the containment and crossing tests and as the network snapping contributors).
 * Point-to-edge distances use `geodesicSegmentFraction` and `geodesicSegmentDistance` of
 * `GEODESIC_WGSL`. The crossing and containment tests stay exact and planar in degrees; before the
 * crossing test the second edge is moved by a multiple of 360 degrees to overlap the first, so an
 * edge across the antimeridian still crosses its neighbors.
 *
 * @internal
 */
export const NEAREST_GEODESIC_WGSL = /* wgsl */ `${NEAREST_ORIENTATION_WGSL}
${SPATIAL_GEODESIC_WGSL}
fn unwrapLongitude(x: f32, reference: f32) -> f32 {
  return x + 360.0 * round((reference - x) / 360.0);
}
fn wrapLongitude(x: f32) -> f32 {
  return x - 360.0 * round(x / 360.0);
}
struct AlignedEdges { b: vec2f, c: vec2f, d: vec2f, overlaps: bool }
// Edges a-b and c-d with b, c and d shifted by multiples of 360 degrees so that b is within 180 of
// a, c within 180 of a and d within 180 of c, and c-d on the copy that overlaps a-b in longitude.
// Coordinates that need no shift are returned unchanged.
fn alignEdges(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> AlignedEdges {
  let alignedB = vec2f(unwrapLongitude(b.x, a.x), b.y);
  var alignedC = vec2f(unwrapLongitude(c.x, a.x), c.y);
  var alignedD = vec2f(unwrapLongitude(d.x, alignedC.x), d.y);
  let low = min(a.x, alignedB.x);
  let high = max(a.x, alignedB.x);
  var overlaps = max(alignedC.x, alignedD.x) >= low && min(alignedC.x, alignedD.x) <= high;
  if (!overlaps) {
    let shift = select(360.0, -360.0, alignedC.x > a.x);
    alignedC.x = alignedC.x + shift;
    alignedD.x = alignedD.x + shift;
    overlaps = max(alignedC.x, alignedD.x) >= low && min(alignedC.x, alignedD.x) <= high;
  }
  return AlignedEdges(alignedB, alignedC, alignedD, overlaps);
}
fn edgesIntersect(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  let edges = alignEdges(a, b, c, d);
  return edges.overlaps && segmentsIntersect(a, edges.b, edges.c, edges.d);
}
fn closestPointOnSegment(p: vec2f, a: vec2f, b: vec2f) -> vec2f {
  let fraction = geodesicSegmentFraction(p, a, b);
  if (fraction <= 0.0) { return a; }
  if (fraction >= 1.0) { return b; }
  let foot = a + vec2f(geodesicWrapLongitudeDelta(b.x - a.x), b.y - a.y) * fraction;
  return vec2f(wrapLongitude(foot.x), foot.y);
}
// Central angle from p to the segment a-b.
fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  return geodesicSegmentDistance(p, a, b, geodesicSegmentFraction(p, a, b), 1.0);
}
fn segmentPairDistanceSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  if (a.x == b.x && a.y == b.y) { return pointSegmentDistanceSq(a, c, d); }
  if (c.x == d.x && c.y == d.y) { return pointSegmentDistanceSq(c, a, b); }
  if (edgesIntersect(a, b, c, d)) { return 0.0; }
  return min(
    min(pointSegmentDistanceSq(a, c, d), pointSegmentDistanceSq(b, c, d)),
    min(pointSegmentDistanceSq(c, a, b), pointSegmentDistanceSq(d, a, b))
  );
}
// The point of segment c-d nearest to segment a-b (the crossing point when they cross).
fn segmentFoot(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec2f {
  if (c.x == d.x && c.y == d.y) { return c; }
  if (a.x == b.x && a.y == b.y) { return closestPointOnSegment(a, c, d); }
  let edges = alignEdges(a, b, c, d);
  if (edges.overlaps) {
    let ab = edges.b - a;
    let cd = edges.d - edges.c;
    let denominator = crossProduct(cd, ab);
    if (denominator != 0.0 && segmentsIntersect(a, edges.b, edges.c, edges.d)) {
      let crossing = edges.c + cd * clamp(crossProduct(a - edges.c, ab) / denominator, 0.0, 1.0);
      return vec2f(wrapLongitude(crossing.x), crossing.y);
    }
  }
  var best = closestPointOnSegment(a, c, d);
  var bestAngle = geodesicCentralAngle(a, best);
  let second = closestPointOnSegment(b, c, d);
  let secondAngle = geodesicCentralAngle(b, second);
  if (secondAngle < bestAngle) { best = second; bestAngle = secondAngle; }
  let thirdAngle = pointSegmentDistanceSq(c, a, b);
  if (thirdAngle < bestAngle) { best = c; bestAngle = thirdAngle; }
  if (pointSegmentDistanceSq(d, a, b) < bestAngle) { best = d; }
  return best;
}
`;

function getMinimumRingVertices(kind: NearestSide['kind']): number {
  return kind === 'points' ? 1 : kind === 'polygons' ? 3 : 2;
}

/**
 * Emits nested loops over every usable edge of `feature`, binding `a` and `b` to the endpoints.
 *
 * Points yield one degenerate edge, linestrings and segments yield open edges, and polygon rings
 * close implicitly. Rings that are too short and edges with a non-finite endpoint are skipped.
 * `body` may `return` or `continue`; `${prefix}vs + ${prefix}k` is the absolute edge-start vertex.
 */
function forEachEdge(side: NearestSide, feature: string, a: string, b: string, body: string) {
  const p = side.prefix;
  const edgeCount =
    side.kind === 'points' ? '1u' : side.kind === 'polygons' ? `${p}n` : `${p}n - 1u`;
  const next =
    side.kind === 'points'
      ? `${p}k`
      : side.kind === 'polygons'
        ? `(${p}k + 1u) % ${p}n`
        : `${p}k + 1u`;
  return `for (var ${p}r = ${p}RingStart(${feature}); ${p}r < ${p}RingEnd(${feature}); ${p}r++) {
    let ${p}vs = ${p}RingVertexStart(${p}r);
    let ${p}ve = ${p}RingVertexEnd(${p}r);
    if (${p}ve <= ${p}vs || ${p}ve - ${p}vs < ${getMinimumRingVertices(side.kind)}u) { continue; }
    let ${p}n = ${p}ve - ${p}vs;
    for (var ${p}k = 0u; ${p}k < ${edgeCount}; ${p}k++) {
      let ${a} = ${p}Vertex(${p}vs + ${p}k);
      let ${b} = ${p}Vertex(${p}vs + ${next});
      if (!(isFiniteValue(${a}.x) && isFiniteValue(${a}.y) && isFiniteValue(${b}.x) && isFiniteValue(${b}.y))) { continue; }
      ${body}
    }
  }`;
}

/**
 * Declares the accessors of one geometry side: ring and vertex ranges, vertices, the first usable
 * vertex, and (for polygons) an even/odd containment test.
 *
 * Bindings read by the accessors: points `${p}Positions`; segments `${p}Starts`, `${p}Ends`; lines
 * `${p}Positions`, `${p}LineOffsets`; polygons `${p}Positions`, `${p}FeatureRings`, `${p}RingOffsets`.
 * With `packedRingOffsetsBase`, polygon ring offsets share the `${p}FeatureRings` binding.
 *
 * @internal
 */
export function getNearestSideWGSL(side: NearestSide): string {
  const p = side.prefix;
  const P = p.toUpperCase();
  const {kind} = side;
  let ringStart = 'f';
  let ringEnd = 'f + 1u';
  let vertexStart = 'r';
  let vertexEnd = 'r + 1u';
  let vertex = `vec2f(${p}Positions[${p}PositionsOffset + i * 2u], ${p}Positions[${p}PositionsOffset + i * 2u + 1u])`;
  if (kind === 'segments') {
    vertexStart = '2u * r';
    vertexEnd = '2u * r + 2u';
    vertex = `select(
    vec2f(${p}Ends[${p}EndsOffset + (i >> 1u) * 2u], ${p}Ends[${p}EndsOffset + (i >> 1u) * 2u + 1u]),
    vec2f(${p}Starts[${p}StartsOffset + (i >> 1u) * 2u], ${p}Starts[${p}StartsOffset + (i >> 1u) * 2u + 1u]),
    (i & 1u) == 0u)`;
  } else if (kind === 'lines') {
    vertexStart = `${p}LineOffsets[${p}LineOffsetsOffset + r]`;
    vertexEnd = `min(${p}LineOffsets[${p}LineOffsetsOffset + r + 1u], ${P}_VERTEX_COUNT)`;
  } else if (kind === 'polygons') {
    ringStart = `${p}FeatureRings[${p}FeatureRingsOffset + f * 2u]`;
    ringEnd = `${p}FeatureRings[${p}FeatureRingsOffset + f * 2u + 1u]`;
    const ringOffsets =
      side.packedRingOffsetsBase === undefined
        ? `${p}RingOffsets[${p}RingOffsetsOffset`
        : `${p}FeatureRings[${p}FeatureRingsOffset + ${side.packedRingOffsetsBase}u`;
    vertexStart = `${ringOffsets} + r]`;
    vertexEnd = `min(${ringOffsets} + r + 1u], ${P}_VERTEX_COUNT)`;
  }
  const contains =
    kind === 'polygons'
      ? `
fn ${p}Contains(q: vec2f, f: u32) -> bool {
  var inside = false;
  ${forEachEdge(
    side,
    'f',
    'a',
    'b',
    `if ((a.y > q.y) != (b.y > q.y)) {
      let side = orientSide(a, b, q);
      if ((b.y > a.y && side > 0) || (b.y < a.y && side < 0)) { inside = !inside; }
    }`
  )}
  return inside;
}`
      : '';
  return `
const ${P}_VERTEX_COUNT: u32 = ${side.vertexCount}u;
fn ${p}RingStart(f: u32) -> u32 { return ${ringStart}; }
fn ${p}RingEnd(f: u32) -> u32 { return ${ringEnd}; }
fn ${p}RingVertexStart(r: u32) -> u32 { return ${vertexStart}; }
fn ${p}RingVertexEnd(r: u32) -> u32 { return ${vertexEnd}; }
fn ${p}Vertex(i: u32) -> vec2f { return ${vertex}; }
// (x, y, valid): first finite vertex of the first usable ring; valid is 0 for an empty feature.
fn ${p}First(f: u32) -> vec3f {
  for (var r = ${p}RingStart(f); r < ${p}RingEnd(f); r++) {
    let vs = ${p}RingVertexStart(r);
    let ve = ${p}RingVertexEnd(r);
    if (ve > vs && ve - vs >= ${getMinimumRingVertices(kind)}u) {
      let v = ${p}Vertex(vs);
      return vec3f(v, select(0.0, 1.0, isFiniteValue(v.x) && isFiniteValue(v.y)));
    }
  }
  return vec3f(0.0);
}${contains}
`;
}

/**
 * Declares `pairResult(queryRow, featureRow)`: the minimum planar distance between two geometries,
 * the nearest point on the feature, the nearest point on the query, and the absolute start-vertex index of the feature edge that
 * attains it.
 *
 * Distance is the minimum over all usable edge pairs (the first minimum in vertex order wins, so
 * the foot point is deterministic). Polygon containment either way gives distance 0 with the
 * contained vertex as foot point and no segment. A pair with no usable edges returns
 * `FLOAT32_MAXIMUM`.
 *
 * @internal
 */
export function getNearestPairWGSL(query: NearestSide, feature: NearestSide): string {
  const featureSegment =
    feature.kind === 'points' ? 'NO_SEGMENT' : feature.kind === 'segments' ? '0u' : 'rvs + rk';
  const inner = forEachEdge(
    feature,
    'rowF',
    'ra',
    'rb',
    `let distanceSq = segmentPairDistanceSq(qa, qb, ra, rb);
      if (distanceSq < result.distanceSq) {
        result.distanceSq = distanceSq;
        result.foot = segmentFoot(qa, qb, ra, rb);
        result.queryFoot = segmentFoot(ra, rb, qa, qb);
        result.segment = ${featureSegment};
        if (distanceSq == 0.0) { return result; }
      }`
  );
  const outer = forEachEdge(query, 'rowQ', 'qa', 'qb', inner);
  const featureInQuery =
    query.kind === 'polygons'
      ? `
  if (result.distanceSq > 0.0) {
    let first = rFirst(rowF);
    if (first.z > 0.0 && qContains(first.xy, rowQ)) {
      return PairResult(0.0, first.xy, NO_SEGMENT, first.xy);
    }
  }`
      : '';
  const queryInFeature =
    feature.kind === 'polygons'
      ? `
  if (result.distanceSq > 0.0) {
    let first = qFirst(rowQ);
    if (first.z > 0.0 && rContains(first.xy, rowF)) {
      return PairResult(0.0, first.xy, NO_SEGMENT, first.xy);
    }
  }`
      : '';
  return `
struct PairResult { distanceSq: f32, foot: vec2f, segment: u32, queryFoot: vec2f }
fn pairResult(rowQ: u32, rowF: u32) -> PairResult {
  var result = PairResult(FLOAT32_MAXIMUM, vec2f(0.0), NO_SEGMENT, vec2f(0.0));
  ${outer}${queryInFeature}${featureInQuery}
  return result;
}
`;
}

/** Options of {@link getNearestTraversalWGSL}. @internal */
export type NearestTraversalOptions = {
  /** Neighbors wanted per query. */
  k: number;
  /** Result slots per query, at least `k`; larger than `k` only in all-ties mode. */
  capacity: number;
  /** Number of features. */
  featureCount: number;
  /** Number of internal BVH nodes. */
  internalNodeCount: number;
  /** Maximum traversal stack depth. */
  stackSize: number;
  /** Row of the first query of this dispatch. Unused when `scratchChunk` is set. */
  chunkFirstRow: number;
  /**
   * Set when the `topk` scratch holds only one chunk of this dispatch's queries. The chunk's first
   * query (relative to the dispatch) is then read from `topk[CURSOR_WORD]`, a GPU word, and the
   * dispatch's first row and query count are pipeline overrides (`CHUNK_FIRST_ROW`,
   * `CHUNK_QUERY_COUNT`), so no chunk index reaches the WGSL text. `cursorWord` is the
   * `topk` element that holds the cursor.
   */
  scratchChunk?: {cursorWord: number};
  /**
   * Set when the kernel binds `queryOrder` (a permutation of the dispatch's query rows, see
   * `getHilbertQueryOrderNodes`). Invocation `index` (or scratch position `cursor + index`) then
   * handles query `queryOrder[position]`. Without `scratchChunk` the `topk` row of a query stays
   * its own row, so the clear and column kernels do not change; with it, the scratch row is the
   * position, and the column kernels map it back through the same order.
   */
  queryOrder?: boolean;
  /** Query geometry kind, for the usable-ring minimum. */
  queryKind: NearestSide['kind'];
  /**
   * Candidate filter read from the tail of `nodeData`: `[id, key]` words per query row from
   * `queryBase`, then per feature row from `featureBase` (word offsets inside `nodeData`).
   */
  filter?: {exclusive: boolean; onAttribute: boolean; queryBase: number; featureBase: number};
  /**
   * Measure distances as great-circle central angles between longitude/latitude positions instead
   * of squared planar distances; the node bounds are then lower-bounded with
   * `boxLowerAngle`. The kernel must include {@link NEAREST_GEODESIC_WGSL}.
   */
  greatCircle?: boolean;
};

/**
 * Lower bound, as a central angle in radians, of the great-circle distance between two
 * longitude/latitude boxes, for the great-circle traversal.
 *
 * The haversine of any two points is `hav(dlat) + cos(lat1) cos(lat2) hav(dlon)`. With the latitude
 * gap of the boxes, the circular longitude gap (through the antimeridian when shorter) and the
 * cosine of the latitude farthest from the equator over both boxes (a lower bound of both
 * cosines), the sum never overestimates. A box 180 degrees or wider in longitude is treated as
 * covering every longitude, which also covers features whose box straddles the antimeridian.
 *
 * @internal
 */
export const NEAREST_BOX_LOWER_ANGLE_WGSL = /* wgsl */ `
fn boxLowerAngle(
  nodeMinimum: vec2f, nodeMaximum: vec2f, queryMinimum: vec2f, queryMaximum: vec2f
) -> f32 {
  let latitudeGap = max(max(nodeMinimum.y - queryMaximum.y, queryMinimum.y - nodeMaximum.y), 0.0);
  var longitudeGap = 0.0;
  if (nodeMaximum.x - nodeMinimum.x < 180.0 && queryMaximum.x - queryMinimum.x < 180.0) {
    let direct = max(nodeMinimum.x - queryMaximum.x, queryMinimum.x - nodeMaximum.x);
    let around = 360.0 - (max(nodeMaximum.x, queryMaximum.x) - min(nodeMinimum.x, queryMinimum.x));
    longitudeGap = max(min(direct, around), 0.0);
  }
  let farLatitude = max(
    max(abs(nodeMinimum.y), abs(nodeMaximum.y)),
    max(abs(queryMinimum.y), abs(queryMaximum.y))
  );
  let cosine = geodesicCosLatitude(farLatitude);
  let sinHalfLatitude = sin(0.5 * latitudeGap * GEODESIC_DEGREES_TO_RADIANS);
  let sinHalfLongitude = sin(0.5 * longitudeGap * GEODESIC_DEGREES_TO_RADIANS);
  let term = sinHalfLatitude * sinHalfLatitude + cosine * cosine * sinHalfLongitude * sinHalfLongitude;
  return 2.0 * asin(sqrt(clamp(term, 0.0, 1.0)));
}`;

/**
 * Declarations and body of the per-query branch-and-bound BVH traversal.
 *
 * Each invocation walks the packed node table `nodeData` with an explicit stack, visiting the
 * nearer child first and pruning every node whose box distance exceeds the current k-th distance
 * (slightly widened so f32 rounding in the box bound never drops a tied candidate). It keeps the
 * best `capacity` rows in a list ordered by `(distance squared, feature row)` and writes the
 * result into `topk`: `[count, flags, bound bits, (row, distance-squared bits) * capacity]`.
 * Slot `[2]` must hold the initial squared bound (the clear pass writes it).
 *
 * @internal
 */
export function getNearestTraversalWGSL(options: NearestTraversalOptions): {
  declarations: string;
  body: string;
} {
  const {k, capacity} = options;
  const stride = NEAREST_HEADER_WORDS + 2 * capacity;
  const declarations = `
const K: u32 = ${k}u;
const CAPACITY: u32 = ${capacity}u;
const STRIDE: u32 = ${stride}u;
const FEATURE_COUNT: u32 = ${options.featureCount}u;
const INTERNAL_NODE_COUNT: u32 = ${options.internalNodeCount}u;
const STACK_SIZE: u32 = ${options.stackSize}u;
${
  options.scratchChunk
    ? `override CHUNK_FIRST_ROW: u32;
override CHUNK_QUERY_COUNT: u32;
const CURSOR_WORD: u32 = ${options.scratchChunk.cursorWord}u;`
    : `const CHUNK_FIRST_ROW: u32 = ${options.chunkFirstRow}u;`
}
const BOUND_SLACK: f32 = 1.000001;
${options.greatCircle ? NEAREST_BOX_LOWER_ANGLE_WGSL : ''}
// ${options.greatCircle ? 'Central-angle lower bound' : 'Squared distance'} from the query box to a node box, or FLOAT32_MAXIMUM for an empty node.
fn boxLower(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> f32 {
  let base = nodeDataOffset + node * ${NEAREST_NODE_WORDS}u;
  let nodeMinimum = vec2f(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  let nodeMaximum = vec2f(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u]));
  if (nodeMinimum.x > nodeMaximum.x || nodeMinimum.y > nodeMaximum.y) { return FLOAT32_MAXIMUM; }
  ${
    options.greatCircle
      ? 'return boxLowerAngle(nodeMinimum, nodeMaximum, queryMinimum, queryMaximum);'
      : `let gap = max(max(nodeMinimum - queryMaximum, queryMinimum - nodeMaximum), vec2f(0.0));
  return dot(gap, gap);`
  }
}`;
  // `localQuery` is the query's index inside this dispatch's query side, `row` its global row, and
  // `header` its slot in the (possibly chunked) `topk` scratch.
  const body = `${
    options.scratchChunk
      ? options.queryOrder
        ? `let position = topk[topkOffset + CURSOR_WORD] + index;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = queryOrder[queryOrderOffset + position];
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + index * STRIDE;`
        : `let localQuery = topk[topkOffset + CURSOR_WORD] + index;
  if (localQuery >= CHUNK_QUERY_COUNT) { return; }
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + index * STRIDE;`
      : options.queryOrder
        ? `let localQuery = queryOrder[queryOrderOffset + index];
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + row * STRIDE;`
        : `let localQuery = index;
  let row = CHUNK_FIRST_ROW + index;
  let header = topkOffset + row * STRIDE;`
  }
  var bound = bitcast<f32>(topk[header + 2u]);
  topk[header] = 0u;
  topk[header + 1u] = 0u;
  if (!(bound >= 0.0)) { return; }
  var queryMinimum = vec2f(FLOAT32_MAXIMUM);
  var queryMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var qRing = qRingStart(localQuery); qRing < qRingEnd(localQuery); qRing++) {
    let vertexStart = qRingVertexStart(qRing);
    let vertexEnd = qRingVertexEnd(qRing);
    if (vertexEnd <= vertexStart || vertexEnd - vertexStart < ${getMinimumRingVertices(options.queryKind)}u) { continue; }
    for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
      let position = qVertex(vertex);
      if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
        queryMinimum = min(queryMinimum, position);
        queryMaximum = max(queryMaximum, position);
      }
    }
  }
  if (queryMinimum.x > queryMaximum.x) { return; }

  var listDistance: array<f32, ${capacity}>;
  var listRow: array<u32, ${capacity}>;
  var count = 0u;
  var droppedMinimum = FLOAT32_MAXIMUM;
  var stackNode: array<u32, ${options.stackSize}>;
  var stackLower: array<f32, ${options.stackSize}>;
  var top = 0u;
  let rootLower = boxLower(0u, queryMinimum, queryMaximum);
  if (rootLower < FLOAT32_MAXIMUM) {
    stackNode[0] = 0u;
    stackLower[0] = rootLower;
    top = 1u;
  }
  loop {
    if (top == 0u) { break; }
    top = top - 1u;
    let node = stackNode[top];
    let limit = select(bound, bound * BOUND_SLACK, bound < 1.0e37);
    if (stackLower[top] > limit) { continue; }
    if (node < INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      let firstLower = boxLower(first, queryMinimum, queryMaximum);
      let secondLower = boxLower(second, queryMinimum, queryMaximum);
      let firstUsable = firstLower < FLOAT32_MAXIMUM && firstLower <= limit;
      let secondUsable = secondLower < FLOAT32_MAXIMUM && secondLower <= limit;
      // Push the farther child first so the nearer one is popped first.
      if (firstUsable && secondUsable && top + 2u <= STACK_SIZE) {
        if (firstLower <= secondLower) {
          stackNode[top] = second; stackLower[top] = secondLower;
          stackNode[top + 1u] = first; stackLower[top + 1u] = firstLower;
        } else {
          stackNode[top] = first; stackLower[top] = firstLower;
          stackNode[top + 1u] = second; stackLower[top + 1u] = secondLower;
        }
        top = top + 2u;
      } else if (firstUsable && top + 1u <= STACK_SIZE) {
        stackNode[top] = first; stackLower[top] = firstLower;
        top = top + 1u;
      } else if (secondUsable && top + 1u <= STACK_SIZE) {
        stackNode[top] = second; stackLower[top] = secondLower;
        top = top + 1u;
      }
      continue;
    }
    let featureRow = nodeData[nodeDataOffset + node * ${NEAREST_NODE_WORDS}u + 4u];
    if (featureRow >= FEATURE_COUNT) { continue; }
    ${
      options.filter
        ? `let queryFilter = nodeDataOffset + ${options.filter.queryBase}u + row * 2u;
    let featureFilter = nodeDataOffset + ${options.filter.featureBase}u + featureRow * 2u;
    ${options.filter.exclusive ? 'if (nodeData[queryFilter] == nodeData[featureFilter]) { continue; }' : ''}
    ${options.filter.onAttribute ? 'if (nodeData[queryFilter + 1u] != nodeData[featureFilter + 1u]) { continue; }' : ''}`
        : ''
    }
    let candidate = pairResult(localQuery, featureRow).distanceSq;
    if (!(candidate < FLOAT32_MAXIMUM) || candidate > bound) { continue; }
    // Find the slot of (candidate, featureRow) in the ordered list.
    var position = count;
    loop {
      if (position == 0u) { break; }
      let previousDistance = listDistance[position - 1u];
      if (previousDistance < candidate || (previousDistance == candidate && listRow[position - 1u] < featureRow)) { break; }
      position = position - 1u;
    }
    if (position >= CAPACITY) {
      droppedMinimum = min(droppedMinimum, candidate);
      continue;
    }
    var last = count;
    if (count == CAPACITY) {
      droppedMinimum = min(droppedMinimum, listDistance[CAPACITY - 1u]);
      last = CAPACITY - 1u;
    }
    var slot = last;
    loop {
      if (slot <= position) { break; }
      listDistance[slot] = listDistance[slot - 1u];
      listRow[slot] = listRow[slot - 1u];
      slot = slot - 1u;
    }
    listDistance[position] = candidate;
    listRow[position] = featureRow;
    count = min(count + 1u, CAPACITY);
    // Beyond k, keep only rows tied with the k-th distance.
    loop {
      if (count > K && listDistance[count - 1u] > listDistance[K - 1u]) { count = count - 1u; } else { break; }
    }
    if (count >= K) { bound = min(bound, listDistance[K - 1u]); }
  }
  var overflowFlag = 0u;
  if (CAPACITY > K && count >= K && droppedMinimum <= listDistance[K - 1u]) { overflowFlag = 1u; }
  for (var slot = 0u; slot < count; slot++) {
    topk[header + 3u + slot * 2u] = listRow[slot];
    topk[header + 4u + slot * 2u] = bitcast<u32>(listDistance[slot]);
  }
  topk[header] = count;
  topk[header + 1u] = overflowFlag;`;
  return {declarations, body};
}
