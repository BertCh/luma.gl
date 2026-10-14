// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {SPATIAL_GEODESIC_MAXIMUM_ANGLE, SPATIAL_GEODESIC_WGSL} from './spatial-geodesic-wgsl';
import {SPATIAL_JOIN_WGSL_HELPERS} from './spatial-join-passes';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';

/** One side of a predicate join as seen by generated WGSL. @internal */
export type SpatialPredicateSide = {
  /** WGSL identifier prefix, `left` or `right`. Bindings are `${prefix}Positions` and so on. */
  prefix: 'left' | 'right';
  /** Geometry kind. */
  kind: GPUSpatialJoinGeometry['kind'];
  /** Number of vertices in `positions`, used to clamp malformed offsets. */
  vertexCount: number;
};

/**
 * Distance model of the `dwithin` kernels: planar Euclidean distance in coordinate units, or
 * great-circle distance between longitude/latitude positions on a sphere. In great-circle mode the
 * helpers `pointSegmentDistanceSq`, `segmentDistanceSq` and `distanceLimitSq` keep their names but
 * work in central angle (radians): edges are straight in longitude/latitude (like the containment
 * and intersection tests), the edge distance of two non-crossing edges is the smallest angle from
 * an endpoint of one to the other edge, and the limit is the distance divided by the sphere radius.
 *
 * @internal
 */
export type SpatialDistanceMetric = {kind: 'planar'} | {kind: 'great-circle'; radius: number};

const PLANAR_METRIC: SpatialDistanceMetric = {kind: 'planar'};

/** Public predicates of the join. @internal */
export type SpatialPredicateName =
  | SpatialLegacyPredicateName
  | 'covers'
  | 'coveredBy'
  | 'touches'
  | 'crosses'
  | 'overlaps'
  | 'equals'
  | 'containsProperly'
  | 'relate';

/** Predicates with dedicated short-circuiting kernels. @internal */
export type SpatialLegacyPredicateName = 'intersects' | 'contains' | 'within' | 'dwithin';

/**
 * Planar f32 predicates shared by every side and predicate.
 *
 * `orient` is the exact sign (see {@link getSpatialPredicateCommonWGSL}). A plain f32 determinant
 * is not usable here: compilers fuse `a * b - c * d` into an FMA, so even the orientation of an edge
 * against itself is not exactly zero, and edges shared by two polygons stopped being collinear
 * (the fast `contains`/`within` kernels then rejected polygons that share a boundary run).
 *
 * @internal
 */
export const SPATIAL_PREDICATE_COMMON_WGSL = getSpatialPredicateCommonWGSL(true);

/**
 * Returns {@link SPATIAL_PREDICATE_COMMON_WGSL}, optionally with exact orientation tests.
 *
 * With `trackUncertainty`, `orient` returns the exact sign (`-1.0`, `0.0` or `1.0`) of the
 * determinant from the shared `orientSign` (an f32 filter with an exact 256-bit integer fallback,
 * see `exact-orientation-wgsl.ts`), so `orient == 0` proves collinearity for any finite f32 input.
 * It sets the invocation-private `uncertainOrientation` flag only when no sign exists: non-finite
 * coordinates, or product exponents spanning more than 200 bits. Callers must not use the
 * magnitude of the result.
 *
 * With a great-circle `metric` the distance helpers measure central angles between longitude/latitude
 * positions (see {@link SpatialDistanceMetric}); the orientation tests stay planar in degrees.
 *
 * @internal
 */
export function getSpatialPredicateCommonWGSL(
  trackUncertainty: boolean,
  metric: SpatialDistanceMetric = PLANAR_METRIC
): string {
  const orient = trackUncertainty
    ? `${EXACT_ORIENTATION_WGSL}
var<private> uncertainOrientation: bool = false;
fn orient(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  // Coincident points make the determinant exactly zero. Shared edges hit this constantly, and an
  // exact zero is the one answer the f32 filter cannot certify (it falls to the slow exact path).
  if ((c.x == a.x && c.y == a.y) || (c.x == b.x && c.y == b.y) || (a.x == b.x && a.y == b.y)) {
    return 0.0;
  }
  let sign = orientSign(a, b, c);
  if (sign == 2) {
    uncertainOrientation = true;
    return 0.0;
  }
  return f32(sign);
}`
    : `fn orient(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}`;
  const pointSegmentDistance =
    metric.kind === 'great-circle'
      ? `// Central angle in radians from p to the segment a-b, straight in longitude/latitude.
fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  return geodesicSegmentDistance(p, a, b, geodesicSegmentFraction(p, a, b), 1.0);
}`
      : `fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (lengthSq == 0.0) { let d = p - a; return dot(d, d); }
  let t = clamp(dot(p - a, ab) / lengthSq, 0.0, 1.0);
  let d = p - (a + ab * t);
  return dot(d, d);
}`;
  return /* wgsl */ `${SPATIAL_JOIN_WGSL_HELPERS}
${metric.kind === 'great-circle' ? SPATIAL_GEODESIC_WGSL : ''}
${orient}
// Differs from nearest-wgsl: range test first, orient() may flag an uncertain sign.
fn onSegment(a: vec2f, b: vec2f, q: vec2f) -> bool {
  // The range test comes first so a sign that cannot matter is never reported as uncertain.
  return q.x >= min(a.x, b.x) && q.x <= max(a.x, b.x) &&
    q.y >= min(a.y, b.y) && q.y <= max(a.y, b.y) &&
    orient(a, b, q) == 0.0;
}
// True when the bounding boxes of segments a-b and c-d do not touch: no intersection is possible.
fn boxesDisjoint(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  return max(a.x, b.x) < min(c.x, d.x) || max(c.x, d.x) < min(a.x, b.x) ||
    max(a.y, b.y) < min(c.y, d.y) || max(c.y, d.y) < min(a.y, b.y);
}
fn segmentsIntersect(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> bool {
  if (boxesDisjoint(a, b, c, d)) { return false; }
  let o1 = orient(a, b, c);
  let o2 = orient(a, b, d);
  let o3 = orient(c, d, a);
  let o4 = orient(c, d, b);
  if (((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) &&
      ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))) {
    return true;
  }
  return onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b);
}
${pointSegmentDistance}
fn segmentDistanceSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  if (segmentsIntersect(a, b, c, d)) { return 0.0; }
  return min(
    min(pointSegmentDistanceSq(a, c, d), pointSegmentDistanceSq(b, c, d)),
    min(pointSegmentDistanceSq(c, a, b), pointSegmentDistanceSq(d, a, b))
  );
}
fn crossProduct(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }
// Parameters t in the open interval (0, 1) along a-b where segment c-d crosses or overlaps it.
// Returns (count, t1, t2). Requires a != b.
fn edgeBreaks(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> vec3f {
  if (boxesDisjoint(a, b, c, d)) { return vec3f(0.0); }
  let ab = b - a;
  // Every branch is decided by orientation signs only. Rounded arithmetic (the cross product of the
  // two directions, divisions) only places a break that the signs already proved to exist: an f32
  // \`denominator == 0\` test lost near-parallel crossings and missed exactly collinear overlaps.
  let o1 = orient(a, b, c);
  let o2 = orient(a, b, d);
  if (o1 == 0.0 && o2 == 0.0) {
    // Collinear: c and d break a-b where they lie strictly between a and b. On a common line the
    // coordinate along a non-degenerate axis orders the points exactly.
    let useX = a.x != b.x;
    let low = select(min(a.y, b.y), min(a.x, b.x), useX);
    let high = select(max(a.y, b.y), max(a.x, b.x), useX);
    let cValue = select(c.y, c.x, useX);
    let dValue = select(d.y, d.x, useX);
    let lengthSq = dot(ab, ab);
    var count = 0.0;
    var first = 0.0;
    var second = 0.0;
    let tc = dot(c - a, ab) / lengthSq;
    let td = dot(d - a, ab) / lengthSq;
    // A break that rounds onto an endpoint is dropped: its piece is below the parameter resolution.
    if (cValue > low && cValue < high && tc > 0.0 && tc < 1.0) { first = tc; count = 1.0; }
    if (dValue > low && dValue < high && td > 0.0 && td < 1.0) {
      if (count == 0.0) { first = td; } else { second = td; }
      count = count + 1.0;
    }
    return vec3f(count, first, second);
  }
  if ((o1 > 0.0 && o2 > 0.0) || (o1 < 0.0 && o2 < 0.0)) { return vec3f(0.0); }
  // c-d must cross the open segment a-b: a and b strictly on opposite sides of c-d.
  let o3 = orient(c, d, a);
  let o4 = orient(c, d, b);
  if (!((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))) { return vec3f(0.0); }
  let t = crossingParameter(a, b, c, d);
  if (t > 0.0 && t < 1.0) { return vec3f(1.0, t, 0.0); }
  return vec3f(0.0);
}
// Rounded parameter along a-b of its crossing with c-d, for a and b strictly on opposite sides of
// c-d: t = s3 / (s3 - s4) with the signed areas of a and b against c-d. Algebraically equal to
// cross(c - a, cd) / cross(ab, cd), but it cannot divide by a denominator that rounded to zero,
// and when the rounded areas keep opposite signs t stays inside [0, 1]. edgeBreaks and
// shiftedLineCrossing share it, so a crossing gets the same f32 value as its break.
fn crossingParameter(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  let cd = d - c;
  let s3 = crossProduct(cd, a - c);
  let s4 = crossProduct(cd, b - c);
  let span = s3 - s4;
  return select(0.5, clamp(s3 / span, 0.0, 1.0), span != 0.0 && (s3 * s4) < 0.0);
}
const NO_SHIFTED_CROSSING: f32 = -2.0;
// Where the line through a-b, shifted infinitesimally toward its positive side, crosses the edge c-d:
// NO_SHIFTED_CROSSING when it does not, -1 at or before a, 2 at or beyond b, otherwise the
// crossingParameter of a crossing strictly inside a-b. A vertex on the line counts as on its
// negative side, so the shifted line crosses c-d when exactly one endpoint is strictly positive;
// crossings of all edges then flip the even-odd location exactly, through vertices and collinear
// runs too. Every decision is an exact orientation sign; only the inside parameter is rounded.
fn shiftedLineCrossing(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  let o1 = orient(a, b, c);
  let o2 = orient(a, b, d);
  if ((o1 > 0.0) == (o2 > 0.0)) { return NO_SHIFTED_CROSSING; }
  // orient(c, d, a + s (b - a)) is linear in s, increasing exactly when c is the positive endpoint.
  let direction = select(-1.0, 1.0, o1 > 0.0);
  if (orient(c, d, b) * direction <= 0.0) { return 2.0; }
  if (orient(c, d, a) * direction >= 0.0) { return -1.0; }
  return crossingParameter(a, b, c, d);
}
`;
}

/**
 * Returns WGSL declaring `${name}(a, b, f, tm) -> bool`: whether the point `a + tm (b - a)` of an
 * open piece of segment a-b (no break of a-b against polygon feature `f` inside the piece, and not
 * on a collinear boundary edge) lies in the polygon.
 *
 * The rounded f32 point itself is not located: next to a boundary edge within a few ulps of the
 * piece (near-parallel or nearly shared edges) it can round onto the wrong side. Instead an
 * endpoint off the boundary is located exactly and flipped by every crossing between it and the
 * piece ({@link getSpatialPredicateCommonWGSL} `shiftedLineCrossing`). When both endpoints are on
 * the boundary, every crossing of the ray from the piece through b counts, from the exterior at
 * infinity. Crossing order along a-b comes from the same rounded parameters as the breaks.
 *
 * @param locate WGSL name of the exact point location of the side (`0` exterior, `1` interior,
 *   `2` boundary).
 * @param forEachCandidateEdge Emits `body` for every edge `c-d` of `f` whose y range may reach
 *   `[yLow, yHigh]` (WGSL expressions); the body may `continue`.
 * @internal
 */
export function getPieceInsideWGSL(
  name: string,
  locate: string,
  forEachCandidateEdge: (yLow: string, yHigh: string, body: string) => string
): string {
  return `fn ${name}(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  // Reference b counts crossings in (tm, 1], reference a crossings in [0, tm).
  var reference = ${locate}(b, f);
  var low = tm;
  var high = 1.5;
  var bounded = true;
  var yLow = min(a.y, b.y);
  var yHigh = max(a.y, b.y);
  if (reference == 2u) {
    reference = ${locate}(a, f);
    low = -0.5;
    high = tm;
    if (reference == 2u) {
      reference = 0u;
      low = tm;
      high = 3.0;
      bounded = false;
      yLow = -FLOAT32_MAXIMUM;
      yHigh = FLOAT32_MAXIMUM;
    }
  }
  var inside = reference == 1u;
  ${forEachCandidateEdge(
    'yLow',
    'yHigh',
    `if (bounded && boxesDisjoint(a, b, c, d)) { continue; }
    let crossing = shiftedLineCrossing(a, b, c, d);
    if (crossing > low && crossing < high) { inside = !inside; }`
  )}
  return inside;
}`;
}

export function getMinimumRingVertices(kind: SpatialPredicateSide['kind']): number {
  return kind === 'points' ? 1 : kind === 'lines' ? 2 : 3;
}

/**
 * Emits nested loops over every edge of `feature`, binding `a` and `b` to the edge endpoints.
 *
 * Points yield one degenerate edge, linestrings yield open edges and polygon rings close
 * implicitly. Rings with too few vertices are skipped. `body` may `return` or `continue`.
 * With `strided`, each ring's edges are visited `relateStride` apart from `relateLane` (private
 * variables of the relate engine), which splits the loop across the lanes of a workgroup.
 */
export function forEachEdge(
  side: SpatialPredicateSide,
  feature: string,
  a: string,
  b: string,
  body: string,
  strided = false
): string {
  const p = side.prefix;
  const edgeCount = side.kind === 'points' ? '1u' : side.kind === 'lines' ? `${p}n - 1u` : `${p}n`;
  const next =
    side.kind === 'points'
      ? `${p}k`
      : side.kind === 'lines'
        ? `${p}k + 1u`
        : `(${p}k + 1u) % ${p}n`;
  return `for (var ${p}r = ${p}RingStart(${feature}); ${p}r < ${p}RingEnd(${feature}); ${p}r++) {
    let ${p}vs = ${p}RingVertexStart(${p}r);
    let ${p}ve = ${p}RingVertexEnd(${p}r);
    if (${p}ve <= ${p}vs || ${p}ve - ${p}vs < ${getMinimumRingVertices(side.kind)}u) { continue; }
    let ${p}n = ${p}ve - ${p}vs;
    for (var ${p}k = ${strided ? 'relateLane' : '0u'}; ${p}k < ${edgeCount}; ${p}k += ${strided ? 'relateStride' : '1u'}) {
      let ${a} = ${p}Vertex(${p}vs + ${p}k);
      let ${b} = ${p}Vertex(${p}vs + ${next});
      ${body}
    }
  }`;
}

/** Declares the per-side accessors and geometry queries used by the predicate functions. */
export function getSideWGSL(side: SpatialPredicateSide): string {
  const p = side.prefix;
  const P = p.toUpperCase();
  const {kind} = side;
  const ringStart = kind === 'polygons' ? `${p}FeatureRings[${p}FeatureRingsOffset + f * 2u]` : 'f';
  const ringEnd =
    kind === 'polygons' ? `${p}FeatureRings[${p}FeatureRingsOffset + f * 2u + 1u]` : 'f + 1u';
  const vertexStart = kind === 'points' ? 'r' : `${p}RingOffsets[${p}RingOffsetsOffset + r]`;
  const vertexEnd =
    kind === 'points'
      ? 'r + 1u'
      : `min(${p}RingOffsets[${p}RingOffsetsOffset + r + 1u], ${P}_VERTEX_COUNT)`;

  let locateBody: string;
  if (kind === 'points') {
    locateBody = `let v = ${p}Vertex(f);
  return select(0u, 1u, v.x == q.x && v.y == q.y);`;
  } else if (kind === 'lines') {
    locateBody = `${forEachEdge(
      side,
      'f',
      'a',
      'b',
      `if (onSegment(a, b, q)) {
        let first = ${p}Vertex(${p}vs);
        let last = ${p}Vertex(${p}ve - 1u);
        let closed = first.x == last.x && first.y == last.y;
        if (!closed && ((q.x == first.x && q.y == first.y) || (q.x == last.x && q.y == last.y))) {
          return 2u;
        }
        return 1u;
      }`
    )}
  return 0u;`;
  } else {
    locateBody = `var inside = false;
  ${forEachEdge(
    side,
    'f',
    'a',
    'b',
    `if (onSegment(a, b, q)) { return 2u; }
      if ((a.y > q.y) != (b.y > q.y)) {
        let o = orient(a, b, q);
        if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
      }`
  )}
  return select(0u, 1u, inside);`;
  }

  let segmentWithin = '';
  if (kind === 'lines') {
    segmentWithin = `
// Bit 0: segment a-b lies entirely on the linestring. Bit 1: it overlaps the linestring interior.
fn ${p}SegmentWithin(a: vec2f, b: vec2f, f: u32) -> u32 {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (lengthSq == 0.0) {
    let location = ${p}Locate(a, f);
    return select(0u, select(1u, 3u, location == 1u), location != 0u);
  }
  var reach = 0.0;
  var passes = 0u;
  loop {
    if (reach >= 1.0 || passes > ${P}_VERTEX_COUNT) { break; }
    passes = passes + 1u;
    var progressed = false;
    ${forEachEdge(
      side,
      'f',
      'c',
      'd',
      `if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
        let tc = dot(c - a, ab) / lengthSq;
        let td = dot(d - a, ab) / lengthSq;
        let low = min(tc, td);
        let high = max(tc, td);
        if (low <= reach && high > reach) { reach = min(high, 1.0); progressed = true; }
      }`
    )}
    if (!progressed) { break; }
  }
  return select(0u, 3u, reach >= 1.0);
}`;
  } else if (kind === 'polygons') {
    segmentWithin = `
fn ${p}CollinearCovers(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  ${forEachEdge(
    side,
    'f',
    'c',
    'd',
    `if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
      let tc = dot(c - a, ab) / lengthSq;
      let td = dot(d - a, ab) / lengthSq;
      if (min(tc, td) <= tm && tm <= max(tc, td)) { return true; }
    }`
  )}
  return false;
}
fn ${p}NextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 {
  var best = 1.0;
  ${forEachEdge(
    side,
    'f',
    'c',
    'd',
    `let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0 && breaks.y > t && breaks.y < best) { best = breaks.y; }
    if (breaks.x >= 2.0 && breaks.z > t && breaks.z < best) { best = breaks.z; }`
  )}
  return best;
}
${getPieceInsideWGSL(`${p}PieceInside`, `${p}Locate`, (_yLow, _yHigh, body) =>
  forEachEdge(side, 'f', 'c', 'd', body)
)}
// 0: the piece starting at t0 lies outside, 1: on the boundary, 2: strictly inside.
fn ${p}PieceCode(a: vec2f, b: vec2f, f: u32, t0: f32) -> u32 {
  let tm = 0.5 * (t0 + ${p}NextBreak(a, b, f, t0));
  if (${p}CollinearCovers(a, b, f, tm)) { return 1u; }
  return select(0u, 2u, ${p}PieceInside(a, b, f, tm));
}
// Bit 0: segment a-b lies in the closed polygon. Bit 1: part of it lies strictly inside.
fn ${p}SegmentWithin(a: vec2f, b: vec2f, f: u32) -> u32 {
  let ab = b - a;
  if (dot(ab, ab) == 0.0) {
    let location = ${p}Locate(a, f);
    return select(0u, select(1u, 3u, location == 1u), location != 0u);
  }
  var interior = false;
  var code = ${p}PieceCode(a, b, f, 0.0);
  if (code == 0u) { return 0u; }
  interior = code == 2u;
  ${forEachEdge(
    side,
    'f',
    'c',
    'd',
    `let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0) {
      code = ${p}PieceCode(a, b, f, breaks.y);
      if (code == 0u) { return 0u; }
      interior = interior || code == 2u;
    }
    if (breaks.x >= 2.0) {
      code = ${p}PieceCode(a, b, f, breaks.z);
      if (code == 0u) { return 0u; }
      interior = interior || code == 2u;
    }`
  )}
  return select(1u, 3u, interior);
}`;
  }

  return `
const ${P}_VERTEX_COUNT: u32 = ${side.vertexCount}u;
fn ${p}RingStart(f: u32) -> u32 { return ${ringStart}; }
fn ${p}RingEnd(f: u32) -> u32 { return ${ringEnd}; }
fn ${p}RingVertexStart(r: u32) -> u32 { return ${vertexStart}; }
fn ${p}RingVertexEnd(r: u32) -> u32 { return ${vertexEnd}; }
fn ${p}Vertex(i: u32) -> vec2f {
  return vec2f(${p}Positions[${p}PositionsOffset + i * 2u], ${p}Positions[${p}PositionsOffset + i * 2u + 1u]);
}
// (x, y, valid): first vertex of the first usable ring; valid is 0 for an empty feature.
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
}
// 0: exterior, 1: interior, 2: boundary (OGC point location).
fn ${p}Locate(q: vec2f, f: u32) -> u32 {
  ${locateBody}
}
fn ${p}EdgesIntersect(qa: vec2f, qb: vec2f, f: u32) -> bool {
  ${forEachEdge(side, 'f', 'c', 'd', 'if (segmentsIntersect(qa, qb, c, d)) { return true; }')}
  return false;
}
fn ${p}MinimumDistanceSq(qa: vec2f, qb: vec2f, f: u32) -> f32 {
  var best = FLOAT32_MAXIMUM;
  ${forEachEdge(side, 'f', 'c', 'd', 'best = min(best, segmentDistanceSq(qa, qb, c, d));')}
  return best;
}${segmentWithin}
`;
}

/** Declares `pairIntersects(left, right)`. */
function getIntersectsWGSL(left: SpatialPredicateSide, right: SpatialPredicateSide): string {
  return `
fn pairIntersects(l: u32, r: u32) -> bool {
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return false; }
  ${forEachEdge(left, 'l', 'la', 'lb', 'if (rightEdgesIntersect(la, lb, r)) { return true; }')}
  ${right.kind === 'polygons' ? 'if (rightLocate(leftFirst(l).xy, r) != 0u) { return true; }' : ''}
  ${left.kind === 'polygons' ? 'if (leftLocate(rightFirst(r).xy, l) != 0u) { return true; }' : ''}
  return false;
}`;
}

/** Declares `pairDistanceSq(left, right)`: squared distance between the closed geometries. */
function getDistanceWGSL(left: SpatialPredicateSide): string {
  return `
fn pairDistanceSq(l: u32, r: u32) -> f32 {
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return FLOAT32_MAXIMUM; }
  if (pairIntersects(l, r)) { return 0.0; }
  var best = FLOAT32_MAXIMUM;
  ${forEachEdge(left, 'l', 'la', 'lb', 'best = min(best, rightMinimumDistanceSq(la, lb, r));')}
  return best;
}`;
}

/**
 * Declares `pairContains(outer, inner)` for the given kinds (OGC `ST_Contains`:
 * `inner` lies in the closure of `outer` and the interiors intersect).
 */
function getContainsWGSL(outer: SpatialPredicateSide, inner: SpatialPredicateSide): string {
  const o = outer.prefix;
  const i = inner.prefix;
  let body: string;
  if (inner.kind === 'points') {
    body = `return ${o}Locate(${i}First(inner).xy, outer) == 1u;`;
  } else if (outer.kind === 'points' || (outer.kind === 'lines' && inner.kind === 'polygons')) {
    body = 'return false;';
  } else if (outer.kind === 'lines') {
    body = `${forEachEdge(
      inner,
      'inner',
      'ia',
      'ib',
      `if ((${o}SegmentWithin(ia, ib, outer) & 1u) == 0u) { return false; }`
    )}
  return true;`;
  } else if (inner.kind === 'lines') {
    body = `var interior = false;
  ${forEachEdge(
    inner,
    'inner',
    'ia',
    'ib',
    `let within = ${o}SegmentWithin(ia, ib, outer);
    if ((within & 1u) == 0u) { return false; }
    interior = interior || (within & 2u) != 0u;`
  )}
  return interior;`;
  } else {
    // Polygon contains polygon. Every inner edge must lie in the closed outer polygon, no outer
    // vertex may lie strictly inside the inner polygon (a hole), and an inner edge that runs along
    // the outer boundary must have the outer interior on the inner interior's side (this rejects an
    // inner polygon that coincides with a hole).
    body = `${forEachEdge(
      inner,
      'inner',
      'ia',
      'ib',
      `if ((${o}SegmentWithin(ia, ib, outer) & 1u) == 0u) { return false; }
      let middle = (ia + ib) * 0.5;
      if (${o}Locate(middle, outer) == 2u) {
        let edge = ib - ia;
        let normal = vec2f(-edge.y, edge.x) * 0.001;
        var probe = middle + normal;
        var found = ${i}Locate(probe, inner) == 1u;
        if (!found) { probe = middle - normal; found = ${i}Locate(probe, inner) == 1u; }
        if (found && ${o}Locate(probe, outer) != 1u) { return false; }
      }`
    )}
  ${forEachEdge(outer, 'outer', 'oa', 'ob', `if (${i}Locate(oa, inner) == 1u) { return false; }`)}
  return true;`;
  }
  return `
fn pairContains(outer: u32, inner: u32) -> bool {
  if (${o}First(outer).z == 0.0 || ${i}First(inner).z == 0.0) { return false; }
  ${body}
}`;
}

/** Reads a `dwithin` distance from a bound storage row each time instead of a WGSL constant. @internal */
export type SpatialDistanceParameter = {
  /** WGSL expression of type `f32` that reads the distance, such as `row[rowOffset]`. */
  expression: string;
};

/**
 * Returns WGSL declaring `distanceLimitSq() -> f32`, the squared `dwithin` distance in f32. With a
 * number it is a compile-time constant. With a {@link SpatialDistanceParameter} it is read from the
 * bound row on every call; a negative, NaN or infinite distance gives -1, which no pair satisfies.
 * Both squares are one f32 multiply, so the two forms agree bit for bit.
 *
 * With a great-circle `metric` the distance is in meters and the function returns the central angle
 * `distance / radius` in radians instead of a square. A distance above
 * {@link SPATIAL_GEODESIC_MAXIMUM_ANGLE} (a per-frame value; a compile-time one is rejected by the
 * join) gives -1.
 *
 * @internal
 */
export function getDistanceLimitWGSL(
  distance: number | SpatialDistanceParameter,
  metric: SpatialDistanceMetric = PLANAR_METRIC
): string {
  if (metric.kind === 'great-circle') {
    const radius = getWGSLFloatLiteral(Math.fround(metric.radius));
    const maximum = getWGSLFloatLiteral(Math.fround(SPATIAL_GEODESIC_MAXIMUM_ANGLE));
    if (typeof distance === 'number') {
      const angle = Math.fround(Math.fround(distance) / Math.fround(metric.radius));
      return `fn distanceLimitSq() -> f32 { return ${getWGSLFloatLiteral(angle <= Math.fround(SPATIAL_GEODESIC_MAXIMUM_ANGLE) ? angle : -1)}; }`;
    }
    return `fn distanceLimitSq() -> f32 {
  let distance = ${distance.expression};
  let angle = distance / ${radius};
  return select(-1.0, angle, distance >= 0.0 && angle <= ${maximum});
}`;
  }
  if (typeof distance === 'number') {
    const squared = Math.fround(Math.fround(distance) * Math.fround(distance));
    return `fn distanceLimitSq() -> f32 { return ${getWGSLFloatLiteral(squared)}; }`;
  }
  return `fn distanceLimitSq() -> f32 {
  let distance = ${distance.expression};
  return select(-1.0, distance * distance, distance >= 0.0 && distance <= 3.4028234e38);
}`;
}

/**
 * Returns module-scope WGSL declaring `pairMatches(left, right) -> bool` for one predicate.
 *
 * The caller must bind `${prefix}Positions` for each side, plus `${prefix}RingOffsets` for lines and
 * polygons and `${prefix}FeatureRings` for polygons.
 *
 * @internal
 */
export function getSpatialPredicateWGSL(
  left: SpatialPredicateSide,
  right: SpatialPredicateSide,
  predicate: SpatialPredicateName,
  distance: number | SpatialDistanceParameter,
  metric: SpatialDistanceMetric = PLANAR_METRIC
): string {
  let predicateWGSL: string;
  switch (predicate) {
    case 'intersects':
      predicateWGSL = `${getIntersectsWGSL(left, right)}
fn pairMatches(l: u32, r: u32) -> bool { return pairIntersects(l, r); }`;
      break;
    case 'contains':
      predicateWGSL = `${getContainsWGSL(left, right)}
fn pairMatches(l: u32, r: u32) -> bool { return pairContains(l, r); }`;
      break;
    case 'within':
      predicateWGSL = `${getContainsWGSL(right, left)}
fn pairMatches(l: u32, r: u32) -> bool { return pairContains(r, l); }`;
      break;
    case 'dwithin': {
      predicateWGSL = `${getIntersectsWGSL(left, right)}
${getDistanceWGSL(left)}
${getDistanceLimitWGSL(distance, metric)}
fn pairMatches(l: u32, r: u32) -> bool { return pairDistanceSq(l, r) <= distanceLimitSq(); }`;
      break;
    }
    default:
      throw new Error(`predicate ${predicate} is evaluated by the relate engine`);
  }
  return `${metric.kind === 'planar' ? SPATIAL_PREDICATE_COMMON_WGSL : getSpatialPredicateCommonWGSL(true, metric)}
${getSideWGSL(left)}
${getSideWGSL(right)}
${predicateWGSL}`;
}
