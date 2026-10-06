// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EXACT_ORIENTATION_WGSL} from '../segment-intersection/exact-orientation-wgsl';
import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
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
 * @internal
 */
export function getSpatialPredicateCommonWGSL(trackUncertainty: boolean): string {
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
  return /* wgsl */ `${SPATIAL_JOIN_WGSL_HELPERS}
${orient}
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
fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (lengthSq == 0.0) { let d = p - a; return dot(d, d); }
  let t = clamp(dot(p - a, ab) / lengthSq, 0.0, 1.0);
  let d = p - (a + ab * t);
  return dot(d, d);
}
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
  let cd = d - c;
  let denominator = crossProduct(ab, cd);
  if (denominator == 0.0) {
    if (orient(a, b, c) != 0.0 || orient(a, b, d) != 0.0) { return vec3f(0.0); }
    let lengthSq = dot(ab, ab);
    let tc = dot(c - a, ab) / lengthSq;
    let td = dot(d - a, ab) / lengthSq;
    var count = 0.0;
    var first = 0.0;
    var second = 0.0;
    if (tc > 0.0 && tc < 1.0) { first = tc; count = 1.0; }
    if (td > 0.0 && td < 1.0) {
      if (count == 0.0) { first = td; } else { second = td; }
      count = count + 1.0;
    }
    return vec3f(count, first, second);
  }
  // Orientation signs decide whether c-d crosses the open segment a-b. The divisions below only
  // place the break: deciding with them let a shared endpoint round to t = 0.99999994, whose
  // sliver piece was then classified by a rounded midpoint.
  let o3 = orient(c, d, a);
  let o4 = orient(c, d, b);
  if (!((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))) { return vec3f(0.0); }
  let o1 = orient(a, b, c);
  let o2 = orient(a, b, d);
  if ((o1 > 0.0 && o2 > 0.0) || (o1 < 0.0 && o2 < 0.0)) { return vec3f(0.0); }
  let t = crossProduct(c - a, cd) / denominator;
  if (t > 0.0 && t < 1.0) { return vec3f(1.0, t, 0.0); }
  return vec3f(0.0);
}
`;
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
// 0: the piece starting at t0 lies outside, 1: on the boundary, 2: strictly inside.
fn ${p}PieceCode(a: vec2f, b: vec2f, f: u32, t0: f32) -> u32 {
  let tm = 0.5 * (t0 + ${p}NextBreak(a, b, f, t0));
  if (${p}CollinearCovers(a, b, f, tm)) { return 1u; }
  let location = ${p}Locate(a + (b - a) * tm, f);
  if (location == 0u) { return 0u; }
  return select(1u, 2u, location == 1u);
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
  distance: number
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
      const squared = Math.fround(Math.fround(distance) * Math.fround(distance));
      predicateWGSL = `${getIntersectsWGSL(left, right)}
${getDistanceWGSL(left)}
const DISTANCE_SQ: f32 = ${getWGSLFloatLiteral(squared)};
fn pairMatches(l: u32, r: u32) -> bool { return pairDistanceSq(l, r) <= DISTANCE_SQ; }`;
      break;
    }
    default:
      throw new Error(`predicate ${predicate} is evaluated by the relate engine`);
  }
  return `${SPATIAL_PREDICATE_COMMON_WGSL}
${getSideWGSL(left)}
${getSideWGSL(right)}
${predicateWGSL}`;
}
