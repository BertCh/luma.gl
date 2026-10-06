// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  forEachEdge,
  getMinimumRingVertices,
  getSideWGSL,
  getSpatialPredicateCommonWGSL,
  type SpatialPredicateSide
} from './spatial-predicate-wgsl';
import {GPU_SPATIAL_RELATE_UNCERTAIN_BIT} from './spatial-relate-types';

/** Matrix cell index in `raise(a, b, dim)`: interior, boundary, exterior. */
const INTERIOR = '0u';
const BOUNDARY = '1u';
const EXTERIOR = '2u';

/** Emits a loop over every vertex of `feature`, binding `v` and the OGC part class `vc`. */
function forEachVertex(side: SpatialPredicateSide, feature: string, body: string): string {
  const p = side.prefix;
  const lineClosed =
    side.kind === 'lines'
      ? `let ${p}first = ${p}Vertex(${p}vs);
    let ${p}last = ${p}Vertex(${p}ve - 1u);
    let ${p}closed = ${p}first.x == ${p}last.x && ${p}first.y == ${p}last.y;`
      : '';
  const partClass =
    side.kind === 'points'
      ? INTERIOR
      : side.kind === 'polygons'
        ? BOUNDARY
        : `select(0u, 1u, !${p}closed && ((v.x == ${p}first.x && v.y == ${p}first.y) || (v.x == ${p}last.x && v.y == ${p}last.y)))`;
  return `for (var ${p}r = ${p}RingStart(${feature}); ${p}r < ${p}RingEnd(${feature}); ${p}r++) {
    let ${p}vs = ${p}RingVertexStart(${p}r);
    let ${p}ve = ${p}RingVertexEnd(${p}r);
    if (${p}ve <= ${p}vs || ${p}ve - ${p}vs < ${getMinimumRingVertices(side.kind)}u) { continue; }
    let ${p}n = ${p}ve - ${p}vs;
    ${lineClosed}
    for (var ${p}k = relateLane; ${p}k < ${p}n; ${p}k += relateStride) {
      let v = ${p}Vertex(${p}vs + ${p}k);
      let vc = ${partClass};
      ${body}
    }
  }`;
}

/**
 * WGSL boolean: the crossing of edges `a-b` and `c-d` is an endpoint of the unclosed line `side`.
 * That point belongs to the line boundary (the vertex passes record it), not to the interior,
 * even when the line passes over it again. False for other kinds.
 */
function endpointCrossing(side: SpatialPredicateSide): string {
  if (side.kind !== 'lines') {
    return 'false';
  }
  const p = side.prefix;
  const endpoint = (vertex: string) => `(onSegment(a, b, ${vertex}) && onSegment(c, d, ${vertex}))`;
  const first = `${p}Vertex(${p}vs)`;
  const last = `${p}Vertex(${p}ve - 1u)`;
  return `(!(${first}.x == ${last}.x && ${first}.y == ${last}.y) && (${endpoint(first)} || ${endpoint(last)}))`;
}

/** Declares the piece helpers of `side` acting as the "other" geometry of a pass. */
function getPieceHelpersWGSL(side: SpatialPredicateSide): string {
  const p = side.prefix;
  const box = `fn ${p}RelateBox(f: u32) -> vec4f {
  var box = vec4f(FLOAT32_MAXIMUM, FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM);
  ${forEachEdge(
    side,
    'f',
    'a',
    'b',
    'box = vec4f(min(box.xy, min(a, b)), max(box.zw, max(a, b)));'
  )}
  return box;
}
// Locate, short-cut for points outside the bounding box of the feature (the box is closed).
fn ${p}RelateLocate(q: vec2f, f: u32) -> u32 {
  if (q.x < ${p}Box.x || q.x > ${p}Box.z || q.y < ${p}Box.y || q.y > ${p}Box.w) { return 0u; }
  return ${p}Locate(q, f);
}
// False when the segment a-b is outside the bounding box of the feature: nothing there can touch it.
fn ${p}RelateNearEdge(a: vec2f, b: vec2f) -> bool {
  return !(max(a.x, b.x) < ${p}Box.x || min(a.x, b.x) > ${p}Box.z ||
    max(a.y, b.y) < ${p}Box.y || min(a.y, b.y) > ${p}Box.w);
}`;
  if (side.kind === 'points') {
    return `${box}
fn ${p}RelateNextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 { return 1.0; }
fn ${p}RelatePieceCode(a: vec2f, b: vec2f, f: u32, t0: f32, t1: f32) -> u32 { return 0u; }`;
  }
  const classify =
    side.kind === 'polygons'
      ? `let location = ${p}RelateLocate(a + (b - a) * tm, f);
  if (location == 0u) { return 0u; }
  return select(1u, 2u, location == 1u);`
      : 'return 0u;';
  return `${box}
// Next parameter after t in (0, 1] where segment a-b crosses or starts to overlap an edge.
fn ${p}RelateNextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 {
  var best = 1.0;
  if (!${p}RelateNearEdge(a, b)) { return best; }
  ${forEachEdge(
    side,
    'f',
    'c',
    'd',
    `if (dot(d - c, d - c) == 0.0 || boxesDisjoint(a, b, c, d)) { continue; }
    let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0 && breaks.y > t && breaks.y < best) { best = breaks.y; }
    if (breaks.x >= 2.0 && breaks.z > t && breaks.z < best) { best = breaks.z; }`
  )}
  return best;
}
fn ${p}RelateCovers(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (!${p}RelateNearEdge(a, b)) { return false; }
  ${forEachEdge(
    side,
    'f',
    'c',
    'd',
    `if (boxesDisjoint(a, b, c, d)) { continue; }
    if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
      let tc = dot(c - a, ab) / lengthSq;
      let td = dot(d - a, ab) / lengthSq;
      if (min(tc, td) <= tm && tm <= max(tc, td)) { return true; }
    }`
  )}
  return false;
}
// 0: the piece [t0, t1] of a-b lies outside, 1: on the boundary or line, 2: strictly inside.
fn ${p}RelatePieceCode(a: vec2f, b: vec2f, f: u32, t0: f32, t1: f32) -> u32 {
  let tm = 0.5 * (t0 + t1);
  if (${p}RelateCovers(a, b, f, tm)) { return 1u; }
  ${classify}
}`;
}

/**
 * One direction of the relate computation: every part of `x` is located against `y`.
 *
 * - Vertices of `x` give point-sized (dimension 0) contributions.
 * - Proper crossings and collinear overlaps of `x` edges with `y` edges give dimension 0 and 1
 *   contributions between the two edge classes (polygon ring edges are boundary, line edges are
 *   interior).
 * - Pieces of each `x` edge between its crossings with `y` lie outside `y`, in its interior or on it,
 *   which gives dimension 1 contributions. For two polygons, the same pieces decide the areal cells:
 *   a piece inside `y` has `x` interior and `x` exterior both in the interior of `y`, a piece outside
 *   `y` has the `x` interior side outside `y`, and a piece shared with the boundary of `y` is probed
 *   on both sides.
 *
 * Running both directions covers every cell, because every area of the arrangement of two
 * polygons is bounded by edge pieces of one of them.
 */
function getDirectionWGSL(
  x: SpatialPredicateSide,
  y: SpatialPredicateSide,
  xIsLeft: boolean
): string {
  const fx = xIsLeft ? 'l' : 'r';
  const fy = xIsLeft ? 'r' : 'l';
  const raiseXY = (xClass: string, yClass: string, dimension: number) =>
    xIsLeft
      ? `raise(${xClass}, ${yClass}, ${dimension}u);`
      : `raise(${yClass}, ${xClass}, ${dimension}u);`;
  const xEdgeClass = x.kind === 'polygons' ? BOUNDARY : INTERIOR;
  const yEdgeClass = y.kind === 'polygons' ? BOUNDARY : INTERIOR;
  const parts: string[] = [];

  parts.push(
    forEachVertex(x, fx, raiseXY('vc', `locationClass(${y.prefix}RelateLocate(v, ${fy}))`, 0))
  );

  if (x.kind !== 'points' && y.kind !== 'points') {
    parts.push(
      forEachEdge(
        x,
        fx,
        'a',
        'b',
        `if (!${y.prefix}RelateNearEdge(a, b)) { continue; }
        ${forEachEdge(
          y,
          fy,
          'c',
          'd',
          `if (boxesDisjoint(a, b, c, d)) { continue; }
      let o1 = orient(a, b, c);
      let o2 = orient(a, b, d);
      let o3 = orient(c, d, a);
      let o4 = orient(c, d, b);
      if (((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) &&
          ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0)) &&
          !(${endpointCrossing(x)} || ${endpointCrossing(y)})) {
        ${raiseXY(xEdgeClass, yEdgeClass, 0)}
      }
      if (o1 == 0.0 && o2 == 0.0) {
        let ab = b - a;
        let sc = dot(c - a, ab);
        let sd = dot(d - a, ab);
        let low = max(min(sc, sd), 0.0);
        let high = min(max(sc, sd), dot(ab, ab));
        if (high > low) { ${raiseXY(xEdgeClass, yEdgeClass, 1)} }
      }`
        )}`,
        true
      )
    );
  }

  if (x.kind !== 'points') {
    const bothPolygons = x.kind === 'polygons' && y.kind === 'polygons';
    const areaCases = bothPolygons
      ? `if (code == 2u) {
        ${raiseXY(INTERIOR, INTERIOR, 2)}
        ${raiseXY(EXTERIOR, INTERIOR, 2)}
      }
      if (code == 0u) { ${raiseXY(INTERIOR, EXTERIOR, 2)} }
      ${xIsLeft ? 'if (code == 1u) {' : 'if (false) {'}
        // Shared boundary piece: probe both sides along the normal, closer than any other edge.
        // Every shared segment lies on an edge of the left polygon, so the left pass sees all of them.
        let middle = a + ab * (0.5 * (t + next));
        let normal = vec2f(-ab.y, ab.x) / length(ab);
        let reach = relateProbeReach(middle, a, b, l, r);
        for (var side = 0u; side < 2u; side++) {
          let probe = middle + normal * select(reach, -reach, side == 1u);
          let inX = ${x.prefix}RelateLocate(probe, ${fx}) == 1u;
          let inY = ${y.prefix}RelateLocate(probe, ${fy}) == 1u;
          if (inX && inY) { ${raiseXY(INTERIOR, INTERIOR, 2)} }
          else if (inX) { ${raiseXY(INTERIOR, EXTERIOR, 2)} }
          else if (inY) { ${raiseXY(EXTERIOR, INTERIOR, 2)} }
        }
      }`
      : '';
    parts.push(
      forEachEdge(
        x,
        fx,
        'a',
        'b',
        `let ab = b - a;
    if (dot(ab, ab) == 0.0) { continue; }
    var t = 0.0;
    for (var guard = 0u; guard < ${y.prefix.toUpperCase()}_VERTEX_COUNT * 2u + 4u; guard++) {
      if (t >= 1.0) { break; }
      let next = ${y.prefix}RelateNextBreak(a, b, ${fy}, t);
      let code = ${y.prefix}RelatePieceCode(a, b, ${fy}, t, next);
      if (code == 0u) { ${raiseXY(xEdgeClass, EXTERIOR, 1)} }
      if (code == 2u) { ${raiseXY(xEdgeClass, INTERIOR, 1)} }
      ${areaCases}
      t = next;
    }`,
        true
      )
    );
  }

  if (x.kind === 'polygons' && y.kind !== 'polygons') {
    // The interior of a polygon always extends outside a point or a line.
    parts.push(raiseXY(INTERIOR, EXTERIOR, 2));
  }
  return parts.join('\n  ');
}

/**
 * Returns module-scope WGSL declaring `pairRelate(l, r) -> u32` for one pair of geometry kinds.
 *
 * The result is the packed DE-9IM matrix (see `packGPUSpatialRelate`), `0` when either feature is
 * empty or non-finite, with {@link GPU_SPATIAL_RELATE_UNCERTAIN_BIT} set when an orientation test
 * could not be certified. The caller must bind `${prefix}Positions` for each side, plus
 * `${prefix}RingOffsets` for lines and polygons and `${prefix}FeatureRings` for polygons.
 *
 * The matrix is derived from structure, not sampling: vertex locations, edge crossings and
 * collinear overlaps, and edge pieces between crossings located against the other geometry. Every
 * orientation sign is exact (shared `orientSign`), so the uncertain bit is set only for input with
 * no sign (non-finite coordinates, product exponents spanning more than 200 bits). Piece
 * midpoints and break parameters are still computed in f32.
 *
 * @internal
 */
export function getSpatialRelateWGSL(
  left: SpatialPredicateSide,
  right: SpatialPredicateSide
): string {
  const bothPolygons = left.kind === 'polygons' && right.kind === 'polygons';
  const probeReach = bothPolygons
    ? `
// Squared distance from q to the bounding box of segment c-d, a lower bound of its distance.
fn boxDistanceSq(q: vec2f, c: vec2f, d: vec2f) -> f32 {
  let gap = max(max(min(c, d) - q, q - max(c, d)), vec2f(0.0));
  return dot(gap, gap);
}
// Distance to probe from a shared piece: half the distance to the nearest non-collinear edge.
fn relateProbeReach(middle: vec2f, a: vec2f, b: vec2f, l: u32, r: u32) -> f32 {
  // The result is capped at length(b - a), so edges farther than twice that cannot change it.
  var best = 4.0 * dot(b - a, b - a);
  ${forEachEdge(
    left,
    'l',
    'c',
    'd',
    `if (boxDistanceSq(middle, c, d) >= best) { continue; }
    if (!(orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0)) { best = min(best, pointSegmentDistanceSq(middle, c, d)); }`
  )}
  ${forEachEdge(
    right,
    'r',
    'c',
    'd',
    `if (boxDistanceSq(middle, c, d) >= best) { continue; }
    if (!(orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0)) { best = min(best, pointSegmentDistanceSq(middle, c, d)); }`
  )}
  return min(0.5 * sqrt(best), length(b - a));
}`
    : '';
  return `${getSpatialPredicateCommonWGSL(true)}
${getSideWGSL(left)}
${getSideWGSL(right)}
${getPieceHelpersWGSL(left)}
${getPieceHelpersWGSL(right)}
var<private> relateMatrix: u32 = 0u;
// Raises cell (a, b) of the matrix to at least dimension \`dimension\`.
fn raise(a: u32, b: u32, dimension: u32) {
  let shift = (a * 3u + b) * 2u;
  let current = (relateMatrix >> shift) & 3u;
  if (dimension + 1u > current) {
    relateMatrix = (relateMatrix & ~(3u << shift)) | ((dimension + 1u) << shift);
  }
}
// Maps an OGC point location (0 exterior, 1 interior, 2 boundary) to a matrix part (I=0, B=1, E=2).
fn locationClass(location: u32) -> u32 {
  return select(select(1u, 0u, location == 1u), 2u, location == 0u);
}${probeReach}
var<private> relateLane: u32 = 0u;
var<private> relateStride: u32 = 1u;
var<private> leftBox: vec4f;
var<private> rightBox: vec4f;
// One lane of the matrix of (l, r): this lane handles every relateStride-th edge of the outer loops.
fn pairRelateLane(l: u32, r: u32) -> u32 {
  relateMatrix = 0u;
  uncertainOrientation = false;
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return 0u; }
  leftBox = leftRelateBox(l);
  rightBox = rightRelateBox(r);
  raise(${EXTERIOR}, ${EXTERIOR}, 2u);
  ${getDirectionWGSL(left, right, true)}
  ${getDirectionWGSL(right, left, false)}
  return relateMatrix | select(0u, ${GPU_SPATIAL_RELATE_UNCERTAIN_BIT}u, uncertainOrientation);
}
// The whole matrix of (l, r) in one invocation.
fn pairRelate(l: u32, r: u32) -> u32 {
  relateLane = 0u;
  relateStride = 1u;
  return pairRelateLane(l, r);
}`;
}

/** Invocations per pair of the workgroup relate kernel. @internal */
export const SPATIAL_RELATE_WORKGROUP_SIZE = 256;

/**
 * Returns module-scope WGSL declaring `relateWorkgroupPair(l, r, valid, lane) -> u32`, which splits
 * one pair across the `SPATIAL_RELATE_WORKGROUP_SIZE` lanes of a workgroup and merges the lane
 * matrices (cell-wise maximum, uncertain bit ORed). Append after {@link getSpatialRelateWGSL}.
 *
 * Every lane of the workgroup must call it with the same `(l, r, valid)` from uniform control
 * flow (it contains a barrier); only lane 0 receives the merged matrix.
 *
 * @internal
 */
export function getSpatialRelateWorkgroupWGSL(): string {
  const size = SPATIAL_RELATE_WORKGROUP_SIZE;
  return `
// Lane matrices merge with atomic ORs: each cell holds a thermometer code of its value (a value v
// sets the low v bits of a 3-bit field), so the OR of the lanes is the cell-wise maximum.
var<workgroup> relateMergedCells: atomic<u32>;
var<workgroup> relateMergedUncertain: atomic<u32>;
fn relateWorkgroupPair(l: u32, r: u32, valid: bool, lane: u32) -> u32 {
  relateLane = lane;
  relateStride = ${size}u;
  if (valid) {
    let lanes = pairRelateLane(l, r);
    var thermometer = 0u;
    for (var cell = 0u; cell < 9u; cell++) {
      thermometer = thermometer | (((1u << ((lanes >> (cell * 2u)) & 3u)) - 1u) << (cell * 3u));
    }
    if (thermometer != 0u) { atomicOr(&relateMergedCells, thermometer); }
    if ((lanes & ${GPU_SPATIAL_RELATE_UNCERTAIN_BIT}u) != 0u) { atomicOr(&relateMergedUncertain, 1u); }
  }
  workgroupBarrier();
  var merged = 0u;
  if (lane == 0u) {
    let cells = atomicLoad(&relateMergedCells);
    for (var cell = 0u; cell < 9u; cell++) {
      merged = merged | (countOneBits((cells >> (cell * 3u)) & 7u) << (cell * 2u));
    }
    merged = merged | select(0u, ${GPU_SPATIAL_RELATE_UNCERTAIN_BIT}u, atomicLoad(&relateMergedUncertain) != 0u);
  }
  return merged;
}`;
}
