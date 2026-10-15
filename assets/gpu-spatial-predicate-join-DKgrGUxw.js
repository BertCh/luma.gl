import{a as e,c as t,i as n,n as r,o as i,r as a,s as o,t as s}from"./spatial-join-geometry-C3pdxIH2.js";import{J as c}from"./src-CNCKFW2b.js";import{D as l,E as u,T as d,u as f,w as p}from"./analysis-frame-Blu0tYdw.js";import{_ as m,b as h,c as g,d as _,f as v,w as y,y as b}from"./resources-CwvhEOBe.js";import{t as x}from"./gpu-scan-BuGhTLDQ.js";import{a as S,l as C,n as ee,s as w}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as T}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{c as E}from"./spatial-weights-D4bd9AQ6.js";import{n as te,r as ne}from"./lon-lat-ring-wgsl-C5XyUe9j.js";import{n as D,t as re}from"./gpu-pairwise-point-in-polygon-DeHbidtn.js";import{h as O,l as ie,t as k,v as ae}from"./spatial-join-passes-C0ZwcuNs.js";import{n as A,t as j}from"./spatial-geodesic-wgsl-BI6DREOO.js";import{t as oe}from"./spatial-join-candidates-DK8ZndRC.js";var M={kind:`planar`},N=P(!0);function P(e,t=M){let n=e?`${T}
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
}`:`fn orient(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}`,r=t.kind===`great-circle`?`// Central angle in radians from p to the segment a-b, straight in longitude/latitude.
fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  return geodesicSegmentDistance(p, a, b, geodesicSegmentFraction(p, a, b), 1.0);
}`:`fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (lengthSq == 0.0) { let d = p - a; return dot(d, d); }
  let t = clamp(dot(p - a, ab) / lengthSq, 0.0, 1.0);
  let d = p - (a + ab * t);
  return dot(d, d);
}`;return`${k}
${t.kind===`great-circle`?A:``}
${n}
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
${r}
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
`}function F(e,t,n){return`fn ${e}(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  // Reference b counts crossings in (tm, 1], reference a crossings in [0, tm).
  var reference = ${t}(b, f);
  var low = tm;
  var high = 1.5;
  var bounded = true;
  var yLow = min(a.y, b.y);
  var yHigh = max(a.y, b.y);
  if (reference == 2u) {
    reference = ${t}(a, f);
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
  ${n(`yLow`,`yHigh`,`if (bounded && boxesDisjoint(a, b, c, d)) { continue; }
    let crossing = shiftedLineCrossing(a, b, c, d);
    if (crossing > low && crossing < high) { inside = !inside; }`)}
  return inside;
}`}function I(e){return e===`points`?1:e===`lines`?2:3}function L(e,t,n,r,i,a=!1){let o=e.prefix,s=e.kind===`points`?`1u`:e.kind===`lines`?`${o}n - 1u`:`${o}n`,c=e.kind===`points`?`${o}k`:e.kind===`lines`?`${o}k + 1u`:`(${o}k + 1u) % ${o}n`;return`for (var ${o}r = ${o}RingStart(${t}); ${o}r < ${o}RingEnd(${t}); ${o}r++) {
    let ${o}vs = ${o}RingVertexStart(${o}r);
    let ${o}ve = ${o}RingVertexEnd(${o}r);
    if (${o}ve <= ${o}vs || ${o}ve - ${o}vs < ${I(e.kind)}u) { continue; }
    let ${o}n = ${o}ve - ${o}vs;
    for (var ${o}k = ${a?`relateLane`:`0u`}; ${o}k < ${s}; ${o}k += ${a?`relateStride`:`1u`}) {
      let ${n} = ${o}Vertex(${o}vs + ${o}k);
      let ${r} = ${o}Vertex(${o}vs + ${c});
      ${i}
    }
  }`}function R(e){let t=e.prefix,n=t.toUpperCase(),{kind:r}=e,i=r===`polygons`?`${t}FeatureRings[${t}FeatureRingsOffset + f * 2u]`:`f`,a=r===`polygons`?`${t}FeatureRings[${t}FeatureRingsOffset + f * 2u + 1u]`:`f + 1u`,o=r===`points`?`r`:`${t}RingOffsets[${t}RingOffsetsOffset + r]`,s=r===`points`?`r + 1u`:`min(${t}RingOffsets[${t}RingOffsetsOffset + r + 1u], ${n}_VERTEX_COUNT)`,c;c=r===`points`?`let v = ${t}Vertex(f);
  return select(0u, 1u, v.x == q.x && v.y == q.y);`:r===`lines`?`${L(e,`f`,`a`,`b`,`if (onSegment(a, b, q)) {
        let first = ${t}Vertex(${t}vs);
        let last = ${t}Vertex(${t}ve - 1u);
        let closed = first.x == last.x && first.y == last.y;
        if (!closed && ((q.x == first.x && q.y == first.y) || (q.x == last.x && q.y == last.y))) {
          return 2u;
        }
        return 1u;
      }`)}
  return 0u;`:`var inside = false;
  ${L(e,`f`,`a`,`b`,`if (onSegment(a, b, q)) { return 2u; }
      if ((a.y > q.y) != (b.y > q.y)) {
        let o = orient(a, b, q);
        if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
      }`)}
  return select(0u, 1u, inside);`;let l=``;return r===`lines`?l=`
// Bit 0: segment a-b lies entirely on the linestring. Bit 1: it overlaps the linestring interior.
fn ${t}SegmentWithin(a: vec2f, b: vec2f, f: u32) -> u32 {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (lengthSq == 0.0) {
    let location = ${t}Locate(a, f);
    return select(0u, select(1u, 3u, location == 1u), location != 0u);
  }
  var reach = 0.0;
  var passes = 0u;
  loop {
    if (reach >= 1.0 || passes > ${n}_VERTEX_COUNT) { break; }
    passes = passes + 1u;
    var progressed = false;
    ${L(e,`f`,`c`,`d`,`if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
        let tc = dot(c - a, ab) / lengthSq;
        let td = dot(d - a, ab) / lengthSq;
        let low = min(tc, td);
        let high = max(tc, td);
        if (low <= reach && high > reach) { reach = min(high, 1.0); progressed = true; }
      }`)}
    if (!progressed) { break; }
  }
  return select(0u, 3u, reach >= 1.0);
}`:r===`polygons`&&(l=`
fn ${t}CollinearCovers(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  ${L(e,`f`,`c`,`d`,`if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
      let tc = dot(c - a, ab) / lengthSq;
      let td = dot(d - a, ab) / lengthSq;
      if (min(tc, td) <= tm && tm <= max(tc, td)) { return true; }
    }`)}
  return false;
}
fn ${t}NextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 {
  var best = 1.0;
  ${L(e,`f`,`c`,`d`,`let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0 && breaks.y > t && breaks.y < best) { best = breaks.y; }
    if (breaks.x >= 2.0 && breaks.z > t && breaks.z < best) { best = breaks.z; }`)}
  return best;
}
${F(`${t}PieceInside`,`${t}Locate`,(t,n,r)=>L(e,`f`,`c`,`d`,r))}
// 0: the piece starting at t0 lies outside, 1: on the boundary, 2: strictly inside.
fn ${t}PieceCode(a: vec2f, b: vec2f, f: u32, t0: f32) -> u32 {
  let tm = 0.5 * (t0 + ${t}NextBreak(a, b, f, t0));
  if (${t}CollinearCovers(a, b, f, tm)) { return 1u; }
  return select(0u, 2u, ${t}PieceInside(a, b, f, tm));
}
// Bit 0: segment a-b lies in the closed polygon. Bit 1: part of it lies strictly inside.
fn ${t}SegmentWithin(a: vec2f, b: vec2f, f: u32) -> u32 {
  let ab = b - a;
  if (dot(ab, ab) == 0.0) {
    let location = ${t}Locate(a, f);
    return select(0u, select(1u, 3u, location == 1u), location != 0u);
  }
  var interior = false;
  var code = ${t}PieceCode(a, b, f, 0.0);
  if (code == 0u) { return 0u; }
  interior = code == 2u;
  ${L(e,`f`,`c`,`d`,`let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0) {
      code = ${t}PieceCode(a, b, f, breaks.y);
      if (code == 0u) { return 0u; }
      interior = interior || code == 2u;
    }
    if (breaks.x >= 2.0) {
      code = ${t}PieceCode(a, b, f, breaks.z);
      if (code == 0u) { return 0u; }
      interior = interior || code == 2u;
    }`)}
  return select(1u, 3u, interior);
}`),`
const ${n}_VERTEX_COUNT: u32 = ${e.vertexCount}u;
fn ${t}RingStart(f: u32) -> u32 { return ${i}; }
fn ${t}RingEnd(f: u32) -> u32 { return ${a}; }
fn ${t}RingVertexStart(r: u32) -> u32 { return ${o}; }
fn ${t}RingVertexEnd(r: u32) -> u32 { return ${s}; }
fn ${t}Vertex(i: u32) -> vec2f {
  return vec2f(${t}Positions[${t}PositionsOffset + i * 2u], ${t}Positions[${t}PositionsOffset + i * 2u + 1u]);
}
// (x, y, valid): first vertex of the first usable ring; valid is 0 for an empty feature.
fn ${t}First(f: u32) -> vec3f {
  for (var r = ${t}RingStart(f); r < ${t}RingEnd(f); r++) {
    let vs = ${t}RingVertexStart(r);
    let ve = ${t}RingVertexEnd(r);
    if (ve > vs && ve - vs >= ${I(r)}u) {
      let v = ${t}Vertex(vs);
      return vec3f(v, select(0.0, 1.0, isFiniteValue(v.x) && isFiniteValue(v.y)));
    }
  }
  return vec3f(0.0);
}
// 0: exterior, 1: interior, 2: boundary (OGC point location).
fn ${t}Locate(q: vec2f, f: u32) -> u32 {
  ${c}
}
fn ${t}EdgesIntersect(qa: vec2f, qb: vec2f, f: u32) -> bool {
  ${L(e,`f`,`c`,`d`,`if (segmentsIntersect(qa, qb, c, d)) { return true; }`)}
  return false;
}
fn ${t}MinimumDistanceSq(qa: vec2f, qb: vec2f, f: u32) -> f32 {
  var best = FLOAT32_MAXIMUM;
  ${L(e,`f`,`c`,`d`,`best = min(best, segmentDistanceSq(qa, qb, c, d));`)}
  return best;
}${l}
`}function z(e,t){return`
fn pairIntersects(l: u32, r: u32) -> bool {
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return false; }
  ${L(e,`l`,`la`,`lb`,`if (rightEdgesIntersect(la, lb, r)) { return true; }`)}
  ${t.kind===`polygons`?`if (rightLocate(leftFirst(l).xy, r) != 0u) { return true; }`:``}
  ${e.kind===`polygons`?`if (leftLocate(rightFirst(r).xy, l) != 0u) { return true; }`:``}
  return false;
}`}function se(e){return`
fn pairDistanceSq(l: u32, r: u32) -> f32 {
  if (leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0) { return FLOAT32_MAXIMUM; }
  if (pairIntersects(l, r)) { return 0.0; }
  var best = FLOAT32_MAXIMUM;
  ${L(e,`l`,`la`,`lb`,`best = min(best, rightMinimumDistanceSq(la, lb, r));`)}
  return best;
}`}function B(e,t){let n=e.prefix,r=t.prefix,i;return i=t.kind===`points`?`return ${n}Locate(${r}First(inner).xy, outer) == 1u;`:e.kind===`points`||e.kind===`lines`&&t.kind===`polygons`?`return false;`:e.kind===`lines`?`${L(t,`inner`,`ia`,`ib`,`if ((${n}SegmentWithin(ia, ib, outer) & 1u) == 0u) { return false; }`)}
  return true;`:t.kind===`lines`?`var interior = false;
  ${L(t,`inner`,`ia`,`ib`,`let within = ${n}SegmentWithin(ia, ib, outer);
    if ((within & 1u) == 0u) { return false; }
    interior = interior || (within & 2u) != 0u;`)}
  return interior;`:`${L(t,`inner`,`ia`,`ib`,`if ((${n}SegmentWithin(ia, ib, outer) & 1u) == 0u) { return false; }
      let middle = (ia + ib) * 0.5;
      if (${n}Locate(middle, outer) == 2u) {
        let edge = ib - ia;
        let normal = vec2f(-edge.y, edge.x) * 0.001;
        var probe = middle + normal;
        var found = ${r}Locate(probe, inner) == 1u;
        if (!found) { probe = middle - normal; found = ${r}Locate(probe, inner) == 1u; }
        if (found && ${n}Locate(probe, outer) != 1u) { return false; }
      }`)}
  ${L(e,`outer`,`oa`,`ob`,`if (${r}Locate(oa, inner) == 1u) { return false; }`)}
  return true;`,`
fn pairContains(outer: u32, inner: u32) -> bool {
  if (${n}First(outer).z == 0.0 || ${r}First(inner).z == 0.0) { return false; }
  ${i}
}`}function ce(e,t=M){if(t.kind===`great-circle`){let n=C(Math.fround(t.radius)),r=C(Math.fround(j));if(typeof e==`number`){let n=Math.fround(Math.fround(e)/Math.fround(t.radius));return`fn distanceLimitSq() -> f32 { return ${C(n<=Math.fround(1.2566)?n:-1)}; }`}return`fn distanceLimitSq() -> f32 {
  let distance = ${e.expression};
  let angle = distance / ${n};
  return select(-1.0, angle, distance >= 0.0 && angle <= ${r});
}`}return typeof e==`number`?`fn distanceLimitSq() -> f32 { return ${C(Math.fround(Math.fround(e)*Math.fround(e)))}; }`:`fn distanceLimitSq() -> f32 {
  let distance = ${e.expression};
  return select(-1.0, distance * distance, distance >= 0.0 && distance <= 3.4028234e38);
}`}function le(e,t,n,r,i=M){let a;switch(n){case`intersects`:a=`${z(e,t)}
fn pairMatches(l: u32, r: u32) -> bool { return pairIntersects(l, r); }`;break;case`contains`:a=`${B(e,t)}
fn pairMatches(l: u32, r: u32) -> bool { return pairContains(l, r); }`;break;case`within`:a=`${B(t,e)}
fn pairMatches(l: u32, r: u32) -> bool { return pairContains(r, l); }`;break;case`dwithin`:a=`${z(e,t)}
${se(e)}
${ce(r,i)}
fn pairMatches(l: u32, r: u32) -> bool { return pairDistanceSq(l, r) <= distanceLimitSq(); }`;break;default:throw Error(`predicate ${n} is evaluated by the relate engine`)}return`${i.kind===`planar`?N:P(!0,i)}
${R(e)}
${R(t)}
${a}`}var V=2147483648,H=[`F`,`0`,`1`,`2`];function U(e){if(!/^[F012]{9}$/.test(e))throw Error(`DE-9IM matrix must be nine characters of F, 0, 1 or 2, got "${e}"`);let t=0;for(let n=0;n<9;n++)t|=H.indexOf(e[n])<<2*n;return t}function ue(e){let t=``;for(let n=0;n<9;n++)t+=H[e>>>2*n&3];return t}function W(e,t){let n=de(e);if(n.length===0)throw Error(`DE-9IM pattern must not be empty`);if(n.length>t)throw Error(`${n.length} DE-9IM patterns exceed the ${t} pattern slots`);if(K(n))throw Error(`pattern admits disjoint geometries, which are not bounding-box candidates; use how: 'anti' with predicate 'intersects' for disjoint`);let r=new Uint32Array(t*2);return n.forEach((e,t)=>{let n=G(e),i=0;for(let e=0;e<8;e++)i|=n[e]<<4*e;r[t*2]=i>>>0,r[t*2+1]=n[8]}),r}function de(e){return typeof e==`string`?[e]:e}function G(e){if(!/^[TF012*]{9}$/.test(e))throw Error(`DE-9IM pattern must be nine characters of T, F, 0, 1, 2 or *, got "${e}"`);let t={"*":15,T:14,F:1,0:2};return t[1]=4,t[2]=8,[...e].map(e=>t[e])}function K(e){return e.some(e=>{let t=G(e);return[0,1,3,4].every(e=>(t[e]&1)!=0)&&(t[8]&8)!=0})}function q(e){return e===`points`?0:e===`lines`?1:2}function fe(e,t,n){switch(e){case`intersects`:return[`T********`,`*T*******`,`***T*****`,`****T****`];case`contains`:return[`T*****FF*`];case`within`:return[`T*F**F***`];case`covers`:return[`T*****FF*`,`*T****FF*`,`***T**FF*`,`****T*FF*`];case`coveredBy`:return[`T*F**F***`,`*TF**F***`,`**FT*F***`,`**F*TF***`];case`touches`:return t===0&&n===0?[]:[`FT*******`,`F**T*****`,`F***T****`];case`crosses`:return t<n?[`T*T******`]:t>n?[`T*****T**`]:t===1?[`0********`]:[];case`overlaps`:return t===n?t===1?[`1*T***T**`]:[`T*T***T**`]:[];case`equals`:return[`T*F**FFF*`];case`containsProperly`:return[`T**FF*FF*`]}}function pe(e){let t=e.map(e=>{let t=G(e).map((e,t)=>e===15?void 0:`((${e}u >> ((m >> ${2*t}u) & 3u)) & 1u) != 0u`).filter(Boolean);return t.length?`(${t.join(` && `)})`:`true`});return`fn relateMatches(m: u32) -> bool { return ${t.length?t.join(` || `):`false`}; }`}function me(e){return`fn relateMatches(m: u32) -> bool {
  for (var slot = 0u; slot < ${e}u; slot++) {
    let low = patternWords[patternWordsOffset + slot * 2u];
    let high = patternWords[patternWordsOffset + slot * 2u + 1u];
    var matched = true;
    for (var cell = 0u; cell < 9u; cell++) {
      let mask = select((low >> (cell * 4u)) & 15u, high & 15u, cell == 8u);
      if (((mask >> ((m >> (cell * 2u)) & 3u)) & 1u) == 0u) { matched = false; break; }
    }
    if (matched) { return true; }
  }
  return false;
}`}function he(e,t,n){return`${le(e,t,`dwithin`,n)}
var<private> relateLane: u32 = 0u;
var<private> relateStride: u32 = 1u;
var<workgroup> dwithinMatched: atomic<u32>;
// Squared gap between the bounding boxes of segments a-b and c-d (zero when they overlap).
fn boxGapSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  let gap = max(max(min(a, b) - max(c, d), min(c, d) - max(a, b)), vec2f(0.0));
  return dot(gap, gap);
}
fn dwithinWorkgroupPair(l: u32, r: u32, valid: bool, lane: u32) -> bool {
  relateLane = lane;
  relateStride = 64u;
  let limitSq = distanceLimitSq();
  let reachSq = limitSq * 1.001 + 1e-30;
  if (valid && limitSq >= 0.0 && !(leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0)) {
    // Bounding box of the right feature.
    var rightLow = vec2f(FLOAT32_MAXIMUM);
    var rightHigh = vec2f(-FLOAT32_MAXIMUM);
    ${L(t,`r`,`c`,`d`,`rightLow = min(rightLow, min(c, d));
    rightHigh = max(rightHigh, max(c, d));`)}
    var matched = false;
    ${L(e,`l`,`a`,`b`,`if (matched) { break; }
    if (boxGapSq(a, b, rightLow, rightHigh) > reachSq) { continue; }
    ${L(t,`r`,`c`,`d`,`if (boxGapSq(a, b, c, d) > reachSq) { continue; }
      if (segmentDistanceSq(a, b, c, d) <= limitSq) { matched = true; break; }`)}`,!0)}
    // Containment without an edge within reach (one polygon inside the other, or a point in one).
    if (lane == 0u && !matched) {
      ${t.kind===`polygons`?`if (rightLocate(leftFirst(l).xy, r) != 0u) { matched = true; }`:``}
      ${e.kind===`polygons`?`if (leftLocate(rightFirst(r).xy, l) != 0u) { matched = true; }`:``}
    }
    if (matched) { atomicOr(&dwithinMatched, 1u); }
  }
  workgroupBarrier();
  return atomicLoad(&dwithinMatched) != 0u;
}`}var J=`0u`,ge=`1u`,Y=`2u`;function _e(e,t,n){let r=e.prefix,i=e.kind===`lines`?`let ${r}first = ${r}Vertex(${r}vs);
    let ${r}last = ${r}Vertex(${r}ve - 1u);
    let ${r}closed = ${r}first.x == ${r}last.x && ${r}first.y == ${r}last.y;`:``,a=e.kind===`points`?J:e.kind===`polygons`?ge:`select(0u, 1u, !${r}closed && ((v.x == ${r}first.x && v.y == ${r}first.y) || (v.x == ${r}last.x && v.y == ${r}last.y)))`;return`for (var ${r}r = ${r}RingStart(${t}); ${r}r < ${r}RingEnd(${t}); ${r}r++) {
    let ${r}vs = ${r}RingVertexStart(${r}r);
    let ${r}ve = ${r}RingVertexEnd(${r}r);
    if (${r}ve <= ${r}vs || ${r}ve - ${r}vs < ${I(e.kind)}u) { continue; }
    let ${r}n = ${r}ve - ${r}vs;
    ${i}
    for (var ${r}k = relateLane; ${r}k < ${r}n; ${r}k += relateStride) {
      let v = ${r}Vertex(${r}vs + ${r}k);
      let vc = ${a};
      ${n}
    }
  }`}function ve(e){if(e.kind!==`lines`)return`false`;let t=e.prefix,n=e=>`(onSegment(a, b, ${e}) && onSegment(c, d, ${e}))`,r=`${t}Vertex(${t}vs)`,i=`${t}Vertex(${t}ve - 1u)`;return`(!(${r}.x == ${i}.x && ${r}.y == ${i}.y) && (${n(r)} || ${n(i)}))`}function ye(e,t,n,r,i,a,o,s){let c=L(e,t,n,r,o);if(!s||e.kind!==`polygons`)return c;let l=e.prefix;return`if (${l}SlabActive) {
    for (var ${l}r = ${l}RingStart(${t}); ${l}r < ${l}RingEnd(${t}); ${l}r++) {
      let ${l}vs = ${l}RingVertexStart(${l}r);
      let ${l}ve = ${l}RingVertexEnd(${l}r);
      if (${l}ve <= ${l}vs || ${l}ve - ${l}vs < ${I(e.kind)}u) { continue; }
      let ${n} = ${l}Vertex(${l}ve - 1u);
      let ${r} = ${l}Vertex(${l}vs);
      ${o}
    }
    let ${l}QueryFirst = ${l}SlabOf(${i});
    let ${l}QueryLast = ${l}SlabOf(${a});
    for (var ${l}Cursor = ${l}QueryFirst; ${l}Cursor <= ${l}QueryLast; ${l}Cursor++) {
      for (var ${l}Entry = ${l}SlabStarts[${l}Cursor]; ${l}Entry < ${l}SlabStarts[${l}Cursor + 1u]; ${l}Entry++) {
        let ${l}EntryEdge = ${l}SlabEntries[${l}Entry];
        let ${n} = ${l}Vertex(${l}EntryEdge);
        let ${r} = ${l}Vertex(${l}EntryEdge + 1u);
        if (max(${l}QueryFirst, ${l}SlabOf(min(${n}.y, ${r}.y))) != ${l}Cursor) { continue; }
        ${o}
      }
    }
  } else {
    ${c}
  }`}function be(e,t){let n=e.prefix,r=`fn ${n}RelateBox(f: u32) -> vec4f {
  var box = vec4f(FLOAT32_MAXIMUM, FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM);
  ${L(e,`f`,`a`,`b`,`box = vec4f(min(box.xy, min(a, b)), max(box.zw, max(a, b)));`)}
  return box;
}
// Locate, short-cut for points outside the bounding box of the feature (the box is closed).
fn ${n}RelateLocate(q: vec2f, f: u32) -> u32 {
  if (q.x < ${n}Box.x || q.x > ${n}Box.z || q.y < ${n}Box.y || q.y > ${n}Box.w) { return 0u; }
  ${t&&e.kind===`polygons`?`if (${n}SlabActive) { return ${n}SlabLocate(q, f); }`:``}
  return ${n}Locate(q, f);
}
// False when the segment a-b is outside the bounding box of the feature: nothing there can touch it.
fn ${n}RelateNearEdge(a: vec2f, b: vec2f) -> bool {
  return !(max(a.x, b.x) < ${n}Box.x || min(a.x, b.x) > ${n}Box.z ||
    max(a.y, b.y) < ${n}Box.y || min(a.y, b.y) > ${n}Box.w);
}`;if(e.kind===`points`)return`${r}
fn ${n}RelateNextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 { return 1.0; }
fn ${n}RelateCollectBreaks(a: vec2f, b: vec2f, f: u32) -> u32 { return 0u; }
fn ${n}RelateBreakAfter(count: u32, t: f32, cursor: ptr<function, u32>) -> f32 { return 1.0; }
fn ${n}RelatePieceCode(a: vec2f, b: vec2f, f: u32, t0: f32, t1: f32) -> u32 { return 0u; }`;let i=e.kind===`polygons`?`return select(0u, 2u, ${n}RelatePieceInside(a, b, f, tm));`:`return 0u;`;return`${r}
${e.kind===`polygons`?F(`${n}RelatePieceInside`,`${n}RelateLocate`,(n,r,i)=>ye(e,`f`,`c`,`d`,n,r,i,t)):``}
// Next parameter after t in (0, 1] where segment a-b crosses or starts to overlap an edge.
fn ${n}RelateNextBreak(a: vec2f, b: vec2f, f: u32, t: f32) -> f32 {
  var best = 1.0;
  if (!${n}RelateNearEdge(a, b)) { return best; }
  ${L(e,`f`,`c`,`d`,`if (dot(d - c, d - c) == 0.0 || boxesDisjoint(a, b, c, d)) { continue; }
    let breaks = edgeBreaks(a, b, c, d);
    if (breaks.x >= 1.0 && breaks.y > t && breaks.y < best) { best = breaks.y; }
    if (breaks.x >= 2.0 && breaks.z > t && breaks.z < best) { best = breaks.z; }`)}
  return best;
}
// Every break parameter of a-b against the edges of the feature, ascending, in one pass over those
// edges: the piece loop then reads them in order instead of rescanning the edges for each piece.
// Returns the break count, or RELATE_BREAK_OVERFLOW when more than RELATE_BREAK_CAPACITY exist (the
// caller then falls back to \`RelateNextBreak\`, which finds each break by its own scan).
var<private> ${n}Breaks: array<f32, RELATE_BREAK_CAPACITY>;
// Whether some edge of the feature may be collinear with a-b, so \`RelateCovers\` has to run. The
// collection scan answers it once per edge instead of once per piece.
var<private> ${n}CoverNeeded: bool = false;
fn ${n}RelateCollectBreaks(a: vec2f, b: vec2f, f: u32) -> u32 {
  var count = 0u;
  ${n}CoverNeeded = false;
  if (!${n}RelateNearEdge(a, b)) { return 0u; }
  ${ye(e,`f`,`c`,`d`,`min(a.y, b.y)`,`max(a.y, b.y)`,`if (boxesDisjoint(a, b, c, d)) { continue; }
    if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) { ${n}CoverNeeded = true; }
    if (dot(d - c, d - c) == 0.0) { continue; }
    let breaks = edgeBreaks(a, b, c, d);
    let found = u32(breaks.x);
    for (var index = 0u; index < found; index++) {
      if (count == RELATE_BREAK_CAPACITY) { ${n}CoverNeeded = true; return RELATE_BREAK_OVERFLOW; }
      var position = count;
      let value = select(breaks.y, breaks.z, index == 1u);
      loop {
        if (position == 0u || ${n}Breaks[position - 1u] <= value) { break; }
        ${n}Breaks[position] = ${n}Breaks[position - 1u];
        position = position - 1u;
      }
      ${n}Breaks[position] = value;
      count = count + 1u;
    }`,t)}
  return count;
}
// First collected break after t, or 1.0. \`cursor\` only moves forward as t increases.
fn ${n}RelateBreakAfter(count: u32, t: f32, cursor: ptr<function, u32>) -> f32 {
  while (*cursor < count && ${n}Breaks[*cursor] <= t) { *cursor = *cursor + 1u; }
  return select(1.0, ${n}Breaks[*cursor], *cursor < count);
}
fn ${n}RelateCovers(a: vec2f, b: vec2f, f: u32, tm: f32) -> bool {
  let ab = b - a;
  let lengthSq = dot(ab, ab);
  if (!${n}RelateNearEdge(a, b)) { return false; }
  ${L(e,`f`,`c`,`d`,`if (boxesDisjoint(a, b, c, d)) { continue; }
    if (orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0) {
      let tc = dot(c - a, ab) / lengthSq;
      let td = dot(d - a, ab) / lengthSq;
      if (min(tc, td) <= tm && tm <= max(tc, td)) { return true; }
    }`)}
  return false;
}
// 0: the piece [t0, t1] of a-b lies outside, 1: on the boundary or line, 2: strictly inside.
fn ${n}RelatePieceCode(a: vec2f, b: vec2f, f: u32, t0: f32, t1: f32) -> u32 {
  let tm = 0.5 * (t0 + t1);
  if (${n}CoverNeeded && ${n}RelateCovers(a, b, f, tm)) { return 1u; }
  ${i}
}`}function xe(e,t,n,r){let i=n?`l`:`r`,a=n?`r`:`l`,o=(e,t,r)=>n?`raise(${e}, ${t}, ${r}u);`:`raise(${t}, ${e}, ${r}u);`,s=e.kind===`polygons`?ge:J,c=t.kind===`polygons`?ge:J,l=[];if(l.push(_e(e,i,o(`vc`,`locationClass(${t.prefix}RelateLocate(v, ${a}))`,0))),e.kind!==`points`&&t.kind!==`points`&&l.push(L(e,i,`a`,`b`,`if (!${t.prefix}RelateNearEdge(a, b)) { continue; }
        ${ye(t,a,`c`,`d`,`min(a.y, b.y)`,`max(a.y, b.y)`,`if (boxesDisjoint(a, b, c, d)) { continue; }
      let o1 = orient(a, b, c);
      let o2 = orient(a, b, d);
      let o3 = orient(c, d, a);
      let o4 = orient(c, d, b);
      if (((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) &&
          ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0)) &&
          !(${ve(e)} || ${ve(t)})) {
        ${o(s,c,0)}
      }
      if (o1 == 0.0 && o2 == 0.0) {
        let ab = b - a;
        let sc = dot(c - a, ab);
        let sd = dot(d - a, ab);
        let low = max(min(sc, sd), 0.0);
        let high = min(max(sc, sd), dot(ab, ab));
        if (high > low) { ${o(s,c,1)} }
      }`,r)}`,!0)),e.kind!==`points`){let r=e.kind===`polygons`&&t.kind===`polygons`?`if (code == 2u) {
        ${o(J,J,2)}
        ${o(Y,J,2)}
      }
      if (code == 0u) { ${o(J,Y,2)} }
      ${n?`if (code == 1u) {`:`if (false) {`}
        // Shared boundary piece: probe both sides along the normal, closer than any other edge.
        // Every shared segment lies on an edge of the left polygon, so the left pass sees all of them.
        let middle = a + ab * (0.5 * (t + next));
        let normal = vec2f(-ab.y, ab.x) / length(ab);
        let reach = relateProbeReach(middle, a, b, l, r);
        for (var side = 0u; side < 2u; side++) {
          let probe = middle + normal * select(reach, -reach, side == 1u);
          let inX = ${e.prefix}RelateLocate(probe, ${i}) == 1u;
          let inY = ${t.prefix}RelateLocate(probe, ${a}) == 1u;
          if (inX && inY) { ${o(J,J,2)} }
          else if (inX) { ${o(J,Y,2)} }
          else if (inY) { ${o(Y,J,2)} }
        }
      }`:``;l.push(L(e,i,`a`,`b`,`let ab = b - a;
    if (dot(ab, ab) == 0.0) { continue; }
    var t = 0.0;
    let breakCount = ${t.prefix}RelateCollectBreaks(a, b, ${a});
    var breakCursor = 0u;
    for (var guard = 0u; guard < ${t.prefix.toUpperCase()}_VERTEX_COUNT * 2u + 4u; guard++) {
      if (t >= 1.0) { break; }
      var next = 1.0;
      if (breakCount == RELATE_BREAK_OVERFLOW) {
        next = ${t.prefix}RelateNextBreak(a, b, ${a}, t);
      } else {
        next = ${t.prefix}RelateBreakAfter(breakCount, t, &breakCursor);
      }
      let code = ${t.prefix}RelatePieceCode(a, b, ${a}, t, next);
      if (code == 0u) { ${o(s,Y,1)} }
      if (code == 2u) { ${o(s,J,1)} }
      ${r}
      t = next;
    }`,!0))}return e.kind===`polygons`&&t.kind!==`polygons`&&l.push(o(J,Y,2)),l.join(`
  `)}var X=32,Se=1280,Ce=48;function we(e,t){let n=e.prefix;return`for (var ${n}r = ${n}RingStart(f); ${n}r < ${n}RingEnd(f); ${n}r++) {
    let ${n}vs = ${n}RingVertexStart(${n}r);
    let ${n}ve = ${n}RingVertexEnd(${n}r);
    if (${n}ve <= ${n}vs || ${n}ve - ${n}vs < ${I(e.kind)}u) { continue; }
    for (var ${n}k = relateLane; ${n}k + 1u < ${n}ve - ${n}vs; ${n}k += relateStride) {
      let a = ${n}Vertex(${n}vs + ${n}k);
      let b = ${n}Vertex(${n}vs + ${n}k + 1u);
      let edge = ${n}vs + ${n}k;
      ${t}
    }
  }`}function Te(e){let t=e.prefix;return`
var<workgroup> ${t}SlabCounts: array<atomic<u32>, ${X+1}>;
var<workgroup> ${t}SlabFill: array<atomic<u32>, ${X}>;
var<workgroup> ${t}SlabStarts: array<u32, ${X+1}>;
var<workgroup> ${t}SlabEntries: array<u32, ${Se}>;
var<workgroup> ${t}SlabFlag: u32;
var<private> ${t}SlabActive: bool = false;
var<private> ${t}SlabLow: f32 = 0.0;
var<private> ${t}SlabScale: f32 = 0.0;
fn ${t}SlabOf(y: f32) -> u32 {
  let t = (y - ${t}SlabLow) * ${t}SlabScale;
  return select(0u, u32(min(t, ${X-1}.0)), t > 0.0);
}
fn ${t}BuildSlabs(f: u32, enabled: bool) {
  ${t}SlabActive = false;
  if (relateLane <= ${X}u) { atomicStore(&${t}SlabCounts[relateLane], 0u); }
  if (relateLane < ${X}u) { atomicStore(&${t}SlabFill[relateLane], 0u); }
  workgroupBarrier();
  var usable = false;
  if (enabled) {
    let box = ${t}RelateBox(f);
    let range = box.w - box.y;
    var edges = 0u;
    for (var r = ${t}RingStart(f); r < ${t}RingEnd(f); r++) {
      let vs = ${t}RingVertexStart(r);
      let ve = ${t}RingVertexEnd(r);
      if (ve > vs && ve - vs >= ${I(e.kind)}u) { edges += ve - vs; }
    }
    ${t}SlabLow = box.y;
    ${t}SlabScale = ${X}.0 / range;
    usable = edges >= ${Ce}u && box.y >= -FLOAT32_MAXIMUM && box.w <= FLOAT32_MAXIMUM &&
      range > 0.0 && ${t}SlabScale <= FLOAT32_MAXIMUM;
  }
  if (usable) {
    ${we(e,`let first = ${t}SlabOf(min(a.y, b.y));
      let last = ${t}SlabOf(max(a.y, b.y));
      for (var slab = first; slab <= last; slab++) { atomicAdd(&${t}SlabCounts[slab], 1u); }`)}
  }
  workgroupBarrier();
  if (relateLane == 0u) {
    var running = 0u;
    for (var slab = 0u; slab < ${X}u; slab++) {
      ${t}SlabStarts[slab] = running;
      running += atomicLoad(&${t}SlabCounts[slab]);
    }
    ${t}SlabStarts[${X}] = running;
    ${t}SlabFlag = select(0u, 1u, usable && running <= ${Se}u);
  }
  workgroupBarrier();
  ${t}SlabActive = ${t}SlabFlag != 0u;
  if (${t}SlabActive) {
    ${we(e,`let first = ${t}SlabOf(min(a.y, b.y));
      let last = ${t}SlabOf(max(a.y, b.y));
      for (var slab = first; slab <= last; slab++) {
        let position = atomicAdd(&${t}SlabFill[slab], 1u);
        ${t}SlabEntries[${t}SlabStarts[slab] + position] = edge;
      }`)}
  }
  workgroupBarrier();
}
// Same answer as ${t}Locate for a polygon: 2 on the boundary, 1 inside, 0 outside.
fn ${t}SlabLocate(q: vec2f, f: u32) -> u32 {
  var inside = false;
  for (var r = ${t}RingStart(f); r < ${t}RingEnd(f); r++) {
    let vs = ${t}RingVertexStart(r);
    let ve = ${t}RingVertexEnd(r);
    if (ve <= vs || ve - vs < ${I(e.kind)}u) { continue; }
    let a = ${t}Vertex(ve - 1u);
    let b = ${t}Vertex(vs);
    if (onSegment(a, b, q)) { return 2u; }
    if ((a.y > q.y) != (b.y > q.y)) {
      let o = orient(a, b, q);
      if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
    }
  }
  let slab = ${t}SlabOf(q.y);
  for (var i = ${t}SlabStarts[slab]; i < ${t}SlabStarts[slab + 1u]; i++) {
    let edge = ${t}SlabEntries[i];
    let a = ${t}Vertex(edge);
    let b = ${t}Vertex(edge + 1u);
    if (onSegment(a, b, q)) { return 2u; }
    if ((a.y > q.y) != (b.y > q.y)) {
      let o = orient(a, b, q);
      if ((b.y > a.y && o > 0.0) || (b.y < a.y && o < 0.0)) { inside = !inside; }
    }
  }
  return select(0u, 1u, inside);
}`}function Ee(e,t,n=!1){let r=e.kind===`polygons`&&t.kind===`polygons`?`
// Squared distance from q to the bounding box of segment c-d, a lower bound of its distance.
fn boxDistanceSq(q: vec2f, c: vec2f, d: vec2f) -> f32 {
  let gap = max(max(min(c, d) - q, q - max(c, d)), vec2f(0.0));
  return dot(gap, gap);
}
// Distance to probe from a shared piece: half the distance to the nearest non-collinear edge.
fn relateProbeReach(middle: vec2f, a: vec2f, b: vec2f, l: u32, r: u32) -> f32 {
  // The result is capped at length(b - a), so edges farther than twice that cannot change it.
  var best = 4.0 * dot(b - a, b - a);
  ${L(e,`l`,`c`,`d`,`if (boxDistanceSq(middle, c, d) >= best) { continue; }
    if (!(orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0)) { best = min(best, pointSegmentDistanceSq(middle, c, d)); }`)}
  ${L(t,`r`,`c`,`d`,`if (boxDistanceSq(middle, c, d) >= best) { continue; }
    if (!(orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0)) { best = min(best, pointSegmentDistanceSq(middle, c, d)); }`)}
  return min(0.5 * sqrt(best), length(b - a));
}`:``;return`${P(!0)}
const RELATE_BREAK_CAPACITY: u32 = 24u;
const RELATE_BREAK_OVERFLOW: u32 = 0xffffffffu;
${R(e)}
${R(t)}
${be(e,n)}
${be(t,n)}
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
}${r}
${n?[e,t].filter(e=>e.kind===`polygons`).map(Te).join(`
`):``}
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
  raise(${Y}, ${Y}, 2u);
  ${xe(e,t,!0,n)}
  ${xe(t,e,!1,n)}
  return relateMatrix | select(0u, ${V}u, uncertainOrientation);
}
// The whole matrix of (l, r) in one invocation.
fn pairRelate(l: u32, r: u32) -> u32 {
  relateLane = 0u;
  relateStride = 1u;
  return pairRelateLane(l, r);
}`}function De(e=[]){let t=e.filter(e=>e.kind===`polygons`).map(e=>`${e.prefix}BuildSlabs(${e.prefix===`left`?`l`:`r`}, usable);`).join(`
  `);return`
// Lane matrices merge with atomic ORs: each cell holds a thermometer code of its value (a value v
// sets the low v bits of a 3-bit field), so the OR of the lanes is the cell-wise maximum.
var<workgroup> relateMergedCells: atomic<u32>;
var<workgroup> relateMergedUncertain: atomic<u32>;
fn relateWorkgroupPair(l: u32, r: u32, valid: bool, lane: u32) -> u32 {
  relateLane = lane;
  relateStride = 256u;
  ${t?`// Per-pair y-slab indexes of the polygon sides (barriers: reached by every lane).
  let usable = valid && leftFirst(l).z != 0.0 && rightFirst(r).z != 0.0;
  ${t}`:``}
  if (valid) {
    let lanes = pairRelateLane(l, r);
    var thermometer = 0u;
    for (var cell = 0u; cell < 9u; cell++) {
      thermometer = thermometer | (((1u << ((lanes >> (cell * 2u)) & 3u)) - 1u) << (cell * 3u));
    }
    if (thermometer != 0u) { atomicOr(&relateMergedCells, thermometer); }
    if ((lanes & ${V}u) != 0u) { atomicOr(&relateMergedUncertain, 1u); }
  }
  workgroupBarrier();
  var merged = 0u;
  if (lane == 0u) {
    let cells = atomicLoad(&relateMergedCells);
    for (var cell = 0u; cell < 9u; cell++) {
      merged = merged | (countOneBits((cells >> (cell * 3u)) & 7u) << (cell * 2u));
    }
    merged = merged | select(0u, ${V}u, atomicLoad(&relateMergedUncertain) != 0u);
  }
  return merged;
}`}var Oe=16,ke=4,Ae=8,je=16,Me=8,Ne=16,Pe=16,Fe=8,Ie=12,Le=20,Re=8;function ze(e){let t=Z(e.leftCount,`leftCount`),n=Z(e.rightCount,`rightCount`),r=e.leftKind??`lines`,i=e.rightKind??`lines`,a=He(t,e.expectedCandidatesPerLeft??Math.min(n,Re),e.observedCandidateCount,`observedCandidateCount`),o=He(t,e.expectedMatchesPerLeft??Math.min(n,1),e.observedRequiredCount,`observedRequiredCount`),s=r===`points`&&i===`polygons`||r===`polygons`&&i===`points`,c=Oe+(e.relate?ke:0)+(s?2*Ne:0),l=Ve(a,e.candidateCapacity,0,c),u=Ve(o,e.pairCapacity,Ae,0),d=e.leafCapacity??g(Math.max(n,1)),f=(r===`polygons`?t:0)*Me+(i===`polygons`?n:0)*Me,p=s?Le+Ie:0,m=(t+n)*je+t*Fe+f+O(d)+Pe+Ie+(e.relate?Ie:0)+p;return{candidates:l,pairs:u,estimatedWorkItems:Q(t+n+a),estimatedOutputBytes:u.estimatedOutputBytes,estimatedPeakTransientBytes:Q(m+l.estimatedPeakTransientBytes)}}function Be(e,t){return Math.max(1,ze({leftCount:e,rightCount:t}).candidates.capacity)}function Ve(e,t,n,r){let i=t===void 0?e:Z(t,`capacity`);return{capacity:i,estimatedRequiredCount:e,limited:i<e,estimatedOutputBytes:Q(i*n),estimatedPeakTransientBytes:Q(i*r)}}function He(e,t,n,r){if(n!==void 0)return Z(n,r);if(!Number.isFinite(t)||t<0)throw Error(`expansionFactor must be a non-negative finite number`);return Q(Math.ceil(e*t))}function Z(e,t){if(!Number.isSafeInteger(e)||e<0)throw Error(`${t} must be a non-negative safe integer`);return e}function Q(e){return Math.min(2**53-1,e)}var $=`GPUSpatialPredicateJoin`,Ue=(262143|V)>>>0,We=8,Ge=Object.freeze({coordinateSpaces:[`planar`,`longitude-latitude`],edgeModels:f,metrics:[`none`,`native`,`great-circle`]}),Ke=Object.freeze({...Ge,measure:`distance`}),qe=[`intersects`,`contains`,`within`,`dwithin`],Je=[`covers`,`coveredBy`,`touches`,`crosses`,`overlaps`,`equals`,`containsProperly`,`relate`],Ye=1024;function Xe(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function Ze(e){return e.positions.length/Math.max(a(e),1)}function Qe(e,t){return e.kind===`points`||t.kind===`points`?!1:Ze(e)*Ze(t)>=Ye}var $e=class{id;props;views;leftCount;rightCount;candidateCapacity;pairCapacity;leafCapacity;how;usesRelateEngine;usesWorkgroupDistance;spatialContext;sphereRadius;relatePatterns;patternView;distanceView;constructor(n){this.id=n.id??`spatial-predicate-join`,this.props=n;let{output:r,...s}=n,c={...s,...r,...r.status};this.views=c;let{id:f}=this;i(f,`left`,c.left),i(f,`right`,c.right),this.leftCount=a(c.left),this.rightCount=a(c.right);let{predicate:m}=c,y=qe.includes(m);if(this.distanceView=m===`dwithin`&&Xe(c.distance)?c.distance:void 0,this.patternView=Xe(c.pattern)?c.pattern:void 0,m===`dwithin`){if(this.distanceView){if(b(this.distanceView,[`float32`],`${f} distance`),this.distanceView.length<1)throw Error(`${f} distance must contain one float32 row`)}else if(typeof c.distance!=`number`||!Number.isFinite(c.distance)||c.distance<0)throw Error(`${f} dwithin requires a finite, non-negative distance`)}else if(!y&&!Je.includes(m))throw Error(`${f} unknown predicate ${String(m)}`);let x=u(f,c.spatialContext,m===`dwithin`?Ke:Ge);if(this.spatialContext=x,d(f,`right`,c.rightSpatialContext,this.spatialContext,Ge.edgeModels),x.coordinateSpace===`longitude-latitude`){if(m===`dwithin`&&x.metric!==`great-circle`)throw Error(`${f} longitude-latitude dwithin needs metric 'great-circle'`);if(m!==`dwithin`&&x.metric!==`none`)throw Error(`${f} ${m} needs metric 'none' on longitude-latitude`)}if(m===`dwithin`&&typeof c.distance==`number`&&l(`GPUSpatialPredicateJoin`,this.spatialContext,c.distance),x.metric===`great-circle`){if(x.units===`kilometers`)throw Error(`${f} great-circle dwithin distances are in meters`);if(this.sphereRadius=p(x),typeof c.distance==`number`&&c.distance/this.sphereRadius>1.2566)throw Error(`${f} distance exceeds ${Math.floor(j*this.sphereRadius)} m`)}else this.sphereRadius=void 0;if(this.how=c.how??`inner`,this.how!==`inner`&&this.how!==`anti`)throw Error(`${f} unknown how ${String(this.how)}`);if(m===`relate`){if(c.pattern===void 0)throw Error(`${f} predicate 'relate' requires a pattern`)}else if(c.pattern!==void 0)throw Error(`${f} pattern requires predicate 'relate'`);if(c.relate&&m===`dwithin`)throw Error(`${f} relate output is not available for dwithin`);let S=c.engine??`auto`;if(S!==`auto`&&S!==`fast`&&S!==`relate`)throw Error(`${f} unknown engine ${String(S)}`);if(S===`fast`&&!y)throw Error(`${f} engine 'fast' needs intersects, contains, within or dwithin`);if(this.sphereRadius!==void 0&&S===`relate`)throw Error(`${f} engine 'relate' is not available for great-circle dwithin`);let C=S===`relate`||S===`auto`&&Qe(c.left,c.right);if(this.usesRelateEngine=!y||c.relate!==void 0||m!==`dwithin`&&C,this.usesWorkgroupDistance=m===`dwithin`&&C&&this.sphereRadius===void 0,this.patternView){let e=this.patternView;if(b(e,[`uint32`],`${f} pattern`),e.length<2||e.length%2!=0)throw Error(`${f} pattern view must hold a positive multiple of 2 words`);this.relatePatterns=[]}else if(this.usesRelateEngine){let e=m===`relate`?[...de(c.pattern)]:[...fe(m,q(c.left.kind),q(c.right.kind))];if(m===`relate`&&e.length===0)throw Error(`${f} pattern must not be empty`);if(K(e))throw Error(`${f} pattern admits disjoint pairs; use how: 'anti'`);this.relatePatterns=e}else this.relatePatterns=[];this.candidateCapacity=c.candidateCapacity??Be(this.leftCount,this.rightCount),_(f,`candidateCapacity`,this.candidateCapacity,1,2147483646);let ee=m!==`dwithin`&&(c.left.kind===`points`&&c.right.kind===`polygons`||c.left.kind===`polygons`&&c.right.kind===`points`),w=this.usesWorkgroupDistance?64:this.usesRelateEngine&&!ee?256:1;if(this.candidateCapacity*w>4294967295)throw Error(`${f} candidateCapacity exceeds ${Math.floor(4294967295/w)}`);if(this.leafCapacity=c.prepared?.leafCapacity??c.leafCapacity??g(Math.max(this.rightCount,1)),h(f,`leafCapacity`,this.leafCapacity),c.prepared&&!e(c.prepared.geometry,c.right))throw Error(`${f} prepared must index the same geometry views as right`);if(this.how===`anti`){if(!c.unmatched)throw Error(`${f} how 'anti' needs output.unmatched`);if(c.pairs||c.weights||c.relate)throw Error(`${f} how 'anti' writes only output.unmatched`);v(f,c.unmatched),this.pairCapacity=0}else{if(c.unmatched)throw Error(`${f} output.unmatched needs how 'anti'`);if(!c.pairs&&!c.weights)throw Error(`${f} needs output.pairs or output.weights`);if(!c.pairs&&!c.overflow)throw Error(`${f} needs output.status.overflow without pairs`);if(c.pairs){let{pairs:e}=c;t(`${f} output.pairs`,e);for(let t of[e.leftIds,e.rightIds,e.status.count,e.status.overflow,e.status.candidateOverflow,e.status.requiredCount])t&&b(t,[`uint32`],`${f} output.pairs`);if(e.leftIds.length<1)throw Error(`${f} output.pairs needs at least one slot`)}if(c.weights){if(E(f,c.weights)!==this.leftCount)throw Error(`${f} output.weights needs one row per left feature`);if(c.weights.distances)throw Error(`${f} output.weights.distances is not supported`);if(c.pairs&&c.pairs.leftIds.length!==c.weights.neighbors.length)throw Error(`${f} output.weights capacity must equal pairs`)}if(this.pairCapacity=c.pairs?c.pairs.leftIds.length:c.weights.neighbors.length,c.relate){if(!c.pairs)throw Error(`${f} output.relate needs output.pairs`);if(b(c.relate,[`uint32`],`${f} output.relate`),c.relate.length!==this.pairCapacity)throw Error(`${f} output.relate needs one row per pair`)}}if(o(f,n.output.status??{}),c.onAttribute){let{left:e,right:t}=c.onAttribute;if(b(e,[`uint32`],`${f} onAttribute.left`),b(t,[`uint32`],`${f} onAttribute.right`),e.length!==this.leftCount||t.length!==this.rightCount)throw Error(`${f} onAttribute needs one key per feature`)}ae(f,this.getInputViews(),this.getOutputViews())}getInputViews(){return[...n(this.views.left),...n(this.views.right),...this.distanceView?[this.distanceView]:[],...this.patternView?[this.patternView]:[],...this.views.onAttribute?[this.views.onAttribute.left,this.views.onAttribute.right]:[]]}getOutputViews(){let{pairs:e,weights:t,overflow:n,candidateCount:r,uncertainCount:i,unmatched:a,relate:o}=this.views;return[e?.leftIds,e?.rightIds,e?.status.count,e?.status.overflow,e?.status.candidateOverflow,e?.status.requiredCount,o,t?.offsets,t?.neighbors,t?.weights,a?.ids,a?.status.count,a?.status.overflow,a?.status.requiredCount,n,r,i,this.views.antimeridianUncertainCount,this.views.poleUncertainCount]}getCapacityPlan(e={}){let{views:t}=this;return ze({leftCount:this.leftCount,rightCount:this.rightCount,leftKind:t.left.kind,rightKind:t.right.kind,relate:this.usesRelateEngine,leafCapacity:this.leafCapacity,observedCandidateCount:e.candidateCount,observedRequiredCount:e.requiredCount})}getCommandNodes(e){let{id:t,views:n,leftCount:i,rightCount:a,candidateCapacity:o,pairCapacity:l,how:u}=this,{left:d,right:f,predicate:p,pairs:h,weights:g,prepared:_,unmatched:v,relate:b}=n;if(m(t,e,[...this.getInputViews(),...this.getOutputViews()]),_&&!_.isDeclaredIn(e))throw Error(`${t} requires prepared to be added to the graph first`);let C=[];if(this.spatialContext.coordinateSpace===`longitude-latitude`&&ne(n)){for(let r of[`antimeridianUncertainCount`,`poleUncertainCount`]){let i=n[r];i&&C.push(ee(e,{id:`${t}-clear-${r}`,operation:$,view:i,type:`u32`,value:`0u`,componentCount:1}))}for(let[r,i]of[d,f].filter(e=>e.kind===`polygons`).entries())C.push(te(e,{id:`${t}-lon-lat-polygon-check-${r}`,operation:$,positions:i.positions,ringOffsets:i.ringOffsets,status:{antimeridianUncertainCount:n.antimeridianUncertainCount,poleUncertainCount:n.poleUncertainCount}}))}let T=u===`anti`,E=[{name:`left`,geometry:d,count:i,isPrepared:!1},{name:`right`,geometry:f,count:a,isPrepared:!!_}].map(n=>{let i,a,o;return n.isPrepared?(i=_?.storage.featureRings,{...n,minima:a,maxima:o,featureRings:i}):(a=y(e,`${t}-${n.name}-minima`,`float32x2`,n.count),o=y(e,`${t}-${n.name}-maxima`,`float32x2`,n.count),n.geometry.kind===`polygons`&&(i=y(e,`${t}-${n.name}-feature-rings`,`uint32x2`,n.count),C.push(r(e,`${t}-${n.name}-rings`,{featureCount:n.count,geometry:n.geometry,featureRings:i}))),C.push(s(e,`${t}-${n.name}-bounds`,{featureCount:n.count,geometry:n.geometry,featureRings:i,minima:a,maxima:o})),{...n,minima:a,maxima:o,featureRings:i})}),[O,ae]=E,A,j;if(_)A={nodeMinima:_.storage.nodeMinima,nodeMaxima:_.storage.nodeMaxima,leafIds:_.storage.leafIds,internalNodeCount:_.bvhInternalNodeCount,spatialSort:_.spatialSort},j=_.storage.overflow;else{let{bvh:n,nodes:r}=ie(e,t,ae.minima,ae.maxima,this.leafCapacity);C.push(...r),A={nodeMinima:n.nodeMinima,nodeMaxima:n.nodeMaxima,leafIds:n.leafIds,internalNodeCount:n.internalNodeCount},j=n.overflow}let M=y(e,`${t}-state`,`uint32`,4),N=y(e,`${t}-candidate-pairs`,`uint32x2`,o+(this.distanceView?1:0)),P=y(e,`${t}-flags`,`uint32`,o),F=T?void 0:y(e,`${t}-flag-offsets`,`uint32`,o),I=T?void 0:h?.leftIds??y(e,`${t}-left-ids`,`uint32`,l),L=T?void 0:h?.rightIds??y(e,`${t}-right-ids`,`uint32`,l),R=g||T?y(e,`${t}-left-match-counts`,`uint32`,i+1):void 0,z=this.usesRelateEngine?y(e,`${t}-matrices`,`uint32`,o):void 0,se=p!==`dwithin`&&(d.kind===`points`&&f.kind===`polygons`||d.kind===`polygons`&&f.kind===`points`),B=2*o+1,ce=d.kind===`polygons`?d:f.kind===`polygons`?f:null,V=se?{pairPoints:y(e,`${t}-pair-points`,`float32x2`,B),pairGeometryOffsets:y(e,`${t}-pair-geometry-offsets`,`uint32`,B+1),pairClassifications:y(e,`${t}-pair-classifications`,`uint32`,B)}:void 0;{let n=[{name:`state`,view:M,type:`u32`,access:`read_write`},{name:`candidatePairs`,view:N,type:`u32`,access:`read_write`},{name:`flags`,view:P,type:`u32`,access:`read_write`}];this.distanceView&&n.push({name:`distanceRow`,view:this.distanceView,type:`f32`,access:`read`}),R&&n.push({name:`leftMatchCounts`,view:R,type:`u32`,access:`read_write`}),V&&n.push({name:`pairPoints`,view:V.pairPoints,type:`f32`,access:`read_write`},{name:`pairGeometryOffsets`,view:V.pairGeometryOffsets,type:`u32`,access:`read_write`});let r=[4,o,R?.length??0,V?.pairGeometryOffsets.length??0];C.push(w(e,{id:`${t}-clear`,operation:$,variant:`clear`,bindings:n,invocationCount:Math.max(...r),declarations:k,body:`if (index < 4u) { state[stateOffset + index] = 0u; }
  if (index < ${o}u) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
    flags[flagsOffset + index] = 0u;
  }
  ${this.distanceView?`if (index == 0u) { candidatePairs[candidatePairsOffset + ${o*2}u] = bitcast<u32>(distanceRow[distanceRowOffset]); }`:``}
  ${R?`if (index <= ${i}u) { leftMatchCounts[leftMatchCountsOffset + index] = 0u; }`:``}
  ${V?`if (index < ${V.pairPoints.length}u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
    pairPoints[pairPointsOffset + index * 2u] = quietNaN;
    pairPoints[pairPointsOffset + index * 2u + 1u] = quietNaN;
  }
  if (index < ${V.pairGeometryOffsets.length}u) {
    pairGeometryOffsets[pairGeometryOffsetsOffset + index] =
      select(0u, ${ce.polygonOffsets.length-1}u, index == ${V.pairGeometryOffsets.length-1}u);
  }`:``}`}))}C.push(...oe(e,{id:t,leftCount:i,rightCount:a,leftMinima:O.minima,leftMaxima:O.maxima,tree:A,candidateCapacity:o,margin:p===`dwithin`?this.distanceView??n.distance:0,sphereRadius:this.sphereRadius,state:M,candidatePairs:N}));let H=S(e,{id:`${t}-candidate-dispatch`,operation:$,count:M,maximumItemCount:o});C.push(H.updateNode);let U=new Map,ue=n=>{let r=U.get(n);return r||(r=S(e,{id:`${t}-candidate-dispatch-${n}`,operation:$,count:M,maximumItemCount:o,workgroupSize:n,invocationsPerItem:n}),U.set(n,r),C.push(r.updateNode)),r};n.onAttribute&&C.push(w(e,{id:`${t}-on-attribute`,operation:$,variant:`on-attribute`,bindings:[{name:`candidatePairs`,view:N,type:`u32`,access:`read_write`},{name:`leftKeys`,view:n.onAttribute.left,type:`u32`,access:`read`},{name:`rightKeys`,view:n.onAttribute.right,type:`u32`,access:`read`}],invocationCount:o,condition:H.condition,declarations:k,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (leftKeys[leftKeysOffset + left] != rightKeys[rightKeysOffset + right]) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
  }`}));let W=n.excludeSameRow?` && left != right`:``,de=this.sphereRadius===void 0?{kind:`planar`}:{kind:`great-circle`,radius:this.sphereRadius},G=E.map(e=>({prefix:e.name,kind:e.geometry.kind,vertexCount:e.geometry.positions.length})),K=[],q=this.distanceView?{expression:`bitcast<f32>(candidatePairs[candidatePairsOffset + ${o*2}u])`}:n.distance??0;for(let e of E){let t=e.name;K.push({name:`${t}Positions`,view:e.geometry.positions,type:`f32`,access:`read`}),e.geometry.kind===`lines`?K.push({name:`${t}RingOffsets`,view:e.geometry.lineOffsets,type:`u32`,access:`read`}):e.geometry.kind===`polygons`&&K.push({name:`${t}FeatureRings`,view:e.featureRings,type:`u32`,access:`read`},{name:`${t}RingOffsets`,view:e.geometry.ringOffsets,type:`u32`,access:`read`})}if(V){let n=d.kind===`points`,r=n?d:f,i=n?f:d;C.push(w(e,{id:`${t}-expand`,operation:$,variant:`expand`,bindings:[{name:`candidatePairs`,view:N,type:`u32`,access:`read`},{name:`pointPositions`,view:r.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:i.featureOffsets,type:`u32`,access:`read`},{name:`pairPoints`,view:V.pairPoints,type:`f32`,access:`read_write`},{name:`pairGeometryOffsets`,view:V.pairGeometryOffsets,type:`u32`,access:`read_write`}],invocationCount:o,condition:H.condition,declarations:k,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let pointRow = ${n?`left`:`right`};
  let polygonRow = ${n?`right`:`left`};
  let row = index * 2u + 1u;
  pairPoints[pairPointsOffset + row * 2u] = pointPositions[pointPositionsOffset + pointRow * 2u];
  pairPoints[pairPointsOffset + row * 2u + 1u] = pointPositions[pointPositionsOffset + pointRow * 2u + 1u];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row] = featureOffsets[featureOffsetsOffset + polygonRow];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row + 1u] = featureOffsets[featureOffsetsOffset + polygonRow + 1u];`}));let a=S(e,{id:`${t}-robust-dispatch`,operation:$,count:M,maximumItemCount:o,invocationsPerItem:2,additionalInvocationCount:1});C.push(a.updateNode),C.push(...new re({id:`${t}-classify`,queryPositions:V.pairPoints,positions:i.positions,featureOffsets:V.pairGeometryOffsets,polygonOffsets:i.polygonOffsets,ringOffsets:i.ringOffsets,output:V.pairClassifications}).getCommandNodes(e).map(e=>(c(e.type===`compute`),{...e,condition:a.condition})))}if(z)this.addRelateNodes(e,C,{sides:E,sideSpecs:G,sideBindings:K,candidatePairs:N,matrices:z,flags:P,state:M,robust:V,sameRowTest:W,candidateCondition:H.condition,cooperativeCondition:V?void 0:ue(256).condition});else if(V){let n=d.kind===`points`,r=p===`contains`&&n||p===`within`&&!n,i=p===`intersects`?`classification == INSIDE || classification == BOUNDARY`:`classification == INSIDE`;C.push(w(e,{id:`${t}-resolve`,operation:$,variant:`resolve`,bindings:[{name:`candidatePairs`,view:N,type:`u32`,access:`read`},{name:`pairClassifications`,view:V.pairClassifications,type:`u32`,access:`read`},{name:`flags`,view:P,type:`u32`,access:`read_write`},{name:`state`,view:M,type:`atomic<u32>`,access:`read_write`}],invocationCount:o,condition:H.condition,declarations:`${k}
const INSIDE: u32 = ${D.inside}u;
const BOUNDARY: u32 = ${D.boundary}u;
const UNCERTAIN: u32 = ${D.uncertain}u;`,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    if (classification == UNCERTAIN) {
      atomicAdd(&state[stateOffset + 1u], 1u);
    } else {
      matched = ${r?`false`:`(${i})`}${W};
    }
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`})),C.push(w(e,{id:`${t}-fallback`,operation:$,variant:`fallback-${d.kind}-${f.kind}-${p}`,bindings:[...K,{name:`candidatePairs`,view:N,type:`u32`,access:`read`},{name:`pairClassifications`,view:V.pairClassifications,type:`u32`,access:`read`},{name:`flags`,view:P,type:`u32`,access:`read_write`}],invocationCount:o,condition:H.condition,declarations:`${le(G[0],G[1],p,0)}
const UNCERTAIN: u32 = ${D.uncertain}u;`,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  flags[flagsOffset + index] = select(0u, 1u, pairMatches(left, right)${W});`}))}else{let n=!this.usesWorkgroupDistance&&K.length+3<=We;C.push(w(e,{id:`${t}-exact`,operation:$,variant:`exact-${d.kind}-${f.kind}-${p}${this.sphereRadius===void 0?``:`-great-circle`}`,bindings:[...K,{name:`candidatePairs`,view:N,type:`u32`,access:`read`},{name:`flags`,view:P,type:`u32`,access:`read_write`},...n?[{name:`state`,view:M,type:`atomic<u32>`,access:`read_write`}]:[]],...this.usesWorkgroupDistance?{invocationCount:o*64,workgroupSize:64,guardIndex:!1,condition:ue(64).condition,declarations:he(G[0],G[1],q),body:`let slot = index / 64u;
  let left = candidatePairs[candidatePairsOffset + slot * 2u];
  let right = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  let matched = dwithinWorkgroupPair(left, right, left != NO_FEATURE, localInvocationIndex);
  if (localInvocationIndex == 0u) {
    flags[flagsOffset + slot] = select(0u, 1u, matched${W});
  }`}:{invocationCount:o,condition:H.condition,declarations:le(G[0],G[1],p,q,de),body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    matched = pairMatches(left, right)${W};
    ${n?`if (uncertainOrientation) { atomicAdd(&state[stateOffset + 1u], 1u); }`:``}
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`}}))}if(T&&v&&R)this.addAntiNodes(e,C,{candidatePairs:N,flags:P,leftMatchCounts:R,unmatched:v,state:M,candidateCondition:H.condition});else if(I&&L&&F){C.push(...new x({id:`${t}-scan-flags`,input:P,output:F,mode:`exclusive`}).getCommandNodes(e)),C.push(w(e,{id:`${t}-match-total`,operation:$,variant:`match-total`,bindings:[{name:`state`,view:M,type:`u32`,access:`read_write`},{name:`flags`,view:P,type:`u32`,access:`read`},{name:`flagOffsets`,view:F,type:`u32`,access:`read`}],invocationCount:1,declarations:`const CANDIDATE_CAPACITY: u32 = ${o}u;`,body:`state[stateOffset + 2u] =
    flagOffsets[flagOffsetsOffset + CANDIDATE_CAPACITY - 1u] +
    flags[flagsOffset + CANDIDATE_CAPACITY - 1u];`}));let n=[{name:`candidatePairs`,view:N,type:`u32`,access:`read`},{name:`flags`,view:P,type:`u32`,access:`read`},{name:`flagOffsets`,view:F,type:`u32`,access:`read`},{name:`leftIds`,view:I,type:`u32`,access:`read_write`},{name:`rightIds`,view:L,type:`u32`,access:`read_write`}];b&&z&&n.push({name:`matrices`,view:z,type:`u32`,access:`read`},{name:`relateOut`,view:b,type:`u32`,access:`read_write`}),C.push(w(e,{id:`${t}-scatter`,operation:$,variant:b?`scatter-relate`:`scatter`,bindings:n,invocationCount:o,condition:H.condition,declarations:`const PAIR_CAPACITY: u32 = ${l}u;`,body:`if (flags[flagsOffset + index] != 0u) {
    let slot = flagOffsets[flagOffsetsOffset + index];
    if (slot < PAIR_CAPACITY) {
      leftIds[leftIdsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u];
      rightIds[rightIdsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u + 1u];
      ${b&&z?`relateOut[relateOutOffset + slot] = matrices[matricesOffset + index] & ${Ue}u;`:``}
    }
  }`}))}{let r=[[`overflow`,n.overflow],[`pairsOverflow`,h?.status.overflow],[`pairsCandidateOverflow`,h?.status.candidateOverflow],[`pairsCount`,h?.status.count],[`pairsTotal`,h?.status.requiredCount],[`unmatchedOverflow`,v?.status.overflow],[`unmatchedCount`,v?.status.count],[`unmatchedTotal`,v?.status.requiredCount],[`candidateCount`,n.candidateCount],[`uncertainCount`,n.uncertainCount]].filter(e=>!!e[1]);for(let n=0;n===0||n<r.length;n+=6){let i=r.slice(n,n+6),a=[{name:`state`,view:M,type:`u32`,access:`read`},{name:`bvhOverflow`,view:j,type:`u32`,access:`read`},...i.map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`}))],s=e=>i.some(([t])=>t===e);C.push(w(e,{id:n===0?`${t}-finalize`:`${t}-finalize-${n/6}`,operation:$,variant:`finalize`,bindings:a,invocationCount:1,declarations:`const CANDIDATE_CAPACITY: u32 = ${o}u;
const PAIR_CAPACITY: u32 = ${l}u;
const UNMATCHED_CAPACITY: u32 = ${v?.ids.length??0}u;`,body:`let candidateTotal = state[stateOffset];
  let matchTotal = ${T?`0u`:`state[stateOffset + 2u]`};
  let unmatchedTotalValue = ${T?`state[stateOffset + 3u]`:`0u`};
  let candidateIncomplete = bvhOverflow[bvhOverflowOffset] != 0u || candidateTotal > CANDIDATE_CAPACITY;
  let pairIncomplete = matchTotal > PAIR_CAPACITY;
  let unmatchedIncomplete = unmatchedTotalValue > UNMATCHED_CAPACITY;
  let overflowValue = select(0u, 1u, candidateIncomplete || pairIncomplete || unmatchedIncomplete);
  ${s(`overflow`)?`overflow[overflowOffset] = overflowValue;`:``}
  ${s(`pairsOverflow`)?`pairsOverflow[pairsOverflowOffset] = select(0u, 1u, pairIncomplete);`:``}
  ${s(`pairsCandidateOverflow`)?`pairsCandidateOverflow[pairsCandidateOverflowOffset] = select(0u, 1u, candidateIncomplete);`:``}
  ${s(`pairsCount`)?`pairsCount[pairsCountOffset] = min(matchTotal, PAIR_CAPACITY);`:``}
  ${s(`pairsTotal`)?`pairsTotal[pairsTotalOffset] = matchTotal;`:``}
  ${s(`unmatchedOverflow`)?`unmatchedOverflow[unmatchedOverflowOffset] = overflowValue;`:``}
  ${s(`unmatchedCount`)?`unmatchedCount[unmatchedCountOffset] = min(unmatchedTotalValue, UNMATCHED_CAPACITY);`:``}
  ${s(`unmatchedTotal`)?`unmatchedTotal[unmatchedTotalOffset] = unmatchedTotalValue;`:``}
  ${s(`candidateCount`)?`candidateCount[candidateCountOffset] = candidateTotal;`:``}
  ${s(`uncertainCount`)?`uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];`:``}`}))}}if(g&&R&&I&&L){let n=S(e,{id:`${t}-pair-dispatch`,operation:$,count:M,countRow:2,maximumItemCount:l});C.push(n.updateNode),C.push(w(e,{id:`${t}-count-matches`,operation:$,variant:`count-matches`,bindings:[{name:`leftIds`,view:I,type:`u32`,access:`read`},{name:`state`,view:M,type:`u32`,access:`read`},{name:`leftMatchCounts`,view:R,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,condition:n.condition,declarations:`const PAIR_CAPACITY: u32 = ${l}u;`,body:`if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  atomicAdd(&leftMatchCounts[leftMatchCountsOffset + leftIds[leftIdsOffset + index]], 1u);`})),C.push(...new x({id:`${t}-scan-matches`,input:R,output:g.offsets,mode:`exclusive`}).getCommandNodes(e)),C.push(w(e,{id:`${t}-fill-weights`,operation:$,variant:`fill-weights`,bindings:[{name:`rightIds`,view:L,type:`u32`,access:`read`},{name:`state`,view:M,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read_write`},{name:`weightValues`,view:g.weights,type:`f32`,access:`read_write`}],invocationCount:l,condition:n.condition,declarations:`const PAIR_CAPACITY: u32 = ${l}u;`,body:`if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  neighbors[neighborsOffset + index] = rightIds[rightIdsOffset + index];
  weightValues[weightValuesOffset + index] = 1.0;`}))}return C}addRelateNodes(e,t,n){let{id:r,views:i,candidateCapacity:a}=this,{left:o,right:s}=i,{sideSpecs:c,sideBindings:l,candidatePairs:u,matrices:d,flags:f,state:p,robust:m}=n,h=`${V}u`,g=D;if(m){let[i,f,p]=o.kind===`points`?[`0FFFFF212`,`F0FFFF212`,`FF0FFF212`]:[`0F2FF1FF2`,`FF20F1FF2`,`FF2FF10F2`];t.push(w(e,{id:`${r}-resolve-relate`,operation:$,variant:`resolve-relate`,bindings:[{name:`candidatePairs`,view:u,type:`u32`,access:`read`},{name:`pairClassifications`,view:m.pairClassifications,type:`u32`,access:`read`},{name:`matrices`,view:d,type:`u32`,access:`read_write`}],invocationCount:a,condition:n.candidateCondition,declarations:`${k}
const INSIDE: u32 = ${g.inside}u;
const BOUNDARY: u32 = ${g.boundary}u;
const UNCERTAIN: u32 = ${g.uncertain}u;`,body:`var matrix = 0u;
  if (candidatePairs[candidatePairsOffset + index * 2u] != NO_FEATURE) {
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    matrix = select(select(select(${U(p)}u, ${U(f)}u, classification == BOUNDARY), ${U(i)}u, classification == INSIDE), ${h}, classification == UNCERTAIN);
  }
  matrices[matricesOffset + index] = matrix;`})),t.push(w(e,{id:`${r}-fallback-relate`,operation:$,variant:`fallback-relate-${o.kind}-${s.kind}`,bindings:[...l,{name:`candidatePairs`,view:u,type:`u32`,access:`read`},{name:`pairClassifications`,view:m.pairClassifications,type:`u32`,access:`read`},{name:`matrices`,view:d,type:`u32`,access:`read_write`}],invocationCount:a,condition:n.candidateCondition,declarations:`${Ee(c[0],c[1])}
const UNCERTAIN: u32 = ${g.uncertain}u;`,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  matrices[matricesOffset + index] = pairRelate(left, right) | ${h};`}))}else t.push(w(e,{id:`${r}-relate`,operation:$,variant:`relate-${o.kind}-${s.kind}`,bindings:[...l,{name:`candidatePairs`,view:u,type:`u32`,access:`read`},{name:`matrices`,view:d,type:`u32`,access:`read_write`}],invocationCount:a*256,workgroupSize:256,guardIndex:!1,condition:n.cooperativeCondition,declarations:`${Ee(c[0],c[1],!0)}
${De(c)}`,body:`let slot = index / 256u;
  let left = candidatePairs[candidatePairsOffset + slot * 2u];
  let right = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  let matrix = relateWorkgroupPair(left, right, left != NO_FEATURE, localInvocationIndex);
  if (localInvocationIndex == 0u) {
    matrices[matricesOffset + slot] = matrix;
  }`}));t.push(w(e,{id:`${r}-mask`,operation:$,variant:`mask`,record:this.props.predicate===`intersects`?{formula:`gpu-spatial-analysis.spatial-predicate-join`}:void 0,bindings:[{name:`candidatePairs`,view:u,type:`u32`,access:`read`},{name:`matrices`,view:d,type:`u32`,access:`read`},{name:`flags`,view:f,type:`u32`,access:`read_write`},{name:`state`,view:p,type:`atomic<u32>`,access:`read_write`},...this.patternView?[{name:`patternWords`,view:this.patternView,type:`u32`,access:`read`}]:[]],invocationCount:a,condition:n.candidateCondition,declarations:`${k}
${this.patternView?me(this.patternView.length/2):pe(this.relatePatterns)}`,body:`let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let raw = matrices[matricesOffset + index];
    if ((raw & ${h}) != 0u) { atomicAdd(&state[stateOffset + 1u], 1u); }
    let matrix = raw & 0x3ffffu;
    // An all-F matrix marks an empty or invalid feature.
    matched = matrix != 0u && relateMatches(matrix)${n.sameRowTest};
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`}))}addAntiNodes(e,t,n){let{id:r,leftCount:i,candidateCapacity:a}=this,{candidatePairs:o,flags:s,leftMatchCounts:c,unmatched:l,state:u}=n,d=y(e,`${r}-unmatched-flags`,`uint32`,i),f=y(e,`${r}-unmatched-offsets`,`uint32`,i);t.push(w(e,{id:`${r}-count-left-matches`,operation:$,variant:`count-left-matches`,bindings:[{name:`candidatePairs`,view:o,type:`u32`,access:`read`},{name:`flags`,view:s,type:`u32`,access:`read`},{name:`leftMatchCounts`,view:c,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,condition:n.candidateCondition,body:`if (flags[flagsOffset + index] != 0u) {
    atomicAdd(&leftMatchCounts[leftMatchCountsOffset + candidatePairs[candidatePairsOffset + index * 2u]], 1u);
  }`})),t.push(w(e,{id:`${r}-unmatched-flags`,operation:$,variant:`unmatched-flags`,bindings:[{name:`leftMatchCounts`,view:c,type:`u32`,access:`read`},{name:`unmatchedFlags`,view:d,type:`u32`,access:`read_write`}],invocationCount:i,body:`unmatchedFlags[unmatchedFlagsOffset + index] = select(0u, 1u, leftMatchCounts[leftMatchCountsOffset + index] == 0u);`})),t.push(...new x({id:`${r}-scan-unmatched`,input:d,output:f,mode:`exclusive`}).getCommandNodes(e)),t.push(w(e,{id:`${r}-scatter-unmatched`,operation:$,variant:`scatter-unmatched`,bindings:[{name:`unmatchedFlags`,view:d,type:`u32`,access:`read`},{name:`unmatchedOffsets`,view:f,type:`u32`,access:`read`},{name:`ids`,view:l.ids,type:`u32`,access:`read_write`},{name:`state`,view:u,type:`u32`,access:`read_write`}],invocationCount:i,declarations:`const UNMATCHED_CAPACITY: u32 = ${l.ids.length}u;`,body:`if (unmatchedFlags[unmatchedFlagsOffset + index] != 0u) {
    let slot = unmatchedOffsets[unmatchedOffsetsOffset + index];
    if (slot < UNMATCHED_CAPACITY) { ids[idsOffset + slot] = index; }
  }
  if (index == ${i-1}u) {
    state[stateOffset + 3u] = unmatchedOffsets[unmatchedOffsetsOffset + index] + unmatchedFlags[unmatchedFlagsOffset + index];
  }`}))}};export{ue as n,W as r,$e as t};