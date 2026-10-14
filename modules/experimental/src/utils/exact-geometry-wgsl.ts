// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact point-in-ring and segment-crossing helpers in WGSL, shared by every kernel that decides a
 * topological classification (overlay membership, ring hole assignment, cell cover containment,
 * zone crossings, make-valid face parity, line-in-polygon pieces).
 *
 * Every decision is an exact sign from `orientSign` (`EXACT_ORIENTATION_WGSL`, which the kernel
 * must declare once). f32 values only place points. Ties are broken by Simulation of Simplicity
 * (SoS), so the parity of a ray test and the crossings of a segment test agree by construction:
 *
 * - A query point `P = (x, y)` is read as `P' = (x - e^2, y - e)` for an infinitesimal `e > 0`;
 *   polygon vertices are not perturbed.
 * - Side of `P'` against edge `c -> d`: `orient(c, d, P)` when that is not zero. When `P` is on the
 *   edge's line the sign is that of `cross(d - c, (-e^2, -e))`: `-sign(dx)` for a non-vertical edge
 *   and `sign(dy)` for a vertical one.
 * - Ray from `P'` toward +x against edge `c -> d`: the edge straddles when `c.y >= y` differs from
 *   `d.y >= y`, and it crosses when `P'` is left of an upward edge (+1) or right of a downward edge
 *   (-1). The signed sum over a ring is its winding number about `P'`; its parity is the even-odd
 *   inside test.
 * - Segment `p'q'` against edge `c -> d` crosses when `p'` and `q'` are on opposite sides of
 *   `c -> d` and `c`, `d` are on opposite sides of `p'q'`, the latter tie broken by
 *   `sign(dx)` of the segment (`-sign(dy)` when vertical).
 *
 * A second probe kind classifies the point infinitesimally beside the midpoint of a segment
 * `s -> t` (left side `side = 1`, right side `side = -1`), which is how overlay and ring assembly
 * label a boundary piece without rounding a probe point off the boundary. Its side against an
 * edge comes from the exact signs of both segment endpoints: a segment collinear with the edge
 * takes the side from the two directions, and only a segment that crosses the edge's line (a
 * noding miss) falls back to the rounded midpoint.
 *
 * One call site rule: `orientSign` is a large function (a 256-bit integer fallback), and shader
 * compilers inline it at every call. Everything here funnels through `exactOrientSigns`, which
 * holds the only call, inside a loop. A kernel that uses these helpers must reach
 * `exactOrientSigns` from a single call site (call `exactProbeCrossing` or `exactSegmentsCross`
 * from one place, typically inside its edge loop), and must not call `orientSign` elsewhere.
 *
 * Sign code 2 from `orientSign` (non-finite input, or terms spanning more than 200 binary orders)
 * falls back to the f32 cross-product sign and raises `EXACT_FLAG_UNCERTAIN`. Ray crossings
 * reject non-finite edges before orientation, contributing zero and raising the same flag.
 *
 * Sampling code that only places a point (label candidates, dot rejection) may keep an f32
 * crossing test, with a comment saying so.
 *
 * @internal
 */
export const EXACT_GEOMETRY_WGSL = /* wgsl */ `
const EXACT_FLAG_BOUNDARY: u32 = 1u;
const EXACT_FLAG_UNCERTAIN: u32 = 2u;
const EXACT_PROBE_POINT: i32 = 0;

// Probe for the ray tests. side 0: the SoS-perturbed point 'point'. side 1 or -1: the point
// infinitesimally left (1) or right (-1) of the midpoint 'point' of segment start -> end.
struct ExactProbe {
  point: vec2f,
  start: vec2f,
  end: vec2f,
  side: i32,
}

struct ExactSigns {
  signs: vec4i,
  uncertain: bool,
}

// Exact signs of orient(starts[k], ends[k], points[k]) for k < count (at most 4). The only call
// site of orientSign for kernels that use these helpers. An uncertain sign (code 2) becomes the
// f32 cross-product sign and sets 'uncertain'.
fn exactOrientSigns(
  count: u32,
  starts: array<vec2f, 4>,
  ends: array<vec2f, 4>,
  points: array<vec2f, 4>
) -> ExactSigns {
  var startValues = starts;
  var endValues = ends;
  var pointValues = points;
  var result = ExactSigns(vec4i(0), false);
  for (var k = 0u; k < min(count, 4u); k++) {
    let a = startValues[k];
    let b = endValues[k];
    let c = pointValues[k];
    var value = orientSign(a, b, c);
    if (value == 2) {
      result.uncertain = true;
      let direction = b - a;
      let offset = c - a;
      value = i32(sign(direction.x * offset.y - direction.y * offset.x));
    }
    result.signs[k] = value;
  }
  return result;
}

fn exactCompareSign(value: f32, reference: f32) -> i32 {
  return select(select(0, -1, value < reference), 1, value > reference);
}

// Side (-1 right, 1 left) of the SoS-perturbed point P' against edge c -> d, given the exact
// sign of orient(c, d, P). Never zero for an edge of positive length.
fn sosPointSide(sign: i32, c: vec2f, d: vec2f) -> i32 {
  if (sign != 0) { return sign; }
  if (d.x != c.x) { return -exactCompareSign(d.x, c.x); }
  return exactCompareSign(d.y, c.y);
}

// Side (-1 right, 1 left) of vertex c against the SoS-perturbed segment p' -> q', given the
// exact sign of orient(p, q, c). Zero only for p == q.
fn sosSegmentSide(sign: i32, p: vec2f, q: vec2f) -> i32 {
  if (sign != 0) { return sign; }
  if (q.x != p.x) { return exactCompareSign(q.x, p.x); }
  return -exactCompareSign(q.y, p.y);
}

// A side probe of a zero-length segment has no side: it is read as the point probe at 'point', so
// both of its sides classify alike (a zero-length boundary piece never separates two regions).
fn isExactPointProbe(probe: ExactProbe) -> bool {
  return probe.side == EXACT_PROBE_POINT || all(probe.start == probe.end);
}

// Whether vertex v is above the probe's horizontal line (the ray's straddle test).
fn exactProbeAbove(probe: ExactProbe, v: vec2f) -> bool {
  if (v.y != probe.point.y) { return v.y > probe.point.y; }
  if (isExactPointProbe(probe)) { return true; }
  // The side probe sits at y + delta * side * dx - e (delta >> e > 0): v is above when that
  // offset is negative or zero.
  return probe.side * exactCompareSign(probe.end.x, probe.start.x) <= 0;
}

// Contribution of edge c -> d to the probe's ray toward +x: x is the signed crossing (+1 upward,
// -1 downward, 0 none), y the flags (EXACT_FLAG_BOUNDARY when a point probe lies exactly on the
// closed edge, EXACT_FLAG_UNCERTAIN).
fn exactProbeCrossing(probe: ExactProbe, c: vec2f, d: vec2f) -> vec2i {
  // A non-finite edge has no geometric crossing. Preserve the diagnostic instead of
  // converting a NaN determinant to an implementation-dependent integer sign.
  if (!isFiniteBits(c.x) || !isFiniteBits(c.y) || !isFiniteBits(d.x) || !isFiniteBits(d.y)) {
    return vec2i(0, i32(EXACT_FLAG_UNCERTAIN));
  }
  let p = probe.point;
  let straddles = exactProbeAbove(probe, c) != exactProbeAbove(probe, d);
  let isPoint = isExactPointProbe(probe);
  let inBox = isPoint && p.x >= min(c.x, d.x) && p.x <= max(c.x, d.x) &&
    p.y >= min(c.y, d.y) && p.y <= max(c.y, d.y);
  if (!straddles && !inBox) { return vec2i(0, 0); }
  var count = 1u;
  var points = array<vec2f, 4>(p, p, p, p);
  if (!isPoint) {
    count = 3u;
    points = array<vec2f, 4>(probe.start, probe.end, p, p);
  }
  let exact = exactOrientSigns(count, array<vec2f, 4>(c, c, c, c), array<vec2f, 4>(d, d, d, d), points);
  var flags = select(0u, EXACT_FLAG_UNCERTAIN, exact.uncertain);
  var side = 0;
  if (isPoint) {
    if (inBox && exact.signs.x == 0) { flags = flags | EXACT_FLAG_BOUNDARY; }
    side = sosPointSide(exact.signs.x, c, d);
  } else {
    let startSign = exact.signs.x;
    let endSign = exact.signs.y;
    if (startSign == 0 && endSign == 0) {
      // Collinear: the probe is on its own side of the edge when both run the same way.
      var same = exactCompareSign(probe.end.x, probe.start.x) * exactCompareSign(d.x, c.x);
      if (probe.end.x == probe.start.x) {
        same = exactCompareSign(probe.end.y, probe.start.y) * exactCompareSign(d.y, c.y);
      }
      side = probe.side * select(-1, 1, same > 0);
    } else if (startSign >= 0 && endSign >= 0) {
      side = 1;
    } else if (startSign <= 0 && endSign <= 0) {
      side = -1;
    } else {
      // The segment crosses the edge's line (a noding miss): decide at the rounded midpoint, then
      // by the probe offset (sign of the dot product of the two directions), then by SoS.
      side = exact.signs.z;
      if (side == 0) {
        let edge = d - c;
        let segment = probe.end - probe.start;
        side = probe.side * i32(sign(dot(edge, segment)));
      }
      side = sosPointSide(side, c, d);
    }
  }
  if (!straddles) { return vec2i(0, i32(flags)); }
  let upward = d.y > c.y;
  var crossing = 0;
  if (upward && side > 0) { crossing = 1; }
  if (!upward && side < 0) { crossing = -1; }
  return vec2i(crossing, i32(flags));
}

// Whether the SoS-perturbed segment p' -> q' crosses edge c -> d. x is 1 for a crossing, y the
// flags (EXACT_FLAG_UNCERTAIN). A zero-length segment never crosses; collinear overlaps do not.
fn exactSegmentsCross(p: vec2f, q: vec2f, c: vec2f, d: vec2f) -> vec2i {
  let exact = exactOrientSigns(
    4u,
    array<vec2f, 4>(c, c, p, p),
    array<vec2f, 4>(d, d, q, q),
    array<vec2f, 4>(p, q, c, d)
  );
  let flags = select(0, i32(EXACT_FLAG_UNCERTAIN), exact.uncertain);
  if (all(p == q)) { return vec2i(0, flags); }
  let separatesTrack = sosPointSide(exact.signs.x, c, d) != sosPointSide(exact.signs.y, c, d);
  let separatesEdge = sosSegmentSide(exact.signs.z, p, q) != sosSegmentSide(exact.signs.w, p, q);
  return vec2i(select(0, 1, separatesTrack && separatesEdge), flags);
}
`;

/**
 * Generates `fn <name>(probe: ExactProbe, begin: u32, end: u32) -> vec2i`: the winding number
 * (x) and OR of flags (y) of the ring whose vertices are rows `[begin, end)`, read through the
 * kernel's `fn <readVertex>(row: u32) -> vec2f`. The ring is closed implicitly (a repeated closing
 * vertex adds a zero-length edge, which never crosses). Its parity `x & 1` is the even-odd inside
 * test. Requires `EXACT_ORIENTATION_WGSL` and {@link EXACT_GEOMETRY_WGSL}. Call it from one site.
 * @internal
 */
export function getExactRingWindingWGSL(name: string, readVertex: string): string {
  return /* wgsl */ `
fn ${name}(probe: ExactProbe, begin: u32, end: u32) -> vec2i {
  var winding = 0;
  var flags = 0;
  if (end < begin + 3u) { return vec2i(0, 0); }
  var previous = ${readVertex}(end - 1u);
  for (var row = begin; row < end; row++) {
    let current = ${readVertex}(row);
    let crossing = exactProbeCrossing(probe, previous, current);
    winding += crossing.x;
    flags = flags | crossing.y;
    previous = current;
  }
  return vec2i(winding, flags);
}
`;
}
