var e=`
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
`;function t(e,t){return`
fn ${e}(probe: ExactProbe, begin: u32, end: u32) -> vec2i {
  var winding = 0;
  var flags = 0;
  if (end < begin + 3u) { return vec2i(0, 0); }
  var previous = ${t}(end - 1u);
  for (var row = begin; row < end; row++) {
    let current = ${t}(row);
    let crossing = exactProbeCrossing(probe, previous, current);
    winding += crossing.x;
    flags = flags | crossing.y;
    previous = current;
  }
  return vec2i(winding, flags);
}
`}export{t as n,e as t};