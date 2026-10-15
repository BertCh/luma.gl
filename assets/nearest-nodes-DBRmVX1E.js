import{m as e,s as t,w as n}from"./resources-CwvhEOBe.js";import{l as r,s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as a}from"./wgsl-common-BNRVzgd8.js";import{t as o}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{a as s,n as c,p as l,u}from"./spatial-join-passes-C0ZwcuNs.js";import{n as d}from"./spatial-geodesic-wgsl-BI6DREOO.js";var f=`
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const NO_FEATURE: u32 = 0xffffffffu;
const NO_SEGMENT: u32 = 0xffffffffu;
${a}
`,p=`${f}
${o}
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
`,m=`${p}
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
`,h=`${p}
${d}
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
`;function g(e){return e===`points`?1:e===`polygons`?3:2}function _(e,t,n,r,i){let a=e.prefix,o=e.kind===`points`?`1u`:e.kind===`polygons`?`${a}n`:`${a}n - 1u`,s=e.kind===`points`?`${a}k`:e.kind===`polygons`?`(${a}k + 1u) % ${a}n`:`${a}k + 1u`;return`for (var ${a}r = ${a}RingStart(${t}); ${a}r < ${a}RingEnd(${t}); ${a}r++) {
    let ${a}vs = ${a}RingVertexStart(${a}r);
    let ${a}ve = ${a}RingVertexEnd(${a}r);
    if (${a}ve <= ${a}vs || ${a}ve - ${a}vs < ${g(e.kind)}u) { continue; }
    let ${a}n = ${a}ve - ${a}vs;
    for (var ${a}k = 0u; ${a}k < ${o}; ${a}k++) {
      let ${n} = ${a}Vertex(${a}vs + ${a}k);
      let ${r} = ${a}Vertex(${a}vs + ${s});
      if (!(isFiniteValue(${n}.x) && isFiniteValue(${n}.y) && isFiniteValue(${r}.x) && isFiniteValue(${r}.y))) { continue; }
      ${i}
    }
  }`}function v(e){let t=e.prefix,n=t.toUpperCase(),{kind:r}=e,i=`f`,a=`f + 1u`,o=`r`,s=`r + 1u`,c=`vec2f(${t}Positions[${t}PositionsOffset + i * 2u], ${t}Positions[${t}PositionsOffset + i * 2u + 1u])`;if(r===`segments`)o=`2u * r`,s=`2u * r + 2u`,c=`select(
    vec2f(${t}Ends[${t}EndsOffset + (i >> 1u) * 2u], ${t}Ends[${t}EndsOffset + (i >> 1u) * 2u + 1u]),
    vec2f(${t}Starts[${t}StartsOffset + (i >> 1u) * 2u], ${t}Starts[${t}StartsOffset + (i >> 1u) * 2u + 1u]),
    (i & 1u) == 0u)`;else if(r===`lines`)o=`${t}LineOffsets[${t}LineOffsetsOffset + r]`,s=`min(${t}LineOffsets[${t}LineOffsetsOffset + r + 1u], ${n}_VERTEX_COUNT)`;else if(r===`polygons`){i=`${t}FeatureRings[${t}FeatureRingsOffset + f * 2u]`,a=`${t}FeatureRings[${t}FeatureRingsOffset + f * 2u + 1u]`;let r=e.packedRingOffsetsBase===void 0?`${t}RingOffsets[${t}RingOffsetsOffset`:`${t}FeatureRings[${t}FeatureRingsOffset + ${e.packedRingOffsetsBase}u`;o=`${r} + r]`,s=`min(${r} + r + 1u], ${n}_VERTEX_COUNT)`}let l=r===`polygons`?`
fn ${t}Contains(q: vec2f, f: u32) -> bool {
  var inside = false;
  ${_(e,`f`,`a`,`b`,`if ((a.y > q.y) != (b.y > q.y)) {
      let side = orientSide(a, b, q);
      if ((b.y > a.y && side > 0) || (b.y < a.y && side < 0)) { inside = !inside; }
    }`)}
  return inside;
}`:``;return`
const ${n}_VERTEX_COUNT: u32 = ${e.vertexCount}u;
fn ${t}RingStart(f: u32) -> u32 { return ${i}; }
fn ${t}RingEnd(f: u32) -> u32 { return ${a}; }
fn ${t}RingVertexStart(r: u32) -> u32 { return ${o}; }
fn ${t}RingVertexEnd(r: u32) -> u32 { return ${s}; }
fn ${t}Vertex(i: u32) -> vec2f { return ${c}; }
// (x, y, valid): first finite vertex of the first usable ring; valid is 0 for an empty feature.
fn ${t}First(f: u32) -> vec3f {
  for (var r = ${t}RingStart(f); r < ${t}RingEnd(f); r++) {
    let vs = ${t}RingVertexStart(r);
    let ve = ${t}RingVertexEnd(r);
    if (ve > vs && ve - vs >= ${g(r)}u) {
      let v = ${t}Vertex(vs);
      return vec3f(v, select(0.0, 1.0, isFiniteValue(v.x) && isFiniteValue(v.y)));
    }
  }
  return vec3f(0.0);
}${l}
`}function ee(e,t){let n=_(e,`rowQ`,`qa`,`qb`,_(t,`rowF`,`ra`,`rb`,`let distanceSq = segmentPairDistanceSq(qa, qb, ra, rb);
      if (distanceSq < result.distanceSq) {
        result.distanceSq = distanceSq;
        result.foot = segmentFoot(qa, qb, ra, rb);
        result.queryFoot = segmentFoot(ra, rb, qa, qb);
        result.segment = ${t.kind===`points`?`NO_SEGMENT`:t.kind===`segments`?`0u`:`rvs + rk`};
        if (distanceSq == 0.0) { return result; }
      }`)),r=e.kind===`polygons`?`
  if (result.distanceSq > 0.0) {
    let first = rFirst(rowF);
    if (first.z > 0.0 && qContains(first.xy, rowQ)) {
      return PairResult(0.0, first.xy, NO_SEGMENT, first.xy);
    }
  }`:``;return`
struct PairResult { distanceSq: f32, foot: vec2f, segment: u32, queryFoot: vec2f }
fn pairResult(rowQ: u32, rowF: u32) -> PairResult {
  var result = PairResult(FLOAT32_MAXIMUM, vec2f(0.0), NO_SEGMENT, vec2f(0.0));
  ${n}${t.kind===`polygons`?`
  if (result.distanceSq > 0.0) {
    let first = qFirst(rowQ);
    if (first.z > 0.0 && rContains(first.xy, rowF)) {
      return PairResult(0.0, first.xy, NO_SEGMENT, first.xy);
    }
  }`:``}${r}
  return result;
}
`}var y=`
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
}`;function te(e){let{k:t,capacity:n}=e;return{declarations:`
const K: u32 = ${t}u;
const CAPACITY: u32 = ${n}u;
const STRIDE: u32 = ${3+2*n}u;
const FEATURE_COUNT: u32 = ${e.featureCount}u;
const INTERNAL_NODE_COUNT: u32 = ${e.internalNodeCount}u;
const STACK_SIZE: u32 = ${e.stackSize}u;
${e.scratchChunk?`override CHUNK_FIRST_ROW: u32;
override CHUNK_QUERY_COUNT: u32;
const CURSOR_WORD: u32 = ${e.scratchChunk.cursorWord}u;`:`const CHUNK_FIRST_ROW: u32 = ${e.chunkFirstRow}u;`}
const BOUND_SLACK: f32 = 1.000001;
${e.greatCircle?y:``}
// ${e.greatCircle?`Central-angle lower bound`:`Squared distance`} from the query box to a node box, or FLOAT32_MAXIMUM for an empty node.
fn boxLower(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> f32 {
  let base = nodeDataOffset + node * 5u;
  let nodeMinimum = vec2f(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  let nodeMaximum = vec2f(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u]));
  if (nodeMinimum.x > nodeMaximum.x || nodeMinimum.y > nodeMaximum.y) { return FLOAT32_MAXIMUM; }
  ${e.greatCircle?`return boxLowerAngle(nodeMinimum, nodeMaximum, queryMinimum, queryMaximum);`:`let gap = max(max(nodeMinimum - queryMaximum, queryMinimum - nodeMaximum), vec2f(0.0));
  return dot(gap, gap);`}
}`,body:`${e.scratchChunk?e.queryOrder?`let position = topk[topkOffset + CURSOR_WORD] + index;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = queryOrder[queryOrderOffset + position];
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + index * STRIDE;`:`let localQuery = topk[topkOffset + CURSOR_WORD] + index;
  if (localQuery >= CHUNK_QUERY_COUNT) { return; }
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + index * STRIDE;`:e.queryOrder?`let localQuery = queryOrder[queryOrderOffset + index];
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + row * STRIDE;`:`let localQuery = index;
  let row = CHUNK_FIRST_ROW + index;
  let header = topkOffset + row * STRIDE;`}
  var bound = bitcast<f32>(topk[header + 2u]);
  topk[header] = 0u;
  topk[header + 1u] = 0u;
  if (!(bound >= 0.0)) { return; }
  var queryMinimum = vec2f(FLOAT32_MAXIMUM);
  var queryMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var qRing = qRingStart(localQuery); qRing < qRingEnd(localQuery); qRing++) {
    let vertexStart = qRingVertexStart(qRing);
    let vertexEnd = qRingVertexEnd(qRing);
    if (vertexEnd <= vertexStart || vertexEnd - vertexStart < ${g(e.queryKind)}u) { continue; }
    for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
      let position = qVertex(vertex);
      if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
        queryMinimum = min(queryMinimum, position);
        queryMaximum = max(queryMaximum, position);
      }
    }
  }
  if (queryMinimum.x > queryMaximum.x) { return; }

  var listDistance: array<f32, ${n}>;
  var listRow: array<u32, ${n}>;
  var count = 0u;
  var droppedMinimum = FLOAT32_MAXIMUM;
  var stackNode: array<u32, ${e.stackSize}>;
  var stackLower: array<f32, ${e.stackSize}>;
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
    let featureRow = nodeData[nodeDataOffset + node * 5u + 4u];
    if (featureRow >= FEATURE_COUNT) { continue; }
    ${e.filter?`let queryFilter = nodeDataOffset + ${e.filter.queryBase}u + row * 2u;
    let featureFilter = nodeDataOffset + ${e.filter.featureBase}u + featureRow * 2u;
    ${e.filter.exclusive?`if (nodeData[queryFilter] == nodeData[featureFilter]) { continue; }`:``}
    ${e.filter.onAttribute?`if (nodeData[queryFilter + 1u] != nodeData[featureFilter + 1u]) { continue; }`:``}`:``}
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
  if (${e.ties===`all`?`true`:`CAPACITY > K`} && count >= K && droppedMinimum <= listDistance[K - 1u]) { overflowFlag = 1u; }
  for (var slot = 0u; slot < count; slot++) {
    topk[header + 3u + slot * 2u] = listRow[slot];
    topk[header + 4u + slot * 2u] = bitcast<u32>(listDistance[slot]);
  }
  topk[header] = count;
  topk[header + 1u] = overflowFlag;`}}var b=5,x=64;function S(e){switch(e.kind){case`points`:return e.positions.length;case`segments`:return e.starts.length;case`lines`:return e.lineOffsets.length-1;case`polygons`:return e.featureOffsets.length-1}}function C(e){switch(e.kind){case`points`:return[e.positions];case`segments`:return[e.starts,e.ends];case`lines`:return[e.positions,e.lineOffsets];case`polygons`:return[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]}}function w(e){return e.kind===`segments`?e.starts.length*2:e.positions.length}function T(e,t,n){let r=(t,n,r)=>({name:`${e}${t}`,view:n,type:r,access:`read`});switch(t.kind){case`points`:return[r(`Positions`,t.positions,`f32`)];case`segments`:return[r(`Starts`,t.starts,`f32`),r(`Ends`,t.ends,`f32`)];case`lines`:return[r(`Positions`,t.positions,`f32`),r(`LineOffsets`,t.lineOffsets,`u32`)];case`polygons`:return[r(`Positions`,t.positions,`f32`),r(`FeatureRings`,n,`u32`),r(`RingOffsets`,t.ringOffsets,`u32`)]}}function E(e,t,n,r,a){return i(e,{id:t,operation:n,variant:`polygon-rings`,bindings:[{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`featureRings`,view:a,type:`u32`,access:`read_write`}],invocationCount:a.length,declarations:`const POLYGON_COUNT: u32 = ${r.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${r.ringOffsets.length-1}u;`,body:`let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var ringStart = 0u;
  var ringEnd = 0u;
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let first = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let last = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (first <= last && last <= RING_COUNT) { ringStart = first; ringEnd = last; }
  }
  featureRings[featureRingsOffset + index * 2u] = ringStart;
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`})}function D(e,t,n,r,a,o,s){return r.kind===`lines`?i(e,{id:t,operation:n,variant:`bounds-lines`,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:r.lineOffsets,type:`u32`,access:`read`},{name:`featureMinima`,view:o,type:`f32`,access:`read_write`},{name:`featureMaxima`,view:s,type:`f32`,access:`read_write`}],invocationCount:a,declarations:`${f}
const VERTEX_COUNT: u32 = ${r.positions.length}u;`,body:`var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let vertexStart = lineOffsets[lineOffsetsOffset + index];
  let vertexEnd = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    let position = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
      minimum = min(minimum, position);
      maximum = max(maximum, position);
    }
  }
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`}):c(e,{id:t,operation:n,featureCount:a,source:r,minima:o,maxima:s})}function O(e,t){let{id:r,operation:i,features:a,featureCount:o,prepared:s}=t;if(s){if(!s.isDeclaredIn(e))throw Error(`${r} prepared handle must be added to the graph before the join`);let{storage:t}=s;return{bvh:{nodeMinima:t.nodeMinima,nodeMaxima:t.nodeMaxima,leafIds:t.leafIds,overflow:t.overflow,internalNodeCount:s.leafCapacity-1,nodeCount:2*s.leafCapacity-1,levelCount:Math.log2(s.leafCapacity)+1},nodes:[]}}let c=n(e,`${r}-feature-minima`,`float32x2`,o),u=n(e,`${r}-feature-maxima`,`float32x2`,o),d=[];o>0&&d.push(D(e,`${r}-bounds`,i,a,o,c,u));let f=l(e,r,i,c,u,t.leafCapacity,t.spatialSort,void 0,t.spatialSortCurve);return d.push(...f.nodes),{bvh:f.bvh,nodes:d}}function k(a,o){let{id:c,operation:l,features:d,featureCount:f,queryCount:p,k:g,capacity:_,sphereRadius:y}=o,S=y!==void 0,C=S?r(Math.fround(y)):``,k=[],{bvh:A,nodes:ne}=O(a,{id:c,operation:l,features:d,featureCount:f,leafCapacity:o.leafCapacity,spatialSort:o.spatialSort,spatialSortCurve:o.spatialSortCurve,prepared:o.prepared});k.push(...ne);let j=!!(o.exclusive||o.onAttribute),M=A.nodeCount*5,N=M+2*p,P=n(a,`${c}-node-data`,`uint32`,N+(j?2*f:0));if(k.push(i(a,{id:`${c}-node-data`,operation:l,variant:`node-data`,bindings:[{name:`nodeMinima`,view:A.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:A.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:A.leafIds,type:`u32`,access:`read`},{name:`nodeData`,view:P,type:`u32`,access:`read_write`}],invocationCount:A.nodeCount,declarations:`const INTERNAL_NODE_COUNT: u32 = ${A.internalNodeCount}u;
const NO_FEATURE: u32 = 0xffffffffu;`,body:`let base = nodeDataOffset + index * 5u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_FEATURE, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`})),j){let e=[{name:`nodeData`,view:P,type:`u32`,access:`read_write`}],t=[[`queryIds`,o.exclusive?o.queryIds:void 0],[`featureIds`,o.exclusive?o.featureIds:void 0],[`queryKeys`,o.onAttribute?.left],[`featureKeys`,o.onAttribute?.right]];for(let[n,r]of t)r&&e.push({name:n,view:r,type:`u32`,access:`read`});let n=(e,n)=>t.find(([t,n])=>t===e&&n)?`${e}[${e}Offset + index]`:n;k.push(i(a,{id:`${c}-filter`,operation:l,variant:`filter`,bindings:e,invocationCount:Math.max(p,f,1),declarations:`const QUERY_COUNT: u32 = ${p}u;
const FEATURE_COUNT: u32 = ${f}u;`,body:`if (index < QUERY_COUNT) {
    let base = nodeDataOffset + ${M}u + index * 2u;
    nodeData[base] = ${n(`queryIds`,`index`)};
    nodeData[base + 1u] = ${n(`queryKeys`,`0u`)};
  }
  if (index < FEATURE_COUNT) {
    let base = nodeDataOffset + ${N}u + index * 2u;
    nodeData[base] = ${n(`featureIds`,`index`)};
    nodeData[base + 1u] = ${n(`featureKeys`,`0u`)};
  }`}))}let F;d.kind===`polygons`&&o.prepared?F=o.prepared.storage.featureRings:d.kind===`polygons`&&(F=n(a,`${c}-feature-rings`,`uint32x2`,f),f>0&&k.push(E(a,`${c}-feature-rings`,l,d,F)));let I=o.queries.kind===`geometry`?o.queries.geometry:void 0,L;I?.kind===`polygons`&&(L=n(a,`${c}-query-rings`,`uint32x2`,p),p>0&&k.push(E(a,`${c}-query-rings`,l,I,L)));let R=3+2*_,z=!!(o.neighborFootPoints||o.neighborQueryPoints||o.neighborSegmentIndices),B=Math.floor(a.device.limits.maxStorageBufferBindingSize/4),V=Math.max(R,z?_*b:0),H=o.scratchChunkQueryCount;if(H!==void 0&&(!Number.isSafeInteger(H)||H<x||H%x!==0))throw Error(`${c} scratchChunkQueryCount must be a positive multiple of 64`);let U=H===void 0?p*V>B:p>H,W=U?H??Math.floor(Math.floor((B-1)/V)/x)*x:Math.max(p,1);if(W<1)throw Error(`${c} needs more scratch per query than one binding holds; reduce k`);let G=W*R;e(a.device,c,`topk`,(W*R+(U?1:0))*4,`reduce the query count or k`);let K=n(a,`${c}-topk`,`uint32`,W*R+(U?1:0)),q=[{name:`topk`,view:K,type:`u32`,access:`read_write`},{name:`overflow`,view:o.overflow,type:`u32`,access:`read_write`},{name:`bvhOverflow`,view:A.overflow,type:`u32`,access:`read`}];o.maxDistance&&q.push({name:`maxDistance`,view:o.maxDistance,type:`f32`,access:`read`});let J=`// A NaN, negative or infinite maxDistance sets a negative bound that matches nothing.
    ${o.maxDistance?`let limit = maxDistance[maxDistanceOffset];
    var bound = -1.0;
    if (limit >= 0.0 && limit <= FLOAT32_MAXIMUM) { bound = ${S?`// A central angle: the traversal compares angles, not squares.
    limit / SPHERE_RADIUS`:`min(limit * limit, FLOAT32_MAXIMUM)`}; }`:`let bound = FLOAT32_MAXIMUM;`}
    topk[header] = 0u;
    topk[header + 1u] = 0u;
    topk[header + 2u] = bitcast<u32>(bound);`,Y=`const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const STRIDE: u32 = ${R}u;${S?`\nconst SPHERE_RADIUS: f32 = ${C};`:``}`;U?k.push(i(a,{id:`${c}-clear-overflow`,operation:l,variant:`clear-overflow`,bindings:q.filter(e=>e.name.endsWith(`verflow`)),invocationCount:1,body:`overflow[overflowOffset] = bvhOverflow[bvhOverflowOffset];`})):k.push(i(a,{id:`${c}-clear`,operation:l,variant:S?`clear-great-circle`:`clear`,bindings:q,invocationCount:Math.max(p,1),declarations:`${Y}
const QUERY_COUNT: u32 = ${p}u;`,body:`if (index == 0u) { overflow[overflowOffset] = bvhOverflow[bvhOverflowOffset]; }
  if (index < QUERY_COUNT) {
    let header = topkOffset + index * STRIDE;
    ${J}
  }`}));let X=[];if(o.queries.kind===`points`){let e=0;for(let[n,r]of t(o.queries.points).entries()){if(r.length>0){let t={kind:`points`,positions:r},i;if(o.querySort&&r.length>=2){let e=u(a,{id:`${s(c,o.queries.points,n)}-order`,operation:l,points:r,bvh:A});k.push(...e.nodes),i=e.order}X.push({chunkId:s(c,o.queries.points,n),side:{prefix:`q`,kind:`points`,vertexCount:r.length},bindings:T(`q`,t,void 0),count:r.length,firstRow:e,kind:`points`,order:i})}e+=r.length}}else if(p>0){let t=o.queries.geometry,r={prefix:`q`,kind:t.kind,vertexCount:w(t)},s=T(`q`,t,L),d;if(o.querySort&&p>=2){if(t.kind===`polygons`){let o=p*2;e(a.device,c,`query metadata`,(o+t.ringOffsets.length)*4,`reduce the polygon query count or disable querySort`);let u=n(a,`${c}-query-metadata`,`uint32`,o+t.ringOffsets.length);k.push(i(a,{id:`${c}-query-metadata`,operation:l,variant:`query-metadata`,bindings:[{name:`featureRings`,view:L,type:`u32`,access:`read`},{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`metadata`,view:u,type:`u32`,access:`read_write`}],invocationCount:u.length,body:`if (index < ${o}u) {
    metadata[metadataOffset + index] = featureRings[featureRingsOffset + index];
  } else {
    metadata[metadataOffset + index] = ringOffsets[ringOffsetsOffset + index - ${o}u];
  }`})),r.packedRingOffsetsBase=o,s=[{name:`qPositions`,view:t.positions,type:`f32`,access:`read`},{name:`qFeatureRings`,view:u,type:`u32`,access:`read`}]}let o=n(a,`${c}-query-minima`,`float32x2`,p),f=n(a,`${c}-query-maxima`,`float32x2`,p),m=n(a,`${c}-query-centers`,`float32x2`,p);k.push(D(a,`${c}-query-bounds`,l,t,p,o,f),i(a,{id:`${c}-query-centers`,operation:l,variant:`query-centers`,bindings:[{name:`minima`,view:o,type:`f32`,access:`read`},{name:`maxima`,view:f,type:`f32`,access:`read`},{name:`centers`,view:m,type:`u32`,access:`read_write`}],invocationCount:p,body:`let minimum = vec2f(minima[minimaOffset + index * 2u], minima[minimaOffset + index * 2u + 1u]);
  let maximum = vec2f(maxima[maximaOffset + index * 2u], maxima[maximaOffset + index * 2u + 1u]);
  // Halve before adding to avoid overflow for large finite bounds. Empty bounds sort last.
  let center = minimum * 0.5 + maximum * 0.5;
  let valid = all(minimum <= maximum);
  centers[centersOffset + index * 2u] = select(0x7fc00000u, bitcast<u32>(center.x), valid);
  centers[centersOffset + index * 2u + 1u] = select(0x7fc00000u, bitcast<u32>(center.y), valid);`}));let h=u(a,{id:`${c}-order`,operation:l,points:m,bvh:A});k.push(...h.nodes),d=h.order}X.push({chunkId:c,side:r,bindings:s,count:p,firstRow:0,kind:t.kind,order:d})}let Z={prefix:`r`,kind:d.kind,vertexCount:w(d)},re=T(`r`,d,F),ie={name:`nodeData`,view:P,type:`u32`,access:`read`},Q=e=>({name:`topk`,view:K,type:`u32`,access:e}),ae=A.levelCount+2;z&&e(a.device,c,`extras`,W*_*b*4,`reduce the query count or k, or omit the neighbor foot-point outputs`);let $=z?n(a,`${c}-extras`,`uint32`,W*_*b):void 0,oe=(e,t,n,r)=>{let s=n?r:void 0,c=s?[{name:`queryOrder`,view:s,type:`u32`,access:`read`}]:[],u=[Q(`read`),{name:`neighborIds`,view:o.neighborIds,type:`u32`,access:`read_write`},{name:`neighborCounts`,view:o.neighborCounts,type:`u32`,access:`read_write`},{name:`overflow`,view:o.overflow,type:`atomic<u32>`,access:`read_write`}];o.featureIds&&!o.writeFeatureRows&&u.push({name:`featureIds`,view:o.featureIds,type:`u32`,access:`read`}),o.neighborDistances&&u.push({name:`neighborDistances`,view:o.neighborDistances,type:`f32`,access:`read_write`});let d=n?`let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let position = topk[topkOffset + CURSOR_WORD] + query;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = ${s?`queryOrder[queryOrderOffset + position]`:`position`};
  let row = CHUNK_FIRST_ROW + localQuery;
  let outputIndex = row * CAPACITY + slot;`:`let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let row = query;
  let outputIndex = index;`;if(k.push(i(a,{id:`${e}-columns`,operation:l,variant:`${S?`columns-great-circle`:`columns`}${s?`-ordered`:``}`,bindings:[...u,...c],invocationCount:t*_,declarations:`const CAPACITY: u32 = ${_}u;
const STRIDE: u32 = ${R}u;
const NO_FEATURE: u32 = 0xffffffffu;${S?`\nconst SPHERE_RADIUS: f32 = ${C};`:``}${n?`\nconst CURSOR_WORD: u32 = ${G}u;
override CHUNK_FIRST_ROW: u32;
override CHUNK_QUERY_COUNT: u32;`:``}`,...n?{constants:{CHUNK_FIRST_ROW:n.firstRow,CHUNK_QUERY_COUNT:n.queryCount}}:{},body:`${d}
  let header = topkOffset + query * STRIDE;
  let count = topk[header];
  if (slot == 0u) {
    neighborCounts[neighborCountsOffset + row] = count;
    if (topk[header + 1u] != 0u) { atomicOr(&overflow[overflowOffset], 1u); }
  }
  var featureId = NO_FEATURE;
  var distance = -1.0;
  if (slot < count) {
    featureId = topk[header + 3u + slot * 2u];
    ${o.featureIds&&!o.writeFeatureRows?`featureId = featureIds[featureIdsOffset + featureId];`:``}
    distance = ${S?`bitcast<f32>(topk[header + 4u + slot * 2u]) * SPHERE_RADIUS`:`sqrt(bitcast<f32>(topk[header + 4u + slot * 2u]))`};
  }
  neighborIds[neighborIdsOffset + outputIndex] = featureId;
  ${o.neighborDistances?`neighborDistances[neighborDistancesOffset + outputIndex] = distance;`:``}`})),$){let r=[Q(`read`),{name:`extras`,view:$,type:`u32`,access:`read`}];o.neighborFootPoints&&r.push({name:`footPoints`,view:o.neighborFootPoints,type:`u32`,access:`read_write`}),o.neighborQueryPoints&&r.push({name:`queryPoints`,view:o.neighborQueryPoints,type:`u32`,access:`read_write`}),o.neighborSegmentIndices&&r.push({name:`segmentIndices`,view:o.neighborSegmentIndices,type:`u32`,access:`read_write`}),k.push(i(a,{id:`${e}-foot-columns`,operation:l,variant:s?`foot-columns-ordered`:`foot-columns`,bindings:[...r,...c],invocationCount:t*_,declarations:`const CAPACITY: u32 = ${_}u;
const STRIDE: u32 = ${R}u;
const NO_SEGMENT: u32 = 0xffffffffu;${n?`\nconst CURSOR_WORD: u32 = ${G}u;
override CHUNK_FIRST_ROW: u32;
override CHUNK_QUERY_COUNT: u32;`:``}`,...n?{constants:{CHUNK_FIRST_ROW:n.firstRow,CHUNK_QUERY_COUNT:n.queryCount}}:{},body:`${d}
  let valid = slot < topk[topkOffset + query * STRIDE];
  let source = extrasOffset + index * ${b}u;
  ${o.neighborFootPoints?`// Unused slots hold a quiet NaN so a missing neighbor is never mistaken for the origin.
  footPoints[footPointsOffset + outputIndex * 2u] = select(0x7fc00000u, extras[source], valid);
  footPoints[footPointsOffset + outputIndex * 2u + 1u] = select(0x7fc00000u, extras[source + 1u], valid);`:``}
  ${o.neighborQueryPoints?`queryPoints[queryPointsOffset + outputIndex * 2u] = select(0x7fc00000u, extras[source + 3u], valid);
  queryPoints[queryPointsOffset + outputIndex * 2u + 1u] = select(0x7fc00000u, extras[source + 4u], valid);`:``}
  ${o.neighborSegmentIndices?`segmentIndices[segmentIndicesOffset + outputIndex] = select(NO_SEGMENT, extras[source + 2u], valid);`:``}`}))}},se=(e,t,n,r)=>{let s=e.order?[{name:`queryOrder`,view:e.order,type:`u32`,access:`read`}]:[],c=te({ties:o.ties,k:g,capacity:_,featureCount:f,internalNodeCount:A.internalNodeCount,stackSize:ae,chunkFirstRow:e.firstRow,scratchChunk:r?{cursorWord:G}:void 0,queryOrder:!!e.order,queryKind:e.kind,greatCircle:S,filter:j?{exclusive:!!o.exclusive,onAttribute:!!o.onAttribute,queryBase:M,featureBase:N}:void 0}),u=`${S?h:m}${v(e.side)}${v(Z)}${ee(e.side,Z)}`,p=r?{CHUNK_FIRST_ROW:r.firstRow,CHUNK_QUERY_COUNT:r.queryCount}:void 0;k.push(i(a,{id:`${t}-traverse`,operation:l,variant:`traverse-${e.kind}-${d.kind}${S?`-great-circle`:``}${e.order?`-ordered`:``}`,record:!S&&g===1&&e.kind===`points`&&d.kind===`points`?{formula:`gpu-spatial-analysis.nearest-feature-join`}:void 0,bindings:[...e.bindings,ie,...re,Q(`read_write`),...s],invocationCount:n,declarations:`${u}${c.declarations}`,...p?{constants:p}:{},body:c.body})),$&&k.push(i(a,{id:`${t}-foot-points`,operation:l,variant:`foot-${e.kind}-${d.kind}${S?`-great-circle`:``}${r&&e.order?`-ordered`:``}`,bindings:[...e.bindings,...re,Q(`read`),{name:`extras`,view:$,type:`u32`,access:`read_write`},...r?s:[]],invocationCount:n,declarations:`${u}
const CAPACITY: u32 = ${_}u;
const STRIDE: u32 = ${R}u;${r?`\nconst CURSOR_WORD: u32 = ${G}u;
override CHUNK_QUERY_COUNT: u32;`:`\nconst CHUNK_FIRST_ROW: u32 = ${e.firstRow}u;`}`,...r?{constants:{CHUNK_QUERY_COUNT:r.queryCount}}:{},body:`${r?`let position = topk[topkOffset + CURSOR_WORD] + index;
  if (position >= CHUNK_QUERY_COUNT) { return; }
  let localQuery = ${e.order?`queryOrder[queryOrderOffset + position]`:`position`};
  let row = index;`:`let localQuery = index;
  let row = CHUNK_FIRST_ROW + localQuery;`}
  let header = topkOffset + row * STRIDE;
  let count = topk[header];
  for (var slot = 0u; slot < count; slot++) {
    let result = pairResult(localQuery, topk[header + 3u + slot * 2u]);
    let output = extrasOffset + (row * CAPACITY + slot) * ${b}u;
    extras[output] = bitcast<u32>(result.foot.x);
    extras[output + 1u] = bitcast<u32>(result.foot.y);
    extras[output + 2u] = result.segment;
    extras[output + 3u] = bitcast<u32>(result.queryFoot.x);
    extras[output + 4u] = bitcast<u32>(result.queryFoot.y);
  }`}))};if(!U){for(let e of X)se(e,e.chunkId,e.count);return p>0&&oe(c,p),{nodes:k,bvh:A}}for(let e of X){let t=Math.ceil(e.count/W),n=Math.min(e.count,W),r={firstRow:e.firstRow,queryCount:e.count};k.push(i(a,{id:`${e.chunkId}-reset-cursor`,operation:l,variant:`reset-cursor`,bindings:[Q(`read_write`)],invocationCount:1,declarations:`const CURSOR_WORD: u32 = ${G}u;`,body:`topk[topkOffset + CURSOR_WORD] = 0u;`}));for(let o=0;o<t;o++){let s=`${e.chunkId}-scratch-${o}`;k.push(i(a,{id:`${s}-clear`,operation:l,variant:S?`clear-chunk-great-circle`:`clear-chunk`,bindings:q.filter(e=>e.name===`topk`||e.name===`maxDistance`),invocationCount:n,declarations:`${Y}
const CURSOR_WORD: u32 = ${G}u;
override CHUNK_QUERY_COUNT: u32;`,constants:{CHUNK_QUERY_COUNT:e.count},body:`if (topk[topkOffset + CURSOR_WORD] + index < CHUNK_QUERY_COUNT) {
    let header = topkOffset + index * STRIDE;
    ${J}
  }`})),se(e,s,n,r),oe(s,n,r,e.order),o<t-1&&k.push(i(a,{id:`${s}-advance-cursor`,operation:l,variant:`advance-cursor`,bindings:[Q(`read_write`)],invocationCount:1,declarations:`const CURSOR_WORD: u32 = ${G}u;
const SCRATCH_QUERY_COUNT: u32 = ${W}u;`,body:`topk[topkOffset + CURSOR_WORD] = topk[topkOffset + CURSOR_WORD] + SCRATCH_QUERY_COUNT;`}))}}return{nodes:k,bvh:A}}export{y as a,k as i,S as n,C as r,O as t};