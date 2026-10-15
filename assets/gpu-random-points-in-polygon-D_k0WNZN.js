import{E as e,h as t}from"./analysis-frame-Blu0tYdw.js";import{_ as n,d as r,f as i,h as a,l as o,v as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{t as u}from"./gpu-scan-BuGhTLDQ.js";import{a as d,n as f,r as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as h,u as g}from"./wgsl-common-BNRVzgd8.js";import{t as _}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{t as v}from"./exact-geometry-wgsl-DOqxBQzc.js";import{n as y}from"./sorted-segment-sums-DwGkmbx2.js";import{i as ee,n as b,r as x,t as S}from"./polygon-edge-slabs-Cmf_6bAe.js";import{t as C}from"./stable-lexicographic-index-sort-CcJLc7b1.js";import{n as w,t as T}from"./philox-random-CHwyk0dd.js";var te=T.dotDensityRemainder.id,E=T.dotDensityCandidate.id;function ne(e,t){return Math.min(t+1,Math.floor(4294967295/Math.max(e,1)),2**31)}var D={routed:0,clean:1,pieces:2,jumpPairs:3,maximumClean:4,jumpActive:5},O={nonFinite:1,shell:2,orientation:4,ray:8,diagonal:16,piece:32,stack:64},k={start:0,end:1,split:2,merge:3,regularLeft:4,regularRight:5};function A(e){return`
const NONE: u32 = 0xffffffffu;
const OVERFLOW_BIT: u32 = 0x80000000u;
const VERTEX_COUNT: u32 = ${e.vertexCount}u;
const POLYGON_COUNT: u32 = ${e.polygonCount}u;
const PREPARED_LAYOUT: bool = ${!!e.preparedLayout};
const RING_COUNT: u32 = ${e.ringCount}u;
const CAPACITY: u32 = ${e.capacity}u;
const INDEX_LENGTH: u32 = ${e.indexLength}u;
const MINIMUM_VERTICES: u32 = ${e.minimumVertices}u;
const COUNT_ROUTED: u32 = ${D.routed}u;
const COUNT_CLEAN: u32 = ${D.clean}u;
const COUNT_PIECES: u32 = ${D.pieces}u;
const COUNT_JUMP_PAIRS: u32 = ${D.jumpPairs}u;
const COUNT_MAXIMUM_CLEAN: u32 = ${D.maximumClean}u;
const COUNT_JUMP_ACTIVE: u32 = ${D.jumpActive}u;
const COUNT_WORDS: u32 = 8u;
const STATS_STRIDE: u32 = 8u;
const STATS_FAIL: u32 = 0u;
const STATS_CLEAN: u32 = 1u;
const STATS_HOLES: u32 = 2u;
const STATS_TRIANGLES: u32 = 3u;
const STATS_WRITTEN: u32 = 4u;
const FAIL_NON_FINITE: u32 = ${O.nonFinite}u;
const FAIL_SHELL: u32 = ${O.shell}u;
const FAIL_ORIENTATION: u32 = ${O.orientation}u;
const FAIL_RAY: u32 = ${O.ray}u;
const FAIL_DIAGONAL: u32 = ${O.diagonal}u;
const FAIL_PIECE: u32 = ${O.piece}u;
const FAIL_STACK: u32 = ${O.stack}u;
const TYPE_START: u32 = ${k.start}u;
const TYPE_END: u32 = ${k.end}u;
const TYPE_SPLIT: u32 = ${k.split}u;
const TYPE_MERGE: u32 = ${k.merge}u;
const TYPE_REGULAR_LEFT: u32 = ${k.regularLeft}u;
const TYPE_REGULAR_RIGHT: u32 = ${k.regularRight}u;
const A_POLYGON: u32 = 0u;
const A_RING: u32 = 1u;
const A_ROW: u32 = 2u;
const A_TYPE: u32 = 3u;
const B_RANK: u32 = 0u;
const B_NEXT: u32 = 1u;
const B_PREVIOUS: u32 = 2u;
const B_LEFT: u32 = 3u;
const PIECE_STRIDE: u32 = 8u;
`}var j=`
fn polygonOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_POLYGON]; }
fn ringOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_RING]; }
fn rowOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_ROW]; }
fn typeOf(vertex: u32) -> u32 { return vertexA[vertexAOffset + 4u * vertex + A_TYPE] & 255u; }
fn isFlipped(vertex: u32) -> bool { return ((vertexA[vertexAOffset + 4u * vertex + A_TYPE] >> 8u) & 1u) == 1u; }
`,M=`
fn rankOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_RANK]; }
fn nextOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_NEXT]; }
fn previousOf(vertex: u32) -> u32 { return vertexB[vertexBOffset + 4u * vertex + B_PREVIOUS]; }
`,N=`
fn positionOf(vertex: u32) -> vec2f {
  return vec2f(vertexPosition[vertexPositionOffset + 2u * vertex], vertexPosition[vertexPositionOffset + 2u * vertex + 1u]);
}
`,P=`
var<private> orientUncertain: bool;
fn orientOf(a: vec2f, b: vec2f, c: vec2f) -> i32 {
  let result = orientSign(a, b, c);
  if (result == 2) {
    orientUncertain = true;
    return 0;
  }
  return result;
}
`,F=`
fn failPolygon(failedPolygon: u32, failureBit: u32) {
  atomicOr(&polygonStats[polygonStatsOffset + STATS_STRIDE * failedPolygon + STATS_FAIL], failureBit);
}
`,I=`
fn tailOf(halfEdge: u32) -> u32 {
  if (halfEdge < CAPACITY) { return halfEdge; }
  return (halfEdge - CAPACITY) / 4u;
}
`,L=`
fn orderedKey(x: f32) -> u32 {
  var bits = bitcast<u32>(x);
  if (x == 0.0) { bits = 0u; }
  return bits ^ select(0x80000000u, 0xffffffffu, (bits & 0x80000000u) != 0u);
}
`,R=`
fn baseSlotOf(polygon: u32) -> i32 {
  if (PREPARED_LAYOUT) {
    return 3 * i32(polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT + 1u + polygon]);
  }
  let ringPrefix = i32(polygonOffsets[polygonOffsetsOffset + polygon]);
  let vertexPrefix = i32(ringOffsets[ringOffsetsOffset + u32(ringPrefix)]);
  return 3 * (vertexPrefix + 2 * ringPrefix - 4 * i32(polygon));
}
`,re=`
fn polygonFirstRow(polygon: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + polygonOffsets[polygonOffsetsOffset + polygon]], VERTEX_COUNT);
}
fn polygonEndRow(polygon: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + polygonOffsets[polygonOffsetsOffset + polygon + 1u]], VERTEX_COUNT);
}
`;function z(e,t=!1){return`
const MONOTONE_MINIMUM_VERTICES: u32 = ${e}u;
fn sizeRouteOf(polygon: u32) -> u32 {
  ${t?`let count = polygonOffsets[polygonOffsetsOffset + 2u * (POLYGON_COUNT + 1u) + polygon];
  return select(0u, 1u, count >= MONOTONE_MINIMUM_VERTICES);`:`
  let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
  let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  if (endRing <= firstRing) { return 0u; }
  let first = min(ringOffsets[ringOffsetsOffset + firstRing], VERTEX_COUNT);
  let end = min(ringOffsets[ringOffsetsOffset + endRing], VERTEX_COUNT);
  return select(0u, 1u, end >= first + MONOTONE_MINIMUM_VERTICES);`}
}
`}function ie(e){return{declarations:`${A(e)}${re}`,body:`var count = 0u;
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
  }`}}function ae(e){return{declarations:A(e),body:`let count = routedCounts[routedCountsOffset + index];
  if (count > 0u && routedOffsets[routedOffsetsOffset + index] + count > CAPACITY) {
    routing[routingOffset + index] = 3u;
    routedCounts[routedCountsOffset + index] = 0u;
  }
  if (index == 0u) {
    counts[countsOffset + COUNT_ROUTED] = routedOffsets[routedOffsetsOffset + POLYGON_COUNT];
  }`}}function oe(e){return{declarations:`${A(e)}${F}
fn vertexAt(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn isFiniteBits(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }`,body:`let j = index;
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
  ${e.preparedLayout?`if (ringOffsets[ringOffsetsOffset + RING_COUNT + 1u + ring] == 0u) { return; }`:``}
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
  let point = vertexAt(row);
  if (!isFiniteBits(point.x) || !isFiniteBits(point.y)) {
    failPolygon(polygon, FAIL_NON_FINITE);
    return;
  }
  let previousRow = select(row - 1u, ringEnd - 1u, row == ringBegin);
  let previous = vertexAt(previousRow);
  if (point.x != previous.x || point.y != previous.y) { keepFlags[keepFlagsOffset + j] = 1u; }`}}function se(e){return{declarations:`${A(e)}${F}`,body:`let info = jInfoOffset + 4u * index;
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
  if (ring == shellRing && shellCount < 3u) { failPolygon(polygon, FAIL_SHELL); }`}}function ce(e){return{declarations:A(e),body:`let routed = min(counts[countsOffset + COUNT_ROUTED], VERTEX_COUNT);
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
  counts[countsOffset + COUNT_JUMP_PAIRS] = (rounds + 1u) / 2u;`}}function le(e){return{declarations:A(e),body:`let info = jInfoOffset + 4u * index;
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
  rowToClean[rowToCleanOffset + row] = vertex;`}}function ue(e){return{declarations:`${A(e)}${L}`,body:`let vertex = index;
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
  identity[identityOffset + vertex] = vertex;`}}function de(e){return{declarations:`${A(e)}${j}`,body:`let vertex = sortedIndices[sortedIndicesOffset + index];
  if (vertex >= CAPACITY) { return; }
  if (polygonOf(vertex) == NONE) { return; }
  vertexB[vertexBOffset + 4u * vertex + B_RANK] = index;
  atomicMin(&ringTop[ringTopOffset + ringOf(vertex)], index);`}}function fe(e){return{declarations:`${A(e)}${j}${M}${N}${F}${P}`,body:`let vertex = index;
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
  if (flip) { ringInfo[info + 2u] = ringInfo[info + 2u] | 2u; }`}}function pe(e){return{declarations:`${A(e)}${j}${M}${N}${F}${P}`,body:`let vertex = index;
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
  }`}}var me=`
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
`;function he(e){return`let vertex = index;
  let polygon = polygonOf(vertex);
  if (polygon == NONE) { return; }
  let kind = typeOf(vertex);
  if (kind != TYPE_SPLIT && kind != TYPE_MERGE && kind != TYPE_REGULAR_RIGHT) { return; }
  let point = positionOf(vertex);
  let queryRank = rankOf(vertex);
  var best = NONE;
  ${e(`let cleanEnd = rowToClean[rowToCleanOffset + edgeEnd];
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
        if (closer) { best = upperVertex; }`)}
  if (orientUncertain) { best = NONE; }
  vertexB[vertexBOffset + 4u * vertex + B_LEFT] = best;`}function ge(e){return{declarations:`${A(e)}${j}${M}${F}`,body:`let rank = index;
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
  memberIdentity[memberIdentityOffset + 3u * rank + 2u] = 3u * rank + 2u;`}}function _e(e){return{declarations:`${A(e)}${j}${F}`,body:`let member = sortedMembers[sortedMembersOffset + index];
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
  vertexDiagonal[vertexDiagonalOffset + 4u * earlier + earlierSlot] = (later << 2u) | laterSlot;`}}function ve(e){return{declarations:`${A(e)}${j}${M}${N}${F}${P}
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
}`,body:`let vertex = index;
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
  }`}}function ye(e){return{declarations:`${A(e)}${j}`,body:`let vertex = index;
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
  }`}}function be(e){return{declarations:A(e),body:`let vertex = index;
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    let pointer = min(jumpSource[jumpSourceOffset + 2u * halfEdge], 5u * CAPACITY - 1u);
    let label = jumpSource[jumpSourceOffset + 2u * halfEdge + 1u];
    jumpTarget[jumpTargetOffset + 2u * halfEdge] = jumpSource[jumpSourceOffset + 2u * pointer];
    jumpTarget[jumpTargetOffset + 2u * halfEdge + 1u] = min(label, jumpSource[jumpSourceOffset + 2u * pointer + 1u]);
  }`}}function xe(e){return{declarations:A(e),body:`let remaining = counts[countsOffset + COUNT_JUMP_PAIRS];
  if (remaining > 0u) {
    counts[countsOffset + COUNT_JUMP_PAIRS] = remaining - 1u;
    counts[countsOffset + COUNT_JUMP_ACTIVE] = counts[countsOffset + COUNT_CLEAN];
  } else {
    counts[countsOffset + COUNT_JUMP_ACTIVE] = 0u;
  }`}}function Se(e){return{declarations:A(e),body:`let vertex = index;
  var count = 0u;
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    if (jumpA[jumpAOffset + 2u * halfEdge + 1u] == halfEdge) { count += 1u; }
  }
  repCounts[repCountsOffset + vertex] = count;`}}function Ce(e){return{declarations:A(e),body:`let vertex = index;
  var piece = repOffsets[repOffsetsOffset + vertex];
  for (var slot = 0u; slot < 5u; slot++) {
    let halfEdge = select(CAPACITY + 4u * vertex + slot - 1u, vertex, slot == 0u);
    if (jumpA[jumpAOffset + 2u * halfEdge + 1u] == halfEdge) {
      if (piece < CAPACITY) { pieceTable[pieceTableOffset + PIECE_STRIDE * piece] = halfEdge; }
      piece += 1u;
    }
  }`}}function we(e){return{declarations:A(e),body:`let clean = counts[countsOffset + COUNT_CLEAN];
  counts[countsOffset + COUNT_PIECES] = min(repOffsets[repOffsetsOffset + clean], CAPACITY);`}}function Te(e){return{declarations:`${A(e)}${j}${F}${I}`,body:`let piece = index;
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
  pieceTriangles[pieceTrianglesOffset + piece] = select(0u, vertexCountOfPiece - 2u, vertexCountOfPiece >= 3u);`}}function Ee(e){return{declarations:`${A(e)}${R}`,body:`let route = routing[routingOffset + index];
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
  polygonPlan[plan + 3u] = 1u;`}}function De(e){return{declarations:`${A(e)}${j}${F}${I}`,body:`let piece = index;
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
  pieceTable[table + 5u] = polygon;`}}function Oe(e){return{declarations:`${A(e)}${j}${M}${N}${F}${I}${P}
const SCRATCH_REGION: u32 = ${3*e.capacity}u;
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
`,body:`triangulatePiece(index);`}}function ke(e,t){return{declarations:A(e),body:`let route = routing[routingOffset + index];
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
  ${t?`triangleCount[triangleCountOffset + index] = select(0u, triangles, ok);`:``}`}}function Ae(e){return{declarations:`${A(e)}${R}${re}
fn padUnit(padIndex: u32, padSliceStart: u32, padCapacity: u32, padWritten: u32, padVertex: u32) {
  if (padIndex >= padCapacity || padIndex < padWritten) { return; }
  let padSlot = indicesOffset + padSliceStart + 3u * padIndex;
  indices[padSlot] = padVertex;
  indices[padSlot + 1u] = padVertex;
  indices[padSlot + 2u] = padVertex;
}`,body:`let info = jInfoOffset + 4u * index;
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
  }`}}function je(e){return o(e,40)}function B(e,t,n){s(t,`monotoneMaximumVertices`,n,je(e))}var Me=256,Ne=64;function Pe(e){return{preparedLayout:!!e.layout,vertexCount:e.positions.length,polygonCount:e.polygonOffsets.length-1,ringCount:e.ringOffsets.length-1,capacity:e.capacity,indexLength:e.output.indices.length,minimumVertices:e.minimumVertices}}function Fe(e){return Math.max(1,32-Math.clz32(e))}function Ie(e){let t=Math.ceil(Math.log2(3*e+3));return Math.ceil(t/2)}function V(e,t){let{id:n,operation:r,positions:i,routing:a,output:o}=t,s=t.layout?.polygonOffsets??t.polygonOffsets,l=t.layout?.ringOffsets??t.ringOffsets,f=Pe(t),{vertexCount:p,polygonCount:h,ringCount:g,capacity:v}=f,y=[],w=t.recorder,T=(t,r)=>c(e,`${n}-${t}`,`uint32`,Math.max(r,1)),te=e=>Math.max(e,1),E=T(`counts`,8),ne=T(`routed-counts`,h+1),O=T(`routed-offsets`,h+1),k=T(`polygon-stats`,8*h),F=T(`polygon-plan`,4*h),I=T(`keep-flags`,p+1),L=T(`keep-offsets`,p+1),R=T(`vertex-info`,4*p),re=T(`row-to-clean`,p),z=T(`ring-info`,4*g),je=T(`ring-top`,g),B=T(`vertex-a`,4*v),V=T(`vertex-b`,4*v),H=c(e,`${n}-vertex-position`,`float32x2`,te(v)),U=T(`sort-key-high`,v),Re=T(`sort-key-low`,v),ze=T(`identity`,v),W=T(`member-group`,3*v),Be=T(`member-identity`,3*v),Ve=T(`vertex-diagonal`,4*v),G=T(`next-half-edge`,5*v),He=T(`jump-a`,10*v),Ue=T(`jump-b`,10*v),We=T(`rep-counts`,v+1),K=T(`rep-offsets`,v+1),Ge=T(`piece-table`,8*v),Ke=T(`piece-triangles`,v+1),qe=T(`piece-triangle-offsets`,v+1),Je=T(`piece-scratch`,6*v),Ye=(t,n)=>{y.push(m(e,t)),w?.({kind:`kernel`,id:t.id,props:t,gate:n})},q=(e,t,i,a,o,s={})=>Ye({id:`${n}-${e}`,operation:r,variant:t,bindings:i,invocationCount:Math.max(a,1),declarations:o.declarations,body:o.body,...s}),Xe=(t,r,i)=>{y.push(...new u({id:`${n}-${t}`,input:r,output:i,mode:`exclusive`}).getCommandNodes(e)),w?.({kind:`scan`,id:`${n}-${t}`,input:r,output:i})},Ze=(t,i,a,o,s=1)=>{let c=d(e,{id:`${n}-${t}-dispatch`,operation:r,count:E,countRow:i,maximumItemCount:a,workgroupSize:o,invocationsPerItem:s});return y.push(c.updateNode),{condition:c.condition,workgroupSize:o,gate:{countRow:i,invocationsPerItem:s,workgroupSize:o}}},J=(e,t,i,a,o,s)=>Ye({id:`${n}-${t}`,operation:r,variant:i,bindings:a,invocationCount:Math.max(o,1),workgroupSize:e.workgroupSize,condition:{...e.condition,id:`${n}-${t}-gate`},declarations:s.declarations,body:s.body},e.gate),Y=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),X=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),Z=(e,t)=>X(e,t,`atomic<u32>`);q(`route-counts`,`monotone-route-counts`,[Y(`polygonOffsets`,s),Y(`ringOffsets`,l),Y(`routing`,a),X(`routedCounts`,ne),X(`polygonStats`,k),X(`counts`,E)],h+1,ie(f)),Xe(`route-scan`,ne,O),q(`route-overflow`,`monotone-route-overflow`,[X(`routing`,a),X(`routedCounts`,ne),Y(`routedOffsets`,O),X(`counts`,E)],h,ae(f));let Qe=Ze(`routed`,D.routed,p,Me),Q=x(e,{id:`${n}-slabs`,operation:r,owner:`polygon`,positions:i,polygonOffsets:t.polygonOffsets,ringOffsets:t.ringOffsets,gate:{counts:ne,minimum:1}});y.push(...Q.nodes),w?.({kind:`slabs`,id:`${n}-slabs`,params:Q.slabs.params,offsets:Q.slabs.slabOffsets,entries:Q.slabs.slabEntries}),J(Qe,`index`,`monotone-index`,[Y(`polygonOffsets`,s),Y(`ringOffsets`,l),Y(`positions`,i,`f32`),Y(`routing`,a),Y(`routedOffsets`,O),X(`keepFlags`,I),X(`jInfo`,R),Z(`polygonStats`,k)],p,oe(f)),Xe(`keep-scan`,I,L),J(Qe,`ring`,`monotone-ring`,[Y(`jInfo`,R),Y(`ringOffsets`,l),Y(`routedOffsets`,O),Y(`keepOffsets`,L),X(`ringInfo`,z),Z(`ringTop`,je),Z(`polygonStats`,k),Z(`counts`,E)],p,se(f)),q(`counts`,`monotone-counts`,[X(`counts`,E),Y(`keepOffsets`,L)],1,ce(f));let $=Ze(`clean`,D.clean,v,Me),$e=Ze(`members`,D.clean,v,Me,3);J(Qe,`compact`,`monotone-compact`,[Y(`jInfo`,R),Y(`keepFlags`,I),Y(`keepOffsets`,L),Y(`ringInfo`,z),Y(`positions`,i,`f32`),X(`vertexA`,B),X(`vertexPosition`,H,`f32`),X(`rowToClean`,re)],p,le(f)),q(`keys`,`monotone-keys`,[Y(`counts`,E),X(`vertexA`,B),Y(`vertexPosition`,H,`f32`),X(`sortKeyHigh`,U),X(`sortKeyLow`,Re),X(`identity`,ze)],v,ue(f));let et=C(e,{id:`${n}-sweep-sort`,operation:r,indices:ze,keys:[{view:U},{view:Re}]});y.push(...et.nodes);let tt=et.sortedIndices;w?.({kind:`sort`,id:`${n}-sweep-sort`,keys:[U,Re],indices:ze,sortedIndices:tt}),J($,`rank`,`monotone-rank`,[Y(`sortedIndices`,tt),Y(`vertexA`,B),X(`vertexB`,V),Z(`ringTop`,je)],v,de(f)),J($,`flip`,`monotone-flip`,[Y(`vertexA`,B),Y(`vertexB`,V),X(`ringInfo`,z),Z(`ringTop`,je),Y(`vertexPosition`,H,`f32`),Y(`polygonOffsets`,s),Z(`polygonStats`,k)],v,Le(fe(f))),J($,`link`,`monotone-link`,[X(`vertexA`,B),X(`vertexB`,V),Y(`ringInfo`,z),Y(`vertexPosition`,H,`f32`),X(`vertexDiagonal`,Ve),X(`nextHalfEdge`,G),Z(`polygonStats`,k)],v,Le(pe(f)));let nt=S(Q.slabs),rt=`${_}${A(f)}${j}${M}${N}${P}${ee(Q.slabs)}${me}`;J($,`left-edge`,`monotone-left-edge`,[...nt,Y(`rowToClean`,re),Y(`vertexA`,B),X(`vertexB`,V),Y(`vertexPosition`,H,`f32`)],v,{declarations:rt,body:he(e=>b({owner:`polygon`,lowY:`point.y`,highY:`point.y`,body:e}))}),q(`members`,`monotone-members`,[Y(`counts`,E),Y(`sortedIndices`,tt),Y(`vertexA`,B),Y(`vertexB`,V),X(`memberGroup`,W),X(`memberIdentity`,Be),Z(`polygonStats`,k)],v,ge(f));let it=C(e,{id:`${n}-member-sort`,operation:r,indices:Be,keys:[{view:W,keyBits:Fe(v)}]});y.push(...it.nodes),w?.({kind:`sort`,id:`${n}-member-sort`,keys:[W],indices:Be,sortedIndices:it.sortedIndices}),J($e,`diagonals`,`monotone-diagonals`,[Y(`sortedMembers`,it.sortedIndices),Y(`memberGroup`,W),Y(`sortedIndices`,tt),Y(`vertexA`,B),X(`vertexDiagonal`,Ve),Z(`polygonStats`,k)],3*v,_e(f)),J($,`next`,`monotone-next`,[Y(`vertexA`,B),Y(`vertexB`,V),Y(`vertexPosition`,H,`f32`),Y(`vertexDiagonal`,Ve),X(`nextHalfEdge`,G),Z(`polygonStats`,k)],v,Le(ve(f))),J($,`jump-init`,`monotone-jump-init`,[Y(`vertexA`,B),Y(`vertexDiagonal`,Ve),Y(`nextHalfEdge`,G),X(`jumpA`,He)],v,ye(f));let at=t.jumpPairCount??Ie(v),ot=be(f),st=xe(f);for(let e=0;e<at;e++){q(`jump-${e}-state`,`monotone-jump-state`,[X(`counts`,E)],1,st);let t=Ze(`jump-${e}`,D.jumpActive,v,Me);J(t,`jump-${e}-forward`,`monotone-jump`,[Y(`jumpSource`,He),X(`jumpTarget`,Ue)],v,ot),J(t,`jump-${e}-backward`,`monotone-jump`,[Y(`jumpSource`,Ue),X(`jumpTarget`,He)],v,ot)}J($,`rep-count`,`monotone-rep-count`,[Y(`jumpA`,He),X(`repCounts`,We)],v,Se(f)),Xe(`rep-scan`,We,K),J($,`pieces`,`monotone-pieces`,[Y(`jumpA`,He),Y(`repOffsets`,K),X(`pieceTable`,Ge)],v,Ce(f)),q(`piece-count`,`monotone-piece-count`,[X(`counts`,E),Y(`repOffsets`,K)],1,we(f));let ct=Ze(`pieces`,D.pieces,v,Ne);return J(ct,`measure`,`monotone-measure`,[Y(`counts`,E),X(`pieceTable`,Ge),Y(`nextHalfEdge`,G),Y(`vertexA`,B),Z(`polygonStats`,k),X(`pieceTriangles`,Ke)],v,Te(f)),Xe(`piece-scan`,Ke,qe),q(`polygon-plan`,`monotone-polygon-plan`,[Y(`polygonOffsets`,s),Y(`ringOffsets`,l),Y(`routing`,a),Y(`routedOffsets`,O),Y(`keepOffsets`,L),Y(`repOffsets`,K),Y(`pieceTriangleOffsets`,qe),X(`polygonPlan`,F)],h,Ee(f)),J(ct,`piece-plan`,`monotone-piece-plan`,[Y(`counts`,E),X(`pieceTable`,Ge),Y(`pieceTriangleOffsets`,qe),Y(`polygonPlan`,F),Y(`vertexA`,B),Z(`polygonStats`,k)],v,De(f)),J(ct,`triangulate`,`monotone-triangulate`,[Y(`pieceTable`,Ge),Y(`nextHalfEdge`,G),Y(`vertexA`,B),Y(`vertexB`,V),Y(`vertexPosition`,H,`f32`),X(`pieceScratch`,Je),X(`indices`,o.indices),Z(`polygonStats`,k)],v,Le(Oe(f))),q(`final`,`monotone-final`,[X(`routing`,a),X(`polygonStats`,k),X(`valid`,o.valid),...o.triangleCount?[X(`triangleCount`,o.triangleCount)]:[]],h,ke(f,!!o.triangleCount)),J(Qe,`pad`,`monotone-pad`,[Y(`jInfo`,R),Y(`polygonOffsets`,s),Y(`ringOffsets`,l),Y(`routedOffsets`,O),Y(`polygonStats`,k),X(`indices`,o.indices)],p,Ae(f)),y}function Le(e){return{declarations:`${_}${e.declarations}`,body:e.body}}function H(e,t){let{id:n,positions:r,polygonOffsets:i,ringOffsets:a,triangleOffsets:o}=t,s=i.length-1,l=a.length-1,d=`GPUPolygonTriangulation`,f={polygonOffsets:c(e,`${n}-polygon-layout`,`uint32`,4*s+2),ringOffsets:c(e,`${n}-ring-layout`,`uint32`,2*l+1)},p=c(e,`${n}-triangle-capacities`,`uint32`,s+1);return{nodes:[m(e,{id:`${n}-classify-rings`,operation:d,variant:`classify-rings`,bindings:[{name:`positions`,view:r,type:`f32`,access:`read`},{name:`ringOffsets`,view:a,type:`u32`,access:`read`},{name:`ringLayout`,view:f.ringOffsets,type:`u32`,access:`read_write`}],invocationCount:l+1,declarations:`${_}
fn readVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}`,body:`let first = ringOffsets[ringOffsetsOffset + index];
  ringLayout[ringLayoutOffset + index] = first;
  if (index == ${l}u) { return; }
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], ${r.length}u);
  var nonDegenerate = 0u;
  if (end >= first + 3u) {
    let origin = readVertex(first);
    for (var vertex = first + 1u; vertex + 1u < end; vertex++) {
      // Match study-window preparation: uncertified (2) is not exact zero.
      if (orientSign(origin, readVertex(vertex), readVertex(vertex + 1u)) != 0) {
        nonDegenerate = 1u;
        break;
      }
    }
  }
  ringLayout[ringLayoutOffset + ${l+1}u + index] = nonDegenerate;`}),m(e,{id:`${n}-polygon-capacities`,operation:d,variant:`polygon-capacities`,bindings:[{name:`polygonOffsets`,view:i,type:`u32`,access:`read`},{name:`ringLayout`,view:f.ringOffsets,type:`u32`,access:`read`},{name:`polygonLayout`,view:f.polygonOffsets,type:`u32`,access:`read_write`},{name:`counts`,view:p,type:`u32`,access:`read_write`}],invocationCount:s+1,body:`let first = polygonOffsets[polygonOffsetsOffset + index];
  polygonLayout[polygonLayoutOffset + index] = first;
  counts[countsOffset + index] = 0u;
  if (index == ${s}u) { return; }
  let end = polygonOffsets[polygonOffsetsOffset + index + 1u];
  var vertices = 0u;
  var rings = 0u;
  if (first < end && ringLayout[ringLayoutOffset + ${l+1}u + first] != 0u) {
    for (var ring = first; ring < end; ring++) {
      if (ringLayout[ringLayoutOffset + ${l+1}u + ring] == 0u) { continue; }
      vertices += ringLayout[ringLayoutOffset + ring + 1u] - ringLayout[ringLayoutOffset + ring];
      rings += 1u;
    }
  }
  if (rings > 0u) { counts[countsOffset + index] = vertices + 2u * rings - 4u; }
  polygonLayout[polygonLayoutOffset + ${2*(s+1)}u + index] = vertices;
  polygonLayout[polygonLayoutOffset + ${3*s+2}u + index] = rings;`}),...new u({id:`${n}-triangle-offsets`,input:p,output:o,mode:`exclusive`}).getCommandNodes(e),m(e,{id:`${n}-publish-layout`,operation:d,variant:`publish-layout`,bindings:[{name:`triangleOffsets`,view:o,type:`u32`,access:`read`},{name:`polygonLayout`,view:f.polygonOffsets,type:`u32`,access:`read_write`}],invocationCount:s+1,body:`polygonLayout[polygonLayoutOffset + ${s+1}u + index] = triangleOffsets[triangleOffsetsOffset + index];`})],layout:f}}var U=`GPUPolygonTriangulation`;function Re(e,t,n){return n>0?Math.max(0,3*(e+2*t-4)):0}function ze(e,t=!0){return o(e,t?28:16)}function W(e,t,n){return 2*(e+2*(t-n))+8*n}function Be(e,t,n,r,i,a=!0){s(t,`scratch nodes`,W(n,r,i),ze(e,a))}var Ve=80,G=class{id;props;spatialContext;polygonCount;ringCount;indexCount;constructor(t){this.id=t.id??`polygon-triangulation`,this.spatialContext=e(this.id,t.spatialContext),this.props=t;let{id:n}=this,{output:i}=t;l(t.positions,[`float32x2`],`${n} positions`);for(let e of[`featureOffsets`,`polygonOffsets`,`ringOffsets`])if(l(t[e],[`uint32`],`${n} ${e}`),t[e].length<1)throw Error(`${n} ${e} requires a terminal entry`);this.polygonCount=t.polygonOffsets.length-1,this.ringCount=t.ringOffsets.length-1,this.indexCount=Re(t.positions.length,this.ringCount,this.polygonCount),l(i.indices,[`uint32`],`${n} output.indices`);let a=Math.max(0,3*(t.positions.length+2*this.ringCount-4*this.polygonCount));if(i.indices.length<a)throw Error(`${n} output.indices needs at least ${a} rows`);if(i.triangleOffsets&&(l(i.triangleOffsets,[`uint32`],`${n} output.triangleOffsets`),i.triangleOffsets.length!==this.polygonCount+1))throw Error(`${n} triangleOffsets length mismatch`);if(l(i.valid,[`uint32`],`${n} output.valid`),i.valid.length!==this.polygonCount)throw Error(`${n} output.valid length must equal the polygon count`);if(i.triangleCount&&(l(i.triangleCount,[`uint32`],`${n} output.triangleCount`),i.triangleCount.length!==this.polygonCount))throw Error(`${n} output.triangleCount length must equal the polygon count`);if(t.maximumWork!==void 0&&r(n,`maximumWork`,t.maximumWork),t.largePolygonMethod!==void 0&&![`monotone`,`none`].includes(t.largePolygonMethod))throw Error(`${n} largePolygonMethod must be 'monotone' or 'none'`);t.monotoneMinimumVertices!==void 0&&r(n,`monotoneMinimumVertices`,t.monotoneMinimumVertices),t.monotoneMaximumVertices!==void 0&&r(n,`monotoneMaximumVertices`,t.monotoneMaximumVertices)}getViews(){let{positions:e,featureOffsets:t,polygonOffsets:n,ringOffsets:r,output:i}=this.props,{indices:a,valid:o,triangleCount:s,triangleOffsets:c}=i;return[e,t,n,r,a,o,s,c]}getCommandNodes(e){let{id:t,props:r,polygonCount:i,ringCount:a}=this,{output:o}=r;n(t,e,this.getViews()),Be(e.device,t,r.positions.length,a,i,r.useZOrderHash??!0);let s=r.positions.length,l=W(s,a,i),u=r.useZOrderHash??!0,d=c(e,`${t}-scratch`,`uint32`,Math.max((u?7:4)*l,4)),p=(r.largePolygonMethod??`monotone`)===`monotone`&&s<=je(e.device),h=o.triangleOffsets??c(e,`${t}-polygon-triangle-offsets`,`uint32`,i+1),{nodes:g,layout:y}=H(e,{id:t,positions:r.positions,polygonOffsets:r.polygonOffsets,ringOffsets:r.ringOffsets,triangleOffsets:h});if(g.push(f(e,{id:`${t}-clear-indices`,operation:U,view:o.indices,type:`u32`,value:`0u`})),i===0)return g;let ee=[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:y.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:y.polygonOffsets,type:`u32`,access:`read`},{name:`indices`,view:o.indices,type:`u32`,access:`read_write`},{name:`valid`,view:o.valid,type:`u32`,access:`read_write`},{name:`scratch`,view:d,type:`u32`,access:`read_write`}];o.triangleCount&&ee.push({name:`triangleCount`,view:o.triangleCount,type:`u32`,access:`read_write`});let b=p?c(e,`${t}-routing`,`uint32`,Math.max(i,1)):void 0;b&&ee.push({name:`routing`,view:b,type:`u32`,access:`read_write`});let x=[];if(b){let n={id:`${t}-monotone`,operation:U,positions:r.positions,polygonOffsets:r.polygonOffsets,ringOffsets:r.ringOffsets,routing:b,layout:y,output:o,capacity:Math.max(1,Math.min(r.monotoneMaximumVertices??s,s)),minimumVertices:r.monotoneMinimumVertices??4096};r.monotoneMaximumVertices!==void 0&&B(e.device,t,r.monotoneMaximumVertices),x=V(e,n)}let S=r.monotoneMinimumVertices??4096,C=(t,n)=>m(e,{id:t,operation:U,variant:`earcut`,bindings:ee,invocationCount:i,workgroupSize:64,declarations:`${_}
// Only sosPointSide and exactCompareSign are called; exactOrientSigns (the helper's orientSign
// call site) is never called here and is dropped as dead code.
${v}
${He({vertexCount:s,polygonCount:i,ringCount:a,indexLength:o.indices.length,nodeCount:l,useZOrderHash:u,maximumWork:r.maximumWork??4e6,writeTriangleCount:!!o.triangleCount,routingPass:n,minimumVertices:S})}`,body:`triangulatePolygon(index);`});return g.push(C(`${t}-earcut`,b?`first`:void 0)),b&&x.push(C(`${t}-earcut-fallback`,`fallback`)),[...g,...x]}};function He(e){return`
const VERTEX_COUNT: u32 = ${e.vertexCount}u;
const POLYGON_COUNT: u32 = ${e.polygonCount}u;
const INDEX_LENGTH: u32 = ${e.indexLength}u;
const SCRATCH_NODE_COUNT: u32 = ${e.nodeCount}u;
const Z_BASE: u32 = ${4*e.nodeCount}u;
const USE_Z_HASH: bool = ${e.useZOrderHash};
const Z_HASH_MINIMUM_VERTICES: u32 = ${Ve}u;
const WORK_LIMIT: u32 = ${e.maximumWork}u;
const NONE: u32 = 0xffffffffu;
${e.routingPass===`first`?z(e.minimumVertices,!0):``}
var<private> nodeBase: u32;
var<private> nodeCursor: u32;
var<private> nodeLimit: u32;
var<private> workDone: u32;
var<private> failed: bool;
var<private> incomplete: bool;
var<private> filteredOut: bool;
var<private> triangleBase: u32;
var<private> triangleSlots: u32;
var<private> trianglesWritten: u32;
var<private> hashed: bool;
var<private> hashMinimum: vec2f;
var<private> hashScale: f32;

// Node n lives in scratch words 4n..4n+3: next, previous, vertex row, spare (hole queue, ring stack).
fn nextOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node]; }
fn prevOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node + 1u]; }
fn vertexOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node + 2u]; }
fn setNext(node: u32, value: u32) { scratch[scratchOffset + 4u * node] = value; }
// z-order list of node n: z, previous in z order, next in z order (NONE-terminated).
fn zOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node]; }
fn prevZOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node + 1u]; }
fn nextZOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node + 2u]; }
fn setZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node] = value; }
fn setPrevZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node + 1u] = value; }
fn setNextZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node + 2u] = value; }
fn setPrev(node: u32, value: u32) { scratch[scratchOffset + 4u * node + 1u] = value; }
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn pointOf(node: u32) -> vec2f { return vertexAt(vertexOf(node)); }
fn samePoint(first: vec2f, second: vec2f) -> bool { return first.x == second.x && first.y == second.y; }
fn spend(amount: u32) {
  workDone += amount;
  if (workDone > WORK_LIMIT) { failed = true; }
}

// Sign of Earcut's area(p, q, r), which is minus the orientation: negative at a convex vertex
// of the (counter-clockwise) working order. Uncertain predicates count as zero.
fn areaSign(p: vec2f, q: vec2f, r: vec2f) -> i32 {
  let result = orientSign(p, q, r);
  if (result == 2) { return 0; }
  return -result;
}
fn orientOf(p: vec2f, q: vec2f, r: vec2f) -> i32 {
  let result = orientSign(p, q, r);
  if (result == 2) { return 0; }
  return result;
}

fn createNode(vertex: u32) -> u32 {
  if (nodeCursor >= nodeLimit) {
    failed = true;
    return nodeBase;
  }
  let node = nodeCursor;
  nodeCursor += 1u;
  scratch[scratchOffset + 4u * node + 2u] = vertex;
  if (hashed) {
    setPrevZ(node, NONE);
    setNextZ(node, NONE);
  }
  return node;
}
fn insertNode(vertex: u32, last: u32) -> u32 {
  let node = createNode(vertex);
  if (last == NONE) {
    setPrev(node, node);
    setNext(node, node);
  } else {
    let following = nextOf(last);
    setNext(node, following);
    setPrev(node, last);
    setPrev(following, node);
    setNext(last, node);
  }
  return node;
}
fn removeNode(node: u32) {
  if (hashed) {
    let previousZ = prevZOf(node);
    let followingZ = nextZOf(node);
    if (previousZ != NONE) { setNextZ(previousZ, followingZ); }
    if (followingZ != NONE) { setPrevZ(followingZ, previousZ); }
  }
  setPrev(nextOf(node), prevOf(node));
  setNext(prevOf(node), nextOf(node));
}
fn equalNodes(first: u32, second: u32) -> bool { return samePoint(pointOf(first), pointOf(second)); }

// Winding of a ring at its lexicographically smallest vertex: 1 counter-clockwise, -1 clockwise,
// 0 undetermined, 2 uncertain.
fn ringWinding(start: u32, count: u32) -> i32 {
  var extreme = 0u;
  for (var k = 1u; k < count; k++) {
    let candidate = vertexAt(start + k);
    let best = vertexAt(start + extreme);
    if (candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y)) { extreme = k; }
  }
  let center = vertexAt(start + extreme);
  var previous = extreme;
  var next = extreme;
  var foundPrevious = false;
  var foundNext = false;
  for (var step = 1u; step < count; step++) {
    if (!foundPrevious) {
      previous = (extreme + count - step) % count;
      foundPrevious = !samePoint(vertexAt(start + previous), center);
    }
    if (!foundNext) {
      next = (extreme + step) % count;
      foundNext = !samePoint(vertexAt(start + next), center);
    }
  }
  if (!foundPrevious || !foundNext) { return 0; }
  return orientSign(vertexAt(start + previous), center, vertexAt(start + next));
}

// Circular list of a ring in counter-clockwise (shell) or clockwise (hole) working order.
fn linkedList(start: u32, count: u32, wantCounterClockwise: bool) -> u32 {
  if (count == 0u) { return NONE; }
  let winding = ringWinding(start, count);
  var forward = true;
  if (winding == 1 || winding == -1) { forward = (winding == 1) == wantCounterClockwise; }
  var last = NONE;
  for (var k = 0u; k < count; k++) {
    let row = select(start + count - 1u - k, start + k, forward);
    last = insertNode(row, last);
  }
  if (last != NONE && equalNodes(last, nextOf(last))) {
    removeNode(last);
    last = nextOf(last);
  }
  return last;
}

// Removes collinear and coincident nodes; with end == start the whole ring until stable.
fn filterPoints(start: u32, endIn: u32) -> u32 {
  var end = endIn;
  let full = end == start;
  var p = start;
  loop {
    if (failed) { break; }
    spend(1u);
    var again = false;
    let following = nextOf(p);
    if (p != following &&
        (equalNodes(p, following) || areaSign(pointOf(prevOf(p)), pointOf(p), pointOf(following)) == 0)) {
      if (full || p == end) { end = prevOf(p); }
      filteredOut = true;
      removeNode(p);
      p = prevOf(p);
      again = true;
    } else if (full || p != end) {
      p = following;
      again = !full;
    }
    if (!(again || p != end)) { break; }
  }
  return end;
}

fn pointInTriangle(a: vec2f, b: vec2f, c: vec2f, p: vec2f) -> bool {
  return orientOf(p, c, a) >= 0 && orientOf(p, a, b) >= 0 && orientOf(p, b, c) >= 0;
}

fn isEar(ear: u32) -> bool {
  let a = prevOf(ear);
  let c = nextOf(ear);
  let pa = pointOf(a);
  let pb = pointOf(ear);
  let pc = pointOf(c);
  let low = min(pa, min(pb, pc));
  let high = max(pa, max(pb, pc));
  var p = nextOf(c);
  while (p != a) {
    spend(1u);
    if (failed) { return false; }
    let pp = pointOf(p);
    if (pp.x >= low.x && pp.x <= high.x && pp.y >= low.y && pp.y <= high.y && !samePoint(pa, pp) &&
        pointInTriangle(pa, pb, pc, pp) &&
        areaSign(pointOf(prevOf(p)), pp, pointOf(nextOf(p))) >= 0) {
      return false;
    }
    p = nextOf(p);
  }
  return true;
}

// Morton code of a point on a 32768 x 32768 grid over the shell's bounding box. Quantisation is
// monotone in each coordinate (and clamped), so every point inside a box has a code between the
// codes of the box's low and high corners.
fn spreadBits(value: u32) -> u32 {
  var x = value;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
fn zOrderOf(point: vec2f) -> u32 {
  let scaled = clamp((point - hashMinimum) * hashScale, vec2f(0.0), vec2f(32767.0));
  return spreadBits(u32(scaled.x)) | (spreadBits(u32(scaled.y)) << 1u);
}

// Earcut's indexCurves: every node of the ring gets a code and joins a list sorted by it (stable
// bottom-up merge sort of the linked list).
fn indexCurves(start: u32) {
  var p = start;
  loop {
    spend(1u);
    setZ(p, zOrderOf(pointOf(p)));
    setPrevZ(p, prevOf(p));
    setNextZ(p, nextOf(p));
    p = nextOf(p);
    if (p == start) { break; }
  }
  setNextZ(prevZOf(start), NONE);
  setPrevZ(start, NONE);
  var list = start;
  var inSize = 1u;
  loop {
    var left = list;
    list = NONE;
    var tail = NONE;
    var merges = 0u;
    while (left != NONE) {
      merges += 1u;
      var right = left;
      var leftSize = 0u;
      for (var i = 0u; i < inSize; i++) {
        leftSize += 1u;
        right = nextZOf(right);
        if (right == NONE) { break; }
      }
      var rightSize = inSize;
      while (leftSize > 0u || (rightSize > 0u && right != NONE)) {
        spend(1u);
        var element = NONE;
        if (leftSize != 0u && (rightSize == 0u || right == NONE || zOf(left) <= zOf(right))) {
          element = left;
          left = nextZOf(left);
          leftSize -= 1u;
        } else {
          element = right;
          right = nextZOf(right);
          rightSize -= 1u;
        }
        if (tail != NONE) { setNextZ(tail, element); } else { list = element; }
        setPrevZ(element, tail);
        tail = element;
      }
      left = right;
    }
    setNextZ(tail, NONE);
    if (failed || merges <= 1u) { break; }
    inSize *= 2u;
  }
}

fn blocksEar(p: u32, a: u32, c: u32, pa: vec2f, pb: vec2f, pc: vec2f, low: vec2f, high: vec2f) -> bool {
  spend(1u);
  let pp = pointOf(p);
  return p != a && p != c && pp.x >= low.x && pp.x <= high.x && pp.y >= low.y && pp.y <= high.y &&
    !samePoint(pa, pp) && pointInTriangle(pa, pb, pc, pp) &&
    areaSign(pointOf(prevOf(p)), pp, pointOf(nextOf(p))) >= 0;
}

// Same decision as isEar, but only visits the nodes whose z-order code lies in the code range of
// the triangle's bounding box (Earcut's isEarHashed).
fn isEarHashed(ear: u32) -> bool {
  let a = prevOf(ear);
  let c = nextOf(ear);
  let pa = pointOf(a);
  let pb = pointOf(ear);
  let pc = pointOf(c);
  let low = min(pa, min(pb, pc));
  let high = max(pa, max(pb, pc));
  let minZ = zOrderOf(low);
  let maxZ = zOrderOf(high);
  var before = prevZOf(ear);
  var after = nextZOf(ear);
  while (before != NONE && zOf(before) >= minZ && after != NONE && zOf(after) <= maxZ) {
    if (failed) { return false; }
    if (blocksEar(before, a, c, pa, pb, pc, low, high)) { return false; }
    before = prevZOf(before);
    if (blocksEar(after, a, c, pa, pb, pc, low, high)) { return false; }
    after = nextZOf(after);
  }
  while (before != NONE && zOf(before) >= minZ) {
    if (failed) { return false; }
    if (blocksEar(before, a, c, pa, pb, pc, low, high)) { return false; }
    before = prevZOf(before);
  }
  while (after != NONE && zOf(after) <= maxZ) {
    if (failed) { return false; }
    if (blocksEar(after, a, c, pa, pb, pc, low, high)) { return false; }
    after = nextZOf(after);
  }
  return true;
}

fn isEarTest(ear: u32) -> bool {
  if (hashed) { return isEarHashed(ear); }
  return isEar(ear);
}

// Bounding-box test only: the caller has already established collinearity with areaSign().
fn onSegment(p: vec2f, q: vec2f, r: vec2f) -> bool {
  return q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) && q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y);
}
fn segmentsIntersect(p1: vec2f, q1: vec2f, p2: vec2f, q2: vec2f, includeBoundary: bool) -> bool {
  let o1 = areaSign(p1, q1, p2);
  let o2 = areaSign(p1, q1, q2);
  let o3 = areaSign(p2, q2, p1);
  let o4 = areaSign(p2, q2, q1);
  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) { return true; }
  if (!includeBoundary) { return false; }
  if (o1 == 0 && onSegment(p1, p2, q1)) { return true; }
  if (o2 == 0 && onSegment(p1, q2, q1)) { return true; }
  if (o3 == 0 && onSegment(p2, p1, q2)) { return true; }
  if (o4 == 0 && onSegment(p2, q1, q2)) { return true; }
  return false;
}

fn locallyInside(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let before = pointOf(prevOf(a));
  let after = pointOf(nextOf(a));
  if (areaSign(before, pa, after) < 0) {
    return areaSign(pa, pb, after) >= 0 && areaSign(pa, before, pb) >= 0;
  }
  return areaSign(pa, pb, before) < 0 || areaSign(pa, after, pb) < 0;
}

// Whether the diagonal midpoint is inside the working polygon: a topology decision, so the parity
// of a ray toward +x uses the exact rule of utils/exact-geometry-wgsl.ts. The midpoint is only a
// rounded f32 placement; its side of each straddling edge is the exact orientSign (through
// orientOf, no new orientSign call site) with Simulation-of-Simplicity ties from sosPointSide.
fn middleInside(a: u32, b: u32) -> bool {
  let middle = (pointOf(a) + pointOf(b)) * 0.5;
  var p = a;
  var inside = false;
  loop {
    spend(1u);
    if (failed) { return false; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    if ((edgeStart.y >= middle.y) != (edgeEnd.y >= middle.y)) {
      // orientOf is orientSign (positive when the midpoint is left of edgeStart -> edgeEnd) with
      // uncertain counted as zero, which sosPointSide then breaks by perturbation.
      let side = sosPointSide(orientOf(edgeStart, edgeEnd, middle), edgeStart, edgeEnd);
      let isUpward = edgeEnd.y > edgeStart.y;
      if ((isUpward && side > 0) || (!isUpward && side < 0)) {
        inside = !inside;
      }
    }
    p = following;
    if (p == a) { break; }
  }
  return inside;
}

fn intersectsPolygon(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let low = min(pa, pb);
  let high = max(pa, pb);
  let indexA = vertexOf(a);
  let indexB = vertexOf(b);
  var p = a;
  loop {
    spend(1u);
    if (failed) { return true; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    let outside = (edgeStart.x > high.x && edgeEnd.x > high.x) || (edgeStart.x < low.x && edgeEnd.x < low.x) ||
      (edgeStart.y > high.y && edgeEnd.y > high.y) || (edgeStart.y < low.y && edgeEnd.y < low.y);
    if (!outside) {
      let indexFrom = vertexOf(p);
      let indexTo = vertexOf(following);
      if (indexFrom != indexA && indexTo != indexA && indexFrom != indexB && indexTo != indexB &&
          segmentsIntersect(edgeStart, edgeEnd, pa, pb, true)) {
        return true;
      }
    }
    p = following;
    if (p == a) { break; }
  }
  return false;
}

fn isValidDiagonal(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let zeroLength = samePoint(pa, pb) &&
    areaSign(pointOf(prevOf(a)), pa, pointOf(nextOf(a))) > 0 &&
    areaSign(pointOf(prevOf(b)), pb, pointOf(nextOf(b))) > 0;
  if (vertexOf(nextOf(a)) == vertexOf(b)) { return false; }
  if (!(zeroLength || (locallyInside(a, b) && locallyInside(b, a) &&
      (areaSign(pointOf(prevOf(a)), pa, pointOf(prevOf(b))) != 0 ||
       areaSign(pa, pointOf(prevOf(b)), pb) != 0)))) {
    return false;
  }
  if (intersectsPolygon(a, b)) { return false; }
  return zeroLength || middleInside(a, b);
}

// Splits (a, b) in two rings (or merges a hole ring into the shell); returns b's duplicate.
fn splitPolygon(a: u32, b: u32) -> u32 {
  let a2 = createNode(vertexOf(a));
  let b2 = createNode(vertexOf(b));
  let an = nextOf(a);
  let bp = prevOf(b);
  setNext(a, b);
  setPrev(b, a);
  setNext(a2, an);
  setPrev(an, a2);
  setNext(b2, a2);
  setPrev(a2, b2);
  setNext(bp, b2);
  setPrev(b2, bp);
  return b2;
}

fn emitTriangle(first: u32, second: u32, third: u32) {
  if (trianglesWritten >= triangleSlots) {
    failed = true;
    return;
  }
  let slot = triangleBase + 3u * trianglesWritten;
  indices[indicesOffset + slot] = first;
  indices[indicesOffset + slot + 1u] = second;
  indices[indicesOffset + slot + 2u] = third;
  trianglesWritten += 1u;
}

fn cureLocalIntersections(startIn: u32) -> u32 {
  var start = startIn;
  var p = start;
  var cured = false;
  loop {
    spend(1u);
    if (failed) { break; }
    let a = prevOf(p);
    let following = nextOf(p);
    let b = nextOf(following);
    if (segmentsIntersect(pointOf(a), pointOf(p), pointOf(following), pointOf(b), false) &&
        locallyInside(a, b) && locallyInside(b, a)) {
      emitTriangle(vertexOf(a), vertexOf(p), vertexOf(b));
      removeNode(p);
      removeNode(following);
      start = b;
      p = b;
      cured = true;
    }
    p = nextOf(p);
    if (p == start) { break; }
  }
  if (cured) { return filterPoints(p, p); }
  return p;
}

// First valid diagonal of the ring as (a, b), or (NONE, NONE).
fn findSplit(start: u32) -> vec2<u32> {
  var a = start;
  loop {
    var b = nextOf(nextOf(a));
    while (b != prevOf(a)) {
      spend(1u);
      if (failed) { return vec2<u32>(NONE, NONE); }
      if (vertexOf(a) != vertexOf(b) && isValidDiagonal(a, b)) { return vec2<u32>(a, b); }
      b = nextOf(b);
    }
    a = nextOf(a);
    if (a == start) { break; }
  }
  return vec2<u32>(NONE, NONE);
}

fn pushRing(depth: u32, head: u32) {
  if (nodeBase + depth >= nodeLimit) {
    failed = true;
    return;
  }
  scratch[scratchOffset + 4u * (nodeBase + depth) + 3u] = head;
}
fn popRing(depth: u32) -> u32 {
  return scratch[scratchOffset + 4u * (nodeBase + depth) + 3u];
}

// Earcut's earcutLinked, with the recursion of splitEarcut replaced by a stack of ring heads.
fn clipEars(first: u32) {
  var ear = first;
  var depth = 0u;
  loop {
    if (hashed) { indexCurves(ear); }
    var stop = ear;
    var cured = false;
    loop {
      if (failed) { break; }
      spend(1u);
      let before = prevOf(ear);
      let after = nextOf(ear);
      if (before == after) { break; }
      var earFound = false;
      if (areaSign(pointOf(before), pointOf(ear), pointOf(after)) < 0) {
        earFound = isEarTest(ear);
      }
      if (earFound) {
        emitTriangle(vertexOf(before), vertexOf(ear), vertexOf(after));
        removeNode(ear);
        ear = after;
        stop = after;
        continue;
      }
      ear = after;
      if (ear == stop) {
        filteredOut = false;
        ear = filterPoints(ear, ear);
        if (filteredOut) {
          stop = ear;
          continue;
        }
        if (!cured) {
          ear = cureLocalIntersections(ear);
          stop = ear;
          cured = true;
          continue;
        }
        let diagonal = findSplit(ear);
        if (diagonal.x == NONE) {
          incomplete = true;
          break;
        }
        let other = splitPolygon(diagonal.x, diagonal.y);
        let keptHead = filterPoints(diagonal.x, nextOf(diagonal.x));
        let otherHead = filterPoints(other, nextOf(other));
        pushRing(depth, otherHead);
        depth += 1u;
        ear = keptHead;
        stop = keptHead;
        cured = false;
      }
    }
    if (failed || incomplete || depth == 0u) { break; }
    depth -= 1u;
    ear = popRing(depth);
  }
}

fn isLeftOf(first: u32, second: u32) -> bool {
  let a = pointOf(first);
  let b = pointOf(second);
  if (a.x != b.x) { return a.x < b.x; }
  if (a.y != b.y) { return a.y < b.y; }
  let nextA = pointOf(nextOf(first)) - a;
  let nextB = pointOf(nextOf(second)) - b;
  let slopeA = select(select(-1.0e30, 1.0e30, nextA.y > 0.0), nextA.y / nextA.x, nextA.x != 0.0);
  let slopeB = select(select(-1.0e30, 1.0e30, nextB.y > 0.0), nextB.y / nextB.x, nextB.x != 0.0);
  return slopeA < slopeB;
}

fn sectorContainsSector(m: u32, p: u32) -> bool {
  let pm = pointOf(m);
  return areaSign(pointOf(prevOf(m)), pm, pointOf(prevOf(p))) < 0 &&
    areaSign(pointOf(nextOf(p)), pm, pointOf(nextOf(m))) < 0;
}

// David Eberly's bridge between a hole's leftmost vertex and the outer ring, or NONE. The f32
// ray intersection x below only places the candidate bridge vertex (it is not a parity or
// containment decision); validity of the resulting diagonal is decided exactly by isValidDiagonal.
fn findHoleBridge(hole: u32, outerNode: u32) -> u32 {
  let hp = pointOf(hole);
  var p = outerNode;
  var qx = -3.0e38;
  var m = NONE;
  if (equalNodes(hole, p)) { return p; }
  loop {
    spend(1u);
    if (failed) { return NONE; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    if (equalNodes(hole, following)) { return following; }
    if (hp.y <= edgeStart.y && hp.y >= edgeEnd.y && edgeEnd.y != edgeStart.y) {
      let x = edgeStart.x + (hp.y - edgeStart.y) * (edgeEnd.x - edgeStart.x) / (edgeEnd.y - edgeStart.y);
      if (x <= hp.x && x > qx) {
        qx = x;
        m = select(following, p, edgeStart.x < edgeEnd.x);
        if (x == hp.x) { return m; }
      }
    }
    p = following;
    if (p == outerNode) { break; }
  }
  if (m == NONE) { return NONE; }
  let pm = pointOf(m);
  let stop = m;
  var tanMin = 3.0e38;
  p = m;
  loop {
    spend(1u);
    if (failed) { return NONE; }
    let pp = pointOf(p);
    if (hp.x >= pp.x && pp.x >= pm.x && hp.x != pp.x) {
      let corner = select(vec2f(qx, hp.y), vec2f(hp.x, hp.y), hp.y < pm.y);
      let corner2 = select(vec2f(hp.x, hp.y), vec2f(qx, hp.y), hp.y < pm.y);
      if (pointInTriangle(corner, pm, corner2, pp)) {
        let tangent = abs(hp.y - pp.y) / (hp.x - pp.x);
        let touchesEdge = pp.y == hp.y && pointOf(nextOf(p)).y == hp.y && pointOf(nextOf(p)).x > hp.x;
        if ((locallyInside(p, hole) || touchesEdge) &&
            (tangent < tanMin || (tangent == tanMin &&
              (pp.x > pm.x || (pp.x == pm.x && sectorContainsSector(m, p)))))) {
          m = p;
          tanMin = tangent;
        }
      }
    }
    p = nextOf(p);
    if (p == stop) { break; }
  }
  return m;
}

fn getLeftmost(start: u32) -> u32 {
  var leftmost = start;
  var p = start;
  loop {
    spend(1u);
    if (failed) { return leftmost; }
    let a = pointOf(p);
    let b = pointOf(leftmost);
    if (a.x < b.x || (a.x == b.x && a.y < b.y)) { leftmost = p; }
    p = nextOf(p);
    if (p == start) { break; }
  }
  return leftmost;
}

// Links every hole ring (rings ringFirst + 1 ..< ringLast) into the outer ring.
fn eliminateHoles(ringFirst: u32, ringLast: u32, outerIn: u32) -> u32 {
  var outerNode = outerIn;
  var queued = 0u;
  for (var ring = ringFirst + 1u; ring < ringLast; ring++) {
    let start = ringOffsets[ringOffsetsOffset + ring];
    let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (ringOffsets[ringOffsetsOffset + ${e.ringCount+1}u + ring] == 0u) { continue; }
    let list = linkedList(start, end - start, false);
    if (list == NONE || failed) { continue; }
    scratch[scratchOffset + 4u * (nodeBase + queued) + 3u] = getLeftmost(list);
    queued += 1u;
  }
  for (var round = 0u; round < queued; round++) {
    // Selection sort by Earcut's compareXYSlope: leftmost, then lowest, then smallest slope.
    var best = NONE;
    var bestSlot = 0u;
    for (var slot = 0u; slot < queued; slot++) {
      spend(1u);
      let candidate = scratch[scratchOffset + 4u * (nodeBase + slot) + 3u];
      if (candidate == NONE) { continue; }
      if (best == NONE || isLeftOf(candidate, best)) {
        best = candidate;
        bestSlot = slot;
      }
    }
    if (best == NONE || failed) { break; }
    scratch[scratchOffset + 4u * (nodeBase + bestSlot) + 3u] = NONE;
    let bridge = findHoleBridge(best, outerNode);
    if (bridge == NONE) { continue; }
    let bridgeReverse = splitPolygon(bridge, best);
    filterPoints(bridgeReverse, nextOf(bridgeReverse));
    outerNode = filterPoints(bridge, nextOf(bridge));
  }
  return filterPoints(outerNode, outerNode);
}

fn baseNodeOf(polygon: u32) -> i32 {
  let ringPrefix = i32(polygonOffsets[polygonOffsetsOffset + polygon]);
  let vertexPrefix = i32(ringOffsets[ringOffsetsOffset + u32(ringPrefix)]);
  return 2 * (vertexPrefix + 2 * (ringPrefix - i32(polygon))) + 8 * i32(polygon);
}
fn baseSlotOf(polygon: u32) -> i32 {
  return 3 * i32(polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT + 1u + polygon]);
}

fn triangulatePolygon(polygon: u32) {
  ${e.routingPass===`first`?`routing[routingOffset + polygon] = sizeRouteOf(polygon);
  if (routing[routingOffset + polygon] == 1u) { return; }`:``}
  ${e.routingPass===`fallback`?`if (routing[routingOffset + polygon] != 4u) { return; }`:``}
  failed = false;
  incomplete = false;
  filteredOut = false;
  workDone = 0u;
  trianglesWritten = 0u;
  let slotStart = baseSlotOf(polygon);
  let slotEnd = min(baseSlotOf(polygon + 1u), i32(INDEX_LENGTH));
  let slotCount = max(slotEnd - slotStart, 0);
  triangleBase = u32(max(slotStart, 0));
  triangleSlots = u32(slotCount) / 3u;
  let nodeStart = baseNodeOf(polygon);
  let nodeEnd = min(baseNodeOf(polygon + 1u), i32(SCRATCH_NODE_COUNT));
  nodeBase = u32(max(nodeStart, 0));
  nodeCursor = nodeBase;
  nodeLimit = u32(max(nodeEnd, nodeStart));
  let ringFirst = polygonOffsets[polygonOffsetsOffset + polygon];
  let ringLast = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  let shellStart = ringOffsets[ringOffsetsOffset + ringFirst];
  var shellEnd = shellStart;
  if (ringLast > ringFirst) {
    shellEnd = min(ringOffsets[ringOffsetsOffset + ringFirst + 1u], VERTEX_COUNT);
  }
  let padVertex = shellStart;
  let liveVertices = polygonOffsets[polygonOffsetsOffset + 2u * (POLYGON_COUNT + 1u) + polygon];
  let liveRings = polygonOffsets[polygonOffsetsOffset + 3u * POLYGON_COUNT + 2u + polygon];
  hashed = USE_Z_HASH && liveVertices > Z_HASH_MINIMUM_VERTICES;
  if (liveRings == 0u) {
    valid[validOffset + polygon] = 1u;
    ${e.writeTriangleCount?`triangleCount[triangleCountOffset + polygon] = 0u;`:``}
    return;
  }
  if (baseSlotOf(polygon + 1u) > i32(INDEX_LENGTH) || triangleSlots == 0u ||
      nodeLimit < nodeBase + ringLast - ringFirst) {
    failed = true;
  } else {
    var outerNode = linkedList(shellStart, shellEnd - shellStart, true);
    if (outerNode == NONE || nextOf(outerNode) == prevOf(outerNode)) {
      failed = true;
    } else {
      if (hashed) {
        var low = vertexAt(shellStart);
        var high = low;
        for (var row = shellStart + 1u; row < shellEnd; row++) {
          low = min(low, vertexAt(row));
          high = max(high, vertexAt(row));
        }
        let extent = max(high.x - low.x, high.y - low.y);
        hashMinimum = low;
        hashScale = select(0.0, 32767.0 / extent, extent > 0.0);
      }
      if (liveRings > 1u) { outerNode = eliminateHoles(ringFirst, ringLast, outerNode); }
      clipEars(outerNode);
    }
  }
  let ok = !failed && !incomplete;
  ${e.routingPass===`first`?`if (!ok && ringLast > ringFirst && shellEnd >= shellStart + 3u) { routing[routingOffset + polygon] = 2u; }`:``}
  let written = select(0u, trianglesWritten, ok);
  // Failed polygons overwrite everything with degenerate triangles; partial output is unusable.
  for (var slot = written; slot < triangleSlots; slot++) {
    let base = indicesOffset + triangleBase + 3u * slot;
    indices[base] = padVertex;
    indices[base + 1u] = padVertex;
    indices[base + 2u] = padVertex;
  }
  valid[validOffset + polygon] = select(0u, 1u, ok);
  ${e.writeTriangleCount?`triangleCount[triangleCountOffset + polygon] = written;`:``}
}
`}function Ue(e,t=new Uint32Array(8)){if(t.length<8)throw Error(`Dot density parameter target must hold 8 elements`);if(!Number.isInteger(e.seed)||e.seed<0||e.seed>4294967295)throw Error(`Dot density seed must be a uint32 integer`);let n=new Float32Array(t.buffer,t.byteOffset,t.length),r=e.maskExtent??[0,0,1,1];return t[0]=e.seed,n[1]=e.dotsPerUnit??1,n[2]=r[0],n[3]=r[1],n[4]=r[2],n[5]=r[3],t[6]=0,t[7]=0,t}var We=`0xffffffffu`,K=4;function Ge(e){return o(e,K*4)}var Ke=`4.0e9`;function qe(e){let{id:t,output:n,mask:o}=e;l(e.positions,[`float32x2`],`${t} positions`),l(e.featureOffsets,[`uint32`],`${t} featureOffsets`),l(e.polygonOffsets,[`uint32`],`${t} polygonOffsets`),l(e.ringOffsets,[`uint32`],`${t} ringOffsets`);let s=e.featureOffsets.length-1;if(s<1)throw Error(`${t} featureOffsets must describe at least one feature`);r(t,`categoryCount`,e.categoryCount);let c=s*e.categoryCount;if(e.source.kind===`values`){if(l(e.source.values,[`float32`],`${t} values`),e.source.values.length!==c)throw Error(`${t} values must have featureCount * categoryCount rows`)}else if(l(e.source.counts,[`uint32`],`${t} counts`),e.source.counts.length!==s)throw Error(`${t} counts must have one row per feature`);if(l(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<8)throw Error(`${t} parameters must contain 8 uint32 rows`);if(r(t,`maximumAttempts`,e.maximumAttempts),o&&(l(o.weights,[`float32`],`${t} mask.weights`),!Number.isInteger(o.width)||!Number.isInteger(o.height)||o.width<1||o.height<1||o.weights.length!==o.width*o.height))throw Error(`${t} mask.weights must have mask.width * mask.height rows`);l(n.positions,[`float32x2`],`${t} output.positions`),i(t,n.dots);let u=n.positions.length;if(u<1||n.dots.ids.length!==u)throw Error(`${t} output.positions and output.dots.ids must share a positive capacity`);if(n.categories&&(l(n.categories,[`uint32`],`${t} output.categories`),n.categories.length!==u))throw Error(`${t} output.categories must have capacity rows`);let d=n.status?.failedCount;if(d&&(l(d,[`uint32`],`${t} output.status.failedCount`),d.length<1))throw Error(`${t} output.status.failedCount must contain one uint32 row`);for(let[e,r]of[[`slotCounts`,n.slotCounts],[`slotOffsets`,n.slotOffsets]])if(r&&(l(r,[`uint32`],`${t} output.${e}`),r.length!==c))throw Error(`${t} output.${e} must have one row per slot`);return a(t,[n.positions,n.dots.ids,n.dots.status.count,n.dots.status.overflow,n.dots.status.requiredCount,n.categories,n.status?.failedCount,n.slotCounts,n.slotOffsets],Je(e)),s}function Je(e){return[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets,e.source.kind===`values`?e.source.values:e.source.counts,e.parameters,...e.mask?[e.mask.weights]:[]]}function Ye(e,t,r){let{id:i,operation:a,output:o,mask:l,categoryCount:d}=t;n(i,e,[...Je(t),o.positions,o.dots.ids,o.dots.status.count,o.dots.status.overflow,o.dots.status.requiredCount,o.categories,o.status?.failedCount,o.slotCounts,o.slotOffsets]);let _=r*d,v=o.positions.length;s(i,`output.positions rows`,v,Ge(e.device));let ee=ne(_,v),b=t.polygonOffsets.length-1,x=Math.floor(Re(t.positions.length,t.ringOffsets.length-1,b)/3),S=[],C=c(e,`${i}-triangle-indices`,`uint32`,Math.max(3*x,1)),T=c(e,`${i}-polygon-valid`,`uint32`,b),D=c(e,`${i}-polygon-triangle-offsets`,`uint32`,b+1),O=c(e,`${i}-triangle-areas`,`float32`,Math.max(x,1)),k=c(e,`${i}-cumulative-areas`,`uint32`,Math.max(x,1)),A=c(e,`${i}-feature-triangle-offsets`,`uint32`,r+1),j=c(e,`${i}-feature-valid`,`uint32`,r),M=c(e,`${i}-feature-areas`,`float32`,r),N=o.slotCounts??c(e,`${i}-slot-counts`,`uint32`,_),P=o.slotOffsets??c(e,`${i}-slot-offsets`,`uint32`,_),F=c(e,`${i}-clamped`,`uint32`,1),I=c(e,`${i}-total`,`uint32`,1),L=c(e,`${i}-dot-slots`,`uint32`,v*K),R=o.status?.failedCount??c(e,`${i}-failed`,`uint32`,1);S.push(...new G({id:`${i}-triangulation`,positions:t.positions,featureOffsets:t.featureOffsets,polygonOffsets:t.polygonOffsets,ringOffsets:t.ringOffsets,output:{indices:C,valid:T,triangleOffsets:D}}).getCommandNodes(e)),S.push(m(e,{id:`${i}-triangle-areas`,operation:a,variant:`triangle-areas`,bindings:[{name:`polygonPositions`,view:t.positions,type:`f32`,access:`read`},{name:`triangleIndices`,view:C,type:`u32`,access:`read`},{name:`polygonTriangleOffsets`,view:D,type:`u32`,access:`read`},{name:`triangleAreas`,view:O,type:`f32`,access:`read_write`}],invocationCount:Math.max(x,1),declarations:`
fn readVertex(vertex: u32) -> vec2<f32> {
  return vec2<f32>(polygonPositions[polygonPositionsOffset + 2u * vertex], polygonPositions[polygonPositionsOffset + 2u * vertex + 1u]);
}`,body:`
  var area = 0.0;
  if (index < polygonTriangleOffsets[polygonTriangleOffsetsOffset + ${b}u]) {
    let base = triangleIndicesOffset + 3u * index;
    let firstIndex = triangleIndices[base];
    let secondIndex = triangleIndices[base + 1u];
    let thirdIndex = triangleIndices[base + 2u];
    if (firstIndex == secondIndex && firstIndex == thirdIndex) {
      triangleAreas[triangleAreasOffset + index] = 0.0;
      return;
    }
    let first = readVertex(firstIndex);
    // Edge vectors from one vertex keep the cross product accurate far from the origin.
    let edgeOne = readVertex(triangleIndices[base + 1u]) - first;
    let edgeTwo = readVertex(triangleIndices[base + 2u]) - first;
    area = 0.5 * abs(edgeOne.x * edgeTwo.y - edgeOne.y * edgeTwo.x);
  }
  triangleAreas[triangleAreasOffset + index] = area;`})),S.push(m(e,{id:`${i}-feature-triangles`,operation:a,variant:`feature-triangles`,bindings:[{name:`featureOffsets`,view:t.featureOffsets,type:`u32`,access:`read`},{name:`polygonTriangleOffsets`,view:D,type:`u32`,access:`read`},{name:`polygonValid`,view:T,type:`u32`,access:`read`},{name:`featureTriangleOffsets`,view:A,type:`u32`,access:`read_write`},{name:`featureValid`,view:j,type:`u32`,access:`read_write`}],invocationCount:r+1,declarations:`
const FEATURE_COUNT: u32 = ${r}u;
const POLYGON_COUNT: u32 = ${b}u;
const TRIANGLE_COUNT: u32 = ${x}u;
fn getTriangleStart(polygon: u32) -> u32 {
  return min(polygonTriangleOffsets[polygonTriangleOffsetsOffset + polygon], TRIANGLE_COUNT);
}`,body:`
  let firstPolygon = min(featureOffsets[featureOffsetsOffset + index], POLYGON_COUNT);
  featureTriangleOffsets[featureTriangleOffsetsOffset + index] = getTriangleStart(firstPolygon);
  if (index < FEATURE_COUNT) {
    let endPolygon = min(featureOffsets[featureOffsetsOffset + index + 1u], POLYGON_COUNT);
    var valid = 1u;
    for (var polygon = firstPolygon; polygon < endPolygon; polygon = polygon + 1u) {
      valid = valid & polygonValid[polygonValidOffset + polygon];
    }
    featureValid[featureValidOffset + index] = valid;
  }`})),S.push(y(e,{id:`${i}-feature-areas`,operation:a,segmentCount:r,input:O,segmentOffsets:A,output:M,offsetsWithinInput:!0})),S.push(m(e,{id:`${i}-cumulative-areas`,operation:a,variant:`cumulative-areas`,bindings:[{name:`triangleAreas`,view:O,type:`f32`,access:`read`},{name:`featureTriangleOffsets`,view:A,type:`u32`,access:`read`},{name:`featureValid`,view:j,type:`u32`,access:`read`},{name:`featureAreas`,view:M,type:`f32`,access:`read`},{name:`cumulativeAreas`,view:k,type:`u32`,access:`read_write`}],invocationCount:r,declarations:h,body:`
  let first = featureTriangleOffsets[featureTriangleOffsetsOffset + index];
  let end = featureTriangleOffsets[featureTriangleOffsetsOffset + index + 1u];
  let featureArea = featureAreas[featureAreasOffset + index];
  let usable = featureValid[featureValidOffset + index] == 1u && isFiniteBits(featureArea) && featureArea > 0.0;
  var running = 0u;
  for (var triangle = first; triangle < end; triangle = triangle + 1u) {
    let area = triangleAreas[triangleAreasOffset + triangle];
    if (usable && isFiniteBits(area)) {
      running = running + u32(clamp(area / featureArea, 0.0, 1.0) * ${Ke});
    }
    cumulativeAreas[cumulativeAreasOffset + triangle] = running;
  }`})),S.push(f(e,{id:`${i}-clear-failed`,operation:a,view:R,type:`u32`,value:`0u`,componentCount:1})),S.push(f(e,{id:`${i}-clear-clamped`,operation:a,view:F,type:`u32`,value:`0u`}));let re=t.source.kind===`values`?{name:`sourceValues`,view:t.source.values,type:`f32`,access:`read`}:{name:`sourceCounts`,view:t.source.counts,type:`u32`,access:`read`};S.push(m(e,{id:`${i}-counts`,operation:a,variant:`counts`,bindings:[re,{name:`featureAreas`,view:M,type:`f32`,access:`read`},{name:`failed`,view:R,type:`atomic<u32>`,access:`read_write`},{name:`parameters`,view:t.parameters,type:`u32`,access:`read`},{name:`slotCounts`,view:N,type:`u32`,access:`read_write`},{name:`clampFlag`,view:F,type:`u32`,access:`read_write`}],invocationCount:_,declarations:`${w}
${g}
const SLOT_LIMIT: u32 = ${ee}u;
const CATEGORY_COUNT: u32 = ${d}u;`,body:t.source.kind===`values`?`
  let seed = parameters[parametersOffset];
  let dotsPerUnit = bitcast<f32>(parameters[parametersOffset + 1u]);
  let scaled = sourceValues[sourceValuesOffset + index] * dotsPerUnit;
  var dots = 0u;
  var clamped = false;
  if (!isNanBits(scaled) && scaled > 0.0) {
    let remainder = philoxUnitFloat(
      philox4x32(vec4<u32>(index, 0u, 0u, 0u), vec2<u32>(seed, ${te}u)).x
    );
    let ceiling = ceil(scaled - remainder);
    if (ceiling > f32(SLOT_LIMIT)) {
      dots = SLOT_LIMIT;
      clamped = true;
    } else if (ceiling > 0.0) {
      dots = u32(ceiling);
      if (dots > SLOT_LIMIT) {
        dots = SLOT_LIMIT;
        clamped = true;
      }
    }
  }
  if (featureAreas[featureAreasOffset + index / CATEGORY_COUNT] == 0.0) {
    atomicAdd(&failed[failedOffset], dots);
    dots = 0u;
  }
  slotCounts[slotCountsOffset + index] = dots;
  if (clamped) {
    clampFlag[clampFlagOffset] = 1u;
  }`:`
  _ = parameters[parametersOffset];
  var dots = sourceCounts[sourceCountsOffset + index];
  if (dots > SLOT_LIMIT) {
    dots = SLOT_LIMIT;
    clampFlag[clampFlagOffset] = 1u;
  }
  if (featureAreas[featureAreasOffset + index / CATEGORY_COUNT] == 0.0) {
    atomicAdd(&failed[failedOffset], dots);
    dots = 0u;
  }
  slotCounts[slotCountsOffset + index] = dots;`})),S.push(...new u({id:`${i}-slot-scan`,input:N,output:P,mode:`exclusive`}).getCommandNodes(e)),S.push(m(e,{id:`${i}-total`,operation:a,variant:`total`,bindings:[{name:`slotCounts`,view:N,type:`u32`,access:`read`},{name:`slotOffsets`,view:P,type:`u32`,access:`read`},{name:`totalOut`,view:I,type:`u32`,access:`read_write`}],invocationCount:1,body:`let last = ${_-1}u;
  totalOut[totalOutOffset] = slotOffsets[slotOffsetsOffset + last] + slotCounts[slotCountsOffset + last];`}));let z=[{name:`slotCounts`,view:N,type:`u32`,access:`read`},{name:`slotOffsets`,view:P,type:`u32`,access:`read`},{name:`total`,view:I,type:`u32`,access:`read`},{name:`featureTriangleOffsets`,view:A,type:`u32`,access:`read`},{name:`dotSlots`,view:L,type:`u32`,access:`read_write`},{name:`featureIds`,view:o.dots.ids,type:`u32`,access:`read_write`}];o.categories&&z.push({name:`categories`,view:o.categories,type:`u32`,access:`read_write`}),S.push(m(e,{id:`${i}-locate`,operation:a,variant:`locate`,bindings:z,invocationCount:v,declarations:`const SLOT_COUNT: u32 = ${_}u;
const CATEGORY_COUNT: u32 = ${d}u;`,body:`
  let count = min(total[totalOffset], ${v}u);
  let table = dotSlotsOffset + ${K}u * index;
  if (index >= count) {
    dotSlots[table] = ${We};
    dotSlots[table + 1u] = 0u;
    dotSlots[table + 2u] = 0u;
    dotSlots[table + 3u] = 0u;
    featureIds[featureIdsOffset + index] = ${We};
    ${o.categories?`categories[categoriesOffset + index] = ${We};`:``}
    return;
  }
  // First slot whose inclusive prefix exceeds the dot index; empty slots never match.
  var low = 0u;
  var high = SLOT_COUNT - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (slotOffsets[slotOffsetsOffset + middle] + slotCounts[slotCountsOffset + middle] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  let feature = low / CATEGORY_COUNT;
  dotSlots[table] = low;
  dotSlots[table + 1u] = index - slotOffsets[slotOffsetsOffset + low];
  dotSlots[table + 2u] = featureTriangleOffsets[featureTriangleOffsetsOffset + feature];
  dotSlots[table + 3u] = featureTriangleOffsets[featureTriangleOffsetsOffset + feature + 1u];
  featureIds[featureIdsOffset + index] = feature;
  ${o.categories?`categories[categoriesOffset + index] = low % CATEGORY_COUNT;`:``}`}));let ie=[{name:`polygonPositions`,view:t.positions,type:`f32`,access:`read`},{name:`triangleIndices`,view:C,type:`u32`,access:`read`},{name:`cumulativeAreas`,view:k,type:`u32`,access:`read`},{name:`dotSlots`,view:L,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:`u32`,access:`read`},{name:`positions`,view:o.positions,type:`f32`,access:`read_write`},{name:`failed`,view:R,type:`atomic<u32>`,access:`read_write`}];return l&&ie.push({name:`maskWeights`,view:l.weights,type:`f32`,access:`read`}),S.push(m(e,{id:`${i}-sample`,operation:a,variant:l?`sample-masked`:`sample`,bindings:ie,invocationCount:v,declarations:`${w}
const CATEGORY_COUNT: u32 = ${d}u;
// Without a mask the first candidate is always kept; with one, candidates are rejected by weight.
const MAXIMUM_ATTEMPTS: u32 = ${l?t.maximumAttempts:1}u;
fn readVertex(vertex: u32) -> vec2<f32> {
  return vec2<f32>(polygonPositions[polygonPositionsOffset + 2u * vertex], polygonPositions[polygonPositionsOffset + 2u * vertex + 1u]);
}
// Uniform point of a triangle: r1 = sqrt(u1), P = (1 - r1) A + r1 (1 - u2) B + r1 u2 C, written
// relative to A (the same point) for accuracy far from the origin.
fn getTrianglePoint(triangle: u32, u1: f32, u2: f32) -> vec2<f32> {
  let base = triangleIndicesOffset + 3u * triangle;
  let a = readVertex(triangleIndices[base]);
  let edgeB = readVertex(triangleIndices[base + 1u]) - a;
  let edgeC = readVertex(triangleIndices[base + 2u]) - a;
  return a + sqrt(u1) * ((1.0 - u2) * edgeB + u2 * edgeC);
}
${l?`${g}
const MASK_WIDTH: u32 = ${l.width}u;
const MASK_HEIGHT: u32 = ${l.height}u;
fn getMaskWeight(point: vec2<f32>) -> f32 {
  let origin = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 2u]), bitcast<f32>(parameters[parametersOffset + 3u]));
  let cell = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 4u]), bitcast<f32>(parameters[parametersOffset + 5u]));
  let local = (point - origin) / cell;
  if (isNanBits(local.x) || isNanBits(local.y) || local.x < 0.0 || local.y < 0.0 ||
      local.x >= f32(MASK_WIDTH) || local.y >= f32(MASK_HEIGHT)) {
    return 0.0;
  }
  let weight = maskWeights[maskWeightsOffset + u32(local.y) * MASK_WIDTH + u32(local.x)];
  return select(weight, 0.0, isNanBits(weight));
}`:``}`,body:`
  let table = dotSlotsOffset + ${K}u * index;
  let slot = dotSlots[table];
  var nanBits = 0x7fc00000u;
  var position = vec2<f32>(bitcast<f32>(nanBits));
  if (slot != ${We}) {
    let rank = dotSlots[table + 1u];
    let firstTriangle = dotSlots[table + 2u];
    let endTriangle = dotSlots[table + 3u];
    let seed = parameters[parametersOffset];
    // Fixed-point area of the whole feature; 0 when it failed to triangulate or has no area.
    var featureArea = 0u;
    if (endTriangle > firstTriangle) {
      featureArea = cumulativeAreas[cumulativeAreasOffset + endTriangle - 1u];
    }
    var accepted = false;
    if (featureArea > 0u) {
      for (var attempt = 0u; attempt < MAXIMUM_ATTEMPTS; attempt = attempt + 1u) {
        let random = philox4x32(vec4<u32>(slot, rank, attempt, 0u), vec2<u32>(seed, ${E}u));
        let areaTarget = min(u32(philoxUnitFloat(random.x) * f32(featureArea)), featureArea - 1u);
        // First triangle whose cumulative area exceeds the target; zero-area triangles never match.
        var low = firstTriangle;
        var high = endTriangle - 1u;
        while (low < high) {
          let middle = (low + high) / 2u;
          if (cumulativeAreas[cumulativeAreasOffset + middle] > areaTarget) {
            high = middle;
          } else {
            low = middle + 1u;
          }
        }
        let candidate = getTrianglePoint(low, philoxUnitFloat(random.y), philoxUnitFloat(random.z));
        ${l?`accepted = philoxUnitFloat(random.w) < getMaskWeight(candidate);`:`accepted = true;`}
        if (accepted) {
          position = candidate;
          break;
        }
      }
    }
    if (!accepted) {
      atomicAdd(&failed[failedOffset], 1u);
    }
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;`})),S.push(p(e,{id:`${i}-publish`,operation:a,requiredCount:I,output:o.dots,overflowSources:[F]})),S}var q=Object.freeze({...t,measure:`area`}),Xe=class{id;spatialContext;featureCount;config;constructor(t){let n=t.id??`random-points-in-polygon`;this.spatialContext=e(n,t.spatialContext,q),this.config={id:n,operation:`GPURandomPointsInPolygon`,positions:t.positions,featureOffsets:t.featureOffsets,polygonOffsets:t.polygonOffsets,ringOffsets:t.ringOffsets,source:{kind:`counts`,counts:t.counts},categoryCount:1,parameters:t.parameters,maximumAttempts:t.maximumAttempts??32,mask:t.mask,output:t.output},this.featureCount=qe(this.config),this.id=n}getCommandNodes(e){let{positions:t,ringOffsets:n,polygonOffsets:r}=this.config;return Be(e.device,this.id,t.length,n.length-1,r.length-1),Ye(e,this.config,this.featureCount)}};export{Ue as i,Ye as n,qe as r,Xe as t};