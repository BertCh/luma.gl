import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{i as t}from"./earth-xB16XgAa.js";import{E as n,h as r,y as i}from"./analysis-frame-Blu0tYdw.js";import{N as a,_ as o,d as s,h as c,n as l,t as u,v as d,w as f,y as p}from"./resources-CwvhEOBe.js";import{n as m,s as h}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c as g}from"./spatial-weights-D4bd9AQ6.js";import{t as _}from"./gpu-contiguity-weights-BI6hkB9v.js";import{t as v}from"./gpu-geometry-measures-nkjV4Jk0.js";import{n as y,r as b,t as x}from"./gpu-segment-intersection-BynYcZbe.js";import{d as S,i as C,o as w,p as T,r as E,s as D}from"./raster-relaxation-BJ40xgro.js";import{n as O,t as ee}from"./gpu-lon-lat-polygon-preparation-CInGqYKK.js";import{n as k,t as A}from"./gpu-shape-descriptors-CS6f3L_b.js";import{a as j}from"./index-Bjk18V2I.js";import{t as te}from"./graph-swap-41Ru7LVu.js";import{t as ne}from"./summary-reader-CsgWxENf.js";import{t as M}from"./mode-kernels-Dxm2GWvI.js";import{m as N}from"./class-table-BVEo3-Vs.js";import{t as re}from"./b3-palette-APlD7x7U.js";import{r as P,t as ie}from"./b3-layers-B2V1ljKW.js";import{_ as ae,a as F,c as I,f as L,g as oe,h as R,m as se,n as ce,r as le,u as z}from"./b3-common-BbHtVprn.js";var B={nonFinite:1,unclosedRing:2,shortRing:4,repeatedVertex:8,selfIntersection:16,crossingRings:32,holeOutsideShell:64,badOrientation:128,uncertain:256,tooFewPoints:512},ue=B.nonFinite|B.unclosedRing|B.shortRing|B.selfIntersection|B.crossingRings|B.holeOutsideShell|B.uncertain|B.tooFewPoints,V=`GPUGeometryValidity`,H=64,de=class{id;props;spatialContext;featureCount;ringCount;polygons;constructor(e){this.id=e.id??`geometry-validity`,this.spatialContext=n(this.id,e.spatialContext),this.props=e;let{id:t}=this,{polygons:r,lines:i,points:a}=e;if(this.polygons=r,[r,i,a].filter(Boolean).length!==1)throw Error(`${t} requires exactly one of polygons, lines and points`);if(r){p(r.positions,[`float32x2`],`${t} polygons.positions`);for(let[e,n]of[[`featureOffsets`,r.featureOffsets],[`polygonOffsets`,r.polygonOffsets],[`ringOffsets`,r.ringOffsets]])if(p(n,[`uint32`],`${t} polygons.${e}`),n.length<1)throw Error(`${t} polygons.${e} requires a terminal entry`);if(this.featureCount=r.featureOffsets.length-1,this.ringCount=r.ringOffsets.length-1,r.positions.length<1)throw Error(`${t} polygons must contain at least one vertex`);if(!e.output.status?.overflow||e.intersectionCapacity===void 0)throw Error(`${t} polygons require status.overflow and intersectionCapacity`)}else if(i){if(p(i.positions,[`float32x2`],`${t} lines.positions`),p(i.lineOffsets,[`uint32`],`${t} lines.lineOffsets`),i.lineOffsets.length<1)throw Error(`${t} lines.lineOffsets requires a terminal entry`);this.featureCount=i.lineOffsets.length-1,this.ringCount=0}else{let e=a;p(e.positions,[`float32x2`],`${t} points.positions`),this.featureCount=e.positions.length,this.ringCount=0}if(this.featureCount<1)throw Error(`${t} geometry must contain at least one feature`);let{mask:o,status:c}=e.output;if(p(o,[`uint32`],`${t} output.mask`),o.length!==this.featureCount)throw Error(`${t} output.mask length must equal the feature count`);for(let[e,n]of[[`overflow`,c?.overflow],[`intersectionCount`,c?.intersectionCount]])if(n&&(p(n,[`uint32`],`${t} output.status.${e}`),n.length<1))throw Error(`${t} output.status.${e} must contain one uint32 row`);if(r&&s(t,`intersectionCapacity`,e.intersectionCapacity??0),e.ringClosure&&![`implicit`,`explicit`].includes(e.ringClosure))throw Error(`${t} ringClosure must be implicit or explicit`);if(e.orientation&&![`counter-clockwise-shell`,`clockwise-shell`,`ignore`].includes(e.orientation))throw Error(`${t} orientation must be counter-clockwise-shell, clockwise-shell or ignore`)}getViews(){let{polygons:e,props:t}=this,n=e?[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]:t.lines?[t.lines.positions,t.lines.lineOffsets]:[t.points.positions],{mask:r,status:i}=t.output;return[...n,r,i?.overflow,i?.intersectionCount]}getLineOrPointNode(e){let{id:t,props:n,featureCount:r}=this,i=B,{lines:a}=n,o=(a??n.points).positions,s=[{name:`positions`,view:o,type:`f32`,access:`read`}];return a&&s.push({name:`lineOffsets`,view:a.lineOffsets,type:`u32`,access:`read`}),s.push({name:`mask`,view:n.output.mask,type:`u32`,access:`read_write`}),h(e,{id:`${t}-simple-kinds`,operation:V,variant:a?`lines`:`points`,bindings:s,invocationCount:r,declarations:`${y}
const VERTEX_COUNT: u32 = ${o.length}u;
const BIT_NON_FINITE: u32 = ${i.nonFinite}u;
const BIT_TOO_FEW: u32 = ${i.tooFewPoints}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`,body:a?`let start = min(lineOffsets[lineOffsetsOffset + index], VERTEX_COUNT);
  let end = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);
  var bits = 0u;
  var distinct = false;
  if (end > start) {
    let first = vertexAt(start);
    for (var k = start; k < end; k++) {
      let vertex = vertexAt(k);
      if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { bits = bits | BIT_NON_FINITE; }
      if (vertex.x != first.x || vertex.y != first.y) { distinct = true; }
    }
    if (!distinct) { bits = bits | BIT_TOO_FEW; }
  }
  mask[maskOffset + index] = bits;`:`let vertex = vertexAt(index);
  mask[maskOffset + index] = select(BIT_NON_FINITE, 0u, orientIsFinite(vertex.x) && orientIsFinite(vertex.y));`})}getCommandNodes(e){let{id:t,props:n,polygons:r,featureCount:i,ringCount:a}=this;if(o(t,e,this.getViews()),n.polygons&&d(t,`polygons.positions rows`,n.polygons.positions.length,b(e.device)),!r)return[this.getLineOrPointNode(e)];let{mask:s,status:c}=n.output,l=n.intersectionCapacity,u=n.orientation??`counter-clockwise-shell`,p=n.ringClosure===`explicit`,m=[];m.push(h(e,{id:`${t}-clear`,operation:V,variant:`clear`,bindings:[{name:`mask`,view:s,type:`u32`,access:`read_write`}],invocationCount:i,body:`mask[maskOffset + index] = 0u;`}));let g=B,_=f(e,`${t}-ring-bounds`,`float32`,Math.max(a,1)*4);m.push(h(e,{id:`${t}-ring-bounds`,operation:V,variant:`ring-bounds`,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`bounds`,view:_,type:`f32`,access:`read_write`}],invocationCount:Math.ceil(a/H)*H,workgroupSize:H,guardIndex:!1,declarations:`${y}
const VERTEX_COUNT: u32 = ${r.positions.length}u;
const POLYGON_COUNT: u32 = ${r.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${a}u;
const RING_LANES: u32 = ${H}u;
const BOUNDS_LARGE: f32 = 3.4e38;
var<workgroup> queueBits: array<atomic<u32>, 2>;
var<workgroup> queueBitsCopy: array<u32, 2>;
var<workgroup> queuedFirst: array<u32, ${H}>;
var<workgroup> queuedLast: array<u32, ${H}>;
var<workgroup> partialMinimum: array<vec2f, ${H}>;
var<workgroup> partialMaximum: array<vec2f, ${H}>;
var<workgroup> partialNonFinite: array<u32, ${H}>;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
// Last polygon whose first ring is at most ring, or SEGMENT_NONE.
fn lastPolygonAtMost(ring: u32) -> u32 {
  if (polygonOffsets[polygonOffsetsOffset] > ring) { return SEGMENT_NONE; }
  var low = 0u;
  var high = POLYGON_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) { low = middle; } else { high = middle; }
  }
  return low;
}
// Non-finite or empty rings get the whole plane, so the prefilter never rejects through them.
fn writeBounds(ring: u32, minimum: vec2f, maximum: vec2f, nonFinite: u32) {
  let unbounded = nonFinite != 0u || minimum.x > maximum.x;
  let low = select(minimum, vec2f(-BOUNDS_LARGE), unbounded);
  let high = select(maximum, vec2f(BOUNDS_LARGE), unbounded);
  bounds[boundsOffset + 4u * ring] = low.x;
  bounds[boundsOffset + 4u * ring + 1u] = low.y;
  bounds[boundsOffset + 4u * ring + 2u] = high.x;
  bounds[boundsOffset + 4u * ring + 3u] = high.y;
}`,body:`let lane = localInvocationIndex;
  let ringBase = workgroupIndex * RING_LANES;
  let ownRing = ringBase + lane;
  // Phase 1: the shell of a polygon with holes is bounded serially by its own lane when small,
  // and queued for the cooperative phase otherwise.
  if (ownRing < RING_COUNT && POLYGON_COUNT > 0u) {
    let polygon = lastPolygonAtMost(ownRing);
    if (polygon != SEGMENT_NONE && ownRing < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
      let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let nextRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
      if (ownRing == firstRing && nextRing - firstRing >= 2u) {
        let first = ringOffsets[ringOffsetsOffset + ownRing];
        let last = min(ringOffsets[ringOffsetsOffset + ownRing + 1u], VERTEX_COUNT);
        if (last <= first || last - first <= RING_LANES) {
          var minimum = vec2f(BOUNDS_LARGE);
          var maximum = vec2f(-BOUNDS_LARGE);
          var nonFinite = 0u;
          for (var vertex = first; vertex < last; vertex++) {
            let point = vertexAt(vertex);
            if (!orientIsFinite(point.x) || !orientIsFinite(point.y)) {
              nonFinite = 1u;
            } else {
              minimum = min(minimum, point);
              maximum = max(maximum, point);
            }
          }
          writeBounds(ownRing, minimum, maximum, nonFinite);
        } else {
          queuedFirst[lane] = first;
          queuedLast[lane] = last;
          atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u));
        }
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    queueBitsCopy[0] = atomicLoad(&queueBits[0]);
    queueBitsCopy[1] = atomicLoad(&queueBits[1]);
  }
  // Phase 2: queued rings in ascending order; all barriers are in workgroup-uniform control flow.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&queueBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let first = workgroupUniformLoad(&queuedFirst[slot]);
      let last = workgroupUniformLoad(&queuedLast[slot]);
      var minimum = vec2f(BOUNDS_LARGE);
      var maximum = vec2f(-BOUNDS_LARGE);
      var nonFinite = 0u;
      for (var vertex = first + lane; vertex < last; vertex += RING_LANES) {
        let point = vertexAt(vertex);
        if (!orientIsFinite(point.x) || !orientIsFinite(point.y)) {
          nonFinite = 1u;
        } else {
          minimum = min(minimum, point);
          maximum = max(maximum, point);
        }
      }
      partialMinimum[lane] = minimum;
      partialMaximum[lane] = maximum;
      partialNonFinite[lane] = nonFinite;
      workgroupBarrier();
      for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
          partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
          partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
          partialNonFinite[lane] = partialNonFinite[lane] | partialNonFinite[lane + stride];
        }
        workgroupBarrier();
      }
      if (lane == 0u) {
        writeBounds(ringBase + slot, partialMinimum[0], partialMaximum[0], partialNonFinite[0]);
      }
    }
  }`})),m.push(h(e,{id:`${t}-rings`,operation:V,variant:`rings`,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`},{name:`ringBounds`,view:_,type:`f32`,access:`read`},{name:`mask`,view:s,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(a/H)*H,workgroupSize:H,guardIndex:!1,declarations:`${y}
const VERTEX_COUNT: u32 = ${r.positions.length}u;
const POLYGON_COUNT: u32 = ${r.polygonOffsets.length-1}u;
const FEATURE_COUNT: u32 = ${r.featureOffsets.length-1}u;
const RING_COUNT: u32 = ${a}u;
const RING_LANES: u32 = ${H}u;
const EXPLICIT_CLOSURE: bool = ${p};
const CHECK_ORIENTATION: bool = ${u!==`ignore`};
const SHELL_COUNTER_CLOCKWISE: bool = ${u!==`clockwise-shell`};
const BIT_NON_FINITE: u32 = ${g.nonFinite}u;
const BIT_UNCLOSED: u32 = ${g.unclosedRing}u;
const BIT_SHORT: u32 = ${g.shortRing}u;
const BIT_REPEATED: u32 = ${g.repeatedVertex}u;
const BIT_HOLE_OUTSIDE: u32 = ${g.holeOutsideShell}u;
const BIT_ORIENTATION: u32 = ${g.badOrientation}u;
const BIT_UNCERTAIN: u32 = ${g.uncertain}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn samePoint(first: vec2f, second: vec2f) -> bool { return first.x == second.x && first.y == second.y; }
fn readOffset(kind: u32, row: u32) -> u32 {
  if (kind == 0u) { return polygonOffsets[polygonOffsetsOffset + row]; }
  return featureOffsets[featureOffsetsOffset + row];
}
// Last row in [0, count) whose offset is at most value, or SEGMENT_NONE.
fn lastAtMost(kind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u || readOffset(kind, 0u) > value) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(kind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn ringStartOf(ring: u32) -> u32 { return ringOffsets[ringOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT); }
// Vertex count without a closing duplicate of the first vertex.
fn effectiveCount(start: u32, end: u32) -> u32 {
  if (end <= start) { return 0u; }
  let count = end - start;
  if (count >= 2u && samePoint(vertexAt(start), vertexAt(end - 1u))) { return count - 1u; }
  return count;
}
const NO_INDEX: u32 = 0xffffffffu;
// Per-ring state of the block, indexed by the ring's lane. Written by the owner lane in phase 1 and
// by lane 0 for cooperatively processed rings; read through workgroupUniformLoad when it steers a barrier.
var<workgroup> ringActive: array<u32, ${H}>;
var<workgroup> ringFeature: array<u32, ${H}>;
var<workgroup> ringIsHole: array<u32, ${H}>;
var<workgroup> ringStart: array<u32, ${H}>;
var<workgroup> ringEnd: array<u32, ${H}>;
var<workgroup> ringFirstRing: array<u32, ${H}>;
var<workgroup> ringLarge: array<u32, ${H}>;
var<workgroup> ringBits: array<u32, ${H}>;
var<workgroup> ringNeeds: array<u32, ${H}>;
var<workgroup> ringEffective: array<u32, ${H}>;
var<workgroup> ringShellStart: array<u32, ${H}>;
var<workgroup> ringShellCount: array<u32, ${H}>;
var<workgroup> ringShellRing: array<u32, ${H}>;
var<workgroup> queueBits: array<atomic<u32>, 2>;
var<workgroup> queueBitsCopy: array<u32, 2>;
var<workgroup> sharedPoint: vec2f;
var<workgroup> sharedResult: array<u32, 2>;
var<workgroup> partialFlags: array<u32, ${H}>;
var<workgroup> partialExtreme: array<u32, ${H}>;
var<workgroup> partialPoint: array<vec2f, ${H}>;
var<workgroup> partialEdge: array<u32, ${H}>;
var<workgroup> partialKind: array<u32, ${H}>;
var<workgroup> partialParity: array<u32, ${H}>;
fn isLexicographicallyLess(candidate: vec2f, best: vec2f) -> bool {
  return candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y);
}
// Winding of the ring at its lexicographically smallest vertex (first one on ties): 1
// counter-clockwise, -1 clockwise, 0 undetermined, 2 uncertain. A known extreme (from the
// cooperative search) skips the serial search when it is not NO_INDEX.
fn ringWinding(start: u32, count: u32, knownExtreme: u32) -> i32 {
  var extreme = 0u;
  if (knownExtreme != NO_INDEX) {
    extreme = knownExtreme;
  } else {
    for (var k = 1u; k < count; k++) {
      if (isLexicographicallyLess(vertexAt(start + k), vertexAt(start + extreme))) { extreme = k; }
    }
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
    // Both neighbours are normally the adjacent vertices: stop instead of walking the whole ring.
    if (foundPrevious && foundNext) { break; }
  }
  if (!foundPrevious || !foundNext) { return 0; }
  return orientSign(vertexAt(start + previous), center, vertexAt(start + next));
}
// Folds the ring-local checks of ring slot into ringBits and decides whether the ring is a hole
// whose containment in its shell must be tested (ringNeeds, with the shell in the other slots arrays).
fn finishRing(slot: u32, start: u32, end: u32, isHole: bool, polygonFirstRing: u32, finiteAll: bool, repeated: bool, extreme: u32) {
  let count = select(0u, end - start, end > start);
  var bits = 0u;
  if (!finiteAll) { bits = bits | BIT_NON_FINITE; }
  if (repeated) { bits = bits | BIT_REPEATED; }
  if (EXPLICIT_CLOSURE && count >= 1u && !samePoint(vertexAt(start), vertexAt(end - 1u))) { bits = bits | BIT_UNCLOSED; }
  let effective = effectiveCount(start, end);
  let isShort = effective < 3u;
  if (isShort) { bits = bits | BIT_SHORT; }
  var needsContainment = 0u;
  if (finiteAll && !isShort) {
    if (CHECK_ORIENTATION) {
      let winding = ringWinding(start, effective, extreme);
      if (winding == 2) { bits = bits | BIT_UNCERTAIN; }
      else if (winding != 0) {
        let wantsCounterClockwise = select(SHELL_COUNTER_CLOCKWISE, !SHELL_COUNTER_CLOCKWISE, isHole);
        if ((winding > 0) != wantsCounterClockwise) { bits = bits | BIT_ORIENTATION; }
      }
    }
    if (isHole) {
      let shellStart = ringStartOf(polygonFirstRing);
      let shellCount = effectiveCount(shellStart, ringEndOf(polygonFirstRing));
      if (shellCount >= 3u) {
        needsContainment = 1u;
        ringEffective[slot] = effective;
        ringShellStart[slot] = shellStart;
        ringShellCount[slot] = shellCount;
        ringShellRing[slot] = polygonFirstRing;
      }
    }
  }
  ringBits[slot] = bits;
  ringNeeds[slot] = needsContainment;
}
// Workgroup-cooperative point-in-ring: 1 inside, 0 outside, 2 on the boundary, 3 uncertain, with
// the serial walk's result: the event (boundary or uncertain) at the lowest edge index wins,
// otherwise the crossing parity decides. A point outside the shell's bounds is outside it.
// Must be called from workgroup-uniform control flow with a uniform argument.
fn locateInShell(point: vec2f, shellStart: u32, count: u32, shellRing: u32, lane: u32) -> u32 {
  if (lane == 0u) {
    let base = ringBoundsOffset + 4u * shellRing;
    sharedResult[0] = select(0u, 1u,
      point.x < ringBounds[base] || point.x > ringBounds[base + 2u] ||
      point.y < ringBounds[base + 1u] || point.y > ringBounds[base + 3u]);
  }
  if (workgroupUniformLoad(&sharedResult[0]) == 1u) { return 0u; }
  var eventEdge = NO_INDEX;
  var eventKind = 0u;
  var parity = 0u;
  for (var k = lane; k < count; k += RING_LANES) {
    let a = vertexAt(shellStart + k);
    let b = vertexAt(shellStart + (k + 1u) % count);
    let inBox = point.x >= min(a.x, b.x) && point.x <= max(a.x, b.x) &&
      point.y >= min(a.y, b.y) && point.y <= max(a.y, b.y);
    let straddles = (a.y > point.y) != (b.y > point.y);
    if (!inBox && !straddles) { continue; }
    let side = orientSign(a, b, point);
    if (side == 2) { eventEdge = k; eventKind = 3u; break; }
    if (inBox && side == 0) { eventEdge = k; eventKind = 2u; break; }
    if (straddles && ((b.y > a.y) == (side > 0))) { parity = parity ^ 1u; }
  }
  partialEdge[lane] = eventEdge;
  partialKind[lane] = eventKind;
  partialParity[lane] = parity;
  workgroupBarrier();
  for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      if (partialEdge[lane + stride] < partialEdge[lane]) {
        partialEdge[lane] = partialEdge[lane + stride];
        partialKind[lane] = partialKind[lane + stride];
      }
      partialParity[lane] = partialParity[lane] ^ partialParity[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    sharedResult[1] = select(partialParity[0], partialKind[0], partialEdge[0] != NO_INDEX);
  }
  return workgroupUniformLoad(&sharedResult[1]);
}`,body:`let lane = localInvocationIndex;
  let ring = workgroupIndex * RING_LANES + lane;
  // Phase 1: this lane's own ring. Small rings run the original serial checks; large rings and
  // holes that need a containment walk are queued (bitmask over the block's lanes).
  if (ring < RING_COUNT) {
    let polygon = lastAtMost(0u, POLYGON_COUNT, ring);
    if (polygon != SEGMENT_NONE && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
        polygon < featureOffsets[featureOffsetsOffset + FEATURE_COUNT]) {
      let feature = lastAtMost(1u, FEATURE_COUNT, polygon);
      if (feature != SEGMENT_NONE) {
        let start = ringStartOf(ring);
        let end = ringEndOf(ring);
        let isHole = ring != polygonOffsets[polygonOffsetsOffset + polygon];
        let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
        ringActive[lane] = 1u;
        ringFeature[lane] = feature;
        ringIsHole[lane] = select(0u, 1u, isHole);
        ringStart[lane] = start;
        ringEnd[lane] = end;
        ringFirstRing[lane] = firstRing;
        let count = select(0u, end - start, end > start);
        if (count <= RING_LANES) {
          var finiteAll = true;
          var repeated = false;
          for (var k = 0u; k < count; k++) {
            let vertex = vertexAt(start + k);
            if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { finiteAll = false; }
            if (k + 1u < count && samePoint(vertex, vertexAt(start + k + 1u))) { repeated = true; }
          }
          finishRing(lane, start, end, isHole, firstRing, finiteAll, repeated, NO_INDEX);
          if (ringNeeds[lane] == 1u) { atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u)); }
        } else {
          ringLarge[lane] = 1u;
          atomicOr(&queueBits[lane / 32u], 1u << (lane % 32u));
        }
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    queueBitsCopy[0] = atomicLoad(&queueBits[0]);
    queueBitsCopy[1] = atomicLoad(&queueBits[1]);
  }
  // Phase 2: queued rings in ascending order. Every barrier is in workgroup-uniform control flow:
  // the queue and everything that steers a loop or branch go through workgroupUniformLoad.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&queueBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let start = workgroupUniformLoad(&ringStart[slot]);
      if (workgroupUniformLoad(&ringLarge[slot]) == 1u) {
        // Large ring: lanes scan strided vertices for non-finite coordinates, repeated neighbours
        // and the lexicographic minimum, then a tree combines them.
        let end = workgroupUniformLoad(&ringEnd[slot]);
        let count = select(0u, end - start, end > start);
        let effective = effectiveCount(start, end);
        var flags = 0u;
        var best = NO_INDEX;
        var bestPoint = vec2f(0.0);
        for (var k = lane; k < count; k += RING_LANES) {
          let vertex = vertexAt(start + k);
          if (!orientIsFinite(vertex.x) || !orientIsFinite(vertex.y)) { flags = flags | 1u; }
          if (k + 1u < count && samePoint(vertex, vertexAt(start + k + 1u))) { flags = flags | 2u; }
          if (k < effective && (best == NO_INDEX || isLexicographicallyLess(vertex, bestPoint))) {
            best = k;
            bestPoint = vertex;
          }
        }
        partialFlags[lane] = flags;
        partialExtreme[lane] = best;
        partialPoint[lane] = bestPoint;
        workgroupBarrier();
        for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
          if (lane < stride) {
            partialFlags[lane] = partialFlags[lane] | partialFlags[lane + stride];
            let other = partialExtreme[lane + stride];
            let current = partialExtreme[lane];
            if (other != NO_INDEX && (current == NO_INDEX ||
                isLexicographicallyLess(partialPoint[lane + stride], partialPoint[lane]) ||
                (partialPoint[lane + stride].x == partialPoint[lane].x &&
                 partialPoint[lane + stride].y == partialPoint[lane].y && other < current))) {
              partialExtreme[lane] = other;
              partialPoint[lane] = partialPoint[lane + stride];
            }
          }
          workgroupBarrier();
        }
        if (lane == 0u) {
          finishRing(slot, start, end, ringIsHole[slot] == 1u, ringFirstRing[slot],
            (partialFlags[0] & 1u) == 0u, (partialFlags[0] & 2u) != 0u, partialExtreme[0]);
        }
      }
      // The hole-in-shell test runs on the whole workgroup whatever the hole's size, because the
      // shell may be huge. One hole vertex off the shell boundary decides containment once no ring
      // edges cross.
      if (workgroupUniformLoad(&ringNeeds[slot]) == 1u) {
        let effective = workgroupUniformLoad(&ringEffective[slot]);
        let shellStart = workgroupUniformLoad(&ringShellStart[slot]);
        let shellCount = workgroupUniformLoad(&ringShellCount[slot]);
        let shellRing = workgroupUniformLoad(&ringShellRing[slot]);
        var located = 2u;
        for (var k = 0u; k < effective && located == 2u; k++) {
          if (lane == 0u) { sharedPoint = vertexAt(start + k); }
          let point = workgroupUniformLoad(&sharedPoint);
          located = locateInShell(point, shellStart, shellCount, shellRing, lane);
        }
        if (lane == 0u) {
          if (located == 3u) { ringBits[slot] = ringBits[slot] | BIT_UNCERTAIN; }
          else if (located != 1u) { ringBits[slot] = ringBits[slot] | BIT_HOLE_OUTSIDE; }
        }
      }
    }
  }
  workgroupBarrier();
  if (ringActive[lane] == 1u && ringBits[lane] != 0u) {
    atomicOr(&mask[maskOffset + ringFeature[lane]], ringBits[lane]);
  }`}));let v=f(e,`${t}-pair-left`,`uint32`,l),S=f(e,`${t}-pair-right`,`uint32`,l),C=f(e,`${t}-pair-count`,`uint32`,1),w=f(e,`${t}-kinds`,`uint32`,l),T=f(e,`${t}-left-features`,`uint32`,l),E=f(e,`${t}-left-rings`,`uint32`,l),D=f(e,`${t}-right-rings`,`uint32`,l),O=new x({id:`${t}-intersection`,left:r,sameFeatureOnly:!0,leafCapacity:n.leafCapacity,output:{pairs:{leftIds:v,rightIds:S,status:{count:C,overflow:c?.overflow,requiredCount:c?.intersectionCount}},kinds:w,leftFeatures:T,leftRings:E,rightRings:D}});return m.push(...O.getCommandNodes(e)),m.push(h(e,{id:`${t}-fold`,operation:V,variant:`fold`,bindings:[{name:`pairCount`,view:C,type:`u32`,access:`read`},{name:`kinds`,view:w,type:`u32`,access:`read`},{name:`leftFeatures`,view:T,type:`u32`,access:`read`},{name:`leftRings`,view:E,type:`u32`,access:`read`},{name:`rightRings`,view:D,type:`u32`,access:`read`},{name:`mask`,view:s,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,declarations:`${y}
const BIT_SELF: u32 = ${g.selfIntersection}u;
const BIT_CROSSING: u32 = ${g.crossingRings}u;
const BIT_UNCERTAIN: u32 = ${g.uncertain}u;`,body:`if (index >= pairCount[pairCountOffset]) { return; }
  let kind = kinds[kindsOffset + index];
  var bits = 0u;
  if (kind == KIND_UNCERTAIN) {
    bits = BIT_UNCERTAIN;
  } else if (leftRings[leftRingsOffset + index] == rightRings[rightRingsOffset + index]) {
    bits = BIT_SELF;
  } else if (kind == KIND_PROPER || kind == KIND_OVERLAP) {
    bits = BIT_CROSSING;
  }
  if (bits != 0u) { atomicOr(&mask[maskOffset + leftFeatures[leftFeaturesOffset + index]], bits); }`})),m}},U=`GPULabelPoint`,W=Object.freeze({...r,measure:`distance`}),fe=64,pe=class{id;props;spatialContext;featureCount;initialGridSize;refinementGridSize;refinementRounds;refinementCandidates;smallFeatureVertexLimit;constructor(e){this.id=e.id??`label-point`,this.props=e,this.spatialContext=n(this.id,e.spatialContext,W),this.initialGridSize=e.initialGridSize??16,this.refinementGridSize=e.refinementGridSize??8,this.refinementRounds=e.refinementRounds??6,this.refinementCandidates=e.refinementCandidates??4,this.smallFeatureVertexLimit=e.smallFeatureVertexLimit??32;let{id:t}=this;if(s(t,`initialGridSize`,this.initialGridSize,2),s(t,`refinementGridSize`,this.refinementGridSize,3),s(t,`refinementRounds`,this.refinementRounds,0),s(t,`refinementCandidates`,this.refinementCandidates,1,16),s(t,`smallFeatureVertexLimit`,this.smallFeatureVertexLimit,0),p(e.positions,[`float32x2`],`${t} positions`),e.positions.length<1)throw Error(`${t} needs at least one position`);for(let n of[`ringOffsets`,`polygonOffsets`,`featureOffsets`])if(p(e[n],[`uint32`],`${t} ${n}`),e[n].length<2)throw Error(`${t} ${n} must contain at least two rows`);this.featureCount=e.featureOffsets.length-1;let{points:r,distances:i,degenerate:a}=e.output;if(p(r,[`float32x2`],`${t} output.points`),r.length!==this.featureCount)throw Error(`${t} output.points must hold ${this.featureCount} rows`);if(i&&(p(i,[`float32`],`${t} output.distances`),i.length!==this.featureCount))throw Error(`${t} output.distances must hold ${this.featureCount} rows`);if(a&&(p(a,[`uint32`],`${t} output.degenerate`),a.length!==this.featureCount))throw Error(`${t} output.degenerate must hold ${this.featureCount} rows`);c(t,[r,i,a],[e.positions,e.ringOffsets,e.featureOffsets,e.polygonOffsets])}getCommandNodes(e){let{props:t,id:n}=this,{points:r,distances:i,degenerate:a}=t.output;o(n,e,[t.positions,t.ringOffsets,t.featureOffsets,t.polygonOffsets,r,i,a]);let s=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:t.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`}];s.push({name:`pointsOut`,view:r,type:`f32`,access:`read_write`}),i&&s.push({name:`distancesOut`,view:i,type:`f32`,access:`read_write`}),a&&s.push({name:`degenerateOut`,view:a,type:`u32`,access:`read_write`});let c=`const ROW_COUNT: u32 = ${t.positions.length}u;
const RING_COUNT: u32 = ${t.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${t.polygonOffsets.length-1}u;
const FEATURE_COUNT: u32 = ${this.featureCount}u;
const INITIAL_GRID_SIZE: u32 = ${this.initialGridSize}u;
const REFINEMENT_GRID_SIZE: u32 = ${this.refinementGridSize}u;
const REFINEMENT_ROUNDS: u32 = ${this.refinementRounds}u;
const CANDIDATE_COUNT: u32 = ${this.refinementCandidates}u;
const SMALL_FEATURE_VERTEX_LIMIT: u32 = ${this.smallFeatureVertexLimit}u;
const WORKGROUP_SIZE: u32 = ${fe}u;
const LARGE: f32 = 3.0e38;
${me}
// Feature -> polygons -> rings.
fn getRingStart(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}
fn getRingEnd(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature + 1u], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}`,l=(e,t)=>`
  pointsOut[pointsOutOffset + 2u * ${e}] = result.x;
  pointsOut[pointsOutOffset + 2u * ${e} + 1u] = result.y;
  ${i?`distancesOut[distancesOutOffset + ${e}] = select(max(bestDistance, 0.0), nan, ${t});`:``}
  ${a?`degenerateOut[degenerateOutOffset + ${e}] = isDegenerate;`:``}`;return[h(e,{id:`${n}-search`,operation:U,variant:`small`,bindings:s,invocationCount:this.featureCount,declarations:c,body:`let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  if (rowEnd - rowStart > SMALL_FEATURE_VERTEX_LIMIT) {
    return;
  }
  var result = vec2<f32>(nan);
  var bestDistance = nan;
  var isDegenerate = 1u;
  if (rowEnd > rowStart) {
    var minimum = getPosition(rowStart);
    var maximum = minimum;
    for (var row = rowStart + 1u; row < rowEnd; row++) {
      let p = getPosition(row);
      minimum = min(minimum, p);
      maximum = max(maximum, p);
    }
    let origin = 0.5 * (minimum + maximum);
    let size = maximum - minimum;
    var halfSize = 0.5 * max(size.x, size.y);
    // Seeds: first vertex (on the boundary, the fallback), then scanline midpoints.
    var best = vec3<f32>(getPosition(rowStart) - origin, 0.0);
    var bestIsSet = false;
    for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
      let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
      let seed = getScanlineSeed(ringStart, ringEnd, origin, fraction * size.y);
      if (seed.z > 0.0) {
        let candidate = vec3<f32>(seed.xy, getSignedDistance(seed.xy, ringStart, ringEnd, origin));
        if (!bestIsSet || candidate.z > best.z) {
          best = candidate;
          bestIsSet = true;
        }
      }
    }
    if (!bestIsSet) {
      // Archipelago fallback: every part missed the three feature scanlines, so repeat them through
      // the bounds of the largest ring instead of settling for a boundary vertex.
      let part = getLargestRingBounds(ringStart, ringEnd, origin);
      for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
        let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
        let seed = getScanlineSeed(ringStart, ringEnd, origin, 0.5 * (part.y + part.w) + fraction * (part.w - part.y));
        if (seed.z > 0.0) {
          let candidate = vec3<f32>(seed.xy, getSignedDistance(seed.xy, ringStart, ringEnd, origin));
          if (!bestIsSet || candidate.z > best.z) {
            best = candidate;
            bestIsSet = true;
          }
        }
      }
    }
    if (!bestIsSet) {
      best.z = getSignedDistance(best.xy, ringStart, ringEnd, origin);
    }
    if (halfSize > 0.0) {
      var top = array<vec3<f32>, CANDIDATE_COUNT>();
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        top[slot] = vec3<f32>(0.0, 0.0, -LARGE);
      }
      scoreGrid(vec2<f32>(0.0), halfSize, INITIAL_GRID_SIZE, ringStart, ringEnd, origin, &top);
      let cellHalfSize = halfSize / f32(INITIAL_GRID_SIZE);
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        var local = top[slot];
        // Candidates are sorted by distance, so once the cell bound cannot beat the best, stop.
        if (local.z + 1.4142135 * cellHalfSize <= best.z) {
          break;
        }
        if (local.z > best.z) {
          best = local;
        }
        var windowHalfSize = 2.0 * cellHalfSize;
        for (var round = 0u; round < REFINEMENT_ROUNDS; round++) {
          local = searchWindow(local.xy, windowHalfSize, REFINEMENT_GRID_SIZE, ringStart, ringEnd, origin, local);
          windowHalfSize = 2.0 * windowHalfSize / f32(REFINEMENT_GRID_SIZE);
        }
        if (local.z > best.z) {
          best = local;
        }
      }
    }
    result = origin + best.xy;
    bestDistance = best.z;
    isDegenerate = select(0u, 1u, best.z <= 0.0);
  }${l(`index`,`rowEnd <= rowStart`)}`}),h(e,{id:`${n}-search-workgroup`,operation:U,variant:`workgroup`,bindings:s,invocationCount:this.featureCount*fe,workgroupSize:fe,guardIndex:!1,declarations:c,body:`let feature = workgroupIndex;
  let lane = localInvocationIndex;
  if (feature >= FEATURE_COUNT) {
    return;
  }
  let nan = bitcast<f32>(0x7fc00000u | (feature & 0u));
  let ringStart = getRingStart(feature);
  let ringEnd = max(getRingEnd(feature), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  if (lane == 0u) {
    sharedUint[0] = select(0u, 1u, rowEnd - rowStart > SMALL_FEATURE_VERTEX_LIMIT);
  }
  if (workgroupUniformLoad(&sharedUint[0]) == 0u) {
    return;
  }
  var localMinimum = vec2<f32>(LARGE);
  var localNegativeMaximum = vec2<f32>(LARGE);
  for (var row = rowStart + lane; row < rowEnd; row += WORKGROUP_SIZE) {
    let p = getPosition(row);
    localMinimum = min(localMinimum, p);
    localNegativeMaximum = min(localNegativeMaximum, -p);
  }
  let minimum = vec2<f32>(reduceMinFloat(lane, localMinimum.x), reduceMinFloat(lane, localMinimum.y));
  let maximum = -vec2<f32>(reduceMinFloat(lane, localNegativeMaximum.x), reduceMinFloat(lane, localNegativeMaximum.y));
  let origin = 0.5 * (minimum + maximum);
  let size = maximum - minimum;
  let halfSize = 0.5 * max(size.x, size.y);
  var best = vec3<f32>(getPosition(rowStart) - origin, 0.0);
  var bestIsSet = false;
  for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
    let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
    let seed = getScanlineSeedCooperative(lane, ringStart, ringEnd, origin, fraction * size.y);
    if (seed.z > 0.0) {
      let candidate = vec3<f32>(seed.xy, getSignedDistanceCooperative(lane, seed.xy, ringStart, ringEnd, origin));
      if (!bestIsSet || candidate.z > best.z) {
        best = candidate;
        bestIsSet = true;
      }
    }
  }
  if (!bestIsSet) {
    // Archipelago fallback (see the serial kernel): scanlines through the largest ring's bounds.
    let part = getLargestRingBoundsCooperative(lane, ringStart, ringEnd, origin);
    for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
      let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
      let seed = getScanlineSeedCooperative(lane, ringStart, ringEnd, origin, 0.5 * (part.y + part.w) + fraction * (part.w - part.y));
      if (seed.z > 0.0) {
        let candidate = vec3<f32>(seed.xy, getSignedDistanceCooperative(lane, seed.xy, ringStart, ringEnd, origin));
        if (!bestIsSet || candidate.z > best.z) {
          best = candidate;
          bestIsSet = true;
        }
      }
    }
  }
  if (!bestIsSet) {
    best.z = getSignedDistanceCooperative(lane, best.xy, ringStart, ringEnd, origin);
  }
  if (halfSize > 0.0) {
    var topDistance = array<f32, CANDIDATE_COUNT>();
    var topCell = array<u32, CANDIDATE_COUNT>();
    for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
      topDistance[slot] = -LARGE;
      topCell[slot] = NO_CELL;
    }
    // Each lane scores the first-grid cells lane, lane + 64, ... in order and keeps its best few.
    let initialCellSize = 2.0 * halfSize / f32(INITIAL_GRID_SIZE);
    for (var cellIndex = lane; cellIndex < INITIAL_GRID_SIZE * INITIAL_GRID_SIZE; cellIndex += WORKGROUP_SIZE) {
      let p = getGridPoint(vec2<f32>(0.0), halfSize, initialCellSize, INITIAL_GRID_SIZE, cellIndex);
      var entryDistance = getSignedDistance(p, ringStart, ringEnd, origin);
      var entryCell = cellIndex;
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        if (entryDistance > topDistance[slot]) {
          let displacedDistance = topDistance[slot];
          let displacedCell = topCell[slot];
          topDistance[slot] = entryDistance;
          topCell[slot] = entryCell;
          entryDistance = displacedDistance;
          entryCell = displacedCell;
        }
      }
    }
    let cellHalfSize = halfSize / f32(INITIAL_GRID_SIZE);
    for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
      // Next best first-grid cell over all lanes: highest distance, earliest cell on ties.
      let winner = reduceArgMax(lane, topDistance[0], topCell[0]);
      if (winner.cell != NO_CELL && lane == winner.cell % WORKGROUP_SIZE) {
        for (var shift = 0u; shift + 1u < CANDIDATE_COUNT; shift++) {
          topDistance[shift] = topDistance[shift + 1u];
          topCell[shift] = topCell[shift + 1u];
        }
        topDistance[CANDIDATE_COUNT - 1u] = -LARGE;
        topCell[CANDIDATE_COUNT - 1u] = NO_CELL;
      }
      var local = vec3<f32>(0.0, 0.0, -LARGE);
      if (winner.cell != NO_CELL) {
        local = vec3<f32>(getGridPoint(vec2<f32>(0.0), halfSize, initialCellSize, INITIAL_GRID_SIZE, winner.cell), winner.distance);
      }
      if (local.z + 1.4142135 * cellHalfSize <= best.z) {
        break;
      }
      if (local.z > best.z) {
        best = local;
      }
      var windowHalfSize = 2.0 * cellHalfSize;
      for (var round = 0u; round < REFINEMENT_ROUNDS; round++) {
        local = searchWindowCooperative(lane, local.xy, windowHalfSize, REFINEMENT_GRID_SIZE, ringStart, ringEnd, origin, local);
        windowHalfSize = 2.0 * windowHalfSize / f32(REFINEMENT_GRID_SIZE);
      }
      if (local.z > best.z) {
        best = local;
      }
    }
  }
  if (lane == 0u) {
    let result = origin + best.xy;
    let bestDistance = best.z;
    let isDegenerate = select(0u, 1u, best.z <= 0.0);${l(`feature`,`false`)}
  }`})]}},me=`
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getRingRowStart(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
}

fn getRingRowEnd(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring + 1u], ROW_COUNT);
}

fn getSegmentDistanceSquared(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let edge = b - a;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) {
    t = clamp(dot(p - a, edge) / lengthSquared, 0.0, 1.0);
  }
  let offset = p - (a + t * edge);
  return dot(offset, offset);
}

// Signed distance to the boundary of all rings: positive inside under the even-odd rule.
fn getSignedDistance(p: vec2<f32>, ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> f32 {
  var minimumSquared = LARGE;
  var inside = false;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let rowStart = getRingRowStart(ring);
    let rowEnd = getRingRowEnd(ring);
    if (rowEnd > rowStart) {
      var previous = getPosition(rowEnd - 1u) - origin;
      for (var row = rowStart; row < rowEnd; row++) {
        let current = getPosition(row) - origin;
        minimumSquared = min(minimumSquared, getSegmentDistanceSquared(p, previous, current));
        // f32 sampling test: it only places the candidate label point; topology uses utils/exact-geometry-wgsl.ts.
        if ((current.y > p.y) != (previous.y > p.y)) {
          let crossingX = previous.x + (p.y - previous.y) * (current.x - previous.x) / (current.y - previous.y);
          if (p.x < crossingX) {
            inside = !inside;
          }
        }
        previous = current;
      }
    }
  }
  let distance = sqrt(minimumSquared);
  return select(-distance, distance, inside);
}

// Bounds (minimum.xy, maximum.xy) relative to origin of ring 'ring'.
fn getRingBounds(ring: u32, origin: vec2<f32>) -> vec4<f32> {
  let rowStart = getRingRowStart(ring);
  let rowEnd = getRingRowEnd(ring);
  var minimum = vec2<f32>(LARGE);
  var maximum = vec2<f32>(-LARGE);
  for (var row = rowStart; row < rowEnd; row++) {
    let p = getPosition(row) - origin;
    minimum = min(minimum, p);
    maximum = max(maximum, p);
  }
  return vec4<f32>(minimum, maximum);
}

// Bounds of the ring with the largest extent (first on ties), used only when the feature-wide
// scanlines found no interior seed.
fn getLargestRingBounds(ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> vec4<f32> {
  var largest = vec4<f32>(0.0);
  var largestExtent = -LARGE;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let bounds = getRingBounds(ring, origin);
    let extent = max(bounds.z - bounds.x, bounds.w - bounds.y);
    if (extent > largestExtent) {
      largestExtent = extent;
      largest = bounds;
    }
  }
  return largest;
}

// Midpoint of the leftmost interior interval on the horizontal line y = lineY (relative to the
// bounds center). f32 sampling code like getSignedDistance (see utils/exact-geometry-wgsl.ts for topology). Returns (x, y, 1) or z = 0 when there is none.
fn getScanlineSeed(ringStart: u32, ringEnd: u32, origin: vec2<f32>, lineY: f32) -> vec3<f32> {
  var firstX = LARGE;
  var firstCount = 0u;
  for (var scanPass = 0u; scanPass < 2u; scanPass++) {
    var secondX = LARGE;
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let rowStart = getRingRowStart(ring);
      let rowEnd = getRingRowEnd(ring);
      if (rowEnd > rowStart) {
        var previous = getPosition(rowEnd - 1u) - origin;
        for (var row = rowStart; row < rowEnd; row++) {
          let current = getPosition(row) - origin;
          if ((current.y > lineY) != (previous.y > lineY)) {
            let crossingX = previous.x + (lineY - previous.y) * (current.x - previous.x) / (current.y - previous.y);
            if (scanPass == 0u) {
              if (crossingX < firstX) {
                firstX = crossingX;
                firstCount = 1u;
              } else if (crossingX == firstX) {
                firstCount += 1u;
              }
            } else if (crossingX > firstX) {
              secondX = min(secondX, crossingX);
            }
          }
          previous = current;
        }
      }
    }
    if (scanPass == 1u) {
      if (firstCount % 2u == 1u && secondX < LARGE) {
        return vec3<f32>(0.5 * (firstX + secondX), lineY, 1.0);
      }
    }
  }
  return vec3<f32>(0.0);
}

// Scores an n x n grid like searchWindow but keeps the best CANDIDATE_COUNT cells, sorted by
// signed distance descending (ties keep the earlier cell).
fn scoreGrid(
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  top: ptr<function, array<vec3<f32>, CANDIDATE_COUNT>>
) {
  let cell = 2.0 * halfSize / f32(gridSize);
  for (var j = 0u; j < gridSize; j++) {
    for (var i = 0u; i < gridSize; i++) {
      let p = center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
      var entry = vec3<f32>(p, getSignedDistance(p, ringStart, ringEnd, origin));
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        if (entry.z > (*top)[slot].z) {
          let displaced = (*top)[slot];
          (*top)[slot] = entry;
          entry = displaced;
        }
      }
    }
  }
}

// Scores an n x n grid of cell centers over [center - halfSize, center + halfSize]^2 and returns
// the better of the grid winner and \`best\` (ties keep the earlier candidate).
fn searchWindow(
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  bestIn: vec3<f32>
) -> vec3<f32> {
  var best = bestIn;
  let cell = 2.0 * halfSize / f32(gridSize);
  for (var j = 0u; j < gridSize; j++) {
    for (var i = 0u; i < gridSize; i++) {
      let p = center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
      let distance = getSignedDistance(p, ringStart, ringEnd, origin);
      if (distance > best.z) {
        best = vec3<f32>(p, distance);
      }
    }
  }
  return best;
}

// Workgroup-cooperative variants used by the one-workgroup-per-feature kernel. Reductions end in
// workgroupUniformLoad so their results are workgroup-uniform, which keeps later barriers legal.
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> sharedUint: array<u32, 1>;
var<workgroup> reduceFloat: array<f32, WORKGROUP_SIZE>;
var<workgroup> reduceUint: array<u32, WORKGROUP_SIZE>;

struct ArgMax {
  distance: f32,
  cell: u32
}

fn reduceMinFloat(lane: u32, value: f32) -> f32 {
  reduceFloat[lane] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceFloat[lane] = min(reduceFloat[lane], reduceFloat[lane + stride]);
    }
    workgroupBarrier();
  }
  return workgroupUniformLoad(&reduceFloat[0]);
}

fn reduceSumUint(lane: u32, value: u32) -> u32 {
  reduceUint[lane] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceUint[lane] = reduceUint[lane] + reduceUint[lane + stride];
    }
    workgroupBarrier();
  }
  return workgroupUniformLoad(&reduceUint[0]);
}

// Highest distance over the workgroup, the lowest cell index among equal distances.
fn reduceArgMax(lane: u32, distance: f32, cell: u32) -> ArgMax {
  reduceFloat[lane] = distance;
  reduceUint[lane] = cell;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      let otherDistance = reduceFloat[lane + stride];
      let otherCell = reduceUint[lane + stride];
      if (otherDistance > reduceFloat[lane] || (otherDistance == reduceFloat[lane] && otherCell < reduceUint[lane])) {
        reduceFloat[lane] = otherDistance;
        reduceUint[lane] = otherCell;
      }
    }
    workgroupBarrier();
  }
  let winnerDistance = workgroupUniformLoad(&reduceFloat[0]);
  let winnerCell = workgroupUniformLoad(&reduceUint[0]);
  return ArgMax(winnerDistance, winnerCell);
}

fn getGridPoint(center: vec2<f32>, halfSize: f32, cell: f32, gridSize: u32, cellIndex: u32) -> vec2<f32> {
  let i = cellIndex % gridSize;
  let j = cellIndex / gridSize;
  return center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
}

// Same value as getSignedDistance: lanes stride over the segments of every ring.
fn getSignedDistanceCooperative(lane: u32, p: vec2<f32>, ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> f32 {
  var minimumSquared = LARGE;
  var crossings = 0u;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let ringRowStart = getRingRowStart(ring);
    let ringRowEnd = getRingRowEnd(ring);
    for (var row = ringRowStart + lane; row < ringRowEnd; row += WORKGROUP_SIZE) {
      let current = getPosition(row) - origin;
      let previous = getPosition(select(row - 1u, ringRowEnd - 1u, row == ringRowStart)) - origin;
      minimumSquared = min(minimumSquared, getSegmentDistanceSquared(p, previous, current));
      // f32 sampling test: it only places the candidate label point; topology uses utils/exact-geometry-wgsl.ts.
      if ((current.y > p.y) != (previous.y > p.y)) {
        let crossingX = previous.x + (p.y - previous.y) * (current.x - previous.x) / (current.y - previous.y);
        if (p.x < crossingX) {
          crossings += 1u;
        }
      }
    }
  }
  let distance = sqrt(reduceMinFloat(lane, minimumSquared));
  return select(-distance, distance, (reduceSumUint(lane, crossings) & 1u) == 1u);
}

// Same ring as getLargestRingBounds: lanes stride over the rings, the largest extent wins (lowest
// ring on ties), and every lane then reads that ring's bounds. Workgroup-uniform result.
fn getLargestRingBoundsCooperative(lane: u32, ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> vec4<f32> {
  var laneExtent = -LARGE;
  var laneRing = NO_CELL;
  for (var ring = ringStart + lane; ring < ringEnd; ring += WORKGROUP_SIZE) {
    let bounds = getRingBounds(ring, origin);
    let extent = max(bounds.z - bounds.x, bounds.w - bounds.y);
    if (extent > laneExtent) {
      laneExtent = extent;
      laneRing = ring;
    }
  }
  let winner = reduceArgMax(lane, laneExtent, laneRing);
  if (winner.cell == NO_CELL) { return vec4<f32>(0.0); }
  return getRingBounds(winner.cell, origin);
}

// Same value as getScanlineSeed. Pass 0 finds the leftmost crossing, pass 1 counts the crossings
// sharing it, pass 2 finds the next crossing to its right.
fn getScanlineSeedCooperative(lane: u32, ringStart: u32, ringEnd: u32, origin: vec2<f32>, lineY: f32) -> vec3<f32> {
  var firstX = LARGE;
  var firstCount = 0u;
  for (var scanPass = 0u; scanPass < 3u; scanPass++) {
    var localX = LARGE;
    var localCount = 0u;
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let ringRowStart = getRingRowStart(ring);
      let ringRowEnd = getRingRowEnd(ring);
      for (var row = ringRowStart + lane; row < ringRowEnd; row += WORKGROUP_SIZE) {
        let current = getPosition(row) - origin;
        let previous = getPosition(select(row - 1u, ringRowEnd - 1u, row == ringRowStart)) - origin;
        if ((current.y > lineY) != (previous.y > lineY)) {
          let crossingX = previous.x + (lineY - previous.y) * (current.x - previous.x) / (current.y - previous.y);
          if (scanPass == 0u) {
            localX = min(localX, crossingX);
          } else if (scanPass == 1u) {
            if (crossingX == firstX) {
              localCount += 1u;
            }
          } else if (crossingX > firstX) {
            localX = min(localX, crossingX);
          }
        }
      }
    }
    if (scanPass == 0u) {
      firstX = reduceMinFloat(lane, localX);
    } else if (scanPass == 1u) {
      firstCount = reduceSumUint(lane, localCount);
    } else {
      let secondX = reduceMinFloat(lane, localX);
      if (firstCount % 2u == 1u && secondX < LARGE) {
        return vec3<f32>(0.5 * (firstX + secondX), lineY, 1.0);
      }
    }
  }
  return vec3<f32>(0.0);
}

// Same result as searchWindow: every lane scores cells lane, lane + 64, ... and the best cell
// (earliest on ties) replaces bestIn only by a strictly larger distance.
fn searchWindowCooperative(
  lane: u32,
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  bestIn: vec3<f32>
) -> vec3<f32> {
  let cell = 2.0 * halfSize / f32(gridSize);
  var laneDistance = -LARGE;
  var laneCell = NO_CELL;
  for (var cellIndex = lane; cellIndex < gridSize * gridSize; cellIndex += WORKGROUP_SIZE) {
    let distance = getSignedDistance(getGridPoint(center, halfSize, cell, gridSize, cellIndex), ringStart, ringEnd, origin);
    if (distance > laneDistance) {
      laneDistance = distance;
      laneCell = cellIndex;
    }
  }
  let winner = reduceArgMax(lane, laneDistance, laneCell);
  if (winner.cell != NO_CELL && winner.distance > bestIn.z) {
    return vec3<f32>(getGridPoint(center, halfSize, cell, gridSize, winner.cell), winner.distance);
  }
  return bestIn;
}
`,G=`GPUMapColoring`,he=`
  if (otherDegree != rowDegree) {
    return otherDegree > rowDegree;
  }`,K=4294967295;function ge(e={}){let t=e.seed??0;if(!Number.isInteger(t)||t<0||t>4294967295)throw Error(`GPUMapColoring seed must be a uint32`);return Uint32Array.of(t)}var _e=class{id;props;maximumIterations;priority;constructor(e){let t=e.id??`map-coloring`;if(this.id=t,this.props=e,this.maximumIterations=e.maximumIterations??64,T(t,`maximumIterations`,this.maximumIterations),this.priority=e.priority??`largest-degree-first`,this.priority!==`largest-degree-first`&&this.priority!==`random`)throw Error(`${t} priority must be 'largest-degree-first' or 'random'`);let n=g(t,e.weights),{output:r}=e;if(p(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<1)throw Error(`${t} parameters must hold 1 uint32`);if(p(r.colors,[`uint32`],`${t} output.colors`),r.colors.length!==n)throw Error(`${t} output.colors length must equal the weights row count`);for(let[e,n]of[[`colorCount`,r.colorCount],[`conflictCount`,r.conflictCount],[`status.converged`,r.status?.converged],[`status.iterationCount`,r.status?.iterationCount]])if(n&&(p(n,[`uint32`],`${t} output.${e}`),n.length<1))throw Error(`${t} output.${e} must hold one uint32`);c(t,ve(r),[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.weights.distances,e.parameters])}getCommandNodes(e){let{id:t,props:n,maximumIterations:r,priority:i}=this,{weights:a,parameters:s,output:c}=n,{colors:l}=c;o(t,e,[a.offsets,a.neighbors,a.weights,a.distances,s,...ve(c)]);let u=l.length,d=D(e,`${t}-rounds`,G,u,t),p=c.colorCount??f(e,`${t}-color-count`,`uint32`,1),g=i===`largest-degree-first`,_=`const UNCOLORED: u32 = ${K}u;

fn priorityHash(value: u32, seed: u32) -> u32 {
  var h = value ^ seed;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

// True when row 'other' is claimed before row 'index' (a strict total order: no two rows tie).
fn outranks(other: u32, otherDegree: u32, index: u32, rowDegree: u32, seed: u32) -> bool {${g?he:``}
  let otherHash = priorityHash(other, seed);
  let rowHash = priorityHash(index, seed);
  return otherHash > rowHash || (otherHash == rowHash && other < index);
}`,v=[m(e,{id:`${t}-colors-clear`,operation:G,view:l,type:`u32`,value:`${K}u`}),m(e,{id:`${t}-color-count-clear`,operation:G,view:p,type:`u32`,value:`0u`,componentCount:1}),w(e,{id:`${t}-rounds-reset`,operation:G,state:d})];for(let n=0;n<r;n++){let i=`${t}-round-${n}`,o=S(d,i);v.push(h(e,{id:`${i}-select`,operation:G,variant:`select-apply`,bindings:[{name:`offsets`,view:a.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:a.neighbors,type:`u32`,access:`read`},{name:`parameters`,view:s,type:`u32`,access:`read`},{name:`colors`,view:l,type:`atomic<u32>`,access:`read_write`},{name:`status`,view:d.status,type:`atomic<u32>`,access:`read_write`},{name:`colorCount`,view:p,type:`atomic<u32>`,access:`read_write`}],invocationCount:u,declarations:_,body:`if (atomicLoad(&colors[colorsOffset + index]) != UNCOLORED) {
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  let seed = parameters[parametersOffset];
  let rowDegree = ${g?`end - begin`:`0u`};
  // One neighbor pass finds both a blocking (uncolored, higher-priority) neighbor and the used
  // colors of the first 32-color window.
  var used = 0u;
  var blocked = false;
  for (var slot = begin; slot < end; slot++) {
    let other = neighbors[neighborsOffset + slot];
    if (other == index) {
      continue;
    }
    let color = atomicLoad(&colors[colorsOffset + other]);
    if (color == UNCOLORED) {
      let otherDegree = ${g?`offsets[offsetsOffset + other + 1u] - offsets[offsetsOffset + other]`:`0u`};
      if (outranks(other, otherDegree, index, rowDegree, seed)) {
        blocked = true;
        break;
      }
    } else if (color < 32u) {
      used = used | (1u << color);
    }
  }
  if (blocked) {
    atomicStore(&status[statusOffset], 1u);
    return;
  }
  var chosen = UNCOLORED;
  if (used != 0xffffffffu) {
    chosen = firstTrailingBit(~used);
  } else {
    // Rare: 32 or more colors used around this row; scan the following windows.
    var base = 32u;
    for (var window = 1u; window <= (end - begin) / 32u; window++) {
      var windowUsed = 0u;
      for (var slot = begin; slot < end; slot++) {
        let other = neighbors[neighborsOffset + slot];
        let color = atomicLoad(&colors[colorsOffset + other]);
        if (other != index && color != UNCOLORED && color >= base && color < base + 32u) {
          windowUsed = windowUsed | (1u << (color - base));
        }
      }
      if (windowUsed != 0xffffffffu) {
        chosen = base + firstTrailingBit(~windowUsed);
        break;
      }
      base += 32u;
    }
  }
  if (chosen != UNCOLORED) {
    atomicStore(&colors[colorsOffset + index], chosen);
    atomicMax(&colorCount[colorCountOffset], chosen + 1u);
  }`,condition:o.condition,extraResources:o.extraResources}),C(e,{id:`${i}-gate`,operation:G,state:d,maxIterations:r}))}return(c.status?.converged||c.status?.iterationCount)&&v.push(E(e,{id:`${t}-status`,operation:G,state:d,converged:c.status?.converged,iterationCount:c.status?.iterationCount})),c.conflictCount&&v.push(m(e,{id:`${t}-conflicts-clear`,operation:G,view:c.conflictCount,type:`u32`,value:`0u`,componentCount:1}),h(e,{id:`${t}-conflicts`,operation:G,variant:`conflicts`,bindings:[{name:`offsets`,view:a.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:a.neighbors,type:`u32`,access:`read`},{name:`colors`,view:l,type:`u32`,access:`read`},{name:`conflictCount`,view:c.conflictCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:u,body:`let color = colors[colorsOffset + index];
  var conflicts = 0u;
  if (color != ${K}u) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let other = neighbors[neighborsOffset + slot];
      if (other > index && colors[colorsOffset + other] == color) {
        conflicts++;
      }
    }
  }
  if (conflicts > 0u) {
    atomicAdd(&conflictCount[conflictCountOffset], conflicts);
  }`})),v}};function ve(e){return[e.colors,e.colorCount,e.conflictCount,e.status?.converged,e.status?.iterationCount]}var ye={coordinateSpace:`longitude-latitude`,datum:`EPSG:4326`,metric:`great-circle`,units:`meters`,edgeModel:`linear`},be=[`Fiji box`,`Antarctica band`,`Arctic ring`],q=3,J=1024;function xe(){let e=[177,-19,-178,-19,-178,-16,177,-16],t=[];for(let e=-180;e<=180;e+=10)t.push(e,-70);t.push(180,-90,-180,-90);let n=[];for(let e=-180;e<180;e+=10)n.push(e,80);return{positions:Float32Array.from([...e,...t,...n]),featureOffsets:new Uint32Array([0,1,2,3]),polygonOffsets:new Uint32Array([0,1,2,3]),ringOffsets:new Uint32Array([0,4,43,79])}}function Se(){let e=i(ye),t=Math.PI/180,n=2*Math.PI*e**2;return[e**2*5*t*Math.abs(Math.sin(-16*t)-Math.sin(-19*t)),n*(1-Math.sin(70*t)),n*(1-Math.sin(80*t))]}async function Ce(t){let n=new u(t.device,`polygon-measures-global`);try{let r=xe(),i=Se(),o={positions:n.createBuffer(`positions`,r.positions),featureOffsets:n.createBuffer(`features`,r.featureOffsets),polygonOffsets:n.createBuffer(`polygons`,r.polygonOffsets),ringOffsets:n.createBuffer(`rings`,r.ringOffsets)},s=n.createBuffer(`raw-areas`,q*4),c=n.createBuffer(`prepared-areas`,q*4),l=n.createBuffer(`relative-errors`,q*4),u=Float32Array.from(i.flatMap(e=>{let t=Math.fround(e);return[t,e-t]})),d=n.createBuffer(`reference-areas`,u),f=n.createBuffer(`raw-invalid`,4),p=n.createBuffer(`prepared-invalid`,4),m=n.createBuffer(`preparation-invalid`,4),h=n.createBuffer(`overflow`,4),g=n.createBuffer(`required-count`,4),_=n.createBuffer(`ring-flags`,q*4),y={positions:n.createBuffer(`prepared-positions`,J*8),featureOffsets:n.createBuffer(`prepared-features`,(q+1)*4),polygonOffsets:n.createBuffer(`prepared-polygons`,(2*q+1)*4),ringOffsets:n.createBuffer(`prepared-rings`,(2*q+1)*4)},b=n.createBuffer(`corners`,J*6*4),x=n.createBuffer(`feature-rows`,J*4),S=n.createBuffer(`starts`,J*8),C=n.createBuffer(`ends`,J*8),w=e=>({positions:e(`positions`,o.positions,`float32x2`,r.positions.length/2),featureOffsets:e(`features`,o.featureOffsets,`uint32`,q+1),polygonOffsets:e(`polygons`,o.polygonOffsets,`uint32`,q+1),ringOffsets:e(`rings`,o.ringOffsets,`uint32`,q+1)}),T=new a(t.device,{id:`global-raw-measures`}),E=F(T);T.add(new v({...w(E),spatialContext:ye,geometryType:`polygons`,output:{areas:E(`areas`,s,`float32`,q),status:{invalidCount:E(`invalid`,f,`uint32`,1)}}}));let D=n.track(await T.compileAsync()),k=new a(t.device,{id:`global-prepared-measures`}),A=F(k),j={positions:A(`prepared-positions`,y.positions,`float32x2`,J),featureOffsets:A(`prepared-features`,y.featureOffsets,`uint32`,q+1),polygonOffsets:A(`prepared-polygons`,y.polygonOffsets,`uint32`,2*q+1),ringOffsets:A(`prepared-rings`,y.ringOffsets,`uint32`,2*q+1)},te=A(`overflow`,h,`uint32`,1),N=n.createParameterBuffer(`preparation-parameters`,`float32`,4,O(`linear`)),re=new ee({...w(A),spatialContext:ye,parameters:N.importToGraph(k),output:{...j,ringFlags:A(`ring-flags`,_,`uint32`,q),status:{overflow:te,requiredCount:A(`required-count`,g,`uint32`,1),invalidCount:A(`preparation-invalid`,m,`uint32`,1)}}});k.add(re),k.add(new v({...j,spatialContext:{...re.spatialContext,metric:`great-circle`,units:`meters`},geometryType:`polygons`,output:{areas:A(`areas`,c,`float32`,q),status:{invalidCount:A(`measures-invalid`,p,`uint32`,1)}}})),M(k,{id:`global-reference-comparison`,invocationCount:q,bindings:[{name:`areas`,view:A(`areas`,c,`float32`,q),type:`f32`,access:`read`},{name:`references`,view:A(`reference-areas`,d,`float32`,q*2),type:`f32`,access:`read`},{name:`errors`,view:A(`relative-errors`,l,`float32`,q),type:`f32`,access:`read_write`}],body:`
  let high = references[referencesOffset + index * 2u];
  let low = references[referencesOffset + index * 2u + 1u];
  errors[errorsOffset + index] = abs((areas[areasOffset + index] - high) - low) / high;`}),M(k,{id:`global-display`,invocationCount:J,bindings:[{name:`positions`,view:j.positions,type:`f32`,access:`read`},{name:`rings`,view:j.ringOffsets,type:`u32`,access:`read`},{name:`overflow`,view:te,type:`u32`,access:`read`},{name:`corners`,view:A(`corners`,b,`float32`,J*6),type:`f32`,access:`read_write`},{name:`features`,view:A(`feature-rows`,x,`uint32`,J),type:`u32`,access:`read_write`},{name:`starts`,view:A(`starts`,S,`float32`,J*2),type:`f32`,access:`read_write`},{name:`ends`,view:A(`ends`,C,`float32`,J*2),type:`f32`,access:`read_write`}],declarations:`
fn displayPosition(vertex: u32) -> vec2<f32> {
  // Web Mercator display clamp only. Measures consume the original prepared latitudes at ±90.
  return vec2<f32>(positions[positionsOffset + vertex * 2u],
    clamp(positions[positionsOffset + vertex * 2u + 1u], -85.0511, 85.0511));
}`,body:`
  // WGSL forbids NaN constants, so bitcast a runtime bit pattern.
  var nanBits = 0x7fc00000u;
  var first = vec2<f32>(bitcast<f32>(nanBits));
  var start = first;
  var end = first;
  var feature = 0u;
  if (overflow[overflowOffset] == 0u) {
    for (var ring = 0u; ring < 6u; ring++) {
      let begin = rings[ringsOffset + ring];
      let finish = rings[ringsOffset + ring + 1u];
      if (index >= begin && index < finish && finish - begin >= 3u) {
        first = displayPosition(begin);
        start = displayPosition(index);
        var following = index + 1u;
        if (following == finish) { following = begin; }
        end = displayPosition(following);
        // One input polygon and shell per feature: main/east copies retain this fixed order.
        feature = ring / 2u;
      }
    }
  }
  features[featuresOffset + index] = feature;
  corners[cornersOffset + index * 6u] = first.x;
  corners[cornersOffset + index * 6u + 1u] = first.y;
  corners[cornersOffset + index * 6u + 2u] = start.x;
  corners[cornersOffset + index * 6u + 3u] = start.y;
  corners[cornersOffset + index * 6u + 4u] = end.x;
  corners[cornersOffset + index * 6u + 5u] = end.y;
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`});let ae=n.track(await k.compileAsync()),I=new ne(n,`global-areas`,[{buffer:s,size:12},{buffer:c,size:12},{buffer:l,size:12},{buffer:f,size:4},{buffer:m,size:4},{buffer:p,size:4},{buffer:h,size:4},{buffer:g,size:4},{buffer:_,size:12}],e=>{let n=new Float32Array(e,0,q),r=new Float32Array(e,12,q),a=new Float32Array(e,24,q),o=new Uint32Array(e,36,8),s=e=>Number.isFinite(e)?(e/1e6).toLocaleString(`en-US`,{maximumFractionDigits:1}):`rejected`;for(let e=0;e<q;e++){let c=a[e];t.setReadout(`globalArea${e}`,[`Known km²     ${s(i[e])}`,`Unprepared    ${s(n[e])}`,`Prepared      ${o[3]?`overflow`:s(r[e])}`,`Relative error ${!o[3]&&Number.isFinite(c)?`${(c*100).toFixed(5)}%`:`—`}`].join(`
`))}t.setReadout(`globalRejected`,`${o[0]} features rejected: ambiguous or over-wide rings`),t.setReadout(`globalPreparation`,[`Preparation: ${o[1]} rejected; measures: ${o[2]} rejected`,...be.map((e,t)=>{let n=o[5+t];return`${e}: ${[n&1?`split`:``,n&2?`pole-capped`:``].filter(Boolean).join(`, `)||`kept`}`}),`Overflow: ${o[3]}; required: ${o[4]} / ${J}`].join(`
`)),t.setStatus(o[3]?`Preparation overflow; requires ${o[4]} positions`:``)}),L=!0;return{getCompiledGraphs:()=>[D,ae],encode(e){L?(D.encode(e,{parameters:void 0}),ae.encode(e,{parameters:void 0}),I.request(e),L=!1):I.flush(e)},setOption(){t.requestLayers()},onThemeChange(){t.requestLayers()},getLayers(){let n=[new ie({id:`global-polygon-fill`,coordinateSystem:e.LNGLAT,corners:b,featureRows:x,instanceCount:J,values:c,valueMapping:`ramp`,valueRange:[0,0xe8d4a510000],color:[255,255,255,Math.round(t.options.opacity*255)]})];return t.options.showOutlines&&n.push(new P({id:`global-polygon-outline`,coordinateSystem:e.LNGLAT,starts:S,ends:C,instanceCount:J,color:t.theme()===`dark`?[235,240,248,220]:[40,52,70,220],widthPixels:1.1})),n},destroy(){I.stop(),n.destroy()}}}catch(e){throw n.destroy(),e}}var Y=32,we=t,Te=[3,5,7];Te.map(e=>re[e]);var X=N({breaks:[5e8,1e9,2e9,5e9,1e10],scheme:`YlOrBr`,unit:`km²`,extent:[0,1e10],labels:[`0–500`,`500–1k`,`1k–2k`,`2k–5k`,`5k–10k`,`10k+`],method:`Fixed log-spaced reference classes`}),Ee=N({breaks:[1.05,1.2,1.5,2],scheme:`PuRd`,unit:`×`,extent:[1,2.5],labels:[`1.00–1.05×`,`1.05–1.20×`,`1.20–1.50×`,`1.50–2.00×`,`2.00×+`],method:`Planar / WGS84; ratios begin at 1.0`}),De=N({breaks:[.1,.25,.4,.6],scheme:`BuGn`,unit:`Polsby-Popper`,extent:[0,1],labels:[`< 0.10`,`0.10–0.25`,`0.25–0.40`,`0.40–0.60`,`0.60–1.00`],method:`Fixed compactness reference classes`}),Oe=N({breaks:[.5,1.5],colors:[[140,149,160,255],[230,159,0,255],[213,94,0,255]],labels:[`Valid`,`Orientation convention only`,`Structural defect`],method:`Amber convention, vermilion structure`});function ke(e){return e===`area`?X:e===`areaDistortion`?Ee:e===`compactness`?De:e===`validity`?Oe:null}var Ae={"us-counties":{space:`mercator`,groupColumn:`stateFips`,describe:e=>`${e.name} County, ${e.state}`,groupName:e=>`state ${e}`},"chicago-community-areas":{space:`local`,groupColumn:null,describe:e=>String(e.name),groupName:e=>`area ${e}`},"chicago-tracts":{space:`local`,groupColumn:`communityArea`,describe:e=>`Tract ${e.GEOID} (community area ${e.communityArea})`,groupName:e=>`community area ${e}`}};async function je(t){let{device:n}=t,r=!1,i=null,o=new Map,s=e=>{let n=o.get(e);return n||(n=R(t.datasets.get(e)),o.set(e,n)),n};async function c(e){let o=t.datasets.get(e),c=Ae[e],l=s(e),{featureCount:d,ringCount:f,vertexCount:p,partCount:m}=l,h=new u(n,`polygon-measures-${e}`),b=o.defaultOrigin,x=c.space===`mercator`?ce:o.getProjection(b),S=c.space===`mercator`?oe(l.lngLat):o.projectColumn(`vertices`,b),C=new Uint32Array(d),w=1;if(c.groupColumn){let e=o.column(c.groupColumn),t=new Map;for(let n=0;n<d;n++){let r=e[n];t.has(r)||t.set(r,t.size),C[n]=t.get(r)}w=t.size}let T=[],E=o.geojson?.features??[];for(let e=0;e<d;e++)T.push(c.describe(E[e]?.properties??{},e));let D=T.findIndex(e=>/o['’]?hare/i.test(e)),O=h.createBuffer(`lnglat`,l.lngLat),ee=h.createBuffer(`planar`,S),j=h.createBuffer(`ring-offsets`,l.ringOffsets),te=h.createBuffer(`feature-rings`,l.featureRingOffsets),N=h.createBuffer(`polygon-offsets`,l.polygonOffsets),P=h.createBuffer(`feature-offsets`,l.featureOffsets),ie=h.createBuffer(`merged-feature-offsets`,Uint32Array.from({length:d+1},(e,t)=>t)),I=h.createBuffer(`group-ids`,C),L=h.createBuffer(`validity-positions`,l.lngLat),R=ae(l),se=le(l),z={corners:h.createBuffer(`fill-corners`,R.corners),deflatedCorners:h.createBuffer(`deflated-fill-corners`,R.corners.byteLength),sourceCorners:R.corners,sourceFeatureRows:R.featureRows,featureRows:h.createBuffer(`fill-feature-rows`,R.featureRows),triangleCount:R.triangleCount},V={starts:h.createBuffer(`edge-starts`,se.starts),ends:h.createBuffer(`edge-ends`,se.ends),edgeCount:se.edgeCount},H=1;for(let e=0;e<d;e++){let t=l.ringOffsets[l.featureRingOffsets[e]],n=l.ringOffsets[l.featureRingOffsets[e+1]];H=Math.max(H,n-t)}let U=(e,t=d)=>h.createBuffer(e,Math.max(t,1)*4),W={areaPlanar:U(`area-planar`),areaSphere:U(`area-sphere`),areaWgs84:U(`area-wgs84`),lengthPlanar:U(`length-planar`),lengthSphere:U(`length-sphere`),lengthWgs84:U(`length-wgs84`),ratio:U(`ratio`),vertexF:U(`vertex-f`),groupAreaF:U(`group-area-f`),polsbyPopper:U(`polsby-popper`),schwartzberg:U(`schwartzberg`),elongation:U(`elongation`),convexity:U(`convexity`),sliverF:U(`sliver-f`),validityF:U(`validity-f`),colorF:U(`color-f`),naiveColorF:U(`naive-color-f`)},fe=U(`centroids-planar`,d*2),me=U(`centroids-wgs84`,d*2),G=U(`bounds-wgs84`,d*4),he=U(`vertex-counts`),K=U(`planar-invalid-count`,1),ve=U(`sphere-invalid-count`,1),ye=U(`wgs84-invalid-count`,1),be=U(`group-areas`,w),q=U(`group-centroids`,w*2),J=U(`group-feature-counts`,w),xe=U(`label-points`,d*2),Se=U(`label-radii`),Ce=U(`label-degenerate`),X=U(`clockwise`),Ee=U(`sliver`),De=U(`orientation`),Oe=U(`circle-starts`,d*Y*2),ke=U(`circle-ends`,d*Y*2),je=U(`axis-starts`,d*2),Me=U(`axis-ends`,d*2),Ne=U(`label-lnglat`,d*2),Pe=U(`bounds-starts`,d*4*2),Fe=U(`bounds-ends`,d*4*2),Ie=U(`selected-starts`,H*2),Le=U(`selected-ends`,H*2),Re=U(`label-grid-starts`,264),ze=U(`label-grid-ends`,264),Be=e===`us-counties`?o.column(`fips`):null,Ve=new Float32Array(d);for(let e=0;e<d;e++)Ve[e]=(Be?.[e]??e)%re.length;W.naiveColorF.write(Ve);let He=U(`validity-mask`),Ue=h.createBuffer(`validity-overflow`,4),We=h.createParameterBuffer(`shape-parameters`,`float32`,4,k({sliverThreshold:t.options.sliverThreshold})),Ge=h.createParameterBuffer(`coloring-parameters`,`uint32`,1,ge({seed:t.options.colorSeed})),Ke=(e,t)=>({featureOffsets:t===`parts`?e(`feature-offsets`,P,`uint32`,d+1):e(`feature-offsets`,ie,`uint32`,d+1),polygonOffsets:t===`parts`?e(`polygon-offsets`,N,`uint32`,m+1):e(`polygon-offsets`,te,`uint32`,d+1),ringOffsets:e(`ring-offsets`,j,`uint32`,f+1)}),qe=(()=>{if(c.space===`mercator`)return`
fn toLngLat(p: vec2<f32>) -> vec2<f32> {
  let r = ${we.toFixed(1)};
  return vec2<f32>(p.x / r * 57.29578, (2.0 * atan(exp(p.y / r)) - 1.5707963) * 57.29578);
}`;let[e]=x.project(b[0]+1,b[1]),[,t]=x.project(b[0],b[1]+1);return`
fn toLngLat(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(${b[0]}, ${b[1]}) + p / vec2<f32>(${e.toFixed(3)}, ${t.toFixed(3)});
}`})(),Je=async({ringGrouping:t})=>{let r=new a(n,{id:`measures-${e}`}),i=F(r),o=i(`lnglat`,O,`float32x2`,p),s=i(`planar`,ee,`float32x2`,p),l=Ke(i,t),u=(e,t,n=d)=>i(e,t,`float32`,n),f=u(`area-planar`,W.areaPlanar),m=u(`area-wgs84`,W.areaWgs84),g={geometryType:`polygons`,...l};r.add(new v({spatialContext:x.spatialContext,id:`measures-planar`,positions:s,...g,output:{areas:f,lengths:u(`length-planar`,W.lengthPlanar),centroids:i(`centroids-planar`,fe,`float32x2`,d),status:{invalidCount:i(`planar-invalid-count`,K,`uint32`,1)}}})),r.add(new v({spatialContext:{coordinateSpace:`longitude-latitude`,datum:`EPSG:4326`,metric:`great-circle`,units:`meters`},id:`measures-sphere`,positions:o,...g,output:{areas:u(`area-sphere`,W.areaSphere),lengths:u(`length-sphere`,W.lengthSphere),status:{invalidCount:i(`sphere-invalid-count`,ve,`uint32`,1)}}}));let _=c.groupColumn?{groupIds:i(`group-ids`,I,`uint32`,d),groupCount:w,groupOutput:{areas:u(`group-areas`,be,w),centroids:i(`group-centroids`,q,`float32x2`,w),featureCounts:i(`group-feature-counts`,J,`uint32`,w)}}:{};r.add(new v({spatialContext:{coordinateSpace:`longitude-latitude`,datum:`EPSG:4326`,metric:`ellipsoidal`,units:`meters`},ellipsoidalEdgeModel:`coordinate-linear`,id:`measures-wgs84`,positions:o,...g,output:{areas:m,lengths:u(`length-wgs84`,W.lengthWgs84),centroids:i(`centroids-wgs84`,me,`float32x2`,d),bounds:i(`bounds-wgs84`,G,`float32x4`,d),vertexCounts:i(`vertex-counts`,he,`uint32`,d),status:{invalidCount:i(`wgs84-invalid-count`,ye,`uint32`,1)}},..._}));let y=u(`ratio`,W.ratio);return M(r,{id:`area-distortion`,invocationCount:d,bindings:[{name:`planar`,view:f,type:`f32`,access:`read`},{name:`ellipsoid`,view:m,type:`f32`,access:`read`},{name:`ratio`,view:y,type:`f32`,access:`read_write`}],body:`ratio[ratioOffset + index] = planar[planarOffset + index] / max(ellipsoid[ellipsoidOffset + index], 1e-9);`}),M(r,{id:`vertex-count-to-float`,invocationCount:d,bindings:[{name:`counts`,view:i(`vertex-counts`,he,`uint32`,d),type:`u32`,access:`read`},{name:`values`,view:u(`vertex-f`,W.vertexF),type:`f32`,access:`read_write`}],body:`values[valuesOffset + index] = f32(counts[countsOffset + index]);`}),M(r,{id:`gather-group-area`,invocationCount:d,bindings:[{name:`groups`,view:i(`group-ids`,I,`uint32`,d),type:`u32`,access:`read`},{name:`totals`,view:u(`group-areas`,be,w),type:`f32`,access:`read`},{name:`values`,view:u(`group-area-f`,W.groupAreaF),type:`f32`,access:`read_write`}],declarations:`const HAS_GROUPS: bool = ${c.groupColumn!==null};`,body:`
  var value = f32(index) * 0.0;
  value = value / value;
  if (HAS_GROUPS) {
    value = totals[totalsOffset + groups[groupsOffset + index]];
  }
  values[valuesOffset + index] = value;`}),M(r,{id:`bounds-boxes`,invocationCount:d,bindings:[{name:`bounds`,view:i(`bounds-wgs84`,G,`float32x4`,d),type:`f32`,access:`read`},{name:`starts`,view:i(`bounds-starts`,Pe,`float32`,d*8),type:`f32`,access:`read_write`},{name:`ends`,view:i(`bounds-ends`,Fe,`float32`,d*8),type:`f32`,access:`read_write`}],body:`
  let b = vec4<f32>(
    bounds[boundsOffset + index * 4u], bounds[boundsOffset + index * 4u + 1u],
    bounds[boundsOffset + index * 4u + 2u], bounds[boundsOffset + index * 4u + 3u]
  );
  var corners = array<vec2<f32>, 4>(
    vec2<f32>(b.x, b.y), vec2<f32>(b.z, b.y), vec2<f32>(b.z, b.w), vec2<f32>(b.x, b.w)
  );
  for (var side = 0u; side < 4u; side++) {
    let a = corners[side];
    let c = corners[(side + 1u) % 4u];
    let slot = (index * 4u + side) * 2u;
    starts[startsOffset + slot] = a.x;
    starts[startsOffset + slot + 1u] = a.y;
    ends[endsOffset + slot] = c.x;
    ends[endsOffset + slot + 1u] = c.y;
  }`}),h.track(await r.compileAsync())},Ye=async t=>{let r=new a(n,{id:`shape-${e}`}),i=F(r),o=i(`planar`,ee,`float32x2`,p),s=Ke(i,`parts`),c=Ke(i,t.ringGrouping),l=i(`label-points`,xe,`float32x2`,d),u=i(`label-radii`,Se,`float32`,d),f=i(`label-degenerate`,Ce,`uint32`,d);r.add(new pe({spatialContext:x.spatialContext,id:`label-point`,positions:o,...s,initialGridSize:t.initialGridSize,refinementCandidates:t.refinementCandidates,output:{points:l,distances:u,degenerate:f}}));let m=(e,t)=>i(e,t,`float32`,d),g=m(`elongation`,W.elongation),_=m(`orientation`,De);return r.add(new A({spatialContext:x.spatialContext,id:`shape-descriptors`,positions:o,...c,parameters:We.importToGraph(r),output:{polsbyPopper:m(`polsby-popper`,W.polsbyPopper),schwartzberg:m(`schwartzberg`,W.schwartzberg),elongation:g,orientation:_,convexity:m(`convexity`,W.convexity),clockwise:i(`clockwise`,X,`uint32`,d),sliver:i(`sliver`,Ee,`uint32`,d)}})),M(r,{id:`sliver-to-float`,invocationCount:d,bindings:[{name:`flags`,view:i(`sliver`,Ee,`uint32`,d),type:`u32`,access:`read`},{name:`values`,view:m(`sliver-f`,W.sliverF),type:`f32`,access:`read_write`}],body:`values[valuesOffset + index] = f32(flags[flagsOffset + index]);`}),M(r,{id:`label-drawing`,invocationCount:d*Y,bindings:[{name:`points`,view:l,type:`f32`,access:`read`},{name:`radii`,view:u,type:`f32`,access:`read`},{name:`degenerate`,view:f,type:`u32`,access:`read`},{name:`lngLat`,view:i(`label-lnglat`,Ne,`float32`,d*2),type:`f32`,access:`read_write`},{name:`starts`,view:i(`circle-starts`,Oe,`float32`,d*Y*2),type:`f32`,access:`read_write`},{name:`ends`,view:i(`circle-ends`,ke,`float32`,d*Y*2),type:`f32`,access:`read_write`}],declarations:`const SEGMENTS: u32 = ${Y}u;\n${qe}`,body:`
  let feature = index / SEGMENTS;
  let step = index % SEGMENTS;
  let center = vec2<f32>(points[pointsOffset + feature * 2u], points[pointsOffset + feature * 2u + 1u]);
  let radius = radii[radiiOffset + feature];
  if (step == 0u) {
    let anchor = toLngLat(center);
    lngLat[lngLatOffset + feature * 2u] = anchor.x;
    lngLat[lngLatOffset + feature * 2u + 1u] = anchor.y;
  }
  let zero = f32(index) * 0.0;
  var start = vec2<f32>(zero / zero);
  var end = start;
  if (degenerate[degenerateOffset + feature] == 0u && radius == radius) {
    let angle0 = 6.2831853 * f32(step) / f32(SEGMENTS);
    let angle1 = 6.2831853 * f32(step + 1u) / f32(SEGMENTS);
    start = toLngLat(center + radius * vec2<f32>(cos(angle0), sin(angle0)));
    end = toLngLat(center + radius * vec2<f32>(cos(angle1), sin(angle1)));
  }
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`}),M(r,{id:`axis-glyphs`,invocationCount:d,bindings:[{name:`points`,view:l,type:`f32`,access:`read`},{name:`radii`,view:u,type:`f32`,access:`read`},{name:`orientation`,view:_,type:`f32`,access:`read`},{name:`elongation`,view:g,type:`f32`,access:`read`},{name:`starts`,view:i(`axis-starts`,je,`float32`,d*2),type:`f32`,access:`read_write`},{name:`ends`,view:i(`axis-ends`,Me,`float32`,d*2),type:`f32`,access:`read_write`}],declarations:qe,body:`
  let center = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let angle = orientation[orientationOffset + index];
  let halfLength = max(radii[radiiOffset + index], 0.0) * (1.0 + 3.0 * elongation[elongationOffset + index]);
  let direction = vec2<f32>(cos(angle), sin(angle)) * halfLength;
  let start = toLngLat(center - direction);
  let end = toLngLat(center + direction);
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`}),h.track(await r.compileAsync())},Xe=async t=>{let r=new a(n,{id:`validity-${e}`}),i=F(r),o=i(`mask`,He,`uint32`,d);return r.add(new de({id:`validity`,polygons:{kind:`polygons`,positions:i(`validity-positions`,L,`float32x2`,p),...Ke(i,`parts`)},output:{mask:o,status:{overflow:i(`validity-overflow`,Ue,`uint32`,1)}},intersectionCapacity:Math.max(4096,d*4),ringClosure:t.ringClosure,orientation:t.orientation})),M(r,{id:`validity-category`,invocationCount:d,bindings:[{name:`mask`,view:o,type:`u32`,access:`read`},{name:`category`,view:i(`validity-f`,W.validityF,`float32`,d),type:`f32`,access:`read_write`}],declarations:`const STRUCTURAL: u32 = ${ue}u;\nconst REPEATED: u32 = ${B.repeatedVertex}u;`,body:`
  let bits = mask[maskOffset + index] & ~REPEATED;
  var value = ${Te[0]}.0;
  if ((bits & STRUCTURAL) != 0u) {
    value = ${Te[2]}.0;
  } else if (bits != 0u) {
    value = ${Te[1]}.0;
  }
  category[categoryOffset + index] = value;`}),h.track(await r.compileAsync())},Ze=Math.max(1024,d*24),Qe=h.createBuffer(`weight-offsets`,(d+1)*4),$e=h.createBuffer(`weight-neighbors`,Ze*4),et=h.createBuffer(`weight-values`,Ze*4),tt=h.createBuffer(`weight-overflow`,4),nt=h.createBuffer(`colors`,d*4),rt=h.createBuffer(`color-summary`,4),Z={conflict:h.createBuffer(`conflict-count`,4),converged:h.createBuffer(`coloring-converged`,4),rounds:h.createBuffer(`coloring-rounds`,4)},it=async t=>{let r=new a(n,{id:`coloring-${e}`}),i=F(r),o={offsets:i(`weight-offsets`,Qe,`uint32`,d+1),neighbors:i(`weight-neighbors`,$e,`uint32`,Ze),weights:i(`weight-values`,et,`float32`,Ze),status:{overflow:i(`weight-overflow`,tt,`uint32`,1)}};r.add(new _({id:`contiguity`,criterion:t.contiguity,polygons:{kind:`polygons`,positions:i(`lnglat`,O,`float32x2`,p),...Ke(i,`parts`)},pairCapacity:Math.max(65536,p*16),output:o}));let s=i(`colors`,nt,`uint32`,d);return r.add(new _e({id:`coloring`,weights:o,parameters:Ge.importToGraph(r),maximumIterations:t.maxRounds,output:{colors:s,colorCount:i(`color-count`,rt,`uint32`,1),conflictCount:i(`conflict-count`,Z.conflict,`uint32`,1),status:{converged:i(`coloring-converged`,Z.converged,`uint32`,1),iterationCount:i(`coloring-rounds`,Z.rounds,`uint32`,1)}}})),M(r,{id:`colors-to-float`,invocationCount:d,bindings:[{name:`colors`,view:s,type:`u32`,access:`read`},{name:`values`,view:i(`color-f`,W.colorF,`float32`,d),type:`f32`,access:`read_write`}],body:`values[valuesOffset + index] = f32(colors[colorsOffset + index]);`}),h.track(await r.compileAsync())},Q={id:e,layout:l,names:T,groupIds:C,groupCount:w,resources:h,fill:z,edges:V,columns:W,overlays:{labelPoints:Ne,circleStarts:Oe,circleEnds:ke,axisStarts:je,axisEnds:Me,centroids:me,boundsStarts:Pe,boundsEnds:Fe,selectedStarts:Ie,selectedEnds:Le,labelGridStarts:Re,labelGridEnds:ze},labelFeature:D,labelGridCount:0,validityOverflow:0,measuresInvalidCounts:{planar:0,spherical:0,wgs84:0},selectedEdgeCapacity:H,builders:{measures:Je,shape:Ye,validity:Xe,coloring:it},measures:await Je(t.options),shape:await Ye(t.options),validity:await Xe(t.options),coloring:await it(t.options),shapeParameters:We,coloringParameters:Ge,validityPositions:L,tableReader:null,coloringReader:null,table:null,coloringTable:null,dirty:{measures:!0,shape:!0,validity:!0,coloring:!0,table:!0},selected:-1,selectedCount:0},$=(e,t)=>({buffer:e,size:t*4}),at=[$(W.areaPlanar,d),$(W.areaSphere,d),$(W.areaWgs84,d),$(W.lengthPlanar,d),$(W.lengthSphere,d),$(W.lengthWgs84,d),$(he,d),$(W.polsbyPopper,d),$(W.schwartzberg,d),$(W.elongation,d),$(W.convexity,d),$(Ee,d),$(X,d),$(xe,d*2),$(Se,d),$(me,d*2),$(He,d),$(be,w),{buffer:Ue,size:4},{buffer:K,size:4},{buffer:ve,size:4},{buffer:ye,size:4}];return Q.tableReader=new ne(h,`table-${e}`,at,e=>{if(r||i!==Q)return;let t=0,n=n=>{let r=new Float32Array(e,t,n);return t+=n*4,r},a=n=>{let r=new Uint32Array(e,t,n);return t+=n*4,r};Q.table={areaPlanar:n(d),areaSphere:n(d),areaWgs84:n(d),lengthPlanar:n(d),lengthSphere:n(d),lengthWgs84:n(d),vertexCount:a(d),polsbyPopper:n(d),schwartzberg:n(d),elongation:n(d),convexity:n(d),sliver:a(d),clockwise:a(d),labelPoint:n(d*2),radius:n(d),centroid:n(d*2),mask:a(d),groupAreas:n(w)},Q.validityOverflow=a(1)[0],Q.measuresInvalidCounts={planar:a(1)[0],spherical:a(1)[0],wgs84:a(1)[0]},g(Q)}),Q.coloringReader=new ne(h,`coloring-${e}`,[{buffer:rt,size:4},{buffer:Z.conflict,size:4},{buffer:Z.converged,size:4},{buffer:Z.rounds,size:4},{buffer:tt,size:4},{buffer:Qe,size:(d+1)*4}],e=>{if(r||i!==Q)return;let t=new Uint32Array(e);Q.coloringTable={colorCount:t[0],conflictCount:t[1],converged:t[2],roundCount:t[3],overflow:t[4],adjacencies:t[5+d]},y(Q)}),Q}function d(e,t){let n=t.areaSystem;switch(t.metric){case`area`:return n===`planar`?e.areaPlanar:n===`spherical`?e.areaSphere:e.areaWgs84;case`perimeter`:return n===`planar`?e.lengthPlanar:n===`spherical`?e.lengthSphere:e.lengthWgs84;case`areaDistortion`:{let t=new Float32Array(e.areaPlanar.length);for(let n=0;n<t.length;n++)t[n]=e.areaPlanar[n]/Math.max(e.areaWgs84[n],1e-9);return t}case`vertices`:return Float32Array.from(e.vertexCount);case`groupArea`:{let t=i;if(!t)return null;let n=new Float32Array(t.groupIds.length);for(let r=0;r<n.length;r++)n[r]=e.groupAreas[t.groupIds[r]];return n}case`compactness`:return e.polsbyPopper;case`schwartzberg`:return e.schwartzberg;case`elongation`:return e.elongation;case`convexity`:return e.convexity;default:return null}}function f(e,t){let n=L(t,e===`areaDistortion`?0:.02),r=L(t,.98);return e===`compactness`||e===`schwartzberg`||e===`convexity`||e===`elongation`?[0,Math.max(r,.05)]:e===`areaDistortion`?[Math.min(n,1),Math.max(r,1.01)]:[Math.min(n,r),Math.max(r,n+1e-6)]}let p=[0,1];function m(e,t){let n=new Float32Array(e.fill.sourceCorners.length);for(let r=0;r<n.length/2;r++){let i=e.fill.sourceFeatureRows[Math.floor(r/3)],a=t.areaPlanar[i]/Math.max(t.areaWgs84[i],1e-9),o=1/Math.sqrt(Math.max(a,1)),s=e.fill.sourceCorners[r*2],c=e.fill.sourceCorners[r*2+1],l=t.centroid[i*2],u=t.centroid[i*2+1];n[r*2]=l+(s-l)*o,n[r*2+1]=u+(c-u)*o}e.fill.deflatedCorners.write(n)}function h(e,t){let n=e.labelFeature;if(n<0){e.labelGridCount=0;return}let r=e.layout,i=1/0,a=1/0,o=-1/0,s=-1/0;for(let e=r.featureRingOffsets[n];e<r.featureRingOffsets[n+1];e++)for(let t=r.ringOffsets[e];t<r.ringOffsets[e+1];t++){let e=r.lngLat[t*2],n=r.lngLat[t*2+1];i=Math.min(i,e),o=Math.max(o,e),a=Math.min(a,n),s=Math.max(s,n)}let c=[],l=[],u=(e,t,n,r)=>{c.push(e,t),l.push(n,r)},d=t.initialGridSize;for(let e=0;e<=d;e++){let t=e/d;u(i+(o-i)*t,a,i+(o-i)*t,s),u(i,a+(s-a)*t,o,a+(s-a)*t)}let f=e.table?.labelPoint;if(f&&Number.isFinite(f[n*2])){let e=Math.max((o-i)/d,(s-a)/d)*1.5;for(let t=0;t<=4;t++){let r=t/4-.5;u(f[n*2]+r*e,f[n*2+1]-e/2,f[n*2]+r*e,f[n*2+1]+e/2),u(f[n*2]-e/2,f[n*2+1]+r*e,f[n*2]+e/2,f[n*2+1]+r*e)}}e.labelGridCount=c.length/2,e.overlays.labelGridStarts.write(Float32Array.from(c)),e.overlays.labelGridEnds.write(Float32Array.from(l))}function g(e){let n=e.table;if(!n)return;let r=t.options,i=e.layout.featureCount,a=d(n,r);a&&(p=f(r.metric,a),t.setLegendExtent(`value`,p),t.requestLayers()),m(e,n),h(e,r);let o=e.measuresInvalidCounts[r.areaSystem];t.setReadout(`features`,o>0?`${l(i)}; ${o} features rejected: ambiguous or over-wide rings`:i),t.setReadout(`vertices`,e.layout.vertexCount);let s=0,c=0,u=0,g=0,_=0,v=0,y=0;for(let e=0;e<i;e++){s+=n.areaPlanar[e],c+=n.areaSphere[e],u+=n.areaWgs84[e];let t=n.areaPlanar[e]/Math.max(n.areaWgs84[e],1e-9);t>g&&(g=t,_=e),v+=n.sliver[e],y+=n.polsbyPopper[e]}t.setReadout(`areaPlanar`,`${z(s/1e6)} km² (${Ae[e.id].space===`mercator`?`Web Mercator`:`Mercator-local`})`),t.setReadout(`areaSphere`,`${z(c/1e6)} km²`),t.setReadout(`areaWgs84`,`${z(u/1e6)} km²`);let x=Array(X.breaks.length+1).fill(0);for(let e of n.areaWgs84){let t=0;for(;t<X.breaks.length&&e>=X.breaks[t];)t++;x[t]++}t.setReadout(`areaClasses`,x.map(l).join(` · `)),t.setReadout(`distortion`,`${g.toFixed(2)}× (${e.names[_]})`);let S=new Float32Array(i),C=new Float32Array(i);for(let e=0;e<i;e++)S[e]=n.centroid[e*2+1],C[e]=n.areaPlanar[e]/Math.max(n.areaWgs84[e],1e-9);t.setChart(`distortionScatter`,{kind:`scatter`,x:S,y:C,xLabel:`Latitude`,yLabel:`Planar / WGS84`,fit:{slope:0,intercept:1,label:`Mercator theory reference`},description:`Planar-to-WGS84 area ratio by GPU centroid latitude with a Mercator theory reference.`}),t.setReadout(`meanCompactness`,(y/i).toFixed(3)),t.setReadout(`slivers`,`${l(v)} (${(100*v/i).toFixed(1)}%)`);let w=0;for(let t=0;t<i;t++){let r=n.centroid[t*2],i=n.centroid[t*2+1];Number.isFinite(r)&&!se(e.layout,t,r,i)&&w++}if(t.setReadout(`centroidOutside`,`${l(w)} of ${l(i)}`),e.labelFeature>=0){let i=e.labelFeature;t.setReadout(`labelClearance`,`${z(n.radius[i],1)} m (${Ae[e.id].space===`mercator`?`Web Mercator`:`Mercator-local`})`),t.setReadout(`labelSearch`,`${r.initialGridSize} × ${r.initialGridSize}; ${r.refinementCandidates} candidates`)}let T={};for(let[e,t]of Object.entries(B)){let r=0;for(let e=0;e<i;e++)n.mask[e]&t&&r++;r>0&&(T[e]=r)}let E=Object.entries(T);t.setReadout(`validity`,E.length?E.map(([e,t])=>`${e} ${l(t)}`).join(`, `):`all valid`);let D=0,O=0;for(let e of n.mask){let t=e&~B.repeatedVertex;(t&ue)===0?t!==0&&D++:O++}t.setReadout(`validityConvention`,l(D)),t.setReadout(`validityStructural`,l(O)),t.setReadout(`validityOverflow`,e.validityOverflow?l(e.validityOverflow):`none`),t.setChart(`validityBits`,{kind:`bars`,labels:E.map(([e])=>e),values:E.map(([,e])=>e),xLabel:`Validity bit`,yLabel:`Polygons`,description:`Per-bit validity counts from the active GPU mask. repeatedVertex is informational: GEOS treats it as valid.`}),e.selected>=0&&b(e)}function y(e){let n=e.coloringTable;n&&(t.setReadout(`coloring`,`${n.colorCount} colors, ${n.conflictCount} conflicts, ${n.roundCount} rounds${n.converged?``:` (not converged)`}`),t.setReadout(`adjacencies`,n.adjacencies),t.setChart(`coloringRounds`,{kind:`bars`,labels:[`Rounds`,`Conflicts`],values:[n.roundCount,n.conflictCount],yLabel:`Count`,description:`Parallel greedy colouring rounds and remaining conflicts.`}),n.overflow&&t.setStatus(`Contiguity capacity overflowed.`))}function b(e){let n=e.table,r=e.selected;if(!n||r<0){t.setReadout(`selected`,null);return}t.setReadout(`selected`,`${e.names[r]}: ${z(n.areaWgs84[r]/1e6,1)} km² (planar ${z(n.areaPlanar[r]/1e6,1)}), compactness ${n.polsbyPopper[r].toFixed(2)}`)}function x(e,t){let n=e.layout,r=Float32Array.from(n.lngLat);if(t){let e=n.featureCount,t=new Set,i=r=>{let i=Math.floor((r+1)*e/7);for(let r=0;r<e;r++,i=(i+1)%e){let e=n.featureRingOffsets[i];if(!t.has(e)&&n.ringOffsets[e+1]-n.ringOffsets[e]>=12)return t.add(e),e}return n.featureRingOffsets[Math.floor((r+1)*e/7)]},a=e=>[n.ringOffsets[e],n.ringOffsets[e+1]],o=(e,t,n)=>{r[e*2]=t,r[e*2+1]=n},s=e=>[r[e*2],r[e*2+1]];{let[e,t]=a(i(0)),n=e+Math.floor((t-e)/2),r=s(e+1),c=s(n);o(e+1,c[0],c[1]),o(n,r[0],r[1])}{let[e]=a(i(1));o(e+1,NaN,NaN)}{let[e]=a(i(2)),t=s(e+1);o(e+2,t[0],t[1])}{let[,e]=a(i(3)),t=s(e-1);o(e-1,t[0]+.01,t[1]+.006)}{let[e,t]=a(i(4)),n=s(e+Math.floor((t-e)/2)+1);o(e+2,n[0],n[1])}{let[e,t]=a(i(5));for(let n=e,r=t-1;n<r;n++,r--){let e=s(n),t=s(r);o(n,t[0],t[1]),o(r,e[0],e[1])}}}e.validityPositions.write(r)}let S=e=>{i&&(i.tableReader?.stop(),i.coloringReader?.stop(),i.resources.destroy()),i=e,C.clear(),x(e,t.options.injectDefects),t.setStatus(``)},C=new Set,w=new te,T=()=>{w.request({build:async()=>{let e=i;if(t.options.dataset!==e?.id)return{kind:`dataset`,next:await c(t.options.dataset)};let n=new Map;for(let r of[...C])n.set(r,await e.builders[r](t.options));return{kind:`graphs`,dataset:e,graphs:n}},commit:e=>{if(e){if(e.kind===`dataset`)S(e.next);else{let n=e.dataset;for(let[r,i]of e.graphs){let e=n[r];n[r]=i,e&&n.resources.release(e),C.delete(r),r===`measures`&&(n.dirty.measures=n.dirty.table=!0),r===`shape`&&(n.dirty.shape=n.dirty.table=!0,h(n,t.options)),r===`validity`&&(n.dirty.validity=n.dirty.table=!0),r===`coloring`&&(n.dirty.coloring=!0)}D()}t.requestLayers()}},discard:e=>{if(e)if(e.kind===`dataset`)e.next.tableReader?.stop(),e.next.coloringReader?.stop(),e.next.resources.destroy();else for(let t of e.graphs.values())e.dataset.resources.release(t)}})};S(await c(t.options.dataset));let E={lastChange:performance.now(),pendingTable:!0,pendingColoring:!0},D=()=>{E.lastChange=performance.now()},O=e=>{let n=i;if(n){if(n.selected=e,e>=0){let t=n.layout,r=[],i=[];for(let n=t.featureRingOffsets[e];n<t.featureRingOffsets[e+1];n++){let e=t.ringOffsets[n],a=t.ringOffsets[n+1];for(let n=e;n<a;n++){let o=n+1<a?n+1:e;r.push(t.lngLat[n*2],t.lngLat[n*2+1]),i.push(t.lngLat[o*2],t.lngLat[o*2+1])}}n.selectedCount=r.length/2,n.overlays.selectedStarts.write(Float32Array.from(r)),n.overlays.selectedEnds.write(Float32Array.from(i))}else n.selectedCount=0;b(n),t.requestLayers()}};return{getCompiledGraphs(){return i?[i.measures,i.shape,i.validity,i.coloring].filter(Boolean):[]},setOption(e,n,r){let a=i;if(a){if(e===`dataset`){T();return}if(e===`ringGrouping`){C.add(`measures`).add(`shape`),T();return}else if(e===`sliverThreshold`)a.shapeParameters.write(k({sliverThreshold:r.sliverThreshold})),a.dirty.shape=a.dirty.table=!0;else if(e===`initialGridSize`||e===`refinementCandidates`){C.add(`shape`),T();return}else if(e===`orientation`||e===`ringClosure`){C.add(`validity`),T();return}else if(e===`injectDefects`)x(a,r.injectDefects),a.dirty.validity=a.dirty.table=!0;else if(e===`colorSeed`)a.coloringParameters.write(ge({seed:r.colorSeed})),a.dirty.coloring=!0;else if(e===`contiguity`||e===`maxRounds`){C.add(`coloring`),T();return}else (e===`areaSystem`||e===`metric`||e===`areaClassification`)&&(a.dirty.table=!0,t.setLegendExtent(`value`,p));D(),t.requestLayers()}},onThemeChange(){t.requestLayers()},getTooltip(e){let t=i;if(!t||!e.coordinate)return null;let n=I(t.layout,e.coordinate[0],e.coordinate[1]);if(n<0)return null;let r=t.table;return r?`${t.names[n]} · ${z(r.areaWgs84[n]/1e6,1)} km² · compactness ${r.polsbyPopper[n].toFixed(2)}`:t.names[n]},onClick(e){let t=i;if(!t||!e.coordinate)return!1;let n=I(t.layout,e.coordinate[0],e.coordinate[1]);return O(n===t.selected?-1:n),!0},encode(e){let t=i;t&&(t.dirty.measures&&(t.measures.encode(e,{parameters:void 0}),t.dirty.measures=!1),t.dirty.shape&&t.shape&&(t.shape.encode(e,{parameters:void 0}),t.dirty.shape=!1),t.dirty.validity&&t.validity&&(t.validity.encode(e,{parameters:void 0}),t.dirty.validity=!1),t.dirty.coloring&&t.coloring&&(t.coloring.encode(e,{parameters:void 0}),t.dirty.coloring=!1,t.coloringReader?.request(e)),t.dirty.table&&performance.now()-E.lastChange>120?(t.tableReader?.request(e),t.dirty.table=!1):(t.tableReader?.flush(e),t.coloringReader?.flush(e)))},getLayers(){let n=i;if(!n)return[];let r=t.options,a=t.theme()===`dark`,o=[],s=r.metric,c=n.columns,l=null,u=`ramp`,d=n.fill.featureRows,f=p,m=!1,h,g=r.areaSystem;switch(s){case`area`:l=g===`planar`?c.areaPlanar:g===`spherical`?c.areaSphere:c.areaWgs84,m=!0;break;case`perimeter`:l=g===`planar`?c.lengthPlanar:g===`spherical`?c.lengthSphere:c.lengthWgs84;break;case`areaDistortion`:l=c.ratio;break;case`vertices`:l=c.vertexF,m=!0;break;case`groupArea`:l=c.groupAreaF;break;case`compactness`:l=c.polsbyPopper;break;case`schwartzberg`:l=c.schwartzberg;break;case`elongation`:l=c.elongation;break;case`convexity`:l=c.convexity;break;case`sliver`:l=c.sliverF,u=`flag`,h=a?[70,80,96,150]:[196,204,214,150];break;case`validity`:l=c.validityF,u=`class`;break;case`mapColor`:l=r.coloringMethod===`algorithm`?c.colorF:c.naiveColorF,u=`category`;break}let _=ke(s);_&&(u=`class`),o.push(new ie({id:`polygon-fill`,coordinateSystem:e.LNGLAT,corners:n.fill.corners,featureRows:d,instanceCount:n.fill.triangleCount,values:l,valueMapping:u,classTable:_??void 0,valueRange:f,sqrtScale:m,color:s===`sliver`?[226,96,80,Math.round(r.opacity*255)]:[255,255,255,Math.round(r.opacity*255)],secondaryColor:h,updateTriggers:{values:[s,g]}})),r.showDeflate&&s===`areaDistortion`&&o.push(new ie({id:`deflated-equal-true-size-reference`,coordinateSystem:e.LNGLAT,corners:n.fill.deflatedCorners,featureRows:n.fill.featureRows,instanceCount:n.fill.triangleCount,color:a?[77,182,172,145]:[0,137,123,145]}));let v=s===`mapColor`?[255,255,255,210]:a?[235,240,248,150]:[40,52,70,150];return r.showOutlines&&o.push(new P({id:`polygon-outline`,coordinateSystem:e.LNGLAT,starts:n.edges.starts,ends:n.edges.ends,instanceCount:n.edges.edgeCount,color:v,widthPixels:n.id===`us-counties`?.8:1.1})),r.showBounds&&o.push(new P({id:`polygon-bounds`,coordinateSystem:e.LNGLAT,starts:n.overlays.boundsStarts,ends:n.overlays.boundsEnds,instanceCount:n.layout.featureCount*4,color:[78,168,222,180],widthPixels:1})),r.showInscribed&&o.push(new P({id:`inscribed-circles`,coordinateSystem:e.LNGLAT,starts:n.overlays.circleStarts,ends:n.overlays.circleEnds,instanceCount:n.layout.featureCount*Y,color:a?[255,214,120,230]:[180,90,10,230],widthPixels:1.4})),r.showAxes&&o.push(new P({id:`axis-glyphs`,coordinateSystem:e.LNGLAT,starts:n.overlays.axisStarts,ends:n.overlays.axisEnds,instanceCount:n.layout.featureCount,color:a?[150,230,255,235]:[0,90,140,235],widthPixels:2})),r.showLabelSearch&&n.labelGridCount&&o.push(new P({id:`label-search-grid-and-refinement`,coordinateSystem:e.LNGLAT,starts:n.overlays.labelGridStarts,ends:n.overlays.labelGridEnds,instanceCount:n.labelGridCount,color:a?[124,200,242,210]:[0,118,173,210],widthPixels:1})),n.selectedCount>0&&o.push(new P({id:`selected-outline`,coordinateSystem:e.LNGLAT,starts:n.overlays.selectedStarts,ends:n.overlays.selectedEnds,instanceCount:n.selectedCount,color:a?[255,255,255,255]:[0,0,0,255],widthPixels:3})),r.showCentroids&&o.push(new j({id:`centroids`,coordinateSystem:e.LNGLAT,positions:n.overlays.centroids,instanceCount:n.layout.featureCount,radiusPixels:n.id===`us-counties`?1.6:3,color:[232,90,140,255]})),r.showLabels&&o.push(new j({id:`label-points`,coordinateSystem:e.LNGLAT,positions:n.overlays.labelPoints,instanceCount:n.layout.featureCount,radiusPixels:n.id===`us-counties`?1.6:3,color:a?[255,214,120,255]:[180,90,10,255]})),o},destroy(){r=!0,w.destroy(),i&&=(i.tableReader?.stop(),i.coloringReader?.stop(),i.resources.destroy(),null)}}}async function Me(e){let t=new te,n=new Set,r=new Set,i=null;async function a(){let t={...e.options},a={...e,get options(){return t},setReadout(t,r){i?.context===a&&(n.add(t),e.setReadout(t,r))},setChart(t,n){i?.context===a&&(r.add(t),e.setChart(t,n))},setStatus(t){i?.context===a&&e.setStatus(t)}},o=t.dataset;return{context:a,instance:o===`antimeridian-and-poles`?await Ce(a):await je({...a,get options(){return{...t,dataset:o}}}),dataset:o,updateOptions(e){t={...e}}}}function o(t){let a=i;i=t,a?.instance.destroy();for(let t of n)e.setReadout(t,null);for(let t of r)e.setChart(t,null);n.clear(),r.clear(),e.setStatus(``),t.dataset===`antimeridian-and-poles`&&(e.flyTo({longitude:0,latitude:0,pitch:0,bearing:0}),e.fitBounds([-180,-85.0511,180,85.0511],{padding:20})),e.requestLayers()}return o(await a()),{getCompiledGraphs:()=>i?.instance.getCompiledGraphs()??[],encode:(e,t)=>i?.instance.encode(e,t),getLayers:()=>i?.instance.getLayers()??[],getTooltip:e=>i?.instance.getTooltip?.(e)??null,onClick:e=>i?.instance.onClick?.(e)??!1,onThemeChange:e=>i?.instance.onThemeChange?.(e),setOption(n,r,s){if(n===`dataset`||t.pending){t.request({build:a,commit:o,discard:e=>e.instance.destroy()}).catch(t=>e.setStatus(String(t)));return}i?.updateOptions(s),i?.instance.setOption?.(n,r,s)},destroy(){t.destroy(),i?.instance.destroy(),i=null}}}export{Me as createPolygonMeasures};