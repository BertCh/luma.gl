import{E as e}from"./analysis-frame-Blu0tYdw.js";import{_ as t,b as n,c as r,j as i,l as a,v as o,w as s,y as c}from"./resources-CwvhEOBe.js";import{t as l}from"./gpu-scan-BuGhTLDQ.js";import{t as u}from"./gpu-sort-Bl3Dztdb.js";import{t as d}from"./gpu-bvh-C68VuZY1.js";import{s as f}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{d as p}from"./wgsl-common-BNRVzgd8.js";import{t as m}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{n as h}from"./scan-total-BdolCtK7.js";import{t as g}from"./bounds-reduction-cUiPIxi7.js";var _={proper:1,touch:2,collinearTouch:3,overlap:4,uncertain:5},v=4294967295;function y(e){return a(e,32)}var b=`
const SEGMENT_NONE: u32 = ${v}u;
const KIND_NONE: u32 = 0u;
const KIND_PROPER: u32 = ${_.proper}u;
const KIND_TOUCH: u32 = ${_.touch}u;
const KIND_COLLINEAR_TOUCH: u32 = ${_.collinearTouch}u;
const KIND_OVERLAP: u32 = ${_.overlap}u;
const KIND_UNCERTAIN: u32 = ${_.uncertain}u;
${m}

struct SegmentHit {
  kind: u32,
  point: vec2f,
  endPoint: vec2f,
}

fn axisValue(point: vec2f, useX: bool) -> f32 { return select(point.y, point.x, useX); }

// Picks the point among the four endpoints whose axis coordinate equals value.
fn pointAtAxis(a: vec2f, b: vec2f, c: vec2f, d: vec2f, useX: bool, value: f32) -> vec2f {
  if (axisValue(a, useX) == value) { return a; }
  if (axisValue(b, useX) == value) { return b; }
  if (axisValue(c, useX) == value) { return c; }
  return d;
}

// Classifies segments a-b and c-d, both non-degenerate. withGeometry also fills the points.
fn classifySegments(a: vec2f, b: vec2f, c: vec2f, d: vec2f, withGeometry: bool) -> SegmentHit {
  var hit = SegmentHit(KIND_NONE, a, a);
  if (max(a.x, b.x) < min(c.x, d.x) || max(c.x, d.x) < min(a.x, b.x) ||
      max(a.y, b.y) < min(c.y, d.y) || max(c.y, d.y) < min(a.y, b.y)) {
    return hit;
  }
  let o1 = orientSign(a, b, c);
  let o2 = orientSign(a, b, d);
  if (o1 == 2 || o2 == 2) { hit.kind = KIND_UNCERTAIN; return hit; }
  if (o1 != 0 && o1 == o2) { return hit; }
  let o3 = orientSign(c, d, a);
  let o4 = orientSign(c, d, b);
  if (o3 == 2 || o4 == 2) { hit.kind = KIND_UNCERTAIN; return hit; }
  if (o3 != 0 && o3 == o4) { return hit; }
  if (o1 == 0 && o2 == 0) {
    // All four orientations vanish: collinear. Compare along the axis the line is not parallel to.
    let useX = a.x != b.x;
    let low = max(min(axisValue(a, useX), axisValue(b, useX)), min(axisValue(c, useX), axisValue(d, useX)));
    let high = min(max(axisValue(a, useX), axisValue(b, useX)), max(axisValue(c, useX), axisValue(d, useX)));
    if (low > high) { return hit; }
    hit.kind = select(KIND_OVERLAP, KIND_COLLINEAR_TOUCH, low == high);
    hit.point = pointAtAxis(a, b, c, d, useX, low);
    hit.endPoint = pointAtAxis(a, b, c, d, useX, high);
    return hit;
  }
  if (o1 != 0 && o2 != 0 && o3 != 0 && o4 != 0) {
    hit.kind = KIND_PROPER;
    if (withGeometry) {
      let r = b - a;
      let s = d - c;
      let denominator = r.x * s.y - r.y * s.x;
      let t = clamp(((c.x - a.x) * s.y - (c.y - a.y) * s.x) / denominator, 0.0, 1.0);
      let crossing = a + r * t;
      hit.point = clamp(crossing, max(min(a, b), min(c, d)), min(max(a, b), max(c, d)));
      hit.endPoint = hit.point;
    }
    return hit;
  }
  hit.kind = KIND_TOUCH;
  if (o1 == 0) { hit.point = c; } else if (o2 == 0) { hit.point = d; } else if (o3 == 0) { hit.point = a; } else { hit.point = b; }
  hit.endPoint = hit.point;
  return hit;
}

`;function x(e){let t=`${e}Table`,n=e=>`${t}Offset + ${e} * 8u`;return`
fn ${e}Ring(row: u32) -> u32 { return ${t}[${n(`row`)} + 4u]; }
fn ${e}Valid(row: u32) -> bool { return ${e}Ring(row) != SEGMENT_NONE; }
fn ${e}Successor(row: u32) -> u32 { return ${t}[${n(`row`)} + 5u]; }
fn ${e}Feature(row: u32) -> u32 { return ${t}[${n(`row`)} + 6u]; }
fn ${e}Start(row: u32) -> vec2f {
  return vec2f(bitcast<f32>(${t}[${n(`row`)}]), bitcast<f32>(${t}[${n(`row`)} + 1u]));
}
fn ${e}End(row: u32) -> vec2f {
  return vec2f(bitcast<f32>(${t}[${n(`row`)} + 2u]), bitcast<f32>(${t}[${n(`row`)} + 3u]));
}`}function S(e,t){let{id:n,operation:r,minima:i,maxima:a,leafCapacity:o,spatialSort:c}=t,l=i.length,p=[],m=2*o-1,h=i,_=a,v;if(c&&l>=2){let t=s(e,`${n}-sort-scene-bounds`,`float32`,4),o=s(e,`${n}-sort-keys`,`uint32`,l),c=s(e,`${n}-sort-rows`,`uint32`,l),d=s(e,`${n}-sort-sorted-keys`,`uint32`,l),m=s(e,`${n}-sort-sorted-rows`,`uint32`,l);h=s(e,`${n}-sort-minima`,`float32x2`,l),_=s(e,`${n}-sort-maxima`,`float32x2`,l),v=m;let y=[{name:`segmentMinima`,view:i,type:`f32`,access:`read`},{name:`segmentMaxima`,view:a,type:`f32`,access:`read`}],b=`const SEGMENT_COUNT: u32 = ${l}u;
fn readCenter(row: u32) -> vec3f {
  // (center.x, center.y, valid). Skipped segments have inverted bounds.
  let minimum = vec2f(segmentMinima[segmentMinimaOffset + row * 2u], segmentMinima[segmentMinimaOffset + row * 2u + 1u]);
  let maximum = vec2f(segmentMaxima[segmentMaximaOffset + row * 2u], segmentMaxima[segmentMaximaOffset + row * 2u + 1u]);
  let valid = minimum.x <= maximum.x && minimum.y <= maximum.y;
  let center = minimum * 0.5 + maximum * 0.5;
  return vec3f(center, select(0.0, 1.0, valid));
}`;p.push(...g(e,{id:`${n}-sort-scene`,operation:r,variant:`sort-scene-bounds`,bindings:y,declarations:`${b}
fn readItem(row: u32) -> vec4f {
  let center = readCenter(row);
  return vec4f(center.xy, center.z, 0.0);
}`,itemCount:l,output:t})),p.push(f(e,{id:`${n}-sort-keys`,operation:r,variant:`sort-keys`,bindings:[...y,{name:`sceneBounds`,view:t,type:`f32`,access:`read`},{name:`sortKeys`,view:o,type:`u32`,access:`read_write`},{name:`sortRows`,view:c,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`${b}
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
fn spreadBits(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let extent = maximum - minimum;
  if (!(extent > 0.0) || extent > FLOAT32_MAXIMUM) { return 0u; }
  let normalized = clamp((value - minimum) / extent, 0.0, 1.0);
  return min(u32(normalized * 65535.0 + 0.5), 65535u);
}`,body:`sortRows[sortRowsOffset + index] = index;
  let center = readCenter(index);
  var key = 0xffffffffu;
  if (center.z > 0.5) {
    let quantizedX = quantizeAxis(center.x, sceneBounds[sceneBoundsOffset], sceneBounds[sceneBoundsOffset + 2u]);
    let quantizedY = quantizeAxis(center.y, sceneBounds[sceneBoundsOffset + 1u], sceneBounds[sceneBoundsOffset + 3u]);
    key = spreadBits(quantizedX) | (spreadBits(quantizedY) << 1u);
  }
  sortKeys[sortKeysOffset + index] = key;`})),p.push(...new u({id:`${n}-sort-order`,keys:o,values:c,outputKeys:d,outputValues:m,algorithm:`radix`}).getCommandNodes(e)),p.push(f(e,{id:`${n}-sort-gather`,operation:r,variant:`sort-gather`,bindings:[...y,{name:`sortedRows`,view:m,type:`u32`,access:`read`},{name:`sortedMinima`,view:h,type:`f32`,access:`read_write`},{name:`sortedMaxima`,view:_,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`const SEGMENT_COUNT: u32 = ${l}u;`,body:`let row = min(sortedRows[sortedRowsOffset + index], SEGMENT_COUNT - 1u);
  sortedMinima[sortedMinimaOffset + index * 2u] = segmentMinima[segmentMinimaOffset + row * 2u];
  sortedMinima[sortedMinimaOffset + index * 2u + 1u] = segmentMinima[segmentMinimaOffset + row * 2u + 1u];
  sortedMaxima[sortedMaximaOffset + index * 2u] = segmentMaxima[segmentMaximaOffset + row * 2u];
  sortedMaxima[sortedMaximaOffset + index * 2u + 1u] = segmentMaxima[segmentMaximaOffset + row * 2u + 1u];`}))}let y=new d({id:`${n}-bvh`,minima:h,maxima:_,sourceIds:v,leafCapacity:o,nodeMinima:s(e,`${n}-bvh-node-minima`,`float32x2`,m),nodeMaxima:s(e,`${n}-bvh-node-maxima`,`float32x2`,m),nodeChildren:s(e,`${n}-bvh-node-children`,`uint32x2`,m),leafIds:s(e,`${n}-bvh-leaf-ids`,`uint32`,o),count:s(e,`${n}-bvh-count`,`uint32`,1),overflow:s(e,`${n}-bvh-overflow`,`uint32`,1)});return p.push(...y.getCommandNodes(e)),{bvh:y,nodes:p}}var C=1024;function w(e){return e.positions.length}function T(e,t){let{geometry:n}=t,r=n.kind===`polygons`,i=[{name:`positions`,view:n.positions,type:`f32`,access:`read`}],a;r?(i.push({name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:n.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:n.featureOffsets,type:`u32`,access:`read`}),a=`
const RING_COUNT: u32 = ${n.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${n.polygonOffsets.length-1}u;
const FEATURE_COUNT: u32 = ${n.featureOffsets.length-1}u;
const MINIMUM_RING_VERTICES: u32 = 3u;
fn lastAtMost(offsetsKind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  if (readOffset(offsetsKind, 0u) > value) { return SEGMENT_NONE; }
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(offsetsKind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn readOffset(offsetsKind: u32, row: u32) -> u32 {
  if (offsetsKind == 0u) { return ringOffsets[ringOffsetsOffset + row]; }
  if (offsetsKind == 1u) { return polygonOffsets[polygonOffsetsOffset + row]; }
  return featureOffsets[featureOffsetsOffset + row];
}
fn ringStartOf(ring: u32) -> u32 { return ringOffsets[ringOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT); }
fn featureOfRing(ring: u32) -> u32 {
  if (ring >= polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) { return SEGMENT_NONE; }
  let polygon = lastAtMost(1u, POLYGON_COUNT, ring);
  if (polygon == SEGMENT_NONE) { return SEGMENT_NONE; }
  if (polygon >= featureOffsets[featureOffsetsOffset + FEATURE_COUNT]) { return SEGMENT_NONE; }
  return lastAtMost(2u, FEATURE_COUNT, polygon);
}
fn ringOfVertex(vertex: u32) -> u32 {
  let ring = lastAtMost(0u, RING_COUNT, vertex);
  if (ring == SEGMENT_NONE || vertex >= ringEndOf(ring)) { return SEGMENT_NONE; }
  return ring;
}`):(i.push({name:`lineOffsets`,view:n.lineOffsets,type:`u32`,access:`read`}),a=`
const RING_COUNT: u32 = ${n.lineOffsets.length-1}u;
const MINIMUM_RING_VERTICES: u32 = 2u;
fn readOffset(offsetsKind: u32, row: u32) -> u32 { return lineOffsets[lineOffsetsOffset + row]; }
fn lastAtMost(offsetsKind: u32, count: u32, value: u32) -> u32 {
  if (count == 0u) { return SEGMENT_NONE; }
  var low = 0u;
  var high = count;
  if (readOffset(offsetsKind, 0u) > value) { return SEGMENT_NONE; }
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (readOffset(offsetsKind, middle) <= value) { low = middle; } else { high = middle; }
  }
  return low;
}
fn ringStartOf(ring: u32) -> u32 { return lineOffsets[lineOffsetsOffset + ring]; }
fn ringEndOf(ring: u32) -> u32 { return min(lineOffsets[lineOffsetsOffset + ring + 1u], VERTEX_COUNT); }
fn featureOfRing(ring: u32) -> u32 { return ring; }
fn ringOfVertex(vertex: u32) -> u32 {
  let ring = lastAtMost(0u, RING_COUNT, vertex);
  if (ring == SEGMENT_NONE || vertex >= ringEndOf(ring)) { return SEGMENT_NONE; }
  return ring;
}`),i.push({name:`table`,view:t.table,type:`u32`,access:`read_write`}),t.minima&&t.maxima&&i.push({name:`segmentMinima`,view:t.minima,type:`f32`,access:`read_write`},{name:`segmentMaxima`,view:t.maxima,type:`f32`,access:`read_write`});let o=!!(t.minima&&t.maxima);return f(e,{id:t.id,operation:t.operation,variant:`segment-table-${n.kind}`,bindings:i,invocationCount:w(n),declarations:`${b}
const VERTEX_COUNT: u32 = ${n.positions.length}u;
const CYCLIC: bool = ${r};
const MAXIMUM_SUCCESSOR_SCAN: u32 = ${C}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
${a}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn finiteValue(value: f32) -> bool { return isFiniteBits(value); }
fn finitePoint(point: vec2f) -> bool { return finiteValue(point.x) && finiteValue(point.y); }
// The vertex that ends the segment starting at vertex, or SEGMENT_NONE when there is none.
fn nextVertex(vertex: u32, ringStart: u32, ringEnd: u32, closedLine: bool) -> u32 {
  if (vertex + 1u < ringEnd) { return vertex + 1u; }
  if (CYCLIC || closedLine) { return ringStart; }
  return SEGMENT_NONE;
}
fn segmentIsUsable(vertex: u32, ringStart: u32, ringEnd: u32, closedLine: bool) -> bool {
  let next = nextVertex(vertex, ringStart, ringEnd, closedLine);
  if (next == SEGMENT_NONE) { return false; }
  // A closed linestring's last vertex repeats its first and owns no segment.
  if (!CYCLIC && vertex + 1u == ringEnd) { return false; }
  let start = vertexAt(vertex);
  let end = vertexAt(next);
  return finitePoint(start) && finitePoint(end) && (start.x != end.x || start.y != end.y);
}`,body:`var row0 = 0u; var row1 = 0u; var row2 = 0u; var row3 = 0u;
  var rowRing = SEGMENT_NONE;
  var rowSuccessor = SEGMENT_NONE;
  var rowFeature = SEGMENT_NONE;
  var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let ring = ringOfVertex(index);
  if (ring != SEGMENT_NONE) {
    let ringStart = ringStartOf(ring);
    let ringEnd = ringEndOf(ring);
    let feature = featureOfRing(ring);
    var closedLine = false;
    if (!CYCLIC && ringEnd - ringStart >= 3u) {
      let first = vertexAt(ringStart);
      let last = vertexAt(ringEnd - 1u);
      closedLine = first.x == last.x && first.y == last.y;
    }
    if (ringEnd - ringStart >= MINIMUM_RING_VERTICES && feature != SEGMENT_NONE &&
        segmentIsUsable(index, ringStart, ringEnd, closedLine)) {
      let start = vertexAt(index);
      let end = vertexAt(nextVertex(index, ringStart, ringEnd, closedLine));
      row0 = bitcast<u32>(start.x); row1 = bitcast<u32>(start.y);
      row2 = bitcast<u32>(end.x); row3 = bitcast<u32>(end.y);
      rowRing = ring;
      rowFeature = feature;
      minimum = min(start, end);
      maximum = max(start, end);
      var candidate = index;
      for (var step = 0u; step < min(ringEnd - ringStart, MAXIMUM_SUCCESSOR_SCAN); step++) {
        let following = nextVertex(candidate, ringStart, ringEnd, closedLine);
        if (following == SEGMENT_NONE) { break; }
        candidate = following;
        if (segmentIsUsable(candidate, ringStart, ringEnd, closedLine)) {
          rowSuccessor = candidate;
          break;
        }
      }
    }
  }
  let base = tableOffset + index * 8u;
  table[base] = row0; table[base + 1u] = row1; table[base + 2u] = row2; table[base + 3u] = row3;
  table[base + 4u] = rowRing; table[base + 5u] = rowSuccessor; table[base + 6u] = rowFeature;
  table[base + 7u] = 0u;
  ${o?`segmentMinima[segmentMinimaOffset + index * 2u] = minimum.x;
  segmentMinima[segmentMinimaOffset + index * 2u + 1u] = minimum.y;
  segmentMaxima[segmentMaximaOffset + index * 2u] = maximum.x;
  segmentMaxima[segmentMaximaOffset + index * 2u + 1u] = maximum.y;`:``}`})}var E=`GPUSegmentIntersection`,D=1024,O=class{id;props;spatialContext;isSelfMode;leftSlotCount;rightSlotCount;pairCapacity;leafCapacity;spatialSort;constructor(t){this.id=t.id??`segment-intersection`,this.spatialContext=e(this.id,t.spatialContext),this.props=t;let{id:a}=this;this.isSelfMode=!t.right;for(let[e,n]of[[`left`,t.left],[`right`,t.right]]){if(!n)continue;c(n.positions,[`float32x2`],`${a} ${e}.positions`);let t=n.kind===`lines`?[[`lineOffsets`,n.lineOffsets]]:[[`featureOffsets`,n.featureOffsets],[`polygonOffsets`,n.polygonOffsets],[`ringOffsets`,n.ringOffsets]];for(let[n,r]of t)if(i(r,`${a} ${e}.${n}`),r.length<1)throw Error(`${a} ${e}.${n} requires a terminal entry`);if(n.positions.length<1)throw Error(`${a} ${e} must contain at least one vertex`)}if(t.sameFeatureOnly&&!this.isSelfMode)throw Error(`${a} sameFeatureOnly requires self mode (omit right)`);this.leftSlotCount=w(t.left),this.rightSlotCount=w(t.right??t.left),this.spatialSort=t.spatialSort??this.rightSlotCount>=D,this.leafCapacity=t.leafCapacity??r(this.rightSlotCount),n(a,`leafCapacity`,this.leafCapacity);let{pairs:o}=t.output;if(i(o.leftIds,`${a} pairs.leftIds`),i(o.rightIds,`${a} pairs.rightIds`),this.pairCapacity=o.leftIds.length,this.pairCapacity<1||o.rightIds.length!==this.pairCapacity)throw Error(`${a} pairs.leftIds and pairs.rightIds must have equal nonzero length`);for(let[e,n]of[[`count`,o.status.count],[`overflow`,o.status.overflow],[`requiredCount`,o.status.requiredCount],[`status.uncertainCount`,t.output.status?.uncertainCount]])if(n&&(i(n,`${a} ${e}`),n.length<1))throw Error(`${a} ${e} must contain one uint32 row`);for(let[e,n]of[[`kinds`,t.output.kinds],[`leftFeatures`,t.output.leftFeatures],[`rightFeatures`,t.output.rightFeatures],[`leftRings`,t.output.leftRings],[`rightRings`,t.output.rightRings]])if(n&&(i(n,`${a} ${e}`),n.length!==this.pairCapacity))throw Error(`${a} ${e} length must equal the pair capacity`);for(let[e,n]of[[`points`,t.output.points],[`endPoints`,t.output.endPoints]])if(n&&(c(n,[`float32x2`],`${a} ${e}`),n.length!==this.pairCapacity))throw Error(`${a} ${e} length must equal the pair capacity`)}getInputViews(){let e=[];for(let t of[this.props.left,this.props.right])t&&(e.push(t.positions),t.kind===`lines`?e.push(t.lineOffsets):e.push(t.featureOffsets,t.polygonOffsets,t.ringOffsets));return e}getOutputViews(){let{pairs:e,status:t,...n}=this.props.output,r=t?.uncertainCount;return[e.leftIds,e.rightIds,e.status.count,e.status.overflow,e.status.requiredCount,r,n.kinds,n.points,n.endPoints,n.leftFeatures,n.rightFeatures,n.leftRings,n.rightRings]}getCommandNodes(e){let{id:n,props:r,isSelfMode:i,leftSlotCount:a,rightSlotCount:c,pairCapacity:u}=this,{pairs:d}=r.output;t(n,e,[...this.getInputViews(),...this.getOutputViews()]);for(let[t,i]of[[`left`,r.left],[`right`,r.right]])i&&o(n,`${t}.positions rows`,i.positions.length,y(e.device));let m=[],g=r.right??r.left,_=s(e,`${n}-right-table`,`uint32`,c*8),v=s(e,`${n}-bounds-minima`,`float32x2`,c),C=s(e,`${n}-bounds-maxima`,`float32x2`,c);m.push(T(e,{id:`${n}-right-table`,operation:E,geometry:g,table:_,minima:v,maxima:C}));let w=_;i||(w=s(e,`${n}-left-table`,`uint32`,a*8),m.push(T(e,{id:`${n}-left-table`,operation:E,geometry:r.left,table:w})));let{bvh:D,nodes:O}=S(e,{id:n,operation:E,minima:v,maxima:C,leafCapacity:this.leafCapacity,spatialSort:this.spatialSort});m.push(...O);let k=s(e,`${n}-state`,`uint32`,4),A=s(e,`${n}-counts`,`uint32`,a),j=s(e,`${n}-offsets`,`uint32`,a);m.push(f(e,{id:`${n}-clear`,operation:E,variant:`clear`,bindings:[{name:`state`,view:k,type:`u32`,access:`read_write`}],invocationCount:4,body:`state[stateOffset + index] = 0u;`}));let M=`${b}
${x(`left`)}
${x(`right`)}
const SELF: bool = ${i};
const SAME_FEATURE: bool = ${!!r.sameFeatureOnly};
const RIGHT_COUNT: u32 = ${c}u;
const INTERNAL_NODE_COUNT: u32 = ${D.internalNodeCount}u;
const PAIR_CAPACITY: u32 = ${u}u;
fn pairHit(left: u32, right: u32, withGeometry: bool) -> SegmentHit {
  var none = SegmentHit(KIND_NONE, vec2f(0.0), vec2f(0.0));
  // The ID test needs no table read, so it goes first: half of all self-mode candidates fail it.
  if (SELF && right <= left) { return none; }
  if (!leftValid(left) || !rightValid(right)) { return none; }
  if (SAME_FEATURE && leftFeature(left) != rightFeature(right)) { return none; }
  var hit = classifySegments(leftStart(left), leftEnd(left), rightStart(right), rightEnd(right), withGeometry);
  if (SELF && (hit.kind == KIND_TOUCH || hit.kind == KIND_COLLINEAR_TOUCH) &&
      (leftSuccessor(left) == right || rightSuccessor(right) == left)) {
    hit.kind = KIND_NONE;
  }
  return hit;
}
${p}`,N=[{name:`leftTable`,view:w,type:`u32`,access:`read`},{name:`rightTable`,view:_,type:`u32`,access:`read`},{name:`nodeMinima`,view:D.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:D.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:D.leafIds,type:`u32`,access:`read`}],P=i&&this.spatialSort&&this.leafCapacity>=a,F=(e,t,n)=>`${P?`let leftRow = leafIds[leafIdsOffset + index];
  if (leftRow >= RIGHT_COUNT) {
    return;
  }`:`let leftRow = index;`}
  ${t}
  if (leftValid(leftRow)) {
    let start = leftStart(leftRow);
    let end = leftEnd(leftRow);
    let queryMinimum = min(start, end);
    let queryMaximum = max(start, end);
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT && pairHit(leftRow, rightRow, false).kind != KIND_NONE) {
          ${e}
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${n}`;m.push(f(e,{id:`${n}-count`,operation:E,variant:`count`,bindings:[...N,{name:`counts`,view:A,type:`u32`,access:`read_write`}],invocationCount:a,declarations:M,body:F(`found = found + 1u;`,`var found = 0u;`,`counts[countsOffset + leftRow] = found;`)})),m.push(...new l({id:`${n}-scan`,input:A,output:j,mode:`exclusive`}).getCommandNodes(e)),m.push(f(e,{id:`${n}-total`,operation:E,variant:`total`,bindings:[{name:`counts`,view:A,type:`u32`,access:`read`},{name:`offsets`,view:j,type:`u32`,access:`read`},{name:`state`,view:k,type:`u32`,access:`read_write`}],invocationCount:1,body:`state[stateOffset] = offsets[offsetsOffset + ${a-1}u] + counts[countsOffset + ${a-1}u];`})),m.push(h(e,{id:n,operation:E,counts:A,offsets:j,rowCount:a,total:k})),m.push(f(e,{id:`${n}-write`,operation:E,variant:`write`,bindings:[...N,{name:`offsets`,view:j,type:`u32`,access:`read`},{name:`pairLeft`,view:d.leftIds,type:`u32`,access:`read_write`},{name:`pairRight`,view:d.rightIds,type:`u32`,access:`read_write`}],invocationCount:a,declarations:M,body:F(`let slot = offsets[offsetsOffset + leftRow] + found;
          if (slot < PAIR_CAPACITY) {
            pairLeft[pairLeftOffset + slot] = leftRow;
            pairRight[pairRightOffset + slot] = rightRow;
          }
          found = found + 1u;`,`var found = 0u;`,``)})),this.spatialSort&&m.push(f(e,{id:`${n}-sort-hits`,operation:E,variant:`sort-hits`,bindings:[{name:`counts`,view:A,type:`u32`,access:`read`},{name:`offsets`,view:j,type:`u32`,access:`read`},{name:`pairRight`,view:d.rightIds,type:`u32`,access:`read_write`}],invocationCount:a,declarations:`const PAIR_CAPACITY: u32 = ${u}u;
const GAPS = array<u32, 8>(701u, 301u, 132u, 57u, 23u, 10u, 4u, 1u);`,body:`let first = offsets[offsetsOffset + index];
  let last = min(first + counts[countsOffset + index], PAIR_CAPACITY);
  if (last <= first + 1u) { return; }
  // Shell sort with Ciura gaps; ranges are short unless one segment crosses very many others.
  for (var gapIndex = 0u; gapIndex < 8u; gapIndex++) {
    let gap = GAPS[gapIndex];
    for (var slot = first + gap; slot < last; slot++) {
      let value = pairRight[pairRightOffset + slot];
      var hole = slot;
      loop {
        if (hole < first + gap) { break; }
        let earlier = pairRight[pairRightOffset + hole - gap];
        if (earlier <= value) { break; }
        pairRight[pairRightOffset + hole] = earlier;
        hole = hole - gap;
      }
      pairRight[pairRightOffset + hole] = value;
    }
  }`}));let{kinds:I,points:L,endPoints:R}=r.output,z=r.output.status?.uncertainCount;if(I||L||R||z){let t=[{name:`pairLeft`,view:d.leftIds,type:`u32`,access:`read`},{name:`pairRight`,view:d.rightIds,type:`u32`,access:`read`},{name:`leftTable`,view:w,type:`u32`,access:`read`},{name:`rightTable`,view:_,type:`u32`,access:`read`},{name:`state`,view:k,type:`atomic<u32>`,access:`read_write`}];I&&t.push({name:`kinds`,view:I,type:`u32`,access:`read_write`}),L&&t.push({name:`points`,view:L,type:`f32`,access:`read_write`}),R&&t.push({name:`endPoints`,view:R,type:`f32`,access:`read_write`}),m.push(f(e,{id:`${n}-classify`,operation:E,variant:`classify`,bindings:t,invocationCount:u,declarations:M.replace(/fn nodeOverlaps[\s\S]*$/,``),body:`if (index >= min(atomicLoad(&state[stateOffset]), PAIR_CAPACITY)) { return; }
  let hit = pairHit(pairLeft[pairLeftOffset + index], pairRight[pairRightOffset + index], true);
  if (hit.kind == KIND_UNCERTAIN) { atomicAdd(&state[stateOffset + 1u], 1u); }
  ${I?`kinds[kindsOffset + index] = hit.kind;`:``}
  ${L?`points[pointsOffset + index * 2u] = hit.point.x; points[pointsOffset + index * 2u + 1u] = hit.point.y;`:``}
  ${R?`endPoints[endPointsOffset + index * 2u] = hit.endPoint.x; endPoints[endPointsOffset + index * 2u + 1u] = hit.endPoint.y;`:``}`}))}for(let[t,i,a,o,s]of[[`left`,d.leftIds,w,r.output.leftFeatures,r.output.leftRings],[`right`,d.rightIds,_,r.output.rightFeatures,r.output.rightRings]]){if(!o&&!s)continue;let r=[{name:`ids`,view:i,type:`u32`,access:`read`},{name:`${t}Table`,view:a,type:`u32`,access:`read`},{name:`state`,view:k,type:`u32`,access:`read`}];o&&r.push({name:`featureColumn`,view:o,type:`u32`,access:`read_write`}),s&&r.push({name:`ringColumn`,view:s,type:`u32`,access:`read_write`}),m.push(f(e,{id:`${n}-${t}-columns`,operation:E,variant:`${t}-columns`,bindings:r,invocationCount:u,declarations:`const SEGMENT_NONE: u32 = 0xffffffffu;
const PAIR_CAPACITY: u32 = ${u}u;
${x(t)}`,body:`if (index >= min(state[stateOffset], PAIR_CAPACITY)) { return; }
  let row = ids[idsOffset + index];
  ${o?`featureColumn[featureColumnOffset + index] = ${t}Feature(row);`:``}
  ${s?`ringColumn[ringColumnOffset + index] = ${t}Ring(row);`:``}`}))}let B=[{name:`state`,view:k,type:`u32`,access:`read`},{name:`bvhOverflow`,view:D.overflow,type:`u32`,access:`read`}],V=[[`count`,d.status.count],[`overflow`,d.status.overflow],[`requiredCount`,d.status.requiredCount],[`uncertainCount`,z]];for(let[e,t]of V)t&&B.push({name:e,view:t,type:`u32`,access:`read_write`});let H=e=>V.some(([t,n])=>t===e&&n);return m.push(f(e,{id:`${n}-finalize`,operation:E,variant:`finalize`,bindings:B,invocationCount:1,declarations:`const PAIR_CAPACITY: u32 = ${u}u;`,body:`let total = state[stateOffset];
  let overflowed = bvhOverflow[bvhOverflowOffset] != 0u || total > PAIR_CAPACITY;
  ${H(`count`)?`count[countOffset] = min(total, PAIR_CAPACITY);`:``}
  ${H(`overflow`)?`overflow[overflowOffset] = select(0u, 1u, overflowed);`:``}
  ${H(`requiredCount`)?`requiredCount[requiredCountOffset] = total;`:``}
  ${H(`uncertainCount`)?`uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];`:``}`})),m}};export{b as n,y as r,O as t};