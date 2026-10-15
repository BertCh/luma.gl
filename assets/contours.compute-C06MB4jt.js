import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{M as e,N as t,R as n,_ as r,t as i,w as a}from"./resources-CwvhEOBe.js";import{t as o}from"./gpu-scan-BuGhTLDQ.js";import{t as s}from"./draw-command-buffer-BeqRkhBa.js";import{l as c,n as l,r as u,s as d}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as f}from"./wgsl-common-BNRVzgd8.js";import{a as p,d as m}from"./raster-utils-DhdnkkOa.js";import{i as h,s as g}from"./raster-band-utils-CA-sh8xt.js";import{n as _,o as v,s as y}from"./raster-relaxation-BJ40xgro.js";import{n as b,r as x,t as S}from"./gpu-isoband-rings-DJPyQm8X.js";import{c as C,h as ee,r as w,s as T,t as E}from"./index-Bjk18V2I.js";import{t as D}from"./graph-buffers-B4_80RlX.js";import{t as O}from"./graph-swap-41Ru7LVu.js";import{t as k}from"./summary-reader-CsgWxENf.js";import{n as A,t as j}from"./gpu-terrain-derivatives-BZk8xB3m.js";import{f as M,l as N,m as P,s as F,u as te}from"./b15-common-Do8LzRH9.js";function ne(e,t,n=new Float32Array(8)){h(`getGPUIsolinesParameterValues`,n,8);let{width:r,height:i}=t,{levelCount:a,extent:o}=e;for(let[e,t]of[[`width`,r],[`height`,i]])if(!Number.isSafeInteger(t)||t<2)throw Error(`getGPUIsolinesParameterValues grid ${e} must be an integer of at least 2`);if(!Number.isSafeInteger(a)||a<0)throw Error(`Isolines levelCount must be a non-negative integer`);m(`Isolines`,o);let[s,c,l,u]=o,[d,f]=p(`Isolines`,e.localOrigin);return n.fill(0,0,8),n[0]=a,n[1]=s-d,n[2]=c-f,n[3]=(l-s)/r,n[4]=(u-c)/i,n}function I(t,r,i,a,o){if(i){if(i instanceof n)throw Error(`${t} ${r} must be a single packed view, not a chunked vector`);if(e(i,[a],`${t} ${r}`),i.length<o)throw Error(`${t} ${r} must hold at least ${o} rows`)}}function L(e,t,n){let r=t.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(r).size!==r.length)throw Error(`${e} outputs must not share buffers`);let i=new Set(n.filter(e=>e!==void 0).map(e=>e.buffer));if(r.some(e=>i.has(e)))throw Error(`${e} outputs must not share buffers with inputs`)}function R(e,t){return`
${f}
fn readCell(cell: u32, value: ptr<function, f32>) -> bool {
  let sample = raster[rasterOffset + cell];
  *value = sample;
  // Compare bits: NaN and infinities are nodata, since a crossing cannot interpolate them.
  var isValid = isFiniteValue(sample);
  ${t===void 0?``:`isValid = isValid && sample != ${c(t)};`}
  ${e?`isValid = isValid && validity[validityOffset + cell] != 0u;`:``}
  return isValid;
}

struct Corners {
  valid: bool,
  v: vec4<f32>
}

fn loadCorners(cx: u32, cy: u32) -> Corners {
  var v0 = 0.0;
  var v1 = 0.0;
  var v2 = 0.0;
  var v3 = 0.0;
  let ok0 = readCell(cy * WIDTH + cx, &v0);
  let ok1 = readCell(cy * WIDTH + cx + 1u, &v1);
  let ok2 = readCell((cy + 1u) * WIDTH + cx + 1u, &v2);
  let ok3 = readCell((cy + 1u) * WIDTH + cx, &v3);
  var corners: Corners;
  corners.valid = ok0 && ok1 && ok2 && ok3;
  corners.v = vec4<f32>(v0, v1, v2, v3);
  return corners;
}

// Bit i is set when corner i (v0..v3) is high, i.e. value >= level.
fn getHighMask(v: vec4<f32>, level: f32) -> u32 {
  return select(0u, 1u, v.x >= level) | select(0u, 2u, v.y >= level) |
    select(0u, 4u, v.z >= level) | select(0u, 8u, v.w >= level);
}

// Counter-clockwise boundary walk v0 -> v1 -> v2 -> v3 -> v0: edge k runs from corner k to corner
// (k + 1) % 4, i.e. edges e0 bottom, e1 right, e2 top, e3 left. An exit is high -> low (the
// segment starts there, keeping high on its left); an entry is low -> high (the segment ends).
fn getRotatedMask(mask: u32) -> u32 {
  return ((mask >> 1u) | (mask << 3u)) & 15u;
}
fn getExitMask(mask: u32) -> u32 { return mask & ~getRotatedMask(mask) & 15u; }
fn getEntryMask(mask: u32) -> u32 { return ~mask & getRotatedMask(mask) & 15u; }

fn getEdgeId(k: u32, cx: u32, cy: u32) -> u32 {
  let horizontalCount = HEIGHT * (WIDTH - 1u);
  if (k == 0u) { return cy * (WIDTH - 1u) + cx; }
  if (k == 2u) { return (cy + 1u) * (WIDTH - 1u) + cx; }
  if (k == 3u) { return horizontalCount + cy * WIDTH + cx; }
  return horizontalCount + cy * WIDTH + cx + 1u;
}`}var z=`GPUIsolines`,B=class{id;props;requiredHalo=1;segmentCapacity;maximumLevelCount;constructor(e){this.id=e.id??`isolines`,this.props=e;let{id:t}=this,{output:n}=e,r=n?.polylines;g(t,e.grid);let{width:i,height:a}=e.grid;if(i<2||a<2)throw Error(`${t} grid width and height must be at least 2`);if(i*a>4294967295/2)throw Error(`${t} raster is too large for 32-bit edge identifiers`);if(e.noDataValue!==void 0&&!Number.isFinite(e.noDataValue))throw Error(`${t} noDataValue must be finite (NaN cells are always nodata)`);if(I(t,`values`,e.values,`float32`,i*a),I(t,`validity`,e.validity,`uint32`,i*a),I(t,`levels`,e.levels,`float32`,1),I(t,`parameters`,e.parameters,`float32`,8),!n?.segments)throw Error(`${t} needs output.segments`);this.maximumLevelCount=e.levels.length,this.segmentCapacity=n.segments.length;let o=this.segmentCapacity;if(o<1)throw Error(`${t} output.segments must hold at least one row`);if(I(t,`output.segments`,n.segments,`float32x4`,1),I(t,`output.segmentLevels`,n.segmentLevels,`uint32`,o),I(t,`output.segmentEdges`,n.segmentEdges,`uint32x2`,o),!n.status.count)throw Error(`${t} needs output.status.count`);if(!n.status?.overflow)throw Error(`${t} needs output.status.overflow`);for(let[e,r]of[[`status.count`,n.status.count],[`status.overflow`,n.status.overflow],[`status.requiredCount`,n.status.requiredCount]])I(t,`output.${e}`,r,`uint32`,1);if(r){I(t,`output.polylines.vertices`,r.vertices,`float32x2`,1),I(t,`output.polylines.polylineOffsets`,r.polylineOffsets,`uint32`,o+1),I(t,`output.polylines.polylineLevels`,r.polylineLevels,`uint32`,o),I(t,`output.polylines.polylineClosed`,r.polylineClosed,`uint32`,o);for(let e of[`polylineCount`,`vertexCount`]){if(!r[e])throw Error(`${t} needs output.polylines.${e}`);I(t,`output.polylines.${e}`,r[e],`uint32`,1)}if(!n.status.polylineOverflow)throw Error(`${t} needs output.status.polylineOverflow`);I(t,`output.status.polylineOverflow`,n.status.polylineOverflow,`uint32`,1)}L(t,[...this.getOutputViews()],[e.values,e.validity,e.levels,e.parameters])}getOutputViews(){let{output:e}=this.props,{polylines:t,status:n}=e;return[e.segments,e.segmentLevels,e.segmentEdges,e.status.count,e.status.requiredCount,n.overflow,n.polylineOverflow,t?.vertices,t?.polylineOffsets,t?.polylineLevels,t?.polylineClosed,t?.polylineCount,t?.vertexCount].filter(e=>e!==void 0)}getCommandNodes(e){let{id:t,props:n,segmentCapacity:i,maximumLevelCount:s}=this,{output:c}=n,{polylines:l}=c,{width:f,height:p}=n.grid;r(t,e,[n.values,n.validity,n.levels,n.parameters,...this.getOutputViews()]);let m=[],h=(f-1)*(p-1),g=`const WIDTH: u32 = ${f}u;
const HEIGHT: u32 = ${p}u;
const MAXIMUM_LEVELS: u32 = ${s}u;
const CAPACITY: u32 = ${i}u;
${R(!!n.validity,n.noDataValue)}`,_=[{name:`params`,view:n.parameters,type:`f32`,access:`read`},{name:`levels`,view:n.levels,type:`f32`,access:`read`},{name:`raster`,view:n.values,type:`f32`,access:`read`},...n.validity?[{name:`validity`,view:n.validity,type:`u32`,access:`read`}]:[]],v=a(e,`${t}-cell-counts`,`uint32`,h),y=a(e,`${t}-cell-offsets`,`uint32`,h),b=a(e,`${t}-total`,`uint32`,1),x=!!l,S=c.segmentEdges??(x?a(e,`${t}-edges`,`uint32x2`,i):void 0);return m.push(d(e,{id:`${t}-count`,operation:z,variant:`count`,bindings:[..._,{name:`counts`,view:v,type:`u32`,access:`read_write`}],invocationCount:h,declarations:g,body:`let cx = index % (WIDTH - 1u);
  let cy = index / (WIDTH - 1u);
  let corners = loadCorners(cx, cy);
  var cellTotal = 0u;
  if (corners.valid) {
    let levelCount = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_LEVELS);
    for (var levelIndex = 0u; levelIndex < levelCount; levelIndex++) {
      let mask = getHighMask(corners.v, levels[levelsOffset + levelIndex]);
      cellTotal += countOneBits(getExitMask(mask));
    }
  }
  counts[countsOffset + index] = cellTotal;`}),...new o({id:`${t}-cell-scan`,input:v,output:y}).getCommandNodes(e),d(e,{id:`${t}-total`,operation:z,variant:`total`,bindings:[{name:`counts`,view:v,type:`u32`,access:`read`},{name:`offsets`,view:y,type:`u32`,access:`read`},{name:`totalOut`,view:b,type:`u32`,access:`read_write`}],invocationCount:1,body:`totalOut[totalOutOffset] =
    offsets[offsetsOffset + ${h-1}u] + counts[countsOffset + ${h-1}u];`})),m.push(d(e,{id:`${t}-scatter`,operation:z,variant:`scatter`,bindings:[..._,{name:`offsets`,view:y,type:`u32`,access:`read`},{name:`segments`,view:c.segments,type:`f32`,access:`read_write`},{name:`segmentLevels`,view:c.segmentLevels,type:`u32`,access:`read_write`},...S?[{name:`edges`,view:S,type:`u32`,access:`read_write`}]:[]],invocationCount:h,declarations:`${g}
${V()}`,body:`let cx = index % (WIDTH - 1u);
  let cy = index / (WIDTH - 1u);
  let corners = loadCorners(cx, cy);
  if (!corners.valid) {
    return;
  }
  let v = corners.v;
  var position = offsets[offsetsOffset + index];
  let levelCount = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_LEVELS);
  for (var levelIndex = 0u; levelIndex < levelCount; levelIndex++) {
    let level = levels[levelsOffset + levelIndex];
    let mask = getHighMask(v, level);
    let exits = getExitMask(mask);
    if (exits == 0u) {
      continue;
    }
    let entries = getEntryMask(mask);
    // Saddles have two exits. High corners are joined when the centre is high.
    let isJoined = countOneBits(exits) < 2u || isCentreHigh(v, level);
    for (var exitEdge = 0u; exitEdge < 4u; exitEdge++) {
      if (((exits >> exitEdge) & 1u) == 0u) {
        continue;
      }
      let entryEdge = getPairedEntry(exitEdge, entries, isJoined);
      // A corner exactly at the level gives a zero-length segment. It stays: it is the bridge
      // that links the neighbouring cells' chains in the edge-based stitching.
      if (position < CAPACITY) {
        let start = getEdgeCrossing(exitEdge, level, v, cx, cy);
        let end = getEdgeCrossing(entryEdge, level, v, cx, cy);
        let base = segmentsOffset + 4u * position;
        segments[base] = start.x;
        segments[base + 1u] = start.y;
        segments[base + 2u] = end.x;
        segments[base + 3u] = end.y;
        segmentLevels[segmentLevelsOffset + position] = levelIndex;
        ${S?`edges[edgesOffset + 2u * position] = getEdgeId(exitEdge, cx, cy);
        edges[edgesOffset + 2u * position + 1u] = getEdgeId(entryEdge, cx, cy);`:``}
      }
      position++;
    }
  }`}),u(e,{id:`${t}-publish`,operation:z,requiredCount:b,output:{ids:c.segmentLevels,status:{count:c.status.count,overflow:c.status.overflow,requiredCount:c.status.requiredCount}}})),l&&S&&m.push(...this.getStitchNodes(e,{counts:v,offsets:y,edges:S,polylines:l,common:H(f,p)})),m}getStitchNodes(e,t){let{id:n,props:r,segmentCapacity:i}=this,{output:s}=r,{counts:c,offsets:u,edges:f,polylines:p}=t,m=[],h=(t,r=i)=>a(e,`${n}-${t}`,`uint32`,r),g=(r,a,o,s=``,c=i)=>m.push(d(e,{id:`${n}-${r}`,operation:z,variant:r,bindings:a,invocationCount:c,declarations:`const CAPACITY: u32 = ${i}u;
${t.common}
${s}`,body:o})),b=(e,t)=>({name:e,view:t,type:`u32`,access:`read`}),x=(e,t)=>({name:e,view:t,type:`u32`,access:`read_write`}),S=h(`active`,1);g(`stitch-active`,[b(`countIn`,s.status.count),b(`overflowIn`,s.status.overflow),x(`activeOut`,S)],`activeOut[activeOutOffset] =
    select(min(countIn[countInOffset], CAPACITY), 0u, overflowIn[overflowInOffset] != 0u);`,``,1);let C=h(`previous`);g(`links`,[b(`edges`,f),b(`levels`,s.segmentLevels),b(`offsets`,u),b(`counts`,c),b(`activeCount`,S),x(`previousOut`,C)],`var previousSegment = NONE;
  if (index < activeCount[activeCountOffset]) {
    let startEdge = edges[edgesOffset + 2u * index];
    let endEdge = edges[edgesOffset + 2u * index + 1u];
    let level = levels[levelsOffset + index];
    previousSegment = findLinkedSegment(getNeighbourCell(endEdge, startEdge), level, 1u, startEdge);
  }
  previousOut[previousOutOffset + index] = previousSegment;`,re());let ee=2*Math.ceil((Math.ceil(Math.log2(i))+1)/2),w=[0,1].map(e=>h(`jump-${e}-state`,4*i));g(`jump-init`,[b(`previous`,C),x(`stateOut`,w[0])],`let hasPrevious = previous[previousOffset + index] != NONE;
  let base = stateOutOffset + 4u * index;
  stateOut[base] = select(index, previous[previousOffset + index], hasPrevious);
  stateOut[base + 1u] = select(0u, index + 1u, hasPrevious);
  stateOut[base + 2u] = 0u;
  stateOut[base + 3u] = select(0u, 1u, hasPrevious);`);let T=y(e,`${n}-jump`,z,i,n);m.push(v(e,{id:`${n}-jump-reset`,operation:z,state:T}),..._(e,{id:`${n}-jump`,operation:z,state:T,maxIterations:ee,groupSize:2,createRound:({nodeId:t,iteration:n,slotBinding:r,roundSlotWGSL:i,condition:a,extraResources:o})=>d(e,{id:t,operation:z,variant:`jump`,bindings:[b(`stateIn`,w[n%2]),x(`stateOut`,w[(n+1)%2]),{name:`status`,view:T.status,type:`atomic<u32>`,access:`read_write`},r],invocationCount:T.invocationCount,body:`let ownBase = stateInOffset + 4u * index;
  let jump = stateIn[ownBase];
  let jumpBase = stateInOffset + 4u * jump;
  let ownKey = stateIn[ownBase + 1u];
  let jumpKey = stateIn[jumpBase + 1u];
  let ownWins = ownKey <= jumpKey;
  let outBase = stateOutOffset + 4u * index;
  stateOut[outBase] = stateIn[jumpBase];
  stateOut[outBase + 1u] = select(jumpKey, ownKey, ownWins);
  stateOut[outBase + 2u] = select(
    stateIn[jumpBase + 2u] + stateIn[ownBase + 3u],
    stateIn[ownBase + 2u],
    ownWins
  );
  stateOut[outBase + 3u] = stateIn[ownBase + 3u] + stateIn[jumpBase + 3u];
  let roundBit = 1u << ${i};
  if (!ownWins && (atomicLoad(&status[statusOffset]) & roundBit) == 0u) {
    atomicOr(&status[statusOffset], roundBit);
  }`,condition:a,extraResources:o})}));let E=w[0],D=h(`head-of`),O=h(`rank-of`);g(`resolve`,[b(`stateIn`,E),x(`headOut`,D),x(`rankOut`,O)],`let base = stateInOffset + 4u * index;
  let key = stateIn[base + 1u];
  let isRing = key != 0u;
  headOut[headOutOffset + index] = select(stateIn[base], key - 1u, isRing);
  rankOut[rankOutOffset + index] =
    stateIn[base + 2u] | select(0u, 0x80000000u, isRing);`);let k=h(`chain-length`),A=h(`head-kind`);m.push(l(e,{id:`${n}-chain-length-clear`,operation:z,view:k,type:`u32`,value:`0u`})),m.push(d(e,{id:`${n}-chain-length`,operation:z,variant:`chain-length`,bindings:[b(`headOf`,D),b(`rankOf`,O),b(`activeCount`,S),{name:`chainLength`,view:k,type:`atomic<u32>`,access:`read_write`},x(`headKindOut`,A)],invocationCount:i,body:`var kind = 0u;
  if (index < activeCount[activeCountOffset]) {
    let packed = rankOf[rankOfOffset + index];
    let rank = packed & 0x7fffffffu;
    atomicMax(&chainLength[chainLengthOffset + headOf[headOfOffset + index]], rank + 1u);
    kind = select(0u, select(1u, 2u, (packed >> 31u) != 0u), rank == 0u);
  }
  headKindOut[headKindOutOffset + index] = kind;`}));let j=h(`head-flag`),M=h(`head-vertex-count`),N=h(`head-index`),P=h(`vertex-offset`);g(`scan-input`,[b(`headKind`,A),b(`chainLength`,k),x(`flagOut`,j),x(`vertexCountOut`,M)],`let isHead = headKind[headKindOffset + index] != 0u;
  flagOut[flagOutOffset + index] = select(0u, 1u, isHead);
  vertexCountOut[vertexCountOutOffset + index] =
    select(0u, chainLength[chainLengthOffset + index] + 1u, isHead);`),m.push(...new o({id:`${n}-head-scan`,input:j,output:N}).getCommandNodes(e),...new o({id:`${n}-vertex-scan`,input:M,output:P}).getCommandNodes(e)),g(`polyline-records`,[b(`headKind`,A),b(`headIndex`,N),b(`vertexOffset`,P),b(`segmentLevels`,s.segmentLevels),x(`polylineOffsets`,p.polylineOffsets),x(`polylineLevels`,p.polylineLevels),x(`polylineClosed`,p.polylineClosed)],`let kind = headKind[headKindOffset + index];
  if (kind != 0u) {
    let polyline = headIndex[headIndexOffset + index];
    polylineOffsets[polylineOffsetsOffset + polyline] = vertexOffset[vertexOffsetOffset + index];
    polylineLevels[polylineLevelsOffset + polyline] = segmentLevels[segmentLevelsOffset + index];
    polylineClosed[polylineClosedOffset + polyline] = select(0u, 1u, kind == 2u);
  }`);let F=p.vertices.length,te=a(e,`${n}-staged-vertices`,`float32x2`,F);m.push(d(e,{id:`${n}-vertices`,operation:z,variant:`vertices`,bindings:[b(`headOf`,D),b(`rankOf`,O),b(`activeCount`,S),b(`chainLength`,k),b(`vertexOffset`,P),{name:`segments`,view:s.segments,type:`f32`,access:`read`},{name:`vertices`,view:te,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`const VERTEX_CAPACITY: u32 = ${F}u;`,body:`if (index >= activeCount[activeCountOffset]) {
    return;
  }
  let head = headOf[headOfOffset + index];
  let packed = rankOf[rankOfOffset + index];
  let rank = packed & 0x7fffffffu;
  let length = chainLength[chainLengthOffset + head];
  let base = vertexOffset[vertexOffsetOffset + head];
  if (base + length + 1u > VERTEX_CAPACITY) {
    return;
  }
  let own = segmentsOffset + 4u * index;
  let slot = verticesOffset + 2u * (base + rank);
  vertices[slot] = segments[own];
  vertices[slot + 1u] = segments[own + 1u];
  if (rank + 1u == length) {
    // The last vertex of a ring repeats the first one; an open chain ends at its last p1.
    let closing = (packed >> 31u) != 0u;
    let source = select(own + 2u, segmentsOffset + 4u * head, closing);
    vertices[slot + 2u] = segments[source];
    vertices[slot + 3u] = segments[source + 1u];
  }`}));let ne=h(`stitch-totals`,2);g(`stitch-totals`,[b(`headIndex`,N),b(`headFlag`,j),b(`vertexOffset`,P),b(`vertexCount`,M),x(`totalsOut`,ne)],`totalsOut[totalsOutOffset] =
    headIndex[headIndexOffset + CAPACITY - 1u] + headFlag[headFlagOffset + CAPACITY - 1u];
  totalsOut[totalsOutOffset + 1u] =
    vertexOffset[vertexOffsetOffset + CAPACITY - 1u] +
    vertexCount[vertexCountOffset + CAPACITY - 1u];`,``,1),g(`stitch-publish`,[b(`totals`,ne),b(`segmentOverflow`,s.status.overflow),x(`polylineOffsets`,p.polylineOffsets),x(`polylineCountOut`,p.polylineCount),x(`vertexCountOut`,p.vertexCount),x(`overflowOut`,s.status.polylineOverflow)],`let polylineTotal = totals[totalsOffset];
  let vertexTotal = totals[totalsOffset + 1u];
  polylineOffsets[polylineOffsetsOffset + polylineTotal] = vertexTotal;
  // Offsets are nondecreasing: keep the longest prefix of polylines that fits the vertex capacity.
  var low = 0u;
  var high = polylineTotal;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high + 1u) / 2u;
    if (polylineOffsets[polylineOffsetsOffset + middle] <= VERTEX_CAPACITY) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  polylineCountOut[polylineCountOutOffset] = low;
  vertexCountOut[vertexCountOutOffset] = polylineOffsets[polylineOffsetsOffset + low];
  overflowOut[overflowOutOffset] =
    select(0u, 1u, segmentOverflow[segmentOverflowOffset] != 0u || low < polylineTotal);`,`const VERTEX_CAPACITY: u32 = ${F}u;`,1);let I=h(`start-flag`,F),L=h(`keep-flag`,F),R=h(`keep-offset`,F),B=`const VERTEX_CAPACITY: u32 = ${F}u;`,V={name:`staged`,view:te,type:`f32`,access:`read`};return m.push(l(e,{id:`${n}-start-flag-clear`,operation:z,view:I,type:`u32`,value:`0u`})),g(`start-flags`,[b(`headKind`,A),b(`vertexOffset`,P),x(`startOut`,I)],`if (headKind[headKindOffset + index] != 0u) {
    let first = vertexOffset[vertexOffsetOffset + index];
    if (first < VERTEX_CAPACITY) {
      startOut[startOutOffset + first] = 1u;
    }
  }`,B),g(`keep-flags`,[V,b(`startFlag`,I),b(`vertexCount`,p.vertexCount),x(`keepOut`,L)],`var keep = 0u;
  if (index < vertexCount[vertexCountOffset]) {
    let current = staged[stagedOffset + 2u * index];
    let currentY = staged[stagedOffset + 2u * index + 1u];
    let isStart = startFlag[startFlagOffset + index] != 0u || index == 0u;
    // Bit-identical neighbours inside one polyline collapse; polyline starts always stay.
    let isDuplicate = !isStart &&
      staged[stagedOffset + 2u * index - 2u] == current &&
      staged[stagedOffset + 2u * index - 1u] == currentY;
    keep = select(1u, 0u, isDuplicate);
  }
  keepOut[keepOutOffset + index] = keep;`,B,F),m.push(...new o({id:`${n}-keep-scan`,input:L,output:R}).getCommandNodes(e)),g(`compact-vertices`,[V,b(`keepFlag`,L),b(`keepOffset`,R),{name:`vertices`,view:p.vertices,type:`f32`,access:`read_write`}],`if (keepFlag[keepFlagOffset + index] != 0u) {
    let destination = verticesOffset + 2u * keepOffset[keepOffsetOffset + index];
    vertices[destination] = staged[stagedOffset + 2u * index];
    vertices[destination + 1u] = staged[stagedOffset + 2u * index + 1u];
  }`,B,F),g(`compact-offsets`,[b(`keepFlag`,L),b(`keepOffset`,R),b(`polylineCount`,p.polylineCount),b(`vertexCount`,p.vertexCount),x(`polylineOffsets`,p.polylineOffsets)],`if (index <= polylineCount[polylineCountOffset]) {
    let old = polylineOffsets[polylineOffsetsOffset + index];
    let total = keepOffset[keepOffsetOffset + VERTEX_CAPACITY - 1u] +
      keepFlag[keepFlagOffset + VERTEX_CAPACITY - 1u];
    polylineOffsets[polylineOffsetsOffset + index] =
      select(keepOffset[keepOffsetOffset + old], total, old >= vertexCount[vertexCountOffset]);
  }`,B,i+1),g(`compact-totals`,[b(`keepFlag`,L),b(`keepOffset`,R),x(`vertexCountOut`,p.vertexCount)],`vertexCountOut[vertexCountOutOffset] =
    keepOffset[keepOffsetOffset + VERTEX_CAPACITY - 1u] +
    keepFlag[keepFlagOffset + VERTEX_CAPACITY - 1u];`,B,1),m}};function V(){return`
// Saddles have two exits. High corners are joined when the centre is high.
fn isCentreHigh(v: vec4<f32>, level: f32) -> bool {
  return ((v.x + v.y) + (v.z + v.w)) * 0.25 >= level;
}

// The entry edge a segment leaving through exitEdge ends on, walking the cell boundary.
fn getPairedEntry(exitEdge: u32, entries: u32, isJoined: bool) -> u32 {
  for (var step = 1u; step < 4u; step++) {
    let candidate = select((exitEdge + 4u - step) % 4u, (exitEdge + step) % 4u, isJoined);
    if (((entries >> candidate) & 1u) != 0u) {
      return candidate;
    }
  }
  return exitEdge;
}

fn getEdgeCrossing(k: u32, level: f32, v: vec4<f32>, cx: u32, cy: u32) -> vec2<f32> {
  let minX = params[paramsOffset + 1u];
  let minY = params[paramsOffset + 2u];
  let cellWidth = params[paramsOffset + 3u];
  let cellHeight = params[paramsOffset + 4u];
  // Canonical orientation: horizontal edges run +x, vertical edges run +y.
  var a = v.x;
  var b = v.y;
  if (k == 1u) {
    a = v.y;
    b = v.z;
  } else if (k == 2u) {
    a = v.w;
    b = v.z;
  } else if (k == 3u) {
    a = v.x;
    b = v.w;
  }
  let t = (level - a) / (b - a);
  if (k == 0u || k == 2u) {
    let gridX = f32(cx) + t;
    let gridY = f32(cy + select(0u, 1u, k == 2u));
    return vec2<f32>(minX + (gridX + 0.5) * cellWidth, minY + (gridY + 0.5) * cellHeight);
  }
  let gridX = f32(cx + select(0u, 1u, k == 1u));
  let gridY = f32(cy) + t;
  return vec2<f32>(minX + (gridX + 0.5) * cellWidth, minY + (gridY + 0.5) * cellHeight);
}`}function H(e,t){return`
const WIDTH: u32 = ${e}u;
const HEIGHT: u32 = ${t}u;
const NONE: u32 = 0xffffffffu;

// The two cells that share edge e: x is below or left of it, y above or right of it.
fn getEdgeCells(e: u32) -> vec2<u32> {
  let cellColumns = WIDTH - 1u;
  let cellRows = HEIGHT - 1u;
  let horizontalCount = HEIGHT * cellColumns;
  var low = NONE;
  var high = NONE;
  if (e < horizontalCount) {
    let r = e / cellColumns;
    let c = e % cellColumns;
    if (r >= 1u) { low = (r - 1u) * cellColumns + c; }
    if (r < cellRows) { high = r * cellColumns + c; }
  } else {
    let i = e - horizontalCount;
    let r = i / WIDTH;
    let c = i % WIDTH;
    if (c >= 1u && r < cellRows) { low = r * cellColumns + c - 1u; }
    if (c < cellColumns && r < cellRows) { high = r * cellColumns + c; }
  }
  return vec2<u32>(low, high);
}

fn cellHasEdge(cell: u32, e: u32) -> bool {
  let cellColumns = WIDTH - 1u;
  let cx = cell % cellColumns;
  let cy = cell / cellColumns;
  let bottom = cy * cellColumns + cx;
  let left = HEIGHT * cellColumns + cy * WIDTH + cx;
  return e == bottom || e == bottom + cellColumns || e == left || e == left + 1u;
}

// Cell on the other side of edge \`across\`, given another edge of the segment's own cell.
fn getNeighbourCell(known: u32, across: u32) -> u32 {
  let cells = getEdgeCells(across);
  if (cells.x != NONE && cellHasEdge(cells.x, known)) {
    return cells.y;
  }
  return cells.x;
}`}function re(){return`
// Segment of \`level\` in \`cell\` whose start (field 0) or end (field 1) edge is \`edge\`.
fn findLinkedSegment(cell: u32, level: u32, field: u32, edge: u32) -> u32 {
  if (cell == NONE) {
    return NONE;
  }
  let first = offsets[offsetsOffset + cell];
  let segmentCount = counts[countsOffset + cell];
  for (var i = 0u; i < segmentCount; i++) {
    let position = first + i;
    if (position >= CAPACITY) {
      break;
    }
    if (levels[levelsOffset + position] == level &&
        edges[edgesOffset + 2u * position + field] == edge) {
      return position;
    }
  }
  return NONE;
}`}var U=`@group(0) @binding(auto) var<storage, read> rasterBounds`,ie=class extends T{static layerName=`IsobandTriangleLayer`;constructor(e){super(e)}getShaderSource(){let e=super.getShaderSource(),t=e.indexOf(U);if(t<0)throw Error(`IsobandTriangleLayer could not find the raster binding block`);let{triangleFacilities:n}=this.props,r=n?`@group(0) @binding(auto) var<storage, read> bandFacilities: array<u32>;`:``,i=n?`let facility = bandFacilities[vertexIndex / 3u];
  let base = spatialAnalysisStyle.palette[facility % max(spatialAnalysisStyle.paletteSize, 1u)];
  let shade = mix(1.15, 0.5, clamp(f32(band) / (breakCount + 1.0), 0.0, 1.0));
  output.color = vec4<f32>(clamp(base.rgb * shade, vec3<f32>(0.0), vec3<f32>(1.0)), base.a);`:`let t = clamp((f32(band) + 0.5) / (breakCount + 1.0), 0.0, 0.9999);
  let packedColor = styleValues[u32(t * f32(paletteSize))];
  output.color = vec4<f32>(
    f32(packedColor & 255u),
    f32((packedColor >> 8u) & 255u),
    f32((packedColor >> 16u) & 255u),
    f32(packedColor >> 24u)
  ) / 255.0;`;return`${e.slice(0,t)}
@group(0) @binding(auto) var<storage, read> bandTriangles: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> bandIds: array<u32>;
${r}

struct BandVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> BandVertexOutput {
  var output: BandVertexOutput;
  let band = bandIds[vertexIndex / 3u];
  let breakCount = max(styleExtent[0], 0.0);
  let paletteSize = max(arrayLength(&styleValues), 1u);
  ${i}
  output.position = projectSpatialAnalysisPosition(getSpatialAnalysisPosition(bandTriangles[vertexIndex]));
  return output;
}

@fragment fn fragmentMain(input: BandVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity);
}
`}getBindings(e){let{triangles:t,triangleBands:n,triangleFacilities:r}=this.props;return{...this.getStyleBindings(e),bandTriangles:t,bandIds:n,...r?{bandFacilities:r}:{}}}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;if(!t||!n)return;this.writeLayerStyle(n);let{drawCommands:r,drawCommandIndex:i=0}=this.props;r&&(t.setInstanceCount(0),t.draw(e),r.draw(e,i))}},ae=class extends C{static layerName=`PolylineLayer`;constructor(e){super(e)}getShaderSource(){let e=super.getShaderSource().replace(`var<storage, read> segmentPositions: array<vec4<f32>>`,`var<storage, read> segmentPositions: array<vec2<f32>>`);if(e=e.replace(/let segment = segmentPositions\[row\];\s*var color = getSpatialAnalysisColor\(getSpatialAnalysisValueRow\(row\)\);/,`let polylineCount = bitcast<u32>(styleExtent[0]);
  var segment = vec4<f32>(0.0);
  var polylineValid = false;
  var polylineLevel = 0u;
  if (polylineCount > 0u) {
    // Last polyline whose first vertex is at or before this instance.
    var low = 0u;
    var high = polylineCount;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (segmentIds[middle] <= row) {
        low = middle;
      } else {
        high = middle;
      }
    }
    if (row + 1u < segmentIds[low + 1u]) {
      segment = vec4<f32>(segmentPositions[row], segmentPositions[row + 1u]);
      polylineLevel = styleValueIndices[low];
      polylineValid = true;
    }
  }
  var color = getSpatialAnalysisColor(polylineLevel);`),e=e.replace(`segment.x != segment.x || segment.z != segment.z`,`!polylineValid`),!e.includes(`polylineValid`)||e.includes(`segment.x != segment.x`))throw Error(`PolylineLayer could not patch the segment vertex shader`);return e}getBindings(e){let{polylineOffsets:t}=this.props;return{...super.getBindings(e),segmentIds:t}}},oe=2,W=12e5,G=3e6,K=12e5,q=32768,J=256,se=315,ce=40,le=0,ue=1,de=2,fe=16,pe=[255,255,255,235],me=[255,60,200,235],he={light:{line:[40,40,50,170],index:[10,10,20,255]},dark:{line:[235,235,245,160],index:[255,255,255,255]}},ge={fine:1e-4,"half-cell":15,cell:31};async function _e(e){let{device:n}=e,r=F(e.datasets.get(`grand-canyon-dem`),oe),{width:a,height:o,cellCount:c}=r,l=[r.origin[0],r.origin[1],0],u=r.bounds,d=r.rasterGrid,f=N(r),p=new i(n,`contours`),m=!1,h=1/0,g=-1/0;for(let e of r.elevation)e<h&&(h=e),e>g&&(g=e);let _=e=>p.createBuffer(e,c*4),v=p.createBuffer(`elevation-south-first`,te(r.elevation,a,o)),y=_(`slope`),I=_(`hillshade`),L=_(`values`),R=p.createParameterBuffer(`derivatives-settings`,`float32`,8),z=p.createParameterBuffer(`isoline-parameters`,`float32`,8),V=p.createParameterBuffer(`isoband-parameters`,`float32`,12),H=p.createBuffer(`levels`,256),re=p.createBuffer(`level-style`,256),U=p.createBuffer(`palette`,new Uint32Array(J)),_e=p.createBuffer(`segments`,W*16),ve=p.createBuffer(`segment-levels`,W*4),Y=p.createBuffer(`segment-count`,4),ye=p.createBuffer(`segment-overflow`,4),be=p.createBuffer(`triangles`,G*3*8),xe=p.createBuffer(`triangle-bands`,G*4),Se=p.createBuffer(`triangle-count`,4),Ce=p.createBuffer(`triangle-overflow`,4),we=p.createBuffer(`band-vertex-count`,4),Te=p.createBuffer(`polyline-vertices`,W*2*8),Ee=p.createBuffer(`polyline-offsets`,(W+1)*4),De=p.createBuffer(`polyline-levels`,W*4),Oe=p.createBuffer(`polyline-closed`,W*4),ke=p.createBuffer(`polyline-count`,4),Ae=p.createBuffer(`polyline-vertex-count`,4),je=p.createBuffer(`polyline-overflow`,4),Me=p.createBuffer(`ring-offsets`,(q+1)*4),Ne=p.createBuffer(`ring-positions`,K*8),Pe=p.createBuffer(`ring-bands`,q*4),Fe=p.createBuffer(`ring-is-hole`,q*4),Ie=p.createBuffer(`ring-count`,4),Le=p.createBuffer(`ring-overflow`,4),Re=p.createBuffer(`ring-open`,4),ze=p.createBuffer(`ring-edge-count`,4),Be=p.createBuffer(`ring-edge-overflow`,4),Ve=p.createBuffer(`ring-style`,Uint32Array.of(0,1)),X=p.track(new s(n,{id:`contours-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0},{vertexCount:6,instanceCount:0},{vertexCount:0,instanceCount:1}]})),Z=new t(n,{id:`contours-setup`});Z.add(new j({id:`derivatives`,grid:f,elevation:{id:`elevation`,format:`float32`,storage:{kind:`buffer`,values:D(Z,`elevation`,v,`float32`,c)}},parameters:R.importToGraph(Z),output:{slope:D(Z,`slope`,y,`float32`,c),hillshade:D(Z,`hillshade`,I,`float32`,c)},cellSizeMode:r.cellSizeMode,rowDirection:`north`}));let He=p.track(await Z.compileAsync()),Ue=async e=>{let r=new t(n,{id:e?`contours-stitched`:`contours-plain`}),i=D(r,`values`,L,`float32`,c),a=D(r,`levels`,H,`float32`,64);return r.add(new b({id:`bands`,grid:d,values:i,breaks:a,parameters:V.importToGraph(r),output:{triangles:D(r,`triangles`,be,`float32x2`,G*3),triangleBands:D(r,`triangle-bands`,xe,`uint32`,G),status:{count:D(r,`triangle-count`,Se,`uint32`,1),vertexCount:D(r,`band-vertex-count`,we,`uint32`,1),overflow:D(r,`triangle-overflow`,Ce,`uint32`,1)}}})),r.add(new B({id:`lines`,grid:d,values:i,levels:a,parameters:z.importToGraph(r),output:{segments:D(r,`segments`,_e,`float32x4`,W),segmentLevels:D(r,`segment-levels`,ve,`uint32`,W),polylines:e?{vertices:D(r,`polyline-vertices`,Te,`float32x2`,W*2),polylineOffsets:D(r,`polyline-offsets`,Ee,`uint32`,W+1),polylineLevels:D(r,`polyline-levels`,De,`uint32`,W),polylineClosed:D(r,`polyline-closed`,Oe,`uint32`,W),polylineCount:D(r,`polyline-count`,ke,`uint32`,1),vertexCount:D(r,`polyline-vertex-count`,Ae,`uint32`,1)}:void 0,status:{count:D(r,`segment-count`,Y,`uint32`,1),overflow:D(r,`segment-overflow`,ye,`uint32`,1),polylineOverflow:e?D(r,`polyline-overflow`,je,`uint32`,1):void 0}}})),p.track(await r.compileAsync())},We=await Ue(!1),Ge=await Ue(!0),Ke=async(e,r)=>{let i=new t(n,{id:`contours-rings`});return i.add(new S({id:`band-rings`,grid:d,values:D(i,`values`,L,`float32`,c),breaks:D(i,`levels`,H,`float32`,64),parameters:V.importToGraph(i),edgeCapacity:K,vertexTolerance:ge[e],splitTouchingRings:r,output:{ringOffsets:D(i,`ring-offsets`,Me,`uint32`,q+1),positions:D(i,`ring-positions`,Ne,`float32x2`,K),ringGroups:D(i,`ring-bands`,Pe,`uint32`,q),ringIsHole:D(i,`ring-is-hole`,Fe,`uint32`,q),status:{count:D(i,`ring-count`,Ie,`uint32`,1),overflow:D(i,`ring-overflow`,Le,`uint32`,1),openSegmentCount:D(i,`ring-open`,Re,`uint32`,1),edgeCount:D(i,`edge-count`,ze,`uint32`,1),edgeOverflow:D(i,`edge-overflow`,Be,`uint32`,1)}}})),await i.compileAsync()},Q=!0,qe=!1,Je=``,Ye=``,$=null,Xe=!1,Ze=!1;function Qe(e){return`${e.ringTolerance}-${e.splitTouchingRings}`}let $e=p.track(new O);function et(){let t=e.options,n=Qe(t);return $&&n===Ye?Promise.resolve(!0):$e.request({build:async()=>p.track(await Ke(t.ringTolerance,t.splitTouchingRings)),commit:t=>{let r=$;$=t,Ye=n,r&&p.release(r),Q=!0,e.requestLayers()},discard:e=>p.release(e)})}function tt(){let t=ee[e.options.ramp],n=new Uint32Array(J);for(let e=0;e<J;e++){let r=e/(J-1)*(t.length-1),i=Math.min(Math.floor(r),t.length-2),a=r-i,o=[0,1,2].map(e=>Math.round(t[i][e]*(1-a)+t[i+1][e]*a));n[e]=(o[0]|o[1]<<8|o[2]<<16|255<<24)>>>0}U.write(n)}function nt(e){let t=e.source===`elevation`,n=t?e.elevationInterval:e.slopeInterval,r=t?h:0,i=t?g:85,a=Math.floor(r/n)+1,o=Math.max(a,Math.floor(i/n)),s=Math.min(64,o-a+1);return{values:Array.from({length:s},(e,t)=>n*(a+t)),offset:a}}function rt(e,t){let[n,r]=e.source===`elevation`?e.elevationWindow:e.slopeWindow,i=e=>t.filter(t=>t<=e).length;return[i(n),Math.min(t.length,i(r))]}function it(){let t=e.options,{values:n,offset:r}=nt(t),i=n.length,a=new Float32Array(64),o=new Uint32Array(64),s=t.source===`elevation`?t.elevationInterval:t.slopeInterval;for(let e=0;e<64;e++)a[e]=e<i?n[e]:n[i-1]+(e-i+1)*s,o[e]=t.indexEvery>0&&(r+e)%t.indexEvery===0?1:0;let[c,l]=rt(t,n),f=`${t.source}|${Array.from(a).join(`,`)}|${o.join(``)}|${c}|${l}`;f!==Je&&(Je=f,H.write(a),re.write(o),z.write(ne({levelCount:i,extent:u},d)),V.write(x({breakCount:i,extent:u,firstBand:c,lastBand:l},d)),e.setReadout(`levels`,`${i} levels every ${s} ${t.source===`elevation`?`m`:`°`}${i>=64?` (capped at 64)`:``}`),e.setReadout(`bandWindow`,`bands ${c} to ${l} of ${i} drawn as filled geometry`),Q=!0)}function at(t){t.copyBufferToBuffer({sourceBuffer:e.options.source===`elevation`?v:y,sourceOffset:0,destinationBuffer:L,destinationOffset:0,size:c*4})}let ot=new k(p,`contours`,[{buffer:Y,size:4},{buffer:ye,size:4},{buffer:Se,size:4},{buffer:Ce,size:4},{buffer:ke,size:4},{buffer:Ae,size:4},{buffer:je,size:4},{buffer:Ie,size:4},{buffer:Le,size:4},{buffer:Re,size:4},{buffer:ze,size:4},{buffer:Be,size:4},{buffer:Fe,size:q*4}],t=>{if(m)return;let n=new Uint32Array(t),r=e.options;if(e.setReadout(`segments`,`${M(n[0])} of ${M(W)}${n[1]?` OVERFLOW`:``}`),e.setReadout(`triangles`,`${M(n[2])} of ${M(G)}${n[3]?` OVERFLOW`:``}`),e.setReadout(`polylines`,r.stitchLines?`${M(n[4])} lines, ${M(n[5])} vertices${n[6]?` OVERFLOW`:``}`:`off (plain segments)`),r.showRings&&Xe){let t=Math.min(n[7],q),r=0;for(let e=0;e<t;e++)r+=n[12+e]?1:0;e.setReadout(`rings`,`${M(t-r)} shells, ${M(r)} holes${n[8]?` OVERFLOW`:``}`),e.setReadout(`ringEdges`,`${M(n[10])} boundary edges of ${M(K)}${n[11]?` OVERFLOW`:``}; ${M(n[9])} open segments`)}else e.setReadout(`rings`,`off`),e.setReadout(`ringEdges`,`off`)});async function st(){if(!(Ze||m)){Ze=!0,e.setReadout(`timings`,`measuring...`);try{let t=async t=>w(n,t,{parameters:void 0,completionBuffer:Y,signal:e.signal,runs:3,warmUpRuns:1,repetitions:2}),r=await t(We),i=await t(Ge);await et();let a=await t($);e.setReadout(`timings`,`lines + bands ${E(r)} · stitched ${E(i)} · rings ${E(a)}`)}catch{}finally{Ze=!1,Q=!0}}}return R.write(A({azimuthDegrees:se,altitudeDegrees:ce},f)),tt(),it(),e.setReadout(`grid`,`${a} × ${o} cells of ${r.groundCellSize[0].toFixed(0)} m`),e.setReadout(`domain`,`${M(h)} to ${M(g)} m elevation, 0 to 85° slope`),e.options.showRings&&await et(),{getCompiledGraphs:()=>[He,e.options.stitchLines?Ge:We,...$&&e.options.showRings?[$]:[]],setOption(t,n,r){switch(t){case`ramp`:tt(),Q=!0;break;case`source`:case`elevationInterval`:case`slopeInterval`:case`elevationWindow`:case`slopeWindow`:case`indexEvery`:it(),Q=!0;break;case`showRings`:case`ringTolerance`:case`splitTouchingRings`:r.showRings&&et(),Q=!0;break;case`stitchLines`:Q=!0;break;default:}e.requestLayers()},onAction(e){e===`time`&&st()},onThemeChange(){e.requestLayers()},encode(t,n){if(qe||(He.encode(t,{parameters:void 0}),qe=!0,Q=!0),!Q){ot.flush(t);return}at(t);let r=(e.options.stitchLines?Ge:We).encode(t,{parameters:void 0}),i=(e,n,r)=>t.copyBufferToBuffer({sourceBuffer:e,sourceOffset:0,destinationBuffer:X.buffer,destinationOffset:n*fe+r*4,size:4});i(Y,le,1),i(Ae,ue,1),i(we,de,0),Xe=!1,e.options.showRings&&$&&($.encode(t,{parameters:void 0}),Xe=!0),e.setReadout(`encode`,`${r.stats.cpuEncodeTimeMilliseconds.toFixed(2)} ms CPU`),Q=!1,ot.request(t)},getLayers(){let t=e.options,n=e.theme()===`dark`,r=[];if(t.showRelief&&r.push(new T({id:`contours-hillshade`,coordinateOrigin:l,gridSize:[a,o],bounds:u,rowOrigin:`south`,values:I,valueFormat:`float32`,colormap:`grayscale`,valueRange:[0,1],color:[255,255,255,n?190:175]})),t.showBands&&r.push(new ie({id:`contours-bands`,coordinateOrigin:l,gridSize:[1,1],bounds:u,triangles:be,triangleBands:xe,values:U,valueFormat:`uint32`,colormap:`category`,extent:V.buffer,drawCommands:X,drawCommandIndex:de,opacity:t.bandOpacity})),t.showRings&&r.push(new ae({id:`contours-ring-outlines`,coordinateOrigin:l,segments:Ne,polylineOffsets:Me,valueIndices:Fe,extent:Ie,values:Ve,valueFormat:`uint32`,colormap:`category`,palette:[pe,me],instanceCount:K,widthPixels:1.8})),t.showLines){let e=n?he.dark:he.light,i={coordinateOrigin:l,values:re,valueFormat:`uint32`,colormap:`category`,palette:[e.line,e.index],widthPixels:t.lineWidth};r.push(t.stitchLines?new ae({id:`contours-polylines`,...i,segments:Te,polylineOffsets:Ee,valueIndices:De,extent:ke,drawCommands:X,drawCommandIndex:ue}):new C({id:`contours-segments`,...i,segments:_e,valueIndices:ve,drawCommands:X,drawCommandIndex:le}))}return r},getTooltip(t){if(!t.coordinate)return null;let n=P(r,t.coordinate[0],t.coordinate[1]);if(n<0)return null;let i=r.elevation[n],a=e.options.elevationInterval,o=Math.floor(i/a)*a;return`Elevation ${M(i)} m\nBand ${M(o)} to ${M(o+a)} m`},destroy(){m=!0,ot.stop(),p.destroy()}}}export{_e as createContours};