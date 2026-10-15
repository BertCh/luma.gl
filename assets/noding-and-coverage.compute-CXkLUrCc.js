import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{E as t,h as n,i as r,t as i}from"./analysis-frame-Blu0tYdw.js";import{N as a,_ as o,d as s,h as c,j as l,l as u,n as d,t as f,v as p,w as m,y as h}from"./resources-CwvhEOBe.js";import{t as g}from"./gpu-scan-BuGhTLDQ.js";import{t as _}from"./gpu-sort-Bl3Dztdb.js";import{t as v}from"./draw-command-buffer-BeqRkhBa.js";import{l as y,n as b,s as x}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c as S,s as C}from"./wgsl-common-BNRVzgd8.js";import{t as w}from"./gpu-segment-ring-assembly-CMjctMio.js";import{t as T}from"./sorted-segment-offsets-HVRgs63A.js";import{i as E,n as D,t as O}from"./key-pair-grouping--fUW7Rco.js";import{t as k}from"./stable-lexicographic-index-sort-CcJLc7b1.js";import{r as A,t as j}from"./gpu-segment-intersection-BynYcZbe.js";import{n as M,t as N}from"./gpu-line-simplification-3HMpu60w.js";import{W as P,a as F}from"./index-Bjk18V2I.js";import{t as I}from"./graph-swap-41Ru7LVu.js";import{t as L}from"./summary-reader-CsgWxENf.js";import{t as R}from"./mode-kernels-Dxm2GWvI.js";import{i as z,n as B,r as V,t as H}from"./b3-layers-B2V1ljKW.js";import{t as U}from"./network-workgroup-counters-CFyh5ZJ-.js";import{_ as W,a as G,h as ee,i as K,l as te,r as q,s as ne}from"./b3-common-BbHtVprn.js";import{i as J,r as Y,t as re}from"./b3-city-data-BBHr728V.js";import{g as ie,h as ae}from"./b6-polygons-Bf2djBQx.js";var X=`GPULineSplit`,oe=2147483648,se=8;function ce(e){return u(e,se*4)}var le=4;function ue(e){return Math.max(1,Math.ceil(Math.log2(e+1)))}var de=class{id;props;spatialContext;vertexCount;lineCount;pieceCapacity;vertexCapacity;constructor(e){this.id=e.id??`line-split`,this.spatialContext=t(this.id,e.spatialContext),this.props=e;let{id:n}=this,{lines:r}=e,{pieces:i}=e.output;if(r.kind!==`lines`)throw Error(`${n} lines must be linestring geometry`);if(h(r.positions,[`float32x2`],`${n} lines.positions`),l(r.lineOffsets,`${n} lines.lineOffsets`),r.lineOffsets.length<2)throw Error(`${n} lines.lineOffsets requires at least one line and a terminal entry`);if(r.positions.length<1)throw Error(`${n} lines.positions must hold at least one vertex`);if(s(n,`intersectionCapacity`,e.intersectionCapacity),i.geometry.kind!==`lines`)throw Error(`${n} pieces.geometry must be linestring geometry`);if(this.vertexCount=r.positions.length,this.lineCount=r.lineOffsets.length-1,l(i.sourceIds,`${n} pieces.sourceIds`),l(i.geometry.lineOffsets,`${n} pieces.geometry.lineOffsets`),h(i.geometry.positions,[`float32x2`],`${n} pieces.geometry.positions`),this.pieceCapacity=i.sourceIds.length,this.vertexCapacity=i.geometry.positions.length,this.pieceCapacity<1||i.geometry.lineOffsets.length!==this.pieceCapacity+1)throw Error(`${n} pieces.geometry.lineOffsets length must be pieces.sourceIds.length + 1`);if(this.vertexCapacity<1)throw Error(`${n} pieces.geometry.positions must be non-empty`);if(i.vertexSourceEdges&&(l(i.vertexSourceEdges,`${n} pieces.vertexSourceEdges`),i.vertexSourceEdges.length!==this.vertexCapacity))throw Error(`${n} pieces.vertexSourceEdges length must equal pieces.geometry.positions length`);for(let[t,r]of[[`count`,i.status.count],[`vertexCount`,i.status.vertexCount],[`overflow`,i.status.overflow],[`candidateOverflow`,i.status.candidateOverflow],[`requiredCount`,i.status.requiredCount],[`requiredVertexCount`,i.status.requiredVertexCount],[`status.uncertainCount`,e.output.status?.uncertainCount]])if(r&&(l(r,`${n} ${t}`),r.length<1))throw Error(`${n} ${t} must contain one uint32 row`)}getCommandNodes(e){let{id:t,props:n,vertexCount:r,lineCount:i,pieceCapacity:a,vertexCapacity:s}=this,{lines:c}=n,{pieces:l}=n.output;o(t,e,[c.positions,c.lineOffsets,l.sourceIds,l.geometry.lineOffsets,l.geometry.positions,l.status.count,l.status.vertexCount,l.status.overflow,l.status.candidateOverflow,l.status.requiredCount,l.status.requiredVertexCount,l.vertexSourceEdges,n.output.status?.uncertainCount]),p(t,`lines.positions rows`,c.positions.length,A(e.device)),p(t,`lines + 4 x intersectionCapacity`,i+4*n.intersectionCapacity,ce(e.device));let u=[],d=n.intersectionCapacity,f=d*4,h=i+f,_=(n,r,i)=>m(e,`${t}-${n}`,r,i),v=_(`pair-left`,`uint32`,d),y=_(`pair-right`,`uint32`,d),b=_(`pair-count`,`uint32`,1),S=_(`pair-overflow`,`uint32`,1),w=n.output.status?.uncertainCount??_(`uncertain-count`,`uint32`,1),T=_(`non-finite-flag`,`uint32`,1);u.push(x(e,{id:`${t}-clear-non-finite`,operation:X,variant:`clear-non-finite`,bindings:[{name:`flag`,view:T,type:`u32`,access:`read_write`}],invocationCount:1,body:`flag[flagOffset] = 0u;`}),x(e,{id:`${t}-flag-non-finite`,operation:X,variant:`flag-non-finite`,bindings:[{name:`positions`,view:c.positions,type:`f32`,access:`read`},{name:`flag`,view:T,type:`atomic<u32>`,access:`read_write`}],invocationCount:r,declarations:`const VERTEX_COUNT: u32 = ${r}u;
${C}`,body:`if (index >= VERTEX_COUNT) { return; }
  if (!isFiniteBits(positions[positionsOffset + index * 2u]) || !isFiniteBits(positions[positionsOffset + index * 2u + 1u])) {
    atomicOr(&flag[flagOffset], 1u);
  }`}));let E=_(`kinds`,`uint32`,d),D=_(`points`,`float32x2`,d),O=_(`end-points`,`float32x2`,d);u.push(...new j({id:`${t}-intersection`,left:c,spatialSort:n.spatialSort,leafCapacity:n.leafCapacity,output:{pairs:{leftIds:v,rightIds:y,status:{count:b,overflow:S}},status:{uncertainCount:w},kinds:E,points:D,endPoints:O}}).getCommandNodes(e));let M=_(`event-segments`,`uint32`,f),N=_(`event-points`,`float32x2`,f),P=_(`event-indices`,`uint32`,f),F=_(`event-slots`,`uint32`,f),I=_(`event-phases`,`uint32`,f),L=_(`event-order-keys`,`uint32`,f),R=_(`event-table`,`uint32`,f*2);u.push(x(e,{id:`${t}-event-segments`,operation:X,variant:`event-segments`,bindings:[{name:`pairLeft`,view:v,type:`u32`,access:`read`},{name:`pairRight`,view:y,type:`u32`,access:`read`},{name:`kinds`,view:E,type:`u32`,access:`read`},{name:`pairCount`,view:b,type:`u32`,access:`read`},{name:`eventSegments`,view:M,type:`u32`,access:`read_write`}],invocationCount:f,body:`let pair = index / 4u;
  let which = index % 4u;
  let kind = kinds[kindsOffset + pair];
  // Kinds 1 to 4 intersect at a point or span; 5 (uncertain) and 0 produce no split.
  var valid = pair < pairCount[pairCountOffset] && kind >= 1u && kind <= 4u;
  if (which >= 2u && kind != 4u) { valid = false; }
  var segment = pairLeft[pairLeftOffset + pair];
  if ((which & 1u) == 1u) { segment = pairRight[pairRightOffset + pair]; }
  eventSegments[eventSegmentsOffset + index] = select(0xffffffffu, segment, valid);`}),x(e,{id:`${t}-event-points`,operation:X,variant:`event-points`,bindings:[{name:`eventSegments`,view:M,type:`u32`,access:`read`},{name:`points`,view:D,type:`f32`,access:`read`},{name:`endPoints`,view:O,type:`f32`,access:`read`},{name:`eventPoints`,view:N,type:`f32`,access:`read_write`},{name:`eventIndices`,view:P,type:`u32`,access:`read_write`}],invocationCount:f,body:`let pair = index / 4u;
  var point = vec2f(points[pointsOffset + pair * 2u], points[pointsOffset + pair * 2u + 1u]);
  if (index % 4u >= 2u) {
    point = vec2f(endPoints[endPointsOffset + pair * 2u], endPoints[endPointsOffset + pair * 2u + 1u]);
  }
  eventPoints[eventPointsOffset + index * 2u] = point.x;
  eventPoints[eventPointsOffset + index * 2u + 1u] = point.y;
  eventIndices[eventIndicesOffset + index] = index;`}));let z=`
const VERTEX_COUNT: u32 = ${r}u;
const LINE_COUNT: u32 = ${i}u;
// Largest line whose first vertex is at most vertex.
fn lineOfVertex(vertex: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle - 1u; }
  }
  return low;
}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`;u.push(x(e,{id:`${t}-event-slots`,operation:X,variant:`event-slots`,bindings:[{name:`eventSegments`,view:M,type:`u32`,access:`read`},{name:`eventPoints`,view:N,type:`f32`,access:`read`},{name:`positions`,view:c.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},{name:`eventSlots`,view:F,type:`u32`,access:`read_write`},{name:`eventPhases`,view:I,type:`u32`,access:`read_write`},{name:`eventOrderKeys`,view:L,type:`u32`,access:`read_write`},{name:`eventTable`,view:R,type:`u32`,access:`read_write`}],invocationCount:f,declarations:z,body:`let segment = eventSegments[eventSegmentsOffset + index];
  var slot = VERTEX_COUNT;
  var phase = 0u;
  var key = 0.0;
  if (segment != 0xffffffffu) {
    let line = lineOfVertex(segment);
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = lineOffsets[lineOffsetsOffset + line + 1u];
    let a = vertexAt(segment);
    let b = vertexAt(segment + 1u);
    let p = vec2f(eventPoints[eventPointsOffset + index * 2u], eventPoints[eventPointsOffset + index * 2u + 1u]);
    if (p.x == a.x && p.y == a.y) {
      if (segment > lineStart) { slot = segment; }
    } else if (p.x == b.x && p.y == b.y) {
      if (segment + 1u < lineEnd - 1u) { slot = segment + 1u; }
    } else {
      slot = segment;
      phase = 1u;
      let d = b - a;
      if (abs(d.x) >= abs(d.y)) { key = p.x * select(-1.0, 1.0, d.x > 0.0); }
      else { key = p.y * select(-1.0, 1.0, d.y > 0.0); }
    }
  }
  eventSlots[eventSlotsOffset + index] = slot;
  eventPhases[eventPhasesOffset + index] = phase;
  let keyBits = bitcast<u32>(key);
  eventOrderKeys[eventOrderKeysOffset + index] = select(keyBits ^ 0xffffffffu, keyBits ^ 0x80000000u, (keyBits & 0x80000000u) == 0u);
  eventTable[eventTableOffset + index * 2u] = phase;
  eventTable[eventTableOffset + index * 2u + 1u] = bitcast<u32>(key);`}));let B=k(e,{id:`${t}-event-order`,operation:X,indices:P,keys:[{view:F,keyBits:ue(r)},{view:I,keyBits:1},{view:L}]});u.push(...B.nodes);let V=B.sortedPrimaryKeys,H=B.sortedIndices,U=`
const VERTEX_COUNT: u32 = ${r}u;
fn eventPhase(event: u32) -> u32 { return eventTable[eventTableOffset + event * 2u]; }
fn eventKey(event: u32) -> f32 { return bitcast<f32>(eventTable[eventTableOffset + event * 2u + 1u]); }
fn eventSame(first: u32, second: u32) -> bool {
  return eventPhase(first) == eventPhase(second) && eventKey(first) == eventKey(second);
}`,W=_(`unique-flags`,`uint32`,f),G=_(`unique-ranks`,`uint32`,f),ee=_(`split-table`,`uint32`,f*le);u.push(x(e,{id:`${t}-unique-flags`,operation:X,variant:`unique-flags`,bindings:[{name:`sortedSlots`,view:V,type:`u32`,access:`read`},{name:`sortedEvents`,view:H,type:`u32`,access:`read`},{name:`eventTable`,view:R,type:`u32`,access:`read`},{name:`uniqueFlags`,view:W,type:`u32`,access:`read_write`}],invocationCount:f,declarations:U,body:`let slot = sortedSlots[sortedSlotsOffset + index];
  var flag = 0u;
  if (slot < VERTEX_COUNT) {
    if (index == 0u || sortedSlots[sortedSlotsOffset + index - 1u] != slot) { flag = 1u; }
    else if (!eventSame(sortedEvents[sortedEventsOffset + index], sortedEvents[sortedEventsOffset + index - 1u])) { flag = 1u; }
  }
  uniqueFlags[uniqueFlagsOffset + index] = flag;`}),...new g({id:`${t}-unique-scan`,input:W,output:G,mode:`exclusive`}).getCommandNodes(e),x(e,{id:`${t}-compact-splits`,operation:X,variant:`compact-splits`,bindings:[{name:`sortedSlots`,view:V,type:`u32`,access:`read`},{name:`sortedEvents`,view:H,type:`u32`,access:`read`},{name:`eventTable`,view:R,type:`u32`,access:`read`},{name:`eventPoints`,view:N,type:`u32`,access:`read`},{name:`uniqueFlags`,view:W,type:`u32`,access:`read`},{name:`uniqueRanks`,view:G,type:`u32`,access:`read`},{name:`splitTable`,view:ee,type:`u32`,access:`read_write`}],invocationCount:f,declarations:U,body:`if (uniqueFlags[uniqueFlagsOffset + index] == 0u) { return; }
  let rank = uniqueRanks[uniqueRanksOffset + index];
  let event = sortedEvents[sortedEventsOffset + index];
  let row = splitTableOffset + rank * ${le}u;
  splitTable[row] = sortedSlots[sortedSlotsOffset + index];
  splitTable[row + 1u] = eventPhase(event);
  splitTable[row + 2u] = eventPoints[eventPointsOffset + event * 2u];
  splitTable[row + 3u] = eventPoints[eventPointsOffset + event * 2u + 1u];`}));let K=_(`split-bases`,`uint32`,i),te=_(`piece-counts`,`uint32`,i),q=_(`piece-starts`,`uint32`,i);u.push(x(e,{id:`${t}-line-splits`,operation:X,variant:`line-splits`,bindings:[{name:`lineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},{name:`sortedSlots`,view:V,type:`u32`,access:`read`},{name:`uniqueFlags`,view:W,type:`u32`,access:`read`},{name:`uniqueRanks`,view:G,type:`u32`,access:`read`},{name:`splitBases`,view:K,type:`u32`,access:`read_write`},{name:`pieceCounts`,view:te,type:`u32`,access:`read_write`}],invocationCount:i,declarations:`const EVENT_COUNT: u32 = ${f}u;
fn lowerBound(value: u32) -> u32 {
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedSlots[sortedSlotsOffset + middle] < value) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}
fn uniqueBefore(position: u32) -> u32 {
  if (position < EVENT_COUNT) { return uniqueRanks[uniqueRanksOffset + position]; }
  return uniqueRanks[uniqueRanksOffset + EVENT_COUNT - 1u] + uniqueFlags[uniqueFlagsOffset + EVENT_COUNT - 1u];
}`,body:`let lineStart = lineOffsets[lineOffsetsOffset + index];
  let lineEnd = lineOffsets[lineOffsetsOffset + index + 1u];
  let first = uniqueBefore(lowerBound(lineStart));
  let last = uniqueBefore(lowerBound(lineEnd));
  splitBases[splitBasesOffset + index] = first;
  pieceCounts[pieceCountsOffset + index] = select(0u, last - first + 1u, lineEnd >= lineStart + 2u);`}),...new g({id:`${t}-piece-scan`,input:te,output:q,mode:`exclusive`}).getCommandNodes(e));let ne=_(`piece-table`,`uint32`,h*se),J=_(`piece-vertex-counts`,`uint32`,h),Y=_(`piece-vertex-starts`,`uint32`,h);u.push(x(e,{id:`${t}-pieces`,operation:X,variant:`pieces`,bindings:[{name:`lineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},{name:`pieceStarts`,view:q,type:`u32`,access:`read`},{name:`pieceCounts`,view:te,type:`u32`,access:`read`},{name:`splitBases`,view:K,type:`u32`,access:`read`},{name:`splitTable`,view:ee,type:`u32`,access:`read`},{name:`pieceTable`,view:ne,type:`u32`,access:`read_write`},{name:`pieceVertexCounts`,view:J,type:`u32`,access:`read_write`}],invocationCount:h,declarations:`const LINE_COUNT: u32 = ${i}u;`,body:`let total = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
  if (index >= total) { pieceVertexCounts[pieceVertexCountsOffset + index] = 0u; return; }
  // Last line whose first piece is at most index.
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (pieceStarts[pieceStartsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; }
  }
  let line = low;
  let k = index - pieceStarts[pieceStartsOffset + line];
  let lastPiece = pieceCounts[pieceCountsOffset + line] - 1u;
  let base = splitBases[splitBasesOffset + line];
  var firstVertex = lineOffsets[lineOffsetsOffset + line];
  var lastVertex = lineOffsets[lineOffsetsOffset + line + 1u] - 1u;
  var prefix = 0u;
  var suffix = 0u;
  var prefixPoint = vec2u(0u);
  var suffixPoint = vec2u(0u);
  if (k > 0u) {
    let row = splitTableOffset + (base + k - 1u) * ${le}u;
    let slot = splitTable[row];
    let phase = splitTable[row + 1u];
    firstVertex = slot + phase;
    prefix = phase;
    prefixPoint = vec2u(splitTable[row + 2u], splitTable[row + 3u]);
  }
  if (k < lastPiece) {
    let row = splitTableOffset + (base + k) * ${le}u;
    lastVertex = splitTable[row];
    suffix = splitTable[row + 1u];
    suffixPoint = vec2u(splitTable[row + 2u], splitTable[row + 3u]);
  }
  let row = pieceTableOffset + index * ${se}u;
  pieceTable[row] = line;
  pieceTable[row + 1u] = firstVertex;
  pieceTable[row + 2u] = lastVertex;
  pieceTable[row + 3u] = prefix | (suffix << 1u);
  pieceTable[row + 4u] = prefixPoint.x;
  pieceTable[row + 5u] = prefixPoint.y;
  pieceTable[row + 6u] = suffixPoint.x;
  pieceTable[row + 7u] = suffixPoint.y;
  pieceVertexCounts[pieceVertexCountsOffset + index] = lastVertex - firstVertex + 1u + prefix + suffix;`}),...new g({id:`${t}-vertex-scan`,input:J,output:Y,mode:`exclusive`}).getCommandNodes(e));let{vertexSourceEdges:re}=l,ie=re?_(`self-contacts`,`uint32`,r):null;ie&&u.push(x(e,{id:`${t}-self-contacts-clear`,operation:X,variant:`self-contacts-clear`,bindings:[{name:`selfContacts`,view:ie,type:`u32`,access:`read_write`}],invocationCount:r,body:`selfContacts[selfContactsOffset + index] = 0u;`}),x(e,{id:`${t}-self-contacts`,operation:X,variant:`self-contacts`,bindings:[{name:`pairLeft`,view:v,type:`u32`,access:`read`},{name:`pairRight`,view:y,type:`u32`,access:`read`},{name:`pairCount`,view:b,type:`u32`,access:`read`},{name:`lineOffsets`,view:c.lineOffsets,type:`u32`,access:`read`},{name:`selfContacts`,view:ie,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,declarations:`const LINE_COUNT: u32 = ${i}u;
// Largest line whose first vertex is at most vertex.
fn lineOfVertex(vertex: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle - 1u; }
  }
  return low;
}`,body:`if (index >= pairCount[pairCountOffset]) { return; }
  let left = pairLeft[pairLeftOffset + index];
  let right = pairRight[pairRightOffset + index];
  if (lineOfVertex(left) != lineOfVertex(right)) { return; }
  atomicStore(&selfContacts[selfContactsOffset + left], 1u);
  atomicStore(&selfContacts[selfContactsOffset + right], 1u);`}));let ae=_(`state`,`uint32`,8);u.push(x(e,{id:`${t}-finalize`,operation:X,variant:`finalize`,bindings:[{name:`pieceStarts`,view:q,type:`u32`,access:`read`},{name:`pieceCounts`,view:te,type:`u32`,access:`read`},{name:`vertexStarts`,view:Y,type:`u32`,access:`read`},{name:`vertexCounts`,view:J,type:`u32`,access:`read`},{name:`pairOverflow`,view:S,type:`u32`,access:`read`},{name:`state`,view:ae,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const LINE_COUNT: u32 = ${i}u;
const PIECE_CAPACITY: u32 = ${a}u;
const VERTEX_CAPACITY: u32 = ${s}u;
fn pieceEnd(piece: u32) -> u32 { return vertexStarts[vertexStartsOffset + piece] + vertexCounts[vertexCountsOffset + piece]; }`,body:`let totalPieces = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
  var totalVertices = 0u;
  if (totalPieces > 0u) { totalVertices = pieceEnd(totalPieces - 1u); }
  var low = 0u;
  var high = min(totalPieces, PIECE_CAPACITY);
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (pieceEnd(middle - 1u) <= VERTEX_CAPACITY) { low = middle; } else { high = middle - 1u; }
  }
  var written = 0u;
  if (low > 0u) { written = pieceEnd(low - 1u); }
  state[stateOffset] = low;
  state[stateOffset + 1u] = written;
  state[stateOffset + 2u] = totalPieces;
  state[stateOffset + 3u] = totalVertices;
  state[stateOffset + 4u] = select(0u, 1u, low < totalPieces);
  state[stateOffset + 5u] = pairOverflow[pairOverflowOffset];`}),x(e,{id:`${t}-write-pieces`,operation:X,variant:`write-pieces`,bindings:[{name:`state`,view:ae,type:`u32`,access:`read`},{name:`pieceTable`,view:ne,type:`u32`,access:`read`},{name:`vertexStarts`,view:Y,type:`u32`,access:`read`},{name:`lineIds`,view:l.sourceIds,type:`u32`,access:`read_write`},{name:`offsets`,view:l.geometry.lineOffsets,type:`u32`,access:`read_write`}],invocationCount:a+1,declarations:`const PIECE_CAPACITY: u32 = ${a}u;`,body:`let kept = state[stateOffset];
  if (index < kept) {
    lineIds[lineIdsOffset + index] = pieceTable[pieceTableOffset + index * ${se}u];
    offsets[offsetsOffset + index] = vertexStarts[vertexStartsOffset + index];
  } else {
    if (index < PIECE_CAPACITY) { lineIds[lineIdsOffset + index] = 0xffffffffu; }
    offsets[offsetsOffset + index] = state[stateOffset + 1u];
  }`}),x(e,{id:`${t}-write-vertices`,operation:X,variant:ie?`write-vertices-with-edges`:`write-vertices`,bindings:[{name:`state`,view:ae,type:`u32`,access:`read`},{name:`pieceTable`,view:ne,type:`u32`,access:`read`},{name:`vertexStarts`,view:Y,type:`u32`,access:`read`},{name:`positions`,view:c.positions,type:`f32`,access:`read`},{name:`outPositions`,view:l.geometry.positions,type:`f32`,access:`read_write`},...ie&&re?[{name:`selfContacts`,view:ie,type:`u32`,access:`read`},{name:`sourceEdges`,view:re,type:`u32`,access:`read_write`}]:[]],invocationCount:s,body:`if (index >= state[stateOffset + 1u]) { return; }
  // Last kept piece whose first vertex is at most index.
  var low = 0u;
  var high = state[stateOffset] - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (vertexStarts[vertexStartsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; }
  }
  let row = pieceTableOffset + low * ${se}u;
  let firstVertex = pieceTable[row + 1u];
  let runLength = pieceTable[row + 2u] - firstVertex + 1u;
  let flags = pieceTable[row + 3u];
  var local = index - vertexStarts[vertexStartsOffset + low];
  ${ie?`// A prefix point lies inside the edge before firstVertex; input vertex v starts edge v.
  let edge = firstVertex - (flags & 1u) + local;
  var contact = state[stateOffset + 5u] != 0u;
  if (edge < ${r}u && selfContacts[selfContactsOffset + edge] != 0u) { contact = true; }
  sourceEdges[sourceEdgesOffset + index] = edge | select(0u, ${oe}u, contact);`:``}
  var point = vec2f(0.0);
  var resolved = false;
  if ((flags & 1u) != 0u) {
    if (local == 0u) {
      point = vec2f(bitcast<f32>(pieceTable[row + 4u]), bitcast<f32>(pieceTable[row + 5u]));
      resolved = true;
    } else { local = local - 1u; }
  }
  if (!resolved) {
    if (local < runLength) {
      let source = positionsOffset + (firstVertex + local) * 2u;
      point = vec2f(positions[source], positions[source + 1u]);
    } else {
      point = vec2f(bitcast<f32>(pieceTable[row + 6u]), bitcast<f32>(pieceTable[row + 7u]));
    }
  }
  outPositions[outPositionsOffset + index * 2u] = point.x;
  outPositions[outPositionsOffset + index * 2u + 1u] = point.y;`})),u.push(x(e,{id:`${t}-incomplete`,operation:X,variant:`incomplete`,bindings:[{name:`uncertainPairs`,view:w,type:`u32`,access:`read`},{name:`incomplete`,view:T,type:`u32`,access:`read_write`}],invocationCount:1,body:`if (uncertainPairs[uncertainPairsOffset] != 0u) { incomplete[incompleteOffset] = 1u; }`}));let de=[{name:`state`,view:ae,type:`u32`,access:`read`},{name:`incomplete`,view:T,type:`u32`,access:`read`}],fe=[[`count`,l.status.count,0],[`vertexCount`,l.status.vertexCount,1],[`requiredCount`,l.status.requiredCount,2],[`requiredVertexCount`,l.status.requiredVertexCount,3],[`overflow`,l.status.overflow,4],[`candidateOverflow`,l.status.candidateOverflow,5]];for(let[e,t]of fe)t&&de.push({name:e,view:t,type:`u32`,access:`read_write`});return u.push(x(e,{id:`${t}-scalars`,operation:X,variant:`scalars`,bindings:de,invocationCount:1,body:fe.filter(([,e])=>e).map(([e,,t])=>e===`overflow`?`overflow[overflowOffset] = select(0u, 1u, state[stateOffset + ${t}u] != 0u || incomplete[incompleteOffset] != 0u);`:`${e}[${e}Offset] = state[stateOffset + ${t}u];`).join(`
  `)})),u}};function fe(e,t,n,r,i){p(e,`lines.positions rows`,n,A(t)),p(e,`lines + 4 x intersectionCapacity`,r+4*i,ce(t))}var pe=`GPULineMerge`,me=class{id;props;spatialContext;lineCount;constructor(e){this.id=e.id??`line-merge`,this.spatialContext=t(this.id,e.spatialContext),this.props=e;let{id:n}=this,{output:r}=e;h(e.positions,[`float32x2`],`${n} positions`),h(r.positions,[`float32x2`],`${n} output.positions`);for(let[t,i]of[[`lineOffsets`,e.lineOffsets],[`output.chainOffsets`,r.chainOffsets],[`output.status.count`,r.status.count],[`output.lineChains`,r.lineChains],[`output.lineOrders`,r.lineOrders],[`output.lineReversed`,r.lineReversed]])i&&h(i,[`uint32`],`${n} ${t}`);if(e.lineOffsets.length<2)throw Error(`${n} lineOffsets needs at least one line`);if(this.lineCount=e.lineOffsets.length-1,r.chainOffsets.length<this.lineCount+1)throw Error(`${n} output.chainOffsets must contain lineCount + 1 rows`);if(r.positions.length<e.positions.length)throw Error(`${n} output.positions must hold at least the input vertex count`);if(r.status.count.length!==1)throw Error(`${n} output.status.count must contain exactly one row`);for(let[e,t]of[[`lineChains`,r.lineChains],[`lineOrders`,r.lineOrders],[`lineReversed`,r.lineReversed]])if(t&&t.length!==this.lineCount)throw Error(`${n} output.${e} length must equal the line count`);let i=ge(e);c(n,i,he(e));let a=i.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(a).size!==a.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(e){let{id:t,props:n,lineCount:r}=this,{output:i}=n;o(t,e,[...he(n),...ge(n)]);let a=r*2,s=n.directed===!0,c=(n,r,i)=>m(e,`${t}-${n}`,r,Math.max(i,1)),l=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),u=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),d=(n,i,a,o,c=``,l=``)=>x(e,{id:`${t}-${n}${l}`,operation:pe,variant:n,bindings:i,invocationCount:a,declarations:`const LINE_COUNT: u32 = ${r}u;
const NONE: u32 = 0xffffffffu;
const INVALID_KEY: u32 = 0xffffffffu;
const DIRECTED: bool = ${s};
${c}`,body:o}),f=c(`key-y`,`uint32`,a),p=c(`key-x`,`uint32`,a),h=c(`endpoint-ids`,`uint32`,a),v=c(`sorted-y`,`uint32`,a),y=c(`order-y`,`uint32`,a),S=c(`second-keys`,`uint32`,a),w=c(`sorted-x`,`uint32`,a),T=c(`order`,`uint32`,a),E=c(`partner`,`uint32`,a),D=c(`leader-exit`,`uint32`,r),O=c(`leader-vertices`,`uint32`,r),k=c(`flags`,`uint32`,r),A=c(`flag-scan`,`uint32`,r),j=c(`vertex-scan`,`uint32`,r),M=[];M.push(d(`keys`,[l(`positions`,n.positions,`f32`),l(`lineOffsets`,n.lineOffsets),u(`keyY`,f),u(`keyX`,p),u(`endpointIds`,h)],a,`let line = index / 2u;
  let first = lineOffsets[lineOffsetsOffset + line];
  let last = lineOffsets[lineOffsetsOffset + line + 1u];
  endpointIds[endpointIdsOffset + index] = index;
  var x = 0.0;
  var y = 0.0;
  var valid = last >= first + 2u;
  if (valid) {
    let vertex = select(first, last - 1u, (index & 1u) == 1u);
    x = positions[positionsOffset + vertex * 2u];
    y = positions[positionsOffset + vertex * 2u + 1u];
    valid = isFiniteBits(x) && isFiniteBits(y);
  }
  if (valid) {
    x = select(x, 0.0, x == 0.0);
    y = select(y, 0.0, y == 0.0);
    keyX[keyXOffset + index] = bitcast<u32>(x);
    keyY[keyYOffset + index] = bitcast<u32>(y);
  } else {
    keyX[keyXOffset + index] = INVALID_KEY;
    keyY[keyYOffset + index] = INVALID_KEY;
  }`,C),...new _({id:`${t}-sort-y`,keys:f,values:h,outputKeys:v,outputValues:y}).getCommandNodes(e),d(`gather-x`,[l(`keyX`,p),l(`orderY`,y),u(`secondKeys`,S)],a,`secondKeys[secondKeysOffset + index] = keyX[keyXOffset + orderY[orderYOffset + index]];`),...new _({id:`${t}-sort-x`,keys:S,values:y,outputKeys:w,outputValues:T}).getCommandNodes(e),d(`partners`,[l(`sortedX`,w),l(`order`,T),l(`keyY`,f),u(`partner`,E)],a,`let end = order[orderOffset + index];
  var result = NONE;
  if (sortedX[sortedXOffset + index] != INVALID_KEY) {
    let isLeftSame = index > 0u && isSame(index - 1u, index);
    let isRightSame = index + 1u < ENDPOINT_COUNT && isSame(index, index + 1u);
    if (isLeftSame && !isRightSame && !(index > 1u && isSame(index - 2u, index - 1u))) {
      result = order[orderOffset + index - 1u];
    } else if (isRightSame && !isLeftSame &&
        !(index + 2u < ENDPOINT_COUNT && isSame(index + 1u, index + 2u))) {
      result = order[orderOffset + index + 1u];
    }
    if (result != NONE && (result >> 1u) == (end >> 1u)) {
      result = NONE;
    }
    // Directed joins pair a line end (odd id) with a line start (even id).
    if (DIRECTED && result != NONE && ((result ^ end) & 1u) == 0u) {
      result = NONE;
    }
  }
  partner[partnerOffset + end] = result;`,`const ENDPOINT_COUNT: u32 = ${a}u;
fn isSame(a: u32, b: u32) -> bool {
  return sortedX[sortedXOffset + a] == sortedX[sortedXOffset + b] &&
    keyY[keyYOffset + order[orderOffset + a]] == keyY[keyYOffset + order[orderOffset + b]];
}`));let N=Math.max(1,Math.ceil(Math.log2(Math.max(r,2)))),P=!!i.lineOrders,F=`
fn vertexCount(line: u32) -> u32 {
  return lineOffsets[lineOffsetsOffset + line + 1u] - lineOffsets[lineOffsetsOffset + line];
}`,I=e=>[c(`${e}-a`,`uint32`,a),c(`${e}-b`,`uint32`,a)],L=I(`min-predecessor`),R=I(`min-key`),z=I(`rank-predecessor`),B=I(`rank-head`),V=I(`rank-sum`),H=I(`rank-count`),U=c(`leader-bits`,`uint32`,a),W=e=>`(${e} >> 1u) * 2u + select(1u, 0u, (${e} & 1u) == 1u)`;M.push(d(`min-init`,[l(`partner`,E),u(`predecessor`,L[0]),u(`minKey`,R[0])],a,`predecessor[predecessorOffset + index] = partner[partnerOffset + (index ^ 1u)];
  minKey[minKeyOffset + index] = ${W(`index`)};`));for(let e=0;e<N;e++){let t=e%2,n=1-t;M.push(d(`min-round`,[l(`predecessorIn`,L[t]),l(`keyIn`,R[t]),u(`predecessorOut`,L[n]),u(`keyOut`,R[n])],a,`let previous = predecessorIn[predecessorInOffset + index];
  var key = keyIn[keyInOffset + index];
  var nextPrevious = previous;
  if (previous != NONE) {
    key = min(key, keyIn[keyInOffset + previous]);
    nextPrevious = predecessorIn[predecessorInOffset + previous];
  }
  predecessorOut[predecessorOutOffset + index] = nextPrevious;
  keyOut[keyOutOffset + index] = key;`,``,`-${e}`))}let G=N%2;M.push(d(`rank-init`,[l(`partner`,E),l(`lineOffsets`,n.lineOffsets),l(`cyclePredecessor`,L[G]),l(`cycleKey`,R[G]),u(`predecessor`,z[0]),u(`head`,B[0]),u(`sum`,V[0]),u(`leaderBits`,U)],a,`let isCycle = cyclePredecessor[cyclePredecessorOffset + index] != NONE;
  let minimumKey = cycleKey[cycleKeyOffset + index];
  let isCanonicalCycle = isCycle && (minimumKey & 1u) == 0u;
  let vertices = vertexCount(index >> 1u);
  let isActive = vertices >= 2u && (!isCycle || isCanonicalCycle);
  let isLeader = isActive && isCycle && minimumKey == ${W(`index`)};
  var previous = NONE;
  if (isActive && !isLeader) {
    previous = partner[partnerOffset + (index ^ 1u)];
  }
  predecessor[predecessorOffset + index] = previous;
  head[headOffset + index] = index;
  sum[sumOffset + index] = select(0u, vertices - 1u, isActive);
  leaderBits[leaderBitsOffset + index] = select(0u, select(1u, 3u, isLeader), isActive);`,F)),P&&M.push(b(e,{id:`${t}-rank-count-init`,operation:pe,view:H[0],type:`u32`,value:`1u`}));for(let e=0;e<N;e++){let t=e%2,n=1-t;M.push(d(`rank-round`,[l(`predecessorIn`,z[t]),l(`headIn`,B[t]),l(`sumIn`,V[t]),u(`predecessorOut`,z[n]),u(`headOut`,B[n]),u(`sumOut`,V[n])],a,`let previous = predecessorIn[predecessorInOffset + index];
  var head = headIn[headInOffset + index];
  var sum = sumIn[sumInOffset + index];
  var nextPrevious = previous;
  if (previous != NONE) {
    head = headIn[headInOffset + previous];
    sum += sumIn[sumInOffset + previous];
    nextPrevious = predecessorIn[predecessorInOffset + previous];
  }
  predecessorOut[predecessorOutOffset + index] = nextPrevious;
  headOut[headOutOffset + index] = head;
  sumOut[sumOutOffset + index] = sum;`,``,`-${e}`)),P&&M.push(d(`rank-count-round`,[l(`predecessorIn`,z[t]),l(`countIn`,H[t]),u(`countOut`,H[n])],a,`let previous = predecessorIn[predecessorInOffset + index];
  var count = countIn[countInOffset + index];
  if (previous != NONE) {
    count += countIn[countInOffset + previous];
  }
  countOut[countOutOffset + index] = count;`,``,`-${e}`))}let ee=N%2,K=B[ee],te=V[ee];if(M.push(b(e,{id:`${t}-clear-leaders`,operation:pe,view:O,type:`u32`,value:`0u`}),d(`tails`,[l(`partner`,E),l(`leaderBits`,U),l(`head`,K),l(`sum`,te),u(`leaderExit`,D),u(`leaderVertices`,O)],a,`if (leaderBits[leaderBitsOffset + index] == 0u) {
    return;
  }
  let next = partner[partnerOffset + index];
  if (next != NONE && (leaderBits[leaderBitsOffset + (next ^ 1u)] & 2u) == 0u) {
    return;
  }
  let first = head[headOffset + index];
  var isCanonical = (leaderBits[leaderBitsOffset + first] & 2u) != 0u;
  if (!isCanonical) {
    isCanonical = select((first ^ 1u) < index, (first & 1u) == 1u, DIRECTED);
  }
  if (isCanonical) {
    leaderVertices[leaderVerticesOffset + (first >> 1u)] = sum[sumOffset + index] + 1u;
    leaderExit[leaderExitOffset + (first >> 1u)] = first;
  }`),d(`flags`,[l(`leaderVertices`,O),u(`flags`,k)],r,`flags[flagsOffset + index] = select(0u, 1u, leaderVertices[leaderVerticesOffset + index] > 0u);`),...new g({id:`${t}-chain-scan`,input:k,output:A,mode:`inclusive`}).getCommandNodes(e),...new g({id:`${t}-vertex-scan`,input:O,output:j,mode:`inclusive`}).getCommandNodes(e),d(`offsets`,[l(`leaderVertices`,O),l(`flagScan`,A),l(`vertexScan`,j),u(`chainOffsets`,i.chainOffsets),u(`count`,i.status.count)],r+1,`let chainCount = flagScan[flagScanOffset + LINE_COUNT - 1u];
  let totalVertices = vertexScan[vertexScanOffset + LINE_COUNT - 1u];
  if (index == 0u) {
    count[countOffset] = chainCount;
  }
  if (index >= chainCount) {
    chainOffsets[chainOffsetsOffset + index] = totalVertices;
  }
  if (index < LINE_COUNT && leaderVertices[leaderVerticesOffset + index] > 0u) {
    let chain = flagScan[flagScanOffset + index] - 1u;
    chainOffsets[chainOffsetsOffset + chain] =
      vertexScan[vertexScanOffset + index] - leaderVertices[leaderVerticesOffset + index];
  }`),d(`positions`,[l(`positions`,n.positions,`f32`),l(`lineOffsets`,n.lineOffsets),l(`head`,K),l(`sum`,te),l(`leaderExit`,D),l(`leaderVertices`,O),l(`vertexScan`,j),u(`outPositions`,i.positions,`f32`)],a,`let first = head[headOffset + index];
  let headLine = first >> 1u;
  if (leaderVertices[leaderVerticesOffset + headLine] == 0u ||
      leaderExit[leaderExitOffset + headLine] != first) {
    return;
  }
  let line = index >> 1u;
  let firstVertex = lineOffsets[lineOffsetsOffset + line];
  let count = vertexCount(line);
  let isForward = (index & 1u) == 1u;
  let base = vertexScan[vertexScanOffset + headLine] - leaderVertices[leaderVerticesOffset + headLine];
  let cursor = base + sum[sumOffset + index] - (count - 1u);
  for (var k = select(1u, 0u, index == first); k < count; k++) {
    let vertex = select(firstVertex + count - 1u - k, firstVertex + k, isForward);
    let slot = cursor + k;
    outPositions[outPositionsOffset + slot * 2u] = positions[positionsOffset + vertex * 2u];
    outPositions[outPositionsOffset + slot * 2u + 1u] = positions[positionsOffset + vertex * 2u + 1u];
  }`,F)),i.lineChains||i.lineOrders||i.lineReversed){let n=[l(`head`,K),l(`leaderExit`,D),l(`leaderVertices`,O),l(`flagScan`,A)],r=`let first = head[headOffset + index];
  let headLine = first >> 1u;
  if (leaderVertices[leaderVerticesOffset + headLine] == 0u ||
      leaderExit[leaderExitOffset + headLine] != first) {
    return;
  }
  let line = index >> 1u;`;i.lineChains&&(n.push(u(`lineChains`,i.lineChains)),r+=`
  lineChains[lineChainsOffset + line] = flagScan[flagScanOffset + headLine] - 1u;`,M.push(b(e,{id:`${t}-clear-line-chains`,operation:pe,view:i.lineChains,type:`u32`,value:`0xffffffffu`}))),i.lineOrders&&(n.push(l(`lineCountPrefix`,H[ee]),u(`lineOrders`,i.lineOrders)),r+=`
  lineOrders[lineOrdersOffset + line] = lineCountPrefix[lineCountPrefixOffset + index] - 1u;`),i.lineReversed&&(n.push(u(`lineReversed`,i.lineReversed)),r+=`
  lineReversed[lineReversedOffset + line] = select(1u, 0u, (index & 1u) == 1u);`),M.push(d(`line-info`,n,a,r))}return M}};function he(e){return[e.positions,e.lineOffsets]}function ge(e){let{output:t}=e;return[t.chainOffsets,t.positions,t.status.count,t.lineChains,t.lineOrders,t.lineReversed]}var _e=Object.freeze({...n,measure:`distance`}),Z=`GPUCoverageSimplification`,Q=`${O}u`,ve=`0x80000000u`,ye=4,be=4,xe=be,Se=class{id;props;spatialContext;constructor(e){let n=e.id??`coverage-simplification`;if(this.id=n,this.spatialContext=t(this.id,e.spatialContext,_e),this.props=e,h(e.positions,[`float32x2`],`${n} positions`),e.positions.length<1)throw Error(`${n} positions must hold at least one vertex`);l(e.ringOffsets,`${n} ringOffsets`),l(e.polygonOffsets,`${n} polygonOffsets`),l(e.featureOffsets,`${n} featureOffsets`);for(let[t,r]of[[`ringOffsets`,e.ringOffsets],[`polygonOffsets`,e.polygonOffsets],[`featureOffsets`,e.featureOffsets]])if(r.length<2)throw Error(`${n} ${t} must hold at least two entries`);if(e.topologyIterationCount!==void 0&&s(n,`topologyIterationCount`,e.topologyIterationCount,0,16),e.topologyPairCapacity!==void 0&&s(n,`topologyPairCapacity`,e.topologyPairCapacity),e.simplifyBoundary!==void 0&&typeof e.simplifyBoundary!=`boolean`)throw Error(`${n} simplifyBoundary must be a boolean`);if(e.snapTolerance!==void 0&&(!Number.isFinite(e.snapTolerance)||e.snapTolerance<0))throw Error(`${n} snapTolerance must be a finite number >= 0`);if(h(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<4)throw Error(`${n} parameters must hold 4 float32 values`);let{output:r}=e;if(h(r.positions,[`float32x2`],`${n} output.positions`),r.positions.length<1)throw Error(`${n} output.positions must hold at least one row`);if(l(r.ringOffsets,`${n} output.ringOffsets`),r.ringOffsets.length!==e.ringOffsets.length)throw Error(`${n} output.ringOffsets length must equal ringOffsets length`);let{overflow:i,requiredCount:a}=r.status;if(l(i,`${n} output.status.overflow`),i.length<1)throw Error(`${n} output.status.overflow must hold one uint32`);if(r.outputMask&&(l(r.outputMask,`${n} output.outputMask`),r.outputMask.length!==e.positions.length))throw Error(`${n} output.outputMask length must equal positions length`);if(r.topologyStatistics&&(l(r.topologyStatistics,`${n} output.topologyStatistics`),r.topologyStatistics.length<be))throw Error(`${n} output.topologyStatistics must hold ${be} uint32 rows`);if(a&&(l(a,`${n} output.status.requiredCount`),a.length<1))throw Error(`${n} output.status.requiredCount must hold one uint32`);if(r.status.converged&&(l(r.status.converged,`${n} output.status.converged`),r.status.converged.length<1))throw Error(`${n} output.status.converged must hold one uint32`);c(n,[r.positions,r.ringOffsets,r.outputMask,i,a,r.topologyStatistics,r.status.converged],[e.positions,e.ringOffsets,e.polygonOffsets,e.featureOffsets,e.parameters])}getCommandNodes(e){let{id:t,props:n}=this,{positions:r,ringOffsets:i,polygonOffsets:a,featureOffsets:s,output:c}=n;o(t,e,[r,i,a,s,n.parameters,c.status.converged,c.positions,c.ringOffsets,c.outputMask,c.status.overflow,c.status.requiredCount,c.topologyStatistics]),p(t,`positions rows`,n.positions.length,A(e.device));let l=r.length,u=i.length-1,d=s.length-1,f=c.positions.length,h=n.snapTolerance??0,_=2*l,v=(n,r)=>m(e,`${t}-${n}`,`uint32`,r),C=v(`feature-ring-starts`,d+1),w=v(`vertex-high`,l),T=v(`vertex-low`,l),O=v(`vertex-polygon`,l),k=v(`vertex-next`,l),j=v(`vertex-prev`,l),M=[x(e,{id:`${t}-feature-ring-starts`,operation:Z,variant:`feature-ring-starts`,bindings:[{name:`featureOffsets`,view:s,type:`u32`,access:`read`},{name:`polygonOffsets`,view:a,type:`u32`,access:`read`},{name:`featureRingStarts`,view:C,type:`u32`,access:`read_write`}],invocationCount:d+1,body:`featureRingStarts[featureRingStartsOffset + index] = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index]];`}),x(e,{id:`${t}-vertex-topology`,operation:Z,variant:`vertex-topology`,bindings:[{name:`positions`,view:r,type:`f32`,access:`read`},{name:`ringOffsets`,view:i,type:`u32`,access:`read`},{name:`featureRingStarts`,view:C,type:`u32`,access:`read`},{name:`vertexHigh`,view:w,type:`u32`,access:`read_write`},{name:`vertexLow`,view:T,type:`u32`,access:`read_write`},{name:`vertexPolygon`,view:O,type:`u32`,access:`read_write`},{name:`vertexNext`,view:k,type:`u32`,access:`read_write`},{name:`vertexPrev`,view:j,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`const RING_COUNT: u32 = ${u}u;
const FEATURE_COUNT: u32 = ${d}u;
const SNAP_TOLERANCE: f32 = ${y(h)};

${S}

fn quantise(value: f32) -> u32 {
  if (SNAP_TOLERANCE <= 0.0) {
    return bitcast<u32>(select(value, 0.0, value == 0.0));
  }
  let snapped = clamp(floor(value / SNAP_TOLERANCE + 0.5), -2147483520.0, 2147483520.0);
  return bitcast<u32>(i32(snapped)) ^ 0x80000000u;
}`,body:`var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  // The coverage identity of a ring is its feature: first ring of feature f is featureRingStarts[f].
  var polygon = 0u;
  high = FEATURE_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (featureRingStarts[featureRingStartsOffset + middle] <= ring) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let inRing = index >= ringBegin && index < ringEnd &&
    ring >= featureRingStarts[featureRingStartsOffset] && ring < featureRingStarts[featureRingStartsOffset + FEATURE_COUNT] &&
    ring < featureRingStarts[featureRingStartsOffset + polygon + 1u];
  let valid = inRing && isFiniteFloat(x) && isFiniteFloat(y);
  vertexHigh[vertexHighOffset + index] = select(${Q}, quantise(x), valid);
  vertexLow[vertexLowOffset + index] = select(${Q}, quantise(y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${Q}, polygon, valid);
  vertexNext[vertexNextOffset + index] = select(index + 1u, ringBegin, index + 1u == ringEnd);
  let previous = select(index - 1u, ringEnd - 1u, index == ringBegin);
  vertexPrev[vertexPrevOffset + index] = select(previous, previous | ${ve}, index == ringBegin);`})],P=E(e,`${t}-vertex`,Z,l,w,T),F=D(e,`${t}-vertex`,Z,l,w,T,P.sortedItems),I=v(`point-ids`,l),L=v(`edge-high`,l),R=v(`edge-low`,l);M.push(...P.nodes,...F.nodes,x(e,{id:`${t}-point-ids`,operation:Z,variant:`point-ids`,bindings:[{name:`sortedItems`,view:P.sortedItems,type:`u32`,access:`read`},{name:`groupIndex`,view:F.groupIndex,type:`u32`,access:`read`},{name:`vertexPolygon`,view:O,type:`u32`,access:`read`},{name:`pointIds`,view:I,type:`u32`,access:`read_write`}],invocationCount:l,body:`let vertex = sortedItems[sortedItemsOffset + index];
  pointIds[pointIdsOffset + vertex] = select(${Q},
    groupIndex[groupIndexOffset + index] - 1u, vertexPolygon[vertexPolygonOffset + vertex] != ${Q});`}),x(e,{id:`${t}-edge-keys`,operation:Z,variant:`edge-keys`,bindings:[{name:`vertexPolygon`,view:O,type:`u32`,access:`read`},{name:`vertexNext`,view:k,type:`u32`,access:`read`},{name:`pointIds`,view:I,type:`u32`,access:`read`},{name:`edgeHigh`,view:L,type:`u32`,access:`read_write`},{name:`edgeLow`,view:R,type:`u32`,access:`read_write`}],invocationCount:l,body:`let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + vertexNext[vertexNextOffset + index]];
  let valid = first != ${Q} && second != ${Q} && first != second &&
    vertexPolygon[vertexPolygonOffset + index] != ${Q};
  edgeHigh[edgeHighOffset + index] = select(${Q}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${Q}, max(first, second), valid);`}));let z=E(e,`${t}-edge`,Z,l,L,R),B=D(e,`${t}-edge`,Z,l,L,R,z.sortedItems);M.push(...z.nodes,...B.nodes);let V=v(`edge-partner`,l);M.push(x(e,{id:`${t}-edge-partner`,operation:Z,variant:`edge-partner`,bindings:[{name:`sortedItems`,view:z.sortedItems,type:`u32`,access:`read`},{name:`edgeHigh`,view:L,type:`u32`,access:`read`},{name:`vertexPolygon`,view:O,type:`u32`,access:`read`},{name:`groupIndex`,view:B.groupIndex,type:`u32`,access:`read`},{name:`groupStarts`,view:B.groupStarts,type:`u32`,access:`read`},{name:`edgePartner`,view:V,type:`u32`,access:`read_write`}],invocationCount:l,body:`let item = sortedItems[sortedItemsOffset + index];
  let polygon = vertexPolygon[vertexPolygonOffset + item];
  var partner = ${Q};
  if (polygon != ${Q} && edgeHigh[edgeHighOffset + item] != ${Q}) {
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    for (var other = groupStarts[groupStartsOffset + groupId]; other < end; other++) {
      let otherPolygon = vertexPolygon[vertexPolygonOffset + sortedItems[sortedItemsOffset + other]];
      if (otherPolygon != ${Q} && otherPolygon != polygon) {
        partner = min(partner, otherPolygon);
      }
    }
  }
  edgePartner[edgePartnerOffset + item] = partner;`}));let H=v(`arc-flags`,l),U=v(`start-flag`,l),W=v(`start-rank`,l),G=v(`start-list`,l+1),ee=v(`owned-length`,l),K=v(`owned-flag`,l);M.push(x(e,{id:`${t}-arc-flags`,operation:Z,variant:`arc-flags`,bindings:[{name:`vertexPrev`,view:j,type:`u32`,access:`read`},{name:`vertexPolygon`,view:O,type:`u32`,access:`read`},{name:`edgePartner`,view:V,type:`u32`,access:`read`},{name:`arcFlags`,view:H,type:`u32`,access:`read_write`},{name:`startFlag`,view:U,type:`u32`,access:`read_write`}],invocationCount:l,body:`let prevEntry = vertexPrev[vertexPrevOffset + index];
  let previous = prevEntry & 0x7fffffffu;
  let ringBegin = (prevEntry & ${ve}) != 0u;
  let valid = vertexPolygon[vertexPolygonOffset + index] != ${Q};
  let junction = valid && edgePartner[edgePartnerOffset + previous] != edgePartner[edgePartnerOffset + index];
  arcFlags[arcFlagsOffset + index] = select(0u, 1u, ringBegin || junction) | select(0u, 2u, junction);
  startFlag[startFlagOffset + index] = select(0u, 1u, ringBegin || junction);`}),...new g({id:`${t}-start-rank-scan`,input:U,output:W,mode:`exclusive`}).getCommandNodes(e),x(e,{id:`${t}-start-list`,operation:Z,variant:`start-list`,bindings:[{name:`startFlag`,view:U,type:`u32`,access:`read`},{name:`startRank`,view:W,type:`u32`,access:`read`},{name:`ringOffsets`,view:i,type:`u32`,access:`read`},{name:`startList`,view:G,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`const VERTEX_COUNT: u32 = ${l}u;
const RING_COUNT: u32 = ${u}u;`,body:`let flag = startFlag[startFlagOffset + index];
  let rank = startRank[startRankOffset + index];
  if (flag != 0u) {
    startList[startListOffset + rank] = index;
  }
  if (index == VERTEX_COUNT - 1u) {
    startList[startListOffset + rank + flag] = min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT);
  }`}),x(e,{id:`${t}-arc-lengths`,operation:Z,variant:`arc-lengths`,bindings:[{name:`arcFlags`,view:H,type:`u32`,access:`read`},{name:`startRank`,view:W,type:`u32`,access:`read`},{name:`startList`,view:G,type:`u32`,access:`read`},{name:`vertexPolygon`,view:O,type:`u32`,access:`read`},{name:`edgePartner`,view:V,type:`u32`,access:`read`},{name:`ownedLength`,view:ee,type:`u32`,access:`read_write`},{name:`ownedFlag`,view:K,type:`u32`,access:`read_write`}],invocationCount:l,body:`var length = 0u;
  var owned = 0u;
  let polygon = vertexPolygon[vertexPolygonOffset + index];
  if ((arcFlags[arcFlagsOffset + index] & 1u) != 0u && polygon != ${Q}) {
    let partner = edgePartner[edgePartnerOffset + index];
    if (partner == ${Q} || polygon < partner) {
      owned = 1u;
      // The arc runs to the next arc start in vertex order, or to the end of the ring, whose
      // first vertex is itself an arc start. The closing vertex is repeated.
      let arcEnd = startList[startListOffset + startRank[startRankOffset + index] + 1u];
      length = arcEnd - index + 1u;
    }
  }
    ownedLength[ownedLengthOffset + index] = length;
  ownedFlag[ownedFlagOffset + index] = owned;`}));let te=v(`arc-index`,l),q=v(`line-start`,l),ne=v(`track-offsets`,l+1),J=m(e,`${t}-line-positions`,`float32x2`,_),Y=v(`line-point`,_);M.push(...new g({id:`${t}-arc-index-scan`,input:K,output:te,mode:`exclusive`}).getCommandNodes(e),...new g({id:`${t}-line-start-scan`,input:ee,output:q,mode:`exclusive`}).getCommandNodes(e),x(e,{id:`${t}-track-offsets-fill`,operation:Z,variant:`track-offsets-fill`,bindings:[{name:`lineStart`,view:q,type:`u32`,access:`read`},{name:`ownedLength`,view:ee,type:`u32`,access:`read`},{name:`trackOffsets`,view:ne,type:`u32`,access:`read_write`}],invocationCount:l+1,declarations:`const VERTEX_COUNT: u32 = ${l}u;`,body:`trackOffsets[trackOffsetsOffset + index] =
    lineStart[lineStartOffset + VERTEX_COUNT - 1u] + ownedLength[ownedLengthOffset + VERTEX_COUNT - 1u];`}),x(e,{id:`${t}-track-offsets`,operation:Z,variant:`track-offsets`,bindings:[{name:`ownedFlag`,view:K,type:`u32`,access:`read`},{name:`arcIndex`,view:te,type:`u32`,access:`read`},{name:`lineStart`,view:q,type:`u32`,access:`read`},{name:`trackOffsets`,view:ne,type:`u32`,access:`read_write`}],invocationCount:l,body:`if (ownedFlag[ownedFlagOffset + index] != 0u) {
    trackOffsets[trackOffsetsOffset + arcIndex[arcIndexOffset + index]] = lineStart[lineStartOffset + index];
  }`}),x(e,{id:`${t}-copy-arcs`,operation:Z,variant:`copy-arcs`,bindings:[{name:`positions`,view:r,type:`f32`,access:`read`},{name:`vertexNext`,view:k,type:`u32`,access:`read`},{name:`ownedLength`,view:ee,type:`u32`,access:`read`},{name:`lineStart`,view:q,type:`u32`,access:`read`},{name:`pointIds`,view:I,type:`u32`,access:`read`},{name:`linePositions`,view:J,type:`f32`,access:`read_write`},{name:`linePoint`,view:Y,type:`u32`,access:`read_write`}],invocationCount:_,declarations:`const VERTEX_COUNT: u32 = ${l}u;`,body:`let total = lineStart[lineStartOffset + VERTEX_COUNT - 1u] + ownedLength[ownedLengthOffset + VERTEX_COUNT - 1u];
  if (index < total) {
    var low = 0u;
    var high = VERTEX_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (lineStart[lineStartOffset + middle] <= index) {
        low = middle;
      } else {
        high = middle;
      }
    }
    let step = index - lineStart[lineStartOffset + low];
    let vertex = select(vertexNext[vertexNextOffset + low + step - 1u], low, step == 0u);
    let pointId = pointIds[pointIdsOffset + vertex];
    let x = positions[positionsOffset + vertex * 2u];
    let y = positions[positionsOffset + vertex * 2u + 1u];
    let finite = pointId != ${Q};
    linePositions[linePositionsOffset + index * 2u] = select(0.0, x, finite);
    linePositions[linePositionsOffset + index * 2u + 1u] = select(0.0, y, finite);
    linePoint[linePointOffset + index] = pointId;
  }`}));let re=m(e,`${t}-line-importance`,`float32`,_),ie=v(`line-keep`,_),ae=v(`selection-ids`,_),X=v(`selection-count`,1),oe=v(`selection-overflow`,1),se={id:`${t}-arcs`,spatialContext:this.spatialContext,positions:J,lineOffsets:ne,importance:re,maximumIterations:n.maximumIterations,parameters:n.parameters,output:{ids:ae,outputMask:ie,status:{count:X,overflow:oe,...c.status.converged?{converged:c.status.converged}:{}}}};M.push(...new N(se).getCommandNodes(e));let ce=v(`keep-point`,l);M.push(b(e,{id:`${t}-keep-point-clear`,operation:Z,view:ce,type:`u32`,value:`0u`}),x(e,{id:`${t}-mark-junctions`,operation:Z,variant:`mark-junctions`,bindings:[{name:`arcFlags`,view:H,type:`u32`,access:`read`},{name:`pointIds`,view:I,type:`u32`,access:`read`},{name:`vertexPrev`,view:j,type:`u32`,access:`read`},{name:`edgePartner`,view:V,type:`u32`,access:`read`},{name:`keepPoint`,view:ce,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,declarations:`const KEEP_BOUNDARY: bool = ${n.simplifyBoundary===!1};`,body:`let pointId = pointIds[pointIdsOffset + index];
  let previous = vertexPrev[vertexPrevOffset + index] & 0x7fffffffu;
  let isBoundary = KEEP_BOUNDARY && (edgePartner[edgePartnerOffset + index] == ${Q} ||
    edgePartner[edgePartnerOffset + previous] == ${Q});
  if (((arcFlags[arcFlagsOffset + index] & 2u) != 0u || isBoundary) && pointId != ${Q}) {
    atomicMax(&keepPoint[keepPointOffset + pointId], 1u);
  }`}),x(e,{id:`${t}-mark-arcs`,operation:Z,variant:`mark-arcs`,bindings:[{name:`lineKeep`,view:ie,type:`u32`,access:`read`},{name:`linePoint`,view:Y,type:`u32`,access:`read`},{name:`trackOffsets`,view:ne,type:`u32`,access:`read`},{name:`keepPoint`,view:ce,type:`atomic<u32>`,access:`read_write`}],invocationCount:_,declarations:`const VERTEX_COUNT: u32 = ${l}u;`,body:`let pointId = linePoint[linePointOffset + index];
  if (index < trackOffsets[trackOffsetsOffset + VERTEX_COUNT] && lineKeep[lineKeepOffset + index] != 0u &&
      pointId != ${Q}) {
    atomicMax(&keepPoint[keepPointOffset + pointId], 1u);
  }`}));let le=n.topologyIterationCount??ye;if(M.push(...Ee(e,{id:t,positions:r,ringOffsets:i,vertexPolygon:O,pointIds:I,keepPoint:ce,vertexCount:l,ringCount:u})),le===0&&c.topologyStatistics&&M.push(b(e,{id:`${t}-topology-stats-zero`,operation:Z,view:c.topologyStatistics,type:`u32`,value:`0u`})),le>0){let o=v(`topology-work`,be+1),d=v(`owned-vertex`,l),f=n.topologyPairCapacity??Math.max(256,4*l);M.push(b(e,{id:`${t}-topology-stats-clear`,operation:Z,view:o,type:`u32`,value:`0u`}),x(e,{id:`${t}-owned-vertex`,operation:Z,variant:`owned-vertex`,bindings:[{name:`vertexPolygon`,view:O,type:`u32`,access:`read`},{name:`edgePartner`,view:V,type:`u32`,access:`read`},{name:`ownedVertex`,view:d,type:`u32`,access:`read_write`}],invocationCount:l,body:`let polygon = vertexPolygon[vertexPolygonOffset + index];
  let partner = edgePartner[edgePartnerOffset + index];
  ownedVertex[ownedVertexOffset + index] =
    select(0u, select(0u, 1u, partner == ${Q} || polygon < partner), polygon != ${Q});`}));let p=c.topologyStatistics?le:le-1;for(let n=0;n<=p;n++){let c=n===0,p=n===le;M.push(...ke(e,{id:`${t}-round-${n}`,positions:r,ringOffsets:i,polygonOffsets:a,featureOffsets:s,pointIds:I,vertexNext:k,ownedVertex:d,keepPoint:ce,vertexCount:l,ringCount:u,pairCapacity:f,repair:!p,statsView:o,crossingSlot:c?0:p?1:xe}))}c.topologyStatistics&&M.push(x(e,{id:`${t}-topology-stats-copy`,operation:Z,variant:`topology-stats-copy`,bindings:[{name:`workStats`,view:o,type:`u32`,access:`read`},{name:`topologyStatistics`,view:c.topologyStatistics,type:`u32`,access:`read_write`}],invocationCount:be,body:`topologyStatistics[topologyStatisticsOffset + index] = workStats[workStatsOffset + index];`}))}let ue=c.outputMask??v(`keep-mask`,l),de=v(`keep-rank`,l);return M.push(Ce(e,`${t}-keep-mask`,I,ce,ue),...new g({id:`${t}-keep-rank-scan`,input:ue,output:de,mode:`exclusive`}).getCommandNodes(e),we(e,{id:`${t}-emit-positions`,positions:r,keepMask:ue,keepRank:de,outputPositions:c.positions,vertexCount:l}),Te(e,{id:`${t}-emit-offsets`,ringOffsets:i,keepMask:ue,keepRank:de,outputRingOffsets:c.ringOffsets,overflow:c.status.overflow,requiredCount:c.status.requiredCount,vertexCount:l,ringCount:u,capacity:f})),M}};function Ce(e,t,n,r,i){return x(e,{id:t,operation:Z,variant:`keep-mask`,bindings:[{name:`pointIds`,view:n,type:`u32`,access:`read`},{name:`keepPoint`,view:r,type:`u32`,access:`read`},{name:`keepMask`,view:i,type:`u32`,access:`read_write`}],invocationCount:n.length,body:`let pointId = pointIds[pointIdsOffset + index];
  keepMask[keepMaskOffset + index] =
    select(0u, keepPoint[keepPointOffset + pointId], pointId != ${Q});`})}function we(e,t){return x(e,{id:t.id,operation:Z,variant:`emit-positions`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`keepMask`,view:t.keepMask,type:`u32`,access:`read`},{name:`keepRank`,view:t.keepRank,type:`u32`,access:`read`},{name:`outputPositions`,view:t.outputPositions,type:`f32`,access:`read_write`}],invocationCount:t.vertexCount,declarations:`const CAPACITY: u32 = ${t.outputPositions.length}u;`,body:`let rank = keepRank[keepRankOffset + index];
  if (keepMask[keepMaskOffset + index] != 0u && rank < CAPACITY) {
    outputPositions[outputPositionsOffset + rank * 2u] = positions[positionsOffset + index * 2u];
    outputPositions[outputPositionsOffset + rank * 2u + 1u] = positions[positionsOffset + index * 2u + 1u];
  }`})}function Te(e,t){let{requiredCount:n}=t;return x(e,{id:t.id,operation:Z,variant:`emit-offsets`,bindings:[{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`keepMask`,view:t.keepMask,type:`u32`,access:`read`},{name:`keepRank`,view:t.keepRank,type:`u32`,access:`read`},{name:`outputRingOffsets`,view:t.outputRingOffsets,type:`u32`,access:`read_write`},{name:`overflow`,view:t.overflow,type:`u32`,access:`read_write`},...n?[{name:`requiredCount`,view:n,type:`u32`,access:`read_write`}]:[]],invocationCount:t.ringCount+1,declarations:`const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const RING_COUNT: u32 = ${t.ringCount}u;
const CAPACITY: u32 = ${t.capacity}u;`,body:`let total = keepRank[keepRankOffset + VERTEX_COUNT - 1u] + keepMask[keepMaskOffset + VERTEX_COUNT - 1u];
  var begin = total;
  if (index < RING_COUNT) {
    let ringBegin = ringOffsets[ringOffsetsOffset + index];
    if (ringBegin < VERTEX_COUNT) {
      begin = keepRank[keepRankOffset + ringBegin];
    }
  }
  outputRingOffsets[outputRingOffsetsOffset + index] = min(begin, CAPACITY);
  if (index == RING_COUNT) {
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${n?`requiredCount[requiredCountOffset] = total;`:``}
  }`})}function Ee(e,t){let{id:n,vertexCount:r,ringCount:i}=t,a=m(e,`${n}-restore-point`,`uint32`,r);return[b(e,{id:`${n}-restore-point-clear`,operation:Z,view:a,type:`u32`,value:`0u`}),x(e,{id:`${n}-ring-minimum`,operation:Z,variant:`ring-minimum`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`pointIds`,view:t.pointIds,type:`u32`,access:`read`},{name:`keepPoint`,view:t.keepPoint,type:`u32`,access:`read`},{name:`restorePoint`,view:a,type:`atomic<u32>`,access:`read_write`}],invocationCount:i,declarations:`const VERTEX_COUNT: u32 = ${r}u;
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn isKept(vertex: u32) -> bool {
  let pointId = pointIds[pointIdsOffset + vertex];
  return pointId != ${Q} && keepPoint[keepPointOffset + pointId] != 0u;
}`,body:`let begin = ringOffsets[ringOffsetsOffset + index];
  var end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  if (end > begin + 1u && pointIds[pointIdsOffset + end - 1u] == pointIds[pointIdsOffset + begin]) {
    end = end - 1u;
  }
  var keptCount = 0u;
  var firstKept = ${Q};
  var secondKept = ${Q};
  for (var vertex = begin; vertex < end; vertex++) {
    if (isKept(vertex)) {
      if (keptCount == 0u) { firstKept = vertex; } else if (keptCount == 1u) { secondKept = vertex; }
      keptCount++;
    }
  }
  if (keptCount >= 1u && keptCount < 3u && end > begin + 2u) {
    let anchor = vertexAt(firstKept);
    var other = vertexAt(firstKept);
    var otherVertex = secondKept;
    var haveOther = keptCount >= 2u;
    if (haveOther) {
      other = vertexAt(secondKept);
    } else {
      var best = 0.0;
      for (var vertex = begin; vertex < end; vertex++) {
        if (!isKept(vertex) && pointIds[pointIdsOffset + vertex] != ${Q}) {
          let distanceToAnchor = distance(vertexAt(vertex), anchor);
          if (distanceToAnchor > best) { best = distanceToAnchor; otherVertex = vertex; }
        }
      }
      if (otherVertex != ${Q}) {
        haveOther = true;
        other = vertexAt(otherVertex);
        atomicMax(&restorePoint[restorePointOffset + pointIds[pointIdsOffset + otherVertex]], 1u);
      }
    }
    if (haveOther) {
      let direction = other - anchor;
      let directionLength = max(length(direction), 1e-30);
      var best = 0.0;
      var thirdVertex = ${Q};
      for (var vertex = begin; vertex < end; vertex++) {
        let pointId = pointIds[pointIdsOffset + vertex];
        if (!isKept(vertex) && pointId != ${Q} && vertex != otherVertex &&
            pointId != pointIds[pointIdsOffset + otherVertex]) {
          let offset = vertexAt(vertex) - anchor;
          let distanceToLine = abs(direction.x * offset.y - direction.y * offset.x) / directionLength;
          if (distanceToLine > best) { best = distanceToLine; thirdVertex = vertex; }
        }
      }
      if (thirdVertex != ${Q}) {
        atomicMax(&restorePoint[restorePointOffset + pointIds[pointIdsOffset + thirdVertex]], 1u);
      }
    }
  }`}),x(e,{id:`${n}-merge-restored`,operation:Z,variant:`merge-restored`,bindings:[{name:`restorePoint`,view:a,type:`u32`,access:`read`},{name:`keepPoint`,view:t.keepPoint,type:`u32`,access:`read_write`}],invocationCount:r,body:`keepPoint[keepPointOffset + index] = max(keepPoint[keepPointOffset + index], restorePoint[restorePointOffset + index]);`})]}function De(e,t){let{id:n,vertexCount:r,ringCount:i,keepMask:a,keepRank:o,ringOffsets:s}=t,c=m(e,`${n}-kept-vertex`,`uint32`,r),l=m(e,`${n}-distance-bits`,`uint32`,r),u=m(e,`${n}-span-best`,`uint32`,r),d=`const VERTEX_COUNT: u32 = ${r}u;
const RING_COUNT: u32 = ${i}u;
fn getRing(vertex: u32) -> u32 {
  var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= vertex) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  return ring;
}
fn getKeptTotal() -> u32 {
  return keepRank[keepRankOffset + VERTEX_COUNT - 1u] + keepMask[keepMaskOffset + VERTEX_COUNT - 1u];
}
// Number of kept vertices before 'position'.
fn getKeptBefore(position: u32) -> u32 {
  return select(keepRank[keepRankOffset + position], getKeptTotal(), position >= VERTEX_COUNT);
}`,f=(e,t)=>({name:e,view:t,type:`u32`,access:`read`});return[x(e,{id:`${n}-kept-vertex`,operation:Z,variant:`kept-vertex`,bindings:[f(`keepMask`,a),f(`keepRank`,o),{name:`keptVertex`,view:c,type:`u32`,access:`read_write`}],invocationCount:r,body:`if (keepMask[keepMaskOffset + index] != 0u) {
    keptVertex[keptVertexOffset + keepRank[keepRankOffset + index]] = index;
  }`}),x(e,{id:`${n}-span-ends`,operation:Z,variant:`span-ends`,bindings:[f(`ringOffsets`,s),f(`keepMask`,a),f(`keepRank`,o),f(`keptVertex`,c),{name:`spanEnd`,view:t.spanEnd,type:`u32`,access:`read_write`}],invocationCount:r,declarations:d,body:`var end = ${Q};
  if (keepMask[keepMaskOffset + index] != 0u) {
    let ring = getRing(index);
    let ringBegin = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    let rank = keepRank[keepRankOffset + index];
    end = keptVertex[keptVertexOffset + getKeptBefore(ringBegin)];
    if (rank + 1u < getKeptTotal()) {
      let candidate = keptVertex[keptVertexOffset + rank + 1u];
      if (candidate < ringEnd) {
        end = candidate;
      }
    }
  }
  spanEnd[spanEndOffset + index] = end;`}),b(e,{id:`${n}-span-best-clear`,operation:Z,view:u,type:`u32`,value:`0u`}),b(e,{id:`${n}-span-key-clear`,operation:Z,view:t.spanKey,type:`u32`,value:Q}),x(e,{id:`${n}-span-distance`,operation:Z,variant:`span-distance`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},f(`ringOffsets`,s),f(`keepMask`,a),f(`keepRank`,o),f(`keptVertex`,c),f(`spanEnd`,t.spanEnd),{name:`distanceBits`,view:l,type:`u32`,access:`read_write`},{name:`spanBest`,view:u,type:`atomic<u32>`,access:`read_write`}],invocationCount:r,declarations:`${d}
${Oe}
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}`,body:`var bits = 0u;
  if (keepMask[keepMaskOffset + index] == 0u) {
    let owner = getSpanOwner(index);
    if (owner != ${Q}) {
      let a = vertexAt(owner);
      let p = vertexAt(index);
      let direction = vertexAt(spanEnd[spanEndOffset + owner]) - a;
      let segmentLength = length(direction);
      let offset = p - a;
      let separation = select(
        distance(p, a),
        abs(direction.x * offset.y - direction.y * offset.x) / max(segmentLength, 1e-30),
        segmentLength > 0.0);
      if (separation > 0.0) {
        bits = bitcast<u32>(separation);
        atomicMax(&spanBest[spanBestOffset + owner], bits);
      }
    }
  }
  distanceBits[distanceBitsOffset + index] = bits;`}),x(e,{id:`${n}-span-split`,operation:Z,variant:`span-split`,bindings:[f(`ringOffsets`,s),f(`keepMask`,a),f(`keepRank`,o),f(`keptVertex`,c),f(`distanceBits`,l),f(`spanBest`,u),{name:`spanKey`,view:t.spanKey,type:`atomic<u32>`,access:`read_write`}],invocationCount:r,declarations:`${d}
${Oe}`,body:`let bits = distanceBits[distanceBitsOffset + index];
  if (bits != 0u && keepMask[keepMaskOffset + index] == 0u) {
    let owner = getSpanOwner(index);
    if (owner != ${Q} && bits == spanBest[spanBestOffset + owner]) {
      atomicMin(&spanKey[spanKeyOffset + owner], select(index + VERTEX_COUNT, index, index > owner));
    }
  }`})]}var Oe=`
fn getSpanOwner(vertex: u32) -> u32 {
  let ring = getRing(vertex);
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let firstRank = getKeptBefore(ringBegin);
  let endRank = getKeptBefore(ringEnd);
  if (vertex < ringBegin || vertex >= ringEnd || endRank <= firstRank) {
    return ${Q};
  }
  // The kept vertex before it in the ring, else the ring's last one (the span wraps over the end).
  let before = keepRank[keepRankOffset + vertex];
  return keptVertex[keptVertexOffset + select(endRank - 1u, before - 1u, before > firstRank)];
}`;function ke(e,t){let{id:n,vertexCount:r,ringCount:i,pairCapacity:a,statsView:o}=t,s=(t,r)=>m(e,`${n}-${t}`,`uint32`,r),c=s(`mask`,r),l=s(`rank`,r),u=m(e,`${n}-compact-positions`,`float32x2`,r),d=s(`compact-ring-offsets`,i+1),f=s(`compact-overflow`,1),p=s(`span-end`,r),h=s(`span-split`,r),_=s(`row-info`,r*4),v=s(`bad`,r),y=s(`pair-left`,a),S=s(`pair-right`,a),C=s(`pair-count`,1),w=s(`pair-overflow`,1),T=s(`pair-kinds`,a),{crossingSlot:E}=t,D=[Ce(e,`${n}-keep-mask`,t.pointIds,t.keepPoint,c),...new g({id:`${n}-rank-scan`,input:c,output:l,mode:`exclusive`}).getCommandNodes(e),we(e,{id:`${n}-emit-positions`,positions:t.positions,keepMask:c,keepRank:l,outputPositions:u,vertexCount:r}),Te(e,{id:`${n}-emit-offsets`,ringOffsets:t.ringOffsets,keepMask:c,keepRank:l,outputRingOffsets:d,overflow:f,vertexCount:r,ringCount:i,capacity:r}),...De(e,{id:n,positions:t.positions,ringOffsets:t.ringOffsets,keepMask:c,keepRank:l,spanEnd:p,spanKey:h,vertexCount:r,ringCount:i}),x(e,{id:`${n}-row-info`,operation:Z,variant:`row-info`,bindings:[{name:`keepMask`,view:c,type:`u32`,access:`read`},{name:`keepRank`,view:l,type:`u32`,access:`read`},{name:`spanEnd`,view:p,type:`u32`,access:`read`},{name:`pointIds`,view:t.pointIds,type:`u32`,access:`read`},{name:`ownedVertex`,view:t.ownedVertex,type:`u32`,access:`read`},{name:`rowInfo`,view:_,type:`u32`,access:`read_write`}],invocationCount:r,body:`if (keepMask[keepMaskOffset + index] != 0u) {
    let base = rowInfoOffset + keepRank[keepRankOffset + index] * 4u;
    rowInfo[base] = pointIds[pointIdsOffset + index];
    rowInfo[base + 1u] = pointIds[pointIdsOffset + spanEnd[spanEndOffset + index]];
    rowInfo[base + 2u] = index;
    rowInfo[base + 3u] = ownedVertex[ownedVertexOffset + index];
  }`}),b(e,{id:`${n}-bad-clear`,operation:Z,view:v,type:`u32`,value:`0u`}),...new j({id:`${n}-intersect`,left:{kind:`polygons`,positions:u,featureOffsets:t.featureOffsets,polygonOffsets:t.polygonOffsets,ringOffsets:d},output:{pairs:{leftIds:y,rightIds:S,status:{count:C,overflow:w}},kinds:T}}).getCommandNodes(e),x(e,{id:`${n}-classify`,operation:Z,variant:`classify-${E}`,bindings:[{name:`pairLeft`,view:y,type:`u32`,access:`read`},{name:`pairRight`,view:S,type:`u32`,access:`read`},{name:`pairKinds`,view:T,type:`u32`,access:`read`},{name:`pairCount`,view:C,type:`u32`,access:`read`},{name:`pairOverflow`,view:w,type:`u32`,access:`read`},{name:`rowInfo`,view:_,type:`u32`,access:`read`},{name:`badFlag`,view:v,type:`atomic<u32>`,access:`read_write`},{name:`stats`,view:o,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,declarations:`const SLOT: u32 = ${E}u;`,body:`if (index == 0u && pairOverflow[pairOverflowOffset] != 0u) {
    atomicMax(&stats[statsOffset + 3u], 1u);
  }
  if (index < pairCount[pairCountOffset]) {
    let kind = pairKinds[pairKindsOffset + index];
    let leftBase = rowInfoOffset + pairLeft[pairLeftOffset + index] * 4u;
    let rightBase = rowInfoOffset + pairRight[pairRightOffset + index] * 4u;
    let a0 = rowInfo[leftBase];
    let a1 = rowInfo[leftBase + 1u];
    let b0 = rowInfo[rightBase];
    let b1 = rowInfo[rightBase + 1u];
    let shares = a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1;
    let identical = (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0);
    let offending = select(!shares, !identical, kind == 4u);
    if (offending) {
      atomicMax(&badFlag[badFlagOffset + rowInfo[leftBase + 2u]], 1u);
      atomicMax(&badFlag[badFlagOffset + rowInfo[rightBase + 2u]], 1u);
      if (rowInfo[leftBase + 3u] != 0u && rowInfo[rightBase + 3u] != 0u) {
        atomicAdd(&stats[statsOffset + SLOT], 1u);
      }
    }
  }`})];return t.repair&&D.push(x(e,{id:`${n}-repair`,operation:Z,variant:`repair`,bindings:[{name:`badFlag`,view:v,type:`u32`,access:`read`},{name:`spanSplit`,view:h,type:`u32`,access:`read`},{name:`pointIds`,view:t.pointIds,type:`u32`,access:`read`},{name:`keepPoint`,view:t.keepPoint,type:`atomic<u32>`,access:`read_write`},{name:`stats`,view:o,type:`atomic<u32>`,access:`read_write`}],invocationCount:r,declarations:`const VERTEX_COUNT: u32 = ${r}u;`,body:`let key = spanSplit[spanSplitOffset + index];
  let split = select(key, key - VERTEX_COUNT, key >= VERTEX_COUNT && key != ${Q});
  if (badFlag[badFlagOffset + index] != 0u && key != ${Q}) {
    let pointId = pointIds[pointIdsOffset + split];
    if (pointId != ${Q}) {
      if (atomicMax(&keepPoint[keepPointOffset + pointId], 1u) == 0u) {
        atomicAdd(&stats[statsOffset + 2u], 1u);
      }
    }
  }`})),D}var Ae=Object.freeze({...n,measure:`distance`}),je=`GPUNetworkNoding`;function Me(e){return Math.floor(ce(e)/4)}var Ne=`
fn isFiniteCoordinate(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
fn getOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getOrderedValue(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}`,Pe=class{id;props;nodeCapacity;edgeCapacity;spatialContext;constructor(e){this.id=e.id??`network-noding`,this.props=e;let{id:n}=this;this.spatialContext=t(n,e.spatialContext,Ae);let{nodes:r,edges:i,topology:a,pieces:o}=e.output;if(s(n,`intersectionCapacity`,e.intersectionCapacity),h(e.tolerance,[`float32`],`${n} tolerance`),e.tolerance.length<1)throw Error(`${n} tolerance must contain one float32 row`);h(r.positions,[`float32x2`],`${n} output.nodes.positions`);for(let[t,i]of[[`output.nodes.status.count`,r.status.count],[`output.nodes.status.requiredCount`,r.status.requiredCount],[`output.status.overflow`,e.output.status?.overflow],[`output.status.uncertainCount`,e.output.status?.uncertainCount]])if(i&&(h(i,[`uint32`],`${n} ${t}`),i.length<1))throw Error(`${n} ${t} must contain one uint32 row`);if(this.nodeCapacity=r.positions.length,this.edgeCapacity=o.sourceIds.length,this.nodeCapacity<1)throw Error(`${n} output.nodes.positions must be non-empty`);if(this.edgeCapacity<1)throw Error(`${n} output.pieces.sourceIds must be non-empty`);if(a.offsets.length!==this.nodeCapacity+1)throw Error(`${n} output.topology.offsets length must be output.nodes.positions.length + 1`);h(a.offsets,[`uint32`],`${n} output.topology.offsets`),h(a.neighbors,[`uint32`],`${n} output.topology.neighbors`),h(a.weights,[`float32`],`${n} output.topology.weights`);for(let[e,t]of[[`topology.neighbors`,a.neighbors],[`topology.weights`,a.weights],[`topology.edgeIds`,a.edgeIds]])if(t&&t.length!==this.edgeCapacity*2)throw Error(`${n} output.${e} length must be twice the edge capacity`);h(i.fromNodes,[`uint32`],`${n} output.edges.fromNodes`),h(i.toNodes,[`uint32`],`${n} output.edges.toNodes`),h(i.lengths,[`float32`],`${n} output.edges.lengths`);for(let[e,t]of[[`output.edges.fromNodes`,i.fromNodes],[`output.edges.toNodes`,i.toNodes],[`output.edges.lengths`,i.lengths]])if(t.length!==this.edgeCapacity)throw Error(`${n} ${e} length must equal output.pieces.sourceIds.length`)}getCommandNodes(e){let{id:t,props:n,nodeCapacity:r,edgeCapacity:i}=this,{tolerance:a,output:c}=n,{pieces:l,nodes:u,edges:d,topology:f}=c;s(t,`intersectionCapacity`,n.intersectionCapacity,1,Me(e.device)),fe(t,e.device,n.lines.positions.length,Math.max(n.lines.lineOffsets.length-1,0),n.intersectionCapacity),o(t,e,[a,u.positions,u.status.count,u.status.requiredCount,d.fromNodes,d.toNodes,d.lengths,f.offsets,f.neighbors,f.weights,f.edgeIds,n.output.status?.overflow]);let p=[],h=i*2,v=(n,r,i)=>m(e,`${t}-${n}`,r,i),y=l.status.overflow;p.push(...new de({id:`${t}-split`,lines:n.lines,intersectionCapacity:n.intersectionCapacity,spatialSort:n.spatialSort,leafCapacity:n.leafCapacity,output:{pieces:l,status:c.status?.uncertainCount?{uncertainCount:c.status.uncertainCount}:void 0}}).getCommandNodes(e));let S=v(`keys-x`,`uint32`,h),C=v(`keys-y`,`uint32`,h),w=v(`endpoint-points`,`float32x2`,h),E=v(`endpoint-indices`,`uint32`,h),D=v(`snap-origin`,`uint32`,2),O=v(`snap-overflow`,`uint32`,1),k=U({counters:[{target:`snapOrigin`,word:0,operation:`min`},{target:`snapOrigin`,word:1,operation:`min`}],rowCount:`ENDPOINT_COUNT`,rowSource:`let edge = index / 2u;
  endpointIndices[endpointIndicesOffset + index] = index;
  var point = vec2f(0.0);
  if (edge < pieceCount[pieceCountOffset]) {
    var vertex = pieceOffsets[pieceOffsetsOffset + edge];
    if ((index & 1u) == 1u) { vertex = pieceOffsets[pieceOffsetsOffset + edge + 1u] - 1u; }
    point = vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
    if (isFiniteCoordinate(point.x) && isFiniteCoordinate(point.y)) {
      minCounter(0u, getOrderedKey(point.x));
      minCounter(1u, getOrderedKey(point.y));
    }
  }
  endpointPoints[endpointPointsOffset + index * 2u] = point.x;
  endpointPoints[endpointPointsOffset + index * 2u + 1u] = point.y;`});p.push(b(e,{id:`${t}-clear-snap-origin`,operation:je,view:D,type:`u32`,value:`0xffffffffu`}),b(e,{id:`${t}-clear-snap-overflow`,operation:je,view:O,type:`u32`,value:`0u`}),x(e,{id:`${t}-endpoints`,operation:je,variant:`endpoints`,bindings:[{name:`pieceCount`,view:l.status.count,type:`u32`,access:`read`},{name:`pieceOffsets`,view:l.geometry.lineOffsets,type:`u32`,access:`read`},{name:`piecePositions`,view:l.geometry.positions,type:`f32`,access:`read`},{name:`endpointPoints`,view:w,type:`f32`,access:`read_write`},{name:`endpointIndices`,view:E,type:`u32`,access:`read_write`},{name:`snapOrigin`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:h,workgroupSize:256,guardIndex:!1,declarations:`${Ne}
const ENDPOINT_COUNT: u32 = ${h}u;
${k.declarations}`,body:k.body}),x(e,{id:`${t}-endpoint-keys`,operation:je,variant:`endpoint-keys`,bindings:[{name:`pieceCount`,view:l.status.count,type:`u32`,access:`read`},{name:`endpointPoints`,view:w,type:`f32`,access:`read`},{name:`tolerance`,view:a,type:`f32`,access:`read`},{name:`snapOrigin`,view:D,type:`u32`,access:`read`},{name:`snapOverflow`,view:O,type:`u32`,access:`read_write`},{name:`keysX`,view:S,type:`u32`,access:`read_write`},{name:`keysY`,view:C,type:`u32`,access:`read_write`}],invocationCount:h,declarations:`${Ne}
// Integer tolerance cell of value counted from origin, or the bit pattern for tolerance 0
// (only bit-identical end points merge). Sets isOverflow past 2^24 cells, where the f32 quotient
// no longer resolves unit cells.
fn snapKey(value: f32, origin: f32, size: f32, isOverflow: ptr<function, bool>) -> u32 {
  if (!isFiniteCoordinate(value)) {
    return 0xfffffffeu;
  }
  if (size > 0.0) {
    let cell = floor(max(value - origin, 0.0) / size);
    if (!(cell < 16777216.0)) {
      *isOverflow = true;
      return 16777215u;
    }
    return u32(cell);
  }
  return min(bitcast<u32>(select(value, 0.0, value == 0.0)), 0xfffffffeu);
}`,body:`var keyX = 0xffffffffu;
  var keyY = 0xffffffffu;
  if (index / 2u < pieceCount[pieceCountOffset]) {
    let point = vec2f(endpointPoints[endpointPointsOffset + index * 2u], endpointPoints[endpointPointsOffset + index * 2u + 1u]);
    let origin = vec2f(
      getOrderedValue(snapOrigin[snapOriginOffset]),
      getOrderedValue(snapOrigin[snapOriginOffset + 1u])
    );
    let size = tolerance[toleranceOffset];
    var isOverflow = false;
    keyX = snapKey(point.x, origin.x, size, &isOverflow);
    keyY = snapKey(point.y, origin.y, size, &isOverflow);
    if (isOverflow) {
      snapOverflow[snapOverflowOffset] = 1u;
    }
  }
  keysX[keysXOffset + index] = keyX;
  keysY[keysYOffset + index] = keyY;`}));let A=v(`sorted-keys-y`,`uint32`,h),j=v(`order-y`,`uint32`,h),M=v(`gathered-keys-x`,`uint32`,h),N=v(`sorted-keys-x`,`uint32`,h),P=v(`order`,`uint32`,h);p.push(...new _({id:`${t}-sort-y`,keys:C,values:E,outputKeys:A,outputValues:j}).getCommandNodes(e),x(e,{id:`${t}-gather-x`,operation:je,variant:`gather-x`,bindings:[{name:`orderY`,view:j,type:`u32`,access:`read`},{name:`keysX`,view:S,type:`u32`,access:`read`},{name:`gatheredKeysX`,view:M,type:`u32`,access:`read_write`}],invocationCount:h,body:`gatheredKeysX[gatheredKeysXOffset + index] = keysX[keysXOffset + orderY[orderYOffset + index]];`}),...new _({id:`${t}-sort-x`,keys:M,values:j,outputKeys:N,outputValues:P}).getCommandNodes(e));let F=v(`head-flags`,`uint32`,h),I=v(`head-ranks`,`uint32`,h),L=v(`endpoint-nodes`,`uint32`,h);p.push(x(e,{id:`${t}-head-flags`,operation:je,variant:`head-flags`,bindings:[{name:`order`,view:P,type:`u32`,access:`read`},{name:`keysX`,view:S,type:`u32`,access:`read`},{name:`keysY`,view:C,type:`u32`,access:`read`},{name:`headFlags`,view:F,type:`u32`,access:`read_write`}],invocationCount:h,body:`let current = order[orderOffset + index];
  var flag = select(0u, 1u, keysX[keysXOffset + current] != 0xffffffffu);
  if (flag == 1u && index > 0u) {
    let previous = order[orderOffset + index - 1u];
    if (keysX[keysXOffset + previous] == keysX[keysXOffset + current] &&
        keysY[keysYOffset + previous] == keysY[keysYOffset + current]) { flag = 0u; }
  }
  headFlags[headFlagsOffset + index] = flag;`}),...new g({id:`${t}-head-scan`,input:F,output:I,mode:`exclusive`}).getCommandNodes(e),x(e,{id:`${t}-assign-nodes`,operation:je,variant:`assign-nodes`,bindings:[{name:`order`,view:P,type:`u32`,access:`read`},{name:`keysX`,view:S,type:`u32`,access:`read`},{name:`headFlags`,view:F,type:`u32`,access:`read`},{name:`headRanks`,view:I,type:`u32`,access:`read`},{name:`endpointPoints`,view:w,type:`f32`,access:`read`},{name:`endpointNodes`,view:L,type:`u32`,access:`read_write`},{name:`nodePositions`,view:u.positions,type:`f32`,access:`read_write`}],invocationCount:h,declarations:`const NODE_CAPACITY: u32 = ${r}u;`,body:`let endpoint = order[orderOffset + index];
  if (keysX[keysXOffset + endpoint] == 0xffffffffu) {
    endpointNodes[endpointNodesOffset + endpoint] = 0xffffffffu;
    return;
  }
  let group = headRanks[headRanksOffset + index] + headFlags[headFlagsOffset + index] - 1u;
  endpointNodes[endpointNodesOffset + endpoint] = group;
  if (headFlags[headFlagsOffset + index] == 1u && group < NODE_CAPACITY) {
    nodePositions[nodePositionsOffset + group * 2u] = endpointPoints[endpointPointsOffset + endpoint * 2u];
    nodePositions[nodePositionsOffset + group * 2u + 1u] = endpointPoints[endpointPointsOffset + endpoint * 2u + 1u];
  }`})),p.push(x(e,{id:`${t}-edges`,operation:je,variant:`edges`,bindings:[{name:`pieceCount`,view:l.status.count,type:`u32`,access:`read`},{name:`pieceOffsets`,view:l.geometry.lineOffsets,type:`u32`,access:`read`},{name:`piecePositions`,view:l.geometry.positions,type:`f32`,access:`read`},{name:`endpointNodes`,view:L,type:`u32`,access:`read`},{name:`fromNodes`,view:d.fromNodes,type:`u32`,access:`read_write`},{name:`toNodes`,view:d.toNodes,type:`u32`,access:`read_write`},{name:`lengths`,view:d.lengths,type:`f32`,access:`read_write`}],invocationCount:i,body:`var length = 0.0;
  var fromNode = 0xffffffffu;
  var toNode = 0xffffffffu;
  if (index < pieceCount[pieceCountOffset]) {
    fromNode = endpointNodes[endpointNodesOffset + index * 2u];
    toNode = endpointNodes[endpointNodesOffset + index * 2u + 1u];
    let first = pieceOffsets[pieceOffsetsOffset + index];
    let last = pieceOffsets[pieceOffsetsOffset + index + 1u];
    for (var vertex = first + 1u; vertex < last; vertex++) {
      let a = vec2f(piecePositions[piecePositionsOffset + (vertex - 1u) * 2u], piecePositions[piecePositionsOffset + (vertex - 1u) * 2u + 1u]);
      let b = vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
      length = length + distance(a, b);
    }
  }
  fromNodes[fromNodesOffset + index] = fromNode;
  toNodes[toNodesOffset + index] = toNode;
  lengths[lengthsOffset + index] = length;`}));let R=v(`csr-flags`,`uint32`,h),z=v(`csr-positions`,`uint32`,h),B=v(`sorted-starts`,`uint32`,h),V=v(`sorted-entries`,`uint32`,h);p.push(x(e,{id:`${t}-csr-flags`,operation:je,variant:`csr-flags`,bindings:[{name:`order`,view:P,type:`u32`,access:`read`},{name:`fromNodes`,view:d.fromNodes,type:`u32`,access:`read`},{name:`toNodes`,view:d.toNodes,type:`u32`,access:`read`},{name:`csrFlags`,view:R,type:`u32`,access:`read_write`}],invocationCount:h,declarations:`const NODE_CAPACITY: u32 = ${r}u;`,body:`let edge = order[orderOffset + index] / 2u;
  let isValid = fromNodes[fromNodesOffset + edge] < NODE_CAPACITY &&
    toNodes[toNodesOffset + edge] < NODE_CAPACITY;
  csrFlags[csrFlagsOffset + index] = select(0u, 1u, isValid);`}),...new g({id:`${t}-csr-scan`,input:R,output:z,mode:`exclusive`}).getCommandNodes(e),x(e,{id:`${t}-csr-compact`,operation:je,variant:`csr-compact`,bindings:[{name:`order`,view:P,type:`u32`,access:`read`},{name:`fromNodes`,view:d.fromNodes,type:`u32`,access:`read`},{name:`toNodes`,view:d.toNodes,type:`u32`,access:`read`},{name:`csrFlags`,view:R,type:`u32`,access:`read`},{name:`csrPositions`,view:z,type:`u32`,access:`read`},{name:`sortedStarts`,view:B,type:`u32`,access:`read_write`},{name:`sortedEntries`,view:V,type:`u32`,access:`read_write`}],invocationCount:h,declarations:`const NODE_CAPACITY: u32 = ${r}u;
const LAST: u32 = ${h-1}u;`,body:`let total = csrPositions[csrPositionsOffset + LAST] + csrFlags[csrFlagsOffset + LAST];
  if (csrFlags[csrFlagsOffset + index] != 0u) {
    let endpoint = order[orderOffset + index];
    let edge = endpoint / 2u;
    let position = csrPositions[csrPositionsOffset + index];
    sortedStarts[sortedStartsOffset + position] =
      select(fromNodes[fromNodesOffset + edge], toNodes[toNodesOffset + edge], (endpoint & 1u) == 1u);
    sortedEntries[sortedEntriesOffset + position] = endpoint;
  }
  if (index >= total) {
    sortedStarts[sortedStartsOffset + index] = NODE_CAPACITY;
    sortedEntries[sortedEntriesOffset + index] = 0xffffffffu;
  }`}),T(e,{id:`${t}-csr-offsets`,operation:je,variant:`csr-offsets`,segmentCount:r,sortedKeys:B,segmentOffsets:f.offsets}));let H=[{name:`sortedStarts`,view:B,type:`u32`,access:`read`},{name:`sortedEntries`,view:V,type:`u32`,access:`read`},{name:`fromNodes`,view:d.fromNodes,type:`u32`,access:`read`},{name:`toNodes`,view:d.toNodes,type:`u32`,access:`read`},{name:`lengths`,view:d.lengths,type:`f32`,access:`read`},{name:`neighbors`,view:f.neighbors,type:`u32`,access:`read_write`},{name:`weights`,view:f.weights,type:`f32`,access:`read_write`}];f.edgeIds&&H.push({name:`edgeIds`,view:f.edgeIds,type:`u32`,access:`read_write`}),p.push(x(e,{id:`${t}-csr-entries`,operation:je,variant:`csr-entries`,bindings:H,invocationCount:h,declarations:`const NODE_CAPACITY: u32 = ${r}u;`,body:`var neighbor = 0xffffffffu;
  var weight = 3.4028234e38;
  var edgeId = 0xffffffffu;
  if (sortedStarts[sortedStartsOffset + index] < NODE_CAPACITY) {
    let entry = sortedEntries[sortedEntriesOffset + index];
    let edge = entry / 2u;
    neighbor = select(toNodes[toNodesOffset + edge], fromNodes[fromNodesOffset + edge], (entry & 1u) == 1u);
    weight = lengths[lengthsOffset + edge];
    edgeId = edge;
  }
  neighbors[neighborsOffset + index] = neighbor;
  weights[weightsOffset + index] = weight;
  ${f.edgeIds?`edgeIds[edgeIdsOffset + index] = edgeId;`:``}`}));let W=[{name:`headFlags`,view:F,type:`u32`,access:`read`},{name:`headRanks`,view:I,type:`u32`,access:`read`},{name:`splitOverflow`,view:y,type:`u32`,access:`read`},{name:`snapOverflow`,view:O,type:`u32`,access:`read`},{name:`nodeCount`,view:u.status.count,type:`u32`,access:`read_write`}];return u.status.requiredCount&&W.push({name:`requiredCount`,view:u.status.requiredCount,type:`u32`,access:`read_write`}),n.output.status?.overflow&&W.push({name:`overflow`,view:n.output.status.overflow,type:`u32`,access:`read_write`}),p.push(x(e,{id:`${t}-scalars`,operation:je,variant:`scalars`,bindings:W,invocationCount:1,declarations:`const NODE_CAPACITY: u32 = ${r}u;
const LAST: u32 = ${h-1}u;`,body:`let total = headRanks[headRanksOffset + LAST] + headFlags[headFlagsOffset + LAST];
  nodeCount[nodeCountOffset] = min(total, NODE_CAPACITY);
  ${u.status.requiredCount?`requiredCount[requiredCountOffset] = total;`:``}
  ${n.output.status?.overflow?`overflow[overflowOffset] = select(0u, 1u, total > NODE_CAPACITY || splitOverflow[splitOverflowOffset] != 0u || snapOverflow[snapOverflowOffset] != 0u);`:``}`})),p}};function Fe(e,t,n){let r=e.properties.routes??[],i=e.column(`shapeRoute`),a=e.column(`shapeDirection`),o=e.column(`shapePathOffsets`),s=P(t,e.column(`shapeVertices`)),c=t.project(re[0],re[1]),l=e=>Math.abs(s[e*2]-c[0])<=3500&&Math.abs(s[e*2+1]-c[1])<=3500,u=[],d=[0],f=new Set,p=0,m=new Set;for(let e=0;e+1<o.length;e++){let t=i[e],c=r[t]?.type===`rail`;if(a[e]!==0||n===`rail`&&!c)continue;let h=[],g=()=>{if(h.length>=2){let e=h.map(e=>`${s[e*2].toFixed(1)},${s[e*2+1].toFixed(1)}`).join(`;`);if(f.has(e))p++;else{f.add(e);for(let e of h)u.push(s[e*2],s[e*2+1]);d.push(u.length/2),m.add(t)}}h=[]};for(let t=o[e];t<o[e+1];t++)n===`rail`||l(t)?h.push(t):g();g()}let h=Float32Array.from(u);return{local:h,lngLat:new Float32Array,offsets:Uint32Array.from(d),pathCount:d.length-1,vertexCount:h.length/2,routeCount:m.size,duplicatesRemoved:p}}function Ie(e){let t=[],n=[],{lngLat:r,ringOffsets:i}=e;for(let a=0;a<e.ringCount;a++){let o=i[a],s=i[a+1];for(let i=o;i<s;i++){let c=i+1<s?i+1:o,l=r[i*2],u=r[i*2+1],d=r[c*2],f=r[c*2+1];l===d&&u===f||(t.push(l,u,d,f),n.push(e.ringFeature[a]))}}return{endpoints:Float32Array.from(t),featureRows:Uint32Array.from(n),count:n.length}}var Le=e.METER_OFFSETS,Re=e.LNGLAT,ze=[``,`proper crossing`,`touch`,`collinear touch`,`overlap`,`uncertain`],Be=[[255,70,70,255],[255,200,40,255],[255,90,220,255],[70,205,150,255],[255,255,255,255]],Ve=[[226,96,80,255],[150,160,180,255],[255,200,40,255],[70,205,150,255],[78,168,222,255],[190,120,255,255]];async function He(e){let t=e.datasets.get(`chicago-roads`),n=e.datasets.get(`cta-transit`),r=e.datasets.get(`us-counties`),i=t.defaultOrigin,a=t.getProjection(i),o=null,s=null,c={projection:a,ctx:e,origin:i,coordinateOrigin:[i[0],i[1],0],getStreets:()=>o??=J(t,a),rail:Y(n,a),buildNoding:e=>Fe(n,a,e),getCounties:()=>{if(!s){let e=ee(r),t=new Uint32Array(e.featureCount),n=r.column(`stateFips`),i=new Map;for(let r=0;r<e.featureCount;r++){let e=n[r];i.has(e)||i.set(e,i.size),t[r]=i.get(e)}let a=new Uint32Array(e.featureCount);if(r.hasColumn(`rucc2023`)){let t=r.column(`rucc2023`);for(let n=0;n<e.featureCount;n++)a[n]=t[n]}s={layout:e,groups:{state:t,rucc:a}}}return s}},l={crossings:()=>We(c),noding:()=>Ge(c),dissolve:()=>Ke(c),generalise:()=>qe(c)},u=new Map,d=!1,f=new I,p=await l[e.options.view]();u.set(e.options.view,p);let m=e.options.view,h=new Map,g=()=>p,_=t=>{f.request({build:async()=>u.get(t)??await l[t](),commit:n=>{u.set(t,n),p=n,m=t;for(let r of h.get(t)??[])n.setOption(r,e.options);h.delete(t),e.requestLayers()},discard:e=>{u.get(t)!==e&&e.destroy()}})},v=(e,t)=>{g().setOption(e,t);for(let t of Object.keys(l)){if(t===m)continue;let n=h.get(t)??new Set;n.add(e),h.set(t,n)}};return{getCompiledGraphs:()=>g().getCompiledGraphs(),setOption(t,n,r){t===`view`?_(r.view):v(t,r),e.requestLayers()},onThemeChange:()=>e.requestLayers(),encode(e,t){d||g().encode(e,t)},getLayers:()=>g().getLayers(),destroy(){d=!0,f.destroy();for(let e of u.values())e.destroy();u.clear()}}}var Ue=1<<19;async function We(e){let{ctx:t}=e,{device:n}=t,r=new f(n,`crossings`),i=e.getStreets(),o=e.rail,s=r.createBuffer(`street-positions`,i.local),c=r.createBuffer(`street-offsets`,i.offsets),l=r.createBuffer(`rail-positions`,o.local),u=r.createBuffer(`rail-offsets`,o.offsets),p=ne(r,`crossing-streets`,i.local,i.offsets),m=ne(r,`crossing-rail`,o.local,o.offsets),h=r.createParameterBuffer(`kind-filter`,`float32`,1),g=new Map,_=null,y=!0,b=async f=>{let p=f.crossMode===`streets-rail`,m=!p&&f.sameFeatureOnly,_=`${f.crossMode}|${m}|${f.crossSpatialSort}`,y=g.get(_);if(y)return y;let b=r.createBuffer(`pair-left-${_}`,Ue*4),x=r.createBuffer(`pair-right-${_}`,Ue*4),S=r.createBuffer(`count-${_}`,4),C=r.createBuffer(`overflow-${_}`,4),w=r.createBuffer(`total-${_}`,4),T=r.createBuffer(`kinds-${_}`,Ue*4),E=r.createBuffer(`points-${_}`,Ue*8),D=r.createBuffer(`shown-${_}`,Ue*4),O=r.createBuffer(`uncertain-${_}`,4),k=new a(n,{id:`crossings-${_}`}),A=G(k),M=A(`kinds`,T,`uint32`,Ue);k.add(new j({spatialContext:e.projection.spatialContext,id:`crossings`,left:{kind:`lines`,positions:A(`street-positions`,s,`float32x2`,i.vertexCount),lineOffsets:A(`street-offsets`,c,`uint32`,i.pathCount+1)},...p?{right:{kind:`lines`,positions:A(`rail-positions`,l,`float32x2`,o.vertexCount),lineOffsets:A(`rail-offsets`,u,`uint32`,o.pathCount+1)}}:{},sameFeatureOnly:m,spatialSort:f.crossSpatialSort,output:{pairs:{leftIds:A(`pair-left`,b,`uint32`,Ue),rightIds:A(`pair-right`,x,`uint32`,Ue),status:{count:A(`count`,S,`uint32`,1),overflow:A(`overflow`,C,`uint32`,1),requiredCount:A(`total`,w,`uint32`,1)}},kinds:M,points:A(`points`,E,`float32x2`,Ue),status:{uncertainCount:A(`uncertain`,O,`uint32`,1)}}})),R(k,{id:`kind-filter`,invocationCount:Ue,bindings:[{name:`kinds`,view:M,type:`u32`,access:`read`},{name:`kindFilter`,view:h.importToGraph(k),type:`f32`,access:`read`},{name:`shown`,view:A(`shown`,D,`uint32`,Ue),type:`u32`,access:`read_write`}],body:`
  let kind = kinds[kindsOffset + index];
  var value = 0xffffffffu;
  if (kind >= 1u && kind <= 5u && ((u32(kindFilter[kindFilterOffset]) >> (kind - 1u)) & 1u) != 0u) {
    value = kind - 1u;
  }
  shown[shownOffset + index] = value;`});let N=r.track(new v(n,{id:`crossing-draw-${_}`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),P=i.vertexCount-i.pathCount,F=new L(r,`crossings-${_}`,[{buffer:S,size:4},{buffer:C,size:4},{buffer:w,size:4},{buffer:O,size:4},{buffer:T,size:Ue*4}],e=>{let n=new Uint32Array(e),r=Math.min(n[0],Ue),i=[0,0,0,0,0,0];for(let e of n.subarray(4,4+r))e<i.length&&i[e]++;t.setReadout(`crossPairs`,`${d(n[2])} intersecting pairs among ${d(P)} street segments${p?` and the L routes`:``}`),t.setReadout(`crossKinds`,[1,2,3,4,5].map(e=>`${ze[e]} ${d(i[e])}`).join(`, `)),t.setReadout(`crossFlags`,`${n[1]?`OVERFLOW`:`no overflow`}, ${d(n[3])} uncertain`)}),I={key:_,compiled:r.track(await k.compileAsync()),kinds:T,points:E,shown:D,count:S,overflow:C,total:w,uncertain:O,drawCommands:N,reader:F,segments:P};return g.set(_,I),I},x=e=>{let t={proper:1,touches:6,overlaps:8,all:31}[e.crossKinds];h.write(Float32Array.of(t)),y=!0},S=(e,t)=>{_=e,x(t),y=!0};S(await b(t.options),t.options);let C=r.track(new I);return{getCompiledGraphs:()=>[...g.values()].map(e=>e.compiled),setOption(e,n){e===`crossKinds`?x(n):C.request({build:()=>b(n),commit:e=>{S(e,n),t.requestLayers()}})},encode(e){_&&(y?(_.compiled.encode(e,{parameters:void 0}),K(e,_.count,_.drawCommands),_.reader.request(e),y=!1):_.reader.flush(e))},getLayers(){if(!_)return[];let n=t.options,r=t.theme()===`dark`,i=[new z({id:`crossing-streets`,coordinateSystem:Le,coordinateOrigin:e.coordinateOrigin,positions:p.positions,pathOffsets:p.offsets,pathOffsetCount:p.offsetCount,vertexCount:p.vertexCount,drawCommands:p.drawCommands,color:r?[150,160,180,90]:[90,100,120,100],widthPixels:.8})];return n.crossMode===`streets-rail`&&i.push(new z({id:`crossing-rail`,coordinateSystem:Le,coordinateOrigin:e.coordinateOrigin,positions:m.positions,pathOffsets:m.offsets,pathOffsetCount:m.offsetCount,vertexCount:m.vertexCount,drawCommands:m.drawCommands,color:r?[120,200,255,220]:[20,100,170,220],widthPixels:2})),i.push(new F({id:`crossing-points`,coordinateSystem:Le,coordinateOrigin:e.coordinateOrigin,positions:_.points,drawCommands:_.drawCommands,values:_.shown,valueFormat:`uint32`,colormap:`category`,palette:Be,noDataValue:4294967295,noDataColor:[0,0,0,0],radiusPixels:n.crossKinds===`proper`?4.5:2.6})),i},destroy(){for(let e of g.values())e.reader.stop();r.destroy()}}}async function Ge(e){let{ctx:t}=e,{device:n}=t,r=new f(n,`noding`),i=null,o=null,s=null,c=null,l=[],u=!0,p=!0,m=async r=>{let o=`${r.nodingInput}|${r.nodingCapacity}|${r.nodingSpatialSort}`;if(i?.key===o)return null;let s=new f(n,`noding-${o}`),c=e.buildNoding(r.nodingInput),l=c.pathCount,m=c.vertexCount,h=m-l,g=Number(r.nodingCapacity),_=h*3+4096,y=m+2*_,b=m+_,x=s.createBuffer(`line-positions`,c.local),S=s.createBuffer(`line-offsets`,c.offsets),C=ne(s,`noding-lines`,c.local,c.offsets),w=s.createBuffer(`split-line-ids`,_*4),T=s.createBuffer(`split-offsets`,(_+1)*4),E=s.createBuffer(`split-positions`,y*8),D=s.createBuffer(`split-count`,4),O=s.createBuffer(`split-vertices`,4),k=s.createBuffer(`split-overflow`,4),A=s.createBuffer(`split-total`,4),j=s.createBuffer(`split-uncertain`,4),M=s.createBuffer(`chain-offsets`,(_+1)*4),N=s.createBuffer(`chain-positions`,y*8),P=s.createBuffer(`chain-count`,4),I=s.createBuffer(`chain-vertices`,4),B=s.track(new v(n,{id:`split-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),V=s.track(new v(n,{id:`chain-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),H=new a(n,{id:`noding-split-${o}`});{let t=G(H),n=t(`split-offsets`,T,`uint32`,_+1),i=t(`split-positions`,E,`float32x2`,y);H.add(new de({spatialContext:e.projection.spatialContext,id:`split`,lines:{kind:`lines`,positions:t(`line-positions`,x,`float32x2`,m),lineOffsets:t(`line-offsets`,S,`uint32`,l+1)},intersectionCapacity:g,spatialSort:r.nodingSpatialSort,output:{pieces:{geometry:{kind:`lines`,positions:i,lineOffsets:n},sourceIds:t(`split-line-ids`,w,`uint32`,_),status:{count:t(`split-count`,D,`uint32`,1),overflow:t(`split-overflow`,k,`uint32`,1),requiredCount:t(`split-total`,A,`uint32`,1),vertexCount:t(`split-vertices`,O,`uint32`,1)}},status:{uncertainCount:t(`split-uncertain`,j,`uint32`,1)}}}));let a=t(`chain-offsets`,M,`uint32`,_+1),o=t(`chain-count`,P,`uint32`,1);H.add(new me({spatialContext:e.projection.spatialContext,id:`merge`,positions:i,lineOffsets:n,output:{chainOffsets:a,positions:t(`chain-positions`,N,`float32x2`,y),status:{count:o}}})),R(H,{id:`chain-vertex-count`,invocationCount:1,bindings:[{name:`offsets`,view:a,type:`u32`,access:`read`},{name:`count`,view:o,type:`u32`,access:`read`},{name:`vertices`,view:t(`chain-vertices`,I,`uint32`,1),type:`u32`,access:`read_write`}],body:`vertices[verticesOffset] = offsets[offsetsOffset + min(count[countOffset], ${_}u)];`})}let U=s.track(await H.compileAsync()),W=s.createParameterBuffer(`tolerance`,`float32`,1),ee=s.createBuffer(`net-line-ids`,_*4),te=s.createBuffer(`net-offsets`,(_+1)*4),q=s.createBuffer(`net-positions`,y*8),J=s.createBuffer(`net-count`,4),Y=s.createBuffer(`net-vertices`,4),re=s.createBuffer(`net-total`,4),ie=s.createBuffer(`node-positions`,b*8),ae=s.createBuffer(`node-count`,4),X=s.createBuffer(`node-total`,4),oe=s.createBuffer(`from-nodes`,_*4),se=s.createBuffer(`to-nodes`,_*4),ce=s.createBuffer(`edge-lengths`,_*4),le=s.createBuffer(`csr-offsets`,(b+1)*4),ue=s.createBuffer(`csr-neighbors`,_*8),fe=s.createBuffer(`csr-weights`,_*8),pe=s.createBuffer(`net-overflow`,4),he=s.createBuffer(`net-uncertain`,4),ge=s.createBuffer(`node-degrees`,b*4),_e=s.track(new v(n,{id:`net-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),Z=s.track(new v(n,{id:`node-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),Q=new a(n,{id:`noding-network-${o}`});{let t=G(Q),n=t(`node-count`,ae,`uint32`,1),i=t(`csr-offsets`,le,`uint32`,b+1);Q.add(new Pe({spatialContext:e.projection.spatialContext,id:`network`,lines:{kind:`lines`,positions:t(`line-positions`,x,`float32x2`,m),lineOffsets:t(`line-offsets`,S,`uint32`,l+1)},intersectionCapacity:g,spatialSort:r.nodingSpatialSort,tolerance:W.importToGraph(Q),output:{pieces:{geometry:{kind:`lines`,positions:t(`net-positions`,q,`float32x2`,y),lineOffsets:t(`net-offsets`,te,`uint32`,_+1)},sourceIds:t(`net-line-ids`,ee,`uint32`,_),status:{count:t(`net-count`,J,`uint32`,1),overflow:t(`net-piece-overflow`,k,`uint32`,1),requiredCount:t(`net-total`,re,`uint32`,1),vertexCount:t(`net-vertices`,Y,`uint32`,1)}},nodes:{positions:t(`node-positions`,ie,`float32x2`,b),status:{count:n,requiredCount:t(`node-total`,X,`uint32`,1)}},edges:{fromNodes:t(`from-nodes`,oe,`uint32`,_),toNodes:t(`to-nodes`,se,`uint32`,_),lengths:t(`edge-lengths`,ce,`float32`,_)},topology:{offsets:i,neighbors:t(`csr-neighbors`,ue,`uint32`,_*2),weights:t(`csr-weights`,fe,`float32`,_*2)},status:{uncertainCount:t(`net-uncertain`,he,`uint32`,1),overflow:t(`net-overflow`,pe,`uint32`,1)}}})),R(Q,{id:`node-degree`,invocationCount:b,bindings:[{name:`offsets`,view:i,type:`u32`,access:`read`},{name:`count`,view:n,type:`u32`,access:`read`},{name:`degrees`,view:t(`node-degrees`,ge,`uint32`,b),type:`u32`,access:`read_write`}],body:`
  var value = 0xffffffffu;
  if (index < count[countOffset]) {
    value = min(offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index], 6u) - 1u;
  }
  degrees[degreesOffset + index] = value;`})}let ve=s.track(await Q.compileAsync()),ye=e=>W.write(Float32Array.of(e)),be=new L(s,`noding-a`,[{buffer:D,size:4},{buffer:k,size:4},{buffer:A,size:4},{buffer:j,size:4},{buffer:P,size:4}],e=>{let n=new Uint32Array(e);t.setReadout(`nodingPieces`,`${d(n[0])} pieces from ${d(l)} lines${n[1]?` (OVERFLOW: raise the intersection capacity)`:``}`),t.setReadout(`nodingMerged`,`${d(n[4])} chains`),t.setReadout(`nodingUncertain`,d(n[3]))}),xe=new L(s,`noding-b`,[{buffer:J,size:4},{buffer:ae,size:4},{buffer:pe,size:4},{buffer:ge,size:b*4}],e=>{let n=new Uint32Array(e),r=[,,,,,,].fill(0);for(let e of n.subarray(3,3+Math.min(n[1],b)))e<6&&r[e]++;t.setReadout(`nodingEdges`,`${d(n[0])} edges between ${d(n[1])} nodes${n[2]?` (OVERFLOW)`:``}`),t.setReadout(`nodingDegrees`,`dead ends ${d(r[0])}, degree 2 ${d(r[1])}, 3 ${d(r[2])}, 4 ${d(r[3])}, 5+ ${d(r[4]+r[5])}`)}),Se=`${d(l)} polylines (${c.routeCount} routes${c.duplicatesRemoved?`, ${d(c.duplicatesRemoved)} exact duplicates removed`:``}), ${d(m)} vertices`;return{key:o,resources:s,compiledA:U,compiledB:ve,readers:[be,xe],toleranceWrite:ye,inputReadout:Se,encodeFor:e=>{let n=t.options;u?(U.encode(e,{parameters:void 0}),K(e,O,B),K(e,I,V),be.request(e),u=!1):be.flush(e),(n.nodingOutput===`network`||p)&&(p||n.nodingOutput===`network`)&&(ve.encode(e,{parameters:void 0}),K(e,Y,_e),K(e,ae,Z),p&&xe.request(e),p=!1),xe.flush(e)},layersFor:()=>{let n=t.options,r=t.theme()===`dark`,i=[];return i.push(new z({id:`noding-input`,coordinateSystem:Le,coordinateOrigin:e.coordinateOrigin,positions:C.positions,pathOffsets:C.offsets,pathOffsetCount:C.offsetCount,vertexCount:C.vertexCount,drawCommands:C.drawCommands,color:r?[150,160,180,70]:[90,100,120,70],widthPixels:6})),n.nodingOutput===`pieces`?i.push(new z({id:`noding-pieces`,coordinateSystem:Le,coordinateOrigin:e.coordinateOrigin,positions:E,pathOffsets:T,pathOffsetCount:_+1,vertexCount:O,drawCommands:B,colorSource:`path-index`,color:[255,255,255,255],widthPixels:2.6})):n.nodingOutput===`merged`?i.push(new z({id:`noding-chains`,coordinateSystem:Le,coordinateOrigin:e.coordinateOrigin,positions:N,pathOffsets:M,pathOffsetCount:_+1,vertexCount:I,drawCommands:V,colorSource:`path-index`,color:[255,255,255,255],widthPixels:3})):i.push(new z({id:`noding-edges`,coordinateSystem:Le,coordinateOrigin:e.coordinateOrigin,positions:q,pathOffsets:te,pathOffsetCount:_+1,vertexCount:Y,drawCommands:_e,colorSource:`path-index`,color:[255,255,255,200],widthPixels:2.2}),new F({id:`noding-nodes`,coordinateSystem:Le,coordinateOrigin:e.coordinateOrigin,positions:ie,drawCommands:Z,values:ge,valueFormat:`uint32`,colormap:`category`,palette:Ve,noDataValue:4294967295,noDataColor:[0,0,0,0],radiusPixels:4.5})),i}}},h=e=>{if(e){for(let t of e.readers)t.stop();e.resources.destroy()}},g=e=>{if(!e)return;let n=r,a=l;setTimeout(()=>{for(let e of a)e.stop();n.destroy()},500),r=e.resources,l=e.readers,i={key:e.key,compiledA:e.compiledA,compiledB:e.compiledB},c=e.toleranceWrite,s=e.encodeFor,o=e.layersFor,p=!0,u=!0,t.setReadout(`nodingInput`,e.inputReadout),c(t.options.nodingTolerance)};g(await m(t.options));let _=new I;return{getCompiledGraphs:()=>i?[i.compiledA,i.compiledB]:[],setOption(e,n){e===`nodingInput`||e===`nodingCapacity`||e===`nodingSpatialSort`?_.request({build:()=>m(n),commit:e=>{g(e),t.requestLayers()},discard:h}):e===`nodingTolerance`&&(c?.(n.nodingTolerance),p=!0)},encode(e){s?.(e)},getLayers:()=>o?o():[],destroy(){_.destroy();for(let e of l)e.stop();r.destroy()}}}var $=8192;async function Ke(e){let{ctx:t}=e,{device:n}=t,r=new f(n,`dissolve`),{layout:i,groups:o}=e.getCounties(),s=Ie(i),c=s.count,l=c+2*$,u=r.createBuffer(`endpoints`,s.endpoints),p=r.createBuffer(`segment-groups`,c*4),m=r.createBuffer(`county-groups`,i.featureCount*4),h=W(i),g=r.createBuffer(`fill-corners`,h.corners),_=r.createBuffer(`fill-rows`,h.featureRows),y=q(i),b=r.createBuffer(`outline-starts`,y.starts),x=r.createBuffer(`outline-ends`,y.ends),S=new Map,C=null,T=!0,E=e=>{let t=e.dissolveBy===`state`?o.state:e.dissolveBy===`rucc`?o.rucc:new Uint32Array(i.featureCount),n=new Uint32Array(c);for(let e=0;e<c;e++)n[e]=t[s.featureRows[e]];p.write(n),m.write(Float32Array.from(t)),T=!0},D=async e=>{let i=[e.cancelOpposing,e.splitTouching,e.vertexTolerance,e.interiorSide,e.normalizeWinding].join(`|`),o=S.get(i);if(o)return o;let s=r.createBuffer(`ring-offsets-${i}`,($+1)*4),f=r.createBuffer(`positions-${i}`,l*8),m=r.createBuffer(`count-${i}`,4),h=r.createBuffer(`overflow-${i}`,4),g=r.createBuffer(`total-${i}`,4),_=r.createBuffer(`open-${i}`,4),y=r.createBuffer(`touching-${i}`,4),b=r.createBuffer(`ring-groups-${i}`,$*4),x=r.createBuffer(`ring-holes-${i}`,$*4),C=r.createBuffer(`ring-areas-${i}`,$*4),T=r.createBuffer(`flags-${i}`,c*4),E=r.createBuffer(`vertices-${i}`,4),D=r.createBuffer(`group-values-${i}`,$*4),O=r.createBuffer(`hole-values-${i}`,$*4),k=new a(n,{id:`dissolve-${i}`}),A=G(k),j=A(`ring-offsets`,s,`uint32`,$+1),M=A(`count`,m,`uint32`,1),N=A(`ring-groups`,b,`uint32`,$),P=A(`ring-holes`,x,`uint32`,$);k.add(new w({id:`dissolve`,endpoints:A(`endpoints`,u,`float32x4`,c),groups:A(`segment-groups`,p,`uint32`,c),spatialContext:{coordinateSpace:`longitude-latitude`,datum:`EPSG:4326`,metric:`none`},vertexTolerance:Number(e.vertexTolerance),interiorSide:e.interiorSide,normalizeWinding:e.normalizeWinding,cancelOpposingSegments:e.cancelOpposing,splitTouchingRings:e.splitTouching,output:{ringOffsets:j,positions:A(`positions`,f,`float32x2`,l),ringAreas:A(`ring-areas`,C,`float32`,$),ringIsHole:P,ringGroups:N,segmentFlags:A(`flags`,T,`uint32`,c),status:{count:M,overflow:A(`overflow`,h,`uint32`,1),requiredCount:A(`total`,g,`uint32`,1),openSegmentCount:A(`open`,_,`uint32`,1),touchingSegmentCount:A(`touching`,y,`uint32`,1)}}})),R(k,{id:`ring-values`,invocationCount:$,bindings:[{name:`groups`,view:N,type:`u32`,access:`read`},{name:`holes`,view:P,type:`u32`,access:`read`},{name:`groupValues`,view:A(`group-values`,D,`float32`,$),type:`f32`,access:`read_write`},{name:`holeValues`,view:A(`hole-values`,O,`float32`,$),type:`f32`,access:`read_write`}],body:`
  groupValues[groupValuesOffset + index] = f32(groups[groupsOffset + index]);
  holeValues[holeValuesOffset + index] = f32(holes[holesOffset + index]);`}),R(k,{id:`ring-vertex-count`,invocationCount:1,bindings:[{name:`offsets`,view:j,type:`u32`,access:`read`},{name:`count`,view:M,type:`u32`,access:`read`},{name:`vertices`,view:A(`vertices`,E,`uint32`,1),type:`u32`,access:`read_write`}],body:`vertices[verticesOffset] = offsets[offsetsOffset + min(count[countOffset], ${$}u)];`});let F=r.track(new v(n,{id:`dissolve-draw-${i}`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),I=new L(r,`dissolve-${i}`,[{buffer:m,size:4},{buffer:h,size:4},{buffer:g,size:4},{buffer:_,size:4},{buffer:y,size:4},{buffer:E,size:4},{buffer:x,size:$*4},{buffer:T,size:c*4}],e=>{let n=new Uint32Array(e),r=Math.min(n[0],$),i=0;for(let e of n.subarray(6,6+r))i+=e;let a=n.subarray(6+$),o=0,s=0,l=0;for(let e of a)e&8&&o++,e&2&&s++,e&4&&l++;t.setReadout(`dissolveRings`,`${d(n[2])} rings (${d(r-i)} shells, ${d(i)} holes), ${d(n[5])} vertices${n[1]?`, OVERFLOW`:``}`),t.setReadout(`dissolveSegments`,`${d(c)} in, ${d(o)} cancelled, ${d(n[3])} on no ring, ${d(n[4])} at touching vertices`),t.setReadout(`dissolveFlags`,`${d(s)} dangling, ${d(l)} conflicts`)}),z={key:i,compiled:r.track(await k.compileAsync()),ringOffsets:s,positions:f,count:m,vertices:E,groupValues:D,holeValues:O,drawCommands:F,reader:I};return S.set(i,z),z};C=await D(t.options),E(t.options);let O=r.track(new I);return{getCompiledGraphs:()=>[...S.values()].map(e=>e.compiled),setOption(e,n){e===`dissolveBy`&&E(n),(e===`cancelOpposing`||e===`splitTouching`||e===`vertexTolerance`||e===`interiorSide`||e===`normalizeWinding`)&&O.request({build:()=>D(n),commit:e=>{C=e,T=!0,t.requestLayers()}})},encode(e){C&&(T?(C.compiled.encode(e,{parameters:void 0}),K(e,C.vertices,C.drawCommands),C.reader.request(e),T=!1):C.reader.flush(e))},getLayers(){if(!C)return[];let e=t.options,n=t.theme()===`dark`,r=[];return e.showFill&&r.push(new H({id:`dissolve-fill`,coordinateSystem:Re,corners:g,featureRows:_,instanceCount:h.triangleCount,values:m,valueMapping:`category`,color:[255,255,255,255],opacity:.55})),r.push(new V({id:`dissolve-county-edges`,coordinateSystem:Re,starts:b,ends:x,instanceCount:y.edgeCount,color:n?[255,255,255,60]:[20,30,50,60],widthPixels:.6}),new z({id:`dissolve-rings`,coordinateSystem:Re,positions:C.positions,pathOffsets:C.ringOffsets,pathOffsetCount:$+1,vertexCount:C.vertices,drawCommands:C.drawCommands,values:e.dissolveColor===`hole`?C.holeValues:null,colorSource:e.dissolveColor===`hole`?`path-value`:`uniform`,valueMapping:`flag`,color:e.dissolveColor===`hole`?[226,96,80,255]:n?[255,255,255,255]:[10,20,40,255],secondaryColor:n?[255,255,255,255]:[10,20,40,255],widthPixels:2.2})),r},destroy(){for(let e of S.values())e.reader.stop();r.destroy()}}}async function qe(e){let{ctx:t}=e,{device:n}=t,o=new f(n,`generalise`),{layout:s}=e.getCounties(),{vertexCount:c,ringCount:l,partCount:u}=s,p=o.createBuffer(`positions`,ie(s.lngLat,ae)),m=o.createBuffer(`ring-offsets`,s.ringOffsets),h=o.createBuffer(`polygon-offsets`,s.polygonOffsets),g=o.createBuffer(`feature-offsets`,s.featureOffsets),_=new Uint32Array(c);for(let e=0;e<l;e++)for(let t=s.ringOffsets[e];t<s.ringOffsets[e+1];t++)_[t]=e;let y=o.createBuffer(`vertex-rings`,_),b=o.createParameterBuffer(`tolerance`,`float32`,4),x=ne(o,`original-rings`,s.lngLat,s.ringOffsets),S=W(s),C=o.createBuffer(`fill-corners`,S.corners),w=o.createBuffer(`fill-rows`,S.featureRows),T=o.createBuffer(`fill-values`,Float32Array.from({length:s.featureCount},(e,t)=>t%7)),E=o.createBuffer(`importance`,c*4),D=o.createBuffer(`importance-converged`,4),O=o.createBuffer(`importance-rounds`,4),k=o.createBuffer(`kept-ids`,c*4),A=o.createBuffer(`kept-count`,4),j=o.createBuffer(`kept-overflow`,4),P=o.track(new v(n,{id:`kept-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),F=new Map,V=async e=>{let t=e.coverageRounds,s=F.get(t);if(s)return s;let u=Number(e.coverageRounds),d=e=>{let t=G(e);return{imp:t,base:{positions:t(`positions`,p,`float32x2`,c),lineOffsets:t(`ring-offsets`,m,`uint32`,l+1),importance:t(`importance`,E,`float32`,c)}}},f=new a(n,{id:`independent-importance-${t}`});{let{imp:e,base:t}=d(f);f.add(new N({spatialContext:r(i),id:`independent-importance`,...t,maximumIterations:u,output:{status:{converged:e(`converged`,D,`uint32`,1),iterationCount:e(`rounds`,O,`uint32`,1)}}}))}let h=new a(n,{id:`independent-selection-${t}`});{let{imp:e,base:t}=d(h);h.add(new N({spatialContext:r(i),id:`independent-selection`,...t,computeImportance:!1,parameters:b.importToGraph(h),output:{ids:e(`kept-ids`,k,`uint32`,c),status:{count:e(`kept-count`,A,`uint32`,1),overflow:e(`kept-overflow`,j,`uint32`,1)}}}))}return s={importance:o.track(await f.compileAsync()),selection:o.track(await h.compileAsync()),encoded:!1},F.set(t,s),s},U=new Map,ee=async e=>{let f=`${e.topologyRounds}|${e.coverageSnap}|${e.coverageRounds}`,_=U.get(f);if(_)return _;let y=o.createBuffer(`out-positions-${f}`,c*8),x=o.createBuffer(`out-rings-${f}`,(l+1)*4),S=o.createBuffer(`keep-mask-${f}`,c*4),C=o.createBuffer(`overflow-${f}`,4),w=o.createBuffer(`converged-${f}`,4),T=o.createBuffer(`topology-${f}`,16),E=o.createBuffer(`vertices-${f}`,4),O=new a(n,{id:`coverage-${f}`}),k=G(O),M=k(`out-rings`,x,`uint32`,l+1),N=k(`out-positions`,y,`float32x2`,c);O.add(new Se({spatialContext:r(i),id:`coverage`,positions:k(`positions`,p,`float32x2`,c),featureOffsets:k(`feature-offsets`,g,`uint32`,s.featureCount+1),polygonOffsets:k(`polygon-offsets`,h,`uint32`,u+1),ringOffsets:k(`ring-offsets`,m,`uint32`,l+1),snapTolerance:Number(e.coverageSnap),parameters:b.importToGraph(O),maximumIterations:Number(e.coverageRounds),topologyIterationCount:Number(e.topologyRounds),output:{positions:N,ringOffsets:M,outputMask:k(`keep-mask`,S,`uint32`,c),topologyStatistics:k(`topology`,T,`uint32`,4),status:{converged:k(`converged`,w,`uint32`,1),overflow:k(`overflow`,C,`uint32`,1)}}})),R(O,{id:`coverage-vertex-count`,invocationCount:1,bindings:[{name:`offsets`,view:M,type:`u32`,access:`read`},{name:`vertices`,view:k(`vertices`,E,`uint32`,1),type:`u32`,access:`read_write`}],body:`vertices[verticesOffset] = offsets[offsetsOffset + ${l}u];`});let P=o.track(new v(n,{id:`coverage-draw-${f}`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),F=new L(o,`coverage-${f}`,[{buffer:E,size:4},{buffer:w,size:4},{buffer:C,size:4},{buffer:T,size:16},{buffer:A,size:4},{buffer:j,size:4},{buffer:D,size:4}],e=>{let n=new Uint32Array(e);t.setReadout(`coverageKept`,`${d(n[0])} of ${d(c)} vertices (${(100*n[0]/c).toFixed(1)}%)${n[2]?`, OVERFLOW`:``}`),t.setReadout(`coverageConverged`,n[1]?`yes`:`no: stopped at the round cap`),t.setReadout(`coverageTopology`,`${d(n[3])} crossings found, ${d(n[4])} remain, ${d(n[5])} vertices restored${n[6]?`, pair capacity overflow`:``}`),t.setReadout(`independentKept`,`${d(n[7])} of ${d(c)} vertices (${(100*n[7]/c).toFixed(1)}%)`)}),I={key:f,compiled:o.track(await O.compileAsync()),outPositions:y,outRings:x,vertices:E,topology:T,converged:w,draw:P,reader:F};return U.set(f,I),I},q=null,J=null,Y=NaN,re=!0,X=new Float32Array(4),oe=async e=>({coverage:await ee(e),independentEntry:await V(e)}),se=e=>{q=e.coverage,J=e.independentEntry,Y=NaN,re=!0};se(await oe(t.options));let ce=o.track(new I);return t.setReadout(`coverageInput`,`${d(s.featureCount)} counties, ${d(l)} rings, ${d(c)} vertices`),{getCompiledGraphs(){let e=[];for(let t of U.values())e.push(t.compiled);for(let t of F.values())e.push(t.importance,t.selection);return e},setOption(e,n){(e===`topologyRounds`||e===`coverageSnap`||e===`coverageRounds`)&&ce.request({build:()=>oe(n),commit:e=>{se(e),t.requestLayers()}})},encode(e){if(!q||!J)return;let n=Math.fround(10**t.options.coverageTolerance);J.encoded||(J.importance.encode(e,{parameters:void 0}),J.encoded=!0,Y=NaN),n!==Y&&(Y=n,b.write(M({tolerance:n},X)),q.compiled.encode(e,{parameters:void 0}),J.selection.encode(e,{parameters:void 0}),K(e,q.vertices,q.draw),K(e,A,P),t.setReadout(`coverageTolerance`,`${te(n)} on Albers CONUS (scale error up to 1.4%)`),re=!0),re?(q.reader.request(e),re=!1):q.reader.flush(e)},getLayers(){if(!q)return[];let e=t.options,n=t.theme()===`dark`,r=[new H({id:`generalise-fill`,coordinateSystem:Re,corners:C,featureRows:w,instanceCount:S.triangleCount,values:T,valueMapping:`category`,color:[255,255,255,255],opacity:n?.18:.3})];return e.coverageOutline===`original`?r.push(new z({id:`generalise-original`,coordinateSystem:Re,positions:x.positions,pathOffsets:x.offsets,pathOffsetCount:x.offsetCount,vertexCount:x.vertexCount,drawCommands:x.drawCommands,color:n?[255,255,255,200]:[20,30,50,200],widthPixels:1})):r.push(new z({id:`generalise-original-faint`,coordinateSystem:Re,positions:x.positions,pathOffsets:x.offsets,pathOffsetCount:x.offsetCount,vertexCount:x.vertexCount,drawCommands:x.drawCommands,color:n?[255,255,255,40]:[20,30,50,40],widthPixels:.6})),J&&(e.coverageOutline===`independent`||e.coverageOutline===`both`)&&r.push(new B({id:`generalise-independent`,coordinateSystem:Re,positions:x.positions,keptIds:k,vertexLines:y,keptCount:A,drawCommands:P,color:[226,96,80,255],widthPixels:1.6})),(e.coverageOutline===`coverage`||e.coverageOutline===`both`)&&r.push(new z({id:`generalise-coverage`,analysisFrame:i,positions:q.outPositions,pathOffsets:q.outRings,pathOffsetCount:l+1,vertexCount:q.vertices,drawCommands:q.draw,closed:!0,color:[30,130,190,255],widthPixels:1.6})),r},destroy(){for(let e of U.values())e.reader.stop();o.destroy()}}}export{He as createNodingAndCoverage};