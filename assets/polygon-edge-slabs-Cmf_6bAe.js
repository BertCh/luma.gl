import{l as e,w as t}from"./resources-CwvhEOBe.js";import{t as n}from"./gpu-scan-BuGhTLDQ.js";import{n as r,s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as a}from"./wgsl-common-BNRVzgd8.js";import{t as o}from"./scan-total-BdolCtK7.js";var s=4096,c=`
const EDGE_SLAB_NO_VERTEX: u32 = 0xffffffffu;
const EDGE_SLAB_PARAMS_STRIDE: u32 = 6u;
const EDGE_SLAB_COUNT_WORD: u32 = 0u;
const EDGE_SLAB_BASE_WORD: u32 = 1u;
const EDGE_SLAB_ORIGIN_WORD: u32 = 2u;
const EDGE_SLAB_SCALE_WORD: u32 = 3u;
const EDGE_SLAB_VERTEX_BEGIN_WORD: u32 = 4u;
const EDGE_SLAB_VERTEX_END_WORD: u32 = 5u;

// Monotone in y for a positive scale (subtract, multiply, floor, clamp). Do not rewrite it as an
// fma or in any other form: edges are listed by this exact expression.
fn edgeSlabOf(y: f32, slabCount: u32, origin: f32, scale: f32) -> u32 {
  if (slabCount <= 1u) { return 0u; }
  return min(u32(clamp(floor((y - origin) * scale), 0.0, f32(slabCount - 1u))), slabCount - 1u);
}
`;function l(e){return[{name:`edgeSlabParams`,view:e.params,type:`u32`,access:`read`},{name:`edgeSlabOffsets`,view:e.slabOffsets,type:`u32`,access:`read`},{name:`edgeSlabEntries`,view:e.slabEntries,type:`u32`,access:`read`}]}function u(e,t){if(!t)return e;let n={edgeSlabParams:t.params.array,edgeSlabParamsOffset:t.params.offset,edgeSlabOffsets:t.offsets.array,edgeSlabOffsetsOffset:t.offsets.offset,edgeSlabEntries:t.entries.array,edgeSlabEntriesOffset:t.entries.offset};return e.replace(/\bedgeSlab(?:Params|Offsets|Entries)(?:Offset)?\b/g,e=>n[e])}function d(e,t){return u(`${c}
const EDGE_SLAB_ENTRY_CAPACITY: u32 = ${e.entryCapacity}u;
const EDGE_SLAB_VERTEX_COUNT: u32 = ${e.vertexCount}u;

fn edgeSlabIsSlabbed(owner: u32) -> bool {
  let ownerBase = edgeSlabParamsOffset + owner * EDGE_SLAB_PARAMS_STRIDE;
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  return slabCount != 0u &&
    edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD] + slabCount] <= EDGE_SLAB_ENTRY_CAPACITY;
}

fn edgeSlabOfHeight(owner: u32, y: f32) -> u32 {
  let ownerBase = edgeSlabParamsOffset + owner * EDGE_SLAB_PARAMS_STRIDE;
  return edgeSlabOf(
    y,
    edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD],
    bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD]),
    bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD])
  );
}
`,t)}function f(e){let[t,n]=e.indexNames??[`edgeStart`,`edgeEnd`],[r,i]=e.pointNames??[`a`,`b`],a=e.getVertex?`let ${r} = ${e.getVertex}(${t});
      let ${i} = ${e.getVertex}(${n});`:``;return u(`{
    let edgeSlabOwner = ${e.owner};
    let edgeSlabOwnerBase = edgeSlabParamsOffset + edgeSlabOwner * EDGE_SLAB_PARAMS_STRIDE;
    let edgeSlabIsWalked = !edgeSlabIsSlabbed(edgeSlabOwner);
    var edgeSlabFirst = 0u;
    var edgeSlabLast = 0u;
    if (!edgeSlabIsWalked) {
      edgeSlabFirst = edgeSlabOfHeight(edgeSlabOwner, min(${e.lowY}, ${e.highY}));
      edgeSlabLast = edgeSlabOfHeight(edgeSlabOwner, max(${e.lowY}, ${e.highY}));
    }
    let edgeSlabBase = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_BASE_WORD];
    for (var edgeSlabIndex = edgeSlabFirst; edgeSlabIndex <= edgeSlabLast; edgeSlabIndex++) {
      var edgeSlabBegin = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_VERTEX_BEGIN_WORD];
      var edgeSlabEnd = edgeSlabParams[edgeSlabOwnerBase + EDGE_SLAB_VERTEX_END_WORD];
      var edgeSlabShift = 0u;
      if (!edgeSlabIsWalked) {
        edgeSlabBegin = edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabBase + edgeSlabIndex];
        edgeSlabEnd = edgeSlabOffsets[edgeSlabOffsetsOffset + edgeSlabBase + edgeSlabIndex + 1u];
        edgeSlabShift = EDGE_SLAB_VERTEX_COUNT;
      }
      for (var edgeSlabEntry = edgeSlabBegin; edgeSlabEntry < edgeSlabEnd; edgeSlabEntry++) {
        let ${t} = edgeSlabEntries[edgeSlabEntriesOffset + 2u * (edgeSlabShift + edgeSlabEntry)];
        if (${t} == EDGE_SLAB_NO_VERTEX) { continue; }
        let ${n} = edgeSlabEntries[edgeSlabEntriesOffset + 2u * (edgeSlabShift + edgeSlabEntry) + 1u];
        ${a}
        ${e.dedupe?`if (!edgeSlabIsWalked && edgeSlabIndex != max(edgeSlabFirst, edgeSlabOfHeight(edgeSlabOwner, ${e.dedupe.edgeLowY}))) { continue; }\n        `:``}${e.body}
      }
    }
  }`,e.storage)}function p(e){return e===`feature`?`let firstPolygon = featureOffsets[featureOffsetsOffset + index];
  let endPolygon = featureOffsets[featureOffsetsOffset + index + 1u];`:`let firstPolygon = index;
  let endPolygon = index + 1u;`}function m(){return`
fn edgeSlabGetVertex(vertex: u32) -> vec2f {
  let vertexBase = positionsOffset + 2u * vertex;
  return vec2f(positions[vertexBase], positions[vertexBase + 1u]);
}
`}function h(e){if(e.owner===`feature`&&!e.featureOffsets)throw Error(`${e.operation} polygon edge slabs with owner 'feature' need featureOffsets`);if(e.polygonOffsets.length<1||e.ringOffsets.length<1)throw Error(`${e.operation} polygon edge slabs need terminal polygon and ring offsets`);if(e.entryCapacity!==void 0&&!(e.entryCapacity>=1))throw Error(`${e.operation} polygon edge slab entryCapacity must be at least 1`)}function g(l,u){h(u);let{id:d,operation:f,owner:g,positions:_,polygonOffsets:v,ringOffsets:y}=u,b=u.featureOffsets,x=_.length,S=(g===`feature`?b.length:v.length)-1,C=Math.floor(x/4)+1,w=Math.min(u.entryCapacity??Math.max(4*x,1024),Math.max(0,e(l.device,8,x))),T=u.minimumVertexCount??8,E=y.length-1,D=v.length-1,O=t(l,`${d}-slab-params`,`uint32`,Math.max(S,1)*6),k=t(l,`${d}-slab-numbers`,`uint32`,S+1),A=t(l,`${d}-slab-starts`,`uint32`,S+1),j=t(l,`${d}-ring-owner`,`uint32`,Math.max(E,1)),M=t(l,`${d}-vertex-owner`,`uint32`,x),N=t(l,`${d}-slab-counters`,`uint32`,C+1),P=t(l,`${d}-slab-offsets`,`uint32`,C+1),F=t(l,`${d}-slab-entries`,`uint32x2`,x+w),I={owner:g,ownerCount:S,vertexCount:x,entryCapacity:w,slabCapacity:C,params:O,slabOffsets:P,slabEntries:F},L=[{name:`positions`,view:_,type:`f32`,access:`read`},...g===`feature`?[{name:`featureOffsets`,view:b,type:`u32`,access:`read`}]:[],{name:`polygonOffsets`,view:v,type:`u32`,access:`read`},{name:`ringOffsets`,view:y,type:`u32`,access:`read`}],R=i(l,{id:`${d}-slab-params`,operation:f,variant:`slab-params`,bindings:[...L,{name:`edgeSlabParams`,view:O,type:`u32`,access:`read_write`},{name:`slabNumbers`,view:k,type:`u32`,access:`read_write`},...u.gate?[{name:`gateCounts`,view:u.gate.counts,type:`u32`,access:`read`}]:[]],invocationCount:S+1,declarations:`${a}${c}${m()}
const OWNER_COUNT: u32 = ${S}u;
const MINIMUM_VERTEX_COUNT: u32 = ${T}u;
const MAXIMUM_SLABS: u32 = ${s}u;
const ENTRY_BUDGET_PER_VERTEX: f32 = ${Math.max(0,w/Math.max(x,1)).toFixed(8)};
fn edgeSlabIsFinite(p: vec2f) -> bool {
  return isFiniteBits(p.x) && isFiniteBits(p.y) && abs(p.x) < 1e30 && abs(p.y) < 1e30;
}`,body:`if (index == OWNER_COUNT) { slabNumbers[slabNumbersOffset + index] = 0u; return; }
  ${p(g)}
  var vertexBegin = 0u;
  var vertexEnd = 0u;
  var vertexCount = 0u;
  var isFirstRing = true;
  var hasNonFinite = false;
  var verticalTravel = 0.0;
  var minY = 1e30;
  var maxY = -1e30;
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      if (isFirstRing) {
        vertexBegin = firstVertex;
        isFirstRing = false;
      }
      vertexEnd = endVertex;
      if (endVertex > firstVertex) {
        vertexCount += endVertex - firstVertex;
      }
      var previousY = 0.0;
      if (endVertex > firstVertex) { previousY = edgeSlabGetVertex(endVertex - 1u).y; }
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let p = edgeSlabGetVertex(vertex);
        if (edgeSlabIsFinite(p)) {
          minY = min(minY, p.y);
          maxY = max(maxY, p.y);
          verticalTravel += abs(p.y - previousY);
        } else {
          hasNonFinite = true;
        }
        previousY = p.y;
      }
    }
  }
  ${u.gate?`let isGated = gateCounts[gateCountsOffset + index] >= ${u.gate.minimum}u;`:`let isGated = true;`}
  var slabCount = 0u;
  var slabScale = 0.0;
  if (isGated && !hasNonFinite && vertexCount >= MINIMUM_VERTEX_COUNT) {
    // An edge contributes at most 2 + abs(dy) * scale entries. Reserve two entries per
    // vertex, and use half the remaining budget for travel (rounding/accumulation headroom).
    // Vertex count alone chose 4096 slabs even for jagged rings, overflowing 4N entries and
    // turning the refine into a full indirect walk. All statistics here are GPU-resident.
    let travelBudget = max(0.0, ENTRY_BUDGET_PER_VERTEX - 2.0) * 0.5 * f32(vertexCount);
    var wanted = min(vertexCount / 4u, MAXIMUM_SLABS);
    if (verticalTravel > 0.0) {
      wanted = min(wanted, max(1u, u32(floor(travelBudget * ((maxY - minY) / verticalTravel)))));
    }
    let scale = f32(wanted) / (maxY - minY);
    if (scale > 0.0 && scale < 1e30) {
      slabCount = wanted;
      slabScale = scale;
    }
  }
  let ownerBase = edgeSlabParamsOffset + index * EDGE_SLAB_PARAMS_STRIDE;
  edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] = slabCount;
  edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD] = bitcast<u32>(minY);
  edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD] = bitcast<u32>(slabScale);
  edgeSlabParams[ownerBase + EDGE_SLAB_VERTEX_BEGIN_WORD] = vertexBegin;
  edgeSlabParams[ownerBase + EDGE_SLAB_VERTEX_END_WORD] = vertexEnd;
  slabNumbers[slabNumbersOffset + index] = slabCount;`}),z=i(l,{id:`${d}-slab-bases`,operation:f,variant:`slab-bases`,bindings:[{name:`edgeSlabParams`,view:O,type:`u32`,access:`read_write`},{name:`slabStarts`,view:A,type:`u32`,access:`read`}],invocationCount:Math.max(S,1),declarations:`${c}
const OWNER_COUNT: u32 = ${S}u;
const SLAB_CAPACITY: u32 = ${C}u;`,body:`if (index >= OWNER_COUNT) { return; }
  let ownerBase = edgeSlabParamsOffset + index * EDGE_SLAB_PARAMS_STRIDE;
  let base = slabStarts[slabStartsOffset + index];
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  if (slabCount != 0u && base + slabCount > SLAB_CAPACITY) {
    edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] = 0u;
  }
  edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD] = base;`}),B=i(l,{id:`${d}-ring-owner`,operation:f,variant:`ring-owner`,bindings:[...g===`feature`?[{name:`featureOffsets`,view:b,type:`u32`,access:`read`}]:[],{name:`polygonOffsets`,view:v,type:`u32`,access:`read`},{name:`ringOwner`,view:j,type:`u32`,access:`read_write`}],invocationCount:Math.max(E,1),declarations:`const OWNER_COUNT: u32 = ${S}u;
const POLYGON_COUNT: u32 = ${D}u;
const RING_COUNT: u32 = ${E}u;
const NO_OWNER: u32 = 0xffffffffu;`,body:`if (index >= RING_COUNT) { return; }
  var polygon = 0u;
  var high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  var ringOwnerValue = NO_OWNER;
  if (POLYGON_COUNT > 0u && index >= polygonOffsets[polygonOffsetsOffset + polygon] &&
      index < polygonOffsets[polygonOffsetsOffset + polygon + 1u]) {
    ${g===`feature`?`var low = 0u;
    high = OWNER_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (featureOffsets[featureOffsetsOffset + middle] <= polygon) {
        low = middle;
      } else {
        high = middle;
      }
    }
    if (OWNER_COUNT > 0u && polygon >= featureOffsets[featureOffsetsOffset + low] &&
        polygon < featureOffsets[featureOffsetsOffset + low + 1u]) {
      ringOwnerValue = low;
    }`:`ringOwnerValue = polygon;`}
  }
  ringOwner[ringOwnerOffset + index] = ringOwnerValue;`}),V=e=>r(l,{id:`${d}-slab-clear-${e}`,operation:f,view:N,type:`u32`,value:`0u`}),H=`${c}${m()}
const RING_COUNT: u32 = ${E}u;
const ENTRY_CAPACITY: u32 = ${w}u;
const VERTEX_COUNT: u32 = ${x}u;

/** Slab range [first, last] of the edge a-b within the owner's lattice. */
fn edgeSlabGetEdgeRange(ownerBase: u32, a: vec2f, b: vec2f) -> vec2u {
  let slabCount = edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD];
  let origin = bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_ORIGIN_WORD]);
  let scale = bitcast<f32>(edgeSlabParams[ownerBase + EDGE_SLAB_SCALE_WORD]);
  return vec2u(
    edgeSlabOf(min(a.y, b.y), slabCount, origin, scale),
    edgeSlabOf(max(a.y, b.y), slabCount, origin, scale)
  );
}`,U=i(l,{id:`${d}-slab-edges-count`,operation:f,variant:`slab-edges-count`,bindings:[{name:`positions`,view:_,type:`f32`,access:`read`},{name:`edgeSlabParams`,view:O,type:`u32`,access:`read`},{name:`ringOffsets`,view:y,type:`u32`,access:`read`},{name:`ringOwner`,view:j,type:`u32`,access:`read`},{name:`vertexOwner`,view:M,type:`u32`,access:`read_write`},{name:`edgeSlabEntries`,view:F,type:`u32`,access:`read_write`},{name:`slabCounters`,view:N,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.max(x,1),declarations:H,body:`if (index >= VERTEX_COUNT) { return; }
  var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  var ownerIndex = EDGE_SLAB_NO_VERTEX;
  var next = EDGE_SLAB_NO_VERTEX;
  if (RING_COUNT > 0u && index >= ringBegin && index < ringEnd) {
    ownerIndex = ringOwner[ringOwnerOffset + ring];
    if (ownerIndex != EDGE_SLAB_NO_VERTEX) {
      next = select(index + 1u, ringBegin, index + 1u == ringEnd);
    }
  }
  edgeSlabEntries[edgeSlabEntriesOffset + 2u * index] = select(EDGE_SLAB_NO_VERTEX, index, next != EDGE_SLAB_NO_VERTEX);
  edgeSlabEntries[edgeSlabEntriesOffset + 2u * index + 1u] = next;
  vertexOwner[vertexOwnerOffset + index] = select(EDGE_SLAB_NO_VERTEX, ownerIndex, next != EDGE_SLAB_NO_VERTEX);
  if (next == EDGE_SLAB_NO_VERTEX) { return; }
  let ownerBase = edgeSlabParamsOffset + ownerIndex * EDGE_SLAB_PARAMS_STRIDE;
  if (edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] == 0u) { return; }
  let edgeRange = edgeSlabGetEdgeRange(ownerBase, edgeSlabGetVertex(index), edgeSlabGetVertex(next));
  let base = edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD];
  for (var slab = edgeRange.x; slab <= edgeRange.y; slab++) {
    atomicAdd(&slabCounters[slabCountersOffset + base + slab], 1u);
  }`}),W=i(l,{id:`${d}-slab-edges-fill`,operation:f,variant:`slab-edges-fill`,bindings:[{name:`positions`,view:_,type:`f32`,access:`read`},{name:`edgeSlabParams`,view:O,type:`u32`,access:`read`},{name:`vertexOwner`,view:M,type:`u32`,access:`read`},{name:`edgeSlabOffsets`,view:P,type:`u32`,access:`read`},{name:`edgeSlabEntries`,view:F,type:`u32`,access:`read_write`},{name:`slabCounters`,view:N,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.max(x,1),declarations:H,body:`if (index >= VERTEX_COUNT) { return; }
  let ownerIndex = vertexOwner[vertexOwnerOffset + index];
  if (ownerIndex == EDGE_SLAB_NO_VERTEX) { return; }
  let ownerBase = edgeSlabParamsOffset + ownerIndex * EDGE_SLAB_PARAMS_STRIDE;
  if (edgeSlabParams[ownerBase + EDGE_SLAB_COUNT_WORD] == 0u) { return; }
  let next = edgeSlabEntries[edgeSlabEntriesOffset + 2u * index + 1u];
  let edgeRange = edgeSlabGetEdgeRange(ownerBase, edgeSlabGetVertex(index), edgeSlabGetVertex(next));
  let base = edgeSlabParams[ownerBase + EDGE_SLAB_BASE_WORD];
  for (var slab = edgeRange.x; slab <= edgeRange.y; slab++) {
    let slabIndex = base + slab;
    let position = edgeSlabOffsets[edgeSlabOffsetsOffset + slabIndex] +
      atomicAdd(&slabCounters[slabCountersOffset + slabIndex], 1u);
    if (position < ENTRY_CAPACITY) {
      edgeSlabEntries[edgeSlabEntriesOffset + 2u * (VERTEX_COUNT + position)] = index;
      edgeSlabEntries[edgeSlabEntriesOffset + 2u * (VERTEX_COUNT + position) + 1u] = next;
    }
  }`}),G=t(l,`${d}-slab-entry-total`,`uint32`,1);return{nodes:[R,...new n({id:`${d}-slab-number-scan`,input:k,output:A,mode:`exclusive`}).getCommandNodes(l),z,B,V(`count`),U,...new n({id:`${d}-slab-offset-scan`,input:N,output:P,mode:`exclusive`}).getCommandNodes(l),...o(l,{id:`${d}-slab-entry`,operation:f,counts:N,offsets:P,rowCount:C,total:G}),i(l,{id:`${d}-slab-entry-fallback`,operation:f,variant:`slab-entry-fallback`,bindings:[{name:`scanTotal`,view:G,type:`u32`,access:`read`},{name:`edgeSlabParams`,view:O,type:`u32`,access:`read_write`}],invocationCount:S,body:`if (scanTotal[scanTotalOffset] == 0xffffffffu) {
    edgeSlabParams[edgeSlabParamsOffset + index * 6u] = 0u;
  }`}),V(`cursor`),W],slabs:I}}export{d as i,f as n,g as r,l as t};