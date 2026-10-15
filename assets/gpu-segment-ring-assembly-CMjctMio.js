import{E as e}from"./analysis-frame-Blu0tYdw.js";import{_ as t,h as n,j as r,w as i,y as a}from"./resources-CwvhEOBe.js";import{t as o}from"./gpu-scan-BuGhTLDQ.js";import{t as s}from"./gpu-sort-Bl3Dztdb.js";import{l as c,s as l}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as u}from"./wgsl-common-BNRVzgd8.js";import{t as d}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{n as f,t as p}from"./exact-geometry-wgsl-DOqxBQzc.js";var m=8388608,h=1.0625;function g(e,t,n){return`${u}
const TOLERANCE_CELL: f32 = ${e};
const CELL_SLACK: f32 = ${h};
const CELL_BIAS: f32 = ${m}.0;
fn ringOrigin() -> vec2f { return vec2f(endpoints[endpointsOffset], endpoints[endpointsOffset + 1u]); }
// Largest |x - originX| or |y - originY| of any endpoint, measured on the GPU.
fn ringFrameExtent() -> f32 { return bitcast<f32>(${t?`atomicLoad(&counters[countersOffset + ${n}u])`:`counters[countersOffset + ${n}u]`}); }
fn ringInverseCell() -> f32 {
  // The 1e-30 floor keeps a zero extent with the derived (zero) tolerance finite.
  return 1.0 / (CELL_SLACK * max(max(TOLERANCE_CELL, ringFrameExtent() * ${2**-22}), 1.0e-30));
}
fn ringQuantize(value: f32, origin: f32) -> u32 {
  let offset = value - origin;
  if (!isFiniteValue(offset)) { return u32(CELL_BIAS); }
  return u32(floor(offset * ringInverseCell()) + CELL_BIAS);
}
fn ringHash(qx: u32, qy: u32, group: u32) -> u32 {
  var h = qx * 0x9E3779B1u;
  h = h ^ (h >> 15u);
  h = (h + qy * 0x85EBCA77u) * 0xC2B2AE3Du;
  h = h ^ (h >> 13u);
  h = (h + group * 0x27D4EB2Fu) * 0x165667B1u;
  h = h ^ (h >> 16u);
  return select(h, 0xfffffffeu, h == 0xffffffffu);
}`}var _={coordinateSpaces:[`planar`,`longitude-latitude`],metrics:[`none`,`native`]},v=`GPUSegmentRingAssembly`,y=64,ee=4294967295,te=class{id;props;spatialContext;constructor(t){this.id=t.id??`segment-ring-assembly`,this.spatialContext=e(this.id,t.spatialContext,_),this.props=t;let i=this.id,{output:o}=t;a(t.endpoints,[`float32x4`],`${i} endpoints`);let s=t.endpoints.length;if(s<1)throw Error(`${i} needs at least one segment row`);if(t.vertexTolerance!==void 0&&!(t.vertexTolerance>0))throw Error(`${i} vertexTolerance must be positive`);if(t.interiorSide!==void 0&&![`left`,`right`].includes(t.interiorSide))throw Error(`${i} interiorSide must be 'left' or 'right'`);for(let[e,n]of[[`count`,t.count],[`output.status.count`,o.status.count],[`output.status.overflow`,o.status.overflow],[`output.status.requiredCount`,o.status.requiredCount],[`output.status.openSegmentCount`,o.status.openSegmentCount],[`output.status.touchingSegmentCount`,o.status.touchingSegmentCount]])if(n&&(r(n,`${i} ${e}`),n.length<1))throw Error(`${i} ${e} must contain one uint32 row`);if(t.groups&&(r(t.groups,`${i} groups`),t.groups.length!==s))throw Error(`${i} groups length must equal the segment row count`);if(o.ringGroups&&!t.groups)throw Error(`${i} output.ringGroups needs the groups input`);if(r(o.ringOffsets,`${i} output.ringOffsets`),o.ringOffsets.length<2)throw Error(`${i} output.ringOffsets needs at least two rows (one ring)`);if(a(o.positions,[`float32x2`],`${i} output.positions`),o.positions.length<1)throw Error(`${i} output.positions must not be empty`);let c=o.ringOffsets.length-1;for(let[e,t]of[[`ringIsHole`,o.ringIsHole],[`ringShells`,o.ringShells],[`ringGroups`,o.ringGroups]])if(t&&(r(t,`${i} output.${e}`),t.length!==c))throw Error(`${i} output.${e} length must equal the ring capacity`);if(o.ringAreas&&(a(o.ringAreas,[`float32`],`${i} output.ringAreas`),o.ringAreas.length!==c))throw Error(`${i} output.ringAreas length must equal the ring capacity`);if(o.polygons){let{polygons:e}=o;if(a(e.positions,[`float32x2`],`${i} output.polygons.positions`),e.positions.length!==o.positions.length)throw Error(`${i} output.polygons.positions must match output.positions length`);if(e.sourceIds){if(!t.groups)throw Error(`${i} output.polygons.sourceIds needs the groups input`);if(r(e.sourceIds,`${i} output.polygons.sourceIds`),e.sourceIds.length!==c)throw Error(`${i} output.polygons.sourceIds must have ${c} rows (the ring capacity)`)}for(let[t,n,a]of[[`ringOffsets`,e.ringOffsets,c+1],[`polygonOffsets`,e.polygonOffsets,c+1],[`featureOffsets`,e.featureOffsets,c+1]])if(r(n,`${i} output.polygons.${t}`),n.length!==a)throw Error(`${i} output.polygons.${t} must have ${a} rows`)}for(let[e,t]of[[`segmentRings`,o.segmentRings],[`segmentVertices`,o.segmentVertices],[`segmentFlags`,o.segmentFlags]])if(t&&(r(t,`${i} output.${e}`),t.length!==s))throw Error(`${i} output.${e} length must equal the segment row count`);n(i,ne(o),[t.endpoints,t.count,t.groups])}getCommandNodes(e){let{id:n,props:r}=this,{output:a}=r;t(n,e,[r.endpoints,r.count,r.groups,...ne(a)]);let m=r.endpoints.length,h=a.ringOffsets.length-1,_=a.positions.length,te=r.vertexTolerance??0,b=r.interiorSide??`left`,re=!!r.normalizeWinding&&b===`right`,ie=r.normalizeWinding||b===`left`?1:-1,ae=this.spatialContext.coordinateSpace===`longitude-latitude`,x=!!r.groups,S=`${ee}u`,oe=(t,r,a)=>i(e,`${n}-${t}`,r,a),C=(e,t=m)=>oe(e,`uint32`,t),se=C(`keys`),ce=C(`ids`),w=C(`sorted-keys`),T=C(`sorted-ids`),E=C(`claim`),D=!!r.cancelOpposingSegments,O=C(`counters`,6),k=C(`next`),le=C(`alternatives`),ue=C(`swapped`),A=a.segmentFlags??C(`flags`),j=C(`leader`),de=C(`leader-flags`),fe=C(`leader-vertices`),M=C(`ring-scan`),N=C(`vertex-scan`),P=C(`ring-starts`),pe=C(`ring-leaders`),F=C(`ring-count`,1),me=a.segmentRings??C(`segment-rings`),he=a.segmentVertices??C(`segment-vertices`),ge=C(`close-vertices`),I=[C(`pointer-a`),C(`pointer-b`)],L=[C(`value-a`),C(`value-b`)],R=c(te),_e=r.vertexTolerance===void 0?`fn ringTolerance() -> f32 { return ringFrameExtent() * ${c(2**-21)}; }`:`fn ringTolerance() -> f32 { return ${R}; }`,ve=g(R,!1,4),ye=g(R,!0,4),be=e=>`min(${e}[${e}Offset], ${m}u)`,z=r.count?[{name:`rowCount`,view:r.count,type:`u32`,access:`read`}]:[],B=r.groups?[{name:`groups`,view:r.groups,type:`u32`,access:`read`}]:[],V=r.count?be(`rowCount`):`${m}u`,H=e=>x?`groups[groupsOffset + ${e}]`:`0u`,U=(t,r,i,a,o=``)=>l(e,{id:`${n}-${t}`,operation:v,variant:t,bindings:r,invocationCount:i,declarations:o,body:a}),W=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),G=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),K=[];K.push(U(`quantize-init`,[...z,G(`counters`,O)],1,`for (var slot = 0u; slot < 5u; slot++) { counters[countersOffset + slot] = 0u; }
  counters[countersOffset + 5u] = ${V};`),U(`quantize-extent`,[W(`endpoints`,r.endpoints,`f32`),...z,{name:`counters`,view:O,type:`atomic<u32>`,access:`read_write`}],m,`if (index >= ${V}) { return; }
  let origin = vec2f(endpoints[endpointsOffset], endpoints[endpointsOffset + 1u]);
  let start = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]) - origin;
  let end = vec2f(endpoints[endpointsOffset + 4u * index + 2u], endpoints[endpointsOffset + 4u * index + 3u]) - origin;
  if (!isFiniteValue(start.x) || !isFiniteValue(start.y) || !isFiniteValue(end.x) || !isFiniteValue(end.y)) {
    atomicAdd(&counters[countersOffset + 3u], 1u);
    return;
  }
  let extent = max(max(abs(start.x), abs(start.y)), max(abs(end.x), abs(end.y)));
  atomicMax(&counters[countersOffset + 4u], bitcast<u32>(extent));`,u)),K.push(U(`hash`,[W(`endpoints`,r.endpoints,`f32`),...z,...B,G(`keys`,se),G(`ids`,ce),G(`claim`,E),G(`counters`,O)],m,`ids[idsOffset + index] = index;
  claim[claimOffset + index] = 0xffffffffu;
  var key = 0xffffffffu;
  if (index < ${V}) {
    let start = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]);
    let origin = ringOrigin();
    key = ringHash(ringQuantize(start.x, origin.x), ringQuantize(start.y, origin.y), ${H(`index`)});
  }
  keys[keysOffset + index] = key;`,ve)),K.push(...new s({id:`${n}-sort`,keys:se,values:ce,outputKeys:w,outputValues:T}).getCommandNodes(e)),D&&K.push(U(`cancel`,[W(`endpoints`,r.endpoints,`f32`),...z,...B,W(`sortedKeys`,w),W(`sortedIds`,T),G(`flags`,A),{name:`counters`,view:O,type:`atomic<u32>`,access:`read_write`}],m,`flags[flagsOffset + index] = 0u;
  if (index >= ${V}) { return; }
  let partner = findReverse(index);
  if (partner != ${S} && findReverse(partner) == index) {
    flags[flagsOffset + index] = 8u;
    atomicAdd(&counters[countersOffset + 2u], 1u);
  }`,`${ye}
${_e}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}
// Lowest segment that starts where 'index' ends and ends where it starts.
fn findReverse(index: u32) -> u32 {
  let valid = ${V};
  let origin = ringStart(index);
  let vertex = ringEnd(index);
  let group = ${H(`index`)};
  let cellOrigin = ringOrigin();
  let qx = ringQuantize(vertex.x, cellOrigin.x);
  let qy = ringQuantize(vertex.y, cellOrigin.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var best = ${S};
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let hash = ringHash(qx + bitcast<u32>(dx), qy + bitcast<u32>(dy), group);
      var duplicate = false;
      for (var j = 0u; j < seenCount; j++) { duplicate = duplicate || seen[j] == hash; }
      if (duplicate) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${m}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${m}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate == index) { continue; }
        let candidateStart = ringStart(candidate);
        let candidateEnd = ringEnd(candidate);
        if (abs(candidateStart.x - vertex.x) > ringTolerance() || abs(candidateStart.y - vertex.y) > ringTolerance()) { continue; }
        if (abs(candidateEnd.x - origin.x) > ringTolerance() || abs(candidateEnd.y - origin.y) > ringTolerance()) { continue; }
        if (${x?`groups[groupsOffset + candidate] != group`:`false`}) { continue; }
        best = min(best, candidate);
      }
    }
  }
  return best;
}`));let xe=b===`left`?`1.0`:`-1.0`;K.push(U(`match`,[W(`endpoints`,r.endpoints,`f32`),W(`counters`,O),...B,W(`sortedKeys`,w),W(`sortedIds`,T),G(`next`,k),G(`flags`,A),G(`alt`,le)],m,`next[nextOffset + index] = ${S};
  alt[altOffset + index] = ${S};
  ${D?`let carried = flags[flagsOffset + index] & 8u;
  flags[flagsOffset + index] = carried;
  if (carried != 0u) { return; }`:`flags[flagsOffset + index] = 0u;`}
  let valid = min(counters[countersOffset + 5u], ${m}u);
  if (index >= valid) { return; }
  let origin = ringStart(index);
  let vertex = ringEnd(index);
  let group = ${H(`index`)};
  let incoming = ringScaled(vertex - origin, vertex.y);
  let cellOrigin = ringOrigin();
  let qx = ringQuantize(vertex.x, cellOrigin.x);
  let qy = ringQuantize(vertex.y, cellOrigin.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var best = ${S};
  var bestTurn = 0.0;
  var matches = 0u;
  var first = ${S};
  var second = ${S};
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let hash = ringHash(qx + bitcast<u32>(dx), qy + bitcast<u32>(dy), group);
      var duplicate = false;
      for (var j = 0u; j < seenCount; j++) { duplicate = duplicate || seen[j] == hash; }
      if (duplicate) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${m}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${m}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate == index) { continue; }
        ${r.pairedOpposites?`if (candidate == (index ^ 1u)) { continue; }`:``}
        ${D?`if ((flags[flagsOffset + candidate] & 8u) != 0u) { continue; }`:``}
        let candidateStart = ringStart(candidate);
        if (abs(candidateStart.x - vertex.x) > ringTolerance() || abs(candidateStart.y - vertex.y) > ringTolerance()) { continue; }
        if (${x?`groups[groupsOffset + candidate] != group`:`false`}) { continue; }
        matches++;
        if (first == ${S}) { first = candidate; } else if (second == ${S}) { second = candidate; }
        let outgoing = ringScaled(ringEnd(candidate) - candidateStart, vertex.y);
        let turn = SIDE * ringTurnKey(incoming.x * outgoing.y - incoming.y * outgoing.x, dot(incoming, outgoing));
        if (best == ${S} || turn > bestTurn || (turn == bestTurn && candidate < best)) {
          best = candidate;
          bestTurn = turn;
        }
      }
    }
  }
  next[nextOffset + index] = best;
  if (matches == 2u) { alt[altOffset + index] = select(first, second, best == first); }
  var flagBits = 0u;
  if (matches > 1u) { flagBits = 1u; }
  if (matches == 0u) { flagBits = 2u; }
  flags[flagsOffset + index] = flagBits;`,`${ve}
${_e}
const SIDE: f32 = ${xe};
// Monotone in the turn angle from the incoming to the outgoing direction (counter-clockwise
// positive, u-turn at +2). atan2 is avoided: it misbehaves on signed zeros under fast math.
fn ringTurnKey(cross: f32, dot: f32) -> f32 {
  let denominator = abs(cross) + abs(dot);
  if (denominator == 0.0) { return 0.0; }
  let ratio = cross / denominator;
  if (dot >= 0.0) { return ratio; }
  return select(-2.0, 2.0, cross >= 0.0) - ratio;
}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}
fn ringScaled(delta: vec2f, latitude: f32) -> vec2f {
  ${ae?`return vec2f(delta.x * max(cos(radians(latitude)), 0.01), delta.y);`:`return delta;`}
}`)),K.push(l(e,{id:`${n}-claim`,operation:v,variant:`claim`,bindings:[W(`next`,k),{name:`claim`,view:E,type:`atomic<u32>`,access:`read_write`}],invocationCount:m,body:`let claimed = next[nextOffset + index];
  if (claimed != ${S}) { atomicMin(&claim[claimOffset + claimed], index); }`}),l(e,{id:`${n}-resolve`,operation:v,variant:`resolve`,bindings:[G(`next`,k),W(`claim`,E),G(`flags`,A),{name:`counters`,view:O,type:`atomic<u32>`,access:`read_write`}],invocationCount:m,body:`let claimed = next[nextOffset + index];
  if (claimed != ${S} && claim[claimOffset + claimed] != index) {
    next[nextOffset + index] = ${S};
    flags[flagsOffset + index] = flags[flagsOffset + index] | 4u;
  }
  if ((flags[flagsOffset + index] & 1u) != 0u) {
    atomicAdd(&counters[countersOffset + 1u], 1u);
  }`}));let Se=Math.max(1,Math.ceil(Math.log2(m))),q=(t,r,i)=>{K.push(U(`jump-init-${t}`,[W(`next`,r),G(`pointer`,I[0]),G(`value`,L[0])],m,`pointer[pointerOffset + index] = next[nextOffset + index];
  value[valueOffset + index] = index;`));let a=0;for(let e=0;e<Se;e++)K.push(Te(`min`,`${t}-${e}`,a)),a=1-a;let o=1-a;K.push(l(e,{id:`${n}-cut-${t}`,operation:v,variant:`cut`,bindings:[W(`next`,r),W(`pointerIn`,I[a]),W(`valueIn`,L[a]),...z,G(`leader`,j),G(`pointerOut`,I[o]),G(`valueOut`,L[o]),{name:`counters`,view:O,type:`atomic<u32>`,access:`read_write`}],invocationCount:m,body:`let valid = ${V};
  var leaderIndex = ${S};
  var tail = ${S};
  var hopCount = 0u;
  if (index < valid) {
    if (pointerIn[pointerInOffset + index] != ${S}) {
      leaderIndex = valueIn[valueInOffset + index];
      tail = next[nextOffset + index];
      if (tail == leaderIndex) { tail = ${S}; }
      if (tail != ${S}) { hopCount = 1u; }
    } else {
      ${i?`atomicAdd(&counters[countersOffset], 1u);`:``}
    }
  }
  leader[leaderOffset + index] = leaderIndex;
  pointerOut[pointerOutOffset + index] = tail;
  valueOut[valueOutOffset + index] = hopCount;`})),a=o;for(let e=0;e<Se;e++)K.push(Te(`sum`,`${t}-${e}`,a)),a=1-a;return L[a]},J=q(1,k,r.splitTouchingRings===!1);r.splitTouchingRings!==!1&&(K.push(U(`split-touching`,[W(`next`,k),W(`alt`,le),W(`claim`,E),W(`leader`,j),G(`swapped`,ue)],m,`let chosen = next[nextOffset + index];
  var result = chosen;
  let other = alt[altOffset + index];
  if (chosen != ${S} && other != ${S}) {
    let partner = claim[claimOffset + other];
    if (partner != ${S} && partner != index &&
        alt[altOffset + partner] == chosen && next[nextOffset + partner] == other &&
        leader[leaderOffset + index] != ${S} &&
        leader[leaderOffset + index] == leader[leaderOffset + partner]) {
      result = other;
    }
  }
  swapped[swappedOffset + index] = result;`)),J=q(2,ue,!0)),K.push(U(`leaders`,[W(`leader`,j),W(`hops`,J),G(`leaderFlags`,de),G(`leaderVertices`,fe)],m,`let isLeader = leader[leaderOffset + index] == index;
  leaderFlags[leaderFlagsOffset + index] = select(0u, 1u, isLeader);
  leaderVertices[leaderVerticesOffset + index] = select(0u, hops[hopsOffset + index] + 2u, isLeader);`),...new o({id:`${n}-ring-scan`,input:de,output:M,mode:`inclusive`}).getCommandNodes(e),...new o({id:`${n}-vertex-scan`,input:fe,output:N,mode:`inclusive`}).getCommandNodes(e),U(`ring-starts`,[W(`leader`,j),W(`hops`,J),W(`ringScan`,M),W(`vertexScan`,N),G(`ringStarts`,P),G(`ringLeaders`,pe)],m,`if (leader[leaderOffset + index] != index) { return; }
  let ring = ringScan[ringScanOffset + index] - 1u;
  ringStarts[ringStartsOffset + ring] = vertexScan[vertexScanOffset + index] - (hops[hopsOffset + index] + 2u);
  ringLeaders[ringLeadersOffset + ring] = index;`),U(`ring-count`,[W(`ringScan`,M),W(`vertexScan`,N),W(`ringStarts`,P),G(`ringCount`,F)],1,`let total = ringScan[ringScanOffset + ${m-1}u];
  let vertices = vertexScan[vertexScanOffset + ${m-1}u];
  var low = 0u;
  var high = min(total, ${h}u);
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    var end = vertices;
    if (middle < total) { end = ringStarts[ringStartsOffset + middle]; }
    if (end <= ${_}u) { low = middle; } else { high = middle - 1u; }
  }
  ringCount[ringCountOffset] = low;`)),K.push(U(`place`,[W(`leader`,j),W(`hops`,J),W(`ringScan`,M),W(`vertexScan`,N),W(`ringCount`,F),G(`segmentRings`,me),G(`segmentVertices`,he),G(`closeVertices`,ge)],m,`segmentRings[segmentRingsOffset + index] = ${S};
  segmentVertices[segmentVerticesOffset + index] = ${S};
  closeVertices[closeVerticesOffset + index] = ${S};
  let head = leader[leaderOffset + index];
  if (head == ${S}) { return; }
  let ring = ringScan[ringScanOffset + head] - 1u;
  if (ring >= ringCount[ringCountOffset]) { return; }
  let ringLength = hops[hopsOffset + head] + 1u;
  let start = vertexScan[vertexScanOffset + head] - (ringLength + 1u);
  let position = hops[hopsOffset + head] - hops[hopsOffset + index];
  ${re?`let slot = (ringLength - position) % ringLength;`:`let slot = position;`}
  segmentRings[segmentRingsOffset + index] = ring;
  segmentVertices[segmentVerticesOffset + index] = start + slot;
  if (position == 0u) { closeVertices[closeVerticesOffset + index] = start + ringLength; }`),U(`write`,[W(`endpoints`,r.endpoints,`f32`),W(`segmentVertices`,he),W(`closeVertices`,ge),G(`positions`,a.positions,`f32`)],m,`let slot = segmentVertices[segmentVerticesOffset + index];
  if (slot == ${S}) { return; }
  let x = endpoints[endpointsOffset + 4u * index];
  let y = endpoints[endpointsOffset + 4u * index + 1u];
  positions[positionsOffset + 2u * slot] = x;
  positions[positionsOffset + 2u * slot + 1u] = y;
  let close = closeVertices[closeVerticesOffset + index];
  if (close != ${S}) {
    positions[positionsOffset + 2u * close] = x;
    positions[positionsOffset + 2u * close + 1u] = y;
  }`),U(`offsets`,[W(`ringStarts`,P),W(`ringScan`,M),W(`vertexScan`,N),W(`ringCount`,F),G(`ringOffsets`,a.ringOffsets)],h+1,`let total = ringScan[ringScanOffset + ${m-1}u];
  let written = min(index, ringCount[ringCountOffset]);
  var offset = vertexScan[vertexScanOffset + ${m-1}u];
  if (written < total) { offset = ringStarts[ringStartsOffset + written]; }
  ringOffsets[ringOffsetsOffset + index] = offset;`));let Y=a.ringAreas??oe(`ring-areas`,`float32`,h),X=a.ringIsHole??C(`ring-is-hole`,h),Ce=!!(a.ringShells??a.polygons),we=C(`ring-lowest-vertices`,h),Z=Ce?i(e,`${n}-ring-bounds`,`float32`,h*4):void 0;K.push(l(e,{id:`${n}-ring-stats`,operation:v,variant:`ring-stats`,bindings:[W(`ringOffsets`,a.ringOffsets),W(`positions`,a.positions,`f32`),W(`ringCount`,F),G(`areas`,Y,`f32`),G(`lowestVertices`,we),...Z?[G(`bounds`,Z,`f32`)]:[]],invocationCount:Math.ceil(h/y)*y,workgroupSize:y,guardIndex:!1,declarations:`const RING_CAPACITY: u32 = ${h}u;
const RING_LANES: u32 = ${y}u;
const BOUNDS_LARGE: f32 = 3.0e38;
var<workgroup> largeBits: array<atomic<u32>, 2>;
var<workgroup> largeBitsCopy: array<u32, 2>;
var<workgroup> largeFirst: array<u32, ${y}>;
var<workgroup> largeLast: array<u32, ${y}>;
var<workgroup> partialSum: array<f32, ${y}>;
var<workgroup> partialMinimum: array<vec2f, ${y}>;
var<workgroup> partialMaximum: array<vec2f, ${y}>;
var<workgroup> partialLowest: array<u32, ${y}>;
fn getVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}
// Lexicographic (y, x, row) order of vertex rows: the lowest vertex is an extreme point of the ring.
fn isLowerVertex(vertex: u32, other: u32) -> bool {
  let p = getVertex(vertex);
  let q = getVertex(other);
  return p.y < q.y || (p.y == q.y && (p.x < q.x || (p.x == q.x && vertex < other)));
}
fn publishRing(ring: u32, sum: f32, minimum: vec2f, maximum: vec2f, lowest: u32) {
  let area = 0.5 * sum;
  areas[areasOffset + ring] = area;
  lowestVertices[lowestVerticesOffset + ring] = lowest;
  ${Z?`bounds[boundsOffset + 4u * ring] = minimum.x; bounds[boundsOffset + 4u * ring + 1u] = minimum.y; bounds[boundsOffset + 4u * ring + 2u] = maximum.x; bounds[boundsOffset + 4u * ring + 3u] = maximum.y;`:``}
}`,body:`let lane = localInvocationIndex;
  let ringBase = workgroupIndex * RING_LANES;
  let ownRing = ringBase + lane;
  // Phase 1: this lane's own ring, serially when small; larger rings are queued for phase 2.
  if (ownRing < RING_CAPACITY) {
    if (ownRing >= ringCount[ringCountOffset]) {
      publishRing(ownRing, 0.0, vec2f(0.0), vec2f(0.0), 0u);
    } else {
      let first = ringOffsets[ringOffsetsOffset + ownRing];
      let last = ringOffsets[ringOffsetsOffset + ownRing + 1u];
      if (last - first <= RING_LANES) {
        let origin = getVertex(first);
        var sum = 0.0;
        var minimum = origin;
        var maximum = origin;
        var lowest = first;
        for (var vertex = first; vertex + 1u < last; vertex++) {
          let p = getVertex(vertex);
          let q = getVertex(vertex + 1u);
          minimum = min(minimum, q);
          maximum = max(maximum, q);
          if (isLowerVertex(vertex + 1u, lowest)) { lowest = vertex + 1u; }
          let a = p - origin;
          let b = q - origin;
          sum += a.x * b.y - b.x * a.y;
        }
        publishRing(ownRing, sum, minimum, maximum, lowest);
      } else {
        largeFirst[lane] = first;
        largeLast[lane] = last;
        atomicOr(&largeBits[lane / 32u], 1u << (lane % 32u));
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    largeBitsCopy[0] = atomicLoad(&largeBits[0]);
    largeBitsCopy[1] = atomicLoad(&largeBits[1]);
  }
  // Phase 2: every barrier below is in workgroup-uniform control flow (the queue is read through
  // workgroupUniformLoad), and the queued rings are visited in ascending order.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&largeBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let first = workgroupUniformLoad(&largeFirst[slot]);
      let last = workgroupUniformLoad(&largeLast[slot]);
      let origin = getVertex(first);
      var sum = 0.0;
      var minimum = vec2f(BOUNDS_LARGE);
      var maximum = vec2f(-BOUNDS_LARGE);
      var lowest = first;
      for (var vertex = first + lane; vertex + 1u < last; vertex += RING_LANES) {
        let p = getVertex(vertex);
        let q = getVertex(vertex + 1u);
        minimum = min(minimum, q);
        maximum = max(maximum, q);
        if (isLowerVertex(vertex + 1u, lowest)) { lowest = vertex + 1u; }
        let a = p - origin;
        let b = q - origin;
        sum += a.x * b.y - b.x * a.y;
      }
      partialSum[lane] = sum;
      partialMinimum[lane] = minimum;
      partialMaximum[lane] = maximum;
      partialLowest[lane] = lowest;
      workgroupBarrier();
      for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
          partialSum[lane] += partialSum[lane + stride];
          partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
          partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
          if (isLowerVertex(partialLowest[lane + stride], partialLowest[lane])) {
            partialLowest[lane] = partialLowest[lane + stride];
          }
        }
        workgroupBarrier();
      }
      if (lane == 0u) {
        publishRing(ringBase + slot, partialSum[0], min(partialMinimum[0], origin), max(partialMaximum[0], origin), partialLowest[0]);
      }
    }
  }`})),K.push(U(`ring-orientation`,[W(`ringOffsets`,a.ringOffsets),W(`positions`,a.positions,`f32`),W(`ringCount`,F),W(`areas`,Y,`f32`),W(`lowestVertices`,we),G(`isHole`,X)],h,`isHole[isHoleOffset + index] = 0u;
  if (index >= ringCount[ringCountOffset]) { return; }
  let first = ringOffsets[ringOffsetsOffset + index];
  let last = ringOffsets[ringOffsetsOffset + index + 1u];
  let area = areas[areasOffset + index];
  var orientation = i32(sign(area));
  // Closed ring: rows [first, last - 1) are its distinct vertex slots, row last - 1 repeats first.
  if (last >= first + 4u) {
    let slots = last - first - 1u;
    var lowest = lowestVertices[lowestVerticesOffset + index];
    if (lowest == last - 1u) { lowest = first; }
    let vertex = getRingVertex(lowest);
    var previous = lowest;
    var next = lowest;
    for (var step = 1u; step < slots; step++) {
      previous = select(previous - 1u, last - 2u, previous == first);
      if (any(getRingVertex(previous) != vertex)) { break; }
    }
    for (var step = 1u; step < slots; step++) {
      next = select(next + 1u, first, next == last - 2u);
      if (any(getRingVertex(next) != vertex)) { break; }
    }
    let turn = exactOrientSigns(
      1u,
      array<vec2f, 4>(getRingVertex(previous), vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(vertex, vec2f(0.0), vec2f(0.0), vec2f(0.0)),
      array<vec2f, 4>(getRingVertex(next), vec2f(0.0), vec2f(0.0), vec2f(0.0))
    ).signs.x;
    if (turn != 0) { orientation = turn; }
  }
  isHole[isHoleOffset + index] = select(0u, 1u, orientation * ${ie} < 0);`,`${d}
${p}
fn getRingVertex(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`));let Q=a.ringGroups??(x?C(`ring-groups`,h):void 0);Q&&r.groups&&K.push(U(`ring-groups`,[W(`ringLeaders`,pe),W(`ringCount`,F),...B,G(`ringGroups`,Q)],h,`ringGroups[ringGroupsOffset + index] = 0u;
  if (index < ringCount[ringCountOffset]) {
    ringGroups[ringGroupsOffset + index] = groups[groupsOffset + ringLeaders[ringLeadersOffset + index]];
  }`));let $=a.ringShells??(a.polygons?C(`ring-shells`,h):void 0);if($&&K.push(U(`ring-shells`,[W(`ringOffsets`,a.ringOffsets),W(`positions`,a.positions,`f32`),W(`ringCount`,F),W(`areas`,Y,`f32`),W(`isHole`,X),W(`ringBounds`,Z,`f32`),G(`shells`,$),...Q?[W(`ringGroups`,Q)]:[]],h,`shells[shellsOffset + index] = ${S};
  let rings = ringCount[ringCountOffset];
  if (index >= rings) { return; }
  if (isHole[isHoleOffset + index] == 0u) {
    shells[shellsOffset + index] = index;
    return;
  }
  let first = ringOffsets[ringOffsetsOffset + index];
  let a = getRingVertex(first);
  let b = getRingVertex(first + 1u);
  // The probe is the symbolic point infinitesimally beside the midpoint of the hole's first edge,
  // on the side away from the hole's interior (left of a clockwise hole), so it lies in the face
  // the hole bounds even when that edge runs along another ring. Every decision below is exact.
  // Holes wind against the shells (exact orientation), so a hole is clockwise exactly when shells
  // are counter-clockwise.
  let probe = ExactProbe(0.5 * (a + b), a, b, ${ie>0?1:-1});
  var best = ${S};
  var bestArea = 0.0;
  let holeArea = abs(areas[areasOffset + index]);
  for (var shell = 0u; shell < rings; shell++) {
    if (isHole[isHoleOffset + shell] != 0u) { continue; }
    ${Q?`if (ringGroups[ringGroupsOffset + shell] != ringGroups[ringGroupsOffset + index]) { continue; }`:``}
    let area = abs(areas[areasOffset + shell]);
    // A bidirected boundary produces an equal-area reverse walk for the unbounded side. It cannot
    // own itself as a hole; a containing shell must have strictly larger area.
    if (area <= holeArea) { continue; }
    if (best != ${S} && area >= bestArea) { continue; }
    // A probe outside the shell's bounding box (or left of it, where a closed ring crosses the ray
    // an even number of times) cannot be inside: skip the vertex walk.
    let shellBounds = vec4f(
      ringBounds[ringBoundsOffset + 4u * shell], ringBounds[ringBoundsOffset + 4u * shell + 1u],
      ringBounds[ringBoundsOffset + 4u * shell + 2u], ringBounds[ringBoundsOffset + 4u * shell + 3u]);
    let point = probe.point;
    if (point.y < shellBounds.y || point.y > shellBounds.w || point.x < shellBounds.x || point.x > shellBounds.z) { continue; }
    let shellFirst = ringOffsets[ringOffsetsOffset + shell];
    let shellLast = ringOffsets[ringOffsetsOffset + shell + 1u];
    let inside = (getShellWinding(probe, shellFirst, shellLast).x & 1) != 0;
    if (inside) {
      best = shell;
      bestArea = area;
    }
  }
  shells[shellsOffset + index] = best;`,`${d}
${p}
fn getRingVertex(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
${f(`getShellWinding`,`getRingVertex`)}`)),a.polygons&&$){let{polygons:t}=a,r=C(`polygon-keys`,h),i=C(`polygon-ids`,h),c=C(`sorted-polygon-keys`,h),l=C(`sorted-polygon-ids`,h),u=C(`polygon-lengths`,h),d=C(`polygon-shell-flags`,h),f=C(`polygon-length-scan`,h),p=C(`polygon-shell-scan`,h),m=C(`polygon-starts`,h),g=C(`polygon-valid-rings`,1);K.push(U(`polygon-keys`,[W(`ringCount`,F),W(`areas`,Y,`f32`),W(`isHole`,X),W(`shells`,$),G(`keys`,r),G(`ids`,i)],h,`ids[idsOffset + index] = index;
  var key = ${S};
  if (index < ringCount[ringCountOffset] && abs(areas[areasOffset + index]) > 0.0 && shells[shellsOffset + index] != ${S}) {
    key = shells[shellsOffset + index] * 2u + isHole[isHoleOffset + index];
  }
  keys[keysOffset + index] = key;`),...new s({id:`${n}-polygon-sort`,keys:r,values:i,outputKeys:c,outputValues:l}).getCommandNodes(e),U(`polygon-lengths`,[W(`ringOffsets`,a.ringOffsets),W(`sortedKeys`,c),W(`sortedIds`,l),G(`lengths`,u),G(`shellFlags`,d)],h,`let key = sortedKeys[sortedKeysOffset + index];
  let ring = sortedIds[sortedIdsOffset + index];
  let isValid = key != ${S};
  lengths[lengthsOffset + index] = select(0u, ringOffsets[ringOffsetsOffset + ring + 1u] - ringOffsets[ringOffsetsOffset + ring] - 1u, isValid);
  shellFlags[shellFlagsOffset + index] = select(0u, 1u, isValid && (key & 1u) == 0u);`),...new o({id:`${n}-polygon-length-scan`,input:u,output:f,mode:`inclusive`}).getCommandNodes(e),...new o({id:`${n}-polygon-shell-scan`,input:d,output:p,mode:`inclusive`}).getCommandNodes(e),U(`polygon-starts`,[W(`shellFlags`,d),W(`shellScan`,p),G(`polygonStarts`,m)],h,`if (shellFlags[shellFlagsOffset + index] != 0u) {
    polygonStarts[polygonStartsOffset + shellScan[shellScanOffset + index] - 1u] = index;
  }`),...t.sourceIds&&Q?[U(`polygon-groups`,[W(`shellScan`,p),W(`polygonStarts`,m),W(`sortedIds`,l),W(`ringGroups`,Q),G(`sourceIds`,t.sourceIds)],h,`var group = ${S};
  if (index < shellScan[shellScanOffset + ${h-1}u]) {
    group = ringGroups[ringGroupsOffset + sortedIds[sortedIdsOffset + polygonStarts[polygonStartsOffset + index]]];
  }
  sourceIds[sourceIdsOffset + index] = group;`)]:[],U(`polygon-counts`,[W(`sortedKeys`,c),G(`validRings`,g)],1,`var low = 0u;
  var high = ${h}u;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] == ${S}) { high = middle; } else { low = middle + 1u; }
  }
  validRings[validRingsOffset] = low;`),U(`polygon-offsets`,[W(`lengthScan`,f),W(`shellScan`,p),W(`polygonStarts`,m),W(`validRings`,g),G(`ringOffsetsOut`,t.ringOffsets),G(`polygonOffsetsOut`,t.polygonOffsets),G(`featureOffsets`,t.featureOffsets)],h+1,`let polygonCount = shellScan[shellScanOffset + ${h-1}u];
  // Identity feature table: one feature per polygon, padded flat with the polygon count.
  featureOffsets[featureOffsetsOffset + index] = min(index, polygonCount);
  var ringOffset = 0u;
  if (index > 0u) { ringOffset = lengthScan[lengthScanOffset + index - 1u]; }
  ringOffsetsOut[ringOffsetsOutOffset + index] = ringOffset;
  var polygonOffset = validRings[validRingsOffset];
  if (index < polygonCount) { polygonOffset = polygonStarts[polygonStartsOffset + index]; }
  polygonOffsetsOut[polygonOffsetsOutOffset + index] = polygonOffset;`),U(`polygon-gather`,[W(`sortedIds`,l),W(`ringOffsets`,a.ringOffsets),W(`positions`,a.positions,`f32`),W(`ringOffsetsOut`,t.ringOffsets),G(`positionsOut`,t.positions,`f32`)],_,`if (index >= ringOffsetsOut[ringOffsetsOutOffset + ${h}u]) { return; }
  var low = 0u;
  var high = ${h}u;
  while (low + 1u < high) {
    let middle = low + (high - low) / 2u;
    if (ringOffsetsOut[ringOffsetsOutOffset + middle] <= index) { low = middle; } else { high = middle; }
  }
  let ring = sortedIds[sortedIdsOffset + low];
  let source = ringOffsets[ringOffsetsOffset + ring] + (index - ringOffsetsOut[ringOffsetsOutOffset + low]);
  positionsOut[positionsOutOffset + 2u * index] = positions[positionsOffset + 2u * source];
  positionsOut[positionsOutOffset + 2u * index + 1u] = positions[positionsOffset + 2u * source + 1u];`))}return K.push(U(`publish`,[W(`ringScan`,M),W(`ringCount`,F),W(`counters`,O),G(`outCount`,a.status.count),G(`outOverflow`,a.status.overflow),...a.status.requiredCount?[G(`outTotal`,a.status.requiredCount)]:[]],1,`let total = ringScan[ringScanOffset + ${m-1}u];
  let written = ringCount[ringCountOffset];
  outCount[outCountOffset] = written;
  outOverflow[outOverflowOffset] = select(0u, 1u, written < total || counters[countersOffset + 3u] != 0u);
  ${a.status.requiredCount?`outTotal[outTotalOffset] = total;`:``}`)),(a.status.openSegmentCount||a.status.touchingSegmentCount)&&K.push(U(`publish-counters`,[W(`counters`,O),...a.status.openSegmentCount?[G(`outOpen`,a.status.openSegmentCount)]:[],...a.status.touchingSegmentCount?[G(`outTouching`,a.status.touchingSegmentCount)]:[]],1,`${a.status.openSegmentCount?`outOpen[outOpenOffset] = counters[countersOffset] - counters[countersOffset + 2u];`:``}
  ${a.status.touchingSegmentCount?`outTouching[outTouchingOffset] = counters[countersOffset + 1u];`:``}`)),K;function Te(t,r,i){let a=1-i;return l(e,{id:`${n}-${t}-jump-${r}`,operation:v,variant:`${t}-jump`,bindings:[W(`pointerIn`,I[i]),W(`valueIn`,L[i]),G(`pointerOut`,I[a]),G(`valueOut`,L[a])],invocationCount:m,body:`let successor = pointerIn[pointerInOffset + index];
  var value = valueIn[valueInOffset + index];
  var pointer = successor;
  if (successor != ${S}) {
    value = ${t===`min`?`min(value, valueIn[valueInOffset + successor])`:`value + valueIn[valueInOffset + successor]`};
    pointer = pointerIn[pointerInOffset + successor];
  }
  pointerOut[pointerOutOffset + index] = pointer;
  valueOut[valueOutOffset + index] = value;`})}}};function ne(e){let{kind:t,...n}=e.polygons??{};return[e.ringOffsets,e.positions,e.ringAreas,e.ringIsHole,e.ringShells,e.ringGroups,...Object.values(n),e.segmentRings,e.segmentVertices,e.segmentFlags,e.status.count,e.status.overflow,e.status.requiredCount,e.status.openSegmentCount,e.status.touchingSegmentCount]}export{te as t};