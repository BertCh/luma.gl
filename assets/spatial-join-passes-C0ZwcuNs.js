import{R as e,T as t,s as n,w as r}from"./resources-CwvhEOBe.js";import{t as i}from"./gpu-scan-BuGhTLDQ.js";import{t as a}from"./gpu-sort-Bl3Dztdb.js";import{t as o}from"./gpu-bvh-C68VuZY1.js";import{s}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{d as c,f as l,l as u}from"./wgsl-common-BNRVzgd8.js";import{n as d}from"./scan-total-BdolCtK7.js";var f=`
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const NO_FEATURE: u32 = 0xffffffffu;
${u}`;function p(e,t,r,i){if(!i)return;let a=n(r).map(e=>e.length),o=n(i).map(e=>e.length);if(a.length!==o.length||a.some((e,t)=>e!==o[t]))throw Error(`${e} ${t} must match the chunk topology of points`)}function m(e,r,i){let a=e=>e.flatMap(e=>e?n(e):[]),o=a(r),s=a(i);for(let[n,r]of s.entries())if(o.some(e=>t(e,r))||s.some((e,i)=>i!==n&&t(e,r)))throw Error(`${e} outputs must not overlap inputs or each other`)}function h(e){return(2*e-1)*24+e*4+8}function g(e,t,n,i,a,s,c){let l=2*a-1,u=new o({id:`${t}-bvh`,minima:n,maxima:i,sourceIds:s,leafCapacity:a,nodeMinima:c?.nodeMinima??r(e,`${t}-bvh-node-minima`,`float32x2`,l),nodeMaxima:c?.nodeMaxima??r(e,`${t}-bvh-node-maxima`,`float32x2`,l),nodeChildren:c?.nodeChildren??r(e,`${t}-bvh-node-children`,`uint32x2`,l),leafIds:c?.leafIds??r(e,`${t}-bvh-leaf-ids`,`uint32`,a),count:c?.count??r(e,`${t}-bvh-count`,`uint32`,1),overflow:c?.overflow??r(e,`${t}-bvh-overflow`,`uint32`,1)});return{bvh:u,nodes:u.getCommandNodes(e)}}var _=`fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let extent = maximum - minimum;
  if (!(extent > 0.0) || !isFiniteValue(extent)) { return 0u; }
  let normalized = clamp((value - minimum) / extent, 0.0, 1.0);
  return min(u32(normalized * 65535.0 + 0.5), 65535u);
}`,v=`fn getHilbertIndex(cellX: u32, cellY: u32) -> u32 {
  var x = cellX;
  var y = cellY;
  var key = 0u;
  for (var s = 32768u; s > 0u; s = s >> 1u) {
    let rx = select(0u, 1u, (x & s) != 0u);
    let ry = select(0u, 1u, (y & s) != 0u);
    key += s * s * ((3u * rx) ^ ry);
    if (ry == 0u) {
      if (rx == 1u) {
        x = 65535u - x;
        y = 65535u - y;
      }
      let swap = x;
      x = y;
      y = swap;
    }
  }
  return key;
}`;function y(e){return e>=256}var b=`hilbert`;function x(e,t,n,i,o,c,u,d,p=b){let m=i.length;if(!u||m<2)return g(e,t,i,o,c,void 0,d);let h=r(e,`${t}-sort-scene-bounds`,`float32`,4),y=r(e,`${t}-sort-keys`,`uint32`,m),x=r(e,`${t}-sort-rows`,`uint32`,m),S=r(e,`${t}-sort-sorted-keys`,`uint32`,m),C=r(e,`${t}-sort-sorted-rows`,`uint32`,m),w=r(e,`${t}-sort-sorted-minima`,`float32x2`,m),T=r(e,`${t}-sort-sorted-maxima`,`float32x2`,m),E=[],D=[{name:`featureMinima`,view:i,type:`f32`,access:`read`},{name:`featureMaxima`,view:o,type:`f32`,access:`read`}],O=`${f}
const FEATURE_COUNT: u32 = ${m}u;
fn readFeatureCenter(row: u32) -> vec4f {
  // Returns (center.x, center.y, valid, 0). Halving before the sum avoids overflow near FLT_MAX.
  let minimum = vec2f(featureMinima[featureMinimaOffset + row * 2u], featureMinima[featureMinimaOffset + row * 2u + 1u]);
  let maximum = vec2f(featureMaxima[featureMaximaOffset + row * 2u], featureMaxima[featureMaximaOffset + row * 2u + 1u]);
  let valid = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) &&
    minimum.x <= maximum.x && minimum.y <= maximum.y;
  let center = minimum * 0.5 + maximum * 0.5;
  return vec4f(center, select(0.0, 1.0, valid), 0.0);
}`;E.push(s(e,{id:`${t}-sort-scene-bounds`,operation:n,variant:`sort-scene-bounds`,bindings:[...D,{name:`sceneBounds`,view:h,type:`f32`,access:`read_write`}],invocationCount:256,guardIndex:!1,declarations:`${O}
var<workgroup> sharedMinima: array<vec2f, 256>;
var<workgroup> sharedMaxima: array<vec2f, 256>;`,body:`// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var row = localInvocationIndex; row < FEATURE_COUNT; row += 256u) {
    let center = readFeatureCenter(row);
    if (center.z > 0.5) {
      localMinimum = min(localMinimum, center.xy);
      localMaximum = max(localMaximum, center.xy);
    }
  }
  sharedMinima[localInvocationIndex] = localMinimum;
  sharedMaxima[localInvocationIndex] = localMaximum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedMinima[localInvocationIndex] = min(sharedMinima[localInvocationIndex], sharedMinima[localInvocationIndex + stride]);
      sharedMaxima[localInvocationIndex] = max(sharedMaxima[localInvocationIndex], sharedMaxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let minimum = sharedMinima[0];
    let maximum = sharedMaxima[0];
    let hasValid = minimum.x <= maximum.x && minimum.y <= maximum.y;
    sceneBounds[sceneBoundsOffset] = select(0.0, minimum.x, hasValid);
    sceneBounds[sceneBoundsOffset + 1u] = select(0.0, minimum.y, hasValid);
    sceneBounds[sceneBoundsOffset + 2u] = select(0.0, maximum.x, hasValid);
    sceneBounds[sceneBoundsOffset + 3u] = select(0.0, maximum.y, hasValid);
  }`})),E.push(s(e,{id:`${t}-sort-keys`,operation:n,variant:p===`hilbert`?`sort-keys-hilbert`:`sort-keys`,bindings:[...D,{name:`sceneBounds`,view:h,type:`f32`,access:`read`},{name:`sortKeys`,view:y,type:`u32`,access:`read_write`},{name:`sortRows`,view:x,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${O}
${l}
${_}
${v}`,body:`sortRows[sortRowsOffset + index] = index;
  let center = readFeatureCenter(index);
  var key = 0xffffffffu;
  if (center.z > 0.5) {
    let quantizedX = quantizeAxis(center.x, sceneBounds[sceneBoundsOffset], sceneBounds[sceneBoundsOffset + 2u]);
    let quantizedY = quantizeAxis(center.y, sceneBounds[sceneBoundsOffset + 1u], sceneBounds[sceneBoundsOffset + 3u]);
    key = ${p===`hilbert`?`getHilbertIndex(quantizedX, quantizedY)`:`spreadBits(quantizedX) | (spreadBits(quantizedY) << 1u)`};
  }
  sortKeys[sortKeysOffset + index] = key;`})),E.push(...new a({id:`${t}-sort-order`,keys:y,values:x,outputKeys:S,outputValues:C,algorithm:`radix`}).getCommandNodes(e)),E.push(s(e,{id:`${t}-sort-gather`,operation:n,variant:`sort-gather`,bindings:[...D,{name:`sortedRows`,view:C,type:`u32`,access:`read`},{name:`sortedMinima`,view:w,type:`f32`,access:`read_write`},{name:`sortedMaxima`,view:T,type:`f32`,access:`read_write`}],invocationCount:m,declarations:`const FEATURE_COUNT: u32 = ${m}u;`,body:`let row = min(sortedRows[sortedRowsOffset + index], FEATURE_COUNT - 1u);
  sortedMinima[sortedMinimaOffset + index * 2u] = featureMinima[featureMinimaOffset + row * 2u];
  sortedMinima[sortedMinimaOffset + index * 2u + 1u] = featureMinima[featureMinimaOffset + row * 2u + 1u];
  sortedMaxima[sortedMaximaOffset + index * 2u] = featureMaxima[featureMaximaOffset + row * 2u];
  sortedMaxima[sortedMaximaOffset + index * 2u + 1u] = featureMaxima[featureMaximaOffset + row * 2u + 1u];`}));let{bvh:k,nodes:A}=g(e,t,w,T,c,C,d);return E.push(...A),{bvh:k,nodes:E}}function S(e,t){return e>=4096&&t>=256}function C(e,t){let{id:n,operation:i,points:o,bvh:c}=t,l=o.length,u=r(e,`${n}-keys`,`uint32`,l),d=r(e,`${n}-rows`,`uint32`,l),p=r(e,`${n}-sorted-keys`,`uint32`,l),m=r(e,`${n}-order`,`uint32`,l),h=[s(e,{id:`${n}-keys`,operation:i,variant:`query-keys-hilbert`,bindings:[{name:`points`,view:o,type:`f32`,access:`read`},{name:`nodeMinima`,view:c.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:c.nodeMaxima,type:`f32`,access:`read`},{name:`sortKeys`,view:u,type:`u32`,access:`read_write`},{name:`sortRows`,view:d,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`${f}
${_}
${v}`,body:`sortRows[sortRowsOffset + index] = index;
  let point = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var key = 0xffffffffu;
  if (isFiniteValue(point.x) && isFiniteValue(point.y)) {
    let quantizedX = quantizeAxis(point.x, nodeMinima[nodeMinimaOffset], nodeMaxima[nodeMaximaOffset]);
    let quantizedY = quantizeAxis(point.y, nodeMinima[nodeMinimaOffset + 1u], nodeMaxima[nodeMaximaOffset + 1u]);
    key = getHilbertIndex(quantizedX, quantizedY);
  }
  sortKeys[sortKeysOffset + index] = key;`})];return h.push(...new a({id:`${n}-sort`,keys:u,values:d,outputKeys:p,outputValues:m,algorithm:`radix`}).getCommandNodes(e)),{nodes:h,order:m}}function w(e,t){let{source:n}=t,r=[],i=f,a;return n.kind===`polygons`?(r.push({name:`polygonPositions`,view:n.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:n.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:n.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`}),i+=`
const POLYGON_COUNT: u32 = ${n.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${n.ringOffsets.length-1}u;
const VERTEX_COUNT: u32 = ${n.positions.length}u;`,a=`var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let ringStart = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let ringEnd = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (ringStart <= ringEnd && ringEnd <= RING_COUNT) {
      let vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
      let vertexEnd = min(ringOffsets[ringOffsetsOffset + ringEnd], VERTEX_COUNT);
      for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
        let position = vec2f(
          polygonPositions[polygonPositionsOffset + vertex * 2u],
          polygonPositions[polygonPositionsOffset + vertex * 2u + 1u]
        );
        if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
          minimum = min(minimum, position);
          maximum = max(maximum, position);
        }
      }
    }
  }`):n.kind===`points`?(r.push({name:`positions`,view:n.positions,type:`f32`,access:`read`}),a=`var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let position = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
    minimum = position;
    maximum = position;
  }`):(r.push({name:`starts`,view:n.starts,type:`f32`,access:`read`},{name:`ends`,view:n.ends,type:`f32`,access:`read`}),a=`var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let start = vec2f(starts[startsOffset + index * 2u], starts[startsOffset + index * 2u + 1u]);
  let end = vec2f(ends[endsOffset + index * 2u], ends[endsOffset + index * 2u + 1u]);
  if (isFiniteValue(start.x) && isFiniteValue(start.y) && isFiniteValue(end.x) && isFiniteValue(end.y)) {
    minimum = min(start, end);
    maximum = max(start, end);
  }`),r.push({name:`featureMinima`,view:t.minima,type:`f32`,access:`read_write`},{name:`featureMaxima`,view:t.maxima,type:`f32`,access:`read_write`}),s(e,{id:t.id,operation:t.operation,variant:`bounds-${n.kind}`,bindings:r,invocationCount:t.featureCount,declarations:i,body:`${a}
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`})}function T(e,t){let{pairs:n,featureCounts:r,bestDistanceBits:i}=t,a=[{name:`state`,view:t.state,type:`atomic<u32>`,access:`read_write`},{name:`assignment`,view:t.assignment,type:`u32`,access:`read_write`}];n&&a.push({name:`pairPoints`,view:n.points,type:`f32`,access:`read_write`},{name:`pairGeometryOffsets`,view:n.geometryOffsets,type:`u32`,access:`read_write`}),i&&a.push({name:`bestDistanceBits`,view:i,type:`u32`,access:`read_write`}),r&&a.push({name:`featureCounts`,view:r,type:`u32`,access:`read_write`});let o=[4,t.pointCount,n?n.geometryOffsets.length:0,r?.length??0];return s(e,{id:t.id,operation:t.operation,variant:`clear`,bindings:a,invocationCount:Math.max(...o),declarations:f,body:`if (index < 4u) { atomicStore(&state[stateOffset + index], 0u); }
  if (index < ${t.pointCount}u) {
    assignment[assignmentOffset + index] = NO_FEATURE;
    ${i?`bestDistanceBits[bestDistanceBitsOffset + index] = 0xffffffffu;`:``}
  }
  ${n?`if (index < ${n.points.length}u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
    pairPoints[pairPointsOffset + index * 2u] = quietNaN;
    pairPoints[pairPointsOffset + index * 2u + 1u] = quietNaN;
  }
  if (index < ${n.geometryOffsets.length}u) {
    pairGeometryOffsets[pairGeometryOffsetsOffset + index] =
      select(0u, ${n.polygonCount}u, index == ${n.geometryOffsets.length-1}u);
  }`:``}
  ${r?`if (index < ${r.length}u) { featureCounts[featureCountsOffset + index] = 0u; }`:``}`})}function E(t,n,r){return n instanceof e?`${t}-${r}`:t}function D(e,t){let a=[],{bvh:o}=t,l=r(e,`${t.id}-point-counts`,`uint32`,t.points.length),u=r(e,`${t.id}-point-offsets`,`uint32`,t.points.length),p=(e,n,r)=>[{name:`points`,view:e,type:`f32`,access:`read`},{name:`nodeMinima`,view:o.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:o.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:o.leafIds,type:`u32`,access:`read`},...t.radius?[{name:`radius`,view:t.radius,type:`f32`,access:`read`}]:[],...r?[{name:`queryOrder`,view:r,type:`u32`,access:`read`}]:[],...n],m=e=>`${f}
const CHUNK_FIRST_ROW: u32 = ${e}u;
const INTERNAL_NODE_COUNT: u32 = ${o.internalNodeCount}u;
const FEATURE_COUNT: u32 = ${t.featureCount}u;
const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;
${c}`,h=t.radius?`let searchRadius = radius[radiusOffset];
  if (!(searchRadius >= 0.0) || searchRadius > FLOAT32_MAXIMUM) { return; }
  let queryMinimum = point - vec2f(searchRadius);
  let queryMaximum = point + vec2f(searchRadius);`:`let queryMinimum = point;
  let queryMaximum = point;`,g=(e,t,n,r)=>`${r?`let localRow = queryOrder[queryOrderOffset + index];
  let row = CHUNK_FIRST_ROW + localRow;`:`let row = CHUNK_FIRST_ROW + index;`}
  ${t}
  var found = 0u;
  let point = vec2f(points[pointsOffset + ${r?`localRow`:`index`} * 2u], points[pointsOffset + ${r?`localRow`:`index`} * 2u + 1u]);
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y)) { return; }
  ${h}
  var node = 0u;
  loop {
    if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
      if (node < INTERNAL_NODE_COUNT) {
        node = node * 2u + 1u;
        continue;
      }
      let featureRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (featureRow < FEATURE_COUNT) {
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
  ${n}`;if(t.points.length===0)return a;let _=new Map;if(t.querySort){for(let[r,i]of n(t.points).entries())if(i.length>=2){let n=C(e,{id:`${E(t.id,t.points,r)}-order`,operation:t.operation,points:i,bvh:o});a.push(...n.nodes),_.set(r,n.order)}}let v=0;for(let[r,i]of n(t.points).entries()){if(i.length>0){let n=_.get(r);a.push(s(e,{id:E(t.id,t.points,r),operation:t.operation,variant:n?`probe-count-ordered`:`probe-count`,bindings:p(i,[{name:`pointCounts`,view:l,type:`u32`,access:`read_write`}],n),invocationCount:i.length,declarations:m(v),body:g(`found = found + 1u;`,`pointCounts[pointCountsOffset + row] = 0u;`,`pointCounts[pointCountsOffset + row] = found;`,!!n)}))}v+=i.length}a.push(...new i({id:`${t.id}-scan`,input:l,output:u,mode:`exclusive`}).getCommandNodes(e)),a.push(s(e,{id:`${t.id}-total`,operation:t.operation,variant:`probe-total`,bindings:[{name:`pointCounts`,view:l,type:`u32`,access:`read`},{name:`pointOffsets`,view:u,type:`u32`,access:`read`},{name:`state`,view:t.state,type:`u32`,access:`read_write`}],invocationCount:1,body:`state[stateOffset] = pointOffsets[pointOffsetsOffset + ${t.points.length-1}u] + pointCounts[pointCountsOffset + ${t.points.length-1}u];`})),a.push(d(e,{id:t.id,operation:t.operation,counts:l,offsets:u,rowCount:t.points.length,total:t.state})),v=0;for(let[r,i]of n(t.points).entries()){if(i.length>0){let n=t.radius?void 0:_.get(r);a.push(s(e,{id:`${E(t.id,t.points,r)}-write`,operation:t.operation,variant:n?`probe-write-ordered`:`probe-write`,bindings:p(i,[{name:`pointOffsets`,view:u,type:`u32`,access:`read`},{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read_write`},...t.packedCandidates?[]:[{name:`candidatePoints`,view:t.candidatePoints,type:`f32`,access:`read_write`}]],n),invocationCount:i.length,declarations:m(v),body:g(t.packedCandidates?`let slot = slotStart + found;
        candidatePairs[candidatePairsOffset + slot * 4u] = row;
        candidatePairs[candidatePairsOffset + slot * 4u + 1u] = featureRow;
        candidatePairs[candidatePairsOffset + slot * 4u + 2u] = bitcast<u32>(point.x);
        candidatePairs[candidatePairsOffset + slot * 4u + 3u] = bitcast<u32>(point.y);
        found = found + 1u;
        if (slot + 1u >= slotEnd) { break; }`:`let slot = slotStart + found;
        candidatePairs[candidatePairsOffset + slot * 2u] = row;
        candidatePairs[candidatePairsOffset + slot * 2u + 1u] = featureRow;
        let pointRow = slot * ${t.pointRowScale}u + ${t.pointRowOffset}u;
        candidatePoints[candidatePointsOffset + pointRow * 2u] = point.x;
        candidatePoints[candidatePointsOffset + pointRow * 2u + 1u] = point.y;
        found = found + 1u;
        if (slot + 1u >= slotEnd) { break; }`,`let slotStart = pointOffsets[pointOffsetsOffset + row];
  var slotEnd = CANDIDATE_CAPACITY;
  if (row + 1u < ${t.points.length}u) {
    slotEnd = min(pointOffsets[pointOffsetsOffset + row + 1u], CANDIDATE_CAPACITY);
  }
  if (slotStart >= slotEnd) { return; }`,``,!!n)}))}v+=i.length}return a}function O(e,t){let r=[],i=t.distances?n(t.distances):[],a=0;for(let[o,c]of n(t.pointFeatureIds).entries()){if(c.length>0){let n=[{name:`assignment`,view:t.assignment,type:`u32`,access:`read`},{name:`pointFeatureIds`,view:c,type:`u32`,access:`read_write`}];t.featureIds&&n.push({name:`featureIds`,view:t.featureIds,type:`u32`,access:`read`}),t.featureCounts&&n.push({name:`featureCounts`,view:t.featureCounts,type:`atomic<u32>`,access:`read_write`});let l=i[o];t.bestDistanceBits&&l&&n.push({name:`bestDistanceBits`,view:t.bestDistanceBits,type:`u32`,access:`read`},{name:`distances`,view:l,type:`f32`,access:`read_write`}),r.push(s(e,{id:E(t.id,t.pointFeatureIds,o),operation:t.operation,variant:`assign`,bindings:n,invocationCount:c.length,declarations:`${f}
const CHUNK_FIRST_ROW: u32 = ${a}u;`,body:`let row = CHUNK_FIRST_ROW + index;
  let featureRow = assignment[assignmentOffset + row];
  let matched = featureRow != NO_FEATURE;
  var featureId = featureRow;
  ${t.featureIds?`if (matched) { featureId = featureIds[featureIdsOffset + featureRow]; }`:``}
  pointFeatureIds[pointFeatureIdsOffset + index] = featureId;
  ${t.featureCounts?`if (matched) { atomicAdd(&featureCounts[featureCountsOffset + featureRow], 1u); }`:``}
  ${t.bestDistanceBits&&l?`let bestDistance = bitcast<f32>(bestDistanceBits[bestDistanceBitsOffset + row]);
  distances[distancesOffset + index] = select(-1.0, ${t.squaredDistances?`sqrt(bestDistance)`:`bestDistance`}, matched);`:``}`}))}a+=c.length}return r}function k(e,t){let a=[],o=t.sourceIds?n(t.sourceIds):[],c=t.matches.ids.length,l=t.points.length;if(l===0)return a;let u=r(e,`${t.id}-flags`,`uint32`,l),d=r(e,`${t.id}-offsets`,`uint32`,l),p=0;for(let[r,i]of n(t.points).entries())i.length>0&&a.push(s(e,{id:E(t.id,t.points,r),operation:t.operation,variant:`collect-match-flags`,bindings:[{name:`assignment`,view:t.assignment,type:`u32`,access:`read`},{name:`flags`,view:u,type:`u32`,access:`read_write`}],invocationCount:i.length,declarations:`${f}
const CHUNK_FIRST_ROW: u32 = ${p}u;`,body:`let row = CHUNK_FIRST_ROW + index;
  flags[flagsOffset + row] = select(0u, 1u, assignment[assignmentOffset + row] != NO_FEATURE);`})),p+=i.length;if(a.push(...new i({id:`${t.id}-scan`,input:u,output:d,mode:`exclusive`}).getCommandNodes(e)),a.push(s(e,{id:`${t.id}-total`,operation:t.operation,variant:`collect-match-total`,bindings:[{name:`flags`,view:u,type:`u32`,access:`read`},{name:`offsets`,view:d,type:`u32`,access:`read`},{name:`state`,view:t.state,type:`u32`,access:`read_write`}],invocationCount:1,body:`state[stateOffset + 2u] = offsets[offsetsOffset + ${l-1}u] + flags[flagsOffset + ${l-1}u];`})),c===0)return a;p=0;for(let[r,i]of n(t.points).entries()){if(i.length>0){let n=o[r],l=[{name:`flags`,view:u,type:`u32`,access:`read`},{name:`offsets`,view:d,type:`u32`,access:`read`},{name:`matchIds`,view:t.matches.ids,type:`u32`,access:`read_write`}];n&&l.push({name:`sourceIds`,view:n,type:`u32`,access:`read`}),a.push(s(e,{id:`${E(t.id,t.points,r)}-write`,operation:t.operation,variant:`collect-match-write`,bindings:l,invocationCount:i.length,declarations:`${f}
const CHUNK_FIRST_ROW: u32 = ${p}u;
const MATCH_CAPACITY: u32 = ${c}u;`,body:`let row = CHUNK_FIRST_ROW + index;
  if (flags[flagsOffset + row] == 0u) { return; }
  let slot = offsets[offsetsOffset + row];
  if (slot < MATCH_CAPACITY) {
    matchIds[matchIdsOffset + slot] = ${n?`sourceIds[sourceIdsOffset + index]`:`row`};
  }`}))}p+=i.length}return a}function A(e,t){let{matches:n}=t,r=[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`bvhOverflow`,view:t.bvhOverflow,type:`u32`,access:`read`},{name:`overflow`,view:t.overflow,type:`u32`,access:`read_write`}];t.candidateCount&&r.push({name:`candidateCount`,view:t.candidateCount,type:`u32`,access:`read_write`}),t.uncertainCount&&r.push({name:`uncertainCount`,view:t.uncertainCount,type:`u32`,access:`read_write`}),n&&(r.push({name:`matchCount`,view:n.status.count,type:`u32`,access:`read_write`},{name:`matchOverflow`,view:n.status.overflow,type:`u32`,access:`read_write`}),n.status.requiredCount&&r.push({name:`matchTotalCount`,view:n.status.requiredCount,type:`u32`,access:`read_write`}));let i=n?.ids.length??0;return s(e,{id:t.id,operation:t.operation,variant:`finalize`,bindings:r,invocationCount:1,declarations:`const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;
const MATCH_CAPACITY: u32 = ${i}u;`,body:`let candidateTotal = state[stateOffset];
  let matchTotal = state[stateOffset + 2u];
  let overflowed = bvhOverflow[bvhOverflowOffset] != 0u || candidateTotal > CANDIDATE_CAPACITY${n?` || matchTotal > MATCH_CAPACITY`:``};
  let overflowValue = select(0u, 1u, overflowed);
  overflow[overflowOffset] = overflowValue;
  ${t.candidateCount?`candidateCount[candidateCountOffset] = candidateTotal;`:``}
  ${t.uncertainCount?`uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];`:``}
  ${n?`matchCount[matchCountOffset] = min(matchTotal, MATCH_CAPACITY);
  matchOverflow[matchOverflowOffset] = overflowValue;
  ${n.status.requiredCount?`matchTotalCount[matchTotalCountOffset] = matchTotal;`:``}`:``}`})}function j(e,t,n){return e*t>=4e6&&e>=8*Math.max(n,1)}function M(e,t){return Math.max(Math.ceil(e/32),1)+2*t}function N(e,t){let{id:n,operation:i,polygons:a}=t,o=a.positions.length,c=a.featureOffsets.length-1,l=Math.max(Math.ceil(o/32),1),u=t.layout??r(e,`${n}-edge-layout`,`uint32`,M(o,c)),d=a.ringOffsets.length-1;return{nodes:[s(e,{id:`${n}-edge-layout-bits`,operation:i,variant:`edge-layout-bits`,bindings:[{name:`ringOffsets`,view:a.ringOffsets,type:`u32`,access:`read`},{name:`edgeLayout`,view:u,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`const VERTEX_COUNT: u32 = ${o}u;
const RING_COUNT: u32 = ${d}u;`,body:`let first = index * 32u;
  let last = min(first + 32u, VERTEX_COUNT);
  var bits = 0u;
  if (RING_COUNT > 0u) {
    // Ring that contains \`first\`: the last ring that starts at or before it.
    var low = 0u;
    var high = RING_COUNT;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (ringOffsets[ringOffsetsOffset + middle] <= first) { low = middle; } else { high = middle; }
    }
    var ring = low;
    var ringBegin = ringOffsets[ringOffsetsOffset + ring];
    var ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    for (var vertex = first; vertex < last; vertex++) {
      loop {
        if (vertex < ringEnd || ring + 1u >= RING_COUNT) { break; }
        ring = ring + 1u;
        ringBegin = ringOffsets[ringOffsetsOffset + ring];
        ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
      }
      let usable = vertex >= ringBegin && vertex < ringEnd && ringEnd >= ringBegin + 3u && ringEnd <= VERTEX_COUNT;
      bits = bits | select(0u, 1u << (vertex - first), usable);
    }
  }
  edgeLayout[edgeLayoutOffset + index] = bits;`}),s(e,{id:`${n}-edge-layout-features`,operation:i,variant:`edge-layout-features`,bindings:[{name:`featureOffsets`,view:a.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:a.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:a.ringOffsets,type:`u32`,access:`read`},{name:`edgeLayout`,view:u,type:`u32`,access:`read_write`}],invocationCount:Math.max(c,1),declarations:`const FEATURE_COUNT: u32 = ${c}u;
const POLYGON_COUNT: u32 = ${a.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${d}u;
const VERTEX_COUNT: u32 = ${o}u;
const FEATURE_BASE: u32 = ${l}u;`,body:`if (index >= FEATURE_COUNT) { return; }
  let partStart = featureOffsets[featureOffsetsOffset + index];
  let partEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var wellFormed = partStart <= partEnd && partEnd <= POLYGON_COUNT;
  if (wellFormed) {
    for (var part = partStart; part < partEnd && wellFormed; part++) {
      let ringStart = polygonOffsets[polygonOffsetsOffset + part];
      let ringEnd = polygonOffsets[polygonOffsetsOffset + part + 1u];
      if (ringStart > ringEnd || ringEnd > RING_COUNT) { wellFormed = false; }
      for (var ring = ringStart; ring < ringEnd && wellFormed; ring++) {
        let vertexStart = ringOffsets[ringOffsetsOffset + ring];
        let vertexEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
        if (vertexEnd < vertexStart || vertexEnd > VERTEX_COUNT) { wellFormed = false; }
      }
    }
  }
  edgeLayout[edgeLayoutOffset + FEATURE_BASE + index * 2u] = select(0xffffffffu, partStart, wellFormed);
  edgeLayout[edgeLayoutOffset + FEATURE_BASE + index * 2u + 1u] = partEnd;`})],layout:u}}export{D as _,E as a,y as c,M as d,N as f,k as g,h,A as i,g as l,O as m,w as n,j as o,x as p,T as r,S as s,f as t,C as u,m as v,p as y};