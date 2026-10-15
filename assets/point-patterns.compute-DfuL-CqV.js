import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{J as e,mt as t}from"./src-CNCKFW2b.js";import{E as n,h as r,n as i,o as a,s as o}from"./analysis-frame-Blu0tYdw.js";import{_ as s,a as c,b as l,c as u,d,f,g as p,h as m,i as h,l as g,m as _,n as v,o as y,p as b,s as x,t as S,u as C,v as w,y as T}from"./pair-statistics-parameters-CO--_nmZ.js";import{C as E,N as D,S as O,_ as k,d as A,h as j,j as M,k as N,t as ee,u as te,w as P,x as F,y as I}from"./resources-CwvhEOBe.js";import{a as L,i as R}from"./gpu-subgroup-utils-BuV3X9Xp.js";import{t as ne}from"./gpu-scan-BuGhTLDQ.js";import{n as re,r as ie,t as ae}from"./gpu-chunk-utils-Bz-6HA7y.js";import{t as oe}from"./gpu-sort-Bl3Dztdb.js";import{t as se}from"./gpu-reduction-BD8l40jM.js";import{n as ce,s as z}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as le,r as B,s as ue}from"./wgsl-common-BNRVzgd8.js";import{t as V}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{t as de}from"./exact-geometry-wgsl-DOqxBQzc.js";import{r as fe}from"./polygon-edge-slabs-Cmf_6bAe.js";import{i as pe,t as me}from"./gpu-random-points-in-polygon-D_k0WNZN.js";import{n as he,t as ge}from"./gpu-neighbor-search-Fct8W8hB.js";import{a as _e,n as ve,t as ye}from"./gpu-polygon-rasterization-0IHMY9Te.js";import{t as be}from"./gpu-rectangle-clip-D_EZ8d4-.js";import{a as xe,c as Se,o as Ce,r as we,s as Te,t as Ee,x as De}from"./index-Bjk18V2I.js";import{t as H}from"./graph-buffers-B4_80RlX.js";import{t as Oe}from"./graph-swap-41Ru7LVu.js";import{t as ke}from"./summary-reader-CsgWxENf.js";import{t as U}from"./mode-kernels-Dxm2GWvI.js";import{t as Ae}from"./polygon-mesh-48KB9j_1.js";import"./anchors-D9E9ucGL.js";import{a as je,c as Me,g as Ne}from"./gazetteer-OhisynFF.js";import{a as Pe,l as Fe,m as W,o as Ie,s as Le}from"./class-table-BVEo3-Vs.js";import{a as Re,n as G,o as ze,r as Be,s as Ve,t as He}from"./live-text-83XBWyr1.js";import{n as Ue}from"./picking-DRmnefrA.js";import"./segments-BtINkq-i.js";import{t as We}from"./polygon-buffers-DTLnArS7.js";import{n as Ge}from"./b1-nature-data-Clqa3fop.js";import{a as Ke,o as qe,r as Je,s as Ye,t as Xe}from"./b1-points-look-B9wT7IoA.js";import{t as Ze}from"./b1-city-geometry-1vs1SQDu.js";var K=class{id;input;values;lengths;count;constructor(e){this.id=e.id??`gpu-run-length-encode`,this.input=e.input,this.values=e.values,this.lengths=e.lengths,this.count=e.count;for(let[e,t]of Object.entries({input:this.input,values:this.values,lengths:this.lengths,count:this.count}))for(let n of L(t))M(n,`${this.id} ${e}`);if(this.values.length<this.input.length||this.lengths.length<this.input.length)throw Error(`${this.id} values and lengths must have capacity >= input.length`);if(this.count.length<1)throw Error(`${this.id} count must contain at least one row`);if([...L(this.values),...L(this.lengths),this.count].some(e=>L(this.input).some(t=>t.buffer===e.buffer)))throw Error(`${this.id} outputs must use separate buffers from input`)}getCommandNodes(e){ie(e,[this.input],[this.values,this.lengths,this.count]);let t=[],n=re(e,this.input,0,this.input.length),r=E(e,`${this.id}-flags`,n),i=E(e,`${this.id}-runs`,n),a=e.device.limits.maxComputeWorkgroupsPerDimension;for(let[i,o]of n.data.entries()){let s=n.data[Math.max(0,i-1)],c=F(this.id,o.length,256,a);t.push(ae(e,{id:`${this.id}-flags-${i}`,inputs:{inputValues:o,...i?{previous:s}:{}},outputs:{flags:r.data[i]},dispatch:c,source:`
@group(0) @binding(0) var<storage, read> inputValues: array<u32>;
${i?`@group(0) @binding(1) var<storage, read> previous: array<u32>;`:``}
@group(0) @binding(${i?2:1}) var<storage, read_write> flags: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${O(c,256)}
  if (index >= ${o.length}u) { return; }
  var head = true;
  if (index > 0u) {
    head = inputValues[${N(o)}u + index] != inputValues[${N(o)}u + index - 1u];
  } ${i?`else {
    head = inputValues[${N(o)}u] != previous[${N(s)+s.length-1}u];
  }`:``}
  flags[index] = select(0u, 1u, head);
}`}))}n.length&&t.push(...new ne({id:`${this.id}-scan`,input:r,output:i,mode:`inclusive`}).getCommandNodes(e));let o=i.data[i.data.length-1];if(t.push(ae(e,{id:`${this.id}-count`,inputs:o?{runs:o}:{},outputs:{count:this.count},dispatch:{x:1,y:1,z:1},workgroupSize:1,source:`
${o?`@group(0) @binding(0) var<storage, read> runs: array<u32>;`:``}
@group(0) @binding(${o?1:0}) var<storage, read_write> count: array<u32>;
@compute @workgroup_size(1)
fn main() {
  count[${N(this.count)}u] = ${o?`runs[${o.length-1}u]`:`0u`};
}`})),!n.length)return t;let s=0;for(let[n,r]of L(this.lengths).entries()){if(!r.length)continue;let i=F(this.id,r.length,256,a);t.push(ae(e,{id:`${this.id}-clear-${n}`,inputs:{count:this.count},outputs:{lengths:r},dispatch:i,source:`
@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read_write> lengths: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${O(i,256)}
  if (index < ${r.length}u && ${s}u + index < count[${N(this.count)}u]) {
    lengths[${N(r)}u + index] = 0u;
  }
}`})),s+=r.length}for(let[o,[s,c,l]]of R(e,[n,r,i]).entries()){let n=F(this.id,s.length,256,a);for(let[r,i]of[[`values`,this.values],[`lengths`,this.lengths]]){let a=0;for(let[u,d]of L(i).entries())d.length&&t.push(ae(e,{id:`${this.id}-${r}-${o}-${u}`,inputs:{...r===`values`?{values:s,heads:c}:{},ranks:l},outputs:{output:d},dispatch:n,source:`
${r===`values`?`@group(0) @binding(0) var<storage, read> values: array<u32>;
@group(0) @binding(1) var<storage, read> heads: array<u32>;`:``}
@group(0) @binding(${r===`values`?2:0}) var<storage, read> ranks: array<u32>;
@group(0) @binding(${r===`values`?3:1}) var<storage, read_write> output: array<${r===`lengths`?`atomic<u32>`:`u32`}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${O(n,256)}
  if (index >= ${s.length}u) { return; }
  let run = ranks[${N(l)}u + index] - 1u;
  if (run < ${a}u || run - ${a}u >= ${d.length}u) { return; }
  let destination = ${N(d)}u + run - ${a}u;
  ${r===`lengths`?`atomicAdd(&output[destination], 1u);`:`if (heads[${N(c)}u + index] != 0u) {
    output[destination] = values[${N(s)}u + index];
  }`}
}`})),a+=d.length}}return t}},q=`GPUStudyWindowPreparation`,Qe=10,$e=class{kind=`polygon`;id;spatialContext;vertexCount;props;sampleGrid;encodedBuildCount=0;graph;views;ownedBuffers=[];dirty=!0;constructor(e){this.id=e.id??`study-window`,this.spatialContext=n(this.id,e.spatialContext,r),this.props=e,this.vertexCount=e.positions.length,this.sampleGrid=e.samplingGrid??[256,256];let{id:t,sampleGrid:i}=this;I(e.positions,[`float32x2`],`${t} positions`);for(let n of[`featureOffsets`,`polygonOffsets`,`ringOffsets`])I(e[n],[`uint32`],`${t} ${n}`);if(e.featureOffsets.length!==2)throw Error(`${t} study window needs exactly one feature (featureOffsets of length 2)`);if(this.vertexCount<3||e.ringOffsets.length<2||e.polygonOffsets.length<2)throw Error(`${t} study window needs at least one ring of 3 or more vertices`);if(i.length!==2||!i.every(e=>Number.isInteger(e)&&e>0)||i[0]*i[1]>1048576)throw Error(`${t} samplingGrid must be two positive integers with a product of at most 1048576`);if(e.output?.summary&&(I(e.output.summary,[`float32`],`${t} output.summary`),e.output.summary.length!==8))throw Error(`${t} output.summary must contain 8 float32 values`);j(t,[e.output?.summary],[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets])}getViews(e){let t=this.views;if(!this.isDeclaredIn(e)||!t)throw Error(`${this.id} study window needs graph.add before its consumers`);return t}isDeclaredIn(e){return this.graph!==void 0&&this.graph===e}invalidate(){this.dirty=!0}destroy(){for(let e of this.ownedBuffers)e.destroy();this.views=void 0,this.ownedBuffers=[],this.graph=void 0,this.dirty=!0}createPersistentView(e,n,r,i){let a=e.device.createBuffer({id:`${this.id}-${n}`,byteLength:Math.max(i,1)*(r===`uint32x2`?8:4),usage:t.STORAGE|t.COPY_SRC|t.COPY_DST});return this.ownedBuffers.push(a),te(e,`${this.id}-${n}`,a,r,i)}getCommandNodes(t){let{id:n,props:r,vertexCount:i,sampleGrid:a}=this;if(this.graph)throw Error(`${n} is already added to a graph`);let o=r.ringOffsets.length-1,c=a[0]*a[1];k(n,t,[r.positions,r.featureOffsets,r.polygonOffsets,r.ringOffsets,r.output?.summary]),this.graph=t;let l=this.createPersistentView(t,`edges`,`float32`,4*i),u=this.createPersistentView(t,`summary`,`float32`,8),d=P(t,`${n}-terms`,`float32`,9*i),f=P(t,`${n}-rings`,`float32`,Qe*o),m=this.createPersistentView(t,`sample-cells`,`uint32`,c),h=this.createPersistentView(t,`sample-cell-count`,`uint32`,1),g={name:`positions`,view:r.positions,type:`f32`,access:`read`},v={name:`ringOffsets`,view:r.ringOffsets,type:`u32`,access:`read`},y={name:`rings`,view:f,type:`f32`,access:`read`},b={name:`summary`,view:u,type:`f32`,access:`read`},x=_e(`findRing`,`ringOffsets`,o),S=`fn readVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}`,C=[z(t,{id:`${n}-ring-terms`,operation:q,variant:`ring-terms`,bindings:[g,v,{name:`terms`,view:d,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`${x}${S}${V}${B}`,body:`let base = termsOffset + 9u * index;
  for (var slot = 0u; slot < 4u; slot++) { terms[base + slot] = 0.0; }
  terms[base + 4u] = 3.402823466e38;
  terms[base + 5u] = 3.402823466e38;
  terms[base + 6u] = -3.402823466e38;
  terms[base + 7u] = -3.402823466e38;
  terms[base + 8u] = 0.0;
  let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  let next = select(index + 1u, first, index + 1u == end);
  let startPosition = readVertex(index);
  let endPosition = readVertex(next);
  let origin = readVertex(first);
  // One exact predicate call site; uncertified orientations must not discard a ring.
  terms[base + 8u] = select(0.0, 1.0, orientSign(origin, startPosition, endPosition) != 0);
  let relativeStart = startPosition - origin;
  let relativeEnd = endPosition - origin;
  let forward = relativeStart.x * relativeEnd.y;
  let backward = relativeStart.y * relativeEnd.x;
  var determinant = forward - backward;
  if (!isFiniteBits(startPosition.x) || !isFiniteBits(startPosition.y) ||
      !isFiniteBits(endPosition.x) || !isFiniteBits(endPosition.y)) { determinant = getNaN(); }
  terms[base] = determinant;
  terms[base + 1u] = abs(forward) + abs(backward);
  terms[base + 2u] = distance(startPosition, endPosition);
  terms[base + 3u] = 1.0;
  if (isFiniteBits(startPosition.x) && isFiniteBits(startPosition.y)) {
    terms[base + 4u] = startPosition.x;
    terms[base + 5u] = startPosition.y;
    terms[base + 6u] = startPosition.x;
    terms[base + 7u] = startPosition.y;
  }`})];for(let e=1;e<i;e*=2)C.push(z(t,{id:`${n}-ring-reduce-${e}`,operation:q,variant:`ring-reduce`,bindings:[v,{name:`terms`,view:d,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`${x}\noverride TREE_STRIDE: u32;`,constants:{TREE_STRIDE:e},body:`let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  if ((index - first) % (2u * TREE_STRIDE) != 0u || index + TREE_STRIDE >= end) { return; }
  let base = termsOffset + 9u * index;
  let partner = termsOffset + 9u * (index + TREE_STRIDE);
  for (var slot = 0u; slot < 4u; slot++) { terms[base + slot] += terms[partner + slot]; }
  for (var slot = 4u; slot < 6u; slot++) { terms[base + slot] = min(terms[base + slot], terms[partner + slot]); }
  for (var slot = 6u; slot < 8u; slot++) { terms[base + slot] = max(terms[base + slot], terms[partner + slot]); }
  terms[base + 8u] += terms[partner + 8u];`}));C.push(z(t,{id:`${n}-ring-orientation`,operation:q,variant:`ring-orientation`,bindings:[v,{name:`polygonOffsets`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`},{name:`terms`,view:d,type:`f32`,access:`read`},{...y,access:`read_write`}],invocationCount:o,declarations:`${ue}${_e(`findPolygon`,`polygonOffsets`,r.polygonOffsets.length-1)}`,body:`let base = ringsOffset + ${Qe}u * index;
  for (var slot = 0u; slot < ${Qe}u; slot++) { rings[base + slot] = 0.0; }
  let polygon = findPolygon(index);
  if (polygon == 0xffffffffu || polygon < featureOffsets[featureOffsetsOffset] ||
      polygon >= featureOffsets[featureOffsetsOffset + 1u]) { return; }
  let first = ringOffsets[ringOffsetsOffset + index];
  let end = ringOffsets[ringOffsetsOffset + index + 1u];
  if (end - first < 3u) { return; }
  let source = termsOffset + 9u * first;
  if (terms[source + 8u] == 0.0) { return; }
  rings[base + 9u] = 1.0;
  let doubledArea = terms[source];
  let role = select(-1.0, 1.0, index == polygonOffsets[polygonOffsetsOffset + polygon]);
  rings[base] = select(0.0, abs(doubledArea) * 0.5 * role, end - first >= 3u);
  rings[base + 1u] = terms[source + 2u];
  rings[base + 2u] = terms[source + 3u];
  let errorBound = ${8*2**-23*(1+Math.ceil(Math.log2(i)))} * terms[source + 1u];
  if (!isFiniteBits(doubledArea) || abs(doubledArea) <= errorBound) {
    rings[base + 3u] = 1.0;
  }
  for (var slot = 4u; slot < 8u; slot++) { rings[base + slot] = terms[source + slot]; }
  rings[base + 8u] = sign(doubledArea) * role;`}),z(t,{id:`${n}-orient-edges`,operation:q,variant:`orient-edges`,bindings:[g,v,y,{name:`edges`,view:l,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`${x}${S}${B}`,body:`let base = edgesOffset + 4u * index;
  for (var slot = 0u; slot < 4u; slot++) { edges[base + slot] = getNaN(); }
  let ring = findRing(index);
  if (ring == 0xffffffffu) { return; }
  let ringBase = ringsOffset + ${Qe}u * ring;
  if (rings[ringBase + 9u] == 0.0) { return; }
  let first = ringOffsets[ringOffsetsOffset + ring];
  let end = ringOffsets[ringOffsetsOffset + ring + 1u];
  let next = select(index + 1u, first, index + 1u == end);
  let reverse = rings[ringBase + 8u] < 0.0;
  let startPosition = readVertex(select(index, next, reverse));
  let endPosition = readVertex(select(next, index, reverse));
  edges[base] = startPosition.x;
  edges[base + 1u] = startPosition.y;
  edges[base + 2u] = endPosition.x;
  edges[base + 3u] = endPosition.y;`}),z(t,{id:`${n}-summarize`,operation:q,variant:`summarize`,bindings:[y,{...b,access:`read_write`}],invocationCount:1,declarations:`${ue}${B}`,body:`var totals = vec3f(0.0);
  var compensation = vec3f(0.0);
  var minimum = vec2f(3.402823466e38);
  var maximum = vec2f(-3.402823466e38);
  var flags = 0u;
  for (var ring = 0u; ring < ${o}u; ring++) {
    let base = ringsOffset + ${Qe}u * ring;
    if (rings[base + 9u] == 0.0) { continue; }
    let values = vec3f(rings[base], rings[base + 1u], rings[base + 2u]);
    let adjusted = values - compensation;
    let nextTotals = totals + adjusted;
    compensation = (nextTotals - totals) - adjusted;
    totals = nextTotals;
    flags |= u32(rings[base + 3u]);
    if (values.z > 0.0) {
      minimum = min(minimum, vec2f(rings[base + 4u], rings[base + 5u]));
      maximum = max(maximum, vec2f(rings[base + 6u], rings[base + 7u]));
    }
  }
  if (!isFiniteBits(totals.x) || totals.x <= 0.0) { flags |= 2u; }
  if (any(minimum > maximum)) { minimum = vec2f(getNaN()); maximum = minimum; }
  summary[summaryOffset] = minimum.x;
  summary[summaryOffset + 1u] = minimum.y;
  summary[summaryOffset + 2u] = maximum.x;
  summary[summaryOffset + 3u] = maximum.y;
  summary[summaryOffset + 4u] = totals.x;
  summary[summaryOffset + 5u] = totals.y;
  summary[summaryOffset + 6u] = totals.z;
  summary[summaryOffset + 7u] = f32(flags);`}));let{nodes:w,slabs:T}=fe(t,{id:n,operation:q,owner:`feature`,positions:r.positions,featureOffsets:r.featureOffsets,polygonOffsets:r.polygonOffsets,ringOffsets:r.ringOffsets,entryCapacity:r.slabEntryCapacity});C.push(...w);let E={...T,params:this.createPersistentView(t,`edge-slab-params`,`uint32`,T.params.length),slabOffsets:this.createPersistentView(t,`edge-slab-offsets`,`uint32`,T.slabOffsets.length),slabEntries:this.createPersistentView(t,`edge-slab-entries`,`uint32x2`,T.slabEntries.length)};for(let e of[`params`,`slabOffsets`,`slabEntries`]){let r=T[e].length*(e===`slabEntries`?2:1);C.push(z(t,{id:`${n}-edges-copy-${e}`,operation:q,variant:`copy-words`,bindings:[{name:`source`,view:T[e],type:`u32`,access:`read`},{name:`destination`,view:E[e],type:`u32`,access:`read_write`}],invocationCount:Math.max(r,1),declarations:`const WORD_COUNT: u32 = ${r}u;`,body:`if (index >= WORD_COUNT) { return; }
  destination[destinationOffset + index] = source[sourceOffset + index];`}))}let D=P(t,`${n}-sample-parameters`,`float32`,4),O=P(t,`${n}-sample-zones`,`uint32`,c),A=P(t,`${n}-sample-boundary`,`uint32`,c),j=P(t,`${n}-sample-overflow`,`uint32`,1);C.push(z(t,{id:`${n}-sample-parameters`,operation:q,variant:`sample-parameters`,bindings:[b,{name:`parameters`,view:D,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`fn readSummary(slot: u32) -> f32 { return summary[summaryOffset + slot]; }
${s(a,`readSummary`)}`,body:`let cell = getStudyWindowSampleCell(0u);
  for (var slot = 0u; slot < 4u; slot++) { parameters[parametersOffset + slot] = cell[slot]; }`}),...new ye({id:`${n}-raster`,spatialContext:this.spatialContext,grid:{width:a[0],height:a[1],affine:[1,0,0,0,1,0],pixelInterpretation:`area`},parameters:D,positions:r.positions,featureOffsets:r.featureOffsets,polygonOffsets:r.polygonOffsets,ringOffsets:r.ringOffsets,crossingCapacity:r.crossingCapacity??4*a[1]*o,output:{zones:O,boundary:A,status:{overflow:j}}}).getCommandNodes(t)),C.push(z(t,{id:`${n}-sample-overflow-recovery`,operation:q,variant:`sample-overflow-recovery`,bindings:[b,{name:`edges`,view:l,type:`f32`,access:`read`},{name:`overflow`,view:j,type:`u32`,access:`read`},{name:`zones`,view:O,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${V}${de}
fn readSummary(slot: u32) -> f32 { return summary[summaryOffset + slot]; }
${s(a,`readSummary`)}`,body:`if (overflow[overflowOffset] == 0u) { return; }
  let cell = getStudyWindowSampleCell(index);
  let center = cell.xy + 0.5 * cell.zw;
  let probe = ExactProbe(center, center, center, EXACT_PROBE_POINT);
  var winding = 0;
  for (var vertex = 0u; vertex < ${i}u; vertex++) {
    let base = edgesOffset + 4u * vertex;
    let startPosition = vec2f(edges[base], edges[base + 1u]);
    let endPosition = vec2f(edges[base + 2u], edges[base + 3u]);
    if (!isFiniteBits(startPosition.x) || !isFiniteBits(startPosition.y) ||
        !isFiniteBits(endPosition.x) || !isFiniteBits(endPosition.y)) { continue; }
    winding += exactProbeCrossing(probe, startPosition, endPosition).x;
  }
  zones[zonesOffset + index] = select(${ve}u, 0u, (winding & 1) != 0);`}));let M=P(t,`${n}-sample-counts`,`uint32`,c+1),N=P(t,`${n}-sample-offsets`,`uint32`,c+1);C.push(z(t,{id:`${n}-sample-count`,operation:q,variant:`sample-count`,bindings:[{name:`zones`,view:O,type:`u32`,access:`read`},{name:`boundary`,view:A,type:`u32`,access:`read`},{name:`counts`,view:M,type:`u32`,access:`read_write`}],invocationCount:c+1,body:`var count = 0u;
  if (index < ${c}u) {
    count = select(0u, 1u, zones[zonesOffset + index] != ${ve}u || boundary[boundaryOffset + index] != 0u);
  }
  counts[countsOffset + index] = count;`}),...new ne({id:`${n}-sample-scan`,input:M,output:N,mode:`exclusive`}).getCommandNodes(t),z(t,{id:`${n}-sample-compact`,operation:q,variant:`sample-compact`,bindings:[{name:`counts`,view:M,type:`u32`,access:`read`},{name:`offsets`,view:N,type:`u32`,access:`read`},{name:`boundary`,view:A,type:`u32`,access:`read`},{name:`sampleCells`,view:m,type:`u32`,access:`read_write`},{name:`sampleCellCount`,view:h,type:`u32`,access:`read_write`}],invocationCount:c+1,body:`let count = offsets[offsetsOffset + ${c}u];
  if (index == ${c}u) { sampleCellCount[sampleCellCountOffset] = count; return; }
  if (index >= count) { sampleCells[sampleCellsOffset + index] = 0u; }
  if (counts[countsOffset + index] != 0u) {
    sampleCells[sampleCellsOffset + offsets[offsetsOffset + index]] = index |
      select(0u, ${_}u, boundary[boundaryOffset + index] != 0u);
  }`}));let ee=p({vertexCount:i,slabOffsetCount:E.slabOffsets.length,slabEntryCount:E.slabEntries.length,sampleCellCapacity:c}),te=this.createPersistentView(t,`packed`,`uint32`,ee.length);for(let[e,r,i]of[[`summary`,u,u.length],[`edges`,l,l.length],[`slabParams`,E.params,E.params.length],[`slabOffsets`,E.slabOffsets,E.slabOffsets.length],[`slabEntries`,E.slabEntries,2*E.slabEntries.length],[`sampleCellCount`,h,1],[`sampleCells`,m,c]])C.push(z(t,{id:`${n}-pack-${e}`,operation:q,variant:`pack`,bindings:[{name:`source`,view:r,type:`u32`,access:`read`},{name:`packed`,view:te,type:`u32`,access:`read_write`}],invocationCount:i,declarations:`override DESTINATION_OFFSET: u32;`,constants:{DESTINATION_OFFSET:ee[e]},body:`packed[packedOffset + DESTINATION_OFFSET + index] = source[sourceOffset + index];`}));r.output?.summary&&C.push(z(t,{id:`${n}-copy-summary`,operation:q,variant:`copy-summary`,bindings:[b,{name:`destination`,view:r.output.summary,type:`f32`,access:`read_write`}],invocationCount:8,body:`destination[destinationOffset + index] = summary[summaryOffset + index];`})),this.views={edges:l,summary:u,slabs:E,sampleCells:m,sampleCellCount:h,sampleGrid:a,packed:te};let F=C.length,I=0,L=!0,R=e=>(I===0&&(L=this.dirty||(this.props.rebuildWhen?.(e)??!1),L&&this.encodedBuildCount++),I++,I===F&&(I=0,L&&(this.dirty=!1)),L);return C.map(t=>{let r=t.condition;return r?(e(r.source===`cpu`),{...t,condition:{id:`${n}-rebuild`,source:`cpu`,evaluate:e=>R(e)&&r.evaluate(e)}}):(e(!t.condition),{...t,condition:{id:`${n}-rebuild`,source:`cpu`,evaluate:R}})})}},et={none:0,border:1,isotropic:2};function tt(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPURipley target must hold 8 elements`);S(`GPURipley`,e.bounds,e.maximumDistance);let n=e.edgeCorrection??`isotropic`;if(!Object.prototype.hasOwnProperty.call(et,n))throw Error(`GPURipley edgeCorrection must be 'none', 'border' or 'isotropic'`);let r=e.seed??0;if(!Number.isInteger(r)||r<0||r>4294967295)throw Error(`GPURipley seed must be a uint32`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,et[n],0,0]),new Uint32Array(t.buffer,t.byteOffset,8)[6]=r,t}var J=`GPURipley`,nt=256,rt=5,it=131072,at=class{id;props;spatialContext;studyWindow;constructor(e){this.id=e.id??`ripley`,this.props=e;let t=this.id;this.spatialContext=n(t,e.spatialContext),this.studyWindow=l(t,e.studyWindow,this.spatialContext),f(t,{...e,parameterLength:8});let{radiusCount:r,output:i}=e;A(t,`radiusCount`,r,1,nt),h(t,r,rt);for(let[e,n]of[[`output.k`,i.k],[`output.l`,i.l],[`output.lMinusR`,i.lMinusR],[`output.pairCorrelation`,i.pairCorrelation],[`output.radii`,i.radii]])if(n&&(I(n,[`float32`],`${t} ${e}`),n.length<r))throw Error(`${t} ${e} must hold radiusCount rows`);if(i.pairCounts&&(M(i.pairCounts,`${t} output.pairCounts`),i.pairCounts.length<r))throw Error(`${t} output.pairCounts must hold radiusCount rows`);j(t,[i.k,i.l,i.lMinusR,i.pairCorrelation,i.pairCounts,i.radii],[e.positions,e.parameters,e.mask])}getCommandNodes(e){let{id:t,props:n}=this,{positions:r,parameters:i,gridSize:a,radiusCount:o,output:s}=n;b(t,e.device,r.length),k(t,e,[r,i,n.mask,s.k,s.l,s.lMinusR,s.pairCorrelation,s.pairCounts,s.radii]);let l=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);l&&k(t,e,[l.edges,l.summary,l.slabs.params,l.slabs.slabOffsets,l.slabs.slabEntries,l.sampleCells,l.sampleCellCount,l.packed]);let u=l?d(e,{id:t,operation:J,parameters:i,parameterLength:i.length,window:l}):void 0,f=u?.parameters??i,p=x(e,{id:t,operation:J,positions:r,parameters:f,gridSize:a,mask:n.mask,studyWindow:l}),h=[...u?.nodes??[],...p.nodes],g=`
const RADIUS_COUNT: u32 = ${o}u;
const ISOTROPIC_SCALE: f32 = ${it}.0;
const WEIGHT_CAP: f32 = 100.0;
// The raw per-annulus pair counts feed K only for the uncorrected estimator, so the other modes
// tally them only when the caller asked for pairCounts.
const RAW_PAIRS: bool = ${!!s.pairCounts};
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;
`,_=l?`${V}
${de}
${w(T(l))}`:``,y=v(e,{id:`${t}-histogram`,operation:J,positions:r,parameters:f,gridSize:a,sortedRows:p.sortedRows,cellOffsets:p.cellOffsets,slotCount:o,channelCount:rt,pairOrder:`ordered`,extraBindings:l?[m(l)]:void 0,declarations:`${g}${_}
// Angle of the arc of a circle of radius d (centered at distance h from a side line) outside it.
fn getSideAngle(h: f32, d: f32) -> f32 {
  if (d > h) {
    return atan2(sqrt((d - h) * (d + h)), h);
  }
  return 0.0;
}

fn getArcOverlap(first: f32, second: f32) -> f32 {
  return max(0.0, first + second - HALF_PI);
}

// Ripley (1977) weight 1 / (fraction of the circle inside the rectangle), capped.
fn getIsotropicWeight(left: f32, bottom: f32, right: f32, top: f32, d: f32) -> f32 {
  let a = getSideAngle(left, d);
  let b = getSideAngle(bottom, d);
  let c = getSideAngle(right, d);
  let e = getSideAngle(top, d);
  let overlap = getArcOverlap(a, b) + getArcOverlap(b, c) + getArcOverlap(c, e) + getArcOverlap(e, a);
  let inside = 1.0 - (a + b + c + e) / PI + overlap / (2.0 * PI);
  return min(1.0 / max(inside, 1.0 / WEIGHT_CAP), WEIGHT_CAP);
}`,focusPrologue:`let mode = u32(readParameter(5u) + 0.5);
    ${l?`var boundaryDistance = lattice.maximumDistance;
    if (mode == 1u || mode == 2u) {
      boundaryDistance = getStudyWindowBoundaryDistance(vec2f(x, y), lattice.maximumDistance);
    }`:`let left = x - lattice.minimumX;
    let bottom = y - lattice.minimumY;
    let right = lattice.maximumX - x;
    let top = lattice.maximumY - y;`}
    var borderCount = 0u;
    // Border estimator: pairs of this focus below its border count, tallied in a register and
    // added once per focus instead of once per pair (integer sums, so the result is identical).
    var borderPairs = 0u;
    if (mode == 1u) {
      let borderDistance = ${l?`min(boundaryDistance, lattice.maximumDistance)`:`min(min(left, right), min(bottom, top))`};
      borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);
    }`,pairAction:`let scaled = pairDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
            let annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
            if (RAW_PAIRS || mode == 0u) {
              accumulate(annulus, 3u, 1u);
            }
            if (mode == 1u) {
              if (annulus < borderCount) {
                accumulate(annulus, 0u, 1u);
                borderPairs++;
              }
            } else if (mode == 2u) {
              ${l?`var weight = 1.0;
              if (boundaryDistance < pairDistance) {
                weight = 1.0 / max(getStudyWindowCircleFraction(vec2f(x, y), pairDistance), 1.0 / WEIGHT_CAP);
              }
              accumulate(annulus, 4u, quantizePairAmount(weight * ISOTROPIC_SCALE));`:`accumulate(annulus, 4u, quantizePairAmount(getIsotropicWeight(left, bottom, right, top, pairDistance) * ISOTROPIC_SCALE));`}
            }`,focusEpilogue:`if (mode == 1u) {
      accumulate(borderCount, 1u, borderPairs);
      accumulate(borderCount, 2u, 1u);
    }`});h.push(...y.nodes);let S=[...l?[m(l)]:[],{name:`accumulators`,view:y.accumulators,type:`u32`,access:`read`},{name:`cellOffsets`,view:p.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`k`,view:s.k,type:`f32`,access:`read_write`}];s.pairCounts&&S.push({name:`pairCounts`,view:s.pairCounts,type:`u32`,access:`read_write`}),h.push(z(e,{id:`${t}-finish`,operation:J,variant:`finish`,bindings:S,invocationCount:o,declarations:`${c}${_}
${g}
const CELL_COUNT: u32 = ${a[0]*a[1]}u;

fn readAccumulator(slot: u32, channel: u32) -> vec2<u32> {
  let word = (slot * ${rt}u + channel) * 2u;
  return vec2<u32>(accumulators[accumulatorsOffset + word], accumulators[accumulatorsOffset + word + 1u]);
}

fn add64(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  let low = first.x + second.x;
  return vec2<u32>(low, first.y + second.y + select(0u, 1u, low < first.x));
}

fn subtract64(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(first.x - second.x, first.y - second.y - select(0u, 1u, first.x < second.x));
}

fn toFloat64(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}

// Exact 64-bit prefix sum of one channel over slots [0, last].
fn getPrefix(channel: u32, last: u32) -> vec2<u32> {
  var sum = vec2<u32>(0u, 0u);
  for (var slot = 0u; slot <= last; slot++) {
    sum = add64(sum, readAccumulator(slot, channel));
  }
  return sum;
}`,body:`let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let area = ${l?`studyWindowSummaryValue(4u)`:`(parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u])`};
  let mode = u32(parameters[parametersOffset + 5u] + 0.5);
  var kValue = getQuietNaN(index);
  if (n >= 2.0 && isFiniteFloat(area) && area > 0.0) {
    if (mode == 1u) {
      let numerator = subtract64(getPrefix(0u, index), getPrefix(1u, index));
      let borderCount = n - toFloat64(getPrefix(2u, index));
      if (borderCount >= 1.0) {
        // spatstat Kest border: lambda = n / A.
        kValue = area * toFloat64(numerator) / (n * borderCount);
      }
    } else if (mode == 2u) {
      kValue = area / (n * (n - 1.0)) * (toFloat64(getPrefix(4u, index)) / ISOTROPIC_SCALE);
    } else {
      kValue = area / (n * (n - 1.0)) * toFloat64(getPrefix(3u, index));
    }
  }
  k[kOffset + index] = kValue;
  ${s.pairCounts?`let annulusCount = readAccumulator(index, 3u);
  pairCounts[pairCountsOffset + index] = select(annulusCount.x, 0xffffffffu, annulusCount.y != 0u);`:``}`}));let C=[{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`k`,view:s.k,type:`f32`,access:`read`}];for(let[e,t]of[[`l`,s.l],[`lMinusR`,s.lMinusR],[`pairCorrelation`,s.pairCorrelation],[`radii`,s.radii]])t&&C.push({name:e,view:t,type:`f32`,access:`read_write`});return C.length>2&&h.push(z(e,{id:`${t}-derive`,operation:J,variant:`derive`,bindings:C,invocationCount:o,declarations:`${c}
${g}`,body:`let nan = getQuietNaN(index);
  let maximumDistance = parameters[parametersOffset + 4u];
  let radius = maximumDistance * f32(index + 1u) / f32(RADIUS_COUNT);
  let kValue = k[kOffset + index];
  let finite = isFiniteFloat(kValue);
  let lValue = select(nan, sqrt(max(kValue, 0.0) / PI), finite);
  ${s.l?`l[lOffset + index] = lValue;`:``}
  ${s.lMinusR?`lMinusR[lMinusROffset + index] = select(nan, lValue - radius, finite);`:``}
  ${s.radii?`radii[radiiOffset + index] = radius;`:``}
  ${s.pairCorrelation?`var previousK = 0.0;
  var previousFinite = true;
  if (index > 0u) {
    previousK = k[kOffset + index - 1u];
    previousFinite = isFiniteFloat(previousK);
  }
  let previousRadius = maximumDistance * f32(index) / f32(RADIUS_COUNT);
  let annulusArea = PI * (radius - previousRadius) * (radius + previousRadius);
  pairCorrelation[pairCorrelationOffset + index] =
    select(nan, (kValue - previousK) / annulusArea, finite && previousFinite);`:``}`})),h}},Y={none:0,donnelly:1,guard:2},ot={"two-sided":0,greater:1,lesser:2},X=65536;function st(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUPointPatternIndices target must hold 8 elements`);if(S(`GPUPointPatternIndices`,e.bounds,e.maximumDistance),e.bounds===`study-window`){if(e.clarkEvansCorrection===void 0)throw Error(`GPUPointPatternIndices needs an explicit clarkEvansCorrection ('guard' or 'none') with a study window`);if(e.clarkEvansCorrection===`donnelly`)throw Error(`GPUPointPatternIndices clarkEvansCorrection 'donnelly' is defined for rectangular windows only; use 'guard' or 'none' with a study window`)}let n=e.clarkEvansCorrection??`donnelly`;if(!Object.prototype.hasOwnProperty.call(Y,n))throw Error(`GPUPointPatternIndices clarkEvansCorrection must be 'none', 'donnelly' or 'guard'`);if(n===`guard`&&(!Number.isFinite(e.guardDistance)||!(e.guardDistance>0)))throw Error(`GPUPointPatternIndices clarkEvansCorrection 'guard' needs a positive guardDistance`);let r=e.alternative??`two-sided`;if(!Object.prototype.hasOwnProperty.call(ot,r))throw Error(`GPUPointPatternIndices alternative must be 'two-sided', 'greater' or 'lesser'`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,Y[n],ot[r],e.guardDistance??0]),t}var Z=`GPUPointPatternIndices`,ct=`
fn getNormalPValue(z: f32, alternative: u32) -> f32 {
  let root = 0.70710678118655;
  if (alternative == 1u) {
    return 0.5 * erfcValue(z * root);
  }
  if (alternative == 2u) {
    return 0.5 * erfcValue(-z * root);
  }
  return erfcValue(abs(z) * root);
}
`,lt=`// Stirling's log-gamma expansion (DLMF 5.11.1), centered at x = a to avoid subtracting
// terms of order a * log(a). The log1p(delta) - delta series also avoids cancellation.
fn getGammaPrefix(shape: f32, value: f32) -> f32 {
  if (shape <= 50.0) {
    return exp(shape * log(value) - value - getLogGamma(shape));
  }
  let delta = (value - shape) / shape;
  var centeredLog: f32;
  if (abs(delta) <= 0.25) {
    var power = -delta * delta;
    centeredLog = 0.5 * power;
    for (var order = 3; order <= 20; order++) {
      power *= -delta;
      centeredLog += power / f32(order);
    }
  } else {
    centeredLog = log(value / shape) - delta;
  }
  let inverse = 1.0 / shape;
  let correction = inverse * (0.08333333333333 - inverse * inverse * 0.00277777777778);
  return exp(shape * centeredLog + 0.5 * log(shape) - 0.91893853320467 - correction);
}

`;function ut(e){let t=e>101;return`
fn getLogGamma(value: f32) -> f32 {
  var x = value;
  var shift = 0.0;
  loop {
    if (x >= 8.0) { break; }
    shift += log(x);
    x += 1.0;
  }
  let inverse = 1.0 / x;
  let inverseSquared = inverse * inverse;
  return (x - 0.5) * log(x) - x + 0.91893853320467 +
    inverse * (0.08333333333333 - inverseSquared * (0.00277777777778 - inverseSquared * 0.00079365079365)) - shift;
}

${t?lt:``}// Returns (lower tail P(a, x), upper tail Q(a, x)) of the regularized incomplete gamma function.
fn getGammaTails(a: f32, x: f32) -> vec2<f32> {
  if (x <= 0.0) {
    return vec2<f32>(0.0, 1.0);
  }
  let prefix = ${t?`getGammaPrefix(a, x)`:`exp(a * log(x) - x - getLogGamma(a))`};
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    var denominator = a;
    for (var iteration = 0; iteration < ${t?2048:500}; iteration++) {
      denominator += 1.0;
      term *= x / denominator;
      sum += term;
      if (abs(term) < abs(sum) * 1.0e-7) { break; }
    }
    let lower = clamp(sum * prefix, 0.0, 1.0);
    return vec2<f32>(lower, 1.0 - lower);
  }
  var b = x + 1.0 - a;
  var c = 1.0e30;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1; iteration < 500; iteration++) {
    let an = -f32(iteration) * (f32(iteration) - a);
    b += 2.0;
    d = an * d + b;
    if (abs(d) < 1.0e-30) { d = 1.0e-30; }
    c = b + an / c;
    if (abs(c) < 1.0e-30) { c = 1.0e-30; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < 1.0e-7) { break; }
  }
  let upper = clamp(prefix * h, 0.0, 1.0);
  return vec2<f32>(1.0 - upper, upper);
}

fn getChiSquarePValue(x: f32, degreesOfFreedom: f32, alternative: u32) -> f32 {
${t?`  let tails = getGammaTails(0.5 * degreesOfFreedom, 0.5 * x);
  let lower = tails.x;
  let upper = tails.y;`:`  var lower: f32;
  var upper: f32;
  if (degreesOfFreedom <= 100.0) {
    let tails = getGammaTails(0.5 * degreesOfFreedom, 0.5 * x);
    lower = tails.x;
    upper = tails.y;
  } else {
    let scale = 2.0 / (9.0 * degreesOfFreedom);
    let z = (pow(x / degreesOfFreedom, 1.0 / 3.0) - (1.0 - scale)) / sqrt(scale);
    lower = getNormalPValue(z, 2u);
    upper = getNormalPValue(z, 1u);
  }`}
  if (alternative == 1u) { return upper; }
  if (alternative == 2u) { return lower; }
  return min(1.0, 2.0 * min(lower, upper));
}
`}var dt=class{id;props;spatialContext;studyWindow;constructor(e){this.id=e.id??`point-pattern-indices`,this.props=e;let t=this.id;this.spatialContext=n(t,e.spatialContext),this.studyWindow=l(t,e.studyWindow,this.spatialContext);let{output:r}=e,i=f(t,{...e,parameterLength:8}),a=r.nearestNeighborDistances||r.nearestNeighborIds||r.clarkEvans,o=r.quadratCounts||r.quadratStatistics||r.quadratAreas;if(!a&&!o)throw Error(`${t} output needs at least one view`);if(r.nearestNeighborDistances&&(I(r.nearestNeighborDistances,[`float32`],`${t} output.nearestNeighborDistances`),r.nearestNeighborDistances.length!==i))throw Error(`${t} output.nearestNeighborDistances length must equal positions length`);if(r.nearestNeighborIds&&(M(r.nearestNeighborIds,`${t} output.nearestNeighborIds`),r.nearestNeighborIds.length!==i))throw Error(`${t} output.nearestNeighborIds length must equal positions length`);if(r.clarkEvans&&(I(r.clarkEvans,[`float32`],`${t} output.clarkEvans`),r.clarkEvans.length<7))throw Error(`${t} output.clarkEvans must hold 7 rows`);if(o){let n=e.quadratGrid;if(!n||!Number.isInteger(n[0])||!Number.isInteger(n[1])||n[0]<1||n[1]<1||n[0]*n[1]>65536)throw Error(`${t} quadratGrid must be two positive integers with columns * rows <= ${X}`);let i=e.quadratSamples??16;if(!Number.isInteger(i)||i<1||i**2*n[0]*n[1]>16777216)throw Error(`${t} quadratSamples squared times the quadrat count must be at most 16777216`);if(r.quadratAreas&&(I(r.quadratAreas,[`float32`],`${t} output.quadratAreas`),r.quadratAreas.length!==n[0]*n[1]))throw Error(`${t} output.quadratAreas must hold columns * rows rows`);if(r.quadratCounts&&(M(r.quadratCounts,`${t} output.quadratCounts`),r.quadratCounts.length!==n[0]*n[1]))throw Error(`${t} output.quadratCounts must hold columns * rows rows`);if(r.quadratStatistics&&(I(r.quadratStatistics,[`float32`],`${t} output.quadratStatistics`),r.quadratStatistics.length<7))throw Error(`${t} output.quadratStatistics must hold 7 rows`)}else if(e.quadratGrid)throw Error(`${t} quadratGrid needs output.quadratCounts, output.quadratAreas or output.quadratStatistics`);j(t,[r.nearestNeighborDistances,r.nearestNeighborIds,r.clarkEvans,r.quadratCounts,r.quadratAreas,r.quadratStatistics],[e.positions,e.parameters,e.mask])}getCommandNodes(e){let{id:t,props:n}=this,{positions:r,parameters:i,gridSize:a,output:o}=n;b(t,e.device,r.length),k(t,e,[r,i,n.mask,o.nearestNeighborDistances,o.nearestNeighborIds,o.clarkEvans,o.quadratCounts,o.quadratAreas,o.quadratStatistics]);let s=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);s&&k(t,e,[s.edges,s.summary,s.slabs.params,s.slabs.slabOffsets,s.slabs.slabEntries,s.sampleCells,s.sampleCellCount,s.packed]);let l=s?d(e,{id:t,operation:Z,parameters:i,parameterLength:i.length,window:s}):void 0,f=l?.parameters??i,p=s?`${V}
${de}
${w(T(s))}`:``,h=r.length,_=a[0]*a[1],v=x(e,{id:t,operation:Z,positions:r,parameters:f,gridSize:a,mask:n.mask,studyWindow:s}),S=[...l?.nodes??[],...v.nodes],E=u(a);if(o.nearestNeighborDistances||o.nearestNeighborIds||o.clarkEvans){let n=o.nearestNeighborDistances??P(e,`${t}-nearest-distances`,`float32`,h),i=o.nearestNeighborIds??P(e,`${t}-nearest-ids`,`uint32`,h),l=g(e,{id:t,operation:Z,positions:r,sortedRows:v.sortedRows,cellOffsets:v.cellOffsets,gridSize:a});if(S.push(...l.nodes,z(e,{id:`${t}-nearest`,operation:Z,variant:`nearest-neighbor`,bindings:[{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`sortedRows`,view:v.sortedRows,type:`u32`,access:`read`},{name:`sortedPoints`,view:l.sortedPoints,type:`u32`,access:`read`},{name:`cellOffsets`,view:v.cellOffsets,type:`u32`,access:`read`},{name:`distances`,view:n,type:`f32`,access:`read_write`},{name:`ids`,view:i,type:`u32`,access:`read_write`}],invocationCount:h,declarations:E,body:`let lattice = readLattice();
  let focus = sortedRows[sortedRowsOffset + index];
  var nearestDistance = getQuietNaN(index);
  var nearestId = 0xffffffffu;
  if (index < cellOffsets[cellOffsetsOffset + CELL_COUNT] && lattice.valid) {
    let x = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u]);
    let y = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u + 1u]);
    let queryColumn = i32(getCellColumn(lattice, x));
    let queryRow = i32(getCellRow(lattice, y));
    let lastColumn = i32(lattice.columns) - 1;
    let lastRow = i32(lattice.rows) - 1;
    var bestSquared = 3.0e38;
    var bestId = 0xffffffffu;
    let ringCount = max(lattice.columns, lattice.rows);
    // Absolute slack covering f32 rounding of the cell assignment (as in the kNN search).
    let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
      (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
    for (var ringIndex = 0u; ringIndex < ringCount; ringIndex++) {
      let ring = i32(ringIndex);
      let firstColumn = max(queryColumn - ring, 0);
      let endColumn = min(queryColumn + ring, lastColumn);
      let firstRow = max(queryRow - ring, 0);
      let endRow = min(queryRow + ring, lastRow);
      for (var cellRow = firstRow; cellRow <= endRow; cellRow++) {
        let rowBase = u32(cellRow) * lattice.columns;
        // The top and bottom rows of the ring are full (one contiguous slot range, as cells are
        // row major); the others only hold its two end cells.
        let isFullRow = abs(cellRow - queryRow) == ring;
        let step = select(max(2 * ring, 1), 1, isFullRow);
        var columnRangeEnd = endColumn;
        if (!isFullRow) {
          columnRangeEnd = queryColumn + ring;
        }
        var cellColumn = select(queryColumn - ring, firstColumn, isFullRow);
        loop {
          if (cellColumn > columnRangeEnd) { break; }
          if (cellColumn >= 0 && cellColumn <= lastColumn) {
            var begin = cellOffsets[cellOffsetsOffset + rowBase + u32(cellColumn)];
            var end = cellOffsets[cellOffsetsOffset + rowBase + u32(cellColumn) + 1u];
            if (isFullRow) {
              end = cellOffsets[cellOffsetsOffset + rowBase + u32(endColumn) + 1u];
            }
            for (var slot = begin; slot < end; slot++) {
              let neighbor = sortedPoints[sortedPointsOffset + slot * 3u + 2u];
              if (neighbor == focus) {
                continue;
              }
              let deltaX = bitcast<f32>(sortedPoints[sortedPointsOffset + slot * 3u]) - x;
              let deltaY = bitcast<f32>(sortedPoints[sortedPointsOffset + slot * 3u + 1u]) - y;
              let distanceSquared = deltaX * deltaX + deltaY * deltaY;
              if (distanceSquared < bestSquared || (distanceSquared == bestSquared && neighbor < bestId)) {
                bestSquared = distanceSquared;
                bestId = neighbor;
              }
            }
          }
          if (isFullRow) { break; }
          cellColumn += step;
        }
      }
      // Clearance of the visited box on each side; a side at the lattice edge has nothing beyond
      // it. Every unvisited point is at least that far away, so stop once the best is strictly
      // closer (ties at the best distance therefore always resolve to the lowest ID).
      var clearance = 3.0e38;
      var open = false;
      if (queryColumn - ring > 0) {
        clearance = min(clearance, x - (lattice.minimumX + f32(queryColumn - ring) * lattice.cellWidth));
        open = true;
      }
      if (queryColumn + ring < lastColumn) {
        clearance = min(clearance, lattice.minimumX + f32(queryColumn + ring + 1) * lattice.cellWidth - x);
        open = true;
      }
      if (queryRow - ring > 0) {
        clearance = min(clearance, y - (lattice.minimumY + f32(queryRow - ring) * lattice.cellHeight));
        open = true;
      }
      if (queryRow + ring < lastRow) {
        clearance = min(clearance, lattice.minimumY + f32(queryRow + ring + 1) * lattice.cellHeight - y);
        open = true;
      }
      if (!open) {
        break;
      }
      let safeClearance = max(clearance - slack, 0.0);
      if (bestId != 0xffffffffu && bestSquared < safeClearance * safeClearance) {
        break;
      }
    }
    if (bestId != 0xffffffffu) {
      nearestDistance = sqrt(bestSquared);
      nearestId = bestId;
    }
  }
  distances[distancesOffset + focus] = nearestDistance;
  ids[idsOffset + focus] = nearestId;`})),o.clarkEvans){let i=P(e,`${t}-nearest-terms`,`float32`,h),a=P(e,`${t}-nearest-total`,`float32`,1);S.push(z(e,{id:`${t}-nearest-terms`,operation:Z,variant:`nearest-terms`,bindings:[{name:`distances`,view:n,type:`f32`,access:`read`},{name:`terms`,view:i,type:`f32`,access:`read_write`}],invocationCount:h,declarations:c,body:`let distance = distances[distancesOffset + index];
  terms[termsOffset + index] = select(0.0, distance, isFiniteFloat(distance));`}),...C(e,{id:`${t}-nearest-sum`,operation:Z,input:i,output:a}),z(e,{id:`${t}-clark-evans`,operation:Z,variant:`clark-evans`,bindings:[...s?[m(s)]:[],{name:`total`,view:a,type:`f32`,access:`read`},{name:`cellOffsets`,view:v.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`clarkEvans`,view:o.clarkEvans,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${c}${p}
${le}
${ct}
const CELL_COUNT: u32 = ${_}u;`,body:`let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let width = parameters[parametersOffset + 2u] - parameters[parametersOffset];
  let height = parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u];
  let area = ${s?`studyWindowSummaryValue(4u)`:`width * height`};
  let correction = u32(parameters[parametersOffset + 5u] + 0.5);
  let alternative = u32(parameters[parametersOffset + 6u] + 0.5);
  let nan = getQuietNaN(index);
  var observed = nan;
  var expected = nan;
  var ratio = nan;
  var standardError = nan;
  var zScore = nan;
  var pValue = nan;
  if (n >= 2.0 && isFiniteFloat(area) && area > 0.0) {
    observed = total[totalOffset] / n;
    expected = 0.5 * sqrt(area / n);
    standardError = 0.26136 * sqrt(area) / n;
    if (correction == 1u) {
      // Donnelly (1978) edge correction for a rectangle of perimeter P. sqrt(A / n^5) is
      // evaluated as sqrt(A) / (n^2 sqrt(n)) to stay inside the f32 range for large n.
      let perimeter = 2.0 * (width + height);
      expected += (0.0514 + 0.041 / sqrt(n)) * perimeter / n;
      standardError = sqrt(0.0703 * area / (n * n) + 0.037 * perimeter * sqrt(area) / (n * n * sqrt(n)));
    }
    ${s?`if (correction == 1u) { expected = nan; standardError = nan; }`:``}${s?`
    `:``}ratio = observed / expected;
    zScore = (observed - expected) / standardError;
    pValue = getNormalPValue(zScore, alternative);
  }
  clarkEvans[clarkEvansOffset] = n;
  clarkEvans[clarkEvansOffset + 1u] = observed;
  clarkEvans[clarkEvansOffset + 2u] = expected;
  clarkEvans[clarkEvansOffset + 3u] = ratio;
  clarkEvans[clarkEvansOffset + 4u] = standardError;
  clarkEvans[clarkEvansOffset + 5u] = zScore;
  clarkEvans[clarkEvansOffset + 6u] = pValue;`}));let l=P(e,`${t}-guard-terms`,`float32`,h),u=P(e,`${t}-guard-focus`,`float32`,h),d=P(e,`${t}-guard-total`,`float32`,1),g=P(e,`${t}-guard-count`,`float32`,1);S.push(z(e,{id:`${t}-guard-terms`,operation:Z,variant:`guard-terms`,bindings:[{name:`positions`,view:r,type:`f32`,access:`read`},{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`distances`,view:n,type:`f32`,access:`read`},{name:`guardTerms`,view:l,type:`f32`,access:`read_write`},{name:`guardFocus`,view:u,type:`f32`,access:`read_write`},...s?[m(s)]:[]],invocationCount:h,declarations:`${c}${p}`,body:`let distance = distances[distancesOffset + index];
  let guardDistance = parameters[parametersOffset + 7u];
  var focus = false;
  if (parameters[parametersOffset + 5u] == 2.0 && isFiniteFloat(distance) &&
      isFiniteFloat(guardDistance) && guardDistance > 0.0) {
    let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
    let boundaryDistance = ${s?`getStudyWindowBoundaryDistance(point, guardDistance)`:`min(min(point.x - parameters[parametersOffset], parameters[parametersOffset + 2u] - point.x),
      min(point.y - parameters[parametersOffset + 1u], parameters[parametersOffset + 3u] - point.y))`};
    focus = boundaryDistance >= guardDistance;
  }
  guardTerms[guardTermsOffset + index] = select(0.0, distance, focus);
  guardFocus[guardFocusOffset + index] = select(0.0, 1.0, focus);`}),...C(e,{id:`${t}-guard-sum`,operation:Z,input:l,output:d}),...C(e,{id:`${t}-guard-count-sum`,operation:Z,input:u,output:g}),z(e,{id:`${t}-guard-clark-evans`,operation:Z,variant:`guard-clark-evans`,bindings:[{name:`guardTotal`,view:d,type:`f32`,access:`read`},{name:`guardCount`,view:g,type:`f32`,access:`read`},{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`clarkEvans`,view:o.clarkEvans,type:`f32`,access:`read_write`},...s?[m(s)]:[]],invocationCount:1,declarations:`${c}${p}
${le}
${ct}`,body:`if (parameters[parametersOffset + 5u] != 2.0) { return; }
  let count = clarkEvans[clarkEvansOffset];
  let focusCount = guardCount[guardCountOffset];
  let area = ${s?`studyWindowSummaryValue(4u)`:`(parameters[parametersOffset + 2u] - parameters[parametersOffset]) * (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u])`};
  let nan = getQuietNaN(index);
  for (var row = 1u; row < 7u; row++) { clarkEvans[clarkEvansOffset + row] = nan; }
  if (count >= 2.0 && focusCount > 0.0 && isFiniteFloat(area) && area > 0.0) {
    let observed = guardTotal[guardTotalOffset] / focusCount;
    let expected = 0.5 * sqrt(area / count);
    let standardError = 0.26136 * sqrt(area / (count * focusCount));
    let zScore = (observed - expected) / standardError;
    clarkEvans[clarkEvansOffset + 1u] = observed;
    clarkEvans[clarkEvansOffset + 2u] = expected;
    clarkEvans[clarkEvansOffset + 3u] = observed / expected;
    clarkEvans[clarkEvansOffset + 4u] = standardError;
    clarkEvans[clarkEvansOffset + 5u] = zScore;
    clarkEvans[clarkEvansOffset + 6u] = getNormalPValue(zScore, u32(parameters[parametersOffset + 6u] + 0.5));
  }`}))}}if(o.quadratCounts||o.quadratStatistics||o.quadratAreas){let[i,a]=n.quadratGrid,l=i*a,u=o.quadratCounts??P(e,`${t}-quadrat-counts`,`uint32`,l);S.push(y(e,`${t}-quadrat-clear`,Z,u),z(e,{id:`${t}-quadrat-counts`,operation:Z,variant:`quadrat-counts`,bindings:[{name:`positions`,view:r,type:`f32`,access:`read`},{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`cellKeys`,view:v.cellKeys,type:`u32`,access:`read`},{name:`quadratCounts`,view:u,type:`atomic<u32>`,access:`read_write`}],invocationCount:h,declarations:`${E}
const QUADRAT_COLUMNS: u32 = ${i}u;
const QUADRAT_ROWS: u32 = ${a}u;`,body:`if (cellKeys[cellKeysOffset + index] < CELL_COUNT) {
    let x = positions[positionsOffset + index * 2u];
    let y = positions[positionsOffset + index * 2u + 1u];
    let width = readParameter(2u) - readParameter(0u);
    let height = readParameter(3u) - readParameter(1u);
    var column = 0u;
    var row = 0u;
    if (width > 0.0) {
      column = min(u32(floor((x - readParameter(0u)) / width * f32(QUADRAT_COLUMNS))), QUADRAT_COLUMNS - 1u);
    }
    if (height > 0.0) {
      row = min(u32(floor((y - readParameter(1u)) / height * f32(QUADRAT_ROWS))), QUADRAT_ROWS - 1u);
    }
    atomicAdd(&quadratCounts[quadratCountsOffset + row * QUADRAT_COLUMNS + column], 1u);
  }`}));let d=s?P(e,`${t}-quadrat-area-samples`,`uint32`,l):void 0,g=s?P(e,`${t}-quadrat-area-totals`,`uint32`,3):void 0;if(s&&d&&g){let r=n.quadratSamples??16;S.push(z(e,{id:`${t}-quadrat-area-samples`,operation:Z,variant:`quadrat-area-samples`,bindings:[m(s),{name:`areaSamples`,view:d,type:`u32`,access:`read_write`}],invocationCount:l,declarations:p,body:`let minimum = vec2f(studyWindowSummaryValue(0u), studyWindowSummaryValue(1u));
  let size = (vec2f(studyWindowSummaryValue(2u), studyWindowSummaryValue(3u)) - minimum) / vec2f(${i}.0, ${a}.0);
  let cell = vec2f(f32(index % ${i}u), f32(index / ${i}u));
  var count = 0u;
  for (var sample = 0u; sample < ${r**2}u; sample++) {
    let fraction = (vec2f(f32(sample % ${r}u), f32(sample / ${r}u)) + vec2f(0.5)) / ${r}.0;
    let point = minimum + (cell + fraction) * size;
    if (isInsideStudyWindow(point)) { count++; }
  }
  areaSamples[areaSamplesOffset + index] = count;`}),z(e,{id:`${t}-quadrat-area-totals`,operation:Z,variant:`quadrat-area-totals`,bindings:[{name:`areaSamples`,view:d,type:`u32`,access:`read`},{name:`areaTotals`,view:g,type:`u32`,access:`read_write`}],invocationCount:1,body:`var total = 0u;
  var count = 0u;
  var equal = true;
  for (var quadrat = 0u; quadrat < ${l}u; quadrat++) {
    let samples = areaSamples[areaSamplesOffset + quadrat];
    total += samples;
    count += select(0u, 1u, samples > 0u);
    equal = equal && samples == areaSamples[areaSamplesOffset];
  }
  areaTotals[areaTotalsOffset] = total;
  areaTotals[areaTotalsOffset + 1u] = count;
  areaTotals[areaTotalsOffset + 2u] = select(0u, 1u, equal);`}))}o.quadratAreas&&S.push(z(e,{id:`${t}-quadrat-areas`,operation:Z,variant:`quadrat-areas`,bindings:[{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`quadratAreas`,view:o.quadratAreas,type:`f32`,access:`read_write`},...s&&d&&g?[m(s),{name:`areaSamples`,view:d,type:`u32`,access:`read`},{name:`areaTotals`,view:g,type:`u32`,access:`read`}]:[]],invocationCount:l,declarations:`${c}${p}`,body:s?`let total = areaTotals[areaTotalsOffset];
  var area = getQuietNaN(index);
  if (total > 0u && isStudyWindowUsable()) {
    area = f32(areaSamples[areaSamplesOffset + index]) * studyWindowSummaryValue(4u) / f32(total);
  }
  quadratAreas[quadratAreasOffset + index] = area;`:`quadratAreas[quadratAreasOffset + index] =
    (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]) / ${l}.0;`})),o.quadratStatistics&&S.push(z(e,{id:`${t}-quadrat-statistics`,operation:Z,variant:`quadrat-statistics`,bindings:[{name:`quadratCounts`,view:u,type:`u32`,access:`read`},...d&&g?[{name:`areaSamples`,view:d,type:`u32`,access:`read`},{name:`areaTotals`,view:g,type:`u32`,access:`read`}]:[],{name:`parameters`,view:f,type:`f32`,access:`read`},{name:`cellOffsets`,view:v.cellOffsets,type:`u32`,access:`read`},{name:`quadratStatistics`,view:o.quadratStatistics,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${c}
${le}
${ct}
${ut(l)}
const CELL_COUNT: u32 = ${_}u;
const QUADRAT_COUNT: u32 = ${l}u;`,body:s?`let nan = getQuietNaN(index);
  let m = f32(areaTotals[areaTotalsOffset + 1u]);
  let total = f32(areaTotals[areaTotalsOffset]);
  let equalAreas = areaTotals[areaTotalsOffset + 2u] != 0u;
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mean = select(nan, n / m, m > 0.0);
  var squares = 0.0;
  var chiSquare = 0.0;
  for (var quadrat = 0u; quadrat < QUADRAT_COUNT; quadrat++) {
    let samples = areaSamples[areaSamplesOffset + quadrat];
    if (samples > 0u) {
      let expected = select(n * f32(samples) / total, mean, equalAreas);
      let centered = f32(quadratCounts[quadratCountsOffset + quadrat]) - expected;
      squares += centered * centered;
      chiSquare += centered * centered / expected;
    }
  }
  let variance = select(nan, squares / (m - 1.0), m > 1.0);
  var ratio = select(nan, chiSquare / (m - 1.0), m > 1.0 && n > 0.0);
  if (equalAreas) {
    // Preserve the legacy operations exactly when every quadrat has the same positive area.
    ratio = select(nan, variance / mean, m > 1.0 && mean > 0.0);
    chiSquare = (m - 1.0) * ratio;
  }
  if (!(m > 1.0 && n > 0.0)) { chiSquare = nan; }
  quadratStatistics[quadratStatisticsOffset] = m;
  quadratStatistics[quadratStatisticsOffset + 1u] = mean;
  quadratStatistics[quadratStatisticsOffset + 2u] = variance;
  quadratStatistics[quadratStatisticsOffset + 3u] = ratio;
  quadratStatistics[quadratStatisticsOffset + 4u] = chiSquare;
  quadratStatistics[quadratStatisticsOffset + 5u] = m - 1.0;
  quadratStatistics[quadratStatisticsOffset + 6u] = select(nan,
    getChiSquarePValue(chiSquare, m - 1.0, u32(parameters[parametersOffset + 6u] + 0.5)),
    isFiniteFloat(chiSquare));`:`let nan = getQuietNaN(index);
  let m = f32(QUADRAT_COUNT);
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mean = n / m;
  // One serial fixed-order loop: bitwise reproducible.
  var squares = 0.0;
  for (var quadrat = 0u; quadrat < QUADRAT_COUNT; quadrat++) {
    let centered = f32(quadratCounts[quadratCountsOffset + quadrat]) - mean;
    squares += centered * centered;
  }
  let hasVariance = QUADRAT_COUNT > 1u;
  let variance = select(nan, squares / (m - 1.0), hasVariance);
  let ratio = select(nan, variance / mean, hasVariance && mean > 0.0);
  quadratStatistics[quadratStatisticsOffset] = m;
  quadratStatistics[quadratStatisticsOffset + 1u] = mean;
  quadratStatistics[quadratStatisticsOffset + 2u] = variance;
  quadratStatistics[quadratStatisticsOffset + 3u] = ratio;
  let chiSquare = select(nan, (m - 1.0) * ratio, isFiniteFloat(ratio));
  quadratStatistics[quadratStatisticsOffset + 4u] = chiSquare;
  quadratStatistics[quadratStatisticsOffset + 5u] = m - 1.0;
  quadratStatistics[quadratStatisticsOffset + 6u] = select(
    nan,
    getChiSquarePValue(chiSquare, m - 1.0, u32(parameters[parametersOffset + 6u] + 0.5)),
    isFiniteFloat(chiSquare));`}))}return S}},ft={none:0,border:1,"kaplan-meier":2,hanisch:3};function pt(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPURipleyDistanceFunctions target must hold 8 elements`);S(`GPURipleyDistanceFunctions`,e.bounds,e.maximumDistance);let n=e.edgeCorrection??`border`;if(!Object.prototype.hasOwnProperty.call(ft,n))throw Error(`GPURipleyDistanceFunctions edgeCorrection must be 'none', 'border', 'kaplan-meier' or 'hanisch'`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,ft[n],0,0]),t}var Q=`GPURipleyDistanceFunctions`,mt=256,ht=2**24,$=8,gt=64,_t=class{id;props;spatialContext;studyWindow;constructor(e){this.id=e.id??`ripley-distance-functions`,this.props=e;let t=this.id;this.spatialContext=n(t,e.spatialContext),this.studyWindow=l(t,e.studyWindow,this.spatialContext),f(t,{...e,parameterLength:8});let{radiusCount:r,referenceGrid:i,output:a}=e;if(A(t,`radiusCount`,r,1,mt),h(t,r,$),i.length!==2||!Number.isInteger(i[0])||!Number.isInteger(i[1])||i[0]<1||i[1]<1||i[0]*i[1]>ht)throw Error(`${t} referenceGrid must be two positive integers with a product of at most ${ht}`);if(!a.g&&!a.f&&!a.j)throw Error(`${t} output needs at least one of g, f or j`);for(let[e,n]of[[`output.g`,a.g],[`output.f`,a.f],[`output.j`,a.j],[`output.radii`,a.radii]])if(n&&(I(n,[`float32`],`${t} ${e}`),n.length<r))throw Error(`${t} ${e} must hold radiusCount rows`);j(t,[a.g,a.f,a.j,a.radii],[e.positions,e.parameters,e.mask])}getCommandNodes(e){let{id:t,props:n}=this,{positions:r,parameters:i,gridSize:a,radiusCount:o,referenceGrid:s,output:l}=n;b(t,e.device,r.length),k(t,e,[r,i,n.mask,l.g,l.f,l.j,l.radii]);let f=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);f&&k(t,e,[f.edges,f.summary,f.slabs.params,f.slabs.slabOffsets,f.slabs.slabEntries,f.sampleCells,f.sampleCellCount,f.packed]);let p=f?d(e,{id:t,operation:Q,parameters:i,parameterLength:i.length,window:f}):void 0,h=p?.parameters??i,g=f?T(f):void 0,_=g?`${V}
${de}
${w(g)}`:``,y=s[0]*s[1],S=x(e,{id:t,operation:Q,positions:r,parameters:h,gridSize:a,mask:n.mask,studyWindow:f}),C=[...p?.nodes??[],...S.nodes],E=`
const RADIUS_COUNT: u32 = ${o}u;
const PI: f32 = 3.14159265358979;
`,D=f?P(e,`${t}-hanisch-counts`,`uint32`,4*o+1):void 0,O=f?[m(f),{name:`erosionCounts`,view:D,type:`u32`,access:`read`}]:[];if(f){let n=P(e,`${t}-hanisch-histogram`,`uint32`,4*o+1);C.push(ce(e,{id:`${t}-hanisch-clear`,operation:Q,view:n,type:`u32`,value:`0u`}),z(e,{id:`${t}-hanisch-reference`,operation:Q,variant:`hanisch-reference`,bindings:[m(f),{name:`parameters`,view:h,type:`f32`,access:`read`},{name:`borderHistogram`,view:n,type:`atomic<u32>`,access:`read_write`}],invocationCount:y,declarations:`${u(a)}${_}`,body:`let lattice = readLattice();
  if (!lattice.valid || u32(parameters[parametersOffset + 5u] + 0.5) != 3u) { return; }
  let point = vec2f(lattice.minimumX, lattice.minimumY) +
    (vec2f(f32(index % ${s[0]}u), f32(index / ${s[0]}u)) + vec2f(0.5)) *
    vec2f(lattice.maximumX - lattice.minimumX, lattice.maximumY - lattice.minimumY) /
    vec2f(${s[0]}.0, ${s[1]}.0);
  if (isInsideStudyWindow(point)) {
    let distance = min(getStudyWindowBoundaryDistance(point, lattice.maximumDistance), lattice.maximumDistance);
    let slot = min(u32(floor(distance / lattice.maximumDistance * ${4*o}.0)), ${4*o}u);
    atomicAdd(&borderHistogram[borderHistogramOffset + slot], 1u);
  }`}),z(e,{id:`${t}-hanisch-suffix`,operation:Q,variant:`hanisch-suffix`,bindings:[{name:`borderHistogram`,view:n,type:`u32`,access:`read`},{name:`erosionCounts`,view:D,type:`u32`,access:`read_write`}],invocationCount:4*o+1,body:`var total = 0u;
  for (var slot = index; slot <= ${4*o}u; slot++) { total += borderHistogram[borderHistogramOffset + slot]; }
  erosionCounts[erosionCountsOffset + index] = total;`}))}let A=`${E}${_}
const HANISCH_SCALE: f32 = 16384.0;
const HANISCH_CAP: f32 = 1024.0;

// Fixed-point Hanisch weight |W| / |W eroded by distance| of one observation; at most 2^24.
fn getHanischAmount(lattice: Lattice, distance: f32) -> u32 {
  ${f?`let scaled = clamp(distance / lattice.maximumDistance * ${4*o}.0, 0.0, ${4*o}.0);
  let lower = u32(floor(scaled));
  let upper = min(lower + 1u, ${4*o}u);
  let remaining = mix(f32(erosionCounts[erosionCountsOffset + lower]), f32(erosionCounts[erosionCountsOffset + upper]), scaled - f32(lower));
  let total = f32(erosionCounts[erosionCountsOffset]);
  // A cancels in |W| / (A * remaining / total).
  let ratio = select(HANISCH_CAP, min(total / remaining, HANISCH_CAP), remaining > 0.0);`:`let width = lattice.maximumX - lattice.minimumX;
  let height = lattice.maximumY - lattice.minimumY;
  let eroded = max(width - 2.0 * distance, 0.0) * max(height - 2.0 * distance, 0.0);
  let ratio = select(HANISCH_CAP, min(width * height / eroded, HANISCH_CAP), eroded > 0.0);`}
  return u32(ratio * HANISCH_SCALE + 0.5);
}

// Slot of a censoring distance; slots at or above RADIUS_COUNT lie beyond every radius.
fn getCensorSlot(distance: f32, step: f32) -> u32 {
  return min(u32(max(ceil(distance / step) - 1.0, 0.0)), RADIUS_COUNT);
}
`,j=v(e,{id:`${t}-g`,operation:Q,positions:r,parameters:h,gridSize:a,sortedRows:S.sortedRows,cellOffsets:S.cellOffsets,slotCount:o,channelCount:$,pairOrder:`ordered`,extraBindings:O,declarations:A,focusPrologue:`var nearestDistance = lattice.maximumDistance;
    var hasNeighbor = false;
    let borderDistance = ${f?`min(getStudyWindowBoundaryDistance(vec2f(x, y), lattice.maximumDistance), lattice.maximumDistance)`:`min(min(x - lattice.minimumX, lattice.maximumX - x), min(y - lattice.minimumY, lattice.maximumY - y))`};
    let borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);`,pairAction:`if (!hasNeighbor || pairDistance < nearestDistance) {
              nearestDistance = pairDistance;
            }
            hasNeighbor = true;`,focusEpilogue:`accumulate(borderCount, 3u, 1u);
    var annulus = 0u;
    if (hasNeighbor) {
      let scaled = nearestDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
      annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
      accumulate(annulus, 0u, 1u);
      if (annulus < borderCount) {
        accumulate(annulus, 1u, 1u);
        accumulate(borderCount, 2u, 1u);
      }
    }
    // Kaplan-Meier and Hanisch: observed when the neighbor is no farther than the border.
    if (hasNeighbor && nearestDistance <= borderDistance) {
      accumulate(annulus, 4u, 1u);
      accumulate(annulus, 6u, getHanischAmount(lattice, nearestDistance));
    } else {
      accumulate(getCensorSlot(borderDistance, lattice.maximumDistance / f32(RADIUS_COUNT)), 5u, 1u);
      if (!hasNeighbor && borderDistance >= lattice.maximumDistance) {
        accumulate(0u, 7u, getHanischAmount(lattice, lattice.maximumDistance));
      }
    }`});C.push(...j.nodes);let M=P(e,`${t}-f-accumulators`,`uint32`,(o+1)*$*2);C.push(ce(e,{id:`${t}-f-clear`,operation:Q,view:M,type:`u32`,value:`0u`}),z(e,{id:`${t}-f-reference`,operation:Q,variant:`f-reference`,bindings:[...O,{name:`parameters`,view:h,type:`f32`,access:`read`},{name:`sortedPoints`,view:j.sortedPoints,type:`u32`,access:`read`},{name:`cellOffsets`,view:S.cellOffsets,type:`u32`,access:`read`},{name:`referenceAccumulators`,view:M,type:`atomic<u32>`,access:`read_write`}],invocationCount:y,workgroupSize:gt,guardIndex:!1,declarations:`${u(a)}
${A}
const REFERENCE_COLUMNS: u32 = ${s[0]}u;
const REFERENCE_ROWS: u32 = ${s[1]}u;
const CHANNELS: u32 = ${$}u;

const ACCUMULATOR_COUNT: u32 = ${(o+1)*$}u;
var<workgroup> localReferences: array<atomic<u32>, ${(o+1)*$}>;

// Workgroup partial sums (they fit 32 bits, see REFERENCE_WORKGROUP_SIZE).
fn addReference(slot: u32, channel: u32, amount: u32) {
  atomicAdd(&localReferences[slot * CHANNELS + channel], amount);
}`,body:`for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += ${gt}u) {
    atomicStore(&localReferences[word], 0u);
  }
  workgroupBarrier();
  let lattice = readLattice();
  if (index < INVOCATION_COUNT && lattice.valid) {
    let referenceColumn = index % REFERENCE_COLUMNS;
    let referenceRow = index / REFERENCE_COLUMNS;
    let x = lattice.minimumX + (f32(referenceColumn) + 0.5) * (lattice.maximumX - lattice.minimumX) / f32(REFERENCE_COLUMNS);
    let y = lattice.minimumY + (f32(referenceRow) + 0.5) * (lattice.maximumY - lattice.minimumY) / f32(REFERENCE_ROWS);
    ${f?`if (isInsideStudyWindow(vec2f(x, y))) {
    `:``}let borderDistance = ${f?`min(getStudyWindowBoundaryDistance(vec2f(x, y), lattice.maximumDistance), lattice.maximumDistance)`:`min(min(x - lattice.minimumX, lattice.maximumX - x), min(y - lattice.minimumY, lattice.maximumY - y))`};
    let borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);
    var nearestSquared = lattice.radiusSquared;
    var hasEvent = false;
    let column = getCellColumn(lattice, x);
    let row = getCellRow(lattice, y);
    let firstColumn = max(column, 1u) - 1u;
    let lastColumn = min(column + 1u, lattice.columns - 1u);
    let firstRow = max(row, 1u) - 1u;
    let lastRow = min(row + 1u, lattice.rows - 1u);
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = cellRow * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      for (var candidateSlot = begin; candidateSlot < end; candidateSlot++) {
        let deltaX = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u]) - x;
        let deltaY = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u + 1u]) - y;
        let distanceSquared = deltaX * deltaX + deltaY * deltaY;
        if (distanceSquared <= lattice.radiusSquared && (!hasEvent || distanceSquared < nearestSquared)) {
          nearestSquared = distanceSquared;
          hasEvent = true;
        }
      }
    }
    addReference(borderCount, 3u, 1u);
    let nearestDistance = sqrt(nearestSquared);
    var annulus = 0u;
    if (hasEvent) {
      let scaled = nearestDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
      annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
      addReference(annulus, 0u, 1u);
      if (annulus < borderCount) {
        addReference(annulus, 1u, 1u);
        addReference(borderCount, 2u, 1u);
      }
    }
    if (hasEvent && nearestDistance <= borderDistance) {
      addReference(annulus, 4u, 1u);
      addReference(annulus, 6u, getHanischAmount(lattice, nearestDistance));
    } else {
      addReference(getCensorSlot(borderDistance, lattice.maximumDistance / f32(RADIUS_COUNT)), 5u, 1u);
      if (!hasEvent && borderDistance >= lattice.maximumDistance) {
        addReference(0u, 7u, getHanischAmount(lattice, lattice.maximumDistance));
      }
    }
${f?`  }
`:``}  }
  workgroupBarrier();
  // One global 64-bit add per nonzero accumulator and workgroup: exact, so order-independent.
  for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += ${gt}u) {
    let amount = atomicLoad(&localReferences[word]);
    if (amount != 0u) {
      let globalWord = referenceAccumulatorsOffset + word * 2u;
      let previous = atomicAdd(&referenceAccumulators[globalWord], amount);
      if (previous > 0xffffffffu - amount) {
        atomicAdd(&referenceAccumulators[globalWord + 1u], 1u);
      }
    }
  }`}));let N=[{name:`accumulators`,view:j.accumulators,type:`u32`,access:`read`},{name:`referenceAccumulators`,view:M,type:`u32`,access:`read`},{name:`cellOffsets`,view:S.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:h,type:`f32`,access:`read`}];for(let[e,t]of[[`g`,l.g],[`f`,l.f],[`j`,l.j],[`radii`,l.radii]])t&&N.push({name:e,view:t,type:`f32`,access:`read_write`});return C.push(z(e,{id:`${t}-finish`,operation:Q,variant:`finish`,bindings:N,invocationCount:o,declarations:`${c}
${E}
const CELL_COUNT: u32 = ${a[0]*a[1]}u;
${f?``:`const REFERENCE_COUNT: f32 = ${y}.0;`}
const CHANNELS: u32 = ${$}u;

// Event counts stay below 2^24, so the low word of each 64-bit accumulator is exact.
fn readEvents(slot: u32, channel: u32) -> u32 {
  return accumulators[accumulatorsOffset + (slot * CHANNELS + channel) * 2u];
}

fn readReferences(slot: u32, channel: u32) -> u32 {
  return referenceAccumulators[referenceAccumulatorsOffset + (slot * CHANNELS + channel) * 2u];
}

// Full 64-bit sum as f32, for the fixed-point Hanisch weights.
fn readWeight(reference: bool, slot: u32, channel: u32) -> f32 {
  let word = (slot * CHANNELS + channel) * 2u;
  if (reference) {
    return f32(referenceAccumulators[referenceAccumulatorsOffset + word + 1u]) * 4294967296.0 +
      f32(referenceAccumulators[referenceAccumulatorsOffset + word]);
  }
  return f32(accumulators[accumulatorsOffset + word + 1u]) * 4294967296.0 +
    f32(accumulators[accumulatorsOffset + word]);
}

fn readObserved(reference: bool, slot: u32, channel: u32) -> f32 {
  return select(f32(readEvents(slot, channel)), f32(readReferences(slot, channel)), reference);
}

// Return CDF and survival together. Preserve exact empty/full probabilities even on devices
// whose floating-point division uses an approximate reciprocal. J needs the surviving mass,
// not a subtraction from an already rounded CDF.
fn getDistribution(observed: f32, remaining: f32, slot: u32) -> vec2f {
  let total = observed + remaining;
  if (!(total > 0.0)) { return vec2f(getQuietNaN(slot)); }
  if (observed == 0.0) { return vec2f(0.0, 1.0); }
  if (remaining == 0.0) { return vec2f(1.0, 0.0); }
  return vec2f(observed / total, remaining / total);
}

// Kaplan-Meier estimate of the distribution function at radius slot last, with hazards
// binned on the radius grid (events before censoring within a slot), as spatstat does.
fn getKaplanMeier(reference: bool, total: f32, last: u32) -> vec2f {
  var survival = 1.0;
  var atRisk = total;
  for (var slot = 0u; slot <= last; slot++) {
    let observed = readObserved(reference, slot, 4u);
    if (atRisk >= 1.0 && observed > 0.0) {
      survival = survival * getDistribution(observed, atRisk - observed, last).y;
    }
    atRisk = atRisk - observed - readObserved(reference, slot, 5u);
  }
  return vec2f(1.0 - survival, survival);
}

// Hanisch estimate: weighted observed fraction up to last over the total observed weight.
fn getHanisch(reference: bool, last: u32) -> vec2f {
  var observed = 0.0;
  var remaining = readWeight(reference, 0u, 7u);
  for (var slot = 0u; slot < RADIUS_COUNT; slot++) {
    let weight = readWeight(reference, slot, 6u);
    if (slot <= last) {
      observed = observed + weight;
    } else {
      remaining = remaining + weight;
    }
  }
  return getDistribution(observed, remaining, last);
}

fn getEventPrefix(channel: u32, last: u32) -> u32 {
  var sum = 0u;
  for (var slot = 0u; slot <= last; slot++) {
    sum += readEvents(slot, channel);
  }
  return sum;
}

fn getReferencePrefix(channel: u32, last: u32) -> u32 {
  var sum = 0u;
  for (var slot = 0u; slot <= last; slot++) {
    sum += readReferences(slot, channel);
  }
  return sum;
}`,body:`${f?`let REFERENCE_COUNT = f32(getReferencePrefix(3u, RADIUS_COUNT));
  `:``}let nan = getQuietNaN(index);
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mode = u32(parameters[parametersOffset + 5u] + 0.5);
  let border = mode == 1u;
  let maximumDistance = parameters[parametersOffset + 4u];
  let valid = isFiniteFloat(parameters[parametersOffset]) && isFiniteFloat(maximumDistance) && maximumDistance > 0.0;
  var gDistribution = vec2f(nan);
  var fDistribution = vec2f(nan);
  if (valid && n >= 2.0) {
    if (border) {
      let denominator = n - f32(getEventPrefix(3u, index));
      if (denominator >= 1.0) {
        let observed = f32(getEventPrefix(1u, index) - getEventPrefix(2u, index));
        gDistribution = getDistribution(observed, denominator - observed, index);
      }
    } else {
      let observed = f32(getEventPrefix(0u, index));
      gDistribution = getDistribution(observed, n - observed, index);
    }
    if (mode == 2u) {
      gDistribution = getKaplanMeier(false, n, index);
    } else if (mode == 3u) {
      gDistribution = getHanisch(false, index);
    }
  }
  if (valid && n >= 1.0${f?` && REFERENCE_COUNT > 0.0`:``}) {
    if (border) {
      let denominator = REFERENCE_COUNT - f32(getReferencePrefix(3u, index));
      if (denominator >= 1.0) {
        let observed = f32(getReferencePrefix(1u, index) - getReferencePrefix(2u, index));
        fDistribution = getDistribution(observed, denominator - observed, index);
      }
    } else {
      let observed = f32(getReferencePrefix(0u, index));
      fDistribution = getDistribution(observed, REFERENCE_COUNT - observed, index);
    }
    if (mode == 2u) {
      fDistribution = getKaplanMeier(true, REFERENCE_COUNT, index);
    } else if (mode == 3u) {
      fDistribution = getHanisch(true, index);
    }
  }
  ${l.g?`g[gOffset + index] = gDistribution.x;`:``}
  ${l.f?`f[fOffset + index] = fDistribution.x;`:``}
  ${l.j?`j[jOffset + index] = nan;
  if (isFiniteFloat(gDistribution.y) && isFiniteFloat(fDistribution.y) && fDistribution.y > 0.0) {
    j[jOffset + index] = gDistribution.y / fDistribution.y;
  }`:``}
  ${l.radii?`radii[radiiOffset + index] = maximumDistance * f32(index + 1u) / f32(RADIUS_COUNT);`:``}`})),C}},vt=[`lMinusR`,`g`,`f`,`j`],yt=class{radiusCount=0;runs=0;curves={lMinusR:[],g:[],f:[],j:[]};get count(){return this.runs}get isComplete(){return this.runs>=39}reset(e){this.radiusCount=e,this.runs=0;for(let e of vt)this.curves[e]=[]}add(e){for(let t of vt)this.curves[t].push(Float32Array.from(e[t]));this.runs++}getEnvelopes(){if(!this.isComplete)return null;let e={};for(let t of vt){let n=new Float32Array(this.radiusCount).fill(NaN),r=new Float32Array(this.radiusCount).fill(NaN);for(let e of this.curves[t])for(let t=0;t<this.radiusCount;t++){let i=e[t];Number.isFinite(i)&&(i>=n[t]||(n[t]=i),i<=r[t]||(r[t]=i))}e[t]={low:n,high:r}}return e}};function bt(e){if(!Number.isFinite(e))return``;if(e<1e3)return`${Math.round(e)} m`;let t=e/1e3;return`${Number.isInteger(t)?t:t.toFixed(1)} km`}function xt(e){let t=[];return e.previous&&e.previous.label!==e.label&&t.push({label:e.previous.label,x:e.previous.radii,y:e.previous.lMinusR,ghost:!0,dashed:!0,directLabel:!0}),e.corrections?t.push({label:`No correction`,x:e.radii,y:e.corrections.none,color:e.slot,dashed:!0,width:1,directLabel:!0},{label:`Border`,x:e.radii,y:e.corrections.border,color:e.slot,dashed:!0,width:1.75,directLabel:!0},{label:`Isotropic`,x:e.radii,y:e.corrections.isotropic,color:e.slot,width:2.5,directLabel:!0}):t.push({label:e.label,x:e.radii,y:e.lMinusR,color:e.slot,width:2.5,directLabel:!0}),{kind:`line`,height:156,title:e.envelope===null?`${e.simulationsDone} of 39 random patterns run`:void 0,xLabel:`radius r`,yLabel:`L(r) − r`,formatX:bt,formatY:e=>`${Ve(e,0)} m`,series:t,band:e.envelope?{x:e.radii,low:e.envelope.low,high:e.envelope.high,label:`39 random patterns (${e.nullName})`}:void 0,guides:[{y:0,label:`random (CSR)`}],link:{option:`radius`,label:e=>`r = ${bt(e)}`},description:`Besag L(r) minus r for ${e.label}. Zero is a random pattern; the grey band is the pointwise range of 39 random patterns (${e.nullName}), a 5 % envelope. Above the band the pattern is clustered at that radius.`}}var St=e=>Number.isFinite(e)?Math.min(Math.max(e,0),2):NaN;function Ct(e){let t=Array.from(e.radii),n=t.map(t=>1-Math.exp(-e.intensity*Math.PI*t*t)),r=(e,n)=>e?{x:t,low:n?Array.from(e.low,n):e.low,high:n?Array.from(e.high,n):e.high,label:`39 random patterns`}:void 0,i={kind:`line`,height:120,formatX:bt,table:!1};return{kind:`multiples`,charts:[{...i,yDomain:[0,1],series:[{label:`Observed`,x:t,y:Array.from(e.g),color:e.slot,width:2.25},{label:`CSR`,x:t,y:n,dashed:!0,ghost:!0}],band:r(e.envelopes?.g)},{...i,yDomain:[0,1],series:[{label:`Observed`,x:t,y:Array.from(e.f),color:e.slot,width:2.25},{label:`CSR`,x:t,y:n,dashed:!0,ghost:!0}],band:r(e.envelopes?.f)},{...i,yDomain:[0,2],series:[{label:`Observed`,x:t,y:Array.from(e.j,St),color:e.slot,width:2.25},{label:`CSR`,x:t,y:t.map(()=>1),dashed:!0,ghost:!0}],band:r(e.envelopes?.j,St)}],titles:[`G nearest neighbour`,`F empty space`,`J = (1 − G) / (1 − F)`],columns:3,shareDomains:!1,description:`G, F and J of the selection against complete spatial randomness (${e.nullName}). G rising early and F rising late mean clustering; J below 1 agrees.`}}var wt=e=>e.replace(/&/g,`&amp;`).replace(/</g,`&lt;`).replace(/>/g,`&gt;`);function Tt(e,t,n){let r=e=>16+Math.min(Math.max(e,0),2)/2*288,i=r(e),a=Math.max(2,r(1+(Number.isFinite(n)?n:0))-r(1)),o=Math.min(Math.max(i,56),264),s=[0,.5,1,1.5,2].map(e=>`<line class="diagram-muted" x1="${r(e)}" y1="34" x2="${r(e)}" y2="46"/><text class="diagram-muted" x="${r(e)}" y="62" text-anchor="middle">${e}</text>`).join(``),c=Number.isFinite(e)?`R = ${e.toFixed(2)}, z = ${Ve(t,0)}`:`R undefined`;return{kind:`diagram`,width:320,height:92,svg:`<rect class="diagram-fill" x="${r(1)-a}" y="30" width="${a*2}" height="20"/>
<line class="diagram-muted" x1="16" y1="40" x2="304" y2="40"/>${s}
${Number.isFinite(e)?`<line class="diagram-accent" x1="${i}" y1="26" x2="${i}" y2="54"/><circle class="diagram-accent" cx="${i}" cy="40" r="4.5"/>`:``}
<text class="diagram-accent" x="${o}" y="16" text-anchor="middle">${wt(c)}</text>
<text class="diagram-muted" x="${r(.5)}" y="82" text-anchor="middle">clustered</text>
<text class="diagram-muted" x="${r(1)}" y="82" text-anchor="middle">random</text>
<text class="diagram-muted" x="${r(1.5)}" y="82" text-anchor="middle">dispersed</text>`,description:`Clark-Evans gauge: R is ${Number.isFinite(e)?e.toFixed(2):`undefined`}. Below 1 is clustered, 1 is random, above 1 is dispersed; the shaded band is where a random pattern would fall.`}}function Et(e,t,n,r){let i=1;for(let t=0;t<e.length;t++)Number.isFinite(e[t])&&(i=Math.max(i,e[t]));let a=i,o=Array(16).fill(0);for(let t=0;t<e.length;t++){if(!Number.isFinite(e[t]))continue;let n=Math.min(15,Math.floor(e[t]/a*16));o[n]++}return{kind:`histogram`,height:120,values:o,xDomain:[0,a],breaks:r.breaks,classColors:r.colors.map((e,t)=>t===0?[128,136,148,150]:e),now:t,nowLabel:`random expects`,xLabel:`observed / area-weighted expected`,yLabel:`quadrats`,description:`Quadrat counts relative to area-weighted expectation (1). Zero-area quadrats are excluded. The reduced chi-square is ${Number.isFinite(n)?n.toFixed(1):`undefined`}.`}}var Dt=[.25,.5,1,2],Ot=.01,kt=[`Same coordinate (0 m)`,`< 0.25 × expected`,`0.25 to 0.5 × expected`,`0.5 to 1 × expected`,`1 to 2 × expected`,`≥ 2 × expected`];function At(e,t){let n=Dt.map(t=>t*e),r=Le(`YlGnBu`,6,{ground:t,reverse:!0}).slice(0,5),i=Ke(t,255);return W({breaks:[Ot,...n],colors:[i,...r],labels:kt,unit:`m`,method:`Multiples of the distance a random pattern gives, E = ${Be(e)}`,noData:{label:`Not in the analysis`,color:[128,128,128,60]}})}function jt(e,t){let n=[1e-20,.5,1,2,4],r=W({breaks:n,colors:Le(`YlGnBu`,6,{ground:t}),transparent:[0],labels:[`No records`,`< 0.5 × expected`,`0.5 to 1 × expected`,`1 to 2 × expected`,`2 to 4 × expected`,`≥ 4 × expected`],unit:`observed / expected`,method:`Expected count = included records × quadrat area / window area; areas measured on the GPU`,noData:{label:`Outside the window`,color:[0,0,0,0]}}),i=[,,,,,,].fill(0);for(let t=0;t<e.length;t++){if(!Number.isFinite(e[t]))continue;let r=0;for(;r<n.length&&e[t]>=n[r];)r++;i[r]++}return{table:r,counts:i}}function Mt(e,t){let{positions:n,nearest:r,spatialContext:i,parameters:a,output:o}=t,s=n.length,c=(t,n=s)=>P(e,t,`uint32`,n),l=c(`radius-included`);U(e,{id:`radius-inclusion`,invocationCount:s,bindings:[{name:`nearest`,view:r,type:`f32`,access:`read`},{name:`included`,view:l,type:`u32`,access:`read_write`}],body:`let distanceBits = bitcast<u32>(nearest[nearestOffset + index]) & 0x7fffffffu;
      included[includedOffset + index] = select(0u, 1u, distanceBits < 0x7f800000u);`});let u=c(`radius-counts`);e.add(new ge({id:`radius-neighbors`,mode:`radius`,positions:n,mask:l,parameters:a,spatialContext:{...i,frame:void 0},gridSize:[96,96],output:{neighborCounts:u,offsets:c(`radius-offsets`,s+1),neighbors:c(`radius-neighbors`,1),weights:P(e,`radius-weights`,`float32`,1),status:{overflow:c(`radius-overflow`,1)}}}));let d=c(`radius-maximum`,1);e.add(new se({id:`maximum-radius-count`,input:u,mask:l,output:d,operation:`max`}));let f=c(`densest-candidates`);U(e,{id:`densest-ties`,invocationCount:s,bindings:[{name:`counts`,view:u,type:`u32`,access:`read`},{name:`included`,view:l,type:`u32`,access:`read`},{name:`maximum`,view:d,type:`u32`,access:`read`},{name:`candidates`,view:f,type:`u32`,access:`read_write`}],body:`candidates[candidatesOffset + index] = select(0xffffffffu, index,
      included[includedOffset + index] != 0u && counts[countsOffset + index] == maximum[maximumOffset]);`});let p=c(`densest-row`,1);e.add(new se({id:`lowest-densest-row`,input:f,output:p,operation:`min`})),U(e,{id:`densest-summary`,invocationCount:1,bindings:[{name:`row`,view:p,type:`u32`,access:`read`},{name:`maximum`,view:d,type:`u32`,access:`read`},{name:`result`,view:o,type:`u32`,access:`read_write`}],body:`result[resultOffset] = row[rowOffset]; result[resultOffset + 1u] = maximum[maximumOffset];`})}function Nt(e,t,n,r){let i=t.length,a=(t,n=i)=>P(e,`duplicates-${t}`,`uint32`,n),o=a(`keys`),s=a(`rows`);U(e,{id:`duplicate-y-keys`,invocationCount:i,bindings:[{name:`base`,view:t,type:`f32`,access:`read`},{name:`mask`,view:n,type:`u32`,access:`read`},{name:`keys`,view:o,type:`u32`,access:`read_write`},{name:`rows`,view:s,type:`u32`,access:`read_write`}],body:`let coordinate = base[baseOffset + 2u * index + 1u];
      keys[keysOffset + index] = select(0xffffffffu, select(bitcast<u32>(coordinate), 0u, coordinate == 0.0), mask[maskOffset + index] != 0u);
      rows[rowsOffset + index] = index;`});let c=a(`sorted-y`),l=a(`rows-by-y`);e.add(new oe({id:`duplicate-sort-y`,keys:o,values:s,outputKeys:c,outputValues:l}));let u=a(`keys-x`);U(e,{id:`duplicate-x-keys`,invocationCount:i,bindings:[{name:`base`,view:t,type:`f32`,access:`read`},{name:`mask`,view:n,type:`u32`,access:`read`},{name:`rows`,view:l,type:`u32`,access:`read`},{name:`keys`,view:u,type:`u32`,access:`read_write`}],body:`let row = rows[rowsOffset + index]; let coordinate = base[baseOffset + 2u * row];
      keys[keysOffset + index] = select(0xffffffffu, select(bitcast<u32>(coordinate), 0u, coordinate == 0.0), mask[maskOffset + row] != 0u);`});let d=a(`sorted-x`),f=a(`sorted-rows`);e.add(new oe({id:`duplicate-sort-x`,keys:u,values:l,outputKeys:d,outputValues:f}));let p=a(`heads`);U(e,{id:`duplicate-pair-heads`,invocationCount:i,bindings:[{name:`base`,view:t,type:`f32`,access:`read`},{name:`mask`,view:n,type:`u32`,access:`read`},{name:`rows`,view:f,type:`u32`,access:`read`},{name:`flags`,view:p,type:`u32`,access:`read_write`}],body:`let row = rows[rowsOffset + index]; var head = index == 0u;
      if (index > 0u) {
        let previous = rows[rowsOffset + index - 1u];
        let selected = mask[maskOffset + row] != 0u;
        head = selected != (mask[maskOffset + previous] != 0u) || (selected &&
          (base[baseOffset + 2u * row] != base[baseOffset + 2u * previous] ||
           base[baseOffset + 2u * row + 1u] != base[baseOffset + 2u * previous + 1u]));
      }
      flags[flagsOffset + index] = select(0u, 1u, head);`});let m=a(`groups`);e.add(new ne({id:`duplicate-group-ids`,input:p,output:m,mode:`inclusive`}));let h=a(`run-count`,1),g=a(`lengths`);e.add(new K({id:`duplicate-runs`,input:m,values:a(`values`),lengths:g,count:h}));let _=a(`selected`,1);e.add(new se({id:`duplicate-selected-count`,input:n,output:_,operation:`sum`}));let v=a(`terms`);U(e,{id:`duplicate-shared-terms`,invocationCount:i,bindings:[{name:`lengths`,view:g,type:`u32`,access:`read`},{name:`runCount`,view:h,type:`u32`,access:`read`},{name:`selectedCount`,view:_,type:`u32`,access:`read`},{name:`terms`,view:v,type:`u32`,access:`read_write`}],body:`var amount = 0u;
      let validRuns = runCount[runCountOffset] - select(0u, 1u, selectedCount[selectedCountOffset] < ${i}u);
      if (index < validRuns) { let runLength = lengths[lengthsOffset + index]; amount = select(0u, runLength, runLength > 1u); }
      terms[termsOffset + index] = amount;`});let y=a(`shared`,1);e.add(new se({id:`duplicate-shared-count`,input:v,output:y,operation:`sum`})),U(e,{id:`duplicate-share`,invocationCount:1,bindings:[{name:`sharedCount`,view:y,type:`u32`,access:`read`},{name:`selectedCount`,view:_,type:`u32`,access:`read`},{name:`result`,view:r,type:`f32`,access:`read_write`}],body:`var nanBits = 0x7fc00000u;
      let total = selectedCount[selectedCountOffset];
      result[resultOffset] = select(bitcast<f32>(nanBits), f32(sharedCount[sharedCountOffset]) / f32(total), total > 0u);`})}async function Pt(e,t,n,r){let i=new D(e.device,{id:`${t.id}-jitter`}),a=(n,r)=>e.createBuffer(`${t.id}-jitter-${n}`,r),o=new Float32Array(256*2);for(let e=0;e<256;e++){let t=e*Math.PI*2/256;o[e*2]=Math.cos(t),o[e*2+1]=Math.sin(t)}let s=e.createParameterBuffer(`${t.id}-jitter-radius`,`float32`,1),c=P(i,`offsets`,`float32x2`,t.count);return i.add(new me({id:`jitter-offsets`,spatialContext:n,positions:H(i,`disc`,a(`disc`,o),`float32x2`),featureOffsets:H(i,`features`,a(`features`,Uint32Array.of(0,1)),`uint32`),polygonOffsets:H(i,`polygons`,a(`polygons`,Uint32Array.of(0,1)),`uint32`),ringOffsets:H(i,`rings`,a(`rings`,Uint32Array.of(0,256)),`uint32`),counts:H(i,`counts`,a(`counts`,Uint32Array.of(t.count)),`uint32`),parameters:H(i,`parameters`,a(`parameters`,pe({seed:r})),`uint32`),output:{positions:c,dots:{ids:P(i,`ids`,`uint32`,t.count),status:{count:P(i,`count`,`uint32`,1),overflow:P(i,`overflow`,`uint32`,1)}}}})),U(i,{id:`apply-jitter`,invocationCount:t.count*2,bindings:[{name:`base`,view:H(i,`base`,t.baseBuffer,`float32x2`),type:`f32`,access:`read`},{name:`offsets`,view:c,type:`f32`,access:`read`},{name:`radius`,view:s.importToGraph(i),type:`f32`,access:`read`},{name:`positions`,view:H(i,`positions`,t.positionsBuffer,`float32x2`),type:`f32`,access:`read_write`}],body:`let amount = radius[radiusOffset];
      var value = base[baseOffset + index];
      if (amount > 0.0) { value += amount * offsets[offsetsOffset + index]; }
      positions[positionsOffset + index] = value;`}),{compiled:e.track(await i.compileAsync()),radius:s}}function Ft(e){return Number.isFinite(e)?e<.001?`p < 0.001`:`p = ${e.toFixed(e<.1?3:2)}`:`p undefined`}var It=[96,96],Lt=350,Rt=4242,zt=[`none`,`border`,`isotropic`],Bt=[[10,1],[12,1.6],[14,2.6]],Vt={restaurant_cafe:`Restaurants and cafes`,bar_nightlife:`Bars and nightlife`,grocery:`Grocery and convenience`,health:`Health care`,school_education:`Schools and education`,park_recreation:`Parks and recreation`,transit:`Transit stops`,retail:`Retail`,finance_business:`Finance and business`,personal_services:`Personal services`,arts_culture:`Arts and culture`,worship_community:`Worship and community`,lodging:`Lodging`,other:`Other`};async function Ht(e){let{device:t}=e,n=e.datasets.get(`chicago-nature`),r=e.datasets.get(`chicago-places`),s=n.defaultOrigin,c=i({kind:`lambert-equal-area`,origin:s}),l=await Ze(e,s,c.project),u=new ee(t,`point-patterns`),d=(e,t,n)=>{let r=o(c,n.column(`position`)),i=r.length/2;return{id:e,label:t,count:i,baseBuffer:u.createBuffer(`${e}-base`,r),jitter:null,jitterDirty:!1,lngLat:n.column(`position`),category:n.column(`category`),categoryNames:n.categories(`category`),positionsBuffer:u.createBuffer(`${e}-positions`,r.slice()),maskBuffer:u.createBuffer(`${e}-mask`,new Uint32Array(i).fill(1)),randomBuffer:u.createBuffer(`${e}-random`,i*8),selected:new Uint32Array(i).fill(1),jitterApplied:0,index:null,indexRows:null}},f={observations:d(`observations`,`Nature observations`,n),places:d(`places`,`Places (Overture)`,r)};for(let e of Object.values(f))e.jitter=await Pt(u,e,c.spatialContext,e.id===`observations`?11:23);let p=[...l.cityBoundsMeters],m={mesh:We(u,(()=>{let[e,t,n,r]=p,i=[c.unproject(e,t),c.unproject(n,t),c.unproject(n,r),c.unproject(e,r)];return Ae([{type:`Feature`,properties:null,geometry:{type:`Polygon`,coordinates:[[...i,i[0]],...l.cityPolygons.map(e=>e[0])]}},...l.cityPolygons.flatMap(e=>e.slice(1).map(e=>({type:`Feature`,properties:null,geometry:{type:`Polygon`,coordinates:[e]}})))],(e,t)=>c.project(e,t))})(),`outside-window`)},h={city:m,rectangle:m,view:m},g=u.createBuffer(`hatch-values`,new Uint32Array(1+l.cityPolygons.reduce((e,t)=>e+t.length-1,0))),_=u.createParameterBuffer(`ripley`,`float32`,8),v=u.createParameterBuffer(`distance`,`float32`,8),y=u.createParameterBuffer(`indices`,`float32`,8),b=u.createParameterBuffer(`neighbors`,`float32`,12),x=u.createParameterBuffer(`window`,`float32`,4),S=[],C=u.createParameterBuffer(`clip-window`,`float32`,4),w=u.createParameterBuffer(`random`,`uint32`,8),T=u.createBuffer(`random-counts`,Uint32Array.of(0)),E=u.createBuffer(`rectangle-positions`,32),O=l.cityWindowColumns,k={positions:u.createBuffer(`city-positions`,O.positions),featureOffsets:u.createBuffer(`city-features`,O.featureOffsets),polygonOffsets:u.createBuffer(`city-polygons`,O.polygonOffsets),ringOffsets:u.createBuffer(`city-rings`,O.ringOffsets)},A=u.createBuffer(`rectangle-features`,Uint32Array.of(0,1)),j=u.createBuffer(`rectangle-polygons`,Uint32Array.of(0,1)),M=u.createBuffer(`rectangle-rings`,Uint32Array.of(0,4)),N=!1,te=!1,F=performance.now(),I=[...p],L=null,R=null,ne=null,re=new Map,ie=new Map,ae=!0,oe=!0,se=!0,ce=0,z=0,le=[],B=null,ue=0,V=0,de=-1,fe=-1,ge=null,_e=NaN,ve=NaN,ye=NaN,W=0,Le=null,K=null,q={},Qe=null,et=null,J=new yt,nt=!1,rt=0,it=0,Y=null,ot=NaN,X=null,Z=``,ct=``,lt=e=>f[e.subject],ut=e=>e.subject===`observations`?e.groupCategory:e.placeCategory,ft=e=>{let t=ut(e);return t===`all`?e.subject===`observations`?`All observations`:`All places`:e.subject===`observations`?Ge(t):Vt[t]??t},Q=t=>t===`city`?e.options.window===`view`?`random on city land within the view`:`random on city land`:e.options.window===`view`?`random in the viewport rectangle`:`random in the rectangle`,mt=e=>`${e.subject}|${e.radiusCount}|${e.quadratGrid}|${e.referenceGrid}|${e.window}`,ht=e=>`${e.subject}|${e.radiusCount}|${e.referenceGrid}|${e.window}|${e.nullModel}`,$=e=>[Number(e),Number(e)],gt=()=>_e,vt=(e,t)=>c.unproject(e,t);function St(e){return{cityGeometry:{positions:H(e,`city-positions`,k.positions,`float32x2`),featureOffsets:H(e,`city-features`,k.featureOffsets,`uint32`),polygonOffsets:H(e,`city-polygons`,k.polygonOffsets,`uint32`),ringOffsets:H(e,`city-rings`,k.ringOffsets,`uint32`)},rectangleGeometry:{positions:H(e,`rectangle-positions`,E,`float32x2`),featureOffsets:H(e,`rectangle-features`,A,`uint32`),polygonOffsets:H(e,`rectangle-polygons`,j,`uint32`),ringOffsets:H(e,`rectangle-rings`,M,`uint32`)}}}function wt(e,t){let{cityGeometry:n,rectangleGeometry:r}=St(e),i=n;if(t.window===`view`){let t=P(e,`clipped-positions`,`float32x2`,n.positions.length*16),r=P(e,`clipped-rings`,`uint32`,n.ringOffsets.length);e.add(new be({id:`clip-city`,geometryType:`polygons`,...n,spatialContext:c.spatialContext,parameters:C.importToGraph(e),output:{positions:t,lineOffsets:r,status:{count:P(e,`clip-count`,`uint32`,1),overflow:P(e,`clip-overflow`,`uint32`,1)}}})),i={...n,positions:t,ringOffsets:r}}let a=t.window===`rectangle`?r:i,o=u.createBuffer(`${e.id}-window-summary`,32),s=H(e,`window-summary`,o,`float32`),l=u.track(new $e({id:`study-window`,...a,spatialContext:c.spatialContext,output:{summary:s}}));e.add(l),t.window!==`city`&&S.push(l);let d=o;if(t.window===`rectangle`){d=u.createBuffer(`${e.id}-city-summary`,32);let t=u.track(new $e({id:`city-land-window`,...n,spatialContext:c.spatialContext,output:{summary:H(e,`city-summary`,d,`float32`)}}));e.add(t)}return{preparation:l,summary:s,summaryBuffer:o,landSummaryBuffer:t.window===`rectangle`?d:o,studyWindow:t.window===`rectangle`?`rectangle`:l,randomGeometry:t.nullModel===`city`?i:r}}async function Dt(e){let n=mt(e),r=lt(e),i=Number(e.radiusCount),a=$(e.quadratGrid),o=$(e.referenceGrid),s=n.replace(/\|/g,`-`),l=a[0]*a[1],d=(e,t)=>u.createBuffer(`${s}-${e}`,t*4),f={k:d(`k`,i),l:d(`l`,i),lMinusR:d(`l-minus-r`,i),radii:d(`radii`,i),g:d(`g`,i),f:d(`f`,i),j:d(`j`,i),nearest:d(`nearest`,r.count),clarkEvans:d(`clark-evans`,7),quadratCounts:d(`quadrat-counts`,l),quadratAreas:d(`quadrat-areas`,l),quadratRatios:d(`quadrat-ratios`,l),quadratStatistics:d(`quadrat-statistics`,7),densest:d(`densest`,2),duplicateShare:d(`duplicate-share`,1)},p=new D(t,{id:`point-patterns-${s}`}),m=wt(p,e),h=H(p,`positions`,r.positionsBuffer,`float32x2`,r.count),g=H(p,`mask`,r.maskBuffer,`uint32`,r.count),x=(e,t,n)=>H(p,e,f[e],t,n),S=x(`quadratCounts`,`uint32`,l),C=x(`quadratAreas`,`float32`,l),w=x(`quadratRatios`,`float32`,l),T=x(`clarkEvans`,`float32`,7),E=x(`nearest`,`float32`,r.count);p.add(new at({spatialContext:c.spatialContext,studyWindow:m.studyWindow,id:`ripley`,positions:h,mask:g,parameters:_.importToGraph(p),gridSize:It,radiusCount:i,output:{k:x(`k`,`float32`,i),l:x(`l`,`float32`,i),lMinusR:x(`lMinusR`,`float32`,i),radii:x(`radii`,`float32`,i)}})),p.add(new _t({spatialContext:c.spatialContext,studyWindow:m.studyWindow,id:`ripley-distance`,positions:h,mask:g,parameters:v.importToGraph(p),gridSize:It,referenceGrid:o,radiusCount:i,output:{g:x(`g`,`float32`,i),f:x(`f`,`float32`,i),j:x(`j`,`float32`,i)}})),p.add(new dt({spatialContext:c.spatialContext,studyWindow:m.studyWindow,id:`indices`,positions:h,mask:g,parameters:y.importToGraph(p),gridSize:It,quadratGrid:a,output:{nearestNeighborDistances:E,clarkEvans:T,quadratCounts:S,quadratAreas:C,quadratStatistics:x(`quadratStatistics`,`float32`,7)}})),Mt(p,{positions:h,nearest:E,spatialContext:c.spatialContext,parameters:b.importToGraph(p),output:x(`densest`,`uint32`,2)}),Nt(p,H(p,`base`,r.baseBuffer,`float32x2`),g,x(`duplicateShare`,`float32`,1)),U(p,{id:`quadrat-relative-counts`,invocationCount:l,bindings:[{name:`counts`,view:S,type:`u32`,access:`read`},{name:`areas`,view:C,type:`f32`,access:`read`},{name:`clark`,view:T,type:`f32`,access:`read`},{name:`summary`,view:m.summary,type:`f32`,access:`read`},{name:`ratios`,view:w,type:`f32`,access:`read_write`}],body:`var nanBits = 0x7fc00000u;
        let expected = clark[clarkOffset] * areas[areasOffset + index] / summary[summaryOffset + 4u];
        ratios[ratiosOffset + index] = select(bitcast<f32>(nanBits), f32(counts[countsOffset + index]) / expected, expected > 0.0);`});let O=m.preparation.getViews(p).edges,k=u.createBuffer(`${s}-outline`,O.length*4),A=u.createBuffer(`${s}-outline-weights`,O.length);U(p,{id:`window-outline`,invocationCount:O.length/4,bindings:[{name:`edges`,view:O,type:`f32`,access:`read`},{name:`weights`,view:H(p,`outline-weights`,A,`float32`),type:`f32`,access:`read_write`},{name:`outline`,view:H(p,`outline`,k,`float32`),type:`f32`,access:`read_write`}],body:`var valid = true;
        for (var component = 0u; component < 4u; component++) {
          valid = valid && (bitcast<u32>(edges[edgesOffset + 4u * index + component]) & 0x7fffffffu) < 0x7f800000u;
        }
        weights[weightsOffset + index] = select(0.0, 1.0, valid);
        for (var component = 0u; component < 4u; component++) {
          outline[outlineOffset + 4u * index + component] = select(0.0, edges[edgesOffset + 4u * index + component], valid);
        }`});let j={key:n,subject:r,radiusCount:i,quadratGrid:a,compiled:u.track(await p.compileAsync()),nearestDistances:f.nearest,quadratCounts:f.quadratCounts,quadratRatios:f.quadratRatios,windowOutline:k,windowOutlineWeights:A,windowEdgeCapacity:O.length/4,reader:new ke(u,`summary-${s}`,[{buffer:f.lMinusR,size:i*4},{buffer:f.g,size:i*4},{buffer:f.f,size:i*4},{buffer:f.j,size:i*4},{buffer:f.radii,size:i*4},{buffer:f.clarkEvans,size:28},{buffer:f.quadratStatistics,size:28},{buffer:f.quadratCounts,size:l*4},{buffer:f.quadratAreas,size:l*4},{buffer:f.quadratRatios,size:l*4},{buffer:m.summaryBuffer,size:32},{buffer:m.landSummaryBuffer,size:32},{buffer:f.densest,size:8},{buffer:f.duplicateShare,size:4}],e=>nn(j,e))};return j}async function Ot(e){let n=ht(e),r=lt(e),i=Number(e.radiusCount),a=$(e.referenceGrid),o=`${n.replace(/\|/g,`-`)}-random`,s=(e,t)=>u.createBuffer(`${o}-${e}`,t*4),l=u.createBuffer(`${o}-positions`,r.count*8),d={k:s(`k`,i),l:s(`l`,i),lMinusR:s(`l-minus-r`,i),radii:s(`radii`,i),g:s(`g`,i),f:s(`f`,i),j:s(`j`,i)},f=new D(t,{id:`point-patterns-${o}`}),p=wt(f,e),m=H(f,`positions`,l,`float32x2`,r.count),h=u.createBuffer(`${o}-failed`,4),g=w.importToGraph(f),y=H(f,`random-counts`,T,`uint32`),b=H(f,`random-failed`,h,`uint32`);f.add(new me({id:`random-points`,...p.randomGeometry,spatialContext:c.spatialContext,counts:y,parameters:g,output:{positions:m,dots:{ids:P(f,`random-ids`,`uint32`,r.count),status:{count:P(f,`random-count`,`uint32`,1),overflow:P(f,`random-overflow`,`uint32`,1)}},status:{failedCount:b}}})),U(f,{id:`retain-first-random-pattern`,invocationCount:r.count*2,bindings:[{name:`positions`,view:m,type:`f32`,access:`read`},{name:`parameters`,view:g,type:`u32`,access:`read`},{name:`displayPositions`,view:H(f,`random-display`,r.randomBuffer,`float32x2`),type:`f32`,access:`read_write`}],body:`if (parameters[parametersOffset] == ${Rt}u) {
        displayPositions[displayPositionsOffset + index] = positions[positionsOffset + index];
      }`});let x=e=>H(f,e,d[e],`float32`,i);f.add(new at({spatialContext:c.spatialContext,studyWindow:p.studyWindow,id:`ripley`,positions:m,parameters:_.importToGraph(f),gridSize:It,radiusCount:i,output:{k:x(`k`),l:x(`l`),lMinusR:x(`lMinusR`),radii:x(`radii`)}})),f.add(new _t({spatialContext:c.spatialContext,studyWindow:p.studyWindow,id:`ripley-distance`,positions:m,parameters:v.importToGraph(f),gridSize:It,referenceGrid:a,radiusCount:i,output:{g:x(`g`),f:x(`f`),j:x(`j`)}}));let S={key:n,subject:r,radiusCount:i,compiled:u.track(await f.compileAsync()),positionsBuffer:l,reader:new ke(u,`summary-${o}`,[{buffer:d.lMinusR,size:i*4},{buffer:d.g,size:i*4},{buffer:d.f,size:i*4},{buffer:d.j,size:i*4},{buffer:h,size:4},{buffer:p.summaryBuffer,size:32}],e=>rn(S,e))};return S}let kt=u.track(new Oe);function Ht(t){let n={...e.options};return Wt(),Gt(),kt.request({build:async()=>{let e=mt(n),t=re.get(e);t||(t=await Dt(n),re.set(e,t));let r=ht(n),i=ie.get(r);return i||(i=await Ot(n),ie.set(r,i)),{set:t,simulation:i}},commit:n=>{R=n.set,ne=n.simulation,t?.(),e.requestLayers()}})}function Ut(){ln(lt(e.options),e.options.jitter),cn(),K=null,X=null;for(let e of zt)delete q[e];Wt(),Gt(),ct=``}function Wt(){ce++,V++,de=-1,ge=null,le=[],oe=!0,ae=!0,F=performance.now()}function Gt(){z++,nt=!1,it=0,se=!0,ae=!0,J.reset(ne?.radiusCount??Number(e.options.radiusCount)),F=performance.now(),e.setReadout(`simulations`,`0 of 39 random patterns`)}function Kt(){se=!1,rt=0,nt=W>=3,it=0,Qt(),$t(),e.options.showRandom&&e.requestLayers()}function qt(){ae=!1,Jt(),Xt(),e.setCost({records:W,passes:R?R.compiled.stats.nodeOrder.length:void 0})}function Jt(){let t=e.options,n=gt();e.setReadout(`windowArea`,He(n)),e.setReadout(`windowLandShare`,Re(ve)),e.setReadout(`outsideShare`,Re(1-ve)),e.setReadout(`nullName`,Q(t.nullModel)),e.setReadout(`intensity`,`${ze(W/(n/1e6),`km²`)} of window`);let[r,i]=vt(I[0],I[1]),[a,o]=vt(I[2],I[3]),s=t.window===`rectangle`?[{kind:`frame`,id:`window-frame`,bounds:[r,i,a,o],text:`Study window, ${He(n)}`}]:[];e.setAnnotations(`window`,s.length?s:null);let c=je(Ne,`lake-michigan`);e.setAnnotations(`outside`,t.emphasizeEdge&&c?[{kind:`note`,id:`outside-note`,coordinate:c.lngLat,title:`${Re(1-ve)} outside city land in the bounding box`,text:`Hatching marks areas outside the city boundary`}]:null),Yt()}function Yt(){let t=e.options,n=lt(t),r=n.id===`observations`?`${G(n.count)} ${Xe}`:`${G(n.count)} Overture places, Chicago`,i=t.showRadius?t.radius:0,a=`${r}|${i}`;a!==ct&&(ct=a,e.setFurniture({title:{sample:r},scaleBar:{units:`metric`,ticks:i?[i]:void 0}}))}function Xt(){let t=e.options;if(Yt(),!t.showRadius||!Le){e.setAnnotations(`radius`,null),e.setReadout(`withinRadius`,null);return}let n=vt(Le.x,Le.y),r=t.subject===`observations`?`sightings`:`places`;e.setReadout(`withinRadius`,`${G(Le.neighbours)} ${r} within ${Be(t.radius)}`),e.setAnnotations(`radius`,[{kind:`ring`,id:`radius-ring`,coordinate:n,radiusMeters:t.radius,text:`r = ${Be(t.radius)}`,dashed:!0},{kind:`note`,id:`radius-note`,coordinate:n,title:`${G(Le.neighbours)} ${r} within r`,text:Me(Ne,n,{maxDistanceMeters:8e3})??void 0,tone:`accent`}])}function Zt(){return e.options.subject===`observations`?4:0}function Qt(){if(!K)return;let t=e.options,n=zt.every(e=>q[e])&&t.compareCorrections?{none:q.none,border:q.border,isotropic:q.isotropic}:null;e.setChart(`lCurve`,xt({label:ft(t),radii:K.radii,lMinusR:K.lMinusR,slot:Zt(),previous:et,envelope:J.getEnvelopes()?.lMinusR??null,simulationsDone:J.count,nullName:Q(t.nullModel),corrections:n}))}function $t(){if(!K)return;let t=e.options,n=J.getEnvelopes();e.setChart(`gfjCurves`,Ct({radii:K.radii,g:K.g,f:K.f,j:K.j,intensity:K.clark[0]/gt(),slot:Zt(),envelopes:n?{g:n.g,f:n.f,j:n.j}:null,nullName:Q(t.nullModel)}))}function en(){e.setReadout(`simulations`,`${J.count} of 39 random patterns`),J.isComplete&&e.setReadout(`graphs`,`2 analysis graphs and 2 jitter graphs compiled once; ${G(J.count)} random runs`),Qt(),J.isComplete&&$t()}function tn(e,t,n){for(let r=0;r<e.length;r++)if(e[r]>=n){if(r===0)return t[0];let i=e[r]-e[r-1],a=i>0?(n-e[r-1])/i:1;return t[r-1]+a*(t[r]-t[r-1])}return NaN}function nn(t,n){let r=B;if(B=null,N||!r||r.kind!==`observed`||r.generation!==ce||t!==R)return;let i=new Float32Array(n),a=new Uint32Array(n),o=t.radiusCount,s=i.slice(0,o);if(q[r.correction]=s,!r.primary){Qt();return}let l=i.slice(o,o*2),u=i.slice(o*2,o*3),d=i.slice(o*3,o*4),f=i.slice(o*4,o*5),p=o*5,m=p+7,h=m+7,g=i.slice(p,m),_=i.slice(m,h),v=t.quadratGrid[0]*t.quadratGrid[1],y=a.slice(h,h+v),b=h+v,S=i.slice(b,b+v),C=i.slice(b+v,b+v*2),w=b+v*2,T=i.subarray(w,w+8),E=i[w+8+4];_e=T[4],ve=E/((I[2]-I[0])*(I[3]-I[1]));let D=w+16,O=a[D],k=O<t.subject.count?c.project(t.subject.lngLat[O*2],t.subject.lngLat[O*2+1]):null;Le=k?{row:O,neighbours:a[D+1],x:k[0],y:k[1]}:null;let A=i[D+2];if(e.setReadout(`duplicateShare`,Number.isFinite(A)?Re(A,1):null),Xt(),de=V,W=g[0],x.write(T.subarray(0,4)),K={radii:f,lMinusR:s,g:l,f:u,j:d,clark:g,quadrat:_,counts:y,areas:S,ratios:C},ye!==an()){oe=!0,Jt();return}se&&Kt(),e.setCost({records:W,passes:t.compiled.stats.nodeOrder.length});let j=e.options;e.setReadout(`included`,G(g[0]));let M=-1,ee=-1/0;for(let e=0;e<o;e++)Number.isFinite(s[e])&&s[e]>ee&&(ee=s[e],M=e);e.setReadout(`lPeak`,M>=0?`${Ve(ee,0)} m at r = ${Be(f[M])}`:`undefined`),e.setReadout(`peakR`,M>=0?Be(f[M]):null);let te=tn(f,s,1e3);e.setReadout(`lAt1km`,Number.isFinite(te)?`${Ve(te,0)} m`:null),Qe={label:ft(j),radii:Array.from(f),lMinusR:Array.from(s)};let P=g[3];e.setReadout(`clarkEvansR`,Number.isFinite(P)?P.toFixed(2):null),e.setReadout(`clarkEvans`,Number.isFinite(P)?`R = ${P.toFixed(3)} (${P<1?`clustered`:`dispersed`}), z = ${Ve(g[5],1)}, ${Ft(g[6])}`:`undefined`),e.setReadout(`expectedNN`,Be(g[2])),e.setReadout(`observedNN`,Be(g[1])),e.setChart(`clarkEvansGauge`,Tt(P,g[5],1.96*g[4]/g[2]));let F=e.ground();Number.isFinite(g[2])&&(!Y||Math.abs(g[2]-ot)>ot*.05)&&(ot=g[2],Y=At(ot,F),e.setLegendData(`nearestTable`,Y),e.setLegendData(`nearestExpected`,ot),e.requestLayers()),e.setReadout(`vmr`,Number.isFinite(_[3])?_[3].toFixed(1):null),e.setReadout(`chiSquare`,Number.isFinite(_[3])?`chi-square ${G(_[4])} on ${G(_[5])} degrees of freedom, ${Ft(_[6])}`:null);let L=`${t.key}|${ut(j)}|${I.join()}|${j.jitter}`;(!X||L!==Z)&&(Z=L,X=jt(C,F),e.setLegendData(`quadrat`,X),e.requestLayers()),e.setChart(`quadratHistogram`,Et(C,1,_[3],X.table)),Jt(),Qt(),$t()}function rn(t,n){let r=B;if(B=null,N||!r||r.kind!==`simulation`||r.generation!==z||t!==ne)return;let i=new Float32Array(n),a=t.radiusCount,o=new Uint32Array(n)[a*4];if(o>0){nt=!1,it=0,e.setReadout(`simulations`,`GPU sampler could not place ${G(o)} points; envelope stopped`),e.requestLayers();return}r.index===0&&(it=W,e.requestLayers()),J.add({lMinusR:i.subarray(0,a),g:i.subarray(a,a*2),f:i.subarray(a*2,a*3),j:i.subarray(a*3,a*4)}),J.isComplete&&(nt=!1),en()}function an(){let t=e.options.maximumDistance/Number(e.options.radiusCount);return W>0&&_e>0?Math.max(2*Math.sqrt(_e/W),t):t}function on(t){let n=e.options,r=n.maximumDistance,i=n.window===`rectangle`?I:`study-window`;b.write(he({bounds:I,radius:n.radius})),ye=an(),_.write(tt({bounds:i,maximumDistance:r,edgeCorrection:t})),v.write(pt({bounds:i,maximumDistance:r,edgeCorrection:n.distanceCorrection})),y.write(st({bounds:i,clarkEvansCorrection:`guard`,guardDistance:ye,maximumDistance:Math.min(r,400)}))}function sn(){x.write(Float32Array.from(I));let[e,t,n,r]=I;C.write(Float32Array.from(I)),E.write(Float32Array.of(e,t,n,t,n,r,e,r));for(let e of S)e.invalidate()}function cn(){let t=e.options,n=lt(t),r=ut(t),i=r===`all`?-1:n.categoryNames.indexOf(r),a=new Uint32Array(n.count);for(let e=0;e<n.count;e++)a[e]=i<0||n.category[e]===i?1:0;n.selected=a,n.index=null,n.indexRows=null,n.maskBuffer.write(a)}function ln(e,t){e.jitterApplied!==t&&(e.jitterApplied=t,e.jitter?.radius.write(Float32Array.of(t)),e.jitterDirty=!0)}function un(){Qe&&(et=Qe)}function dn(e){if(e.index&&e.indexRows)return e.index;let t=[];for(let n=0;n<e.count;n++)e.selected[n]&&t.push(n);let n=new Float32Array(t.length*2);return t.forEach((t,r)=>{n[r*2]=e.lngLat[t*2],n[r*2+1]=e.lngLat[t*2+1]}),e.indexRows=Uint32Array.from(t),e.index=Ue(n),e.index}cn(),ln(f.observations,e.options.subject===`observations`?e.options.jitter:0),ln(f.places,e.options.subject===`places`?e.options.jitter:0);let fn=e.getViewport();e.options.window===`view`&&fn&&(I=a(fn,c)),sn(),await Ht(),J.reset(Number(e.options.radiusCount)),qt(),F=performance.now();async function pn(){let n=R;if(!(!n||te||N)){te=!0,e.setReadout(`timing`,`measuring…`),on(e.options.ripleyCorrection);try{let r=await we(t,n.compiled,{parameters:void 0,completionBuffer:n.quadratCounts,signal:e.signal});N||e.setReadout(`timing`,`${Ee(r).split(` · `)[0]} for ${G(n.subject.count)} points, r up to ${bt(e.options.maximumDistance)}`)}catch{}finally{te=!1}}}e.setReadout(`isotropicCap`,`Isotropic weights are capped at 100. Polygon Hanisch and quadrat areas use lattices. Positions are Lambert equal-area float32 metres; city-scale distance distortion is below 0.01%.`);function mn(){let t=e.options,n=t.ripleyCorrection,r=t.compareCorrections?[...zt.filter(e=>e!==n),n]:[n];le=r.map((e,t)=>({kind:`observed`,correction:e,primary:t===r.length-1,generation:ce})),oe=!1}function hn(e,t,n){on(e.correction),V++,de=-1,ge=null,t.compiled.encode(n,{parameters:void 0}),t.reader.request(n)}function gn(t,n,r){T.write(Uint32Array.of(W)),w.write(pe({seed:Rt+t.index})),on(e.options.ripleyCorrection),n.compiled.encode(r,{parameters:void 0}),n.reader.request(r)}function _n(){let e=Math.max(W,1e3);return Math.round(Math.min(70,Math.max(14,40*Math.sqrt(11e3/e))))}return{getCompiledGraphs:()=>[R?.compiled,ne?.compiled,...Object.values(f).map(e=>e.jitter?.compiled)].filter(Boolean),setOption(t,n,r){switch(t){case`subject`:case`groupCategory`:case`placeCategory`:un(),Ht(Ut);break;case`radiusCount`:case`quadratGrid`:case`referenceGrid`:Ht(Ut);break;case`jitter`:ln(lt(r),r.jitter),Wt(),Gt();break;case`window`:I=[...p],L=null,sn(),Ht(Ut);break;case`maximumDistance`:case`ripleyCorrection`:case`distanceCorrection`:for(let e of zt)delete q[e];Wt(),Gt();break;case`nullModel`:Ht(Ut);break;case`compareCorrections`:for(let e of zt)delete q[e];Wt();break;case`radius`:Wt(),Xt();break;case`showRadius`:Xt();break;case`emphasizeEdge`:Jt();break;default:break}e.requestLayers()},onAction(e){e===`measure`&&pn()},onThemeChange(){e.requestLayers()},onGroundChange(t){Number.isFinite(ot)&&(Y=At(ot,t),e.setLegendData(`nearestTable`,Y)),K&&(X=jt(K.ratios,t),e.setLegendData(`quadrat`,X),Qt()),e.setLegendData(`ground`,t),e.requestLayers()},getTooltip(t){let n=e.options,r=lt(n);if(!t.coordinate||!K)return null;let i=dn(r).nearest(t.coordinate,30),a=r.indexRows;if(!i||!a)return null;let o=a[i.index],s=[r.lngLat[o*2],r.lngLat[o*2+1]];if(R&&de===V&&fe!==V){let t=V,n=R;fe=t,n.nearestDistances.readAsync().then(r=>{!N&&R===n&&t===V&&(ge=new Float32Array(r.slice().buffer),e.requestLayers())}).catch(()=>{})}let c=ge?.[o]??NaN,l=K.clark[2],u=Y?Pe(Y,c):-1,d=[{label:`Nearest neighbour`,value:ge?Number.isFinite(c)?Be(c):`none`:`…`,swatch:Y&&u>=0?Y.colors[u]:void 0,emphasis:!0}];return Y&&u>=0&&d.push({label:`Class`,value:Ie(Y,u)}),Number.isFinite(c)&&l>0&&d.push({label:`Compared with random`,value:`${(c/l).toFixed(2)} ×`,unit:`expected`}),{title:ft(n),subtitle:Me(Ne,s,{maxDistanceMeters:8e3})??void 0,rows:d,highlight:{kind:`point`,coordinate:s}}},encode(t,n){for(let e of Object.values(f))e.jitterDirty&&e.jitter&&(e.jitter.compiled.encode(t,{parameters:void 0}),e.jitterDirty=!1);let r=R,i=ne;if(!r||!i)return;if(e.options.window===`view`){let e=a(n.viewport,c),t=e.map(e=>Math.round(e/5)).join(`,`);t!==L&&(L=t,I=[...e],sn(),Wt(),Gt())}r.reader.flush(t),i.reader.flush(t);let o=performance.now();if(kt.pending||te)return;if(B){o-ue>4e3&&!r.reader.isPending&&!i.reader.isPending&&(B=null);return}if(o-F<=Lt)return;ae&&qt(),oe&&le.length===0&&mn();let s=le.shift()??(nt?{kind:`simulation`,index:rt++,generation:z}:void 0);s&&(B=s,ue=o,s.kind===`observed`?hn(s,r,t):gn(s,i,t))},getLayers(){let t=R;if(!t)return[];let n=e.options,r=t.subject,i=e.ground(),a=i===`dark`,o=[s[0],s[1],0],l=[];n.showQuadrats&&X&&l.push(new Te({id:`pattern-quadrats-${t.key}`,coordinateOrigin:o,analysisFrame:c.frame,gridSize:t.quadratGrid,bounds:x.buffer,values:t.quadratRatios,valueFormat:`float32`,noDataColor:[0,0,0,0],colormap:`uniform`,...Fe(X.table),outlineClasses:{color:Ke(i,90),widthPixels:.7},color:[255,255,255,a?214:224]}));let u=h[n.window];l.push(new Ce({id:`pattern-outside-${n.window}`,coordinateOrigin:o,analysisFrame:c.frame,triangles:u.mesh.triangles,features:u.mesh.triangleFeatures,vertexCount:u.mesh.vertexCount,values:g,valueFormat:`uint32`,colormap:`uniform`,classBreaks:[1],classColors:[[0,0,0,0],[0,0,0,0]],hatchClasses:[0],hatchColor:Ke(i,n.emphasizeEdge?102:56),hatchSpacingPixels:5,hatchWidthPixels:1})),n.showOthers&&n.colorPoints!==`nearest`&&l.push(new xe({id:`pattern-others-${r.id}`,coordinateOrigin:o,analysisFrame:c.frame,positions:r.positionsBuffer,instanceCount:r.count,radiusPixels:.8,color:Je(i)}));let d=a?De:Bt;if(n.colorPoints===`nearest`&&Y){let e=Y.colors.map((e,t)=>t===0?[0,0,0,0]:[e[0],e[1],e[2],e[3]??255]);l.push(new xe({id:`pattern-nearest-${r.id}`,coordinateOrigin:o,analysisFrame:c.frame,positions:r.positionsBuffer,instanceCount:r.count,values:t.nearestDistances,valueFormat:`float32`,colormap:`uniform`,classBreaks:[...Y.breaks],classColors:e,noDataColor:[0,0,0,0],radiusPixels:d.map(([e,t])=>[e,t+.6])}),new xe({id:`pattern-duplicates-${r.id}`,coordinateOrigin:o,analysisFrame:c.frame,positions:r.positionsBuffer,instanceCount:r.count,values:t.nearestDistances,valueFormat:`float32`,colormap:`uniform`,classBreaks:[...Y.breaks.slice(0,1)],classColors:[Y.colors[0],[0,0,0,0]],noDataColor:[0,0,0,0],shape:`ring`,outlineWidthPixels:1.2,radiusPixels:d.map(([e,t])=>[e,t+3])}))}else if(n.showRandom&&it>0){let e=a?_n():200;l.push(new xe({id:`pattern-random-${r.id}`,coordinateOrigin:o,analysisFrame:c.frame,positions:r.randomBuffer,instanceCount:it,radiusPixels:d,blending:a?`additive`:`normal`,color:r.id===`observations`?qe(i,e):[233,236,240,e]}))}else{let e=a?_n():210;l.push(new xe({id:`pattern-selected-${r.id}`,coordinateOrigin:o,analysisFrame:c.frame,positions:r.positionsBuffer,instanceCount:r.count,values:r.maskBuffer,valueFormat:`uint32`,colormap:`mask`,color:r.id===`observations`?Ye(i,e):qe(i,e),noDataColor:[0,0,0,0],radiusPixels:d,blending:a?`additive`:`normal`}))}return l.push(new Se({id:`pattern-window`,coordinateOrigin:o,analysisFrame:c.frame,segments:t.windowOutline,weights:t.windowOutlineWeights,instanceCount:t.windowEdgeCapacity,widthPixels:1.5,color:Ke(i,235)})),l},destroy(){N=!0;for(let e of re.values())e.reader.stop();for(let e of ie.values())e.reader.stop();u.destroy()}}}export{Ht as createPointPatterns};