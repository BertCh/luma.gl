import{a as e}from"./arithmetic-expression-Kknn3j_i.js";import{D as t,F as n,S as r,k as i,w as a,x as o}from"./resources-CwvhEOBe.js";import{u as s}from"./interleave-brZZCO-f.js";import{n as c,t as l}from"./gpu-scan-BuGhTLDQ.js";import{t as u}from"./gpu-reduction-BD8l40jM.js";import{c as d,l as f,u as p}from"./gpu-graph-topology-view-DhuT3ULg.js";var m=256,h=4294967295;function g(e,t,n){let r={sourceVertices:t.importGPUVector(`${e.id}-source-vertices`,e.graph.sourceVertices),targetVertices:t.importGPUVector(`${e.id}-target-vertices`,e.graph.targetVertices),...e.graph.edgeIds?{edgeIds:t.importGPUVector(`${e.id}-source-edge-ids`,e.graph.edgeIds)}:{},...e.graph.edgeWeights?{edgeWeights:t.importGPUVector(`${e.id}-source-edge-weights`,e.graph.edgeWeights)}:{}},i=t.importGPUVector(`${e.id}-invalid-edge-count`,e.invalidEdgeCount).data[0];v(t,{id:`${e.id}-forward`,vertexCount:e.graph.vertexCount,directed:e.graph.directed,reverse:!1,sources:r,adjacency:_(t,`${e.id}-forward`,e.forward),invalidEdgeCount:i,maxComputeWorkgroupsPerDimension:n}),e.reverse&&v(t,{id:`${e.id}-reverse`,vertexCount:e.graph.vertexCount,directed:e.graph.directed,reverse:!0,sources:r,adjacency:_(t,`${e.id}-reverse`,e.reverse),maxComputeWorkgroupsPerDimension:n})}function _(e,t,n){return{offsets:e.importGPUVector(`${t}-offsets`,n.offsets).data[0],neighbors:e.importGPUVector(`${t}-neighbors`,n.neighbors).data[0],edgeIds:e.importGPUVector(`${t}-edge-ids`,n.edgeIds).data[0],...n.edgeWeights?{edgeWeights:e.importGPUVector(`${t}-edge-weights`,n.edgeWeights).data[0]}:{},count:e.importGPUVector(`${t}-count`,n.count).data[0],overflow:e.importGPUVector(`${t}-overflow`,n.overflow).data[0]}}function v(e,t){let r=a(e,`${t.id}-degrees`,`uint32`,t.vertexCount),i=a(e,`${t.id}-cursors`,`uint32`,t.vertexCount),o=a(e,`${t.id}-scanned-offsets`,`uint32`,t.vertexCount);y(e,{id:`${t.id}-initialize`,degrees:r,cursors:i,invalidEdgeCount:t.invalidEdgeCount,dispatchLayout:S(Math.max(t.vertexCount,1),t.maxComputeWorkgroupsPerDimension)});for(let[n,i]of t.sources.sourceVertices.data.entries())i.length>0&&b(e,{id:`${t.id}-degrees-${n}`,sourceVertices:i,targetVertices:t.sources.targetVertices.data[n],degrees:r,invalidEdgeCount:t.invalidEdgeCount,vertexCount:t.vertexCount,directed:t.directed,reverse:t.reverse,dispatchLayout:S(i.length,t.maxComputeWorkgroupsPerDimension)});n(e,c(new l({id:`${t.id}-scan`,input:r,output:o}),e,t.maxComputeWorkgroupsPerDimension)),ee(e,{id:`${t.id}-finalize`,degrees:r,scannedOffsets:o,adjacency:t.adjacency,vertexCount:t.vertexCount,dispatchLayout:S(Math.max(t.vertexCount,1),t.maxComputeWorkgroupsPerDimension)});let s=0;for(let[n,r]of t.sources.sourceVertices.data.entries()){if(r.length>0&&t.adjacency.neighbors.length>0){let c=t.sources.edgeWeights?.data[n],l=c?a(e,`${t.id}-slots-${n}`,`uint32`,r.length*(t.directed?1:2)):void 0,u=S(r.length,t.maxComputeWorkgroupsPerDimension);te(e,{id:`${t.id}-scatter-${n}`,sourceVertices:r,targetVertices:t.sources.targetVertices.data[n],sourceEdgeIds:t.sources.edgeIds?.data[n],sourceBase:s,scannedOffsets:o,cursors:i,adjacency:t.adjacency,slots:l,vertexCount:t.vertexCount,directed:t.directed,reverse:t.reverse,dispatchLayout:u}),c&&l&&t.adjacency.edgeWeights&&ne(e,{id:`${t.id}-weights-${n}`,sourceWeights:c,outputWeights:t.adjacency.edgeWeights,slots:l,directed:t.directed,dispatchLayout:u})}s+=r.length}}function y(e,t){let n=t.invalidEdgeCount?`const INVALID_OFFSET: u32 = ${i(t.invalidEdgeCount)}u;
@group(0) @binding(2) var<storage, read_write> invalidEdges: array<atomic<u32>>;`:``,r=t.invalidEdgeCount?`if (index == 0u) { atomicStore(&invalidEdges[INVALID_OFFSET], 0u); }`:``,a=`
const VERTEX_COUNT: u32 = ${t.degrees.length}u;
const DEGREES_OFFSET: u32 = ${i(t.degrees)}u;
const CURSORS_OFFSET: u32 = ${i(t.cursors)}u;
@group(0) @binding(0) var<storage, read_write> degrees: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cursors: array<atomic<u32>>;
${n}

@compute @workgroup_size(${m})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${C(t.dispatchLayout)}
  if (index < VERTEX_COUNT) {
    atomicStore(&degrees[DEGREES_OFFSET + index], 0u);
    atomicStore(&cursors[CURSORS_OFFSET + index], 0u);
  }
  ${r}
}`,o={degrees:t.degrees,cursors:t.cursors},s=[{buffer:t.degrees,usage:`storage-write`},{buffer:t.cursors,usage:`storage-write`}];t.invalidEdgeCount&&(o.invalidEdges=t.invalidEdgeCount,s.push({buffer:t.invalidEdgeCount,usage:`storage-write`})),x(e,{operation:`GPUGraphTopology`,workgroupSize:m,datum:{keySpace:`vertex`},invocationCount:t.degrees.length,id:t.id,source:a,bindings:o,resources:s,dispatchLayout:t.dispatchLayout})}function b(e,t){let n=t.invalidEdgeCount?`const INVALID_OFFSET: u32 = ${i(t.invalidEdgeCount)}u;
@group(0) @binding(3) var<storage, read_write> invalidEdges: array<atomic<u32>>;`:``,r=t.invalidEdgeCount?`atomicAdd(&invalidEdges[INVALID_OFFSET], 1u);`:``,a=t.reverse?`targetVertex`:`sourceVertex`,o=t.reverse?`sourceVertex`:`targetVertex`,s=t.directed?``:`if (sourceVertex != targetVertex) {
    atomicAdd(&degrees[DEGREES_OFFSET + ${o}], 1u);
  }`,c=`
const EDGE_COUNT: u32 = ${t.sourceVertices.length}u;
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const SOURCES_OFFSET: u32 = ${i(t.sourceVertices)}u;
const TARGETS_OFFSET: u32 = ${i(t.targetVertices)}u;
const DEGREES_OFFSET: u32 = ${i(t.degrees)}u;
@group(0) @binding(0) var<storage, read> sourceVertices: array<u32>;
@group(0) @binding(1) var<storage, read> targetVertices: array<u32>;
@group(0) @binding(2) var<storage, read_write> degrees: array<atomic<u32>>;
${n}

@compute @workgroup_size(${m})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${C(t.dispatchLayout)}
  if (index >= EDGE_COUNT) { return; }
  let sourceVertex = sourceVertices[SOURCES_OFFSET + index];
  let targetVertex = targetVertices[TARGETS_OFFSET + index];
  if (sourceVertex >= VERTEX_COUNT || targetVertex >= VERTEX_COUNT) {
    ${r}
    return;
  }
  atomicAdd(&degrees[DEGREES_OFFSET + ${a}], 1u);
  ${s}
}`,l={sourceVertices:t.sourceVertices,targetVertices:t.targetVertices,degrees:t.degrees},u=[{buffer:t.sourceVertices,usage:`storage-read`},{buffer:t.targetVertices,usage:`storage-read`},{buffer:t.degrees,usage:`storage-read-write`}];t.invalidEdgeCount&&(l.invalidEdges=t.invalidEdgeCount,u.push({buffer:t.invalidEdgeCount,usage:`storage-read-write`})),x(e,{operation:`GPUGraphTopology`,workgroupSize:m,datum:{keySpace:`edge`},invocationCount:t.sourceVertices.length,id:t.id,source:c,bindings:l,resources:u,dispatchLayout:t.dispatchLayout})}function ee(e,t){let n=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const CAPACITY: u32 = ${t.adjacency.neighbors.length}u;
const DEGREES_OFFSET: u32 = ${i(t.degrees)}u;
const SCANNED_OFFSET: u32 = ${i(t.scannedOffsets)}u;
const OFFSETS_OFFSET: u32 = ${i(t.adjacency.offsets)}u;
const COUNT_OFFSET: u32 = ${i(t.adjacency.count)}u;
const OVERFLOW_OFFSET: u32 = ${i(t.adjacency.overflow)}u;
@group(0) @binding(0) var<storage, read> degrees: array<u32>;
@group(0) @binding(1) var<storage, read> scannedOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputCount: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputOverflow: array<u32>;

@compute @workgroup_size(${m})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${C(t.dispatchLayout)}
  if (index < VERTEX_COUNT) {
    offsets[OFFSETS_OFFSET + index] = scannedOffsets[SCANNED_OFFSET + index];
  }
  if (index == 0u) {
    var total = 0u;
    if (VERTEX_COUNT > 0u) {
      let last = max(VERTEX_COUNT, 1u) - 1u;
      total = scannedOffsets[SCANNED_OFFSET + last] + degrees[DEGREES_OFFSET + last];
    }
    offsets[OFFSETS_OFFSET + VERTEX_COUNT] = total;
    outputCount[COUNT_OFFSET] = total;
    outputOverflow[OVERFLOW_OFFSET] = select(0u, 1u, total > CAPACITY);
  }
}`;x(e,{operation:`GPUGraphTopology`,workgroupSize:m,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:t.id,source:n,resources:[{buffer:t.degrees,usage:`storage-read`},{buffer:t.scannedOffsets,usage:`storage-read`},{buffer:t.adjacency.offsets,usage:`storage-write`},{buffer:t.adjacency.count,usage:`storage-write`},{buffer:t.adjacency.overflow,usage:`storage-write`}],bindings:{degrees:t.degrees,scannedOffsets:t.scannedOffsets,offsets:t.adjacency.offsets,outputCount:t.adjacency.count,outputOverflow:t.adjacency.overflow},dispatchLayout:t.dispatchLayout})}function te(e,t){let n=t.sourceEdgeIds?`const SOURCE_EDGE_IDS_OFFSET: u32 = ${i(t.sourceEdgeIds)}u;
@group(0) @binding(6) var<storage, read> sourceEdgeIds: array<u32>;`:``,r=t.sourceEdgeIds?7:6,a=t.slots?`const SLOTS_OFFSET: u32 = ${i(t.slots)}u;
@group(0) @binding(${r}) var<storage, read_write> slots: array<u32>;`:``,o=t.directed?1:2,s=t.slots?`slots[SLOTS_OFFSET + index * ${o}u] = ${h}u;
  ${t.directed?``:`slots[SLOTS_OFFSET + index * 2u + 1u] = ${h}u;`}`:``,c=t.reverse?`targetVertex`:`sourceVertex`,l=t.reverse?`sourceVertex`:`targetVertex`,u=l,d=c,f=t.slots?`slots[SLOTS_OFFSET + index * ${o}u] = firstSlot;`:``,p=t.slots?`slots[SLOTS_OFFSET + index * 2u + 1u] = secondSlot;`:``,g=t.directed?``:`if (sourceVertex != targetVertex) {
    let secondSlot = scannedOffsets[SCANNED_OFFSET + ${u}] +
      atomicAdd(&cursors[CURSORS_OFFSET + ${u}], 1u);
    if (secondSlot < CAPACITY) {
      neighbors[NEIGHBORS_OFFSET + secondSlot] = ${d};
      outputEdgeIds[OUTPUT_EDGE_IDS_OFFSET + secondSlot] = edgeIdentifier;
      ${p}
    }
  }`,_=t.sourceEdgeIds?`sourceEdgeIds[SOURCE_EDGE_IDS_OFFSET + index]`:`${t.sourceBase}u + index`,v=`
const EDGE_COUNT: u32 = ${t.sourceVertices.length}u;
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const CAPACITY: u32 = ${t.adjacency.neighbors.length}u;
const SOURCES_OFFSET: u32 = ${i(t.sourceVertices)}u;
const TARGETS_OFFSET: u32 = ${i(t.targetVertices)}u;
const SCANNED_OFFSET: u32 = ${i(t.scannedOffsets)}u;
const CURSORS_OFFSET: u32 = ${i(t.cursors)}u;
const NEIGHBORS_OFFSET: u32 = ${i(t.adjacency.neighbors)}u;
const OUTPUT_EDGE_IDS_OFFSET: u32 = ${i(t.adjacency.edgeIds)}u;
@group(0) @binding(0) var<storage, read> sourceVertices: array<u32>;
@group(0) @binding(1) var<storage, read> targetVertices: array<u32>;
@group(0) @binding(2) var<storage, read> scannedOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cursors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> neighbors: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputEdgeIds: array<u32>;
${n}
${a}

@compute @workgroup_size(${m})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${C(t.dispatchLayout)}
  if (index >= EDGE_COUNT) { return; }
  ${s}
  let sourceVertex = sourceVertices[SOURCES_OFFSET + index];
  let targetVertex = targetVertices[TARGETS_OFFSET + index];
  if (sourceVertex >= VERTEX_COUNT || targetVertex >= VERTEX_COUNT) { return; }
  let edgeIdentifier = ${_};
  let firstSlot = scannedOffsets[SCANNED_OFFSET + ${c}] +
    atomicAdd(&cursors[CURSORS_OFFSET + ${c}], 1u);
  if (firstSlot < CAPACITY) {
    neighbors[NEIGHBORS_OFFSET + firstSlot] = ${l};
    outputEdgeIds[OUTPUT_EDGE_IDS_OFFSET + firstSlot] = edgeIdentifier;
    ${f}
  }
  ${g}
}`,y={sourceVertices:t.sourceVertices,targetVertices:t.targetVertices,scannedOffsets:t.scannedOffsets,cursors:t.cursors,neighbors:t.adjacency.neighbors,outputEdgeIds:t.adjacency.edgeIds},b=[{buffer:t.sourceVertices,usage:`storage-read`},{buffer:t.targetVertices,usage:`storage-read`},{buffer:t.scannedOffsets,usage:`storage-read`},{buffer:t.cursors,usage:`storage-read-write`},{buffer:t.adjacency.neighbors,usage:`storage-write`},{buffer:t.adjacency.edgeIds,usage:`storage-write`}];t.sourceEdgeIds&&(y.sourceEdgeIds=t.sourceEdgeIds,b.push({buffer:t.sourceEdgeIds,usage:`storage-read`})),t.slots&&(y.slots=t.slots,b.push({buffer:t.slots,usage:`storage-write`})),x(e,{operation:`GPUGraphTopology`,workgroupSize:m,datum:{keySpace:`edge`},invocationCount:t.sourceVertices.length,id:t.id,source:v,bindings:y,resources:b,dispatchLayout:t.dispatchLayout})}function ne(e,t){let n=t.directed?1:2,r=t.directed?``:`let secondSlot = slots[SLOTS_OFFSET + index * 2u + 1u];
  if (secondSlot != ${h}u) {
    outputWeights[OUTPUT_OFFSET + secondSlot] = weight;
  }`,a=`
const EDGE_COUNT: u32 = ${t.sourceWeights.length}u;
const SOURCE_OFFSET: u32 = ${i(t.sourceWeights)}u;
const OUTPUT_OFFSET: u32 = ${i(t.outputWeights)}u;
const SLOTS_OFFSET: u32 = ${i(t.slots)}u;
@group(0) @binding(0) var<storage, read> sourceWeights: array<f32>;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputWeights: array<f32>;

@compute @workgroup_size(${m})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${C(t.dispatchLayout)}
  if (index >= EDGE_COUNT) { return; }
  let weight = sourceWeights[SOURCE_OFFSET + index];
  let firstSlot = slots[SLOTS_OFFSET + index * ${n}u];
  if (firstSlot != ${h}u) {
    outputWeights[OUTPUT_OFFSET + firstSlot] = weight;
  }
  ${r}
}`;x(e,{operation:`GPUGraphTopology`,workgroupSize:m,datum:{keySpace:`edge`},invocationCount:t.sourceWeights.length,id:t.id,source:a,resources:[{buffer:t.sourceWeights,usage:`storage-read`},{buffer:t.slots,usage:`storage-read`},{buffer:t.outputWeights,usage:`storage-write`}],bindings:{sourceWeights:t.sourceWeights,slots:t.slots,outputWeights:t.outputWeights},dispatchLayout:t.dispatchLayout})}function x(e,n){let r=e.recording?d(e,{operation:n.operation,source:n.source,bindings:Object.entries(n.bindings).map(([e,t],r)=>({name:e,view:t,usage:n.resources[r].usage})),workgroupSize:n.workgroupSize,dispatch:n.dispatchLayout,invocationCount:n.invocationCount,datum:n.datum}):void 0;e.addComputePass({...r&&{annotation:r},id:n.id,resources:n.resources,compile:({device:e})=>{let r=new s(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,r]of Object.entries(n.bindings))a[e]=t(r,i);r.setBindings(a),r.dispatch(e,n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z)},destroy:()=>r.destroy()}}})}function S(e,t){return o(`GPUGraphTopology`,e,m,t)}function C(e){return r(e,m)}var w=4294967295,T=4,re=class{id;graph;forward;reverse;invalidEdgeCount;constructor(e){if(this.id=e.id??`gpu-graph-topology`,this.graph=e.graph,this.forward=e.forward,this.reverse=e.reverse,this.invalidEdgeCount=e.invalidEdgeCount,!Number.isSafeInteger(this.graph.edgeCount)||this.graph.edgeCount<0)throw Error(`${this.id} source edge count must be a non-negative uint32`);if(this.graph.edgeCount>w||!this.graph.directed&&this.graph.edgeCount>Math.floor(w/2))throw Error(`${this.id} directed or symmetrized adjacency counts must fit in uint32`);if(this.reverse&&!this.graph.directed)throw Error(`${this.id} reverse adjacency requires a directed graph`);if(E(this.graph,this.forward,`${this.id} forward`),this.reverse&&E(this.graph,this.reverse,`${this.id} reverse`),D(this.invalidEdgeCount,`uint32`,`${this.id} invalidEdgeCount`),this.invalidEdgeCount.length!==1)throw Error(`${this.id} invalidEdgeCount must contain exactly one uint32 row`);ie(this)}addToGraph(e){p(e,`GPUGraphTopology`,this.id,()=>{g(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function E(e,t,n){if(D(t.offsets,`uint32`,`${n} offsets`),D(t.neighbors,`uint32`,`${n} neighbors`),D(t.edgeIds,`uint32`,`${n} edgeIds`),D(t.count,`uint32`,`${n} count`),D(t.overflow,`uint32`,`${n} overflow`),t.offsets.length!==e.vertexCount+1)throw Error(`${n} offsets must contain vertexCount + 1 rows`);if(t.neighbors.length!==t.edgeIds.length)throw Error(`${n} neighbors and edgeIds must have matching capacities`);if(t.count.length!==1||t.overflow.length!==1)throw Error(`${n} count and overflow must each contain exactly one uint32 row`);if(!!t.edgeWeights!=!!e.edgeWeights)throw Error(`${n} edgeWeights must match the source graph edge weights`);if(t.edgeWeights&&(D(t.edgeWeights,`float32`,`${n} edgeWeights`),t.edgeWeights.length!==t.neighbors.length))throw Error(`${n} edgeWeights and neighbors must have matching capacities`)}function D(e,t,n){if(e.data.length!==1||e.format!==t||e.stride!==1||e.byteStride!==T||e.rowByteLength!==T||e.valueLength!==e.length||e.bufferLayout||!Number.isSafeInteger(e.length)||e.length<0||e.length>w)throw Error(`${n} must contain one packed ${t} chunk with uint32 capacity`);let r=e.data[0];if(r.format!==t||r.length!==e.length||r.stride!==1||r.byteStride!==T||r.rowByteLength!==T||r.valueLength!==r.length||!Number.isSafeInteger(r.byteOffset)||r.byteOffset<0||r.byteOffset%T!==0)throw Error(`${n} must contain one packed, uint32-aligned ${t} chunk`)}function ie(e){let t=new Map,n=[e.graph.sourceVertices,e.graph.targetVertices,...e.graph.edgeWeights?[e.graph.edgeWeights]:[],...e.graph.edgeIds?[e.graph.edgeIds]:[]];for(let e of n)for(let n of e.data){let r=k(n);t.has(r)||t.set(r,e.name)}let r=[...O(e.forward,`forward`),...e.reverse?O(e.reverse,`reverse`):[],{name:`invalidEdgeCount`,vector:e.invalidEdgeCount}];for(let{name:n,vector:i}of r){let r=k(i.data[0]);if(t.get(r)!==void 0)throw Error(`${e.id} ${n} must use a distinct physical buffer allocation`);t.set(r,n)}}function O(e,t){return[{name:`${t}.offsets`,vector:e.offsets},{name:`${t}.neighbors`,vector:e.neighbors},{name:`${t}.edgeIds`,vector:e.edgeIds},...e.edgeWeights?[{name:`${t}.edgeWeights`,vector:e.edgeWeights}]:[],{name:`${t}.count`,vector:e.count},{name:`${t}.overflow`,vector:e.overflow}]}function k(t){return t.buffer instanceof e?t.buffer.buffer:t.buffer}var A=256;function ae(e,t,n){let r=e.graph.vertexCount,i={id:e.id,vertexCount:r,directed:e.graph.directed,resolution:Math.fround(e.resolution),sourceVertices:t.importGPUVector(`${e.id}-source-vertices`,e.graph.sourceVertices),targetVertices:t.importGPUVector(`${e.id}-target-vertices`,e.graph.targetVertices),...e.graph.edgeWeights?{edgeWeights:t.importGPUVector(`${e.id}-source-weights`,e.graph.edgeWeights)}:{},communities:t.importGPUVector(`${e.id}-communities`,e.communities).data[0],outgoingVolumes:a(t,`${e.id}-outgoing-volumes`,`float32`,r),...e.graph.directed?{incomingVolumes:a(t,`${e.id}-incoming-volumes`,`float32`,r)}:{},internalWeights:a(t,`${e.id}-internal-weights`,`float32`,r),contributions:e.communityContributions?t.importGPUVector(`${e.id}-community-contributions`,e.communityContributions).data[0]:a(t,`${e.id}-contribution-scratch`,`float32`,r),totalVolume:a(t,`${e.id}-total-volume`,`float32`,1),status:a(t,`${e.id}-invalid-status`,`uint32`,1),output:t.importGPUVector(`${e.id}-output`,e.output).data[0],...e.valid?{valid:t.importGPUVector(`${e.id}-valid`,e.valid).data[0]}:{},maxComputeWorkgroupsPerDimension:n};oe(t,i),r>0&&se(t,i);for(let[e,n]of i.sourceVertices.data.entries())n.length!==0&&ce(t,{state:i,chunkIndex:e,sources:n,targets:i.targetVertices.data[e],weights:i.edgeWeights?.data[e]});t.add(new u({id:`${i.id}-total-volume-reduction`,input:i.outgoingVolumes,output:i.totalVolume,operation:`sum`})),r>0&&j(t,i),t.add(new u({id:`${i.id}-score-reduction`,input:i.contributions,output:i.output,operation:`sum`})),M(t,i)}function oe(e,t){let n={status:{view:t.status,usage:`storage-write`,atomic:!0},output:{view:t.output,usage:`storage-write`},...t.valid?{validity:{view:t.valid,usage:`storage-write`}}:{}},r=t.valid?`validity[${i(t.valid)}u] = 0u;`:``,a=`
${P(n)}

@compute @workgroup_size(1)
fn main() {
  atomicStore(&status[${i(t.status)}u], 0u);
  output[${i(t.output)}u] = 0.0;
  ${r}
}`;F(e,{operation:`GPUGraphModularity`,workgroupSize:1,invocationCount:1,id:`${t.id}-initialize-status`,source:a,bindings:n,dispatchLayout:{x:1,y:1,z:1}})}function se(e,t){let n={communities:{view:t.communities,usage:`storage-read`},outgoingVolumes:{view:t.outgoingVolumes,usage:`storage-write`,atomic:!0},...t.incomingVolumes?{incomingVolumes:{view:t.incomingVolumes,usage:`storage-write`,atomic:!0}}:{},internalWeights:{view:t.internalWeights,usage:`storage-write`,atomic:!0},contributions:{view:t.contributions,usage:`storage-write`},status:{view:t.status,usage:`storage-read-write`,atomic:!0}},a=t.incomingVolumes?`atomicStore(&incomingVolumes[${i(t.incomingVolumes)}u + index], 0u);`:``,o=I(t,t.vertexCount),s=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
${P(n)}

@compute @workgroup_size(${A})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,A)}
  if (index >= VERTEX_COUNT) { return; }
  atomicStore(&outgoingVolumes[${i(t.outgoingVolumes)}u + index], 0u);
  ${a}
  atomicStore(&internalWeights[${i(t.internalWeights)}u + index], 0u);
  contributions[${i(t.contributions)}u + index] = 0.0;
  if (communities[${i(t.communities)}u + index] >= VERTEX_COUNT) {
    atomicStore(&status[${i(t.status)}u], 1u);
  }
}`;F(e,{operation:`GPUGraphModularity`,workgroupSize:A,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-initialize-communities`,source:s,bindings:n,dispatchLayout:o})}function ce(e,t){let{state:n}=t,a={sourceVertices:{view:t.sources,usage:`storage-read`},targetVertices:{view:t.targets,usage:`storage-read`},...t.weights?{edgeWeights:{view:t.weights,usage:`storage-read`}}:{},communities:{view:n.communities,usage:`storage-read`},outgoingVolumes:{view:n.outgoingVolumes,usage:`storage-read-write`,atomic:!0},...n.incomingVolumes?{incomingVolumes:{view:n.incomingVolumes,usage:`storage-read-write`,atomic:!0}}:{},internalWeights:{view:n.internalWeights,usage:`storage-read-write`,atomic:!0},status:{view:n.status,usage:`storage-read-write`,atomic:!0}},o=t.weights?`edgeWeights[${i(t.weights)}u + index]`:`1.0`,s=n.incomingVolumes?`atomicAddFloat(&incomingVolumes[${i(n.incomingVolumes)}u + targetCommunity], edgeWeight);`:`atomicAddFloat(&outgoingVolumes[${i(n.outgoingVolumes)}u + targetCommunity], edgeWeight);`,c=I(n,t.sources.length),l=`
const EDGE_COUNT: u32 = ${t.sources.length}u;
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
${P(a)}

${N()}

@compute @workgroup_size(${A})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(c,A)}
  if (index >= EDGE_COUNT) { return; }
  let sourceVertex = sourceVertices[${i(t.sources)}u + index];
  let targetVertex = targetVertices[${i(t.targets)}u + index];
  if (sourceVertex >= VERTEX_COUNT || targetVertex >= VERTEX_COUNT) { return; }

  let edgeWeight = ${o};
  let weightMagnitudeBits = bitcast<u32>(edgeWeight) & 0x7fffffffu;
  if (weightMagnitudeBits >= 0x7f800000u || edgeWeight < 0.0) {
    atomicStore(&status[${i(n.status)}u], 1u);
    return;
  }

  let sourceCommunity = communities[${i(n.communities)}u + sourceVertex];
  let targetCommunity = communities[${i(n.communities)}u + targetVertex];
  if (sourceCommunity >= VERTEX_COUNT || targetCommunity >= VERTEX_COUNT) {
    atomicStore(&status[${i(n.status)}u], 1u);
    return;
  }

  atomicAddFloat(&outgoingVolumes[${i(n.outgoingVolumes)}u + sourceCommunity], edgeWeight);
  ${s}
  if (sourceCommunity == targetCommunity) {
    atomicAddFloat(&internalWeights[${i(n.internalWeights)}u + sourceCommunity], edgeWeight);
  }
}`;F(e,{operation:`GPUGraphModularity`,workgroupSize:A,datum:{keySpace:`edge`},invocationCount:t.sources.length,id:`${n.id}-accumulate-chunk-${t.chunkIndex}`,source:l,bindings:a,dispatchLayout:c})}function j(e,t){let n={outgoingVolumes:{view:t.outgoingVolumes,usage:`storage-read`},...t.incomingVolumes?{incomingVolumes:{view:t.incomingVolumes,usage:`storage-read`}}:{},internalWeights:{view:t.internalWeights,usage:`storage-read`},totalVolume:{view:t.totalVolume,usage:`storage-read`},contributions:{view:t.contributions,usage:`storage-write`},status:{view:t.status,usage:`storage-read-write`,atomic:!0}},a=t.incomingVolumes?`incomingVolumes[${i(t.incomingVolumes)}u + index]`:`outgoingVolumes[${i(t.outgoingVolumes)}u + index]`,o=t.directed?`1.0`:`2.0`,s=I(t,t.vertexCount),c=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const RESOLUTION: f32 = ${t.resolution.toExponential()};
${P(n)}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

@compute @workgroup_size(${A})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(s,A)}
  if (index >= VERTEX_COUNT) { return; }

  let volume = totalVolume[${i(t.totalVolume)}u];
  if (atomicLoad(&status[${i(t.status)}u]) != 0u ||
      !isFiniteValue(volume) || volume <= 0.0) {
    contributions[${i(t.contributions)}u + index] = 0.0;
    return;
  }

  let internal = internalWeights[${i(t.internalWeights)}u + index];
  let outgoing = outgoingVolumes[${i(t.outgoingVolumes)}u + index];
  let incoming = ${a};
  let contribution =
    (internal / volume) * ${o} - RESOLUTION * (outgoing / volume) * (incoming / volume);
  if (!isFiniteValue(contribution)) {
    atomicStore(&status[${i(t.status)}u], 1u);
    contributions[${i(t.contributions)}u + index] = 0.0;
    return;
  }
  contributions[${i(t.contributions)}u + index] = contribution;
}`;F(e,{operation:`GPUGraphModularity`,workgroupSize:A,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-community-contributions`,source:c,bindings:n,dispatchLayout:s})}function M(e,t){let n={status:{view:t.status,usage:`storage-read`},totalVolume:{view:t.totalVolume,usage:`storage-read`},output:{view:t.output,usage:`storage-read-write`},contributions:{view:t.contributions,usage:`storage-read-write`},...t.valid?{validity:{view:t.valid,usage:`storage-write`}}:{}},a=t.valid?`validity[${i(t.valid)}u] = select(0u, 1u, isValid);`:``,o=I(t,Math.max(t.vertexCount,1)),s=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
${P(n)}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

@compute @workgroup_size(${A})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,A)}
  if (index >= max(VERTEX_COUNT, 1u)) { return; }
  let volume = totalVolume[${i(t.totalVolume)}u];
  let score = output[${i(t.output)}u];
  let isValid = status[${i(t.status)}u] == 0u &&
    isFiniteValue(volume) && volume > 0.0 && isFiniteValue(score);
  if (index == 0u) {
    output[${i(t.output)}u] = select(0.0, score, isValid);
    ${a}
  }
  if (index < VERTEX_COUNT && !isValid) {
    contributions[${i(t.contributions)}u + index] = 0.0;
  }
}`;F(e,{operation:`GPUGraphModularity`,workgroupSize:A,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-finalize-validity`,source:s,bindings:n,dispatchLayout:o})}function N(){return`
fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var previousBits = atomicLoad(destination);
  loop {
    let nextBits = bitcast<u32>(bitcast<f32>(previousBits) + value);
    let exchange = atomicCompareExchangeWeak(destination, previousBits, nextBits);
    if (exchange.exchanged) { return; }
    previousBits = exchange.old_value;
  }
}`}function P(e){return Object.entries(e).map(([e,t],n)=>`@group(0) @binding(${n}) var<storage, ${t.usage===`storage-read`?`read`:`read_write`}> ${e}: array<${t.atomic?`atomic<u32>`:t.view.format===`float32`?`f32`:`u32`}>;`).join(`
`)}function F(e,n){let r=e.recording?d(e,{operation:n.operation,source:n.source,bindings:Object.entries(n.bindings).map(([e,t])=>({name:e,...t})),workgroupSize:n.workgroupSize,dispatch:n.dispatchLayout,invocationCount:n.invocationCount,datum:n.datum}):void 0;e.addComputePass({...r&&{annotation:r},id:n.id,resources:Object.values(n.bindings).map(({view:e,usage:t})=>({buffer:e,usage:t})),compile:({device:e})=>{let r=new s(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,r]of Object.entries(n.bindings))a[e]=t(r.view,i);r.setBindings(a),r.dispatch(e,n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z)},destroy:()=>r.destroy()}}})}function I(e,t){return le(t,e.maxComputeWorkgroupsPerDimension)}function le(e,t){return o(`GPUGraphModularity`,e,A,t)}var L=4,R=class{id;graph;communities;output;resolution;communityContributions;valid;constructor(e){if(this.id=e.id??`gpu-graph-modularity`,this.graph=e.graph,this.communities=e.communities,this.output=e.output,this.resolution=e.resolution??1,this.communityContributions=e.communityContributions,this.valid=e.valid,!Number.isFinite(this.resolution)||this.resolution<0||!Number.isFinite(Math.fround(this.resolution)))throw Error(`${this.id} resolution must be a non-negative finite float32`);z(this.communities,`uint32`,this.graph.vertexCount,`${this.id} communities`),z(this.output,`float32`,1,`${this.id} output`),this.communityContributions&&z(this.communityContributions,`float32`,this.graph.vertexCount,`${this.id} communityContributions`),this.valid&&z(this.valid,`uint32`,1,`${this.id} valid`),ue(this)}addToGraph(e){p(e,`GPUGraphModularity`,this.id,()=>{ae(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function z(e,t,n,r){if(e.format!==t||e.data.length!==1||e.length!==n||e.stride!==1||e.byteStride!==L||e.rowByteLength!==L||e.valueLength!==e.length||e.bufferLayout)throw Error(`${r} must contain exactly ${n} packed ${t} rows in one chunk`);let i=e.data[0];if(i.format!==t||i.length!==n||i.stride!==1||i.byteStride!==L||i.rowByteLength!==L||i.valueLength!==i.length||!Number.isSafeInteger(i.byteOffset)||i.byteOffset<0||i.byteOffset%L!==0)throw Error(`${r} must contain one packed, uint32-aligned ${t} chunk`)}function ue(e){let t=[e.graph.sourceVertices,e.graph.targetVertices,...e.graph.edgeWeights?[e.graph.edgeWeights]:[],...e.graph.edgeIds?[e.graph.edgeIds]:[],e.communities],n=new Set;for(let e of t)for(let t of e.data)n.add(B(t));let r=[{name:`output`,vector:e.output},...e.communityContributions?[{name:`communityContributions`,vector:e.communityContributions}]:[],...e.valid?[{name:`valid`,vector:e.valid}]:[]];for(let{name:t,vector:i}of r){let r=B(i.data[0]);if(n.has(r))throw Error(`${e.id} ${t} must use a distinct physical buffer allocation`);n.add(r)}}function B(t){return t.buffer instanceof e?t.buffer.buffer:t.buffer}var V=256,H=4294967295,U=1,de=2;function fe(e,t,n){let r=e.topology,i=r.graph.vertexCount,o=r.graph.directed?r.reverse:void 0,s=Oe(i,r.forward.neighbors.length,o?.neighbors.length??0),c=Math.min(t.device.limits.maxBufferSize,t.device.limits.maxStorageBufferBindingSize),l=s!==void 0&&s*8<=c&&i*8<=c?s:void 0,u={id:e.id,vertexCount:i,directed:r.graph.directed,resolution:Math.fround(e.resolution),minimumGain:Math.fround(e.minimumGain),iterations:e.iterations,sourceVertices:t.importGPUVector(`${e.id}-source-vertices`,r.graph.sourceVertices),targetVertices:t.importGPUVector(`${e.id}-target-vertices`,r.graph.targetVertices),...r.graph.edgeWeights?{edgeWeights:t.importGPUVector(`${e.id}-edge-weights`,r.graph.edgeWeights)}:{},forwardOffsets:t.importGPUVector(`${e.id}-forward-offsets`,r.forward.offsets).data[0],forwardNeighbors:t.importGPUVector(`${e.id}-forward-neighbors`,r.forward.neighbors).data[0],...r.forward.edgeWeights?{forwardWeights:t.importGPUVector(`${e.id}-forward-weights`,r.forward.edgeWeights).data[0]}:{},forwardOverflow:t.importGPUVector(`${e.id}-forward-overflow`,r.forward.overflow).data[0],packedForward:a(t,`${e.id}-packed-forward`,`uint32x2`,r.forward.neighbors.length),...o?{reverseOffsets:t.importGPUVector(`${e.id}-reverse-offsets`,o.offsets).data[0],reverseNeighbors:t.importGPUVector(`${e.id}-reverse-neighbors`,o.neighbors).data[0],...o.edgeWeights?{reverseWeights:t.importGPUVector(`${e.id}-reverse-weights`,o.edgeWeights).data[0]}:{},reverseOverflow:t.importGPUVector(`${e.id}-reverse-overflow`,o.overflow).data[0],packedReverse:a(t,`${e.id}-packed-reverse`,`uint32x2`,o.neighbors.length)}:{},...e.initialCommunities?{initialCommunities:t.importGPUVector(`${e.id}-initial-communities`,e.initialCommunities).data[0]}:{},output:t.importGPUVector(`${e.id}-output`,e.output).data[0],statistics:a(t,`${e.id}-vertex-community-statistics`,`float32x4`,i),...l===void 0?{}:{aggregatedCommunities:a(t,`${e.id}-aggregated-communities`,`uint32`,l),aggregatedWeights:a(t,`${e.id}-aggregated-community-weights`,`float32x2`,l),aggregationRanges:a(t,`${e.id}-community-aggregation-ranges`,`uint32x2`,i)},candidates:a(t,`${e.id}-candidate-moves`,`uint32x2`,i),control:a(t,`${e.id}-control`,`uint32`,i+5),...e.converged?{converged:t.importGPUVector(`${e.id}-converged`,e.converged).data[0]}:{},maxComputeWorkgroupsPerDimension:n};pe(t,u),i>0&&me(t,u),u.forwardNeighbors.length>0&&W(t,{state:u,direction:`forward`,neighbors:u.forwardNeighbors,weights:u.forwardWeights,packed:u.packedForward}),u.reverseNeighbors&&u.packedReverse&&u.reverseNeighbors.length>0&&W(t,{state:u,direction:`reverse`,neighbors:u.reverseNeighbors,weights:u.reverseWeights,packed:u.packedReverse});for(let n=0;n<Math.max(e.iterations,1);n++)f(t,{loopId:`${u.id}-iterations`,iteration:n,maximumIterations:Math.max(e.iterations,1),gated:!1},()=>{he(t,{state:u,iteration:n}),i>0&&(ge(t,{state:u,iteration:n}),n<e.iterations&&(_e(t,{state:u,iteration:n}),ve(t,{state:u,iteration:n})));for(let[e,r]of u.sourceVertices.data.entries())r.length!==0&&ye(t,{state:u,iteration:n,chunkIndex:e,sources:r,targets:u.targetVertices.data[e],weights:u.edgeWeights?.data[e]});n<e.iterations&&i>0&&(be(u)?(xe(t,{state:u,iteration:n}),Se(t,{state:u,iteration:n})):Ce(t,{state:u,iteration:n}),we(t,{state:u,iteration:n}),Te(t,{state:u,iteration:n}))});Ee(t,u),new R({id:`${e.id}-final-modularity`,graph:r.graph,communities:e.output,output:e.modularity,resolution:e.resolution,...e.valid?{valid:e.valid}:{}}).addToGraph(t)}function pe(e,t){let n={control:{view:t.control,usage:`storage-read-write`,atomic:!0},forwardOverflow:{view:t.forwardOverflow,usage:`storage-read`},...t.reverseOverflow?{reverseOverflow:{view:t.reverseOverflow,usage:`storage-read`}}:{},...t.converged?{converged:{view:t.converged,usage:`storage-write`}}:{}},r=t.reverseOverflow?` || reverseOverflow[${i(t.reverseOverflow)}u] != 0u`:``,a=t.converged?`converged[${i(t.converged)}u] = select(0u, 1u, VERTEX_COUNT == 0u && !hasOverflow);`:``,o=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
${K(n)}

@compute @workgroup_size(1)
fn main() {
  let hasOverflow = forwardOverflow[${i(t.forwardOverflow)}u] != 0u${r};
  atomicStore(&control[0u], 0u);
  atomicStore(&control[1u], select(0u, ${U}u, hasOverflow));
  atomicStore(&control[2u], 0u);
  atomicStore(&control[3u], ${H}u);
  atomicStore(&control[4u], ${H}u);
  ${a}
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:1,invocationCount:1,id:`${t.id}-initialize-control`,source:o,bindings:n,dispatchLayout:{x:1,y:1,z:1}})}function me(e,t){let n={output:{view:t.output,usage:`storage-write`},...t.initialCommunities?{initialCommunities:{view:t.initialCommunities,usage:`storage-read`}}:{},control:{view:t.control,usage:`storage-read-write`,atomic:!0}},a=t.initialCommunities?`initialCommunities[${i(t.initialCommunities)}u + index]`:`index`,o=J(t,t.vertexCount),s=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
${K(n)}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,V)}
  if (index >= VERTEX_COUNT) { return; }
  let community = ${a};
  output[${i(t.output)}u + index] = community;
  if (community >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${U}u);
  }
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-initialize-communities`,source:s,bindings:n,dispatchLayout:o})}function W(e,t){let n={neighbors:{view:t.neighbors,usage:`storage-read`},...t.weights?{weights:{view:t.weights,usage:`storage-read`}}:{},packed:{view:t.packed,usage:`storage-write`}},a=t.weights?`weights[${i(t.weights)}u + index]`:`1.0`,o=J(t.state,t.neighbors.length),s=`
const ADJACENCY_CAPACITY: u32 = ${t.neighbors.length}u;
${K(n)}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,V)}
  if (index >= ADJACENCY_CAPACITY) { return; }
  packed[index] = vec2<u32>(
    neighbors[${i(t.neighbors)}u + index],
    bitcast<u32>(${a})
  );
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`edge`},invocationCount:t.neighbors.length,id:`${t.state.id}-pack-${t.direction}-adjacency`,source:s,bindings:n,dispatchLayout:o})}function he(e,t){let n={control:{view:t.state.control,usage:`storage-read-write`,atomic:!0}},r=`
${K(n)}

@compute @workgroup_size(1)
fn main() {
  if (atomicLoad(&control[1u]) != 0u) { return; }
  atomicStore(&control[0u], 0u);
  atomicStore(&control[2u], 0u);
  atomicStore(&control[3u], ${H}u);
  atomicStore(&control[4u], ${H}u);
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:1,invocationCount:1,id:`${t.state.id}-iteration-${t.iteration}-reset`,source:r,bindings:n,dispatchLayout:{x:1,y:1,z:1}})}function ge(e,t){let{state:n}=t,a={output:{view:n.output,usage:`storage-read`},statistics:{view:n.statistics,usage:`storage-read-write`,atomic:!0},control:{view:n.control,usage:`storage-read-write`,atomic:!0}},o=J(n,n.vertexCount),s=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
${K(a)}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,V)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  atomicStore(&statistics[4u * index], 0u);
  atomicStore(&statistics[4u * index + 1u], 0u);
  atomicStore(&statistics[4u * index + 2u], 0u);
  atomicStore(&statistics[4u * index + 3u], 0u);
  atomicStore(&control[5u + index], 0u);
  if (output[${i(n.output)}u + index] >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${U}u);
  }
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-initialize-statistics`,source:s,bindings:a,dispatchLayout:o})}function _e(e,t){let{state:n}=t,a={communities:{view:n.output,usage:`storage-read`},control:{view:n.control,usage:`storage-read-write`,atomic:!0}},o=J(n,n.vertexCount),s=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
${K(a)}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,V)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  let community = communities[${i(n.output)}u + index];
  if (community >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${U}u);
    return;
  }
  atomicStore(&control[5u + community], 1u);
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-mark-community-occupancy`,source:s,bindings:a,dispatchLayout:o})}function ve(e,t){let{state:n}=t,i={control:{view:n.control,usage:`storage-read-write`,atomic:!0}},a=J(n,n.vertexCount),o=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
${K(i)}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(a,V)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  if (atomicLoad(&control[5u + index]) == 0u) {
    atomicMin(&control[4u], index);
  }
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-select-unused-community`,source:o,bindings:i,dispatchLayout:a})}function ye(e,t){let{state:n}=t,a={sourceVertices:{view:t.sources,usage:`storage-read`},targetVertices:{view:t.targets,usage:`storage-read`},...t.weights?{edgeWeights:{view:t.weights,usage:`storage-read`}}:{},communities:{view:n.output,usage:`storage-read`},statistics:{view:n.statistics,usage:`storage-read-write`,atomic:!0},control:{view:n.control,usage:`storage-read-write`,atomic:!0}},o=t.weights?`edgeWeights[${i(t.weights)}u + index]`:`1.0`,s=n.directed?``:`
  atomicAddFloat(&statistics[4u * targetVertex], edgeWeight);
  atomicAddFloat(&statistics[4u * sourceVertex + 1u], edgeWeight);
  atomicAddFloat(&statistics[4u * targetCommunity + 2u], edgeWeight);
  atomicAddFloat(&statistics[4u * sourceCommunity + 3u], edgeWeight);
  atomicAddFloat(&control[0u], edgeWeight);`,c=J(n,t.sources.length),l=`
const EDGE_COUNT: u32 = ${t.sources.length}u;
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
${K(a)}

${De()}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(c,V)}
  if (index >= EDGE_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  let sourceVertex = sourceVertices[${i(t.sources)}u + index];
  let targetVertex = targetVertices[${i(t.targets)}u + index];
  if (sourceVertex >= VERTEX_COUNT || targetVertex >= VERTEX_COUNT) { return; }

  let edgeWeight = ${o};
  if ((bitcast<u32>(edgeWeight) & 0x7fffffffu) >= 0x7f800000u || edgeWeight < 0.0) {
    atomicOr(&control[1u], ${U}u);
    return;
  }

  let sourceCommunity = communities[${i(n.output)}u + sourceVertex];
  let targetCommunity = communities[${i(n.output)}u + targetVertex];
  if (sourceCommunity >= VERTEX_COUNT || targetCommunity >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${U}u);
    return;
  }

  atomicAddFloat(&statistics[4u * sourceVertex], edgeWeight);
  atomicAddFloat(&statistics[4u * targetVertex + 1u], edgeWeight);
  atomicAddFloat(&statistics[4u * sourceCommunity + 2u], edgeWeight);
  atomicAddFloat(&statistics[4u * targetCommunity + 3u], edgeWeight);
  atomicAddFloat(&control[0u], edgeWeight);
  ${s}
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`edge`},invocationCount:t.sources.length,id:`${n.id}-iteration-${t.iteration}-accumulate-chunk-${t.chunkIndex}`,source:l,bindings:a,dispatchLayout:c})}function be(e){return!!(e.aggregatedCommunities&&e.aggregatedWeights&&e.aggregationRanges)}function xe(e,t){let{state:n}=t,a={forwardOffsets:{view:n.forwardOffsets,usage:`storage-read`},packedForward:{view:n.packedForward,usage:`storage-read`},...n.reverseOffsets&&n.packedReverse?{reverseOffsets:{view:n.reverseOffsets,usage:`storage-read`},packedReverse:{view:n.packedReverse,usage:`storage-read`}}:{},communities:{view:n.output,usage:`storage-read`},aggregatedCommunities:{view:n.aggregatedCommunities,usage:`storage-write`},aggregatedWeights:{view:n.aggregatedWeights,usage:`storage-write`},aggregationRanges:{view:n.aggregationRanges,usage:`storage-write`}},o=J(n,n.vertexCount),s=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const COMMUNITY_OFFSET: u32 = ${i(n.output)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${i(n.forwardOffsets)}u;
const FORWARD_CAPACITY: u32 = ${n.packedForward.length}u;
const DIRECTION_COUNT: u32 = ${n.directed?2:1}u;
${n.reverseOffsets&&n.packedReverse?`const REVERSE_OFFSETS_OFFSET: u32 = ${i(n.reverseOffsets)}u;
const REVERSE_CAPACITY: u32 = ${n.packedReverse.length}u;`:``}
${K(a)}

${G(n)}

fn getAggregationRange(vertex: u32) -> vec2<u32> {
  let forwardStart = getNeighborStart(vertex, 0u);
  let forwardDegree = getNeighborEnd(vertex, 0u) - forwardStart;
  var weakStart = forwardStart;
  var weakDegree = forwardDegree;
  ${n.directed?`let reverseStart = getNeighborStart(vertex, 1u);
  weakStart += reverseStart;
  weakDegree += getNeighborEnd(vertex, 1u) - reverseStart;`:``}
  let first = 2u * weakStart + vertex;
  return vec2<u32>(first, first + 2u * weakDegree + 1u);
}

fn hashCommunity(community: u32) -> u32 {
  var hash = community;
  hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
  hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
  return hash ^ (hash >> 16u);
}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,V)}
  if (index >= VERTEX_COUNT) { return; }

  let range = getAggregationRange(index);
  aggregationRanges[index] = range;
  let segmentCapacity = range.y - range.x;
  for (var slot = range.x; slot < range.y; slot++) {
    aggregatedCommunities[slot] = ${H}u;
    aggregatedWeights[slot] = vec2<f32>(0.0);
  }

  for (var direction = 0u; direction < DIRECTION_COUNT; direction++) {
    let first = getNeighborStart(index, direction);
    let last = getNeighborEnd(index, direction);
    for (var neighborSlot = first; neighborSlot < last; neighborSlot++) {
      let neighbor = getNeighbor(neighborSlot, direction);
      if (neighbor.x >= VERTEX_COUNT || neighbor.x == index) { continue; }
      let neighborCommunity = communities[COMMUNITY_OFFSET + neighbor.x];
      let edgeWeight = bitcast<f32>(neighbor.y);
      let contribution = select(
        vec2<f32>(edgeWeight, 0.0),
        vec2<f32>(0.0, edgeWeight),
        direction == 1u
      );
      var aggregateSlot = range.x + hashCommunity(neighborCommunity) % segmentCapacity;
      for (var attempt = 0u; attempt < segmentCapacity; attempt++) {
        let aggregateCommunity = aggregatedCommunities[aggregateSlot];
        if (aggregateCommunity == ${H}u) {
          aggregatedCommunities[aggregateSlot] = neighborCommunity;
          aggregatedWeights[aggregateSlot] = contribution;
          break;
        }
        if (aggregateCommunity == neighborCommunity) {
          aggregatedWeights[aggregateSlot] += contribution;
          break;
        }
        aggregateSlot = range.x + (aggregateSlot - range.x + 1u) % segmentCapacity;
      }
    }
  }
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-aggregate-neighbor-communities`,source:s,bindings:a,dispatchLayout:o})}function Se(e,t){let{state:n}=t,a={communities:{view:n.output,usage:`storage-read`},statistics:{view:n.statistics,usage:`storage-read`},aggregatedCommunities:{view:n.aggregatedCommunities,usage:`storage-read`},aggregatedWeights:{view:n.aggregatedWeights,usage:`storage-read`},aggregationRanges:{view:n.aggregationRanges,usage:`storage-read`},candidates:{view:n.candidates,usage:`storage-write`},control:{view:n.control,usage:`storage-read-write`,atomic:!0}},o=J(n,n.vertexCount),s=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const COMMUNITY_OFFSET: u32 = ${i(n.output)}u;
const RESOLUTION: f32 = ${n.resolution.toExponential()};
const MINIMUM_GAIN: f32 = ${n.minimumGain.toExponential()};
${K(a)}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

fn isFiniteVector(value: vec4<f32>) -> bool {
  let magnitude = bitcast<vec4<u32>>(value) & vec4<u32>(0x7fffffffu);
  return all(magnitude < vec4<u32>(0x7f800000u));
}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,V)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }

  let total = bitcast<f32>(atomicLoad(&control[0u]));
  if (!isFiniteValue(total) || total <= 0.0) {
    atomicOr(&control[1u], ${U}u);
    return;
  }

  let currentCommunity = communities[COMMUNITY_OFFSET + index];
  if (currentCommunity >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${U}u);
    return;
  }

  let vertexStatistics = statistics[index];
  let currentStatistics = statistics[currentCommunity];
  if (!isFiniteVector(vertexStatistics) || !isFiniteVector(currentStatistics)) {
    atomicOr(&control[1u], ${U}u);
    return;
  }

  let aggregationRange = aggregationRanges[index];
  var currentWeights = vec2<f32>(0.0);
  for (var slot = aggregationRange.x; slot < aggregationRange.y; slot++) {
    if (aggregatedCommunities[slot] == currentCommunity) {
      currentWeights = aggregatedWeights[slot];
      break;
    }
  }
  let currentOutgoingWeight = currentWeights.x;
  let currentIncomingWeight = ${n.directed?`currentWeights.y`:`currentWeights.x`};
  var bestCommunity = currentCommunity;
  var bestGain = 0.0;

  for (var slot = aggregationRange.x; slot < aggregationRange.y; slot++) {
    let candidateCommunity = aggregatedCommunities[slot];
    if (candidateCommunity >= VERTEX_COUNT || candidateCommunity == currentCommunity) {
      continue;
    }

    let candidateStatistics = statistics[candidateCommunity];
    if (!isFiniteVector(candidateStatistics)) {
      atomicOr(&control[1u], ${U}u);
      return;
    }

    let weights = aggregatedWeights[slot];
    let outgoingWeight = weights.x;
    let incomingWeight = ${n.directed?`weights.y`:`weights.x`};
    let observedGain =
      ((outgoingWeight - currentOutgoingWeight) +
       (incomingWeight - currentIncomingWeight)) / total;
    let expectedGain = RESOLUTION * (
      (vertexStatistics.y / total) *
        ((candidateStatistics.z - (currentStatistics.z - vertexStatistics.x)) / total) +
      (vertexStatistics.x / total) *
        ((candidateStatistics.w - (currentStatistics.w - vertexStatistics.y)) / total)
    );
    let gain = observedGain - expectedGain;
    if (!isFiniteValue(gain)) {
      atomicOr(&control[1u], ${U}u);
      return;
    }
    if (gain > MINIMUM_GAIN &&
        (gain > bestGain || (gain == bestGain && candidateCommunity < bestCommunity))) {
      bestGain = gain;
      bestCommunity = candidateCommunity;
    }
  }

  let availableCommunity = atomicLoad(&control[4u]);
  if (availableCommunity < VERTEX_COUNT && availableCommunity != currentCommunity) {
    let availableStatistics = statistics[availableCommunity];
    if (!isFiniteVector(availableStatistics)) {
      atomicOr(&control[1u], ${U}u);
      return;
    }
    let observedGain =
      ((0.0 - currentOutgoingWeight) + (0.0 - currentIncomingWeight)) / total;
    let expectedGain = RESOLUTION * (
      (vertexStatistics.y / total) *
        ((availableStatistics.z - (currentStatistics.z - vertexStatistics.x)) / total) +
      (vertexStatistics.x / total) *
        ((availableStatistics.w - (currentStatistics.w - vertexStatistics.y)) / total)
    );
    let gain = observedGain - expectedGain;
    if (!isFiniteValue(gain)) {
      atomicOr(&control[1u], ${U}u);
      return;
    }
    if (gain > MINIMUM_GAIN &&
        (gain > bestGain || (gain == bestGain && availableCommunity < bestCommunity))) {
      bestGain = gain;
      bestCommunity = availableCommunity;
    }
  }

  candidates[index] = vec2<u32>(bestCommunity, bitcast<u32>(bestGain));
  if (bestGain > MINIMUM_GAIN) {
    atomicMax(&control[2u], bitcast<u32>(bestGain));
  }
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-evaluate-candidates`,source:s,bindings:a,dispatchLayout:o})}function Ce(e,t){let{state:n}=t,a={forwardOffsets:{view:n.forwardOffsets,usage:`storage-read`},packedForward:{view:n.packedForward,usage:`storage-read`},...n.reverseOffsets&&n.packedReverse?{reverseOffsets:{view:n.reverseOffsets,usage:`storage-read`},packedReverse:{view:n.packedReverse,usage:`storage-read`}}:{},communities:{view:n.output,usage:`storage-read`},statistics:{view:n.statistics,usage:`storage-read`},candidates:{view:n.candidates,usage:`storage-write`},control:{view:n.control,usage:`storage-read-write`,atomic:!0}},o=J(n,n.vertexCount),s=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const COMMUNITY_OFFSET: u32 = ${i(n.output)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${i(n.forwardOffsets)}u;
const FORWARD_CAPACITY: u32 = ${n.packedForward.length}u;
const DIRECTION_COUNT: u32 = ${n.directed?2:1}u;
const RESOLUTION: f32 = ${n.resolution.toExponential()};
const MINIMUM_GAIN: f32 = ${n.minimumGain.toExponential()};
${n.reverseOffsets&&n.packedReverse?`const REVERSE_OFFSETS_OFFSET: u32 = ${i(n.reverseOffsets)}u;
const REVERSE_CAPACITY: u32 = ${n.packedReverse.length}u;`:``}
${K(a)}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

fn isFiniteVector(value: vec4<f32>) -> bool {
  let magnitude = bitcast<vec4<u32>>(value) & vec4<u32>(0x7fffffffu);
  return all(magnitude < vec4<u32>(0x7f800000u));
}

${G(n)}

fn getCommunityWeight(vertex: u32, community: u32, direction: u32) -> f32 {
  var weight = 0.0;
  let first = getNeighborStart(vertex, direction);
  let last = getNeighborEnd(vertex, direction);
  for (var slot = first; slot < last; slot++) {
    let neighbor = getNeighbor(slot, direction);
    if (neighbor.x < VERTEX_COUNT && neighbor.x != vertex &&
        communities[COMMUNITY_OFFSET + neighbor.x] == community) {
      weight += bitcast<f32>(neighbor.y);
    }
  }
  return weight;
}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,V)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }

  let total = bitcast<f32>(atomicLoad(&control[0u]));
  if (!isFiniteValue(total) || total <= 0.0) {
    atomicOr(&control[1u], ${U}u);
    return;
  }

  let currentCommunity = communities[COMMUNITY_OFFSET + index];
  if (currentCommunity >= VERTEX_COUNT) {
    atomicOr(&control[1u], ${U}u);
    return;
  }

  let vertexStatistics = statistics[index];
  let currentStatistics = statistics[currentCommunity];
  if (!isFiniteVector(vertexStatistics) || !isFiniteVector(currentStatistics)) {
    atomicOr(&control[1u], ${U}u);
    return;
  }

  let currentOutgoingWeight = getCommunityWeight(index, currentCommunity, 0u);
  let currentIncomingWeight = ${n.directed?`getCommunityWeight(index, currentCommunity, 1u)`:`currentOutgoingWeight`};
  var bestCommunity = currentCommunity;
  var bestGain = 0.0;

  for (var direction = 0u; direction < DIRECTION_COUNT; direction++) {
    let first = getNeighborStart(index, direction);
    let last = getNeighborEnd(index, direction);
    for (var slot = first; slot < last; slot++) {
      let neighborVertex = getNeighbor(slot, direction).x;
      if (neighborVertex >= VERTEX_COUNT || neighborVertex == index) { continue; }
      let candidateCommunity = communities[COMMUNITY_OFFSET + neighborVertex];
      if (candidateCommunity >= VERTEX_COUNT || candidateCommunity == currentCommunity) {
        continue;
      }

      let candidateStatistics = statistics[candidateCommunity];
      if (!isFiniteVector(candidateStatistics)) {
        atomicOr(&control[1u], ${U}u);
        return;
      }

      let outgoingWeight = getCommunityWeight(index, candidateCommunity, 0u);
      let incomingWeight = ${n.directed?`getCommunityWeight(index, candidateCommunity, 1u)`:`outgoingWeight`};
      let observedGain =
        ((outgoingWeight - currentOutgoingWeight) +
         (incomingWeight - currentIncomingWeight)) / total;
      let expectedGain = RESOLUTION * (
        (vertexStatistics.y / total) *
          ((candidateStatistics.z - (currentStatistics.z - vertexStatistics.x)) / total) +
        (vertexStatistics.x / total) *
          ((candidateStatistics.w - (currentStatistics.w - vertexStatistics.y)) / total)
      );
      let gain = observedGain - expectedGain;
      if (!isFiniteValue(gain)) {
        atomicOr(&control[1u], ${U}u);
        return;
      }
      if (gain > MINIMUM_GAIN &&
          (gain > bestGain || (gain == bestGain && candidateCommunity < bestCommunity))) {
        bestGain = gain;
        bestCommunity = candidateCommunity;
      }
    }
  }

  let availableCommunity = atomicLoad(&control[4u]);
  if (availableCommunity < VERTEX_COUNT && availableCommunity != currentCommunity) {
    let availableStatistics = statistics[availableCommunity];
    if (!isFiniteVector(availableStatistics)) {
      atomicOr(&control[1u], ${U}u);
      return;
    }
    let outgoingWeight = getCommunityWeight(index, availableCommunity, 0u);
    let incomingWeight = ${n.directed?`getCommunityWeight(index, availableCommunity, 1u)`:`outgoingWeight`};
    let observedGain =
      ((outgoingWeight - currentOutgoingWeight) +
       (incomingWeight - currentIncomingWeight)) / total;
    let expectedGain = RESOLUTION * (
      (vertexStatistics.y / total) *
        ((availableStatistics.z - (currentStatistics.z - vertexStatistics.x)) / total) +
      (vertexStatistics.x / total) *
        ((availableStatistics.w - (currentStatistics.w - vertexStatistics.y)) / total)
    );
    let gain = observedGain - expectedGain;
    if (!isFiniteValue(gain)) {
      atomicOr(&control[1u], ${U}u);
      return;
    }
    if (gain > MINIMUM_GAIN &&
        (gain > bestGain || (gain == bestGain && availableCommunity < bestCommunity))) {
      bestGain = gain;
      bestCommunity = availableCommunity;
    }
  }

  candidates[index] = vec2<u32>(bestCommunity, bitcast<u32>(bestGain));
  if (bestGain > MINIMUM_GAIN) {
    atomicMax(&control[2u], bitcast<u32>(bestGain));
  }
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-evaluate-candidates-by-rescan`,source:s,bindings:a,dispatchLayout:o})}function we(e,t){let{state:n}=t,i={candidates:{view:n.candidates,usage:`storage-read`},control:{view:n.control,usage:`storage-read-write`,atomic:!0}},a=J(n,n.vertexCount),o=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
${K(i)}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(a,V)}
  if (index >= VERTEX_COUNT || atomicLoad(&control[1u]) != 0u) { return; }
  let maximumGain = atomicLoad(&control[2u]);
  if (maximumGain != 0u && candidates[index].y == maximumGain) {
    atomicMin(&control[3u], index);
  }
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-select-winner`,source:o,bindings:i,dispatchLayout:a})}function Te(e,t){let{state:n}=t,r={communities:{view:n.output,usage:`storage-read-write`},candidates:{view:n.candidates,usage:`storage-read`},control:{view:n.control,usage:`storage-read-write`,atomic:!0},...n.converged?{converged:{view:n.converged,usage:`storage-write`}}:{}},a=n.converged?`converged[${i(n.converged)}u] = 1u;`:``,o=n.converged?`converged[${i(n.converged)}u] = 0u;`:``,s=`
${K(r)}

@compute @workgroup_size(1)
fn main() {
  if (atomicLoad(&control[1u]) != 0u) { return; }
  let winner = atomicLoad(&control[3u]);
  if (atomicLoad(&control[2u]) == 0u || winner == ${H}u) {
    atomicOr(&control[1u], ${de}u);
    ${a}
    return;
  }
  communities[${i(n.output)}u + winner] = candidates[winner].x;
  ${o}
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:1,invocationCount:1,id:`${n.id}-iteration-${t.iteration}-apply-winner`,source:s,bindings:r,dispatchLayout:{x:1,y:1,z:1}})}function Ee(e,t){let n={communities:{view:t.output,usage:`storage-read-write`},control:{view:t.control,usage:`storage-read-write`,atomic:!0},...t.converged?{converged:{view:t.converged,usage:`storage-read-write`}}:{}},a=t.converged?`if (index == 0u && VERTEX_COUNT > 0u) {
    converged[${i(t.converged)}u] = 0u;
  }`:``,o=J(t,Math.max(t.vertexCount,1)),s=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
${K(n)}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

@compute @workgroup_size(${V})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${r(o,V)}
  if (index >= max(VERTEX_COUNT, 1u)) { return; }
  let volume = bitcast<f32>(atomicLoad(&control[0u]));
  let invalid = (atomicLoad(&control[1u]) & ${U}u) != 0u ||
    !isFiniteValue(volume) || volume <= 0.0;
  if (!invalid) { return; }
  if (index < VERTEX_COUNT) {
    communities[${i(t.output)}u + index] = ${H}u;
  }
  ${a}
}`;q(e,{operation:`GPUGraphModularityOptimization`,workgroupSize:V,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-finalize-failures`,source:s,bindings:n,dispatchLayout:o})}function G(e){let t=e.reverseOffsets&&e.packedReverse;return`
fn getNeighborStart(vertex: u32, direction: u32) -> u32 {
  ${t?`if (direction == 1u) {
    return min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex], REVERSE_CAPACITY);
  }`:``}
  return min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex], FORWARD_CAPACITY);
}

fn getNeighborEnd(vertex: u32, direction: u32) -> u32 {
  ${t?`if (direction == 1u) {
    return min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex + 1u], REVERSE_CAPACITY);
  }`:``}
  return min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex + 1u], FORWARD_CAPACITY);
}

fn getNeighbor(slot: u32, direction: u32) -> vec2<u32> {
  ${t?`if (direction == 1u) {
    return packedReverse[slot];
  }`:``}
  return packedForward[slot];
}`}function De(){return`
fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var previousBits = atomicLoad(destination);
  loop {
    let nextBits = bitcast<u32>(bitcast<f32>(previousBits) + value);
    let exchange = atomicCompareExchangeWeak(destination, previousBits, nextBits);
    if (exchange.exchanged) { return; }
    previousBits = exchange.old_value;
  }
}`}function K(e){return Object.entries(e).map(([e,t],n)=>{let r=t.usage===`storage-read`?`read`:`read_write`,i=`u32`;return t.atomic?i=`atomic<u32>`:t.view.format===`float32`?i=`f32`:t.view.format===`float32x2`?i=`vec2<f32>`:t.view.format===`uint32x2`?i=`vec2<u32>`:t.view.format===`float32x4`&&(i=`vec4<f32>`),`@group(0) @binding(${n}) var<storage, ${r}> ${e}: array<${i}>;`}).join(`
`)}function q(e,n){let r=e.recording?d(e,{operation:n.operation,source:n.source,bindings:Object.entries(n.bindings).map(([e,t])=>({name:e,...t})),workgroupSize:n.workgroupSize,dispatch:n.dispatchLayout,invocationCount:n.invocationCount,datum:n.datum}):void 0;e.addComputePass({...r&&{annotation:r},id:n.id,resources:Object.values(n.bindings).map(({view:e,usage:t})=>({buffer:e,usage:t})),compile:({device:e})=>{let r=new s(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,r]of Object.entries(n.bindings))a[e]=t(r.view,i);r.setBindings(a),r.dispatch(e,n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z)},destroy:()=>r.destroy()}}})}function J(e,t){return ke(t,e.maxComputeWorkgroupsPerDimension)}function Oe(e,t,n){if(!Number.isSafeInteger(e)||e<0||!Number.isSafeInteger(t)||t<0||!Number.isSafeInteger(n)||n<0)throw Error(`Community aggregation sizes must be non-negative safe integers`);let r=2*(t+n)+e;if(!(!Number.isSafeInteger(r)||r>4294967295))return r}function ke(e,t){return o(`GPUGraphModularityOptimization`,e,V,t)}var Ae=32,je=1024,Y=4,Me=class{id;topology;output;modularity;initialCommunities;resolution;iterations;minimumGain;converged;valid;constructor(e){if(this.id=e.id??`gpu-graph-modularity-optimization`,this.topology=e.topology,this.output=e.output,this.modularity=e.modularity,this.initialCommunities=e.initialCommunities,this.resolution=e.resolution??1,this.iterations=e.iterations??Ae,this.minimumGain=e.minimumGain??0,this.converged=e.converged,this.valid=e.valid,this.topology.graph.directed&&!this.topology.reverse)throw Error(`${this.id} directed modularity optimization requires reverse adjacency`);if(!Number.isFinite(this.resolution)||this.resolution<0||!Number.isFinite(Math.fround(this.resolution)))throw Error(`${this.id} resolution must be a non-negative finite float32`);if(!Number.isFinite(this.minimumGain)||this.minimumGain<0||!Number.isFinite(Math.fround(this.minimumGain)))throw Error(`${this.id} minimumGain must be a non-negative finite float32`);if(!Number.isSafeInteger(this.iterations)||this.iterations<0||this.iterations>je)throw Error(`${this.id} iterations must be a safe integer between zero and 1024`);let t=this.topology.graph.vertexCount;X(this.output,`uint32`,t,`${this.id} output`),X(this.modularity,`float32`,1,`${this.id} modularity`),this.initialCommunities&&X(this.initialCommunities,`uint32`,t,`${this.id} initialCommunities`),this.converged&&X(this.converged,`uint32`,1,`${this.id} converged`),this.valid&&X(this.valid,`uint32`,1,`${this.id} valid`),Z(this)}addToGraph(e){p(e,`GPUGraphModularityOptimization`,this.id,()=>{fe(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function X(e,t,n,r){if(e.format!==t||e.data.length!==1||e.length!==n||e.stride!==1||e.byteStride!==Y||e.rowByteLength!==Y||e.valueLength!==e.length||e.bufferLayout)throw Error(`${r} must contain exactly ${n} packed ${t} rows in one chunk`);let i=e.data[0];if(i.format!==t||i.length!==n||i.stride!==1||i.byteStride!==Y||i.rowByteLength!==Y||i.valueLength!==i.length||!Number.isSafeInteger(i.byteOffset)||i.byteOffset<0||i.byteOffset%Y!==0)throw Error(`${r} must contain one packed, uint32-aligned ${t} chunk`)}function Z(e){let{topology:t}=e,n=[t.graph.sourceVertices,t.graph.targetVertices,...t.graph.edgeWeights?[t.graph.edgeWeights]:[],...t.graph.edgeIds?[t.graph.edgeIds]:[],...Q(t.forward),...t.reverse?Q(t.reverse):[],t.invalidEdgeCount,...e.initialCommunities?[e.initialCommunities]:[]],r=new Set;for(let e of n)for(let t of e.data)r.add($(t));let i=[{name:`output`,vector:e.output},{name:`modularity`,vector:e.modularity},...e.converged?[{name:`converged`,vector:e.converged}]:[],...e.valid?[{name:`valid`,vector:e.valid}]:[]];for(let{name:t,vector:n}of i){let i=$(n.data[0]);if(r.has(i))throw Error(`${e.id} ${t} must use a distinct physical buffer allocation`);r.add(i)}}function Q(e){return[e.offsets,e.neighbors,e.edgeIds,...e.edgeWeights?[e.edgeWeights]:[],e.count,e.overflow]}function $(t){return t.buffer instanceof e?t.buffer.buffer:t.buffer}export{R as n,re as r,Me as t};