import{a as e}from"./arithmetic-expression-Kknn3j_i.js";import{D as t,S as n,k as r,x as i}from"./resources-CwvhEOBe.js";import{u as a}from"./interleave-brZZCO-f.js";import{a as o,c as s,i as c,l,n as u,o as d,r as f,s as p,u as m}from"./gpu-graph-topology-view-DhuT3ULg.js";var h=256,g=4294967295;function _(e,t,n){if(e.topology.graph.vertexCount===0&&!e.converged)return;let r={id:e.id,vertexCount:e.topology.graph.vertexCount,offsets:u(t,`${e.id}-offsets`,e.topology.forward.offsets),neighbors:u(t,`${e.id}-neighbors`,e.topology.forward.neighbors),overflow:f(t,`${e.id}-overflow`,e.topology.forward.overflow,`GPUGraphConnectedComponents`),output:u(t,`${e.id}-output`,e.output),...e.converged?{converged:u(t,`${e.id}-converged`,e.converged)}:{},maxComputeWorkgroupsPerDimension:n};if(o(e.topology,[e.output,e.converged])&&d(e.id,[r.offsets,r.neighbors,r.overflow],[{name:`output`,view:r.output},...r.converged?[{name:`converged`,view:r.converged}]:[]]),v(t,r),r.vertexCount!==0)for(let n=0;n<e.iterations;n++)l(t,{loopId:`${r.id}-iterations`,iteration:n,maximumIterations:e.iterations,gated:!1},()=>{r.converged&&y(t,{state:r,iteration:n}),b(t,{state:r,iteration:n}),x(t,{state:r,iteration:n})})}function v(e,t){let i={output:{view:t.output,usage:`storage-write`,atomic:!0},overflow:{view:t.overflow,usage:`storage-read`},...t.converged?{converged:{view:t.converged,usage:`storage-write`,atomic:!0}}:{}},a=t.converged?`const CONVERGED_OFFSET: u32 = ${r(t.converged)}u;`:``,o=t.converged?`if (index == 0u) {
    let emptyAndValid = VERTEX_COUNT == 0u && !hasOverflow;
    atomicStore(&converged[CONVERGED_OFFSET], select(0u, 1u, emptyAndValid));
  }`:``,s=w(Math.max(t.vertexCount,1),t.maxComputeWorkgroupsPerDimension),c=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
const OVERFLOW_OFFSET: u32 = ${r(t.overflow)}u;
${a}
${S(i)}

@compute @workgroup_size(${h})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(s,h)}
  let hasOverflow = overflow[OVERFLOW_OFFSET] != 0u;
  if (index < VERTEX_COUNT) {
    let component = select(index, ${g}u, hasOverflow);
    atomicStore(&output[OUTPUT_OFFSET + index], component);
  }
  ${o}
}`;C(e,{operation:`GPUGraphConnectedComponents`,workgroupSize:h,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-initialize`,source:c,bindings:i,dispatchLayout:s})}function y(e,t){let{state:i}=t,a=i.converged,o={overflow:{view:i.overflow,usage:`storage-read`},converged:{view:a,usage:`storage-write`,atomic:!0}},s=w(1,i.maxComputeWorkgroupsPerDimension),c=`
const OVERFLOW_OFFSET: u32 = ${r(i.overflow)}u;
const CONVERGED_OFFSET: u32 = ${r(a)}u;
${S(o)}

@compute @workgroup_size(${h})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(s,h)}
  if (index != 0u) { return; }
  atomicStore(&converged[CONVERGED_OFFSET], select(1u, 0u, overflow[OVERFLOW_OFFSET] != 0u));
}`;C(e,{operation:`GPUGraphConnectedComponents`,workgroupSize:h,invocationCount:1,id:`${i.id}-iteration-${t.iteration}-reset`,source:c,bindings:o,dispatchLayout:s})}function b(e,t){let{state:i}=t,a={offsets:{view:i.offsets,usage:`storage-read`},neighbors:{view:i.neighbors,usage:`storage-read`},output:{view:i.output,usage:`storage-read-write`,atomic:!0},overflow:{view:i.overflow,usage:`storage-read`},...i.converged?{converged:{view:i.converged,usage:`storage-read-write`,atomic:!0}}:{}},o=i.converged?`const CONVERGED_OFFSET: u32 = ${r(i.converged)}u;`:``,s=i.converged?`if (previous > lower) { atomicStore(&converged[CONVERGED_OFFSET], 0u); }`:``,c=w(i.vertexCount,i.maxComputeWorkgroupsPerDimension),l=`
const VERTEX_COUNT: u32 = ${i.vertexCount}u;
const CAPACITY: u32 = ${i.neighbors.length}u;
const OFFSETS_OFFSET: u32 = ${r(i.offsets)}u;
const NEIGHBORS_OFFSET: u32 = ${r(i.neighbors)}u;
const OUTPUT_OFFSET: u32 = ${r(i.output)}u;
const OVERFLOW_OFFSET: u32 = ${r(i.overflow)}u;
${o}
${S(a)}

@compute @workgroup_size(${h})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(c,h)}
  if (index >= VERTEX_COUNT || overflow[OVERFLOW_OFFSET] != 0u) { return; }
  let first = min(offsets[OFFSETS_OFFSET + index], CAPACITY);
  let last = min(offsets[OFFSETS_OFFSET + index + 1u], CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT) { continue; }
    let sourceComponent = atomicLoad(&output[OUTPUT_OFFSET + index]);
    let neighborComponent = atomicLoad(&output[OUTPUT_OFFSET + neighbor]);
    let lower = min(sourceComponent, neighborComponent);
    let higher = max(sourceComponent, neighborComponent);
    if (higher == lower) { continue; }
    let previous = atomicMin(&output[OUTPUT_OFFSET + higher], lower);
    ${s}
  }
}`;C(e,{operation:`GPUGraphConnectedComponents`,workgroupSize:h,datum:{keySpace:`vertex`},invocationCount:i.vertexCount,id:`${i.id}-iteration-${t.iteration}-hook`,source:l,bindings:a,dispatchLayout:c})}function x(e,t){let{state:i}=t,a={output:{view:i.output,usage:`storage-read-write`,atomic:!0},overflow:{view:i.overflow,usage:`storage-read`},...i.converged?{converged:{view:i.converged,usage:`storage-read-write`,atomic:!0}}:{}},o=i.converged?`const CONVERGED_OFFSET: u32 = ${r(i.converged)}u;`:``,s=i.converged?`if (previous > parentComponent) {
    atomicStore(&converged[CONVERGED_OFFSET], 0u);
  }`:``,c=w(i.vertexCount,i.maxComputeWorkgroupsPerDimension),l=`
const VERTEX_COUNT: u32 = ${i.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${r(i.output)}u;
const OVERFLOW_OFFSET: u32 = ${r(i.overflow)}u;
${o}
${S(a)}

@compute @workgroup_size(${h})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(c,h)}
  if (index >= VERTEX_COUNT || overflow[OVERFLOW_OFFSET] != 0u) { return; }
  let component = atomicLoad(&output[OUTPUT_OFFSET + index]);
  let parentComponent = atomicLoad(&output[OUTPUT_OFFSET + component]);
  if (parentComponent >= component) { return; }
  let previous = atomicMin(&output[OUTPUT_OFFSET + index], parentComponent);
  ${s}
}`;C(e,{operation:`GPUGraphConnectedComponents`,workgroupSize:h,datum:{keySpace:`vertex`},invocationCount:i.vertexCount,id:`${i.id}-iteration-${t.iteration}-jump`,source:l,bindings:a,dispatchLayout:c})}function S(e){return Object.entries(e).map(([e,t],n)=>`@group(0) @binding(${n}) var<storage, ${t.usage===`storage-read`?`read`:`read_write`}> ${e}: array<${t.atomic?`atomic<u32>`:`u32`}>;`).join(`
`)}function C(e,n){let r=e.recording?s(e,{operation:n.operation,source:n.source,bindings:Object.entries(n.bindings).map(([e,t])=>({name:e,...t})),workgroupSize:n.workgroupSize,dispatch:n.dispatchLayout,invocationCount:n.invocationCount,datum:n.datum}):void 0;e.addComputePass({...r&&{annotation:r},id:n.id,resources:Object.values(n.bindings).map(({view:e,usage:t})=>({buffer:e,usage:t})),compile:({device:e})=>{let r=new a(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,r]of Object.entries(n.bindings))a[e]=t(r.view,i);r.setBindings(a),r.dispatch(e,n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z)},destroy:()=>r.destroy()}}})}function w(e,t){return i(`GPUGraphConnectedComponents`,e,h,t)}var T=32,E=1024,D=4,O=class{id;topology;output;iterations;converged;constructor(e){if(this.id=e.id??`gpu-graph-connected-components`,this.topology=e.topology,this.output=e.output,this.iterations=e.iterations??T,this.converged=e.converged,!Number.isSafeInteger(this.iterations)||this.iterations<1||this.iterations>E)throw Error(`${this.id} iterations must be a safe integer between one and 1024`);k(this.output,this.topology.graph.vertexCount,`${this.id} output`),this.converged&&k(this.converged,1,`${this.id} converged`),o(this.topology,[this.output,this.converged])||j(this)}addToGraph(e){m(e,`GPUGraphConnectedComponents`,this.id,()=>{_(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function k(e,t,n){c(e)?p(e,`uint32`,t,n):A(e,t,n)}function A(e,t,n){if(e.data.length!==1||e.format!==`uint32`||e.stride!==1||e.byteStride!==D||e.rowByteLength!==D||e.valueLength!==e.length||e.bufferLayout)throw Error(`${n} must contain exactly one packed uint32 chunk`);if(e.length!==t)throw Error(`${n} must contain exactly ${t} uint32 rows`);let r=e.data[0];if(r.format!==`uint32`||r.length!==t||r.stride!==1||r.byteStride!==D||r.rowByteLength!==D||r.valueLength!==r.length||!Number.isSafeInteger(r.byteOffset)||r.byteOffset<0||r.byteOffset%D!==0)throw Error(`${n} must contain one packed, uint32-aligned chunk`)}function j(e){let t=e.topology,n=[t.graph.sourceVertices,t.graph.targetVertices,...t.graph.edgeWeights?[t.graph.edgeWeights]:[],...t.graph.edgeIds?[t.graph.edgeIds]:[],...M(t.forward),...t.reverse?M(t.reverse):[],t.invalidEdgeCount],r=new Set;for(let e of n)for(let t of e.data)r.add(N(t));let i=[{name:`output`,vector:e.output},...e.converged?[{name:`converged`,vector:e.converged}]:[]];for(let{name:t,vector:n}of i){let i=N(n.data[0]);if(r.has(i))throw Error(`${e.id} ${t} must use a distinct physical buffer allocation`);r.add(i)}}function M(e){return[e.offsets,e.neighbors,e.edgeIds,...e.edgeWeights?[e.edgeWeights]:[],e.count,e.overflow]}function N(t){return t.buffer instanceof e?t.buffer.buffer:t.buffer}export{O as t};