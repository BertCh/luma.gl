import{a as e}from"./arithmetic-expression-Kknn3j_i.js";import{D as t,S as n,k as r,w as i,x as a}from"./resources-CwvhEOBe.js";import{u as o}from"./interleave-brZZCO-f.js";import{a as s,c,i as l,l as u,n as d,o as f,r as p,s as m,u as h}from"./gpu-graph-topology-view-DhuT3ULg.js";var g=256,_=4294967295;function v(e,t,n){let r=e.topology.graph.vertexCount;if(r===0&&!e.converged)return;let a=e.topology.graph.directed?e.topology.reverse:void 0,o={id:e.id,vertexCount:r,forwardOffsets:d(t,`${e.id}-forward-offsets`,e.topology.forward.offsets),forwardNeighbors:d(t,`${e.id}-forward-neighbors`,e.topology.forward.neighbors),forwardOverflow:p(t,`${e.id}-forward-overflow`,e.topology.forward.overflow,`GPUGraphLabelPropagation`),...a?{reverseOffsets:d(t,`${e.id}-reverse-offsets`,a.offsets),reverseNeighbors:d(t,`${e.id}-reverse-neighbors`,a.neighbors),reverseOverflow:p(t,`${e.id}-reverse-overflow`,a.overflow,`GPUGraphLabelPropagation`)}:{},output:d(t,`${e.id}-output`,e.output),...e.converged?{converged:d(t,`${e.id}-converged`,e.converged)}:{},...r>0?{scratch:i(t,`${e.id}-next-labels`,`uint32`,r)}:{},maxComputeWorkgroupsPerDimension:n};if(s(e.topology,[e.output,e.converged])&&f(e.id,[o.forwardOffsets,o.forwardNeighbors,o.forwardOverflow,...o.reverseOffsets?[o.reverseOffsets,o.reverseNeighbors,o.reverseOverflow]:[]],[{name:`output`,view:o.output},...o.converged?[{name:`converged`,view:o.converged}]:[]]),y(t,o),r!==0)for(let n=0;n<e.iterations;n++)u(t,{loopId:`${o.id}-iterations`,iteration:n,maximumIterations:e.iterations,gated:!1},()=>{o.converged&&b(t,{state:o,iteration:n}),x(t,{state:o,iteration:n}),S(t,{state:o,iteration:n})})}function y(e,t){let i={output:{view:t.output,usage:`storage-write`},forwardOverflow:{view:t.forwardOverflow,usage:`storage-read`},...t.reverseOverflow?{reverseOverflow:{view:t.reverseOverflow,usage:`storage-read`}}:{},...t.converged?{converged:{view:t.converged,usage:`storage-write`,atomic:!0}}:{}},a=t.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${r(t.reverseOverflow)}u;`:``,o=t.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,s=t.converged?`const CONVERGED_OFFSET: u32 = ${r(t.converged)}u;`:``,c=t.converged?`if (index == 0u) {
    atomicStore(&converged[CONVERGED_OFFSET], select(0u, 1u, VERTEX_COUNT == 0u && !hasOverflow));
  }`:``,l=T(t,Math.max(t.vertexCount,1)),u=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${r(t.forwardOverflow)}u;
${a}
${s}
${C(i)}

@compute @workgroup_size(${g})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(l,g)}
  let hasOverflow = forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${o};
  if (index < VERTEX_COUNT) {
    output[OUTPUT_OFFSET + index] = select(index, ${_}u, hasOverflow);
  }
  ${c}
}`;w(e,{operation:`GPUGraphLabelPropagation`,workgroupSize:g,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-initialize`,source:u,bindings:i,dispatchLayout:l})}function b(e,t){let{state:i}=t,a=i.converged,o={forwardOverflow:{view:i.forwardOverflow,usage:`storage-read`},...i.reverseOverflow?{reverseOverflow:{view:i.reverseOverflow,usage:`storage-read`}}:{},converged:{view:a,usage:`storage-write`,atomic:!0}},s=i.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${r(i.reverseOverflow)}u;`:``,c=i.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,l=T(i,1),u=`
const FORWARD_OVERFLOW_OFFSET: u32 = ${r(i.forwardOverflow)}u;
const CONVERGED_OFFSET: u32 = ${r(a)}u;
${s}
${C(o)}

@compute @workgroup_size(${g})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(l,g)}
  if (index != 0u) { return; }
  let hasOverflow = forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${c};
  atomicStore(&converged[CONVERGED_OFFSET], select(1u, 0u, hasOverflow));
}`;w(e,{operation:`GPUGraphLabelPropagation`,workgroupSize:g,invocationCount:1,id:`${i.id}-iteration-${t.iteration}-reset`,source:u,bindings:o,dispatchLayout:l})}function x(e,t){let{state:i}=t,a=i.scratch,o={output:{view:i.output,usage:`storage-read`},scratch:{view:a,usage:`storage-write`},forwardOffsets:{view:i.forwardOffsets,usage:`storage-read`},forwardNeighbors:{view:i.forwardNeighbors,usage:`storage-read`},forwardOverflow:{view:i.forwardOverflow,usage:`storage-read`},...i.reverseOffsets&&i.reverseNeighbors&&i.reverseOverflow?{reverseOffsets:{view:i.reverseOffsets,usage:`storage-read`},reverseNeighbors:{view:i.reverseNeighbors,usage:`storage-read`},reverseOverflow:{view:i.reverseOverflow,usage:`storage-read`}}:{}},s=!!(i.reverseOffsets&&i.reverseNeighbors&&i.reverseOverflow),c=s?`const REVERSE_CAPACITY: u32 = ${i.reverseNeighbors.length}u;
const REVERSE_OFFSETS_OFFSET: u32 = ${r(i.reverseOffsets)}u;
const REVERSE_NEIGHBORS_OFFSET: u32 = ${r(i.reverseNeighbors)}u;
const REVERSE_OVERFLOW_OFFSET: u32 = ${r(i.reverseOverflow)}u;`:``,l=s?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,u=s?`let reverseFirst = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex], REVERSE_CAPACITY);
  let reverseLast = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex + 1u], REVERSE_CAPACITY);
  for (var slot = reverseFirst; slot < reverseLast; slot++) {
    let neighbor = reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + slot];
    if (neighbor < VERTEX_COUNT && neighbor != vertex && output[OUTPUT_OFFSET + neighbor] == candidate) {
      votes++;
    }
  }`:``,d=s?`let reverseFirst = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + index], REVERSE_CAPACITY);
  let reverseLast = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + index + 1u], REVERSE_CAPACITY);
  for (var slot = reverseFirst; slot < reverseLast; slot++) {
    let neighbor = reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    let candidate = output[OUTPUT_OFFSET + neighbor];
    let votes = countCandidateVotes(index, candidate);
    if (votes > selectedVotes || (votes == selectedVotes && candidate < selectedLabel)) {
      selectedLabel = candidate;
      selectedVotes = votes;
    }
  }`:``,f=T(i,i.vertexCount),p=`
const VERTEX_COUNT: u32 = ${i.vertexCount}u;
const FORWARD_CAPACITY: u32 = ${i.forwardNeighbors.length}u;
const OUTPUT_OFFSET: u32 = ${r(i.output)}u;
const SCRATCH_OFFSET: u32 = ${r(a)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${r(i.forwardOffsets)}u;
const FORWARD_NEIGHBORS_OFFSET: u32 = ${r(i.forwardNeighbors)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${r(i.forwardOverflow)}u;
${c}
${C(o)}

fn countCandidateVotes(vertex: u32, candidate: u32) -> u32 {
  var votes = select(0u, 1u, output[OUTPUT_OFFSET + vertex] == candidate);
  let first = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex], FORWARD_CAPACITY);
  let last = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex + 1u], FORWARD_CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + slot];
    if (neighbor < VERTEX_COUNT && neighbor != vertex && output[OUTPUT_OFFSET + neighbor] == candidate) {
      votes++;
    }
  }
  ${u}
  return votes;
}

@compute @workgroup_size(${g})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(f,g)}
  if (index >= VERTEX_COUNT) { return; }
  if (forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${l}) {
    scratch[SCRATCH_OFFSET + index] = ${_}u;
    return;
  }

  var selectedLabel = output[OUTPUT_OFFSET + index];
  var selectedVotes = countCandidateVotes(index, selectedLabel);
  let first = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + index], FORWARD_CAPACITY);
  let last = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + index + 1u], FORWARD_CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    let candidate = output[OUTPUT_OFFSET + neighbor];
    let votes = countCandidateVotes(index, candidate);
    if (votes > selectedVotes || (votes == selectedVotes && candidate < selectedLabel)) {
      selectedLabel = candidate;
      selectedVotes = votes;
    }
  }
  ${d}
  scratch[SCRATCH_OFFSET + index] = selectedLabel;
}`;w(e,{operation:`GPUGraphLabelPropagation`,workgroupSize:g,datum:{keySpace:`vertex`},invocationCount:i.vertexCount,id:`${i.id}-iteration-${t.iteration}-vote`,source:p,bindings:o,dispatchLayout:f})}function S(e,t){let{state:i}=t,a=i.scratch,o={output:{view:i.output,usage:`storage-read-write`},scratch:{view:a,usage:`storage-read`},...i.converged?{converged:{view:i.converged,usage:`storage-read-write`,atomic:!0}}:{}},s=i.converged?`const CONVERGED_OFFSET: u32 = ${r(i.converged)}u;`:``,c=i.converged?`if (nextLabel != previousLabel) { atomicStore(&converged[CONVERGED_OFFSET], 0u); }`:``,l=T(i,i.vertexCount),u=`
const VERTEX_COUNT: u32 = ${i.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${r(i.output)}u;
const SCRATCH_OFFSET: u32 = ${r(a)}u;
${s}
${C(o)}

@compute @workgroup_size(${g})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(l,g)}
  if (index >= VERTEX_COUNT) { return; }
  let previousLabel = output[OUTPUT_OFFSET + index];
  let nextLabel = scratch[SCRATCH_OFFSET + index];
  output[OUTPUT_OFFSET + index] = nextLabel;
  ${c}
}`;w(e,{operation:`GPUGraphLabelPropagation`,workgroupSize:g,datum:{keySpace:`vertex`},invocationCount:i.vertexCount,id:`${i.id}-iteration-${t.iteration}-publish`,source:u,bindings:o,dispatchLayout:l})}function C(e){return Object.entries(e).map(([e,t],n)=>`@group(0) @binding(${n}) var<storage, ${t.usage===`storage-read`?`read`:`read_write`}> ${e}: array<${t.atomic?`atomic<u32>`:`u32`}>;`).join(`
`)}function w(e,n){let r=e.recording?c(e,{operation:n.operation,source:n.source,bindings:Object.entries(n.bindings).map(([e,t])=>({name:e,...t})),workgroupSize:n.workgroupSize,dispatch:n.dispatchLayout,invocationCount:n.invocationCount,datum:n.datum}):void 0;e.addComputePass({...r&&{annotation:r},id:n.id,resources:Object.values(n.bindings).map(({view:e,usage:t})=>({buffer:e,usage:t})),compile:({device:e})=>{let r=new o(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,r]of Object.entries(n.bindings))a[e]=t(r.view,i);r.setBindings(a),r.dispatch(e,n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z)},destroy:()=>r.destroy()}}})}function T(e,t){return E(t,e.maxComputeWorkgroupsPerDimension)}function E(e,t){return a(`GPUGraphLabelPropagation`,e,g,t)}var D=32,O=1024,k=4,A=class{id;topology;output;iterations;converged;constructor(e){if(this.id=e.id??`gpu-graph-label-propagation`,this.topology=e.topology,this.output=e.output,this.iterations=e.iterations??D,this.converged=e.converged,!Number.isSafeInteger(this.iterations)||this.iterations<1||this.iterations>O)throw Error(`${this.id} iterations must be a safe integer between one and 1024`);if(this.topology.graph.directed&&!this.topology.reverse)throw Error(`${this.id} directed weak-neighbor votes require reverse adjacency`);j(this.output,this.topology.graph.vertexCount,`${this.id} output`),this.converged&&j(this.converged,1,`${this.id} converged`),s(this.topology,[this.output,this.converged])||N(this)}addToGraph(e){h(e,`GPUGraphLabelPropagation`,this.id,()=>{v(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function j(e,t,n){l(e)?m(e,`uint32`,t,n):M(e,t,n)}function M(e,t,n){if(e.data.length!==1||e.format!==`uint32`||e.stride!==1||e.byteStride!==k||e.rowByteLength!==k||e.valueLength!==e.length||e.bufferLayout)throw Error(`${n} must contain exactly one packed uint32 chunk`);if(e.length!==t)throw Error(`${n} must contain exactly ${t} uint32 rows`);let r=e.data[0];if(r.format!==`uint32`||r.length!==t||r.stride!==1||r.byteStride!==k||r.rowByteLength!==k||r.valueLength!==r.length||!Number.isSafeInteger(r.byteOffset)||r.byteOffset<0||r.byteOffset%k!==0)throw Error(`${n} must contain one packed, uint32-aligned chunk`)}function N(e){let t=e.topology,n=[t.graph.sourceVertices,t.graph.targetVertices,...t.graph.edgeWeights?[t.graph.edgeWeights]:[],...t.graph.edgeIds?[t.graph.edgeIds]:[],...P(t.forward),...t.reverse?P(t.reverse):[],t.invalidEdgeCount],r=new Set;for(let e of n)for(let t of e.data)r.add(F(t));let i=[{name:`output`,vector:e.output},...e.converged?[{name:`converged`,vector:e.converged}]:[]];for(let{name:t,vector:n}of i){let i=F(n.data[0]);if(r.has(i))throw Error(`${e.id} ${t} must use a distinct physical buffer allocation`);r.add(i)}}function P(e){return[e.offsets,e.neighbors,e.edgeIds,...e.edgeWeights?[e.edgeWeights]:[],e.count,e.overflow]}function F(t){return t.buffer instanceof e?t.buffer.buffer:t.buffer}export{A as t};