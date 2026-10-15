import{a as e}from"./arithmetic-expression-Kknn3j_i.js";import{D as t,S as n,k as r,w as i,x as a}from"./resources-CwvhEOBe.js";import{u as o}from"./interleave-brZZCO-f.js";import{n as s,t as c}from"./gpu-reduction-BD8l40jM.js";import{a as l,c as u,i as d,l as f,n as p,o as m,r as h,s as g,t as _,u as v}from"./gpu-graph-topology-view-DhuT3ULg.js";var y=256;function ee(e,t,i){if(e.topology.graph.vertexCount===0)return;let a=e.direction===`incoming`&&e.topology.graph.directed?e.topology.reverse:e.topology.forward,o=p(t,`${e.id}-offsets`,a.offsets),s=p(t,`${e.id}-output`,e.output);l(e.topology,[e.output])&&m(e.id,[o],[{name:`output`,view:s}]);let c=ne(e.topology.graph.vertexCount,i),u=`
const VERTEX_COUNT: u32 = ${e.topology.graph.vertexCount}u;
const OFFSETS_OFFSET: u32 = ${r(o)}u;
const OUTPUT_OFFSET: u32 = ${r(s)}u;
@group(0) @binding(0) var<storage, read> offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(${y})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(c,y)}
  if (index >= VERTEX_COUNT) { return; }
  output[OUTPUT_OFFSET + index] =
    offsets[OFFSETS_OFFSET + index + 1u] - offsets[OFFSETS_OFFSET + index];
}`;te(t,{operation:`GPUGraphDegree`,workgroupSize:y,datum:{keySpace:`vertex`},invocationCount:e.topology.graph.vertexCount,id:e.id,offsets:o,output:s,source:u,dispatchLayout:c})}function te(e,n){let r=e.recording?u(e,{operation:n.operation,source:n.source,bindings:[{name:`offsets`,view:n.offsets,usage:`storage-read`},{name:`output`,view:n.output,usage:`storage-write`}],workgroupSize:n.workgroupSize,dispatch:n.dispatchLayout,invocationCount:n.invocationCount,datum:n.datum}):void 0;e.addComputePass({...r&&{annotation:r},id:n.id,resources:[{buffer:n.offsets,usage:`storage-read`},{buffer:n.output,usage:`storage-write`}],compile:({device:e})=>{let r=new o(e,{id:n.id,source:n.source,shaderLayout:{bindings:[{name:`offsets`,type:`storage`,group:0,location:0},{name:`output`,type:`storage`,group:0,location:1}]}});return{encode:({computePass:e,getBuffer:i})=>{let a={offsets:t(n.offsets,i),output:t(n.output,i)};r.setBindings(a),r.dispatch(e,n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z)},destroy:()=>r.destroy()}}})}function ne(e,t){return a(`GPUGraphDegree`,e,y,t)}var b=4,x=class{id;topology;output;direction;constructor(e){if(this.id=e.id??`gpu-graph-degree`,this.topology=e.topology,this.output=e.output,this.direction=e.direction??`outgoing`,this.direction!==`outgoing`&&this.direction!==`incoming`)throw Error(`${this.id} direction must be outgoing or incoming`);if(this.direction===`incoming`&&this.topology.graph.directed&&!this.topology.reverse)throw Error(`${this.id} incoming directed degree requires reverse adjacency`);d(this.output)?g(this.output,`uint32`,this.topology.graph.vertexCount,`${this.id} output`):(re(this.output,this.topology.graph.vertexCount,this.id),this.topology instanceof _||ie(this.topology,this.output,this.id))}addToGraph(e){v(e,`GPUGraphDegree`,this.id,()=>{ee(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function re(e,t,n){if(e.data.length!==1||e.format!==`uint32`||e.stride!==1||e.byteStride!==b||e.rowByteLength!==b||e.valueLength!==e.length||e.bufferLayout)throw Error(`${n} output must contain one packed uint32 chunk`);if(e.length!==t)throw Error(`${n} output length must equal vertexCount`);let r=e.data[0];if(r.format!==`uint32`||r.length!==e.length||r.stride!==1||r.byteStride!==b||r.rowByteLength!==b||r.valueLength!==r.length||!Number.isSafeInteger(r.byteOffset)||r.byteOffset<0||r.byteOffset%b!==0)throw Error(`${n} output must contain one packed, uint32-aligned chunk`)}function ie(e,t,n){let r=[e.graph.sourceVertices,e.graph.targetVertices,...e.graph.edgeWeights?[e.graph.edgeWeights]:[],...e.graph.edgeIds?[e.graph.edgeIds]:[],...S(e.forward),...e.reverse?S(e.reverse):[],e.invalidEdgeCount],i=C(t.data[0]);for(let e of r)if(e.data.some(e=>C(e)===i))throw Error(`${n} output must use a distinct physical buffer allocation`)}function S(e){return[e.offsets,e.neighbors,e.edgeIds,...e.edgeWeights?[e.edgeWeights]:[],e.count,e.overflow]}function C(t){return t.buffer instanceof e?t.buffer.buffer:t.buffer}var w=256,T=4294967295;function E(e,t,n){let r=e.topology.graph,a=r.directed?e.topology.reverse:void 0,o=p(t,`${e.id}-output`,e.output),s={id:e.id,vertexCount:r.vertexCount,edgeCount:r.edgeCount,iterations:e.iterations,forwardOffsets:p(t,`${e.id}-forward-offsets`,e.topology.forward.offsets),forwardNeighbors:p(t,`${e.id}-forward-neighbors`,e.topology.forward.neighbors),forwardOverflow:h(t,`${e.id}-forward-overflow`,e.topology.forward.overflow,`GPUGraphCoreNumber`),...a?{reverseOffsets:p(t,`${e.id}-reverse-offsets`,a.offsets),reverseNeighbors:p(t,`${e.id}-reverse-neighbors`,a.neighbors),reverseOverflow:h(t,`${e.id}-reverse-overflow`,a.overflow,`GPUGraphCoreNumber`)}:{},output:o,converged:e.converged?p(t,`${e.id}-converged`,e.converged):i(t,`${e.id}-convergence-scratch`,`uint32`,1),...e.degeneracy?{degeneracy:p(t,`${e.id}-degeneracy`,e.degeneracy)}:{},...r.vertexCount>0&&e.iterations>0?{scratch:i(t,`${e.id}-next-core-numbers`,`uint32`,r.vertexCount),changed:i(t,`${e.id}-changed`,`uint32`,1)}:{},maxComputeWorkgroupsPerDimension:n},{output:u,converged:d,degeneracy:g}=e;if(l(e.topology,[u,d,g])&&m(e.id,[s.forwardOffsets,s.forwardNeighbors,s.forwardOverflow,...s.reverseOffsets?[s.reverseOffsets,s.reverseNeighbors,s.reverseOverflow]:[]],[{name:`output`,view:s.output},{name:`converged`,view:s.converged},...s.degeneracy?[{name:`degeneracy`,view:s.degeneracy}]:[]]),D(t,s),s.vertexCount>0)for(let e=0;e<s.iterations;e++)f(t,{loopId:`${s.id}-iterations`,iteration:e,maximumIterations:s.iterations,gated:!1},()=>{O(t,s,e),k(t,s,e),ae(t,s,e),oe(t,s,e)});s.degeneracy&&t.add(new c({id:`${s.id}-degeneracy`,input:s.output,output:s.degeneracy,operation:`max`}))}function D(e,t){let i={output:{view:t.output,usage:`storage-write`},forwardOffsets:{view:t.forwardOffsets,usage:`storage-read`},forwardNeighbors:{view:t.forwardNeighbors,usage:`storage-read`},forwardOverflow:{view:t.forwardOverflow,usage:`storage-read`},...t.reverseOffsets&&t.reverseNeighbors&&t.reverseOverflow?{reverseOffsets:{view:t.reverseOffsets,usage:`storage-read`},reverseNeighbors:{view:t.reverseNeighbors,usage:`storage-read`},reverseOverflow:{view:t.reverseOverflow,usage:`storage-read`}}:{},converged:{view:t.converged,usage:`storage-write`,atomic:!0}},a=P(t,Math.max(t.vertexCount,1)),o=t.vertexCount===0||t.edgeCount===0,s=`
${A(t)}
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${r(t.forwardOverflow)}u;
const CONVERGED_OFFSET: u32 = ${r(t.converged)}u;
${t.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${r(t.reverseOverflow)}u;`:``}
${M(i)}
${j(t)}

@compute @workgroup_size(${w})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(a,w)}
  let hasOverflow = forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${t.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``};
  if (index == 0u) {
    atomicStore(&converged[CONVERGED_OFFSET], select(0u, ${o?`1u`:`0u`}, !hasOverflow));
  }
  if (index >= VERTEX_COUNT) { return; }
  if (hasOverflow) {
    output[OUTPUT_OFFSET + index] = ${T}u;
    return;
  }

  var degree = 0u;
  for (var direction = 0u; direction < NEIGHBOR_DIRECTION_COUNT; direction++) {
    let first = getFirstNeighborSlot(index, direction);
    let last = getLastNeighborSlot(index, direction);
    for (var slot = first; slot < last; slot++) {
      let neighbor = getNeighbor(direction, slot);
      if (isFirstDistinctNeighbor(index, direction, slot, neighbor)) { degree++; }
    }
  }
  output[OUTPUT_OFFSET + index] = degree;
}`;N(e,{operation:`GPUGraphCoreNumber`,workgroupSize:w,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-initialize`,source:s,bindings:i,dispatchLayout:a})}function O(e,t,n){let i=t.changed,a={changed:{view:i,usage:`storage-write`,atomic:!0}},o=`
const CHANGED_OFFSET: u32 = ${r(i)}u;
${M(a)}

@compute @workgroup_size(1)
fn main() {
  atomicStore(&changed[CHANGED_OFFSET], 0u);
}`;N(e,{operation:`GPUGraphCoreNumber`,workgroupSize:1,invocationCount:1,id:`${t.id}-iteration-${n}-reset`,source:o,bindings:a,dispatchLayout:{x:1,y:1,z:1}})}function k(e,t,i){let a=t.scratch,o=t.changed,s={output:{view:t.output,usage:`storage-read`},scratch:{view:a,usage:`storage-write`},forwardOffsets:{view:t.forwardOffsets,usage:`storage-read`},forwardNeighbors:{view:t.forwardNeighbors,usage:`storage-read`},...t.reverseOffsets&&t.reverseNeighbors?{reverseOffsets:{view:t.reverseOffsets,usage:`storage-read`},reverseNeighbors:{view:t.reverseNeighbors,usage:`storage-read`}}:{},changed:{view:o,usage:`storage-read-write`,atomic:!0},converged:{view:t.converged,usage:`storage-read`}},c=P(t,t.vertexCount),l=`
${A(t)}
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
const SCRATCH_OFFSET: u32 = ${r(a)}u;
const CHANGED_OFFSET: u32 = ${r(o)}u;
const CONVERGED_OFFSET: u32 = ${r(t.converged)}u;
${M(s)}
${j(t)}

fn countSupportingNeighbors(vertex: u32, candidate: u32) -> u32 {
  var supporters = 0u;
  for (var direction = 0u; direction < NEIGHBOR_DIRECTION_COUNT; direction++) {
    let first = getFirstNeighborSlot(vertex, direction);
    let last = getLastNeighborSlot(vertex, direction);
    for (var slot = first; slot < last; slot++) {
      let neighbor = getNeighbor(direction, slot);
      if (!isFirstDistinctNeighbor(vertex, direction, slot, neighbor)) { continue; }
      let neighborCore = output[OUTPUT_OFFSET + neighbor];
      if (neighborCore != ${T}u && neighborCore >= candidate) {
        supporters++;
      }
    }
  }
  return supporters;
}

@compute @workgroup_size(${w})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(c,w)}
  if (index >= VERTEX_COUNT || converged[CONVERGED_OFFSET] != 0u) { return; }
  let previous = output[OUTPUT_OFFSET + index];
  if (previous == ${T}u) {
    scratch[SCRATCH_OFFSET + index] = ${T}u;
    return;
  }

  var lower = 0u;
  var upper = previous;
  while (lower < upper) {
    let candidate = lower + (upper - lower + 1u) / 2u;
    if (countSupportingNeighbors(index, candidate) >= candidate) {
      lower = candidate;
    } else {
      upper = candidate - 1u;
    }
  }
  scratch[SCRATCH_OFFSET + index] = lower;
  if (lower != previous) { atomicStore(&changed[CHANGED_OFFSET], 1u); }
}`;N(e,{operation:`GPUGraphCoreNumber`,workgroupSize:w,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-iteration-${i}-refine`,source:l,bindings:s,dispatchLayout:c})}function ae(e,t,i){let a=t.scratch,o={output:{view:t.output,usage:`storage-write`},scratch:{view:a,usage:`storage-read`},converged:{view:t.converged,usage:`storage-read`}},s=P(t,t.vertexCount),c=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
const SCRATCH_OFFSET: u32 = ${r(a)}u;
const CONVERGED_OFFSET: u32 = ${r(t.converged)}u;
${M(o)}

@compute @workgroup_size(${w})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(s,w)}
  if (index >= VERTEX_COUNT || converged[CONVERGED_OFFSET] != 0u) { return; }
  output[OUTPUT_OFFSET + index] = scratch[SCRATCH_OFFSET + index];
}`;N(e,{operation:`GPUGraphCoreNumber`,workgroupSize:w,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-iteration-${i}-publish`,source:c,bindings:o,dispatchLayout:s})}function oe(e,t,n){let i=t.changed,a={changed:{view:i,usage:`storage-read`},converged:{view:t.converged,usage:`storage-read-write`,atomic:!0},forwardOverflow:{view:t.forwardOverflow,usage:`storage-read`},...t.reverseOverflow?{reverseOverflow:{view:t.reverseOverflow,usage:`storage-read`}}:{}},o=`
const CHANGED_OFFSET: u32 = ${r(i)}u;
const CONVERGED_OFFSET: u32 = ${r(t.converged)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${r(t.forwardOverflow)}u;
${t.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${r(t.reverseOverflow)}u;`:``}
${M(a)}

@compute @workgroup_size(1)
fn main() {
  let hasOverflow = forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${t.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``};
  atomicStore(&converged[CONVERGED_OFFSET], select(0u, 1u, !hasOverflow && changed[CHANGED_OFFSET] == 0u));
}`;N(e,{operation:`GPUGraphCoreNumber`,workgroupSize:1,invocationCount:1,id:`${t.id}-iteration-${n}-convergence`,source:o,bindings:a,dispatchLayout:{x:1,y:1,z:1}})}function A(e){return[`const VERTEX_COUNT: u32 = ${e.vertexCount}u;`,`const FORWARD_CAPACITY: u32 = ${e.forwardNeighbors.length}u;`,`const FORWARD_OFFSETS_OFFSET: u32 = ${r(e.forwardOffsets)}u;`,`const FORWARD_NEIGHBORS_OFFSET: u32 = ${r(e.forwardNeighbors)}u;`,`const NEIGHBOR_DIRECTION_COUNT: u32 = ${e.reverseOffsets?2:1}u;`,...e.reverseOffsets&&e.reverseNeighbors?[`const REVERSE_CAPACITY: u32 = ${e.reverseNeighbors.length}u;`,`const REVERSE_OFFSETS_OFFSET: u32 = ${r(e.reverseOffsets)}u;`,`const REVERSE_NEIGHBORS_OFFSET: u32 = ${r(e.reverseNeighbors)}u;`]:[]].join(`
`)}function j(e){let t=!!(e.reverseOffsets&&e.reverseNeighbors);return`
fn getFirstNeighborSlot(vertex: u32, direction: u32) -> u32 {
  ${t?`if (direction == 1u) {
    return min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex], REVERSE_CAPACITY);
  }`:``}
  return min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex], FORWARD_CAPACITY);
}

fn getLastNeighborSlot(vertex: u32, direction: u32) -> u32 {
  ${t?`if (direction == 1u) {
    return min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex + 1u], REVERSE_CAPACITY);
  }`:``}
  return min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex + 1u], FORWARD_CAPACITY);
}

fn getNeighbor(direction: u32, slot: u32) -> u32 {
  ${t?`if (direction == 1u) {
    return reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + slot];
  }`:``}
  return forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + slot];
}

fn isFirstDistinctNeighbor(vertex: u32, direction: u32, slot: u32, candidate: u32) -> bool {
  if (candidate >= VERTEX_COUNT || candidate == vertex) { return false; }
  let forwardFirst = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex], FORWARD_CAPACITY);
  let forwardEnd = select(
    slot,
    min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex + 1u], FORWARD_CAPACITY),
    direction != 0u
  );
  for (var previous = forwardFirst; previous < forwardEnd; previous++) {
    if (forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + previous] == candidate) { return false; }
  }
  ${t?`if (direction == 1u) {
    let reverseFirst = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex], REVERSE_CAPACITY);
    for (var previous = reverseFirst; previous < slot; previous++) {
      if (reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + previous] == candidate) { return false; }
    }
  }`:``}
  return true;
}`}function M(e){return Object.entries(e).map(([e,t],n)=>`@group(0) @binding(${n}) var<storage, ${t.usage===`storage-read`?`read`:`read_write`}> ${e}: array<${t.atomic?`atomic<u32>`:`u32`}>;`).join(`
`)}function N(e,n){let r=e.recording?u(e,{operation:n.operation,source:n.source,bindings:Object.entries(n.bindings).map(([e,t])=>({name:e,...t})),workgroupSize:n.workgroupSize,dispatch:n.dispatchLayout,invocationCount:n.invocationCount,datum:n.datum}):void 0;e.addComputePass({...r&&{annotation:r},id:n.id,resources:Object.values(n.bindings).map(({view:e,usage:t})=>({buffer:e,usage:t})),compile:({device:e})=>{let r=new o(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,r]of Object.entries(n.bindings))a[e]=t(r.view,i);r.setBindings(a),r.dispatch(e,n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z)},destroy:()=>r.destroy()}}})}function P(e,t){return se(t,e.maxComputeWorkgroupsPerDimension)}function se(e,t){return a(`GPUGraphCoreNumber`,e,w,t)}var ce=32,F=1024,I=4,L=class{id;topology;output;iterations;converged;degeneracy;constructor(e){if(this.id=e.id??`gpu-graph-core-number`,this.topology=e.topology,this.output=e.output,this.iterations=e.iterations??ce,this.converged=e.converged,this.degeneracy=e.degeneracy,this.topology.graph.directed&&!this.topology.reverse)throw Error(`${this.id} directed weak-neighbor core numbers require reverse adjacency`);if(!Number.isSafeInteger(this.iterations)||this.iterations<0||this.iterations>F)throw Error(`${this.id} iterations must be a safe integer between zero and 1024`);R(this.output,this.topology.graph.vertexCount,`${this.id} output`),this.converged&&R(this.converged,1,`${this.id} converged`),this.degeneracy&&R(this.degeneracy,1,`${this.id} degeneracy`),l(this.topology,[this.output,this.converged,this.degeneracy])||B(this)}addToGraph(e){v(e,`GPUGraphCoreNumber`,this.id,()=>{E(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function R(e,t,n){d(e)?g(e,`uint32`,t,n):z(e,t,n)}function z(e,t,n){if(e.format!==`uint32`||e.data.length!==1||e.length!==t||e.stride!==1||e.byteStride!==I||e.rowByteLength!==I||e.valueLength!==e.length||e.bufferLayout)throw Error(`${n} must contain exactly ${t} packed uint32 rows in one chunk`);let r=e.data[0];if(r.format!==`uint32`||r.length!==t||r.stride!==1||r.byteStride!==I||r.rowByteLength!==I||r.valueLength!==r.length||!Number.isSafeInteger(r.byteOffset)||r.byteOffset<0||r.byteOffset%I!==0)throw Error(`${n} must contain one packed, uint32-aligned chunk`)}function B(e){let t=e.topology,n=[t.graph.sourceVertices,t.graph.targetVertices,...t.graph.edgeWeights?[t.graph.edgeWeights]:[],...t.graph.edgeIds?[t.graph.edgeIds]:[],...V(t.forward),...t.reverse?V(t.reverse):[],t.invalidEdgeCount],r=new Set;for(let e of n)for(let t of e.data)r.add(H(t));let i=[{name:`output`,vector:e.output},...e.converged?[{name:`converged`,vector:e.converged}]:[],...e.degeneracy?[{name:`degeneracy`,vector:e.degeneracy}]:[]];for(let{name:t,vector:n}of i){let i=H(n.data[0]);if(r.has(i))throw Error(`${e.id} ${t} must use a distinct physical buffer allocation`);r.add(i)}}function V(e){return[e.offsets,e.neighbors,e.edgeIds,...e.edgeWeights?[e.edgeWeights]:[],e.count,e.overflow]}function H(t){return t.buffer instanceof e?t.buffer.buffer:t.buffer}var U=256;function W(e,t,n){if(e.topology.graph.vertexCount===0&&!e.residual)return;let r=e.topology.graph.directed,a=p(t,`${e.id}-forward-offsets`,e.topology.forward.offsets),o=r?e.topology.reverse:e.topology.forward,s={id:e.id,vertexCount:e.topology.graph.vertexCount,damping:e.damping,forwardOffsets:a,incomingOffsets:r?p(t,`${e.id}-incoming-offsets`,o.offsets):a,incomingNeighbors:p(t,`${e.id}-incoming-neighbors`,o.neighbors),overflow:h(t,`${e.id}-forward-overflow`,e.topology.forward.overflow,`GPUGraphPageRank`),...r?{reverseOverflow:h(t,`${e.id}-incoming-overflow`,o.overflow,`GPUGraphPageRank`)}:{},output:p(t,`${e.id}-output`,e.output),...e.residual?{residual:p(t,`${e.id}-residual`,e.residual)}:{},maxComputeWorkgroupsPerDimension:n};if(l(e.topology,[e.output,e.residual])&&m(e.id,[s.forwardOffsets,s.incomingOffsets,s.incomingNeighbors,s.overflow,...s.reverseOverflow?[s.reverseOverflow]:[]],[{name:`output`,view:s.output},...s.residual?[{name:`residual`,view:s.residual}]:[]]),le(t,s),s.vertexCount===0)return;let c=i(t,`${s.id}-workspace`,`float32`,s.vertexCount),u=ue(t,s);for(let n=0;n<e.iterations;n++)f(t,{loopId:`${s.id}-iterations`,iteration:n,maximumIterations:e.iterations,gated:!1},()=>{de(t,{state:s,workspace:c,iteration:n}),fe(t,{state:s,workspace:c,danglingMass:G(t,{id:`${s.id}-iteration-${n}-dangling`,state:s,input:c,levels:u}),iteration:n});let r=G(t,{id:`${s.id}-iteration-${n}-sum`,state:s,input:c,levels:u}),i=!!(s.residual&&n===e.iterations-1);pe(t,{state:s,workspace:c,rankSum:r,iteration:n,collectResidual:i}),i&&G(t,{id:`${s.id}-residual`,state:s,input:c,levels:u,output:s.residual})})}function le(e,t){let i={output:{view:t.output,usage:`storage-write`},overflow:{view:t.overflow,usage:`storage-read`},...t.reverseOverflow?{reverseOverflow:{view:t.reverseOverflow,usage:`storage-read`}}:{},...t.residual?{residual:{view:t.residual,usage:`storage-write`}}:{}},a=t.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${r(t.reverseOverflow)}u;`:``,o=t.residual?`const RESIDUAL_OFFSET: u32 = ${r(t.residual)}u;`:``,s=t.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,c=t.residual?`if (index == 0u) { residual[RESIDUAL_OFFSET] = 0.0; }`:``,l=J(Math.max(t.vertexCount,1),t.maxComputeWorkgroupsPerDimension),u=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
const OVERFLOW_OFFSET: u32 = ${r(t.overflow)}u;
${a}
${o}
${K(i)}

@compute @workgroup_size(${U})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(l,U)}
  if (index < VERTEX_COUNT) {
    let hasOverflow = overflow[OVERFLOW_OFFSET] != 0u${s};
    let uniformScore = 1.0 / f32(max(VERTEX_COUNT, 1u));
    output[OUTPUT_OFFSET + index] = select(uniformScore, 0.0, hasOverflow);
  }
  ${c}
}`;q(e,{operation:`GPUGraphPageRank`,workgroupSize:U,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-initialize`,source:u,bindings:i,dispatchLayout:l})}function ue(e,t){let n=[],r=t.vertexCount;do r=Math.ceil(r/U),n.push(i(e,`${t.id}-reduction-level-${n.length}`,`float32`,r));while(r>1);return n}function de(e,t){let{state:i,workspace:a}=t,o={output:{view:i.output,usage:`storage-read`},forwardOffsets:{view:i.forwardOffsets,usage:`storage-read`},workspace:{view:a,usage:`storage-write`},overflow:{view:i.overflow,usage:`storage-read`},...i.reverseOverflow?{reverseOverflow:{view:i.reverseOverflow,usage:`storage-read`}}:{}},s=i.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${r(i.reverseOverflow)}u;`:``,c=i.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,l=J(i.vertexCount,i.maxComputeWorkgroupsPerDimension),u=`
const VERTEX_COUNT: u32 = ${i.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${r(i.output)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${r(i.forwardOffsets)}u;
const WORKSPACE_OFFSET: u32 = ${r(a)}u;
const OVERFLOW_OFFSET: u32 = ${r(i.overflow)}u;
${s}
${K(o)}

@compute @workgroup_size(${U})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(l,U)}
  if (index >= VERTEX_COUNT) { return; }
  let hasOverflow = overflow[OVERFLOW_OFFSET] != 0u${c};
  var contribution = 0.0;
  if (!hasOverflow) {
    let degree =
      forwardOffsets[FORWARD_OFFSETS_OFFSET + index + 1u] -
      forwardOffsets[FORWARD_OFFSETS_OFFSET + index];
    if (degree == 0u) { contribution = output[OUTPUT_OFFSET + index]; }
  }
  workspace[WORKSPACE_OFFSET + index] = contribution;
}`;q(e,{operation:`GPUGraphPageRank`,workgroupSize:U,datum:{keySpace:`vertex`},invocationCount:i.vertexCount,id:`${i.id}-iteration-${t.iteration}-gather-dangling`,source:u,bindings:o,dispatchLayout:l})}function fe(e,t){let{state:i,workspace:a,danglingMass:o}=t,s={output:{view:i.output,usage:`storage-read`},workspace:{view:a,usage:`storage-write`},forwardOffsets:{view:i.forwardOffsets,usage:`storage-read`},...i.reverseOverflow?{incomingOffsets:{view:i.incomingOffsets,usage:`storage-read`}}:{},incomingNeighbors:{view:i.incomingNeighbors,usage:`storage-read`},danglingMass:{view:o,usage:`storage-read`},overflow:{view:i.overflow,usage:`storage-read`},...i.reverseOverflow?{reverseOverflow:{view:i.reverseOverflow,usage:`storage-read`}}:{}},c=i.reverseOverflow?`const INCOMING_OFFSETS_OFFSET: u32 = ${r(i.incomingOffsets)}u;`:``,l=i.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${r(i.reverseOverflow)}u;`:``,u=i.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,d=i.reverseOverflow?`incomingOffsets`:`forwardOffsets`,f=i.reverseOverflow?`INCOMING_OFFSETS_OFFSET`:`FORWARD_OFFSETS_OFFSET`,p=J(i.vertexCount,i.maxComputeWorkgroupsPerDimension),m=`
const VERTEX_COUNT: u32 = ${i.vertexCount}u;
const CAPACITY: u32 = ${i.incomingNeighbors.length}u;
const DAMPING: f32 = ${i.damping};
const OUTPUT_OFFSET: u32 = ${r(i.output)}u;
const WORKSPACE_OFFSET: u32 = ${r(a)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${r(i.forwardOffsets)}u;
const INCOMING_NEIGHBORS_OFFSET: u32 = ${r(i.incomingNeighbors)}u;
const DANGLING_MASS_OFFSET: u32 = ${r(o)}u;
const OVERFLOW_OFFSET: u32 = ${r(i.overflow)}u;
${c}
${l}
${K(s)}

@compute @workgroup_size(${U})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(p,U)}
  if (index >= VERTEX_COUNT) { return; }
  let hasOverflow = overflow[OVERFLOW_OFFSET] != 0u${u};
  if (hasOverflow) {
    workspace[WORKSPACE_OFFSET + index] = 0.0;
    return;
  }
  let first = min(${d}[${f} + index], CAPACITY);
  let last = min(${d}[${f} + index + 1u], CAPACITY);
  var incomingMass = 0.0;
  for (var slot = first; slot < last; slot++) {
    let neighbor = incomingNeighbors[INCOMING_NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT) { continue; }
    let degree =
      forwardOffsets[FORWARD_OFFSETS_OFFSET + neighbor + 1u] -
      forwardOffsets[FORWARD_OFFSETS_OFFSET + neighbor];
    if (degree > 0u) {
      incomingMass += output[OUTPUT_OFFSET + neighbor] / f32(degree);
    }
  }
  let vertexCount = f32(VERTEX_COUNT);
  let redistributedDangling = danglingMass[DANGLING_MASS_OFFSET] / vertexCount;
  let teleportation = (1.0 - DAMPING) / vertexCount;
  workspace[WORKSPACE_OFFSET + index] =
    teleportation + DAMPING * (incomingMass + redistributedDangling);
}`;q(e,{operation:`GPUGraphPageRank`,workgroupSize:U,datum:{keySpace:`vertex`},invocationCount:i.vertexCount,id:`${i.id}-iteration-${t.iteration}-pull`,source:m,bindings:s,dispatchLayout:p})}function pe(e,t){let{state:i,workspace:a,rankSum:o}=t,s={output:{view:i.output,usage:`storage-read-write`},workspace:{view:a,usage:t.collectResidual?`storage-read-write`:`storage-read`},rankSum:{view:o,usage:`storage-read`},overflow:{view:i.overflow,usage:`storage-read`},...i.reverseOverflow?{reverseOverflow:{view:i.reverseOverflow,usage:`storage-read`}}:{}},c=i.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${r(i.reverseOverflow)}u;`:``,l=i.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,u=t.collectResidual?`workspace[WORKSPACE_OFFSET + index] = difference;`:``,d=J(i.vertexCount,i.maxComputeWorkgroupsPerDimension),f=`
const VERTEX_COUNT: u32 = ${i.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${r(i.output)}u;
const WORKSPACE_OFFSET: u32 = ${r(a)}u;
const RANK_SUM_OFFSET: u32 = ${r(o)}u;
const OVERFLOW_OFFSET: u32 = ${r(i.overflow)}u;
${c}
${K(s)}

@compute @workgroup_size(${U})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(d,U)}
  if (index >= VERTEX_COUNT) { return; }
  let hasOverflow = overflow[OVERFLOW_OFFSET] != 0u${l};
  let total = rankSum[RANK_SUM_OFFSET];
  let validTotal = total > 0.0 && total == total && abs(total) <= 3.402823466e+38;
  var next = 0.0;
  var difference = 0.0;
  if (!hasOverflow && validTotal) {
    next = workspace[WORKSPACE_OFFSET + index] / total;
    difference = abs(next - output[OUTPUT_OFFSET + index]);
  }
  output[OUTPUT_OFFSET + index] = next;
  ${u}
}`;q(e,{operation:`GPUGraphPageRank`,workgroupSize:U,datum:{keySpace:`vertex`},invocationCount:i.vertexCount,id:`${i.id}-iteration-${t.iteration}-normalize`,source:f,bindings:s,dispatchLayout:d})}function G(e,t){let n=t.input;for(let[r,i]of t.levels.entries()){let a=r===t.levels.length-1&&t.output?t.output:i;me(e,{id:`${t.id}-level-${r}`,input:n,output:a,maxComputeWorkgroupsPerDimension:t.state.maxComputeWorkgroupsPerDimension}),n=a}return n}function me(e,t){let i={inputValues:{view:t.input,usage:`storage-read`},outputValues:{view:t.output,usage:`storage-write`}},a=J(t.input.length,t.maxComputeWorkgroupsPerDimension),o=s(e.device)===`subgroups`,c=o?`let subgroupTotal = subgroupAdd(value);
  if (subgroupInvocationId == 0u) {
    reductionValues[subgroupId] = subgroupTotal;
  }
  workgroupBarrier();
  let subgroupCount = ${U}u / subgroupSize;
  if (subgroupCount > 1u) {
    for (var stride = subgroupCount / 2u; stride > 0u; stride /= 2u) {
      if (localInvocationIndex < stride) {
        reductionValues[localInvocationIndex] += reductionValues[localInvocationIndex + stride];
      }
      workgroupBarrier();
    }
  }`:`reductionValues[localInvocationIndex] = value;
  workgroupBarrier();

  for (var stride = ${U/2}u; stride > 0u; stride /= 2u) {
    if (localInvocationIndex < stride) {
      reductionValues[localInvocationIndex] += reductionValues[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`,l=`
${o?`enable subgroups;
requires subgroup_id;`:``}
const INPUT_COUNT: u32 = ${t.input.length}u;
const OUTPUT_COUNT: u32 = ${t.output.length}u;
const INPUT_OFFSET: u32 = ${r(t.input)}u;
const OUTPUT_OFFSET: u32 = ${r(t.output)}u;
${K(i)}
var<workgroup> reductionValues: array<f32, ${U}>;

@compute @workgroup_size(${U})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  ${o?`@builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32,`:``}
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${n(a,U)}
  if (workgroupIndex >= OUTPUT_COUNT) { return; }
  var value = 0.0;
  if (index < INPUT_COUNT) { value = inputValues[INPUT_OFFSET + index]; }
  ${c}
  if (localInvocationIndex == 0u) {
    outputValues[OUTPUT_OFFSET + workgroupIndex] = reductionValues[0];
  }
}`;q(e,{operation:`GPUGraphPageRank`,workgroupSize:U,invocationCount:t.input.length,id:t.id,source:l,bindings:i,dispatchLayout:a})}function K(e){return Object.entries(e).map(([e,t],n)=>`@group(0) @binding(${n}) var<storage, ${t.usage===`storage-read`?`read`:`read_write`}> ${e}: array<${t.view.format===`float32`?`f32`:`u32`}>;`).join(`
`)}function q(e,n){let r=e.recording?u(e,{operation:n.operation,source:n.source,bindings:Object.entries(n.bindings).map(([e,t])=>({name:e,...t})),workgroupSize:n.workgroupSize,dispatch:n.dispatchLayout,invocationCount:n.invocationCount,datum:n.datum}):void 0;e.addComputePass({...r&&{annotation:r},id:n.id,resources:Object.values(n.bindings).map(({view:e,usage:t})=>({buffer:e,usage:t})),compile:({device:e})=>{let r=new o(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,r]of Object.entries(n.bindings))a[e]=t(r.view,i);r.setBindings(a),r.dispatch(e,n.dispatchLayout.x,n.dispatchLayout.y,n.dispatchLayout.z)},destroy:()=>r.destroy()}}})}function J(e,t){return a(`GPUGraphPageRank`,e,U,t)}var he=.85,Y=40,ge=1024,X=4,_e=class{id;topology;output;damping;iterations;residual;constructor(e){if(this.id=e.id??`gpu-graph-page-rank`,this.topology=e.topology,this.output=e.output,this.damping=e.damping??he,this.iterations=e.iterations??Y,this.residual=e.residual,this.topology.graph.directed&&!this.topology.reverse)throw Error(`${this.id} directed PageRank requires reverse adjacency`);if(!Number.isFinite(this.damping)||this.damping<0||this.damping>1)throw Error(`${this.id} damping must be a finite number between zero and one`);if(!Number.isSafeInteger(this.iterations)||this.iterations<1||this.iterations>ge)throw Error(`${this.id} iterations must be a safe integer between one and 1024`);Z(this.output,this.topology.graph.vertexCount,`${this.id} output`),this.residual&&Z(this.residual,1,`${this.id} residual`);let{topology:t,output:n,residual:r}=this;!(t instanceof _)&&!d(n)&&!(r&&d(r))&&ye(this.id,t,n,r)}addToGraph(e){v(e,`GPUGraphPageRank`,this.id,()=>{W(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function Z(e,t,n){d(e)?g(e,`float32`,t,n):ve(e,t,n)}function ve(e,t,n){if(e.data.length!==1||e.format!==`float32`||e.stride!==1||e.byteStride!==X||e.rowByteLength!==X||e.valueLength!==e.length||e.bufferLayout)throw Error(`${n} must contain exactly one packed float32 chunk`);if(e.length!==t)throw Error(`${n} must contain exactly ${t} float32 rows`);let r=e.data[0];if(r.format!==`float32`||r.length!==t||r.stride!==1||r.byteStride!==X||r.rowByteLength!==X||r.valueLength!==r.length||!Number.isSafeInteger(r.byteOffset)||r.byteOffset<0||r.byteOffset%X!==0)throw Error(`${n} must contain one packed, float32-aligned chunk`)}function ye(e,t,n,r){let i=[t.graph.sourceVertices,t.graph.targetVertices,...t.graph.edgeWeights?[t.graph.edgeWeights]:[],...t.graph.edgeIds?[t.graph.edgeIds]:[],...Q(t.forward),...t.reverse?Q(t.reverse):[],t.invalidEdgeCount],a=new Set;for(let e of i)for(let t of e.data)a.add($(t));let o=[{name:`output`,vector:n},...r?[{name:`residual`,vector:r}]:[]];for(let{name:t,vector:n}of o){let r=$(n.data[0]);if(a.has(r))throw Error(`${e} ${t} must use a distinct physical buffer allocation`);a.add(r)}}function Q(e){return[e.offsets,e.neighbors,e.edgeIds,...e.edgeWeights?[e.edgeWeights]:[],e.count,e.overflow]}function $(t){return t.buffer instanceof e?t.buffer.buffer:t.buffer}export{L as n,x as r,_e as t};