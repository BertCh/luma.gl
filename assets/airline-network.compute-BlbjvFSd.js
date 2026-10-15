import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as t}from"./src-CNCKFW2b.js";import"./analysis-frame-Blu0tYdw.js";import{S as n,a as r,w as i}from"./arithmetic-expression-Kknn3j_i.js";import{D as a,F as o,N as s,S as c,k as l,t as u,w as d,x as f}from"./resources-CwvhEOBe.js";import{u as p}from"./interleave-brZZCO-f.js";import{n as m,t as h}from"./gpu-grid-index-D1JNO067.js";import{n as g,t as _}from"./gpu-great-circle-arcs-D0WZYJJk.js";import{a as ee,c as te}from"./index-Bjk18V2I.js";import{t as v}from"./graph-buffers-B4_80RlX.js";import{t as ne}from"./graph-swap-41Ru7LVu.js";import{t as re}from"./summary-reader-CsgWxENf.js";import{t as ie}from"./mode-kernels-Dxm2GWvI.js";import{n as ae}from"./anchors-D9E9ucGL.js";import"./class-table-BVEo3-Vs.js";import"./hue-registry-4FjIJ9xz.js";import{t as oe}from"./gpu-graph-rLdn6WKb.js";import{c as se,l as ce,u as le}from"./gpu-graph-topology-view-DhuT3ULg.js";import{n as ue,r as de,t as fe}from"./gpu-graph-modularity-optimization-uPa9xSWp.js";import{n as pe,r as me,t as he}from"./gpu-graph-page-rank-BugwiSFh.js";import{t as ge}from"./gpu-graph-label-propagation-BJ25J5mS.js";import{r as _e,t as ve}from"./gpu-edge-bundling-DTYx6Cwo.js";import{c as ye,i as be,n as y}from"./live-text-83XBWyr1.js";import{n as xe}from"./picking-DRmnefrA.js";import{t as Se}from"./proportional-C9adWYq6.js";import{a as Ce,i as we,o as Te,p as Ee}from"./b3-common-BbHtVprn.js";import{r as De,t as Oe}from"./b11-geography-nN3A5Qz3.js";import{i as ke,t as Ae}from"./b11-flight-data-ClfAV2uI.js";import"./flows-style-DZTmwMyJ.js";import{a as je,c as Me,n as Ne,o as Pe,r as Fe,s as Ie,t as Le}from"./airline-network-palette-Bkn9dW7J.js";import{i as Re,n as ze,r as Be,t as Ve}from"./airline-network-layers-DRoqCpci.js";import{t as He}from"./stable-hues-CtE8dbLG.js";var b=256,Ue=1e-4;function We(e,t,n){if(e.topology.graph.vertexCount===0&&!e.reset)return;let r=e.topology.graph.directed,i=e.topology.reverse,a={id:e.id,vertexCount:e.topology.graph.vertexCount,seed:e.seed,repulsion:e.repulsion,attraction:e.attraction,gravity:e.gravity,damping:e.damping,maxVelocity:e.maxVelocity,timeStep:e.timeStep,positions:t.importGPUVector(`${e.id}-positions`,e.positions).data[0],velocities:t.importGPUVector(`${e.id}-velocities`,e.velocities).data[0],...e.pinned?{pinned:t.importGPUVector(`${e.id}-pinned`,e.pinned).data[0]}:{},...e.reset?{reset:t.importGPUVector(`${e.id}-reset`,e.reset).data[0]}:{},forwardOffsets:t.importGPUVector(`${e.id}-forward-offsets`,e.topology.forward.offsets).data[0],forwardNeighbors:t.importGPUVector(`${e.id}-forward-neighbors`,e.topology.forward.neighbors).data[0],overflow:t.importGPUVector(`${e.id}-forward-overflow`,e.topology.forward.overflow).data[0],...r&&i?{reverseOffsets:t.importGPUVector(`${e.id}-reverse-offsets`,i.offsets).data[0],reverseNeighbors:t.importGPUVector(`${e.id}-reverse-neighbors`,i.neighbors).data[0],reverseOverflow:t.importGPUVector(`${e.id}-reverse-overflow`,i.overflow).data[0]}:{},maxComputeWorkgroupsPerDimension:n};if(a.reset&&a.vertexCount>0&&Ge(t,a),a.reset&&Ke(t,a),a.vertexCount!==0)for(let n=0;n<e.iterationsPerFrame;n++)ce(t,{loopId:`${a.id}-iterations`,iteration:n,maximumIterations:e.iterationsPerFrame,gated:!1},()=>{qe(t,{state:a,iteration:n}),Je(t,{state:a,iteration:n})})}function Ge(e,t){let n=t.reset,r={positions:{view:t.positions,usage:`storage-read-write`},velocities:{view:t.velocities,usage:`storage-write`},reset:{view:n,usage:`storage-read`},overflow:{view:t.overflow,usage:`storage-read`},...t.reverseOverflow?{reverseOverflow:{view:t.reverseOverflow,usage:`storage-read`}}:{},...t.pinned?{pinned:{view:t.pinned,usage:`storage-read`}}:{}},i=t.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${l(t.reverseOverflow)}u;`:``,a=t.pinned?`const PINNED_OFFSET: u32 = ${l(t.pinned)}u;`:``,o=t.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,s=t.pinned?`pinned[PINNED_OFFSET + index] != 0u`:`false`,u=C(t.vertexCount,t.maxComputeWorkgroupsPerDimension),d=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const SEED: u32 = ${t.seed}u;
const POSITIONS_OFFSET: u32 = ${l(t.positions)}u;
const VELOCITIES_OFFSET: u32 = ${l(t.velocities)}u;
const RESET_OFFSET: u32 = ${l(n)}u;
const OVERFLOW_OFFSET: u32 = ${l(t.overflow)}u;
${i}
${a}
${x(r)}

fn hash(value: u32) -> u32 {
  var result = value;
  result ^= result >> 16u;
  result *= 0x7feb352du;
  result ^= result >> 15u;
  result *= 0x846ca68bu;
  result ^= result >> 16u;
  return result;
}

@compute @workgroup_size(${b})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(u,b)}
  if (index >= VERTEX_COUNT || reset[RESET_OFFSET] == 0u) { return; }
  let velocityOffset = VELOCITIES_OFFSET + index * 2u;
  velocities[velocityOffset] = 0.0;
  velocities[velocityOffset + 1u] = 0.0;
  let hasOverflow = overflow[OVERFLOW_OFFSET] != 0u${o};
  if (hasOverflow || ${s}) { return; }
  let first = hash(SEED ^ (index * 2u));
  let second = hash(SEED ^ (index * 2u + 1u));
  let positionOffset = POSITIONS_OFFSET + index * 2u;
  positions[positionOffset] = f32(first & 0x00ffffffu) / 16777216.0 * 2.0 - 1.0;
  positions[positionOffset + 1u] = f32(second & 0x00ffffffu) / 16777216.0 * 2.0 - 1.0;
}`;S(e,{operation:`GPUGraphForceLayout`,workgroupSize:b,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-initialize`,source:d,bindings:r,dispatchLayout:u})}function Ke(e,t){let n=t.reset,r={reset:{view:n,usage:`storage-write`}},i=C(1,t.maxComputeWorkgroupsPerDimension),a=`
const RESET_OFFSET: u32 = ${l(n)}u;
${x(r)}

@compute @workgroup_size(${b})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(i,b)}
  if (index == 0u) { reset[RESET_OFFSET] = 0u; }
}`;S(e,{operation:`GPUGraphForceLayout`,workgroupSize:b,invocationCount:1,id:`${t.id}-clear-reset`,source:a,bindings:r,dispatchLayout:i})}function qe(e,t){let{state:n}=t,r={positions:{view:n.positions,usage:`storage-read`},velocities:{view:n.velocities,usage:`storage-read-write`},forwardOffsets:{view:n.forwardOffsets,usage:`storage-read`},forwardNeighbors:{view:n.forwardNeighbors,usage:`storage-read`},overflow:{view:n.overflow,usage:`storage-read`},...n.reverseOffsets&&n.reverseNeighbors&&n.reverseOverflow?{reverseOffsets:{view:n.reverseOffsets,usage:`storage-read`},reverseNeighbors:{view:n.reverseNeighbors,usage:`storage-read`},reverseOverflow:{view:n.reverseOverflow,usage:`storage-read`}}:{}},i=n.reverseOffsets&&n.reverseNeighbors&&n.reverseOverflow?`const REVERSE_CAPACITY: u32 = ${n.reverseNeighbors.length}u;
const REVERSE_OFFSETS_OFFSET: u32 = ${l(n.reverseOffsets)}u;
const REVERSE_NEIGHBORS_OFFSET: u32 = ${l(n.reverseNeighbors)}u;
const REVERSE_OVERFLOW_OFFSET: u32 = ${l(n.reverseOverflow)}u;`:``,a=n.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,o=n.reverseOffsets&&n.reverseNeighbors?`let reverseFirst = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + index], REVERSE_CAPACITY);
  let reverseLast = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + index + 1u], REVERSE_CAPACITY);
  for (var slot = reverseFirst; slot < reverseLast; slot++) {
    let neighbor = reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + slot];
    if (neighbor < VERTEX_COUNT) {
      force += ATTRACTION * (readPosition(neighbor) - position);
    }
  }`:``,s=Math.ceil(n.vertexCount/b),u=C(n.vertexCount,n.maxComputeWorkgroupsPerDimension),d=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const TILE_COUNT: u32 = ${s}u;
const FORWARD_CAPACITY: u32 = ${n.forwardNeighbors.length}u;
const POSITIONS_OFFSET: u32 = ${l(n.positions)}u;
const VELOCITIES_OFFSET: u32 = ${l(n.velocities)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${l(n.forwardOffsets)}u;
const FORWARD_NEIGHBORS_OFFSET: u32 = ${l(n.forwardNeighbors)}u;
const OVERFLOW_OFFSET: u32 = ${l(n.overflow)}u;
const REPULSION: f32 = ${n.repulsion};
const ATTRACTION: f32 = ${n.attraction};
const GRAVITY: f32 = ${n.gravity};
const DAMPING: f32 = ${n.damping};
const MAX_VELOCITY: f32 = ${n.maxVelocity};
const TIME_STEP: f32 = ${n.timeStep};
const MINIMUM_DISTANCE_SQUARED: f32 = ${Ue};
${i}
${x(r)}
var<workgroup> tilePositions: array<vec2<f32>, ${b}>;

fn readPosition(vertex: u32) -> vec2<f32> {
  let positionOffset = POSITIONS_OFFSET + vertex * 2u;
  return vec2<f32>(positions[positionOffset], positions[positionOffset + 1u]);
}

@compute @workgroup_size(${b})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(u,b)}
  if (workgroupIndex >= TILE_COUNT) { return; }
  let isActiveVertex = index < VERTEX_COUNT;
  let hasOverflow = overflow[OVERFLOW_OFFSET] != 0u${a};
  var position = vec2<f32>(0.0);
  if (isActiveVertex) { position = readPosition(index); }
  var force = -GRAVITY * position;

  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let sourceVertex = tile * ${b}u + localInvocationIndex;
    if (sourceVertex < VERTEX_COUNT) {
      tilePositions[localInvocationIndex] = readPosition(sourceVertex);
    } else {
      tilePositions[localInvocationIndex] = vec2<f32>(0.0);
    }
    workgroupBarrier();

    if (isActiveVertex && !hasOverflow) {
      let firstVertex = tile * ${b}u;
      let count = min(${b}u, VERTEX_COUNT - firstVertex);
      for (var localVertex = 0u; localVertex < count; localVertex++) {
        if (firstVertex + localVertex != index) {
          let difference = position - tilePositions[localVertex];
          let distanceSquared = max(dot(difference, difference), MINIMUM_DISTANCE_SQUARED);
          force += REPULSION * difference / distanceSquared;
        }
      }
    }
    workgroupBarrier();
  }

  if (!isActiveVertex) { return; }
  let velocityOffset = VELOCITIES_OFFSET + index * 2u;
  if (hasOverflow) {
    velocities[velocityOffset] = 0.0;
    velocities[velocityOffset + 1u] = 0.0;
    return;
  }

  let first = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + index], FORWARD_CAPACITY);
  let last = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + index + 1u], FORWARD_CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + slot];
    if (neighbor < VERTEX_COUNT) {
      force += ATTRACTION * (readPosition(neighbor) - position);
    }
  }
  ${o}
  let previousVelocity = vec2<f32>(velocities[velocityOffset], velocities[velocityOffset + 1u]);
  var velocity = (previousVelocity + force * TIME_STEP) * DAMPING;
  let speed = length(velocity);
  if (speed > MAX_VELOCITY) { velocity *= MAX_VELOCITY / speed; }
  velocities[velocityOffset] = velocity.x;
  velocities[velocityOffset + 1u] = velocity.y;
}`;S(e,{operation:`GPUGraphForceLayout`,workgroupSize:b,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-forces`,source:d,bindings:r,dispatchLayout:u})}function Je(e,t){let{state:n}=t,r={positions:{view:n.positions,usage:`storage-read-write`},velocities:{view:n.velocities,usage:`storage-read-write`},overflow:{view:n.overflow,usage:`storage-read`},...n.reverseOverflow?{reverseOverflow:{view:n.reverseOverflow,usage:`storage-read`}}:{},...n.pinned?{pinned:{view:n.pinned,usage:`storage-read`}}:{}},i=n.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${l(n.reverseOverflow)}u;`:``,a=n.pinned?`const PINNED_OFFSET: u32 = ${l(n.pinned)}u;`:``,o=n.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,s=n.pinned?` || pinned[PINNED_OFFSET + index] != 0u`:``,u=C(n.vertexCount,n.maxComputeWorkgroupsPerDimension),d=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const TIME_STEP: f32 = ${n.timeStep};
const POSITIONS_OFFSET: u32 = ${l(n.positions)}u;
const VELOCITIES_OFFSET: u32 = ${l(n.velocities)}u;
const OVERFLOW_OFFSET: u32 = ${l(n.overflow)}u;
${i}
${a}
${x(r)}

@compute @workgroup_size(${b})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(u,b)}
  if (index >= VERTEX_COUNT) { return; }
  let velocityOffset = VELOCITIES_OFFSET + index * 2u;
  let blocked = overflow[OVERFLOW_OFFSET] != 0u${o}${s};
  if (blocked) {
    velocities[velocityOffset] = 0.0;
    velocities[velocityOffset + 1u] = 0.0;
    return;
  }
  let positionOffset = POSITIONS_OFFSET + index * 2u;
  positions[positionOffset] += velocities[velocityOffset] * TIME_STEP;
  positions[positionOffset + 1u] += velocities[velocityOffset + 1u] * TIME_STEP;
}`;S(e,{operation:`GPUGraphForceLayout`,workgroupSize:b,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-integrate`,source:d,bindings:r,dispatchLayout:u})}function x(e){return Object.entries(e).map(([e,t],n)=>`@group(0) @binding(${n}) var<storage, ${t.usage===`storage-read`?`read`:`read_write`}> ${e}: array<${t.view.format===`uint32`?`u32`:`f32`}>;`).join(`
`)}function S(e,t){let n=e.recording?se(e,{operation:t.operation,source:t.source,bindings:Object.entries(t.bindings).map(([e,t])=>({name:e,...t})),workgroupSize:t.workgroupSize,dispatch:t.dispatchLayout,invocationCount:t.invocationCount,datum:t.datum}):void 0;e.addComputePass({...n&&{annotation:n},id:t.id,resources:Object.values(t.bindings).map(({view:e,usage:t})=>({buffer:e,usage:t})),compile:({device:e})=>{let n=new p(e,{id:t.id,source:t.source,shaderLayout:{bindings:Object.keys(t.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>{let i={};for(let[e,n]of Object.entries(t.bindings))i[e]=a(n.view,r);n.setBindings(i),n.dispatch(e,t.dispatchLayout.x,t.dispatchLayout.y,t.dispatchLayout.z)},destroy:()=>n.destroy()}}})}function C(e,t){return f(`GPUGraphForceLayout`,e,b,t)}var Ye=4294967295,Xe=1024,w=4,Ze=class{id;topology;positions;velocities;pinned;reset;seed;iterationsPerFrame;repulsion;attraction;gravity;damping;maxVelocity;timeStep;constructor(e){if(this.id=e.id??`gpu-graph-force-layout`,this.topology=e.topology,this.positions=e.positions,this.velocities=e.velocities,this.pinned=e.pinned,this.reset=e.reset,this.seed=e.seed??0,this.iterationsPerFrame=e.iterationsPerFrame??4,this.repulsion=e.repulsion??1,this.attraction=e.attraction??.1,this.gravity=e.gravity??.01,this.damping=e.damping??.9,this.maxVelocity=e.maxVelocity??1,this.timeStep=e.timeStep??1,this.topology.graph.directed&&!this.topology.reverse)throw Error(`${this.id} directed force layout requires reverse adjacency`);if(!Number.isSafeInteger(this.seed)||this.seed<0||this.seed>Ye)throw Error(`${this.id} seed must be an unsigned 32-bit integer`);if(!Number.isSafeInteger(this.iterationsPerFrame)||this.iterationsPerFrame<1||this.iterationsPerFrame>Xe)throw Error(`${this.id} iterationsPerFrame must be an integer between one and 1024`);if(Qe(this.repulsion,`${this.id} repulsion`),Qe(this.attraction,`${this.id} attraction`),Qe(this.gravity,`${this.id} gravity`),!Number.isFinite(this.damping)||this.damping<0||this.damping>1)throw Error(`${this.id} damping must be a finite number between zero and one`);T(this.maxVelocity,`${this.id} maxVelocity`),T(this.timeStep,`${this.id} timeStep`);let n=this.topology.graph.vertexCount;E(this.positions,`float32x2`,n,`${this.id} positions`),E(this.velocities,`float32x2`,n,`${this.id} velocities`),this.pinned&&E(this.pinned,`uint32`,n,`${this.id} pinned`),this.reset&&E(this.reset,`uint32`,1,`${this.id} reset`);let r=O(this.positions.data[0]).usage,i=t.STORAGE|t.VERTEX;if((r&i)!==i)throw Error(`${this.id} positions require both STORAGE and VERTEX buffer usage`);if((O(this.velocities.data[0]).usage&t.STORAGE)===0)throw Error(`${this.id} velocities require STORAGE buffer usage`);$e(this)}addToGraph(e){le(e,`GPUGraphForceLayout`,this.id,()=>{We(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function Qe(e,t){if(!Number.isFinite(e)||e<0)throw Error(`${t} must be a finite non-negative number`)}function T(e,t){if(!Number.isFinite(e)||e<=0)throw Error(`${t} must be a finite positive number`)}function E(e,t,n,r){let i=t===`float32x2`?2:1,a=i*w;if(e.data.length!==1||e.format!==t||e.stride!==i||e.byteStride!==a||e.rowByteLength!==a||e.valueLength!==e.length||e.bufferLayout)throw Error(`${r} must contain exactly one packed ${t} chunk`);if(e.length!==n)throw Error(`${r} must contain exactly ${n} ${t} rows`);let o=e.data[0];if(o.format!==t||o.length!==n||o.stride!==i||o.byteStride!==a||o.rowByteLength!==a||o.valueLength!==o.length||!Number.isSafeInteger(o.byteOffset)||o.byteOffset<0||o.byteOffset%w!==0)throw Error(`${r} must contain one packed, uint32-aligned ${t} chunk`)}function $e(e){let t=e.topology,n=[t.graph.sourceVertices,t.graph.targetVertices,...t.graph.edgeWeights?[t.graph.edgeWeights]:[],...t.graph.edgeIds?[t.graph.edgeIds]:[],...D(t.forward),...t.reverse?D(t.reverse):[],t.invalidEdgeCount],r=new Set;for(let e of n)for(let t of e.data)r.add(O(t));let i=[{name:`positions`,vector:e.positions},{name:`velocities`,vector:e.velocities},...e.pinned?[{name:`pinned`,vector:e.pinned}]:[],...e.reset?[{name:`reset`,vector:e.reset}]:[]];for(let{name:t,vector:n}of i){let i=O(n.data[0]);if(r.has(i))throw Error(`${e.id} ${t} must use a distinct physical buffer allocation`);r.add(i)}}function D(e){return[e.offsets,e.neighbors,e.edgeIds,...e.edgeWeights?[e.edgeWeights]:[],e.count,e.overflow]}function O(e){return e.buffer instanceof r?e.buffer.buffer:e.buffer}var k=256,A=1e-4;function et(e,t,n){let r=e.layout,i=r.topology.reverse,a={id:e.id,vertexCount:r.topology.graph.vertexCount,cellCount:e.cellCount,gridSize:e.gridSize,bounds:e.bounds,theta:e.theta,nearCellRadius:e.nearCellRadius,seed:r.seed,repulsion:r.repulsion,attraction:r.attraction,gravity:r.gravity,damping:r.damping,maxVelocity:r.maxVelocity,timeStep:r.timeStep,positions:t.importGPUVector(`${e.id}-positions`,r.positions).data[0],velocities:t.importGPUVector(`${e.id}-velocities`,r.velocities).data[0],...r.pinned?{pinned:t.importGPUVector(`${e.id}-pinned`,r.pinned).data[0]}:{},...r.reset?{reset:t.importGPUVector(`${e.id}-reset`,r.reset).data[0]}:{},forwardOffsets:t.importGPUVector(`${e.id}-forward-offsets`,r.topology.forward.offsets).data[0],forwardNeighbors:t.importGPUVector(`${e.id}-forward-neighbors`,r.topology.forward.neighbors).data[0],forwardOverflow:t.importGPUVector(`${e.id}-forward-overflow`,r.topology.forward.overflow).data[0],...r.topology.graph.directed&&i?{reverseOffsets:t.importGPUVector(`${e.id}-reverse-offsets`,i.offsets).data[0],reverseNeighbors:t.importGPUVector(`${e.id}-reverse-neighbors`,i.neighbors).data[0],reverseOverflow:t.importGPUVector(`${e.id}-reverse-overflow`,i.overflow).data[0]}:{},cellOffsets:t.importGPUVector(`${e.id}-cell-offsets`,e.cellOffsets).data[0],vertexIds:t.importGPUVector(`${e.id}-vertex-ids`,e.vertexIds).data[0],cellCenters:t.importGPUVector(`${e.id}-cell-centers`,e.cellCenters).data[0],count:t.importGPUVector(`${e.id}-index-count`,e.count).data[0],overflow:t.importGPUVector(`${e.id}-index-overflow`,e.overflow).data[0],validity:d(t,`${e.id}-validity`,`uint32`,1),maxComputeWorkgroupsPerDimension:n};a.reset&&a.vertexCount>0&&tt(t,a),a.reset&&nt(t,a);let s=a.vertexCount===0?1:r.iterationsPerFrame;for(let e=0;e<s;e++)ce(t,{loopId:`${a.id}-iterations`,iteration:e,maximumIterations:s,gated:!1},()=>{o(t,m(new h({id:`${a.id}-iteration-${e}-index`,positions:a.positions,gridSize:a.gridSize,bounds:a.bounds,cellOffsets:a.cellOffsets,objectIds:a.vertexIds,count:a.count,overflow:a.overflow}),t,a.maxComputeWorkgroupsPerDimension)),rt(t,{state:a,iteration:e}),it(t,{state:a,iteration:e}),a.vertexCount>0&&(at(t,{state:a,iteration:e}),j(t,{state:a,iteration:e}),M(t,{state:a,iteration:e}))})}function tt(e,t){let n=t.reset,r={positions:{view:t.positions,usage:`storage-read-write`},velocities:{view:t.velocities,usage:`storage-write`},reset:{view:n,usage:`storage-read`},forwardOverflow:{view:t.forwardOverflow,usage:`storage-read`},...t.reverseOverflow?{reverseOverflow:{view:t.reverseOverflow,usage:`storage-read`}}:{},...t.pinned?{pinned:{view:t.pinned,usage:`storage-read`}}:{}},i=t.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${l(t.reverseOverflow)}u;`:``,a=t.pinned?`const PINNED_OFFSET: u32 = ${l(t.pinned)}u;`:``,o=t.reverseOverflow?` || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u`:``,s=t.pinned?` || pinned[PINNED_OFFSET + index] != 0u`:``,u=I(t,t.vertexCount),d=`
const VERTEX_COUNT: u32 = ${t.vertexCount}u;
const SEED: u32 = ${t.seed}u;
const POSITIONS_OFFSET: u32 = ${l(t.positions)}u;
const VELOCITIES_OFFSET: u32 = ${l(t.velocities)}u;
const RESET_OFFSET: u32 = ${l(n)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${l(t.forwardOverflow)}u;
${i}
${a}
${N(r)}

fn hash(value: u32) -> u32 {
  var result = value;
  result ^= result >> 16u;
  result *= 0x7feb352du;
  result ^= result >> 15u;
  result *= 0x846ca68bu;
  result ^= result >> 16u;
  return result;
}

@compute @workgroup_size(${k})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(u,k)}
  if (index >= VERTEX_COUNT || reset[RESET_OFFSET] == 0u) { return; }
  let velocityOffset = VELOCITIES_OFFSET + index * 2u;
  velocities[velocityOffset] = 0.0;
  velocities[velocityOffset + 1u] = 0.0;
  if (forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${o}${s}) { return; }
  let first = hash(SEED ^ (index * 2u));
  let second = hash(SEED ^ (index * 2u + 1u));
  let positionOffset = POSITIONS_OFFSET + index * 2u;
  positions[positionOffset] = f32(first & 0x00ffffffu) / 16777216.0 * 2.0 - 1.0;
  positions[positionOffset + 1u] = f32(second & 0x00ffffffu) / 16777216.0 * 2.0 - 1.0;
}`;P(e,{operation:`GPUGraphSpatialForceLayout`,workgroupSize:k,datum:{keySpace:`vertex`},invocationCount:t.vertexCount,id:`${t.id}-initialize`,source:d,bindings:r,dispatchLayout:u})}function nt(e,t){let n=t.reset,r={reset:{view:n,usage:`storage-write`}},i=I(t,1),a=`
const RESET_OFFSET: u32 = ${l(n)}u;
${N(r)}

@compute @workgroup_size(${k})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(i,k)}
  if (index == 0u) { reset[RESET_OFFSET] = 0u; }
}`;P(e,{operation:`GPUGraphSpatialForceLayout`,workgroupSize:k,invocationCount:1,id:`${t.id}-clear-reset`,source:a,bindings:r,dispatchLayout:i})}function rt(e,t){let{state:n}=t,r={count:{view:n.count,usage:`storage-read`},indexOverflow:{view:n.overflow,usage:`storage-read`},forwardOverflow:{view:n.forwardOverflow,usage:`storage-read`},validity:{view:n.validity,usage:`storage-write`},...n.reverseOverflow?{reverseOverflow:{view:n.reverseOverflow,usage:`storage-read`}}:{}},i=n.reverseOverflow?`const REVERSE_OVERFLOW_OFFSET: u32 = ${l(n.reverseOverflow)}u;`:``,a=n.reverseOverflow?` && reverseOverflow[REVERSE_OVERFLOW_OFFSET] == 0u`:``,o=I(n,1),s=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const COUNT_OFFSET: u32 = ${l(n.count)}u;
const INDEX_OVERFLOW_OFFSET: u32 = ${l(n.overflow)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${l(n.forwardOverflow)}u;
const VALIDITY_OFFSET: u32 = ${l(n.validity)}u;
${i}
${N(r)}

@compute @workgroup_size(${k})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(o,k)}
  if (index != 0u) { return; }
  let isValid =
    count[COUNT_OFFSET] == VERTEX_COUNT &&
    indexOverflow[INDEX_OVERFLOW_OFFSET] == 0u &&
    forwardOverflow[FORWARD_OVERFLOW_OFFSET] == 0u${a};
  validity[VALIDITY_OFFSET] = select(0u, 1u, isValid);
}`;P(e,{operation:`GPUGraphSpatialForceLayout`,workgroupSize:k,invocationCount:1,id:`${n.id}-iteration-${t.iteration}-validate`,source:s,bindings:r,dispatchLayout:o})}function it(e,t){let{state:n}=t,r={positions:{view:n.positions,usage:`storage-read`},cellOffsets:{view:n.cellOffsets,usage:`storage-read`},vertexIds:{view:n.vertexIds,usage:`storage-read`},cellCenters:{view:n.cellCenters,usage:`storage-write`},validity:{view:n.validity,usage:`storage-read`}},i=I(n,n.cellCount),a=`
const CELL_COUNT: u32 = ${n.cellCount}u;
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const CAPACITY: u32 = ${n.vertexIds.length}u;
const POSITIONS_OFFSET: u32 = ${l(n.positions)}u;
const CELL_OFFSETS_OFFSET: u32 = ${l(n.cellOffsets)}u;
const VERTEX_IDS_OFFSET: u32 = ${l(n.vertexIds)}u;
const CELL_CENTERS_OFFSET: u32 = ${l(n.cellCenters)}u;
const VALIDITY_OFFSET: u32 = ${l(n.validity)}u;
${N(r)}

@compute @workgroup_size(${k})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(i,k)}
  if (index >= CELL_COUNT) { return; }
  var center = vec2<f32>(0.0);
  var mass = 0u;
  if (validity[VALIDITY_OFFSET] != 0u) {
    let first = min(cellOffsets[CELL_OFFSETS_OFFSET + index], CAPACITY);
    let last = min(cellOffsets[CELL_OFFSETS_OFFSET + index + 1u], CAPACITY);
    for (var slot = first; slot < last; slot++) {
      let vertex = vertexIds[VERTEX_IDS_OFFSET + slot];
      if (vertex < VERTEX_COUNT) {
        let positionOffset = POSITIONS_OFFSET + vertex * 2u;
        center += vec2<f32>(positions[positionOffset], positions[positionOffset + 1u]);
        mass++;
      }
    }
  }
  if (mass > 0u) { center /= f32(mass); }
  let centerOffset = CELL_CENTERS_OFFSET + index * 2u;
  cellCenters[centerOffset] = center.x;
  cellCenters[centerOffset + 1u] = center.y;
}`;P(e,{operation:`GPUGraphSpatialForceLayout`,workgroupSize:k,datum:{keySpace:`cell`},invocationCount:n.cellCount,id:`${n.id}-iteration-${t.iteration}-cell-centers`,source:a,bindings:r,dispatchLayout:i})}function at(e,t){let{state:n}=t,r={positions:{view:n.positions,usage:`storage-read`},velocities:{view:n.velocities,usage:`storage-read-write`},cellOffsets:{view:n.cellOffsets,usage:`storage-read`},vertexIds:{view:n.vertexIds,usage:`storage-read`},cellCenters:{view:n.cellCenters,usage:`storage-read`},validity:{view:n.validity,usage:`storage-read`}},i=I(n,n.vertexCount),a=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const CELL_COUNT: u32 = ${n.cellCount}u;
const WIDTH: u32 = ${n.gridSize[0]}u;
const HEIGHT: u32 = ${n.gridSize[1]}u;
const CAPACITY: u32 = ${n.vertexIds.length}u;
const NEAR_CELL_RADIUS: u32 = ${n.nearCellRadius}u;
const MINIMUM_X: f32 = ${F(n.bounds[0])};
const MINIMUM_Y: f32 = ${F(n.bounds[1])};
const MAXIMUM_X: f32 = ${F(n.bounds[2])};
const MAXIMUM_Y: f32 = ${F(n.bounds[3])};
const THETA: f32 = ${F(n.theta)};
const REPULSION: f32 = ${F(n.repulsion)};
const GRAVITY: f32 = ${F(n.gravity)};
const TIME_STEP: f32 = ${F(n.timeStep)};
const MINIMUM_DISTANCE_SQUARED: f32 = ${A};
const POSITIONS_OFFSET: u32 = ${l(n.positions)}u;
const VELOCITIES_OFFSET: u32 = ${l(n.velocities)}u;
const CELL_OFFSETS_OFFSET: u32 = ${l(n.cellOffsets)}u;
const VERTEX_IDS_OFFSET: u32 = ${l(n.vertexIds)}u;
const CELL_CENTERS_OFFSET: u32 = ${l(n.cellCenters)}u;
const VALIDITY_OFFSET: u32 = ${l(n.validity)}u;
${N(r)}

fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}

fn readPosition(vertex: u32) -> vec2<f32> {
  let positionOffset = POSITIONS_OFFSET + vertex * 2u;
  return vec2<f32>(positions[positionOffset], positions[positionOffset + 1u]);
}

@compute @workgroup_size(${k})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(i,k)}
  if (index >= VERTEX_COUNT) { return; }
  let velocityOffset = VELOCITIES_OFFSET + index * 2u;
  if (validity[VALIDITY_OFFSET] == 0u) {
    velocities[velocityOffset] = 0.0;
    velocities[velocityOffset + 1u] = 0.0;
    return;
  }

  let position = readPosition(index);
  let sourceColumn = getCoordinate(position.x, MINIMUM_X, MAXIMUM_X, WIDTH);
  let sourceRow = getCoordinate(position.y, MINIMUM_Y, MAXIMUM_Y, HEIGHT);
  let cellWidth = (MAXIMUM_X - MINIMUM_X) / f32(WIDTH);
  let cellHeight = (MAXIMUM_Y - MINIMUM_Y) / f32(HEIGHT);
  let cellDiameterSquared = cellWidth * cellWidth + cellHeight * cellHeight;
  var force = -GRAVITY * position;

  for (var cell = 0u; cell < CELL_COUNT; cell++) {
    let first = min(cellOffsets[CELL_OFFSETS_OFFSET + cell], CAPACITY);
    let last = min(cellOffsets[CELL_OFFSETS_OFFSET + cell + 1u], CAPACITY);
    let mass = last - first;
    if (mass == 0u) { continue; }
    let column = cell % WIDTH;
    let row = cell / WIDTH;
    let columnDistance = max(column, sourceColumn) - min(column, sourceColumn);
    let rowDistance = max(row, sourceRow) - min(row, sourceRow);
    let isNear = columnDistance <= NEAR_CELL_RADIUS && rowDistance <= NEAR_CELL_RADIUS;
    let centerOffset = CELL_CENTERS_OFFSET + cell * 2u;
    let center = vec2<f32>(cellCenters[centerOffset], cellCenters[centerOffset + 1u]);
    let centerDifference = position - center;
    let distanceSquared = dot(centerDifference, centerDifference);
    let useMonopole =
      !isNear && THETA > 0.0 && cellDiameterSquared < THETA * THETA * distanceSquared;
    if (useMonopole) {
      force += REPULSION * f32(mass) * centerDifference /
        max(distanceSquared, MINIMUM_DISTANCE_SQUARED);
      continue;
    }
    for (var slot = first; slot < last; slot++) {
      let otherVertex = vertexIds[VERTEX_IDS_OFFSET + slot];
      if (otherVertex >= VERTEX_COUNT || otherVertex == index) { continue; }
      let difference = position - readPosition(otherVertex);
      let separationSquared = max(dot(difference, difference), MINIMUM_DISTANCE_SQUARED);
      force += REPULSION * difference / separationSquared;
    }
  }

  let previous = vec2<f32>(velocities[velocityOffset], velocities[velocityOffset + 1u]);
  let velocity = previous + force * TIME_STEP;
  velocities[velocityOffset] = velocity.x;
  velocities[velocityOffset + 1u] = velocity.y;
}`;P(e,{operation:`GPUGraphSpatialForceLayout`,workgroupSize:k,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-repulsion`,source:a,bindings:r,dispatchLayout:i})}function j(e,t){let{state:n}=t,r={positions:{view:n.positions,usage:`storage-read`},velocities:{view:n.velocities,usage:`storage-read-write`},forwardOffsets:{view:n.forwardOffsets,usage:`storage-read`},forwardNeighbors:{view:n.forwardNeighbors,usage:`storage-read`},validity:{view:n.validity,usage:`storage-read`},...n.reverseOffsets&&n.reverseNeighbors?{reverseOffsets:{view:n.reverseOffsets,usage:`storage-read`},reverseNeighbors:{view:n.reverseNeighbors,usage:`storage-read`}}:{}},i=n.reverseOffsets&&n.reverseNeighbors?`const REVERSE_CAPACITY: u32 = ${n.reverseNeighbors.length}u;
const REVERSE_OFFSETS_OFFSET: u32 = ${l(n.reverseOffsets)}u;
const REVERSE_NEIGHBORS_OFFSET: u32 = ${l(n.reverseNeighbors)}u;`:``,a=n.reverseOffsets&&n.reverseNeighbors?`let reverseFirst = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + index], REVERSE_CAPACITY);
  let reverseLast = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + index + 1u], REVERSE_CAPACITY);
  for (var slot = reverseFirst; slot < reverseLast; slot++) {
    let neighbor = reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + slot];
    if (neighbor < VERTEX_COUNT) {
      force += ATTRACTION * (readPosition(neighbor) - position);
    }
  }`:``,o=I(n,n.vertexCount),s=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const FORWARD_CAPACITY: u32 = ${n.forwardNeighbors.length}u;
const POSITIONS_OFFSET: u32 = ${l(n.positions)}u;
const VELOCITIES_OFFSET: u32 = ${l(n.velocities)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${l(n.forwardOffsets)}u;
const FORWARD_NEIGHBORS_OFFSET: u32 = ${l(n.forwardNeighbors)}u;
const VALIDITY_OFFSET: u32 = ${l(n.validity)}u;
const ATTRACTION: f32 = ${F(n.attraction)};
const DAMPING: f32 = ${F(n.damping)};
const MAX_VELOCITY: f32 = ${F(n.maxVelocity)};
const TIME_STEP: f32 = ${F(n.timeStep)};
${i}
${N(r)}

fn readPosition(vertex: u32) -> vec2<f32> {
  let positionOffset = POSITIONS_OFFSET + vertex * 2u;
  return vec2<f32>(positions[positionOffset], positions[positionOffset + 1u]);
}

@compute @workgroup_size(${k})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(o,k)}
  if (index >= VERTEX_COUNT) { return; }
  let velocityOffset = VELOCITIES_OFFSET + index * 2u;
  if (validity[VALIDITY_OFFSET] == 0u) {
    velocities[velocityOffset] = 0.0;
    velocities[velocityOffset + 1u] = 0.0;
    return;
  }
  let position = readPosition(index);
  var force = vec2<f32>(0.0);
  let first = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + index], FORWARD_CAPACITY);
  let last = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + index + 1u], FORWARD_CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + slot];
    if (neighbor < VERTEX_COUNT) {
      force += ATTRACTION * (readPosition(neighbor) - position);
    }
  }
  ${a}
  let previous = vec2<f32>(velocities[velocityOffset], velocities[velocityOffset + 1u]);
  var velocity = (previous + force * TIME_STEP) * DAMPING;
  let speed = length(velocity);
  if (speed > MAX_VELOCITY) { velocity *= MAX_VELOCITY / speed; }
  velocities[velocityOffset] = velocity.x;
  velocities[velocityOffset + 1u] = velocity.y;
}`;P(e,{operation:`GPUGraphSpatialForceLayout`,workgroupSize:k,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-attraction`,source:s,bindings:r,dispatchLayout:o})}function M(e,t){let{state:n}=t,r={positions:{view:n.positions,usage:`storage-read-write`},velocities:{view:n.velocities,usage:`storage-read-write`},validity:{view:n.validity,usage:`storage-read`},...n.pinned?{pinned:{view:n.pinned,usage:`storage-read`}}:{}},i=n.pinned?`const PINNED_OFFSET: u32 = ${l(n.pinned)}u;`:``,a=n.pinned?` || pinned[PINNED_OFFSET + index] != 0u`:``,o=I(n,n.vertexCount),s=`
const VERTEX_COUNT: u32 = ${n.vertexCount}u;
const POSITIONS_OFFSET: u32 = ${l(n.positions)}u;
const VELOCITIES_OFFSET: u32 = ${l(n.velocities)}u;
const VALIDITY_OFFSET: u32 = ${l(n.validity)}u;
const TIME_STEP: f32 = ${F(n.timeStep)};
${i}
${N(r)}

@compute @workgroup_size(${k})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(o,k)}
  if (index >= VERTEX_COUNT) { return; }
  let velocityOffset = VELOCITIES_OFFSET + index * 2u;
  if (validity[VALIDITY_OFFSET] == 0u${a}) {
    velocities[velocityOffset] = 0.0;
    velocities[velocityOffset + 1u] = 0.0;
    return;
  }
  let positionOffset = POSITIONS_OFFSET + index * 2u;
  positions[positionOffset] += velocities[velocityOffset] * TIME_STEP;
  positions[positionOffset + 1u] += velocities[velocityOffset + 1u] * TIME_STEP;
}`;P(e,{operation:`GPUGraphSpatialForceLayout`,workgroupSize:k,datum:{keySpace:`vertex`},invocationCount:n.vertexCount,id:`${n.id}-iteration-${t.iteration}-integrate`,source:s,bindings:r,dispatchLayout:o})}function N(e){return Object.entries(e).map(([e,t],n)=>`@group(0) @binding(${n}) var<storage, ${t.usage===`storage-read`?`read`:`read_write`}> ${e}: array<${t.view.format===`uint32`?`u32`:`f32`}>;`).join(`
`)}function P(e,t){let n=e.recording?se(e,{operation:t.operation,source:t.source,bindings:Object.entries(t.bindings).map(([e,t])=>({name:e,...t})),workgroupSize:t.workgroupSize,dispatch:t.dispatchLayout,invocationCount:t.invocationCount,datum:t.datum}):void 0;e.addComputePass({...n&&{annotation:n},id:t.id,resources:Object.values(t.bindings).map(({view:e,usage:t})=>({buffer:e,usage:t})),compile:({device:e})=>{let n=new p(e,{id:t.id,source:t.source,shaderLayout:{bindings:Object.keys(t.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:r})=>{let i={};for(let[e,n]of Object.entries(t.bindings))i[e]=a(n.view,r);n.setBindings(i),n.dispatch(e,t.dispatchLayout.x,t.dispatchLayout.y,t.dispatchLayout.z)},destroy:()=>n.destroy()}}})}function F(e){let t=`${Math.fround(e)}`;return t.includes(`.`)||t.includes(`e`)?t:`${t}.0`}function I(e,t){return L(t,e.maxComputeWorkgroupsPerDimension)}function L(e,t){return f(`GPUGraphSpatialForceLayout`,e,k,t)}var R=4294967295,z=4,ot=class{id;layout;gridSize;bounds;theta;nearCellRadius;cellCount;cellOffsets;vertexIds;cellCenters;count;overflow;constructor(e){if(this.id=e.id??`gpu-graph-spatial-force-layout`,this.layout=e.layout,this.gridSize=e.gridSize,this.bounds=e.bounds,this.theta=e.theta??.6,this.nearCellRadius=e.nearCellRadius??1,this.cellOffsets=e.cellOffsets,this.vertexIds=e.vertexIds,this.cellCenters=e.cellCenters,this.count=e.count,this.overflow=e.overflow,this.gridSize.length!==2||this.gridSize.some(e=>!Number.isSafeInteger(e)||e<1))throw Error(`${this.id} gridSize requires two positive integer dimensions`);let t=this.gridSize[0]*this.gridSize[1];if(!Number.isSafeInteger(t)||t>=R)throw Error(`${this.id} grid cell count and trailing offset must fit in uint32`);if(this.cellCount=t,this.bounds.length!==4||!this.bounds.every(Number.isFinite)||this.bounds[0]>=this.bounds[2]||this.bounds[1]>=this.bounds[3])throw Error(`${this.id} bounds require finite, strictly increasing x and y extents`);if(!Number.isFinite(this.theta)||this.theta<0)throw Error(`${this.id} theta must be a finite non-negative number`);if(!Number.isSafeInteger(this.nearCellRadius)||this.nearCellRadius<0||this.nearCellRadius>R)throw Error(`${this.id} nearCellRadius must be a non-negative uint32`);B(this.cellOffsets,`uint32`,`${this.id} cellOffsets`,t+1),B(this.vertexIds,`uint32`,`${this.id} vertexIds`),B(this.cellCenters,`float32x2`,`${this.id} cellCenters`,t),B(this.count,`uint32`,`${this.id} count`,1),B(this.overflow,`uint32`,`${this.id} overflow`,1),V(this)}addToGraph(e){le(e,`GPUGraphSpatialForceLayout`,this.id,()=>{et(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)})}};function B(e,t,n,r){let i=t===`float32x2`?2:1,a=i*z;if(e.data.length!==1||e.format!==t||e.stride!==i||e.byteStride!==a||e.rowByteLength!==a||e.valueLength!==e.length||e.bufferLayout||!Number.isSafeInteger(e.length)||e.length<0||e.length>R)throw Error(`${n} must contain exactly one packed ${t} chunk`);if(r!==void 0&&e.length!==r)throw Error(`${n} must contain exactly ${r} ${t} rows`);let o=e.data[0];if(o.format!==t||o.length!==e.length||o.stride!==i||o.byteStride!==a||o.rowByteLength!==a||o.valueLength!==o.length||!Number.isSafeInteger(o.byteOffset)||o.byteOffset<0||o.byteOffset%z!==0)throw Error(`${n} must contain one packed, uint32-aligned ${t} chunk`)}function V(e){let t=e.layout,n=t.topology,r=[n.graph.sourceVertices,n.graph.targetVertices,...n.graph.edgeWeights?[n.graph.edgeWeights]:[],...n.graph.edgeIds?[n.graph.edgeIds]:[],...st(n.forward),...n.reverse?st(n.reverse):[],n.invalidEdgeCount,t.positions,t.velocities,...t.pinned?[t.pinned]:[],...t.reset?[t.reset]:[]],i=new Set;for(let e of r)for(let t of e.data)i.add(ct(t));let a=[{name:`cellOffsets`,vector:e.cellOffsets},{name:`vertexIds`,vector:e.vertexIds},{name:`cellCenters`,vector:e.cellCenters},{name:`count`,vector:e.count},{name:`overflow`,vector:e.overflow}];for(let{name:t,vector:n}of a){let r=ct(n.data[0]);if(i.has(r))throw Error(`${e.id} ${t} must use a distinct physical buffer allocation`);i.add(r)}}function st(e){return[e.offsets,e.neighbors,e.edgeIds,...e.edgeWeights?[e.edgeWeights]:[],e.count,e.overflow]}function ct(e){return e.buffer instanceof r?e.buffer.buffer:e.buffer}var H=4,U=36,lt=class{buffers=[];vectors=[];constructor(e,t){this.device=e,this.prefix=t}scalar(e,r,i,a){let o=t.STORAGE|t.COPY_SRC|t.COPY_DST,s=this.device.createBuffer({id:`${this.prefix}-${e}`,usage:o,...a&&a.length>0?{data:a}:{byteLength:Math.max(i,1)*H}});this.buffers.push(s);let c=new n({type:`buffer`,name:e,format:r,buffer:s,length:i,ownsBuffer:!1});return this.vectors.push(c),c}coordinates(e,r,i=0){let a=this.device.createBuffer({id:`${this.prefix}-${e}`,data:r,usage:t.STORAGE|t.COPY_SRC|t.COPY_DST|i});this.buffers.push(a);let o=new n({type:`buffer`,name:e,format:`float32x2`,buffer:a,length:r.length/2,ownsBuffer:!1});return this.vectors.push(o),o}edgeColumn(e,r){let a=this.device.createBuffer({id:`${this.prefix}-${e}`,data:r,usage:t.STORAGE|t.COPY_DST});this.buffers.push(a);let o=new n({type:`data`,name:e,format:`uint32`,data:[new i({buffer:a,format:`uint32`,length:r.length,ownsBuffer:!1})],ownsData:!1});return this.vectors.push(o),o}adjacency(e,t,n){return{offsets:this.scalar(`${e}-offsets`,`uint32`,t+1),neighbors:this.scalar(`${e}-neighbors`,`uint32`,n),edgeIds:this.scalar(`${e}-edge-ids`,`uint32`,n),count:this.scalar(`${e}-count`,`uint32`,1),overflow:this.scalar(`${e}-overflow`,`uint32`,1)}}getBuffer(e){return e.data[0].buffer}destroy(){for(let e of this.vectors)e.destroy();for(let e of this.buffers)e.destroy();this.vectors.length=0,this.buffers.length=0}};function ut(e,t,n,r,i){let a=n.length,o=new oe({vertexCount:t,sourceVertices:e.edgeColumn(`sources`,n),targetVertices:e.edgeColumn(`targets`,r),directed:!1}),s=e.adjacency(`forward`,t,a*2);return{factory:e,vertexCount:t,edgeCount:a,graph:o,topology:new de({id:`airline-topology`,graph:o,forward:s,invalidEdgeCount:e.scalar(`invalid-edges`,`uint32`,1)}),forwardOverflow:s.overflow,degree:e.scalar(`degree`,`uint32`,t),pageRank:e.scalar(`page-rank`,`float32`,t),pageRankResidual:e.scalar(`page-rank-residual`,`float32`,1),coreNumber:e.scalar(`core-number`,`uint32`,t),coreConverged:e.scalar(`core-converged`,`uint32`,1),degeneracy:e.scalar(`degeneracy`,`uint32`,1),propagation:e.scalar(`propagation`,`uint32`,t),propagationConverged:e.scalar(`propagation-converged`,`uint32`,1),continents:e.scalar(`continents`,`uint32`,t,Uint32Array.from(i)),propagationModularity:e.scalar(`propagation-modularity`,`float32`,1),continentModularity:e.scalar(`continent-modularity`,`float32`,1),propagationValid:e.scalar(`propagation-valid`,`uint32`,1),continentValid:e.scalar(`continent-valid`,`uint32`,1)}}async function dt(e,t,n,r){let{topology:i}=t,a=new s(e,{id:r});return i.addToGraph(a),new me({id:`airline-degree`,topology:i,output:t.degree}).addToGraph(a),new he({id:`airline-page-rank`,topology:i,output:t.pageRank,damping:n.pageRankDamping,iterations:n.pageRankIterations,residual:t.pageRankResidual}).addToGraph(a),new pe({id:`airline-core`,topology:i,output:t.coreNumber,iterations:32,converged:t.coreConverged,degeneracy:t.degeneracy}).addToGraph(a),new ge({id:`airline-propagation`,topology:i,output:t.propagation,iterations:n.propagationRounds,converged:t.propagationConverged}).addToGraph(a),new ue({id:`airline-modularity-propagation`,graph:t.graph,communities:t.propagation,output:t.propagationModularity,resolution:1,valid:t.propagationValid}).addToGraph(a),new ue({id:`airline-modularity-continents`,graph:t.graph,communities:t.continents,output:t.continentModularity,resolution:1,valid:t.continentValid}).addToGraph(a),await a.compileAsync()}function ft(e){let{factory:t,vertexCount:n}=e,r=e=>({buffer:t.getBuffer(e),size:n*H}),i=e=>({buffer:t.getBuffer(e),size:H});return[r(e.pageRank),r(e.degree),r(e.coreNumber),r(e.propagation),i(e.pageRankResidual),i(e.degeneracy),i(e.coreConverged),i(e.propagationConverged),i(e.propagationModularity),i(e.continentModularity),i(e.propagationValid),i(e.continentValid),i(e.forwardOverflow)]}function pt(e,t){let n=t*H,r=new Float32Array(e,0,t),i=new Uint32Array(e,n,t),a=new Uint32Array(e,n*2,t),o=new Uint32Array(e,n*3,t),s=new Float32Array(e,n*4),c=new Uint32Array(e,n*4);return{pageRank:r,degree:i,coreNumber:a,propagation:o,residual:s[0],degeneracy:c[1],coreConverged:c[2]===1,propagationConverged:c[3]===1,propagationModularity:s[4],continentModularity:s[5],valid:c[6]===1&&c[7]===1,adjacencyOverflow:c[8]!==0}}async function mt(e,t,n,r,i){let a=new u(e,r),o=a.track(new lt(e,r)),{vertexCount:c}=t,l=o.scalar(`labels`,`uint32`,c),d=o.scalar(`modularity`,`float32`,1),f=o.scalar(`converged`,`uint32`,1),p=o.scalar(`valid`,`uint32`,1),m=o.scalar(`propagation-score`,`float32`,1),h=o.scalar(`continent-score`,`float32`,1),g=new s(e,{id:r});new fe({id:`airline-optimization`,topology:t.topology,output:l,modularity:d,initialCommunities:t.propagation,resolution:n.resolution,iterations:n.rounds,minimumGain:n.minimumGain,converged:f,valid:p}).addToGraph(g),new ue({id:`airline-score-propagation`,graph:t.graph,communities:t.propagation,output:m,resolution:n.resolution}).addToGraph(g),new ue({id:`airline-score-continents`,graph:t.graph,communities:t.continents,output:h,resolution:n.resolution}).addToGraph(g);let _;try{_=a.track(await g.compileAsync())}catch(e){throw a.destroy(),e}let ee=_.createExecution({maximumInvocationCount:2**53-1,maximumNodeCount:U},{latencyPriority:`background`}),te=c*H,v=new re(a,r,[{buffer:o.getBuffer(l),size:te},{buffer:o.getBuffer(d),size:H},{buffer:o.getBuffer(f),size:H},{buffer:o.getBuffer(p),size:H},{buffer:o.getBuffer(m),size:H},{buffer:o.getBuffer(h),size:H}],e=>{let t=new Float32Array(e,te),n=new Uint32Array(e,te);i({labels:new Uint32Array(e.slice(0,te)),modularity:t[0],converged:n[1]===1,valid:n[2]===1,propagationModularity:t[3],continentModularity:t[4]})});return{resources:a,compiled:_,execution:ee,reader:v,settings:n}}function ht(e){return Uint32Array.from({length:e.length},(e,t)=>t).sort((t,n)=>e[n]-e[t]||t-n)}function gt(e){let t=Array.from({length:e.length},(e,t)=>t).sort((t,n)=>e[t]-e[n]),n=new Float64Array(e.length),r=0;for(;r<t.length;){let i=r;for(;i+1<t.length&&e[t[i+1]]===e[t[r]];)i++;let a=(r+i)/2+1;for(let e=r;e<=i;e++)n[t[e]]=a;r=i+1}return n}function _t(e,t){let n=e.length,r=0,i=0;for(let a=0;a<n;a++)r+=e[a],i+=t[a];r/=n,i/=n;let a=0,o=0,s=0;for(let c=0;c<n;c++){let n=e[c]-r,l=t[c]-i;a+=n*l,o+=n*n,s+=l*l}return a/Math.sqrt(o*s)}function vt(e,t){return _t(gt(e),gt(t))}function yt(e,t){let n=1;for(let t=0;t<e.length;t++)n=Math.max(n,e[t]);let r=Math.max(Math.log10(n),1e-6),i=Array(t).fill(0);for(let n=0;n<e.length;n++){if(e[n]<1)continue;let a=Math.min(t-1,Math.floor(Math.log10(e[n])/r*t));i[a]++}return{counts:i,maximumLog:r}}function bt(e,t,n,r,i,a){let o=Oe.length,s=new Map;for(let e=0;e<t.length;e++){let i=s.get(t[e]);i||(i={size:0,top:e,byContinent:new Uint32Array(o)},s.set(t[e],i)),i.size++,i.byContinent[r[e]]++,n[e]>n[i.top]&&(i.top=e)}let c=[],l=0;for(let[e,t]of s){let n=0;for(let e=1;e<o;e++)t.byContinent[e]>t.byContinent[n]&&(n=e);l+=t.byContinent[n],c.push({label:e,size:t.size,topAirport:t.top,dominantContinent:n,dominantShare:t.byContinent[n]/t.size,slot:7})}if(c.sort((e,t)=>t.size-e.size||e.label-t.label),e===`continent`)for(let e of c)e.slot=Math.min(e.label,7);else{let e=c.slice(0,7),n=new Map;e.forEach((e,t)=>n.set(e.label,t));let i=He(r,Int32Array.from(t,e=>n.get(e)??-1),7);e.forEach((e,t)=>{e.slot=i[t]??7})}let u=new Map;for(let e of c)u.set(e.label,e.slot);let d=new Uint32Array(t.length);for(let e=0;e<t.length;e++)d[e]=u.get(t[e])??7;let f=new Uint32Array(i.length),p=0;for(let e=0;e<i.length;e++){let n=i[e],r=a[e];t[n]===t[r]?f[e]=d[n]:(f[e]=8,p++)}return{groups:c,nodeSlots:d,labels:t,edgeSlots:f,betweenCount:p,continentPurity:t.length>0?l/t.length:0}}function xt(e,t,n){let r=new Float32Array(e.length);for(let i=0;i<t.length;i++)e[t[i]]!==e[n[i]]&&(r[t[i]]++,r[n[i]]++);return r}function St(e,t,n){let r=0;for(let i=0;i<t.length;i++)(e.has(t[i])||e.has(n[i]))&&r++;return t.length>0?r/t.length:0}var W=[.25,.5,.75,1,1.25,1.5,2,3],Ct=32,wt=64,Tt=[-360,0,360],Et=[14,14],Dt=[-6,-6,6,6],Ot=4,kt=20,At=[10,15],jt=[165,58],Mt=e.LNGLAT,Nt=20,Pt=3,Ft=1.5,It=4,Lt=5,Rt=5,zt=80,Bt=260,Vt=16,Ht=24;async function G(e){let{device:n}=e,r=new u(n,`airline-network`),i=await De(`openflights`,e.signal),a=ke(e.datasets.get(`openflights`),i),o=a.nodeCount,c=a.edgeCount,l=Uint32Array.from(a.continent),d=r.track(new lt(n,`airline-network`)),f=ut(d,o,a.source,a.target,a.continent),p=r.createBuffer(`geo-positions`,a.lonLat),m=d.coordinates(`layout-positions`,new Float32Array(o*2),t.VERTEX),h=d.coordinates(`layout-velocities`,new Float32Array(o*2)),oe=d.scalar(`layout-pinned`,`uint32`,o),se=d.scalar(`layout-reset`,`uint32`,1,Uint32Array.of(1)),ce=Et[0]*Et[1],le=d.scalar(`spatial-offsets`,`uint32`,ce+1),ue=d.scalar(`spatial-ids`,`uint32`,o),de=d.coordinates(`spatial-centers`,new Float32Array(ce*2)),fe=d.scalar(`spatial-count`,`uint32`,1),pe=d.scalar(`spatial-overflow`,`uint32`,1),me=r.createBuffer(`morph-positions`,o*8),he=r.createBuffer(`morph-segments`,c*16),ge=r.createBuffer(`morph-weights`,c*4),He=r.createBuffer(`edge-mask`,new Uint32Array(c).fill(1)),b=r.createBuffer(`edge-order`,Uint32Array.from({length:c},(e,t)=>t)),Ue=r.createBuffer(`edge-index`,new Uint32Array(c)),We=r.createBuffer(`edge-category-continent`,new Float32Array(c)),Ge=r.createBuffer(`edge-category-community`,new Float32Array(c)),Ke=r.createBuffer(`node-slots-continent`,new Uint32Array(o)),qe=r.createBuffer(`node-slots-community`,new Uint32Array(o)),Je=r.createBuffer(`size-values`,new Float32Array(o)),x=r.createBuffer(`draw-order`,Uint32Array.from({length:o},(e,t)=>t)),S=r.createBuffer(`highlight-channel`,new Float32Array(o)),C=r.createParameterBuffer(`morph-parameters`,`float32`,8),Ye=new Float32Array(c*2),Xe=new Float32Array(c*2);for(let e=0;e<c;e++){let t=a.source[e],n=a.target[e];Ye.set(a.lonLat.subarray(t*2,t*2+2),e*2),Xe.set(a.lonLat.subarray(n*2,n*2+2),e*2)}let w=Te(r,`arcs`,c*(wt+1),c),Qe=r.createParameterBuffer(`arc-parameters`,`float32`,4),T=new s(n,{id:`airline-arcs`});{let e=Ce(T);T.add(new _({id:`airline-arcs`,sources:e(`sources`,r.createBuffer(`arc-sources`,Ye),`float32x2`,c),targets:e(`targets`,r.createBuffer(`arc-targets`,Xe),`float32x2`,c),maximumSegmentCount:wt,parameters:Qe.importToGraph(T),output:Ee(T,w)}))}let E=await T.compileAsync(),$e=new re(r,`arcs`,[{buffer:w.count,size:4},{buffer:w.overflow,size:4}],t=>{let n=new Uint32Array(t);e.setReadout(`arcVertices`,`${y(n[0])} vertices, ${(n[0]/c).toFixed(1)} per arc${n[1]?` (overflow)`:``}`)}),D=new s(n,{id:`airline-morph`});{let e=Ce(D),t=C.importToGraph(D),n=e(`geo`,p,`float32x2`,o),r=e(`morphed`,me,`float32x2`,o);ie(D,{id:`airline-morph-positions`,invocationCount:o,bindings:[{name:`geo`,view:n,type:`f32`,access:`read`},{name:`layoutPositions`,view:e(`layout`,d.getBuffer(m),`float32x2`,o),type:`f32`,access:`read`},{name:`parameters`,view:t,type:`f32`,access:`read`},{name:`morphed`,view:r,type:`f32`,access:`read_write`}],body:`
  let blend = parameters[parametersOffset];
  let eased = blend * blend * (3.0 - 2.0 * blend);
  let scale = parameters[parametersOffset + 1u];
  let layoutCenter = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let mapCenter = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
  let geoPoint = vec2<f32>(geo[geoOffset + 2u * index], geo[geoOffset + 2u * index + 1u]);
  let layoutPoint = vec2<f32>(layoutPositions[layoutPositionsOffset + 2u * index], layoutPositions[layoutPositionsOffset + 2u * index + 1u]);
  let mapped = mapCenter + (layoutPoint - layoutCenter) * scale;
  let result = mix(geoPoint, mapped, eased);
  morphed[morphedOffset + 2u * index] = result.x;
  morphed[morphedOffset + 2u * index + 1u] = result.y;`}),ie(D,{id:`airline-morph-segments`,invocationCount:c,bindings:[{name:`morphed`,view:r,type:`f32`,access:`read`},{name:`geo`,view:n,type:`f32`,access:`read`},{name:`sources`,view:e(`edge-sources`,d.getBuffer(f.graph.sourceVertices),`uint32`,c),type:`u32`,access:`read`},{name:`targets`,view:e(`edge-targets`,d.getBuffer(f.graph.targetVertices),`uint32`,c),type:`u32`,access:`read`},{name:`mask`,view:e(`edge-mask`,He,`uint32`,c),type:`u32`,access:`read`},{name:`parameters`,view:t,type:`f32`,access:`read`},{name:`segments`,view:e(`segments`,he,`float32`,c*4),type:`f32`,access:`read_write`},{name:`weights`,view:e(`weights`,ge,`float32`,c),type:`f32`,access:`read_write`}],body:`
  let a = sources[sourcesOffset + index];
  let b = targets[targetsOffset + index];
  segments[segmentsOffset + 4u * index] = morphed[morphedOffset + 2u * a];
  segments[segmentsOffset + 4u * index + 1u] = morphed[morphedOffset + 2u * a + 1u];
  segments[segmentsOffset + 4u * index + 2u] = morphed[morphedOffset + 2u * b];
  segments[segmentsOffset + 4u * index + 3u] = morphed[morphedOffset + 2u * b + 1u];
  // A pair that spans the antimeridian is a straight line across the whole map in geography.
  // Hide it at the start of the morph and fade it in once the endpoints have moved.
  let blend = parameters[parametersOffset];
  var weight = f32(mask[maskOffset + index]);
  if (abs(geo[geoOffset + 2u * a] - geo[geoOffset + 2u * b]) > 180.0) {
    weight = weight * smoothstep(0.35, 0.8, blend);
  }
  weights[weightsOffset + index] = weight;`})}let O=await D.compileAsync(),k=Ae(a),A=k.source.length,et=r.createBuffer(`bundle-positions`,k.positions),tt=r.createBuffer(`bundle-sources`,k.source),nt=r.createBuffer(`bundle-targets`,k.target),rt=r.createBuffer(`bundle-mask`,new Uint32Array(A).fill(1)),it=r.createBuffer(`bundle-category`,new Float32Array(A)),at=r.createParameterBuffer(`bundle-parameters`,`uint32`,5),j=null,M=0,N=0,P=[],F=e=>P.push({item:e,frames:0}),I=e=>{e.reader.stop(),F(e.resources)},L=!1,R=null,z=null,B=0,V=new Uint32Array,st=new Uint32Array,ct=new Uint32Array,H=null,U=null,gt=``,G=null,Ut=0,K=[],q=null,Wt=0,Gt=null,J={scale:40,centerX:0,centerY:0},Kt=!1,qt=0,Jt=e.options.morph,Yt=1,Xt=-1/0,Y=e.options.morph,X=null,Z=null,Zt=1,Qt=!1,$t=null,en={pagerank:1,degree:1,core:1,bridges:1},Q=-1,tn=null,nn=!1,rn,an,on=xe(a.lonLat),sn=new re(r,`layout-positions`,[{buffer:d.getBuffer(m),size:o*8},{buffer:d.getBuffer(fe),size:4},{buffer:d.getBuffer(pe),size:4}],e=>{if(L)return;let t=new Uint32Array(e,o*8);Gt={positions:new Float32Array(e.slice(0,o*8)),spatialCount:t[0],spatialOverflow:t[1]!==0},qn(Gt)}),cn=new re(r,`analysis`,ft(f),e=>{L||(R=pt(e.slice(0),o),zn(R))});function ln(){let{pageRankDamping:t,pageRankIterations:r,propagationRounds:i}=e.options;return dt(n,f,{pageRankDamping:t,pageRankIterations:r,propagationRounds:i},`airline-analysis-${++B}`)}function un(e){z&&F(z),z=e,cn.markStale(),dn=!0}let dn=!1,fn=!1,pn=r.track(new ne);function mn(){return pn.request({build:ln,commit:t=>{un(t),fn&&(fn=!1,H&&I(H),H=null,U=null,Sn()),e.requestLayers()},discard:e=>e.destroy()})}function hn(){let{resolution:t,rounds:n,minimumGain:r}=e.options;return`${t}|${n}|${r}`}async function gn(){let t=hn(),r=await mt(n,f,{resolution:e.options.resolution,rounds:Number(e.options.rounds),minimumGain:e.options.minimumGain},`airline-opt-${t}`,e=>{L||H!==r||(U=e,Bn(e))});return{run:r,key:t}}function _n(t){H&&I(H),H=t.run,gt=t.key,e.setReadout(`optimizerStatus`,`running 0%`)}function vn(e){e.run.reader.stop(),e.run.resources.destroy()}let yn=!1,bn=r.track(new ne);function xn(){return R?bn.request({build:gn,commit:t=>{_n(t),yn&&(yn=!1,Sn(),Un()),e.requestLayers()},discard:vn}):Promise.resolve(!1)}function Sn(){Ut++,K.length=0,G&&I(G),G=null}function Cn(){Sn(),Un(),wn(Ut)}function wn(e){setTimeout(()=>void Tn(e),60)}async function Tn(t){if(L||t!==Ut||!R)return;let r=K.length;if(r>=W.length)return;let i=W[r],a=await mt(n,f,{resolution:i,rounds:Number(e.options.rounds),minimumGain:e.options.minimumGain},`airline-sweep-${t}-${r}`,e=>{L||t!==Ut||(K.push({resolution:i,optimized:e.modularity,propagation:e.propagationModularity,continents:e.continentModularity}),I(a),G===a&&(G=null),Un(),wn(t))});if(L||t!==Ut){a.reader.stop(),a.resources.destroy();return}G=a}function En(){let t=e.options,r=new s(n,{id:`airline-layout-${++qt}`}),i=new Ze({id:`airline-layout`,topology:f.topology,positions:m,velocities:h,pinned:oe,reset:se,seed:20140601,iterationsPerFrame:Number(t.iterationsPerFrame),repulsion:t.repulsion,attraction:t.attraction,gravity:t.gravity,damping:t.layoutDamping,maxVelocity:t.maxVelocity});return t.layoutMode===`spatial`?new ot({id:`airline-layout-spatial`,layout:i,gridSize:Et,bounds:Dt,theta:t.theta,nearCellRadius:1,cellOffsets:le,vertexIds:ue,cellCenters:de,count:fe,overflow:pe}).addToGraph(r):i.addToGraph(r),r.compileAsync()}let Dn=r.track(new ne);function On(){return Dn.request({build:En,commit:t=>{q&&F(q),q=t,e.requestLayers()},discard:e=>e.destroy()})}async function kn(){let t=Number(e.options.pointsPerEdge),r=Number(e.options.densityResolution),i=new u(n,`airline-bundling-${++N}`),a=i.createBuffer(`paths`,A*t*8),o=new s(n,{id:`airline-bundling-${N}`});o.add(new ve({id:`bundling`,nodePositions:v(o,`positions`,et,`float32x2`,k.positions.length/2),sourceNodes:v(o,`sources`,tt,`uint32`,A),targetNodes:v(o,`targets`,nt,`uint32`,A),edgeMask:v(o,`mask`,rt,`uint32`,A),spatialContext:{coordinateSpace:`longitude-latitude`,datum:`EPSG:4326`,metric:`none`},pointsPerEdge:t,maximumIterations:Ct,densityResolution:r,parameters:at.importToGraph(o),output:{paths:v(o,`paths`,a,`float32x2`,A*t)}}));let c;try{c=i.track(await o.compileAsync())}catch(e){throw i.destroy(),e}return{resources:i,compiled:c,paths:a,pointsPerEdge:t}}function An(e){j&&F(j.resources),j=e,Nn(),M=3}let jn=r.track(new ne);function Mn(){return jn.request({build:kn,commit:t=>{An(t),e.requestLayers()},discard:e=>e.resources.destroy()})}function Nn(){let{bundleIterations:t,kernelRadius:n,decay:r,stiffness:i,stepScale:a}=e.options;at.write(_e({activeIterations:t,kernelRadius:n,lambda:r,smoothing:i,stepScale:a},`uint32`)),M=Math.max(M,2)}function Pn(){Qe.write(g({maximumSegmentLength:e.options.arcSegmentKm*1e3,minimumSegments:2})),Fn=!0}let Fn=!0;function In(){let t=e.options,n=t.autoFit?J:{scale:t.layoutScale,centerX:0,centerY:0};C.write(Float32Array.of(Y,n.scale,n.centerX,n.centerY,t.autoFit?At[0]:0,t.autoFit?At[1]:15,0,0))}function Ln(t,n){let r=e.options;r.morph!==Jt&&(Y=r.morph,Jt=r.morph),r.animateMorph&&(Y+=Yt*t/Math.max(r.morphSeconds,.5),Y>=1?(Y=1,Yt=-1):Y<=0&&(Y=0,Yt=1),n-Xt>1/20&&(Xt=n,Jt=Math.round(Y*100)/100,e.setOptions({morph:Jt})))}function Rn(e){let t=a.airports[e];return`${t.iata} (${t.city})`}function zn(t){V=ht(t.pageRank),st=ht(t.degree),ct=new Uint32Array(o),V.forEach((e,t)=>{ct[e]=t}),e.setReadout(`airports`,o),e.setReadout(`routes`,c),e.setReadout(`engineStatus`,t.adjacencyOverflow?`adjacency overflow: results are zeroed`:t.valid?`CSR complete, partitions valid`:`a partition failed validation`),e.setReadout(`residual`,t.residual.toExponential(1)),e.setReadout(`degeneracy`,t.degeneracy),e.setReadout(`coreStatus`,t.coreConverged?`core numbers converged`:`core numbers are upper bounds`),e.setReadout(`propagationStatus`,t.propagationConverged?`converged`:`stopped at its round budget`),e.setReadout(`topHub`,Rn(V[0])),e.setReadout(`topHubs`,Array.from(V.subarray(0,8),e=>a.airports[e].iata).join(`, `)),e.setReadout(`topByDegree`,Array.from(st.subarray(0,8),e=>a.airports[e].iata).join(`, `)),e.setReadout(`rankAgreement`,vt(t.pageRank,t.degree)),e.setReadout(`hubShare`,St(new Set(st.subarray(0,10)),a.source,a.target));let n=Array.from(t.degree).sort((e,t)=>e-t);e.setReadout(`medianDegree`,n[Math.floor(o/2)]),e.setReadout(`maxDegree`,n[o-1]),e.setReadout(`leafShare`,t.degree.reduce((e,t)=>e+(t===1?1:0),0)/o),e.setCost({records:c,note:`${e.options.pageRankIterations} PageRank rounds, ${e.options.propagationRounds} propagation rounds`}),Vn(t),Hn(t),Zn(),(!H||gt!==hn())&&xn()}function Bn(t){e.setReadout(`optimizerStatus`,t.valid?t.converged?`local optimum reached`:`stopped at its round budget`:`failed validation`),Zn(),Un()}function Vn(t){let n=Array.from(V.subarray(0,Nt)),r=new Set(st.subarray(0,Nt));e.setChart(`pageRankChart`,{kind:`bars`,values:n.map(e=>t.pageRank[e]*1e3),labels:n.map(e=>a.airports[e].iata),highlight:n.flatMap((e,t)=>r.has(e)?[]:[t]),yLabel:`PageRank x 1000`,height:150,formatY:e=>e.toFixed(1),onBarClick:e=>ir(n[e],!0),description:`The 20 airports with the highest PageRank. Highlighted bars are not among the 20 airports with the most connections. Click a bar to mark the airport on the map.`})}function Hn(t){let{counts:n,maximumLog:r}=yt(t.degree,24);e.setChart(`degreeChart`,{kind:`histogram`,values:n.map(e=>Math.log10(1+e)),xDomain:[0,r],xLabel:`connections per airport (log scale)`,yLabel:`airports (log scale)`,height:130,formatX:e=>String(Math.round(10**e)),formatY:e=>y(10**e-1),description:`Histogram of the number of distinct airports each airport connects to, with logarithmic bins on both axes.`})}function Un(){let t=K.map(e=>e.resolution),n=K.length===0?null:{kind:`line`,xLabel:`resolution gamma`,yLabel:`modularity Q`,height:160,xDomain:[W[0],W[W.length-1]],series:[{label:`optimized`,x:t,y:K.map(e=>e.optimized),color:0,points:!0},{label:`propagation`,x:t,y:K.map(e=>e.propagation),color:1},{label:`continents`,x:t,y:K.map(e=>e.continents),color:2,dashed:!0}],markers:[{x:e.options.resolution,label:`now`}],link:{option:`resolution`,label:e=>`gamma = ${e.toFixed(1)}`},formatY:e=>e.toFixed(2),description:`Modularity of three partitions scored at each resolution: the optimized communities, label propagation, and the six continents. Click or drag to set the resolution.`};e.setChart(`modularityChart`,n),e.setReadout(`sweepStatus`,K.length>=W.length?`${W.length} resolutions done`:`${K.length} of ${W.length} resolutions done`)}function Wn(){return R?e.options.communityMethod===`modularity`&&U?.valid?U.labels:R.propagation:null}function Gn(){return e.options.colorBy===`continent`?X:Z}function Kn(){let t=Wn(),n=Z;if(!t||!R||!n)return;let r=t!==R.propagation;e.setReadout(`communities`,n.groups.length),e.setReadout(`purity`,n.continentPurity),e.setReadout(`qCommunities`,r&&U?U.modularity:R.propagationModularity),e.setReadout(`qContinents`,r&&U?U.continentModularity:R.continentModularity),e.setReadout(`withinShare`,1-n.betweenCount/c),e.setReadout(`communityList`,n.groups.slice(0,6).map((e,t)=>{let n=a.airports[e.topAirport].iata,r=Oe[e.dominantContinent];return`${t+1}. ${y(e.size)} airports, hub ${n}, ${Math.round(e.dominantShare*100)}% ${r}`}).join(`
`))}function qn(t){let n=t.positions,r=new Float32Array(o),i=new Float32Array(o);for(let e=0;e<o;e++)r[e]=n[e*2],i[e]=n[e*2+1];r.sort(),i.sort();let s=Math.floor(o*.02),l=Math.min(o-1,Math.ceil(o*.98)),u=Math.max((r[l]-r[s])/2,1e-4)*1.05,d=Math.max((i[l]-i[s])/2,1e-4)*1.05,f={scale:Math.min(jt[0]/u,jt[1]/d),centerX:(r[l]+r[s])/2,centerY:(i[l]+i[s])/2},p=Kt?.3:1;Kt=!0,J={scale:J.scale+(f.scale-J.scale)*p,centerX:J.centerX+(f.centerX-J.centerX)*p,centerY:J.centerY+(f.centerY-J.centerY)*p};let m=new Float32Array(c),h=new Float32Array(c);for(let e=0;e<c;e++){let t=a.source[e],r=a.target[e];m[e]=a.distanceKm[e],h[e]=Math.hypot(n[t*2]-n[r*2],n[t*2+1]-n[r*2+1])}e.setReadout(`layoutCorrelation`,_t(m,h)),e.setReadout(`layoutSteps`,Wt),e.setReadout(`layoutStatus`,e.options.layoutMode===`spatial`?t.spatialOverflow||t.spatialCount<o?`spatial index: ${y(o-t.spatialCount)} airports outside the grid, layout frozen`:`spatial index holds every airport`:`exact all-pairs repulsion`)}function Jn(t,n){let{edgeFilter:r,minRecords:i}=e.options;return a.traffic[t]<i?!1:r===`all`?!0:r===`within`?n!==8:n===8}function Yn(e){let t=new Float32Array(c);for(let n=0;n<c;n++){let r=e.edgeSlots[n];t[n]=Jn(n,r)?tn?.[n]?Vt+r:r:NaN}return t}function Xn(){if(!X||!Z)return;Ke.write(X.nodeSlots),qe.write(Z.nodeSlots);let t=Yn(X),n=Yn(Z);We.write(t),Ge.write(n);let r=Gn(),i=r===X?t:n,a=r??Z,o=new Uint32Array(c),s=new Float32Array(A),l=new Uint32Array(A),u=0;for(let e=0;e<c;e++)o[e]=Number.isNaN(i[e])?0:1,u+=o[e];for(let e=0;e<A;e++){let t=k.original[e];s[e]=i[t],l[e]=o[t]}let d=new Uint32Array(c),f=0,p=c;for(let e=0;e<c;e++)a.edgeSlots[e]===8?d[--p]=e:d[f++]=e;Ue.write(a.edgeSlots),He.write(o),b.write(d),it.write(s),rt.write(l),j&&(M=Math.max(M,2)),e.setReadout(`edges`,`${y(u)} of ${y(c)}`)}function Zn(){let t=Wn();!R||!t||(X=bt(`continent`,l,R.pageRank,a.continent,a.source,a.target),Z=bt(`community`,t,R.pageRank,a.continent,a.source,a.target),Xn(),Kn(),Qn(),er(),nr(),sr(),e.requestLayers())}function Qn(){let t=Gn()??Z;if(!t)return;e.setReadout(`betweenShare`,t.betweenCount/c);let n=ht(xt(t.labels,a.source,a.target)).subarray(0,Lt);e.setReadout(`bridgeAirports`,Array.from(n,e=>a.airports[e].iata).join(`, `))}function $n(e){if(!R)return null;switch(e){case`pagerank`:return R.pageRank;case`degree`:return Float32Array.from(R.degree);case`core`:return Float32Array.from(R.coreNumber);case`bridges`:{let e=Gn()??Z;return e?xt(e.labels,a.source,a.target):null}default:return null}}function er(){if(!R)return;for(let e of[`pagerank`,`degree`,`core`,`bridges`]){let t=$n(e);if(!t)continue;let n=0;for(let e=0;e<o;e++)n=Math.max(n,t[e]);en[e]=Math.max(n,1e-9)}let t=e.options.sizeBy,n=$n(t);if(!n||t===`uniform`){Qt=!1,$t=null;return}$t=n,Je.write(n),x.write(ht(n)),Zt=en[t],Qt=!0}function tr(t){return e.options.sizeBy===`uniform`||!Qt||!$t?Pt:Se($t[t],Zt,e.options.maxRadius,{minRadiusPixels:Ft})}function nr(){if(!X||!Z)return;let t=Ie(e.ground()),n=(e,n)=>{let r=[],i=0,o=0;for(let s of e.groups){if(s.slot===7&&!n){i+=s.size,o++;continue}r.push({color:t[s.slot],label:n?Oe[s.label]:`${a.airports[s.topAirport].iata} community`,count:s.size})}return o>0&&r.push({color:t[7],label:`${y(o)} smaller communities`,count:i}),r},r={continent:n(X,!0),community:n(Z,!1),between:t[8],betweenCount:(Gn()??Z).betweenCount,comparing:nn,sizeMaxima:{...en}};e.setLegendData(`network`,r)}function rr(t){Q=t;let n=new Float32Array(o),r=[];if(t>=0){tn=new Uint8Array(c);let e=[];for(let n=0;n<c;n++){let r=a.source[n],i=a.target[n];(r===t||i===t)&&(tn[n]=1,e.push(r===t?i:r))}n[t]=1;for(let t of e)n[t]=1;e.sort((e,t)=>(R?.pageRank[t]??0)-(R?.pageRank[e]??0)||e-t);for(let t of e.slice(0,zt))r.push({kind:`point`,coordinate:$(t),radiusPixels:tr(t)+2.5});r.push({kind:`point`,coordinate:$(t),radiusPixels:tr(t)+4,pulse:!0})}else tn=null;S.write(n),e.setHighlight(r.length?r:null),Xn(),sr(),e.requestLayers()}function ir(t,n=!1){if(e.options.ego&&e.options.view===`arcs`){rr(t===Q&&!n?-1:t);return}e.setHighlight({kind:`point`,coordinate:$(t),radiusPixels:tr(t)+4,pulse:!0})}function $(e){return[a.lonLat[e*2],a.lonLat[e*2+1]]}function ar(e,t){let n=a.airports[e];return{kind:`point`,id:`airport:${n.iata}`,coordinate:$(e),text:n.iata,detail:n.city,marker:`dot`,rank:`subject`,priority:t}}function or(t){if(!R)return[];let n=e.options.sizeBy,r=n===`uniform`||n===`bridges`?R.degree:$n(n);return r?Array.from(ht(r).subarray(0,t)):[]}function sr(){let t=e.options,n=[],r=Gn()??Z;if(R&&t.view!==`morph`&&t.labels!==`none`){if(t.labels===`hubs`)for(let e of or(It))n.push(ar(e,6));else if(t.labels===`bridges`&&r){let e=xt(r.labels,a.source,a.target);for(let t of ht(e).subarray(0,Lt))n.push(ar(t,6))}else if(r){for(let e of r.groups.slice(0,Rt))n.push(ar(e.topAirport,6));if(t.labels===`disagreement`){let e=r.groups.slice(0,7).reduce((e,t)=>t.dominantShare<e.dominantShare?t:e);n.push({kind:`note`,id:`note:disagreement`,coordinate:$(e.topAirport),title:ye(`{share:percent} {continent}`,{share:e.dominantShare,continent:Oe[e.dominantContinent]}),text:`of the ${a.airports[e.topAirport].iata} community; the rest sit on other continents`,priority:8})}}}Q>=0&&t.view===`arcs`&&n.push({...ar(Q,9),marker:`ring`,tone:`signal`}),e.setAnnotations(`airline-network`,n.length?n:null)}function cr(t){let n=a.lonLat[t*2],r=a.lonLat[t*2+1];if(e.options.view!==`morph`||!Gt)return[n,r];let i=e.options,o=i.autoFit?J:{scale:i.layoutScale,centerX:0,centerY:0},s=i.autoFit?At[0]:0,c=i.autoFit?At[1]:15,l=Y*Y*(3-2*Y),u=s+(Gt.positions[t*2]-o.centerX)*o.scale,d=c+(Gt.positions[t*2+1]-o.centerY)*o.scale;return[n+(u-n)*l,r+(d-r)*l]}function lr(t){let n=e.getViewport();if(!n||!R)return-1;let r=t.pixel,i;if(e.options.view===`morph`)i=Array.from({length:o},(e,t)=>t);else{if(!t.coordinate)return-1;let e=((t.coordinate[0]+180)%360+360)%360-180,a=n.unproject([r[0]+Ht,r[1]]),o=ae([t.coordinate[0],t.coordinate[1]],[a[0],a[1]]);i=on.within([e,t.coordinate[1]],o)}let a=-1,s=1/0;for(let e of i){let[i,o]=cr(e),c=t.coordinate?Math.round((t.coordinate[0]-i)/360)*360:0,[l,u]=n.project([i+c,o]),d=Math.hypot(l-r[0],u-r[1]),f=tr(e);if(d>Math.max(f+3,7))continue;let p=d-f;p<s&&(s=p,a=e)}return a}function ur(t){let n=lr(t);if(n<0||!R)return null;let r=a.airports[n],i=Wn(),s=Z?.groups.find(e=>e.label===i?.[n]),c=Ie(e.ground()),l=(e.options.colorBy===`continent`?X:Z)?.nodeSlots[n],u=e.options.sizeBy,d=l===void 0||e.options.colorBy===`none`?void 0:c[l],f={label:`Connections`,value:y(R.degree[n]),unit:`airports`},p={label:`PageRank rank`,value:be(ct[n]+1),unit:`of ${y(o)}`},m={label:`Core number`,value:y(R.coreNumber[n])},h=u===`pagerank`?p:u===`core`?m:f,g=[{...h,swatch:d,emphasis:!0},...[f,p,m].filter(e=>e!==h),{label:`Continent`,value:Oe[a.continent[n]]}];return s&&g.push({label:`Community`,value:`${a.airports[s.topAirport].iata} community`,unit:`${y(s.size)} airports`,swatch:c[s.slot]}),{title:r.iata,subtitle:`${r.name}, ${r.city}, ${r.country}`,rows:g,anchor:$(n),highlight:{kind:`point`,coordinate:$(n),radiusPixels:tr(n)+3}}}Pn(),Nn(),un(await ln()),q=await En(),e.options.view===`bundles`&&An(await kn()),e.setReadout(`airports`,o),e.setReadout(`routes`,c),e.setReadout(`resolutionNow`,e.options.resolution.toFixed(1)),e.setReadout(`edges`,`${y(c)} of ${y(c)}`),e.setFurniture({title:{sample:`${y(o)} airports, ${y(c)} route pairs`}});function dr(e){clearTimeout(rn),e&&(fn=!0),rn=setTimeout(()=>{L||mn()},Bt)}function fr(e){clearTimeout(an),e&&(yn=!0),an=setTimeout(()=>{L||xn()},Bt)}return{getCompiledGraphs:()=>[z,H?.compiled,q,O,E,j?.compiled].filter(Boolean),setOption(t){switch(t){case`pageRankDamping`:case`pageRankIterations`:dr(!1);break;case`propagationRounds`:dr(!0);break;case`resolution`:e.setReadout(`resolutionNow`,e.options.resolution.toFixed(1)),Un(),fr(!1);break;case`rounds`:case`minimumGain`:fr(!0);break;case`communityMethod`:Zn();break;case`colorBy`:case`edgeFilter`:case`minRecords`:Xn(),Qn(),er(),sr();break;case`sizeBy`:er(),sr();break;case`labels`:sr();break;case`ego`:!e.options.ego&&Q>=0&&rr(-1);break;case`arcSegmentKm`:Pn();break;case`view`:e.options.view===`bundles`&&!j&&!jn.pending&&Mn(),e.options.view===`arcs`&&(Fn=!0),e.options.view!==`arcs`&&Q>=0&&rr(-1),sr();break;case`repulsion`:case`attraction`:case`gravity`:case`layoutDamping`:case`maxVelocity`:case`iterationsPerFrame`:case`layoutMode`:case`theta`:On();break;case`bundleIterations`:case`kernelRadius`:case`decay`:case`stiffness`:case`stepScale`:Nn();break;case`pointsPerEdge`:case`densityResolution`:Mn();break;default:}e.requestLayers()},onAction(e){e===`resetLayout`?(d.getBuffer(se).write(Uint32Array.of(1)),Wt=0,J={scale:40,centerX:0,centerY:0},Kt=!1):e===`rerunSweep`&&Cn()},onThemeChange(){e.requestLayers()},onGroundChange(){nr(),e.requestLayers()},getTooltip:ur,onClick(t){if(!e.options.ego||e.options.view!==`arcs`||!R)return!1;let n=lr(t);return n<0?Q<0?!1:(rr(-1),!0):(ir(n),!0)},encode(t,n){for(let e=P.length-1;e>=0;e--)++P[e].frames>Ot&&(P[e].item.destroy(),P.splice(e,1));let r=e.options;Ln(n.deltaSeconds,n.timeSeconds),dn&&z?(z.encode(t,{parameters:void 0}),cn.request(t),dn=!1):cn.flush(t);let i=!!(H&&R&&!H.execution.completed);if(H&&i){let n=H.execution.encodeNext(t,{parameters:void 0});e.setReadout(`optimizerStatus`,`running ${Math.round(n.progress*100)}%`),n.completed&&H.reader.request(t)}else H?.reader.flush(t);if(G&&!i&&!G.execution.completed){let n=G.execution.encodeNext(t,{parameters:void 0});e.setReadout(`sweepStatus`,`${K.length} of ${W.length} resolutions done · ${Math.round(n.progress*100)}% current`),n.completed&&G.reader.request(t)}else G?.reader.flush(t);Fn?(E.encode(t,{parameters:void 0}),we(t,w.count,w.drawCommands),$e.request(t),Fn=!1):$e.flush(t),j&&M>0&&r.view===`bundles`&&(j.compiled.encode(t,{parameters:void 0}),M--),q&&r.layoutRunning&&r.view===`morph`?(q.encode(t,{parameters:void 0}),Wt+=Number(r.iterationsPerFrame),n.frameIndex%kt===0||Wt<=8?sn.request(t):sn.flush(t)):sn.flush(t),r.view===`morph`&&(In(),O.encode(t,{parameters:void 0}),!r.layoutRunning&&!Gt&&sn.request(t))},getLayers(){let t=e.options,n=[],r=e.ground(),i=r===`dark`?Ve:Re,a=Ie(r),s=Me(r),l=je[r],u=Pe[r],d=!!(X&&Z),f=t.view===`arcs`&&e.getCompare()!==null&&d;f!==nn&&(nn=f,queueMicrotask(()=>{L||nr()}));let m=!f&&(t.colorBy===`none`||!d),h=f?[{side:`a`,partition:`continent`},{side:`b`,partition:`community`}]:[{side:void 0,partition:t.colorBy===`continent`?`continent`:`community`}],g=Q>=0&&t.view===`arcs`,_=m||t.edgeFilter!==`all`?[`all`]:[`within`,`between`];for(let{side:e,partition:a}of h){let o=`${e??`x`}-${a}`,l=a===`continent`?We:Ge,d={palette:s,neutral:u,colorMode:m?`neutral`:`category`,opacity:t.edgeOpacity,compareSide:e,egoActive:g,dimAlpha:Fe,parameters:i};if(t.view===`arcs`)for(let e of _)for(let t of Tt)n.push(new ze({id:`network-arcs-${o}-${e}-${t}`,...d,drawPass:e,positionOffset:[t,0],positions:w.positions,pathOffsets:w.offsets,pathOffsetCount:w.offsetCount,vertexCount:w.count,drawCommands:w.drawCommands,categories:l,widthPixels:.75}));else if(t.view===`bundles`&&j)for(let e of _)n.push(new Be({id:`network-bundles-${j.pointsPerEdge}-${N}-${e}`,...d,drawPass:e,paths:j.paths,pointsPerPath:j.pointsPerEdge,pathCount:A,categories:it,opacity:Math.min(1,t.edgeOpacity*1.4)}));else t.view===`morph`&&n.push(new te({id:`network-morph-edges`,coordinateSystem:Mt,segments:he,weights:ge,ids:b,instanceCount:c,widthPixels:.8,values:Ue,valueFormat:`uint32`,colormap:m?`uniform`:`category`,color:u,palette:s,blending:r===`dark`?`additive`:`normal`,opacity:t.edgeOpacity}))}let v=t.sizeBy!==`uniform`&&Qt;for(let{side:e,partition:r}of h){let i=r===`continent`?Ke:qe,s=t.view===`morph`?[0]:Tt;for(let r of s)n.push(new ee({id:`network-airports-${e??`x`}-${r}`,coordinateSystem:Mt,positions:t.view===`morph`?me:p,instanceCount:o,ids:v?x:null,sizeValues:v?Je:null,sizeMaximumValue:Zt,radiusPixels:v?t.maxRadius:Pt,radiusMinPixels:v?Ft:void 0,shape:`circle`,colormap:m?`uniform`:`category`,color:l,...m?{}:{values:i,valueFormat:`uint32`},palette:a,outlineColor:Le,outlineWidthPixels:Ne,...g?{instanceChannels:S,channelStride:1,channels:{highlight:0},dimOpacity:.2}:{},positionOffset:[r,0],compareSide:e,opacity:.95}))}return n},destroy(){L=!0,clearTimeout(rn),clearTimeout(an),cn.stop(),sn.stop(),$e.stop(),H?.reader.stop(),G?.reader.stop();for(let e of P)e.item.destroy();P.length=0,G?.resources.destroy(),H?.resources.destroy(),j?.resources.destroy(),z?.destroy(),q?.destroy(),O.destroy(),E.destroy(),r.destroy()}}}export{G as createAirlineNetwork};