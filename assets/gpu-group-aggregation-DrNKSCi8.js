import{D as e,I as t,M as n,R as r,S as i,j as a,k as o,w as s,x as c,z as l}from"./resources-CwvhEOBe.js";import{i as u,n as d,o as f,r as p,t as m}from"./gpu-subgroup-utils-BuV3X9Xp.js";var h=256,g=256,_=16,v=Uint32Array.BYTES_PER_ELEMENT,y=class{id;keys;values;output;mask;operation;constructor(e){this.id=e.id??`gpu-group-aggregation`,this.keys=e.keys,this.values=e.values,this.output=e.output,this.mask=e.mask,this.operation=e.operation??`count`;for(let[e,t]of S(this.keys).entries())b(t,`uint32`,`${this.id} keys chunk ${e}`);if(this.output.length===0)throw Error(`${this.id} output must contain at least one group`);if(![`count`,`sum`,`min`,`max`,`mean`].includes(this.operation))throw Error(`${this.id} operation must be count, sum, min, max, or mean`);if(this.operation===`count`){if(a(this.output,`${this.id} count output`),this.values)throw Error(`${this.id} count operation must not provide values`)}else{if(n(this.output,[`float32`],`${this.id} statistic output`),!this.values)throw Error(`${this.id} ${this.operation} operation requires values`);for(let[e,t]of C(this.values).entries())b(t,`float32`,`${this.id} values chunk ${e}`);if(w(this.keys,this.values,`${this.id} keys and values`),this.operation===`mean`&&this.keys.length>4294967295)throw Error(`${this.id} mean input length must fit in uint32 group counts`)}if(S(this.keys).some(e=>e.buffer===this.output.buffer))throw Error(`${this.id} keys and output must use separate buffers`);if(this.values&&C(this.values).some(e=>e.buffer===this.output.buffer))throw Error(`${this.id} values and output must use separate buffers`);if(this.mask){for(let[e,t]of S(this.mask).entries())if(b(t,`uint32`,`${this.id} mask chunk ${e}`),t.buffer===this.output.buffer)throw Error(`${this.id} mask and output must use separate buffers`);w(this.keys,this.mask,`${this.id} keys and mask`)}}getCommandNodes(e){let t=[],n=S(this.keys),i=this.mask?S(this.mask):void 0,a=this.values?C(this.values):void 0;if(n.some(t=>t.buffer.graph!==e)||i?.some(t=>t.buffer.graph!==e)||a?.some(t=>t.buffer.graph!==e)||this.output.buffer.graph!==e)throw Error(`${this.id} views must belong to the target graph`);if(this.values){let t=u(e,[this.keys,this.values,this.mask??this.keys]);n=t.map(([e])=>e),a=t.map(([,e])=>e),i=this.mask?t.map(([,,e])=>e):void 0}else if(this.mask){let t=u(e,[this.keys,this.mask]);n=t.map(([e])=>e),i=t.map(([,e])=>e)}if(this.operation===`count`){let a=this.output;t.push(...T(e,this.id,a));let o=a.length<=g?`local`:`global`;for(let s=0;s<n.length;s++){let c=n[s];c.length!==0&&t.push(...E(e,{id:this.keys instanceof r||n.length>1?`${this.id}-chunk-${s}-${o}`:`${this.id}-${o}`,keys:c,output:a,mask:i?.[s],dispatchLayout:N(c.length,e.device.limits.maxComputeWorkgroupsPerDimension)}))}return t}let o=this.output,c=this.operation,l=c===`mean`?s(e,`${this.id}-counts`,`uint32`,o.length):void 0;t.push(...D(e,this.id,o,c,l));for(let s=0;s<n.length;s++){let u=n[s];u.length!==0&&t.push(...O(e,{id:this.keys instanceof r||n.length>1?`${this.id}-chunk-${s}`:this.id,keys:u,values:a[s],mask:i?.[s],output:o,operation:c,counts:l,dispatchLayout:N(u.length,e.device.limits.maxComputeWorkgroupsPerDimension)}))}return c!==`sum`&&t.push(...k(e,this.id,o,c,l)),t}};function b(e,t,n){if(e.format!==t||e.rowByteLength!==v||e.byteStride<v||e.byteStride%v!==0||e.byteOffset%v!==0)throw Error(`${n} must be a uint32-aligned scalar ${t} GPU data view`)}function x(e){return e.byteStride/v}function S(e){return e instanceof r?e.data:[e]}function C(e){return e instanceof r?e.data:[e]}function w(e,t,n){if(e.length!==t.length)throw Error(`${n} lengths must match`)}function T(e,t,n){let r=[],a=N(n.length,e.device.limits.maxComputeWorkgroupsPerDimension),s=`
const GROUP_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(0) var<storage, read_write> outputCounts: array<atomic<u32>>;

@compute @workgroup_size(${h})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${i(a,h)}
  if (index < GROUP_COUNT) {
    atomicStore(&outputCounts[OUTPUT_OFFSET + index], 0u);
  }
}`;return r.push(...P(e,{id:`${t}-clear`,source:s,resources:[{buffer:n,usage:`storage-write`}],bindings:{outputCounts:n},dispatchSize:a,record:{moduleId:`GPUGroupAggregation/clear`,invocationCount:n.length}})),r}function E(e,t){let n=[],r=t.output.length<=g,i=r&&t.output.length<=_&&m(e.device)===`subgroups`,a=t.mask?`@group(0) @binding(1) var<storage, read> selectionMask: array<u32>;`:``,s=t.mask?2:1,c=t.mask?`selectionMask[${o(t.mask)}u + index * ${x(t.mask)}u] != 0u`:`true`,l=i?p(`accepted`,`groupIndex`,`localCounts`,t.output.length):`  if (accepted) { atomicAdd(&localCounts[groupIndex], 1u); }`,u=r?`${l}
  workgroupBarrier();
  if (lane < GROUP_COUNT) {
    atomicAdd(&outputCounts[OUTPUT_OFFSET + lane], atomicLoad(&localCounts[lane]));
  }`:`if (accepted) { atomicAdd(&outputCounts[OUTPUT_OFFSET + groupIndex], 1u); }`,f=`
${i?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const GROUP_COUNT: u32 = ${t.output.length}u;
const KEYS_OFFSET: u32 = ${o(t.keys)}u;
const KEYS_STRIDE: u32 = ${x(t.keys)}u;
const OUTPUT_OFFSET: u32 = ${o(t.output)}u;
@group(0) @binding(0) var<storage, read> groupKeys: array<u32>;
${a}
@group(0) @binding(${s}) var<storage, read_write> outputCounts: array<atomic<u32>>;
${r?`var<workgroup> localCounts: array<atomic<u32>, ${t.output.length}>;`:``}
${i?d():``}

@compute @workgroup_size(${h}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>${i?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  let workgroupIndex = (workgroupId.z * ${t.dispatchLayout.y}u + workgroupId.y) * ${t.dispatchLayout.x}u + workgroupId.x;
  let index = workgroupIndex * ${h}u + localId.x;
  let lane = localId.x;
  ${r?`if (lane < GROUP_COUNT) { atomicStore(&localCounts[lane], 0u); }
  workgroupBarrier();`:``}
  var accepted = false;
  var groupIndex = 0u;
  if (index < ELEMENT_COUNT && ${c}) {
    groupIndex = groupKeys[KEYS_OFFSET + index * KEYS_STRIDE];
    accepted = groupIndex < GROUP_COUNT;
  }
  ${u}
}`,v=[{buffer:t.keys,usage:`storage-read`},...t.mask?[{buffer:t.mask,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`}];return n.push(...P(e,{id:t.id,source:f,resources:v,bindings:{groupKeys:t.keys,...t.mask?{selectionMask:t.mask}:{},outputCounts:t.output},dispatchSize:t.dispatchLayout,record:{moduleId:`GPUGroupAggregation/count/${r?`local`:`global`}${i?`/subgroups`:``}`,invocationCount:t.keys.length,formula:`gpu-core.group-aggregation`}})),n}function D(e,t,n,r,a){let s=[],c=N(n.length,e.device.limits.maxComputeWorkgroupsPerDimension),l=r===`min`?`0xffffffffu`:`0u`,u=a?`@group(0) @binding(1) var<storage, read_write> outputCounts: array<atomic<u32>>;`:``,d=a?`atomicStore(&outputCounts[${o(a)}u + index], 0u);`:``,f=`
const GROUP_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<atomic<u32>>;
${u}
@compute @workgroup_size(${h}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${i(c,h)}
  if (index < GROUP_COUNT) {
    atomicStore(&outputValues[OUTPUT_OFFSET + index], ${l});
    ${d}
  }
}`;return s.push(...P(e,{id:r===`sum`?`${t}-clear`:`${t}-initialize`,source:f,resources:[{buffer:n,usage:`storage-write`},...a?[{buffer:a,usage:`storage-write`}]:[]],bindings:{outputValues:n,...a?{outputCounts:a}:{}},dispatchSize:c,record:{moduleId:`GPUGroupAggregation/${r===`sum`?`clear`:`initialize`}`,invocationCount:n.length}})),s}function O(e,t){let n=[],r=t.output.length<=_&&m(e.device)===`subgroups`,i=t.mask?`@group(0) @binding(2) var<storage, read> selectionMask: array<u32>;`:``,a=t.mask?3:2,s=t.counts?`@group(0) @binding(${a+1}) var<storage, read_write> outputCounts: array<atomic<u32>>;`:``,c=t.mask?`selectionMask[${o(t.mask)}u + index * ${x(t.mask)}u] != 0u`:`true`,l=r?M(t.operation,t.output.length,t.counts):`  if (accepted) {
    ${j(t.operation,`groupIndex`)}
    ${t.counts?`atomicAdd(&outputCounts[${o(t.counts)}u + groupIndex], 1u);`:``}
  }`,u=`
${r?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const GROUP_COUNT: u32 = ${t.output.length}u;
const KEYS_OFFSET: u32 = ${o(t.keys)}u;
const VALUES_OFFSET: u32 = ${o(t.values)}u;
const KEYS_STRIDE: u32 = ${x(t.keys)}u;
const VALUES_STRIDE: u32 = ${x(t.values)}u;
const OUTPUT_OFFSET: u32 = ${o(t.output)}u;
@group(0) @binding(0) var<storage, read> groupKeys: array<u32>;
@group(0) @binding(1) var<storage, read> inputValues: array<f32>;
${i}
@group(0) @binding(${a}) var<storage, read_write> outputValues: array<atomic<u32>>;
${s}

${A(t.operation)}
${r?d():``}

@compute @workgroup_size(${h}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>${r?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  let workgroupIndex = (workgroupId.z * ${t.dispatchLayout.y}u + workgroupId.y) * ${t.dispatchLayout.x}u + workgroupId.x;
  let index = workgroupIndex * ${h}u + localId.x;
  var accepted = false;
  var groupIndex = 0u;
  var value = 0.0;
  if (index < ELEMENT_COUNT && ${c}) {
    groupIndex = groupKeys[KEYS_OFFSET + index * KEYS_STRIDE];
    value = inputValues[VALUES_OFFSET + index * VALUES_STRIDE];
    let finiteValue = value == value && abs(value) <= 3.402823466e+38;
    accepted = groupIndex < GROUP_COUNT && finiteValue;
  }
${l}
}`,f=[{buffer:t.keys,usage:`storage-read`},{buffer:t.values,usage:`storage-read`},...t.mask?[{buffer:t.mask,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`},...t.counts?[{buffer:t.counts,usage:`storage-read-write`}]:[]];return n.push(...P(e,{id:`${t.id}-${t.operation}`,source:u,resources:f,bindings:{groupKeys:t.keys,inputValues:t.values,...t.mask?{selectionMask:t.mask}:{},outputValues:t.output,...t.counts?{outputCounts:t.counts}:{}},dispatchSize:t.dispatchLayout,record:{moduleId:`GPUGroupAggregation/accumulate-${t.operation}${r?`/subgroups`:``}`,invocationCount:t.keys.length,formula:`gpu-core.group-aggregation`}})),n}function k(e,t,n,r,a){let s=[],c=N(n.length,e.device.limits.maxComputeWorkgroupsPerDimension),l=a?`@group(0) @binding(1) var<storage, read> outputCounts: array<u32>;`:``,u=r===`mean`?`let count = outputCounts[${o(a)}u + index];
    if (count == 0u) {
      outputValues[OUTPUT_OFFSET + index] = 0x7fc00000u;
    } else {
      let sum = bitcast<f32>(outputValues[OUTPUT_OFFSET + index]);
      outputValues[OUTPUT_OFFSET + index] = bitcast<u32>(sum / f32(count));
    }`:`let orderedValue = outputValues[OUTPUT_OFFSET + index];
    if (orderedValue == ${r===`min`?`0xffffffffu`:`0u`}) {
      outputValues[OUTPUT_OFFSET + index] = 0x7fc00000u;
    } else {
      outputValues[OUTPUT_OFFSET + index] = bitcast<u32>(decodeOrderedFloat(orderedValue));
    }`,d=r===`mean`?``:`fn decodeOrderedFloat(value: u32) -> f32 {
  let bits = select(~value, value ^ 0x80000000u, (value & 0x80000000u) != 0u);
  return bitcast<f32>(bits);
}`,f=`
const GROUP_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<u32>;
${l}
${d}
@compute @workgroup_size(${h}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${i(c,h)}
  if (index < GROUP_COUNT) {
    ${u}
  }
}`;return s.push(...P(e,{id:`${t}-finalize`,source:f,resources:[{buffer:n,usage:`storage-read-write`},...a?[{buffer:a,usage:`storage-read`}]:[]],bindings:{outputValues:n,...a?{outputCounts:a}:{}},dispatchSize:c,record:{moduleId:`GPUGroupAggregation/finalize-${r}`,invocationCount:n.length,formula:`gpu-core.group-aggregation`}})),s}function A(e){return e===`min`||e===`max`?`fn encodeOrderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`:`fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var oldBits = atomicLoad(destination);
  loop {
    let newBits = bitcast<u32>(bitcast<f32>(oldBits) + value);
    let result = atomicCompareExchangeWeak(destination, oldBits, newBits);
    if (result.exchanged) { break; }
    oldBits = result.old_value;
  }
}`}function j(e,t,n=`value`){return e===`min`||e===`max`?`${e===`min`?`atomicMin`:`atomicMax`}(&outputValues[OUTPUT_OFFSET + ${t}], encodeOrderedFloat(${n}));`:`atomicAddFloat(&outputValues[OUTPUT_OFFSET + ${t}], ${n});`}function M(e,t,n){let r=e===`min`||e===`max`;return`
  var subgroupPending = accepted;
  for (var subgroupGroup = 0u; subgroupGroup < ${t}u; subgroupGroup++) {
    let pendingBallot = subgroupBallot(subgroupPending);
    let hasPending = any(pendingBallot != vec4<u32>(0u));
    let leaderInvocation = getFirstBallotLane(pendingBallot);
    let leaderKey = subgroupShuffle(groupIndex, leaderInvocation);
    let matchingKey = hasPending && subgroupPending && groupIndex == leaderKey;
    let matchingBallot = subgroupBallot(matchingKey);
    let selectedValue = ${r?`select(${e===`min`?`0xffffffffu`:`0u`}, encodeOrderedFloat(value), matchingKey)`:`select(0.0, value, matchingKey)`};
    let aggregatedValue = ${e===`min`?`subgroupMin(selectedValue)`:e===`max`?`subgroupMax(selectedValue)`:`subgroupAdd(selectedValue)`};
    if (hasPending && subgroupInvocationId == leaderInvocation) {
      ${r?`${e===`min`?`atomicMin`:`atomicMax`}(&outputValues[OUTPUT_OFFSET + leaderKey], aggregatedValue);`:j(e,`leaderKey`,`aggregatedValue`)}
      ${n?`atomicAdd(&outputCounts[${o(n)}u + leaderKey], getBallotLaneCount(matchingBallot));`:``}
    }
    subgroupPending = subgroupPending && !matchingKey;
  }`}function N(e,t){return c(`GPUGroupAggregation`,e,h,t)}function P(n,r){let i=[],a=r.dispatchSize?r.dispatchSize.x*r.dispatchSize.y*r.dispatchSize.z:r.dispatchCount??1;i.push(t({id:r.id,resources:r.resources,workload:{operation:`GPUGroupAggregation`,commandCount:1,maximumWorkgroupCount:a,maximumInvocationCount:a*h},compile:({device:t})=>{let n=new l(t,{id:r.id,source:r.source,shaderLayout:{bindings:Object.keys(r.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:t,getBuffer:i})=>{let a={};for(let[t,n]of Object.entries(r.bindings))a[t]=e(n,i);r.dispatchSize?n.dispatch(t,{bindings:a,x:r.dispatchSize.x,y:r.dispatchSize.y,z:r.dispatchSize.z}):n.dispatch(t,{bindings:a,x:r.dispatchCount})},destroy:()=>n.destroy()}}}));let o=f(n,()=>({op:`GPUGroupAggregation`,moduleId:r.record.moduleId,source:r.source,workgroupSize:[h,1,1],workgroups:r.dispatchSize?[r.dispatchSize.x,r.dispatchSize.y,r.dispatchSize.z]:[r.dispatchCount??1,1,1],invocationCount:r.record.invocationCount,formula:r.record.formula,bindings:Object.entries(r.bindings).map(([e,t])=>{let n=r.resources.find(e=>e.buffer===t)?.usage;return{name:e,view:t,access:n===`storage-read`?`read`:n===`storage-read-write`?`read_write`:`write`}})}));return o&&(i[0].annotation=o),i}export{y as t};