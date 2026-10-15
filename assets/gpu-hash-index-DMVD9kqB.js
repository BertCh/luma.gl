import{D as e,I as t,T as n,j as r,k as i,w as a,z as o}from"./resources-CwvhEOBe.js";import{a as s,i as c,o as l}from"./gpu-subgroup-utils-BuV3X9Xp.js";var u=256,d=4,f=4294967295,p=f,m=class{id;keys;values;firstValue;tableKeys;tableValues;statistics;maxProbeCount;stats;updatePolicy=`rebuild`;constructor(e){this.id=e.id??`gpu-hash-index`,this.keys=e.keys,this.values=e.values,this.firstValue=e.firstValue??0,this.tableKeys=e.tableKeys,this.tableValues=e.tableValues,this.statistics=e.statistics,this.maxProbeCount=e.maxProbeCount??this.tableKeys.length;for(let[e,t]of[[this.keys,`keys`],...this.values?[[this.values,`values`]]:[],[this.tableKeys,`tableKeys`],[this.tableValues,`tableValues`],[this.statistics,`statistics`]])for(let n of s(e))r(n,`${this.id} ${t}`);if(!w(this.tableKeys.length))throw Error(`${this.id} table capacity must be a positive power of two`);if(this.tableKeys.length>f)throw Error(`${this.id} table capacity must fit in uint32`);if(this.tableValues.length!==this.tableKeys.length)throw Error(`${this.id} table key and value capacities must match`);if(this.statistics.length<6)throw Error(`${this.id} statistics must contain six uint32 rows`);if(this.values&&this.values.length!==this.keys.length)throw Error(`${this.id} keys and values lengths must match`);if(this.values&&e.firstValue!==void 0)throw Error(`${this.id} values and firstValue are mutually exclusive`);if(!Number.isSafeInteger(this.firstValue)||this.firstValue<0||this.firstValue>f||this.keys.length>0&&this.firstValue+this.keys.length-1>f)throw Error(`${this.id} generated values must fit in uint32`);S(this.id,this.maxProbeCount,this.tableKeys.length,this.keys.length),C(this.id,[this.keys,...this.values?[this.values]:[]],[this.tableKeys,this.tableValues,this.statistics]),this.stats=Object.freeze({capacity:this.tableKeys.length,maxProbeCount:this.maxProbeCount,tableByteLength:this.tableKeys.length*8,statisticsByteLength:24,outputByteLength:this.tableKeys.length*8+24})}getCommandNodes(e){let t=[];if([this.keys,...this.values?[this.values]:[],this.tableKeys,this.tableValues,this.statistics].flatMap(s).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);let n=this.values?c(e,[this.keys,this.values]):c(e,[this.keys]),r=this.firstValue,i=[];for(let[e,t]of n)i.push({keys:e,values:t,firstValue:r}),r+=e.length;return t.push(...h(e,this,i)),t}};function h(e,t,n){let r=[],i=a(e,`${t.id}-source-rows`,`uint32`,t.tableKeys.length);r.push(..._(e,t,i));let o=0;for(let[a,s]of n.entries()){if(s.keys.length>0){let n={...s,id:`${t.id}-batch-${a}`,firstSourceRow:o};r.push(...v(e,t,i,n)),r.push(...y(e,t,i,n))}o+=s.keys.length}return r}var g=class{id;index;keys;values;found;probes;statistics;maxProbeCount;constructor(e){this.id=e.id??`gpu-hash-index-query`,this.index=e.index,this.keys=e.keys,this.values=e.values,this.found=e.found,this.probes=e.probes,this.statistics=e.statistics,this.maxProbeCount=e.maxProbeCount??this.index.maxProbeCount;for(let[e,t]of[[this.index.tableKeys,`index.tableKeys`],[this.index.tableValues,`index.tableValues`],[this.keys,`keys`],[this.values,`values`],[this.found,`found`],[this.probes,`probes`],[this.statistics,`statistics`]])for(let n of s(e))r(n,`${this.id} ${t}`);if(this.index.tableKeys.length!==this.index.tableValues.length)throw Error(`${this.id} index table key and value capacities must match`);if(!w(this.index.tableKeys.length))throw Error(`${this.id} index capacity must be a positive power of two`);if(this.values.length!==this.keys.length||this.found.length!==this.keys.length||this.probes.length!==this.keys.length)throw Error(`${this.id} query outputs must match key length`);if(this.statistics.length<4)throw Error(`${this.id} statistics must contain four uint32 rows`);S(this.id,this.maxProbeCount,this.index.tableKeys.length,this.keys.length),C(this.id,[this.index.tableKeys,this.index.tableValues,this.keys],[this.values,this.found,this.probes,this.statistics])}getCommandNodes(e){let t=[];if([this.index.tableKeys,this.index.tableValues,this.keys,this.values,this.found,this.probes,this.statistics].flatMap(s).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);t.push(...b(e,this));let n=c(e,[this.keys,this.values,this.found,this.probes]);for(let[r,[i,a,o,s]]of n.entries())t.push(...x(e,{...this,id:`${this.id}-span-${r}`,keys:i,values:a,found:o,probes:s}));return t}};function _(e,t,n){let r=[],a=T(Math.max(t.tableKeys.length,6),e.device.limits.maxComputeWorkgroupsPerDimension),o=`
const CAPACITY: u32 = ${t.tableKeys.length}u;
const DISPATCH_X: u32 = ${a.x}u;
const DISPATCH_Y: u32 = ${a.y}u;
const TABLE_KEYS_OFFSET: u32 = ${i(t.tableKeys)}u;
const TABLE_VALUES_OFFSET: u32 = ${i(t.tableValues)}u;
const SOURCE_ROWS_OFFSET: u32 = ${i(n)}u;
const STATISTICS_OFFSET: u32 = ${i(t.statistics)}u;
@group(0) @binding(0) var<storage, read_write> tableKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> tableValues: array<u32>;
@group(0) @binding(2) var<storage, read_write> sourceRows: array<u32>;
@group(0) @binding(3) var<storage, read_write> statistics: array<u32>;
@compute @workgroup_size(${u}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let elementIndex = workgroupIndex * ${u}u + localIndex;
  if (elementIndex < CAPACITY) {
    tableKeys[TABLE_KEYS_OFFSET + elementIndex] = ${p}u;
    tableValues[TABLE_VALUES_OFFSET + elementIndex] = ${p}u;
    sourceRows[SOURCE_ROWS_OFFSET + elementIndex] = ${p}u;
  }
  if (elementIndex < 6u) {
    statistics[STATISTICS_OFFSET + elementIndex] = 0u;
  }
}`;return r.push(...E(e,{id:`${t.id}-initialize`,source:o,resources:[{buffer:t.tableKeys,usage:`storage-write`},{buffer:t.tableValues,usage:`storage-write`},{buffer:n,usage:`storage-write`},{buffer:t.statistics,usage:`storage-write`}],bindings:{tableKeys:t.tableKeys,tableValues:t.tableValues,sourceRows:n,statistics:t.statistics},dispatchSize:a,record:{moduleId:`GPUHashIndex/initialize`,invocationCount:t.tableKeys.length}})),r}function v(e,t,n,r){let a=[],o=T(r.keys.length,e.device.limits.maxComputeWorkgroupsPerDimension),s=r.validity?`@group(0) @binding(1) var<storage, read> inputValidity: array<u32>;`:``,c=r.validity?2:1,l=r.validity?i(r.validity):0,f=r.validity?`if (inputValidity[${l}u + inputIndex] == 0u) { return; }`:``,m=`
const ELEMENT_COUNT: u32 = ${r.keys.length}u;
const CAPACITY_MASK: u32 = ${t.tableKeys.length-1}u;
const MAX_PROBES: u32 = ${t.maxProbeCount}u;
const DISPATCH_X: u32 = ${o.x}u;
const DISPATCH_Y: u32 = ${o.y}u;
const KEYS_OFFSET: u32 = ${i(r.keys)}u;
const TABLE_KEYS_OFFSET: u32 = ${i(t.tableKeys)}u;
const SOURCE_ROWS_OFFSET: u32 = ${i(n)}u;
const STATISTICS_OFFSET: u32 = ${i(t.statistics)}u;
const FIRST_SOURCE_ROW: u32 = ${r.firstSourceRow}u;
@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
${s}
@group(0) @binding(${c}) var<storage, read_write> tableKeys: array<atomic<u32>>;
@group(0) @binding(${c+1}) var<storage, read_write> sourceRows: array<atomic<u32>>;
@group(0) @binding(${c+2}) var<storage, read_write> statistics: array<atomic<u32>>;

fn hashKey(key: u32) -> u32 {
  var value = key;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

@compute @workgroup_size(${u}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let inputIndex = workgroupIndex * ${u}u + localIndex;
  if (inputIndex >= ELEMENT_COUNT) { return; }
  ${f}
  let key = inputKeys[KEYS_OFFSET + inputIndex];
  if (key == ${p}u) {
    atomicAdd(&statistics[STATISTICS_OFFSET + 3u], 1u);
    return;
  }
  let start = hashKey(key) & CAPACITY_MASK;
  var probes = 0u;
  var inserted = false;
  var duplicate = false;
  for (var probe = 0u; probe < MAX_PROBES; probe++) {
    probes = probe + 1u;
    let slot = (start + probe) & CAPACITY_MASK;
    var result = atomicCompareExchangeWeak(
      &tableKeys[TABLE_KEYS_OFFSET + slot], ${p}u, key
    );
    // Bound spurious weak-CAS retries without probing onward and risking a duplicate slot.
    var exchangeAttempt = 1u;
    while (
      !result.exchanged &&
      result.old_value == ${p}u &&
      exchangeAttempt < ${d}u
    ) {
      result = atomicCompareExchangeWeak(
        &tableKeys[TABLE_KEYS_OFFSET + slot], ${p}u, key
      );
      exchangeAttempt++;
    }
    if (!result.exchanged && result.old_value == ${p}u) {
      break;
    }
    if (result.exchanged || result.old_value == key) {
      atomicMin(&sourceRows[SOURCE_ROWS_OFFSET + slot], FIRST_SOURCE_ROW + inputIndex);
      inserted = result.exchanged;
      duplicate = !result.exchanged;
      break;
    }
  }
  atomicAdd(&statistics[STATISTICS_OFFSET + 4u], probes);
  atomicMax(&statistics[STATISTICS_OFFSET + 5u], probes);
  if (inserted) {
    atomicAdd(&statistics[STATISTICS_OFFSET], 1u);
  } else if (duplicate) {
    atomicAdd(&statistics[STATISTICS_OFFSET + 1u], 1u);
  } else {
    atomicAdd(&statistics[STATISTICS_OFFSET + 2u], 1u);
  }
}`;return a.push(...E(e,{id:`${r.id}-build`,source:m,resources:[{buffer:r.keys,usage:`storage-read`},...r.validity?[{buffer:r.validity,usage:`storage-read`}]:[],{buffer:t.tableKeys,usage:`storage-read-write`},{buffer:n,usage:`storage-read-write`},{buffer:t.statistics,usage:`storage-read-write`}],bindings:{inputKeys:r.keys,...r.validity?{inputValidity:r.validity}:{},tableKeys:t.tableKeys,sourceRows:n,statistics:t.statistics},dispatchSize:o,record:{moduleId:`GPUHashIndex/build`,invocationCount:r.keys.length}})),a}function y(e,t,n,r){let a=[],o=T(t.tableKeys.length,e.device.limits.maxComputeWorkgroupsPerDimension),s=!!(r.values&&r.keys.length>0),c=s?`@group(0) @binding(3) var<storage, read> inputValues: array<u32>;`:``,l=s?`inputValues[${i(r.values)}u + localSourceRow]`:`${r.firstValue}u + localSourceRow`,d=`
const CAPACITY: u32 = ${t.tableKeys.length}u;
const FIRST_SOURCE_ROW: u32 = ${r.firstSourceRow}u;
const SOURCE_ROW_COUNT: u32 = ${r.keys.length}u;
const DISPATCH_X: u32 = ${o.x}u;
const DISPATCH_Y: u32 = ${o.y}u;
const TABLE_KEYS_OFFSET: u32 = ${i(t.tableKeys)}u;
const TABLE_VALUES_OFFSET: u32 = ${i(t.tableValues)}u;
const SOURCE_ROWS_OFFSET: u32 = ${i(n)}u;
@group(0) @binding(0) var<storage, read> tableKeys: array<u32>;
@group(0) @binding(1) var<storage, read> sourceRows: array<u32>;
@group(0) @binding(2) var<storage, read_write> tableValues: array<u32>;
${c}
@compute @workgroup_size(${u}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let slot = workgroupIndex * ${u}u + localIndex;
  if (slot >= CAPACITY || tableKeys[TABLE_KEYS_OFFSET + slot] == ${p}u) { return; }
  let sourceRow = sourceRows[SOURCE_ROWS_OFFSET + slot];
  if (sourceRow < FIRST_SOURCE_ROW || sourceRow - FIRST_SOURCE_ROW >= SOURCE_ROW_COUNT) { return; }
  let localSourceRow = sourceRow - FIRST_SOURCE_ROW;
  tableValues[TABLE_VALUES_OFFSET + slot] = ${l};
}`;return a.push(...E(e,{id:`${r.id}-finalize`,source:d,resources:[{buffer:t.tableKeys,usage:`storage-read`},{buffer:n,usage:`storage-read`},{buffer:t.tableValues,usage:`storage-write`},...s?[{buffer:r.values,usage:`storage-read`}]:[]],bindings:{tableKeys:t.tableKeys,sourceRows:n,tableValues:t.tableValues,...s?{inputValues:r.values}:{}},dispatchSize:o,record:{moduleId:`GPUHashIndex/finalize`,invocationCount:t.tableKeys.length}})),a}function b(e,t){let n=[],r=`
const STATISTICS_OFFSET: u32 = ${i(t.statistics)}u;
@group(0) @binding(0) var<storage, read_write> statistics: array<u32>;
@compute @workgroup_size(${u}) fn main(
  @builtin(global_invocation_id) globalId: vec3u
) {
  if (globalId.x < 4u) {
    statistics[STATISTICS_OFFSET + globalId.x] = 0u;
  }
}`;return n.push(...E(e,{id:`${t.id}-initialize`,source:r,resources:[{buffer:t.statistics,usage:`storage-write`}],bindings:{statistics:t.statistics},dispatchSize:{x:1,y:1,z:1}})),n}function x(e,t){let n=[],r=T(t.keys.length,e.device.limits.maxComputeWorkgroupsPerDimension),a=`
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const CAPACITY_MASK: u32 = ${t.index.tableKeys.length-1}u;
const MAX_PROBES: u32 = ${t.maxProbeCount}u;
const DISPATCH_X: u32 = ${r.x}u;
const DISPATCH_Y: u32 = ${r.y}u;
const KEYS_OFFSET: u32 = ${i(t.keys)}u;
const TABLE_KEYS_OFFSET: u32 = ${i(t.index.tableKeys)}u;
const TABLE_VALUES_OFFSET: u32 = ${i(t.index.tableValues)}u;
const VALUES_OFFSET: u32 = ${i(t.values)}u;
const FOUND_OFFSET: u32 = ${i(t.found)}u;
const PROBES_OFFSET: u32 = ${i(t.probes)}u;
const STATISTICS_OFFSET: u32 = ${i(t.statistics)}u;
@group(0) @binding(0) var<storage, read> queryKeys: array<u32>;
@group(0) @binding(1) var<storage, read> tableKeys: array<u32>;
@group(0) @binding(2) var<storage, read> tableValues: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputFound: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputProbes: array<u32>;
@group(0) @binding(6) var<storage, read_write> statistics: array<atomic<u32>>;

fn hashKey(key: u32) -> u32 {
  var value = key;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

@compute @workgroup_size(${u}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let queryIndex = workgroupIndex * ${u}u + localIndex;
  if (queryIndex >= ELEMENT_COUNT) { return; }
  let key = queryKeys[KEYS_OFFSET + queryIndex];
  var value = ${p}u;
  var found = false;
  var probes = 0u;
  if (key != ${p}u) {
    let start = hashKey(key) & CAPACITY_MASK;
    for (var probe = 0u; probe < MAX_PROBES; probe++) {
      probes = probe + 1u;
      let slot = (start + probe) & CAPACITY_MASK;
      let storedKey = tableKeys[TABLE_KEYS_OFFSET + slot];
      if (storedKey == key) {
        value = tableValues[TABLE_VALUES_OFFSET + slot];
        found = true;
        break;
      }
      if (storedKey == ${p}u) { break; }
    }
  }
  outputValues[VALUES_OFFSET + queryIndex] = value;
  outputFound[FOUND_OFFSET + queryIndex] = select(0u, 1u, found);
  outputProbes[PROBES_OFFSET + queryIndex] = probes;
  atomicAdd(&statistics[STATISTICS_OFFSET + select(1u, 0u, found)], 1u);
  atomicAdd(&statistics[STATISTICS_OFFSET + 2u], probes);
  atomicMax(&statistics[STATISTICS_OFFSET + 3u], probes);
}`;return n.push(...E(e,{id:`${t.id}-lookup`,source:a,resources:[{buffer:t.keys,usage:`storage-read`},{buffer:t.index.tableKeys,usage:`storage-read`},{buffer:t.index.tableValues,usage:`storage-read`},{buffer:t.values,usage:`storage-write`},{buffer:t.found,usage:`storage-write`},{buffer:t.probes,usage:`storage-write`},{buffer:t.statistics,usage:`storage-read-write`}],bindings:{queryKeys:t.keys,tableKeys:t.index.tableKeys,tableValues:t.index.tableValues,outputValues:t.values,outputFound:t.found,outputProbes:t.probes,statistics:t.statistics},dispatchSize:r})),n}function S(e,t,n,r){if(!Number.isSafeInteger(t)||t<1||t>n)throw Error(`${e} maxProbeCount must be an integer from one through capacity`);if(r*t>f)throw Error(`${e} aggregate probe count must fit in uint32 statistics`)}function C(e,t,r){let i=t.flatMap(s),a=r.flatMap(s);for(let t of i)for(let r of a)if(n(t,r))throw Error(`${e} input and output views must not overlap`);for(let t=0;t<a.length;t++)for(let r=t+1;r<a.length;r++)if(n(a[t],a[r]))throw Error(`${e} output views must not overlap`)}function w(e){return Number.isSafeInteger(e)&&e>0&&Number.isInteger(Math.log2(e))}function T(e,t){let n=Math.floor(t),r=Math.max(1,Math.ceil(e/u)),i=Math.min(r,n),a=Math.min(Math.ceil(r/i),n),o=Math.ceil(r/i/a);if(o>n)throw Error(`GPUHashIndex work exceeds the device 3D dispatch limit`);return{x:i,y:a,z:o}}function E(n,r){let i=[];i.push(t({id:r.id,resources:r.resources,compile:({device:t})=>{let n=new o(t,{id:r.id,source:r.source,shaderLayout:{bindings:Object.keys(r.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:t,getBuffer:i})=>{let a={};for(let[t,n]of Object.entries(r.bindings))a[t]=e(n,i);n.dispatch(t,{bindings:a,x:r.dispatchSize.x,y:r.dispatchSize.y,z:r.dispatchSize.z})},destroy:()=>n.destroy()}}}));let a=r.record?l(n,()=>({op:`GPUHashIndex`,moduleId:r.record.moduleId,source:r.source,workgroupSize:[r.record.workgroupSize??u,1,1],workgroups:[r.dispatchSize.x,r.dispatchSize.y,r.dispatchSize.z],invocationCount:r.record.invocationCount,bindings:Object.entries(r.bindings).map(([e,t])=>{let n=r.resources.find(e=>e.buffer===t)?.usage;return{name:e,view:t,access:n===`storage-read`?`read`:n===`storage-read-write`?`read_write`:`write`}})})):void 0;return a&&(i[0].annotation=a),i}export{h as i,g as n,p as r,m as t};