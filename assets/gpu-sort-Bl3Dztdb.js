import{D as e,I as t,R as n,S as r,T as i,j as a,k as o,w as s,x as c,z as l}from"./resources-CwvhEOBe.js";import{a as u,i as d,o as f,t as p}from"./gpu-subgroup-utils-BuV3X9Xp.js";import{n as m,t as h}from"./gpu-scan-BuGhTLDQ.js";import{t as g}from"./gpu-chunk-utils-Bz-6HA7y.js";function _(e,t,n){let i=[],a=d(e,[t.keys,t.values]),l=d(e,[t.outputKeys,t.outputValues]),u=a.map(([r,a],o)=>{let c=`${t.id}-span-${o}`,l=s(e,`${c}-keys`,`uint32`,r.length),u=s(e,`${c}-values`,`uint32`,r.length),d=s(e,`${c}-ranks`,`uint32`,r.length);return i.push(...O(new D({id:c,keys:r,values:a,outputKeys:l,outputValues:u,algorithm:t.resolvedAlgorithm,direction:t.direction,keyBits:t.keyBits,digitBits:t.digitBits,elementsPerThread:t.elementsPerThread}),e,n)),{keys:l,values:u,ranks:d}}),f=t.resolvedAlgorithm===`radix`&&t.keyBits<32?2**t.keyBits-1:4294967295,p=t.direction===`ascending`?`<`:`>`;for(let[a,s]of u.entries()){let d=c(`GPUSort merge`,s.keys.length,256,n),m=r(d,256);i.push(g(e,{id:`${t.id}-ranks-${a}`,outputs:{ranks:s.ranks},dispatch:d,source:`
@group(0) @binding(0) var<storage, read_write> ranks: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${m}
  if (index < ${s.keys.length}u) { ranks[index] = index; }
}`}));for(let[n,r]of u.entries())n!==a&&i.push(g(e,{id:`${t.id}-rank-${a}-against-${n}`,inputs:{keys:s.keys,otherKeys:r.keys},outputs:{ranks:s.ranks},dispatch:d,source:`
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> otherKeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> ranks: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${m}
  if (index >= ${s.keys.length}u) { return; }
  let key = keys[index] & ${f}u;
  var first = 0u;
  var last = ${r.keys.length}u;
  loop {
    if (first == last) { break; }
    let middle = first + (last - first) / 2u;
    let otherKey = otherKeys[middle] & ${f}u;
    if (otherKey ${p} key${n<a?` || otherKey == key`:``}) { first = middle + 1u; }
    else { last = middle; }
  }
  ranks[index] += first;
}`}));let h=0;for(let[n,[r,c]]of l.entries())i.push(g(e,{id:`${t.id}-publish-${a}-${n}`,inputs:{keys:s.keys,values:s.values,ranks:s.ranks},outputs:{outputKeys:r,outputValues:c},dispatch:d,source:`
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> ranks: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${m}
  if (index >= ${s.keys.length}u) { return; }
  let rank = ranks[index];
  if (rank >= ${h}u && rank - ${h}u < ${r.length}u) {
    let localRank = rank - ${h}u;
    outputKeys[${o(r)}u + localRank] = keys[index];
    outputValues[${o(c)}u + localRank] = values[index];
  }
}`})),h+=r.length}return i}var v=256,y=256,b=y/32,x=4294967295,S=2147483648,C=v,w=[4,8],T=4,E=8,D=class{id;keys;values;outputKeys;outputValues;algorithm;direction;keyBits;digitBits;elementsPerThread;resolvedAlgorithm;constructor(e){this.id=e.id??`gpu-sort`,this.keys=e.keys,this.values=e.values,this.outputKeys=e.outputKeys,this.outputValues=e.outputValues,this.algorithm=e.algorithm??`auto`,this.direction=e.direction??`ascending`,this.keyBits=e.keyBits??32,this.digitBits=e.digitBits??T,this.elementsPerThread=e.elementsPerThread??E;for(let[e,t]of[[`keys`,this.keys],[`values`,this.values],[`outputKeys`,this.outputKeys],[`outputValues`,this.outputValues]])for(let n of u(t))a(n,`${this.id} ${e}`);if(![`auto`,`bitonic`,`radix`].includes(this.algorithm))throw Error(`${this.id} algorithm must be auto, bitonic, or radix`);if(![`ascending`,`descending`].includes(this.direction))throw Error(`${this.id} direction must be ascending or descending`);if(!Number.isInteger(this.keyBits)||this.keyBits<1||this.keyBits>32)throw Error(`${this.id} keyBits must be an integer from 1 to 32`);if(!w.includes(this.digitBits))throw Error(`${this.id} digitBits must be 4 or 8`);if(!Number.isInteger(this.elementsPerThread)||this.elementsPerThread<1||this.elementsPerThread>32)throw Error(`${this.id} elementsPerThread must be an integer from 1 to 32`);if(this.values.length!==this.keys.length||this.outputKeys.length!==this.keys.length||this.outputValues.length!==this.keys.length)throw Error(`${this.id} key, value, and output lengths must match`);if(this.keys.length>S)throw Error(`${this.id} supports at most ${S} rows`);A(this),this.resolvedAlgorithm=this.algorithm===`auto`?this.keys.length<=C?`bitonic`:`radix`:this.algorithm}get radixPlan(){return z(this)}getCommandNodes(e){let t=[];return t.push(...O(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)),t}};function O(e,t,r){for(let n of[e.keys,e.values,e.outputKeys,e.outputValues])for(let r of u(n))if(r.buffer.graph!==t)throw Error(`${e.id} views must belong to the target graph`);if([e.keys,e.values,e.outputKeys,e.outputValues].some(e=>e instanceof n)){let n=_(t,e,r);if(t.recording)for(let e of n){if(e.workload?.operation===`GPUSort`)continue;let t=0,n=0;for(let r of e.resources??[]){if(!(`buffer`in r))continue;let e=`length`in r.buffer?r.buffer.length*r.buffer.rowByteLength:r.buffer.byteLength;(r.usage===`storage-read`||r.usage===`storage-read-write`)&&(t+=e),(r.usage===`storage-write`||r.usage===`storage-read-write`)&&(n+=e)}e.workload={...e.workload,operation:`GPUSort`,variant:e.workload?.variant??`chunk-merge`,readByteLength:t,writeByteLength:n}}return n}let i=k(e,t,r);if(t.recording)for(let e of i)e.workload?.operation!==`GPUSort`&&(e.workload={...e.workload,operation:`GPUSort`,variant:`radix-scan`});return i}function k(e,t,n){let r=[];if(e.keys.length===0)return r;if(e.keys.length===1)return r.push(...j(t,e)),r;if(e.resolvedAlgorithm===`bitonic`){let i=c(`GPUSort`,e.keys.length,y,n);r.push(...M(t,e,i,n))}else r.push(...V(t,e,n));return r}function A(e){let t=u(e.outputKeys),n=u(e.outputValues),r=[...u(e.keys),...u(e.values)];if([...t,...n].some(e=>r.some(t=>t.buffer===e.buffer))||t.some(e=>n.some(t=>e.buffer===t.buffer)))throw Error(`${e.id} outputs must use separate buffers from inputs and each other`);for(let r of[t,n])for(let[t,n]of r.entries())if(r.slice(0,t).some(e=>i(e,n)))throw Error(`${e.id} output chunks must not overlap`)}function j(e,t,n=t.keys,i=t.values,a=`copy-pair`,s={x:1,y:1,z:1}){let c=[],l=`
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const KEYS_OFFSET: u32 = ${o(n)}u;
const VALUES_OFFSET: u32 = ${o(i)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${o(t.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${o(t.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;

@compute @workgroup_size(${y}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${r(s,y)}
  if (index >= ELEMENT_COUNT) { return; }
  outputKeys[OUTPUT_KEYS_OFFSET + index] = keys[KEYS_OFFSET + index];
  outputValues[OUTPUT_VALUES_OFFSET + index] = values[VALUES_OFFSET + index];
}`;return c.push(...K(e,{id:`${t.id}-${a}`,variant:`copy-pair`,invocationCount:t.keys.length,source:l,resources:[{buffer:n,usage:`storage-read`},{buffer:i,usage:`storage-read`},{buffer:t.outputKeys,usage:`storage-write`},{buffer:t.outputValues,usage:`storage-write`}],bindings:{keys:n,values:i,outputKeys:t.outputKeys,outputValues:t.outputValues},dispatchLayout:s})),c}function M(e,t,n,r){let i=[],a=W(t.keys.length);if(a<=v)return i.push(...N(e,t,a)),i;let o=c(`GPUSort bitonic`,a,v,r),l=s(e,`${t.id}-bitonic-indices-a`,`uint32`,a),u=s(e,`${t.id}-bitonic-indices-b`,`uint32`,a);i.push(...I(e,t,l,a,o));let d=l,f=u,p=G(a);for(let[n,r]of p.entries())i.push(...L(e,t,d,f,a,r,o,e.recording?{id:`${t.id}/passes`,iteration:n,phase:`unconditional`,maximumIterations:p.length}:void 0)),[d,f]=[f,d];return i.push(...R(e,t,d,n)),i}function N(e,t,n){let r=[],i=t.direction===`descending`,a=p(e.device,{requiresSubgroupId:!0})===`subgroups`,s=`
${a?`enable subgroups;
requires subgroup_id;`:``}
const INVALID_INDEX: u32 = ${x}u;
const LOGICAL_LENGTH: u32 = ${t.keys.length}u;
const PADDED_LENGTH: u32 = ${n}u;
const KEYS_OFFSET: u32 = ${o(t.keys)}u;
const VALUES_OFFSET: u32 = ${o(t.values)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${o(t.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${o(t.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;
var<workgroup> indices: array<u32, ${n}>;
var<workgroup> cachedKeys: array<u32, ${n}>;

fn comes_before(leftIndex: u32, rightIndex: u32) -> bool {
  let leftValid = leftIndex != INVALID_INDEX && leftIndex < LOGICAL_LENGTH;
  let rightValid = rightIndex != INVALID_INDEX && rightIndex < LOGICAL_LENGTH;
  if (leftValid != rightValid) { return leftValid; }
  if (!leftValid) { return false; }
  let leftKey = cachedKeys[leftIndex];
  let rightKey = cachedKeys[rightIndex];
  if (leftKey == rightKey) { return leftIndex < rightIndex; }
  return ${i?`leftKey > rightKey`:`leftKey < rightKey`};
}

@compute @workgroup_size(${n}) fn main(
  ${a?`@builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32`:`@builtin(local_invocation_index) localInvocationIndex: u32`}
) {
${a?F():P()}
}`;return r.push(...K(e,{id:`${t.id}-bitonic-local`,variant:`bitonic-local`,workgroupSize:n,invocationCount:t.keys.length,source:s,resources:[{buffer:t.keys,usage:`storage-read`},{buffer:t.values,usage:`storage-read`},{buffer:t.outputKeys,usage:`storage-write`},{buffer:t.outputValues,usage:`storage-write`}],bindings:{keys:t.keys,values:t.values,outputKeys:t.outputKeys,outputValues:t.outputValues},dispatchLayout:{x:1,y:1,z:1}})),r}function P(){return`
  indices[localInvocationIndex] = select(
    INVALID_INDEX,
    localInvocationIndex,
    localInvocationIndex < LOGICAL_LENGTH
  );
  if (localInvocationIndex < LOGICAL_LENGTH) {
    cachedKeys[localInvocationIndex] = keys[KEYS_OFFSET + localInvocationIndex];
  } else {
    cachedKeys[localInvocationIndex] = 0u;
  }
  workgroupBarrier();

  for (var blockWidth = 2u; blockWidth <= PADDED_LENGTH; blockWidth <<= 1u) {
    for (var compareStride = blockWidth >> 1u; compareStride > 0u; compareStride >>= 1u) {
      let partnerIndex = localInvocationIndex ^ compareStride;
      if (partnerIndex > localInvocationIndex) {
        let leftIndex = indices[localInvocationIndex];
        let rightIndex = indices[partnerIndex];
        let ascending = (localInvocationIndex & blockWidth) == 0u;
        let shouldSwap = select(
          comes_before(leftIndex, rightIndex),
          comes_before(rightIndex, leftIndex),
          ascending
        );
        indices[localInvocationIndex] = select(leftIndex, rightIndex, shouldSwap);
        indices[partnerIndex] = select(rightIndex, leftIndex, shouldSwap);
      }
      workgroupBarrier();
    }
  }

  if (localInvocationIndex < LOGICAL_LENGTH) {
    let sourceIndex = indices[localInvocationIndex];
    outputKeys[OUTPUT_KEYS_OFFSET + localInvocationIndex] = cachedKeys[sourceIndex];
    outputValues[OUTPUT_VALUES_OFFSET + localInvocationIndex] = values[VALUES_OFFSET + sourceIndex];
  }`}function F(){return`
  let lane = subgroupId * subgroupSize + subgroupInvocationId;
  var currentIndex = select(INVALID_INDEX, lane, lane < LOGICAL_LENGTH);
  if (lane < LOGICAL_LENGTH) {
    cachedKeys[lane] = keys[KEYS_OFFSET + lane];
  } else {
    cachedKeys[lane] = 0u;
  }
  workgroupBarrier();

  for (var blockWidth = 2u; blockWidth <= PADDED_LENGTH; blockWidth <<= 1u) {
    for (var compareStride = blockWidth >> 1u; compareStride > 0u; compareStride >>= 1u) {
      var partnerIndex = INVALID_INDEX;
      if (compareStride < subgroupSize) {
        partnerIndex = subgroupShuffleXor(currentIndex, compareStride);
      } else {
        indices[lane] = currentIndex;
        workgroupBarrier();
        partnerIndex = indices[lane ^ compareStride];
      }

      let lowerLane = (lane & compareStride) == 0u;
      let leftIndex = select(partnerIndex, currentIndex, lowerLane);
      let rightIndex = select(currentIndex, partnerIndex, lowerLane);
      let ascending = (lane & blockWidth) == 0u;
      let shouldSwap = select(
        comes_before(leftIndex, rightIndex),
        comes_before(rightIndex, leftIndex),
        ascending
      );
      let sortedLeft = select(leftIndex, rightIndex, shouldSwap);
      let sortedRight = select(rightIndex, leftIndex, shouldSwap);
      currentIndex = select(sortedRight, sortedLeft, lowerLane);

      if (compareStride >= subgroupSize) {
        workgroupBarrier();
      }
    }
  }

  if (lane < LOGICAL_LENGTH) {
    outputKeys[OUTPUT_KEYS_OFFSET + lane] = cachedKeys[currentIndex];
    outputValues[OUTPUT_VALUES_OFFSET + lane] = values[VALUES_OFFSET + currentIndex];
  }`}function I(e,t,n,i,a){let s=[],c=`
const INVALID_INDEX: u32 = ${x}u;
const LOGICAL_LENGTH: u32 = ${t.keys.length}u;
const PADDED_LENGTH: u32 = ${i}u;
const INDICES_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(0) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(${v}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${r(a,v)}
  if (index < PADDED_LENGTH) {
    indices[INDICES_OFFSET + index] = select(INVALID_INDEX, index, index < LOGICAL_LENGTH);
  }
}`;return s.push(...K(e,{id:`${t.id}-bitonic-initialize`,variant:`bitonic-initialize`,invocationCount:i,source:c,resources:[{buffer:n,usage:`storage-write`}],bindings:{indices:n},dispatchLayout:a})),s}function L(e,t,n,i,a,s,c,l){let u=[],d=t.direction===`descending`,f=`
const INVALID_INDEX: u32 = ${x}u;
const LOGICAL_LENGTH: u32 = ${t.keys.length}u;
const PADDED_LENGTH: u32 = ${a}u;
const BLOCK_WIDTH: u32 = ${s.blockWidth}u;
const COMPARE_STRIDE: u32 = ${s.compareStride}u;
const KEYS_OFFSET: u32 = ${o(t.keys)}u;
const INDICES_IN_OFFSET: u32 = ${o(n)}u;
const INDICES_OUT_OFFSET: u32 = ${o(i)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> indicesIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> indicesOut: array<u32>;

fn is_valid(index: u32) -> bool {
  return index != INVALID_INDEX && index < LOGICAL_LENGTH;
}

fn comes_before(leftIndex: u32, rightIndex: u32) -> bool {
  let leftValid = is_valid(leftIndex);
  let rightValid = is_valid(rightIndex);
  if (leftValid != rightValid) { return leftValid; }
  if (!leftValid) { return false; }
  let leftKey = keys[KEYS_OFFSET + leftIndex];
  let rightKey = keys[KEYS_OFFSET + rightIndex];
  if (leftKey == rightKey) { return leftIndex < rightIndex; }
  return ${d?`leftKey > rightKey`:`leftKey < rightKey`};
}

@compute @workgroup_size(${v}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${r(c,v)}
  if (index >= PADDED_LENGTH) { return; }
  let partnerIndex = index ^ COMPARE_STRIDE;
  if (partnerIndex <= index) { return; }
  let leftIndex = indicesIn[INDICES_IN_OFFSET + index];
  let rightIndex = indicesIn[INDICES_IN_OFFSET + partnerIndex];
  let ascending = (index & BLOCK_WIDTH) == 0u;
  let shouldSwap = select(
    comes_before(leftIndex, rightIndex),
    comes_before(rightIndex, leftIndex),
    ascending
  );
  indicesOut[INDICES_OUT_OFFSET + index] = select(leftIndex, rightIndex, shouldSwap);
  indicesOut[INDICES_OUT_OFFSET + partnerIndex] = select(rightIndex, leftIndex, shouldSwap);
}`;return u.push(...K(e,{id:`${t.id}-bitonic-${s.blockWidth}-${s.compareStride}`,variant:`bitonic-compare`,loop:l,invocationCount:a,source:f,resources:[{buffer:t.keys,usage:`storage-read`},{buffer:n,usage:`storage-read`},{buffer:i,usage:`storage-write`}],bindings:{keys:t.keys,indicesIn:n,indicesOut:i},dispatchLayout:c})),u}function R(e,t,n,i){let a=[],s=`
const LOGICAL_LENGTH: u32 = ${t.keys.length}u;
const KEYS_OFFSET: u32 = ${o(t.keys)}u;
const VALUES_OFFSET: u32 = ${o(t.values)}u;
const INDICES_OFFSET: u32 = ${o(n)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${o(t.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${o(t.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;

@compute @workgroup_size(${v}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${r(i,v)}
  if (index >= LOGICAL_LENGTH) { return; }
  let sourceIndex = indices[INDICES_OFFSET + index];
  outputKeys[OUTPUT_KEYS_OFFSET + index] = keys[KEYS_OFFSET + sourceIndex];
  outputValues[OUTPUT_VALUES_OFFSET + index] = values[VALUES_OFFSET + sourceIndex];
}`;return a.push(...K(e,{id:`${t.id}-bitonic-gather`,variant:`bitonic-gather`,invocationCount:t.keys.length,source:s,resources:[{buffer:t.keys,usage:`storage-read`},{buffer:t.values,usage:`storage-read`},{buffer:n,usage:`storage-read`},{buffer:t.outputKeys,usage:`storage-write`},{buffer:t.outputValues,usage:`storage-write`}],bindings:{keys:t.keys,values:t.values,indices:n,outputKeys:t.outputKeys,outputValues:t.outputValues},dispatchLayout:i})),a}function z(e){let t=e.digitBits,n=2**t,r=e.elementsPerThread,i=y*r,a=Math.max(1,Math.ceil(e.keys.length/i)),o=Math.ceil(e.keyBits/t);return{digitBits:t,bucketCount:n,elementsPerThread:r,tileSize:i,workgroupCount:a,histogramLength:n*a,passCount:o,workgroupStorageBytes:B(t)}}function B(e){let t=2**e;return(t*b+t)*Uint32Array.BYTES_PER_ELEMENT}function V(e,t,n){let r=[],i=z(t),a=c(`GPUSort radix`,t.keys.length,i.tileSize,n),o=i.passCount>1?s(e,`${t.id}-radix-scratch-keys`,`uint32`,t.keys.length):void 0,l=i.passCount>1?s(e,`${t.id}-radix-scratch-values`,`uint32`,t.keys.length):void 0,u=t.keys,d=t.values;for(let c=0;c<i.passCount;c++){let f=e.recording?{id:`${t.id}/passes`,iteration:c,phase:`unconditional`,maximumIterations:i.passCount}:void 0,p=c*i.digitBits,g=Math.min(i.digitBits,t.keyBits-p),_=2**g,v=s(e,`${t.id}-radix-digit-${p}-histogram`,`uint32`,_*i.workgroupCount),y=s(e,`${t.id}-radix-digit-${p}-offsets`,`uint32`,_*i.workgroupCount),b=(i.passCount-c)%2==1,x=b?t.outputKeys:o,S=b?t.outputValues:l;if(!x||!S)throw Error(`${t.id} radix scratch is missing`);r.push(...H(e,t,u,v,{bitOffset:p,digitBits:g,plan:i,dispatchLayout:a,loop:f}));let C=new h({id:`${t.id}-radix-digit-${p}-scan`,input:v,output:y});r.push(...m(C,e,n)),r.push(...U(e,t,u,d,y,x,S,{bitOffset:p,digitBits:g,plan:i,dispatchLayout:a,loop:f})),u=x,d=S}return r}function H(e,t,n,r,i){let a=[],{bitOffset:s,digitBits:c,plan:l,dispatchLayout:u}=i,d=2**c,f=t.direction===`descending`,p=`
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const BIT_OFFSET: u32 = ${s}u;
const BUCKET_COUNT: u32 = ${d}u;
const DIGIT_MASK: u32 = ${d-1}u;
const WORKGROUP_COUNT: u32 = ${l.workgroupCount}u;
const ELEMENTS_PER_THREAD: u32 = ${l.elementsPerThread}u;
const TILE_SIZE: u32 = ${l.tileSize}u;
const KEYS_OFFSET: u32 = ${o(n)}u;
const HISTOGRAM_OFFSET: u32 = ${o(r)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32>;
var<workgroup> digitCounts: array<atomic<u32>, ${d}>;

@compute @workgroup_size(${y}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let workgroupIndex =
    (workgroupId.z * ${u.y}u + workgroupId.y) * ${u.x}u + workgroupId.x;
  if (workgroupIndex >= WORKGROUP_COUNT) { return; }
  for (var bucket = localInvocationIndex; bucket < BUCKET_COUNT; bucket += ${y}u) {
    atomicStore(&digitCounts[bucket], 0u);
  }
  workgroupBarrier();

  let tileBase = workgroupIndex * TILE_SIZE;
  for (var slot = 0u; slot < ELEMENTS_PER_THREAD; slot++) {
    let index = tileBase + slot * ${y}u + localInvocationIndex;
    if (index < ELEMENT_COUNT) {
      let key = keys[KEYS_OFFSET + index];
      let digit = (key >> BIT_OFFSET) & DIGIT_MASK;
      let bucket = ${f?`DIGIT_MASK - digit`:`digit`};
      atomicAdd(&digitCounts[bucket], 1u);
    }
  }
  workgroupBarrier();

  for (var bucket = localInvocationIndex; bucket < BUCKET_COUNT; bucket += ${y}u) {
    histogram[HISTOGRAM_OFFSET + bucket * WORKGROUP_COUNT + workgroupIndex] =
      atomicLoad(&digitCounts[bucket]);
  }
}`;return a.push(...K(e,{id:`${t.id}-radix-digit-${s}-histogram`,variant:`radix-histogram`,loop:i.loop,invocationCount:t.keys.length,source:p,resources:[{buffer:n,usage:`storage-read`},{buffer:r,usage:`storage-write`}],bindings:{keys:n,histogram:r},dispatchLayout:u})),a}function U(e,t,n,r,i,a,s,c){let l=[],{bitOffset:u,digitBits:d,plan:f,dispatchLayout:p}=c,m=2**d,h=t.direction===`descending`,g=`
const ELEMENT_COUNT: u32 = ${t.keys.length}u;
const BIT_OFFSET: u32 = ${u}u;
const BUCKET_COUNT: u32 = ${m}u;
const DIGIT_MASK: u32 = ${m-1}u;
const WORKGROUP_COUNT: u32 = ${f.workgroupCount}u;
const ELEMENTS_PER_THREAD: u32 = ${f.elementsPerThread}u;
const TILE_SIZE: u32 = ${f.tileSize}u;
const MASK_WORD_COUNT: u32 = ${b}u;
const MASK_COUNT: u32 = ${m*b}u;
const KEYS_OFFSET: u32 = ${o(n)}u;
const VALUES_OFFSET: u32 = ${o(r)}u;
const OFFSETS_OFFSET: u32 = ${o(i)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${o(a)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${o(s)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;
var<workgroup> digitMasks: array<atomic<u32>, ${m*b}>;
var<workgroup> bucketCursors: array<u32, ${m}>;

@compute @workgroup_size(${y}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let workgroupIndex =
    (workgroupId.z * ${p.y}u + workgroupId.y) * ${p.x}u + workgroupId.x;
  if (workgroupIndex >= WORKGROUP_COUNT) { return; }
  for (var bucket = localInvocationIndex; bucket < BUCKET_COUNT; bucket += ${y}u) {
    bucketCursors[bucket] = 0u;
  }

  let tileBase = workgroupIndex * TILE_SIZE;
  let maskWord = localInvocationIndex >> 5u;
  let precedingBits = (1u << (localInvocationIndex & 31u)) - 1u;
  for (var slot = 0u; slot < ELEMENTS_PER_THREAD; slot++) {
    for (var mask = localInvocationIndex; mask < MASK_COUNT; mask += ${y}u) {
      atomicStore(&digitMasks[mask], 0u);
    }
    workgroupBarrier();

    let index = tileBase + slot * ${y}u + localInvocationIndex;
    let valid = index < ELEMENT_COUNT;
    var key = 0u;
    var bucket = 0u;
    if (valid) {
      key = keys[KEYS_OFFSET + index];
      let digit = (key >> BIT_OFFSET) & DIGIT_MASK;
      bucket = ${h?`DIGIT_MASK - digit`:`digit`};
      atomicOr(&digitMasks[bucket * MASK_WORD_COUNT + maskWord], 1u << (localInvocationIndex & 31u));
    }
    workgroupBarrier();

    if (valid) {
      let maskBase = bucket * MASK_WORD_COUNT;
      var localRank = 0u;
      for (var word = 0u; word < maskWord; word++) {
        localRank += countOneBits(atomicLoad(&digitMasks[maskBase + word]));
      }
      localRank += countOneBits(atomicLoad(&digitMasks[maskBase + maskWord]) & precedingBits);
      let bucketOffset = offsets[OFFSETS_OFFSET + bucket * WORKGROUP_COUNT + workgroupIndex];
      let outputIndex = bucketOffset + bucketCursors[bucket] + localRank;
      outputKeys[OUTPUT_KEYS_OFFSET + outputIndex] = key;
      outputValues[OUTPUT_VALUES_OFFSET + outputIndex] = values[VALUES_OFFSET + index];
    }
    workgroupBarrier();

    for (var bucket = localInvocationIndex; bucket < BUCKET_COUNT; bucket += ${y}u) {
      var slotCount = 0u;
      for (var word = 0u; word < MASK_WORD_COUNT; word++) {
        slotCount += countOneBits(atomicLoad(&digitMasks[bucket * MASK_WORD_COUNT + word]));
      }
      bucketCursors[bucket] += slotCount;
    }
    workgroupBarrier();
  }
}`;return l.push(...K(e,{id:`${t.id}-radix-digit-${u}-scatter`,variant:`radix-scatter`,loop:c.loop,invocationCount:t.keys.length,source:g,resources:[{buffer:n,usage:`storage-read`},{buffer:r,usage:`storage-read`},{buffer:i,usage:`storage-read`},{buffer:a,usage:`storage-write`},{buffer:s,usage:`storage-write`}],bindings:{keys:n,values:r,offsets:i,outputKeys:a,outputValues:s},dispatchLayout:p})),l}function W(e){let t=1;for(;t<e;)t*=2;return t}function G(e){let t=[];for(let n=2;n<=e;n*=2)for(let e=n/2;e>=1;e/=2)t.push({blockWidth:n,compareStride:e});return t}function K(n,r){let i=[];if(i.push(t({id:r.id,resources:r.resources,compile:({device:t})=>{let n=new l(t,{id:r.id,source:r.source,shaderLayout:{bindings:Object.keys(r.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:t,getBuffer:i})=>{let a={};for(let[t,n]of Object.entries(r.bindings))a[t]=e(n,i);n.dispatch(t,{bindings:a,x:r.dispatchLayout.x,y:r.dispatchLayout.y,z:r.dispatchLayout.z})},destroy:()=>n.destroy()}}})),n.recording)for(let e of i)e.workload={operation:`GPUSort`,variant:r.variant,commandCount:1,maximumWorkgroupCount:r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z,maximumInvocationCount:r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z*(r.workgroupSize??y),readByteLength:r.resources.reduce((e,t)=>e+(t.usage===`storage-read`||t.usage===`storage-read-write`?`length`in t.buffer?t.buffer.length*t.buffer.rowByteLength:t.buffer.byteLength:0),0),writeByteLength:r.resources.reduce((e,t)=>e+(t.usage===`storage-write`||t.usage===`storage-read-write`?`length`in t.buffer?t.buffer.length*t.buffer.rowByteLength:t.buffer.byteLength:0),0)};let a=f(n,()=>({op:`GPUSort`,moduleId:`GPUSort/${r.variant}`,source:r.source,workgroupSize:[r.workgroupSize??y,1,1],workgroups:[r.dispatchLayout.x,r.dispatchLayout.y,r.dispatchLayout.z],invocationCount:r.invocationCount,loop:r.loop,dedupName:r.variant,bindings:Object.entries(r.bindings).map(([e,t])=>{let n=r.resources.find(e=>e.buffer===t)?.usage;return{name:e,view:t,access:n===`storage-read`?`read`:n===`storage-read-write`?`read_write`:`write`}})}));return a&&(i[0].annotation=a),i}export{D as t};