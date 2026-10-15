import{D as e,I as t,R as n,S as r,j as i,k as a,w as o,x as s,z as c}from"./resources-CwvhEOBe.js";import{i as l,o as u,t as d}from"./gpu-subgroup-utils-BuV3X9Xp.js";var f=256,p=64;function m(e,t=!1){return t?`portable`:d(e,{requiresSubgroupId:!0})}var h=class{id;input;output;mode;segmentFlags;constructor(e){if(this.id=e.id??`gpu-scan`,this.input=e.input,this.output=e.output,this.mode=e.mode??`exclusive`,this.segmentFlags=e.segmentFlags,x(this.input,`${this.id} input`),x(this.output,`${this.id} output`),this.output.length<this.input.length)throw Error(`${this.id} output must contain at least input.length rows`);if(this.output instanceof n&&this.output.length!==this.input.length)throw Error(`${this.id} vector output length must equal input length`);if(this.segmentFlags){if(x(this.segmentFlags,`${this.id} segmentFlags`),this.segmentFlags.length<this.input.length)throw Error(`${this.id} segmentFlags must contain at least input.length rows`);if(this.segmentFlags instanceof n&&this.segmentFlags.length!==this.input.length)throw Error(`${this.id} vector segmentFlags length must equal input length`);let e=new Set(S(this.output).map(e=>e.buffer));if(S(this.segmentFlags).some(t=>e.has(t.buffer)))throw Error(`${this.id} segmentFlags and output must use separate buffers`)}}getCommandNodes(e){let t=[];return t.push(...g(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)),t}};function g(e,t,n){let r=[];C(t,e.input,e.id),C(t,e.output,e.id),e.segmentFlags&&C(t,e.segmentFlags,e.id);let i=_(t,e.output,e.input.length),a=e.segmentFlags?_(t,e.segmentFlags,e.input.length):void 0,o=a?l(t,[e.input,i,a]):l(t,[e.input,i]);return r.push(...v(t,{id:e.id,inputChunks:o.map(e=>e[0]),outputChunks:o.map(e=>e[1]),segmentFlagChunks:a?o.map(e=>e[2]):void 0,isVector:o.length>1,mode:e.mode},n)),r}function _(e,t,r){return t.length===r||t instanceof n?t:e.createDataView(t.buffer,{format:t.format,length:r,byteOffset:t.byteOffset,byteStride:t.byteStride,rowByteLength:t.rowByteLength})}function v(e,t,n){let r=[],i=t.inputChunks,a=t.outputChunks,s=t.segmentFlagChunks,c=i.map((e,t)=>({chunkIndex:t,input:e,output:a[t],segmentFlags:s?.[t]})).filter(e=>e.input.length>0);if(c.length===0)return r;if(c.length===1){let i=c[0];return r.push(...y(e,{id:t.isVector?`${t.id}-chunk-${i.chunkIndex}`:t.id,input:i.input,output:i.output,mode:t.mode,segmentFlags:i.segmentFlags,maxComputeWorkgroupsPerDimension:n})),r}let l=o(e,`${t.id}-chunk-totals`,`uint32`,c.length),u=o(e,`${t.id}-chunk-offsets`,`uint32`,c.length),d=s?o(e,`${t.id}-chunk-segment-flags`,`uint32`,c.length):void 0,f=s?c.map(n=>o(e,`${t.id}-chunk-${n.chunkIndex}-segment-prefixes`,`uint32`,n.input.length)):void 0;for(let[i,a]of c.entries())r.push(...y(e,{id:`${t.id}-chunk-${a.chunkIndex}`,input:a.input,output:a.output,mode:t.mode,segmentFlags:a.segmentFlags,outputSegmentPrefixes:f?.[i],finalSum:b(e,l,i),finalSegmentFlag:d?b(e,d,i):void 0,maxComputeWorkgroupsPerDimension:n}));r.push(...y(e,{id:`${t.id}-chunk-carries`,input:l,output:u,mode:`exclusive`,segmentFlags:d,segmentSummaryInput:!!d,maxComputeWorkgroupsPerDimension:n}));for(let[i,a]of c.entries())r.push(...D(e,{id:`${t.id}-chunk-${a.chunkIndex}-add-carry`,output:a.output,offsets:u,length:a.output.length,offsetIndex:i,segmentPrefixes:f?.[i],dispatchLayout:O(a.output.length,n)}));return r}function y(e,t){let n=[];if(t.input.length===0)return n;let r=[],i=t.input,a=t.output,s=t.segmentFlags,c=t.input.length,l=0;for(;;){let u=Math.ceil(c/f),d,p;u>1&&(d=o(e,`${t.id}-level-${l}-block-sums`,`uint32`,u),s&&(p=o(e,`${t.id}-level-${l}-block-segment-flags`,`uint32`,u)));let m=s?l===0&&t.outputSegmentPrefixes?t.outputSegmentPrefixes:u>1||l>0?o(e,`${t.id}-level-${l}-segment-prefixes`,`uint32`,c):void 0:void 0;if(n.push(...w(e,{id:`${t.id}-level-${l}-scan`,input:i,output:a,mode:l===0?t.mode:`exclusive`,segmentFlags:s,segmentSummaryInput:!!s&&(l>0||t.segmentSummaryInput),segmentPrefixes:m,blockSums:d,blockSegmentFlags:p,finalSum:d?void 0:t.finalSum,finalSegmentFlag:d?void 0:t.finalSegmentFlag,length:c,blockCount:u,dispatchLayout:O(c,t.maxComputeWorkgroupsPerDimension)})),r.push({output:a,length:c,segmentPrefixes:m}),!d)break;let h=o(e,`${t.id}-level-${l}-block-offsets`,`uint32`,u);r[r.length-1].blockOffsets=h,i=d,a=h,s=p,c=u,l++}for(let i=r.length-2;i>=0;i--){let a=r[i],o=r[i+1];n.push(...D(e,{id:`${t.id}-level-${i}-add-offsets`,output:a.output,offsets:a.blockOffsets,length:a.length,segmentPrefixes:a.segmentPrefixes,offsetSegmentPrefixes:o.segmentPrefixes,dispatchLayout:O(a.length,t.maxComputeWorkgroupsPerDimension)}))}if(e.recording)for(let e of n){let n=Number(/^-level-(\d+)-/.exec(e.id.slice(t.id.length))?.[1]);e.annotation&&(e.annotation.loop={id:`${t.id}/levels`,iteration:n,phase:`unconditional`,maximumIterations:r.length})}return n}function b(e,t,n){return e.createDataView(t.buffer,{format:`uint32`,length:1,byteOffset:t.byteOffset+n*t.rowByteLength})}function x(e,t){let r=e instanceof n?e.data:[e];for(let e of r)i(e,t)}function S(e){return e instanceof n?e.data:[e]}function C(e,t,r){if((t instanceof n?t.data:[t]).some(t=>t.buffer.graph!==e))throw Error(`${r} views must belong to the target graph`)}function w(n,r){let i=[],o=r.blockSums??r.finalSum,s=r.blockSegmentFlags??r.finalSegmentFlag,l=o?`@group(0) @binding(2) var<storage, read_write> sumValues: array<u32>;`:``,d=r.segmentFlags?`@group(0) @binding(3) var<storage, read> segmentFlags: array<u32>;`:``,p=r.segmentPrefixes?`@group(0) @binding(4) var<storage, read_write> segmentPrefixes: array<u32>;`:``,m=s?`@group(0) @binding(5) var<storage, read_write> summarySegmentFlags: array<u32>;`:``,h=r.blockSums?`sumValues[SUM_OFFSET + workgroupIndex] = scratch[255u];`:r.finalSum?`sumValues[SUM_OFFSET] = scratch[255u];`:``,g=r.blockSegmentFlags?`summarySegmentFlags[SUMMARY_SEGMENT_FLAGS_OFFSET + workgroupIndex] = segmentScratch[255u];`:r.finalSegmentFlag?`summarySegmentFlags[SUMMARY_SEGMENT_FLAGS_OFFSET] = segmentScratch[255u];`:``,_=r.segmentFlags?`segmentFlags[SEGMENT_FLAGS_OFFSET + index]`:`0u`,v=r.segmentFlags?`if (lane >= stride) {
      scratch[lane] = select(addend + scratch[lane], scratch[lane], segmentScratch[lane] != 0u);
      segmentScratch[lane] = addendSegment | segmentScratch[lane];
    }`:`if (lane >= stride) {
      scratch[lane] = scratch[lane] + addend;
    }`,y=r.mode===`inclusive`?`scratch[lane]`:`scratch[lane] - inputValue`,b=r.segmentFlags&&r.mode===`exclusive`?`var scannedOutput = 0u;
    if (lane > 0u) {
      scannedOutput = scratch[lane - 1u];
    }
    ${r.segmentSummaryInput?``:`if (inputSegmentFlag != 0u) { scannedOutput = 0u; }`}
    outputValues[OUTPUT_OFFSET + index] = scannedOutput;`:`outputValues[OUTPUT_OFFSET + index] = ${y};`,x=r.segmentPrefixes?r.segmentSummaryInput?`var segmentPrefix = 0u;
    if (lane > 0u) {
      segmentPrefix = segmentScratch[lane - 1u];
    }
    segmentPrefixes[SEGMENT_PREFIXES_OFFSET + index] = segmentPrefix;`:`segmentPrefixes[SEGMENT_PREFIXES_OFFSET + index] = segmentScratch[lane];`:``,S=`
const ELEMENT_COUNT: u32 = ${r.length}u;
const BLOCK_COUNT: u32 = ${r.blockCount}u;
const INPUT_OFFSET: u32 = ${a(r.input)}u;
const OUTPUT_OFFSET: u32 = ${a(r.output)}u;
${o?`const SUM_OFFSET: u32 = ${a(o)}u;`:``}
${r.segmentFlags?`const SEGMENT_FLAGS_OFFSET: u32 = ${a(r.segmentFlags)}u;`:``}
${r.segmentPrefixes?`const SEGMENT_PREFIXES_OFFSET: u32 = ${a(r.segmentPrefixes)}u;`:``}
${s?`const SUMMARY_SEGMENT_FLAGS_OFFSET: u32 = ${a(s)}u;`:``}
@group(0) @binding(0) var<storage, read> inputValues: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<u32>;
${l}
${d}
${p}
${m}
var<workgroup> scratch: array<u32, ${f}>;
${r.segmentFlags?`var<workgroup> segmentScratch: array<u32, ${f}>;`:``}

@compute @workgroup_size(${f}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${k(r.dispatchLayout)}
  if (workgroupIndex >= BLOCK_COUNT) { return; }
  let lane = localInvocationIndex;
  var inputValue = 0u;
  var inputSegmentFlag = 0u;
  if (index < ELEMENT_COUNT) {
    inputValue = inputValues[INPUT_OFFSET + index];
    inputSegmentFlag = ${_};
  }
  scratch[lane] = inputValue;
  ${r.segmentFlags?`segmentScratch[lane] = inputSegmentFlag;`:``}
  workgroupBarrier();

  for (var stride = 1u; stride < ${f}u; stride = stride * 2u) {
    var addend = 0u;
    ${r.segmentFlags?`var addendSegment = 0u;`:``}
    if (lane >= stride) {
      addend = scratch[lane - stride];
      ${r.segmentFlags?`addendSegment = segmentScratch[lane - stride];`:``}
    }
    workgroupBarrier();
    ${v}
    workgroupBarrier();
  }

  if (lane == ${f-1}u) {
    ${h}
    ${g}
  }
  if (index < ELEMENT_COUNT) {
    ${b}
    ${x}
  }
}`,C=T(n,!!r.segmentFlags,r.length),w=C===`subgroups`?E(r,o,l):S;i.push(t({id:r.id,workload:{operation:`GPUScan`,variant:C,commandCount:1,maximumWorkgroupCount:r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z,maximumInvocationCount:r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z*f,readByteLength:r.length*Uint32Array.BYTES_PER_ELEMENT,writeByteLength:r.length*Uint32Array.BYTES_PER_ELEMENT+r.blockCount*Uint32Array.BYTES_PER_ELEMENT},resources:[{buffer:r.input,usage:`storage-read`},{buffer:r.output,usage:`storage-write`},...o?[{buffer:o,usage:`storage-write`}]:[],...r.segmentFlags?[{buffer:r.segmentFlags,usage:`storage-read`}]:[],...r.segmentPrefixes?[{buffer:r.segmentPrefixes,usage:`storage-write`}]:[],...s?[{buffer:s,usage:`storage-write`}]:[]],compile:({device:t})=>{let n=new c(t,{id:r.id,source:w,shaderLayout:{bindings:[{name:`inputValues`,type:`storage`,group:0,location:0},{name:`outputValues`,type:`storage`,group:0,location:1},...o?[{name:`sumValues`,type:`storage`,group:0,location:2}]:[],...r.segmentFlags?[{name:`segmentFlags`,type:`storage`,group:0,location:3}]:[],...r.segmentPrefixes?[{name:`segmentPrefixes`,type:`storage`,group:0,location:4}]:[],...s?[{name:`summarySegmentFlags`,type:`storage`,group:0,location:5}]:[]]}});return{encode:({computePass:t,getBuffer:i})=>{let a={inputValues:e(r.input,i),outputValues:e(r.output,i)};o&&(a.sumValues=e(o,i)),r.segmentFlags&&(a.segmentFlags=e(r.segmentFlags,i)),r.segmentPrefixes&&(a.segmentPrefixes=e(r.segmentPrefixes,i)),s&&(a.summarySegmentFlags=e(s,i)),n.dispatch(t,{bindings:a,x:r.dispatchLayout.x,y:r.dispatchLayout.y,z:r.dispatchLayout.z})},destroy:()=>n.destroy()}}}));let D=u(n,()=>({op:`GPUScan`,moduleId:`GPUScan/level-scan/${C}`,source:w,workgroupSize:[f,1,1],workgroups:[r.dispatchLayout.x,r.dispatchLayout.y,r.dispatchLayout.z],invocationCount:r.length,dedupName:`level-scan`,formula:`gpu-core.scan`,bindings:[{name:`inputValues`,view:r.input,access:`read`},{name:`outputValues`,view:r.output,access:`read_write`},...o?[{name:`sumValues`,view:o,access:`read_write`}]:[],...r.segmentFlags?[{name:`segmentFlags`,view:r.segmentFlags,access:`read`}]:[],...r.segmentPrefixes?[{name:`segmentPrefixes`,view:r.segmentPrefixes,access:`read_write`}]:[],...s?[{name:`summarySegmentFlags`,view:s,access:`read_write`}]:[]]}));return D&&(i[0].annotation=D),i}function T(e,t,n){let r=m(e.device,t);if(!e.autotuner||t)return r;let i=r===`subgroups`?`portable`:`subgroups`;return e.autotuner.selectKernel({operation:`GPUScan`,workloadSize:n,candidates:[{id:r},{id:i,supported:i===`portable`||m(e.device,!1)===`subgroups`}]}).variant}function E(e,t,n){let r=e.mode===`inclusive`?`blockPrefix`:`blockPrefix - inputValue`,i=e.blockSums?`sumValues[SUM_OFFSET + workgroupIndex] = blockPrefix;`:e.finalSum?`sumValues[SUM_OFFSET] = blockPrefix;`:``,o=Math.floor(4294967295/f)+1;return`
enable subgroups;
requires subgroup_id;

const ELEMENT_COUNT: u32 = ${e.length}u;
const BLOCK_COUNT: u32 = ${e.blockCount}u;
const INPUT_OFFSET: u32 = ${a(e.input)}u;
const OUTPUT_OFFSET: u32 = ${a(e.output)}u;
${t?`const SUM_OFFSET: u32 = ${a(t)}u;`:``}
@group(0) @binding(0) var<storage, read> inputValues: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<u32>;
${n}
var<workgroup> subgroupOffsets: array<u32, ${p}>;

@compute @workgroup_size(${f}) fn main(
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let workgroupIndex = (workgroupId.z * ${e.dispatchLayout.y}u + workgroupId.y) * ${e.dispatchLayout.x}u + workgroupId.x;
  if (workgroupIndex >= ${o}u || workgroupIndex >= BLOCK_COUNT) { return; }
  let lane = subgroupId * subgroupSize + subgroupInvocationId;
  let index = workgroupIndex * ${f}u + lane;
  var inputValue = 0u;
  if (index < ELEMENT_COUNT) {
    inputValue = inputValues[INPUT_OFFSET + index];
  }

  let subgroupPrefix = subgroupInclusiveAdd(inputValue);
  if (subgroupInvocationId == subgroupSize - 1u) {
    subgroupOffsets[subgroupId] = subgroupPrefix;
  }
  workgroupBarrier();

  let subgroupCount = ${f}u / subgroupSize;
  if (lane == 0u) {
    var runningOffset = 0u;
    for (var subgroupIndex = 0u; subgroupIndex < subgroupCount; subgroupIndex++) {
      let subgroupSum = subgroupOffsets[subgroupIndex];
      subgroupOffsets[subgroupIndex] = runningOffset;
      runningOffset = runningOffset + subgroupSum;
    }
  }
  workgroupBarrier();

  let blockPrefix = subgroupOffsets[subgroupId] + subgroupPrefix;
  if (lane == ${f-1}u) {
    ${i}
  }
  if (index < ELEMENT_COUNT) {
    outputValues[OUTPUT_OFFSET + index] = ${r};
  }
}`}function D(n,r){let i=[],o=r.offsetIndex===void 0?`index / ${f}u`:`${r.offsetIndex}u`,s=`
const ELEMENT_COUNT: u32 = ${r.length}u;
const OUTPUT_OFFSET: u32 = ${a(r.output)}u;
const OFFSETS_OFFSET: u32 = ${a(r.offsets)}u;
${r.segmentPrefixes?`const SEGMENT_PREFIXES_OFFSET: u32 = ${a(r.segmentPrefixes)}u;`:``}
${r.offsetSegmentPrefixes?`const OFFSET_SEGMENT_PREFIXES_OFFSET: u32 = ${a(r.offsetSegmentPrefixes)}u;`:``}
@group(0) @binding(0) var<storage, read_write> outputValues: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
${r.segmentPrefixes?`@group(0) @binding(2) var<storage, ${r.offsetSegmentPrefixes?`read_write`:`read`}> segmentPrefixes: array<u32>;`:``}
${r.offsetSegmentPrefixes?`@group(0) @binding(3) var<storage, read> offsetSegmentPrefixes: array<u32>;`:``}

@compute @workgroup_size(${f}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${k(r.dispatchLayout)}
  if (index < ELEMENT_COUNT) {
    ${r.offsetSegmentPrefixes?`let offsetSegmentPrefix = offsetSegmentPrefixes[OFFSET_SEGMENT_PREFIXES_OFFSET + ${o}];`:``}
    ${r.segmentPrefixes?`let segmentPrefix = segmentPrefixes[SEGMENT_PREFIXES_OFFSET + index];
    if (segmentPrefix == 0u) {`:``}
      outputValues[OUTPUT_OFFSET + index] = outputValues[OUTPUT_OFFSET + index] + offsets[OFFSETS_OFFSET + ${o}];
    ${r.segmentPrefixes?`}`:``}
    ${r.segmentPrefixes&&r.offsetSegmentPrefixes?`segmentPrefixes[SEGMENT_PREFIXES_OFFSET + index] = segmentPrefix | offsetSegmentPrefix;`:``}
  }
}`;i.push(t({id:r.id,workload:{operation:`GPUScan`,commandCount:1,maximumWorkgroupCount:r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z,maximumInvocationCount:r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z*f,readByteLength:r.length*Uint32Array.BYTES_PER_ELEMENT*2,writeByteLength:r.length*Uint32Array.BYTES_PER_ELEMENT},resources:[{buffer:r.output,usage:`storage-read-write`},{buffer:r.offsets,usage:`storage-read`},...r.segmentPrefixes?[{buffer:r.segmentPrefixes,usage:r.offsetSegmentPrefixes?`storage-read-write`:`storage-read`}]:[],...r.offsetSegmentPrefixes?[{buffer:r.offsetSegmentPrefixes,usage:`storage-read`}]:[]],compile:({device:t})=>{let n=new c(t,{id:r.id,source:s,shaderLayout:{bindings:[{name:`outputValues`,type:`storage`,group:0,location:0},{name:`offsets`,type:`storage`,group:0,location:1},...r.segmentPrefixes?[{name:`segmentPrefixes`,type:`storage`,group:0,location:2}]:[],...r.offsetSegmentPrefixes?[{name:`offsetSegmentPrefixes`,type:`storage`,group:0,location:3}]:[]]}});return{encode:({computePass:t,getBuffer:i})=>{let a={outputValues:e(r.output,i),offsets:e(r.offsets,i)};r.segmentPrefixes&&(a.segmentPrefixes=e(r.segmentPrefixes,i)),r.offsetSegmentPrefixes&&(a.offsetSegmentPrefixes=e(r.offsetSegmentPrefixes,i)),n.dispatch(t,{bindings:a,x:r.dispatchLayout.x,y:r.dispatchLayout.y,z:r.dispatchLayout.z})},destroy:()=>n.destroy()}}}));let l=u(n,()=>({op:`GPUScan`,moduleId:`GPUScan/add-offsets`,source:s,workgroupSize:[f,1,1],workgroups:[r.dispatchLayout.x,r.dispatchLayout.y,r.dispatchLayout.z],invocationCount:r.length,dedupName:`add-offsets`,formula:`gpu-core.scan`,bindings:[{name:`outputValues`,view:r.output,access:`read_write`},{name:`offsets`,view:r.offsets,access:`read`},...r.segmentPrefixes?[{name:`segmentPrefixes`,view:r.segmentPrefixes,access:r.offsetSegmentPrefixes?`read_write`:`read`}]:[],...r.offsetSegmentPrefixes?[{name:`offsetSegmentPrefixes`,view:r.offsetSegmentPrefixes,access:`read`}]:[]]}));return l&&(i[0].annotation=l),i}function O(e,t){return s(`GPUScan`,e,f,t)}function k(e){return r(e,f)}export{g as n,h as t};