import{D as e,I as t,M as n,R as r,S as i,j as a,k as o,w as s,x as c,z as l}from"./resources-CwvhEOBe.js";import{i as u,o as d,t as f}from"./gpu-subgroup-utils-BuV3X9Xp.js";var p=256,m=[`uint32`,`sint32`,`float32`];function h(e){return f(e,{requiresSubgroupId:!0})}var g=class{id;input;mask;output;operation;constructor(e){this.id=e.id??`gpu-reduction`,this.input=e.input,this.mask=e.mask,this.output=e.output,this.operation=e.operation;for(let[e,t]of y(this.input).entries()){let i=this.input instanceof r?` input chunk ${e}`:` input`;if(n(t,m,`${this.id}${i}`),t.format!==this.input.format)throw Error(`${this.id}${i} format must match the input format`)}if(n(this.output,m,`${this.id} output`),this.input.format!==this.output.format)throw Error(`${this.id} input and output formats must match`);if(![`sum`,`min`,`max`,`extent`].includes(this.operation))throw Error(`${this.id} operation must be sum, min, max, or extent`);let t=this.operation===`extent`?2:1;if(this.output.length!==t)throw Error(`${this.id} ${this.operation} output must contain ${t} row(s)`);if(y(this.input).some(e=>e.buffer===this.output.buffer))throw Error(`${this.id} inputs and output must use separate buffers`);if(this.mask){for(let[e,t]of b(this.mask).entries())if(a(t,`${this.id} mask chunk ${e}`),t.buffer===this.output.buffer)throw Error(`${this.id} mask and output must use separate buffers`);if(this.input.length!==this.mask.length)throw Error(`${this.id} input and mask lengths must match`)}}getCommandNodes(e){let t=[],n=y(this.input),r=this.mask?b(this.mask):void 0;if(n.some(t=>t.buffer.graph!==e)||r?.some(t=>t.buffer.graph!==e)||this.output.buffer.graph!==e)throw Error(`${this.id} views must belong to the target graph`);if(this.mask){let t=u(e,[this.input,this.mask]);n=t.map(([e])=>e),r=t.map(([,e])=>e)}let i=n.map((e,t)=>({input:e,mask:r?.[t]})).filter(({input:e})=>e.length>0);if(i.length===0)return t.push(...D(e,this.id,this.output)),t;let a=this.operation===`extent`?2:1,o=this.operation!==`sum`&&(this.input.format===`float32`||!!this.mask),c;if(i.length===1)c=_(e,{id:this.id,format:this.input.format,operation:this.operation,inputValues:i[0].input,inputValidity:i[0].mask,inputLength:i[0].input.length,valuesPerRow:a,firstLevel:!0,needsValidity:o}),t.push(...c.nodes);else{let n=s(e,`${this.id}-chunk-values`,this.input.format,i.length*a),r=o?s(e,`${this.id}-chunk-validity`,`uint32`,i.length):void 0;i.forEach(({input:i,mask:s},c)=>{t.push(..._(e,{id:`${this.id}-chunk-${c}`,format:this.input.format,operation:this.operation,inputValues:i,inputValidity:s,inputLength:i.length,valuesPerRow:a,firstLevel:!0,needsValidity:o,finalValues:v(e,n,c*a,a),finalValidity:r?v(e,r,c,1):void 0}).nodes)}),c=_(e,{id:`${this.id}-merge`,format:this.input.format,operation:this.operation,inputValues:n,inputValidity:r,inputLength:i.length,valuesPerRow:a,firstLevel:!1,needsValidity:o}),t.push(...c.nodes)}return t.push(...E(e,{id:`${this.id}-finalize`,format:this.input.format,inputValues:c.values,inputValidity:c.validity,output:this.output,valuesPerRow:a})),t}};function _(e,t){let n=[],r=t.inputValues,i=t.inputValidity,a=t.inputLength,o=0;for(;a>1||o===0;){let c=Math.ceil(a/p),l=c===1,u=l&&t.finalValues?t.finalValues:s(e,`${t.id}-level-${o}-values`,t.format,c*t.valuesPerRow),d=t.needsValidity?l&&t.finalValidity?t.finalValidity:s(e,`${t.id}-level-${o}-validity`,`uint32`,c):void 0;n.push(...x(e,{id:`${t.id}-level-${o}`,format:t.format,operation:t.operation,inputValues:r,inputValidity:i,outputValues:u,outputValidity:d,inputLength:a,valuesPerRow:t.valuesPerRow,firstLevel:t.firstLevel&&o===0})),r=u,i=d,a=c,o++}return e.recording&&n.forEach((e,r)=>{e.annotation&&(e.annotation.loop={id:`${t.id}/levels`,iteration:r,phase:`unconditional`,maximumIterations:n.length})}),{values:r,validity:i,nodes:n}}function v(e,t,n,r){return e.createDataView(t.buffer,{format:t.format,length:r,byteOffset:t.byteOffset+n*t.rowByteLength})}function y(e){return e instanceof r?e.data:[e]}function b(e){return e instanceof r?e.data:[e]}function x(e,t){let n=[],r=c(`GPUReduction`,t.inputLength,p,e.device.limits.maxComputeWorkgroupsPerDimension),a=k(t.format),s=A(t.format),l=t.operation===`sum`,u=t.operation===`extent`,d=t.inputValidity?`@group(0) @binding(1) var<storage, read> inputValidity: array<u32>;`:``,f=t.inputValidity?2:1,m=f+1,g=`value = inputValues[INPUT_OFFSET + index];
    secondValue = value;
    valid = select(0u, 1u, ${t.format===`float32`?`value == value && abs(value) <= 3.402823466e+38`:`true`});${t.firstLevel&&t.inputValidity&&t.format===`float32`&&l?`
    if (valid == 0u) { value = 0.0; secondValue = 0.0; }`:``}`,_=t.firstLevel?t.inputValidity?`if (inputValidity[VALIDITY_OFFSET + index] != 0u) {
      ${g}
    }`:g:`value = inputValues[INPUT_OFFSET + index * VALUES_PER_ROW];
    secondValue = inputValues[INPUT_OFFSET + index * VALUES_PER_ROW + ${u?`1u`:`0u`}];
    ${t.inputValidity?`valid = inputValidity[VALIDITY_OFFSET + index];`:`valid = 1u;`}`,v=l?`firstScratch[lane] = firstScratch[lane] + firstScratch[lane + stride];`:`let rightValid = validityScratch[lane + stride];
      if (rightValid != 0u) {
        if (validityScratch[lane] == 0u) {
          firstScratch[lane] = firstScratch[lane + stride];
          secondScratch[lane] = secondScratch[lane + stride];
        } else {
          firstScratch[lane] = ${t.operation===`max`?`max`:`min`}(firstScratch[lane], firstScratch[lane + stride]);
          secondScratch[lane] = ${u||t.operation===`max`?`max`:`min`}(secondScratch[lane], secondScratch[lane + stride]);
        }
        validityScratch[lane] = 1u;
      }`,y=h(e.device),b=y===`subgroups`?C({format:t.format,operation:t.operation,combine:v}):S(v),x=`
${y===`subgroups`?`enable subgroups;
requires subgroup_id;`:``}
const ELEMENT_COUNT: u32 = ${t.inputLength}u;
const VALUES_PER_ROW: u32 = ${t.valuesPerRow}u;
const INPUT_OFFSET: u32 = ${o(t.inputValues)}u;
${t.inputValidity?`const VALIDITY_OFFSET: u32 = ${o(t.inputValidity)}u;`:``}
const OUTPUT_OFFSET: u32 = ${o(t.outputValues)}u;
${t.outputValidity?`const OUTPUT_VALIDITY_OFFSET: u32 = ${o(t.outputValidity)}u;`:``}
@group(0) @binding(0) var<storage, read> inputValues: array<${a}>;
${d}
@group(0) @binding(${f}) var<storage, read_write> outputValues: array<${a}>;
${t.outputValidity?`@group(0) @binding(${m}) var<storage, read_write> outputValidity: array<u32>;`:``}
var<workgroup> firstScratch: array<${a}, ${p}>;
var<workgroup> secondScratch: array<${a}, ${p}>;
var<workgroup> validityScratch: array<u32, ${p}>;

@compute @workgroup_size(${p}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  ${y===`subgroups`?`@builtin(subgroup_invocation_id) subgroupInvocationId: u32,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_id) subgroupId: u32,`:``}
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${i(r,p)}
  if (workgroupIndex >= ${Math.ceil(t.inputLength/p)}u) {
    return;
  }
  let lane = localInvocationIndex;
  var value = ${s};
  var secondValue = ${s};
  var valid = 0u;
  if (index < ELEMENT_COUNT) {
    ${_}
  }
  ${b}
  if (lane == 0u) {
    outputValues[OUTPUT_OFFSET + workgroupIndex * VALUES_PER_ROW] = firstScratch[0];
    ${u?`outputValues[OUTPUT_OFFSET + workgroupIndex * VALUES_PER_ROW + 1u] = secondScratch[0];`:``}
    ${t.outputValidity?`outputValidity[OUTPUT_VALIDITY_OFFSET + workgroupIndex] = validityScratch[0];`:``}
  }
}`,w=[{buffer:t.inputValues,usage:`storage-read`},...t.inputValidity?[{buffer:t.inputValidity,usage:`storage-read`}]:[],{buffer:t.outputValues,usage:`storage-write`},...t.outputValidity?[{buffer:t.outputValidity,usage:`storage-write`}]:[]],T={inputValues:t.inputValues};return t.inputValidity&&(T.inputValidity=t.inputValidity),T.outputValues=t.outputValues,t.outputValidity&&(T.outputValidity=t.outputValidity),n.push(...O(e,{id:t.id,source:x,resources:w,bindings:T,dispatchLayout:r,record:{moduleId:`GPUReduction/level/${y}`,workgroupSize:p,invocationCount:t.inputLength,dedupName:`level`,formula:`gpu-core.reduction`}})),n}function S(e){return`firstScratch[lane] = value;
  secondScratch[lane] = secondValue;
  validityScratch[lane] = valid;
  workgroupBarrier();
  var stride = ${p/2}u;
  loop {
    if (lane < stride) {
      ${e}
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }`}function C(e){return`let subgroupValid = subgroupMax(valid);
  let subgroupFirst = ${w(e.format,e.operation===`extent`?`min`:e.operation,`value`,`valid`)};
  let subgroupSecond = ${w(e.format,e.operation===`extent`?`max`:e.operation,`secondValue`,`valid`)};
  if (subgroupInvocationId == 0u) {
    firstScratch[subgroupId] = subgroupFirst;
    secondScratch[subgroupId] = subgroupSecond;
    validityScratch[subgroupId] = subgroupValid;
  }
  workgroupBarrier();
  let subgroupCount = ${p}u / subgroupSize;
  if (subgroupCount > 1u) {
    var stride = subgroupCount / 2u;
    loop {
      if (lane < stride) {
        ${e.combine}
      }
      workgroupBarrier();
      if (stride == 1u) { break; }
      stride = stride / 2u;
    }
  }`}function w(e,t,n,r){if(t===`sum`)return`subgroupAdd(${n})`;let i=T(e,t);return`${t===`min`?`subgroupMin`:`subgroupMax`}(select(${i}, ${n}, ${r} != 0u))`}function T(e,t){return e===`uint32`?t===`min`?`4294967295u`:`0u`:e===`sint32`?t===`min`?`2147483647`:`(-2147483647 - 1)`:t===`min`?`3.402823466e+38`:`-3.402823466e+38`}function E(e,t){let n=[],r=k(t.format),i=A(t.format),a=Array.from({length:t.valuesPerRow},(e,n)=>{let r=`inputValues[INPUT_OFFSET + ${n}u]`;return`outputValues[OUTPUT_OFFSET + ${n}u] = ${t.inputValidity?`select(${i}, ${r}, valid)`:r};`}).join(`
  `),s=`
const INPUT_OFFSET: u32 = ${o(t.inputValues)}u;
const OUTPUT_OFFSET: u32 = ${o(t.output)}u;
${t.inputValidity?`const VALIDITY_OFFSET: u32 = ${o(t.inputValidity)}u;`:``}
@group(0) @binding(0) var<storage, read> inputValues: array<${r}>;
${t.inputValidity?`@group(0) @binding(1) var<storage, read> inputValidity: array<u32>;`:``}
@group(0) @binding(${t.inputValidity?2:1}) var<storage, read_write> outputValues: array<${r}>;
@compute @workgroup_size(1) fn main() {
  ${t.inputValidity?`let valid = inputValidity[VALIDITY_OFFSET] != 0u;`:``}
  ${a}
}`;return n.push(...O(e,{id:t.id,source:s,resources:[{buffer:t.inputValues,usage:`storage-read`},...t.inputValidity?[{buffer:t.inputValidity,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-write`}],bindings:{inputValues:t.inputValues,...t.inputValidity?{inputValidity:t.inputValidity}:{},outputValues:t.output},dispatchCount:1,record:{moduleId:`GPUReduction/finalize`,workgroupSize:1,invocationCount:1,formula:`gpu-core.reduction`}})),n}function D(e,t,n){let r=[],i=`${t}-clear`,a=k(n.format),s=A(n.format);return r.push(...O(e,{id:i,source:`const OUTPUT_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<${a}>;
@compute @workgroup_size(1) fn main() {
  for (var index = 0u; index < ${n.length}u; index++) { outputValues[OUTPUT_OFFSET + index] = ${s}; }
}`,resources:[{buffer:n,usage:`storage-write`}],bindings:{outputValues:n},dispatchCount:1,record:{moduleId:`GPUReduction/clear`,workgroupSize:1,invocationCount:1}})),r}function O(n,r){let i=[];i.push(t({id:r.id,resources:r.resources,compile:({device:t})=>{let n=Object.keys(r.bindings),i=new l(t,{id:r.id,source:r.source,shaderLayout:{bindings:n.map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:t,getBuffer:n})=>{let a={};for(let[t,i]of Object.entries(r.bindings))a[t]=e(i,n);r.dispatchLayout?i.dispatch(t,{bindings:a,x:r.dispatchLayout.x,y:r.dispatchLayout.y,z:r.dispatchLayout.z}):i.dispatch(t,{bindings:a,x:r.dispatchCount??1})},destroy:()=>i.destroy()}}}));let a=d(n,()=>({op:`GPUReduction`,moduleId:r.record.moduleId,source:r.source,workgroupSize:[r.record.workgroupSize,1,1],workgroups:r.dispatchLayout?[r.dispatchLayout.x,r.dispatchLayout.y,r.dispatchLayout.z]:[r.dispatchCount??1,1,1],invocationCount:r.record.invocationCount,dedupName:r.record.dedupName,formula:r.record.formula,bindings:Object.entries(r.bindings).map(([e,t])=>{let n=r.resources.find(e=>e.buffer===t)?.usage;return{name:e,view:t,access:n===`storage-read`?`read`:n===`storage-read-write`?`read_write`:`write`}})}));return a&&(i[0].annotation=a),i}function k(e){return e===`uint32`?`u32`:e===`sint32`?`i32`:`f32`}function A(e){return e===`uint32`?`0u`:e===`sint32`?`0`:`0.0`}export{h as n,g as t};