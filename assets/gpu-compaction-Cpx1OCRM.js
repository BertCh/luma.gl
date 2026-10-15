import{C as e,D as t,I as n,R as r,S as i,j as a,k as o,w as s,x as c,z as l}from"./resources-CwvhEOBe.js";import{i as u,o as d}from"./gpu-subgroup-utils-BuV3X9Xp.js";import{n as f,t as p}from"./gpu-scan-BuGhTLDQ.js";var m=256,h=class{id;input;flags;output;count;constructor(e){if(this.id=e.id??`gpu-compaction`,this.input=e.input,this.flags=e.flags,this.output=e.output,this.count=e.count,S(this.input,`${this.id} input`),S(this.flags,`${this.id} flags`),S(this.output,`${this.id} output`),a(this.count,`${this.id} count`),this.flags.length!==this.input.length)throw Error(`${this.id} flags.length must equal input.length`);if(this.output.length<this.input.length)throw Error(`${this.id} output must contain at least input.length rows`);if(this.count.length<1)throw Error(`${this.id} count must contain one uint32 row`)}getCommandNodes(e){let t=[];return t.push(...g(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)),t}};function g(t,n,i){let a=[];for(let e of[...x(t.input),...x(t.flags),...x(t.output),t.count])if(e.buffer.graph!==n)throw Error(`${t.id} views must belong to the target graph`);if(t.input.length===0)return a.push(..._(n,t.id,t.count)),a;let o=t.flags instanceof r?e(n,`${t.id}-offsets`,t.flags):s(n,`${t.id}-offsets`,`uint32`,t.flags.length),c=new p({id:`${t.id}-scan`,input:t.flags,output:o});return a.push(...f(c,n,i)),a.push(...v(n,t.id,t.input,t.flags,o,t.output,t.count,i)),a}function _(e,r,i){let a=[],s=()=>`const COUNT_OFFSET: u32 = ${o(i)}u;
@group(0) @binding(0) var<storage, read_write> outputCount: array<u32>;
@compute @workgroup_size(1) fn main() { outputCount[COUNT_OFFSET] = 0u; }`,c=`${r}-clear-count`;a.push(n({id:c,resources:[{buffer:i,usage:`storage-write`}],compile:({device:e})=>{let n=new l(e,{id:c,source:s(),shaderLayout:{bindings:[{name:`outputCount`,type:`storage`,group:0,location:0}]}});return{encode:({computePass:e,getBuffer:r})=>{n.dispatch(e,{bindings:{outputCount:t(i,r)},x:1})},destroy:()=>n.destroy()}}}));let u=d(e,()=>({op:`GPUCompaction`,moduleId:`GPUCompaction/clear-count`,source:s(),workgroupSize:[1,1,1],workgroups:[1,1,1],invocationCount:1,formula:`gpu-core.compaction`,bindings:[{name:`outputCount`,view:i,access:`read_write`}]}));return u&&(a[0].annotation=u),a}function v(e,t,n,i,a,o,s,l){let d=[],f=u(e,[n,i,a]),p=f.map(e=>e[0]),h=f.map(e=>e[1]),g=f.map(e=>e[2]),_=x(o),v=p.map((e,t)=>e.length>0?t:-1).filter(e=>e>=0),b=_.map((e,t)=>e.length>0?t:-1).filter(e=>e>=0),S=v[v.length-1],C=b[b.length-1],w=f.length>1||o instanceof r,T=0;for(let n=0;n<_.length;n++){let r=_[n],i=T+r.length;if(r.length>0)for(let a of v){let o=a===S&&n===C;d.push(...y(e,{id:w?`${t}-scatter-input-${a}-output-${n}`:`${t}-scatter`,input:p[a],flags:h[a],offsets:g[a],output:r,outputStart:T,outputEnd:i,count:o?s:void 0,dispatchLayout:c(`GPUCompaction`,p[a].length,m,l)}))}T=i}return d}function y(e,t){let n=[],r=t.count?`@group(0) @binding(4) var<storage, read_write> outputCount: array<u32>;`:``,a=t.count?`if (index == ELEMENT_COUNT - 1u) {
    outputCount[COUNT_OFFSET] = outputIndex + flag;
  }`:``,s=`
const ELEMENT_COUNT: u32 = ${t.input.length}u;
const INPUT_OFFSET: u32 = ${o(t.input)}u;
const FLAGS_OFFSET: u32 = ${o(t.flags)}u;
const OFFSETS_OFFSET: u32 = ${o(t.offsets)}u;
const OUTPUT_OFFSET: u32 = ${o(t.output)}u;
const OUTPUT_START: u32 = ${t.outputStart}u;
const OUTPUT_END: u32 = ${t.outputEnd}u;
${t.count?`const COUNT_OFFSET: u32 = ${o(t.count)}u;`:``}
@group(0) @binding(0) var<storage, read> inputValues: array<u32>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;
${r}

@compute @workgroup_size(${m}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${i(t.dispatchLayout,m)}
  if (index >= ELEMENT_COUNT) { return; }
  let flag = min(flags[FLAGS_OFFSET + index], 1u);
  let outputIndex = offsets[OFFSETS_OFFSET + index];
  if (flag != 0u && outputIndex >= OUTPUT_START && outputIndex < OUTPUT_END) {
    outputValues[OUTPUT_OFFSET + outputIndex - OUTPUT_START] = inputValues[INPUT_OFFSET + index];
  }
  ${a}
}`;return n.push(...b(e,{id:t.id,source:s,resources:[{buffer:t.input,usage:`storage-read`},{buffer:t.flags,usage:`storage-read`},{buffer:t.offsets,usage:`storage-read`},{buffer:t.output,usage:`storage-write`},...t.count?[{buffer:t.count,usage:`storage-write`}]:[]],bindings:{inputValues:t.input,flags:t.flags,offsets:t.offsets,outputValues:t.output,...t.count?{outputCount:t.count}:{}},dispatchLayout:t.dispatchLayout})),n}function b(e,r){let i=[];i.push(n({id:r.id,workload:{operation:`GPUCompaction`,commandCount:1,maximumWorkgroupCount:r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z,maximumInvocationCount:r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z*m,readByteLength:r.resources.reduce((e,t)=>e+(t.usage===`storage-read`||t.usage===`storage-read-write`?t.buffer.length*t.buffer.rowByteLength:0),0),writeByteLength:r.resources.reduce((e,t)=>e+(t.usage===`storage-write`||t.usage===`storage-read-write`?t.buffer.length*t.buffer.rowByteLength:0),0)},resources:r.resources,compile:({device:e})=>{let n=new l(e,{id:r.id,source:r.source,shaderLayout:{bindings:Object.keys(r.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,n]of Object.entries(r.bindings))a[e]=t(n,i);n.dispatch(e,{bindings:a,x:r.dispatchLayout.x,y:r.dispatchLayout.y,z:r.dispatchLayout.z})},destroy:()=>n.destroy()}}}));let a=d(e,()=>({op:`GPUCompaction`,moduleId:`GPUCompaction/scatter`,source:r.source,workgroupSize:[m,1,1],workgroups:[r.dispatchLayout.x,r.dispatchLayout.y,r.dispatchLayout.z],invocationCount:r.bindings.inputValues.length,formula:`gpu-core.compaction`,bindings:Object.entries(r.bindings).map(([e,t])=>({name:e,view:t,access:r.resources.find(e=>e.buffer===t)?.usage===`storage-read`?`read`:`read_write`}))}));return a&&(i[0].annotation=a),i}function x(e){return e instanceof r?e.data:[e]}function S(e,t){for(let n of x(e))a(n,t)}export{g as n,h as t};