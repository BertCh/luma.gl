import{a as e,i as t,n}from"./gpu-vector-format-zKtof1xn.js";import{C as r}from"./arithmetic-expression-Kknn3j_i.js";import{D as i,E as a,I as o,O as s,P as c,S as l,j as u,k as d,x as f,z as p}from"./resources-CwvhEOBe.js";import{a as m,i as h,o as g}from"./gpu-subgroup-utils-BuV3X9Xp.js";var _=256,v=Uint32Array.BYTES_PER_ELEMENT,y=class{id;source;indices;output;wordsPerRow;constructor(e){this.id=e.id??`gpu-gather`,this.source=e.source,this.indices=e.indices,this.output=e.output;for(let e of m(this.indices))u(e,`${this.id} indices`);for(let e of[this.source,this.output])for(let t of m(e))x(t,this.id,e===this.output);if(this.source.format!==this.output.format)throw Error(`${this.id} source and output must use the same format`);if(this.output.length<this.indices.length)throw Error(`${this.id} output must contain at least indices.length rows`);let t=new Set([...m(this.source).map(e=>e.buffer),...m(this.indices).map(e=>e.buffer)]);if(m(this.output).some(e=>t.has(e.buffer)))throw Error(`${this.id} output must use a separate buffer`);this.wordsPerRow=n(this.source.format).byteLength/v}getCommandNodes(e){return b(e,this)}};function b(e,t,n=0,l=`GPUGather`){let u=[];for(let n of[t.source,t.indices,t.output])for(let r of m(n))if(r.buffer.graph!==e)throw Error(`${t.id} views must belong to the target graph`);if(t.indices.length===0)return u;let d=a(e,t.output,t.indices.length),y=h(e,[t.indices,d]),b=r(m(t.source)).filter(e=>e.length),x=b.length?b:[void 0];for(let[r,[a,d]]of y.entries()){let m=f(l,a.length,_,e.device.limits.maxComputeWorkgroupsPerDimension);for(let[f,h]of x.entries()){let b=h?.data;for(let t of[d,...b?[b,a]:[]])if(s(t).size>e.device.limits.maxStorageBufferBindingSize)throw Error(`Active chunks must fit a storage binding`);let C=y.length===1&&x.length===1?t.id:`${t.id}-span-${r}-source-${f}`,w=S({source:b,sourceOffset:h?.offset??0,indices:a,output:d,wordsPerRow:t.wordsPerRow,initialize:f===0,invalidValue:n,dispatchLayout:m}),T=c(o({id:C,workload:{operation:l,commandCount:1,maximumWorkgroupCount:m.x*m.y*m.z,maximumInvocationCount:m.x*m.y*m.z*_,readByteLength:b?a.length*(v+b.rowByteLength):0,writeByteLength:a.length*d.rowByteLength},resources:[...b?[{buffer:b,usage:`storage-read`},{buffer:a,usage:`storage-read`}]:[],{buffer:d,usage:`storage-write`}],compile:({device:e})=>{let t=new p(e,{id:C,source:w,shaderLayout:{bindings:[{name:`outputWords`,type:`storage`,group:0,location:0},...b?[{name:`sourceWords`,type:`read-only-storage`,group:0,location:1},{name:`indices`,type:`read-only-storage`,group:0,location:2}]:[]]}});return{encode:({computePass:e,getBuffer:n})=>{let r={outputWords:i(d,n)};b&&(r.sourceWords=i(b,n),r.indices=i(a,n)),t.dispatch(e,{bindings:r,x:m.x,y:m.y,z:m.z})},destroy:()=>t.destroy()}}}),[m.x,m.y,m.z]),E=g(e,()=>({op:l,moduleId:`${l}/${b?`copy-rows`:`fill-invalid`}`,source:w,workgroupSize:[_,1,1],workgroups:[m.x,m.y,m.z],invocationCount:a.length,...l===`GPUGather`&&{formula:`gpu-core.gather`},bindings:[{name:`outputWords`,view:d,access:`write`},...b?[{name:`sourceWords`,view:b,access:`read`},{name:`indices`,view:a,access:`read`}]:[]]}));E&&(T.annotation=E),u.push(T)}}return u}function x(r,i,a){if(e(r.format)||t(r.format))throw Error(`${i} must use a fixed-width GPU data format`);let o=n(r.format);if((a?r.byteStride!==o.byteLength:r.byteStride<o.byteLength||r.byteStride%v!==0)||r.rowByteLength!==o.byteLength||r.byteOffset%v!==0||r.rowByteLength%v!==0)throw Error(`${i} must be packed, uint32-aligned GPU data with 32-bit-word rows`)}function S(e){let{source:t,indices:n,output:r,wordsPerRow:i,initialize:a,invalidValue:o,dispatchLayout:s}=e;return`const INDEX_COUNT: u32 = ${n.length}u;
const WORDS_PER_ROW: u32 = ${i}u;
const OUTPUT_OFFSET: u32 = ${d(r)}u;
@group(0) @binding(0) var<storage, read_write> outputWords: array<u32>;
${t?`
const SOURCE_LENGTH: u32 = ${t.length}u;
const SOURCE_ROW_OFFSET: u32 = ${e.sourceOffset}u;
const SOURCE_OFFSET: u32 = ${d(t)}u;
const INDEX_OFFSET: u32 = ${d(n)}u;
@group(0) @binding(1) var<storage, read> sourceWords: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
`:``}
@compute @workgroup_size(${_})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(s,_)}
  if (index >= INDEX_COUNT) { return; }
  let outputBase = OUTPUT_OFFSET + index * WORDS_PER_ROW;
${t?`
  let sourceIndex = indices[INDEX_OFFSET + index];
  if (sourceIndex >= SOURCE_ROW_OFFSET && sourceIndex - SOURCE_ROW_OFFSET < SOURCE_LENGTH) {
    let sourceBase = SOURCE_OFFSET + (sourceIndex - SOURCE_ROW_OFFSET) * ${t.byteStride/v}u;
    for (var word = 0u; word < WORDS_PER_ROW; word++) {
      outputWords[outputBase + word] = sourceWords[sourceBase + word];
    }
    return;
  }
`:``}
${a?`
  for (var word = 0u; word < WORDS_PER_ROW; word++) {
    outputWords[outputBase + word] = ${o}u;
  }
`:``}
}`}var C=class extends y{invalidValue;constructor(e){if(super({...e,id:e.id??`gpu-uint32-gather`}),this.invalidValue=e.invalidValue??0,this.source.format!==`uint32`)throw Error(`${this.id} source must use uint32`);if(!Number.isSafeInteger(this.invalidValue)||this.invalidValue<0||this.invalidValue>4294967295)throw Error(`${this.id} invalidValue must be a uint32`)}getCommandNodes(e){return b(e,this,this.invalidValue,`GPUUint32Gather`)}};export{y as n,C as t};