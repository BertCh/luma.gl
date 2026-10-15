import{D as e,I as t,S as n,k as r,x as i,z as a}from"./resources-CwvhEOBe.js";function o(o,s){let c=s.workgroupSize??256;if(s.bindings.length>o.device.limits.maxStorageBuffersPerShaderStage)throw Error(`${s.id} exceeds the device storage-buffer limit`);let l=i(s.operation,Math.max(s.invocationCount,1),c,o.device.limits.maxComputeWorkgroupsPerDimension),u=s.bindings.map((e,t)=>`const ${e.name}Offset: u32 = ${r(e.view)}u;\n@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),d=s.bindings.map(e=>`_ = &${e.name};`).join(`
  `),f=`
const INVOCATION_COUNT: u32 = ${s.invocationCount}u;
${u}
${s.declarations??``}

@compute @workgroup_size(${c})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${d}
  ${n(l,c)}
  if (index >= INVOCATION_COUNT) { return; }
  ${s.body}
}`,p=s.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),m=0,h=0;for(let e of s.bindings){let t=e.view.length*e.view.rowByteLength;e.access===`read`?m+=t:h+=t}return t({id:s.id,resources:p,workload:{operation:s.operation,...s.variant?{variant:s.variant}:{},commandCount:1,maximumWorkgroupCount:l.x*l.y*l.z,maximumInvocationCount:s.invocationCount,readByteLength:m,writeByteLength:h},compile:({device:t})=>{let n=new a(t,{id:s.id,source:f,shaderLayout:{bindings:s.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:t,getBuffer:r})=>{let i={};for(let t of s.bindings)i[t.name]=e(t.view,r);n.dispatch(t,{bindings:i,x:l.x,y:l.y,z:l.z})},destroy:()=>n.destroy()}}})}export{o as t};