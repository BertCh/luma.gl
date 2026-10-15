import{D as e,k as t}from"./resources-CwvhEOBe.js";import{u as n}from"./interleave-brZZCO-f.js";var r=256;function i(i,a){let o=Math.ceil(a.invocationCount/r);if(o>i.device.limits.maxComputeWorkgroupsPerDimension)throw Error(`${a.id} needs more workgroups than one dispatch dimension allows`);let s=a.bindings.map((e,n)=>`const ${e.name}Offset: u32 = ${t(e.view)}u;
@group(0) @binding(${n}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),c=`
const INVOCATION_COUNT: u32 = ${a.invocationCount}u;
${s}
${a.declarations??``}

@compute @workgroup_size(${r})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= INVOCATION_COUNT) {
    return;
  }
  ${a.body}
}`,l=0,u=0;for(let e of a.bindings){let t=e.view.length*e.view.rowByteLength;e.access===`read`?l+=t:u+=t}i.addComputePass({id:a.id,workload:{operation:`ExplorerModeKernel`,variant:a.id,commandCount:1,maximumWorkgroupCount:o,maximumInvocationCount:o*r,readByteLength:l,writeByteLength:u},resources:a.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),compile:({device:t})=>{let r=new n(t,{id:a.id,source:c,shaderLayout:{bindings:a.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:t,getBuffer:n})=>{let i={};for(let t of a.bindings)i[t.name]=e(t.view,n);r.setBindings(i),r.dispatch(t,o)},destroy:()=>r.destroy()}}})}export{i as t};