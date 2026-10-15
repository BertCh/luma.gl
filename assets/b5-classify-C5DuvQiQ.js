import{D as e,k as t}from"./resources-CwvhEOBe.js";import{u as n}from"./interleave-brZZCO-f.js";import{t as r}from"./graph-buffers-B4_80RlX.js";function i(e){let t=new Map,n=new Map;return{float:(n,i)=>{let a=t.get(n);return a||(a=r(e,n.id,n,`float32`,i),t.set(n,a)),a},word:(t,i)=>{let a=n.get(t);return a||(a=r(e,t.id,t,`uint32`,i),n.set(t,a)),a}}}function a(r,i){let a=i.maximumClassCount+1,o=[{name:`parameters`,view:i.parameters,access:`read`,type:`f32`},{name:`primaryBreaks`,view:i.primaryBreaks,access:`read`,type:`f32`},{name:`primaryClassCount`,view:i.primaryClassCount,access:`read`,type:`u32`},{name:`alternateBreaks`,view:i.alternateBreaks,access:`read`,type:`f32`},{name:`alternateClassCount`,view:i.alternateClassCount,access:`read`,type:`u32`},{name:`selectedBreaks`,view:i.breaks,access:`read_write`,type:`f32`},{name:`selectedClassCount`,view:i.classCount,access:`read_write`,type:`u32`}],s=`
${o.map((e,n)=>`const ${e.name}Offset: u32 = ${t(e.view)}u;
@group(0) @binding(${n}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`)}
@compute @workgroup_size(${a})
fn main(@builtin(local_invocation_id) invocation: vec3<u32>) {
  let useAlternate = u32(parameters[parametersOffset]) == ${i.alternateMethodCode}u;
  let index = invocation.x;
  selectedBreaks[selectedBreaksOffset + index] = select(
    primaryBreaks[primaryBreaksOffset + index],
    alternateBreaks[alternateBreaksOffset + index],
    useAlternate
  );
  if (index == 0u) {
    selectedClassCount[selectedClassCountOffset] = select(
      primaryClassCount[primaryClassCountOffset],
      alternateClassCount[alternateClassCountOffset],
      useAlternate
    );
  }
}`;r.addComputePass({id:i.id,workload:{operation:`ClassificationBreaksSelect`,variant:i.id,commandCount:1,maximumWorkgroupCount:1,maximumInvocationCount:a,readByteLength:a*8+12,writeByteLength:a*4+4},resources:o.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`})),compile:({device:t})=>{let r=new n(t,{id:i.id,source:s,shaderLayout:{bindings:o.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:t,getBuffer:n})=>{let i={};for(let t of o)i[t.name]=e(t.view,n);r.setBindings(i),r.dispatch(t,1)},destroy:()=>r.destroy()}}})}export{i as n,a as t};