import{J as e,mt as t}from"./src-CNCKFW2b.js";import{D as n,I as r,O as i,R as a,k as o,w as s,x as c,z as l}from"./resources-CwvhEOBe.js";import{t as u}from"./hash-computation-text-DNYT9V8A.js";var d=256;function f(e,n){let r=n.workgroupSize??d,i=n.invocationsPerItem??1,a=n.additionalInvocationCount??0,o=n.countRow??0;if(!Number.isSafeInteger(o)||o<0||o>=n.count.length||!Number.isSafeInteger(n.maximumItemCount)||n.maximumItemCount<0||!Number.isSafeInteger(i)||i<1||!Number.isSafeInteger(a)||a<0||n.maximumItemCount*i+a>4294967295)throw Error(`${n.operation}: invalid active-count dispatch topology`);let l=c(n.operation,Math.max(n.maximumItemCount*i+a,1),r,e.device.limits.maxComputeWorkgroupsPerDimension),u=s(e,`${n.id}-buffer`,`uint32`,3,t.STORAGE|t.INDIRECT);return{updateNode:h(e,{id:n.id,operation:n.operation,variant:`active-count-dispatch`,record:n.record,bindings:[{name:`count`,view:n.count,type:`u32`,access:`read`},{name:`dispatch`,view:u,type:`u32`,access:`read_write`}],invocationCount:1,workgroupSize:2,declarations:`override COUNT_ROW: u32;
override MAXIMUM_ITEM_COUNT: u32;
override INVOCATIONS_PER_ITEM: u32;
override ADDITIONAL_INVOCATION_COUNT: u32;
override TARGET_WORKGROUP_SIZE: u32;
override MAXIMUM_X: u32;
override MAXIMUM_Y: u32;`,constants:{COUNT_ROW:o,MAXIMUM_ITEM_COUNT:n.maximumItemCount,INVOCATIONS_PER_ITEM:i,ADDITIONAL_INVOCATION_COUNT:a,TARGET_WORKGROUP_SIZE:r,MAXIMUM_X:l.x,MAXIMUM_Y:l.y},body:`let itemCount = min(count[countOffset + COUNT_ROW], MAXIMUM_ITEM_COUNT);
  let invocationCount = itemCount * INVOCATIONS_PER_ITEM + ADDITIONAL_INVOCATION_COUNT;
  let workgroupCount = (invocationCount + TARGET_WORKGROUP_SIZE - 1u) / TARGET_WORKGROUP_SIZE;
  var x = min(workgroupCount, MAXIMUM_X);
  var y = 1u;
  var z = 1u;
  if (workgroupCount > MAXIMUM_X) {
    x = MAXIMUM_X;
    y = min((workgroupCount + MAXIMUM_X - 1u) / MAXIMUM_X, MAXIMUM_Y);
  }
  if (workgroupCount > MAXIMUM_X * MAXIMUM_Y) {
    y = MAXIMUM_Y;
    z = (workgroupCount + MAXIMUM_X * MAXIMUM_Y - 1u) / (MAXIMUM_X * MAXIMUM_Y);
  }
  dispatch[dispatchOffset] = x;
  dispatch[dispatchOffset + 1u] = y;
  dispatch[dispatchOffset + 2u] = z;`}),condition:{id:`${n.id}-condition`,source:`gpu`,mode:`indirect`,buffer:u.buffer,byteOffset:u.byteOffset}}}function p(e,n){let{id:r}=n,i=e.device.limits.maxComputeWorkgroupsPerDimension,a=n.shapes.map(([e,t])=>c(n.operation,Math.max(e,1),t??_(e),i)),o=s(e,r,`uint32`,3*a.length,t.STORAGE|t.INDIRECT);return{view:o,binding:{name:`dispatch`,view:o,type:`u32`,access:`read_write`},getGate:e=>({condition:{id:`${r}-${e}`,source:`gpu`,mode:`indirect`,buffer:o.buffer,byteOffset:o.byteOffset+e*12},extraResources:[{buffer:o,usage:`indirect`}]}),armWGSL:a.map((e,t)=>`dispatch[dispatchOffset + ${3*t}u] = ${e.x}u;
  dispatch[dispatchOffset + ${3*t+1}u] = ${e.y}u;
  dispatch[dispatchOffset + ${3*t+2}u] = ${e.z}u;`).join(`
  `),disarmWGSL:a.map((e,t)=>`dispatch[dispatchOffset + ${3*t}u] = 0u;`).join(` `)}}var m=32;function h(e,t){let a=t.workgroupSize??(t.guardIndex===!1?d:_(t.invocationCount)),s=e.device.limits.maxStorageBuffersPerShaderStage;if(t.bindings.length>s)throw Error(`${t.operation}: kernel ${t.id} binds ${t.bindings.length} storage buffers, more than the device limit of ${s}`);let f=e.device.limits.maxStorageBufferBindingSize;for(let e of t.bindings){let n=i(e.view).size;if(n>f)throw Error(`${t.operation}: kernel ${t.id} binding ${e.name} needs ${n} bytes, above the device limit of ${f}`)}let p=t.textures??[],m=e.device.limits.maxSampledTexturesPerShaderStage;if(p.length>m)throw Error(`${t.operation}: kernel ${t.id} binds ${p.length} sampled textures, more than the device limit of ${m}`);let h=c(t.operation,Math.max(t.invocationCount,1),a,e.device.limits.maxComputeWorkgroupsPerDimension),v=[t.bindings.map((e,t)=>`const ${e.name}Offset: u32 = ${o(e.view)}u;
@group(0) @binding(${t}) var<storage, ${e.access}> ${e.name}: array<${e.type}>;`).join(`
`),...p.map((e,n)=>`@group(0) @binding(${t.bindings.length+n}) var ${e.name}: ${g(e)};`)].join(`
`),y=[...t.bindings.map(e=>`_ = &${e.name};`),...p.map(e=>`_ = ${e.name};`)].join(`
  `),b=Math.floor(4294967295/a)+1,x=`
override INVOCATION_COUNT: u32;
override WGSL_KERNEL_DISPATCH_X: u32;
override WGSL_KERNEL_DISPATCH_Y: u32;
${v}
${t.declarations??``}

@compute @workgroup_size(${a})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${y}
  _ = INVOCATION_COUNT;
  let workgroupIndex =
    (workgroupId.z * WGSL_KERNEL_DISPATCH_Y + workgroupId.y) * WGSL_KERNEL_DISPATCH_X +
    workgroupId.x;
  if (workgroupIndex >= ${b}u) { return; }
  let index = workgroupIndex * ${a}u + localInvocationIndex;
  ${t.guardIndex===!1?``:`if (index >= INVOCATION_COUNT) {
    return;
  }`}
  ${t.body}
}`,S=t.bindings.map(e=>({buffer:e.view,usage:e.access===`read`?`storage-read`:`storage-read-write`}));S.push(...p.map(e=>({texture:e.texture,usage:`sampled`}))),S.push(...t.extraResources??[]);let C=0,w=0;for(let e of t.bindings){let t=e.view.length*e.view.rowByteLength;e.access===`read`?C+=t:w+=t}let T;if(e.recording){let e=u(x);T={...Object.fromEntries(Object.entries(t.record??{}).filter(([,e])=>e!==void 0)),op:t.operation,source:{id:`wgsl:${e}`,language:`wgsl`,moduleId:t.variant?`${t.operation}/${t.variant}`:t.operation,hash:e,bodyHash:u(t.body),entryPoint:`main`,text:x,constants:{...t.constants,INVOCATION_COUNT:t.invocationCount,WGSL_KERNEL_DISPATCH_X:h.x,WGSL_KERNEL_DISPATCH_Y:h.y}},workgroupSize:[a,1,1],dispatch:{workgroups:[h.x,h.y,h.z],invocationCount:t.invocationCount,indirect:t.condition?.source===`gpu`},bindings:[...t.bindings.map(e=>({name:e.name,resource:e.view.buffer.id,access:e.access,elementType:e.type,elementOffset:o(e.view),length:e.view.length})),...p.map(e=>({name:e.name,resource:e.texture.texture.id,access:`sampled`}))]}}let E=r({id:t.id,resources:S,condition:t.condition,workload:{operation:t.operation,...t.variant?{variant:t.variant}:{},commandCount:1,maximumWorkgroupCount:h.x*h.y*h.z,maximumInvocationCount:t.invocationCount,readByteLength:C,writeByteLength:w},compile:({device:e})=>{let r=new l(e,{id:t.id,source:x,constants:{...t.constants,INVOCATION_COUNT:t.invocationCount,WGSL_KERNEL_DISPATCH_X:h.x,WGSL_KERNEL_DISPATCH_Y:h.y},shaderLayout:{bindings:[...t.bindings.map((e,t)=>({name:e.name,type:e.access===`read`?`read-only-storage`:`storage`,group:0,location:t})),...p.map((e,n)=>({name:e.name,type:`texture`,group:0,location:t.bindings.length+n,viewDimension:e.texture.dimension,sampleType:e.sampleType}))]}}),i=[],a=[],o={},s={};return{encode:({computePass:e,getBuffer:c,getTextureView:l})=>{let u=t.bindings.map(e=>c(e.view)),d=p.map(e=>l(e.texture));if(u.some((e,t)=>e!==i[t])||d.some((e,t)=>e!==a[t])||u.length!==i.length||d.length!==a.length){i=u,a=d,o={};for(let[e,r]of t.bindings.entries()){let t=u[e];o[r.name]=n(r.view,()=>t)}for(let[e,t]of p.entries())o[t.name]=d[e];s={}}r.dispatch(e,{bindings:o,_bindGroupCacheKeys:{0:s},x:h.x,y:h.y,z:h.z})},destroy:()=>r.destroy()}}});return T&&(E.annotation=T),E}function g(e){let t={float:`f32`,"unfilterable-float":`f32`,uint:`u32`,sint:`i32`}[e.sampleType],n={"1d":`1d`,"2d":`2d`,"2d-array":`2d_array`,"3d":`3d`}[e.texture.dimension];if(!n)throw Error(`WGSL kernel texture ${e.name} must be a 1d, 2d, 2d-array or 3d view`);return`texture_${n}<${t}>`}function _(e){if(!Number.isSafeInteger(e)||e<0)throw Error(`WGSL kernel invocation count must be a non-negative safe integer`);let t=Math.max(e,m);return Math.min(d,2**Math.ceil(Math.log2(t)))}function v(e){let t=Math.fround(e);if(!Number.isFinite(t))throw Error(`WGSL float literals must be finite`);if(Object.is(t,-0))return`-0.0`;let n=String(t);return/[.eE]/.test(n)?n:`${n}.0`}function y(e,t,n){if(!(n instanceof a))return s(e,t,`uint32`,n.length);let r,i=n.data.map((n,i)=>n.length===0?(r??=s(e,`${t}-empty`,`uint32`,0),r):s(e,`${t}-chunk-${i}`,`uint32`,n.length));return new a({id:t,name:t,format:`uint32`,length:n.length,valueLength:n.length,stride:1,byteStride:4,rowByteLength:4,data:i})}function b(e,t){let{output:n,compactIds:r}=t,i=n.ids.length,a=r?Math.min(i,r.length):0,o=t.overflowSources??[],s=t.extraCounts??[],c=[{name:`totalIn`,view:t.requiredCount,type:`u32`,access:`read`}];r&&a>0&&(c.push({name:`compactIn`,view:r,type:`u32`,access:`read`}),c.push({name:`idsOut`,view:n.ids,type:`u32`,access:`read_write`}));for(let[e,t]of o.entries())c.push({name:`overflowIn${e}`,view:t,type:`u32`,access:`read`});c.push({name:`countOut`,view:n.status.count,type:`u32`,access:`read_write`}),c.push({name:`overflowOut`,view:n.status.overflow,type:`u32`,access:`read_write`}),n.status.requiredCount&&c.push({name:`totalOut`,view:n.status.requiredCount,type:`u32`,access:`read_write`});let l=a>0?t.extraColumn:void 0;l&&c.push({name:`columnIn`,view:l.source,type:`u32`,access:`read`},{name:`columnOut`,view:l.destination,type:`u32`,access:`read_write`});for(let[e,t]of s.entries())c.push({name:`extraOut${e}`,view:t,type:`u32`,access:`read_write`});if(c.length>8)throw Error(`${t.id} needs more than 8 storage bindings`);let u=[`total > CAPACITY`,...o.map((e,t)=>`overflowIn${t}[overflowIn${t}Offset] != 0u`)].join(` || `);return h(e,{id:t.id,operation:t.operation,variant:`publish`,record:t.record,bindings:c,invocationCount:Math.max(a,1),declarations:a>0?`override CAPACITY: u32;
override COPY_ROW_COUNT: u32;`:`override CAPACITY: u32;`,constants:a>0?{CAPACITY:i,COPY_ROW_COUNT:a}:{CAPACITY:i},body:`let total = totalIn[totalInOffset];
  let count = min(total, CAPACITY);
  ${a>0?`if (index < COPY_ROW_COUNT && index < count) {
    idsOut[idsOutOffset + index] = compactIn[compactInOffset + index];
    ${l?`columnOut[columnOutOffset + index] = columnIn[columnInOffset + index];`:``}
  }`:``}
  if (index == 0u) {
    countOut[countOutOffset] = count;
    overflowOut[overflowOutOffset] = select(0u, 1u, ${u});
    ${n.status.requiredCount?`totalOut[totalOutOffset] = total;`:``}
    ${s.map((e,t)=>`extraOut${t}[extraOut${t}Offset] = count;`).join(`
    `)}
  }`})}function x(e,t){let n=t.componentCount??t.view.length;return h(e,{id:t.id,operation:t.operation,variant:`fill`,record:t.record,bindings:[{name:`fillTarget`,view:t.view,type:t.type,access:`read_write`}],invocationCount:n,body:`fillTarget[fillTargetOffset + index] = ${t.value};`})}function S(t,r={}){let i=Object.entries(t),a=Object.entries(r),o=!1,s=[],c=[];return(t,r)=>{let l=i.map(([,e])=>t(e)),u=a.map(([,t])=>(e(r),r(t)));if(o&&l.every((e,t)=>e===s[t])&&u.every((e,t)=>e===c[t]))return null;o=!0,s=l,c=u;let d={};for(let[e,[t,r]]of i.entries()){let i=l[e];d[t]=n(r,()=>i)}for(let[e,[t]]of a.entries())d[t]=u[e];return d}}export{f as a,_ as c,y as i,v as l,x as n,p as o,b as r,h as s,S as t};