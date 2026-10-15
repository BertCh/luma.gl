import{w as e}from"./resources-CwvhEOBe.js";import{s as t}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var n=256,r=4096,i=n;function a(a,o){let{id:s,operation:c,variant:l,itemCount:u,output:d}=o,f=Math.min(i,Math.max(1,Math.ceil(u/r))),p=`const ITEM_COUNT: u32 = ${u}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const PARTIAL_COUNT: u32 = ${f}u;
const WORKGROUP_THREADS: u32 = ${n}u;
${o.declarations}
var<workgroup> sharedMinima: array<vec2f, ${n}>;
var<workgroup> sharedMaxima: array<vec2f, ${n}>;`,m=`sharedMinima[localInvocationIndex] = localMinimum;
  sharedMaxima[localInvocationIndex] = localMaximum;
  workgroupBarrier();
  for (var stride = ${n/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedMinima[localInvocationIndex] = min(sharedMinima[localInvocationIndex], sharedMinima[localInvocationIndex + stride]);
      sharedMaxima[localInvocationIndex] = max(sharedMaxima[localInvocationIndex], sharedMaxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }`,h=e=>`if (localInvocationIndex == 0u) {
    let minimum = sharedMinima[0];
    let maximum = sharedMaxima[0];
    let hasValid = minimum.x <= maximum.x && minimum.y <= maximum.y;
    ${e}[${e}Offset] = select(0.0, minimum.x, hasValid);
    ${e}[${e}Offset + 1u] = select(0.0, minimum.y, hasValid);
    ${e}[${e}Offset + 2u] = select(0.0, maximum.x, hasValid);
    ${e}[${e}Offset + 3u] = select(0.0, maximum.y, hasValid);
  }`,g=`var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  // Grid-stride loop: consecutive threads read consecutive items on every trip.
  for (var row = index; row < ITEM_COUNT; row += PARTIAL_COUNT * WORKGROUP_THREADS) {
    let item = readItem(row);
    if (item.z > 0.5) {
      localMinimum = min(localMinimum, item.xy);
      localMaximum = max(localMaximum, item.xy);
    }
  }
  ${m}`;if(f===1)return[t(a,{id:`${s}-bounds`,operation:c,variant:l,bindings:[...o.bindings,{name:`boundsOut`,view:d,type:`f32`,access:`read_write`}],invocationCount:n,guardIndex:!1,declarations:p,body:`// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  ${g}
  ${h(`boundsOut`)}`})];let _=e(a,`${s}-bounds-partials`,`float32`,f*4);return[t(a,{id:`${s}-bounds-blocks`,operation:c,variant:`${l}-blocks`,bindings:[...o.bindings,{name:`partials`,view:_,type:`f32`,access:`read_write`}],invocationCount:f*n,guardIndex:!1,declarations:p,body:`${g}
  if (localInvocationIndex == 0u) {
    let block = index / WORKGROUP_THREADS;
    partials[partialsOffset + block * 4u] = sharedMinima[0].x;
    partials[partialsOffset + block * 4u + 1u] = sharedMinima[0].y;
    partials[partialsOffset + block * 4u + 2u] = sharedMaxima[0].x;
    partials[partialsOffset + block * 4u + 3u] = sharedMaxima[0].y;
  }`}),t(a,{id:`${s}-bounds`,operation:c,variant:`${l}-merge`,bindings:[{name:`partials`,view:_,type:`f32`,access:`read`},{name:`boundsOut`,view:d,type:`f32`,access:`read_write`}],invocationCount:n,guardIndex:!1,declarations:`const PARTIAL_COUNT: u32 = ${f}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
var<workgroup> sharedMinima: array<vec2f, ${n}>;
var<workgroup> sharedMaxima: array<vec2f, ${n}>;`,body:`// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  if (localInvocationIndex < PARTIAL_COUNT) {
    let base = partialsOffset + localInvocationIndex * 4u;
    localMinimum = vec2f(partials[base], partials[base + 1u]);
    localMaximum = vec2f(partials[base + 2u], partials[base + 3u]);
  }
  ${m}
  ${h(`boundsOut`)}`})]}export{a as t};