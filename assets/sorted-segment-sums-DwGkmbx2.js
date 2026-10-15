import{J as e}from"./src-CNCKFW2b.js";import{w as t}from"./resources-CwvhEOBe.js";import{t as n}from"./gpu-sort-Bl3Dztdb.js";import{t as r}from"./gpu-segmented-reduction-BOmwUgZN.js";import{s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as a}from"./sorted-segment-offsets-HVRgs63A.js";var o=65535,s=`var<workgroup> partialSums: array<f32, 256>;`;function c(){return`partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = ${256/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`}function l(e){return[e*256,256]}function u(e){return Math.max(1,Math.floor(e).toString(2).length)}function d(e,r){let{id:o,operation:s,segmentCount:c,segmentKeys:l}=r,d=l.length,p=[],m=t(e,`${o}-sort-keys`,`uint32`,d),h=t(e,`${o}-sort-indices`,`uint32`,d),g=t(e,`${o}-sorted-keys`,`uint32`,d),_=t(e,`${o}-sorted-indices`,`uint32`,d);p.push(i(e,{id:`${o}-sort-prepare`,operation:s,variant:`sort-prepare`,bindings:[{name:`pointFeatureRows`,view:l,type:`u32`,access:`read`},{name:`sortKeys`,view:m,type:`u32`,access:`read_write`},{name:`sortIndices`,view:h,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const FEATURE_COUNT: u32 = ${c}u;`,body:`sortKeys[sortKeysOffset + index] = min(pointFeatureRows[pointFeatureRowsOffset + index], FEATURE_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`})),p.push(...new n({id:`${o}-sort`,keys:m,values:h,outputKeys:g,outputValues:_,keyBits:u(c)}).getCommandNodes(e));let v=t(e,`${o}-segment-offsets`,`uint32`,c+1);p.push(...a(e,{id:o,operation:s,groupCount:c,sortedGroupKeys:g,segmentOffsets:v}));let y=[];r.sums&&r.sumContributions&&y.push({name:`sums`,contributions:r.sumContributions,output:r.sums}),r.weightSums&&r.weightContributions&&y.push({name:`weight-sums`,contributions:r.weightContributions,output:r.weightSums}),y.push(...r.reductions??[]);for(let n of y){let r=t(e,`${o}-sorted-${n.name}`,`float32`,d);p.push(i(e,{id:`${o}-gather-${n.name}`,operation:s,variant:`gather`,bindings:[{name:`sortedIndices`,view:_,type:`u32`,access:`read`},{name:`contributions`,view:n.contributions,type:`f32`,access:`read`},{name:`sortedContributions`,view:r,type:`f32`,access:`read_write`}],invocationCount:d,body:`sortedContributions[sortedContributionsOffset + index] =
    contributions[contributionsOffset + sortedIndices[sortedIndicesOffset + index]];`})),p.push(f(e,{id:`${o}-reduce-${n.name}`,operation:s,segmentCount:c,input:r,segmentOffsets:v,output:n.output,offsetsWithinInput:!0}))}return p}function f(t,n){if(e(n.segmentOffsets.length>=n.segmentCount+1),e(n.output.length>=n.segmentCount),n.implementation!==`kernel`&&!n.condition&&m(n)){let[e]=new r({id:n.id,input:n.input,segmentOffsets:n.segmentOffsets,output:n.output,operation:`sum`}).getCommandNodes(t);return e}return h(t,n)}function p(e){return e.byteStride===4&&e.rowByteLength===4&&e.byteOffset%4==0}function m(e){let{segmentCount:t,input:n,segmentOffsets:r,output:i}=e;return e.offsetsWithinInput===!0&&t>0&&t<=o&&r.length===t+1&&i.length===t&&p(n)&&p(r)&&p(i)&&i.buffer!==n.buffer&&i.buffer!==r.buffer}function h(e,t){return i(e,{id:t.id,operation:t.operation,variant:`segment-sum`,condition:t.condition,extraResources:t.extraResources,bindings:[{name:`segmentInput`,view:t.input,type:`f32`,access:`read`},{name:`segmentOffsets`,view:t.segmentOffsets,type:`u32`,access:`read`},{name:`segmentOutput`,view:t.output,type:`f32`,access:`read_write`}],workgroupSize:256,invocationCount:t.segmentCount*256,guardIndex:!1,declarations:`const INPUT_COUNT: u32 = ${t.input.length}u;
${s}`,body:`let segment = index / 256u;
  let isInRange = index < INVOCATION_COUNT;
  var partialSum = 0.0;
  if (isInRange) {
    let begin = segmentOffsets[segmentOffsetsOffset + segment];
    let end = min(segmentOffsets[segmentOffsetsOffset + segment + 1u], INPUT_COUNT);
    for (var row = begin + localInvocationIndex; row < end; row += 256u) {
      partialSum += segmentInput[segmentInputOffset + row];
    }
  }
  ${c()}
  if (isInRange && localInvocationIndex == 0u) {
    segmentOutput[segmentOutputOffset + segment] = partialSums[0];
  }`})}export{u as a,c as i,f as n,d as o,l as r,s as t};