import{R as e,k as t}from"./resources-CwvhEOBe.js";import{t as n}from"./hash-computation-text-DNYT9V8A.js";function r(e,r){if(!e.recording)return;let{source:i,moduleId:a,entryPoint:o,constants:s,workgroupSize:c,workgroups:l,invocationCount:u,bindings:d,...f}=r(),p=n(i);return{...Object.fromEntries(Object.entries(f).filter(([,e])=>e!==void 0)),source:{id:`wgsl:${p}`,language:`wgsl`,moduleId:a,hash:p,entryPoint:o??`main`,text:i,...s&&{constants:s}},workgroupSize:c,dispatch:{workgroups:l,...u!==void 0&&{invocationCount:u},indirect:!1},...d&&{bindings:d.map(({name:e,view:n,access:r})=>({name:e,resource:`buffer`in n?n.buffer.id:n.id,access:r,...`buffer`in n&&{elementType:n.format===`uint32`?`u32`:n.format===`sint32`?`i32`:n.format===`float32`?`f32`:n.format,elementOffset:t(n),length:n.length}}))}}}function i(t){return t instanceof e?t.data:[t]}function a(e,t){let n=t[0]?.length??0;if(t.some(e=>e.length!==n))throw Error(`Aligned vectors must have equal logical lengths`);let r=t.map(i),a=t.map(()=>0),o=t.map(()=>0),s=[],c=0;for(;c<n;){for(let e=0;e<r.length;e++)for(;r[e][a[e]]?.length===o[e];)a[e]++,o[e]=0;let t=r.map((e,t)=>e[a[t]]),n=Math.min(...t.map((e,t)=>e.length-o[t])),i=[];for(let r=0;r<t.length;r++){let a=t[r],s=t.findIndex((e,t)=>t<r&&e===a&&o[t]===o[r]);i.push(s>=0?i[s]:o[r]===0&&n===a.length?a:e.createDataView(a.buffer,{format:a.format,length:n,byteOffset:a.byteOffset+o[r]*a.byteStride,byteStride:a.byteStride,rowByteLength:a.rowByteLength}))}s.push(i);for(let e=0;e<o.length;e++)o[e]+=n;c+=n}return s}function o(e,t={}){let n=e.features?.has(`subgroups`),r=!t.requiresSubgroupId||e.wgslLanguageFeatures?.has(`subgroup_id`);return n&&r?`subgroups`:`portable`}function s(){return`
fn getFirstBallotLane(mask: vec4<u32>) -> u32 {
  if (mask.x != 0u) { return firstTrailingBit(mask.x); }
  if (mask.y != 0u) { return 32u + firstTrailingBit(mask.y); }
  if (mask.z != 0u) { return 64u + firstTrailingBit(mask.z); }
  if (mask.w != 0u) { return 96u + firstTrailingBit(mask.w); }
  return 0u;
}

fn getBallotLaneCount(mask: vec4<u32>) -> u32 {
  return countOneBits(mask.x) + countOneBits(mask.y) +
    countOneBits(mask.z) + countOneBits(mask.w);
}

fn getBallotPrefixLaneCount(mask: vec4<u32>, invocation: u32) -> u32 {
  let word = invocation >> 5u;
  let bit = invocation & 31u;
  var count = 0u;
  if (word > 0u) { count += countOneBits(mask.x); }
  if (word > 1u) { count += countOneBits(mask.y); }
  if (word > 2u) { count += countOneBits(mask.z); }
  let lowerBits = (1u << bit) - 1u;
  return count + countOneBits(mask[word] & lowerBits);
}`}function c(e,t,n,r){return`
  var subgroupPending = ${e};
  for (var subgroupGroup = 0u; subgroupGroup < ${r}u; subgroupGroup++) {
    let pendingBallot = subgroupBallot(subgroupPending);
    let hasPending = any(pendingBallot != vec4<u32>(0u));
    let leaderInvocation = getFirstBallotLane(pendingBallot);
    let leaderKey = subgroupShuffle(${t}, leaderInvocation);
    let matchingKey = hasPending && subgroupPending && ${t} == leaderKey;
    let matchingBallot = subgroupBallot(matchingKey);
    if (hasPending && subgroupInvocationId == leaderInvocation) {
      atomicAdd(&${n}[leaderKey], getBallotLaneCount(matchingBallot));
    }
    subgroupPending = subgroupPending && !matchingKey;
  }`}export{i as a,a as i,s as n,r as o,c as r,o as t};