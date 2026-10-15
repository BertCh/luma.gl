import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{N as e,_ as t,d as n,f as r,h as i,i as a,n as o,t as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{t as u}from"./gpu-scan-BuGhTLDQ.js";import{t as d}from"./gpu-reduction-BD8l40jM.js";import{n as f,r as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as h}from"./load-balanced-for-each-OquID6hm.js";import{t as g}from"./time-words-C-gLAZQw.js";import{c as _,r as v}from"./index-Bjk18V2I.js";import{t as y}from"./graph-buffers-B4_80RlX.js";import{t as b}from"./graph-swap-41Ru7LVu.js";import{t as x}from"./summary-reader-CsgWxENf.js";import{t as S}from"./mode-kernels-Dxm2GWvI.js";import{l as C,m as w}from"./class-table-BVEo3-Vs.js";import{t as T}from"./gpu-graph-topology-view-DhuT3ULg.js";import{n as E,r as D,t as O}from"./gpu-graph-page-rank-BugwiSFh.js";import{t as k}from"./gpu-graph-connected-components-aw-M009a.js";import{t as A}from"./gpu-graph-label-propagation-BJ25J5mS.js";import{i as j,n as M}from"./network-topology-BXzNuj-N.js";import{t as N}from"./network-workgroup-counters-CFyh5ZJ-.js";import{a as P,i as F,n as I,t as L}from"./b10-road-graph-63jDTifd.js";import{n as R,t as z}from"./b10-scene-constants-DbEpJ5O7.js";var B=class{id;props;nodeCount;metrics;constructor(e){this.id=e.id??`network-analytics-columns`,this.props=e;let{id:t}=this;this.nodeCount=j(t,e.topology).nodeCount;let{nodeCount:n}=this;if(e.reverseTopology&&(l(e.reverseTopology.offsets,[`uint32`],`${t} reverseTopology.offsets`),l(e.reverseTopology.neighbors,[`uint32`],`${t} reverseTopology.neighbors`),e.reverseTopology.offsets.length!==n+1))throw Error(`${t} reverseTopology.offsets must contain the same number of rows as topology.offsets`);if(!e.degree&&!e.inDegree&&!e.pageRank&&!e.coreNumber&&!e.components&&!e.communities)throw Error(`${t} requires at least one metric`);if(e.inDegree&&!e.reverseTopology)throw Error(`${t} inDegree requires reverseTopology`);let r=[],i=[],a=[];for(let[i,o,s]of[[`degree`,e.degree,`uint32`],[`inDegree`,e.inDegree,`uint32`],[`pageRank`,e.pageRank,`float32`],[`coreNumber`,e.coreNumber,`uint32`]])if(o&&(r.push([`${i}.output`,o.output,s]),o.normalized&&(a.push([`${i}.normalized`,o.normalized,n]),l(o.normalized,[`float32`],`${t} ${i}.normalized`)),o.extent&&(a.push([`${i}.extent`,o.extent,2]),l(o.extent,[s],`${t} ${i}.extent`)),o.extent&&!o.normalized))throw Error(`${t} ${i}.extent requires ${i}.normalized`);e.components&&r.push([`components.output`,e.components.output,`uint32`]),e.communities&&r.push([`communities.output`,e.communities.output,`uint32`]),i.push([`pageRank.residual`,e.pageRank?.residual,`float32`],[`coreNumber.status.converged`,e.coreNumber?.status?.converged,`uint32`],[`coreNumber.degeneracy`,e.coreNumber?.degeneracy,`uint32`],[`components.status.converged`,e.components?.status?.converged,`uint32`],[`communities.status.converged`,e.communities?.status?.converged,`uint32`]);for(let[e,i,a]of r)if(l(i,[a],`${t} ${e}`),i.length!==n)throw Error(`${t} ${e} length must equal the node count`);for(let[e,n,r]of i)if(n&&(l(n,[r],`${t} ${e}`),n.length!==1))throw Error(`${t} ${e} must contain exactly one row`);for(let[e,n,r]of a)if(n&&n.length!==r)throw Error(`${t} ${e} must contain ${r===2?`exactly two rows`:`one row per node`}`);let o=[...M(e.topology),e.reverseTopology?.offsets,e.reverseTopology?.neighbors].filter(e=>e!==void 0).map(e=>e.buffer),s=[...r.map(([,e])=>e),...i.map(([,e])=>e),...a.map(([,e])=>e)].filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(s).size!==s.length||s.some(e=>o.includes(e)))throw Error(`${t} outputs must use separate buffers from each other and from inputs`);this.metrics=this.createMetrics()}getCommandNodes(e){let{id:n,props:r,nodeCount:i}=this;t(n,e,V(r));let o=[];for(let t of this.metrics)o.push(...a(e,()=>t.algorithm.addToGraph(e)));for(let t of this.metrics){if(!t.normalized||!t.raw||!t.format)continue;let r=t.extent??c(e,`${n}-${t.step}-extent-scratch`,t.format,2);o.push(...new d({id:`${n}-${t.step}-extent`,input:t.raw,output:r,operation:`extent`}).getCommandNodes(e));let a=t.format===`uint32`?`u32`:`f32`;o.push(m(e,{id:`${n}-${t.step}-normalize`,operation:`GPUNetworkAnalyticsColumns`,variant:`normalize`,record:{datum:{keySpace:`node`},formula:`gpu-network.analytics.normalize`},bindings:[{name:`raw`,view:t.raw,type:a,access:`read`},{name:`extent`,view:r,type:a,access:`read`},{name:`normalized`,view:t.normalized,type:`f32`,access:`read_write`}],invocationCount:i,body:`let lowest = f32(extent[extentOffset]);
  let highest = f32(extent[extentOffset + 1u]);
  let range = highest - lowest;
  var scaled = 0.0;
  if (range > 0.0) {
    scaled = (f32(raw[rawOffset + index]) - lowest) / range;
  }
  normalized[normalizedOffset + index] = scaled;`}))}return o}createMetrics(){let{id:e,props:t,nodeCount:n}=this,r=new T({id:`${e}-topology`,vertexCount:n,forward:{offsets:t.topology.offsets,neighbors:t.topology.neighbors},reverse:t.reverseTopology?{offsets:t.reverseTopology.offsets,neighbors:t.reverseTopology.neighbors}:void 0}),i=[];if(t.degree&&i.push({step:`degree`,algorithm:new D({id:`${e}-degree`,topology:r,output:t.degree.output,direction:`outgoing`}),format:`uint32`,raw:t.degree.output,normalized:t.degree.normalized,extent:t.degree.extent}),t.inDegree&&i.push({step:`in-degree`,algorithm:new D({id:`${e}-in-degree`,topology:r,output:t.inDegree.output,direction:`incoming`}),format:`uint32`,raw:t.inDegree.output,normalized:t.inDegree.normalized,extent:t.inDegree.extent}),t.pageRank){let{pageRank:n}=t;i.push({step:`page-rank`,algorithm:new O({id:`${e}-page-rank`,topology:r,output:n.output,damping:n.damping,iterations:n.iterationCount,residual:n.residual&&n.residual}),format:`float32`,raw:n.output,normalized:n.normalized,extent:n.extent})}if(t.coreNumber){let{coreNumber:n}=t;i.push({step:`core-number`,algorithm:new E({id:`${e}-core-number`,topology:r,output:n.output,iterations:n.maximumIterations,converged:n.status?.converged,degeneracy:n.degeneracy&&n.degeneracy}),format:`uint32`,raw:n.output,normalized:n.normalized,extent:n.extent})}if(t.components){let{components:n}=t;i.push({step:`components`,algorithm:new k({id:`${e}-components`,topology:r,output:n.output,iterations:n.maximumIterations,converged:n.status?.converged})})}if(t.communities){let{communities:n}=t;i.push({step:`communities`,algorithm:new A({id:`${e}-communities`,topology:r,output:n.output,iterations:n.maximumIterations,converged:n.status?.converged})})}return i}};function V(e){return[...M(e.topology),e.reverseTopology?.offsets,e.reverseTopology?.neighbors,...[e.degree,e.inDegree,e.pageRank,e.coreNumber].flatMap(e=>[e?.output,e?.normalized,e?.extent]),e.pageRank?.residual,e.coreNumber?.status?.converged,e.coreNumber?.degeneracy,e.components?.output,e.components?.status?.converged,e.communities?.output,e.communities?.status?.converged]}var H={liveNodeCount:0,liveEdgeCount:1,liveSlotCount:2,componentCount:3,largestComponentSize:4,componentsConverged:5,isolatedNodeCount:6,maximumOutDegree:7,maximumInDegree:8,maximumTotalDegree:9,modularity:10,modularityValid:11,intraCommunitySlotCount:12,selfLoopSlotCount:13};function U(e=32){return 16+3*e}function ee(e={}){let t=new Uint32Array(2);return new Float32Array(t.buffer,0,1)[0]=e.resolution??1,t[1]=Math.max(1,Math.floor(e.degreeBinWidth??1)),t}function te(e,t={}){let n=t.degreeBinCount??32;if(e.length<U(n))throw Error(`decodeGPUNetworkStatistics needs the full summary`);let r=H,i=t=>Array.from({length:n},(r,i)=>e[16+t*n+i]),a=new Uint32Array([e[r.modularity]]);return{liveNodeCount:e[r.liveNodeCount],liveEdgeCount:e[r.liveEdgeCount],liveSlotCount:e[r.liveSlotCount],componentCount:e[r.componentCount],largestComponentSize:e[r.largestComponentSize],componentsConverged:e[r.componentsConverged]!==0,isolatedNodeCount:e[r.isolatedNodeCount],maximumOutDegree:e[r.maximumOutDegree],maximumInDegree:e[r.maximumInDegree],maximumTotalDegree:e[r.maximumTotalDegree],modularity:new Float32Array(a.buffer)[0],modularityValid:e[r.modularityValid]!==0,intraCommunitySlotCount:e[r.intraCommunitySlotCount],selfLoopSlotCount:e[r.selfLoopSlotCount],outDegreeHistogram:i(0),inDegreeHistogram:i(1),totalDegreeHistogram:i(2)}}var W=256,ne=1024,re=class{id;props;nodeCount;degreeBinCount;constructor(e){this.id=e.id??`network-statistics`,this.props=e;let{id:t}=this;this.nodeCount=j(t,e.topology).nodeCount;let{nodeCount:r}=this;if(this.degreeBinCount=e.degreeBinCount??32,n(t,`degreeBinCount`,this.degreeBinCount,1,4096),e.degreeBinning&&e.degreeBinning!==`linear`&&e.degreeBinning!==`log2`)throw Error(`${t} degreeBinning must be 'linear' or 'log2'`);n(t,`maximumComponentIterations`,e.maximumComponentIterations??32,1,1024);for(let[n,i,a]of[[`nodeMask`,e.nodeMask,r],[`communities`,e.communities,r],[`edgeMask`,e.edgeMask,e.topology.neighbors.length]])if(i&&(l(i,[`uint32`],`${t} ${n}`),i.length!==a))throw Error(`${t} ${n} must contain one row per ${n===`edgeMask`?`neighbor slot`:`node`}`);if(e.parameters&&(l(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<2))throw Error(`${t} parameters must contain at least two uint32 rows`);if(l(e.output,[`uint32`],`${t} output`),e.output.length!==U(this.degreeBinCount))throw Error(`${t} output must contain exactly ${U(this.degreeBinCount)} rows`);i(t,[e.output],[...M(e.topology),e.nodeMask,e.edgeMask,e.communities,e.parameters])}getCommandNodes(e){let{id:n,props:r,nodeCount:i,degreeBinCount:o}=this;t(n,e,[...M(r.topology),r.nodeMask,r.edgeMask,r.communities,r.parameters,r.output]);let s=!!r.directed,l=!s,u=!!r.communities,d=H,p=`GPUNetworkStatistics`,{offsets:g,neighbors:_}=r.topology,v=c(e,`${n}-masked-neighbors`,`uint32`,_.length),y=c(e,`${n}-component-labels`,`uint32`,i),b=c(e,`${n}-component-converged`,`uint32`,1),x=t=>c(e,`${n}-${t}`,`uint32`,i),S=x(`out-degree`),C=s?x(`in-degree`):void 0,w=x(`component-size`),E=u?x(`community-out-sum`):void 0,D=u&&s?x(`community-in-sum`):void 0,{output:O}=r,A=[];r.nodeMask&&A.push({name:`vertexMask`,view:r.nodeMask,type:`u32`,access:`read`});let j=e=>r.nodeMask?`vertexMask[vertexMaskOffset + ${e}] != 0u`:`true`,P=[],F=(t,r)=>{r&&P.push(f(e,{id:`${n}-${t}`,operation:p,view:r,type:`u32`,value:`0u`}))};F(`zero-output`,O),F(`zero-in-degree`,C),F(`zero-component-size`,w),F(`zero-out-sum`,E),F(`zero-in-sum`,D);let I=c(e,`${n}-self-loop-counts`,`uint32`,i),L=[{name:`neighbors`,view:_,type:`u32`,access:`read`},...A,{name:`maskedOut`,view:v,type:`u32`,access:`read_write`}];r.edgeMask&&L.push({name:`edgeMask`,view:r.edgeMask,type:`u32`,access:`read`}),C&&L.push({name:`inDegree`,view:C,type:`atomic<u32>`,access:`read_write`}),P.push(...h(e,{id:`${n}-slots`,operation:p,variant:`slots`,offsets:g,itemCapacity:_.length,accumulatorType:`u32`,output:S,bindings:L,declarations:`const NODE_COUNT: u32 = ${i}u;
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  let neighbor = neighbors[neighborsOffset + item];
  var slotLive = ${j(`row`)} && neighbor < NODE_COUNT;
  ${r.nodeMask?`if (slotLive) { slotLive = vertexMask[vertexMaskOffset + neighbor] != 0u; }`:``}
  ${r.edgeMask?`if (slotLive) { slotLive = edgeMask[edgeMaskOffset + item] != 0u; }`:``}
  maskedOut[maskedOutOffset + item] = select(0xffffffffu, neighbor, slotLive);
  if (!slotLive) {
    return 0u;
  }
  ${C?`atomicAdd(&inDegree[inDegreeOffset + neighbor], 1u);`:``}
  return ${l?`select(1u, 2u, neighbor == row)`:`1u`};
}`}),...h(e,{id:`${n}-self-loops`,operation:p,variant:`self-loops`,offsets:g,itemCapacity:_.length,accumulatorType:`u32`,output:I,bindings:[{name:`maskedIn`,view:v,type:`u32`,access:`read`}],declarations:`fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  return select(0u, 1u, maskedIn[maskedInOffset + item] == row);
}`}));let R=N({counters:[{target:`summary`,word:d.liveSlotCount},{target:`summary`,word:d.selfLoopSlotCount}],rowCount:`NODE_COUNT`,rowSource:`let selfCount = selfLoopCounts[selfLoopCountsOffset + index];
  addCounter(0u, outDegree[outDegreeOffset + index]${l?` - selfCount`:``});
  addCounter(1u, selfCount);`});P.push(m(e,{id:`${n}-slots-finalize`,operation:p,variant:`slots-finalize`,bindings:[{name:`selfLoopCounts`,view:I,type:`u32`,access:`read`},{name:`outDegree`,view:S,type:`u32`,access:`read`},{name:`summary`,view:O,type:`atomic<u32>`,access:`read_write`}],invocationCount:i,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${i}u;
${R.declarations}`,body:R.body})),P.push(...a(e,()=>{let t=new T({id:`${n}-topology`,vertexCount:i,forward:{offsets:g,neighbors:v}});new k({id:`${n}-components`,topology:t,output:y,iterations:r.maximumComponentIterations,converged:b}).addToGraph(e)})),P.push(m(e,{id:`${n}-component-sizes`,operation:p,variant:`component-sizes`,bindings:[...A,{name:`labels`,view:y,type:`u32`,access:`read`},{name:`componentSize`,view:w,type:`atomic<u32>`,access:`read_write`}],invocationCount:i,body:`if (!(${j(`index`)})) {
    return;
  }
  atomicAdd(&componentSize[componentSizeOffset + labels[labelsOffset + index]], 1u);`}));let z=r.parameters?`max(parameters[parametersOffset + 1u], 1u)`:`1u`,B=r.degreeBinning===`log2`?`fn getBin(degree: u32) -> u32 {
  if (degree == 0u) {
    return 0u;
  }
  return min(1u + firstLeadingBit(degree), BIN_COUNT - 1u);
}`:`fn getBin(degree: u32) -> u32 {
  return min(degree / ${z}, BIN_COUNT - 1u);
}`,V=[...A,{name:`outDegree`,view:S,type:`u32`,access:`read`}];C&&V.push({name:`inDegree`,view:C,type:`u32`,access:`read`}),r.communities&&V.push({name:`communities`,view:r.communities,type:`u32`,access:`read`}),E&&V.push({name:`outSum`,view:E,type:`atomic<u32>`,access:`read_write`}),D&&V.push({name:`inSum`,view:D,type:`atomic<u32>`,access:`read_write`}),r.parameters&&r.degreeBinning!==`log2`&&V.push({name:`parameters`,view:r.parameters,type:`u32`,access:`read`}),V.push({name:`summary`,view:O,type:`atomic<u32>`,access:`read_write`});let U=3*o<=ne,ee=`const NODE_COUNT: u32 = ${i}u;
const BIN_COUNT: u32 = ${o}u;
const HISTOGRAM_BASE: u32 = 16u;
${B}`,te=u?`let label = communities[communitiesOffset + index];
  if (label >= NODE_COUNT) {
    atomicAdd(&summary[summaryOffset + ${d.modularityValid}u], 1u);
  } else {
    atomicAdd(&outSum[outSumOffset + label], outValue);
    ${D?`atomicAdd(&inSum[inSumOffset + label], inValue);`:``}
  }`:``;if(P.push(m(e,{id:`${n}-degrees`,operation:p,variant:`degrees`,record:{datum:{keySpace:`node`},formula:`gpu-network.statistics.degree-histogram`},bindings:V,invocationCount:i,...U?{workgroupSize:W,guardIndex:!1,declarations:`${ee}
var<workgroup> localHistogram: array<atomic<u32>, ${3*o}>;
var<workgroup> localMaxima: array<atomic<u32>, 3>;
var<workgroup> localCounts: array<atomic<u32>, 2>;`,body:`for (var bin = localInvocationIndex; bin < ${3*o}u; bin += ${W}u) {
    atomicStore(&localHistogram[bin], 0u);
  }
  if (localInvocationIndex < 3u) {
    atomicStore(&localMaxima[localInvocationIndex], 0u);
  }
  if (localInvocationIndex < 2u) {
    atomicStore(&localCounts[localInvocationIndex], 0u);
  }
  workgroupBarrier();
  if (index < NODE_COUNT && ${j(`index`)}) {
    let outValue = outDegree[outDegreeOffset + index];
    let inValue = ${C?`inDegree[inDegreeOffset + index]`:`outValue`};
    let totalValue = ${s?`outValue + inValue`:`outValue`};
    atomicAdd(&localCounts[1], 1u);
    if (totalValue == 0u) {
      atomicAdd(&localCounts[0], 1u);
    }
    atomicMax(&localMaxima[0], outValue);
    atomicMax(&localMaxima[1], inValue);
    atomicMax(&localMaxima[2], totalValue);
    atomicAdd(&localHistogram[getBin(outValue)], 1u);
    atomicAdd(&localHistogram[BIN_COUNT + getBin(inValue)], 1u);
    atomicAdd(&localHistogram[2u * BIN_COUNT + getBin(totalValue)], 1u);
    ${te}
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < ${3*o}u; bin += ${W}u) {
    let binCount = atomicLoad(&localHistogram[bin]);
    if (binCount > 0u) {
      atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + bin], binCount);
    }
  }
  if (localInvocationIndex < 3u) {
    let workgroupMaximum = atomicLoad(&localMaxima[localInvocationIndex]);
    if (workgroupMaximum > 0u) {
      atomicMax(&summary[summaryOffset + ${d.maximumOutDegree}u + localInvocationIndex], workgroupMaximum);
    }
  }
  if (localInvocationIndex == 3u) {
    let workgroupIsolated = atomicLoad(&localCounts[0]);
    if (workgroupIsolated > 0u) {
      atomicAdd(&summary[summaryOffset + ${d.isolatedNodeCount}u], workgroupIsolated);
    }
    let workgroupLive = atomicLoad(&localCounts[1]);
    if (workgroupLive > 0u) {
      atomicAdd(&summary[summaryOffset + ${d.liveNodeCount}u], workgroupLive);
    }
  }`}:{declarations:ee,body:`if (!(${j(`index`)})) {
    return;
  }
  let outValue = outDegree[outDegreeOffset + index];
  let inValue = ${C?`inDegree[inDegreeOffset + index]`:`outValue`};
  let totalValue = ${s?`outValue + inValue`:`outValue`};
  atomicAdd(&summary[summaryOffset + ${d.liveNodeCount}u], 1u);
  if (totalValue == 0u) {
    atomicAdd(&summary[summaryOffset + ${d.isolatedNodeCount}u], 1u);
  }
  atomicMax(&summary[summaryOffset + ${d.maximumOutDegree}u], outValue);
  atomicMax(&summary[summaryOffset + ${d.maximumInDegree}u], inValue);
  atomicMax(&summary[summaryOffset + ${d.maximumTotalDegree}u], totalValue);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + getBin(outValue)], 1u);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + BIN_COUNT + getBin(inValue)], 1u);
  atomicAdd(&summary[summaryOffset + HISTOGRAM_BASE + 2u * BIN_COUNT + getBin(totalValue)], 1u);
  ${te}`}})),r.communities){let t=c(e,`${n}-intra-sums`,`uint32`,i);P.push(...h(e,{id:`${n}-intra-community`,operation:p,variant:`intra-community`,offsets:g,itemCapacity:_.length,accumulatorType:`u32`,output:t,bindings:[{name:`maskedIn`,view:v,type:`u32`,access:`read`},{name:`communities`,view:r.communities,type:`u32`,access:`read`}],declarations:`fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> u32 {
  let neighbor = maskedIn[maskedInOffset + item];
  return select(
    0u,
    1u,
    neighbor != 0xffffffffu &&
      communities[communitiesOffset + neighbor] == communities[communitiesOffset + row]
  );
}`}));let a=N({counters:[{target:`summary`,word:d.intraCommunitySlotCount}],rowCount:`NODE_COUNT`,rowSource:`addCounter(0u, intraSums[intraSumsOffset + index]);`});P.push(m(e,{id:`${n}-intra-community-total`,operation:p,variant:`intra-community-total`,bindings:[{name:`intraSums`,view:t,type:`u32`,access:`read`},{name:`summary`,view:O,type:`atomic<u32>`,access:`read_write`}],invocationCount:i,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${i}u;
${a.declarations}`,body:a.body}))}let re=Math.ceil(i/W),K=c(e,`${n}-finish-partials`,`uint32`,3*re),q=[{name:`componentSize`,view:w,type:`u32`,access:`read`}];E&&q.push({name:`outSum`,view:E,type:`u32`,access:`read`}),D&&q.push({name:`inSum`,view:D,type:`u32`,access:`read`}),q.push({name:`summaryIn`,view:O,type:`u32`,access:`read`},{name:`partials`,view:K,type:`u32`,access:`read_write`});let ie=D?`(f32(outSum[outSumOffset + index]) / totalSlots) * (f32(inSum[inSumOffset + index]) / totalSlots)`:`(f32(outSum[outSumOffset + index]) / totalSlots) * (f32(outSum[outSumOffset + index]) / totalSlots)`,ae=`workgroupBarrier();
  for (var stride = ${W/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedCount[localInvocationIndex] += sharedCount[localInvocationIndex + stride];
      sharedLargest[localInvocationIndex] =
        max(sharedLargest[localInvocationIndex], sharedLargest[localInvocationIndex + stride]);
      sharedSum[localInvocationIndex] += sharedSum[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`,J=`var<workgroup> sharedCount: array<u32, ${W}>;
var<workgroup> sharedLargest: array<u32, ${W}>;
var<workgroup> sharedSum: array<f32, ${W}>;`;P.push(m(e,{id:`${n}-finish-partials`,operation:p,variant:`finish-partials`,bindings:q,invocationCount:re*W,workgroupSize:W,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${i}u;
${J}`,body:`let liveSlots = summaryIn[summaryInOffset + ${d.liveSlotCount}u];
  let selfLoops = summaryIn[summaryInOffset + ${d.selfLoopSlotCount}u];
  let totalSlots = f32(liveSlots${l?` + selfLoops`:``});
  var componentCount = 0u;
  var largest = 0u;
  var termSum = 0.0;
  if (index < NODE_COUNT) {
    let size = componentSize[componentSizeOffset + index];
    if (size > 0u) {
      componentCount = 1u;
      largest = size;
    }
    ${u?`if (liveSlots > 0u) { termSum = ${ie}; }`:``}
  }
  sharedCount[localInvocationIndex] = componentCount;
  sharedLargest[localInvocationIndex] = largest;
  sharedSum[localInvocationIndex] = termSum;
  ${ae}
  if (localInvocationIndex == 0u) {
    let partial = 3u * (index / ${W}u);
    partials[partialsOffset + partial] = sharedCount[0];
    partials[partialsOffset + partial + 1u] = sharedLargest[0];
    partials[partialsOffset + partial + 2u] = bitcast<u32>(sharedSum[0]);
  }`}));let Y=[{name:`partials`,view:K,type:`u32`,access:`read`},{name:`converged`,view:b,type:`u32`,access:`read`}];return r.parameters&&Y.push({name:`parameters`,view:r.parameters,type:`u32`,access:`read`}),Y.push({name:`summary`,view:O,type:`u32`,access:`read_write`}),P.push(m(e,{id:`${n}-finish`,operation:p,variant:`finish`,bindings:Y,invocationCount:W,workgroupSize:W,guardIndex:!1,declarations:`const PARTIAL_COUNT: u32 = ${re}u;
${J}`,body:`let liveSlots = summary[summaryOffset + ${d.liveSlotCount}u];
  let selfLoops = summary[summaryOffset + ${d.selfLoopSlotCount}u];
  let totalSlots = f32(liveSlots${l?` + selfLoops`:``});
  var componentCount = 0u;
  var largest = 0u;
  var termSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${W}u) {
    componentCount += partials[partialsOffset + 3u * partial];
    largest = max(largest, partials[partialsOffset + 3u * partial + 1u]);
    termSum += bitcast<f32>(partials[partialsOffset + 3u * partial + 2u]);
  }
  sharedCount[localInvocationIndex] = componentCount;
  sharedLargest[localInvocationIndex] = largest;
  sharedSum[localInvocationIndex] = termSum;
  ${ae}
  if (localInvocationIndex == 0u) {
    let invalidLabels = summary[summaryOffset + ${d.modularityValid}u];
    let intra = summary[summaryOffset + ${d.intraCommunitySlotCount}u]${l?` + selfLoops`:``};
    summary[summaryOffset + ${d.componentCount}u] = sharedCount[0];
    summary[summaryOffset + ${d.largestComponentSize}u] = sharedLargest[0];
    summary[summaryOffset + ${d.componentsConverged}u] = select(0u, 1u, converged[convergedOffset] != 0u);
    summary[summaryOffset + ${d.liveEdgeCount}u] = ${s?`liveSlots`:`(liveSlots - selfLoops) / 2u + selfLoops`};
    ${G(s,u,!!r.parameters)}
    summary[summaryOffset + 14u] = 0u;
    summary[summaryOffset + 15u] = 0u;
  }`})),P}};function G(e,t,n){let r=H;return t?`let modularityValid = invalidLabels == 0u && liveSlots > 0u;
    var modularity = 0.0;
    if (modularityValid) {
      modularity = f32(intra) / totalSlots - ${n?`bitcast<f32>(parameters[parametersOffset])`:`1.0`} * sharedSum[0];
    }
    summary[summaryOffset + ${r.modularity}u] = bitcast<u32>(modularity);
    summary[summaryOffset + ${r.modularityValid}u] = select(0u, 1u, modularityValid);`:`summary[summaryOffset + ${r.modularity}u] = 0u;
    summary[summaryOffset + ${r.modularityValid}u] = 0u;`}var K={liveNodeCount:0,liveSlotCount:1,selfLoopSlotCount:2,liveEdgeCount:3};function q(e){return(e.nodeColumnCount+e.edgeColumnCount+(e.hasEdgeTimes?1:0))*4}function ie(e,t={},n=new Float32Array(q(e))){if(n.length<q(e))throw Error(`GPUNetworkSubgraphFilter parameter target is too short`);n.fill(0,0,q(e));let r=[];for(let n=0;n<e.nodeColumnCount;n++)r.push(t.nodeRanges?.[n]);for(let n=0;n<e.edgeColumnCount;n++)r.push(t.edgeRanges?.[n]);e.hasEdgeTimes&&r.push(t.edgeTimeWindow);for(let[e,t]of r.entries())t&&(n[e*4]=t[0],n[e*4+1]=t[1],n[e*4+2]=1);return n}function ae(e){let t=K;return{liveNodeCount:e[t.liveNodeCount],liveSlotCount:e[t.liveSlotCount],selfLoopSlotCount:e[t.selfLoopSlotCount],liveEdgeCount:e[t.liveEdgeCount]}}var J=8,Y=6,X=`GPUNetworkSubgraphFilter`,oe=class{id;props;nodeCount;slotCount;constructor(e){this.id=e.id??`network-subgraph-filter`,this.props=e;let{id:t}=this,n=j(t,e.topology);this.nodeCount=n.nodeCount,this.slotCount=n.edgeCount;let{nodeCount:a,slotCount:o}=this;if(o<1)throw Error(`${t} topology.neighbors must contain at least one row`);let s=e.nodeColumns??[],c=e.edgeColumns??[];if(s.length+c.length>64)throw Error(`${t} supports at most 64 columns`);let u=[[`nodeMask`,e.nodeMask,a,`node`],[`edgeMask`,e.edgeMask,o,`neighbor slot`],[`edgeTimes`,e.edgeTimes,o,`neighbor slot`],[`edgeTimeWords`,e.edgeTimeWords,o,`neighbor slot`],[`output.nodeMask`,e.output.nodeMask,a,`node`],[`output.edgeMask`,e.output.edgeMask,o,`neighbor slot`]];for(let[e,n]of s.entries())l(n,[`float32`],`${t} nodeColumns[${e}]`),u.push([`nodeColumns[${e}]`,n,a,`node`]);for(let[e,n]of c.entries())l(n,[`float32`],`${t} edgeColumns[${e}]`),u.push([`edgeColumns[${e}]`,n,o,`neighbor slot`]);for(let[e,n,r,i]of u)if(n&&(e===`edgeTimeWords`?l(n,[`uint32x2`],`${t} ${e}`):e===`edgeTimes`?l(n,[`float32`],`${t} ${e}`):!e.startsWith(`nodeColumns`)&&!e.startsWith(`edgeColumns`)&&l(n,[`uint32`],`${t} ${e}`),n.length!==r))throw Error(`${t} ${e} must contain one row per ${i}`);let d=q({nodeColumnCount:s.length,edgeColumnCount:c.length,hasEdgeTimes:!!e.edgeTimes});if(d>0){if(!e.parameters)throw Error(`${t} parameters are required with columns or edgeTimes`);if(l(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<d)throw Error(`${t} parameters must contain at least ${d} float32 rows`)}if(e.edgeTimeWords){if(!e.uint32Parameters)throw Error(`${t} uint32Parameters are required with edgeTimeWords`);if(l(e.uint32Parameters,[`uint32`],`${t} uint32Parameters`),e.uint32Parameters.length<J)throw Error(`${t} uint32Parameters must contain at least 8 uint32 rows`)}let{output:f}=e;if(f.counts&&(l(f.counts,[`uint32`],`${t} output.counts`),f.counts.length<4))throw Error(`${t} output.counts must contain at least 4 uint32 rows`);for(let[e,n]of[[`liveNodes`,f.liveNodes],[`liveEdgeSlots`,f.liveEdgeSlots]])n&&r(`${t} output.${e}`,n);let p=f.inducedTopology;if(p){if(l(p.offsets,[`uint32`],`${t} output.inducedTopology.offsets`),l(p.neighbors,[`uint32`],`${t} output.inducedTopology.neighbors`),l(p.status.overflow,[`uint32`],`${t} output.inducedTopology.status.overflow`),p.offsets.length!==a+1)throw Error(`${t} output.inducedTopology.offsets must contain nodeCount + 1 rows`);if(p.status.overflow.length<1)throw Error(`${t} output.inducedTopology.status.overflow must contain one uint32 row`);if(p.sourceSlots&&(l(p.sourceSlots,[`uint32`],`${t} output.inducedTopology.sourceSlots`),p.sourceSlots.length!==p.neighbors.length))throw Error(`${t} output.inducedTopology.sourceSlots must match neighbors in length`)}i(t,this.getOutputViews(),this.getInputViews())}getInputViews(){let{props:e}=this;return[...M(e.topology),e.nodeMask,e.edgeMask,...e.nodeColumns??[],...e.edgeColumns??[],e.edgeTimes,e.edgeTimeWords,e.parameters,e.uint32Parameters]}getOutputViews(){let{output:e}=this.props;return[e.nodeMask,e.edgeMask,e.counts,e.liveNodes?.ids,e.liveNodes?.status.count,e.liveNodes?.status.overflow,e.liveNodes?.status.requiredCount,e.liveEdgeSlots?.ids,e.liveEdgeSlots?.status.count,e.liveEdgeSlots?.status.overflow,e.liveEdgeSlots?.status.requiredCount,e.inducedTopology?.offsets,e.inducedTopology?.neighbors,e.inducedTopology?.sourceSlots,e.inducedTopology?.status.overflow]}getCommandNodes(e){let{id:n,props:r,nodeCount:i,slotCount:a}=this;t(n,e,[...this.getInputViews(),...this.getOutputViews()]);let{output:o}=r,s=r.nodeColumns??[],l=r.edgeColumns??[],d=[],h=K,_=(t,r,i,a,o,s,c)=>d.push(m(e,{id:`${n}-${t}`,operation:X,variant:r,record:c,bindings:i,invocationCount:a,body:o,declarations:s})),v=(e,t,n,i,a)=>{for(let o=0;o<n.length;o+=Y){let s=n.slice(o,o+Y),c=[{name:`maskOut`,view:t,type:`u32`,access:`read_write`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},...s.map((e,t)=>({name:`column${t}`,view:e,type:`f32`,access:`read`}))],l=s.map((e,t)=>`{
    let base = parametersOffset + ${(i+o+t)*4}u;
    if (parameters[base + 2u] != 0.0) {
      let value = column${t}[column${t}Offset + index];
      live = live && value >= parameters[base] && value < parameters[base + 1u];
    }
  }`).join(`
  `);_(`${e}-columns-${o/Y}`,e,c,a,`var live = maskOut[maskOutOffset + index] != 0u;
  ${l}
  maskOut[maskOutOffset + index] = select(0u, 1u, live);`,void 0,{datum:{keySpace:e===`vertex`?`node`:`edge`},formula:`gpu-network.subgraph-filter.range`})}};_(`vertex-init`,`vertex-init`,[{name:`maskOut`,view:o.nodeMask,type:`u32`,access:`read_write`},...r.nodeMask?[{name:`maskIn`,view:r.nodeMask,type:`u32`,access:`read`}]:[]],i,r.nodeMask?`maskOut[maskOutOffset + index] = select(0u, 1u, maskIn[maskInOffset + index] != 0u);`:`maskOut[maskOutOffset + index] = 1u;`),v(`vertex`,o.nodeMask,s,0,i);let y=!r.directed&&r.pairUndirectedSlots!==!1,b=y?c(e,`${n}-own-edge-mask`,`uint32`,a):o.edgeMask;if(_(`edge-init`,`edge-init`,[{name:`maskOut`,view:b,type:`u32`,access:`read_write`},...r.edgeMask?[{name:`maskIn`,view:r.edgeMask,type:`u32`,access:`read`}]:[]],a,r.edgeMask?`maskOut[maskOutOffset + index] = select(0u, 1u, maskIn[maskInOffset + index] != 0u);`:`maskOut[maskOutOffset + index] = 1u;`),v(`edge`,b,l,s.length,a),r.edgeTimes){let e=(s.length+l.length)*4;_(`edge-times`,`edge-times`,[{name:`maskOut`,view:b,type:`u32`,access:`read_write`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`times`,view:r.edgeTimes,type:`f32`,access:`read`}],a,`let base = parametersOffset + ${e}u;
  if (parameters[base + 2u] != 0.0) {
    let time = times[timesOffset + index];
    let live = time >= parameters[base] && time <= parameters[base + 1u];
    if (!live) {
      maskOut[maskOutOffset + index] = 0u;
    }
  }`)}r.edgeTimeWords&&_(`edge-time-words`,`edge-time-words`,[{name:`maskOut`,view:b,type:`u32`,access:`read_write`},{name:`timeWindow`,view:r.uint32Parameters,type:`u32`,access:`read`},{name:`timeWords`,view:r.edgeTimeWords,type:`u32`,access:`read`}],a,`let windowStart = vec2<u32>(timeWindow[timeWindowOffset], timeWindow[timeWindowOffset + 1u]);
  let windowEnd = vec2<u32>(timeWindow[timeWindowOffset + 2u], timeWindow[timeWindowOffset + 3u]);
  let startFraction = bitcast<f32>(timeWindow[timeWindowOffset + 4u]);
  let endFraction = bitcast<f32>(timeWindow[timeWindowOffset + 5u]);
  let time = vec2<u32>(
    timeWords[timeWordsOffset + 2u * index],
    timeWords[timeWordsOffset + 2u * index + 1u]
  );
  let live = isTimeWordsAtLeast(time, 0.0, windowStart, startFraction) &&
    isTimeWordsAtLeast(windowEnd, endFraction, time, 0.0);
  if (!live) {
    maskOut[maskOutOffset + index] = 0u;
  }`,g);let x=y?`var live = sourceLive && ownMask[ownMaskOffset + slot] != 0u && neighbor < NODE_COUNT;
    if (live && neighbor != row) {
      // The j-th slot row -> neighbor pairs with the j-th slot neighbor -> row. A missing
      // reverse slot leaves the slot's own result in force.
      var ordinal = 0u;
      for (var earlier = rowBegin; earlier < slot; earlier++) {
        ordinal += select(0u, 1u, neighbors[neighborsOffset + earlier] == neighbor);
      }
      let reverseEnd = min(offsets[offsetsOffset + neighbor + 1u], SLOT_COUNT);
      var seen = 0u;
      for (var reverse = offsets[offsetsOffset + neighbor]; reverse < reverseEnd; reverse++) {
        if (neighbors[neighborsOffset + reverse] == row) {
          if (seen == ordinal) {
            live = ownMask[ownMaskOffset + reverse] != 0u;
            break;
          }
          seen++;
        }
      }
    }`:`var live = sourceLive && edgeMask[edgeMaskOffset + slot] != 0u && neighbor < NODE_COUNT;`;if(_(`edge-endpoints`,`edge-endpoints`,[{name:`offsets`,view:r.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.topology.neighbors,type:`u32`,access:`read`},{name:`vertexMask`,view:o.nodeMask,type:`u32`,access:`read`},...y?[{name:`ownMask`,view:b,type:`u32`,access:`read`}]:[],{name:`edgeMask`,view:o.edgeMask,type:`u32`,access:`read_write`}],a,`let slot = index;
  // Owner row: the last row whose offset is at or before the slot.
  var low = 0u;
  var high = NODE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= slot) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let row = low;
  let rowBegin = offsets[offsetsOffset + row];
  let rowEnd = min(offsets[offsetsOffset + row + 1u], SLOT_COUNT);
  if (slot < rowBegin || slot >= rowEnd) {
    // A slot past offsets[NODE_COUNT] belongs to no row. The edge-init pass wrote 1 there (or the
    // caller's mask), so it must be cleared here or it reads as a live edge (review B-09).
    edgeMask[edgeMaskOffset + slot] = 0u;
    return;
  }
  let sourceLive = vertexMask[vertexMaskOffset + row] != 0u;
  let neighbor = neighbors[neighborsOffset + slot];
  ${x}
  if (live) {
    live = vertexMask[vertexMaskOffset + neighbor] != 0u;
  }
  edgeMask[edgeMaskOffset + slot] = select(0u, 1u, live);`,`const NODE_COUNT: u32 = ${i}u;
const SLOT_COUNT: u32 = ${a}u;`),r.dropIsolated){let t=c(e,`${n}-incident`,`uint32`,i);d.push(f(e,{id:`${n}-zero-incident`,operation:X,view:t,type:`u32`,value:`0u`})),_(`mark-incident`,`mark-incident`,[{name:`offsets`,view:r.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.topology.neighbors,type:`u32`,access:`read`},{name:`edgeMask`,view:o.edgeMask,type:`u32`,access:`read`},{name:`incident`,view:t,type:`atomic<u32>`,access:`read_write`}],i,`let rowBegin = offsets[offsetsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    if (edgeMask[edgeMaskOffset + slot] != 0u) {
      atomicStore(&incident[incidentOffset + index], 1u);
      atomicStore(&incident[incidentOffset + neighbors[neighborsOffset + slot]], 1u);
    }
  }`,`const SLOT_COUNT: u32 = ${a}u;`),_(`drop-isolated`,`drop-isolated`,[{name:`vertexMask`,view:o.nodeMask,type:`u32`,access:`read_write`},{name:`incident`,view:t,type:`u32`,access:`read`}],i,`if (incident[incidentOffset + index] == 0u) {
    vertexMask[vertexMaskOffset + index] = 0u;
  }`)}if(o.counts){d.push(f(e,{id:`${n}-zero-counts`,operation:X,view:o.counts,type:`u32`,value:`0u`,componentCount:4}));let t=N({counters:[{target:`counts`,word:h.liveNodeCount},{target:`counts`,word:h.liveSlotCount},{target:`counts`,word:h.selfLoopSlotCount}],rowCount:`NODE_COUNT`,rowSource:`if (vertexMask[vertexMaskOffset + index] != 0u) {
    addCounter(0u, 1u);
  }
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var liveCount = 0u;
  var selfCount = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    if (edgeMask[edgeMaskOffset + slot] != 0u) {
      liveCount++;
      if (neighbors[neighborsOffset + slot] == index) {
        selfCount++;
      }
    }
  }
  addCounter(1u, liveCount);
  addCounter(2u, selfCount);`});d.push(m(e,{id:`${n}-counts`,operation:X,variant:`counts`,bindings:[{name:`offsets`,view:r.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.topology.neighbors,type:`u32`,access:`read`},{name:`vertexMask`,view:o.nodeMask,type:`u32`,access:`read`},{name:`edgeMask`,view:o.edgeMask,type:`u32`,access:`read`},{name:`counts`,view:o.counts,type:`atomic<u32>`,access:`read_write`}],invocationCount:i,workgroupSize:256,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${i}u;
const SLOT_COUNT: u32 = ${a}u;
${t.declarations}`,body:t.body})),_(`counts-finish`,`counts-finish`,[{name:`counts`,view:o.counts,type:`u32`,access:`read_write`}],1,`let slots = counts[countsOffset + ${h.liveSlotCount}u];
  let selfLoops = counts[countsOffset + ${h.selfLoopSlotCount}u];
  counts[countsOffset + ${h.liveEdgeCount}u] =
    ${r.directed?`slots`:`(slots - selfLoops) / 2u + selfLoops`};`)}let S=(t,r,i,a=c(e,`${n}-${t}-offsets`,`uint32`,r.length))=>{let o=r.length,s=c(e,`${n}-${t}-total`,`uint32`,1);d.push(...new u({id:`${n}-${t}-scan`,input:r,output:a}).getCommandNodes(e)),_(`${t}-scatter`,`scatter`,[{name:`flags`,view:r,type:`u32`,access:`read`},{name:`prefix`,view:a,type:`u32`,access:`read`},{name:`idsOut`,view:i.ids,type:`u32`,access:`read_write`},{name:`totalOut`,view:s,type:`u32`,access:`read_write`}],o,`let live = flags[flagsOffset + index] != 0u;
  let position = prefix[prefixOffset + index];
  if (live && position < CAPACITY) {
    idsOut[idsOutOffset + position] = index;
  }
  if (index == ROW_COUNT - 1u) {
    totalOut[totalOutOffset] = position + select(0u, 1u, live);
  }`,`const CAPACITY: u32 = ${i.ids.length}u;
const ROW_COUNT: u32 = ${o}u;`),d.push(p(e,{id:`${n}-${t}-publish`,operation:X,requiredCount:s,output:i}))};o.liveNodes&&S(`live-vertices`,o.nodeMask,o.liveNodes);let C=o.liveEdgeSlots||o.inducedTopology?c(e,`${n}-live-slots-offsets`,`uint32`,a):void 0;o.liveEdgeSlots?S(`live-slots`,o.edgeMask,o.liveEdgeSlots,C):C&&d.push(...new u({id:`${n}-live-slots-scan`,input:o.edgeMask,output:C}).getCommandNodes(e));let w=o.inducedTopology;if(w&&C){let t=`fn getPrefix(slot: u32) -> u32 {
  if (slot >= SLOT_COUNT) {
    return edgeMask[edgeMaskOffset + SLOT_COUNT - 1u] + slotPrefix[slotPrefixOffset + SLOT_COUNT - 1u];
  }
  return slotPrefix[slotPrefixOffset + slot];
}`;_(`induced-offsets`,`induced-offsets`,[{name:`offsets`,view:r.topology.offsets,type:`u32`,access:`read`},{name:`edgeMask`,view:o.edgeMask,type:`u32`,access:`read`},{name:`slotPrefix`,view:C,type:`u32`,access:`read`},{name:`inducedOffsets`,view:w.offsets,type:`u32`,access:`read_write`},{name:`overflowOut`,view:w.status.overflow,type:`u32`,access:`read_write`}],i+1,`let base = getPrefix(offsets[offsetsOffset]);
  let value = getPrefix(offsets[offsetsOffset + index]) - base;
  inducedOffsets[inducedOffsetsOffset + index] = value;
  if (index == NODE_COUNT) {
    overflowOut[overflowOutOffset] = select(0u, 1u, value > CAPACITY);
  }`,`const NODE_COUNT: u32 = ${i}u;
const SLOT_COUNT: u32 = ${a}u;
const CAPACITY: u32 = ${w.neighbors.length}u;
${t}`);let s=[{name:`offsets`,view:r.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.topology.neighbors,type:`u32`,access:`read`},{name:`edgeMask`,view:o.edgeMask,type:`u32`,access:`read`},{name:`slotPrefix`,view:C,type:`u32`,access:`read`},{name:`inducedNeighbors`,view:w.neighbors,type:`u32`,access:`read_write`}];for(let[t,r]of[[`clear-induced-neighbors`,w.neighbors],[`clear-induced-slots`,w.sourceSlots]])r&&d.push(f(e,{id:`${n}-${t}`,operation:X,view:r,type:`u32`,value:`0xffffffffu`}));w.sourceSlots&&s.push({name:`inducedSlots`,view:w.sourceSlots,type:`u32`,access:`read_write`}),_(`induced-scatter`,`induced-scatter`,s,a,`let firstSlot = min(offsets[offsetsOffset], SLOT_COUNT);
  let endSlot = min(offsets[offsetsOffset + NODE_COUNT], SLOT_COUNT);
  if (index < firstSlot || index >= endSlot || edgeMask[edgeMaskOffset + index] == 0u) {
    return;
  }
  let position = slotPrefix[slotPrefixOffset + index] - getPrefix(firstSlot);
  if (position < CAPACITY) {
    inducedNeighbors[inducedNeighborsOffset + position] = neighbors[neighborsOffset + index];
    ${w.sourceSlots?`inducedSlots[inducedSlotsOffset + position] = index;`:``}
  }`,`const NODE_COUNT: u32 = ${i}u;
const SLOT_COUNT: u32 = ${a}u;
const CAPACITY: u32 = ${w.neighbors.length}u;
${t}`)}return d}},Z=12,se=[1,2,3,4],ce=[...z,[135,140,150,255]];function le(e,t){if(!t)return{directed:t,slotCount:e.slotCount,offsets:e.offsets,neighbors:e.neighbors,reverseOffsets:null,reverseNeighbors:null,slotSource:e.slotSource,slotLength:e.weights,slotClass:e.slotClass,slotSpeed:e.slotSpeed,segmentSlots:e.segmentSlots};let{nodeCount:n,edgeCount:r,edgeSource:i,edgeTarget:a}=e,o=new Uint32Array(n+1);for(let e=0;e<r;e++)o[i[e]+1]++;for(let e=0;e<n;e++)o[e+1]+=o[e];let s=new Uint32Array(n+1);for(let e=0;e<r;e++)s[a[e]+1]++;for(let e=0;e<n;e++)s[e+1]+=s[e];let c=s.slice(0,n),l=new Uint32Array(r);for(let e=0;e<r;e++)l[c[a[e]]++]=i[e];return{directed:t,slotCount:r,offsets:o,neighbors:a,reverseOffsets:s,reverseNeighbors:l,slotSource:i,slotLength:Float32Array.from(e.edgeLength),slotClass:Float32Array.from(e.edgeClass),slotSpeed:Float32Array.from(e.edgeSpeed),segmentSlots:e.segmentEdge}}async function Q(t){let n=t.datasets.get(`chicago-roads`),{device:r}=t,i=n.defaultOrigin,a=F(n,i),{nodeCount:c}=a,l=new s(r,`centrality`),u=new I(a.segments,a.bounds),d=n.getProjection(i),f=l.createBuffer(`segments`,a.segments),p=l.createParameterBuffer(`filter-parameters`,`float32`,q({nodeColumnCount:1,edgeColumnCount:3,hasEdgeTimes:!1})),m=l.createParameterBuffer(`statistics-parameters`,`uint32`,2),h=l.createParameterBuffer(`style-parameters`,`uint32`,3),g=le(a,!1),T=le(a,!0),E=!1,D=null,O=0,k=!0,A=!0,j=!0,M=null,N=0,V=!1,H=null,W={};t.setReadout(`network`,`${o(c)} intersections, ${o(a.edgeCount)} directed edges`),t.setReadout(`length`,P(a.networkLength));function ne(){t.setFurniture({title:{title:`Structural centrality of Chicago streets`},scaleBar:{units:`metric`},credit:`OpenStreetMap contributors (ODbL)`,caveat:`Network clipped at the data boundary; centrality is not observed traffic.`})}function G(e,n){let r=Array.from(e,e=>Number(e)).filter(Number.isFinite).sort((e,t)=>e-t),i=[];for(let e=1;e<7;e++){let t=r[Math.min(r.length-1,Math.floor(e*r.length/7))]??0;(i.length===0||t>i[i.length-1])&&i.push(t)}return w({breaks:i.length?i:[0],scheme:`YlOrRd`,ground:t.ground(),unit:n===`pageRank`?`PageRank (raw score)`:n===`inDegree`?`incoming streets`:`streets meeting`,method:`Quantile breaks frozen from the first completed readback for this topology.`})}function K(){W=Object.fromEntries(Object.entries(W).map(([e,n])=>[e,w({breaks:n.breaks,scheme:`YlOrRd`,ground:t.ground(),unit:n.unit,method:n.method})]))}function J(e,t){switch(t){case`degree`:return e.degree;case`inDegree`:return e.inDegree;case`pageRank`:return e.pageRank;case`core`:return e.core;default:return e.pageRank}}function Y(){let e=D?.nodeData,n=t.options.metric;if(!e||n===`community`||n===`component`)return;let r=n===`core`?w({breaks:se,colors:[[110,110,110],[255,237,160],[254,178,76],[240,59,32],[189,0,38]]}):W[n]??G(J(e,n),n),i=J(e,n),a=Array(r.breaks.length+1).fill(0);for(let e of i){let t=0;for(;t<r.breaks.length&&e>=r.breaks[t];)t++;a[t]++}t.setLegendData(`centralityClasses`,{tables:W,active:n}),t.setChart(`histogram`,{kind:`bars`,values:a,labels:a.map((e,t)=>`${t===0?`≤`:`≥ ${r.breaks[t-1].toPrecision(3)}`} ${t===r.breaks.length?`+`:`< ${r.breaks[t].toPrecision(3)}`}`),xLabel:`${n} class`,yLabel:`intersections`,breaks:r.breaks,description:`Readback-derived ${n} values grouped by the displayed frozen class breaks.`}),t.setReadout(`histogramBins`,`${n} displayed-class counts`)}function X(e){let n=[5,10,40,100],r=new Float64Array(c).fill(1/c),i=new Float64Array(c),a=[],o=Number(t.options.damping),s=0;for(let t=1;t<=100;t++){i.fill((1-o)/c);for(let t=0;t<c;t++){let n=e.offsets[t],a=e.offsets[t+1],s=a>n?o*r[t]/(a-n):o*r[t]/c;if(a>n)for(let t=n;t<a;t++)i[e.neighbors[t]]+=s;else for(let e=0;e<c;e++)i[e]+=s}s=0;for(let e=0;e<c;e++)s+=Math.abs(i[e]-r[e]),r[e]=i[e];n.includes(t)&&a.push(s)}t.setChart(`pageRankResiduals`,{kind:`line`,xLabel:`iteration limit`,yLabel:`final-step residual (L1)`,series:[{label:`CPU reference residual`,x:n,y:a,color:0,width:2}],description:`Scene-local CPU PageRank reference over the current CSR and damping; a residual is not a convergence claim.`})}function Q(e){e.analyticsReader.stop(),e.statisticsReader.stop(),e.resources.destroy()}function ue(){D&&=(Q(D),null)}async function de(){let n=++O,i=t.options.oneWay?T:g,a=new s(r,`centrality-${n}`),o=i.slotCount,l=e=>a.createBuffer(e,c*4),u={degree:l(`degree`),degreeNormalized:l(`degree-normalized`),inDegree:l(`in-degree`),inDegreeNormalized:l(`in-degree-normalized`),pageRank:l(`page-rank`),pageRankNormalized:l(`page-rank-normalized`),core:l(`core`),coreNormalized:l(`core-normalized`),components:l(`components`),communities:l(`communities`),communityDisplay:l(`community-display`),edgeMask:a.createBuffer(`edge-mask`,o*4),slotScalar:a.createBuffer(`slot-scalar`,o*4),slotLabel:a.createBuffer(`slot-label`,o*4),slotMask:a.createBuffer(`slot-mask`,o*4),nodeScalar:l(`node-scalar`),nodeLabel:l(`node-label`),statistics:a.createBuffer(`statistics`,U(Z)*4),counts:a.createBuffer(`counts`,16),segmentSlots:a.createBuffer(`segment-slots`,i.segmentSlots)},d={offsets:a.createBuffer(`offsets`,i.offsets),neighbors:a.createBuffer(`neighbors`,i.neighbors),reverseOffsets:i.reverseOffsets?a.createBuffer(`reverse-offsets`,i.reverseOffsets):null,reverseNeighbors:i.reverseNeighbors?a.createBuffer(`reverse-neighbors`,i.reverseNeighbors):null,slotSource:a.createBuffer(`slot-source`,i.slotSource),slotLength:a.createBuffer(`slot-length`,i.slotLength),slotClass:a.createBuffer(`slot-class`,i.slotClass),slotSpeed:a.createBuffer(`slot-speed`,i.slotSpeed),nodeMask:a.createBuffer(`node-mask`,c*4)},f={residual:a.createBuffer(`residual`,4),degeneracy:a.createBuffer(`degeneracy`,4),coreConverged:a.createBuffer(`core-converged`,4),componentConverged:a.createBuffer(`component-converged`,4),communityConverged:a.createBuffer(`community-converged`,4)},p=new e(r,{id:`centrality-style`}),m=(e,t,n,r)=>y(p,e,t,n,r),_=m(`slot-source`,d.slotSource,`uint32`,o),v=h.importToGraph(p);S(p,{id:`slot-scalar`,bindings:[{name:`nodeScalar`,view:m(`node-scalar`,u.nodeScalar,`uint32`,c),type:`u32`,access:`read`},{name:`slotSource`,view:_,type:`u32`,access:`read`},{name:`neighbors`,view:m(`neighbors`,d.neighbors,`uint32`,o),type:`u32`,access:`read`},{name:`style`,view:v,type:`u32`,access:`read`},{name:`slotScalar`,view:m(`slot-scalar`,u.slotScalar,`float32`,o),type:`f32`,access:`read_write`}],invocationCount:o,body:`let sourceBits = nodeScalar[nodeScalarOffset + slotSource[slotSourceOffset + index]];
  let targetBits = nodeScalar[nodeScalarOffset + neighbors[neighborsOffset + index]];
  // style[2] distinguishes integer metrics from the raw IEEE-754 bits copied for PageRank.
  let integerMetric = style[styleOffset + 2u] != 0u;
  let a = select(bitcast<f32>(sourceBits), f32(sourceBits), integerMetric);
  let b = select(bitcast<f32>(targetBits), f32(targetBits), integerMetric);
  slotScalar[slotScalarOffset + index] = 0.5 * (a + b);`}),S(p,{id:`slot-label`,bindings:[{name:`nodeLabel`,view:m(`node-label`,u.nodeLabel,`uint32`,c),type:`u32`,access:`read`},{name:`slotSource`,view:_,type:`u32`,access:`read`},{name:`style`,view:v,type:`u32`,access:`read`},{name:`slotLabel`,view:m(`slot-label`,u.slotLabel,`uint32`,o),type:`u32`,access:`read_write`}],invocationCount:o,body:`let label = nodeLabel[nodeLabelOffset + slotSource[slotSourceOffset + index]];
  // style[0] = largest component label, style[1] = 1 to highlight islands instead of cycling labels.
  if (style[styleOffset + 1u] == 1u) {
    slotLabel[slotLabelOffset + index] = select(1u, 0u, label == style[styleOffset]);
  } else {
    slotLabel[slotLabelOffset + index] = label;
  }`}),S(p,{id:`slot-mask`,bindings:[{name:`edgeMask`,view:m(`edge-mask`,u.edgeMask,`uint32`,o),type:`u32`,access:`read`},{name:`slotMask`,view:m(`slot-mask`,u.slotMask,`float32`,o),type:`f32`,access:`read_write`}],invocationCount:o,body:`slotMask[slotMaskOffset + index] = select(0.0, 1.0, edgeMask[edgeMaskOffset + index] != 0u);`});let b;try{b=a.track(await p.compileAsync())}catch(e){throw a.destroy(),e}let C=new x(a,`analytics`,[{buffer:u.degree,size:c*4},{buffer:u.pageRank,size:c*4},{buffer:u.core,size:c*4},{buffer:u.communities,size:c*4},{buffer:u.components,size:c*4},{buffer:u.inDegree,size:c*4},{buffer:f.residual,size:4},{buffer:f.degeneracy,size:4},{buffer:f.coreConverged,size:4},{buffer:f.componentConverged,size:4},{buffer:f.communityConverged,size:4}],e=>{E||D?.analyticsReader!==C||Ce(e)}),w=new x(a,`statistics`,[{buffer:u.statistics,size:U(Z)*4},{buffer:u.counts,size:16}],e=>{E||D?.statisticsReader!==w||Ee(e)}),k={topology:i,resources:a,analytics:null,filterStatistics:null,style:b,buffers:u,analyticsReader:C,statisticsReader:w,inputs:{csr:d,scalars:f},nodeData:null};try{k.analytics=await pe(k),k.filterStatistics=await me(k)}catch(e){throw Q(k),e}return k}function fe(e){let n=D;D=e,W={},n&&Q(n),k=!0,A=!0,j=!0,M=null,t.setReadout(`slots`,`${o(e.topology.slotCount)} ${e.topology.directed?`directed edges`:`street slots (both directions)`}`)}async function pe(n){let{csr:i,scalars:a}=n.inputs,{buffers:o,topology:s}=n,l=s.slotCount,u=new e(r,{id:`centrality-analytics`}),d=(e,t,n,r)=>y(u,e,t,n,r);return u.add(new B({id:`centrality`,topology:{offsets:d(`offsets`,i.offsets,`uint32`,c+1),neighbors:d(`neighbors`,i.neighbors,`uint32`,l)},...i.reverseOffsets&&i.reverseNeighbors?{reverseTopology:{offsets:d(`reverse-offsets`,i.reverseOffsets,`uint32`,c+1),neighbors:d(`reverse-neighbors`,i.reverseNeighbors,`uint32`,l)},inDegree:{output:d(`in-degree`,o.inDegree,`uint32`,c),normalized:d(`in-degree-normalized`,o.inDegreeNormalized,`float32`,c)}}:{},degree:{output:d(`degree`,o.degree,`uint32`,c),normalized:d(`degree-normalized`,o.degreeNormalized,`float32`,c)},pageRank:{output:d(`page-rank`,o.pageRank,`float32`,c),normalized:d(`page-rank-normalized`,o.pageRankNormalized,`float32`,c),damping:Number(t.options.damping),iterationCount:Number(t.options.pageRankIterations),residual:d(`residual`,a.residual,`float32`,1)},coreNumber:{output:d(`core`,o.core,`uint32`,c),normalized:d(`core-normalized`,o.coreNormalized,`float32`,c),maximumIterations:64,degeneracy:d(`degeneracy`,a.degeneracy,`uint32`,1),status:{converged:d(`core-converged`,a.coreConverged,`uint32`,1)}},components:{output:d(`components`,o.components,`uint32`,c),maximumIterations:Number(t.options.componentIterations),status:{converged:d(`component-converged`,a.componentConverged,`uint32`,1)}},communities:{output:d(`communities`,o.communities,`uint32`,c),maximumIterations:Number(t.options.communityIterations),status:{converged:d(`community-converged`,a.communityConverged,`uint32`,1)}}})),n.resources.track(await u.compileAsync())}async function me(n){let{csr:i}=n.inputs,{buffers:a,topology:o}=n,s=o.slotCount,l=new e(r,{id:`centrality-filter-statistics`}),u=(e,t,n,r)=>y(l,e,t,n,r),d={offsets:u(`offsets`,i.offsets,`uint32`,c+1),neighbors:u(`neighbors`,i.neighbors,`uint32`,s)},f=u(`node-mask`,i.nodeMask,`uint32`,c),h=u(`edge-mask`,a.edgeMask,`uint32`,s),g=new oe({id:`filter`,topology:d,directed:o.directed,pairUndirectedSlots:t.options.pairSlots,dropIsolated:t.options.dropIsolated,nodeColumns:[u(`page-rank-normalized`,a.pageRankNormalized,`float32`,c)],edgeColumns:[u(`slot-speed`,i.slotSpeed,`float32`,s),u(`slot-length`,i.slotLength,`float32`,s),u(`slot-class`,i.slotClass,`float32`,s)],parameters:p.importToGraph(l),output:{nodeMask:f,edgeMask:h,counts:u(`counts`,a.counts,`uint32`,4)}});l.add(g);let _=new re({id:`statistics`,topology:d,directed:o.directed,nodeMask:f,edgeMask:h,communities:u(`communities`,a.communities,`uint32`,c),parameters:m.importToGraph(l),degreeBinCount:Z,degreeBinning:t.options.degreeBinning,maximumComponentIterations:Number(t.options.componentIterations),output:u(`statistics`,a.statistics,`uint32`,U(Z))});return l.add(_),n.resources.track(await l.compileAsync())}let he=l.track(new b),$={pipeline:!1,analytics:!1,filter:!1};function ge(e){$={pipeline:$.pipeline||!!e.pipeline,analytics:$.analytics||!!e.analytics,filter:$.filter||!!e.filter}}function _e(e){ge(e),he.request({build:ve,commit:ye,discard:be})}async function ve(){let e=$;$={pipeline:!1,analytics:!1,filter:!1};try{if(e.pipeline)return{request:e,next:await de()};let t=D;if(!t)return{request:e};let n={request:e};return e.analytics&&(n.analytics=await pe(t)),e.filter&&(n.filterStatistics=await me(t)),n}catch(t){throw ge(e),t}}function ye(e){if(e.next){fe(e.next),xe(),Se(),t.requestLayers();return}let n=D;if(n){if(e.analytics){let t=n.analytics;n.analytics=e.analytics,t&&n.resources.release(t),k=!0}if(e.filterStatistics){let t=n.filterStatistics;n.filterStatistics=e.filterStatistics,t&&n.resources.release(t),A=!0}}}function be(e){if(ge(e.request),e.next){Q(e.next);return}e.analytics&&D?.resources.release(e.analytics),e.filterStatistics&&D?.resources.release(e.filterStatistics)}function xe(){let{speedRange:e,lengthRange:n,classRange:r}=t.options;p.write(ie({nodeColumnCount:1,edgeColumnCount:3,hasEdgeTimes:!1},{nodeRanges:[M===null?null:[M,1/0]],edgeRanges:[e[0]<=0&&e[1]>=100?null:[e[0],e[1]>=100?1/0:e[1]],n[0]<=0&&n[1]>=1e3?null:[n[0],n[1]>=1e3?1/0:n[1]],r[0]<=0&&r[1]>=L.length-1?null:[r[0],r[1]+1]]})),m.write(ee({resolution:t.options.resolution,degreeBinWidth:t.options.degreeBinWidth})),A=!0}function Se(){let e=t.options.metric===`degree`||t.options.metric===`inDegree`||t.options.metric===`core`;h.write(Uint32Array.of(N,t.options.metric===`component`?1:0,e?1:0)),j=!0}function Ce(e){let n=D;if(!n)return;let r=0,i=(t,n)=>{let i=new t(e.slice(r,r+n*4),0,n);return r+=n*4,i},a=i(Uint32Array,c),s=i(Float32Array,c),l=i(Uint32Array,c),u=i(Uint32Array,c),d=i(Uint32Array,c),f=i(Uint32Array,c),p=i(Float32Array,1)[0],m=i(Uint32Array,1)[0],h=i(Uint32Array,1)[0],g=i(Uint32Array,1)[0],_=i(Uint32Array,1)[0];n.nodeData={degree:a,pageRank:s,core:l,community:u,component:d,inDegree:f},Object.keys(W).length===0&&(W={degree:G(a,`degree`),inDegree:G(f,`inDegree`),pageRank:G(s,`pageRank`)}),X(n.topology);let v=e=>{let t=new Map;for(let n of e)t.set(n,(t.get(n)??0)+1);return t},y=v(u),b=v(d),x=0,S=0;for(let[e,t]of b)t>S&&(x=e,S=t);N=x;let C=new Map;if(H)for(let e=0;e<c;e++){let t=u[e],n=C.get(t)??new Map;n.set(H[e],(n.get(H[e])??0)+1),C.set(t,n)}let w=[...y.entries()].sort((e,t)=>t[1]-e[1]),T=new Map,E=new Set;for(let[e]of w){let t=[...C.get(e)??[]].filter(([e])=>e<z.length&&!E.has(e)).sort((e,t)=>t[1]-e[1])[0];t&&(T.set(e,t[0]),E.add(t[0]))}for(let[e]of w)if(!T.has(e)){let t=Array.from({length:z.length},(e,t)=>t).find(e=>!E.has(e));T.set(e,t??z.length),t!==void 0&&E.add(t)}let O=Uint32Array.from(u,e=>T.get(e)??z.length);H=O,n.buffers.communityDisplay.write(O);let k=[...y.values()].sort((e,t)=>t-e).slice(0,8);t.setChart(`communitySizes`,{kind:`bars`,values:k,labels:k.map((e,t)=>`group ${t+1}`),yLabel:`intersections`,description:`Largest communities; identities retain scene-local display slots across iteration presets.`}),Se();let A=0,j=1/0,M=0;for(let e of s)A=Math.max(A,e),j=Math.min(j,e),M+=e;let P=Float32Array.from(s).sort(),F=A-j;we=e=>F>0?(P[Math.min(c-1,Math.floor(e*c))]-j)/F:0,Te(),t.setReadout(`communities`,`${o(y.size)} found, largest ${o(Math.max(...y.values()))} intersections${_?``:` (not converged)`}`),t.setReadout(`components`,`${o(b.size)}, largest ${(100*S/c).toFixed(1)}% of intersections${g?``:` (not converged)`}`),t.setReadout(`pageRankResidual`,p);let I=a.reduce((e,t)=>e+t,0)/c,L=M/c,R=0,B=0,V=0;for(let e=0;e<c;e++){let t=a[e]-I,n=s[e]-L;R+=t*n,B+=t*t,V+=n*n}let U=B&&V?R/Math.sqrt(B*V):0;t.setChart(`degreeCorrelation`,{kind:`scatter`,x:a,y:s,xLabel:`degree`,yLabel:`PageRank`,radius:1.2,fit:{slope:R/Math.max(B,1e-12),intercept:L-R/Math.max(B,1e-12)*I,label:`r = ${U.toFixed(3)}`},description:`Live degree and PageRank node columns. The fit describes this clipped graph, not traffic.`}),t.setReadout(`pageRankPeak`,`${(A/(M/c)).toFixed(1)}x the average`),t.setReadout(`degeneracy`,`${m}${h?``:` (not converged)`}`),t.setReadout(`analyticsTime`,null),Y()}let we=()=>0;function Te(){let{topPercent:e}=t.options;M=e>=100?null:we(1-e/100),xe()}function Ee(e){let n=U(Z),r=new Uint32Array(e,0,n+4),i=te(r.subarray(0,n),{degreeBinCount:Z}),a=ae(r.subarray(n));t.setReadout(`liveVertices`,`${o(i.liveNodeCount)} of ${o(c)}`),t.setReadout(`liveEdges`,o(i.liveEdgeCount)),t.setReadout(`filterCounts`,`${o(a.liveNodeCount)} nodes, ${o(a.liveSlotCount)} slots`),t.setReadout(`statComponents`,`${o(i.componentCount)}${i.componentsConverged?``:` (not converged)`}`),t.setReadout(`largestComponent`,i.liveNodeCount>0?`${o(i.largestComponentSize)} (${(100*i.largestComponentSize/i.liveNodeCount).toFixed(1)}%)`:null),t.setReadout(`isolated`,i.isolatedNodeCount),t.setReadout(`maxDegree`,i.maximumTotalDegree),t.setReadout(`modularity`,i.modularityValid?i.modularity:`n/a`)}async function De(){let e=D;if(!(V||!e?.filterStatistics||!e.analytics)){V=!0;try{let n={parameters:void 0,completionBuffer:e.buffers.counts,runs:5,repetitions:2},i=await v(r,e.analytics,n),a=await v(r,e.filterStatistics,n);E||(t.setReadout(`analyticsTime`,`${i.milliseconds.toFixed(1)} ms`),t.setReadout(`filterTime`,`${a.milliseconds.toFixed(1)} ms`))}catch{}finally{V=!1}}}fe(await de()),xe(),Se(),ne();let Oe=[i[0],i[1],0],ke=e=>{let n=D,{metric:r}=t.options,i=r===`degree`?n.buffers.degree:r===`inDegree`?n.buffers.inDegree:r===`core`?n.buffers.core:n.buffers.pageRank,a=r===`component`?n.buffers.components:n.buffers.communityDisplay;e.copyBufferToBuffer({sourceBuffer:i,destinationBuffer:n.buffers.nodeScalar,size:c*4}),e.copyBufferToBuffer({sourceBuffer:a,destinationBuffer:n.buffers.nodeLabel,size:c*4})};return{getCompiledGraphs:()=>{let e=D;return e?.analytics&&e.filterStatistics?[e.analytics,e.filterStatistics,e.style]:[]},setOption(e){switch(e){case`oneWay`:_e({pipeline:!0});break;case`damping`:case`pageRankIterations`:case`communityIterations`:_e({analytics:!0});break;case`componentIterations`:_e({analytics:!0,filter:!0});break;case`degreeBinning`:case`dropIsolated`:case`pairSlots`:_e({filter:!0});break;case`speedRange`:case`lengthRange`:case`classRange`:case`resolution`:case`degreeBinWidth`:xe();break;case`topPercent`:Te();break;case`metric`:Se(),Y(),t.requestLayers();break;default:t.requestLayers()}},onAction(e){e===`measure`&&De()},onThemeChange(){K(),Y(),t.requestLayers()},getTooltip(e){let n=D;if(!e.coordinate||!n)return null;let[r,i]=d.project(e.coordinate[0],e.coordinate[1]),o=u.nearest(r,i,35);if(o<0)return null;let s=n.topology.segmentSlots[o],c=a.segmentEdge[o],l=n.topology.slotSource[s],f=n.topology.neighbors[s],p=n.nodeData,m=[`${L[a.edgeClass[c]]}, ${Math.round(a.edgeLength[c])} m, ${a.edgeSpeed[c]} km/h`];if(p){let e=t.options.metric;if(e!==`community`&&e!==`component`){let t=J(p,e),n=e===`core`?{breaks:se}:W[e]??G(t,e),r=.5*(t[l]+t[f]),i=0;for(;i<n.breaks.length&&r>=n.breaks[i];)i++;let a=i===0?`-∞`:n.breaks[i-1].toPrecision(4),o=i===n.breaks.length?`+∞`:n.breaks[i].toPrecision(4);m.push(`${e} class ${i+1}: [${a}, ${o})`),m.push(`${e} endpoints: ${Number(t[l]).toPrecision(4)} / ${Number(t[f]).toPrecision(4)}`)}}return m.join(`
`)},encode(e){let t=D;!t?.analytics||!t.filterStatistics||(k&&(t.analytics.encode(e,{parameters:void 0}),t.analyticsReader.request(e),k=!1,A=!0),A&&(t.filterStatistics.encode(e,{parameters:void 0}),t.statisticsReader.request(e),A=!1,j=!0),j&&=(ke(e),t.style.encode(e,{parameters:void 0}),!1),t.analyticsReader.flush(e),t.statisticsReader.flush(e))},getLayers(){let e=D;if(!e)return[];let{metric:n,showRemoved:r}=t.options,i=t.ground()===`dark`,o=[];r&&o.push(new _({id:`centrality-removed`,coordinateOrigin:Oe,segments:f,instanceCount:a.segmentCount,color:i?[150,158,185,55]:[90,100,125,60],widthPixels:.55}));let s=n===`community`||n===`component`,c=s?{valueFormat:`uint32`,colormap:`category`,palette:n===`component`?R:ce}:{valueFormat:`float32`,...C(w({breaks:n===`core`?se:W[n]?.breaks??[0],scheme:`YlOrRd`,ground:t.ground(),colors:n===`core`?[[110,110,110],[255,237,160],[254,178,76],[240,59,32],[189,0,38]]:void 0}))},l=s?[2.6]:n===`core`?[.7,1,1.4,1.9,2.6]:[.7,1,1.4,1.9,2.6,3.4,3.4];for(let t=0;t<l.length;t++)o.push(new _({id:`centrality-streets-${t}`,coordinateOrigin:Oe,segments:f,instanceCount:a.segmentCount,weights:e.buffers.slotMask,values:s?e.buffers.slotLabel:e.buffers.slotScalar,valueIndices:e.buffers.segmentSlots,...c,widthPixels:l[t],...s?{}:{highlightClasses:[t],dimOpacity:0}}));return o},destroy(){E=!0,ue(),l.destroy()}}}export{Q as createStreetCentrality};