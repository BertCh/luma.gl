import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{E as e,p as t}from"./analysis-frame-Blu0tYdw.js";import{N as n,_ as r,d as i,f as a,h as o,l as s,t as c,v as l,w as u,y as d}from"./resources-CwvhEOBe.js";import{t as f}from"./gpu-scan-BuGhTLDQ.js";import{t as p}from"./gpu-sort-Bl3Dztdb.js";import{t as m}from"./gpu-visibility-workflow-CuWho9Ae.js";import{n as h,r as g,s as _}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as v}from"./scan-total-BdolCtK7.js";import{t as y}from"./geodesic-wgsl-DS0CJZ42.js";import{a as b,c as x}from"./index-Bjk18V2I.js";import{t as ee}from"./graph-buffers-B4_80RlX.js";import{t as te}from"./graph-swap-41Ru7LVu.js";import{t as S}from"./summary-reader-CsgWxENf.js";import{t as C}from"./mode-kernels-Dxm2GWvI.js";import{m as w}from"./class-table-BVEo3-Vs.js";import{i as T,n as E}from"./network-topology-BXzNuj-N.js";import{i as ne,n as D,t as re}from"./gpu-network-reachability-ROiXwwot.js";import{a as ie,c as ae,i as oe,l as se,n as O,r as k,s as ce,t as A}from"./b9-shared-CJ1sDweI.js";var j=`GPUNetworkPathExtraction`,le=`
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`;function M(e,t){let n=[{name:`status`,view:t.status,type:`u32`,access:`read_write`}];return t.converged&&n.push({name:`converged`,view:t.converged,type:`u32`,access:`read`}),_(e,{id:t.id,operation:j,variant:`status`,bindings:n,invocationCount:2,body:`let isConverged = ${t.converged?`converged[convergedOffset] != 0u`:`true`};
  status[statusOffset + index] = select(0u, select(0u, 1u, isConverged), index == 1u);`})}function ue(e,t){let n=[{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`targets`,view:t.targets,type:`u32`,access:`read`},{name:`nodeLengths`,view:t.nodeLengths,type:`u32`,access:`read_write`},{name:`status`,view:t.status,type:`atomic<u32>`,access:`read_write`}];return typeof t.targetCount==`object`&&n.push({name:`targetCount`,view:t.targetCount,type:`u32`,access:`read`}),t.pathFound&&n.push({name:`pathFound`,view:t.pathFound,type:`u32`,access:`read_write`}),t.pathCosts&&n.push({name:`pathCosts`,view:t.pathCosts,type:`u32`,access:`read_write`}),_(e,{id:t.id,operation:j,variant:`measure`,bindings:n,invocationCount:t.targets.length,declarations:`${le}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const MAXIMUM_PATH_LENGTH: u32 = ${t.maxPathLength}u;`,body:`var length = 0u;
  var found = false;
  var costBits = INFINITY_BITS;
  let isActive = ${typeof t.targetCount==`number`?`index < ${t.targetCount}u`:t.targetCount?`index < targetCount[targetCountOffset]`:`true`};
  let targetNode = targets[targetsOffset + index];
  if (isActive && targetNode < NODE_COUNT) {
    let targetCost = costs[costsOffset + targetNode];
    if ((bitcast<u32>(targetCost) & 0x7fffffffu) < INFINITY_BITS) {
      var node = targetNode;
      var walked = 1u;
      // 0 walking, 1 root reached, 2 broken chain, 3 too long.
      var state = 0u;
      for (var step = 0u; step <= MAXIMUM_PATH_LENGTH; step++) {
        let predecessor = predecessors[predecessorsOffset + node];
        if (predecessor == NONE) {
          state = 1u;
          break;
        }
        if (predecessor >= NODE_COUNT) {
          state = 2u;
          break;
        }
        if (walked >= MAXIMUM_PATH_LENGTH) {
          state = 3u;
          break;
        }
        node = predecessor;
        walked++;
      }
      // A NONE predecessor is a source only when the search converged (status word 1).
      if (state == 1u && atomicLoad(&status[statusOffset + 1u]) != 0u) {
        found = true;
        length = walked;
        costBits = bitcast<u32>(targetCost);
      }
      if (state == 3u) {
        atomicOr(&status[statusOffset], 1u);
      }
    }
  }
  nodeLengths[nodeLengthsOffset + index] = length;
  ${t.pathFound?`pathFound[pathFoundOffset + index] = select(0u, 1u, found);`:``}
  ${t.pathCosts?`pathCosts[pathCostsOffset + index] = costBits;`:``}`})}function de(e,t){return _(e,{id:t.id,operation:j,variant:`edge-lengths`,bindings:[{name:`nodeLengths`,view:t.nodeLengths,type:`u32`,access:`read`},{name:`edgeLengths`,view:t.edgeLengths,type:`u32`,access:`read_write`}],invocationCount:t.nodeLengths.length,body:`let length = nodeLengths[nodeLengthsOffset + index];
  edgeLengths[edgeLengthsOffset + index] = select(0u, length - 1u, length > 0u);`})}function N(e,t){let n=[],r=[],i=t.targetCapacity;for(let[e,a]of[[`node`,t.nodes],[`edge`,t.edges]]){if(!a)continue;a.lengths&&a.starts&&n.push({name:`${e}Lengths`,view:a.lengths,type:`u32`,access:`read`},{name:`${e}Starts`,view:a.starts,type:`u32`,access:`read`}),n.push({name:`${e}Total`,view:a.total,type:`u32`,access:`read_write`}),a.pathOffsets&&n.push({name:`${e}PathOffsets`,view:a.pathOffsets,type:`u32`,access:`read_write`});let t=i>0?`${e}Starts[${e}StartsOffset + ${i-1}u] + ${e}Lengths[${e}LengthsOffset + ${i-1}u]`:`0u`;r.push(`if (index == ${i}u) {
    let total = ${t};
    ${e}Total[${e}TotalOffset] = total;
    ${a.pathOffsets?`${e}PathOffsets[${e}PathOffsetsOffset + ${i}u] = total;`:``}
  }
  ${a.pathOffsets&&i>0?`if (index < ${i}u) { ${e}PathOffsets[${e}PathOffsetsOffset + index] = ${e}Starts[${e}StartsOffset + index]; }`:``}`)}return _(e,{id:t.id,operation:j,variant:`totals`,bindings:n,invocationCount:i+1,body:r.join(`
  `)})}function fe(e,t){let n=[{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`targets`,view:t.targets,type:`u32`,access:`read`},{name:`nodeLengths`,view:t.nodeLengths,type:`u32`,access:`read`},{name:`nodeStarts`,view:t.nodeStarts,type:`u32`,access:`read`},{name:`ids`,view:t.ids,type:`u32`,access:`read_write`}];return t.nodeIds&&n.push({name:`nodeIds`,view:t.nodeIds,type:`u32`,access:`read`}),_(e,{id:t.id,operation:j,variant:`write-nodes`,bindings:n,invocationCount:t.targets.length,declarations:`const CAPACITY: u32 = ${t.capacity}u;`,body:`let length = nodeLengths[nodeLengthsOffset + index];
  if (length == 0u) { return; }
  let start = nodeStarts[nodeStartsOffset + index];
  var node = targets[targetsOffset + index];
  for (var step = 0u; step < length; step++) {
    let position = start + length - 1u - step;
    if (position < CAPACITY) {
      ids[idsOffset + position] = ${t.nodeIds?`nodeIds[nodeIdsOffset + node]`:`node`};
    }
    node = predecessors[predecessorsOffset + node];
  }`})}function pe(e,t){return _(e,{id:t.id,operation:j,variant:`write-links`,bindings:[{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`targets`,view:t.targets,type:`u32`,access:`read`},{name:`nodeLengths`,view:t.nodeLengths,type:`u32`,access:`read`},{name:`edgeStarts`,view:t.edgeStarts,type:`u32`,access:`read`},{name:`linkTails`,view:t.linkTails,type:`u32`,access:`read_write`},{name:`linkHeads`,view:t.linkHeads,type:`u32`,access:`read_write`}],invocationCount:t.targets.length,declarations:`const CAPACITY: u32 = ${t.capacity}u;`,body:`let length = nodeLengths[nodeLengthsOffset + index];
  if (length < 2u) { return; }
  let start = edgeStarts[edgeStartsOffset + index];
  var head = targets[targetsOffset + index];
  for (var step = 0u; step + 1u < length; step++) {
    let tail = predecessors[predecessorsOffset + head];
    let position = start + length - 2u - step;
    if (position < CAPACITY) {
      linkTails[linkTailsOffset + position] = tail;
      linkHeads[linkHeadsOffset + position] = head;
    }
    head = tail;
  }`})}function me(e,t){let n=[{name:`linkTails`,view:t.linkTails,type:`u32`,access:`read`},{name:`linkHeads`,view:t.linkHeads,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights,type:`f32`,access:`read`},{name:`ids`,view:t.ids,type:`u32`,access:`read_write`}];return t.edgeIds&&n.push({name:`edgeIds`,view:t.edgeIds,type:`u32`,access:`read`}),_(e,{id:t.id,operation:j,variant:`resolve-edges`,bindings:n,invocationCount:t.ids.length,declarations:`${le}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;`,body:`let tail = linkTails[linkTailsOffset + index];
  var result = NONE;
  if (tail < NODE_COUNT) {
    let head = linkHeads[linkHeadsOffset + index];
    let tailCost = costs[costsOffset + tail];
    let headCost = costs[costsOffset + head];
    let rangeStart = min(offsets[offsetsOffset + tail], EDGE_COUNT);
    let rangeEnd = min(offsets[offsetsOffset + tail + 1u], EDGE_COUNT);
    var exact = NONE;
    var fallback = NONE;
    for (var edge = rangeStart; edge < rangeEnd; edge++) {
      if (neighbors[neighborsOffset + edge] != head) { continue; }
      if (fallback == NONE) { fallback = edge; }
      let weight = weights[weightsOffset + edge];
      if (weight >= 0.0 && tailCost + weight == headCost) {
        exact = edge;
        break;
      }
    }
    result = select(exact, fallback, exact == NONE);
    ${t.edgeIds?`if (result != NONE) { result = edgeIds[edgeIdsOffset + result]; }`:``}
  }
  ids[idsOffset + index] = result;`})}var he=65536,ge=1024,P=`GPUNetworkPathExtraction`,_e=class{id;props;maximumPathLength;constructor(e){this.id=e.id??`network-path-extraction`,this.props=e,this.maximumPathLength=e.maximumPathLength??ge;let{id:t}=this,{edges:n,output:r}=e;for(let[i,a]of[[`predecessors`,e.predecessors],[`targets`,e.targets],[`targetCount`,typeof e.targetCount==`object`?e.targetCount:void 0],[`predecessorsComplete`,e.predecessorsComplete],[`nodeIds`,e.nodeIds],[`output.lineOffsets`,r.lineOffsets],[`output.pathFound`,r.pathFound],[`edges.edgeIds`,n?.edgeIds],[`edges.output.lineOffsets`,n?.output.lineOffsets]])a&&d(a,[`uint32`],`${t} ${i}`);for(let[n,i]of[[`costs`,e.costs],[`output.pathCosts`,r.pathCosts]])i&&d(i,[`float32`],`${t} ${n}`);let o=e.predecessors.length,s=e.targets.length;if(o<1)throw Error(`${t} predecessors must contain at least one node`);if(e.costs.length!==o)throw Error(`${t} costs length must equal predecessors length`);if(e.nodeIds&&e.nodeIds.length!==o)throw Error(`${t} nodeIds length must equal the node count`);if(typeof e.targetCount==`number`)i(t,`targetCount`,e.targetCount,0,s);else if(e.targetCount&&e.targetCount.length!==1)throw Error(`${t} targetCount must contain exactly one row`);if(e.predecessorsComplete&&e.predecessorsComplete.length!==1)throw Error(`${t} predecessorsComplete must contain exactly one row`);for(let[e,n]of[[`output.pathCosts`,r.pathCosts],[`output.pathFound`,r.pathFound]])if(n&&n.length!==s)throw Error(`${t} ${e} length must equal targets length`);for(let[e,i]of[[`output.lineOffsets`,r.lineOffsets],[`edges.output.lineOffsets`,n?.output.lineOffsets]])if(i&&i.length!==s+1)throw Error(`${t} ${e} must contain one more row than targets`);if(n){let e=T(`${t} edges`,n.topology,{weights:!0});if(e.nodeCount!==o)throw Error(`${t} edges.topology must contain one node per predecessors row`);if(n.edgeIds&&n.edgeIds.length!==e.edgeCount)throw Error(`${t} edges.edgeIds length must equal topology.neighbors length`);a(`${t} edges`,n.output)}i(t,`maximumPathLength`,this.maximumPathLength,1,he),a(t,r);let c=[r.ids,r.status.count,r.status.overflow,r.status.requiredCount,r.lineOffsets,r.pathCosts,r.pathFound,n?.output.ids,n?.output.status.count,n?.output.status.overflow,n?.output.status.requiredCount,n?.output.lineOffsets].filter(e=>e!==void 0).map(e=>e.buffer),l=[e.predecessors,e.costs,e.targets,typeof e.targetCount==`object`?e.targetCount:void 0,e.predecessorsComplete,e.nodeIds,...n?E(n.topology):[],n?.edgeIds].filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(c).size!==c.length||c.some(e=>l.includes(e)))throw Error(`${t} outputs must use separate buffers from each other and from inputs`)}getCommandNodes(e){let{id:t,props:n,maximumPathLength:i}=this,{edges:a,output:o}=n;r(t,e,ve(n));let s=n.predecessors.length,c=n.targets.length,l=u(e,`${t}-node-total`,`uint32`,1),d=a?u(e,`${t}-edge-total`,`uint32`,1):void 0,p=[];if(c===0)return p.push(N(e,{id:`${t}-totals`,targetCapacity:c,nodes:{total:l,pathOffsets:o.lineOffsets},edges:a&&d?{total:d,pathOffsets:a.output.lineOffsets}:void 0}),g(e,{id:`${t}-publish`,operation:P,requiredCount:l,output:o})),a&&d&&p.push(g(e,{id:`${t}-edge-publish`,operation:P,requiredCount:d,output:a.output})),p;let m=u(e,`${t}-status`,`uint32`,2),_=u(e,`${t}-node-lengths`,`uint32`,c),v=u(e,`${t}-node-starts`,`uint32`,c);p.push(M(e,{id:`${t}-clear`,status:m,converged:n.predecessorsComplete}),ue(e,{id:`${t}-measure`,nodeCount:s,maxPathLength:i,predecessors:n.predecessors,costs:n.costs,targets:n.targets,targetCount:n.targetCount,nodeLengths:_,status:m,pathFound:o.pathFound,pathCosts:o.pathCosts}),...new f({id:`${t}-node-scan`,input:_,output:v}).getCommandNodes(e));let y,b;if(a&&(y=u(e,`${t}-edge-lengths`,`uint32`,c),b=u(e,`${t}-edge-starts`,`uint32`,c),p.push(de(e,{id:`${t}-edge-lengths`,nodeLengths:_,edgeLengths:y}),...new f({id:`${t}-edge-scan`,input:y,output:b}).getCommandNodes(e))),p.push(N(e,{id:`${t}-totals`,targetCapacity:c,nodes:{lengths:_,starts:v,total:l,pathOffsets:o.lineOffsets},edges:a&&d?{lengths:y,starts:b,total:d,pathOffsets:a.output.lineOffsets}:void 0})),o.ids.length>0&&p.push(fe(e,{id:`${t}-write-nodes`,capacity:o.ids.length,predecessors:n.predecessors,targets:n.targets,nodeLengths:_,nodeStarts:v,nodeIds:n.nodeIds,ids:o.ids})),p.push(g(e,{id:`${t}-publish`,operation:P,requiredCount:l,output:o,overflowSources:[m]})),a&&d&&b){let r=a.output.ids.length;if(r>0){let i=u(e,`${t}-link-tails`,`uint32`,r),o=u(e,`${t}-link-heads`,`uint32`,r);p.push(h(e,{id:`${t}-link-clear`,operation:P,view:i,type:`u32`,value:`0xffffffffu`}),pe(e,{id:`${t}-write-links`,capacity:r,predecessors:n.predecessors,targets:n.targets,nodeLengths:_,edgeStarts:b,linkTails:i,linkHeads:o}),me(e,{id:`${t}-write-edges`,nodeCount:s,edgeCount:a.topology.neighbors.length,linkTails:i,linkHeads:o,costs:n.costs,offsets:a.topology.offsets,neighbors:a.topology.neighbors,weights:a.topology.weights,edgeIds:a.edgeIds,ids:a.output.ids}))}p.push(g(e,{id:`${t}-edge-publish`,operation:P,requiredCount:d,output:a.output,overflowSources:[m]}))}return p}};function ve(e){let{edges:t,output:n}=e;return[e.predecessors,e.costs,e.targets,typeof e.targetCount==`object`?e.targetCount:void 0,e.predecessorsComplete,e.nodeIds,n.ids,n.status.count,n.status.overflow,n.status.requiredCount,n.lineOffsets,n.pathCosts,n.pathFound,...t?E(t.topology):[],t?.edgeIds,t?.output.ids,t?.output.status.count,t?.output.status.overflow,t?.output.status.requiredCount,t?.output.lineOffsets]}var ye=`GPUNetworkNeighborhood`;function be(e,t){let n=[{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`hopsIn`,view:t.hops,type:`u32`,access:`read`},{name:`hopDistances`,view:t.hopDistances,type:`u32`,access:`read_write`}];return t.nodeMask&&n.push({name:`nodeMask`,view:t.nodeMask,type:`u32`,access:`read_write`}),_(e,{id:t.id,operation:ye,variant:`hops`,record:{datum:{keySpace:`node`},formula:`gpu-network.neighborhood.hops`},bindings:n,invocationCount:t.nodeCount,declarations:`const NONE: u32 = 0xffffffffu;
const MAXIMUM_HOPS: u32 = ${t.maximumHops}u;`,body:`let limit = min(hopsIn[hopsInOffset], MAXIMUM_HOPS);
  let cost = costs[costsOffset + index];
  // NaN and +Infinity compare false and stay outside the ego network.
  let inside = cost <= f32(limit);
  hopDistances[hopDistancesOffset + index] = select(NONE, u32(select(0.0, cost, inside)), inside);
  ${t.nodeMask?`nodeMask[nodeMaskOffset + index] = select(0u, 1u, inside);`:``}`})}function xe(e,t){return _(e,{id:t.id,operation:ye,variant:`edge-mask`,bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`nodeMask`,view:t.nodeMask,type:`u32`,access:`read`},{name:`edgeMask`,view:t.edgeMask,type:`u32`,access:`read_write`}],invocationCount:t.nodeCount,declarations:`const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;`,body:`if (nodeMask[nodeMaskOffset + index] == 0u) {
    return;
  }
  let rowStart = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let rowEnd = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = rowStart; edge < rowEnd; edge++) {
    let neighbor = neighbors[neighborsOffset + edge];
    var inside = 0u;
    if (neighbor < NODE_COUNT && nodeMask[nodeMaskOffset + neighbor] != 0u) {
      inside = 1u;
    }
    edgeMask[edgeMaskOffset + edge] = inside;
  }`})}var Se=8,F=`GPUNetworkNeighborhood`,Ce=class{id;props;maximumHops;constructor(e){this.id=e.id??`network-neighborhood`,this.props=e,this.maximumHops=e.maximumHops??Se;let{id:t}=this,{output:n}=e,{nodeCount:r,edgeCount:s}=T(t,e.topology);for(let[r,i]of[[`startNodes`,e.startNodes],[`startCount`,typeof e.startCount==`object`?e.startCount:void 0],[`hops`,e.hops],[`nodeIds`,e.nodeIds],[`edgeIds`,e.edgeIds],[`output.hopDistances`,n.hopDistances],[`output.nodeMask`,n.nodeMask],[`output.edgeMask`,n.edgeMask]])i&&d(i,[`uint32`],`${t} ${r}`);for(let e of[n.nodes,n.edges])e&&a(t,e);let c=n.hopDistances.length;if(c<1)throw Error(`${t} output.hopDistances must contain at least one node`);if(r!==c)throw Error(`${t} topology must contain one node per output.hopDistances row`);typeof e.startCount==`number`&&i(t,`startCount`,e.startCount,0,e.startNodes.length);for(let[n,r]of[[`startCount`,typeof e.startCount==`object`?e.startCount:void 0],[`hops`,e.hops]])if(r&&r.length!==1)throw Error(`${t} ${n} must contain exactly one row`);for(let[r,i,a,o]of[[`nodeIds`,e.nodeIds,c,`output.hopDistances`],[`output.nodeMask`,n.nodeMask,c,`output.hopDistances`],[`edgeIds`,e.edgeIds,s,`topology.neighbors`],[`output.edgeMask`,n.edgeMask,s,`topology.neighbors`]])if(i&&i.length!==a)throw Error(`${t} ${r} length must equal ${o} length`);i(t,`maximumHops`,this.maximumHops,1,D);let l=Ee(n);o(t,l,Te(e));let u=l.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(u).size!==u.length)throw Error(`${t} outputs must use separate buffers from each other`)}getCommandNodes(e){let{id:t,props:n,maximumHops:i}=this,{output:a}=n,{offsets:o,neighbors:s}=n.topology;r(t,e,[...Te(n),...Ee(a)]);let c=a.hopDistances.length,l=s.length,d=!!(a.edges||a.edgeMask),f=!!(a.nodeMask||a.nodes||d),p=[],m=u(e,`${t}-unit-weights`,`float32`,l);l>0&&p.push(h(e,{id:`${t}-unit-weights`,operation:F,view:m,type:`f32`,value:`1.0`}));let _=u(e,`${t}-costs`,`float32`,c);p.push(...new re({id:`${t}-reachability`,topology:{offsets:o,neighbors:s,weights:m},startNodes:n.startNodes,startCount:n.startCount,maximumIterations:i,activeIterations:n.hops,output:{costs:_}}).getCommandNodes(e));let v=a.nodeMask??(f?u(e,`${t}-node-mask`,`uint32`,c):void 0);p.push(be(e,{id:`${t}-hops`,nodeCount:c,maximumHops:i,costs:_,hops:n.hops,hopDistances:a.hopDistances,nodeMask:v}));let y;if(d&&l>0&&v&&(y=a.edgeMask??u(e,`${t}-edge-mask`,`uint32`,l),p.push(h(e,{id:`${t}-edge-clear`,operation:F,view:y,type:`u32`,value:`0u`}),xe(e,{id:`${t}-edge-mask`,nodeCount:c,edgeCount:l,offsets:o,neighbors:s,nodeMask:v,edgeMask:y}))),a.nodes&&v&&p.push(...we(e,{id:`${t}-node`,mask:v,sourceIds:n.nodeIds,output:a.nodes})),a.edges)if(y)p.push(...we(e,{id:`${t}-edge`,mask:y,sourceIds:n.edgeIds,output:a.edges}));else{let n=u(e,`${t}-edge-total`,`uint32`,1);p.push(h(e,{id:`${t}-edge-total`,operation:F,view:n,type:`u32`,value:`0u`}),g(e,{id:`${t}-edge-publish`,operation:F,requiredCount:n,output:a.edges}))}return p}};function we(e,t){let{id:n,mask:r,output:i}=t,a=r.length,o=i.ids.length>=a,s=u(e,`${n}-total`,`uint32`,1),c=o?i.ids:u(e,`${n}-compact-ids`,`uint32`,a);return[...new m({id:`${n}-visibility`,predicates:[{kind:`selection`,mask:r}],output:c,count:s,outputMask:r,sourceIds:t.sourceIds}).getCommandNodes(e),g(e,{id:`${n}-publish`,operation:F,requiredCount:s,compactIds:o?void 0:c,output:i})]}function Te(e){return[...E(e.topology),e.startNodes,typeof e.startCount==`object`?e.startCount:void 0,e.hops,e.nodeIds,e.edgeIds]}function Ee(e){return[e.hopDistances,e.nodeMask,e.edgeMask,e.nodes?.ids,e.nodes?.status.count,e.nodes?.status.overflow,e.nodes?.status.requiredCount,e.edges?.ids,e.edges?.status.count,e.edges?.status.overflow,e.edges?.status.requiredCount]}function De(e={},t=new Float32Array(8)){if(t.length<8)throw Error(`GPUNetworkLineGraph parameter target must hold 8 elements`);let n=e.angleCost??0,r=e.leftTurnCost??0,i=e.rightTurnCost??0;for(let[e,t]of[[`angleCost`,n],[`leftTurnCost`,r],[`rightTurnCost`,i]])if(!(t>=0)||!Number.isFinite(t))throw Error(`GPUNetworkLineGraph ${e} must be a non-negative finite number`);let a=e.bannedTurnCount??0;if(!Number.isSafeInteger(a)||a<0)throw Error(`GPUNetworkLineGraph bannedTurnCount must be a non-negative integer`);return t[0]=n,t[1]=r,t[2]=i,t[3]=e.uTurnCost??-1,t[4]=e.straightAngle??.5,t[5]=a,t[6]=e.uTurnAngle??2.9,t[7]=0,t}var Oe=Object.freeze({...t,measure:`distance`}),I=`GPUNetworkLineGraph`;function ke(e){return s(e,8)}function Ae(e){return s(e,16)}var je=class{id;props;nodeCount;edgeCount;arcCapacity;spatialContext;constructor(t){this.id=t.id??`network-line-graph`,this.props=t;let{id:n}=this;this.spatialContext=e(n,t.spatialContext,Oe);let{output:r}=t,{nodeCount:i,edgeCount:a}=T(n,t.topology,{weights:!0,nodePositions:!0});for(let[e,i]of[[`bannedTurns`,t.bannedTurns],[`output.topology.offsets`,r.topology.offsets],[`output.topology.neighbors`,r.topology.neighbors],[`output.arcCount`,r.arcCount],[`output.status.overflow`,r.status?.overflow]])i&&d(i,[`uint32`],`${n} ${e}`);for(let[e,i]of[[`parameters`,t.parameters],[`output.topology.weights`,r.topology.weights]])d(i,[`float32`],`${n} ${e}`);if(this.nodeCount=i,this.edgeCount=a,this.arcCapacity=r.topology.neighbors.length,this.arcCapacity<1)throw Error(`${n} output.topology.neighbors must hold at least one arc`);if(!r.status?.overflow)throw Error(`${n} output.status.overflow is required`);if(t.parameters.length<8)throw Error(`${n} parameters must hold 8 elements`);if(t.bannedTurns&&t.bannedTurns.length%2!=0)throw Error(`${n} bannedTurns length must be even`);if(r.topology.offsets.length!==this.edgeCount+1)throw Error(`${n} output.topology.offsets must contain edgeCount + 1 rows`);if(r.topology.weights.length!==this.arcCapacity)throw Error(`${n} output.topology.weights length must equal neighbors length`);for(let[e,t]of[[`output.arcCount`,r.arcCount],[`output.status.overflow`,r.status?.overflow]])if(t&&t.length!==1)throw Error(`${n} ${e} must contain exactly one row`);let s=L(t);o(n,s,Pe(t));let c=s.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(c).size!==c.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(e){let{id:t,props:n,nodeCount:i,edgeCount:a,arcCapacity:o}=this;r(t,e,[...Pe(n),...L(n)]),this.spatialContext.coordinateSpace===`longitude-latitude`?l(t,`topology edges with a great-circle spatialContext`,a,Ae(e.device)):l(t,`topology edges`,a,ke(e.device));let s=[],c=n.bannedTurns?.length?n.bannedTurns:void 0,d=!!c,p=c?c.length/2:0,m=c?this.addBanSortNodes(e,s,c):void 0,g=this.spatialContext.coordinateSpace===`longitude-latitude`,b=g?4:2,x=u(e,`${t}-edge-vectors`,`float32`,b*a),ee=u(e,`${t}-counts`,`uint32`,a+1),te=`const NODE_COUNT: u32 = ${i}u;
const EDGE_COUNT: u32 = ${a}u;
const ARC_CAPACITY: u32 = ${o}u;
const EDGE_VECTOR_STRIDE: u32 = ${b}u;
const ARRIVAL_OFFSET: u32 = ${b-2}u;
${Ne}
${d?`const BAN_CAPACITY: u32 = ${p}u;
// sortedBans holds (fromEdge, toEdge) pairs in ascending fromEdge order; inactive rows have
// fromEdge = EDGE_COUNT and sort last. Find the first pair of fromEdge, then scan its run.
fn isBannedTurn(fromEdge: u32, toEdge: u32) -> bool {
  var low = 0u;
  var high = BAN_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedBans[sortedBansOffset + 2u * middle] < fromEdge) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  for (var ban = low; ban < BAN_CAPACITY; ban++) {
    if (sortedBans[sortedBansOffset + 2u * ban] != fromEdge) {
      break;
    }
    if (sortedBans[sortedBansOffset + 2u * ban + 1u] == toEdge) {
      return true;
    }
  }
  return false;
}`:`fn isBannedTurn(fromEdge: u32, toEdge: u32) -> bool {
  return false;
}`}
// Only a negative U-turn cost makes getTurnCost negative (every other term is non-negative), so
// with U-turns priced the angle is never needed, and a turn whose edges point forward (positive dot
// product) is below any U-turn threshold clearly above a right angle (1.6 rad leaves a margin for
// backend atan2 error). Both skip the atan2 that the cost would need; the remaining arcs evaluate
// the exact cost as before.
fn isAllowedTurn(fromEdge: u32, toEdge: u32) -> bool {
  if (isBannedTurn(fromEdge, toEdge)) {
    return false;
  }
  if (parameters[parametersOffset + 3u] >= 0.0) {
    return true;
  }
  if (parameters[parametersOffset + 6u] > 1.6) {
    let a = getArrivalDirection(fromEdge);
    let b = getDepartureDirection(toEdge);
    if (dot(a, a) >= 0.5 && dot(b, b) >= 0.5 && dot(a, b) > 0.0) {
      return true;
    }
  }
  return getTurnCost(fromEdge, toEdge) >= 0.0;
}`,S=[{name:`edgeVectors`,view:x,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`}];m&&S.push({name:`sortedBans`,view:m,type:`u32`,access:`read`});let C=`fn isLiveEdge(edge: u32) -> bool {
  return edge >= offsets[offsetsOffset] && edge < min(offsets[offsetsOffset + NODE_COUNT], EDGE_COUNT);
}`;s.push(_(e,{id:`${t}-edge-vectors`,operation:I,variant:`edge-vectors`,bindings:[{name:`offsets`,view:n.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.topology.neighbors,type:`u32`,access:`read`},{name:`nodePositions`,view:n.topology.nodePositions,type:`f32`,access:`read`},{name:`edgeVectors`,view:x,type:`f32`,access:`read_write`}],invocationCount:i,declarations:`const NODE_COUNT: u32 = ${i}u;
const EDGE_COUNT: u32 = ${a}u;
${g?y:``}`,body:g?Me:`let start = nodePositions[nodePositionsOffset + 2u * index];
  let startY = nodePositions[nodePositionsOffset + 2u * index + 1u];
  let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    let headNode = neighbors[neighborsOffset + edge];
    var direction = vec2<f32>(0.0);
    if (headNode < NODE_COUNT) {
      let delta = vec2<f32>(
        nodePositions[nodePositionsOffset + 2u * headNode] - start,
        nodePositions[nodePositionsOffset + 2u * headNode + 1u] - startY
      );
      let edgeLength = sqrt(dot(delta, delta));
      if (edgeLength > 0.0) {
        direction = delta / edgeLength;
      }
    }
    edgeVectors[edgeVectorsOffset + 2u * edge] = direction.x;
    edgeVectors[edgeVectorsOffset + 2u * edge + 1u] = direction.y;
  }`})),s.push(_(e,{id:`${t}-count`,operation:I,variant:`count`,bindings:[{name:`offsets`,view:n.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.topology.neighbors,type:`u32`,access:`read`},...S,{name:`counts`,view:ee,type:`u32`,access:`read_write`}],invocationCount:a+1,declarations:`${te}
${C}`,body:`var count = 0u;
  if (index < EDGE_COUNT && isLiveEdge(index)) {
    let head = neighbors[neighborsOffset + index];
    if (head < NODE_COUNT) {
      let end = min(offsets[offsetsOffset + head + 1u], EDGE_COUNT);
      for (var next = offsets[offsetsOffset + head]; next < end; next++) {
        if (isAllowedTurn(index, next)) {
          count++;
        }
      }
    }
  }
  counts[countsOffset + index] = count;`})),s.push(...new f({id:`${t}-scan`,input:ee,output:n.output.topology.offsets,mode:`exclusive`}).getCommandNodes(e));let w=u(e,`${t}-scan-total`,`uint32`,1);if(s.push(...v(e,{id:t,operation:I,counts:ee,offsets:n.output.topology.offsets,rowCount:a,total:w})),n.output.arcCount||n.output.status.overflow){let r=[{name:`scanTotal`,view:w,type:`u32`,access:`read`}];n.output.arcCount&&r.push({name:`arcCount`,view:n.output.arcCount,type:`u32`,access:`read_write`}),n.output.status.overflow&&r.push({name:`overflow`,view:n.output.status.overflow,type:`u32`,access:`read_write`}),s.push(_(e,{id:`${t}-publish`,operation:I,variant:`publish`,bindings:r,invocationCount:1,declarations:`const EDGE_COUNT: u32 = ${a}u;
const ARC_CAPACITY: u32 = ${o}u;`,body:`let total = scanTotal[scanTotalOffset];
  ${n.output.arcCount?`arcCount[arcCountOffset] = min(total, ARC_CAPACITY);`:``}
  ${n.output.status.overflow?`overflow[overflowOffset] = select(0u, 1u, total > ARC_CAPACITY);`:``}`}))}let T=8+(m?1:0);s.push(h(e,{id:`${t}-clear-line-neighbors`,operation:I,view:n.output.topology.neighbors,type:`u32`,value:`0xffffffffu`}),h(e,{id:`${t}-clear-line-weights`,operation:I,view:n.output.topology.weights,type:`f32`,value:`-1.0`}));let E=T<=e.device.limits.maxStorageBuffersPerShaderStage;return s.push(_(e,{id:`${t}-fill-neighbors`,operation:I,variant:E?`fill`:`fill-neighbors`,record:{datum:{keySpace:`edge`},formula:E?`gpu-network.line-graph.arc-weight`:void 0},bindings:[{name:`offsets`,view:n.topology.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.topology.neighbors,type:`u32`,access:`read`},...S,{name:`lineOffsets`,view:n.output.topology.offsets,type:`u32`,access:`read`},{name:`lineNeighbors`,view:n.output.topology.neighbors,type:`u32`,access:`read_write`},...E?[{name:`weights`,view:n.topology.weights,type:`f32`,access:`read`},{name:`lineWeights`,view:n.output.topology.weights,type:`f32`,access:`read_write`}]:[]],invocationCount:a,declarations:`${te}
${C}`,body:`if (!isLiveEdge(index)) {
    return;
  }
  let head = neighbors[neighborsOffset + index];
  if (head >= NODE_COUNT) {
    return;
  }
  var slot = lineOffsets[lineOffsetsOffset + index];
  let end = min(offsets[offsetsOffset + head + 1u], EDGE_COUNT);
  for (var next = offsets[offsetsOffset + head]; next < end; next++) {
    ${E?`if (isBannedTurn(index, next)) {
      continue;
    }
    let turnCost = getTurnCost(index, next);
    if (turnCost >= 0.0) {
      if (slot < ARC_CAPACITY) {
        lineNeighbors[lineNeighborsOffset + slot] = next;
        let baseCost = weights[weightsOffset + next];
        lineWeights[lineWeightsOffset + slot] =
          select(-1.0, baseCost + max(turnCost, 0.0), baseCost >= 0.0);
      }
      slot++;
    }`:`if (isAllowedTurn(index, next)) {
      if (slot < ARC_CAPACITY) {
        lineNeighbors[lineNeighborsOffset + slot] = next;
      }
      slot++;
    }`}
  }`})),E||s.push(_(e,{id:`${t}-fill-weights`,operation:I,variant:`fill-weights`,record:{datum:{keySpace:`arc`},formula:`gpu-network.line-graph.arc-weight`},bindings:[{name:`lineOffsets`,view:n.output.topology.offsets,type:`u32`,access:`read`},{name:`lineNeighbors`,view:n.output.topology.neighbors,type:`u32`,access:`read`},{name:`weights`,view:n.topology.weights,type:`f32`,access:`read`},...S,{name:`lineWeights`,view:n.output.topology.weights,type:`f32`,access:`read_write`}],invocationCount:o,declarations:te,body:`if (index >= lineOffsets[lineOffsetsOffset + EDGE_COUNT]) {
    return;
  }
  var low = 0u;
  var high = EDGE_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle + 1u] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  let next = lineNeighbors[lineNeighborsOffset + index];
  let baseCost = weights[weightsOffset + next];
  lineWeights[lineWeightsOffset + index] =
    select(-1.0, baseCost + max(getTurnCost(low, next), 0.0), baseCost >= 0.0);`})),s.push(_(e,{id:`${t}-clamp`,operation:I,variant:`clamp`,bindings:[{name:`lineOffsets`,view:n.output.topology.offsets,type:`u32`,access:`read_write`}],invocationCount:a+1,declarations:`const ARC_CAPACITY: u32 = ${o}u;`,body:`lineOffsets[lineOffsetsOffset + index] =
    min(lineOffsets[lineOffsetsOffset + index], ARC_CAPACITY);`})),s}addBanSortNodes(e,t,n){let{id:r,edgeCount:i}=this,a=n.length/2,o=(t,n)=>u(e,`${r}-${t}`,`uint32`,n),s=o(`ban-keys`,a),c=o(`ban-values`,a),l=o(`ban-sorted-keys`,a),d=o(`ban-sorted-values`,a),f=o(`sorted-bans`,2*a);return t.push(_(e,{id:`${r}-ban-keys`,operation:I,variant:`ban-keys`,bindings:[{name:`parameters`,view:this.props.parameters,type:`f32`,access:`read`},{name:`bannedTurns`,view:n,type:`u32`,access:`read`},{name:`keys`,view:s,type:`u32`,access:`read_write`},{name:`values`,view:c,type:`u32`,access:`read_write`}],invocationCount:a,declarations:`const EDGE_COUNT: u32 = ${i}u;`,body:`let bannedCount = u32(max(parameters[parametersOffset + 5u], 0.0));
  let fromEdge = bannedTurns[bannedTurnsOffset + 2u * index];
  let isActive = index < bannedCount && fromEdge < EDGE_COUNT;
  keys[keysOffset + index] = select(EDGE_COUNT, fromEdge, isActive);
  values[valuesOffset + index] = bannedTurns[bannedTurnsOffset + 2u * index + 1u];`}),...new p({id:`${r}-ban-sort`,keys:s,values:c,outputKeys:l,outputValues:d,keyBits:Math.max(1,Math.ceil(Math.log2(i+1)))}).getCommandNodes(e),_(e,{id:`${r}-ban-pairs`,operation:I,variant:`ban-pairs`,bindings:[{name:`sortedKeys`,view:l,type:`u32`,access:`read`},{name:`sortedValues`,view:d,type:`u32`,access:`read`},{name:`sortedBans`,view:f,type:`u32`,access:`read_write`}],invocationCount:a,body:`sortedBans[sortedBansOffset + 2u * index] = sortedKeys[sortedKeysOffset + index];
  sortedBans[sortedBansOffset + 2u * index + 1u] = sortedValues[sortedValuesOffset + index];`})),f}},Me=`let start = vec2<f32>(
    nodePositions[nodePositionsOffset + 2u * index],
    nodePositions[nodePositionsOffset + 2u * index + 1u]
  );
  let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    let headNode = neighbors[neighborsOffset + edge];
    var departure = vec2<f32>(0.0);
    var arrival = vec2<f32>(0.0);
    if (headNode < NODE_COUNT) {
      let head = vec2<f32>(
        nodePositions[nodePositionsOffset + 2u * headNode],
        nodePositions[nodePositionsOffset + 2u * headNode + 1u]
      );
      if (geodesicCentralAngle(start, head) > 0.0) {
        let departureBearing = geodesicInitialBearingDegrees(start, head) * GEODESIC_DEGREES_TO_RADIANS;
        let returnBearing = geodesicInitialBearingDegrees(head, start) * GEODESIC_DEGREES_TO_RADIANS;
        departure = vec2<f32>(sin(departureBearing), cos(departureBearing));
        arrival = -vec2<f32>(sin(returnBearing), cos(returnBearing));
      }
    }
    edgeVectors[edgeVectorsOffset + 4u * edge] = departure.x;
    edgeVectors[edgeVectorsOffset + 4u * edge + 1u] = departure.y;
    edgeVectors[edgeVectorsOffset + 4u * edge + 2u] = arrival.x;
    edgeVectors[edgeVectorsOffset + 4u * edge + 3u] = arrival.y;
  }`,Ne=`
fn getDepartureDirection(edge: u32) -> vec2<f32> {
  let base = edgeVectorsOffset + EDGE_VECTOR_STRIDE * edge;
  return vec2<f32>(edgeVectors[base], edgeVectors[base + 1u]);
}
fn getArrivalDirection(edge: u32) -> vec2<f32> {
  let base = edgeVectorsOffset + EDGE_VECTOR_STRIDE * edge + ARRIVAL_OFFSET;
  return vec2<f32>(edgeVectors[base], edgeVectors[base + 1u]);
}
fn getTurnAngle(fromEdge: u32, toEdge: u32) -> f32 {
  let a = getArrivalDirection(fromEdge);
  let b = getDepartureDirection(toEdge);
  if (dot(a, a) < 0.5 || dot(b, b) < 0.5) {
    return 0.0;
  }
  let cross = a.x * b.y - a.y * b.x;
  let projection = dot(a, b);
  // atan2 with a signed-zero x is unreliable on some backends; right angles are common on grids.
  if (projection == 0.0) {
    return select(-1.5707963, 1.5707963, cross > 0.0);
  }
  return atan2(cross, projection);
}
// Turn cost of the turn fromEdge -> toEdge, or a negative value when U-turns are banned.
fn getTurnCost(fromEdge: u32, toEdge: u32) -> f32 {
  let angle = getTurnAngle(fromEdge, toEdge);
  let magnitude = abs(angle);
  if (magnitude >= parameters[parametersOffset + 6u]) {
    return parameters[parametersOffset + 3u];
  }
  var cost = parameters[parametersOffset] * magnitude;
  if (magnitude > parameters[parametersOffset + 4u]) {
    cost += select(
      parameters[parametersOffset + 2u],
      parameters[parametersOffset + 1u],
      angle > 0.0
    );
  }
  return cost;
}`;function Pe(e){return[...E(e.topology),e.parameters,e.bannedTurns]}function L(e){let{output:t}=e;return[t.topology.offsets,t.topology.neighbors,t.topology.weights,t.arcCount,t.status.overflow]}var Fe={willis:{label:`Willis Tower (Loop)`,coordinate:A.willisTower},ohare:{label:`O'Hare Airport`,coordinate:A.ohare},midway:{label:`Midway Airport`,coordinate:A.midway},wrigley:{label:`Wrigley Field`,coordinate:A.wrigleyField},soldier:{label:`Soldier Field`,coordinate:A.soldierField},unitedCenter:{label:`United Center`,coordinate:A.unitedCenter},msi:{label:`Museum of Science and Industry`,coordinate:A.scienceIndustry},southShore:{label:`South Shore`,coordinate:A.southShore},austin:{label:`Austin (West Side)`,coordinate:A.austin}},Ie=[`midway`,`wrigley`,`soldier`,`msi`,`unitedCenter`],R=6,z=2048,B=80,Le=24,Re=8192,ze=16384,V=8192,Be=16,Ve=[230,120,48,255],He=[113,72,150,255],Ue=[117,122,132,105],We=[35,38,48,255],Ge=w({breaks:[600,1200,1800,2700,3600],scheme:`YlGnBu`,reverse:!0,labels:[`0–10`,`10–20`,`20–30`,`30–45`,`45–60`,`60–90 min`],unit:`min`,extent:[0,5400],noData:{color:Ue,label:`Unreached`}}),Ke=w({breaks:[0,300,600,1200],scheme:`PuRd`,labels:[`No added time`,`0–5`,`5–10`,`10–20`,`>20 min`],unit:`min`,extent:[0,1200],transparent:[0],noData:{color:Ue,label:`Unreachable in scenario`}});async function qe(e){let{device:t}=e,r=ce(e.datasets.get(`chicago-roads`)),{nodeCount:i,edgeCount:a}=r,o=new c(t,`routing`),s=r.segmentEdges.length,l=o.createBuffer(`offsets`,r.offsets),u=o.createBuffer(`neighbors`,r.targets),d=o.createBuffer(`weights`,a*4),f=o.createBuffer(`free-flow-weights`,a*4),p=o.createBuffer(`node-positions`,r.nodePositions),m=o.createBuffer(`segments`,r.segments),h=o.createBuffer(`major-segments`,r.majorSegments),g=o.createBuffer(`segment-edges`,r.segmentEdges),_=o.createBuffer(`segment-targets`,r.segmentTargetNodes),v=new Uint32Array(i+1);for(let e=0;e<a;e++)v[r.targets[e]+1]++;for(let e=0;e<i;e++)v[e+1]+=v[e];let y=new Uint32Array(a);{let e=v.slice(0,i);for(let t=0;t<a;t++)y[e[r.targets[t]]++]=t}let w=o.createBuffer(`in-offsets`,v),T=o.createBuffer(`in-edges`,y),E=0,D=1;for(let e=0;e<a;e++){let t=r.targets[e];E+=r.offsets[t+1]-r.offsets[t]}for(let e=0;e<i;e++)D=Math.max(D,r.offsets[e+1]-r.offsets[e]);D=Math.min(D,Be),E=Math.max(E,1);let A=[];for(let e=0;e<i&&A.length<V*2;e++)for(let t=r.offsets[e];t<r.offsets[e+1];t++){let n=r.targets[t];if(!(r.roadClass[t]<2||r.roadClass[t]>3))for(let i=r.offsets[n];i<r.offsets[n+1];i++){if(r.roadClass[i]<2||r.roadClass[i]>3||r.targets[i]===e)continue;let n=ae(r,t,i);n>.5&&n<2.9&&A.length<V*2&&A.push(t,i)}}let j=A.length/2,le=o.createBuffer(`banned-turns`,V*2*4);le.write(Uint32Array.from(A));let M=o.createParameterBuffer(`origin`,`uint32`,1),ue=o.createParameterBuffer(`destinations`,`uint32`,R),de=o.createParameterBuffer(`destination-count`,`uint32`,1),N=o.createParameterBuffer(`cost-limit`,`float32`,1),fe=o.createParameterBuffer(`hops`,`uint32`,1),pe=o.createParameterBuffer(`turn-parameters`,`float32`,8),me=o.createParameterBuffer(`line-sources`,`uint32`,D),he=o.createParameterBuffer(`line-source-costs`,`float32`,D),ge=o.createBuffer(`markers`,(1+R)*8),P=o.createBuffer(`marker-ids`,Uint32Array.from({length:1+R},(e,t)=>t)),ve=o.createBuffer(`tree-costs`,i*4),ye=o.createBuffer(`tree-predecessors`,i*4),be=o.createBuffer(`tree-converged`,4),xe=o.createBuffer(`tree-iterations`,4),Se=o.createBuffer(`tree-unresolved`,4),F=R*z,we=o.createBuffer(`path-node-ids`,F*4),Te=o.createBuffer(`path-node-count`,4),Ee=o.createBuffer(`path-node-overflow`,4),Oe=o.createBuffer(`path-edge-ids`,F*4),I=o.createBuffer(`path-edge-count`,4),ke=o.createBuffer(`path-edge-overflow`,4),Ae=o.createBuffer(`path-offsets`,(R+1)*4),Me=o.createBuffer(`path-costs`,R*4),Ne=o.createBuffer(`path-found`,R*4),Pe=o.createBuffer(`route-flags`,a*4),L=o.createBuffer(`free-flow-tree-costs`,i*4),qe=o.createBuffer(`free-flow-predecessors`,i*4),Je=o.createBuffer(`free-flow-converged`,4),Ye=o.createBuffer(`free-flow-iterations`,4),Xe=o.createBuffer(`scenario-delta-costs`,i*4),Ze=o.createBuffer(`scenario-unreached`,i*4),Qe=o.createBuffer(`selected-route-flags`,a*4),$e=o.createBuffer(`hop-distances`,i*4),et=o.createBuffer(`hood-edge-mask`,a*4),tt=o.createBuffer(`hood-node-ids`,Re*4),nt=o.createBuffer(`hood-node-count`,4),rt=o.createBuffer(`hood-node-overflow`,4),it=o.createBuffer(`hood-edge-ids`,ze*4),at=o.createBuffer(`hood-edge-count`,4),ot=o.createBuffer(`hood-edge-overflow`,4),st=o.createBuffer(`line-offsets`,(a+1)*4),ct=o.createBuffer(`line-neighbors`,E*4),lt=o.createBuffer(`line-weights`,E*4),ut=o.createBuffer(`line-arc-count`,4),dt=o.createBuffer(`line-overflow`,4),ft=o.createBuffer(`line-costs`,a*4),pt=o.createBuffer(`line-predecessors`,a*4),mt=o.createBuffer(`line-converged`,4),ht=o.createBuffer(`line-iterations`,4),gt=o.createBuffer(`line-targets`,R*4),_t=o.createBuffer(`line-path-node-ids`,F*4),vt=o.createBuffer(`line-path-node-count`,4),yt=o.createBuffer(`line-path-node-overflow`,4),bt=o.createBuffer(`line-path-offsets`,(R+1)*4),xt=o.createBuffer(`line-path-costs`,R*4),St=o.createBuffer(`line-path-found`,R*4),Ct=o.createBuffer(`turn-flags`,a*4),wt=e=>{let t=(Fe[e]??Fe.willis).coordinate,[n,i]=r.projection.project(t[0],t[1]);return r.findNearestNode(n,i)},H=wt(e.options.originPlace),U=[],Tt=()=>{U.length=0,U.push(wt(e.options.destinationPlace));for(let e of Ie)U.push(wt(e));U.length=R};Tt();let W=new Float32Array(a),Et=new Float32Array(a),G=!1,K=null,q=2,J=2,Y=2,X=2,Z=null,Dt=null,Ot=null,Q={treeFound:new Uint32Array(R),treeCost:new Float32Array(R),turnFound:new Uint32Array(R),turnCost:new Float32Array(R),treeEdges:new Uint32Array,turnEdges:new Uint32Array,treeOffsets:new Uint32Array(R+1),turnOffsets:new Uint32Array(R+1)},kt=()=>{let t=e.options;se(r,{closeExpressways:t.closeExpressways,expresswaySlowdown:t.expresswaySlowdown,intersectionDelay:t.intersectionDelay},W),d.write(W),e.setReadout(`expressways`,t.closeExpressways?`closed`:`open`)},At=()=>{se(r,{closeExpressways:!1,expresswaySlowdown:1,intersectionDelay:0},Et),f.write(Et)},jt=()=>{M.write(Uint32Array.of(H));let e=new Float32Array((1+R)*2);e.set(r.nodePositions.subarray(H*2,H*2+2),0),U.forEach((t,n)=>{e.set(r.nodePositions.subarray(t*2,t*2+2),(1+n)*2)}),ge.write(e),ue.write(Uint32Array.from(U));let t=new Uint32Array(D).fill(ne),n=new Float32Array(D).fill(-1),i=0;for(let e=r.offsets[H];e<r.offsets[H+1]&&i<D;e++)W[e]>=0&&(t[i]=e,n[i]=W[e],i++);me.write(t),he.write(n)},Mt=()=>{let t=e.options;pe.write(De({angleCost:t.angleCost,leftTurnCost:t.leftTurnCost,rightTurnCost:t.rightTurnCost,uTurnCost:t.uTurns===`banned`?-1:90,straightAngle:t.straightAngle,bannedTurnCount:t.banLeftTurns===`arterials`?Math.min(j,V):0}))},Nt=()=>{let t=e.options;N.write(Float32Array.of(t.costLimitMinutes*60)),fe.write(Uint32Array.of(t.hops)),de.write(Uint32Array.of(t.destinationCount))};async function Pt(e){let s=e=>{let t=new Map;return(n,r,i)=>{let a=t.get(n);return a||(a=ee(e,n.id,n,r,i),t.set(n,a)),a}},c=new n(t,{id:`routing-free-flow`});{let t=s(c);c.add(new re({id:`free-flow-tree`,topology:{offsets:t(l,`uint32`,i+1),neighbors:t(u,`uint32`,a),weights:t(f,`float32`,a)},startNodes:M.importToGraph(c),costLimit:N.importToGraph(c),maximumIterations:B,localIterations:e,output:{costs:t(L,`float32`,i),predecessors:t(qe,`uint32`,i),status:{converged:t(Je,`uint32`,1),iterationCount:t(Ye,`uint32`,1)}}}))}let m=new n(t,{id:`routing-tree`});{let t=s(m),n=t(l,`uint32`,i+1),r=t(u,`uint32`,a),o=t(d,`float32`,a),c=t(ve,`float32`,i),f=t(ye,`uint32`,i),p=t(be,`uint32`,1);m.add(new re({id:`tree`,topology:{offsets:n,neighbors:r,weights:o},startNodes:M.importToGraph(m),costLimit:N.importToGraph(m),maximumIterations:B,localIterations:e,output:{costs:c,predecessors:f,unresolvedCount:t(Se,`uint32`,1),status:{converged:p,iterationCount:t(xe,`uint32`,1)}}}));let h=t(Oe,`uint32`,F),g=t(I,`uint32`,1);m.add(new _e({id:`route`,predecessors:f,costs:c,targets:ue.importToGraph(m),targetCount:de.importToGraph(m),predecessorsComplete:p,maximumPathLength:z,output:{ids:t(we,`uint32`,F),status:{count:t(Te,`uint32`,1),overflow:t(Ee,`uint32`,1)},lineOffsets:t(Ae,`uint32`,R+1),pathCosts:t(Me,`float32`,R),pathFound:t(Ne,`uint32`,R)},edges:{topology:{offsets:n,neighbors:r,weights:o},output:{ids:h,status:{count:g,overflow:t(ke,`uint32`,1)}}}})),Rt(m,`tree-flags`,t(Pe,`uint32`,a),h,g),C(m,{id:`scenario-delta`,bindings:[{name:`scenarioCosts`,view:c,type:`f32`,access:`read`},{name:`freeFlowCosts`,view:t(L,`float32`,i),type:`f32`,access:`read`},{name:`deltaCosts`,view:t(Xe,`float32`,i),type:`f32`,access:`read_write`},{name:`unreached`,view:t(Ze,`uint32`,i),type:`u32`,access:`read_write`}],invocationCount:i,body:`let scenario = scenarioCosts[scenarioCostsOffset + index];
  let freeFlow = freeFlowCosts[freeFlowCostsOffset + index];
  let scenarioBits = bitcast<u32>(scenario);
  let freeFlowBits = bitcast<u32>(freeFlow);
  let finite = (scenarioBits & 0x7f800000u) != 0x7f800000u && (freeFlowBits & 0x7f800000u) != 0x7f800000u;
  let delta = scenario - freeFlow;
  let noData = bitcast<f32>(0x7fc00000u | (index & 0u));
  // The class table reserves values at or below zero for the transparent no-added-time class.
  deltaCosts[deltaCostsOffset + index] = select(noData, select(delta, -1.0, delta <= 0.0), finite);
  unreached[unreachedOffset + index] = select(0u, 1u, (freeFlowBits & 0x7f800000u) != 0x7f800000u && (scenarioBits & 0x7f800000u) == 0x7f800000u);`}),C(m,{id:`selected-route-flags-clear`,bindings:[{name:`flags`,view:t(Qe,`uint32`,a),type:`u32`,access:`read_write`}],invocationCount:a,body:`flags[flagsOffset + index] = 0u;`}),C(m,{id:`selected-route-flags-set`,bindings:[{name:`ids`,view:h,type:`u32`,access:`read`},{name:`offsets`,view:t(Ae,`uint32`,R+1),type:`u32`,access:`read`},{name:`flags`,view:t(Qe,`uint32`,a),type:`u32`,access:`read_write`}],invocationCount:z,declarations:`const EDGE_COUNT: u32 = ${a}u;`,body:`if (index < offsets[offsetsOffset + 1u]) {
  let edge = ids[idsOffset + index];
  if (edge < EDGE_COUNT) { flags[flagsOffset + edge] = 1u; }
}`})}let h=new n(t,{id:`routing-hood`});{let e=s(h);h.add(new Ce({id:`hood`,topology:{offsets:e(l,`uint32`,i+1),neighbors:e(u,`uint32`,a)},startNodes:M.importToGraph(h),hops:fe.importToGraph(h),maximumHops:Le,output:{hopDistances:e($e,`uint32`,i),edgeMask:e(et,`uint32`,a),nodes:{ids:e(tt,`uint32`,Re),status:{count:e(nt,`uint32`,1),overflow:e(rt,`uint32`,1)}},edges:{ids:e(it,`uint32`,ze),status:{count:e(at,`uint32`,1),overflow:e(ot,`uint32`,1)}}}}))}let g=new n(t,{id:`routing-turns`});{let t=s(g),n=t(l,`uint32`,i+1),o=t(st,`uint32`,a+1),c=t(ct,`uint32`,E),f=t(lt,`float32`,E);g.add(new je({spatialContext:r.projection.spatialContext,id:`turns`,topology:{offsets:n,neighbors:t(u,`uint32`,a),weights:t(d,`float32`,a),nodePositions:t(p,`float32x2`,i)},parameters:pe.importToGraph(g),bannedTurns:t(le,`uint32`,V*2),output:{topology:{offsets:o,neighbors:c,weights:f},arcCount:t(ut,`uint32`,1),status:{overflow:t(dt,`uint32`,1)}}}));let m=t(ft,`float32`,a),h=t(mt,`uint32`,1),_=t(pt,`uint32`,a);g.add(new re({id:`line-tree`,topology:{offsets:o,neighbors:c,weights:f},startNodes:me.importToGraph(g),startCosts:he.importToGraph(g),costLimit:N.importToGraph(g),maximumIterations:B,localIterations:e,output:{costs:m,predecessors:_,status:{converged:h,iterationCount:t(ht,`uint32`,1)}}}));let v=ue.importToGraph(g),y=t(gt,`uint32`,R);C(g,{id:`arrival-edges`,bindings:[{name:`destinations`,view:v,type:`u32`,access:`read`},{name:`lineCosts`,view:m,type:`f32`,access:`read`},{name:`inOffsets`,view:t(w,`uint32`,i+1),type:`u32`,access:`read`},{name:`inEdges`,view:t(T,`uint32`,a),type:`u32`,access:`read`},{name:`targets`,view:y,type:`u32`,access:`read_write`}],invocationCount:R,declarations:`const NODE_COUNT: u32 = ${i}u;`,body:`let node = destinations[destinationsOffset + index];
  var bestBits = 0x7f800000u;
  var bestEdge = 0xffffffffu;
  if (node < NODE_COUNT) {
    for (var slot = inOffsets[inOffsetsOffset + node]; slot < inOffsets[inOffsetsOffset + node + 1u]; slot++) {
      let edge = inEdges[inEdgesOffset + slot];
      let bits = bitcast<u32>(lineCosts[lineCostsOffset + edge]);
      if (bits < bestBits) {
        bestBits = bits;
        bestEdge = edge;
      }
    }
  }
  targets[targetsOffset + index] = bestEdge;`});let b=t(_t,`uint32`,F),x=t(vt,`uint32`,1);g.add(new _e({id:`line-route`,predecessors:_,costs:m,targets:y,targetCount:de.importToGraph(g),predecessorsComplete:h,maximumPathLength:z,output:{ids:b,status:{count:x,overflow:t(yt,`uint32`,1)},lineOffsets:t(bt,`uint32`,R+1),pathCosts:t(xt,`float32`,R),pathFound:t(St,`uint32`,R)}})),Rt(g,`turn-flags`,t(Ct,`uint32`,a),b,x)}let[_,v,y,b]=await Promise.all([c,m,h,g].map(async e=>o.track(await e.compileAsync())));return{freeFlow:_,tree:{compiled:v,localIterations:e},hood:y,turns:b}}function Ft(t){let n=K;K=t,n&&It(n);let{localIterations:r}=t.tree;q=2,J=2,Y=2,X=2,e.setReadout(`solver`,`${B} rounds x ${r} hops`)}function It(e){o.release(e.freeFlow),o.release(e.tree.compiled),o.release(e.hood),o.release(e.turns)}let Lt=o.track(new te);function Rt(e,t,n,r,i){C(e,{id:`${t}-clear`,bindings:[{name:`flags`,view:n,type:`u32`,access:`read_write`}],invocationCount:a,body:`flags[flagsOffset + index] = 0u;`}),C(e,{id:`${t}-set`,bindings:[{name:`ids`,view:r,type:`u32`,access:`read`},{name:`count`,view:i,type:`u32`,access:`read`},{name:`flags`,view:n,type:`u32`,access:`read_write`}],invocationCount:F,declarations:`const EDGE_COUNT: u32 = ${a}u;`,body:`if (index < count[countOffset]) {
    let edge = ids[idsOffset + index];
    if (edge < EDGE_COUNT) {
      flags[flagsOffset + edge] = 1u;
    }
  }`})}let $=[R*4,R*4,(R+1)*4,4,4,4,4,4,4,4,z*4,i*4],zt=new S(o,`routing-tree`,[{buffer:Ne,size:$[0]},{buffer:Me,size:$[1]},{buffer:Ae,size:$[2]},{buffer:Te,size:4},{buffer:Ee,size:4},{buffer:I,size:4},{buffer:ke,size:4},{buffer:be,size:4},{buffer:xe,size:4},{buffer:Se,size:4},{buffer:Oe,size:$[10]},{buffer:ve,size:$[11]}],t=>{if(G)return;let[n,r,a,o,s,c,l,u,d,f,p,m]=ie(t,$);Q.treeFound=new Uint32Array(n),Q.treeCost=new Float32Array(r),Q.treeOffsets=new Uint32Array(a),Q.treeEdges=new Uint32Array(p),Z=new Float32Array(m);let h=0;for(let e of Z)Number.isFinite(e)&&h++;let g=new Uint32Array(u)[0];e.setReadout(`reached`,`${O(h)} of ${O(i)}`),e.setReadout(`solver`,`${g?`converged`:`NOT converged`} in ${new Uint32Array(d)[0]} of ${B} rounds`+(new Uint32Array(f)[0]?`, ${new Uint32Array(f)[0]} unresolved`:``));let _=new Uint32Array(o)[0],v=new Uint32Array(c)[0];e.setReadout(`routeSize`,`${O(_)} nodes / ${O(v)} edges`+(new Uint32Array(s)[0]||new Uint32Array(l)[0]?` (overflow)`:``)),Kt()}),Bt=new S(o,`routing-free-flow`,[{buffer:L,size:i*4}],e=>{G||(Dt=new Float32Array(e),Kt())}),Vt=[R*4,R*4,(R+1)*4,z*4,4,4,4,4],Ht=new S(o,`routing-turns`,[{buffer:St,size:Vt[0]},{buffer:xt,size:Vt[1]},{buffer:bt,size:Vt[2]},{buffer:_t,size:Vt[3]},{buffer:ut,size:4},{buffer:dt,size:4},{buffer:mt,size:4},{buffer:ht,size:4}],t=>{if(G||!e.options.showTurnRoute)return;let[n,r,i,a,o,s,c]=ie(t,Vt);Q.turnFound=new Uint32Array(n),Q.turnCost=new Float32Array(r),Q.turnOffsets=new Uint32Array(i),Q.turnEdges=new Uint32Array(a),e.setReadout(`arcs`,`${O(new Uint32Array(o)[0])} turns of ${O(E)}`+(new Uint32Array(s)[0]?` (overflow)`:``)+(new Uint32Array(c)[0]?``:` (not converged)`)),Kt()}),Ut=[4,4,4,4,i*4],Wt=new S(o,`routing-hood`,[{buffer:nt,size:4},{buffer:rt,size:4},{buffer:at,size:4},{buffer:ot,size:4},{buffer:$e,size:Ut[4]}],t=>{if(G)return;let[n,r,i,a,o]=ie(t,Ut);Ot=new Uint32Array(o),e.setReadout(`hood`,`${O(new Uint32Array(n)[0])} intersections / ${O(new Uint32Array(i)[0])} street segments`+(new Uint32Array(r)[0]||new Uint32Array(a)[0]?` (capacity reached)`:``))}),Gt=(t,n,i)=>{let o=0,s=0,c=0,l=e.options.straightAngle;for(let e=n+1;e<i&&e<t.length;e++){let n=t[e-1],i=t[e];if(n>=a||i>=a||n===4294967295||i===4294967295)continue;let u=ae(r,n,i);Math.abs(u)>=2.9?c++:u>l?o++:u<-l&&s++}return{left:o,right:s,uTurns:c}};function Kt(){let t=Math.max(1,e.options.destinationCount),n=[];for(let e=0;e<t;e++)n.push(Q.treeFound[e]?k(Q.treeCost[e]):`not found`);if(e.setReadout(`routeTimes`,n.join(` / `)),e.setChart(`routeComparison`,{kind:`bars`,values:Array.from(Q.treeCost.slice(0,t),e=>e/60),labels:Array.from({length:t},(t,n)=>{let r=n===0?e.options.destinationPlace:Ie[n-1];return`${n+1}. ${Fe[r]?.label??`Destination`}`}),yLabel:`free-flow minutes`,highlight:[0]}),Z){let t=[10,20,30,45,60,90];e.setChart(`reachCurve`,{kind:`line`,series:[{label:`intersections reached`,x:t,y:t.map(e=>Z.reduce((t,n)=>t+Number(n<=e*60),0)),color:0,points:!0}],xLabel:`free-flow minutes`,yLabel:`cumulative intersections`})}let i=Dt?.[U[0]]??NaN,o=Z?.[U[0]]??NaN;if(Number.isFinite(i))if(e.setReadout(`freeFlowRoute`,k(i)),Number.isFinite(o)){let t=(o-i)/60;e.setReadout(`scenarioDelta`,`${t>=0?`+`:``}${t.toFixed(1)} min versus free flow`),e.setChart(`scenarioComparison`,{kind:`bars`,values:[i/60,o/60],labels:[`Free flow`,`Scenario`],colors:[Ke.colors[1],Ke.colors[4]],yLabel:`minutes`,description:`Route 1 travel time under retained free-flow and current scenario costs.`})}else e.setReadout(`scenarioDelta`,`scenario route not reached`),e.setChart(`scenarioComparison`,null);else e.setReadout(`freeFlowRoute`,`not reached within the search budget`),e.setReadout(`scenarioDelta`,`-`),e.setChart(`scenarioComparison`,null);let s=Q.treeFound[0]===1;e.setReadout(`routeTime`,s?k(Q.treeCost[0]):`not found (beyond the cost limit?)`);let c=Q.treeOffsets[0],l=Math.min(Q.treeOffsets[1],c+z),u={left:0,right:0,uTurns:0};if(s){u=Gt(Q.treeEdges,c,l);let t=0;for(let e=c;e<l;e++){let n=Q.treeEdges[e];n<a&&(t+=r.length[n])}e.setReadout(`routeLength`,`${(t/1e3).toFixed(1)} km`),e.setReadout(`plainTurns`,`${u.left} left, ${u.right} right, ${u.uTurns} U`)}else e.setReadout(`routeLength`,`-`),e.setReadout(`plainTurns`,`-`);if(e.options.showTurnRoute){let t=Q.turnFound[0]===1;if(e.setReadout(`turnTime`,t?k(Q.turnCost[0]):`not found`),t){let t=Q.turnOffsets[0],n=Math.min(Q.turnOffsets[1],t+z),i=Gt(Q.turnEdges,t,n);e.setReadout(`turnTurns`,`${i.left} left, ${i.right} right, ${i.uTurns} U`),e.setChart(`turnComparison`,{kind:`bars`,values:[u.left,u.right,i.left,i.right],labels:[`Plain left`,`Plain right`,`Turn-aware left`,`Turn-aware right`],colors:[Ve,Ve,He,He],yLabel:`turns`});let o=0,c=0;for(let e=t;e<n;e++){let t=Q.turnEdges[e];t<a&&(o+=W[t],c+=r.length[t])}if(s){let t=o-Q.treeCost[0];e.setReadout(`turnCost`,`${t>=0?`+`:``}${(t/60).toFixed(1)} min, ${(c/1e3).toFixed(1)} km`)}}else e.setReadout(`turnTurns`,`-`),e.setReadout(`turnCost`,`-`)}else{for(let t of[`turnTime`,`turnTurns`,`turnCost`])e.setReadout(t,`off`);e.setChart(`turnComparison`,null)}}e.setReadout(`nodes`,O(i)),e.setReadout(`edges`,O(a)),e.setReadout(`bans`,`${O(j)} left turns between primary or secondary roads`),Ft(await Pt(Number(e.options.localIterations))),At(),kt(),Nt(),jt(),Mt();let qt=()=>{q=Math.max(q,1),J=Math.max(J,1),Y=Math.max(Y,1),X=Math.max(X,1)};return{getCompiledGraphs:()=>K?[K.tree.compiled,K.hood,K.turns]:[],setOption(t,n,r){switch(t){case`originPlace`:H=wt(r.originPlace),jt(),qt(),e.requestLayers();break;case`destinationPlace`:Tt(),jt(),qt(),e.requestLayers();break;case`destinationCount`:case`costLimitMinutes`:case`hops`:Nt(),qt(),e.requestLayers();break;case`closeExpressways`:case`expresswaySlowdown`:case`intersectionDelay`:kt(),jt(),qt();break;case`angleCost`:case`leftTurnCost`:case`rightTurnCost`:case`uTurns`:case`straightAngle`:case`banLeftTurns`:Mt(),X=Math.max(X,1);break;case`showTurnRoute`:X=Math.max(X,1),Kt(),e.requestLayers();break;case`localIterations`:Lt.request({build:()=>Pt(Number(r.localIterations)),commit:t=>{Ft(t),e.requestLayers()},discard:It});break;default:e.requestLayers()}},onClick(e){if(!e.coordinate)return!1;let[t,n]=r.projection.project(e.coordinate[0],e.coordinate[1]),i=r.findNearestNode(t,n);return Math.hypot(r.nodePositions[H*2]-t,r.nodePositions[H*2+1]-n)<400?H=i:U[0]=i,jt(),qt(),!0},getTooltip(t){if(!t.coordinate||!Z)return null;let[n,i]=r.projection.project(t.coordinate[0],t.coordinate[1]),a=r.findNearestNode(n,i);if(Math.hypot(r.nodePositions[a*2]-n,r.nodePositions[a*2+1]-i)>300)return null;let o=[`Drive time from origin: ${k(Z[a])}`];return Ot&&e.options.base===`hops`&&o.push(Ot[a]===4294967295?`Outside the ego network`:`${Ot[a]} intersections from origin`),o.join(`
`)},onThemeChange(){e.requestLayers()},encode(t){if(!K)return;let n=e.options;J>0&&(q>0&&(K.freeFlow.encode(t,{parameters:void 0}),q--,Bt.markStale()),K.tree.compiled.encode(t,{parameters:void 0}),J--,zt.markStale()),q>0&&(K.freeFlow.encode(t,{parameters:void 0}),q--,Bt.markStale()),Y>0&&(K.hood.encode(t,{parameters:void 0}),Y--,Wt.markStale()),X>0&&n.showTurnRoute&&(K.turns.encode(t,{parameters:void 0}),X--,Ht.markStale()),zt.flush(t),Bt.flush(t),Wt.flush(t),n.showTurnRoute&&Ht.flush(t)},getLayers(){let t=e.options,n=oe(e.theme()),i=[r.origin[0],r.origin[1],0],a=[new x({id:`routing-roads`,coordinateOrigin:i,segments:m,instanceCount:s,widthPixels:1,color:n.minor}),new x({id:`routing-major-roads`,coordinateOrigin:i,segments:h,instanceCount:r.majorSegments.length/4,widthPixels:1.6,color:n.major})];t.base===`time`?a.push(new x({id:`routing-cost-tint`,coordinateOrigin:i,segments:m,instanceCount:s,widthPixels:2.4,values:ve,valueFormat:`float32`,valueIndices:_,colormap:`uniform`,classTable:Ge,noDataColor:Ue})):t.base===`hops`?a.push(new x({id:`routing-hop-tint`,coordinateOrigin:i,segments:m,instanceCount:s,widthPixels:2.8,values:$e,valueFormat:`uint32`,valueIndices:_,colormap:`ylgnbu`,valueRange:[0,Math.max(1,t.hops)],color:[255,255,255,240],noDataColor:[0,0,0,0]})):t.base===`delta`&&(a.push(new x({id:`routing-scenario-delta`,coordinateOrigin:i,segments:m,instanceCount:s,widthPixels:2.8,values:Xe,valueFormat:`float32`,valueIndices:_,colormap:`uniform`,classTable:Ke,noDataColor:[0,0,0,0]})),a.push(new x({id:`routing-scenario-unreachable`,coordinateOrigin:i,segments:m,instanceCount:s,widthPixels:2.1,values:Ze,valueFormat:`uint32`,valueIndices:_,colormap:`mask`,color:[112,55,104,230],dashArray:[4,3],noDataColor:[0,0,0,0]})));let o=(e,t,n,r,a)=>new x({id:e,coordinateOrigin:i,segments:m,instanceCount:s,widthPixels:r,values:t,valueFormat:`uint32`,valueIndices:g,colormap:`mask`,color:n,dashArray:a,noDataColor:[0,0,0,0]});return t.showRoute&&a.push(o(`routing-comparison-routes`,Pe,[196,151,109,210],2.2),o(`routing-route-halo`,Qe,n.halo,8),o(`routing-route`,Qe,Ve,5)),t.showTurnRoute&&a.push(o(`routing-turn-route`,Ct,He,3.5,[6,4])),a.push(new b({id:`routing-destination-halo`,coordinateOrigin:i,positions:ge,instanceCount:1+Math.max(1,t.destinationCount),radiusPixels:10,color:n.halo}),new b({id:`routing-destinations`,coordinateOrigin:i,positions:ge,instanceCount:1+Math.max(1,t.destinationCount),radiusPixels:7,values:P,valueFormat:`uint32`,colormap:`category`,palette:[[255,255,255,255],...Array.from({length:R},()=>We)],shape:`ring`})),a},destroy(){G=!0,zt.stop(),Bt.stop(),Wt.stop(),Ht.stop(),o.destroy()}}}export{qe as createRouting};