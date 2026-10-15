import{E as e,d as t}from"./analysis-frame-Blu0tYdw.js";import{R as n,_ as r,d as i,h as a,w as o,y as s}from"./resources-CwvhEOBe.js";import{t as c}from"./gpu-group-aggregation-DrNKSCi8.js";import{l,n as u,s as d}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as f}from"./wgsl-common-BNRVzgd8.js";import{t as p}from"./gpu-cell-aggregation-BFqjxJvj.js";import{t as m}from"./gpu-point-to-cell-DabDU1o1.js";import{t as h}from"./gpu-cell-set-outline-B1MxUkL7.js";import{n as g,t as _}from"./gpu-isoband-rings-DJPyQm8X.js";import{c as v,s as y}from"./index-Bjk18V2I.js";import{i as b,n as x}from"./network-topology-BXzNuj-N.js";import{_ as S,a as C,c as w,d as T,f as E,g as ee,h as D,l as te,m as ne,n as O,o as k,p as re,r as A,s as ie,t as j,u as ae,v as M,y as N}from"./gpu-network-reachability-ROiXwwot.js";var P=`GPUNetworkServiceAreas`,F=`
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`;function I(e,t){let{state:n,phase:r}=t,i=[{name:`labels`,view:t.labels,type:`u32`,access:`read_write`},{name:`tightBits`,view:t.tightBits,type:`u32`,access:`read_write`},{name:`queuedRound`,view:n.queuedRound,type:`u32`,access:`read_write`},{name:`control`,view:n.control,type:`u32`,access:`read_write`},{name:`dispatchEven`,view:n.dispatchEven,type:`u32`,access:`read_write`},{name:`dispatchOdd`,view:n.dispatchOdd,type:`u32`,access:`read_write`}];t.activeIterations&&i.push({name:`activeIterations`,view:t.activeIterations,type:`u32`,access:`read`});let a=t.activeIterations?`min(activeIterations[activeIterationsOffset], ${t.maxIterations}u)`:`${t.maxIterations}u`;return d(e,{id:t.id,operation:P,variant:`label-initialize`,bindings:i,invocationCount:Math.max(t.nodeCount,t.tightBits.length,M(r)),declarations:`${F}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const TIGHT_WORD_COUNT: u32 = ${t.tightBits.length}u;`,body:`${N(r,{limit:a})}
  if (index < NODE_COUNT) {
    labels[labelsOffset + index] = NONE;
    // Stamps never need clearing within one encoding, but a previous encoding left them dirty.
    queuedRound[queuedRoundOffset + index] = 0u;
  }
  if (index < TIGHT_WORD_COUNT) { tightBits[tightBitsOffset + index] = 0u; }`})}function L(e,t){let n=[{name:`facilities`,view:t.facilities,type:`u32`,access:`read`}];return t.facilityCosts&&n.push({name:`facilityCosts`,view:t.facilityCosts,type:`f32`,access:`read`}),typeof t.facilityCount==`object`&&n.push({name:`facilityCount`,view:t.facilityCount,type:`u32`,access:`read`}),n.push({name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`labels`,view:t.labels,type:`atomic<u32>`,access:`read_write`}),d(e,{id:t.id,operation:P,variant:`label-seed`,bindings:n,invocationCount:t.facilities.length,declarations:`${F}
const NODE_COUNT: u32 = ${t.nodeCount}u;`,body:`${typeof t.facilityCount==`number`?`if (index >= ${t.facilityCount}u) { return; }`:t.facilityCount?`if (index >= facilityCount[facilityCountOffset]) { return; }`:``}
  let node = facilities[facilitiesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${t.facilityCosts?`facilityCosts[facilityCostsOffset + index]`:`0.0`};
  if (!(cost >= 0.0)) { return; }
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  if (cost != costs[costsOffset + node]) { return; }
  atomicMin(&labels[labelsOffset + node], index);`})}function R(e,t){let{csr:n}=t;return d(e,{id:t.id,operation:P,variant:`label-tight-edges`,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`weights`,view:n.weights,type:`f32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`tightBits`,view:t.tightBits,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:`${F}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${n.neighbors.length}u;`,body:`let sourceCost = costs[costsOffset + index];
  if (bitcast<u32>(sourceCost) >= INFINITY_BITS) { return; }
  let firstEdge = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    let weight = weights[weightsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT || !(weight >= 0.0)) { continue; }
    let targetCost = costs[costsOffset + targetNode];
    if (bitcast<u32>(targetCost) >= INFINITY_BITS) { continue; }
    if (sourceCost + weight != targetCost) { continue; }
    atomicOr(&tightBits[tightBitsOffset + (edgeIndex >> 5u)], 1u << (edgeIndex & 31u));
  }`})}function z(e,t){return d(e,{id:t.id,operation:P,variant:`label-frontier-seed`,bindings:[{name:`labels`,view:t.labels,type:`u32`,access:`read`},...re(e,t.state,t.phase,0)],invocationCount:t.nodeCount,declarations:`${F}
${S(t.phase,t.nodeCount,e.device.limits.maxComputeWorkgroupsPerDimension)}`,body:`if (labels[labelsOffset + index] != NONE) {
    frontierPush(index);
  }`})}function B(e,t){let{csr:n}=t;return T(e,{id:t.id,operation:P,variant:`label-round`,record:{loop:{id:t.id.replace(/-\d+$/,``),iteration:t.round,phase:`gated`,maximumIterations:t.phase.maxRounds},datum:{keySpace:`node`},formula:`gpu-network.service-areas.label`,attrs:{chaotic:!0}},state:t.state,phase:t.phase,round:t.round,localIterations:t.localIterations,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`tightBits`,view:t.tightBits,type:`u32`,access:`read`},{name:`labels`,view:t.labels,type:`atomic<u32>`,access:`read_write`}],declarations:`${F}
const EDGE_COUNT: u32 = ${n.neighbors.length}u;`,visitSource:`fn visit(node: u32) {
  let sourceLabel = atomicLoad(&labels[labelsOffset + node]);
  if (sourceLabel == NONE) { return; }
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (((tightBits[tightBitsOffset + (edgeIndex >> 5u)] >> (edgeIndex & 31u)) & 1u) == 0u) { continue; }
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    if (sourceLabel < atomicLoad(&labels[labelsOffset + targetNode])) {
      if (atomicMin(&labels[labelsOffset + targetNode], sourceLabel) > sourceLabel) {
        push(targetNode);
      }
    }
  }
}`})}function V(e,t){return d(e,{id:t.id,operation:P,variant:`finalize`,bindings:[{name:`control`,view:t.state.control,type:`u32`,access:`read`},{name:`reachabilityConverged`,view:t.reachabilityConverged,type:`u32`,access:`read`},{name:`converged`,view:t.converged,type:`u32`,access:`read_write`}],invocationCount:1,body:`${ne(t.phase)}
  converged[convergedOffset] = select(
    0u,
    1u,
    reachabilityConverged[reachabilityConvergedOffset] != 0u && frontierConverged != 0u${t.tiePhase?` &&
      control[controlOffset + ${t.tiePhase.controlWordOffset+0}u] == 0u`:``}
  );`})}var H=64,U=32,W=4,G=class{id;props;maximumIterations;maximumLabelIterations;localIterations;maximumTieIterations;constructor(e){this.id=e.id??`network-service-areas`,this.props=e,this.maximumIterations=e.maximumIterations??H,this.maximumLabelIterations=e.maximumLabelIterations??Math.min(this.maximumIterations,U),this.localIterations=e.localIterations??16,this.maximumTieIterations=e.maximumTieIterations??W;let{id:t}=this,{output:n}=e,{nodeCount:r}=b(t,e.topology,{weights:!0});for(let[r,i]of[[`facilities`,e.facilities],[`activeIterations`,e.activeIterations],[`output.assignments`,n.assignments],[`output.predecessors`,n.predecessors],[`output.facilityNodeCounts`,n.facilityNodeCounts],[`output.status.converged`,n.status?.converged]])i&&s(i,[`uint32`],`${t} ${r}`);typeof e.facilityCount==`object`&&s(e.facilityCount,[`uint32`],`${t} facilityCount`);for(let[r,i]of[[`facilityCosts`,e.facilityCosts],[`costLimit`,e.costLimit],[`output.costs`,n.costs],[`output.facilityCostSums`,n.facilityCostSums]])i&&s(i,[`float32`],`${t} ${r}`);let a=n.assignments.length;if(a<1)throw Error(`${t} output.assignments must contain at least one node`);if(r!==a)throw Error(`${t} topology must contain one node per output.assignments row`);if(e.facilityCosts&&e.facilityCosts.length!==e.facilities.length)throw Error(`${t} facilityCosts length must equal facilities length`);typeof e.facilityCount==`number`&&i(t,`facilityCount`,e.facilityCount,0,e.facilities.length);for(let[r,i]of[[`output.facilityNodeCounts`,n.facilityNodeCounts],[`output.facilityCostSums`,n.facilityCostSums]])if(i&&i.length!==e.facilities.length)throw Error(`${t} ${r} length must equal facilities length`);for(let[r,i]of[[`facilityCount`,typeof e.facilityCount==`object`?e.facilityCount:void 0],[`activeIterations`,e.activeIterations],[`costLimit`,e.costLimit],[`output.status.converged`,n.status?.converged]])if(i&&i.length!==1)throw Error(`${t} ${r} must contain exactly one row`);for(let[e,r]of[[`output.costs`,n.costs],[`output.predecessors`,n.predecessors]])if(r&&r.length!==a)throw Error(`${t} ${e} length must equal the node count`);i(t,`maximumIterations`,this.maximumIterations,1,O),i(t,`maximumLabelIterations`,this.maximumLabelIterations,1,O),i(t,`localIterations`,this.localIterations,1,64),i(t,`maximumTieIterations`,this.maximumTieIterations,1,A);let o=[n.assignments,n.costs,n.predecessors,n.facilityNodeCounts,n.facilityCostSums,n.status?.converged].filter(e=>e!==void 0).map(e=>e.buffer),c=[...x(e.topology),e.facilities,e.facilityCosts,typeof e.facilityCount==`object`?e.facilityCount:void 0,e.activeIterations,e.costLimit].filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(o).size!==o.length||o.some(e=>c.includes(e)))throw Error(`${t} outputs must use separate buffers from each other and from inputs`)}getCommandNodes(e){let{id:t,props:n,maximumIterations:i,localIterations:a}=this,{output:s}=n,{offsets:l,neighbors:u,weights:d}=n.topology,f=this.maximumLabelIterations;r(t,e,K(n));let p=s.assignments.length,m=s.costs??o(e,`${t}-costs`,`float32`,p),h=s.status?.converged?o(e,`${t}-reachability-converged`,`uint32`,1):void 0,g=s.assignments,_={maxRounds:f,stampBase:0,controlWordOffset:0,dispatchSlotOffset:0},v=D(f),y=D(this.maximumTieIterations),b=s.predecessors?{maxRounds:this.maximumTieIterations,stampBase:ee(_),controlWordOffset:v.controlWordCount,dispatchSlotOffset:v.dispatchSlotCount}:void 0,x=E(e,`${t}-label`,{nodeCount:p,controlWordCount:v.controlWordCount+(b?y.controlWordCount:0),dispatchSlotCount:v.dispatchSlotCount+(b?y.dispatchSlotCount:0)}),S=o(e,`${t}-label-tight-bits`,`uint32`,Math.max(1,Math.ceil(u.length/32))),C={offsets:l,neighbors:u,weights:d},w=[...new j({id:`${t}-reachability`,topology:n.topology,startNodes:n.facilities,startCosts:n.facilityCosts,startCount:n.facilityCount,costLimit:n.costLimit,maximumIterations:i,localIterations:a,activeIterations:n.activeIterations,output:{costs:m,status:{converged:h}}}).getCommandNodes(e),I(e,{id:`${t}-label-initialize`,nodeCount:p,maxIterations:f,state:x,phase:_,labels:g,tightBits:S,activeIterations:n.activeIterations})];n.facilities.length>0&&w.push(L(e,{id:`${t}-label-seed`,nodeCount:p,facilities:n.facilities,facilityCosts:n.facilityCosts,facilityCount:n.facilityCount,costs:m,labels:g})),w.push(R(e,{id:`${t}-label-tight-edges`,nodeCount:p,csr:C,costs:m,tightBits:S}),z(e,{id:`${t}-label-frontier-seed`,nodeCount:p,state:x,phase:_,labels:g}));for(let n=0;n<f;n++)w.push(B(e,{id:`${t}-label-round-${n}`,nodeCount:p,state:x,phase:_,round:n,localIterations:a,csr:C,tightBits:S,labels:g}));return s.predecessors&&b&&w.push(...this.getPredecessorNodes(e,{state:x,tiePhase:b,costs:m,predecessors:s.predecessors})),s.status?.converged&&h&&w.push(V(e,{id:`${t}-finalize`,state:x,phase:_,tiePhase:b,reachabilityConverged:h,converged:s.status.converged})),s.facilityNodeCounts&&n.facilities.length>0&&w.push(...new c({id:`${t}-facility-node-counts`,keys:s.assignments,output:s.facilityNodeCounts}).getCommandNodes(e)),s.facilityCostSums&&n.facilities.length>0&&w.push(...new c({id:`${t}-facility-cost-sums`,keys:s.assignments,values:m,output:s.facilityCostSums,operation:`sum`}).getCommandNodes(e)),w}getPredecessorNodes(e,t){let{id:n,props:r,localIterations:i}=this,{state:a,tiePhase:s,costs:c,predecessors:l}=t,{offsets:d,neighbors:f,weights:p}=r.topology,m=r.output.assignments.length,h=f.length,g=r.output.assignments,_={offsets:d,neighbors:f,weights:p},v=o(e,`${n}-tie-levels`,`uint32`,m),y=o(e,`${n}-tie-bits`,`uint32`,Math.max(1,Math.ceil(h/32))),b=[u(e,{id:`${n}-clear-predecessors`,operation:`GPUNetworkServiceAreas`,view:l,type:`u32`,value:`0xffffffffu`}),k(e,{id:`${n}-tie-initialize`,nodeCount:m,state:a,tiePhase:s,levels:v,tieBits:y}),C(e,{id:`${n}-predecessors`,nodeCount:m,csr:_,costs:c,predecessors:l,tieBits:y,labels:g})];r.facilities.length>0&&b.push(te(e,{id:`${n}-tie-roots`,nodeCount:m,state:a,phase:s,sources:r.facilities,sourceCosts:r.facilityCosts,sourceCount:r.facilityCount,hasCostLimit:!1,costs:c,levels:v,labels:g})),b.push(ae(e,{id:`${n}-tie-seed`,nodeCount:m,edgeCount:h,state:a,tiePhase:s,offsets:d,tieBits:y,predecessors:l,levels:v}));for(let t=0;t<this.maximumTieIterations;t++)b.push(ie(e,{id:`${n}-tie-level-${t}`,nodeCount:m,edgeCount:h,state:a,tiePhase:s,round:t,localIterations:i,offsets:d,neighbors:f,tieBits:y,levels:v}));return b.push(w(e,{id:`${n}-tie-predecessors`,nodeCount:m,edgeCount:h,offsets:d,neighbors:f,tieBits:y,levels:v,predecessors:l})),b}};function K(e){let{output:t}=e;return[...x(e.topology),e.facilities,e.facilityCosts,typeof e.facilityCount==`object`?e.facilityCount:void 0,e.costLimit,e.activeIterations,t.assignments,t.costs,t.predecessors,t.facilityNodeCounts,t.facilityCostSums,t.status?.converged]}function q(e,t=new Float32Array(12)){if(t.length<12)throw Error(`GPUNetworkIsochrones parameter target must hold 12 elements`);let{breakCount:n}=e,r=e.firstBand??0,i=e.lastBand??Math.max(n-1,0);for(let[e,t]of[[`breakCount`,n],[`firstBand`,r],[`lastBand`,i]])if(!Number.isSafeInteger(t)||t<0)throw Error(`GPUNetworkIsochrones ${e} must be a non-negative integer`);let[a,o,s,c]=e.extent;if(![a,o,s,c].every(Number.isFinite)||s<=a||c<=o)throw Error(`GPUNetworkIsochrones extent must be finite with maxX > minX and maxY > minY`);let l=e.bufferRadius??0,u=e.walkCostPerUnit??0;if(!(l>=0)||!Number.isFinite(l))throw Error(`GPUNetworkIsochrones bufferRadius must be a non-negative finite number`);if(!(u>=0)||!Number.isFinite(u))throw Error(`GPUNetworkIsochrones walkCostPerUnit must be a non-negative finite number`);return t.fill(0),t[0]=n,t[1]=a,t[2]=o,t[3]=s,t[4]=c,t[5]=l,t[6]=u,t[7]=r,t[8]=i,t[9]=Math.min(e.cellCostLimit??1/0,3e38),t}var J=`GPUNetworkIsochrones`;function oe(e){return e===`min`?4294967295:0}function se(e,t){return d(e,{id:t.id,operation:J,variant:`band-parameters`,bindings:[{name:`params`,view:t.parameters,type:`f32`,access:`read`},{name:`bandParams`,view:t.bandParameters,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const WIDTH: f32 = ${t.width}.0;
const HEIGHT: f32 = ${t.height}.0;`,body:`bandParams[bandParamsOffset] = params[paramsOffset];
  bandParams[bandParamsOffset + 1u] = params[paramsOffset + 7u];
  bandParams[bandParamsOffset + 2u] = params[paramsOffset + 8u];
  bandParams[bandParamsOffset + 3u] = 0.0;
  bandParams[bandParamsOffset + 4u] = params[paramsOffset + 1u];
  bandParams[bandParamsOffset + 5u] = params[paramsOffset + 2u];
  bandParams[bandParamsOffset + 6u] = params[paramsOffset + 3u];
  bandParams[bandParamsOffset + 7u] = params[paramsOffset + 4u];
  bandParams[bandParamsOffset + 8u] = (params[paramsOffset + 3u] - params[paramsOffset + 1u]) / WIDTH;
  bandParams[bandParamsOffset + 9u] = (params[paramsOffset + 4u] - params[paramsOffset + 2u]) / HEIGHT;
  bandParams[bandParamsOffset + 10u] = 0.0;
  bandParams[bandParamsOffset + 11u] = 0.0;`})}function ce(e,t){let n=t.mode===`min`,r=!!t.assignments;return d(e,{id:t.id,operation:J,variant:`splat-${t.mode}${r?`-labelled`:``}`,record:{datum:{keySpace:`edge`},formula:n?`gpu-network.isochrones.cost`:void 0},bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights,type:`f32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`positions`,view:t.nodePositions,type:`f32`,access:`read`},{name:`params`,view:t.parameters,type:`f32`,access:`read`},{name:`encoded`,view:t.encoded,type:`atomic<u32>`,access:`read_write`},...t.assignments?[{name:`assignments`,view:t.assignments,type:`u32`,access:`read`}]:[]],invocationCount:t.neighbors.length,declarations:`const NODE_COUNT: u32 = ${t.nodeCount}u;
const WIDTH: i32 = ${t.width};
const HEIGHT: i32 = ${t.height};
const MAXIMUM_BUFFER_PIXELS: u32 = ${t.maximumBufferPixels}u;
const MAXIMUM_SAMPLES: u32 = ${t.maximumSamplesPerEdge}u;
const COST_CEILING: f32 = ${l(3e38)};

${f}
fn isochronesIsFinite(value: f32) -> bool {
  return isFiniteBits(value) && abs(value) < COST_CEILING;
}

fn isochronesFindSource(edge: u32) -> u32 {
  var low = 0u;
  var high = NODE_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) >> 1u;
    if (offsets[offsetsOffset + middle] <= edge) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

fn isochronesGetPosition(node: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * node], positions[positionsOffset + 2u * node + 1u]);
}`,body:`if (index >= offsets[offsetsOffset + NODE_COUNT]) { return; }
  let targetNode = neighbors[neighborsOffset + index];
  if (targetNode >= NODE_COUNT) { return; }
  let weight = weights[weightsOffset + index];
  let source = isochronesFindSource(index);
  let sourceCost = costs[costsOffset + source];
  ${r?`let facility = assignments[assignmentsOffset + source];
  if (facility >= 255u) { return; }`:``}
  if (!(weight >= 0.0) || !isochronesIsFinite(weight) || !isochronesIsFinite(sourceCost) || sourceCost < 0.0) { return; }
  let startPoint = isochronesGetPosition(source);
  let endPoint = isochronesGetPosition(targetNode);
  if (!isochronesIsFinite(startPoint.x) || !isochronesIsFinite(startPoint.y) || !isochronesIsFinite(endPoint.x) || !isochronesIsFinite(endPoint.y)) { return; }
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f(
    (params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT)
  );
  let walk = max(params[paramsOffset + 6u], 0.0);
  let radius = max(max(params[paramsOffset + 5u], 0.0), 0.5 * length(cellSize));
  let reachX = i32(min(u32(ceil(radius / cellSize.x)), MAXIMUM_BUFFER_PIXELS));
  let reachY = i32(min(u32(ceil(radius / cellSize.y)), MAXIMUM_BUFFER_PIXELS));
  // Extent culling (review B-20): an edge, or a sample, farther than the splat reach from the
  // raster extent touches no pixel.
  let maximum = vec2f(params[paramsOffset + 3u], params[paramsOffset + 4u]);
  let reach = vec2f(f32(reachX), f32(reachY)) * abs(cellSize) + vec2f(radius);
  let extentLow = min(minimum, maximum) - reach;
  let extentHigh = max(minimum, maximum) + reach;
  if (any(max(startPoint, endPoint) < extentLow) || any(min(startPoint, endPoint) > extentHigh)) {
    return;
  }
  let spacing = 0.5 * radius;
  let sampleCount = clamp(u32(ceil(length(endPoint - startPoint) / spacing)) + 1u, 2u, MAXIMUM_SAMPLES);
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
    let t = f32(sampleIndex) / f32(sampleCount - 1u);
    let point = mix(startPoint, endPoint, vec2f(t));
    if (any(point < extentLow) || any(point > extentHigh)) { continue; }
    let edgeCost = sourceCost + t * weight;
    let centerPixel = vec2i(floor((point - minimum) / cellSize));
    for (var dy = -reachY; dy <= reachY; dy++) {
      let row = centerPixel.y + dy;
      if (row < 0 || row >= HEIGHT) { continue; }
      for (var dx = -reachX; dx <= reachX; dx++) {
        let column = centerPixel.x + dx;
        if (column < 0 || column >= WIDTH) { continue; }
        let pixelCenter = minimum + (vec2f(f32(column), f32(row)) + 0.5) * cellSize;
        let pixelDistance = length(pixelCenter - point);
        if (pixelDistance > radius) { continue; }
        let cost = abs(edgeCost + walk * pixelDistance);
        if (!isochronesIsFinite(cost)) { continue; }
        let slot = encodedOffset + u32(row * WIDTH + column);
        // Test-before-RMW: overlapping samples and edges rewrite most pixels with a no-op, and a
        // plain atomic load is far cheaper than a contended read-modify-write.
        ${r?`let encodedValue = (bitcast<u32>(cost) & 0xffffff00u) | facility;
        if (encodedValue < atomicLoad(&encoded[slot])) { atomicMin(&encoded[slot], encodedValue); }`:n?`let encodedValue = bitcast<u32>(cost);
        if (encodedValue < atomicLoad(&encoded[slot])) { atomicMin(&encoded[slot], encodedValue); }`:`let encodedValue = bitcast<u32>(cost) + 1u;
        if (encodedValue > atomicLoad(&encoded[slot])) { atomicMax(&encoded[slot], encodedValue); }`}
      }
    }
  }`})}function le(e,t){let n=t.mode===`min`,r=!!t.pixelFacilities;return d(e,{id:t.id,operation:J,variant:`decode-${t.mode}${r?`-labelled`:``}`,bindings:[{name:`encoded`,view:t.encoded,type:`u32`,access:`read`},{name:`values`,view:t.values,type:`f32`,access:`read_write`},...t.pixelFacilities?[{name:`facilitiesOut`,view:t.pixelFacilities,type:`u32`,access:`read_write`}]:[]],invocationCount:t.pixelCount,declarations:`const UNREACHED_COST: f32 = ${l(t.unreachedCost)};`,body:`let word = encoded[encodedOffset + index];
  ${r?`values[valuesOffset + index] = select(bitcast<f32>(word & 0xffffff00u), UNREACHED_COST, word == 0xffffffffu);
  facilitiesOut[facilitiesOutOffset + index] = select(word & 0xffu, 0xffffffffu, word == 0xffffffffu);`:n?`values[valuesOffset + index] = select(bitcast<f32>(word), UNREACHED_COST, word == 0xffffffffu);`:`values[valuesOffset + index] = select(bitcast<f32>(word - 1u), UNREACHED_COST, word == 0u);`}`})}function ue(e,t){return d(e,{id:t.id,operation:J,variant:`node-mask`,bindings:[{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`params`,view:t.parameters,type:`f32`,access:`read`},{name:`mask`,view:t.mask,type:`u32`,access:`read_write`}],invocationCount:t.costs.length,declarations:f,body:`let cost = costs[costsOffset + index];
  let isReached = isFiniteBits(cost) && abs(cost) < ${l(3e38)} && cost <= params[paramsOffset + 9u];
  mask[maskOffset + index] = select(0u, 1u, isReached);`})}function de(e,t){let n=`const NODE_COUNT: u32 = ${t.nodeCount}u;

// Row of the node's cell in the sorted table, or 0xffffffff.
fn isochronesFindCellRow(node: u32) -> u32 {
  let low = nodeCells[nodeCellsOffset + 2u * node];
  let high = nodeCells[nodeCellsOffset + 2u * node + 1u];
  if (low == 0u && high == 0u) {
    return 0xffffffffu;
  }
  var first = 0u;
  var last = tableCount[tableCountOffset];
  while (first < last) {
    let middle = (first + last) >> 1u;
    let rowLow = tableCells[tableCellsOffset + 2u * middle];
    let rowHigh = tableCells[tableCellsOffset + 2u * middle + 1u];
    if (rowHigh < high || (rowHigh == high && rowLow < low)) {
      first = middle + 1u;
    } else {
      last = middle;
    }
  }
  if (first < tableCount[tableCountOffset]
    && tableCells[tableCellsOffset + 2u * first] == low
    && tableCells[tableCellsOffset + 2u * first + 1u] == high) {
    return first;
  }
  return 0xffffffffu;
}`,r=[{name:`nodeCells`,view:t.nodeCells,type:`u32`,access:`read`},{name:`tableCells`,view:t.tableCells,type:`u32`,access:`read`},{name:`tableCount`,view:t.tableCount,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`}];return[u(e,{id:`${t.id}-minimum-fill`,operation:J,view:t.cellMinimumCosts,type:`u32`,value:`0xffffffffu`}),u(e,{id:`${t.id}-facility-fill`,operation:J,view:t.cellFacilities,type:`u32`,value:`0xffffffffu`}),d(e,{id:`${t.id}-minimum`,operation:J,variant:`cell-minimum-cost`,bindings:[...r,{name:`cellMinimum`,view:t.cellMinimumCosts,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:n,body:`let row = isochronesFindCellRow(index);
  let cost = costs[costsOffset + index];
  if (row != 0xffffffffu && cost >= 0.0) {
    atomicMin(&cellMinimum[cellMinimumOffset + row], select(bitcast<u32>(cost), 0u, cost == 0.0));
  }`}),d(e,{id:`${t.id}-facility`,operation:J,variant:`cell-facility`,bindings:[...r,{name:`assignments`,view:t.assignments,type:`u32`,access:`read`},{name:`cellMinimum`,view:t.cellMinimumCosts,type:`u32`,access:`read`},{name:`cellFacility`,view:t.cellFacilities,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:n,body:`let row = isochronesFindCellRow(index);
  let cost = costs[costsOffset + index];
  if (row != 0xffffffffu && cost >= 0.0 &&
      select(bitcast<u32>(cost), 0u, cost == 0.0) == cellMinimum[cellMinimumOffset + row]) {
    atomicMin(&cellFacility[cellFacilityOffset + row], assignments[assignmentsOffset + index]);
  }`})]}function fe(e,t){return d(e,{id:t.id,operation:J,variant:`triangle-facility`,bindings:[{name:`triangles`,view:t.triangles,type:`f32`,access:`read`},{name:`triangleCount`,view:t.triangleCount,type:`u32`,access:`read`},{name:`params`,view:t.parameters,type:`f32`,access:`read`},{name:`pixelFacilities`,view:t.pixelFacilities,type:`u32`,access:`read`},{name:`facilitiesOut`,view:t.triangleFacilities,type:`u32`,access:`read_write`}],invocationCount:t.triangleFacilities.length,declarations:`const WIDTH: i32 = ${t.width};
const HEIGHT: i32 = ${t.height};`,body:`if (index >= triangleCount[triangleCountOffset]) {
    facilitiesOut[facilitiesOutOffset + index] = 0xffffffffu;
    return;
  }
  let base = trianglesOffset + 6u * index;
  let centroid = vec2f(
    (triangles[base] + triangles[base + 2u] + triangles[base + 4u]) / 3.0,
    (triangles[base + 1u] + triangles[base + 3u] + triangles[base + 5u]) / 3.0
  );
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f(
    (params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT)
  );
  // Sample (column, row) sits at minimum + (column + 0.5, row + 0.5) * cellSize.
  let gridPosition = (centroid - minimum) / cellSize - vec2f(0.5);
  let lower = vec2i(floor(gridPosition));
  var best = 0xffffffffu;
  var bestDistance = 1e30;
  for (var corner = 0u; corner < 4u; corner++) {
    let column = lower.x + i32(corner & 1u);
    let row = lower.y + i32(corner >> 1u);
    if (column < 0 || row < 0 || column >= WIDTH || row >= HEIGHT) { continue; }
    let facility = pixelFacilities[pixelFacilitiesOffset + u32(row * WIDTH + column)];
    if (facility == 0xffffffffu) { continue; }
    let offset = gridPosition - vec2f(f32(column), f32(row));
    let distance = dot(offset, offset);
    if (distance < bestDistance || (distance == bestDistance && facility < best)) {
      bestDistance = distance;
      best = facility;
    }
  }
  facilitiesOut[facilitiesOutOffset + index] = best;`})}var pe=`network-isochrones`,Y=6,me=16,X=64,he=1024,Z=1e30,ge=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),measure:`distance`}),_e=class{id;props;nodeCount;mode;spatialContext;constructor(r){this.id=r.id??pe,this.props=r;let{id:o}=this,{raster:c,cellOutline:l}=r;if(!c&&!l)throw Error(`${o} needs raster or cellOutline`);this.spatialContext=e(o,r.spatialContext??(c?void 0:t),ge);let u=this.spatialContext.coordinateSpace;if(c&&u===`longitude-latitude`)throw Error(`${o} raster needs planar positions; project to a planar frame first`);if(l&&u===`planar`&&r.spatialContext)throw Error(`${o} cellOutline needs longitude-latitude positions`);for(let[e,t]of[[`costs`,r.costs],[`costBreaks`,r.costBreaks],[`parameters`,r.parameters],[`startNodes`,r.startNodes],[`assignments`,r.assignments]])if(t instanceof n)throw Error(`${o} ${e} must be a single packed view, not a chunked vector`);let{nodeCount:d,edgeCount:f}=b(o,r.topology,{weights:!0,nodePositions:!0});if(s(r.costs,[`float32`],`${o} costs`),s(r.costBreaks,[`float32`],`${o} costBreaks`),s(r.parameters,[`float32`],`${o} parameters`),this.nodeCount=r.costs.length,this.nodeCount<1)throw Error(`${o} costs must contain at least one node`);if(d!==this.nodeCount)throw Error(`${o} topology must contain one node per costs row`);if(f<1)throw Error(`${o} needs at least one edge`);if(r.costBreaks.length<1)throw Error(`${o} costBreaks must contain at least one row`);if(r.parameters.length<12)throw Error(`${o} parameters must hold 12 float32 rows`);if(r.startNodes)s(r.startNodes,[`uint32`],`${o} startNodes`),typeof r.startCount==`number`&&i(o,`startCount`,r.startCount,0,r.startNodes.length);else for(let e of[`startCosts`,`startCount`,`costLimit`,`output`,`localIterations`])if(r[e]!==void 0)throw Error(`${o} ${e} needs startNodes`);if(r.output?.status?.converged&&(s(r.output.status.converged,[`uint32`],`${o} output.status.converged`),r.output.status.converged.length!==1))throw Error(`${o} output.status.converged must contain exactly one row`);if(r.assignments&&(s(r.assignments,[`uint32`],`${o} assignments`),r.assignments.length!==this.nodeCount))throw Error(`${o} assignments length must equal the node count`);if(r.maximumLabelIterations!==void 0&&!(r.assignments&&r.startNodes))throw Error(`${o} maximumLabelIterations needs startNodes and assignments`);if(this.mode=c?.mode??`min`,this.mode!==`min`&&this.mode!==`max`)throw Error(`${o} raster.mode must be 'min' or 'max'`);if(c){for(let[e,t,n,r]of[[`raster.width`,c.width,2,2147483647],[`raster.height`,c.height,2,2147483647],[`raster.maximumBufferPixels`,c.maximumBufferPixels??Y,1,me],[`raster.maximumSamplesPerEdge`,c.maximumSamplesPerEdge??X,2,he]])i(o,e,t,n,r);let e=c.unreachedCost??Z;if(!Number.isFinite(e)||!(e>0)||e>=3e38)throw Error(`${o} raster.unreachedCost must be a positive finite number below 3e38`);let t=c.width*c.height;if(!Number.isSafeInteger(t)||t>2147483647)throw Error(`${o} raster width * height must fit in 31 bits`);if(c.output.pixelFacilities){if(!r.assignments)throw Error(`${o} raster.output.pixelFacilities needs assignments`);if(this.mode!==`min`)throw Error(`${o} raster.output.pixelFacilities needs raster.mode 'min'`);if(r.startNodes&&r.startNodes.length>255)throw Error(`${o} raster.output.pixelFacilities supports at most 255 startNodes`);if(s(c.output.pixelFacilities,[`uint32`],`${o} raster.output.pixelFacilities`),c.output.pixelFacilities.length<t)throw Error(`${o} raster.output.pixelFacilities must hold width * height rows`)}if(c.output.triangleFacilities){if(!c.output.pixelFacilities||!c.output.triangles)throw Error(`${o} raster.output.triangleFacilities needs pixelFacilities and the triangle output`);if(s(c.output.triangleFacilities,[`uint32`],`${o} raster.output.triangleFacilities`),c.output.triangleFacilities.length!==c.output.triangleBands.length)throw Error(`${o} raster.output.triangleFacilities length must equal triangleBands length`)}if(c.rings&&i(o,`raster.rings.edgeCapacity`,c.rings.edgeCapacity),c.output.values&&(s(c.output.values,[`float32`],`${o} raster.output.values`),c.output.values.length<t))throw Error(`${o} raster.output.values must hold width * height rows`)}if(l){if(l.family!==`h3`&&l.family!==`quadbin`)throw Error(`${o} cellOutline.family must be 'h3' or 'quadbin'`);if(l.byFacility&&!r.assignments)throw Error(`${o} cellOutline.byFacility needs assignments`);if(l.cellFacilities&&!l.byFacility)throw Error(`${o} cellOutline.cellFacilities needs cellOutline.byFacility`);if(l.byFacility&&!l.output.groups)throw Error(`${o} cellOutline.byFacility needs cellOutline.output.groups`);l.tableCapacity!==void 0&&i(o,`cellOutline.tableCapacity`,l.tableCapacity)}let p=[...x(r.topology),r.costBreaks,r.parameters,r.startNodes,r.startCosts,typeof r.startCount==`object`?r.startCount:void 0,r.costLimit,...r.startNodes?[]:[r.costs,r.assignments]];a(o,[...r.startNodes?[r.costs,r.assignments]:[],l?.cellFacilities,...Q(c?.output),...Q(l?.output),...$(l?.rings?.output),...$(c?.rings?.output),...Q(l?.table)],p)}getCommandNodes(e){let{id:t,props:n,nodeCount:i,mode:a}=this,{raster:s,cellOutline:c}=n,{offsets:l,neighbors:d,weights:f,nodePositions:v}=n.topology;r(t,e,[...x(n.topology),n.costs,n.startNodes,n.startCosts,typeof n.startCount==`object`?n.startCount:void 0,n.costLimit,n.output?.status?.converged,n.costBreaks,n.parameters,n.assignments,c?.cellFacilities,...Q(s?.output),...Q(c?.output),...$(c?.rings?.output),...$(s?.rings?.output),...Q(c?.table)]);let y=[];if(n.startNodes&&n.assignments?y.push(...new G({id:`${t}-search`,topology:n.topology,facilities:n.startNodes,facilityCosts:n.startCosts,facilityCount:n.startCount,costLimit:n.costLimit,maximumIterations:n.maximumIterations,maximumLabelIterations:n.maximumLabelIterations,localIterations:n.localIterations,output:{assignments:n.assignments,costs:n.costs,status:{converged:n.output?.status?.converged}}}).getCommandNodes(e)):n.startNodes&&y.push(...new j({id:`${t}-search`,topology:n.topology,startNodes:n.startNodes,startCosts:n.startCosts,startCount:n.startCount,costLimit:n.costLimit,maximumIterations:n.maximumIterations,localIterations:n.localIterations,output:{costs:n.costs,status:{converged:n.output?.status?.converged}}}).getCommandNodes(e)),s){let{width:r,height:c}=s,p=r*c,m={width:r,height:c,affine:[1,0,0,0,1,0],pixelInterpretation:`point`},h=o(e,`${t}-encoded`,`uint32`,p),b=s.output.values??o(e,`${t}-values`,`float32`,p),x=o(e,`${t}-band-parameters`,`float32`,12);y.push(se(e,{id:`${t}-band-parameters`,width:r,height:c,parameters:n.parameters,bandParameters:x}),u(e,{id:`${t}-encoded-fill`,operation:`GPUNetworkIsochrones`,view:h,type:`u32`,value:`${oe(a)}u`}),ce(e,{id:`${t}-splat`,nodeCount:i,width:r,height:c,mode:a,maximumBufferPixels:s.maximumBufferPixels??Y,maximumSamplesPerEdge:s.maximumSamplesPerEdge??X,offsets:l,neighbors:d,weights:f,costs:n.costs,nodePositions:v,parameters:n.parameters,encoded:h,assignments:s.output.pixelFacilities?n.assignments:void 0}),le(e,{id:`${t}-decode`,mode:a,unreachedCost:s.unreachedCost??Z,encoded:h,values:b,pixelCount:p,pixelFacilities:s.output.pixelFacilities}),...new g({id:`${t}-isobands`,grid:m,values:b,breaks:n.costBreaks,parameters:x,output:{bandClasses:s.output.bandClasses,triangles:s.output.triangles,triangleBands:s.output.triangleBands,edges:s.output.edges,edgeBands:s.output.edgeBands,status:s.output.status}}).getCommandNodes(e)),s.output.triangleFacilities&&y.push(fe(e,{id:`${t}-triangle-facilities`,width:r,height:c,triangles:s.output.triangles,triangleCount:s.output.status.count,parameters:n.parameters,pixelFacilities:s.output.pixelFacilities,triangleFacilities:s.output.triangleFacilities})),s.rings&&y.push(...new _({id:`${t}-rings`,grid:m,values:b,breaks:n.costBreaks,parameters:x,edgeCapacity:s.rings.edgeCapacity,vertexTolerance:s.rings.vertexTolerance,output:s.rings.output}).getCommandNodes(e))}if(c){let r=o(e,`${t}-reached-mask`,`uint32`,i);y.push(ue(e,{id:`${t}-reached-mask`,costs:n.costs,parameters:n.parameters,mask:r}));let a=c.tableCapacity??i,s,l=c.table??{cells:o(e,`${t}-reached-cells`,`uint32x2`,a),counts:o(e,`${t}-reached-counts`,`uint32`,a),status:{count:o(e,`${t}-reached-count`,`uint32`,1),overflow:o(e,`${t}-reached-overflow`,`uint32`,1)}},u=!!c.byFacility;if(c.family===`quadbin`&&!u)y.push(...new p({id:`${t}-reached-table`,family:`quadbin`,resolution:c.resolution,positions:v,mask:r,output:l}).getCommandNodes(e));else{let d=o(e,`${t}-node-cells`,`uint32x2`,i);y.push(...new m({id:`${t}-node-cells`,family:c.family,resolution:c.resolution,positions:v,mask:r,output:{cells:d}}).getCommandNodes(e),...new p({id:`${t}-reached-table`,family:c.family,resolution:c.resolution,cells:d,output:l}).getCommandNodes(e)),u&&(s=c.cellFacilities??o(e,`${t}-cell-facilities`,`uint32`,a),y.push(...de(e,{id:`${t}-cell-facilities`,nodeCount:i,nodeCells:d,tableCells:l.cells,tableCount:l.status.count,costs:n.costs,assignments:n.assignments,cellMinimumCosts:o(e,`${t}-cell-minimum-costs`,`uint32`,a),cellFacilities:s})))}y.push(...new h({id:`${t}-outline`,family:c.family,cells:l.cells,count:l.status.count,groups:s,output:c.output,rings:c.rings}).getCommandNodes(e))}return y}};function Q(e){return e?Object.values(e).flatMap(e=>!e||typeof e!=`object`?[]:`buffer`in e?[e]:Q(e)):[]}function $(e){return Q(e)}var ve=`@group(0) @binding(auto) var<storage, read> rasterBounds`,ye=class extends y{static layerName=`IsobandTriangleLayer`;constructor(e){super(e)}getShaderSource(){let e=super.getShaderSource(),t=e.indexOf(ve);if(t<0)throw Error(`IsobandTriangleLayer could not find the raster binding block`);return`${e.slice(0,t)}
@group(0) @binding(auto) var<storage, read> bandTriangles: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> bandIds: array<u32>;

struct BandVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> BandVertexOutput {
  var output: BandVertexOutput;
  let band = bandIds[vertexIndex / 3u];
  let breakCount = max(styleExtent[0], 0.0);
  let paletteSize = max(arrayLength(&styleValues), 1u);
  let t = clamp((f32(band) + 0.5) / (breakCount + 1.0), 0.0, 0.9999);
  let packedColor = styleValues[u32(t * f32(paletteSize))];
  output.color = vec4<f32>(
    f32(packedColor & 255u),
    f32((packedColor >> 8u) & 255u),
    f32((packedColor >> 16u) & 255u),
    f32(packedColor >> 24u)
  ) / 255.0;
  output.position = projectSpatialAnalysisPosition(getSpatialAnalysisPosition(bandTriangles[vertexIndex]));
  return output;
}

@fragment fn fragmentMain(input: BandVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity);
}
`}getBindings(e){let{triangles:t,triangleBands:n}=this.props;return{...this.getStyleBindings(e),bandTriangles:t,bandIds:n}}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;if(!t||!n)return;this.writeLayerStyle(n);let{drawCommands:r,drawCommandIndex:i=0}=this.props;r&&(t.setInstanceCount(0),t.draw(e),r.draw(e,i))}},be=class extends v{static layerName=`PolylineLayer`;constructor(e){super(e)}getShaderSource(){let e=super.getShaderSource().replace(`var<storage, read> segmentPositions: array<vec4<f32>>`,`var<storage, read> segmentPositions: array<vec2<f32>>`);if(e=e.replace(/let segment = segmentPositions\[row\];\s*var color = getSpatialAnalysisColor\(getSpatialAnalysisValueRow\(row\)\);/,`let polylineCount = bitcast<u32>(styleExtent[0]);
  var segment = vec4<f32>(0.0);
  var polylineValid = false;
  var polylineLevel = 0u;
  if (polylineCount > 0u) {
    // Last polyline whose first vertex is at or before this instance.
    var low = 0u;
    var high = polylineCount;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (segmentIds[middle] <= row) {
        low = middle;
      } else {
        high = middle;
      }
    }
    if (row + 1u < segmentIds[low + 1u]) {
      segment = vec4<f32>(segmentPositions[row], segmentPositions[row + 1u]);
      polylineLevel = styleValueIndices[low];
      polylineValid = true;
    }
  }
  var color = getSpatialAnalysisColor(polylineLevel);`),e=e.replace(`segment.x != segment.x || segment.z != segment.z`,`!polylineValid`),!e.includes(`polylineValid`)||e.includes(`segment.x != segment.x`))throw Error(`PolylineLayer could not patch the segment vertex shader`);return e}getBindings(e){let{polylineOffsets:t}=this.props;return{...super.getBindings(e),segmentIds:t}}};export{G as a,q as i,be as n,_e as r,ye as t};