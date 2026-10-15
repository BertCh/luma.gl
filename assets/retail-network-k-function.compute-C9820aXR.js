import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{E as e,p as t}from"./analysis-frame-Blu0tYdw.js";import{N as n,_ as r,d as i,h as a,l as o,m as s,n as c,t as l,v as u,w as d,y as f}from"./resources-CwvhEOBe.js";import{t as p}from"./gpu-scan-BuGhTLDQ.js";import{t as m}from"./gpu-reduction-BD8l40jM.js";import{n as h,s as g}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as _}from"./wgsl-common-BNRVzgd8.js";import{t as v}from"./philox-random-CHwyk0dd.js";import{t as y}from"./permutation-random-C7SSHCbB.js";import{U as b,a as x,c as S,r as C}from"./index-Bjk18V2I.js";import{t as w}from"./graph-buffers-B4_80RlX.js";import{t as T}from"./graph-swap-41Ru7LVu.js";import{t as E}from"./summary-reader-CsgWxENf.js";import{l as D,m as O}from"./class-table-BVEo3-Vs.js";import{i as k,n as A}from"./network-topology-BXzNuj-N.js";import{t as ee}from"./network-accessibility-passes-B3hrEHbU.js";import{t as te}from"./gpu-network-snapping-D735W-3S.js";import{t as ne}from"./gpu-network-reachability-ROiXwwot.js";import{t as j}from"./network-accessibility-lanes-CM32kWYY.js";import{a as M,i as re,n as N,t as P}from"./b10-road-graph-63jDTifd.js";function F(e,t=new Uint32Array(4)){if(t.length<4)throw Error(`GPUNetworkKFunction parameter target must hold 4 elements`);let{seed:n,activeSimulations:r}=e;if(!Number.isSafeInteger(n)||n<0)throw Error(`GPUNetworkKFunction seed must be a non-negative safe integer`);if(!Number.isSafeInteger(r)||r<0)throw Error(`GPUNetworkKFunction activeSimulations must be a non-negative integer`);return t[0]=n%2**32,t[1]=Math.floor(n/2**32),t[2]=r,t[3]=0,t}var I=Object.freeze({...t,measure:`distance`}),L=`GPUNetworkKFunction`,ie=v.networkKFunction.id;function ae(e){let t=Math.floor(Math.log2((2**32-1)/Math.max(e,1)));return 2**Math.max(0,Math.min(29,t))}var R=1024;function oe(e){return o(e,4)}var se=4294967295,z=64,B=32,V=32,ce=16,le=class{id;props;eventCount;nodeCount;edgeCount;bandCount;simulationCount;rowCount;rowsPerBlock;blockCount;spatialContext;constructor(t){this.id=t.id??`network-k-function`,this.props=t;let{id:n}=this;this.spatialContext=e(n,t.spatialContext,I);let{output:r}=t,{nodeCount:o,edgeCount:s}=k(n,t.topology,{weights:!0,nodePositions:!0});for(let[e,i]of[[`parameters`,t.parameters],[`output.pairCounts`,r.pairCounts],[`output.snappedEventCount`,r.snappedEventCount],[`output.status.overflow`,r.status?.overflow],[`output.status.converged`,r.status?.converged]])i&&f(i,[`uint32`],`${n} ${e}`);for(let[e,i]of[[`maximumSnapDistance`,t.maximumSnapDistance],[`maximumDistance`,t.maximumDistance],[`networkLength`,t.networkLength],[`output.kValues`,r.kValues],[`output.envelope`,r.envelope]])i&&f(i,[`float32`],`${n} ${e}`);if(f(t.positions,[`float32x2`],`${n} positions`),this.eventCount=t.positions.length,this.nodeCount=o,this.edgeCount=s,this.bandCount=t.bandCount??B,this.simulationCount=t.simulationCount??0,this.eventCount<2)throw Error(`${n} requires at least two events`);if(this.edgeCount<1)throw Error(`${n} requires at least one edge`);if(i(n,`bandCount`,this.bandCount,2,R),i(n,`simulationCount`,this.simulationCount,0),t.parameters.length<4)throw Error(`${n} parameters must hold 4 elements`);for(let[e,i]of[[`maximumSnapDistance`,t.maximumSnapDistance],[`maximumDistance`,t.maximumDistance],[`networkLength`,t.networkLength],[`output.snappedEventCount`,r.snappedEventCount],[`output.status.overflow`,r.status?.overflow],[`output.status.converged`,r.status?.converged]])if(i&&i.length!==1)throw Error(`${n} ${e} must contain exactly one row`);let c=1+this.simulationCount;for(let[e,t,i]of[[`output.kValues`,r.kValues,c*this.bandCount],[`output.pairCounts`,r.pairCounts,c*this.bandCount],[`output.envelope`,r.envelope,3*this.bandCount]])if(t&&t.length!==i)throw Error(`${n} ${e} length must be ${i}`);if(r.status?.overflow&&t.candidateCapacity===void 0)throw Error(`${n} output.status.overflow requires candidateCapacity`);if(t.candidateCapacity!==void 0&&!r.status?.overflow)throw Error(`${n} candidateCapacity requires output.status.overflow`);this.rowCount=c*this.eventCount,this.rowsPerBlock=t.rowsPerBlock??Math.min(this.rowCount,Math.max(1,Math.min(V,j({rowCount:this.rowCount,nodeCount:this.nodeCount,edgeCount:this.edgeCount})))),i(n,`rowsPerBlock`,this.rowsPerBlock,1,this.rowCount),u(n,`rowsPerBlock x events`,this.rowsPerBlock*this.eventCount,4294967295),this.blockCount=Math.ceil(this.rowCount/this.rowsPerBlock);let l=G(t);a(n,l,W(t));let d=l.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(d).size!==d.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(e){let{id:t,props:n,eventCount:i,nodeCount:a,edgeCount:o,bandCount:c,simulationCount:l}=this,{rowCount:f,rowsPerBlock:v,blockCount:b}=this,{output:x}=n,{offsets:S,neighbors:C,weights:w}=n.topology;r(t,e,[...W(n),...G(n)]);let T=Math.min(n.laneCount??v,v);u(t,`expanded seed words (rows, rowsPerBlock, laneCount)`,b*Math.ceil(v/T)*U(2*T),oe(e.device));let E=[],D=1+l,O=(n,r)=>d(e,`${t}-${n}`,`uint32`,r),k=O(`seed-nodes`,2*f),A=d(e,`${t}-seed-costs`,`float32`,2*f),j=O(`snapped-edges`,i);E.push(...new te({id:`${t}-snap`,spatialContext:this.spatialContext,positions:n.positions,topology:n.topology,maximumSnapDistance:n.maximumSnapDistance,candidateCapacity:n.candidateCapacity,spatialSort:n.spatialSort,startDirection:`both`,output:{snappedEdges:j,startNodes:H(e,k,0,2*i),startCosts:H(e,A,0,2*i),status:{overflow:x.status?.overflow}}}).getCommandNodes(e));let M=x.snappedEventCount??O(`event-count`,1),re=O(`event-flags`,i);if(E.push(g(e,{id:`${t}-event-flags`,operation:L,variant:`event-flags`,bindings:[{name:`seedNodes`,view:k,type:`u32`,access:`read`},{name:`validFlags`,view:re,type:`u32`,access:`read_write`}],invocationCount:i,declarations:`const NODE_COUNT: u32 = ${a}u;`,body:`validFlags[validFlagsOffset + index] =
    select(0u, 1u, seedNodes[seedNodesOffset + 2u * index] < NODE_COUNT);`}),...new m({id:`${t}-event-count`,input:re,output:M,operation:`sum`}).getCommandNodes(e)),l>0){let r=O(`quantized-lengths`,o),s=O(`length-prefix`,o),c=d(e,`${t}-live-weights`,`float32`,o),u=d(e,`${t}-live-maximum`,`float32`,1),f=ae(o),h=`${_}
fn getLiveWeight(edge: u32) -> f32 {
  let weight = weights[weightsOffset + edge];
  let isLive = edge < offsets[offsetsOffset + NODE_COUNT] &&
    neighbors[neighborsOffset + edge] < NODE_COUNT && isFiniteBits(weight) && weight >= 0.0;
  return select(0.0, weight, isLive);
}`;E.push(g(e,{id:`${t}-live-weights`,operation:L,variant:`live-weights`,bindings:[{name:`offsets`,view:S,type:`u32`,access:`read`},{name:`neighbors`,view:C,type:`u32`,access:`read`},{name:`weights`,view:w,type:`f32`,access:`read`},{name:`liveWeights`,view:c,type:`f32`,access:`read_write`}],invocationCount:o,declarations:`const NODE_COUNT: u32 = ${a}u;
${h}`,body:`liveWeights[liveWeightsOffset + index] = getLiveWeight(index);`}),...new m({id:`${t}-live-maximum`,input:c,output:u,operation:`max`}).getCommandNodes(e),g(e,{id:`${t}-quantize`,operation:L,variant:`quantize`,bindings:[{name:`liveWeights`,view:c,type:`f32`,access:`read`},{name:`liveMaximum`,view:u,type:`f32`,access:`read`},{name:`quantized`,view:r,type:`u32`,access:`read_write`}],invocationCount:o,body:`let maximum = liveMaximum[liveMaximumOffset];
  var value = 0u;
  if (maximum > 0.0) {
    value = u32(liveWeights[liveWeightsOffset + index] * (${f}.0 / maximum));
  }
  quantized[quantizedOffset + index] = value;`}),...new p({id:`${t}-length-scan`,input:r,output:s,mode:`inclusive`}).getCommandNodes(e),g(e,{id:`${t}-simulate`,operation:L,variant:`simulate`,bindings:[{name:`prefix`,view:s,type:`u32`,access:`read`},{name:`offsets`,view:S,type:`u32`,access:`read`},{name:`neighbors`,view:C,type:`u32`,access:`read`},{name:`weights`,view:w,type:`f32`,access:`read`},{name:`validCount`,view:M,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`u32`,access:`read`},{name:`outNodes`,view:k,type:`u32`,access:`read_write`},{name:`outCosts`,view:A,type:`f32`,access:`read_write`}],invocationCount:l*i,declarations:`const NODE_COUNT: u32 = ${a}u;
const EDGE_COUNT: u32 = ${o}u;
const EVENT_COUNT: u32 = ${i}u;
const SIMULATION_COUNT: u32 = ${l}u;
const NONE: u32 = ${se}u;
const STREAM_TAG: u32 = ${ie}u;
${y}`,body:`let pattern = 1u + index / EVENT_COUNT;
  let eventIndex = index - (pattern - 1u) * EVENT_COUNT;
  let slot = 2u * (pattern * EVENT_COUNT + eventIndex);
  let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  let total = prefix[prefixOffset + EDGE_COUNT - 1u];
  var firstNode = NONE;
  var secondNode = NONE;
  var firstCost = -1.0;
  var secondCost = -1.0;
  if (pattern <= activeCount && eventIndex < validCount[validCountOffset] && total > 0u) {
    var stream = createPhiloxStream(
      vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]),
      eventIndex,
      pattern,
      STREAM_TAG
    );
    let draw = nextPhiloxBelow(&stream, total);
    let fractionBits = nextPhiloxUint32(&stream);
    var low = 0u;
    var high = EDGE_COUNT;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (prefix[prefixOffset + middle] > draw) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    let edge = low;
    var nodeLow = 0u;
    var nodeHigh = NODE_COUNT;
    while (nodeLow < nodeHigh) {
      let middle = (nodeLow + nodeHigh) / 2u;
      if (offsets[offsetsOffset + middle + 1u] > edge) {
        nodeHigh = middle;
      } else {
        nodeLow = middle + 1u;
      }
    }
    let fraction = f32(fractionBits >> 8u) / 16777216.0;
    let weight = weights[weightsOffset + edge];
    firstNode = nodeLow;
    secondNode = neighbors[neighborsOffset + edge];
    firstCost = fraction * weight;
    secondCost = (1.0 - fraction) * weight;
  }
  outNodes[outNodesOffset + slot] = firstNode;
  outNodes[outNodesOffset + slot + 1u] = secondNode;
  outCosts[outCostsOffset + slot] = firstCost;
  outCosts[outCostsOffset + slot + 1u] = secondCost;`}))}let N=O(`histogram`,D*c);E.push(h(e,{id:`${t}-clear`,operation:L,view:N,type:`u32`,value:`0u`}));let P=Math.min(n.laneCount??v,v);if(P*a>=4294967295)throw Error(`${t} laneCount * nodeCount must fit in uint32`);let F=Math.ceil(v/P),I=b*F,R=U(P*a),B=U(2*P);s(e.device,t,`block costs`,F*R*4,`lower rowsPerBlock`);let V=d(e,`${t}-costs`,`float32`,F*R),le=O(`expanded-seeds`,I*B),K=d(e,`${t}-expanded-seed-costs`,`float32`,I*B);E.push(g(e,{id:`${t}-expand-seeds`,operation:L,variant:`expand-seeds`,bindings:[{name:`seedNodes`,view:k,type:`u32`,access:`read`},{name:`seedCosts`,view:A,type:`f32`,access:`read`},{name:`expandedSeeds`,view:le,type:`u32`,access:`read_write`},{name:`expandedSeedCosts`,view:K,type:`f32`,access:`read_write`}],invocationCount:I*2*P,declarations:`const NODE_COUNT: u32 = ${a}u;
const ROW_COUNT: u32 = ${f}u;
const ROWS_PER_BLOCK: u32 = ${v}u;
const LANE_COUNT: u32 = ${P}u;
const BATCHES_PER_BLOCK: u32 = ${F}u;
const SEED_STRIDE: u32 = ${B}u;`,body:`let batch = index / (2u * LANE_COUNT);
  let slot = index - batch * 2u * LANE_COUNT;
  let lane = slot / 2u;
  let blockIndex = batch / BATCHES_PER_BLOCK;
  let rowInBlock = (batch - blockIndex * BATCHES_PER_BLOCK) * LANE_COUNT + lane;
  let row = blockIndex * ROWS_PER_BLOCK + rowInBlock;
  var node = ${ee}u;
  var cost = -1.0;
  if (rowInBlock < ROWS_PER_BLOCK && row < ROW_COUNT) {
    let seedNode = seedNodes[seedNodesOffset + 2u * row + (slot & 1u)];
    if (seedNode < NODE_COUNT) {
      node = lane * NODE_COUNT + seedNode;
      cost = seedCosts[seedCostsOffset + 2u * row + (slot & 1u)];
    }
  }
  expandedSeeds[expandedSeedsOffset + batch * SEED_STRIDE + slot] = node;
  expandedSeedCosts[expandedSeedCostsOffset + batch * SEED_STRIDE + slot] = cost;`}));let ue=O(`block-info`,b*z);E.push(g(e,{id:`${t}-block-info`,operation:L,variant:`block-info`,bindings:[{name:`blockInfo`,view:ue,type:`u32`,access:`read_write`}],invocationCount:b,declarations:`const ROW_COUNT: u32 = ${f}u;
const ROWS_PER_BLOCK: u32 = ${v}u;
const SLOT_WORDS: u32 = ${z}u;`,body:`let firstRow = index * ROWS_PER_BLOCK;
  blockInfo[blockInfoOffset + index * SLOT_WORDS] = firstRow;
  blockInfo[blockInfoOffset + index * SLOT_WORDS + 1u] = min(ROWS_PER_BLOCK, ROW_COUNT - firstRow);`}));let q=x.status?.converged?O(`batch-converged`,I*z):void 0;for(let r=0;r<b;r++){for(let i=0;i<F;i++){let o=r*F+i;E.push(...new ne({id:`${t}-block-${r}-batch-${i}`,topology:{offsets:S,neighbors:C,weights:w},laneCount:P,startNodes:H(e,le,o*B,2*P),startCosts:H(e,K,o*B,2*P),costLimit:n.maximumDistance,maximumIterations:n.maximumIterations,localIterations:n.localIterations??ce,output:{costs:H(e,V,i*R,P*a),status:{converged:q?H(e,q,o*z,1):void 0}}}).getCommandNodes(e))}E.push(g(e,{id:`${t}-block-${r}-count`,operation:L,variant:`count`,bindings:[{name:`costs`,view:V,type:`f32`,access:`read`},{name:`seedNodes`,view:k,type:`u32`,access:`read`},{name:`seedCosts`,view:A,type:`f32`,access:`read`},{name:`maximumDistanceIn`,view:n.maximumDistance,type:`f32`,access:`read`},{name:`histogram`,view:N,type:`atomic<u32>`,access:`read_write`},{name:`blockInfo`,view:H(e,ue,r*z,2),type:`u32`,access:`read`}],invocationCount:v*i,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${a}u;
const EVENT_COUNT: u32 = ${i}u;
const BAND_COUNT: u32 = ${c}u;
const LANE_COUNT: u32 = ${P}u;
const BATCH_STRIDE: u32 = ${R}u;
const WORKGROUP_SIZE: u32 = 64u;
// Per-workgroup histogram of the pattern that owns the workgroup's first invocation. A workgroup
// spans 64 consecutive (row, column) pairs, so nearly all of them share one pattern and one
// global atomic per pair collapses into one local atomic and at most BAND_COUNT global adds.
var<workgroup> localHistogram: array<atomic<u32>, ${c}>;
fn isFiniteCost(cost: f32) -> bool {
  return bitcast<u32>(cost) < 0x7f800000u;
}
fn getThreshold(maximumDistance: f32, band: u32) -> f32 {
  return maximumDistance * f32(band) / f32(BAND_COUNT - 1u);
}
// The block's first row, read from its info slot (rule 12: the block index is data).
var<private> blockFirstRow: u32;
fn getPattern(index: u32) -> u32 {
  return (blockFirstRow + index / EVENT_COUNT) / EVENT_COUNT;
}
// Start of a block row's costs: batch (localRow / LANE_COUNT) at BATCH_STRIDE, then its lane.
fn getRowCostBase(localRow: u32) -> u32 {
  let batch = localRow / LANE_COUNT;
  return costsOffset + batch * BATCH_STRIDE + (localRow - batch * LANE_COUNT) * NODE_COUNT;
}
// Distance band of the unordered pair of row and column events, or BAND_COUNT when the pair is
// not counted.
fn classifyPair(index: u32) -> u32 {
  let localRow = index / EVENT_COUNT;
  let columnEvent = index - localRow * EVENT_COUNT;
  let row = blockFirstRow + localRow;
  let pattern = row / EVENT_COUNT;
  let eventIndex = row - pattern * EVENT_COUNT;
  let maximumDistance = maximumDistanceIn[maximumDistanceInOffset];
  if (columnEvent <= eventIndex || !(maximumDistance > 0.0)) {
    return BAND_COUNT;
  }
  let rowSlot = 2u * row;
  let columnSlot = 2u * (pattern * EVENT_COUNT + columnEvent);
  let rowFirst = seedNodes[seedNodesOffset + rowSlot];
  let rowSecond = seedNodes[seedNodesOffset + rowSlot + 1u];
  let columnFirst = seedNodes[seedNodesOffset + columnSlot];
  let columnSecond = seedNodes[seedNodesOffset + columnSlot + 1u];
  if (rowFirst >= NODE_COUNT || rowSecond >= NODE_COUNT ||
      columnFirst >= NODE_COUNT || columnSecond >= NODE_COUNT) {
    return BAND_COUNT;
  }
  let rowFirstCost = seedCosts[seedCostsOffset + rowSlot];
  let columnFirstCost = seedCosts[seedCostsOffset + columnSlot];
  let columnSecondCost = seedCosts[seedCostsOffset + columnSlot + 1u];
  var best = 0.0;
  var found = false;
  let viaFirst = costs[getRowCostBase(localRow) + columnFirst];
  if (isFiniteCost(viaFirst)) {
    best = viaFirst + columnFirstCost;
    found = true;
  }
  let viaSecond = costs[getRowCostBase(localRow) + columnSecond];
  if (isFiniteCost(viaSecond)) {
    let candidate = viaSecond + columnSecondCost;
    if (!found || candidate < best) {
      best = candidate;
      found = true;
    }
  }
  var sameEdge = -1.0;
  if (rowFirst != rowSecond) {
    if (rowFirst == columnFirst && rowSecond == columnSecond) {
      sameEdge = abs(rowFirstCost - columnFirstCost);
    } else if (rowFirst == columnSecond && rowSecond == columnFirst) {
      sameEdge = abs(rowFirstCost - columnSecondCost);
    }
  }
  if (sameEdge >= 0.0 && (!found || sameEdge < best)) {
    best = sameEdge;
    found = true;
  }
  if (!found || !(best < maximumDistance)) {
    return BAND_COUNT;
  }
  var band = u32(max(floor(best / (maximumDistance / f32(BAND_COUNT - 1u))), 0.0)) + 1u;
  band = min(band, BAND_COUNT);
  while (band > 0u && best < getThreshold(maximumDistance, band - 1u)) {
    band--;
  }
  while (band < BAND_COUNT && !(best < getThreshold(maximumDistance, band))) {
    band++;
  }
  return band;
}`,workgroupSize:64,body:`blockFirstRow = blockInfo[blockInfoOffset];
  let liveCount = blockInfo[blockInfoOffset + 1u] * EVENT_COUNT;
  let firstIndex = index - localInvocationIndex;
  let ownerPattern = getPattern(min(firstIndex, liveCount - 1u));
  for (var bin = localInvocationIndex; bin < BAND_COUNT; bin += WORKGROUP_SIZE) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  if (index < liveCount) {
    let band = classifyPair(index);
    if (band < BAND_COUNT) {
      let pattern = getPattern(index);
      if (pattern == ownerPattern) {
        atomicAdd(&localHistogram[band], 1u);
      } else {
        atomicAdd(&histogram[histogramOffset + pattern * BAND_COUNT + band], 1u);
      }
    }
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < BAND_COUNT; bin += WORKGROUP_SIZE) {
    let binCount = atomicLoad(&localHistogram[bin]);
    if (binCount > 0u) {
      atomicAdd(&histogram[histogramOffset + ownerPattern * BAND_COUNT + bin], binCount);
    }
  }`}))}let de=[{name:`histogram`,view:N,type:`u32`,access:`read`},{name:`networkLength`,view:n.networkLength,type:`f32`,access:`read`},{name:`validCount`,view:M,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`u32`,access:`read`},{name:`kValues`,view:x.kValues,type:`f32`,access:`read_write`}];return x.pairCounts&&de.push({name:`pairCounts`,view:x.pairCounts,type:`u32`,access:`read_write`}),E.push(g(e,{id:`${t}-finalize`,operation:L,variant:`finalize`,record:{datum:{keySpace:`band`},formula:`gpu-network.k-function.k`},bindings:de,invocationCount:D*c,declarations:`const BAND_COUNT: u32 = ${c}u;
const SIMULATION_COUNT: u32 = ${l}u;`,body:`let pattern = index / BAND_COUNT;
  let band = index - pattern * BAND_COUNT;
  let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  var cumulative = 0u;
  if (pattern <= activeCount) {
    for (var bin = 0u; bin <= band; bin++) {
      cumulative += histogram[histogramOffset + pattern * BAND_COUNT + bin];
    }
  }
  let count = f32(validCount[validCountOffset]);
  var k = 0.0;
  if (count >= 2.0) {
    k = 2.0 * f32(cumulative) * networkLength[networkLengthOffset] / (count * count);
  }
  kValues[kValuesOffset + index] = k;
  ${x.pairCounts?`pairCounts[pairCountsOffset + index] = cumulative;`:``}`})),x.envelope&&E.push(g(e,{id:`${t}-envelope`,operation:L,variant:`envelope`,bindings:[{name:`kValues`,view:x.kValues,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`u32`,access:`read`},{name:`envelope`,view:x.envelope,type:`f32`,access:`read_write`}],invocationCount:c,declarations:`const BAND_COUNT: u32 = ${c}u;
const SIMULATION_COUNT: u32 = ${l}u;`,body:`let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  var lowest = 0.0;
  var highest = 0.0;
  var sum = 0.0;
  for (var pattern = 1u; pattern <= activeCount; pattern++) {
    let value = kValues[kValuesOffset + pattern * BAND_COUNT + index];
    lowest = select(min(lowest, value), value, pattern == 1u);
    highest = select(max(highest, value), value, pattern == 1u);
    sum += value;
  }
  envelope[envelopeOffset + index] = lowest;
  envelope[envelopeOffset + BAND_COUNT + index] = select(0.0, sum / f32(activeCount), activeCount > 0u);
  envelope[envelopeOffset + 2u * BAND_COUNT + index] = highest;`})),x.status?.converged&&q&&E.push(g(e,{id:`${t}-converged`,operation:L,variant:`converged`,bindings:[{name:`batchConverged`,view:q,type:`u32`,access:`read`},{name:`converged`,view:x.status.converged,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const BATCH_COUNT: u32 = ${I}u;
const SLOT_WORDS: u32 = ${z}u;`,body:`var allDone = 1u;
  for (var batch = 0u; batch < BATCH_COUNT; batch++) {
    allDone = min(allDone, batchConverged[batchConvergedOffset + batch * SLOT_WORDS]);
  }
  converged[convergedOffset] = allDone;`})),E}};function H(e,t,n,r){return e.createDataView(t.buffer,{format:t.format,length:r,byteOffset:t.byteOffset+n*t.byteStride})}function U(e){return Math.ceil(e/z)*z}function W(e){return[e.positions,...A(e.topology),e.maximumSnapDistance,e.maximumDistance,e.networkLength,e.parameters]}function G(e){let{output:t}=e;return[t.kValues,t.pairCounts,t.envelope,t.snappedEventCount,t.status?.overflow,t.status?.converged]}var K=1e7,ue=16,q=20,de=O({breaks:[1,3,8,20,50],colors:[[105,112,125,255],[255,255,178,255],[254,204,92,255],[253,141,60,255],[227,26,28,255],[128,0,38,255]]}),fe=O({breaks:[.25,.5,.75],colors:[[215,240,255,255],[145,205,245,255],[70,155,220,255],[20,95,175,255]]}),pe={retail:`Retail`,restaurant_cafe:`Restaurants and cafes`,grocery:`Groceries`,combined:`Shops, cafes and groceries`};async function J(e){let t=e.datasets.get(`chicago-roads`),r=e.datasets.get(`chicago-places`),{device:i}=e,a=t.defaultOrigin,o=re(t,a),s=r.projectColumn(`position`,a),u=r.column(`category`),d=u.length,{nodeCount:f,slotCount:p}=o,m=new l(i,`retail-network-k`),h=new N(o.segments,o.bounds),g=t.getProjection(a),_=(e,t)=>{let n=u[e];return t===`combined`?n===0||n===2||n===7:n===7?t===`retail`:n===0?t===`restaurant_cafe`:n===2&&t===`grocery`},v=new Map,y=e=>{let t=v.get(e);if(!t){let n=[];for(let t=0;t<d;t++)_(t,e)&&n.push(t);t=Uint32Array.from(n),v.set(e,t)}return t},O=m.createBuffer(`events`,256*8),k=m.createBuffer(`node-positions`,o.nodePositions),A=m.createBuffer(`offsets`,o.offsets),ee=m.createBuffer(`neighbors`,o.neighbors),te=m.createBuffer(`weights`,o.weights),ne=m.createBuffer(`segments`,o.segments),j=m.createBuffer(`segment-slots`,o.segmentSlots),I=m.createBuffer(`slot-places`,p*4),L=m.createBuffer(`network-search-ball`,o.segments.byteLength),ie=m.createBuffer(`network-search-ball-distances`,o.segmentCount*4),ae=m.createBuffer(`straight-radius-ring`,1536),R=m.createBuffer(`street-length-null-points`,384),oe=m.createParameterBuffer(`max-distance`,`float32`,1),se=m.createParameterBuffer(`max-snap-distance`,`float32`,1),z=m.createParameterBuffer(`network-length`,`float32`,1,Float32Array.of(o.networkLength)),B=m.createParameterBuffer(`k-parameters`,`uint32`,4),V=m.createBuffer(`snapped-count`,4),ce=m.createBuffer(`snap-overflow`,4),H=!1,U=!0,W=null,G=0,J=0,me=!1,he=0,Y=K,X=K,Z=new Float64Array(o.segmentCount),ge=0;for(let e=0;e<o.segmentCount;e++){let t=o.segments[e*4],n=o.segments[e*4+1],r=o.segments[e*4+2],i=o.segments[e*4+3];ge+=Math.hypot(r-t,i-n),Z[e]=ge}e.setReadout(`network`,`${c(f)} intersections, ${c(o.edgeCount)} directed edges`),e.setReadout(`length`,M(o.networkLength));function _e(){let t=pe[e.options.category];e.setFurniture({title:{title:`Retail places on the street network`,sample:`${t} · ${c(J)} mapped places · sample ${G}`},scaleBar:{units:`metric`,ticks:[e.options.maxDistance]},credit:`Overture Maps Foundation (mixed-permissive) · OpenStreetMap contributors (ODbL)`,caveat:`Street-length null; coverage, footfall and zoning are not controlled.`})}function ve(t,n){let r=h.nearest(t,n,e.options.maxSnapDistance),i=new Float64Array(f).fill(1/0);if(r>=0){let a=o.segmentEdge[r],s=o.segments[r*4],c=o.segments[r*4+1],l=o.segments[r*4+2],u=o.segments[r*4+3],d=l-s,f=u-c,p=Math.max(0,Math.min(1,((t-s)*d+(n-c)*f)/Math.max(d*d+f*f,1e-9))),m=o.edgeLength[a],h=[[o.edgeSource[a],m*p],[o.edgeTarget[a],m*(1-p)]],g=[...h];for(let[e,t]of h)i[t]=Math.min(i[t],e);for(;g.length;){g.sort((e,t)=>e[0]-t[0]);let[t,n]=g.shift();if(!(t!==i[n]||t>e.options.maxDistance))for(let r=o.offsets[n];r<o.offsets[n+1];r++){let n=o.neighbors[r],a=t+o.weights[r];a<i[n]&&a<=e.options.maxDistance&&(i[n]=a,g.push([a,n]))}}}let a=[],s=[];for(let t=0;t<o.segmentCount;t++){let n=o.segmentEdge[t],r=Math.min(i[o.edgeSource[n]],i[o.edgeTarget[n]]);r<=e.options.maxDistance&&(a.push(...o.segments.subarray(t*4,t*4+4)),s.push(r/Math.max(e.options.maxDistance,1)))}he=s.length,L.write(Float32Array.from(a)),ie.write(Float32Array.from(s))}function ye(){let t=b(e.options.envelopeSeed*97+7),n=new Float32Array(96);for(let e=0;e<48;e++){let r=t()*ge,i=0,a=Z.length-1;for(;i<a;){let e=Math.floor((i+a)/2);Z[e]>r?a=e:i=e+1}let s=i,c=s===0?0:Z[s-1],l=(r-c)/Math.max(Z[s]-c,1e-9),u=s*4;n[e*2]=o.segments[u]*(1-l)+o.segments[u+2]*l,n[e*2+1]=o.segments[u+1]*(1-l)+o.segments[u+3]*l}R.write(n)}function be(){let t=[];for(let n=0;n<96;n++){let r=n/96*Math.PI*2,i=(n+1)/96*Math.PI*2;t.push(Y+Math.cos(r)*e.options.maxDistance,X+Math.sin(r)*e.options.maxDistance,Y+Math.cos(i)*e.options.maxDistance,X+Math.sin(i)*e.options.maxDistance)}ae.write(Float32Array.from(t))}function xe(){let{category:t,eventCount:n,sampleSeed:r}=e.options,i=y(t);J=i.length;let a=Array.from(i),c=b(r*7919+13);for(let e=a.length-1;e>0;e--){let t=Math.floor(c()*(e+1));[a[e],a[t]]=[a[t],a[e]]}let l=new Float32Array(256*2).fill(K);G=Math.min(n,256,a.length);for(let e=0;e<G;e++)l[e*2]=s[a[e]*2],l[e*2+1]=s[a[e]*2+1];O.write(l),Y=l[0],X=l[1],ve(Y,X),be(),ye();let u=new Float32Array(p);for(let t of i){let n=h.nearest(s[t*2],s[t*2+1],e.options.maxSnapDistance);if(n<0)continue;let r=o.segmentSlots[n];u[r]+=1e3/Math.max(50,o.edgeLength[o.segmentEdge[n]])}I.write(u),Se=u,e.setReadout(`eligible`,J),e.setReadout(`eventsUsed`,G),e.setReadout(`sampleFraction`,`${G} of ${J} eligible (${J?(100*G/J).toFixed(1):`0`}%)`),_e(),U=!0}let Se=new Float32Array(p);function Q(){let{maxDistance:t,maxSnapDistance:n,simulations:r,envelopeSeed:i}=e.options;oe.write(Float32Array.of(t)),se.write(Float32Array.of(n)),B.write(F({seed:i,activeSimulations:r})),U=!0}async function Ce(){let t=Number(e.options.bandCount),r=Number(e.options.rowsPerBlock),a=m.createBuffer(`k-values-${t}`,20*t*4),o=m.createBuffer(`envelope-${t}`,3*t*4),s=m.createBuffer(`pair-counts-${t}`,20*t*4),c=m.createBuffer(`converged-${t}`,4),l=new n(i,{id:`retail-network-k`});l.add(new le({spatialContext:g.spatialContext,id:`k`,positions:w(l,`events`,O,`float32x2`,256),topology:{offsets:w(l,`offsets`,A,`uint32`,f+1),neighbors:w(l,`neighbors`,ee,`uint32`,p),weights:w(l,`weights`,te,`float32`,p),nodePositions:w(l,`node-positions`,k,`float32x2`,f)},maximumSnapDistance:se.importToGraph(l),maximumDistance:oe.importToGraph(l),networkLength:z.importToGraph(l),parameters:B.importToGraph(l),bandCount:t,simulationCount:19,rowsPerBlock:r,maximumIterations:q,localIterations:ue,spatialSort:e.options.spatialSort,candidateCapacity:256*512,output:{kValues:w(l,`k-values`,a,`float32`,20*t),pairCounts:w(l,`pair-counts`,s,`uint32`,20*t),envelope:w(l,`envelope`,o,`float32`,3*t),snappedEventCount:w(l,`snapped-count`,V,`uint32`,1),status:{overflow:w(l,`snap-overflow`,ce,`uint32`,1),converged:w(l,`converged`,c,`uint32`,1)}}}));let u=m.track(await l.compileAsync()),d=t*4,h=new E(m,`k-${t}`,[{buffer:a,size:d},{buffer:o,size:3*d},{buffer:s,size:d},{buffer:V,size:4},{buffer:c,size:4},{buffer:ce,size:4}],e=>{H||W?.reader!==h||De(e,t)});return{compiled:u,bandCount:t,rowsPerBlock:r,kValues:a,envelope:o,pairCounts:s,converged:c,reader:h}}function we(e){e.reader.stop(),m.release(e.compiled);for(let t of[e.kValues,e.envelope,e.pairCounts,e.converged])m.release(t)}function Te(t){let n=W;W=t,n&&we(n),e.setReadout(`rows`,(1+e.options.simulations)*256),e.setReadout(`blocks`,Math.ceil(20*256/t.rowsPerBlock)),U=!0}let Ee=m.track(new T);function De(t,n){let r=new Float32Array(t),i=new Uint32Array(t),a=r.slice(0,n),o=r.slice(n,2*n),s=r.slice(2*n,3*n),l=r.slice(3*n,4*n),u=i[4*n+n-1],d=5*n,f=i[d],p=i[d+1],m=i[d+2],h=e.options.simulations>0,g=n-1,_=0,v=0,y=-1;for(let e=1;e<n;e++)h&&a[e]>l[e]?(_++,y<0&&(y=e)):h&&a[e]<o[e]&&(v++,y<0&&(y=e));e.setReadout(`events`,`${c(f)} of ${G}${m?` (snap overflow)`:``}`);let b=Array.from({length:n},(t,r)=>(r+1)*e.options.maxDistance/n),x=h&&s.every(e=>e>0&&Number.isFinite(e)),S=e=>x?Array.from(e,(e,t)=>e/s[t]):e,C=S(a),w=S(o),T=S(l);e.setChart(`kCurve`,{kind:`line`,xLabel:`network distance (m)`,yLabel:x?`K / simulated mean`:`K(d)`,series:[{label:`observed`,x:b,y:C,color:0,width:2.4},{label:`simulated mean`,x:b,y:x?b.map(()=>1):Array.from(s),color:1,dashed:!0},{label:`observed outside envelope`,x:b.filter((e,t)=>h&&(a[t]>l[t]||a[t]<o[t])),y:C.filter((e,t)=>h&&(a[t]>l[t]||a[t]<o[t])),color:0,points:!0,width:0}],band:{x:b,low:w,high:T,label:`pointwise min–max envelope`},guides:x?[{y:1,label:`street-length null`}]:void 0,markers:h?b.flatMap((e,t)=>a[t]>l[t]||a[t]<o[t]?[{x:e,label:t===y?`first outside`:`outside`}]:[]):[{x:e.options.maxDistance,label:`maximum d`}],link:{option:`maxDistance`,label:e=>`${e} m`},description:`Observed network K against the active simulated street-length null. The grey band is pointwise, not a global test.`}),e.setReadout(`firstOutside`,y<0?`none in active pointwise envelope`:`${Math.round(b[y])} m (pointwise, not a global test)`),e.setReadout(`bands`,h?`${_} above, ${v} below of ${n-1}`:`no simulations`),e.setReadout(`kAtMaximum`,a[g]),e.setReadout(`envelopeRange`,h?`${o[g].toFixed(0)} to ${l[g].toFixed(0)}`:null),e.setReadout(`ratio`,h&&s[g]>0?`${(a[g]/s[g]).toFixed(2)}x`:null),e.setReadout(`verdict`,h?`${_} above and ${v} below the pointwise envelope; not a global test`:`enable simulations`),e.setReadout(`pairs`,u),e.setReadout(`converged`,p?`yes`:`no: raise iterations or lower distance`)}async function Oe(){if(!(me||!W)){me=!0;try{let t=await C(i,W.compiled,{parameters:void 0,completionBuffer:W.converged,runs:5,repetitions:2});H||e.setReadout(`time`,`${t.milliseconds.toFixed(1)} ms (${t.method===`gpu-timestamps`?`GPU`:`wall clock`})`)}catch{}finally{me=!1}}}xe(),Q(),Te(await Ce());let $=[a[0],a[1],0];return{getCompiledGraphs:()=>W?[W.compiled]:[],setOption(t,n,r){switch(t){case`category`:case`eventCount`:case`sampleSeed`:xe(),e.requestLayers();break;case`maxDistance`:case`simulations`:Q(),t===`maxDistance`&&(ve(Y,X),be(),_e()),e.setReadout(`rows`,(1+r.simulations)*256);break;case`maxSnapDistance`:Q(),xe();break;case`envelopeSeed`:Q(),ye(),_e();break;case`bandCount`:case`rowsPerBlock`:case`spatialSort`:Ee.request({build:Ce,commit:Te,discard:we});break;default:e.requestLayers()}},onAction(e){e===`measure`&&Oe()},onThemeChange(){e.requestLayers()},getTooltip(e){if(!e.coordinate)return null;let[t,n]=g.project(e.coordinate[0],e.coordinate[1]),r=h.nearest(t,n,35);if(r<0)return null;let i=o.segmentEdge[r],a=o.segmentSlots[r],s=Se[a];return`${P[o.edgeClass[i]]} block, ${Math.round(o.edgeLength[i])} m, ${o.edgeSpeed[i]} km/h\n${s.toFixed(1)} mapped places per km`},encode(e){W&&(U?(W.compiled.encode(e,{parameters:void 0}),U=!1,W.reader.request(e)):W.reader.flush(e))},getLayers(){let{roadStyle:t,showEvents:n}=e.options,r=e.ground()===`dark`,i=[new S({id:`retail-network-k-roads`,coordinateOrigin:$,segments:ne,instanceCount:o.segmentCount,widthPixels:.55,color:r?[150,160,190,t===`plain`?130:70]:[120,130,155,t===`plain`?130:80]})];if(t===`perKm`)for(let[e,t]of[.8,1.2,1.8,2.6,3.1,3.4].entries())i.push(new S({id:`retail-network-k-place-load-${e}`,coordinateOrigin:$,segments:ne,instanceCount:o.segmentCount,values:I,valueFormat:`float32`,valueIndices:j,...D(de),highlightClasses:[e],dimOpacity:0,widthPixels:t}));return n&&i.push(...[.8,1.15,1.5,1.9].map((e,t)=>new S({id:`retail-network-search-ball-${t}`,coordinateOrigin:$,segments:L,instanceCount:he,widthPixels:e,values:ie,valueFormat:`float32`,...D(fe),highlightClasses:[t],dimOpacity:0,color:r?[95,185,255,180]:[20,110,175,170]})),new S({id:`retail-straight-radius-ring`,coordinateOrigin:$,segments:ae,instanceCount:96,widthPixels:1,dashArray:[4,3],color:r?[215,225,240,180]:[55,70,95,150]}),new x({id:`retail-street-length-null`,coordinateOrigin:$,positions:R,instanceCount:48,radiusPixels:3,shape:`ring`,color:r?[210,215,225,170]:[80,90,105,170]}),new x({id:`retail-network-k-events`,coordinateOrigin:$,positions:O,instanceCount:G,radiusPixels:3.6,color:r?[90,230,255,235]:[0,120,200,235]})),i},destroy(){H=!0,W?.reader.stop(),m.destroy()}}}export{J as createRetailNetworkKFunction};