import{J as e,mt as t}from"./src-CNCKFW2b.js";import{_ as n,d as r,l as i,v as a,w as o,y as s}from"./resources-CwvhEOBe.js";import{t as c}from"./gpu-histogram-C4GJppav.js";import{s as l}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as u,n as d}from"./network-topology-BXzNuj-N.js";var f=4;function p(e){return{controlWordCount:4+e+1,dispatchSlotCount:Math.floor(e/2)+1}}function m(e,n,r){let i=t.STORAGE|t.INDIRECT;return{nodeCount:r.nodeCount,queues:o(e,`${n}-queues`,`uint32`,2*r.nodeCount),queuedRound:o(e,`${n}-queued-round`,`uint32`,r.nodeCount),control:o(e,`${n}-control`,`uint32`,r.controlWordCount),dispatchEven:o(e,`${n}-dispatch-even`,`uint32`,r.dispatchSlotCount*64,i),dispatchOdd:o(e,`${n}-dispatch-odd`,`uint32`,r.dispatchSlotCount*64,i)}}function h(e){return e.stampBase+e.maxRounds+1}function g(e,t){return 256*(e.dispatchSlotOffset+Math.floor(t/2))}function _(e,t,n,r){let i=r%2==1?t.dispatchOdd:t.dispatchEven,a=e.createDataView(i.buffer,{format:`uint32`,length:64,byteOffset:i.byteOffset+g(n,r)});return[{name:`queues`,view:t.queues,type:`u32`,access:`read_write`},{name:`queuedRound`,view:t.queuedRound,type:`atomic<u32>`,access:`read_write`},{name:`control`,view:t.control,type:`atomic<u32>`,access:`read_write`},{name:`nextDispatch`,view:a,type:`atomic<u32>`,access:`read_write`}]}function v(e,t,n){return`
const FRONTIER_MAXIMUM_X: u32 = ${n}u;
const FRONTIER_TRUNCATED_WORD: u32 = ${e.controlWordOffset+0}u;
const FRONTIER_LIMIT_WORD: u32 = ${e.controlWordOffset+1}u;
const FRONTIER_PARAMETER_0_WORD: u32 = ${e.controlWordOffset+2}u;
const FRONTIER_PARAMETER_1_WORD: u32 = ${e.controlWordOffset+3}u;
const FRONTIER_ROUND_COUNTS_WORD: u32 = ${e.controlWordOffset+4}u;
const FRONTIER_STAMP_BASE: u32 = ${e.stampBase}u;
const FRONTIER_QUEUE_NODE_COUNT: u32 = ${t}u;
const FRONTIER_SLOT_ROUND_WORD: u32 = 3u;

// Round index (within the phase) of the bound nextDispatch slot.
fn frontierNextRound() -> u32 {
  return atomicLoad(&nextDispatch[nextDispatchOffset + FRONTIER_SLOT_ROUND_WORD]);
}

fn frontierPush(targetNode: u32) {
  let nextRound = frontierNextRound();
  if (nextRound >= atomicLoad(&control[controlOffset + FRONTIER_LIMIT_WORD])) {
    atomicStore(&control[controlOffset + FRONTIER_TRUNCATED_WORD], 1u);
    return;
  }
  let nextStamp = FRONTIER_STAMP_BASE + nextRound + 1u;
  if (atomicMax(&queuedRound[queuedRoundOffset + targetNode], nextStamp) >= nextStamp) {
    return;
  }
  let slot = atomicAdd(&control[controlOffset + FRONTIER_ROUND_COUNTS_WORD + nextRound], 1u);
  queues[queuesOffset + (nextRound & 1u) * FRONTIER_QUEUE_NODE_COUNT + slot] = targetNode;
  if (slot % 256u == 0u && slot / 256u < FRONTIER_MAXIMUM_X) {
    atomicAdd(&nextDispatch[nextDispatchOffset], 1u);
  }
}`}function y(e,t){let{controlWordCount:n,dispatchSlotCount:r}=p(e.maxRounds);return`if (index < ${n}u) {
    var controlValue = 0u;
    if (index == 1u) { controlValue = ${t.limit}; }
    if (index == 2u) { controlValue = ${t.parameter0??`0u`}; }
    if (index == 3u) { controlValue = ${t.parameter1??`0u`}; }
    control[controlOffset + ${e.controlWordOffset}u + index] = controlValue;
  }
  if (index < ${r*f}u) {
    let slot = index / ${f}u;
    let slotWord = index % ${f}u;
    let word = (${e.dispatchSlotOffset}u + slot) * 64u + slotWord;
    // x starts at 0 (pushes grow it), y and z at 1; word 3 is the slot's round index.
    var evenValue = select(1u, 0u, slotWord == 0u);
    var oddValue = evenValue;
    if (slotWord == 3u) {
      evenValue = 2u * slot;
      oddValue = 2u * slot + 1u;
    }
    dispatchEven[dispatchEvenOffset + word] = evenValue;
    dispatchOdd[dispatchOddOffset + word] = oddValue;
  }`}function b(e){let{controlWordCount:t,dispatchSlotCount:n}=p(e.maxRounds);return Math.max(t,n*f)}function x(e){let t=`controlOffset + ${e.controlWordOffset}u`;return`let frontierLimit = control[${t} + 1u];
  var frontierRoundCount = 0u;
  for (var roundIndex = 0u; roundIndex < ${e.maxRounds}u; roundIndex++) {
    if (roundIndex >= frontierLimit || control[${t} + 4u + roundIndex] == 0u) {
      break;
    }
    frontierRoundCount++;
  }
  frontierRoundCount = max(frontierRoundCount, min(frontierLimit, 1u));
  let frontierConverged = select(
    0u,
    1u,
    frontierLimit > 0u && control[${t} + 0u] == 0u
  );`}function S(t,n){let{state:r,phase:i,round:a,localIterations:o}=n,{nodeCount:s}=r,c=_(t,r,i,a+1),u=[...n.bindings,...c];e(u.length<=8);let d=t.device.limits.maxComputeWorkgroupsPerDimension,f=a%2==0?r.dispatchEven:r.dispatchOdd,p=g(i,a);return l(t,{id:n.id,operation:n.operation,record:n.record,variant:n.variant,bindings:u,invocationCount:s,guardIndex:!1,condition:{id:`${n.id}-gate`,source:`gpu`,mode:`indirect`,buffer:f.buffer,byteOffset:f.byteOffset+p},extraResources:[{buffer:f,usage:`indirect`}],declarations:`${n.declarations??``}
${v(i,s,d)}
const FRONTIER_LOCAL_ITERATIONS: u32 = ${o}u;

var<workgroup> frontierLocalQueue: array<u32, ${2*256}>;
var<workgroup> frontierLocalCount: array<atomic<u32>, 2>;
var<workgroup> frontierSharedCount: u32;
var<private> frontierPushToLocal: bool;
var<private> frontierLocalWriteHalf: u32;

fn push(targetNode: u32) {
  if (frontierPushToLocal) {
    let slot = atomicAdd(&frontierLocalCount[frontierLocalWriteHalf], 1u);
    if (slot < 256u) {
      frontierLocalQueue[frontierLocalWriteHalf * 256u + slot] = targetNode;
      return;
    }
  }
  frontierPush(targetNode);
}

${n.visitSource}`,body:`// This round's index is one below the round of the slot it pushes into (rule 12: it is data).
  let currentRound = frontierNextRound() - 1u;
  let currentQueueBase = (currentRound & 1u) * FRONTIER_QUEUE_NODE_COUNT;
  if (localInvocationIndex == 0u) {
    frontierSharedCount = atomicLoad(&control[controlOffset + FRONTIER_ROUND_COUNTS_WORD + currentRound]);
  }
  let queueLength = workgroupUniformLoad(&frontierSharedCount);
  let dispatchedGroupCount = min(
    (queueLength + 255u) / 256u,
    FRONTIER_MAXIMUM_X
  );
  for (var chunk = workgroupId.x; chunk * 256u < queueLength; chunk += dispatchedGroupCount) {
    // Hop 0 reads the global queue chunk; hop k reads the local queue written by hop k - 1.
    frontierPushToLocal = FRONTIER_LOCAL_ITERATIONS > 1u;
    frontierLocalWriteHalf = 1u;
    let queueIndex = chunk * 256u + localInvocationIndex;
    if (queueIndex < queueLength) {
      visit(queues[queuesOffset + currentQueueBase + queueIndex]);
    }
    for (var hop = 1u; hop < FRONTIER_LOCAL_ITERATIONS; hop++) {
      workgroupBarrier();
      let readHalf = hop & 1u;
      if (localInvocationIndex == 0u) {
        frontierSharedCount = min(atomicLoad(&frontierLocalCount[readHalf]), 256u);
        atomicStore(&frontierLocalCount[readHalf], 0u);
      }
      let localLength = workgroupUniformLoad(&frontierSharedCount);
      if (localLength == 0u) { break; }
      // The last hop pushes straight to the next global round.
      frontierPushToLocal = hop + 1u < FRONTIER_LOCAL_ITERATIONS;
      frontierLocalWriteHalf = (hop + 1u) & 1u;
      if (localInvocationIndex < localLength) {
        visit(frontierLocalQueue[readHalf * 256u + localInvocationIndex]);
      }
    }
    workgroupBarrier();
  }
  ${n.epilogueSource??``}`})}var C=`GPUNetworkReachability`,w=`
const INFINITY_BITS: u32 = 0x7f800000u;
const NONE: u32 = 0xffffffffu;`,T=`
var<workgroup> counterLow: atomic<u32>;
var<workgroup> counterHigh: atomic<u32>;

// Adds to this workgroup's 64-bit partial count.
fn addCount(count: u32) {
  if (count == 0u) { return; }
  let previous = atomicAdd(&counterLow, count);
  if (previous + count < previous) { atomicAdd(&counterHigh, 1u); }
}`;function E(e){return`if (localInvocationIndex == 0u) {
    let partialLow = atomicLoad(&counterLow);
    let partialHigh = atomicLoad(&counterHigh);
    if (partialLow != 0u || partialHigh != 0u) {
      let previous = atomicAdd(&control[controlOffset + ${e}u], partialLow);
      var carry = partialHigh;
      if (previous + partialLow < previous) { carry += 1u; }
      if (carry != 0u) { atomicAdd(&control[controlOffset + ${e+1}u], carry); }
    }
  }`}function D(e,t){return t?`min(activeIterations[activeIterationsOffset], ${e}u)`:`${e}u`}function O(e,t){let{state:n,phase:r}=t,i=[{name:`costBits`,view:t.costs,type:`u32`,access:`read_write`},{name:`queuedRound`,view:n.queuedRound,type:`u32`,access:`read_write`},{name:`control`,view:n.control,type:`u32`,access:`read_write`},{name:`dispatchEven`,view:n.dispatchEven,type:`u32`,access:`read_write`},{name:`dispatchOdd`,view:n.dispatchOdd,type:`u32`,access:`read_write`}];return t.predecessors&&i.push({name:`predecessors`,view:t.predecessors,type:`u32`,access:`read_write`}),t.activeIterations&&i.push({name:`activeIterations`,view:t.activeIterations,type:`u32`,access:`read`}),t.costLimit&&i.push({name:`costLimit`,view:t.costLimit,type:`f32`,access:`read`}),l(e,{id:t.id,operation:C,variant:`initialize`,record:{datum:{keySpace:`node`}},bindings:i,invocationCount:Math.max(t.nodeCount,b(r),t.counterWordOffset===void 0?0:4),declarations:`${w}
const NODE_COUNT: u32 = ${t.nodeCount}u;`,body:`${y(r,{limit:D(r.maxRounds,!!t.activeIterations),parameter0:t.costLimit?`bitcast<u32>(costLimit[costLimitOffset])`:void 0})}
  ${t.counterWordOffset===void 0?``:`if (index < 4u) { control[controlOffset + ${t.counterWordOffset}u + index] = 0u; }`}
  if (index < NODE_COUNT) {
    costBits[costBitsOffset + index] = INFINITY_BITS;
    // Stamps never need clearing within one encoding, but a previous encoding left them dirty.
    queuedRound[queuedRoundOffset + index] = 0u;
    ${t.predecessors?`predecessors[predecessorsOffset + index] = NONE;`:``}
  }`})}function k(e){return e===void 0?``:typeof e==`number`?`if (index >= ${e}u) { return; }`:`if (index >= sourceCount[sourceCountOffset]) { return; }`}function A(e,t){let n=[{name:`sources`,view:t.sources,type:`u32`,access:`read`}];return t.sourceCosts&&n.push({name:`sourceCosts`,view:t.sourceCosts,type:`f32`,access:`read`}),typeof t.sourceCount==`object`&&n.push({name:`sourceCount`,view:t.sourceCount,type:`u32`,access:`read`}),n.push({name:`costBits`,view:t.costs,type:`atomic<u32>`,access:`read_write`},..._(e,t.state,t.phase,0)),l(e,{id:t.id,operation:C,variant:`seed`,record:{datum:{keySpace:`node`}},bindings:n,invocationCount:t.sources.length,declarations:`${w}
const NODE_COUNT: u32 = ${t.nodeCount}u;
${v(t.phase,t.nodeCount,e.device.limits.maxComputeWorkgroupsPerDimension)}`,body:`${k(t.sourceCount)}
  let node = sources[sourcesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${t.sourceCosts?`sourceCosts[sourceCostsOffset + index]`:`0.0`};
  if (!(cost >= 0.0)) { return; }
  ${t.hasCostLimit?`if (cost > bitcast<f32>(atomicLoad(&control[controlOffset + FRONTIER_PARAMETER_0_WORD]))) { return; }`:``}
  // Fold -0.0 so the u32 bit order of non-negative floats matches numeric order.
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  atomicMin(&costBits[costBitsOffset + node], bits);
  frontierPush(node);`})}function j(e,t){let{csr:n,laneNodeCount:r,counterWordOffset:i}=t,a=i!==void 0;return S(e,{id:t.id,operation:C,record:t.record,variant:`${r===void 0?`relax`:`relax-lanes`}${a?`-counted`:``}`,epilogueSource:a?E(i):void 0,state:t.state,phase:t.phase,round:t.round,localIterations:t.localIterations,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`weights`,view:n.weights,type:`f32`,access:`read`},{name:`costBits`,view:t.costs,type:`atomic<u32>`,access:`read_write`}],declarations:`${w}
const NODE_COUNT: u32 = ${r??t.nodeCount}u;
const EDGE_COUNT: u32 = ${n.neighbors.length}u;${a?T:``}`,visitSource:`fn visit(node: u32) {
  let sourceCost = bitcast<f32>(atomicLoad(&costBits[costBitsOffset + node]));
  ${r===void 0?`let csrNode = node; let laneBase = 0u;`:`let laneBase = (node / NODE_COUNT) * NODE_COUNT; let csrNode = node - laneBase;`}
  let firstEdge = min(offsets[offsetsOffset + csrNode], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + csrNode + 1u], EDGE_COUNT);
  ${t.hasCostLimit?`let costLimitValue = bitcast<f32>(atomicLoad(&control[controlOffset + FRONTIER_PARAMETER_0_WORD]));`:``}
  ${a?`addCount(select(0u, lastEdge - firstEdge, lastEdge > firstEdge));`:``}
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    let targetCsrNode = neighbors[neighborsOffset + edgeIndex];
    let weight = weights[weightsOffset + edgeIndex];
    if (targetCsrNode >= NODE_COUNT || !(weight >= 0.0)) { continue; }
    let targetNode = laneBase + targetCsrNode;
    let candidate = sourceCost + weight;
    ${t.hasCostLimit?`if (candidate > costLimitValue) { continue; }`:``}
    let candidateBits = select(bitcast<u32>(candidate), 0u, candidate == 0.0);
    if (candidateBits >= INFINITY_BITS) { continue; }
    if (candidateBits < atomicLoad(&costBits[costBitsOffset + targetNode])) {
      if (atomicMin(&costBits[costBitsOffset + targetNode], candidateBits) > candidateBits) {
        push(targetNode);
      }
    }
  }
}`})}function M(e,t){let n=[{name:`control`,view:t.state.control,type:`u32`,access:`read`}],r=[];if(t.counters?.relaxationCount){n.push({name:`relaxationCount`,view:t.counters.relaxationCount,type:`u32`,access:`read_write`});let e=t.counters.wordOffset;r.push(`relaxationCount[relaxationCountOffset] = control[controlOffset + ${e}u];`,`relaxationCount[relaxationCountOffset + 1u] = control[controlOffset + ${e+1}u];`)}if(t.counters?.reachedEdgeCount){n.push({name:`reachedEdgeCount`,view:t.counters.reachedEdgeCount,type:`u32`,access:`read_write`});let e=t.counters.wordOffset+2;r.push(`reachedEdgeCount[reachedEdgeCountOffset] = control[controlOffset + ${e}u];`,`reachedEdgeCount[reachedEdgeCountOffset + 1u] = control[controlOffset + ${e+1}u];`)}return t.converged&&n.push({name:`converged`,view:t.converged,type:`u32`,access:`read_write`}),t.iterationCount&&n.push({name:`iterationCount`,view:t.iterationCount,type:`u32`,access:`read_write`}),l(e,{id:t.id,operation:C,variant:`finalize`,bindings:n,invocationCount:1,body:`${x(t.phase)}
  ${t.converged?`converged[convergedOffset] = frontierConverged${t.tiePhase?` & select(0u, 1u, control[controlOffset + ${t.tiePhase.controlWordOffset+0}u] == 0u)`:``};`:``}
  ${t.iterationCount?`iterationCount[iterationCountOffset] = frontierRoundCount;`:``}
  ${r.join(`
  `)}`})}function N(e,t){let{csr:n}=t;return l(e,{id:t.id,operation:C,variant:`reached-edges`,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`control`,view:t.state.control,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,guardIndex:!1,declarations:`${w}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const LANE_NODE_COUNT: u32 = ${t.laneNodeCount??t.nodeCount}u;
const EDGE_COUNT: u32 = ${n.neighbors.length}u;${T}`,body:`if (index < NODE_COUNT && bitcast<u32>(costs[costsOffset + index]) < INFINITY_BITS) {
    let csrNode = index % LANE_NODE_COUNT;
    let firstEdge = min(offsets[offsetsOffset + csrNode], EDGE_COUNT);
    let lastEdge = min(offsets[offsetsOffset + csrNode + 1u], EDGE_COUNT);
    var degree = 0u;
    for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
      if (neighbors[neighborsOffset + edgeIndex] < LANE_NODE_COUNT) { degree++; }
    }
    addCount(degree);
  }
  workgroupBarrier();
  ${E(t.counterWordOffset+2)}`})}function P(e,t){let{csr:n}=t,r=[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`weights`,view:n.weights,type:`f32`,access:`read`},{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`predecessors`,view:t.predecessors,type:`atomic<u32>`,access:`read_write`}];return t.tieBits&&r.push({name:`tieBits`,view:t.tieBits,type:`atomic<u32>`,access:`read_write`}),t.labels&&r.push({name:`labels`,view:t.labels,type:`u32`,access:`read`}),l(e,{id:t.id,operation:C,variant:t.labels?`predecessors-labelled`:`predecessors`,record:{datum:{keySpace:`node`}},bindings:r,invocationCount:t.nodeCount,declarations:`${w}
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
    if (bitcast<u32>(targetCost) >= INFINITY_BITS || sourceCost + weight != targetCost) { continue; }
    ${t.labels?`if (labels[labelsOffset + index] != labels[labelsOffset + targetNode]) { continue; }`:``}
    if (sourceCost < targetCost) {
      atomicMin(&predecessors[predecessorsOffset + targetNode], index);
    }${t.tieBits?` else if (sourceCost == targetCost) {
      atomicOr(&tieBits[tieBitsOffset + (edgeIndex >> 5u)], 1u << (edgeIndex & 31u));
    }`:``}
  }`})}function F(e,t){return l(e,{id:t.id,operation:C,variant:`bands`,record:{datum:{keySpace:`node`},formula:`gpu-network.reachability.bands`},bindings:[{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`thresholds`,view:t.thresholds,type:`f32`,access:`read`},{name:`bands`,view:t.bands,type:`u32`,access:`read_write`}],invocationCount:t.nodeCount,declarations:`${w}
const THRESHOLD_COUNT: u32 = ${t.thresholds.length}u;`,body:`let cost = costs[costsOffset + index];
  var band = NONE;
  if (bitcast<u32>(cost) < INFINITY_BITS) {
    var count = 0u;
    for (var thresholdIndex = 0u; thresholdIndex < THRESHOLD_COUNT; thresholdIndex++) {
      if (thresholds[thresholdsOffset + thresholdIndex] < cost) { count++; }
    }
    if (count < THRESHOLD_COUNT) { band = count; }
  }
  bands[bandsOffset + index] = band;`})}var I=`
fn isTieEdge(edgeIndex: u32) -> bool {
  return ((tieBits[tieBitsOffset + (edgeIndex >> 5u)] >> (edgeIndex & 31u)) & 1u) != 0u;
}

fn hasTieOutEdge(node: u32) -> bool {
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (isTieEdge(edgeIndex)) { return true; }
  }
  return false;
}`;function L(e,t){let{state:n,tiePhase:r}=t,i=[{name:`levels`,view:t.levels,type:`u32`,access:`read_write`},{name:`tieBits`,view:t.tieBits,type:`u32`,access:`read_write`},{name:`control`,view:n.control,type:`u32`,access:`read_write`},{name:`dispatchEven`,view:n.dispatchEven,type:`u32`,access:`read_write`},{name:`dispatchOdd`,view:n.dispatchOdd,type:`u32`,access:`read_write`}];return t.unresolvedCount&&i.push({name:`unresolvedCount`,view:t.unresolvedCount,type:`u32`,access:`read_write`}),l(e,{id:t.id,operation:C,variant:`tie-initialize`,record:{datum:{keySpace:`node`}},bindings:i,invocationCount:Math.max(t.nodeCount,t.tieBits.length,b(r)),declarations:`${w}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const TIE_WORD_COUNT: u32 = ${t.tieBits.length}u;`,body:`${y(r,{limit:`${r.maxRounds}u`})}
  if (index < NODE_COUNT) { levels[levelsOffset + index] = NONE; }
  if (index < TIE_WORD_COUNT) { tieBits[tieBitsOffset + index] = 0u; }${t.unresolvedCount?`
  if (index == 0u) { unresolvedCount[unresolvedCountOffset] = 0u; }`:``}`})}function R(e,t){let n=[{name:`sources`,view:t.sources,type:`u32`,access:`read`}];return t.labels&&n.push({name:`labels`,view:t.labels,type:`u32`,access:`read`}),t.sourceCosts&&n.push({name:`sourceCosts`,view:t.sourceCosts,type:`f32`,access:`read`}),typeof t.sourceCount==`object`&&n.push({name:`sourceCount`,view:t.sourceCount,type:`u32`,access:`read`}),n.push({name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`levels`,view:t.levels,type:`u32`,access:`read_write`}),t.hasCostLimit&&n.push({name:`control`,view:t.state.control,type:`u32`,access:`read`}),l(e,{id:t.id,operation:C,variant:`tie-roots`,record:{datum:{keySpace:`node`}},bindings:n,invocationCount:t.sources.length,declarations:`${w}
const NODE_COUNT: u32 = ${t.nodeCount}u;`,body:`${k(t.sourceCount)}
  let node = sources[sourcesOffset + index];
  if (node >= NODE_COUNT) { return; }
  let cost = ${t.sourceCosts?`sourceCosts[sourceCostsOffset + index]`:`0.0`};
  if (!(cost >= 0.0)) { return; }
  ${t.hasCostLimit?`if (cost > bitcast<f32>(control[controlOffset + ${t.phase.controlWordOffset+2}u])) { return; }`:``}
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= INFINITY_BITS) { return; }
  ${t.labels?`if (labels[labelsOffset + node] != index) { return; }`:``}
  if (bits == bitcast<u32>(costs[costsOffset + node])) {
    levels[levelsOffset + node] = 0u;
  }`})}function z(e,t){return l(e,{id:t.id,operation:C,variant:`tie-seed`,record:{datum:{keySpace:`node`}},bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`tieBits`,view:t.tieBits,type:`u32`,access:`read`},{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`levels`,view:t.levels,type:`u32`,access:`read_write`},..._(e,t.state,t.tiePhase,0)],invocationCount:t.nodeCount,declarations:`${w}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;
${I}
${v(t.tiePhase,t.nodeCount,e.device.limits.maxComputeWorkgroupsPerDimension)}`,body:`if (predecessors[predecessorsOffset + index] != NONE) {
    levels[levelsOffset + index] = 0u;
  }
  if (levels[levelsOffset + index] == 0u && hasTieOutEdge(index)) {
    frontierPush(index);
  }`})}function B(e,t){return S(e,{id:t.id,operation:C,record:t.record,variant:`tie-level`,state:t.state,phase:t.tiePhase,round:t.round,localIterations:t.localIterations,bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`tieBits`,view:t.tieBits,type:`u32`,access:`read`},{name:`levels`,view:t.levels,type:`atomic<u32>`,access:`read_write`}],declarations:`${w}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;
${I}`,visitSource:`fn visit(node: u32) {
  let level = atomicLoad(&levels[levelsOffset + node]);
  if (level == NONE) { return; }
  let nextLevel = level + 1u;
  let firstEdge = min(offsets[offsetsOffset + node], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + node + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (!isTieEdge(edgeIndex)) { continue; }
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT) { continue; }
    if (nextLevel < atomicLoad(&levels[levelsOffset + targetNode])) {
      if (atomicMin(&levels[levelsOffset + targetNode], nextLevel) > nextLevel && hasTieOutEdge(targetNode)) {
        push(targetNode);
      }
    }
  }
}`})}function V(e,t){return l(e,{id:t.id,operation:C,variant:`tie-predecessors`,record:{datum:{keySpace:`node`}},bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`tieBits`,view:t.tieBits,type:`u32`,access:`read`},{name:`levels`,view:t.levels,type:`u32`,access:`read`},{name:`predecessors`,view:t.predecessors,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:`${w}
const NODE_COUNT: u32 = ${t.nodeCount}u;
const EDGE_COUNT: u32 = ${t.edgeCount}u;
${I}`,body:`let level = levels[levelsOffset + index];
  if (level == NONE) { return; }
  let firstEdge = min(offsets[offsetsOffset + index], EDGE_COUNT);
  let lastEdge = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edgeIndex = firstEdge; edgeIndex < lastEdge; edgeIndex++) {
    if (!isTieEdge(edgeIndex)) { continue; }
    let targetNode = neighbors[neighborsOffset + edgeIndex];
    if (targetNode >= NODE_COUNT) { continue; }
    let targetLevel = levels[levelsOffset + targetNode];
    if (targetLevel != 0u && targetLevel != NONE && level + 1u == targetLevel) {
      atomicMin(&predecessors[predecessorsOffset + targetNode], index);
    }
  }`})}function H(e,t){return l(e,{id:t.id,operation:C,variant:`unresolved-count`,bindings:[{name:`costs`,view:t.costs,type:`f32`,access:`read`},{name:`levels`,view:t.levels,type:`u32`,access:`read`},{name:`predecessors`,view:t.predecessors,type:`u32`,access:`read`},{name:`unresolvedCount`,view:t.unresolvedCount,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.nodeCount,declarations:w,body:`if (bitcast<u32>(costs[costsOffset + index]) >= INFINITY_BITS) { return; }
  if (predecessors[predecessorsOffset + index] != NONE) { return; }
  if (levels[levelsOffset + index] == 0u) { return; }
  atomicAdd(&unresolvedCount[unresolvedCountOffset], 1u);`})}var U=4294967295,W=1024,G=1024;function K(e){return i(e,8)}var q=64,J=4,Y=32,X=class{id;props;maximumIterations;localIterations;maximumTieIterations;laneCount;constructor(e){this.id=e.id??`network-reachability`,this.props=e,this.maximumIterations=e.maximumIterations??q,this.localIterations=e.localIterations??Y,this.maximumTieIterations=e.maximumTieIterations??J,this.laneCount=e.laneCount??1;let{id:t}=this,{output:n}=e,{nodeCount:i}=u(t,e.topology,{weights:!0});for(let[r,i]of[[`startNodes`,e.startNodes],[`activeIterations`,e.activeIterations],[`output.predecessors`,n.predecessors],[`output.bands`,n.bands],[`output.bandCounts`,n.bandCounts],[`output.status.converged`,n.status?.converged],[`output.unresolvedCount`,n.unresolvedCount],[`output.status.iterationCount`,n.status?.iterationCount],[`output.status.relaxationCount`,n.status?.relaxationCount],[`output.status.reachedEdgeCount`,n.status?.reachedEdgeCount]])i&&s(i,[`uint32`],`${t} ${r}`);typeof e.startCount==`object`&&s(e.startCount,[`uint32`],`${t} startCount`);for(let[r,i]of[[`startCosts`,e.startCosts],[`costLimit`,e.costLimit],[`output.costs`,n.costs],[`costBreaks`,e.costBreaks]])i&&s(i,[`float32`],`${t} ${r}`);let a=n.costs.length;if(a<1)throw Error(`${t} output.costs must contain at least one node`);if(r(t,`laneCount`,this.laneCount),this.laneCount>1&&(n.predecessors||n.unresolvedCount))throw Error(`${t} laneCount > 1 does not support output.predecessors or output.unresolvedCount`);if(a%this.laneCount!==0)throw Error(`${t} output.costs length must be a multiple of laneCount`);if(i!==a/this.laneCount)throw Error(`${t} topology must contain one node per output.costs row per lane`);if(e.startCosts&&e.startCosts.length!==e.startNodes.length)throw Error(`${t} startCosts length must equal startNodes length`);typeof e.startCount==`number`&&r(t,`startCount`,e.startCount,0,e.startNodes.length);for(let[r,i]of[[`startCount`,typeof e.startCount==`object`?e.startCount:void 0],[`activeIterations`,e.activeIterations],[`costLimit`,e.costLimit],[`output.status.converged`,n.status?.converged],[`output.unresolvedCount`,n.unresolvedCount],[`output.status.iterationCount`,n.status?.iterationCount]])if(i&&i.length!==1)throw Error(`${t} ${r} must contain exactly one row`);for(let[e,r]of[[`output.status.relaxationCount`,n.status?.relaxationCount],[`output.status.reachedEdgeCount`,n.status?.reachedEdgeCount]])if(r&&r.length!==2)throw Error(`${t} ${e} must contain exactly two rows`);for(let[e,r]of[[`output.predecessors`,n.predecessors],[`output.bands`,n.bands]])if(r&&r.length!==a)throw Error(`${t} ${e} length must equal the node count`);if(!!(n.bands||n.bandCounts)!=!!e.costBreaks)throw Error(`${t} costBreaks is required exactly when output.bands or output.bandCounts is given`);if(e.costBreaks&&e.costBreaks.length<1)throw Error(`${t} costBreaks must contain at least one break`);if(n.bandCounts&&n.bandCounts.length!==e.costBreaks?.length)throw Error(`${t} output.bandCounts length must equal costBreaks length`);if(r(t,`maximumIterations`,this.maximumIterations,1,W),r(t,`localIterations`,this.localIterations,1,64),r(t,`maximumTieIterations`,this.maximumTieIterations,1,G),n.unresolvedCount&&!n.predecessors)throw Error(`${t} output.unresolvedCount requires output.predecessors`);let o=[n.costs,n.predecessors,n.bands,n.bandCounts,n.status?.converged,n.unresolvedCount,n.status?.iterationCount,n.status?.relaxationCount,n.status?.reachedEdgeCount].filter(e=>e!==void 0).map(e=>e.buffer),c=[...d(e.topology),e.startNodes,e.startCosts,typeof e.startCount==`object`?e.startCount:void 0,e.activeIterations,e.costLimit,e.costBreaks].filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(o).size!==o.length||o.some(e=>c.includes(e)))throw Error(`${t} outputs must use separate buffers from each other and from inputs`)}getCommandNodes(e){let{id:t,props:r,maximumIterations:i,localIterations:s}=this,{output:l}=r,{offsets:u,neighbors:d,weights:f}=r.topology;n(t,e,Z(r));let g=l.costs.length;a(t,`output.costs rows`,g,K(e.device));let _={offsets:u,neighbors:d,weights:f},v={maxRounds:i,stampBase:0,controlWordOffset:0,dispatchSlotOffset:0},y=p(i),b=p(this.maximumTieIterations),x={maxRounds:this.maximumTieIterations,stampBase:h(v),controlWordOffset:y.controlWordCount,dispatchSlotOffset:y.dispatchSlotCount},S=!!l.predecessors,C=!!l.status?.relaxationCount,w=!!l.status?.reachedEdgeCount,T=C||w,E=y.controlWordCount+(S?b.controlWordCount:0),D=T?E:void 0,k=m(e,t,{nodeCount:g,controlWordCount:E+(T?4:0),dispatchSlotCount:y.dispatchSlotCount+(S?b.dispatchSlotCount:0)}),I=d.length,U=S?o(e,`${t}-tie-levels`,`uint32`,g):void 0,W=S?o(e,`${t}-tie-bits`,`uint32`,Math.max(1,Math.ceil(I/32))):void 0,G=!!r.costLimit,q=[O(e,{id:`${t}-initialize`,nodeCount:g,state:k,phase:v,costs:l.costs,predecessors:l.predecessors,activeIterations:r.activeIterations,costLimit:r.costLimit,counterWordOffset:D})];r.startNodes.length>0&&q.push(A(e,{id:`${t}-seed`,nodeCount:g,state:k,phase:v,sources:r.startNodes,sourceCosts:r.startCosts,sourceCount:r.startCount,hasCostLimit:G,costs:l.costs}));for(let n=0;n<i;n++)q.push(j(e,{id:`${t}-relax-${n}`,record:{loop:{id:`${t}-relax`,iteration:n,phase:`gated`,maximumIterations:i},datum:{keySpace:`node`,relation:`frontier-queue`},formula:`gpu-network.reachability.relax`,attrs:{chaotic:!0}},nodeCount:g,state:k,phase:v,round:n,localIterations:s,csr:_,hasCostLimit:G,costs:l.costs,laneNodeCount:this.laneCount>1?g/this.laneCount:void 0,counterWordOffset:C?D:void 0}));if(w&&D!==void 0&&q.push(N(e,{id:`${t}-reached-edges`,nodeCount:g,state:k,csr:_,costs:l.costs,laneNodeCount:this.laneCount>1?g/this.laneCount:void 0,counterWordOffset:D})),l.predecessors&&U&&W){q.push(L(e,{id:`${t}-tie-initialize`,nodeCount:g,state:k,tiePhase:x,levels:U,tieBits:W,unresolvedCount:l.unresolvedCount}),P(e,{id:`${t}-predecessors`,nodeCount:g,csr:_,costs:l.costs,predecessors:l.predecessors,tieBits:W})),r.startNodes.length>0&&q.push(R(e,{id:`${t}-tie-roots`,nodeCount:g,state:k,phase:v,sources:r.startNodes,sourceCosts:r.startCosts,sourceCount:r.startCount,hasCostLimit:G,costs:l.costs,levels:U})),q.push(z(e,{id:`${t}-tie-seed`,nodeCount:g,edgeCount:I,state:k,tiePhase:x,offsets:u,tieBits:W,predecessors:l.predecessors,levels:U}));for(let n=0;n<this.maximumTieIterations;n++)q.push(B(e,{id:`${t}-tie-level-${n}`,record:{loop:{id:`${t}-tie-level`,iteration:n,phase:`gated`,maximumIterations:this.maximumTieIterations},datum:{keySpace:`node`,relation:`frontier-queue`},attrs:{chaotic:!0}},nodeCount:g,edgeCount:I,state:k,tiePhase:x,round:n,localIterations:s,offsets:u,neighbors:d,tieBits:W,levels:U}));q.push(V(e,{id:`${t}-tie-predecessors`,nodeCount:g,edgeCount:I,offsets:u,neighbors:d,tieBits:W,levels:U,predecessors:l.predecessors}))}if(l.predecessors&&U&&l.unresolvedCount&&q.push(H(e,{id:`${t}-unresolved-count`,nodeCount:g,costs:l.costs,levels:U,predecessors:l.predecessors,unresolvedCount:l.unresolvedCount})),(l.status?.converged||l.status?.iterationCount||l.status?.relaxationCount||l.status?.reachedEdgeCount)&&q.push(M(e,{id:`${t}-finalize`,state:k,phase:v,tiePhase:S?x:void 0,converged:l.status?.converged,iterationCount:l.status?.iterationCount,counters:D===void 0?void 0:{wordOffset:D,relaxationCount:l.status?.relaxationCount,reachedEdgeCount:l.status?.reachedEdgeCount}})),r.costBreaks&&(l.bands||l.bandCounts)){let n=l.bands??o(e,`${t}-bands-scratch`,`uint32`,g);if(q.push(F(e,{id:`${t}-bands`,nodeCount:g,costs:l.costs,thresholds:r.costBreaks,bands:n})),l.bandCounts){let i=r.costBreaks.length;q.push(...new c({id:`${t}-band-counts`,input:n,output:l.bandCounts,domain:[0,i]}).getCommandNodes(e))}}return q}};function Z(e){let{output:t}=e;return[...d(e.topology),e.startNodes,e.startCosts,typeof e.startCount==`object`?e.startCount:void 0,e.activeIterations,e.costLimit,e.costBreaks,t.costs,t.predecessors,t.bands,t.bandCounts,t.status?.converged,t.unresolvedCount,t.status?.iterationCount,t.status?.relaxationCount,t.status?.reachedEdgeCount]}export{v as _,P as a,V as c,S as d,m as f,h as g,p as h,U as i,R as l,x as m,W as n,L as o,_ as p,G as r,B as s,X as t,z as u,b as v,y};