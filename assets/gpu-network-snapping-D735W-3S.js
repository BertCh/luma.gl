import{E as e,S as t,p as n}from"./analysis-frame-Blu0tYdw.js";import{_ as r,d as i,h as a,l as o,v as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{n as u,s as d}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as f}from"./gpu-nearest-feature-join-DscXfyZB.js";import{i as p,n as m}from"./network-topology-BXzNuj-N.js";import{t as h}from"./network-spatial-context-C-8PxYjz.js";import{t as g}from"./network-accessibility-passes-B3hrEHbU.js";function _(e,t){let{id:n,operation:r,offsets:i,edgeSources:a,nodeCount:o}=t;return[u(e,{id:`${n}-clear`,operation:r,view:a,type:`u32`,value:`${g}u`}),d(e,{id:n,operation:r,variant:`edge-sources`,bindings:[{name:`offsets`,view:i,type:`u32`,access:`read`},{name:`edgeSources`,view:a,type:`u32`,access:`read_write`}],invocationCount:o,declarations:`const EDGE_COUNT: u32 = ${a.length}u;`,body:`let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    edgeSources[edgeSourcesOffset + edge] = index;
  }`})]}var v=Object.freeze({...n,measure:`distance`}),y=`GPUNetworkSnapping`,b=4294967295;function x(e){return o(e,16)}var S=class{id;props;pointCount;nodeCount;edgeCount;startDirection;spatialContext;constructor(t){this.id=t.id??`network-snapping`,this.props=t,this.startDirection=t.startDirection??`both`;let{id:n}=this;this.spatialContext=e(n,t.spatialContext,v);let{output:r}=t,{nodeCount:o,edgeCount:s}=p(n,t.topology,{nodePositions:!0});l(t.positions,[`float32x2`],`${n} positions`),r.snappedPositions&&l(r.snappedPositions,[`float32x2`],`${n} output.snappedPositions`);for(let[e,i]of[[`maximumSnapDistance`,t.maximumSnapDistance],[`output.snapFractions`,r.snapFractions],[`output.snapDistances`,r.snapDistances],[`output.sourceCosts`,r.sourceCosts],[`output.targetCosts`,r.targetCosts],[`output.startCosts`,r.startCosts]])i&&l(i,[`float32`],`${n} ${e}`);for(let[e,t]of[[`output.snappedEdges`,r.snappedEdges],[`output.startNodes`,r.startNodes],[`output.status.overflow`,r.status?.overflow]])t&&l(t,[`uint32`],`${n} ${e}`);this.pointCount=t.positions.length,this.nodeCount=o,this.edgeCount=s;for(let[e,t]of[[`output.snappedEdges`,r.snappedEdges],[`output.snapFractions`,r.snapFractions],[`output.snapDistances`,r.snapDistances],[`output.sourceCosts`,r.sourceCosts],[`output.targetCosts`,r.targetCosts],[`output.snappedPositions`,r.snappedPositions]])if(t&&t.length!==this.pointCount)throw Error(`${n} ${e} length must equal the position count`);for(let[e,t]of[[`output.startNodes`,r.startNodes],[`output.startCosts`,r.startCosts]])if(t&&t.length!==2*this.pointCount)throw Error(`${n} ${e} length must be twice the position count`);if(!!r.startNodes!=!!r.startCosts)throw Error(`${n} output.startNodes and output.startCosts must be given together`);for(let[e,i]of[[`maximumSnapDistance`,t.maximumSnapDistance],[`output.status.overflow`,r.status?.overflow]])if(i&&i.length!==1)throw Error(`${n} ${e} must contain exactly one row`);if(t.candidateCapacity!==void 0){if(i(n,`candidateCapacity`,t.candidateCapacity),!t.maximumSnapDistance)throw Error(`${n} candidateCapacity requires maximumSnapDistance`);if(!r.status?.overflow)throw Error(`${n} candidateCapacity requires output.status.overflow`)}if(r.status?.overflow&&t.candidateCapacity===void 0)throw Error(`${n} output.status.overflow requires candidateCapacity`);let c=w(t);a(n,c,C(t));let u=c.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(u).size!==u.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(e){let{id:n,props:i,pointCount:a,nodeCount:o,edgeCount:l}=this,{offsets:u,neighbors:p,weights:m}=i.topology,{nodePositions:v}=i.topology,{output:b}=i;r(n,e,[...C(i),...w(i)]),i.candidateCapacity!==void 0&&s(n,`topology edges with candidateCapacity`,l,x(e.device));let S=[],T=`const NODE_COUNT: u32 = ${o}u;
const EDGE_COUNT: u32 = ${l}u;
const NONE: u32 = ${g}u;`,E=c(e,`${n}-edge-sources`,`uint32`,Math.max(l,1));S.push(..._(e,{id:`${n}-edge-sources`,operation:y,offsets:u,edgeSources:E,nodeCount:o}));let D=E,O=b.snapFractions??c(e,`${n}-fractions`,`float32`,a),k=b.snapDistances??c(e,`${n}-distances`,`float32`,a),A=h(t(this.spatialContext)),j=`
fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`,M=`${j}
${A}
// Fraction of the closest point along a -> b; 0 for a degenerate edge.
fn getFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  return networkSegmentFraction(point, start, end);
}
fn getDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  return networkSegmentDistance(point, start, end, fraction);
}`,N=[{name:`points`,view:i.positions,type:`f32`,access:`read`},{name:`nodePositions`,view:v,type:`f32`,access:`read`},{name:`edgeSources`,view:D,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`}],P=[{name:`snappedEdges`,view:b.snappedEdges,type:`u32`,access:`read_write`},{name:`fractions`,view:O,type:`f32`,access:`read_write`},{name:`distances`,view:k,type:`f32`,access:`read_write`}],F=`if (bestEdge == NONE) {
    snappedEdges[snappedEdgesOffset + index] = NONE;
    fractions[fractionsOffset + index] = -1.0;
    distances[distancesOffset + index] = -1.0;
    return;
  }
  snappedEdges[snappedEdgesOffset + index] = bestEdge;
  fractions[fractionsOffset + index] = bestFraction;
  distances[distancesOffset + index] = bestDistance;`;if(i.candidateCapacity!==void 0&&i.maximumSnapDistance){let t=c(e,`${n}-segment-starts`,`float32x2`,l),r=c(e,`${n}-segment-ends`,`float32x2`,l),o=c(e,`${n}-join-rows`,`uint32`,a);l>0&&S.push(d(e,{id:`${n}-segments`,operation:y,variant:`segments`,bindings:[{name:`nodePositions`,view:v,type:`f32`,access:`read`},{name:`edgeSources`,view:D,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`},{name:`segmentStarts`,view:t,type:`f32`,access:`read_write`},{name:`segmentEnds`,view:r,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`${T}
fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`,body:`let sourceNode = edgeSources[edgeSourcesOffset + index];
  let targetNode = edgeTargets[edgeTargetsOffset + index];
  // WGSL const evaluation rejects an infinite value, so the lon/lat sentinel is built from runtime
  // bits (a var), never from a literal bitcast.
  var infinityBits = 0x7f800000u;
  var start = vec2<f32>(${this.spatialContext.coordinateSpace===`longitude-latitude`?`bitcast<f32>(infinityBits)`:`1.0e30`});
  var end = start;
  if (sourceNode < NODE_COUNT && targetNode < NODE_COUNT) {
    start = readNode(sourceNode);
    end = readNode(targetNode);
  }
  segmentStarts[segmentStartsOffset + index * 2u] = start.x;
  segmentStarts[segmentStartsOffset + index * 2u + 1u] = start.y;
  segmentEnds[segmentEndsOffset + index * 2u] = end.x;
  segmentEnds[segmentEndsOffset + index * 2u + 1u] = end.y;`})),S.push(...new f({id:`${n}-join`,spatialContext:this.spatialContext,queries:{kind:`points`,positions:i.positions},features:{kind:`segments`,starts:t,ends:r},radius:i.maximumSnapDistance,candidateCapacity:i.candidateCapacity,spatialSort:i.spatialSort,output:{nearestFeatureIds:o,status:{overflow:b.status.overflow}}}).getCommandNodes(e)),S.push(d(e,{id:`${n}-project`,operation:y,variant:`project`,bindings:[...N,{name:`joinRows`,view:o,type:`u32`,access:`read`},...P],invocationCount:a,declarations:`${T}
${M}`,body:`let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var bestEdge = joinRows[joinRowsOffset + index];
  var bestFraction = 0.0;
  var bestDistance = 0.0;
  if (bestEdge < EDGE_COUNT) {
    let sourceNode = edgeSources[edgeSourcesOffset + bestEdge];
    let targetNode = edgeTargets[edgeTargetsOffset + bestEdge];
    let start = readNode(sourceNode);
    let end = readNode(targetNode);
    bestFraction = getFraction(point, start, end);
    bestDistance = getDistance(point, start, end, bestFraction);
  } else {
    bestEdge = NONE;
  }
  ${F}`}))}else{let t=[...N,...P];i.maximumSnapDistance&&t.push({name:`maximumSnapDistance`,view:i.maximumSnapDistance,type:`f32`,access:`read`}),S.push(d(e,{id:`${n}-scan`,operation:y,variant:`scan`,record:{datum:{keySpace:`point`},formula:`gpu-network.snapping.nearest`},bindings:t,invocationCount:a,declarations:`${T}
${M}`,body:`let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  ${i.maximumSnapDistance?`let limit = maximumSnapDistance[maximumSnapDistanceOffset];`:``}
  var bestEdge = NONE;
  var bestFraction = 0.0;
  var bestDistance = 0.0;
  for (var edge = 0u; edge < EDGE_COUNT; edge++) {
    let sourceNode = edgeSources[edgeSourcesOffset + edge];
    let targetNode = edgeTargets[edgeTargetsOffset + edge];
    if (sourceNode >= NODE_COUNT || targetNode >= NODE_COUNT) {
      continue;
    }
    let start = readNode(sourceNode);
    let end = readNode(targetNode);
    let fraction = getFraction(point, start, end);
    let distance = getDistance(point, start, end, fraction);
    // Strict less keeps the smallest edge row on ties; NaN never passes.
    if (${i.maximumSnapDistance?`distance <= limit`:`distance >= 0.0`} && (bestEdge == NONE || distance < bestDistance)) {
      bestEdge = edge;
      bestFraction = fraction;
      bestDistance = distance;
    }
  }
  ${F}`}))}let I=!!(b.sourceCosts||b.targetCosts||b.startNodes),L=b.sourceCosts??(I?c(e,`${n}-source-costs`,`float32`,a):null),R=b.targetCosts??(I?c(e,`${n}-target-costs`,`float32`,a):null);if(L&&R){let t=[{name:`snappedEdges`,view:b.snappedEdges,type:`u32`,access:`read`},{name:`fractions`,view:O,type:`f32`,access:`read`},{name:`edgeSources`,view:D,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`},m?{name:`edgeCosts`,view:m,type:`f32`,access:`read`}:{name:`nodePositions`,view:v,type:`f32`,access:`read`},{name:`sourceCosts`,view:L,type:`f32`,access:`read_write`},{name:`targetCosts`,view:R,type:`f32`,access:`read_write`}];S.push(d(e,{id:`${n}-costs`,operation:y,variant:`costs`,bindings:t,invocationCount:a,declarations:`${T}
${m?``:`${j}\n${A}`}`,body:`let edge = snappedEdges[snappedEdgesOffset + index];
  if (edge >= EDGE_COUNT) {
    sourceCosts[sourceCostsOffset + index] = -1.0;
    targetCosts[targetCostsOffset + index] = -1.0;
    return;
  }
  let fraction = fractions[fractionsOffset + index];
  ${m?`let edgeCost = edgeCosts[edgeCostsOffset + edge];`:`let edgeCost = networkDistance(
    readNode(edgeSources[edgeSourcesOffset + edge]),
    readNode(edgeTargets[edgeTargetsOffset + edge])
  );`}
  sourceCosts[sourceCostsOffset + index] = fraction * edgeCost;
  targetCosts[targetCostsOffset + index] = (1.0 - fraction) * edgeCost;`}))}if(b.snappedPositions&&S.push(d(e,{id:`${n}-positions`,operation:y,variant:`positions`,bindings:[{name:`points`,view:i.positions,type:`f32`,access:`read`},{name:`nodePositions`,view:v,type:`f32`,access:`read`},{name:`edgeSources`,view:D,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`},{name:`snappedEdges`,view:b.snappedEdges,type:`u32`,access:`read`},{name:`fractions`,view:O,type:`f32`,access:`read`},{name:`snappedPositions`,view:b.snappedPositions,type:`f32`,access:`read_write`}],invocationCount:a,declarations:`${T}
${j}
${A}`,body:`var position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let edge = snappedEdges[snappedEdgesOffset + index];
  if (edge < EDGE_COUNT) {
    let start = readNode(edgeSources[edgeSourcesOffset + edge]);
    let end = readNode(edgeTargets[edgeTargetsOffset + edge]);
    position = networkInterpolate(start, end, fractions[fractionsOffset + index]);
  }
  snappedPositions[snappedPositionsOffset + index * 2u] = position.x;
  snappedPositions[snappedPositionsOffset + index * 2u + 1u] = position.y;`})),b.startNodes&&b.startCosts&&L&&R){let t=this.startDirection!==`forward`,r=this.startDirection!==`reverse`;S.push(d(e,{id:`${n}-seeds`,operation:y,variant:`seeds-${this.startDirection}`,bindings:[{name:`snappedEdges`,view:b.snappedEdges,type:`u32`,access:`read`},{name:`edgeSources`,view:D,type:`u32`,access:`read`},{name:`edgeTargets`,view:p,type:`u32`,access:`read`},{name:`sourceCosts`,view:L,type:`f32`,access:`read`},{name:`targetCosts`,view:R,type:`f32`,access:`read`},{name:`seedNodes`,view:b.startNodes,type:`u32`,access:`read_write`},{name:`seedCosts`,view:b.startCosts,type:`f32`,access:`read_write`}],invocationCount:a,declarations:T,body:`let edge = snappedEdges[snappedEdgesOffset + index];
  let isSnapped = edge < EDGE_COUNT;
  let useSource = isSnapped && ${t};
  let useTarget = isSnapped && ${r};
  let sourceNode = select(0u, edgeSources[edgeSourcesOffset + min(edge, EDGE_COUNT - 1u)], isSnapped);
  let targetNode = select(0u, edgeTargets[edgeTargetsOffset + min(edge, EDGE_COUNT - 1u)], isSnapped);
  seedNodes[seedNodesOffset + index * 2u] = select(NONE, sourceNode, useSource);
  seedCosts[seedCostsOffset + index * 2u] = select(-1.0, sourceCosts[sourceCostsOffset + index], useSource);
  seedNodes[seedNodesOffset + index * 2u + 1u] = select(NONE, targetNode, useTarget);
  seedCosts[seedCostsOffset + index * 2u + 1u] = select(-1.0, targetCosts[targetCostsOffset + index], useTarget);`}))}return S}};function C(e){return[e.positions,...m(e.topology),e.maximumSnapDistance]}function w(e){let{output:t}=e;return[t.snappedEdges,t.snapFractions,t.snapDistances,t.sourceCosts,t.targetCosts,t.snappedPositions,t.startNodes,t.startCosts,t.status?.overflow]}export{b as n,_ as r,S as t};