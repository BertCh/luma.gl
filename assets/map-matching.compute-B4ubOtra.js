import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{D as e,E as t,S as n,p as r}from"./analysis-frame-Blu0tYdw.js";import{N as i,_ as a,d as o,h as s,l as c,n as l,t as u,v as d,w as f,y as p}from"./resources-CwvhEOBe.js";import{t as m}from"./gpu-scan-BuGhTLDQ.js";import{n as h,s as g}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as _,c as v,r as y}from"./index-Bjk18V2I.js";import{t as b}from"./graph-buffers-B4_80RlX.js";import{t as x}from"./graph-swap-41Ru7LVu.js";import{t as ee}from"./summary-reader-CsgWxENf.js";import{t as S}from"./mode-kernels-Dxm2GWvI.js";import{i as C,n as w}from"./network-topology-BXzNuj-N.js";import{t as T}from"./network-spatial-context-C-8PxYjz.js";import{t as E}from"./network-workgroup-counters-CFyh5ZJ-.js";import{i as D,n as O,r as k}from"./b10-road-graph-63jDTifd.js";var A=Object.freeze({...r,measure:`distance`}),j=`GPUMapMatching`,te=3212836864;function ne(e){return c(e,12)}function M(e){return c(e,4)}function N(t){e(`GPUMapMatching`,t.spatialContext,t.sigma),e(`GPUMapMatching`,t.spatialContext,t.searchRadius);let n=new Float32Array(8);return n[0]=t.sigma,n[1]=t.beta,n[2]=t.searchRadius,n[3]=t.routeFactor??3,n[4]=t.routeSlack??50,n}var P=class{id;props;pointCount;trackCount;nodeCount;edgeCount;candidateCount;gridWidth;gridHeight;spatialContext;constructor(e){this.id=e.id??`map-matching`,this.props=e;let{id:n,props:r}={id:this.id,props:e},{output:i}=e;this.spatialContext=t(n,e.spatialContext,A);let a=C(n,e.topology,{nodePositions:!0});p(e.positions,[`float32x2`],`${n} positions`),p(e.parameters,[`float32`],`${n} parameters`);for(let[t,r]of[[`trackOffsets`,e.trackOffsets],[`output.matchedEdges`,i.matchedEdges],[`output.breaks`,i.breaks],[`output.matchedCount`,i.matchedCount],[`output.breakCount`,i.breakCount],[`output.status.overflow`,i.status?.overflow],[`output.status.routeOverflow`,i.status?.routeOverflow]])r&&p(r,[`uint32`],`${n} ${t}`);for(let[e,t]of[[`output.matchedFractions`,i.matchedFractions],[`output.matchedOffsets`,i.matchedOffsets],[`output.snapDistances`,i.snapDistances],[`output.trackLogLikelihoods`,i.trackLogLikelihoods]])t&&p(t,[`float32`],`${n} ${e}`);if(i.snappedPositions&&p(i.snappedPositions,[`float32x2`],`${n} output.snappedPositions`),this.pointCount=e.positions.length,this.trackCount=e.trackOffsets.length-1,this.nodeCount=a.nodeCount,this.edgeCount=a.edgeCount,this.candidateCount=e.candidateCount??8,this.trackCount<1)throw Error(`${n} trackOffsets needs at least one track`);if(this.pointCount<1||this.edgeCount<1||this.nodeCount<1)throw Error(`${n} needs at least one point, node and edge`);if(e.parameters.length!==8)throw Error(`${n} parameters must contain 8 rows`);if(o(n,`candidateCount`,this.candidateCount,1,8),o(n,`routeNodeBudget`,e.routeNodeBudget??64,8,128),!(e.cellSize>0)||!Number.isFinite(e.cellSize))throw Error(`${n} cellSize must be positive`);let[c,l]=e.bounds.minimum,[u,d]=e.bounds.maximum;if(!(u>c)||!(d>l))throw Error(`${n} bounds must have a positive extent`);if(this.gridWidth=Math.max(1,Math.ceil((u-c)/e.cellSize)),this.gridHeight=Math.max(1,Math.ceil((d-l)/e.cellSize)),this.gridWidth*this.gridHeight>1<<24)throw Error(`${n} cellSize yields more than 2^24 grid cells; use a larger cellSize`);e.entryCapacity!==void 0&&o(n,`entryCapacity`,e.entryCapacity);for(let[e,t,r]of[[`matchedEdges`,i.matchedEdges,this.pointCount],[`matchedFractions`,i.matchedFractions,this.pointCount],[`matchedOffsets`,i.matchedOffsets,this.pointCount],[`snapDistances`,i.snapDistances,this.pointCount],[`snappedPositions`,i.snappedPositions,this.pointCount],[`breaks`,i.breaks,this.pointCount],[`trackLogLikelihoods`,i.trackLogLikelihoods,this.trackCount],[`matchedCount`,i.matchedCount,1],[`breakCount`,i.breakCount,1],[`status.overflow`,i.status?.overflow,1],[`status.routeOverflow`,i.status?.routeOverflow,1]])if(t&&t.length!==r)throw Error(`${n} output.${e} must contain ${r} rows`);let{routeTable:f}=e;if(f){for(let[e,t]of[[`routeTable.offsets`,f.offsets],[`routeTable.targets`,f.targets],[`routeTable.status.overflow`,f.status?.overflow]])t&&p(t,[`uint32`],`${n} ${e}`);if(p(f.costs,[`float32`],`${n} routeTable.costs`),p(f.costLimit,[`float32`],`${n} routeTable.costLimit`),f.offsets.length!==this.nodeCount+1)throw Error(`${n} routeTable.offsets must contain nodeCount + 1 rows`);if(f.costs.length!==f.targets.length)throw Error(`${n} routeTable.costs length must equal routeTable.targets length`);for(let[e,t]of[[`costLimit`,f.costLimit],[`status.overflow`,f.status?.overflow]])if(t&&t.length!==1)throw Error(`${n} routeTable.${e} must contain exactly one row`)}let m=ge(e);s(n,m,he(e));let h=m.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(h).size!==h.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(e){let{id:t,props:r,pointCount:i,trackCount:s,nodeCount:c,edgeCount:l,candidateCount:u}=this,{output:p}=r;a(t,e,[...he(r),...ge(r)]),d(t,`topology edges`,l,ne(e.device));let _=M(e.device);r.entryCapacity!==void 0&&o(t,`entryCapacity`,r.entryCapacity,1,_);let v=r.routeNodeBudget??64,y=this.gridWidth*this.gridHeight,b=r.entryCapacity??Math.min(16*l+256,_),[x,ee]=r.bounds.minimum,S=r.bounds.maximum[0],C=n(this.spatialContext),w=(n,r,i)=>f(e,`${t}-${n}`,r,Math.max(i,1)),D=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),O=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),k=`const POINT_COUNT: u32 = ${i}u;
const NODE_COUNT: u32 = ${c}u;
const EDGE_COUNT: u32 = ${l}u;
const K: u32 = ${u}u;
// Route rows: K route lengths, then the number of nodes the route search dropped (an exact f32).
const ROUTE_STRIDE: u32 = ${u+1}u;
const NONE: u32 = 0xffffffffu;
// Back pointer of a restart where a route search into the point dropped nodes: a budget break.
const BUDGET_BREAK: u32 = 0xfffffffeu;
const NEG: f32 = -1.0e30;
const GRID_WIDTH: u32 = ${this.gridWidth}u;
const GRID_HEIGHT: u32 = ${this.gridHeight}u;
const CELL_COUNT: u32 = ${y}u;
const ENTRY_CAPACITY: u32 = ${b}u;
const GRID_MINIMUM: vec2<f32> = vec2<f32>(${F(x)}, ${F(ee)});
const CELL_SIZE: f32 = ${F(r.cellSize)};
const MAX_EDGE_SPAN: u32 = 16u;
fn getCellX(x: f32) -> u32 {
  return u32(clamp(floor((x - GRID_MINIMUM.x) / CELL_SIZE), 0.0, f32(GRID_WIDTH - 1u)));
}
fn getCellY(y: f32) -> u32 {
  return u32(clamp(floor((y - GRID_MINIMUM.y) / CELL_SIZE), 0.0, f32(GRID_HEIGHT - 1u)));
}
${T(C)}
${ie(C!==void 0,x,S)}`,A=`
fn edgeSource(edge: u32) -> u32 { return edgeTable[edgeTableOffset + edge * 3u]; }
fn edgeTarget(edge: u32) -> u32 { return edgeTable[edgeTableOffset + edge * 3u + 1u]; }
fn edgeLength(edge: u32) -> f32 { return bitcast<f32>(edgeTable[edgeTableOffset + edge * 3u + 2u]); }
fn isEdgeValid(edge: u32) -> bool { return edgeLength(edge) >= 0.0; }`,N=(n,r,i,a,o=``)=>g(e,{id:`${t}-${n}`,operation:j,variant:n,bindings:r,invocationCount:i,declarations:`${k}
${r.some(e=>e.name===`nodePositions`)?`fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`:``}
${o}`,body:a}),P=w(`edge-table`,`uint32`,l*3),I=w(`cell-counts`,`uint32`,y),L=w(`cell-ends`,`uint32`,y),R=w(`cursors`,`uint32`,y),_e=w(`entries`,`uint32`,b),z=w(`candidates`,`uint32`,i*u*3),B=w(`point-starts`,`uint32`,i),V=w(`routes`,`float32`,i*u*(u+1)),ve=w(`scores`,`float32`,i*u),ye=w(`back-pointers`,`uint32`,i*u),H=w(`chosen`,`uint32`,i),be=p.breaks??w(`breaks`,`uint32`,i),U=p.status?.overflow??w(`overflow`,`uint32`,1),xe=()=>{let{routeTable:e}=r;if(!e)return[];let t=e.status?.overflow;return[N(`route-bounds`,[D(`pointStarts`,B),D(`points`,r.positions,`f32`),D(`parameters`,r.parameters,`f32`),D(`candidates`,z),D(`edgeTable`,P),O(`routes`,V,`f32`),D(`tableCostLimit`,e.costLimit,`f32`),...t?[D(`tableOverflow`,t)]:[]],i*u,`let point = index / K;
  let fromSlot = index % K;
  let base = routesOffset + index * ROUTE_STRIDE;
  routes[base + K] = -1.0;
  if (point == 0u || pointStarts[pointStartsOffset + point] != 0u) {
    return;
  }
  ${t?`if (tableOverflow[tableOverflowOffset] != 0u) {
    return;
  }`:``}
  let previous = point - 1u;
  let fromEdge = getCandidateEdge(previous, fromSlot);
  if (fromEdge == NONE) {
    return;
  }
  let straight = networkDistance(readPoint(point), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let tail = (1.0 - getCandidateFraction(previous, fromSlot)) * edgeLength(fromEdge);
  let remaining = bound - tail;
  if (!(remaining >= 0.0) || !(remaining <= tableCostLimit[tableCostLimitOffset]) ||
      edgeTarget(fromEdge) >= NODE_COUNT) {
    return;
  }
  routes[base] = bound;
  routes[base + K] = remaining;`,`${A}
${ce()}`),N(`route-table`,[D(`candidates`,z),D(`edgeTable`,P),O(`routes`,V,`f32`),D(`tableOffsets`,e.offsets),D(`tableTargets`,e.targets),D(`tableCosts`,e.costs,`f32`)],i*u,fe(),`${A}
${de()}`)]},W=[];W.push(h(e,{id:`${t}-clear-edge-table`,operation:j,view:P,type:`u32`,value:`${te}u`}),N(`edge-table`,[D(`adjacency`,r.topology.offsets),D(`edgeTargets`,r.topology.neighbors),D(`nodePositions`,r.topology.nodePositions,`f32`),O(`edgeTable`,P)],c,`let end = min(adjacency[adjacencyOffset + index + 1u], EDGE_COUNT);
  let start = readNode(index);
  for (var edge = adjacency[adjacencyOffset + index]; edge < end; edge++) {
    let targetNode = edgeTargets[edgeTargetsOffset + edge];
    var length = -1.0;
    if (targetNode < NODE_COUNT) {
      length = networkDistance(start, readNode(targetNode));
    }
    edgeTable[edgeTableOffset + edge * 3u] = index;
    edgeTable[edgeTableOffset + edge * 3u + 1u] = targetNode;
    edgeTable[edgeTableOffset + edge * 3u + 2u] = bitcast<u32>(length);
  }`),h(e,{id:`${t}-clear-counts`,operation:j,view:I,type:`u32`,value:`0u`}),h(e,{id:`${t}-clear-cursors`,operation:j,view:R,type:`u32`,value:`0u`}),h(e,{id:`${t}-clear-overflow`,operation:j,view:U,type:`u32`,value:`0u`}));let Se=`let a = readNode(edgeSource(index));
  let b = readNode(edgeTarget(index));
  let gridXA = getGridX(a.x);
  let gridXB = getGridXNear(b.x, a.x, gridXA);
  let cellX0 = getCellX(min(gridXA, gridXB));
  let cellX1 = getCellX(max(gridXA, gridXB));
  let cellY0 = getCellY(min(a.y, b.y));
  let cellY1 = getCellY(max(a.y, b.y));`;if(W.push(N(`cell-count`,[D(`edgeTable`,P),D(`nodePositions`,r.topology.nodePositions,`f32`),O(`cellCounts`,I,`atomic<u32>`)],l,`if (!isEdgeValid(index)) {
    return;
  }
  ${Se}
  for (var cellY = cellY0; cellY <= min(cellY1, cellY0 + MAX_EDGE_SPAN - 1u); cellY++) {
    for (var cellX = cellX0; cellX <= min(cellX1, cellX0 + MAX_EDGE_SPAN - 1u); cellX++) {
      atomicAdd(&cellCounts[cellCountsOffset + cellY * GRID_WIDTH + cellX], 1u);
    }
  }`,A),...new m({id:`${t}-cell-scan`,input:I,output:L,mode:`inclusive`}).getCommandNodes(e),N(`cell-fill`,[D(`edgeTable`,P),D(`nodePositions`,r.topology.nodePositions,`f32`),D(`cellEnds`,L),O(`cursors`,R,`atomic<u32>`),O(`entries`,_e),O(`overflow`,U)],l,`if (!isEdgeValid(index)) {
    return;
  }
  ${Se}
  if (cellX1 - cellX0 >= MAX_EDGE_SPAN || cellY1 - cellY0 >= MAX_EDGE_SPAN) {
    overflow[overflowOffset] = 1u;
  }
  for (var cellY = cellY0; cellY <= min(cellY1, cellY0 + MAX_EDGE_SPAN - 1u); cellY++) {
    for (var cellX = cellX0; cellX <= min(cellX1, cellX0 + MAX_EDGE_SPAN - 1u); cellX++) {
      let cell = cellY * GRID_WIDTH + cellX;
      let start = select(0u, cellEnds[cellEndsOffset + cell - 1u], cell > 0u);
      let slot = start + atomicAdd(&cursors[cursorsOffset + cell], 1u);
      if (slot < ENTRY_CAPACITY) {
        entries[entriesOffset + slot] = index;
      } else {
        overflow[overflowOffset] = 1u;
      }
    }
  }`,A),N(`candidates`,[D(`points`,r.positions,`f32`),D(`nodePositions`,r.topology.nodePositions,`f32`),D(`edgeTable`,P),D(`cellEnds`,L),D(`entries`,_e),D(`parameters`,r.parameters,`f32`),O(`candidates`,z)],i,`let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  ${re(C!==void 0)}
  var bestEdge: array<u32, ${u}>;
  var bestFraction: array<f32, ${u}>;
  var bestDistance: array<f32, ${u}>;
  for (var slot = 0u; slot < K; slot++) {
    bestEdge[slot] = NONE;
    bestFraction[slot] = 0.0;
    bestDistance[slot] = 0.0;
  }
  let cellY0 = getCellY(point.y - latitudeExtent);
  let cellY1 = getCellY(point.y + latitudeExtent);
  for (var windowIndex = 0u; windowIndex < WINDOW_COUNT; windowIndex++) {
  let longitudeRange = getLongitudeRange(getGridX(point.x), longitudeExtent, windowIndex);
  if (longitudeRange.x > longitudeRange.y) {
    continue;
  }
  let cellX0 = getCellX(longitudeRange.x);
  let cellX1 = getCellX(longitudeRange.y);
  for (var cellY = cellY0; cellY <= cellY1; cellY++) {
    for (var cellX = cellX0; cellX <= cellX1; cellX++) {
      let cell = cellY * GRID_WIDTH + cellX;
      let start = select(0u, cellEnds[cellEndsOffset + cell - 1u], cell > 0u);
      let end = min(cellEnds[cellEndsOffset + cell], ENTRY_CAPACITY);
      for (var entry = start; entry < end; entry++) {
        let edge = entries[entriesOffset + entry];
        let origin = readNode(edgeSource(edge));
        let segmentEnd = readNode(edgeTarget(edge));
        let fraction = networkSegmentFraction(point, origin, segmentEnd);
        let distance = networkSegmentDistance(point, origin, segmentEnd, fraction);
        if (!(distance <= radius)) {
          continue;
        }
        var isKnown = false;
        for (var slot = 0u; slot < K; slot++) {
          isKnown = isKnown || bestEdge[slot] == edge;
        }
        if (isKnown) {
          continue;
        }
        // Insertion into the sorted list; the last slot is dropped when full.
        var position = K;
        for (var slot = 0u; slot < K; slot++) {
          if (position == K && (bestEdge[slot] == NONE || distance < bestDistance[slot] ||
              (distance == bestDistance[slot] && edge < bestEdge[slot]))) {
            position = slot;
          }
        }
        if (position < K) {
          for (var slot = K - 1u; slot > position; slot--) {
            bestEdge[slot] = bestEdge[slot - 1u];
            bestFraction[slot] = bestFraction[slot - 1u];
            bestDistance[slot] = bestDistance[slot - 1u];
          }
          bestEdge[position] = edge;
          bestFraction[position] = fraction;
          bestDistance[position] = distance;
        }
      }
    }
  }
  }
  for (var slot = 0u; slot < K; slot++) {
    let base = candidatesOffset + (index * K + slot) * 3u;
    candidates[base] = bestEdge[slot];
    candidates[base + 1u] = bitcast<u32>(bestFraction[slot]);
    candidates[base + 2u] = bitcast<u32>(bestDistance[slot]);
  }`,`${A}
${ae(C!==void 0,S)}`),h(e,{id:`${t}-clear-starts`,operation:j,view:B,type:`u32`,value:`0u`}),N(`starts`,[D(`trackOffsets`,r.trackOffsets),O(`pointStarts`,B)],s,`let first = trackOffsets[trackOffsetsOffset + index];
  if (first < POINT_COUNT) {
    pointStarts[pointStartsOffset + first] = 1u;
  }`),...xe(),N(`routes`,[D(`pointStarts`,B),D(`points`,r.positions,`f32`),D(`nodePositions`,r.topology.nodePositions,`f32`),D(`adjacency`,r.topology.offsets),D(`edgeTable`,P),D(`candidates`,z),D(`parameters`,r.parameters,`f32`),O(`routes`,V,`f32`)],i*u,ue(!!r.routeTable),se(v,oe(v),u,A)),N(`forward`,[D(`trackOffsets`,r.trackOffsets),D(`points`,r.positions,`f32`),D(`candidates`,z),D(`parameters`,r.parameters,`f32`),D(`routes`,V,`f32`),O(`scores`,ve,`f32`),O(`backPointers`,ye)],s,pe(),le(u)),N(`backward`,[D(`trackOffsets`,r.trackOffsets),D(`scores`,ve,`f32`),D(`backPointers`,ye),O(`chosen`,H),O(`breaks`,be),...p.trackLogLikelihoods?[O(`logLikelihoods`,p.trackLogLikelihoods,`f32`)]:[]],s,me(!!p.trackLogLikelihoods))),p.status?.routeOverflow&&W.push(h(e,{id:`${t}-clear-route-overflow`,operation:j,view:p.status.routeOverflow,type:`u32`,value:`0u`})),W.push(N(`route-status`,[D(`routes`,V,`f32`),O(`overflow`,U),...p.status?.routeOverflow?[O(`routeOverflow`,p.status.routeOverflow,`atomic<u32>`)]:[]],i*u,`let dropped = u32(routes[routesOffset + index * ROUTE_STRIDE + K]);
  if (dropped == 0u) {
    return;
  }
  overflow[overflowOffset] = 1u;
  ${p.status?.routeOverflow?`// Saturating sum: an add that wraps pins the total at the maximum, and every later add
  // that wraps from there pins it again.
  let previous = atomicAdd(&routeOverflow[routeOverflowOffset], dropped);
  if (previous + dropped < previous) {
    atomicMax(&routeOverflow[routeOverflowOffset], 0xffffffffu);
  }`:``}`)),W.push(N(`matched`,[D(`candidates`,z),D(`chosen`,H),D(`edgeTable`,P),O(`matchedEdges`,p.matchedEdges),...p.matchedFractions?[O(`fractions`,p.matchedFractions,`f32`)]:[],...p.matchedOffsets?[O(`offsets`,p.matchedOffsets,`f32`)]:[],...p.snapDistances?[O(`distances`,p.snapDistances,`f32`)]:[]],i,`let slot = chosen[chosenOffset + index];
  let isMatched = slot != NONE;
  let base = candidatesOffset + (index * K + min(slot, K - 1u)) * 3u;
  let edge = candidates[base];
  matchedEdges[matchedEdgesOffset + index] = select(NONE, edge, isMatched);
  ${p.matchedFractions?`fractions[fractionsOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 1u]), isMatched);`:``}
  ${p.matchedOffsets?`offsets[offsetsOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 1u]) * edgeLength(select(0u, edge, isMatched)), isMatched);`:``}
  ${p.snapDistances?`distances[distancesOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 2u]), isMatched);`:``}`,A)),p.snappedPositions&&W.push(N(`positions`,[D(`points`,r.positions,`f32`),D(`candidates`,z),D(`chosen`,H),D(`edgeTable`,P),D(`nodePositions`,r.topology.nodePositions,`f32`),O(`snappedPositions`,p.snappedPositions,`f32`)],i,`var position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let slot = chosen[chosenOffset + index];
  if (slot != NONE) {
    let base = candidatesOffset + (index * K + slot) * 3u;
    let edge = candidates[base];
    let origin = readNode(edgeSource(edge));
    position = networkInterpolate(origin, readNode(edgeTarget(edge)), bitcast<f32>(candidates[base + 1u]));
  }
  snappedPositions[snappedPositionsOffset + index * 2u] = position.x;
  snappedPositions[snappedPositionsOffset + index * 2u + 1u] = position.y;`,A)),p.matchedCount||p.breakCount){let n=[p.matchedCount,p.breakCount].filter(e=>e);for(let r of n)W.push(h(e,{id:`${t}-clear-${r===p.matchedCount?`matched`:`break`}-count`,operation:j,view:r,type:`u32`,value:`0u`}));let r=[],a=[];p.matchedCount&&(a.push(`if (chosen[chosenOffset + index] != NONE) { addCounter(${r.length}u, 1u); }`),r.push({target:`matchedCount`,word:0})),p.breakCount&&(a.push(`if (breaks[breaksOffset + index] != 0u) { addCounter(${r.length}u, 1u); }`),r.push({target:`breakCount`,word:0}));let o=E({counters:r,rowCount:`POINT_COUNT`,rowSource:a.join(`
  `)});W.push(g(e,{id:`${t}-summary`,operation:j,variant:`summary`,bindings:[D(`chosen`,H),D(`breaks`,be),...p.matchedCount?[O(`matchedCount`,p.matchedCount,`atomic<u32>`)]:[],...p.breakCount?[O(`breakCount`,p.breakCount,`atomic<u32>`)]:[]],invocationCount:i,workgroupSize:256,guardIndex:!1,declarations:`${k}
${o.declarations}`,body:o.body}))}return W}};function re(e){return e?`let radius = min(
    parameters[parametersOffset + 2u],
    2.0 * CELL_SIZE * GEODESIC_DEGREES_TO_RADIANS * NETWORK_EARTH_RADIUS
  );
  let latitudeExtent = radius / NETWORK_EARTH_RADIUS * GEODESIC_RADIANS_TO_DEGREES;
  let bandLatitude = min(abs(point.y) + latitudeExtent, 90.0);
  let longitudeExtent = min(latitudeExtent / max(geodesicCosLatitude(bandLatitude), 1.0e-6), 180.0);`:`let radius = min(parameters[parametersOffset + 2u], 2.0 * CELL_SIZE);
  let latitudeExtent = radius;
  let longitudeExtent = radius;`}function ie(e,t,n){return e?`const GRID_CENTER_X: f32 = ${F((t+n)/2)};
fn getGridX(x: f32) -> f32 {
  return GRID_CENTER_X + geodesicWrapLongitudeDelta(x - GRID_CENTER_X);
}
fn getGridXNear(x: f32, anchorX: f32, anchorGridX: f32) -> f32 {
  return anchorGridX + geodesicWrapLongitudeDelta(x - anchorX);
}`:`fn getGridX(x: f32) -> f32 {
  return x;
}
fn getGridXNear(x: f32, anchorX: f32, anchorGridX: f32) -> f32 {
  return x;
}`}function ae(e,t){return e?`const WINDOW_COUNT: u32 = 3u;
const GRID_MAXIMUM_X: f32 = ${F(t)};
fn getLongitudeRange(x: f32, extent: f32, windowIndex: u32) -> vec2<f32> {
  let low = x - extent;
  let high = x + extent;
  let empty = vec2<f32>(1.0, 0.0);
  if (windowIndex == 1u) {
    let isWrapped = low < GRID_MINIMUM.x && low + 360.0 < GRID_MAXIMUM_X;
    return select(empty, vec2<f32>(low + 360.0, GRID_MAXIMUM_X), isWrapped);
  }
  if (windowIndex == 2u) {
    let isWrapped = high > GRID_MAXIMUM_X && high - 360.0 > GRID_MINIMUM.x;
    return select(empty, vec2<f32>(GRID_MINIMUM.x, high - 360.0), isWrapped);
  }
  return vec2<f32>(low, high);
}`:`const WINDOW_COUNT: u32 = 1u;
fn getLongitudeRange(x: f32, extent: f32, windowIndex: u32) -> vec2<f32> {
  return vec2<f32>(x - extent, x + extent);
}`}function F(e){let t=String(Math.fround(e));return/[.eE]/.test(t)?t:`${t}.0`}function oe(e){let t=16;for(;t<2*e;)t*=2;return t}function se(e,t,n,r){return`${r}
${ce()}
const S: u32 = ${e}u;
const HASH_MASK: u32 = ${t-1}u;
var<private> tableNode: array<u32, ${e}>;
var<private> tableDistance: array<f32, ${e}>;
var<private> tableKey: array<f32, ${e}>;
var<private> tableDone: array<u32, ${e}>;
var<private> tableHash: array<u32, ${t}>;
var<private> pendingTargets: array<u32, ${n}>;
var<private> targetPositions: array<vec2<f32>, ${n}>;
var<private> targetTotal: u32 = 0u;
// Discovered nodes the full table could not keep in the last search.
var<private> droppedCount: u32 = 0u;

// Consistent A* heuristic: the distance to the nearest target node, planar or great-circle. Every
// edge is as long as the straight line (or great circle) between its nodes, so by the triangle
// inequality the heuristic never overestimates; the factor
// below 1 absorbs f32 rounding so it stays consistent in floating point.
fn getHeuristic(node: u32) -> f32 {
  if (targetTotal == 0u || node >= NODE_COUNT) {
    return 0.0;
  }
  let position = readNode(node);
  var nearest = 3.0e38;
  for (var slot = 0u; slot < targetTotal; slot++) {
    nearest = min(nearest, networkDistance(position, targetPositions[slot]));
  }
  return nearest * 0.999999;
}

fn getHashStart(node: u32) -> u32 {
  return ((node * 2654435761u) >> 7u) & HASH_MASK;
}
// Table slot of a node, or NONE. The hash has at least twice as many cells as table slots.
fn findEntry(node: u32) -> u32 {
  var cell = getHashStart(node);
  loop {
    let stored = tableHash[cell];
    if (stored == 0u) {
      return NONE;
    }
    if (tableNode[stored - 1u] == node) {
      return stored - 1u;
    }
    cell = (cell + 1u) & HASH_MASK;
  }
}
// Removes the hash cell of table slot 'slot' with backward-shift deletion: later cells of the
// probe run move into the hole unless their home cell lies cyclically after it, so every lookup
// still reaches its entry without tombstones. The hash always has an empty cell (it has at least
// twice as many cells as table slots), so the loops end.
fn removeHashEntry(slot: u32) {
  var hole = getHashStart(tableNode[slot]);
  while (tableHash[hole] != slot + 1u) {
    hole = (hole + 1u) & HASH_MASK;
  }
  tableHash[hole] = 0u;
  var cell = (hole + 1u) & HASH_MASK;
  loop {
    let stored = tableHash[cell];
    if (stored == 0u) {
      break;
    }
    let home = getHashStart(tableNode[stored - 1u]);
    if (((cell - home) & HASH_MASK) >= ((cell - hole) & HASH_MASK)) {
      tableHash[hole] = stored;
      tableHash[cell] = 0u;
      hole = cell;
    }
    cell = (cell + 1u) & HASH_MASK;
  }
}
// The open (unsettled) slot with the largest key, ties to the highest slot, or NONE.
fn findWorstOpenEntry(count: u32) -> u32 {
  var worst = NONE;
  for (var slot = 0u; slot < count; slot++) {
    if (tableDone[slot] == 0u && (worst == NONE || tableKey[slot] >= tableKey[worst])) {
      worst = slot;
    }
  }
  return worst;
}
fn addEntry(slot: u32, node: u32, distance: f32) {
  tableNode[slot] = node;
  tableDistance[slot] = distance;
  tableKey[slot] = distance + getHeuristic(node);
  tableDone[slot] = 0u;
  var cell = getHashStart(node);
  while (tableHash[cell] != 0u) {
    cell = (cell + 1u) & HASH_MASK;
  }
  tableHash[cell] = slot + 1u;
}

// Bounded A* from 'start' over the CSR network; distances within 'limit' land in the private
// table (at most S nodes; a full table counts every node it cannot keep in 'droppedCount' and keeps
// the open nodes with the best keys). Nodes are settled in order of distance plus the heuristic, so a node
// settled (done) has its exact network distance, and the search heads toward the targets instead of
// settling the whole disc of radius 'limit' as plain Dijkstra does. It stops once every node in
// 'pendingTargets' (NONE for unused slots; 'pendingCount' of them are set) is settled, or once the
// best key exceeds 'limit' (no remaining target can be within the bound). Returns the table size.
fn searchNetwork(start: u32, limit: f32, pendingCount: u32) -> u32 {
  var pending = pendingCount;
  droppedCount = 0u;
  targetTotal = 0u;
  for (var slot = 0u; slot < K; slot++) {
    if (pendingTargets[slot] != NONE) {
      targetPositions[targetTotal] = readNode(pendingTargets[slot]);
      targetTotal++;
    }
  }
  for (var slot = 0u; slot <= HASH_MASK; slot++) {
    tableHash[slot] = 0u;
  }
  var count = 0u;
  addEntry(count, start, 0.0);
  count = 1u;
  loop {
    var best = NONE;
    var bestKey = limit + 1.0;
    for (var slot = 0u; slot < count; slot++) {
      if (tableDone[slot] == 0u && tableKey[slot] < bestKey) {
        best = slot;
        bestKey = tableKey[slot];
      }
    }
    if (best == NONE || bestKey > limit) {
      break;
    }
    let bestDistance = tableDistance[best];
    tableDone[best] = 1u;
    let node = tableNode[best];
    for (var slot = 0u; slot < K; slot++) {
      if (pendingTargets[slot] == node) {
        pendingTargets[slot] = NONE;
        pending--;
      }
    }
    if (pending == 0u) {
      break;
    }
    if (node >= NODE_COUNT) {
      continue;
    }
    let end = min(adjacency[adjacencyOffset + node + 1u], EDGE_COUNT);
    for (var edge = adjacency[adjacencyOffset + node]; edge < end; edge++) {
      if (!isEdgeValid(edge)) {
        continue;
      }
      let candidateDistance = bestDistance + edgeLength(edge);
      if (candidateDistance > limit) {
        continue;
      }
      let neighbor = edgeTarget(edge);
      let found = findEntry(neighbor);
      if (found != NONE) {
        if (tableDone[found] == 0u && candidateDistance < tableDistance[found]) {
          tableDistance[found] = candidateDistance;
          tableKey[found] = candidateDistance + getHeuristic(neighbor);
        }
      } else if (count < S) {
        addEntry(count, neighbor, candidateDistance);
        count++;
      } else {
        // Full table: the node is dropped, or replaces the open entry with the worst key when its
        // own key is better. Either way the search may now miss or overestimate a route.
        droppedCount++;
        let key = candidateDistance + getHeuristic(neighbor);
        let worst = findWorstOpenEntry(count);
        if (worst != NONE && key < tableKey[worst]) {
          removeHashEntry(worst);
          addEntry(worst, neighbor, candidateDistance);
        }
      }
    }
  }
  return count;
}

fn lookupDistance(node: u32) -> f32 {
  let slot = findEntry(node);
  if (slot != NONE && tableDone[slot] != 0u) {
    return tableDistance[slot];
  }
  return -1.0;
}
`}function ce(){return`fn readPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}
fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
fn getCandidateDistance(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 2u]);
}

`}function le(e){return`fn readPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}
fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
fn getCandidateDistance(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 2u]);
}

const LOG_TWO_PI_HALF: f32 = 0.9189385;
const CANDIDATE_COUNT: u32 = ${e}u;`}function ue(e){return`let point = index / K;
  let fromSlot = index % K;
  ${e?`// A non-negative trailing word marks a row the route table resolved (route-table kernel).
  if (routes[routesOffset + index * ROUTE_STRIDE + K] >= 0.0) {
    return;
  }
  `:``}for (var to = 0u; to < K; to++) {
    routes[routesOffset + index * ROUTE_STRIDE + to] = -1.0;
  }
  routes[routesOffset + index * ROUTE_STRIDE + K] = 0.0;
  // Point 0 has no predecessor even when it lies before the first track (review B-23).
  if (point == 0u || pointStarts[pointStartsOffset + point] != 0u) {
    return;
  }
  let previous = point - 1u;
  let fromEdge = getCandidateEdge(previous, fromSlot);
  if (fromEdge == NONE) {
    return;
  }
  let straight = networkDistance(readPoint(point), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let fromLength = edgeLength(fromEdge);
  let fromFraction = getCandidateFraction(previous, fromSlot);
  let tail = (1.0 - fromFraction) * fromLength;
  let remaining = bound - tail;
  if (remaining >= 0.0) {
    var pendingCount = 0u;
    for (var to = 0u; to < K; to++) {
      let toEdge = getCandidateEdge(point, to);
      let isSameEdgeAhead = toEdge == fromEdge && getCandidateFraction(point, to) >= fromFraction;
      let isPending = toEdge != NONE && !isSameEdgeAhead;
      pendingTargets[to] = select(NONE, edgeSource(select(0u, toEdge, isPending)), isPending);
      pendingCount += select(0u, 1u, isPending);
    }
    searchNetwork(edgeTarget(fromEdge), remaining, pendingCount);
    routes[routesOffset + index * ROUTE_STRIDE + K] = f32(min(droppedCount, 16777216u));
  }
  for (var to = 0u; to < K; to++) {
    let toEdge = getCandidateEdge(point, to);
    if (toEdge == NONE) {
      continue;
    }
    let toFraction = getCandidateFraction(point, to);
    var route = -1.0;
    if (toEdge == fromEdge && toFraction >= fromFraction) {
      route = (toFraction - fromFraction) * fromLength;
    } else if (remaining >= 0.0) {
      let network = lookupDistance(edgeSource(toEdge));
      if (network >= 0.0) {
        route = tail + network + toFraction * edgeLength(toEdge);
      }
    }
    // The bound covers the whole route, including the partial edge at either end (review B-24).
    if (route > bound) {
      route = -1.0;
    }
    routes[routesOffset + index * ROUTE_STRIDE + to] = route;
  }`}function de(){return`fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
// Cost from 'row' to 'targetNode' by binary search of the row's ascending targets, or -1 when the
// table does not list it (the cost then exceeds the table's cost limit).
fn findTableCost(row: u32, targetNode: u32) -> f32 {
  var low = tableOffsets[tableOffsetsOffset + row];
  let end = tableOffsets[tableOffsetsOffset + row + 1u];
  var high = end;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (tableTargets[tableTargetsOffset + middle] < targetNode) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && tableTargets[tableTargetsOffset + low] == targetNode) {
    return tableCosts[tableCostsOffset + low];
  }
  return -1.0;
}`}function fe(){return`let base = routesOffset + index * ROUTE_STRIDE;
  let remaining = routes[base + K];
  if (!(remaining >= 0.0)) {
    return;
  }
  let bound = routes[base];
  let point = index / K;
  let fromEdge = getCandidateEdge(point - 1u, index % K);
  let fromFraction = getCandidateFraction(point - 1u, index % K);
  let fromLength = edgeLength(fromEdge);
  let tail = (1.0 - fromFraction) * fromLength;
  let startNode = edgeTarget(fromEdge);
  for (var to = 0u; to < K; to++) {
    let toEdge = getCandidateEdge(point, to);
    var route = -1.0;
    if (toEdge != NONE) {
      let toFraction = getCandidateFraction(point, to);
      if (toEdge == fromEdge && toFraction >= fromFraction) {
        route = (toFraction - fromFraction) * fromLength;
      } else {
        let network = findTableCost(startNode, edgeSource(toEdge));
        // The A* search only admits costs within the allowance; the table also lists larger ones.
        if (network >= 0.0 && network <= remaining) {
          route = tail + network + toFraction * edgeLength(toEdge);
        }
      }
      // The bound covers the whole route, including the partial edge at either end (review B-24).
      if (route > bound) {
        route = -1.0;
      }
    }
    routes[base + to] = route;
  }
  routes[base + K] = 0.0;`}function pe(){return`let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  let sigma = max(parameters[parametersOffset], 1.0e-6);
  let beta = max(parameters[parametersOffset + 1u], 1.0e-6);
  let emissionConstant = -log(sigma) - LOG_TWO_PI_HALF;
  let transitionConstant = -log(beta);
  for (var point = first; point < last; point++) {
    var newScore: array<f32, CANDIDATE_COUNT>;
    var newBack: array<u32, CANDIDATE_COUNT>;
    for (var slot = 0u; slot < K; slot++) {
      newScore[slot] = NEG;
      newBack[slot] = NONE;
    }
    var isReachable = false;
    var isBudgetLimited = false;
    if (point > first) {
      let previous = point - 1u;
      let straight = networkDistance(readPoint(point), readPoint(previous));
      for (var fromSlot = 0u; fromSlot < K; fromSlot++) {
        let fromEdge = getCandidateEdge(previous, fromSlot);
        let fromScore = scores[scoresOffset + previous * K + fromSlot];
        if (fromEdge == NONE || fromScore < -1.0e29) {
          continue;
        }
        let routeRow = routesOffset + (point * K + fromSlot) * ROUTE_STRIDE;
        isBudgetLimited = isBudgetLimited || routes[routeRow + K] > 0.0;
        for (var to = 0u; to < K; to++) {
          let route = routes[routeRow + to];
          if (route < 0.0) {
            continue;
          }
          let score = fromScore + transitionConstant - abs(straight - route) / beta;
          if (score > newScore[to]) {
            newScore[to] = score;
            newBack[to] = fromSlot;
            isReachable = true;
          }
        }
      }
    }
    for (var slot = 0u; slot < K; slot++) {
      if (getCandidateEdge(point, slot) == NONE) {
        newScore[slot] = NEG;
        newBack[slot] = NONE;
        continue;
      }
      let normalized = getCandidateDistance(point, slot) / sigma;
      let emission = emissionConstant - 0.5 * normalized * normalized;
      if (isReachable) {
        if (newScore[slot] > -1.0e29) {
          newScore[slot] += emission;
        }
      } else {
        // First point of the track or a break: restart from the emission alone. A break where a
        // route search dropped nodes is marked, since the missing route may be the budget's doing.
        newScore[slot] = emission;
        newBack[slot] = select(NONE, BUDGET_BREAK, isBudgetLimited);
      }
    }
    for (var slot = 0u; slot < K; slot++) {
      scores[scoresOffset + point * K + slot] = newScore[slot];
      backPointers[backPointersOffset + point * K + slot] = newBack[slot];
    }
  }`}function me(e){return`let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  var logLikelihood = 0.0;
  var current = NONE;
  var isPicking = true;
  for (var step = 0u; step < last - first; step++) {
    let point = last - 1u - step;
    if (isPicking) {
      var bestScore = -1.0e29;
      current = NONE;
      for (var slot = 0u; slot < K; slot++) {
        let score = scores[scoresOffset + point * K + slot];
        if (score > bestScore) {
          bestScore = score;
          current = slot;
        }
      }
      if (current != NONE) {
        logLikelihood += bestScore;
      }
    }
    chosen[chosenOffset + point] = current;
    var isBreak = current == NONE;
    var isBudgetBreak = false;
    isPicking = true;
    if (current != NONE) {
      let previous = backPointers[backPointersOffset + point * K + current];
      if (previous < K) {
        current = previous;
        isPicking = false;
      } else {
        isBreak = true;
        isBudgetBreak = previous == BUDGET_BREAK;
      }
    }
    breaks[breaksOffset + point] = select(0u, select(1u, 2u, isBudgetBreak), isBreak && point > first);
  }
  ${e?`logLikelihoods[logLikelihoodsOffset + index] = logLikelihood;`:``}`}function he(e){return[e.positions,e.trackOffsets,...w(e.topology),e.parameters,e.routeTable?.offsets,e.routeTable?.targets,e.routeTable?.costs,e.routeTable?.costLimit,e.routeTable?.status?.overflow]}function ge(e){let{output:t}=e;return[t.matchedEdges,t.matchedFractions,t.matchedOffsets,t.snapDistances,t.snappedPositions,t.breaks,t.trackLogLikelihoods,t.matchedCount,t.breakCount,t.status?.overflow,t.status?.routeOverflow]}var I=.15,L=4294967295,R=[0,10,20,30,40,50,60],_e=`
fn hashValue(value: u32) -> u32 {
  let state = value * 747796405u + 2891336653u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn uniformValue(value: u32) -> f32 {
  return (f32(hashValue(value) >> 8u) + 0.5) / 16777216.0;
}`;async function z(e){let t=e.datasets.get(`chicago-roads`),n=e.datasets.get(`chicago-gps-traces`),{device:r}=e,a=t.defaultOrigin,o=D(t,a),s=k(t,o),c=t.getProjection(a),d=new O(o.segments,o.bounds),f=Math.hypot(o.bounds[2]-o.bounds[0],o.bounds[3]-o.bounds[1]),p=n.projectColumn(`vertices`,a),m=n.projectColumn(`truthPosition`,a),h=n.column(`truthEdge`),g=n.column(`pathOffsets`),C=n.column(`timestamp`),w=n.column(`sampleInterval`),T=h.length,E=g.length-1,A=new u(r,`map-matching`),j=new Uint32Array(T),te=new Uint32Array(T);for(let e=0;e<E;e++)for(let t=g[e];t<g[e+1];t++)j[t]=e,te[t]=t+1<g[e+1]?1:0;let[ne,M,re,ie]=o.bounds;ne-=200,M-=200,re+=200,ie+=200;let ae=A.createBuffer(`original-points`,p),F=A.createBuffer(`noisy-points`,T*8),oe=A.createBuffer(`truth-points`,m),se=A.createBuffer(`track-of-point`,j),ce=A.createBuffer(`has-next`,te),le=A.createBuffer(`truth-edges`,h),ue=A.createBuffer(`edge-reverse`,o.edgeReverse),de=A.createBuffer(`row-edge`,s.rowEdge),fe=A.createBuffer(`track-offsets`,g),pe=A.createBuffer(`node-positions`,s.nodePositions),me=A.createBuffer(`offsets`,s.offsets),he=A.createBuffer(`targets`,s.targets),ge=A.createBuffer(`road-segments`,o.segments),z=A.createParameterBuffer(`jitter`,`float32`,2),B=A.createParameterBuffer(`focus`,`uint32`,1),V=A.createParameterBuffer(`parameters`,`float32`,8),ve=A.createBuffer(`matched-edges`,T*4),ye=A.createBuffer(`snap-distances`,T*4),H=A.createBuffer(`snapped-positions`,T*8),be=A.createBuffer(`breaks`,T*4),U=A.createBuffer(`matched-count`,4),xe=A.createBuffer(`break-count`,4),W=A.createBuffer(`overflow`,4),Se=A.createBuffer(`route-overflow`,4),Ce=A.createBuffer(`raw-segments`,T*16),we=A.createBuffer(`raw-weights`,T*4),Te=A.createBuffer(`truth-segments`,T*16),Ee=A.createBuffer(`truth-weights`,T*4),G=A.createBuffer(`matched-segments`,T*16),De=A.createBuffer(`matched-weights`,T*4),Oe=A.createBuffer(`matched-classes`,T*4),ke=A.createBuffer(`wrong-street-weights`,T*4),Ae=A.createBuffer(`correct-weights`,T*4),je=A.createBuffer(`reverse-weights`,T*4),Me=o.segmentCount,Ne=A.createBuffer(`selected-fix-positions`,256*8),Pe=A.createBuffer(`candidate-segments`,128),Fe=A.createBuffer(`snap-leaders`,128),Ie=A.createBuffer(`break-positions`,256*8),Le=A.createBuffer(`sigma-position`,8),Re=A.createBuffer(`search-position`,8),ze=A.createBuffer(`nearest-segments`,256*16),Be=A.createBuffer(`nearest-weights`,256*4),Ve=A.createBuffer(`selected-raw-segments`,256*16),He=A.createBuffer(`selected-raw-weights`,256*4),Ue=new Uint32Array(T),We=new Float32Array(T),K=new Float32Array(T*2),Ge=new Uint32Array(T).fill(L),Ke=new Uint32Array(T),qe=new Float32Array(T),Je=0,Ye=0,Xe=0,q=null,J=-1,Y=new Map,X=0,Ze=null,Qe=null,$e=null,et=!1,tt=!0,nt=-1/0,rt=!1,Z=null,it=()=>2*Number(e.options.cellSize);function Q(t=e.options.extraNoise,n=null){let{noiseSeed:r,sigma:i,beta:a,searchRadius:o,routeFactor:s,routeSlack:c,trackFocus:l}=e.options;z.write(Float32Array.of(t,r)),B.write(Uint32Array.of(l)),V.write(N({sigma:i,beta:a,searchRadius:Math.min(o,it()),routeFactor:s,routeSlack:c})),Ze=n,tt=!0,dt(),ft()}function at(){X++,q=null,J=-1,Ze=null,Qe=null,$e=null,e.options.evidenceMode===`stress`&&(Y.clear(),J=0,q=R[J],Q(q,{generation:X,noise:q}))}function ot(e){q===null||e.generation!==X||e.noise!==q||(J++,J<R.length?(q=R[J],Q(q,{generation:X,noise:q})):(q=null,J=-1,Q()))}function st(e){let t=Math.imul(e,747796405)+2891336653>>>0,n=Math.imul(t>>>(t>>>28)+4^t,277803737)>>>0;return(n>>>22^n)>>>0}function ct(e){return((st(e)>>>8)+.5)/16777216}function lt(t){let n=Math.imul(e.options.noiseSeed,2654435761)>>>0,r=Math.sqrt(-2*Math.log(ct(Math.imul(t,2)+n))),i=2*Math.PI*ct(Math.imul(t,2)+1+n);return[p[t*2]+e.options.extraNoise*r*Math.cos(i),p[t*2+1]+e.options.extraNoise*r*Math.sin(i)]}function ut(e,t,n){let r=e*4,i=o.segments[r],a=o.segments[r+1],s=o.segments[r+2]-i,c=o.segments[r+3]-a,l=s*s+c*c,u=l?Math.max(0,Math.min(1,((t-i)*s+(n-a)*c)/l)):0,d=i+s*u,f=a+c*u;return[d,f,Math.hypot(t-d,n-f)]}function dt(){let t=0,n=0,r=0,i=0,a=-1,c=Math.max(0,Math.min(E-1,e.options.trackFocus-1)),u=g[c],p=g[c+1];for(let e=0;e<T;e++){let[c,l]=lt(e);K[e*2]=c,K[e*2+1]=l;let m=d.nearest(c,l,f),g=m>=0?o.segmentEdge[m]:L;Ue[e]=g,We[e]=m>=0?ut(m,c,l)[2]:NaN;let _=h[e];g===_?t++:_!==L&&g===o.edgeReverse[_]&&n++,Ge[e]!==L&&s.rowEdge[Ge[e]]!==g&&(r++,a<0&&(a=e),e>=u&&e<p&&i++)}e.setReadout(`baseline`,`${(100*(t+n)/T).toFixed(1)}% (either direction)`);let m=a<0?`none`:`trace ${j[a]+1}, fix ${a-g[j[a]]+1}`;e.setReadout(`mismatch`,`${l(r)} disagreements; selected trace: ${i}; first: ${m}`)}function ft(){let t=Math.max(0,Math.min(E-1,e.options.trackFocus-1)),n=g[t],r=Math.min(g[t+1],n+256),i=r-n,a=new Float32Array(256*2),s=new Float32Array(256*4),c=new Float32Array(256),l=new Float32Array(256*4),u=new Float32Array(256),p=new Float32Array(256*2),m=0;for(let e=0;e<i;e++){let t=n+e,r=K[t*2],o=K[t*2+1];a[e*2]=r,a[e*2+1]=o,e+1<i&&(s.set([r,o,K[(t+1)*2],K[(t+1)*2+1]],e*4),c[e]=1);let h=d.nearest(r,o,f);if(h>=0){let[t,n]=ut(h,r,o);l.set([r,o,t,n],e*4),u[e]=1}Ke[t]&&(p[m*2]=r,p[m*2+1]=o,m++)}Ne.write(a),Je=i,Ve.write(s),He.write(c),ze.write(l),Be.write(u),Ie.write(p),Xe=m;let h=Math.min(r-1,n+Math.max(0,e.options.fixFocus-1)),_=K[h*2],v=K[h*2+1];Le.write(Float32Array.of(_,v)),Re.write(Float32Array.of(_,v));let y=[],b=Math.min(e.options.searchRadius,it());for(let e=0;e<Me;e++){let t=ut(e,_,v)[2];t<=b&&y.push({segment:e,distance:t})}y.sort((e,t)=>e.distance-t.distance);let x=new Float32Array(32),ee=new Float32Array(32),S=Math.min(8,y.length);for(let e=0;e<S;e++){let t=y[e].segment;x.set(o.segments.subarray(t*4,t*4+4),e*4);let[n,r]=ut(t,_,v);ee.set([_,v,n,r],e*4)}Pe.write(x),Ye=S,Fe.write(ee),e.requestLayers()}function pt(e){let t=4*e,n=0;for(let r=0;r<s.nodeCount;r++)for(let i=s.offsets[r];i<s.offsets[r+1];i++){let a=s.targets[i],o=Math.abs(s.nodePositions[a*2]-s.nodePositions[r*2]),c=Math.abs(s.nodePositions[a*2+1]-s.nodePositions[r*2+1]);n+=(Math.floor((o+t)/e)+2)*(Math.floor((c+t)/e)+2)}return Math.min(n,4e7)}async function mt(){let t=Math.max(Number(e.options.cellSize),Math.sqrt((re-ne)*(ie-M)/2e6)),n=new i(r,{id:`map-matching`}),a=(e,t,r,i)=>b(n,e,t,r,i),l=a(`original-points`,ae,`float32x2`,T),u=a(`noisy-points`,F,`float32x2`,T),d=a(`matched-edges`,ve,`uint32`,T),f=a(`snapped-positions`,H,`float32x2`,T),p=a(`breaks`,be,`uint32`,T),m=a(`has-next`,ce,`uint32`,T),h=a(`track-of-point`,se,`uint32`,T),g=B.importToGraph(n),_=a(`truth-points`,oe,`float32x2`,T);S(n,{id:`extra-noise`,bindings:[{name:`original`,view:l,type:`f32`,access:`read`},{name:`jitter`,view:z.importToGraph(n),type:`f32`,access:`read`},{name:`noisy`,view:u,type:`f32`,access:`read_write`}],invocationCount:T,declarations:_e,body:`let seed = u32(jitter[jitterOffset + 1u]) * 2654435761u;
  let radius = sqrt(-2.0 * log(uniformValue(index * 2u + seed)));
  let angle = 6.2831853 * uniformValue(index * 2u + 1u + seed);
  noisy[noisyOffset + index * 2u] = original[originalOffset + index * 2u] + jitter[jitterOffset] * radius * cos(angle);
  noisy[noisyOffset + index * 2u + 1u] = original[originalOffset + index * 2u + 1u] + jitter[jitterOffset] * radius * sin(angle);`}),n.add(new P({spatialContext:c.spatialContext,id:`matching`,positions:u,trackOffsets:a(`track-offsets`,fe,`uint32`,E+1),topology:{offsets:a(`offsets`,me,`uint32`,s.nodeCount+1),neighbors:a(`targets`,he,`uint32`,s.rowCount),nodePositions:a(`node-positions`,pe,`float32x2`,s.nodeCount)},parameters:V.importToGraph(n),candidateCount:Number(e.options.candidateCount),routeNodeBudget:Number(e.options.routeNodeBudget),cellSize:t,bounds:{minimum:[ne,M],maximum:[re,ie]},entryCapacity:pt(t),output:{matchedEdges:d,snapDistances:a(`snap-distances`,ye,`float32`,T),snappedPositions:f,breaks:p,matchedCount:a(`matched-count`,U,`uint32`,1),breakCount:a(`break-count`,xe,`uint32`,1),status:{overflow:a(`overflow`,W,`uint32`,1),routeOverflow:a(`route-overflow`,Se,`uint32`,1)}}}));let v=(e,t,r,i)=>{S(n,{id:e,bindings:[{name:`points`,view:t,type:`f32`,access:`read`},{name:`hasNext`,view:m,type:`u32`,access:`read`},{name:`trackOfPoint`,view:h,type:`u32`,access:`read`},{name:`focus`,view:g,type:`u32`,access:`read`},{name:`segments`,view:a(`${e}-segments`,r,`float32`,T*4),type:`f32`,access:`read_write`},{name:`weights`,view:a(`${e}-weights`,i,`float32`,T),type:`f32`,access:`read_write`}],invocationCount:T,declarations:`const POINT_COUNT: u32 = ${T}u;
`,body:`let next = min(index + 1u, POINT_COUNT - 1u);
  segments[segmentsOffset + index * 4u] = points[pointsOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = points[pointsOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = points[pointsOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = points[pointsOffset + next * 2u + 1u];
  let inFocus = focus[focusOffset] == 0u || trackOfPoint[trackOfPointOffset + index] + 1u == focus[focusOffset];
  weights[weightsOffset + index] = select(0.0, 1.0, hasNext[hasNextOffset + index] != 0u && inFocus);`})};v(`raw-rows`,u,Ce,we),v(`truth-rows`,_,Te,Ee);let y=a(`edge-reverse`,ue,`uint32`,o.edgeCount),x=a(`row-edge`,de,`uint32`,s.rowCount),C=a(`truth-edges`,le,`uint32`,T);S(n,{id:`matched-classes`,bindings:[{name:`matched`,view:d,type:`u32`,access:`read`},{name:`rowEdge`,view:x,type:`u32`,access:`read`},{name:`truthEdge`,view:C,type:`u32`,access:`read`},{name:`reverse`,view:y,type:`u32`,access:`read`},{name:`classes`,view:a(`matched-classes`,Oe,`uint32`,T),type:`u32`,access:`read_write`}],invocationCount:T,declarations:`const NONE: u32 = ${L}u;`,body:`var accuracy = 0u;
  if (matched[matchedOffset + index] != NONE) {
    let edge = rowEdge[rowEdgeOffset + matched[matchedOffset + index]];
    let truth = truthEdge[truthEdgeOffset + index];
    if (edge == truth) {
      accuracy = 1u;
    } else if (truth != NONE && edge == reverse[reverseOffset + truth]) {
      accuracy = 2u;
    }
  }
  classes[classesOffset + index] = accuracy;`});let w=a(`matched-segments`,G,`float32`,T*4),D=a(`matched-weights`,De,`float32`,T);S(n,{id:`matched-rows`,bindings:[{name:`points`,view:f,type:`f32`,access:`read`},{name:`matched`,view:d,type:`u32`,access:`read`},{name:`breaks`,view:p,type:`u32`,access:`read`},{name:`hasNext`,view:m,type:`u32`,access:`read`},{name:`trackOfPoint`,view:h,type:`u32`,access:`read`},{name:`focus`,view:g,type:`u32`,access:`read`},{name:`segments`,view:w,type:`f32`,access:`read_write`},{name:`weights`,view:D,type:`f32`,access:`read_write`}],invocationCount:T,declarations:`const POINT_COUNT: u32 = ${T}u;
const NONE: u32 = ${L}u;`,body:`let next = min(index + 1u, POINT_COUNT - 1u);
  segments[segmentsOffset + index * 4u] = points[pointsOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = points[pointsOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = points[pointsOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = points[pointsOffset + next * 2u + 1u];
  let inFocus = focus[focusOffset] == 0u || trackOfPoint[trackOfPointOffset + index] + 1u == focus[focusOffset];
  let isDrawn = hasNext[hasNextOffset + index] != 0u && inFocus && matched[matchedOffset + index] != NONE &&
    matched[matchedOffset + next] != NONE && breaks[breaksOffset + next] == 0u;
  weights[weightsOffset + index] = select(0.0, 1.0, isDrawn);`}),S(n,{id:`matched-outcomes`,bindings:[{name:`matched`,view:d,type:`u32`,access:`read`},{name:`weights`,view:D,type:`f32`,access:`read`},{name:`rowEdge`,view:x,type:`u32`,access:`read`},{name:`truthEdge`,view:C,type:`u32`,access:`read`},{name:`reverse`,view:y,type:`u32`,access:`read`},{name:`wrongWeights`,view:a(`wrong-street-weights`,ke,`float32`,T),type:`f32`,access:`read_write`},{name:`correctWeights`,view:a(`correct-weights`,Ae,`float32`,T),type:`f32`,access:`read_write`},{name:`reverseWeights`,view:a(`reverse-weights`,je,`float32`,T),type:`f32`,access:`read_write`}],invocationCount:T,declarations:`const NONE: u32 = ${L}u;`,body:`let isDrawn = weights[weightsOffset + index] != 0.0;
  var outcome = 0u;
  if (isDrawn) {
    let edge = rowEdge[rowEdgeOffset + matched[matchedOffset + index]];
    let truth = truthEdge[truthEdgeOffset + index];
    if (edge == truth) {
      outcome = 1u;
    } else if (truth != NONE && edge == reverse[reverseOffset + truth]) {
      outcome = 2u;
    }
  }
  wrongWeights[wrongWeightsOffset + index] = select(0.0, 1.0, isDrawn && outcome == 0u);
  correctWeights[correctWeightsOffset + index] = select(0.0, 1.0, outcome == 1u);
  reverseWeights[reverseWeightsOffset + index] = select(0.0, 1.0, outcome == 2u);`});let O=A.track(await n.compileAsync()),k=new ee(A,`match`,[{buffer:U,size:4},{buffer:xe,size:4},{buffer:W,size:4},{buffer:ve,size:T*4},{buffer:ye,size:T*4},{buffer:be,size:T*4},{buffer:Se,size:4}],e=>{if(et||Z?.reader!==k)return;let t=$e;$e=null,vt(e,t)});return{compiled:O,reader:k}}function ht(e){e.reader.stop(),A.release(e.compiled)}function gt(e){let t=Z;t?.reader.stop(),Z=e,t&&A.release(t.compiled),tt=!0}let _t=A.track(new x);function vt(t,n){let r=new Uint32Array(t,0,3+T),[i,a,c]=r,u=r.subarray(3),d=new Float32Array(t,(3+T)*4,T);Ge=u.slice(),Ke=new Uint32Array(t,(3+T*2)*4,T).slice(),qe=d.slice();let f=new Uint32Array(t,(3+T*3)*4,1)[0],p=Ke.reduce((e,t)=>e+(t===2?1:0),0),m=0,g=0,_=0,v=0;for(let e=0;e<T;e++){let t=u[e];if(t===L)continue;let n=s.rowEdge[t],r=h[e];n===r?m++:r!==L&&n===o.edgeReverse[r]&&g++,d[e]>=0&&(_+=d[e],v++)}e.setReadout(`matched`,`${l(i)} of ${l(T)} (${(100*i/T).toFixed(1)}%)`),e.setReadout(`breaks`,p>0?`${l(a)} (${l(p)} route budget exceeded)`:a),e.setReadout(`overflow`,f>0?`route budget exceeded (${l(f)} nodes dropped)`:c?`edge grid overflowed`:`none`),e.setReadout(`exact`,`${(100*m/T).toFixed(1)}%`),e.setReadout(`sameStreet`,`${(100*(m+g)/T).toFixed(1)}%`),e.setReadout(`snap`,v?`${(_/v).toFixed(1)} m`:null);let y=100*(m+g)/T;e.setReadout(`hmm`,`${y.toFixed(1)}% (either direction)`);let b=!!(n&&n.generation===X&&n.noise===q);b&&Y.set(n.noise,y),dt(),ft();let x=Array(8).fill(0);for(let e of d)Number.isFinite(e)&&e>=0&&x[Math.min(7,Math.floor(e/10))]++;e.setChart(`snapHistogram`,{kind:`histogram`,values:x,xDomain:[0,80],markers:[{x:e.options.sigma,label:`sigma`}],xLabel:`snap distance (m)`,yLabel:`fixes`,description:`HMM snap-distance histogram from the loaded traces; marker is the current emission sigma.`}),e.setChart(`noiseComparison`,{kind:`line`,xLabel:`added isotropic noise (m)`,yLabel:`accuracy (%)`,series:[{label:`Nearest baseline`,x:R,y:yt(R),color:1,dashed:!0,points:!0},{label:`HMM`,x:R.slice(0,Y.size),y:R.slice(0,Y.size).map(e=>Y.get(e)??0),color:0,points:!0}],description:`Nearest-edge and HMM accuracy are measured from the loaded traces over the same fixed deterministic added-noise ladder.`}),b&&ot(n)}function yt(t){return t.map(t=>{let n=0;for(let r=0;r<T;r++){let i=Math.imul(e.options.noiseSeed,2654435761)>>>0,a=Math.sqrt(-2*Math.log(ct(Math.imul(r,2)+i))),s=2*Math.PI*ct(Math.imul(r,2)+1+i),c=p[r*2]+t*a*Math.cos(s),l=p[r*2+1]+t*a*Math.sin(s),u=d.nearest(c,l,f),m=u>=0?o.segmentEdge[u]:L,g=h[r];(m===g||g!==L&&m===o.edgeReverse[g])&&n++}return 100*n/T})}async function bt(){if(!(rt||!Z)){rt=!0;try{let t=await y(r,Z.compiled,{parameters:void 0,completionBuffer:U,runs:5,repetitions:2});et||e.setReadout(`time`,`${t.milliseconds.toFixed(1)} ms (${t.method===`gpu-timestamps`?`GPU`:`wall clock`})`)}catch{}finally{rt=!1}}}let xt=0;{let e=0;for(let t=0;t<T;t++){let n=p[t*2]-m[t*2],r=p[t*2+1]-m[t*2+1];e+=n*n+r*r}xt=Math.sqrt(e/T)}e.setReadout(`fixes`,`${l(E)} tracks, ${l(T)} fixes`),e.setReadout(`network`,`${l(s.nodeCount)} nodes, ${l(s.rowCount)} directed edges`),e.setReadout(`noise`,`${xt.toFixed(1)} m RMS (data)`);let St=Array.from(w);St.sort((e,t)=>e-t);let Ct=St[Math.floor(St.length/2)]??0;function wt(){let t=Math.max(0,Math.min(E-1,e.options.trackFocus-1)),r=n.column(`noiseSigma`)[t];e.setReadout(`interval`,`${Ct} s nominal (selected sigma ${r.toFixed(0)} m)`),e.setFurniture({title:{title:`Simulated GPS`,subtitle:`${r.toFixed(0)} m sigma + ${e.options.extraNoise.toFixed(0)} m added · ${Ct} s nominal samples`},scaleBar:{units:`metric`},credit:`Synthetic traces derived from OSM, ODbL; not observed vehicles.`})}wt(),Q(),gt(await mt());let $=[a[0],a[1],0];return{getCompiledGraphs:()=>Z?[Z.compiled]:[],setOption(t){switch(t){case`candidateCount`:case`routeNodeBudget`:case`cellSize`:_t.request({build:mt,commit:e=>{Q(),gt(e),at()},discard:ht});break;case`extraNoise`:case`noiseSeed`:case`sigma`:case`beta`:case`searchRadius`:case`routeFactor`:case`routeSlack`:case`trackFocus`:case`fixFocus`:Q(),wt(),at();break;case`evidenceMode`:Q(),at();break;default:e.requestLayers()}},onAction(e){e===`measure`&&bt()},onThemeChange(){e.requestLayers()},getTooltip(e){if(!e.coordinate)return null;let[t,n]=c.project(e.coordinate[0],e.coordinate[1]),r=-1,i=1225;for(let e=0;e<T;e++){let a=K[e*2]-t,o=K[e*2+1]-n,s=a*a+o*o;s<i&&(i=s,r=e)}if(r<0)return null;let a=0;for(;a+1<g.length&&g[a+1]<=r;)a++;let l=r-g[a]+1,u=Math.hypot(K[r*2]-m[r*2],K[r*2+1]-m[r*2+1]),d=Ge[r],f=d===L?L:s.rowEdge[d],p=h[r],_=f===L?`unmatched`:f===p?`truth edge`:p!==L&&f===o.edgeReverse[p]?`reverse direction`:`wrong street`;return{title:`Trace ${a+1}, fix ${l}`,subtitle:`t + ${C[r]} s`,rows:[{label:`Raw to truth`,value:u.toFixed(1),unit:`m`},{label:`Nearest snap`,value:We[r].toFixed(1),unit:`m`},{label:`HMM snap`,value:d===L?`—`:qe[r].toFixed(1),unit:d===L?void 0:`m`},{label:`Outcome`,value:_,emphasis:!0}]}},encode(e,t){if(!Z)return;tt&&t.timeSeconds-nt>=I&&(nt=t.timeSeconds,Z.compiled.encode(e,{parameters:void 0}),tt=!1,Z.reader.markStale(),Qe=Ze,Ze=null);let n=Z.reader.isPending;Z.reader.flush(e),!n&&Z.reader.isPending&&($e=Qe,Qe=null)},getLayers(){let t=e.ground()===`dark`,n=e.options.evidenceMode===`raw`||e.options.evidenceMode===`nearest`,r=e.options.evidenceMode===`nearest`||e.options.evidenceMode===`tradeoff`,i=e.options.evidenceMode===`candidates`,a=e.options.evidenceMode===`tradeoff`||e.options.evidenceMode===`stress`,o=[];return o.push(new v({id:`mm-roads`,coordinateOrigin:$,segments:ge,instanceCount:Me,color:t?[174,164,150,115]:[130,120,105,120],widthPixels:1}),new v({id:`mm-truth`,coordinateOrigin:$,segments:Te,instanceCount:T,weights:Ee,color:t?[200,205,215,85]:[60,65,80,80],widthPixels:7,outlineColor:t?[20,23,28,160]:[250,247,239,190],outlineWidthPixels:1}),new v({id:`mm-raw-connector`,coordinateOrigin:$,segments:Ve,instanceCount:n?256:0,weights:He,color:[135,83,180,190],widthPixels:.8,dashArray:[3,3]}),new v({id:`mm-nearest-baseline`,coordinateOrigin:$,segments:ze,instanceCount:r?256:0,weights:Be,color:t?[200,185,160,165]:[105,92,75,185],widthPixels:2,dashArray:[4,3]}),new v({id:`mm-hmm-wrong-street`,coordinateOrigin:$,segments:G,instanceCount:T,weights:ke,color:[205,76,47,255],widthPixels:4.5}),new v({id:`mm-hmm-correct`,coordinateOrigin:$,segments:G,instanceCount:T,weights:Ae,color:[40,125,210,255],widthPixels:3.5}),new v({id:`mm-hmm-reverse`,coordinateOrigin:$,segments:G,instanceCount:T,weights:je,color:[122,185,232,255],widthPixels:3.5,dashArray:[5,3]}),new _({id:`mm-raw-halos`,coordinateOrigin:$,positions:Ne,instanceCount:n?Je:0,radiusPixels:3.5,color:[135,83,180,255],outlineColor:t?[28,22,34,220]:[255,248,255,230],outlineWidthPixels:2}),new _({id:`mm-sigma-ring`,coordinateOrigin:$,positions:Le,instanceCount:n?1:0,radiusMeters:e.options.sigma,radiusMinPixels:3,shape:`ring`,color:[135,83,180,255],fillOpacity:0}),new _({id:`mm-search-ring`,coordinateOrigin:$,positions:Re,instanceCount:i?1:0,radiusMeters:Math.min(e.options.searchRadius,it()),shape:`ring`,color:[212,163,45,255],fillOpacity:0}),new v({id:`mm-candidate-segments`,coordinateOrigin:$,segments:Pe,instanceCount:i?Ye:0,color:[212,163,45,245],widthPixels:3}),new v({id:`mm-snap-leaders`,coordinateOrigin:$,segments:Fe,instanceCount:i?Ye:0,color:[212,163,45,180],widthPixels:1,dashArray:[2,2]}),new _({id:`mm-break-symbols`,coordinateOrigin:$,positions:Ie,instanceCount:a?Xe:0,radiusPixels:6,shape:`cross`,color:[205,76,47,255],outlineColor:t?[22,22,22,255]:[255,255,255,255],outlineWidthPixels:1})),o},destroy(){et=!0,Z?.reader.stop(),A.destroy()}}}export{z as createMapMatching};