import{_ as e,d as t,h as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{t as a}from"./gpu-scan-BuGhTLDQ.js";import{l as o,s}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c}from"./wgsl-common-BNRVzgd8.js";import{a as l}from"./sorted-segment-sums-DwGkmbx2.js";import{i as u,n as d,r as f,t as p}from"./key-pair-grouping--fUW7Rco.js";import{a as m,i as h,l as g,n as _,t as v}from"./spatial-weights-D4bd9AQ6.js";import{t as y}from"./scan-total-BdolCtK7.js";var b=`GPUContiguityWeights`,x=`${p}u`,S=4,C=4,w=2,T=4,E=class{id;props;outputWeights;constructor(e){let r=e.id??`contiguity-weights`;if(this.id=r,this.props=e,e.criterion!==`queen`&&e.criterion!==`rook`)throw Error(`${r} criterion must be 'queen' or 'rook'`);let{polygons:a}=e;i(a.positions,[`float32x2`],`${r} polygons.positions`),t(r,`polygons.positions length`,a.positions.length,1,2**31-1);for(let e of[`ringOffsets`,`polygonOffsets`,`featureOffsets`])i(a[e],[`uint32`],`${r} polygons.${e}`),t(r,`polygons.${e} length`,a[e].length,2);if(e.snapTolerance!==void 0&&(!Number.isFinite(e.snapTolerance)||e.snapTolerance<0))throw Error(`${r} snapTolerance must be finite and >= 0`);let o=g(r,e.output);if(o!==a.featureOffsets.length-1)throw Error(`${r} output rows must equal the feature count`);e.pairCapacity!==void 0&&t(r,`pairCapacity`,e.pairCapacity),this.outputWeights=v(e.output,{information:{rowCount:o,transform:`binary`,symmetric:!0,hasSelfWeights:!1,provenance:{source:b,criterion:e.criterion}},status:e.output.status}),n(r,[...m(e.output),...h(e.output)],[a.positions,a.ringOffsets,a.polygonOffsets,a.featureOffsets])}getFeatureCount(){return this.props.polygons.featureOffsets.length-1}getPairCapacity(){return this.props.pairCapacity??Math.max(1,this.props.output.neighbors.length*4)}getCapacityPlan(e={}){let{props:t}=this,n=t.output.neighbors.length,r=_({rowCount:this.getFeatureCount(),capacity:n,requiredCount:e.requiredCount,distances:!!t.output.distances}),i=this.getPairCapacity(),a=t.pairCapacity===void 0?r.capacity*4:i,o=e.candidateOverflow?Math.max(a,i*2):a,s=t.polygons.positions.length,c=D(s,t.criterion),l=O(o);return{slots:r,pairs:{capacity:o,estimatedRequiredCount:o,limited:!1,estimatedOutputBytes:0,estimatedPeakTransientBytes:l*Uint32Array.BYTES_PER_ELEMENT},estimatedWorkItems:s+o+this.getFeatureCount()+1,estimatedOutputBytes:r.estimatedOutputBytes,estimatedPeakTransientBytes:(c+l)*Uint32Array.BYTES_PER_ELEMENT}}getCommandNodes(t){let{id:n,props:i}=this,{output:f}=i,{ringOffsets:p,polygonOffsets:g,featureOffsets:_}=i.polygons,v=i.polygons.positions,{overflow:S,requiredCount:C,candidateOverflow:w}=f.status;e(n,t,[v,p,g,_,...m(f),...h(f)]);let T=v.length,E=this.getFeatureCount(),D=g.length-1,O=p.length-1,k=f.neighbors.length,A=this.getPairCapacity(),j=i.snapTolerance??0,M=r(t,`${n}-vertex-high`,`uint32`,T),N=r(t,`${n}-vertex-low`,`uint32`,T),P=r(t,`${n}-vertex-feature`,`uint32`,T),F=r(t,`${n}-vertex-next`,`uint32`,T),I=[s(t,{id:`${n}-vertex-topology`,operation:b,variant:`vertex-topology`,bindings:[{name:`positions`,view:v,type:`f32`,access:`read`},{name:`ringOffsets`,view:p,type:`u32`,access:`read`},{name:`polygonOffsets`,view:g,type:`u32`,access:`read`},{name:`featureOffsets`,view:_,type:`u32`,access:`read`},{name:`vertexHigh`,view:M,type:`u32`,access:`read_write`},{name:`vertexLow`,view:N,type:`u32`,access:`read_write`},{name:`vertexPolygon`,view:P,type:`u32`,access:`read_write`},{name:`vertexNext`,view:F,type:`u32`,access:`read_write`}],invocationCount:T,declarations:`const RING_COUNT: u32 = ${O}u;
const POLYGON_COUNT: u32 = ${D}u;
const FEATURE_COUNT: u32 = ${E}u;
const SNAP_TOLERANCE: f32 = ${o(j)};

${c}

fn quantise(value: f32) -> u32 {
  if (SNAP_TOLERANCE <= 0.0) {
    return bitcast<u32>(select(value, 0.0, value == 0.0));
  }
  let snapped = clamp(floor(value / SNAP_TOLERANCE + 0.5), -2147483520.0, 2147483520.0);
  return bitcast<u32>(i32(snapped)) ^ 0x80000000u;
}`,body:`var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  var polygon = 0u;
  high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  var feature = 0u;
  high = FEATURE_COUNT;
  while (feature + 1u < high) {
    let middle = (feature + high) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= polygon) {
      feature = middle;
    } else {
      high = middle;
    }
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let valid = index >= ringBegin && index < ringEnd &&
    ring >= polygonOffsets[polygonOffsetsOffset] && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
    ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u] &&
    polygon >= featureOffsets[featureOffsetsOffset] && polygon < featureOffsets[featureOffsetsOffset + FEATURE_COUNT] &&
    polygon < featureOffsets[featureOffsetsOffset + feature + 1u] &&
    isFiniteFloat(x) && isFiniteFloat(y);
  vertexHigh[vertexHighOffset + index] = select(${x}, quantise(x), valid);
  vertexLow[vertexLowOffset + index] = select(${x}, quantise(y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${x}, feature, valid);
  vertexNext[vertexNextOffset + index] = select(index + 1u, ringBegin, index + 1u == ringEnd);`})],L=u(t,`${n}-vertex`,b,T,M,N),R=d(t,`${n}-vertex`,b,T,M,N,L.sortedItems);I.push(...L.nodes,...R.nodes);let z=P,B=L.sortedItems,V=R;if(i.criterion===`rook`){let e=r(t,`${n}-point-ids`,`uint32`,T),i=r(t,`${n}-edge-high`,`uint32`,T),a=r(t,`${n}-edge-low`,`uint32`,T);z=r(t,`${n}-edge-polygon`,`uint32`,T),I.push(s(t,{id:`${n}-point-ids`,operation:b,variant:`point-ids`,bindings:[{name:`sortedItems`,view:L.sortedItems,type:`u32`,access:`read`},{name:`groupIndex`,view:R.groupIndex,type:`u32`,access:`read`},{name:`vertexPolygon`,view:P,type:`u32`,access:`read`},{name:`pointIds`,view:e,type:`u32`,access:`read_write`}],invocationCount:T,body:`let vertex = sortedItems[sortedItemsOffset + index];
  pointIds[pointIdsOffset + vertex] = select(${x},
    groupIndex[groupIndexOffset + index] - 1u, vertexPolygon[vertexPolygonOffset + vertex] != ${x});`}),s(t,{id:`${n}-edge-keys`,operation:b,variant:`edge-keys`,bindings:[{name:`vertexPolygon`,view:P,type:`u32`,access:`read`},{name:`vertexNext`,view:F,type:`u32`,access:`read`},{name:`pointIds`,view:e,type:`u32`,access:`read`},{name:`edgeHigh`,view:i,type:`u32`,access:`read_write`},{name:`edgeLow`,view:a,type:`u32`,access:`read_write`},{name:`edgePolygon`,view:z,type:`u32`,access:`read_write`}],invocationCount:T,body:`let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + vertexNext[vertexNextOffset + index]];
  let valid = first != ${x} && second != ${x} && first != second;
  edgeHigh[edgeHighOffset + index] = select(${x}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${x}, max(first, second), valid);
  edgePolygon[edgePolygonOffset + index] = select(${x}, vertexPolygon[vertexPolygonOffset + index], valid);`}));let o=u(t,`${n}-edge`,b,T,i,a,{high:l(T),low:l(T)});V=d(t,`${n}-edge`,b,T,i,a,o.sortedItems),B=o.sortedItems,I.push(...o.nodes,...V.nodes)}let H=r(t,`${n}-pair-counts`,`uint32`,T),U=r(t,`${n}-pair-starts`,`uint32`,T),W=r(t,`${n}-pair-total`,`uint32`,1),G=r(t,`${n}-pair-high`,`uint32`,A),K=r(t,`${n}-pair-low`,`uint32`,A),q=[{name:`sortedItems`,view:B,type:`u32`,access:`read`},{name:`itemPolygon`,view:z,type:`u32`,access:`read`},{name:`groupIndex`,view:V.groupIndex,type:`u32`,access:`read`},{name:`groupStarts`,view:V.groupStarts,type:`u32`,access:`read`}],J=e=>`let item = sortedItems[sortedItemsOffset + index];
  let polygon = itemPolygon[itemPolygonOffset + item];
  if (polygon != ${x}) {
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let begin = groupStarts[groupStartsOffset + groupId];
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    var firstPolygonOccurrence = index == begin;
    if (!firstPolygonOccurrence) {
      let previousItem = sortedItems[sortedItemsOffset + index - 1u];
      firstPolygonOccurrence = itemPolygon[itemPolygonOffset + previousItem] != polygon;
    }
    if (firstPolygonOccurrence) {
      var previousOtherPolygon = ${x};
      for (var other = begin; other < end; other++) {
        let otherPolygon = itemPolygon[itemPolygonOffset + sortedItems[sortedItemsOffset + other]];
        let firstOtherPolygonOccurrence = otherPolygon != previousOtherPolygon;
        previousOtherPolygon = otherPolygon;
        if (firstOtherPolygonOccurrence && otherPolygon != ${x} && otherPolygon != polygon) {
          ${e}
        }
      }
    }
  }`;I.push(s(t,{id:`${n}-pair-counts`,operation:b,variant:`pair-counts`,bindings:[...q,{name:`pairCounts`,view:H,type:`u32`,access:`read_write`}],invocationCount:T,body:`var count = 0u;
  ${J(`count++;`)}
  pairCounts[pairCountsOffset + index] = count;`}),...new a({id:`${n}-pair-scan`,input:H,output:U,mode:`exclusive`}).getCommandNodes(t),...y(t,{id:`${n}-pair`,operation:b,counts:H,offsets:U,rowCount:T,total:W}),s(t,{id:`${n}-pair-fill`,operation:b,variant:`pair-fill`,bindings:[{name:`pairHigh`,view:G,type:`u32`,access:`read_write`},{name:`pairLow`,view:K,type:`u32`,access:`read_write`}],invocationCount:A,body:`pairHigh[pairHighOffset + index] = ${x};
  pairLow[pairLowOffset + index] = ${x};`}),s(t,{id:`${n}-pair-emit`,operation:b,variant:`pair-emit`,bindings:[...q,{name:`pairStarts`,view:U,type:`u32`,access:`read`},{name:`pairHigh`,view:G,type:`u32`,access:`read_write`},{name:`pairLow`,view:K,type:`u32`,access:`read_write`}],invocationCount:T,declarations:`const PAIR_CAPACITY: u32 = ${A}u;`,body:`var next = pairStarts[pairStartsOffset + index];
  ${J(`if (next < PAIR_CAPACITY) {
          pairHigh[pairHighOffset + next] = polygon;
          pairLow[pairLowOffset + next] = otherPolygon;
        }
        next++;`)}`}));let Y=u(t,`${n}-pair`,b,A,G,K,{high:l(E),low:l(E)}),X=r(t,`${n}-unique`,`uint32`,A),Z=r(t,`${n}-unique-rank`,`uint32`,A);return I.push(...Y.nodes,s(t,{id:`${n}-unique-flags`,operation:b,variant:`unique-flags`,bindings:[{name:`sortedPairs`,view:Y.sortedItems,type:`u32`,access:`read`},{name:`pairHigh`,view:G,type:`u32`,access:`read`},{name:`pairLow`,view:K,type:`u32`,access:`read`},{name:`isUnique`,view:X,type:`u32`,access:`read_write`}],invocationCount:A,body:`let item = sortedPairs[sortedPairsOffset + index];
  let high = pairHigh[pairHighOffset + item];
  let low = pairLow[pairLowOffset + item];
  var first = high != ${x};
  if (first && index > 0u) {
    let previous = sortedPairs[sortedPairsOffset + index - 1u];
    first = high != pairHigh[pairHighOffset + previous] || low != pairLow[pairLowOffset + previous];
  }
  isUnique[isUniqueOffset + index] = select(0u, 1u, first);`}),...new a({id:`${n}-unique-scan`,input:X,output:Z,mode:`exclusive`}).getCommandNodes(t),s(t,{id:`${n}-offsets`,operation:b,variant:`offsets`,bindings:[{name:`sortedPairs`,view:Y.sortedItems,type:`u32`,access:`read`},{name:`pairHigh`,view:G,type:`u32`,access:`read`},{name:`isUnique`,view:X,type:`u32`,access:`read`},{name:`uniqueRank`,view:Z,type:`u32`,access:`read`},{name:`offsets`,view:f.offsets,type:`u32`,access:`read_write`}],invocationCount:E+1,declarations:`const CAPACITY: u32 = ${k}u;
const PAIR_CAPACITY: u32 = ${A}u;`,body:`var low = 0u;
  var high = PAIR_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pairHigh[pairHighOffset + sortedPairs[sortedPairsOffset + middle]] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  var start = 0u;
  if (low < PAIR_CAPACITY) {
    start = uniqueRank[uniqueRankOffset + low];
  } else {
    start = uniqueRank[uniqueRankOffset + PAIR_CAPACITY - 1u] + isUnique[isUniqueOffset + PAIR_CAPACITY - 1u];
  }
  offsets[offsetsOffset + index] = min(start, CAPACITY);`}),s(t,{id:`${n}-totals`,operation:b,variant:`totals`,bindings:[{name:`isUnique`,view:X,type:`u32`,access:`read`},{name:`uniqueRank`,view:Z,type:`u32`,access:`read`},{name:`scanTotal`,view:W,type:`u32`,access:`read`},{name:`overflow`,view:S,type:`u32`,access:`read_write`},...C?[{name:`requiredCount`,view:C,type:`u32`,access:`read_write`}]:[],...w?[{name:`candidateOverflow`,view:w,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const CAPACITY: u32 = ${k}u;
const PAIR_CAPACITY: u32 = ${A}u;
const VERTEX_COUNT: u32 = ${T}u;`,body:`let total = uniqueRank[uniqueRankOffset + PAIR_CAPACITY - 1u] + isUnique[isUniqueOffset + PAIR_CAPACITY - 1u];
  let pairTotal = scanTotal[scanTotalOffset];
  overflow[overflowOffset] = select(0u, 1u, total > CAPACITY || pairTotal > PAIR_CAPACITY);
  ${C?`requiredCount[requiredCountOffset] = total;`:``}
  ${w?`candidateOverflow[candidateOverflowOffset] = select(0u, 1u, pairTotal > PAIR_CAPACITY);`:``}`}),s(t,{id:`${n}-emit`,operation:b,variant:`emit`,bindings:[{name:`sortedPairs`,view:Y.sortedItems,type:`u32`,access:`read`},{name:`pairLow`,view:K,type:`u32`,access:`read`},{name:`isUnique`,view:X,type:`u32`,access:`read`},{name:`uniqueRank`,view:Z,type:`u32`,access:`read`},{name:`neighbors`,view:f.neighbors,type:`u32`,access:`read_write`},{name:`weights`,view:f.weights,type:`f32`,access:`read_write`}],invocationCount:A,declarations:`const CAPACITY: u32 = ${k}u;`,body:`let rank = uniqueRank[uniqueRankOffset + index];
  if (isUnique[isUniqueOffset + index] != 0u && rank < CAPACITY) {
    neighbors[neighborsOffset + rank] = pairLow[pairLowOffset + sortedPairs[sortedPairsOffset + index]];
    weights[weightsOffset + rank] = 1.0;
  }`})),I}};function D(e,t){let n=6*e+f(e),r=t===`rook`?C*e+n:0;return S*e+n+r+w*e}function O(e){return(T+6)*e+1}export{E as t};