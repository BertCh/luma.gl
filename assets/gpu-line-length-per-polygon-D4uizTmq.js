import{n as e}from"./spatial-join-geometry-C3pdxIH2.js";import{E as t,S as n,T as r,p as i,u as a}from"./analysis-frame-Blu0tYdw.js";import{_ as o,d as s,h as c,l,v as u,w as d,y as f}from"./resources-CwvhEOBe.js";import{l as p,n as m,s as h}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{u as g}from"./wgsl-common-BNRVzgd8.js";import{t as _}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{n as v,t as y}from"./exact-geometry-wgsl-DOqxBQzc.js";import{o as b}from"./sorted-segment-sums-DwGkmbx2.js";import{n as x,r as S}from"./lon-lat-ring-wgsl-C5XyUe9j.js";import{t as C}from"./geodesic-wgsl-DS0CJZ42.js";import{t as w}from"./gpu-spatial-join-candidates-0HmMq2ES.js";import{t as T}from"./line-segmentize-kernels-JWzYlmh-.js";var E=`GPULineLengthPerPolygon`,D=Object.freeze({...i,edgeModels:a,measure:`distance`});function O(e){return l(e,16)}function k(e){return l(e,8)}var A=class{id;props;polygonCount;spatialContext;coordinateSystem;radius;pairCapacity;maximumCrossingCount;constructor(e){this.id=e.id??`line-length-per-polygon`,this.props=e;let{id:i}=this;if(this.spatialContext=t(i,e.spatialContext,D),r(i,`polygons`,e.polygonsSpatialContext,this.spatialContext,D.edgeModels),this.coordinateSystem=this.spatialContext.coordinateSpace===`longitude-latitude`?`spherical`:`planar`,this.radius=n(this.spatialContext)??1,this.pairCapacity=e.pairCapacity??Math.max(1024,8*e.positions.length),this.maximumCrossingCount=e.maximumCrossingCount??64,s(i,`pairCapacity`,this.pairCapacity),s(i,`maximumCrossingCount`,this.maximumCrossingCount,1,1024),f(e.positions,[`float32x2`],`${i} positions`),e.positions.length<2)throw Error(`${i} needs at least two positions`);if(f(e.lineOffsets,[`uint32`],`${i} lineOffsets`),e.lineOffsets.length<2)throw Error(`${i} lineOffsets needs two rows`);if(e.lineWeights&&(f(e.lineWeights,[`float32`],`${i} lineWeights`),e.lineWeights.length!==e.lineOffsets.length-1))throw Error(`${i} lineWeights must hold one row per line`);if(e.polygons.kind!==`polygons`)throw Error(`${i} polygons must have kind 'polygons'`);for(let[t,n]of[[`featureOffsets`,e.polygons.featureOffsets],[`polygonOffsets`,e.polygons.polygonOffsets],[`ringOffsets`,e.polygons.ringOffsets]])f(n,[`uint32`],`${i} polygons.${t}`);if(f(e.polygons.positions,[`float32x2`],`${i} polygons.positions`),this.polygonCount=e.polygons.featureOffsets.length-1,this.polygonCount<1)throw Error(`${i} polygons must contain at least one feature`);let{lengths:a,weightedLengths:o,segmentCounts:l,status:u}=e.output,{overflow:d}=u;if(f(a,[`float32`],`${i} output.lengths`),a.length!==this.polygonCount)throw Error(`${i} output.lengths must hold ${this.polygonCount} rows`);if(o){if(!e.lineWeights)throw Error(`${i} output.weightedLengths requires lineWeights`);if(f(o,[`float32`],`${i} output.weightedLengths`),o.length!==this.polygonCount)throw Error(`${i} output.weightedLengths must hold ${this.polygonCount} rows`)}if(l&&(f(l,[`uint32`],`${i} output.segmentCounts`),l.length!==this.polygonCount))throw Error(`${i} output.segmentCounts must hold ${this.polygonCount} rows`);if(f(d,[`uint32`],`${i} output.status.overflow`),d.length<1)throw Error(`${i} output.status.overflow must contain one uint32 row`);for(let e of[`antimeridianUncertainCount`,`poleUncertainCount`]){let t=u[e];if(t&&(f(t,[`uint32`],`${i} output.status.${e}`),t.length<1))throw Error(`${i} output.status.${e} must contain one uint32 row`)}c(i,[a,o,l,d,u.antimeridianUncertainCount,u.poleUncertainCount],[e.positions,e.lineOffsets,e.lineWeights,e.polygons.positions,e.polygons.featureOffsets,e.polygons.polygonOffsets,e.polygons.ringOffsets])}getCommandNodes(t){let{props:n,id:r,polygonCount:i,maximumCrossingCount:a}=this,{polygons:c,output:l}=n;o(r,t,[n.positions,n.lineOffsets,n.lineWeights,c.positions,c.featureOffsets,c.polygonOffsets,c.ringOffsets,l.lengths,l.weightedLengths,l.segmentCounts,l.status.overflow,l.status.antimeridianUncertainCount,l.status.poleUncertainCount]),u(r,`positions rows`,n.positions.length,O(t.device));let f=k(t.device),D=n.pairCapacity??Math.min(this.pairCapacity,f);s(r,`pairCapacity`,D,1,f);let A=this.coordinateSystem===`spherical`,j=n.positions.length,M=n.lineOffsets.length-1,N=[];if(this.spatialContext.coordinateSpace===`longitude-latitude`&&S(l.status)){let e=l.status;for(let n of[`antimeridianUncertainCount`,`poleUncertainCount`]){let i=e[n];i&&N.push(m(t,{id:`${r}-clear-${n}`,operation:E,view:i,type:`u32`,value:`0u`,componentCount:1}))}N.push(x(t,{id:`${r}-lon-lat-polygon-check-0`,operation:E,positions:c.positions,ringOffsets:c.ringOffsets,status:e}))}let P=d(t,`${r}-segment-positions`,`float32x2`,2*j),F=d(t,`${r}-segment-offsets`,`uint32`,j+1),I=d(t,`${r}-segment-weights`,`float32`,j);N.push(h(t,{id:`${r}-build-segments`,operation:E,variant:`build-segments`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:n.lineOffsets,type:`u32`,access:`read`},{name:`lineWeights`,view:n.lineWeights??l.lengths,type:`f32`,access:`read`},{name:`segmentPositions`,view:P,type:`f32`,access:`read_write`},{name:`segmentOffsets`,view:F,type:`u32`,access:`read_write`},{name:`segmentWeights`,view:I,type:`f32`,access:`read_write`}],invocationCount:j+1,declarations:`const LINE_COUNT: u32 = ${M}u;
const ROW_COUNT: u32 = ${j}u;
const HAS_WEIGHTS: bool = ${n.lineWeights?`true`:`false`};
${T}`,body:`segmentOffsets[segmentOffsetsOffset + index] = 2u * index;
  if (index < ROW_COUNT) {
    let line = findLine(index);
    var valid = line != NO_LINE;
    var weight = 1.0;
    if (valid) {
      valid = index + 1u < min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
      if (HAS_WEIGHTS) { weight = lineWeights[lineWeightsOffset + line]; }
    }
    var nanBits = 0x7fc00000u;
    let nan = bitcast<f32>(nanBits);
    var a = vec2f(nan);
    var b = vec2f(nan);
    if (valid) {
      a = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
      b = vec2f(positions[positionsOffset + 2u * index + 2u], positions[positionsOffset + 2u * index + 3u]);
    }
    segmentPositions[segmentPositionsOffset + 4u * index] = a.x;
    segmentPositions[segmentPositionsOffset + 4u * index + 1u] = a.y;
    segmentPositions[segmentPositionsOffset + 4u * index + 2u] = b.x;
    segmentPositions[segmentPositionsOffset + 4u * index + 3u] = b.y;
    segmentWeights[segmentWeightsOffset + index] = select(0.0, weight, valid);
  }`}));let L={leftIds:d(t,`${r}-pair-left`,`uint32`,D),rightIds:d(t,`${r}-pair-right`,`uint32`,D),status:{count:d(t,`${r}-pair-count`,`uint32`,1),overflow:d(t,`${r}-pair-overflow`,`uint32`,1)}};N.push(...new w({id:`${r}-candidates`,left:{kind:`lines`,positions:P,lineOffsets:F},right:c,output:L}).getCommandNodes(t));let R=d(t,`${r}-feature-rings`,`uint32x2`,i);N.push(e(t,`${r}-polygon-rings`,{featureCount:i,geometry:c,featureRings:R}));let z=d(t,`${r}-crossing-overflow`,`uint32`,1),B=d(t,`${r}-inside-lengths`,`float32`,D);N.push(m(t,{id:`${r}-fill-crossing-overflow`,operation:E,view:z,type:`u32`,value:`0u`}),h(t,{id:`${r}-clip`,operation:E,variant:A?`clip-spherical`:`clip-planar`,bindings:[{name:`leftIds`,view:L.leftIds,type:`u32`,access:`read`},{name:`rightIds`,view:L.rightIds,type:`u32`,access:`read`},{name:`segmentPositions`,view:P,type:`f32`,access:`read`},{name:`featureRings`,view:R,type:`u32`,access:`read`},{name:`ringOffsets`,view:c.ringOffsets,type:`u32`,access:`read`},{name:`polygonPositions`,view:c.positions,type:`f32`,access:`read`},{name:`insideLengths`,view:B,type:`f32`,access:`read_write`},{name:`crossingOverflow`,view:z,type:`atomic<u32>`,access:`read_write`}],invocationCount:D,declarations:`const NO_FEATURE: u32 = 0xffffffffu;
const VERTEX_COUNT: u32 = ${c.positions.length}u;
const MAXIMUM_CROSSINGS: u32 = ${a}u;
const RADIUS: f32 = ${p(this.radius)};
const EDGE_TOLERANCE: f32 = 1.0e-6;
${A?C:``}
${g}
${_}
${y}

fn getVertex(row: u32) -> vec2f {
  return vec2f(polygonPositions[polygonPositionsOffset + 2u * row], polygonPositions[polygonPositionsOffset + 2u * row + 1u]);
}

var<private> segmentOrigin: vec2f;

fn getRelativeVertex(row: u32) -> vec2f {
  return getVertex(row) - segmentOrigin;
}

fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

fn getPieceLength(a: vec2f, b: vec2f, t0: f32, t1: f32) -> f32 {
  ${A?`return geodesicCentralAngle(mix(a, b, t0), mix(a, b, t1)) * RADIUS;`:`return (t1 - t0) * length(b - a);`}
}

${v(`getRingWinding`,`getRelativeVertex`)}
// Even/odd fill over every ring of the feature, with the shared exact point-in-ring rule
// (utils/exact-geometry-wgsl.ts). Called from one site.
fn isInside(point: vec2f, ringStart: u32, ringEnd: u32) -> bool {
  let probe = ExactProbe(point, point, point, EXACT_PROBE_POINT);
  var winding = 0;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let first = ringOffsets[ringOffsetsOffset + ring];
    let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    winding += getRingWinding(probe, first, last).x;
  }
  return (winding & 1) != 0;
}`,body:`var inside = 0.0;
  let segment = leftIds[leftIdsOffset + index];
  let feature = rightIds[rightIdsOffset + index];
  if (segment != NO_FEATURE && feature != NO_FEATURE) {
    let a = vec2f(segmentPositions[segmentPositionsOffset + 4u * segment], segmentPositions[segmentPositionsOffset + 4u * segment + 1u]);
    let b = vec2f(segmentPositions[segmentPositionsOffset + 4u * segment + 2u], segmentPositions[segmentPositionsOffset + 4u * segment + 3u]);
    let delta = b - a;
    segmentOrigin = a;
    let ringStart = featureRings[featureRingsOffset + 2u * feature];
    let ringEnd = featureRings[featureRingsOffset + 2u * feature + 1u];
    if (!isNanBits(a.x) && !isNanBits(a.y) && !isNanBits(b.x) && !isNanBits(b.y) && (delta.x != 0.0 || delta.y != 0.0)) {
      var crossings = array<f32, ${a}>();
      var crossingCount = 0u;
      var truncated = false;
      for (var ring = ringStart; ring < ringEnd; ring++) {
        let first = ringOffsets[ringOffsetsOffset + ring];
        let last = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
        if (last < first + 3u) { continue; }
        var previous = getVertex(last - 1u);
        for (var row = first; row < last; row++) {
          let current = getVertex(row);
          let edge = current - previous;
          let denominator = cross2(delta, edge);
          if (denominator != 0.0) {
            let offset = previous - a;
            let t = cross2(offset, edge) / denominator;
            let u = cross2(offset, delta) / denominator;
            // Extra breakpoints are harmless (pieces are tested by midpoint), so edges are
            // inclusive with a small tolerance and a vertex crossing is never dropped.
            if (t > 0.0 && t < 1.0 && u >= -EDGE_TOLERANCE && u <= 1.0 + EDGE_TOLERANCE) {
              if (crossingCount < MAXIMUM_CROSSINGS) {
                var slot = crossingCount;
                crossingCount++;
                loop {
                  if (slot == 0u || crossings[slot - 1u] <= t) { break; }
                  crossings[slot] = crossings[slot - 1u];
                  slot--;
                }
                crossings[slot] = t;
              } else {
                truncated = true;
              }
            }
          }
          previous = current;
        }
      }
      if (truncated) { atomicStore(&crossingOverflow[crossingOverflowOffset], 1u); }
      var start = 0.0;
      for (var piece = 0u; piece <= crossingCount; piece++) {
        let end = select(1.0, crossings[min(piece, MAXIMUM_CROSSINGS - 1u)], piece < crossingCount);
        if (end > start) {
          // Classify in the segment's local frame. Adding the world origin can round a
          // genuine interior midpoint onto a hole edge, losing a contributing segment.
          if (isInside(delta * (0.5 * (start + end)), ringStart, ringEnd)) {
            inside += getPieceLength(a, b, start, end);
          }
          start = end;
        }
      }
    }
  }
  insideLengths[insideLengthsOffset + index] = inside;`}));let V=l.segmentCounts,H=d(t,`${r}-keys`,`uint32`,D),U=d(t,`${r}-weighted`,`float32`,D);return V&&N.push(m(t,{id:`${r}-fill-counts`,operation:E,view:V,type:`u32`,value:`0u`})),N.push(h(t,{id:`${r}-finish-pairs`,operation:E,variant:`finish-pairs`,bindings:[{name:`leftIds`,view:L.leftIds,type:`u32`,access:`read`},{name:`rightIds`,view:L.rightIds,type:`u32`,access:`read`},{name:`insideLengths`,view:B,type:`f32`,access:`read`},{name:`segmentWeights`,view:I,type:`f32`,access:`read`},{name:`keys`,view:H,type:`u32`,access:`read_write`},{name:`weighted`,view:U,type:`f32`,access:`read_write`},...V?[{name:`counts`,view:V,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:D,declarations:`const NO_FEATURE: u32 = 0xffffffffu;
const POLYGON_COUNT: u32 = ${i}u;`,body:`let segment = leftIds[leftIdsOffset + index];
  let feature = rightIds[rightIdsOffset + index];
  let inside = insideLengths[insideLengthsOffset + index];
  var key = POLYGON_COUNT;
  var contribution = 0.0;
  if (segment != NO_FEATURE && feature < POLYGON_COUNT && inside > 0.0) {
    key = feature;
    contribution = inside * segmentWeights[segmentWeightsOffset + segment];
    ${V?`atomicAdd(&counts[countsOffset + feature], 1u);`:``}
  }
  keys[keysOffset + index] = key;
  weighted[weightedOffset + index] = contribution;`}),...b(t,{id:`${r}-sum`,operation:E,segmentCount:i,segmentKeys:H,sumContributions:B,sums:l.lengths,reductions:l.weightedLengths?[{name:`weighted`,contributions:U,output:l.weightedLengths}]:[]}),h(t,{id:`${r}-publish`,operation:E,variant:`publish`,bindings:[{name:`pairOverflow`,view:L.status.overflow,type:`u32`,access:`read`},{name:`crossingOverflow`,view:z,type:`u32`,access:`read`},{name:`overflow`,view:l.status.overflow,type:`u32`,access:`read_write`}],invocationCount:1,body:`overflow[overflowOffset] = select(0u, 1u, pairOverflow[pairOverflowOffset] != 0u || crossingOverflow[crossingOverflowOffset] != 0u);`})),N}};export{A as t};