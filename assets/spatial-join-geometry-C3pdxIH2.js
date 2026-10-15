import{t as e}from"./status-BwNARr9H.js";import{y as t}from"./resources-CwvhEOBe.js";import{s as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as r,t as i}from"./spatial-join-passes-C0ZwcuNs.js";function a(e,t,n,r){if(n.format!==r)throw Error(`${e} ${t} must use ${r}`)}function o(e,t,n){if(t&&(a(e,`sourceIds`,t,`uint32`),t.length!==n))throw Error(`${e} sourceIds length must equal the feature count`)}function s(e,t){if(a(e,`positions`,t.positions,`float32x2`),t.kind===`points`){o(e,t.sourceIds,t.positions.length);return}if(t.kind===`lines`){if(a(e,`lineOffsets`,t.lineOffsets,`uint32`),t.lineOffsets.length<1)throw Error(`${e} lineOffsets must contain a terminal entry`);o(e,t.sourceIds,t.lineOffsets.length-1);return}if(t.kind===`polygons`){if(a(e,`featureOffsets`,t.featureOffsets,`uint32`),a(e,`polygonOffsets`,t.polygonOffsets,`uint32`),a(e,`ringOffsets`,t.ringOffsets,`uint32`),t.featureOffsets.length<1||t.polygonOffsets.length<1||t.ringOffsets.length<1)throw Error(`${e} polygon offsets must contain terminal entries`);o(e,t.sourceIds,t.featureOffsets.length-1);return}if(a(e,`trackOffsets`,t.trackOffsets,`uint32`),t.timestamps.format!==`float32`&&t.timestamps.format!==`uint32x2`)throw Error(`${e} timestamps must use float32 or uint32x2`);if(t.trackOffsets.length<1)throw Error(`${e} trackOffsets must contain a terminal entry`);if(t.timestamps.length!==t.positions.length)throw Error(`${e} timestamps length must equal positions length`);o(e,t.sourceIds,t.trackOffsets.length-1)}function c(t,n){if(a(t,`leftIds`,n.leftIds,`uint32`),a(t,`rightIds`,n.rightIds,`uint32`),n.leftIds.length!==n.rightIds.length)throw Error(`${t} pair ID capacities must match`);e(`${t} status`,n.status)}var l=`GPUSpatialPredicateJoin`;function u(e){switch(e.kind){case`points`:return e.positions.length;case`lines`:return e.lineOffsets.length-1;case`polygons`:return e.featureOffsets.length-1}}function d(e){return e.kind===`lines`?[e.positions,e.lineOffsets]:e.kind===`polygons`?[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets]:[e.positions]}function f(e,n,r,i={}){for(let i of[...d(r),r.sourceIds])i&&t(i,[i.format],`${e} ${n}`);if(s(`${e} ${n}`,r),!i.allowEmpty&&u(r)<1)throw Error(`${e} ${n} needs at least one feature`)}function p(e,n){for(let[r,i]of Object.entries(n))if(i&&(t(i,[`uint32`],`${e} output.status.${r}`),i.length<1))throw Error(`${e} output.status.${r} needs one row`)}function m(e,t){if(e.kind!==t.kind)return!1;let n=d(e),r=d(t);return n.every((e,t)=>e===r[t])}function h(e,t,r){let{geometry:i}=r;return n(e,{id:t,operation:l,variant:`feature-rings`,bindings:[{name:`featureOffsets`,view:i.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:i.polygonOffsets,type:`u32`,access:`read`},{name:`featureRings`,view:r.featureRings,type:`u32`,access:`read_write`}],invocationCount:r.featureCount,declarations:`const POLYGON_COUNT: u32 = ${i.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${i.ringOffsets.length-1}u;`,body:`let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var ringStart = 0u;
  var ringEnd = 0u;
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let first = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let last = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (first <= last && last <= RING_COUNT) { ringStart = first; ringEnd = last; }
  }
  featureRings[featureRingsOffset + index * 2u] = ringStart;
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`})}function g(e,t,a){let{geometry:o}=a;if(o.kind===`points`)return r(e,{id:t,operation:l,featureCount:a.featureCount,source:{kind:`points`,positions:o.positions},minima:a.minima,maxima:a.maxima});let s=[{name:`positions`,view:o.positions,type:`f32`,access:`read`}],c;return o.kind===`lines`?(s.push({name:`lineOffsets`,view:o.lineOffsets,type:`u32`,access:`read`}),c=`let vertexStart = lineOffsets[lineOffsetsOffset + index];
  let vertexEnd = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);`):(s.push({name:`featureRings`,view:a.featureRings,type:`u32`,access:`read`},{name:`ringOffsets`,view:o.ringOffsets,type:`u32`,access:`read`}),c=`let ringStart = featureRings[featureRingsOffset + index * 2u];
  let ringEnd = featureRings[featureRingsOffset + index * 2u + 1u];
  var vertexStart = 0u;
  var vertexEnd = 0u;
  if (ringStart < ringEnd) {
    vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
    vertexEnd = min(ringOffsets[ringOffsetsOffset + ringEnd], VERTEX_COUNT);
  }`),s.push({name:`featureMinima`,view:a.minima,type:`f32`,access:`read_write`},{name:`featureMaxima`,view:a.maxima,type:`f32`,access:`read_write`}),n(e,{id:t,operation:l,variant:`bounds-${o.kind}`,bindings:s,invocationCount:a.featureCount,declarations:`${i}
const VERTEX_COUNT: u32 = ${o.positions.length}u;`,body:`${c}
  var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    let position = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
      minimum = min(minimum, position);
      maximum = max(maximum, position);
    }
  }
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`})}export{m as a,c,d as i,h as n,f as o,u as r,p as s,g as t};