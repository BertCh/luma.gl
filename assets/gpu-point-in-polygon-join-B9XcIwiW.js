import{a as e,i as t,o as n,s as r}from"./spatial-join-geometry-C3pdxIH2.js";import{E as i,T as a,u as o}from"./analysis-frame-Blu0tYdw.js";import{_ as s,b as c,c as l,d as u,f as d,s as f,w as p,y as m}from"./resources-CwvhEOBe.js";import{a as h,n as g,s as _}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as v}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{i as y,n as b,r as x,t as S}from"./polygon-edge-slabs-Cmf_6bAe.js";import{n as C,r as w}from"./lon-lat-ring-wgsl-C5XyUe9j.js";import{n as T}from"./gpu-pairwise-point-in-polygon-DeHbidtn.js";import{_ as E,c as D,f as O,g as k,i as A,m as j,n as M,o as N,p as P,r as F,s as I,v as L,y as R}from"./spatial-join-passes-C0ZwcuNs.js";var z=`GPUPointInPolygonJoin`;function B(e){return e.coordinateSpace===`longitude-latitude`?{coordinateSpace:`longitude-latitude`,metric:`none`,...e.edgeModel===void 0?{}:{edgeModel:e.edgeModel},...e.datum===void 0?{}:{datum:e.datum}}:e}var V=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),edgeModels:o,metrics:Object.freeze([`none`,`native`])}),H=256,U=class{id;props;featureCount;leafCapacity;candidateCapacity;boundaryRule;spatialContext;spatialSort;querySort;edgeSlabs;constructor(t){this.id=t.id??`point-in-polygon-join`,this.props=t;let{id:o}=this,{points:s,polygons:p,output:h}=t,{status:g}=h;if(s.kind!==`points`||p.kind!==`polygons`)throw Error(`${o} needs point and polygon geometry`);for(let e of f(s.positions))m(e,[`float32x2`],`${o} points.positions`);n(o,`polygons`,p,{allowEmpty:!0}),this.featureCount=p.featureOffsets.length-1;for(let[e,t]of[[`output.pointFeatureIds`,h.pointFeatureIds],[`points.sourceIds`,s.sourceIds]]){for(let n of t?f(t):[])m(n,[`uint32`],`${o} ${e}`);R(o,e,s.positions,t)}for(let[e,t]of[[`polygons.sourceIds`,p.sourceIds],[`output.featureCounts`,h.featureCounts]])if(t&&(m(t,[`uint32`],`${o} ${e}`),t.length!==this.featureCount))throw Error(`${o} ${e} needs one row per feature`);if(r(o,g),h.matches&&d(o,h.matches),this.candidateCapacity=t.candidateCapacity,u(o,`candidateCapacity`,this.candidateCapacity,1,2147483646),this.leafCapacity=t.leafCapacity??l(Math.max(this.featureCount,1)),c(o,`leafCapacity`,this.leafCapacity),t.spatialContext?.coordinateSpace===`longitude-latitude`&&t.spatialContext.metric!==`none`)throw Error(`${o} needs metric 'none' on longitude-latitude`);if(this.spatialContext=i(o,t.spatialContext,V),a(o,`polygons`,t.polygonsSpatialContext,this.spatialContext,V.edgeModels),t.boundaryRule!==void 0&&![`half-open`,`closed`,`open`].includes(t.boundaryRule))throw Error(`${o} boundaryRule must be 'half-open', 'closed' or 'open'`);if(this.boundaryRule=t.boundaryRule??`half-open`,this.spatialSort=t.prepared?.spatialSort??t.spatialSort??D(this.featureCount),this.querySort=t.querySort??I(t.points.positions.length,this.featureCount),this.edgeSlabs=t.edgeSlabs??t.prepared?.edgeSlabs??N(p.positions.length,this.candidateCapacity,this.featureCount),t.prepared&&(this.leafCapacity=t.prepared.leafCapacity,!e(t.prepared.geometry,p)))throw Error(`${o} prepared must index the same polygon views`);L(o,W(t),G(t))}getCommandNodes(e){let{id:t,props:n,featureCount:r,leafCapacity:i,candidateCapacity:a}=this,{polygons:o,output:c}=n,l=n.points.positions,{matches:u,status:d}=c;s(t,e,[...W(n),...G(n)]);let f=l.length,m=[];if(this.spatialContext.coordinateSpace===`longitude-latitude`&&w(d)){for(let n of[`antimeridianUncertainCount`,`poleUncertainCount`]){let r=d[n];r&&m.push(g(e,{id:`${t}-clear-${n}`,operation:z,view:r,type:`u32`,value:`0u`,componentCount:1}))}m.push(C(e,{id:`${t}-lon-lat-polygon-check-0`,operation:z,positions:o.positions,ringOffsets:o.ringOffsets,status:d}))}if(n.prepared&&!n.prepared.isDeclaredIn(e))throw Error(`${t} requires prepared to be added to the graph first`);let y;if(n.prepared){let{storage:e}=n.prepared;y={...e,internalNodeCount:n.prepared.bvhInternalNodeCount}}else{let n=p(e,`${t}-feature-minima`,`float32x2`,r),a=p(e,`${t}-feature-maxima`,`float32x2`,r);r>0&&m.push(M(e,{id:`${t}-bounds`,operation:z,featureCount:r,source:o,minima:n,maxima:a}));let s=P(e,t,z,n,a,i,this.spatialSort,void 0,this.props.spatialSortCurve);m.push(...s.nodes),y=s.bvh}let b=p(e,`${t}-state`,`uint32`,4),S=p(e,`${t}-assignment`,`uint32`,f),D=this.edgeSlabs,N=D?4:2,I=p(e,`${t}-candidate-pairs`,`uint32x2`,N/2*a),L=D?void 0:p(e,`${t}-candidate-points`,`float32x2`,a),R,B;if(D){let r=n.prepared?.edgeSlabStorage;if(r)({slabs:R,layout:B}=r);else{let n=x(e,{id:`${t}-edges`,operation:z,owner:`polygon`,positions:o.positions,polygonOffsets:o.polygonOffsets,ringOffsets:o.ringOffsets}),r=O(e,{id:`${t}-edges`,operation:z,polygons:o});m.push(...n.nodes,...r.nodes),R=n.slabs,B=r.layout}}let V=p(e,`${t}-pair-classifications`,`uint32`,a);m.push(F(e,{id:`${t}-clear`,operation:z,state:b,assignment:S,pointCount:f,featureCounts:c.featureCounts})),m.push(...E(e,{id:`${t}-probe`,operation:z,points:l,bvh:y,featureCount:r,candidateCapacity:a,state:b,candidatePairs:I,candidatePoints:L,packedCandidates:D,pointRowScale:1,pointRowOffset:0,querySort:this.querySort}));let U=h(e,{id:`${t}-candidate-dispatch`,operation:z,count:b,maximumItemCount:a});return m.push(U.updateNode),R&&B?m.push(this.createSlabClassifyNode(e,{condition:U.condition,state:b,candidatePairs:I,pairClassifications:V,slabs:R,edgeLayout:B})):m.push(_(e,{id:`${t}-classify`,operation:z,variant:`classify-exact`,record:this.boundaryRule===`half-open`?{formula:`gpu-spatial-analysis.point-in-polygon-join`}:void 0,bindings:[{name:`state`,view:b,type:`u32`,access:`read`},{name:`candidatePairs`,view:I,type:`u32`,access:`read`},{name:`candidatePoints`,view:L,type:`f32`,access:`read`},{name:`featureOffsets`,view:o.featureOffsets,type:`u32`,access:`read`},{name:`polygonPositions`,view:o.positions,type:`f32`,access:`read`},{name:`polygonOffsets`,view:o.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:o.ringOffsets,type:`u32`,access:`read`},{name:`pairClassifications`,view:V,type:`u32`,access:`read_write`}],invocationCount:a,condition:U.condition,declarations:`const CANDIDATE_CAPACITY: u32 = ${a}u;
  const POLYGON_COUNT: u32 = ${o.polygonOffsets.length-1}u;
  const RING_COUNT: u32 = ${o.ringOffsets.length-1}u;
  const VERTEX_COUNT: u32 = ${o.positions.length}u;
  const OUTSIDE: u32 = ${T.outside}u;
  const INSIDE: u32 = ${T.inside}u;
  const BOUNDARY: u32 = ${T.boundary}u;
  const UNCERTAIN: u32 = ${T.uncertain}u;
  const HALF_OPEN_OWNER_BIT: u32 = ${H}u;
  ${v}
  fn readVertex(vertexIndex: u32) -> vec2f {
    return vec2f(
      polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
      polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
    );
  }
  // Exact even/odd classification, or UNCERTAIN when some orientation has no sign. A BOUNDARY result
  // also carries HALF_OPEN_OWNER_BIT when the point belongs to the feature under the half-open rule.
  //
  // Half-open rule: classify the point moved by (epsilon^2, epsilon) with epsilon infinitesimal. The
  // crossing test against an edge only changes when the point is on the edge's supporting line, where
  // the sign of the moved point is the sign of dx (the epsilon term), or of -dy for a vertical edge
  // (the epsilon^2 term). The y comparisons are unchanged because a vertex y is either equal to the
  // point's y or at least one ulp away.
  fn classifyExactly(point: vec2f, geometryStart: u32, geometryEnd: u32) -> u32 {
    var geometryInside = false;
    var geometryOwned = false;
    var onBoundary = false;
    for (var polygonIndex = geometryStart; polygonIndex < geometryEnd; polygonIndex++) {
      let polygonStart = polygonOffsets[polygonOffsetsOffset + polygonIndex];
      let polygonEnd = polygonOffsets[polygonOffsetsOffset + polygonIndex + 1u];
      // Malformed offsets cannot be classified; an empty part contains nothing (as in shapely) and
      // must not hide the other parts of the feature.
      if (polygonStart > polygonEnd || polygonEnd > RING_COUNT) { return UNCERTAIN; }
      var polygonInside = false;
      var polygonOwned = false;
      for (var ringIndex = polygonStart; ringIndex < polygonEnd; ringIndex++) {
        let ringStart = ringOffsets[ringOffsetsOffset + ringIndex];
        let ringEnd = ringOffsets[ringOffsetsOffset + ringIndex + 1u];
        if (ringEnd < ringStart || ringEnd > VERTEX_COUNT) { return UNCERTAIN; }
        // A ring with fewer than 3 vertices bounds no area: skip it, as shapely treats empty parts.
        if (ringEnd < ringStart + 3u) { continue; }
        var previous = readVertex(ringEnd - 1u);
        for (var vertexIndex = ringStart; vertexIndex < ringEnd; vertexIndex++) {
          let current = readVertex(vertexIndex);
          let sign = orientSign(previous, current, point);
          if (sign == 2) { return UNCERTAIN; }
          var side = sign;
          if (sign == 0) {
            if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
                point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
              onBoundary = true;
            }
            side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
            if (side == 0) {
              side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
            }
          }
          if ((previous.y > point.y) != (current.y > point.y)) {
            if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
            if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
            if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
              polygonOwned = !polygonOwned;
            }
          }
          previous = current;
        }
      }
      geometryInside = geometryInside || polygonInside;
      geometryOwned = geometryOwned || polygonOwned;
    }
    if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
    return select(OUTSIDE, INSIDE, geometryInside);
  }`,body:`let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
    if (index >= activeCount) { return; }
    let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let geometryStart = featureOffsets[featureOffsetsOffset + featureRow];
    let geometryEnd = featureOffsets[featureOffsetsOffset + featureRow + 1u];
    var classification = UNCERTAIN;
    if (geometryStart <= geometryEnd && geometryEnd <= POLYGON_COUNT) {
      let point = vec2f(candidatePoints[candidatePointsOffset + index * 2u], candidatePoints[candidatePointsOffset + index * 2u + 1u]);
      classification = classifyExactly(point, geometryStart, geometryEnd);
    }
    pairClassifications[pairClassificationsOffset + index] = classification;`})),m.push(_(e,{id:`${t}-resolve`,operation:z,variant:`resolve`,bindings:[{name:`state`,view:b,type:`atomic<u32>`,access:`read_write`},{name:`candidatePairs`,view:I,type:`u32`,access:`read`},{name:`pairClassifications`,view:V,type:`u32`,access:`read`},{name:`assignment`,view:S,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,condition:U.condition,declarations:`const CANDIDATE_CAPACITY: u32 = ${a}u;
const INSIDE: u32 = ${T.inside}u;
const BOUNDARY: u32 = ${T.boundary}u;
const UNCERTAIN: u32 = ${T.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${H}u;
const BOUNDARY_CLOSED: bool = ${this.boundaryRule===`closed`};
const BOUNDARY_HALF_OPEN: bool = ${this.boundaryRule===`half-open`};`,body:`let activeCount = min(atomicLoad(&state[stateOffset]), CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * ${N}u];
  let featureRow = candidatePairs[candidatePairsOffset + index * ${N}u + 1u];
  let word = pairClassifications[pairClassificationsOffset + index];
  let classification = word & ~HALF_OPEN_OWNER_BIT;
  let owned = BOUNDARY_CLOSED || (BOUNDARY_HALF_OPEN && (word & HALF_OPEN_OWNER_BIT) != 0u);
  if (classification == INSIDE || (owned && classification == BOUNDARY)) {
    atomicMin(&assignment[assignmentOffset + pointRow], featureRow);
  } else if (classification == UNCERTAIN) {
    atomicAdd(&state[stateOffset + 1u], 1u);
  }`})),m.push(...j(e,{id:`${t}-assign`,operation:z,assignment:S,pointFeatureIds:c.pointFeatureIds,featureIds:o.sourceIds,featureCounts:c.featureCounts})),u&&m.push(...k(e,{id:`${t}-collect-matches`,operation:z,points:l,assignment:S,state:b,sourceIds:n.points.sourceIds,matches:u})),m.push(A(e,{id:`${t}-finalize`,operation:z,state:b,bvhOverflow:y.overflow,candidateCapacity:a,overflow:d.overflow,candidateCount:d.candidateCount,uncertainCount:d.uncertainCount,matches:u})),m}createSlabClassifyNode(e,t){let{id:n,candidateCapacity:r}=this,{polygons:i}=this.props,a=b({owner:`polygonIndex`,lowY:`point.y`,highY:`point.y`,getVertex:`readVertex`,pointNames:[`previous`,`current`],body:`let isUsableRing = ((edgeLayout[edgeLayoutOffset + (edgeStart >> 5u)] >> (edgeStart & 31u)) & 1u) != 0u;
        if (!isUsableRing) { continue; }
        let sign = orientSign(previous, current, point);
        if (sign == 2) { return UNCERTAIN; }
        var side = sign;
        if (sign == 0) {
          if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
              point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
            onBoundary = true;
          }
          side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
          if (side == 0) {
            side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
          }
        }
        if ((previous.y > point.y) != (current.y > point.y)) {
          if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
          if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
          if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
            polygonOwned = !polygonOwned;
          }
        }`});return _(e,{id:`${n}-classify`,operation:z,variant:`classify-exact-slabs`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read`},{name:`polygonPositions`,view:i.positions,type:`f32`,access:`read`},...S(t.slabs),{name:`edgeLayout`,view:t.edgeLayout,type:`u32`,access:`read`},{name:`pairClassifications`,view:t.pairClassifications,type:`u32`,access:`read_write`}],invocationCount:r,condition:t.condition,declarations:`const CANDIDATE_CAPACITY: u32 = ${r}u;
const OUTSIDE: u32 = ${T.outside}u;
const INSIDE: u32 = ${T.inside}u;
const BOUNDARY: u32 = ${T.boundary}u;
const UNCERTAIN: u32 = ${T.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${H}u;
const NO_PART: u32 = 0xffffffffu;
const FEATURE_BASE: u32 = ${Math.max(Math.ceil(i.positions.length/32),1)}u;
${v}
${y(t.slabs)}
fn readVertex(vertexIndex: u32) -> vec2f {
  return vec2f(
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
  );
}
// Same classification as the unslabbed classifyExactly, over the edges of the slab at point.y of
// each part: see the edgeSlabs prop. Parity is per part (each part is a slab owner) and combined
// with OR across parts.
fn classifySlabs(point: vec2f, partStart: u32, partEnd: u32) -> u32 {
  var geometryInside = false;
  var geometryOwned = false;
  var onBoundary = false;
  for (var polygonIndex = partStart; polygonIndex < partEnd; polygonIndex++) {
    var polygonInside = false;
    var polygonOwned = false;
    ${a}
    geometryInside = geometryInside || polygonInside;
    geometryOwned = geometryOwned || polygonOwned;
  }
  if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
  return select(OUTSIDE, INSIDE, geometryInside);
}`,body:`let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let candidate = candidatePairsOffset + index * 4u;
  let featureRow = candidatePairs[candidate + 1u];
  let point = vec2f(bitcast<f32>(candidatePairs[candidate + 2u]), bitcast<f32>(candidatePairs[candidate + 3u]));
  let feature = edgeLayoutOffset + FEATURE_BASE + featureRow * 2u;
  var classification = UNCERTAIN;
  let partStart = edgeLayout[feature];
  // A malformed feature (partStart == NO_PART) is UNCERTAIN, as in the unslabbed walk.
  if (partStart != NO_PART) {
    classification = classifySlabs(point, partStart, edgeLayout[feature + 1u]);
  }
  pairClassifications[pairClassificationsOffset + index] = classification;`})}};function W(e){let{points:n,polygons:r}=e;return[n.positions,n.sourceIds,...t(r),r.sourceIds]}function G(e){let{output:t}=e,{matches:n,status:r}=t;return[t.pointFeatureIds,t.featureCounts,r.overflow,r.candidateCount,r.uncertainCount,r.antimeridianUncertainCount,r.poleUncertainCount,n?.ids,n?.status.count,n?.status.overflow,n?.status.requiredCount]}export{B as n,U as t};