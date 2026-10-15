import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{J as r,mt as i}from"./src-CNCKFW2b.js";import{E as a,h as o,n as s,s as c}from"./analysis-frame-Blu0tYdw.js";import{i as l}from"./convert-colors-D-TZjU1C.js";import{N as u,_ as d,d as f,h as p,t as m,w as h,y as g}from"./resources-CwvhEOBe.js";import{t as _}from"./gpu-scan-BuGhTLDQ.js";import{t as v}from"./gpu-sort-Bl3Dztdb.js";import{t as y}from"./gpu-segmented-reduction-BOmwUgZN.js";import{t as b}from"./gpu-spatial-clustering-CnxFo0Om.js";import{l as x,n as S,o as C,s as w}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as T,l as E}from"./wgsl-common-BNRVzgd8.js";import{n as D}from"./sorted-segment-offsets-HVRgs63A.js";import{a as O,n as ee,r as k}from"./sorted-segment-sums-DwGkmbx2.js";import{t as A}from"./gpu-geometry-measures-nkjV4Jk0.js";import{t as j}from"./gpu-group-convex-hull-B1eInSxa.js";import{i as M,r as N,s as P,t as F}from"./recipe-utils-CqprOQld.js";import{n as I}from"./spatial-clustering-parameters-OGQKgHDK.js";import{n as te,t as L}from"./gpu-kmeans-Ba8e911_.js";import{U as R,a as z,c as B,d as ne,o as re,r as ie,t as ae,x as V}from"./index-Bjk18V2I.js";import{t as H}from"./graph-buffers-B4_80RlX.js";import{t as U}from"./summary-reader-CsgWxENf.js";import{t as oe}from"./mode-kernels-Dxm2GWvI.js";import"./anchors-D9E9ucGL.js";import{a as se,c as ce,g as le,s as ue}from"./gazetteer-OhisynFF.js";import{l as de}from"./class-table-BVEo3-Vs.js";import"./hue-registry-4FjIJ9xz.js";import{a as fe,n as W,o as pe,r as G,t as me}from"./live-text-83XBWyr1.js";import"./picking-DRmnefrA.js";import"./segments-BtINkq-i.js";import{i as he}from"./b1-nature-data-Clqa3fop.js";import{i as ge,s as _e}from"./b1-points-look-B9wT7IoA.js";import{t as ve}from"./b1-city-geometry-1vs1SQDu.js";import{a as ye,c as be,d as xe,f as Se,h as Ce,i as we,l as Te,m as Ee,n as De,o as Oe,p as ke,r as Ae,s as je,t as Me,u as Ne}from"./nature-clusters-classes-DxuaS0T2.js";var Pe=`
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;
const TWO_PI: f32 = 6.28318530717959;
// Bit test: NaN and infinity have an all-ones exponent, and a float comparison may be folded away.
${E}
${T}
`,Fe=1e-12,Ie=2**24-1;function Le(e={},t=new Float32Array(8)){if(t.length<8)throw Error(`Geographic distribution target must hold 8 elements`);let[n,r]=e.origin??[0,0],i=e.standardDeviations??1,a=e.ellipseConvention??`arcgis`,o=e.medianTolerance??.001;for(let e of[n,r,i,o])if(!Number.isFinite(e))throw Error(`Geographic distribution origin, standardDeviations and medianTolerance must be finite`);if(i<=0)throw Error(`Geographic distribution standardDeviations must be positive`);if(o<0)throw Error(`Geographic distribution medianTolerance must not be negative`);if(a!==`arcgis`&&a!==`standard`)throw Error(`Geographic distribution ellipseConvention must be arcgis or standard`);return t.set([n,r,i,a===`arcgis`?Math.SQRT2:1,e.orientationOnly?1:0,o,0,0]),t}var K=256,q=64,J=1024,Re=4;function ze(e,t){return{partial:[Math.min(J,Math.max(1,Math.ceil(e/K)))*K,K],fold:[t*K,K]}}function Be(e,t){let{id:n,operation:i,rows:a,groupCount:o,outputs:s}=t,c=s.length;r(o>=1&&o<=256),r(c>=1&&c<=Re);let l=Math.min(J,Math.max(1,Math.ceil(a/K))),u=Math.ceil(a/l/K)*K,d=Math.min(q,Math.floor(K/o)),f=o*d,p=Array.from({length:c},(e,t)=>t),m=h(e,`${n}-partial-sums`,`float32`,c*o*l),g=t.countOutput?h(e,`${n}-partial-counts`,`uint32`,o*l):void 0,_=!!g,v=w(e,{id:`${n}-partial`,operation:i,variant:`grouped-partial`,bindings:[...t.bindings,{name:`partialSums`,view:m,type:`f32`,access:`read_write`},...g?[{name:`partialCounts`,view:g,type:`u32`,access:`read_write`}]:[]],workgroupSize:K,invocationCount:l*K,guardIndex:!1,...t.gates?.partial,declarations:`${t.declarations??``}
const GROUP_COUNT: u32 = ${o}u;
const NO_GROUP: u32 = 0xffffffffu;
const ROWS: u32 = ${a}u;
const CHUNK_SIZE: u32 = ${u}u;
const CHUNK_COUNT: u32 = ${l}u;
const TILE_COUNT: u32 = ${u/K}u;
const SLICES: u32 = ${d}u;
const LANES: u32 = ${f}u;
var<workgroup> tileGroups: array<u32, ${K}>;
var<workgroup> tileValues: array<f32, ${K*c}>;
var<workgroup> laneSums: array<f32, ${K*c}>;
var<workgroup> laneCounts: array<u32, ${K}>;`,body:`let chunk = index / ${K}u;
  let begin = chunk * CHUNK_SIZE;
  let end = min(begin + CHUNK_SIZE, ROWS);
  let lane = localInvocationIndex;
  let laneGroup = lane % GROUP_COUNT;
  let laneSlice = lane / GROUP_COUNT;
  ${p.map(e=>`var sum${e} = 0.0;`).join(`
  `)}
  var count = 0u;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let row = begin + tile * ${K}u + lane;
    var group = NO_GROUP;
    ${p.map(e=>`var value${e} = 0.0;`).join(`
    `)}
    if (row < end) {
      ${t.rowSnippet}
    }
    if (group >= GROUP_COUNT) {
      group = NO_GROUP;
    }
    tileGroups[lane] = group;
    ${p.map(e=>`tileValues[${e*K}u + lane] = value${e};`).join(`
    `)}
    workgroupBarrier();
    if (lane < LANES) {
      // Lane (slice, group) owns tile entries slice, slice + SLICES, ...: a fixed order.
      for (var entry = laneSlice; entry < ${K}u; entry += SLICES) {
        if (tileGroups[entry] == laneGroup) {
          count++;
          ${p.map(e=>`sum${e} += tileValues[${e*K}u + entry];`).join(`
          `)}
        }
      }
    }
    workgroupBarrier();
  }
  ${p.map(e=>`laneSums[${e*K}u + lane] = sum${e};`).join(`
  `)}
  laneCounts[lane] = count;
  workgroupBarrier();
  if (lane < GROUP_COUNT) {
    var chunkCountTotal = 0u;
    ${p.map(e=>`var chunkSum${e} = 0.0;`).join(`
    `)}
    for (var slice = 0u; slice < SLICES; slice++) {
      chunkCountTotal += laneCounts[slice * GROUP_COUNT + lane];
      ${p.map(e=>`chunkSum${e} += laneSums[${e*K}u + slice * GROUP_COUNT + lane];`).join(`
      `)}
    }
    ${p.map(e=>`partialSums[partialSumsOffset + (${e}u * GROUP_COUNT + lane) * CHUNK_COUNT + chunk] = chunkSum${e};`).join(`
    `)}
    ${_?`partialCounts[partialCountsOffset + lane * CHUNK_COUNT + chunk] = chunkCountTotal;`:``}
  }`}),y=t.finish;return[v,w(e,{id:`${n}-${t.foldSuffix??`fold`}`,operation:i,variant:`grouped-fold`,bindings:[{name:`partialSums`,view:m,type:`f32`,access:`read`},...g?[{name:`partialCounts`,view:g,type:`u32`,access:`read`}]:[],...s.map((e,t)=>({name:`output${t}`,view:e,type:`f32`,access:`read_write`})),...t.countOutput?[{name:`countOutput`,view:t.countOutput,type:`u32`,access:`read_write`}]:[],...y?.bindings??[]],workgroupSize:K,invocationCount:o*K,guardIndex:!1,...t.gates?.fold,declarations:`${y?.declarations??``}
const GROUP_COUNT: u32 = ${o}u;
const CHUNK_COUNT: u32 = ${l}u;
var<workgroup> reducedSums: array<f32, ${K*c}>;
var<workgroup> reducedCounts: array<u32, ${K}>;`,body:`let group = index / ${K}u;
  let lane = localInvocationIndex;
  ${p.map(e=>`var sum${e} = 0.0;`).join(`
  `)}
  var count = 0u;
  for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${K}u) {
    ${p.map(e=>`sum${e} += partialSums[partialSumsOffset + (${e}u * GROUP_COUNT + group) * CHUNK_COUNT + chunk];`).join(`
    `)}
    ${_?`count += partialCounts[partialCountsOffset + group * CHUNK_COUNT + chunk];`:``}
  }
  ${p.map(e=>`reducedSums[${e*K}u + lane] = sum${e};`).join(`
  `)}
  reducedCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${K/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      ${p.map(e=>`reducedSums[${e*K}u + lane] += reducedSums[${e*K}u + lane + stride];`).join(`
      `)}
      reducedCounts[lane] += reducedCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    ${p.map(e=>`sum${e} = reducedSums[${e*K}u];`).join(`
    `)}
    count = reducedCounts[0];
    ${p.map(e=>`output${e}[output${e}Offset + group] = sum${e};`).join(`
    `)}
    ${t.countOutput?`countOutput[countOutputOffset + group] = count;`:``}
    ${y?.body??``}
  }`})]}var Y=`GPUGeographicDistribution`,Ve=Object.freeze({...o,measure:`distance`}),He=256,Ue=256,We=256,Ge=24,Ke=64;function qe(e,t,n){return`let coincidentWeight = ${n};
  if (coincidentWeight > 0.0) {
    let pullLength = sqrt(${e} * ${e} + ${t} * ${t});
    var gamma = 1.0;
    if (pullLength > 0.0) {
      gamma = min(1.0, coincidentWeight / pullLength);
    }
    stepX = stepX * (1.0 - gamma);
    stepY = stepY * (1.0 - gamma);
  }`}var Je=64,Ye=class{id;props;groupCount;maximumMedianIterations;polygonVertexCount;reduction;spatialContext;constructor(e){this.id=e.id??`geographic-distribution`,this.props=e;let{id:t}=this;this.spatialContext=a(t,e.spatialContext,Ve);let{output:n}=e;g(e.positions,[`float32x2`],`${t} positions`);let r=e.positions.length;if(r<1)throw Error(`${t} needs at least one row`);if(r>Ie)throw Error(`${t} supports fewer than 2^24 rows`);e.weights&&g(e.weights,[`float32`],`${t} weights`),e.groupIds&&g(e.groupIds,[`uint32`],`${t} groupIds`),e.mask&&g(e.mask,[`uint32`],`${t} mask`),e.lineEnds&&g(e.lineEnds,[`float32x2`],`${t} lineEnds`);for(let[n,i]of[[`weights`,e.weights],[`groupIds`,e.groupIds],[`mask`,e.mask],[`lineEnds`,e.lineEnds]])if(i&&i.length!==r)throw Error(`${t} ${n} length must equal positions length`);if(g(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<8)throw Error(`${t} parameters must hold 8 float32 values`);if(this.groupCount=e.groupCount??1,f(t,`groupCount`,this.groupCount,1,Ie),!e.groupIds&&this.groupCount!==1)throw Error(`${t} groupCount must be 1 without groupIds`);this.maximumMedianIterations=e.maximumMedianIterations??Ge,f(t,`maximumMedianIterations`,this.maximumMedianIterations,1,We);let i=e.reduction??`auto`;if(i!==`auto`&&i!==`chunked`&&i!==`sorted`)throw Error(`${t} reduction must be auto, chunked, or sorted`);if(i===`chunked`&&this.groupCount>256)throw Error(`${t} chunked reduction supports at most 256 groups`);this.reduction=i===`auto`?this.groupCount<=Je?`chunked`:`sorted`:i,this.polygonVertexCount=e.polygonVertexCount??Ke,f(t,`polygonVertexCount`,this.polygonVertexCount,3);let o=this.groupCount,s=o*this.polygonVertexCount,c=[[`counts`,n.counts,[`uint32`],o],[`weightSums`,n.weightSums,[`float32`],o],[`meanCenters`,n.meanCenters,[`float32x2`],o],[`medianCenters`,n.medianCenters,[`float32x2`],o],[`medianConverged`,n.medianConverged,[`uint32`],o],[`standardDistances`,n.standardDistances,[`float32`],o],[`ellipses`,n.ellipses,[`float32`],o*3],[`directionalMeans`,n.directionalMeans,[`float32`],o*3],[`ellipseVertices`,n.ellipseVertices,[`float32x2`],s],[`circleVertices`,n.circleVertices,[`float32x2`],s]],l=0;for(let[e,n,r,i]of c)if(n&&(l++,g(n,r,`${t} output.${e}`),n.length<i))throw Error(`${t} output.${e} must hold at least ${i} rows`);if(l===0)throw Error(`${t} needs at least one output`);if(n.directionalMeans&&!e.lineEnds)throw Error(`${t} output.directionalMeans requires lineEnds`);p(t,c.map(([,e])=>e),[e.positions,e.weights,e.groupIds,e.mask,e.lineEnds,e.parameters])}getCommandNodes(e){let{props:t,id:n,groupCount:r,polygonVertexCount:i}=this,{output:a,positions:o,weights:s,groupIds:c,mask:l,lineEnds:u,parameters:f}=t;d(n,e,[o,s,c,l,u,f,...Object.values(a)]);let p=o.length,m=(t,r)=>h(e,`${n}-${t}`,`float32`,r),g=(t,r)=>h(e,`${n}-${t}`,`uint32`,r),y=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),b=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read_write`}),T=(t,r,i,a,o=``,s,c=t)=>w(e,{id:`${n}-${t}`,operation:Y,variant:c,bindings:i,invocationCount:r,declarations:`${Pe}\n${o}`,body:a,...s}),E=`const GROUP_COUNT: u32 = ${r}u;`,D=[],A=this.reduction===`chunked`,j=g(`group-counts`,r),M=m(`weight-sums`,r),N=h(e,`${n}-mean`,`float32x2`,r),P=a.standardDistances||a.ellipses||a.ellipseVertices||a.circleVertices,F=a.standardDistances??(P?m(`standard-distances`,r):void 0),I=a.ellipses??(a.ellipseVertices?m(`ellipses`,r*3):void 0),te=[y(`positions`,o,`f32`),y(`parameters`,f,`f32`),...s?[y(`weights`,s,`f32`)]:[],...c?[y(`groupIds`,c,`u32`)]:[],...l?[y(`mask`,l,`u32`)]:[]],L=`${Pe}
struct RowData {
  group: u32,
  weight: f32,
  x: f32,
  y: f32
}
// An excluded row has group NO_GROUP (and weight 0); x and y are relative to the local origin.
fn loadRow(row: u32) -> RowData {
  var data = RowData(0xffffffffu, 0.0, 0.0, 0.0);
  let x = positions[positionsOffset + row * 2u];
  let y = positions[positionsOffset + row * 2u + 1u];
  var valid = isFiniteValue(x) && isFiniteValue(y);
  var weight = 1.0;
  ${s?`weight = weights[weightsOffset + row]; valid = valid && isFiniteValue(weight) && weight > 0.0;`:``}
  var group = 0u;
  ${c?`group = groupIds[groupIdsOffset + row]; valid = valid && group < GROUP_COUNT;`:``}
  ${l?`valid = valid && mask[maskOffset + row] != 0u;`:``}
  if (valid) {
    data = RowData(group, weight, x - parameters[parametersOffset], y - parameters[parametersOffset + 1u]);
  }
  return data;
}`,R=(t,i)=>Be(e,{id:`${n}-${t}`,operation:Y,rows:p,groupCount:r,bindings:[...te,...i.bindings??[]],declarations:`${L}\n${i.declarations??``}`,rowSnippet:i.rowSnippet,outputs:i.outputs,countOutput:i.countOutput,finish:i.finish,foldSuffix:i.foldSuffix,gates:i.gates}),z=m(`sum-x`,r),B=m(`sum-y`,r),ne=(e,t,n)=>{let i=[y(`weightSums`,M,`f32`),y(`sumXX`,e,`f32`),y(`sumYY`,t,`f32`),y(`sumXY`,n,`f32`),y(`parameters`,f,`f32`)];return F&&i.push(b(`standardDistances`,F)),I&&i.push(b(`ellipses`,I)),T(`spread`,r,i,`let total = weightSums[weightSumsOffset + index];
  let isEmpty = !(total > 0.0);
  let varianceX = sumXX[sumXXOffset + index] / total;
  let varianceY = sumYY[sumYYOffset + index] / total;
  let covariance = sumXY[sumXYOffset + index] / total;
  let multiplier = parameters[parametersOffset + 2u];
  let scale = multiplier * parameters[parametersOffset + 3u];
  ${F?`standardDistances[standardDistancesOffset + index] = select(multiplier * sqrt(max(varianceX + varianceY, 0.0)), getQuietNaN(index), isEmpty);`:``}
  ${I?`let halfSum = 0.5 * (varianceX + varianceY);
  let halfDifference = 0.5 * (varianceX - varianceY);
  let radius = sqrt(halfDifference * halfDifference + covariance * covariance);
  var majorAngle = 0.0;
  if (radius > 0.0) {
    majorAngle = 0.5 * atan2(covariance, halfDifference);
  }
  var angle = majorAngle + HALF_PI;
  if (angle > HALF_PI) {
    angle = angle - PI;
  }
  let nan = getQuietNaN(index);
  ellipses[ellipsesOffset + index * 3u] = select(angle, nan, isEmpty);
  ellipses[ellipsesOffset + index * 3u + 1u] = select(scale * sqrt(max(halfSum - radius, 0.0)), nan, isEmpty);
  ellipses[ellipsesOffset + index * 3u + 2u] = select(scale * sqrt(max(halfSum + radius, 0.0)), nan, isEmpty);`:``}`)},re=(e,t,n,i)=>T(`median-start`,r,[y(`mean`,N,`f32`),b(`median`,e),b(`lastMoves`,t),b(`dispatch`,n.view,`u32`),...i?[b(`medianActive`,i,`u32`)]:[]],`median[medianOffset + index * 4u] = 0.0;
  median[medianOffset + index * 4u + 1u] = 0.0;
  median[medianOffset + index * 4u + 2u] = mean[meanOffset + index * 2u];
  median[medianOffset + index * 4u + 3u] = mean[meanOffset + index * 2u + 1u];
  lastMoves[lastMovesOffset + index] = 0.0;
  if (index == 0u) {
    median[medianOffset + 4u * ${r}u] = 0.0;
    ${i?`medianActive[medianActiveOffset] = 0u;`:``}
    ${n.armWGSL}
  }`),ie=`fn isMedianMoving(medianX: f32, stepLength: f32, tolerance: f32) -> bool {
  return isFiniteValue(medianX) && !(stepLength <= tolerance);
}`,ae=(e,t)=>{let n=[y(`median`,e,`f32`),y(`lastMoves`,t,`f32`),y(`parameters`,f,`f32`)];return a.medianCenters&&n.push(b(`medianCenters`,a.medianCenters)),a.medianConverged&&n.push(b(`medianConverged`,a.medianConverged,`u32`)),T(`median-publish`,r,n,`${a.medianCenters?`medianCenters[medianCentersOffset + index * 2u] =
    (median[medianOffset + index * 4u + 2u] + median[medianOffset + index * 4u]) + parameters[parametersOffset];
  medianCenters[medianCentersOffset + index * 2u + 1u] =
    (median[medianOffset + index * 4u + 3u] + median[medianOffset + index * 4u + 1u]) + parameters[parametersOffset + 1u];`:``}
  ${a.medianConverged?`let isValid = isFiniteValue(median[medianOffset + index * 4u + 2u]) && isFiniteValue(median[medianOffset + index * 4u]);
  medianConverged[medianConvergedOffset + index] = select(0u, 1u, isValid && lastMoves[lastMovesOffset + index] <= parameters[parametersOffset + 5u]);`:``}`)},V=(e,t,n,i)=>T(`direction-publish`,r,[y(`sumCosine`,e,`f32`),y(`sumSine`,t,`f32`),y(`sumLength`,n,`f32`),y(`sumWeight`,i,`f32`),y(`parameters`,f,`f32`),b(`directionalMeans`,a.directionalMeans)],`let total = sumWeight[sumWeightOffset + index];
  let isEmpty = !(total > 0.0);
  let cosine = sumCosine[sumCosineOffset + index];
  let sine = sumSine[sumSineOffset + index];
  let resultant = min(sqrt(cosine * cosine + sine * sine) / total, 1.0);
  let hasAngle = cosine != 0.0 || sine != 0.0;
  var angle = atan2(sine, cosine);
  if (parameters[parametersOffset + 4u] > 0.5) {
    angle = 0.5 * angle;
  }
  let nan = getQuietNaN(index);
  directionalMeans[directionalMeansOffset + index * 3u] = select(angle, nan, isEmpty || !hasAngle);
  directionalMeans[directionalMeansOffset + index * 3u + 1u] = select(1.0 - resultant, nan, isEmpty);
  directionalMeans[directionalMeansOffset + index * 3u + 2u] = select(sumLength[sumLengthOffset + index] / total, nan, isEmpty);`),H=()=>T(`mean`,r,[y(`weightSums`,M,`f32`),y(`sumX`,z,`f32`),y(`sumY`,B,`f32`),b(`mean`,N)],`let total = weightSums[weightSumsOffset + index];
  let isEmpty = !(total > 0.0);
  mean[meanOffset + index * 2u] = select(sumX[sumXOffset + index] / total, getQuietNaN(index), isEmpty);
  mean[meanOffset + index * 2u + 1u] = select(sumY[sumYOffset + index] / total, getQuietNaN(index), isEmpty);`);if(A){if(D.push(...R(`mean-sums`,{rowSnippet:`let data = loadRow(row);
      group = data.group;
      value0 = data.weight;
      value1 = data.weight * data.x;
      value2 = data.weight * data.y;`,outputs:[M,z,B],countOutput:j}),H()),P){let e=m(`sum-xx`,r),t=m(`sum-yy`,r),n=m(`sum-xy`,r);D.push(...R(`moments`,{bindings:[y(`mean`,N,`f32`)],rowSnippet:`let data = loadRow(row);
      group = data.group;
      if (group != NO_GROUP) {
        let dx = data.x - mean[meanOffset + group * 2u];
        let dy = data.y - mean[meanOffset + group * 2u + 1u];
        value0 = data.weight * dx * dx;
        value1 = data.weight * dy * dy;
        value2 = data.weight * dx * dy;
      }`,outputs:[e,t,n]}),ne(e,t,n))}if(a.medianCenters||a.medianConverged){let t=h(e,`${n}-median`,`float32`,4*r+1),i=m(`median-moves`,r),a=m(`median-sum-weight`,r),o=m(`median-sum-x`,r),s=m(`median-sum-y`,r),c=m(`median-sum-coincident`,r),l=x(Fe),u=ze(p,r),d=C(e,{id:`${n}-median-dispatch`,operation:Y,shapes:[u.partial,u.fold]});D.push(re(t,i,d));for(let r=0;r<this.maximumMedianIterations;r++)D.push(...R(`median-${r}`,{gates:{partial:d.getGate(0),fold:d.getGate(1)},bindings:[y(`median`,t,`f32`)],rowSnippet:`let data = loadRow(row);
      group = data.group;
      if (group != NO_GROUP) {
        let opaqueZero = bitcast<u32>(median[medianOffset + 4u * GROUP_COUNT]);
        let relative = bitcast<vec2f>(bitcast<vec2u>(vec2f(data.x, data.y) -
          vec2f(median[medianOffset + group * 4u + 2u], median[medianOffset + group * 4u + 3u])) ^ vec2u(opaqueZero));
        let dx = relative.x - median[medianOffset + group * 4u];
        let dy = relative.y - median[medianOffset + group * 4u + 1u];
        let distanceSquared = dx * dx + dy * dy;
        if (distanceSquared > ${l}) {
          let inverseWeight = data.weight / sqrt(distanceSquared);
          value0 = inverseWeight;
          value1 = inverseWeight * dx;
          value2 = inverseWeight * dy;
        } else {
          value3 = data.weight;
        }
      }`,outputs:[a,o,s,c],foldSuffix:`update`,finish:{bindings:[b(`median`,t),b(`lastMoves`,i)],body:`var stepX = 0.0;
    var stepY = 0.0;
    if (sum0 > 0.0) {
      stepX = sum1 / sum0;
      stepY = sum2 / sum0;
    }
    ${qe(`sum1`,`sum2`,`sum3`)}
    median[medianOffset + group * 4u] = median[medianOffset + group * 4u] + stepX;
    median[medianOffset + group * 4u + 1u] = median[medianOffset + group * 4u + 1u] + stepY;
    lastMoves[lastMovesOffset + group] = sqrt(stepX * stepX + stepY * stepY);`}}),w(e,{id:`${n}-median-${r}-check`,operation:Y,variant:`median-check`,bindings:[y(`median`,t,`f32`),y(`lastMoves`,i,`f32`),y(`parameters`,f,`f32`),b(`dispatch`,d.view,`u32`)],invocationCount:He,workgroupSize:He,guardIndex:!1,declarations:`${Pe}
${ie}
${E}
var<workgroup> moving: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&moving, 0u);
  }
  workgroupBarrier();
  let group = localInvocationIndex;
  if (group < GROUP_COUNT &&
      isMedianMoving(median[medianOffset + group * 4u + 2u] + median[medianOffset + group * 4u], lastMoves[lastMovesOffset + group], parameters[parametersOffset + 5u])) {
    atomicOr(&moving, 1u);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u && atomicLoad(&moving) == 0u) {
    ${d.disarmWGSL}
  }`}));D.push(ae(t,i))}if(a.directionalMeans&&u){let e=m(`direction-sum-cosine`,r),t=m(`direction-sum-sine`,r),n=m(`direction-sum-length`,r),i=m(`direction-sum-weight`,r);D.push(...R(`direction`,{bindings:[y(`lineEnds`,u,`f32`)],declarations:`
fn getLineWeight(weight: f32, dx: f32, dy: f32) -> f32 {
  let lengthSquared = dx * dx + dy * dy;
  let isValid = weight > 0.0 && isFiniteValue(dx) && isFiniteValue(dy) && isFiniteValue(lengthSquared) && lengthSquared > 0.0;
  return select(0.0, weight, isValid);
}`,rowSnippet:`let data = loadRow(row);
      group = data.group;
      if (group != NO_GROUP) {
        let dx = lineEnds[lineEndsOffset + row * 2u] - positions[positionsOffset + row * 2u];
        let dy = lineEnds[lineEndsOffset + row * 2u + 1u] - positions[positionsOffset + row * 2u + 1u];
        let lineWeight = getLineWeight(data.weight, dx, dy);
        if (lineWeight > 0.0) {
          let lengthSquared = dx * dx + dy * dy;
          var cosine = 0.0;
          var sine = 0.0;
          if (parameters[parametersOffset + 4u] > 0.5) {
            cosine = (dx * dx - dy * dy) / lengthSquared;
            sine = 2.0 * dx * dy / lengthSquared;
          } else {
            let lineLength = sqrt(lengthSquared);
            cosine = dx / lineLength;
            sine = dy / lineLength;
          }
          value0 = lineWeight * cosine;
          value1 = lineWeight * sine;
          value2 = lineWeight * sqrt(lengthSquared);
          value3 = lineWeight;
        }
      }`,outputs:[e,t,n,i]}),V(e,t,n,i))}}else{let t=g(`sort-keys`,p),i=g(`sort-indices`,p),d=m(`row-weights`,p),A=[y(`positions`,o,`f32`)];s&&A.push(y(`weights`,s,`f32`)),c&&A.push(y(`groupIds`,c,`u32`)),l&&A.push(y(`mask`,l,`u32`)),A.push(b(`sortKeys`,t,`u32`),b(`sortIndices`,i,`u32`),b(`rowWeights`,d)),D.push(T(`prepare`,p,A,`let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var valid = isFiniteValue(x) && isFiniteValue(y);
  var weight = 1.0;
  ${s?`weight = weights[weightsOffset + index]; valid = valid && isFiniteValue(weight) && weight > 0.0;`:``}
  var group = 0u;
  ${c?`group = groupIds[groupIdsOffset + index]; valid = valid && group < GROUP_COUNT;`:``}
  ${l?`valid = valid && mask[maskOffset + index] != 0u;`:``}
  sortKeys[sortKeysOffset + index] = select(GROUP_COUNT, group, valid);
  sortIndices[sortIndicesOffset + index] = index;
  rowWeights[rowWeightsOffset + index] = select(0.0, weight, valid);`,E),S(e,{id:`${n}-zero-counts`,operation:Y,view:j,type:`u32`,value:`0u`}),T(`count`,p,[y(`sortKeys`,t,`u32`),b(`groupCounts`,j,`atomic<u32>`)],`let group = sortKeys[sortKeysOffset + index];
  if (group < GROUP_COUNT) {
    atomicAdd(&groupCounts[groupCountsOffset + group], 1u);
  }`,E));let F=g(`sorted-keys`,p),I=g(`sorted-indices`,p);D.push(...new v({id:`${n}-sort`,keys:t,values:i,outputKeys:F,outputValues:I,keyBits:O(r)}).getCommandNodes(e));let te=g(`segment-offsets`,r+1);D.push(...new _({id:`${n}-segment-scan`,input:j,output:te,mode:`exclusive`}).getCommandNodes(e),T(`segment-total`,1,[y(`counts`,j,`u32`),b(`segmentOffsets`,te,`u32`)],`segmentOffsets[segmentOffsetsOffset + ${r}u] =
    segmentOffsets[segmentOffsetsOffset + ${r-1}u] + counts[countsOffset + ${r-1}u];`));let L=(t,i,a,o)=>ee(e,{id:`${n}-sum-${t}`,operation:Y,segmentCount:r,input:i,segmentOffsets:te,output:a,...o,offsetsWithinInput:i.length>=p}),R=m(`sorted-x`,p),U=m(`sorted-y`,p),oe=m(`sorted-weights`,p),se=m(`weighted-x`,p),ce=m(`weighted-y`,p);if(D.push(T(`gather`,p,[y(`sortedIndices`,I,`u32`),y(`positions`,o,`f32`),y(`rowWeights`,d,`f32`),y(`parameters`,f,`f32`),b(`sortedX`,R),b(`sortedY`,U),b(`sortedWeights`,oe)],`let row = sortedIndices[sortedIndicesOffset + index];
  let weight = rowWeights[rowWeightsOffset + row];
  let isIncluded = weight > 0.0;
  sortedX[sortedXOffset + index] = select(0.0, positions[positionsOffset + row * 2u] - parameters[parametersOffset], isIncluded);
  sortedY[sortedYOffset + index] = select(0.0, positions[positionsOffset + row * 2u + 1u] - parameters[parametersOffset + 1u], isIncluded);
  sortedWeights[sortedWeightsOffset + index] = weight;`),T(`weighted-coordinates`,p,[y(`sortedX`,R,`f32`),y(`sortedY`,U,`f32`),y(`sortedWeights`,oe,`f32`),b(`weightedX`,se),b(`weightedY`,ce)],`let weight = sortedWeights[sortedWeightsOffset + index];
  weightedX[weightedXOffset + index] = weight * sortedX[sortedXOffset + index];
  weightedY[weightedYOffset + index] = weight * sortedY[sortedYOffset + index];`),L(`weight`,oe,M),L(`x`,se,z),L(`y`,ce,B)),D.push(H()),P){let e=m(`contribution-xx`,p),t=m(`contribution-yy`,p),n=m(`contribution-xy`,p),i=m(`sum-xx`,r),a=m(`sum-yy`,r),o=m(`sum-xy`,r);D.push(T(`moments-diagonal`,p,[y(`sortedKeys`,F,`u32`),y(`sortedX`,R,`f32`),y(`sortedY`,U,`f32`),y(`sortedWeights`,oe,`f32`),y(`mean`,N,`f32`),b(`contributionXX`,e),b(`contributionYY`,t)],`let group = sortedKeys[sortedKeysOffset + index];
  var xx = 0.0;
  var yy = 0.0;
  if (group < GROUP_COUNT) {
    let weight = sortedWeights[sortedWeightsOffset + index];
    let dx = sortedX[sortedXOffset + index] - mean[meanOffset + group * 2u];
    let dy = sortedY[sortedYOffset + index] - mean[meanOffset + group * 2u + 1u];
    xx = weight * dx * dx;
    yy = weight * dy * dy;
  }
  contributionXX[contributionXXOffset + index] = xx;
  contributionYY[contributionYYOffset + index] = yy;`,E),T(`moments-cross`,p,[y(`sortedKeys`,F,`u32`),y(`sortedX`,R,`f32`),y(`sortedY`,U,`f32`),y(`sortedWeights`,oe,`f32`),y(`mean`,N,`f32`),b(`contributionXY`,n)],`let group = sortedKeys[sortedKeysOffset + index];
  var xy = 0.0;
  if (group < GROUP_COUNT) {
    let weight = sortedWeights[sortedWeightsOffset + index];
    let dx = sortedX[sortedXOffset + index] - mean[meanOffset + group * 2u];
    let dy = sortedY[sortedYOffset + index] - mean[meanOffset + group * 2u + 1u];
    xy = weight * dx * dy;
  }
  contributionXY[contributionXYOffset + index] = xy;`,E),L(`xx`,e,i),L(`yy`,t,a),L(`xy`,n,o)),D.push(ne(i,a,o))}if(a.medianCenters||a.medianConverged){let t=h(e,`${n}-median`,`float32`,4*r+1),i=m(`median-moves`,r),a=m(`median-contribution-weight`,p),o=m(`median-contribution-x`,p),s=m(`median-contribution-y`,p),c=m(`median-sum-weight`,r),l=m(`median-sum-x`,r),u=m(`median-sum-y`,r),d=m(`median-contribution-coincident`,p),_=m(`median-sum-coincident`,r),v=x(Fe),S=C(e,{id:`${n}-median-dispatch`,operation:Y,shapes:[[p],[r,Ue],k(r)]}),O=g(`median-active`,1);D.push(re(t,i,S,O));for(let m=0;m<this.maximumMedianIterations;m++){let h=`median-${m}`;D.push(T(`${h}-x`,p,[y(`sortedKeys`,F,`u32`),y(`sortedX`,R,`f32`),y(`sortedY`,U,`f32`),y(`sortedWeights`,oe,`f32`),y(`median`,t,`f32`),b(`contributionWeight`,a),b(`contributionX`,o),b(`contributionCoincident`,d)],`let group = sortedKeys[sortedKeysOffset + index];
  var inverseWeight = 0.0;
  var deltaX = 0.0;
  var coincidentWeight = 0.0;
  if (group < GROUP_COUNT) {
    let opaqueZero = bitcast<u32>(median[medianOffset + 4u * GROUP_COUNT]);
    let relative = bitcast<vec2f>(bitcast<vec2u>(vec2f(sortedX[sortedXOffset + index], sortedY[sortedYOffset + index]) -
      vec2f(median[medianOffset + group * 4u + 2u], median[medianOffset + group * 4u + 3u])) ^ vec2u(opaqueZero));
    let dx = relative.x - median[medianOffset + group * 4u];
    let dy = relative.y - median[medianOffset + group * 4u + 1u];
    let distanceSquared = dx * dx + dy * dy;
    if (distanceSquared > ${v}) {
      inverseWeight = sortedWeights[sortedWeightsOffset + index] / sqrt(distanceSquared);
      deltaX = inverseWeight * dx;
    } else {
      coincidentWeight = sortedWeights[sortedWeightsOffset + index];
    }
  }
  contributionWeight[contributionWeightOffset + index] = inverseWeight;
  contributionX[contributionXOffset + index] = deltaX;
  contributionCoincident[contributionCoincidentOffset + index] = coincidentWeight;`,E,S.getGate(0),`median-x`),T(`${h}-y`,p,[y(`sortedKeys`,F,`u32`),y(`sortedY`,U,`f32`),y(`contributionWeight`,a,`f32`),y(`median`,t,`f32`),b(`contributionY`,s)],`let group = sortedKeys[sortedKeysOffset + index];
  var deltaY = 0.0;
  if (group < GROUP_COUNT) {
    deltaY = contributionWeight[contributionWeightOffset + index] * (bitcast<f32>(bitcast<u32>(sortedY[sortedYOffset + index] - median[medianOffset + group * 4u + 3u]) ^
      bitcast<u32>(median[medianOffset + 4u * GROUP_COUNT])) - median[medianOffset + group * 4u + 1u]);
  }
  contributionY[contributionYOffset + index] = deltaY;`,E,S.getGate(0),`median-y`),L(`${h}-weight`,a,c,S.getGate(2)),L(`${h}-x`,o,l,S.getGate(2)),L(`${h}-y`,s,u,S.getGate(2)),L(`${h}-coincident`,d,_,S.getGate(2)),w(e,{id:`${n}-${h}-update`,operation:Y,variant:`median-update`,bindings:[y(`sumWeight`,c,`f32`),y(`sumX`,l,`f32`),y(`sumY`,u,`f32`),y(`sumCoincident`,_,`f32`),b(`median`,t),b(`lastMoves`,i),y(`parameters`,f,`f32`),b(`medianActive`,O,`atomic<u32>`)],invocationCount:r,workgroupSize:Ue,guardIndex:!1,...S.getGate(1),declarations:`${Pe}
${ie}
var<workgroup> moving: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&moving, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
  let total = sumWeight[sumWeightOffset + index];
  var stepX = 0.0;
  var stepY = 0.0;
  if (total > 0.0) {
    stepX = sumX[sumXOffset + index] / total;
    stepY = sumY[sumYOffset + index] / total;
  }
  ${qe(`sumX[sumXOffset + index]`,`sumY[sumYOffset + index]`,`sumCoincident[sumCoincidentOffset + index]`)}
  let nextX = median[medianOffset + index * 4u] + stepX;
  median[medianOffset + index * 4u] = nextX;
  median[medianOffset + index * 4u + 1u] = median[medianOffset + index * 4u + 1u] + stepY;
  let stepLength = sqrt(stepX * stepX + stepY * stepY);
  lastMoves[lastMovesOffset + index] = stepLength;
  if (isMedianMoving(median[medianOffset + index * 4u + 2u] + nextX, stepLength, parameters[parametersOffset + 5u])) {
    atomicOr(&moving, 1u);
  }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u && atomicLoad(&moving) != 0u) {
    atomicOr(&medianActive[medianActiveOffset], 1u);
  }`}),w(e,{id:`${n}-${h}-check`,operation:Y,variant:`median-check`,bindings:[b(`medianActive`,O,`u32`),b(`dispatch`,S.view,`u32`)],invocationCount:1,body:`if (medianActive[medianActiveOffset] == 0u) {
    ${S.disarmWGSL}
  }
  medianActive[medianActiveOffset] = 0u;`}))}D.push(ae(t,i))}if(a.directionalMeans&&u){let e=m(`direction-contribution-cosine`,p),t=m(`direction-contribution-sine`,p),n=m(`direction-contribution-length`,p),i=m(`direction-contribution-weight`,p),a=m(`direction-sum-cosine`,r),s=m(`direction-sum-sine`,r),c=m(`direction-sum-length`,r),l=m(`direction-sum-weight`,r),h=`
fn getLineWeight(weight: f32, dx: f32, dy: f32) -> f32 {
  let lengthSquared = dx * dx + dy * dy;
  let isValid = weight > 0.0 && isFiniteValue(dx) && isFiniteValue(dy) && isFiniteValue(lengthSquared) && lengthSquared > 0.0;
  return select(0.0, weight, isValid);
}`,g=e=>[y(`sortedIndices`,I,`u32`),y(`positions`,o,`f32`),y(`lineEnds`,u,`f32`),y(`rowWeights`,d,`f32`),...e],_=`let row = sortedIndices[sortedIndicesOffset + index];
  let dx = lineEnds[lineEndsOffset + row * 2u] - positions[positionsOffset + row * 2u];
  let dy = lineEnds[lineEndsOffset + row * 2u + 1u] - positions[positionsOffset + row * 2u + 1u];
  let weight = getLineWeight(rowWeights[rowWeightsOffset + row], dx, dy);`;D.push(T(`direction-angles`,p,g([y(`parameters`,f,`f32`),b(`contributionCosine`,e),b(`contributionSine`,t)]),`${_}
  var cosine = 0.0;
  var sine = 0.0;
  if (weight > 0.0) {
    let lengthSquared = dx * dx + dy * dy;
    if (parameters[parametersOffset + 4u] > 0.5) {
      cosine = (dx * dx - dy * dy) / lengthSquared;
      sine = 2.0 * dx * dy / lengthSquared;
    } else {
      let lineLength = sqrt(lengthSquared);
      cosine = dx / lineLength;
      sine = dy / lineLength;
    }
  }
  contributionCosine[contributionCosineOffset + index] = weight * cosine;
  contributionSine[contributionSineOffset + index] = weight * sine;`,h),T(`direction-lengths`,p,g([b(`contributionLength`,n),b(`contributionWeight`,i)]),`${_}
  contributionLength[contributionLengthOffset + index] = weight * sqrt(dx * dx + dy * dy) * select(0.0, 1.0, weight > 0.0);
  contributionWeight[contributionWeightOffset + index] = weight;`,h),L(`direction-cosine`,e,a),L(`direction-sine`,t,s),L(`direction-length`,n,c),L(`direction-weight`,i,l),V(a,s,c,l))}}if(a.counts||a.weightSums||a.meanCenters){let e=[y(`groupCounts`,j,`u32`),y(`weightSums`,M,`f32`),y(`mean`,N,`f32`),y(`parameters`,f,`f32`)];a.counts&&e.push(b(`countsOut`,a.counts,`u32`)),a.weightSums&&e.push(b(`weightSumsOut`,a.weightSums)),a.meanCenters&&e.push(b(`meanCentersOut`,a.meanCenters)),D.push(T(`publish`,r,e,`${a.counts?`countsOut[countsOutOffset + index] = groupCounts[groupCountsOffset + index];`:``}
  ${a.weightSums?`weightSumsOut[weightSumsOutOffset + index] = weightSums[weightSumsOffset + index];`:``}
  ${a.meanCenters?`meanCentersOut[meanCentersOutOffset + index * 2u] = mean[meanOffset + index * 2u] + parameters[parametersOffset];
  meanCentersOut[meanCentersOutOffset + index * 2u + 1u] = mean[meanOffset + index * 2u + 1u] + parameters[parametersOffset + 1u];`:``}`))}let U=`const VERTEX_COUNT: u32 = ${i}u;`;return a.ellipseVertices&&I&&D.push(T(`ellipse-vertices`,r*i,[y(`ellipses`,I,`f32`),y(`mean`,N,`f32`),y(`parameters`,f,`f32`),b(`vertices`,a.ellipseVertices)],`let group = index / VERTEX_COUNT;
  let t = TWO_PI * f32(index % VERTEX_COUNT) / f32(VERTEX_COUNT);
  let angle = ellipses[ellipsesOffset + group * 3u];
  let localX = ellipses[ellipsesOffset + group * 3u + 1u] * cos(t);
  let localY = ellipses[ellipsesOffset + group * 3u + 2u] * sin(t);
  vertices[verticesOffset + index * 2u] = mean[meanOffset + group * 2u] + parameters[parametersOffset] + localX * cos(angle) - localY * sin(angle);
  vertices[verticesOffset + index * 2u + 1u] = mean[meanOffset + group * 2u + 1u] + parameters[parametersOffset + 1u] + localX * sin(angle) + localY * cos(angle);`,U)),a.circleVertices&&F&&D.push(T(`circle-vertices`,r*i,[y(`standardDistances`,F,`f32`),y(`mean`,N,`f32`),y(`parameters`,f,`f32`),b(`vertices`,a.circleVertices)],`let group = index / VERTEX_COUNT;
  let t = TWO_PI * f32(index % VERTEX_COUNT) / f32(VERTEX_COUNT);
  let radius = standardDistances[standardDistancesOffset + group];
  vertices[verticesOffset + index * 2u] = mean[meanOffset + group * 2u] + parameters[parametersOffset] + radius * cos(t);
  vertices[verticesOffset + index * 2u + 1u] = mean[meanOffset + group * 2u + 1u] + parameters[parametersOffset + 1u] + radius * sin(t);`,U)),D}},Xe=`GPUGroupGeometry`,Ze=Object.freeze({...o,measure:`distance`}),X=2**24-1,Qe=4096,$e=4294967295,et=class{id;props;spatialContext;groupCount;medoidMaximumGroupSize;constructor(e){this.id=e.id??`group-geometry`,this.props=e,this.spatialContext=a(this.id,e.spatialContext,Ze);let{id:t}=this,{output:n}=e;g(e.positions,[`float32x2`],`${t} positions`);let r=e.positions.length;if(r<1)throw Error(`${t} needs at least one row`);if(r>X)throw Error(`${t} supports fewer than 2^24 rows`);if(g(e.labels,[`uint32`],`${t} labels`),e.labels.length!==r)throw Error(`${t} labels length must equal positions length`);if(e.weights&&(g(e.weights,[`float32`],`${t} weights`),e.weights.length!==r))throw Error(`${t} weights length must equal positions length`);if(g(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<8)throw Error(`${t} parameters must hold 8 float32 values`);this.groupCount=e.groupCount,f(t,`groupCount`,this.groupCount,1,X),e.noiseLabel!==void 0&&f(t,`noiseLabel`,e.noiseLabel,0),this.medoidMaximumGroupSize=e.medoidMaximumGroupSize??Qe,f(t,`medoidMaximumGroupSize`,this.medoidMaximumGroupSize);let i=this.groupCount,o=[[`counts`,n.counts,[`uint32`],i],[`bounds`,n.bounds,[`float32x4`],i],[`meanCenters`,n.meanCenters,[`float32x2`],i],[`weightedCenters`,n.weightedCenters,[`float32x2`],i],[`weightSums`,n.weightSums,[`float32`],i],[`medoidIndices`,n.medoidIndices,[`uint32`],i],[`ellipses`,n.ellipses,[`float32`],i*3],[`standardDistances`,n.standardDistances,[`float32`],i],[`overflow`,n.status?.overflow,[`uint32`],1]],s=0;for(let[e,n,r,i]of o)if(n&&(s++,g(n,r,`${t} output.${e}`),n.length<i))throw Error(`${t} output.${e} must hold at least ${i} rows`);if(s===0)throw Error(`${t} needs at least one output`);if((n.weightedCenters||n.weightSums)&&!e.weights)throw Error(`${t} output.weightedCenters and output.weightSums require weights`);if(n.status?.overflow&&!n.medoidIndices)throw Error(`${t} output.status.overflow requires output.medoidIndices`);p(t,o.map(([,e])=>e),[e.positions,e.labels,e.weights,e.parameters])}getCommandNodes(e){let{props:t,id:n,groupCount:r,medoidMaximumGroupSize:i}=this,{output:a,positions:o,labels:s,weights:c,parameters:l}=t,{status:u,...f}=a;d(n,e,[o,s,c,l,...Object.values(f),u?.overflow]);let p=o.length,m=(t,r)=>h(e,`${n}-${t}`,`float32`,r),g=(t,r)=>h(e,`${n}-${t}`,`uint32`,r),_=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),b=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read_write`}),x=(t,i,a,o,s=``)=>w(e,{id:`${n}-${t}`,operation:Xe,variant:t,bindings:a,invocationCount:i,declarations:`const GROUP_COUNT: u32 = ${r}u;\n${s}`,body:o}),C=[],E=g(`group-keys`,p),ee=g(`row-indices`,p),k=g(`group-counts`,r),A=t.noiseLabel!==void 0&&t.noiseLabel<r?`valid = valid && label != ${t.noiseLabel}u;`:``;C.push(x(`keys`,p,[_(`positions`,o,`f32`),_(`labels`,s,`u32`),b(`groupKeys`,E,`u32`),b(`rowIndices`,ee,`u32`)],`let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let label = labels[labelsOffset + index];
  var valid = (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(y) & 0x7f800000u) != 0x7f800000u;
  valid = valid && label < GROUP_COUNT;
  ${A}
  groupKeys[groupKeysOffset + index] = select(GROUP_COUNT, label, valid);
  rowIndices[rowIndicesOffset + index] = index;`));let j=g(`sorted-keys`,p),M=g(`sorted-indices`,p),N=g(`segment-offsets`,r+1);C.push(...new v({id:`${n}-sort`,keys:E,values:ee,outputKeys:j,outputValues:M,keyBits:O(r)}).getCommandNodes(e),...D(e,{id:n,operation:Xe,groupCount:r,sortedGroupKeys:j,segmentOffsets:N,counts:k}));let P=!!(a.bounds||a.medoidIndices),F=P?m(`sorted-x`,p):void 0,I=P?m(`sorted-y`,p):void 0;if(F&&I&&C.push(x(`gather`,p,[_(`sortedIndices`,M,`u32`),_(`positions`,o,`f32`),b(`sortedX`,F),b(`sortedY`,I)],`let row = sortedIndices[sortedIndicesOffset + index];
  sortedX[sortedXOffset + index] = positions[positionsOffset + row * 2u];
  sortedY[sortedYOffset + index] = positions[positionsOffset + row * 2u + 1u];`)),a.bounds&&F&&I){let t=(t,i,a)=>{let o=m(`bounds-${t}`,r);return C.push(...new y({id:`${n}-bounds-${t}`,input:i,segmentOffsets:N,output:o,operation:a}).getCommandNodes(e)),o},i=t(`min-x`,F,`min`),o=t(`min-y`,I,`min`),s=t(`max-x`,F,`max`),c=t(`max-y`,I,`max`);C.push(x(`publish-bounds`,r,[_(`counts`,k,`u32`),_(`minX`,i,`f32`),_(`minY`,o,`f32`),_(`maxX`,s,`f32`),_(`maxY`,c,`f32`),b(`bounds`,a.bounds)],`let isEmpty = counts[countsOffset + index] == 0u;
  let nan = getQuietNaN(index);
  bounds[boundsOffset + index * 4u] = select(minX[minXOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 1u] = select(minY[minYOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 2u] = select(maxX[maxXOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 3u] = select(maxY[maxYOffset + index], nan, isEmpty);`,T))}a.counts&&C.push(x(`publish-counts`,r,[_(`counts`,k,`u32`),b(`countsOut`,a.counts,`u32`)],`countsOut[countsOutOffset + index] = counts[countsOffset + index];`));let te=!!(a.ellipses||a.standardDistances);if(a.meanCenters||te&&!c){let t=a.meanCenters??h(e,`${n}-mean-centers`,`float32x2`,r);C.push(...new Ye({id:`${n}-unweighted`,positions:o,groupIds:E,groupCount:r,parameters:l,output:{meanCenters:t,ellipses:c?void 0:a.ellipses,standardDistances:c?void 0:a.standardDistances}}).getCommandNodes(e))}if(c&&(a.weightedCenters||a.weightSums||te)&&C.push(...new Ye({id:`${n}-weighted`,positions:o,weights:c,groupIds:E,groupCount:r,parameters:l,output:{meanCenters:a.weightedCenters??h(e,`${n}-weighted-centers`,`float32x2`,r),weightSums:a.weightSums,ellipses:a.ellipses,standardDistances:a.standardDistances}}).getCommandNodes(e)),a.medoidIndices&&F&&I){let t=m(`medoid-costs`,p);C.push(x(`medoid-cost`,p,[_(`sortedKeys`,j,`u32`),_(`segmentOffsets`,N,`u32`),_(`sortedX`,F,`f32`),_(`sortedY`,I,`f32`),b(`costs`,t)],`let group = sortedKeys[sortedKeysOffset + index];
  var cost = 0.0;
  if (group < GROUP_COUNT) {
    let begin = segmentOffsets[segmentOffsetsOffset + group];
    let end = segmentOffsets[segmentOffsetsOffset + group + 1u];
    if (end - begin <= MEDOID_MAXIMUM_GROUP_SIZE) {
      let x = sortedX[sortedXOffset + index];
      let y = sortedY[sortedYOffset + index];
      for (var other = begin; other < end; other++) {
        let dx = sortedX[sortedXOffset + other] - x;
        let dy = sortedY[sortedYOffset + other] - y;
        cost += sqrt(dx * dx + dy * dy);
      }
    }
  }
  costs[costsOffset + index] = cost;`,`const MEDOID_MAXIMUM_GROUP_SIZE: u32 = ${i}u;`)),a.status?.overflow&&C.push(S(e,{id:`${n}-zero-overflow`,operation:Xe,view:a.status.overflow,type:`u32`,value:`0u`,componentCount:1}));let o=[_(`segmentOffsets`,N,`u32`),_(`costs`,t,`f32`),_(`sortedIndices`,M,`u32`),b(`medoids`,a.medoidIndices,`u32`)];a.status?.overflow&&o.push(b(`overflow`,a.status.overflow,`atomic<u32>`)),C.push(x(`medoid-select`,r,o,`let begin = segmentOffsets[segmentOffsetsOffset + index];
  let end = segmentOffsets[segmentOffsetsOffset + index + 1u];
  var best = NO_MEDOID;
  if (end - begin > MEDOID_MAXIMUM_GROUP_SIZE) {
    ${a.status?.overflow?`atomicStore(&overflow[overflowOffset], 1u);`:``}
  } else if (end > begin) {
    var bestCost = costs[costsOffset + begin];
    best = sortedIndices[sortedIndicesOffset + begin];
    for (var row = begin + 1u; row < end; row++) {
      let cost = costs[costsOffset + row];
      if (cost < bestCost) {
        bestCost = cost;
        best = sortedIndices[sortedIndicesOffset + row];
      }
    }
  }
  medoids[medoidsOffset + index] = best;`,`const MEDOID_MAXIMUM_GROUP_SIZE: u32 = ${i}u;
const NO_MEDOID: u32 = ${$e}u;`))}return C}},tt=`GPUClusterAndOutlineRecipe`,nt=Object.freeze({...o,measure:`distance`});function rt(e,t){let n=t.id??`cluster-outline-recipe`,r=a(n,t.spatialContext,nt),i=t.points.positions,o=i.length,s=t.maximumClusterCount,c=t.outputs??{},l=t.scratch??{};N(tt,s>=1,`maximumClusterCount must be positive`);let u=new F(e),d=P(e,`${n}-labels`,`uint32`,o,c.labels),f=P(e,`${n}-cluster-count`,`uint32`,1,c.clusterCount),p=P(e,`${n}-cluster-ids`,`uint32`,s,l.clusterIds),m=P(e,`${n}-cluster-bound-count`,`uint32`,1,l.clusterBoundCount),h=P(e,`${n}-cluster-overflow`,`uint32`,1,c.status?.clusterOverflow);u.add(new b({id:`${n}-clustering`,spatialContext:r,positions:i,sourceIds:t.points.sourceIds,parameters:t.clusteringParameters,gridSize:t.gridSize,output:{labels:d,clusterCount:f,clusters:{ids:p,status:{count:m,overflow:h}}}}));let g=P(e,`${n}-counts`,`uint32`,s,l.counts),_=P(e,`${n}-bounds`,`float32x4`,s,l.bounds),v=P(e,`${n}-mean-centers`,`float32x2`,s,l.meanCenters);u.add(new et({id:`${n}-group-geometry`,spatialContext:r,positions:i,labels:d,groupCount:s,parameters:t.geometryParameters,output:{counts:g,bounds:_,meanCenters:v}}));let y=P(e,`${n}-hull-vertex-indices`,`uint32`,t.hullCapacity,l.hullVertexIndices),x=P(e,`${n}-hull-positions`,`float32x2`,t.hullCapacity,c.hullPositions),S=P(e,`${n}-hull-offsets`,`uint32`,s+1,c.hullOffsets),C=P(e,`${n}-hull-counts`,`uint32`,s,l.hullCounts),w=P(e,`${n}-hull-overflow`,`uint32`,1,c.status?.hullOverflow);u.add(new j({id:`${n}-convex-hull`,spatialContext:r,positions:i,labels:d,groupCount:s,maximumVerticesPerGroup:t.maximumVerticesPerHull,vertexCapacity:t.hullCapacity,output:{vertexIndices:y,vertexPositions:x,offsets:S,counts:C,status:{overflow:w}}}));let T=P(e,`${n}-areas`,`float32`,s,c.areas),E=P(e,`${n}-perimeters`,`float32`,s,c.perimeters),D=P(e,`${n}-centroids`,`float32x2`,s,c.centroids),O=P(e,`${n}-invalid-count`,`uint32`,1,c.status?.invalidCount),ee=M(e,`${n}-hull-identity`,s);return u.add(new A({spatialContext:r,id:`${n}-hull-measures`,geometryType:`polygons`,positions:x,featureOffsets:ee,polygonOffsets:ee,ringOffsets:S,output:{areas:T,lengths:E,centroids:D,status:{invalidCount:O}}})),{contributors:u.contributors,spatialContext:r,maximumClusterCount:s,labels:d,clusterCount:f,counts:g,bounds:_,meanCenters:v,hullVertexIndices:y,hullPositions:x,hullOffsets:S,hullCounts:C,areas:T,perimeters:E,centroids:D,outputs:{labels:d,clusterCount:f,hullPositions:x,hullOffsets:S,areas:T,perimeters:E,centroids:D},intermediates:{counts:g,bounds:_,meanCenters:v,hullVertexIndices:y,hullCounts:C},status:{clusterOverflow:h,hullOverflow:w,invalidCount:O,stages:[{stage:`clustering`,status:{count:m,requiredCount:f,overflow:h}},{stage:`convex-hull`,status:{overflow:w}},{stage:`hull-measures`,status:{invalidCount:O}}]}}}var it=160,at={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},ot=class extends n{static layerName=`StorageModelLayer`;static defaultProps={coordinateSystem:t.METER_OFFSETS,parameters:at};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:it,usage:i.UNIFORM|i.COPY_DST}),r=new l(t,{...this.getShaders({modules:[e],source:this.getShaderSource()}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:this.getVertexCount(),instanceCount:0,bufferLayout:[],bindings:this.getBindings(n),parameters:at});this.setState({model:r,styleBuffer:n})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n}=this.state;t.setBindings(this.getBindings(n))}getModels(){return[this.state.model]}_drawLayer(e){let{shaderModuleProps:n}=e;this.props.analysisFrame&&n?.project&&(n={...n,project:{...n.project,coordinateSystem:t.LNGLAT,coordinateOrigin:[0,0,0]}}),super._drawLayer({...e,shaderModuleProps:n})}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;this.writeStyle(n),t.setInstanceCount(this.getInstanceCount()),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}},st=8,ct=`
struct ShapeStyle {
  palette: array<vec4<f32>, ${st}>,
  // widthPixels, opacity, sizePixels, unused
  numbers: vec4<f32>,
  // groupCount, segments, shape (0 disc, 1 ring, 2 diamond), mode (0 centers, 1 medoid rows)
  config: vec4<u32>,
};

@group(0) @binding(auto) var<uniform> shapeStyle: ShapeStyle;

fn projectShapePosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

fn isFiniteVector(value: vec2<f32>) -> bool {
  return (bitcast<u32>(value.x) & 0x7f800000u) != 0x7f800000u &&
    (bitcast<u32>(value.y) & 0x7f800000u) != 0x7f800000u;
}

`;function lt(e,t){return t?e.replace(/fn projectShapePosition\(position: vec2<f32>\) -> vec4<f32> \{[\s\S]*?\n\}/,ne(t).replaceAll(`projectSpatialAnalysisPosition`,`projectShapePosition`)):e}var ut=`
struct LineOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) side: f32,
};

fn getHiddenLine() -> LineOutput {
  var output: LineOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.side = 0.0;
  return output;
}

// One screen-space quad (six vertices) along the segment from start to end.
fn getLineVertex(start: vec2<f32>, end: vec2<f32>, vertexIndex: u32, color: vec4<f32>) -> LineOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  if (!isFiniteVector(start) || !isFiniteVector(end)) { return getHiddenLine(); }
  let startClip = projectShapePosition(start);
  let endClip = projectShapePosition(end);
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * shapeStyle.numbers.x * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  var output: LineOutput;
  output.position = clipPosition;
  output.color = vec4<f32>(color.rgb, color.a * shapeStyle.numbers.y);
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: LineOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`,dt=`
${ct}
${ut}
@group(0) @binding(auto) var<storage, read> groupCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> groupCenters: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> groupEllipses: array<f32>;

fn getEllipsePoint(group: u32, step: u32) -> vec2<f32> {
  let segments = shapeStyle.config.y;
  let t = 6.283185307179586 * f32(step) / f32(segments);
  let angle = groupEllipses[group * 3u];
  let local = vec2<f32>(
    groupEllipses[group * 3u + 1u] * cos(t),
    groupEllipses[group * 3u + 2u] * sin(t)
  );
  return groupCenters[group] + vec2<f32>(
    local.x * cos(angle) - local.y * sin(angle),
    local.x * sin(angle) + local.y * cos(angle)
  );
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instance: u32
) -> LineOutput {
  let segments = shapeStyle.config.y;
  let group = instance / segments;
  let step = instance % segments;
  if (group >= shapeStyle.config.x || groupCounts[group] < 3u) { return getHiddenLine(); }
  return getLineVertex(
    getEllipsePoint(group, step),
    getEllipsePoint(group, step + 1u),
    vertexIndex,
    shapeStyle.palette[group % ${st}u]
  );
}
`,ft=`
${ct}
${ut}
@group(0) @binding(auto) var<storage, read> groupCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> groupBounds: array<vec4<f32>>;

fn getBoundsCorner(bounds: vec4<f32>, corner: u32) -> vec2<f32> {
  let x = select(bounds.x, bounds.z, corner == 1u || corner == 2u);
  let y = select(bounds.y, bounds.w, corner == 2u || corner == 3u);
  return vec2<f32>(x, y);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instance: u32
) -> LineOutput {
  let group = instance / 4u;
  let edge = instance % 4u;
  if (group >= shapeStyle.config.x || groupCounts[group] == 0u) { return getHiddenLine(); }
  let bounds = groupBounds[group];
  return getLineVertex(
    getBoundsCorner(bounds, edge),
    getBoundsCorner(bounds, (edge + 1u) % 4u),
    vertexIndex,
    shapeStyle.palette[group % ${st}u]
  );
}
`,pt=`
${ct}
@group(0) @binding(auto) var<storage, read> groupCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> markerCenters: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> markerRows: array<u32>;
@group(0) @binding(auto) var<storage, read> markerPositions: array<vec2<f32>>;

struct MarkerOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) group: u32
) -> MarkerOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: MarkerOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.corner = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  if (group >= shapeStyle.config.x || groupCounts[group] == 0u) { return output; }
  var position = markerCenters[group];
  if (shapeStyle.config.w == 1u) {
    let row = markerRows[group];
    if (row == 0xffffffffu) { return output; }
    position = markerPositions[row];
  }
  if (!isFiniteVector(position)) { return output; }
  let corner = corners[vertexIndex];
  var clipPosition = projectShapePosition(position);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * shapeStyle.numbers.z),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = vec4<f32>(shapeStyle.palette[0].rgb, shapeStyle.palette[0].a * shapeStyle.numbers.y);
  return output;
}

@fragment fn fragmentMain(input: MarkerOutput) -> @location(0) vec4<f32> {
  var inside = length(input.corner);
  if (shapeStyle.config.z == 2u) {
    inside = abs(input.corner.x) + abs(input.corner.y);
  }
  if (inside > 1.0) { discard; }
  var alpha = 1.0;
  if (shapeStyle.config.z == 1u) {
    alpha = smoothstep(0.45, 0.6, inside);
  }
  return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
`;function mt(e,t,n={}){let r=new ArrayBuffer(160),i=new Float32Array(r),a=new Uint32Array(r);for(let e=0;e<st;e++){let n=t.palette[e%t.palette.length];i.set([n[0]/255,n[1]/255,n[2]/255,n[3]/255],e*4)}i.set([t.widthPixels??1.5,t.opacity??1,n.sizePixels??6,0],32),a.set([t.groupCount,n.segments??1,n.shape??0,n.mode??0],36),e.write(new Uint8Array(r))}var ht=class extends ot{static layerName=`EllipseOutlineLayer`;getShaderSource(){return lt(dt,this.props.analysisFrame)}getVertexCount(){return 6}getInstanceCount(){return this.props.groupCount*(this.props.segments??48)}getBindings(e){return{shapeStyle:e,groupCounts:this.props.counts,groupCenters:this.props.centers,groupEllipses:this.props.ellipses}}writeStyle(e){mt(e,this.props,{segments:this.props.segments??48})}},gt=class extends ot{static layerName=`BoundsOutlineLayer`;getShaderSource(){return lt(ft,this.props.analysisFrame)}getVertexCount(){return 6}getInstanceCount(){return this.props.groupCount*4}getBindings(e){return{shapeStyle:e,groupCounts:this.props.counts,groupBounds:this.props.bounds}}writeStyle(e){mt(e,this.props)}},_t={disc:0,ring:1,diamond:2},vt=class extends ot{static layerName=`GroupMarkerLayer`;getShaderSource(){return lt(pt,this.props.analysisFrame)}getVertexCount(){return 6}getInstanceCount(){return this.props.groupCount}getBindings(e){return{shapeStyle:e,groupCounts:this.props.counts,markerCenters:this.props.centers,markerRows:this.props.medoidRows,markerPositions:this.props.positions}}writeStyle(e){mt(e,this.props,{sizePixels:this.props.sizePixels??6,shape:_t[this.props.shape??`disc`],mode:this.props.source===`medoids`?1:0})}},yt=[256,256],Z=2048,bt=10,xt=1024,St=256,Q=16384,Ct=250,$=64,wt=48,Tt=3500,Et=2e3,Dt=60,Ot=14.5,kt=.2;async function At(e){let{device:t}=e,n=e.datasets.get(`chicago-nature`),r=he(n),i=r.origin,a=s({kind:`azimuthal-equidistant`,origin:i}),o=c(a,n.column(`position`)),l=se(le,`montrose-point`),d=l?a.project(l.lngLat[0],l.lngLat[1]):[0,0],f=0,p=!1,h=!1,g,_=0,v=[],y=NaN,x=NaN,S=null,C=null,w=[],T=!1,E=``,D=-1,O={serial:-1,running:!1,awaiting:!1,index:0,stale:!1,minimumPoints:0,counts:[],result:null},ee=e=>[e.category,e.method,e.pipeline,e.denseBoxShortcut,e.sumOrder,e.k,e.initialization,e.seed,e.iterationStep,e.tolerance,e.prefilterLevels].join(`|`),k=e=>Me[Math.min(Math.max(Math.round(e.iterationStep),0),5)],A=t=>{let n={origin:[0,0],standardDeviations:Number(e.options.standardDeviations),ellipseConvention:e.options.ellipseConvention};t.geometryParameters.write(Le(n)),t.distributionParameters.write(Le(n))},M=new Map,N=e=>{let t=M.get(e);if(!t){let n=r.categoryNames.indexOf(e),i=[];for(let e=0;e<r.count;e++)r.category[e]===n&&i.push(e);t=Uint32Array.from(i),M.set(e,t)}return t},P=t=>{let n=e.options.weight===`researchGrade`?r.researchGrade:e.options.weight===`introduced`?r.introduced:null,i=new Float32Array(t.length);for(let e=0;e<t.length;e++)i[e]=1+(n?4*n[t[e]]:0);return i},F=e=>{let t=new Map;for(let n=0;n<e.length/2;n++){let r=`${e[n*2]},${e[n*2+1]}`;t.set(r,(t.get(r)??0)+1)}let n=0;for(let e of t.values())e>1&&(n+=e);return e.length?n/(e.length/2):0};async function ne(e){let n=++f,r=new m(t,`nature-clusters-${n}`),i=N(e.category),s=i.length,c=new Float32Array(s*2),l=1/0,d=1/0,p=-1/0,h=-1/0;for(let e=0;e<s;e++){let t=o[i[e]*2],n=o[i[e]*2+1];c[e*2]=t,c[e*2+1]=n,l=Math.min(l,t),p=Math.max(p,t),d=Math.min(d,n),h=Math.max(h,n)}let g=[l-bt,d-bt,p+bt,h+bt],_=e.method===`dbscan`,v=_&&e.pipeline===`recipe`,y=_?Z:e.k,x=r.createBuffer(`positions`,c),S=r.createBuffer(`weights`,P(i)),C=r.createParameterBuffer(`clustering-parameters`,`float32`,8),w=r.createParameterBuffer(`geometry-parameters`,`float32`,8),T=r.createParameterBuffer(`distribution-parameters`,`float32`,8),E=r.createBuffer(`labels`,s*4),D=r.createBuffer(`core-flags`,s*4),O=r.createBuffer(`cluster-count`,4),M=r.createBuffer(`cluster-stored-count`,4),I=r.createBuffer(`cluster-overflow`,4),R=r.createBuffer(`iteration-count`,4),z=r.createBuffer(`converged`,4),B=r.createBuffer(`sizes`,y*4),ne=r.createBuffer(`centroids`,y*8),re=r.createBuffer(`geometry-counts`,y*4),ie=r.createBuffer(`geometry-bounds`,y*16),ae=r.createBuffer(`mean-centers`,y*8),V=r.createBuffer(`weighted-centers`,y*8),se=r.createBuffer(`medoids`,y*4),ce=r.createBuffer(`ellipses`,y*12),le=r.createBuffer(`geometry-overflow`,4),ue=r.createBuffer(`hull-indices`,Q*4),de=r.createBuffer(`hull-positions`,Q*8),fe=r.createBuffer(`hull-offsets`,(y+1)*4),W=r.createBuffer(`hull-counts`,y*4),pe=r.createBuffer(`hull-overflow`,4),G=r.createBuffer(`areas`,y*4),me=r.createBuffer(`perimeters`,y*4),he=r.createBuffer(`hull-centroids`,y*8),ge=r.createBuffer(`point-sizes`,s*4),_e=r.createBuffer(`point-slots`,s*4),ve=r.createBuffer(`group-sizes`,y*4),ye=r.createBuffer(`group-slots`,y*4),be=r.createBuffer(`noise-ids`,s*4),xe=r.createBuffer(`member-ids`,s*4),Se=r.createBuffer(`core-ids`,s*4),Ce=r.createBuffer(`border-ids`,s*4),we=r.createBuffer(`hull-triangles`,Q*3*8),Te=r.createBuffer(`hull-triangle-features`,Q*3*4),Ee=r.createBuffer(`hull-outline`,Q*16),De=r.createBuffer(`hull-outline-features`,Q*4),Oe,ke=null;if(v){let e=new u(t,{id:`nature-recipe-${n}`}),i=(t,n,r,i)=>H(e,t,n,r,i);rt(e,{id:`cluster-outline`,points:{kind:`points`,positions:i(`positions`,x,`float32x2`,s)},spatialContext:a.spatialContext,clusteringParameters:C.importToGraph(e),gridSize:yt,geometryParameters:w.importToGraph(e),maximumClusterCount:Z,maximumVerticesPerHull:St,hullCapacity:Q,outputs:{labels:i(`labels`,E,`uint32`,s),clusterCount:i(`cluster-count`,O,`uint32`,1),hullPositions:i(`hull-positions`,de,`float32x2`,Q),hullOffsets:i(`hull-offsets`,fe,`uint32`,y+1),status:{hullOverflow:i(`hull-overflow`,pe,`uint32`,1)},areas:i(`areas`,G,`float32`,y),perimeters:i(`perimeters`,me,`float32`,y),centroids:i(`hull-centroids`,he,`float32x2`,y)},scratch:{counts:i(`counts`,re,`uint32`,y),bounds:i(`bounds`,ie,`float32x4`,y),meanCenters:i(`mean-centers`,ae,`float32x2`,y),hullVertexIndices:i(`hull-indices`,ue,`uint32`,Q),hullCounts:i(`hull-counts`,W,`uint32`,y)}}),Oe=r.track(await e.compileAsync())}else{if(_){let i=new u(t,{id:`nature-dbscan-${n}`}),o=r.createBuffer(`cluster-ids`,Z*4);i.add(new b({spatialContext:a.spatialContext,id:`clusters`,positions:H(i,`positions`,x,`float32x2`,s),parameters:C.importToGraph(i),gridSize:yt,denseBoxShortcut:e.denseBoxShortcut,sumOrder:e.sumOrder,output:{labels:H(i,`labels`,E,`uint32`,s),coreFlags:H(i,`core-flags`,D,`uint32`,s),clusterCount:H(i,`cluster-count`,O,`uint32`,1),clusters:{ids:H(i,`cluster-ids`,o,`uint32`,Z),status:{count:H(i,`cluster-stored-count`,M,`uint32`,1),overflow:H(i,`cluster-overflow`,I,`uint32`,1)}},clusterSizes:H(i,`cluster-sizes`,B,`uint32`,Z),clusterCentroids:H(i,`cluster-centroids`,ne,`float32x2`,Z)}})),Oe=r.track(await i.compileAsync())}else{let i=new u(t,{id:`nature-kmeans-${n}`}),o=r.createParameterBuffer(`kmeans-parameters`,`uint32`,2,te({seed:e.seed,tolerance:e.tolerance}));i.add(new L({spatialContext:a.spatialContext,id:`kmeans`,positions:H(i,`positions`,x,`float32x2`,s),k:e.k,maximumIterations:k(e),parameters:o.importToGraph(i),initialization:e.initialization,output:{status:{iterationCount:H(i,`iteration-count`,R,`uint32`,1),converged:H(i,`converged`,z,`uint32`,1)},labels:H(i,`labels`,E,`uint32`,s),centers:H(i,`centers`,ne,`float32x2`,e.k),sizes:H(i,`sizes`,B,`uint32`,e.k)}})),Oe=r.track(await i.compileAsync())}let i=new u(t,{id:`nature-shapes-${n}`}),o=H(i,`positions`,x,`float32x2`,s),c=H(i,`labels`,E,`uint32`,s);i.add(new et({spatialContext:a.spatialContext,id:`geometry`,positions:o,labels:c,groupCount:y,weights:H(i,`weights`,S,`float32`,s),parameters:w.importToGraph(i),medoidMaximumGroupSize:xt,output:{counts:H(i,`geometry-counts`,re,`uint32`,y),bounds:H(i,`geometry-bounds`,ie,`float32x4`,y),meanCenters:H(i,`mean-centers`,ae,`float32x2`,y),weightedCenters:H(i,`weighted-centers`,V,`float32x2`,y),medoidIndices:H(i,`medoids`,se,`uint32`,y),ellipses:H(i,`ellipses`,ce,`float32`,y*3),status:{overflow:H(i,`geometry-overflow`,le,`uint32`,1)}}})),i.add(new j({spatialContext:a.spatialContext,id:`hulls`,positions:o,labels:c,groupCount:y,maximumVerticesPerGroup:St,vertexCapacity:Q,prefilterLevels:Number(e.prefilterLevels),output:{vertexIndices:H(i,`hull-indices`,ue,`uint32`,Q),vertexPositions:H(i,`hull-positions`,de,`float32x2`,Q),offsets:H(i,`hull-offsets`,fe,`uint32`,y+1),counts:H(i,`hull-counts`,W,`uint32`,y),status:{overflow:H(i,`hull-overflow`,pe,`uint32`,1)}}})),ke=r.track(await i.compileAsync())}let Ae=r.createBuffer(`distribution-counts`,4),je=r.createBuffer(`distribution-mean`,8),Me=r.createBuffer(`distribution-median`,8),Ne=r.createBuffer(`distribution-standard-distance`,4),Pe=r.createBuffer(`distribution-ellipse`,12),Fe=r.createBuffer(`ellipse-vertices`,$*8),Le=r.createBuffer(`circle-vertices`,$*8),K=r.createBuffer(`ellipse-segments`,$*16),q=r.createBuffer(`circle-segments`,$*16),J=new u(t,{id:`nature-distribution-${n}`});{let e=(e,t,n,r)=>H(J,e,t,n,r),t=e(`ellipse-vertices`,Fe,`float32x2`,$),n=e(`circle-vertices`,Le,`float32x2`,$);J.add(new Ye({spatialContext:a.spatialContext,id:`distribution`,positions:e(`positions`,x,`float32x2`,s),weights:e(`weights`,S,`float32`,s),polygonVertexCount:$,parameters:T.importToGraph(J),output:{counts:e(`counts`,Ae,`uint32`,1),meanCenters:e(`mean`,je,`float32x2`,1),medianCenters:e(`median`,Me,`float32x2`,1),standardDistances:e(`standard-distance`,Ne,`float32`,1),ellipses:e(`ellipse`,Pe,`float32`,3),ellipseVertices:t,circleVertices:n}}));for(let[e,r,i]of[[`ellipse`,t,K],[`circle`,n,q]])oe(J,{id:`${e}-segments`,invocationCount:$,bindings:[{name:`vertices`,view:r,type:`f32`,access:`read`},{name:`segments`,view:H(J,`${e}-segments`,i,`float32`,$*4),type:`f32`,access:`read_write`}],declarations:`const RING: u32 = ${$}u;`,body:`
  let next = (index + 1u) % RING;
  segments[segmentsOffset + index * 4u] = vertices[verticesOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = vertices[verticesOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = vertices[verticesOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = vertices[verticesOffset + next * 2u + 1u];`})}let Re=r.track(await J.compileAsync()),ze=[{name:`labels`,buffer:E,size:s*4},{name:`coreFlags`,buffer:D,size:s*4},{name:`clusterCount`,buffer:O,size:4},{name:`clusterOverflow`,buffer:I,size:4},{name:`geometryOverflow`,buffer:le,size:4},{name:`hullOverflow`,buffer:pe,size:4},{name:`iterationCount`,buffer:R,size:4},{name:`converged`,buffer:z,size:4},{name:`hullOffsets`,buffer:fe,size:(y+1)*4},{name:`hullCounts`,buffer:W,size:y*4},{name:`hullPositions`,buffer:de,size:Q*8},{name:`geometryBounds`,buffer:ie,size:y*16},{name:`meanCenters`,buffer:ae,size:y*8},{name:`weightedCenters`,buffer:V,size:y*8},{name:`medoids`,buffer:se,size:y*4},{name:`areas`,buffer:G,size:y*4},{name:`perimeters`,buffer:me,size:y*4},{name:`distributionCounts`,buffer:Ae,size:4},{name:`distributionMean`,buffer:je,size:8},{name:`distributionMedian`,buffer:Me,size:8},{name:`distributionStandardDistance`,buffer:Ne,size:4},{name:`distributionEllipse`,buffer:Pe,size:12}],Be=new Map,Y=0;for(let e of ze)Be.set(e.name,{offset:Y,size:e.size}),Y+=e.size;let Ve=new U(r,`detail-${n}`,ze.map(({buffer:e,size:t})=>({buffer:e,size:t})),e=>Ie(He,{u32:t=>{let n=Be.get(t);return new Uint32Array(e,n.offset,n.size/4)},f32:t=>{let n=Be.get(t);return new Float32Array(e,n.offset,n.size/4)}})),He={serial:n,key:ee(e),config:{...e},resources:r,pointCount:s,positions:c,bounds:g,clusteringParameters:C,geometryParameters:w,distributionParameters:T,clusterGraph:Oe,shapeGraph:ke,distributionGraph:Re,groupCount:y,weights:S,positionsBuffer:x,labels:E,coreFlags:D,clusterCount:O,clusterStoredCount:M,clusterOverflow:I,iterationCount:R,converged:z,sizes:B,centroids:ne,geometryCounts:re,geometryBounds:ie,meanCenters:ae,weightedCenters:V,medoids:se,ellipses:ce,geometryOverflow:le,hullOffsets:fe,hullCounts:W,hullPositions:de,hullOverflow:pe,areas:G,perimeters:me,distributionCounts:Ae,distributionMean:je,distributionMedian:Me,distributionStandardDistance:Ne,distributionEllipse:Pe,ellipseSegments:K,circleSegments:q,pointSizes:ge,pointSlots:_e,groupSizes:ve,groupSlots:ye,noiseIds:be,memberIds:xe,coreIds:Se,borderIds:Ce,hullTriangles:we,hullTriangleFeatures:Te,hullOutline:Ee,hullOutlineFeatures:De,drawCounts:{noise:0,members:0,core:0,border:0,hullVertices:0,hullSegments:0},stackedShare:F(c),analysis:null,detail:null,partitionSlots:[],discs:null,water:null,sweep:null,reader:Ve,dirty:!0,destroyed:!1};return A(He),He}async function Pe(e){if(e.sweep||e.destroyed||e.config.method!==`dbscan`)return;let{resources:n}=e,r=new u(t,{id:`nature-sweep-${e.serial}`}),i=n.createParameterBuffer(`sweep-parameters`,`float32`,8),o=n.createBuffer(`sweep-labels`,e.pointCount*4),s=n.createBuffer(`sweep-count`,4),c=n.createBuffer(`sweep-ids`,Z*4),l=n.createBuffer(`sweep-stored`,4),d=n.createBuffer(`sweep-overflow`,4);r.add(new b({spatialContext:a.spatialContext,id:`sweep`,positions:H(r,`positions`,e.positionsBuffer,`float32x2`,e.pointCount),parameters:i.importToGraph(r),gridSize:yt,sumOrder:e.config.sumOrder,output:{labels:H(r,`labels`,o,`uint32`,e.pointCount),clusterCount:H(r,`count`,s,`uint32`,1),clusters:{ids:H(r,`ids`,c,`uint32`,Z),status:{count:H(r,`stored`,l,`uint32`,1),overflow:H(r,`overflow`,d,`uint32`,1)}}}}));let f=n.track(await r.compileAsync());e.destroyed||e!==X||(e.sweep={graph:f,parameters:i,clusterCount:s,reader:new U(n,`sweep-${e.serial}`,[{buffer:s,size:4}],t=>Fe(e,new Uint32Array(t)[0]))},O.serial=e.serial,O.stale=!0,O.result=null)}function Fe(e,t){p||e.destroyed||e!==X||!O.running||(O.counts.push(t),O.awaiting=!1,O.index++,O.index>=Ae.length&&(O.running=!1,O.result={minimumPoints:O.minimumPoints,counts:[...O.counts]},Ue(e),Je(e)))}function Ie(t,n){if(p||t.destroyed||t!==X)return;let r=t.config.method===`dbscan`,i=r&&t.config.pipeline===`recipe`,a=Te({positions:t.positions,labels:n.u32(`labels`),coreFlags:r&&!i?n.u32(`coreFlags`):null,groupCount:t.groupCount,hasNoise:r});if(Ne(a,n.u32(`hullOffsets`),n.u32(`hullCounts`),n.f32(`hullPositions`)),t.analysis=a,t.detail=n,t.discs=null,t.water=null,t.drawCounts.noise=K(t.noiseIds,a.noiseIds),t.drawCounts.members=K(t.memberIds,a.memberIds),t.drawCounts.core=K(t.coreIds,a.coreIds),t.drawCounts.border=K(t.borderIds,a.borderIds),t.pointSizes.write(a.pointSizes),t.groupSizes.write(a.groupSizes),!r){let e=Se(a.groupCentres,t.groupCount,w,7);t.partitionSlots=e,w=[],e.forEach((e,t)=>{a.sizes[t]>0&&w.push({center:[a.groupCentres[t*2],a.groupCentres[t*2+1]],color:e})});let r=n.u32(`labels`),i=new Uint32Array(t.pointCount);for(let n=0;n<t.pointCount;n++)i[n]=e[r[n]]??0;t.pointSlots.write(i),t.groupSlots.write(Uint32Array.from(e))}let o=xe(a.hulls);t.drawCounts.hullVertices=o.triangleFeatures.length,t.drawCounts.hullSegments=o.outlineFeatures.length,o.triangleFeatures.length>0&&(t.hullTriangles.write(o.triangles),t.hullTriangleFeatures.write(o.triangleFeatures),t.hullOutline.write(o.outline),t.hullOutlineFeatures.write(o.outlineFeatures)),Re(t),e.requestLayers()}function K(e,t){return t.length>0&&e.write(t),t.length}let q=(e,t)=>a.unproject(e,t),J=(e,t)=>e.analysis?.hulls.find(e=>e.group===t);function Re(e){Be(e),ze(e),Ue(e),qe(e),Je(e),Xe(e)}function ze(e){He(e),Y(e),Xe(e)}function Be(t){let n=t.analysis,r=t.detail;if(!n||!r)return;let i=t.config.method===`dbscan`,a=i&&t.config.pipeline===`recipe`,o=t.pointCount,s=n.ranked.length?n.sizes[n.ranked[0]]:0;e.setReadout(`points`,W(o)),e.setReadout(`stacked`,fe(t.stackedShare,0)),e.setReadout(`clusters`,W(n.clusterCount)),e.setReadout(`clusteredShare`,i?fe(o?n.memberCount/o:0,0):`100%`),e.setReadout(`clustered`,`${W(n.memberCount)} / ${W(n.noiseCount)}`),e.setReadout(`largest`,W(s)),e.setReadout(`meanSize`,n.clusterCount>0?(n.memberCount/n.clusterCount).toFixed(1):null),e.setReadout(`coreCount`,a?`n/a (recipe)`:W(n.coreCount)),e.setReadout(`borderCount`,a?`n/a (recipe)`:W(n.borderCount)),e.setReadout(`noiseCount`,W(n.noiseCount)),e.setReadout(`parameters`,i?`DBSCAN ${e.options.epsilon} m / ${e.options.minimumPoints} points${t.config.denseBoxShortcut?`, dense-box`:``}${a?`, recipe`:``}`:`K-means k = ${t.config.k}, ${t.config.initialization}${t.config.initialization===`kmeans++`?` seed ${t.config.seed}`:``}, cap ${k(t.config)}`);let c=r.u32(`iterationCount`)[0];if(e.setReadout(`convergence`,i?`n/a (DBSCAN)`:r.u32(`converged`)[0]?`converged after ${c} of ${k(t.config)} iterations`:`not converged: stopped at the ${c} iteration cap`),i)e.setReadout(`farthest`,`n/a (DBSCAN leaves noise)`);else{let i=0,a=r.u32(`labels`);for(let e=0;e<t.pointCount;e++){let r=a[e];i=Math.max(i,Math.hypot(t.positions[e*2]-n.groupCentres[r*2],t.positions[e*2+1]-n.groupCentres[r*2+1]))}e.setReadout(`farthest`,`${G(i)} from its group centre`)}let l=[r.u32(`hullOverflow`)[0]&1?`a hull over 256 vertices (dropped)`:``,r.u32(`hullOverflow`)[0]&2?`total capacity`:``].filter(Boolean);e.setReadout(`hulls`,`${W(n.hulls.length)} hulls${l.length?`, OVERFLOW: ${l.join(`, `)}`:``}`);let u=n.ranked[0];if(u!==void 0){let i=r.f32(`meanCenters`),o=i[u*2],s=i[u*2+1];if(a){e.setReadout(`weightedShift`,`n/a (recipe)`),e.setReadout(`medoid`,`n/a (recipe)`);let t=r.f32(`areas`),i=0;for(let e of n.ranked)i+=t[e];e.setReadout(`recipeHulls`,`GPU areas: largest ${me(t[u])}, all ${me(i)}; largest perimeter ${G(r.f32(`perimeters`)[u])}`)}else{let n=r.f32(`weightedCenters`);e.setReadout(`weightedShift`,`${Math.hypot(n[u*2]-o,n[u*2+1]-s).toFixed(1)} m`);let i=r.u32(`medoids`)[u];e.setReadout(`medoid`,i===4294967295?`none (cluster over ${xt} members)`:`${Math.hypot(t.positions[i*2]-o,t.positions[i*2+1]-s).toFixed(0)} m from the mean centre`),e.setReadout(`recipeHulls`,`recipe pipeline only`)}}let d=r.f32(`distributionMean`),f=r.f32(`distributionMedian`);e.setReadout(`medianGap`,`${G(Math.hypot(f[0]-d[0],f[1]-d[1]))} apart`),e.setReadout(`standardDistance`,G(r.f32(`distributionStandardDistance`)[0]));let p=r.f32(`distributionEllipse`);e.setReadout(`ellipse`,`${G(Math.max(p[1],p[2]))} by ${G(Math.min(p[1],p[2]))}, rotated ${(p[0]*180/Math.PI).toFixed(0)} deg`),e.setReadout(`numerics`,`Hull cap ${St} vertices per hot spot, ${W(Q)} in all; sums ${t.config.sumOrder}; dense-box ${t.config.denseBoxShortcut?`on`:`off`} (3 to 9 % at 200,000 points, rarely pays at this size)`)}function Y(t){let n=t.analysis;if(!n||t.config.method!==`dbscan`||n.ranked.length===0){for(let t of[`winnerSize`,`hullArea`,`hullWaterShare`,`density`])e.setReadout(t,null);return}let r=n.ranked[0],i=n.sizes[r];e.setReadout(`winnerSize`,W(i));let a=J(t,r);if(!a){e.setReadout(`hullArea`,`hull over the vertex cap`),e.setReadout(`hullWaterShare`,null),e.setReadout(`density`,null);return}if(e.setReadout(`hullArea`,me(a.areaSquareMeters)),e.setReadout(`density`,pe(i/(a.areaSquareMeters/1e6),`km²`)),e.options.zoomToWinner&&C){let n=`${t.serial}|${a.group}|${a.size}|${a.areaSquareMeters.toFixed(0)}`;t.water?.key!==n&&(t.water={key:n,share:Ee(a,C.containsMeters,R(1),Et)}),e.setReadout(`hullWaterShare`,fe(t.water.share,0))}else e.setReadout(`hullWaterShare`,e.options.zoomToWinner?`measuring...`:null);if(T){T=!1;let n=Ve(t,r);n&&setTimeout(()=>{p||e.fitBounds(n,{transitionMs:1200,maxZoom:Ot})},450)}}function Ve(e,t){let n=e.detail?.f32(`geometryBounds`);if(!n||!((e.analysis?.sizes[t]??0)>0))return null;let[r,i]=q(n[t*4],n[t*4+1]),[a,o]=q(n[t*4+2],n[t*4+3]);return[r,i,a,o]}function He(t){let n=t.analysis,r=t.detail,i=e.options,a=t.config.method===`dbscan`,o=i.view===`hotspots`&&a;if(n&&r&&o&&i.showEpsilonDiscs&&t.config.pipeline===`assembled`){let n=`${t.serial}|${i.epsilon}|${i.minimumPoints}`;t.discs?.key!==n&&(t.discs={key:n,samples:Ce({positions:t.positions,labels:r.u32(`labels`),coreFlags:r.u32(`coreFlags`),epsilon:i.epsilon,minimumPoints:i.minimumPoints,centre:d,windowMeters:Tt})}),e.setAnnotations(`epsilon-discs`,t.discs.samples.map(e=>({kind:`ring`,id:`disc-${e.role}`,coordinate:q(e.x,e.y),radiusMeters:i.epsilon,dashed:!0,text:`${e.neighbours} within ε: ${e.role}`})))}else e.setAnnotations(`epsilon-discs`,null);if(n&&o&&i.labelHotSpots?e.setAnnotations(`top-spots`,n.ranked.slice(0,3).map((e,t)=>{let r=q(n.groupCentres[e*2],n.groupCentres[e*2+1]);return{kind:`note`,id:`hot-spot-${t}`,coordinate:r,title:`${W(n.sizes[e])} records`,text:ce(le,r)??void 0,tone:`accent`}})):e.setAnnotations(`top-spots`,null),r&&i.showDistribution){let t=r.f32(`distributionMean`),n=r.f32(`distributionMedian`),i=q(t[0],t[1]),a=q(n[0],n[1]);e.setAnnotations(`distribution`,[{kind:`point`,id:`mean-centre`,coordinate:i,text:`Mean centre`,detail:ce(le,i)??void 0,marker:`none`,rank:`subject`,priority:3},{kind:`point`,id:`median-centre`,coordinate:a,text:`Median centre`,detail:ce(le,a)??void 0,marker:`none`,rank:`subject`,tone:`signal`,priority:3},{kind:`dimension`,id:`median-gap`,from:i,to:a,text:G(Math.hypot(n[0]-t[0],n[1]-t[1]))}])}else e.setAnnotations(`distribution`,null)}function Ue(t){let n=t.analysis,r=be(e.ground(),n?.noiseCount);if(n&&t.config.method===`dbscan`&&n.ranked.length>0){let i=n.ranked.slice(0,Dt),a={kind:`bars`,values:i.map(e=>n.sizes[e]),colors:i.map(e=>{let t=r.colors[We(n.sizes[e])];return[t[0],t[1],t[2],255]}),yScale:`log`,xLabel:`Hot spots, largest first`,yLabel:`Records`,description:`Sizes of the ${i.length} largest hot spots, largest first, on a log scale, coloured by size class.`,onBarClick:e=>Ge(t,i[e])};e.setChart(`sizeBars`,a)}else e.setChart(`sizeBars`,null);let i=O.result;i&&t.config.method===`dbscan`?e.setChart(`epsilonCurve`,{kind:`line`,series:[{label:`Hot spots at ${i.minimumPoints} points`,x:Ae,y:i.counts,points:!0,directLabel:!1}],xLabel:`Neighbour radius ε (m)`,yLabel:`Hot spots`,yDomain:[0,Math.max(...i.counts,1)*1.1],link:{option:`epsilon`,label:e=>`ε = ${e} m`},description:`Number of hot spots against the neighbour radius epsilon: it rises while small piles appear, then falls as hot spots merge.`,table:!0}):e.setChart(`epsilonCurve`,null)}function We(e){return e>=1e3?3:e>=200?2:e>=50?1:0}function Ge(t,n){let r=Ve(t,n),i=J(t,n);i&&e.setHighlight({kind:`polygon`,rings:[Ke(i)],pulse:!0}),r&&e.fitBounds(r,{transitionMs:1200,maxZoom:Ot})}function Ke(e){let t=[];for(let n=0;n<e.ring.length/2;n++)t.push(q(e.ring[n*2],e.ring[n*2+1]));return t.push(t[0]),t}function qe(t){let n=t.analysis;e.setLegendData(`ground`,e.ground()),n&&(e.setLegendData(`table`,be(e.ground(),n.noiseCount)),e.setLegendData(`classCounts`,n.classCounts))}function Je(t){let n=t.config.method===`dbscan`;e.setCost({records:t.pointCount,passes:n?5:k(t.config),note:n?O.result?`compiled once, ${Ae.length} runs`:`epsilon is a parameter write`:`compiled per iteration cap`})}function Xe(t){let n=e.options,r=t.config.method===`dbscan`,i=n.view===`hotspots`,a=n.category.toLowerCase(),o=i?r?`DBSCAN, ε ${n.epsilon} m, at least ${n.minimumPoints} records · ${a}`:`k-means, k ${t.config.k}, up to ${k(t.config)} iterations · ${a}`:void 0,s=i&&r?n.epsilon:null,c=`${o}|${s}|${t.pointCount}`;c!==E&&(E=c,e.setFurniture({title:{...o?{subtitle:o}:{},sample:`${W(t.pointCount)} iNaturalist ${a} records, Chicago, 2023`},scaleBar:{units:`metric`,...s?{ticks:[s]}:{}}}))}let Ze=e=>{e.destroyed=!0,e.reader.stop(),e.sweep?.reader.stop(),e.resources.destroy()},X=await ne(e.options);E=``;let Qe=e=>{setTimeout(async()=>{if(!(p||e.destroyed||e!==X))try{await Pe(e)}catch{}},150)};Qe(X),ve(e,i,a.project).then(e=>{C=e,p||Y(X)}).catch(()=>{});let $e=()=>{clearTimeout(g),g=setTimeout(()=>{if(p)return;let t=++_;e.setStatus(`Rebuilding the cluster graphs…`),setTimeout(async()=>{if(!(p||t!==_))try{let n=await ne(e.options);if(p||t!==_){Ze(n);return}let r=X;X=n,y=NaN,x=NaN,E=``,O.running=!1,O.awaiting=!1,O.result=null,r.destroyed=!0,r.reader.stop(),r.sweep?.reader.stop(),v.push(r),setTimeout(()=>{let e=v.indexOf(r);e>=0&&(v.splice(e,1),r.resources.destroy())},500),e.setStatus(``),Qe(n),e.requestLayers()}catch(t){e.setStatus(`Could not rebuild: ${String(t)}`)}},30)},Ct)};async function tt(){let n=X;if(h||p)return;if(n.config.method!==`dbscan`||n.config.pipeline===`recipe`){e.setReadout(`timing`,`select DBSCAN with the assembled pipeline`);return}h=!0,e.setReadout(`timing`,`measuring…`);let r=new m(t,`nature-timing-${n.serial}`);try{n.clusteringParameters.write(I({bounds:n.bounds,epsilon:e.options.epsilon,minimumPoints:e.options.minimumPoints}));let i=[];for(let o of[!1,!0]){let s=new u(t,{id:`nature-timing-${o}`}),c=r.createBuffer(`labels-${o}`,n.pointCount*4),l=r.createBuffer(`count-${o}`,4),d=r.createBuffer(`ids-${o}`,Z*4),f=r.createBuffer(`stored-${o}`,4),p=r.createBuffer(`overflow-${o}`,4);s.add(new b({spatialContext:a.spatialContext,id:`clusters`,positions:H(s,`positions`,n.positionsBuffer,`float32x2`,n.pointCount),parameters:n.clusteringParameters.importToGraph(s),gridSize:yt,denseBoxShortcut:o,sumOrder:n.config.sumOrder,output:{labels:H(s,`labels`,c,`uint32`,n.pointCount),clusterCount:H(s,`count`,l,`uint32`,1),clusters:{ids:H(s,`ids`,d,`uint32`,Z),status:{count:H(s,`stored`,f,`uint32`,1),overflow:H(s,`overflow`,p,`uint32`,1)}}}}));let m=await ie(t,r.track(await s.compileAsync()),{parameters:void 0,completionBuffer:l,signal:e.signal});i.push({milliseconds:m.milliseconds,label:ae(m).split(` · `)[0]})}if(p||n.destroyed)return;let[o,s]=i,c=o.milliseconds/Math.max(s.milliseconds,1e-6);e.setReadout(`timing`,`plain ${o.label}, dense-box ${s.label} (${c.toFixed(2)}x) at ${e.options.epsilon} m / ${e.options.minimumPoints}`)}catch{}finally{r.destroy(),h=!1}}async function nt(){let n=X;if(!(h||p)){h=!0,e.setReadout(`shapeTiming`,`measuring…`);try{let r=await ie(t,n.shapeGraph??n.clusterGraph,{parameters:void 0,completionBuffer:n.hullOverflow,signal:e.signal});!p&&!n.destroyed&&e.setReadout(`shapeTiming`,`${ae(r).split(` · `)[0]} (${n.shapeGraph?`geometry + hulls`:`whole recipe`}, prefilter ${n.config.prefilterLevels})`)}catch{}finally{h=!1}}}function it(t,n){let r=t.analysis,i=be(e.ground(),r.noiseCount),a=t.config.method===`dbscan`,o=q(r.groupCentres[n.group*2],r.groupCentres[n.group*2+1]),s=r.ranked.indexOf(n.group)+1,c=n.areaSquareMeters/1e6,l=a?i.colors[n.sizeClass]:Oe(e.ground())[t.partitionSlots[n.group]??0],u=[{label:`Records`,value:W(n.size),swatch:l,emphasis:!0},{label:`Hull area`,value:me(n.areaSquareMeters)},{label:`Density`,value:pe(n.size/Math.max(c,1e-9),`km²`)},{label:`Share of group`,value:fe(n.size/t.pointCount,1)}];return a&&u.splice(1,0,{label:`Size class`,value:De[n.sizeClass]}),{title:ue(le,o)?.name??`Hot spot`,subtitle:a?`Hot spot ${s} of ${r.clusterCount}, ${ce(le,o)??``}`.replace(/, $/,``):`Group ${s} of ${r.clusterCount}`,rows:u,note:`The convex hull encloses the records; it is not the footprint of the place.`,highlight:{kind:`polygon`,rings:[Ke(n)]}}}let at=()=>(D!==-1&&(D=-1),null);return{getCompiledGraphs:()=>[X.clusterGraph,X.shapeGraph,X.distributionGraph].filter(Boolean),setOption(t,n,r){if([`category`,`method`,`pipeline`,`denseBoxShortcut`,`sumOrder`,`k`,`initialization`,`seed`,`iterationStep`,`tolerance`,`prefilterLevels`].includes(t)){ee(r)!==X.key&&$e();return}t===`weight`?(X.weights.write(P(N(r.category))),X.dirty=!0):t===`standardDeviations`||t===`ellipseConvention`?(A(X),X.dirty=!0):t===`epsilon`||(t===`minimumPoints`?O.stale=!0:(t===`zoomToWinner`&&n&&(T=!0),ze(X),e.requestLayers()))},onAction(e){e===`measureShortcut`?tt():e===`measureShapes`&&nt()},onThemeChange(){Ue(X),qe(X),e.requestLayers()},onGroundChange(){Ue(X),qe(X),e.requestLayers()},onLegendFilter(t,n){t===`hot-spot-size`&&(S=n===null?null:[...n],e.requestLayers())},encode(t){let n=X,{epsilon:r,minimumPoints:i}=e.options;n.config.method===`dbscan`&&(r!==y||i!==x)&&(y=r,x=i,n.dirty=!0),n.dirty&&(n.dirty=!1,n.config.method===`dbscan`&&n.clusteringParameters.write(I({bounds:n.bounds,epsilon:r,minimumPoints:i})),n.clusterGraph.encode(t,{parameters:void 0}),n.shapeGraph?.encode(t,{parameters:void 0}),n.distributionGraph.encode(t,{parameters:void 0}),n.reader.markStale());let a=n.sweep;a&&O.serial===n.serial&&(O.stale&&!O.awaiting&&(O.stale=!1,O.running=!0,O.index=0,O.counts=[],O.minimumPoints=e.options.minimumPoints),O.running&&!O.awaiting&&!a.reader.isPending?(a.parameters.write(I({bounds:n.bounds,epsilon:Ae[O.index],minimumPoints:O.minimumPoints})),a.graph.encode(t,{parameters:void 0}),O.awaiting=!0,a.reader.request(t)):a.reader.flush(t)),n.reader.flush(t)},getTooltip(t){let n=X,r=n.analysis,i=e.options;if(!r||i.view!==`hotspots`||!t.coordinate)return at();let[o,s]=a.project(t.coordinate[0],t.coordinate[1]),c=i.showHulls?ke(r.hulls,o,s):null;if(c)return D=c.group,it(n,c);if(n.config.method===`dbscan`&&i.showNoise){let t=Math.max(6*e.getMetersPerPixel(),15),a=1/0;for(let e of r.noiseIds){let t=n.positions[e*2]-o,r=n.positions[e*2+1]-s;a=Math.min(a,t*t+r*r)}if(Math.sqrt(a)<=t)return{title:`Not in a hot spot`,subtitle:`Noise`,note:`Fewer than ${i.minimumPoints} neighbours within ${i.epsilon} m, and no core point among them.`}}return at()},getLayers(){let t=X,n=e.options,r=e.ground(),o=t.analysis,s=[i[0],i[1],0],c=t.config.method===`dbscan`,l=c&&t.config.pipeline===`recipe`,u=[],d=e=>`nature-clusters-${e}-${t.serial}`,f=V.map(([e,t])=>[e,t*.8]),p=de(be(r,o?.noiseCount)),m=ge(r,120);if(n.view===`records`||!o)u.push(new z({id:d(`records`),coordinateOrigin:s,analysisFrame:a.frame,positions:t.positionsBuffer,instanceCount:t.pointCount,colormap:`uniform`,color:_e(r,n.view===`records`?130:60),radiusPixels:V}));else if(c){n.showHulls&&t.drawCounts.hullVertices>0&&u.push(new re({id:d(`hull-fill`),coordinateOrigin:s,analysisFrame:a.frame,triangles:t.hullTriangles,features:t.hullTriangleFeatures,vertexCount:t.drawCounts.hullVertices,values:t.groupSizes,valueFormat:`float32`,colormap:`uniform`,...p,highlightClasses:S,dimOpacity:.15,opacity:r===`dark`?.18:.14})),n.showNoise&&t.drawCounts.noise>0&&u.push(new z({id:d(`noise`),coordinateOrigin:s,analysisFrame:a.frame,positions:t.positionsBuffer,ids:t.noiseIds,instanceCount:t.drawCounts.noise,colormap:`uniform`,color:ye(r,77),radiusPixels:f}));let e=n.showCore?V.map(([e,t])=>[e,t*1.35]):V,i={coordinateOrigin:s,analysisFrame:a.frame,positions:t.positionsBuffer,values:t.pointSizes,valueFormat:`float32`,colormap:`uniform`,...p,highlightClasses:S,dimOpacity:.15};n.showCore&&!l?(t.drawCounts.border>0&&u.push(new z({...i,id:d(`border-fill`),ids:t.borderIds,instanceCount:t.drawCounts.border,radiusPixels:e,opacity:.25}),new z({...i,id:d(`border-ring`),ids:t.borderIds,instanceCount:t.drawCounts.border,shape:`ring`,outlineWidthPixels:1,radiusPixels:e})),t.drawCounts.core>0&&u.push(new z({...i,id:d(`core`),ids:t.coreIds,instanceCount:t.drawCounts.core,radiusPixels:e,outlineColor:m,outlineWidthPixels:.5}))):t.drawCounts.members>0&&u.push(new z({...i,id:d(`members`),ids:t.memberIds,instanceCount:t.drawCounts.members,radiusPixels:e,outlineColor:m,outlineWidthPixels:.5})),n.showHulls&&t.drawCounts.hullSegments>0&&u.push(new B({id:d(`hull-outline`),coordinateOrigin:s,analysisFrame:a.frame,segments:t.hullOutline,instanceCount:t.drawCounts.hullSegments,valueIndices:t.hullOutlineFeatures,values:t.groupSizes,valueFormat:`float32`,colormap:`uniform`,...p,widthPixels:1.4,cap:`round`,highlightClasses:S,dimOpacity:.15,opacity:.9}))}else{let e=Oe(r);n.showHulls&&t.drawCounts.hullVertices>0&&u.push(new re({id:d(`hull-fill`),coordinateOrigin:s,analysisFrame:a.frame,triangles:t.hullTriangles,features:t.hullTriangleFeatures,vertexCount:t.drawCounts.hullVertices,values:t.groupSlots,valueFormat:`uint32`,colormap:`category`,palette:e,opacity:kt})),u.push(new z({id:d(`partition-points`),coordinateOrigin:s,analysisFrame:a.frame,positions:t.positionsBuffer,instanceCount:t.pointCount,values:t.pointSlots,valueFormat:`uint32`,colormap:`category`,palette:e,radiusPixels:V,outlineColor:m,outlineWidthPixels:.5})),n.showHulls&&t.drawCounts.hullSegments>0&&u.push(new B({id:d(`hull-outline`),coordinateOrigin:s,analysisFrame:a.frame,segments:t.hullOutline,instanceCount:t.drawCounts.hullSegments,valueIndices:t.hullOutlineFeatures,values:t.groupSlots,valueFormat:`uint32`,colormap:`category`,palette:e,widthPixels:1.2,cap:`round`,opacity:.9}))}if(n.view===`hotspots`&&o){let e=[we(r)],i=t.groupCount,o={coordinateOrigin:s,analysisFrame:a.frame,counts:t.geometryCounts,centers:t.meanCenters,medoidRows:t.medoids,positions:t.positionsBuffer,groupCount:i};n.showBounds&&u.push(new gt({id:d(`bounds`),coordinateOrigin:s,analysisFrame:a.frame,counts:t.geometryCounts,bounds:t.geometryBounds,groupCount:i,palette:e,widthPixels:1,opacity:.6})),n.showClusterEllipses&&!l&&u.push(new ht({id:d(`ellipses`),coordinateOrigin:s,analysisFrame:a.frame,counts:t.geometryCounts,centers:t.weightedCenters,ellipses:t.ellipses,groupCount:i,segments:wt,palette:e,widthPixels:1.4,opacity:.8})),n.showMeans&&u.push(new vt({...o,id:d(`mean-centers`),source:`centers`,shape:`ring`,sizePixels:7,palette:e})),n.showMedoids&&!l&&u.push(new vt({...o,id:d(`medoids`),source:`medoids`,shape:`disc`,sizePixels:3.5,palette:e})),n.showWeighted&&!l&&n.weight!==`none`&&u.push(new vt({...o,id:d(`weighted-centers`),centers:t.weightedCenters,source:`centers`,shape:`diamond`,sizePixels:6,palette:[je(r)]}))}if(n.showDistribution){let e=we(r);u.push(new B({id:d(`distribution-circle`),coordinateOrigin:s,analysisFrame:a.frame,segments:t.circleSegments,instanceCount:$,widthPixels:1,dashArray:[6,4],color:[e[0],e[1],e[2],200]}),new B({id:d(`distribution-ellipse`),coordinateOrigin:s,analysisFrame:a.frame,segments:t.ellipseSegments,instanceCount:$,widthPixels:2,color:e}),new vt({id:d(`distribution-mean`),coordinateOrigin:s,analysisFrame:a.frame,counts:t.distributionCounts,centers:t.distributionMean,medoidRows:t.distributionCounts,positions:t.positionsBuffer,groupCount:1,source:`centers`,shape:`ring`,sizePixels:10,palette:[e]}),new vt({id:d(`distribution-median`),coordinateOrigin:s,analysisFrame:a.frame,counts:t.distributionCounts,centers:t.distributionMedian,medoidRows:t.distributionCounts,positions:t.positionsBuffer,groupCount:1,source:`centers`,shape:`diamond`,sizePixels:8,palette:[je(r)]}))}return u},destroy(){p=!0,_++,clearTimeout(g),Ze(X);for(let e of v.splice(0))e.resources.destroy()}}}export{At as createNatureClusters};