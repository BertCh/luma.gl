import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{N as e,_ as t,d as n,h as r,t as i,w as a,y as o}from"./resources-CwvhEOBe.js";import{l as s,s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{r as l}from"./wgsl-common-BNRVzgd8.js";import{c as u}from"./spatial-weights-D4bd9AQ6.js";import{t as d}from"./gpu-spatial-weights-transform-B3cihvVH.js";import{n as f,t as p}from"./gpu-neighbor-search-Fct8W8hB.js";import{c as m,o as h}from"./index-Bjk18V2I.js";import{t as g}from"./graph-buffers-B4_80RlX.js";import{t as _}from"./graph-swap-41Ru7LVu.js";import{t as v}from"./summary-reader-CsgWxENf.js";import{t as y}from"./mode-kernels-Dxm2GWvI.js";import{t as b}from"./polygon-mesh-48KB9j_1.js";import{l as x}from"./class-table-BVEo3-Vs.js";import{a as ee,h as S,m as te}from"./hue-registry-4FjIJ9xz.js";import{r as ne}from"./b5-geometry-Dq2nRO88.js";import{a as C,c as re,i as ie,n as w}from"./live-text-83XBWyr1.js";import{r as ae,t as oe}from"./picking-DRmnefrA.js";import{t as se}from"./polygon-buffers-DTLnArS7.js";import{t as ce}from"./draw-order-jJzL6P4d.js";import{t as le}from"./b5-common-Dp9lAUMs.js";import{a as ue,c as de,d as fe,f as pe,i as T,l as me,n as he,o as E,p as ge,r as D,s as O,t as _e,u as ve}from"./segregation.style-zvwel2vg.js";function ye(e){return{entropy:0,multiGroupDissimilarity:1,diversity:2,dissimilarity:3,isolation:3+e,atkinson:3+2*e,interaction:3+3*e,stride:3+3*e+e*e}}var k=(e,t)=>`
const ENVIRONMENT_TOTALS: u32 = ${e*t}u;
const ENVIRONMENT_SCALE: u32 = ${e*t+e}u;
fn getEnvironmentTotal(unit: u32) -> f32 {
  return environment[environmentOffset + ENVIRONMENT_TOTALS + unit];
}
fn getScaleIndex() -> u32 {
  return u32(environment[environmentOffset + ENVIRONMENT_SCALE]);
}`,A=`GPUSegregation`,j=256,M=8,be=class{id;props;scaleCount;unitCount;constructor(e){let t=e.id??`segregation`;this.id=t,this.props=e;let{groupCount:i}=e;n(t,`groupCount`,i,2,16);let a=e.scales??[null];if(a.length<1)throw Error(`${t} scales must not be empty`);this.scaleCount=a.length,o(e.groupCounts,[`float32`],`${t} groupCounts`);let s=e.groupCounts.length/i;if(!Number.isInteger(s)||s<1)throw Error(`${t} groupCounts length must be a positive multiple of groupCount`);this.unitCount=s,a.forEach((e,n)=>{if(e&&u(t,e,`scales[${n}]`)!==s)throw Error(`${t} scales[${n}] must have one row per unit`)});let c=e.atkinsonInequalityAversion??.5;if(!(c>0&&c<1))throw Error(`${t} atkinsonInequalityAversion must be in (0, 1)`);let l=e.selfWeight??1;if(!Number.isFinite(l)||l<0)throw Error(`${t} selfWeight must be finite and non-negative`);let d=ye(i),{output:f}=e,p=[[`output.indices`,f.indices,this.scaleCount*d.stride],[`output.localEnvironment`,f.localEnvironment,this.scaleCount*s*i],[`output.localEntropy`,f.localEntropy,this.scaleCount*s],[`output.localDissimilarity`,f.localDissimilarity,this.scaleCount*s*i],[`output.localTheil`,f.localTheil,this.scaleCount*s]];for(let[e,n,r]of p)if(n&&(o(n,[`float32`],`${t} ${e}`),n.length!==r))throw Error(`${t} ${e} must contain ${r} rows`);r(t,this.getOutputViews(),this.getInputViews())}getInputViews(){let e=[this.props.groupCounts];for(let t of this.props.scales??[])t&&e.push(t.offsets,t.neighbors,t.weights);return e}getOutputViews(){let{output:e}=this.props;return[e.indices,e.localEnvironment,e.localEntropy,e.localDissimilarity,e.localTheil]}getCommandNodes(e){let{id:n,props:r,unitCount:i}=this;t(n,e,[...this.getInputViews(),...this.getOutputViews()]);let{groupCount:o,groupCounts:u,output:d}=r,{indices:f}=d,p={environment:d.localEnvironment,entropy:d.localEntropy,dissimilarity:d.localDissimilarity,theil:d.localTheil},m=r.scales??[null],h=r.selfWeight??1,g=r.atkinsonInequalityAversion??.5,_=ye(o),v=2+2*o+o*o,y=[],b=`${l}
const UNITS: u32 = ${i}u;
const K: u32 = ${o}u;
const TERM_COUNT: u32 = ${v}u;
const ATKINSON_B: f32 = ${s(g)};
const SELF_WEIGHT: f32 = ${s(h)};
fn countAt(unit: u32, group: u32) -> f32 {
  let value = groupCounts[groupCountsOffset + unit * K + group];
  // Negative, NaN and infinite counts are 0.
  let isFinite = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  return select(0.0, value, isFinite && value > 0.0);
}
fn unitTotal(unit: u32) -> f32 {
  var total = 0.0;
  for (var group = 0u; group < K; group++) {
    total += countAt(unit, group);
  }
  return total;
}
fn entropyTerm(share: f32) -> f32 {
  return select(0.0, -share * log(share), share > 0.0);
}`,x=Math.ceil(i/j),ee=(e,t)=>`${t}
const LANES: u32 = ${j}u;
const GROUPS: u32 = ${x}u;
const COLUMNS: u32 = ${e}u;
const COLUMN_BATCH: u32 = ${M}u;
var<private> columnValues: array<f32, ${e}>;
var<workgroup> batchScratch: array<f32, ${M*j}>;
fn reduceColumns(lane: u32, group: u32) {
  for (var first = 0u; first < COLUMNS; first += COLUMN_BATCH) {
    workgroupBarrier();
    for (var slot = 0u; slot < COLUMN_BATCH; slot++) {
      let column = first + slot;
      batchScratch[slot * LANES + lane] = select(0.0, columnValues[min(column, COLUMNS - 1u)], column < COLUMNS);
    }
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        for (var slot = 0u; slot < COLUMN_BATCH; slot++) {
          batchScratch[slot * LANES + lane] += batchScratch[slot * LANES + lane + stride];
        }
      }
      workgroupBarrier();
    }
    if (lane < COLUMN_BATCH && first + lane < COLUMNS) {
      partials[partialsOffset + (first + lane) * GROUPS + group] = batchScratch[lane * LANES];
    }
  }
}`,S=(t,r,a,s,l)=>c(e,{id:`${n}-${t}`,operation:A,variant:`column-totals`,bindings:[{name:`partials`,view:r,type:`f32`,access:`read`},{name:`output`,view:s,type:`f32`,access:`read_write`},...l?[{name:`collected`,view:l.view,type:`f32`,access:`read_write`},{name:`environment`,view:l.environment,type:`f32`,access:`read`}]:[]],invocationCount:a*j,workgroupSize:j,guardIndex:!1,declarations:`const LANES: u32 = ${j}u;
const GROUPS: u32 = ${x}u;
${l?`const COLLECTED_STRIDE: u32 = ${l.stride}u;
${k(i,o)}`:``}
var<workgroup> scratch: array<f32, ${j}>;`,body:`let column = index / LANES;
  let lane = localInvocationIndex;
  var sum = 0.0;
  for (var group = lane; group < GROUPS; group += LANES) {
    sum += partials[partialsOffset + column * GROUPS + group];
  }
  scratch[lane] = sum;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    output[outputOffset + column] = scratch[0];
    ${l?`collected[collectedOffset + getScaleIndex() * COLLECTED_STRIDE + column] = scratch[0];`:``}
  }`}),te=a(e,`${n}-all-totals`,`float32`,m.length*(o+1)),ne=r.spatialForm===`smoothed-population`,C=(t,r,i)=>{let s=a(e,`${n}-${t}-partials`,`float32`,(o+1)*x),l=a(e,`${n}-${t}`,`float32`,o+1);return y.push(c(e,{id:`${n}-${t}-partials`,operation:A,variant:`total-partials`,bindings:[{name:`groupCounts`,view:r,type:`f32`,access:`read`},{name:`partials`,view:s,type:`f32`,access:`read_write`}],invocationCount:x*j,workgroupSize:j,guardIndex:!1,declarations:ee(o+1,b),body:`let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  if (index < UNITS) {
    for (var column = 0u; column < K; column++) {
      columnValues[column] = countAt(index, column);
    }
    columnValues[K] = unitTotal(index);
  }
  reduceColumns(lane, group);`}),S(t,s,o+1,l,i?{view:te,stride:o+1,environment:i}:void 0)),l},re=(t,n,r)=>c(e,{id:`${t}-collect-totals`,operation:A,variant:`collect-totals`,bindings:[{name:`totals`,view:n,type:`f32`,access:`read`},{name:`allTotals`,view:te,type:`f32`,access:`read_write`},{name:`environment`,view:r,type:`f32`,access:`read`}],invocationCount:o+1,declarations:`const K: u32 = ${o}u;
${k(i,o)}`,body:`allTotals[allTotalsOffset + getScaleIndex() * (K + 1u) + index] = totals[totalsOffset + index];`}),ie=ne?void 0:C(`totals`,u),w=`${b}
${k(i,o)}
fn getShare(unit: u32, group: u32) -> f32 {
  let total = getEnvironmentTotal(unit);
  return select(0.0, environment[environmentOffset + unit * K + group] / total, total > 0.0);
}
fn getPopulationShare(group: u32) -> f32 {
  let total = totals[totalsOffset + K];
  return select(0.0, totals[totalsOffset + group] / total, total > 0.0);
}
fn getDiversity() -> f32 {
  var diversity = 0.0;
  for (var group = 0u; group < K; group++) {
    diversity += entropyTerm(getPopulationShare(group));
  }
  return diversity;
}`,ae=a(e,`${n}-term-sums`,`float32`,m.length*v),oe=a(e,`${n}-scale-cursor`,`uint32`,1);return y.push(c(e,{id:`${n}-scale-cursor-clear`,operation:A,variant:`scale-cursor-clear`,bindings:[{name:`scaleCursor`,view:oe,type:`u32`,access:`read_write`}],invocationCount:1,body:`scaleCursor[scaleCursorOffset] = 0xffffffffu;`})),m.forEach((t,r)=>{let s=`${n}-scale-${r}`,l=a(e,`${s}-environment`,`float32`,i*o+i+1),d=a(e,`${s}-sums`,`float32`,v),f=a(e,`${s}-term-partials`,`float32`,v*x),m=ne?l:u;y.push(c(e,{id:`${s}-environment`,operation:A,variant:t?`environment`:`environment-aspatial`,bindings:[{name:`groupCounts`,view:u,type:`f32`,access:`read`},...t?[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t.weights,type:`f32`,access:`read`}]:[],{name:`environment`,view:l,type:`f32`,access:`read_write`},{name:`scaleCursor`,view:oe,type:`u32`,access:`read_write`}],invocationCount:i,declarations:b,body:`if (index == 0u) {
    let scale = scaleCursor[scaleCursorOffset] + 1u;
    scaleCursor[scaleCursorOffset] = scale;
    environment[environmentOffset + UNITS * K + UNITS] = f32(scale);
  }
  ${t?`var total = 0.0;
  for (var group = 0u; group < K; group++) {
    var sum = SELF_WEIGHT * countAt(index, group);
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < UNITS) {
        sum += weights[weightsOffset + slot] * countAt(neighbor, group);
      }
    }
    environment[environmentOffset + index * K + group] = sum;
    total += sum;
  }
  environment[environmentOffset + UNITS * K + index] = total;`:`for (var group = 0u; group < K; group++) {
    environment[environmentOffset + index * K + group] = countAt(index, group);
  }
  environment[environmentOffset + UNITS * K + index] = unitTotal(index);`}`}));let h=ie??C(`scale-${r}-totals`,m,l);if(y.push(...ie?[re(s,h,l)]:[],c(e,{id:`${s}-terms`,operation:A,variant:`terms`,bindings:[{name:`groupCounts`,view:m,type:`f32`,access:`read`},{name:`environment`,view:l,type:`f32`,access:`read`},{name:`totals`,view:h,type:`f32`,access:`read`},{name:`partials`,view:f,type:`f32`,access:`read_write`}],invocationCount:x*j,workgroupSize:j,guardIndex:!1,declarations:ee(v,w),body:`let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  // A unit whose environment is empty has no composition: it contributes nothing to any sum.
  if (index < UNITS && getEnvironmentTotal(index) > 0.0) {
    let unitPopulation = unitTotal(index);
    var entropy = 0.0;
    var absolute = 0.0;
    for (var member = 0u; member < K; member++) {
      let share = getShare(index, member);
      let deviation = abs(share - getPopulationShare(member));
      entropy += entropyTerm(share);
      absolute += deviation;
      columnValues[2u + member] = unitPopulation * deviation;
      let base = max(1.0 - share, 0.0);
      let atkinson = select(0.0, pow(base, 1.0 - ATKINSON_B) * pow(share, ATKINSON_B), base > 0.0 && share > 0.0);
      columnValues[2u + K + member] = unitPopulation * atkinson;
      for (var other = 0u; other < K; other++) {
        columnValues[2u + 2u * K + member * K + other] = countAt(index, member) * getShare(index, other);
      }
    }
    columnValues[0] = unitPopulation * entropy;
    columnValues[1] = unitPopulation * absolute;
  }
  reduceColumns(lane, group);`}),S(`${s}-sums`,f,v,d,{view:ae,stride:v,environment:l})),p.environment||p.entropy||p.dissimilarity||p.theil){let t=[[`localEnvironment`,p.environment],[`localEntropy`,p.entropy],[`localDissimilarity`,p.dissimilarity],[`localTheil`,p.theil]];y.push(c(e,{id:`${s}-local`,operation:A,record:p.theil?{formula:`gpu-spatial-analysis.segregation`}:void 0,variant:`local-${t.map(([,e])=>e?1:0).join(``)}`,bindings:[{name:`groupCounts`,view:m,type:`f32`,access:`read`},{name:`environment`,view:l,type:`f32`,access:`read`},{name:`totals`,view:h,type:`f32`,access:`read`},...t.flatMap(([e,t])=>t?[{name:e,view:t,type:`f32`,access:`read_write`}]:[])],invocationCount:i,declarations:w,body:`let unitPopulation = unitTotal(index);
  let scaleIndex = getScaleIndex();
  // An empty environment has no composition, so every local value of the unit is NaN.
  let hasEnvironment = getEnvironmentTotal(index) > 0.0;
  let populationTotal = totals[totalsOffset + K];
  let diversity = getDiversity();
  var entropy = 0.0;
  for (var group = 0u; group < K; group++) {
    let share = getShare(index, group);
    entropy += entropyTerm(share);
    ${p.environment?`localEnvironment[localEnvironmentOffset + (scaleIndex * UNITS + index) * K + group] =
      select(getNaN(), share, hasEnvironment);`:``}
    ${p.dissimilarity?`let populationShare = getPopulationShare(group);
    let scale = 2.0 * populationTotal * populationShare * (1.0 - populationShare);
    localDissimilarity[localDissimilarityOffset + (scaleIndex * UNITS + index) * K + group] =
      select(getNaN(), unitPopulation * abs(share - populationShare) / scale, scale > 0.0 && hasEnvironment);`:``}
  }
  ${p.entropy?`localEntropy[localEntropyOffset + scaleIndex * UNITS + index] =
    select(getNaN(), entropy, hasEnvironment);`:``}
  ${p.theil?`localTheil[localTheilOffset + scaleIndex * UNITS + index] =
    select(getNaN(), unitPopulation * (diversity - entropy) / (diversity * populationTotal), diversity > 0.0 && populationTotal > 0.0 && hasEnvironment);`:``}`}))}}),y.push(c(e,{id:`${n}-finalize`,operation:A,variant:`finalize`,bindings:[{name:`allTotals`,view:te,type:`f32`,access:`read`},{name:`allSums`,view:ae,type:`f32`,access:`read`},{name:`indices`,view:f,type:`f32`,access:`read_write`}],invocationCount:m.length,declarations:`${l}
const K: u32 = ${o}u;
const TERM_COUNT: u32 = ${v}u;
const STRIDE: u32 = ${_.stride}u;
const ATKINSON_B: f32 = ${s(g)};
fn entropyTerm(share: f32) -> f32 {
  return select(0.0, -share * log(share), share > 0.0);
}`,body:`let populationTotal = allTotals[allTotalsOffset + index * (K + 1u) + K];
  let sums = index * TERM_COUNT;
  let base = index * STRIDE;
  var diversity = 0.0;
  var variance = 0.0;
  for (var group = 0u; group < K; group++) {
    let share = select(0.0, allTotals[allTotalsOffset + index * (K + 1u) + group] / populationTotal, populationTotal > 0.0);
    diversity += entropyTerm(share);
    variance += share * (1.0 - share);
  }
  let hasPopulation = populationTotal > 0.0;
  indices[indicesOffset + base + ${_.entropy}u] = select(
    getNaN(), 1.0 - allSums[allSumsOffset + sums] / (diversity * populationTotal), hasPopulation && diversity > 0.0);
  indices[indicesOffset + base + ${_.multiGroupDissimilarity}u] = select(
    getNaN(), allSums[allSumsOffset + sums + 1u] / (2.0 * populationTotal * variance), hasPopulation && variance > 0.0);
  indices[indicesOffset + base + ${_.diversity}u] = select(getNaN(), diversity, hasPopulation);
  for (var group = 0u; group < K; group++) {
    let groupTotal = allTotals[allTotalsOffset + index * (K + 1u) + group];
    let share = select(0.0, groupTotal / populationTotal, hasPopulation);
    let isMixed = hasPopulation && share > 0.0 && share < 1.0;
    let spread = 2.0 * populationTotal * share * (1.0 - share);
    indices[indicesOffset + base + ${_.dissimilarity}u + group] = select(
      getNaN(), allSums[allSumsOffset + sums + 2u + group] / spread, isMixed);
    let ratio = abs(allSums[allSumsOffset + sums + 2u + K + group] / max(share * populationTotal, 1e-30));
    let atkinson = 1.0 - share / (1.0 - share) * pow(ratio, 1.0 / (1.0 - ATKINSON_B));
    indices[indicesOffset + base + ${_.atkinson}u + group] = select(getNaN(), atkinson, isMixed);
    for (var other = 0u; other < K; other++) {
      let exposure = allSums[allSumsOffset + sums + 2u + 2u * K + group * K + other];
      let value = select(getNaN(), exposure / groupTotal, groupTotal > 0.0);
      indices[indicesOffset + base + ${_.interaction}u + group * K + other] = value;
      if (other == group) {
        indices[indicesOffset + base + ${_.isolation}u + group] = value;
      }
    }
  }`})),y}};function xe(e,t){let n=new Float64Array(D),r=0;for(let i=0;i<t;i++)for(let t=0;t<D;t++){let a=e[i*D+t];n[t]+=a,r+=a}let i=n.map(e=>r>0?e/r:0),a=i.reduce((e,t)=>e+t*(1-t),0),o=new Float64Array(D),s=0;for(let n=0;n<t;n++){let t=0;for(let r=0;r<D;r++)t+=e[n*D+r];if(!(t<=0))for(let c=0;c<D;c++){let l=Math.abs(e[n*D+c]/t-i[c]);s+=t*l/(2*r*a);let u=2*r*i[c]*(1-i[c]);u>0&&(o[c]+=t*l/u)}}return{multigroup:a>0?s:0,perGroup:o,shares:i,total:r}}function Se(e,t,n,r){let i=new Float64Array(r*D);for(let a=0;a<t;a++){let t=n[a];if(!(t<0||t>=r))for(let n=0;n<D;n++)i[t*D+n]+=e[a*D+n]}return i}function Ce(e,t){let n=0,r=0;for(let i=0;i<D;i++){let a=e[t*D+i];n+=a,a>e[t*D+r]&&(r=i)}return n>0?{group:r,share:e[t*D+r]/n,total:n}:null}function we(e,t){let n=new Uint32Array(t);for(let r=0;r<t;r++){let t=Ce(e,r);n[r]=t?E[t.group].registryIndex*3+fe(t.share):T}return n}function Te(e){let t=Array(D*3).fill(0),n=0;for(let r=0;r<e.length;r++)e[r]===4294967295?n++:t[e[r]]++;return{cells:t,empty:n}}function Ee(e,t,n){let r=[];for(let n=0;n<t;n++){let t=0;for(let r=0;r<D;r++)t+=e[n*D+r];t>0&&r.push(n)}let i=ce(r.length,n),a=Float32Array.from(e);return r.forEach((t,n)=>{let o=r[i[n]];for(let n=0;n<D;n++)a[t*D+n]=e[o*D+n]}),a}function De(e,t,n,r){let i=[],a=e[n*2],o=e[n*2+1],s=r*r;for(let r=0;r<t;r++){if(r===n)continue;let t=e[r*2]-a,c=e[r*2+1]-o;t*t+c*c<=s&&i.push(r)}return i}function Oe(e,t,n){let r=0,i=0;for(let a of n)!Number.isFinite(e[a])||!(t[a]>0)||(r+=e[a]*t[a],i+=t[a]);return i>0?r/i:NaN}function ke(e,t){let n=e.length;if(n===0||!Number.isFinite(t))return NaN;let r=0,i=n;for(;r<i;){let n=r+i>>1;e[n]<=t?r=n+1:i=n}return r/n}function Ae(e,t){let n=Array(t.length+1).fill(0);for(let r=0;r<e.length;r++){let i=e[r];if(!Number.isFinite(i))continue;let a=0;for(;a<t.length&&i>=t[a];)a++;n[a]++}return n}function je(e,t){if(!Number.isFinite(e))return-1;let n=0;for(;n<t.length&&e>=t[n];)n++;return n}var N=[1,2,4,8,16],P=N.length+1,Me=Math.log(D),Ne={relative:0,diversity:1,dominant:0},Pe=1e3,Fe=[`Rogers Park`,`Uptown`,`Albany Park`],Ie={light:.88,dark:.9};function Le(e){return new Uint32Array(Float32Array.of(e).buffer)[0]}async function Re(t){let n=t.datasets.get(`chicago-tracts`),r=t.datasets.get(`chicago-community-areas`),a=n.geojson,o=r.geojson;if(!a||!o)throw Error(`segregation needs the tract and community-area polygons`);let{device:s}=t,c=new i(s,`segregation`),l=ne(n),u=l.featureCount,ce=n.defaultOrigin,T=n.getProjection(ce),k=(e,t)=>T.project(e,t),A=[ce[0],ce[1],0],j=b(a,k),M=se(c,j,`tracts`),Re=b(o,k),ze=se(c,Re,`areas`),Be=oe(a),Ve=oe(o),He=a.features,Ue=o.features,F=Ue.length,I=new Float32Array(u*2),We=[];{let{vertices:e,ringOffsets:t,featureRingOffsets:n}=l;for(let r=0;r<u;r++){let i=0,a=0,o=0,s=t[n[r]],c=t[n[r+1]];for(let t=s;t<c;t++){let[n,r]=k(e[t*2],e[t*2+1]);i+=n,a+=r,o++}I[r*2]=i/Math.max(o,1),I[r*2+1]=a/Math.max(o,1),We.push(T.unproject(I[r*2],I[r*2+1]))}}let Ge=1/0,Ke=1/0,qe=-1/0,Je=-1/0;for(let e=0;e<u;e++)Ge=Math.min(Ge,I[e*2]),qe=Math.max(qe,I[e*2]),Ke=Math.min(Ke,I[e*2+1]),Je=Math.max(Je,I[e*2+1]);let Ye=2e3,Xe=[Ge-Ye,Ke-Ye,qe+Ye,Je+Ye],L=new Float32Array(u*D),Ze=E.map(e=>n.column(e.id)),R=new Float64Array(u);for(let e=0;e<u;e++)for(let t=0;t<D;t++){let n=Ze[t][e],r=Number.isFinite(n)&&n>0?n:0;L[e*D+t]=r,R[e]+=r}let Qe=n.column(`communityArea`),$e=Int32Array.from(Qe,e=>e-1),et=e=>String(Ue[e]?.properties?.name??`Area ${e+1}`),tt=e=>et($e[e]),nt=xe(L,u),rt=Se(L,u,$e,F),it=xe(rt,F),at=new Float64Array(F);for(let e=0;e<F;e++)for(let t=0;t<D;t++)at[e]+=rt[e*D+t];let ot=nt.shares.reduce((e,t)=>e+(t>0?t*Math.log(1/t):0),0)/Me,st=we(L,u),ct=Te(st),lt=[2,5,8,11,14].reduce((e,t)=>e+ct.cells[t],0),ut=[0,3,6,9,12].reduce((e,t)=>e+ct.cells[t],0),z=Ee(L,u,t.options.shuffleSeed),dt=we(z,u),ft=e=>{let t=Ue.findIndex(t=>t.properties?.name===e),n=-1;for(let e=0;e<u;e++)$e[e]===t&&(n<0||R[e]>R[n])&&(n=e);return n},B=new Uint32Array(u+1);for(let e of j.outlineFeatures)B[e+1]++;for(let e=0;e<u;e++)B[e+1]+=B[e];let pt=e=>{let t=[],n=0;for(let r of e){let e=j.outlineSegments.subarray(B[r]*4,B[r+1]*4);t.push(e),n+=e.length}let r=new Float32Array(n),i=0;for(let e of t)r.set(e,i),i+=e.length;return r},mt=e=>ae(e?.geometry).flatMap(e=>e.map(e=>e.map(e=>[e[0],e[1]]))),V=ye(D),ht=c.createBuffer(`positions`,I),gt=c.createBuffer(`group-counts`,L),_t=c.createBuffer(`group-counts-shuffled`,z),vt=c.createBuffer(`indices`,P*V.stride*4),yt=c.createBuffer(`indices-shuffled`,P*V.stride*4),bt=c.createBuffer(`local-environment`,P*u*D*4),xt=c.createBuffer(`local-entropy`,P*u*4),St=c.createBuffer(`local-dissimilarity`,P*u*D*4),Ct=c.createBuffer(`local-theil`,P*u*4),wt=c.createBuffer(`display`,u*4),Tt=c.createParameterBuffer(`display-parameter`,`uint32`,6),Et=c.createBuffer(`class-real`,st),Dt=c.createBuffer(`class-shuffled`,dt),Ot=c.createBuffer(`area-values`,F*4),kt=c.createBuffer(`highlight-channel`,new Float32Array(u)),At=c.createBuffer(`selection`,j.outlineSegments.byteLength),jt=c.createBuffer(`members`,j.outlineSegments.byteLength),Mt=0,Nt=0,H=u*u,Pt=N.map((e,t)=>({capacity:H,searchParameter:c.createParameterBuffer(`search-parameter-${t}`,`float32`,12),offsets:c.createBuffer(`offsets-${t}`,(u+1)*4),neighbors:c.createBuffer(`neighbors-${t}`,H*4),weights:c.createBuffer(`weights-${t}`,H*4),distances:c.createBuffer(`distances-${t}`,H*4),overflow:c.createBuffer(`overflow-${t}`,4)})),Ft=(e,t)=>{let n=Pt[t];return{spatialContext:T.spatialContext,offsets:g(e,`offsets-${t}`,n.offsets,`uint32`,u+1),neighbors:g(e,`neighbors-${t}`,n.neighbors,`uint32`,n.capacity),weights:g(e,`weights-${t}`,n.weights,`float32`,n.capacity),distances:g(e,`distances-${t}`,n.distances,`float32`,n.capacity)}},U=new e(s,{id:`segregation-search`}),It=g(U,`positions`,ht,`float32x2`,u);N.forEach((e,t)=>{let n=Pt[t];U.add(new p({spatialContext:T.spatialContext,id:`search-${t}`,mode:`radius`,gridSize:[32,32],positions:It,parameters:n.searchParameter.importToGraph(U),output:{...Ft(U,t),status:{overflow:g(U,`overflow-${t}`,n.overflow,`uint32`,1)}}}))});let Lt=c.track(await U.compileAsync()),Rt=e=>[e.weights,e.rowStandardize?`row`:`-`,e.spatialForm,e.selfWeight,e.atkinsonB].join(`|`),zt=async t=>{let n=Rt(t),r=new e(s,{id:`segregation-${n}`}),i=[null];N.forEach((e,n)=>{let a=Ft(r,n);t.weights!==`band`&&r.add(new d({spatialContext:T.spatialContext,id:`kernel-${n}`,operation:`kernel`,weights:a,kernel:t.weights,bandwidth:`adaptive`})),t.rowStandardize&&r.add(new d({spatialContext:T.spatialContext,id:`row-${n}`,operation:`row`,weights:a})),i.push(a)});let a=(e,t,n)=>g(r,e,t,`float32`,n),o={groupCount:D,scales:i,selfWeight:Number(t.selfWeight),spatialForm:t.spatialForm,atkinsonInequalityAversion:Number(t.atkinsonB)};return r.add(new be({id:`segregation`,...o,groupCounts:a(`group-counts`,gt,u*D),output:{indices:a(`indices`,vt,P*V.stride),localEnvironment:a(`local-environment`,bt,P*u*D),localEntropy:a(`local-entropy`,xt,P*u),localDissimilarity:a(`local-dissimilarity`,St,P*u*D),localTheil:a(`local-theil`,Ct,P*u)}})),r.add(new be({id:`segregation-shuffled`,...o,groupCounts:a(`group-counts-shuffled`,_t,u*D),output:{indices:a(`indices-shuffled`,yt,P*V.stride)}})),{key:n,compiled:c.track(await r.compileAsync())}},Bt=new Map,Vt=async e=>{let t=Rt(e),n=Bt.get(t);return n||(n=await zt(e),Bt.set(t,n)),n},Ht=new e(s,{id:`segregation-display`}),Ut=(e,t,n)=>g(Ht,e,t,`float32`,n);y(Ht,{id:`segregation-display`,invocationCount:u,bindings:[{name:`parameters`,view:Tt.importToGraph(Ht),type:`u32`,access:`read`},{name:`counts`,view:Ut(`group-counts`,gt,u*D),type:`f32`,access:`read`},{name:`environment`,view:Ut(`local-environment`,bt,P*u*D),type:`f32`,access:`read`},{name:`entropy`,view:Ut(`local-entropy`,xt,P*u),type:`f32`,access:`read`},{name:`display`,view:Ut(`display`,wt,u),type:`f32`,access:`read_write`}],declarations:`const UNITS: u32 = ${u}u;
const GROUPS: u32 = ${D}u;
const LOG_GROUPS: f32 = ${Me.toFixed(8)};
const CLIP: f32 = ${2 .toFixed(1)};`,body:`
  let mode = parameters[parametersOffset];
  let scale = parameters[parametersOffset + 1u];
  let group = parameters[parametersOffset + 2u];
  let cityShare = bitcast<f32>(parameters[parametersOffset + 3u]);
  var total = 0.0;
  for (var member = 0u; member < GROUPS; member++) {
    total += counts[countsOffset + index * GROUPS + member];
  }
  let row = scale * UNITS + index;
  var environmentTotal = 0.0;
  for (var member = 0u; member < GROUPS; member++) {
    environmentTotal += environment[environmentOffset + row * GROUPS + member];
  }
  // Keeps the NaN pattern a runtime value because WGSL rejects a constant NaN.
  var value = bitcast<f32>(0x7fc00000u | (parameters[parametersOffset + 4u] >> 31u));
  if (total > 0.0 && environmentTotal > 0.0) {
    if (mode == 0u) {
      let share = environment[environmentOffset + row * GROUPS + group];
      value = clamp(log2(max(share, 0.000001) / max(cityShare, 0.000001)), -CLIP, CLIP);
    } else {
      value = entropy[entropyOffset + row] / LOG_GROUPS;
    }
  }
  display[displayOffset + index] = value;`});let Wt=c.track(await Ht.compileAsync()),Gt=await Vt(t.options),Kt=c.track(new _),W=!0,qt=!0,G=null,Jt=!0,Yt=!1,Xt=null,K=ft(t.options.focusArea),Zt=!1,Qt=[],q=()=>Math.max(0,E.findIndex(e=>e.id===t.options.focalGroup)),J=()=>E[q()].label,$t=e=>e<=0?0:N[e-1]*t.options.bandwidth*1e3,Y=e=>e<=0?`aspatial`:`${Number(($t(e)/1e3).toFixed(2))} km`,en=()=>nt.shares[q()],tn=()=>t.options.units===`areas`&&t.options.localIndex===`relative`,nn=()=>t.options.showRing&&t.options.scale>0,rn=()=>{Pt.forEach((e,n)=>{e.searchParameter.write(f({bounds:Xe,radius:N[n]*t.options.bandwidth*1e3,weightKind:`binary`}))}),W=!0,$.markStale()},an=()=>{let{localIndex:e,scale:n}=t.options;Tt.write(Uint32Array.of(Ne[e],n,q(),Le(en()),0,0)),qt=!0,$.markStale()},on=()=>{let e=q(),t=en(),n=new Float32Array(F);for(let r=0;r<F;r++){let i=at[r]>0?rt[r*D+e]/at[r]:NaN;n[r]=Number.isFinite(i)?Math.max(-2,Math.min(2,Math.log2(Math.max(i,1e-6)/Math.max(t,1e-6)))):NaN}return Ot.write(n),n},sn=on(),X=()=>{if(K<0)Mt=0,Nt=0,Qt=[],t.setAnnotations(`segregation-ring`,null);else{let e=pt([K]);At.write(e),Mt=e.length/4,Qt=nn()?De(I,u,K,$t(t.options.scale)):[];let n=pt(Qt);jt.write(n),Nt=n.length/4;let r=new Float32Array(u);r[K]=1;for(let e of Qt)r[e]=1;kt.write(r),t.setAnnotations(`segregation-ring`,nn()?[{kind:`ring`,id:`segregation-ring`,coordinate:We[K],radiusMeters:$t(t.options.scale),text:`${Y(t.options.scale)} environment`,dashed:!0,tone:`signal`}]:null)}hn(),Yt&&t.requestLayers()},cn=()=>{let{scale:e}=t.options;t.setFurniture({title:{sample:`${w(u)} census tracts, ACS 2018-2022 via CDC SVI 2022`},scaleBar:e>0?{units:`metric`,ticks:[$t(e)]}:!1})},ln=()=>{let{localIndex:e,scale:n,showRing:r}=t.options,i=[];if(G&&e===`relative`&&!r&&!tn()){let e=-1;for(let t=0;t<u;t++)R[t]<Pe||!Number.isFinite(G.display[t])||(e<0||G.display[t]>G.display[e])&&(e=t);if(e>=0){let t=q();i.push({kind:`note`,id:`segregation-note-over`,coordinate:We[e],priority:5,title:re(`{ratio:fixed:1} times the city share`,{ratio:2**G.display[e]}),text:`${tt(e)}: ${C(L[e*D+t]/R[e])} ${J()}${n>0?` within ${Y(n)}`:``}`})}}if(G&&e===`diversity`&&n>0){let e=e=>Array.from({length:u},(e,t)=>t).filter(t=>$e[t]===e);for(let t of Fe){let r=Ue.findIndex(e=>e.properties?.name===t),a=Re.labelPoints[r];if(r<0||!a||!Number.isFinite(a[0]))continue;let o=Oe(G.display,R,e(r));i.push({kind:`note`,id:`segregation-note-${r}`,coordinate:[a[0],a[1]],priority:4,title:re(`{value:fixed:2} of the maximum`,{value:o}),text:`${t}, ${Y(n)} environments`})}}t.setAnnotations(`segregation-notes`,i.length?i:null)},Z=(e,t,n)=>e[t*V.stride+n],un=()=>{if(!G)return null;let e=Array.from({length:P},(e,t)=>Y(t)),n=Array.from({length:P},(e,t)=>t),r=[{label:`Real tracts`,x:n,y:n.map(e=>Z(G.indices,e,V.multiGroupDissimilarity)),color:5,directLabel:!0,points:!0}];return t.options.shuffle&&r.push({label:`Tracts shuffled`,x:n,y:n.map(e=>Z(G.shuffledIndices,e,V.multiGroupDissimilarity)),dashed:!0,ghost:!0,directLabel:!0}),{kind:`line`,title:`Multigroup dissimilarity D by environment radius`,series:r,xDomain:[0,P-1],yDomain:[0,.9],xLabel:`Environment radius (each step doubles)`,yLabel:`D`,formatX:t=>Number.isInteger(t)?e[t]??``:``,formatY:e=>e.toFixed(1),link:{option:`scale`,label:t=>e[Math.round(t)]??``},description:`Multigroup dissimilarity D falls as the environment widens; the same tracts shuffled across the city start at the same D and fall much faster.`}},dn=()=>{if(!G)return null;let e=Array.from({length:P},(e,t)=>Y(t)),t=Array.from({length:P},(e,t)=>t);return{kind:`line`,title:`Dissimilarity D of each group by environment radius`,series:[1,0,2,3,4].map((e,n)=>({label:E[e].label.split(` `)[0],x:t,y:t.map(t=>Z(G.indices,t,V.dissimilarity+e)),color:n,ghost:n===4,directLabel:!0})),xDomain:[0,P-1],yDomain:[0,.9],xLabel:`Environment radius (each step doubles)`,yLabel:`D`,formatX:t=>Number.isInteger(t)?e[t]??``:``,formatY:e=>e.toFixed(1),link:{option:`scale`,label:t=>e[Math.round(t)]??``},description:`Dissimilarity of each group by radius: the Black curve stays high at wide radii, the Hispanic curve falls to near zero.`}},fn=()=>{if(!G||K<0)return null;let e=t.ground(),{scale:n}=t.options,r=t=>E.map((n,r)=>({label:n.label.split(` `)[0],value:t(r),color:pe(r,e)})),i=r(e=>L[K*D+e]),a=r(e=>G.environment[(n*u+K)*D+e]),o=r(e=>nt.shares[e]);return{kind:`multiples`,columns:1,shareDomains:!1,titles:[`This tract (${tt(K)})`,n>0?`Within ${Y(n)}`:`Its environment (aspatial: itself)`,`Chicago`],charts:[{kind:`stacked`,segments:i},{kind:`stacked`,segments:a},{kind:`stacked`,segments:o}],description:`Three stacked share bars: the composition of the selected tract, of everyone within the radius, and of the whole city.`}},Q=()=>{let{scale:e}=t.options,n=tn(),r=n?sn:G?.display;t.setLegendData(`segregation`,{groundIsDark:t.ground()===`dark`,dominanceCells:ct.cells,emptyTracts:ct.empty,relativeCounts:r&&(n||Jt)&&t.options.localIndex===`relative`?Ae(r,ue):null,diversityCounts:G&&Jt&&t.options.localIndex===`diversity`?Ae(G.display,_e):null,focalCityShare:en(),cityDiversity:ot,environmentLabel:n?`the community area`:e===0?`the tract`:`everyone within ${Y(e)}`,focalLabel:J()})},pn=()=>{t.setReadout(`residents`,nt.total),t.setReadout(`supermajority`,lt),t.setReadout(`noMajority`,ut),t.setReadout(`capacity`,`${w(H)} slots per radius, 4 arrays, ${N.length} radii (capacity = tracts squared; a count pass would size it)`)},mn=()=>{let e=q();t.setReadout(`focalLabel`,J()),t.setReadout(`cityShare`,C(en())),t.setReadout(`dAreas`,O(it.multigroup,3)),t.setReadout(`dFocalAreas`,O(it.perGroup[e],3))},hn=()=>{if(K<0){t.setReadout(`selected`,null),t.setReadout(`members`,null),t.setReadout(`ownShare`,null),t.setReadout(`envShare`,null);return}let e=q(),{scale:n}=t.options,r=He[K]?.properties?.GEOID??K;t.setReadout(`selected`,`${tt(K)}, tract ${r}: ${w(R[K])} residents`),t.setReadout(`members`,nn()?Qt.length:null),t.setReadout(`ownShare`,C(L[K*D+e]/R[K])),t.setReadout(`envShare`,G?C(G.environment[(n*u+K)*D+e]):null),t.setChart(`composition`,fn())},gn=()=>{if(!G)return;let{scale:e}=t.options,n=q();t.setReadout(`radius`,Y(e)),t.setReadout(`radiusWide`,Y(4)),t.setReadout(`radiusMid`,Y(3)),t.setReadout(`dAspatial`,O(Z(G.indices,0,V.multiGroupDissimilarity),3)),t.setReadout(`dAspatialShuffled`,O(Z(G.shuffledIndices,0,V.multiGroupDissimilarity),3)),t.setReadout(`dFocal`,O(Z(G.indices,0,V.dissimilarity+n),3)),t.setReadout(`dMidReal`,O(Z(G.indices,3,V.multiGroupDissimilarity),3)),t.setReadout(`dMidShuffled`,O(Z(G.shuffledIndices,3,V.multiGroupDissimilarity),3)),t.setReadout(`dScale`,O(Z(G.indices,e,V.multiGroupDissimilarity),3)),t.setReadout(`entropyIndex`,O(Z(G.indices,e,V.entropy),3)),t.setReadout(`isolation`,O(Z(G.indices,e,V.isolation+n),3)),t.setReadout(`atkinson`,O(Z(G.indices,e,V.atkinson+n),3));let r=E.findIndex(e=>e.id===`nhBlack`),i=E.findIndex(e=>e.id===`hispanic`);t.setReadout(`blackWide`,O(Z(G.indices,4,V.dissimilarity+r),3)),t.setReadout(`hispanicWide`,O(Z(G.indices,4,V.dissimilarity+i),3)),t.setReadout(`cityDiversity`,O(ot,2));let a=0,o=0;for(let t=0;t<u;t++){let r=G.theil[e*u+t],i=G.dissimilarity[(e*u+t)*D+n];Number.isFinite(r)&&(o+=r),Number.isFinite(i)&&(a+=i)}let s=Z(G.indices,e,V.entropy),c=Z(G.indices,e,V.dissimilarity+n),l=Number.isFinite(s)&&Number.isFinite(c)&&Math.abs(o-s)<5e-4&&Math.abs(a-c)<5e-4;t.setReadout(`localSum`,!Number.isFinite(s)||!Number.isFinite(c)?`Local-sum check unavailable: the global index is undefined.`:`${l?`Verified`:`Mismatch`}: local H sums to ${o.toFixed(3)} (global ${O(Z(G.indices,e,V.entropy),3)}); local D of ${J()} sums to ${a.toFixed(3)} (global ${O(Z(G.indices,e,V.dissimilarity+n),3)})`)},_n=()=>{t.setChart(`profile`,un()),t.setChart(`profileAll`,dn()),t.setChart(`composition`,fn())},$=new v(c,`segregation`,[{buffer:vt,size:P*V.stride*4},{buffer:yt,size:P*V.stride*4},...Pt.map(e=>({buffer:e.overflow,size:4})),{buffer:wt,size:u*4},{buffer:bt,size:P*u*D*4},{buffer:Ct,size:P*u*4},{buffer:St,size:P*u*D*4}],e=>{let n=le(e),r=n.floats(P*V.stride),i=n.floats(P*V.stride),a=n.words(N.length),o=n.floats(u),s=n.floats(P*u*D),c=n.floats(P*u),l=n.floats(P*u*D),d=[];for(let e of o)Number.isFinite(e)&&d.push(e);G={indices:r,shuffledIndices:i,display:o,environment:s,theil:c,dissimilarity:l,sortedDisplay:Float64Array.from(d).sort()},t.setReadout(`overflow`,a.some(e=>e)?`YES in radii ${Array.from(a).join(`, `)}`:`no`),Jt=!0,gn(),_n(),Q(),hn(),ln(),t.requestLayers()});t.setCost({records:u,note:`${N.length} radii, real and shuffled counts`}),pn(),mn(),Q(),rn(),an(),cn(),X(),Yt=!0;let vn=()=>t.options.localIndex===`diversity`?me(t.ground()):ge(t.ground()),yn=()=>{let e=t.options,n=t.ground(),r=[],i=ee[n],a=tn(),o={coordinateOrigin:A,opacity:Ie[n],noDataColor:i,hatchNoData:!0,hatchColor:S(n,90),highlightClasses:Xt},s={triangles:M.triangles,features:M.triangleFeatures,vertexCount:M.vertexCount},c=nn(),l={instanceChannels:kt,channelStride:1,channels:{highlight:0},highlightActive:c,dimOpacity:.28};if(a)r.push(new h({id:`segregation-area-fill`,...o,triangles:ze.triangles,features:ze.triangleFeatures,vertexCount:ze.vertexCount,values:Ot,valueFormat:`float32`,colormap:`puor`,...x(vn())}));else if(e.localIndex===`dominant`){let t=ve(n),i=e.shuffle?[{id:`segregation-class-real`,values:Et,side:`a`},{id:`segregation-class-shuffled`,values:Dt,side:`b`}]:[{id:`segregation-class-real`,values:Et}];for(let{id:e,values:n,side:a}of i)r.push(new h({id:e,...o,...s,...l,values:n,valueFormat:`uint32`,colormap:`category`,palette:t,highlightClasses:null,compareSide:a}))}else r.push(new h({id:`segregation-tract-fill`,...o,...s,...l,values:wt,valueFormat:`float32`,colormap:e.localIndex===`relative`?`puor`:`greys`,...x(vn())}));a||r.push(new m({id:`segregation-hairlines`,coordinateOrigin:A,segments:M.outline,instanceCount:M.outlineCount,widthPixels:.5,color:n===`dark`?[14,17,22,128]:[255,255,255,161]}));let u=de(n,a);return r.push(new m({id:`segregation-area-lines`,coordinateOrigin:A,segments:ze.outline,instanceCount:ze.outlineCount,widthPixels:u.widthPixels,color:u.color,...a?{outlineColor:u.casing,outlineWidthPixels:(u.casingPixels-u.widthPixels)/2}:{}})),!a&&K>=0&&(c||Zt)&&(c&&Nt&&r.push(new m({id:`segregation-members`,coordinateOrigin:A,segments:jt,instanceCount:Nt,widthPixels:1.2,color:S(n)})),r.push(new m({id:`segregation-selection`,coordinateOrigin:A,segments:At,instanceCount:Mt,widthPixels:2.5,color:S(n),outlineColor:te(n),outlineWidthPixels:1}))),r},bn=(e,t)=>{let n=e.colors[Math.max(0,je(t,e.breaks))];return[n[0],n[1],n[2],n[3]??255]},xn=(e,n,r)=>E.map((i,a)=>({label:i.label.split(` `)[0],value:C(e[n*D+a]/r),swatch:pe(a,t.ground())})),Sn=e=>{let n=t.options,r=t.ground(),i=R[e],a=He[e]?.properties?.GEOID??e,o=j.labelPoints[e],s=[];if(i<=0)s.push({label:`Residents`,value:0});else if(n.localIndex===`dominant`){let t=Ce(L,e);if(t){let e=E[t.group],n=fe(t.share);s.push({label:`Largest group`,value:e.label.split(` `)[0],unit:`${C(t.share)} of residents`,swatch:ve(r)[e.registryIndex*3+n],emphasis:!0},{label:`Share of the largest group`,value:he[n]})}}else if(G){let t=G.display[e],r=vn(),i=ke(G.sortedDisplay,t);n.localIndex===`relative`?s.push({label:`${J().split(` `)[0]} share of ${n.scale===0?`the tract`:`everyone within ${Y(n.scale)}`}`,value:Number.isFinite(t)?`${(2**t).toFixed(1)}x`:`n/a`,unit:`the city share`,swatch:bn(r,t),emphasis:!0}):s.push({label:`Local diversity`,value:Number.isFinite(t)?t.toFixed(2):`n/a`,unit:`of the maximum`,swatch:bn(r,t),emphasis:!0}),Number.isFinite(i)&&s.push({label:`Rank`,value:`${ie(Math.round(i*100))} percentile`,unit:`of ${w(G.sortedDisplay.length)} tracts`})}if(i>0&&(s.push({label:`Residents`,value:w(i)}),s.push(...xn(L,e,i)),G&&n.scale>0&&n.localIndex!==`dominant`)){let t=q();s.push({label:`${J().split(` `)[0]} within ${Y(n.scale)}`,value:C(G.environment[(n.scale*u+e)*D+t])})}return{title:tt(e),subtitle:`Census tract ${a}`,rows:s,anchor:o&&Number.isFinite(o[0])?[o[0],o[1]]:void 0,highlight:{kind:`polygon`,rings:mt(He[e])}}},Cn=e=>{let n=at[e],r=sn[e],i=ge(t.ground()),a=[{label:`${J().split(` `)[0]} share, relative to the city`,value:Number.isFinite(r)?`${(2**r).toFixed(1)}x`:`n/a`,unit:`the city share`,swatch:bn(i,r),emphasis:!0},{label:`Residents`,value:w(n)},...xn(rt,e,Math.max(n,1))],o=Re.labelPoints[e];return{title:et(e),subtitle:`Community area`,rows:a,anchor:o&&Number.isFinite(o[0])?[o[0],o[1]]:void 0,highlight:{kind:`polygon`,rings:mt(Ue[e])}}};return{getCompiledGraphs:()=>[Lt,...[...Bt.values()].map(e=>e.compiled),Wt],setOption(e){if([`weights`,`rowStandardize`,`spatialForm`,`selfWeight`,`atkinsonB`].includes(e))Kt.request({build:()=>Vt(t.options),commit:e=>{Gt=e,W=!0,$.markStale(),t.requestLayers()}});else if(e===`bandwidth`)rn(),cn(),X();else if(e===`shuffleSeed`)z=Ee(L,u,t.options.shuffleSeed),dt=we(z,u),_t.write(z),Dt.write(dt),W=!0,$.markStale();else if(e===`shuffle`)_n();else if(e===`focalGroup`)Jt=!1,sn=on(),mn(),an(),hn(),Q(),t.setAnnotations(`segregation-notes`,null);else if(e===`localIndex`||e===`scale`)Jt=!1,Xt=null,an(),cn(),e===`scale`&&X(),Q(),t.setAnnotations(`segregation-notes`,null);else if(e===`units`)Xt=null,Q(),ln();else if(e===`focusArea`){let e=ft(t.options.focusArea);e>=0&&(K=e),X()}else e===`showRing`&&(t.options.showRing&&K<0&&(K=ft(t.options.focusArea)),X(),ln());t.requestLayers()},encode(e,t){(W||t.frameIndex<2)&&(Lt.encode(e,{parameters:void 0}),Gt.compiled.encode(e,{parameters:void 0}),W=!1,qt=!0),(qt||t.frameIndex<2)&&(Wt.encode(e,{parameters:void 0}),qt=!1,$.markStale()),t.frameIndex>=2&&$.flush(e)},getLayers:yn,onThemeChange(){t.requestLayers()},onGroundChange(){Q(),hn(),_n(),t.requestLayers()},onLegendFilter(e,n){Xt=n===null?null:[...n],t.requestLayers()},getTooltip(e){if(!e.coordinate)return null;if(tn()){let t=Ve.find(e.coordinate);return t?Cn(t.index):null}let t=Be.find(e.coordinate);return t?Sn(t.index):null},onClick(e){if(!e.coordinate||tn())return!1;let t=Be.find(e.coordinate),n=t?t.index:-1;return K=n===K&&Zt?-1:n,Zt=K>=0,X(),!0},destroy(){$.stop(),c.destroy()}}}export{Re as createSegregation};