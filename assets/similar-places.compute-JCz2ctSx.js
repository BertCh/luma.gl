import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{N as e,_ as t,d as n,h as r,l as i,t as a,v as o,w as s,y as c}from"./resources-CwvhEOBe.js";import{t as l}from"./gpu-sort-Bl3Dztdb.js";import{s as u}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as d,r as f}from"./wgsl-common-BNRVzgd8.js";import{N as p,P as m,c as h,o as g}from"./index-Bjk18V2I.js";import{t as _}from"./graph-buffers-B4_80RlX.js";import{t as v}from"./graph-swap-41Ru7LVu.js";import{t as y}from"./summary-reader-CsgWxENf.js";import{t as b}from"./mode-kernels-Dxm2GWvI.js";import{f as x}from"./class-table-BVEo3-Vs.js";import{a as S,i as C}from"./hue-registry-4FjIJ9xz.js";import"./b5-geometry-Dq2nRO88.js";import{c as w,i as T,n as E,r as D}from"./live-text-83XBWyr1.js";import"./picking-DRmnefrA.js";import{n as O,r as ee,t as k}from"./segments-BtINkq-i.js";import{f as te,l as ne}from"./hot-spots.style-DcEoMUmX.js";import{n as re,r as ie,t as ae}from"./b5-common-Dp9lAUMs.js";import{n as oe,r as A}from"./b5-variables-E0Z2Tw89.js";import{t as se}from"./choropleth-classes.geometry-C8xz_Rhm.js";import{a as ce,i as le,o as ue,r as j,s as de,t as fe}from"./similar-places.style-DT1ohzqX.js";var pe=4294967295;function M(e){return 4+e}function me(e,t,n=new Float32Array(M(t))){let r=M(t);if(n.length<r)throw Error(`Similar locations parameter target must hold ${r} elements`);if(!Number.isSafeInteger(e.resultCount)||e.resultCount<0)throw Error(`Similar locations resultCount must be a non-negative integer`);if(e.weights&&e.weights.length!==t)throw Error(`Similar locations weights must have one entry per attribute`);n[0]=e.resultCount,n[1]=e.direction===`least`?1:0,n[2]=e.excludeReference===!1?0:1,n[3]=0;for(let r=0;r<t;r++){let t=e.weights?e.weights[r]:1;if(!Number.isFinite(t)||t<0)throw Error(`Similar locations weights must be finite and non-negative`);n[4+r]=t}return n}var N=`GPUSimilarLocations`,P=256,F=Math.floor(4294967295/P);function I(e){return i(e,4)}var L=4096,he=`
const LANES: u32 = ${P}u;
var<workgroup> floatScratch: array<f32, ${P}>;
var<workgroup> countScratch: array<u32, ${P}>;
fn reduceSum(value: f32, lane: u32) -> f32 {
  workgroupBarrier();
  floatScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      floatScratch[lane] += floatScratch[lane + stride];
    }
    workgroupBarrier();
  }
  return floatScratch[0];
}
fn reduceCount(value: u32, lane: u32) -> u32 {
  workgroupBarrier();
  countScratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      countScratch[lane] += countScratch[lane + stride];
    }
    workgroupBarrier();
  }
  return countScratch[0];
}`,R=`
const INVALID_KEY: u32 = 0xffffffffu;
fn getOrderedKey(value: f32) -> u32 {
  // Integer test: a float compare of -0 against 0 can be folded away by the shader compiler.
  let raw = bitcast<u32>(value);
  let bits = select(raw, 0u, (raw & 0x7fffffffu) == 0u);
  return select(bits | 0x80000000u, ~bits, (bits >> 31u) == 1u);
}`,ge=class{id;props;rowCount;resultCapacity;constructor(e){this.id=e.id??`similar-locations`,this.props=e;let{id:t}=this;if(n(t,`columnCount`,e.columnCount,1,F),c(e.values,[`float32`],`${t} values`),e.values.length%e.columnCount!==0)throw Error(`${t} values length must be a multiple of columnCount`);if(this.rowCount=e.values.length/e.columnCount,this.rowCount<1)throw Error(`${t} needs at least one row`);if(e.standardization&&![`zscore`,`rank`].includes(e.standardization))throw Error(`${t} standardization must be 'zscore' or 'rank'`);if(this.resultCapacity=e.resultCapacity??Math.min(this.rowCount,32),n(t,`resultCapacity`,this.resultCapacity,1,this.rowCount),e.mask&&(c(e.mask,[`uint32`],`${t} mask`),e.mask.length!==this.rowCount))throw Error(`${t} mask length must equal the row count`);if(c(e.referenceMask,[`uint32`],`${t} referenceMask`),e.referenceMask.length!==this.rowCount)throw Error(`${t} referenceMask length must equal the row count`);c(e.parameters,[`float32`],`${t} parameters`);let i=M(e.columnCount);if(e.parameters.length<i)throw Error(`${t} parameters must hold ${i} float32 values`);if(c(e.output.ranks,[`uint32`],`${t} output.ranks`),c(e.output.distances,[`float32`],`${t} output.distances`),c(e.output.topIds,[`uint32`],`${t} output.topIds`),c(e.output.status.count,[`uint32`],`${t} output.status.count`),e.output.ranks.length<this.rowCount||e.output.distances.length<this.rowCount)throw Error(`${t} output.ranks and output.distances must hold one row per input row`);if(e.output.topIds.length<this.resultCapacity)throw Error(`${t} output.topIds must hold resultCapacity rows`);if(e.output.status.count.length<1)throw Error(`${t} output.status.count must hold one row`);r(t,[e.output.ranks,e.output.distances,e.output.topIds,e.output.status.count],[e.values,e.mask,e.referenceMask,e.parameters])}getCommandNodes(e){let{id:n,props:r,rowCount:i,resultCapacity:a}=this,{output:c,columnCount:p}=r;t(n,e,[r.values,r.mask,r.referenceMask,r.parameters,c.ranks,c.distances,c.topIds,c.status.count]),o(n,`values rows`,r.values.length,I(e.device));let m=r.standardization===`rank`,h=i*p,g=s(e,`${n}-validity`,`uint32`,i),_=s(e,`${n}-standardized`,`float32`,h),v=s(e,`${n}-reference`,`float32`,p),y=s(e,`${n}-keys`,`uint32`,i),b=s(e,`${n}-row-ids`,`uint32`,i),x=s(e,`${n}-sorted-keys`,`uint32`,i),S=s(e,`${n}-sorted-ids`,`uint32`,i),C=`const ROW_COUNT: u32 = ${i}u;
const ATTRIBUTE_COUNT: u32 = ${p}u;
const MAX_RESULTS: u32 = ${a}u;
const NO_RANK: u32 = 0xffffffffu;
${d}
${f}`,w=[],T=(t,r,i,a,o={})=>w.push(u(e,{id:`${n}-${t}`,operation:N,variant:t,bindings:r,invocationCount:o.reduction?i*P:i,workgroupSize:o.reduction?P:64,guardIndex:o.reduction?!1:void 0,declarations:`${C}\n${o.reduction?he:``}\n${o.declarations??``}`,body:a})),E=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),D=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`});if(T(`validate`,[E(`attributes`,r.values,`f32`),...r.mask?[E(`rowMask`,r.mask,`u32`)]:[],D(`validity`,g,`u32`)],i,`var isValid = true;
  ${r.mask?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  for (var channel = 0u; channel < ATTRIBUTE_COUNT; channel++) {
    isValid = isValid && isFiniteValue(attributes[attributesOffset + index * ATTRIBUTE_COUNT + channel]);
  }
  validity[validityOffset + index] = select(0u, 1u, isValid);`),m&&i<L)T(`standardize-rank`,[E(`attributes`,r.values,`f32`),E(`validity`,g,`u32`),D(`standardized`,_,`f32`)],h,`let row = index / ATTRIBUTE_COUNT;
  let channel = index % ATTRIBUTE_COUNT;
  if (validity[validityOffset + row] == 0u) {
    standardized[standardizedOffset + index] = getNaN();
    return;
  }
  let value = attributes[attributesOffset + index];
  var less = 0u;
  var equal = 0u;
  var valid = 0u;
  for (var other = 0u; other < ROW_COUNT; other++) {
    if (validity[validityOffset + other] == 0u) {
      continue;
    }
    valid += 1u;
    let otherValue = attributes[attributesOffset + other * ATTRIBUTE_COUNT + channel];
    less += select(0u, 1u, otherValue < value);
    equal += select(0u, 1u, otherValue == value);
  }
  let denominator = max(f32(valid) - 1.0, 1.0);
  standardized[standardizedOffset + index] = (f32(less) + 0.5 * (f32(equal) - 1.0)) / denominator;`);else if(m){let t=s(e,`${n}-rank-keys`,`uint32`,i),a=s(e,`${n}-rank-row-ids`,`uint32`,i),o=s(e,`${n}-rank-sorted-keys`,`uint32`,i),c=s(e,`${n}-rank-sorted-ids`,`uint32`,i),d=s(e,`${n}-rank-column`,`uint32`,1);T(`rank-column-clear`,[D(`rankColumn`,d,`u32`)],1,`rankColumn[rankColumnOffset] = 0xffffffffu;`);for(let s=0;s<p;s++)w.push(u(e,{id:`${n}-rank-column-${s}`,operation:N,variant:`rank-column`,bindings:[D(`rankColumn`,d,`u32`)],invocationCount:1,body:`rankColumn[rankColumnOffset] = rankColumn[rankColumnOffset] + 1u;`})),T(`rank-keys-${s}`,[E(`attributes`,r.values,`f32`),E(`validity`,g,`u32`),D(`rankKeys`,t,`u32`),D(`rankRowIds`,a,`u32`),E(`rankColumn`,d,`u32`)],i,`let value = attributes[attributesOffset + index * ATTRIBUTE_COUNT + rankColumn[rankColumnOffset]];
  rankKeys[rankKeysOffset + index] =
    select(INVALID_KEY, getOrderedKey(value), validity[validityOffset + index] != 0u);
  rankRowIds[rankRowIdsOffset + index] = index;`,{declarations:R}),w.push(...new l({id:`${n}-rank-sort-${s}`,keys:t,values:a,outputKeys:o,outputValues:c,keyBits:32}).getCommandNodes(e)),T(`rank-assign-${s}`,[E(`sortedKeys`,o,`u32`),E(`sortedIds`,c,`u32`),D(`standardized`,_,`f32`),E(`rankColumn`,d,`u32`)],i,`let key = sortedKeys[sortedKeysOffset + index];
  let row = sortedIds[sortedIdsOffset + index];
  let column = rankColumn[rankColumnOffset];
  if (key == INVALID_KEY) {
    standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column] = getNaN();
    return;
  }
  let below = lowerBound(key);
  let notAbove = lowerBound(key + 1u);
  let valid = lowerBound(INVALID_KEY);
  let denominator = max(f32(valid) - 1.0, 1.0);
  standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column] =
    (f32(below) + 0.5 * (f32(notAbove - below) - 1.0)) / denominator;`,{declarations:`${R}
fn lowerBound(bound: u32) -> u32 {
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < bound) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`})}else{let t=s(e,`${n}-stats`,`float32`,2*p);T(`column-statistics`,[E(`attributes`,r.values,`f32`),E(`validity`,g,`u32`),D(`stats`,t,`f32`)],p,`let column = index / LANES;
  let lane = localInvocationIndex;
  var validPartial = 0u;
  var sumPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      validPartial += 1u;
      sumPartial += attributes[attributesOffset + row * ATTRIBUTE_COUNT + column];
    }
  }
  let valid = reduceCount(validPartial, lane);
  let mean = select(0.0, reduceSum(sumPartial, lane) / f32(valid), valid > 0u);
  var squaresPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u) {
      let delta = attributes[attributesOffset + row * ATTRIBUTE_COUNT + column] - mean;
      squaresPartial += delta * delta;
    }
  }
  let squares = reduceSum(squaresPartial, lane);
  if (lane == 0u) {
    let deviation = select(0.0, sqrt(squares / f32(valid)), valid > 0u);
    stats[statsOffset + 2u * column] = mean;
    stats[statsOffset + 2u * column + 1u] = select(0.0, 1.0 / deviation, deviation > 0.0);
  }`,{reduction:!0}),T(`standardize-zscore`,[E(`attributes`,r.values,`f32`),E(`validity`,g,`u32`),E(`stats`,t,`f32`),D(`standardized`,_,`f32`)],h,`let row = index / ATTRIBUTE_COUNT;
  let channel = index % ATTRIBUTE_COUNT;
  if (validity[validityOffset + row] == 0u) {
    standardized[standardizedOffset + index] = getNaN();
    return;
  }
  standardized[standardizedOffset + index] =
    (attributes[attributesOffset + index] - stats[statsOffset + 2u * channel]) *
    stats[statsOffset + 2u * channel + 1u];`)}return T(`reference`,[E(`standardized`,_,`f32`),E(`validity`,g,`u32`),E(`selection`,r.referenceMask,`u32`),D(`reference`,v,`f32`)],p,`let column = index / LANES;
  let lane = localInvocationIndex;
  var countPartial = 0u;
  var sumPartial = 0.0;
  for (var row = lane; row < ROW_COUNT; row += LANES) {
    if (validity[validityOffset + row] != 0u && selection[selectionOffset + row] != 0u) {
      countPartial += 1u;
      sumPartial += standardized[standardizedOffset + row * ATTRIBUTE_COUNT + column];
    }
  }
  let count = reduceCount(countPartial, lane);
  let sum = reduceSum(sumPartial, lane);
  if (lane == 0u) {
    reference[referenceOffset + column] = select(getNaN(), sum / f32(count), count > 0u);
  }`,{reduction:!0}),T(`distance`,[E(`standardized`,_,`f32`),E(`validity`,g,`u32`),E(`selection`,r.referenceMask,`u32`),E(`reference`,v,`f32`),E(`params`,r.parameters,`f32`),D(`keys`,y,`u32`),D(`rowIds`,b,`u32`),D(`distances`,c.distances,`f32`)],i,`rowIds[rowIdsOffset + index] = index;
  let isLeast = params[paramsOffset + 1u] > 0.5;
  let excludeReference = params[paramsOffset + 2u] > 0.5;
  var eligible = validity[validityOffset + index] != 0u &&
    !(excludeReference && selection[selectionOffset + index] != 0u);
  var distanceSquared = 0.0;
  for (var channel = 0u; channel < ATTRIBUTE_COUNT; channel++) {
    let weight = params[paramsOffset + 4u + channel];
    let delta = standardized[standardizedOffset + index * ATTRIBUTE_COUNT + channel] -
      reference[referenceOffset + channel];
    distanceSquared += weight * delta * delta;
  }
  eligible = eligible && isFiniteValue(distanceSquared);
  let distance = sqrt(distanceSquared);
  // Non-negative f32 bit patterns order like the values; the sentinel sorts after every key.
  let bits = bitcast<u32>(distance);
  keys[keysOffset + index] = select(NO_RANK, select(bits, 0xfffffffeu - bits, isLeast), eligible);
  distances[distancesOffset + index] = select(getNaN(), distance, eligible);`),w.push(...new l({id:`${n}-sort`,keys:y,values:b,outputKeys:x,outputValues:S,keyBits:32}).getCommandNodes(e)),T(`rank`,[E(`sortedKeys`,x,`u32`),E(`sortedIds`,S,`u32`),E(`params`,r.parameters,`f32`),D(`ranks`,c.ranks,`u32`),D(`topIds`,c.topIds,`u32`),D(`resultCount`,c.status.count,`u32`)],i,`let key = sortedKeys[sortedKeysOffset + index];
  let isEligible = key != NO_RANK;
  let row = sortedIds[sortedIdsOffset + index];
  ranks[ranksOffset + row] = select(NO_RANK, index, isEligible);
  let requested = params[paramsOffset];
  let limit = min(u32(clamp(select(0.0, requested, isFiniteValue(requested)), 0.0, f32(MAX_RESULTS))), MAX_RESULTS);
  // Eligible rows sort first, so the boundary row owns the eligible count.
  let isLast = index + 1u == ROW_COUNT || sortedKeys[sortedKeysOffset + index + 1u] == NO_RANK;
  if (isEligible && isLast) {
    resultCount[resultCountOffset] = min(index + 1u, limit);
  }
  if (index == 0u && !isEligible) {
    resultCount[resultCountOffset] = 0u;
  }
  if (index < MAX_RESULTS) {
    topIds[topIdsOffset + index] = select(NO_RANK, row, isEligible && index < limit);
  }`),w}};function _e(e,t){for(let n of e)if(!Number.isFinite(n[t]))return!1;return!0}function ve(e,t,n){let r=[];for(let t=0;t<n;t++)_e(e,t)&&r.push(t);return e.map(e=>{let i=new Float64Array(n).fill(NaN),a=r.length;if(t===`zscore`){let t=0;for(let n of r)t+=e[n];let n=t/a,o=0;for(let t of r)o+=(e[t]-n)**2;let s=Math.sqrt(o/a);for(let t of r)i[t]=s>0?(e[t]-n)/s:0;return i}let o=Float64Array.from(r,t=>e[t]).sort();for(let t of r){let n=e[t],r=0,s=a;for(;r<s;){let e=r+s>>1;o[e]<n?r=e+1:s=e}let c=r;for(;c<a&&o[c]===n;)c++;i[t]=a>1?(r+(c-r-1)/2)/(a-1):0}return i})}function z(e,t){return Float64Array.from(e,e=>{let n=0;for(let r of t)n+=e[r];return t.length?n/t.length:NaN})}function ye(e,t,n,r){let i=e[0]?.length??0,a=new Float64Array(i).fill(NaN),o=[];for(let s=0;s<i;s++){if(r.excluded.has(s)||Number.isNaN(e[0][s]))continue;let i=0;for(let r=0;r<e.length;r++)i+=n[r]*(e[r][s]-t[r])**2;Number.isFinite(i)&&(a[s]=Math.sqrt(i),o.push(s))}let s=r.direction===`most`?1:-1;return o.sort((e,t)=>s*(a[e]-a[t])||e-t),{distances:a,order:Int32Array.from(o)}}function be(e,t,n,r){let i=[];for(let a=0;a<e.length;a++){if(!(n[a]>0))continue;let o=e[a][r]-t[a];i.push({attribute:a,delta:o,contribution:n[a]*o*o})}return i.sort((e,t)=>t.contribution-e.contribution)}function B(e,t){return m(e,t)/1e3}function xe(e){if(!e.length)return NaN;let t=[...e].sort((e,t)=>e-t),n=t.length>>1;return t.length%2==1?t[n]:(t[n-1]+t[n])/2}function Se(e,t,n){let r=new Set(t),i=[];for(let n=0;n<e.length;n++){if(r.has(n)||!Number.isFinite(e[n][0]))continue;let a=1/0;for(let r of t)a=Math.min(a,B(e[r],e[n]));i.push({row:n,kilometers:a})}return i.sort((e,t)=>e.kilometers-t.kilometers||e.row-t.row),i.slice(0,n)}function Ce(e,t){let n=new Set(t),r=0;for(let t of e)n.has(t)&&r++;return r}function we(e,t){let n=e.map((e,t)=>t).filter(e=>t[e]>0),r=null,i=e[0]?.length??0;for(let t=0;t<n.length;t++)for(let a=t+1;a<n.length;a++){let o=0,s=0;for(let r=0;r<i;r++){let i=e[n[t]][r]*e[n[a]][r];Number.isFinite(i)&&(o+=i,s++)}let c=s?o/s:0;(!r||Math.abs(c)>Math.abs(r.r))&&(r={a:n[t],b:n[a],r:c})}return r}function Te(e,t,n){let r=e.createBuffer(t,Math.max(1,n)*16);return{buffer:r,write(e){let t=Math.min(e.length/4,n);return t>0&&r.write(e.subarray(0,t*4)),t}}}function Ee(e,t){let n=new Uint32Array(e.mesh.featureCount);for(let t of e.mesh.outlineFeatures)n[t]++;return[...n].sort((e,t)=>t-e).slice(0,t).reduce((e,t)=>e+t+8,0)}function De(e,t){let n=k(e.origin),r=t.map(t=>ee(e.getRings(t),n)),i=new Float32Array(r.reduce((e,t)=>e+t.length,0)),a=0;for(let e of r)i.set(e,a),a+=e.length;return i}var Oe=400,ke=12;function Ae(e,t,n){let r=k(e.origin),i=[];for(let e of n){let n=Math.min(ke,Math.max(1,Math.ceil(B(t,e)/Oe)));i.push(...p(t,e,n))}return O(i,r)}var V=A.length,H=32,U=20,je=6,Me=6,W=pe;async function Ne(t){let n=t.datasets.get(`us-counties`),{device:r}=t,i=new a(r,`similar-places`),o=se(i,n,t.datasets.get(`us-states`)),s=o.features.length,c=o.mesh.labelPoints,l=Array.from({length:s},(e,t)=>le(o.features[t]?.properties??null)),u=e=>l[e]??`County ${e}`,d=e=>String(o.features[e]?.properties?.state),f=(e,t)=>o.features.findIndex(n=>{let r=n.properties;return r?.name===e&&r?.state===t&&Number(r?.fips)%1e3<500}),p=e=>c[e],m=A.map(e=>n.column(e.id)),O=A.map((e,t)=>e.transform===`log10`?Float32Array.from(m[t],e=>Math.log10(Math.max(e,.001))):m[t]),ee=new Float32Array(s*V);for(let e=0;e<s;e++)for(let t=0;t<V;t++)ee[e*V+t]=O[t][e];let k={zscore:ve(O,`zscore`,s),rank:ve(O,`rank`,s)},pe=i.createBuffer(`attributes`,ee),N=i.createBuffer(`selection`,s*4),P=i.createBuffer(`ranks`,s*4),F=i.createBuffer(`distances`,s*4),I=i.createBuffer(`top-ids`,H*4),L=i.createBuffer(`count`,4),he=i.createBuffer(`rank-classes`,s*4),R=i.createParameterBuffer(`similarity-parameters`,`float32`,M(V)),_e=i.createParameterBuffer(`display-parameters`,`float32`,4),Oe=Te(i,`match-outline`,Ee(o,U)),ke=Te(i,`reference-outline`,Ee(o,je)),Ne=Te(i,`arcs`,U*12),Pe=0,Fe=0,Ie=0,Le=0,Re=async t=>{let n=new e(r,{id:`similar-${t}`}),a=_(n,`ranks`,P,`uint32`,s),o=_(n,`selection`,N,`uint32`,s);return n.add(new ge({id:`similar-locations`,values:_(n,`attributes`,pe,`float32`,s*V),columnCount:V,referenceMask:o,parameters:R.importToGraph(n),standardization:t,resultCapacity:H,output:{ranks:a,distances:_(n,`distances`,F,`float32`,s),topIds:_(n,`top-ids`,I,`uint32`,H),status:{count:_(n,`count`,L,`uint32`,1)}}})),b(n,{id:`rank-classes`,invocationCount:s,bindings:[{name:`parameters`,view:_e.importToGraph(n),type:`f32`,access:`read`},{name:`ranks`,view:a,type:`u32`,access:`read`},{name:`selection`,view:o,type:`u32`,access:`read`},{name:`bands`,view:_(n,`rank-classes`,he,`uint32`,s),type:`u32`,access:`read_write`}],body:`
  let rank = ranks[ranksOffset + index];
  var bandIndex = 0xffffffffu;
  if (selection[selectionOffset + index] != 0u) {
    bandIndex = 0u;
  } else if (rank != ${W}u) {
    let matchLimit = u32(parameters[parametersOffset]);
    bandIndex = select(select(select(select(4u, 3u, rank < 600u), 2u, rank < 200u), 1u, rank < 50u), 0u, rank < matchLimit);
  }
  bands[bandsOffset + index] = bandIndex;`}),{compiled:i.track(await n.compileAsync())}},ze=new Map,Be=async e=>{let t=ze.get(e);return t||(t=await Re(e),ze.set(e,t)),t},Ve=await Be(t.options.standardization),He=i.track(new v),G=[],K=-1,q=null,Ue=null,J=ue({direction:t.options.direction,matchCount:t.options.resultCount,rankedCount:s,ground:t.ground()}),Y=!0,We=()=>{let e=oe.find(e=>e.value===t.options.preset);if(!e)return;let n=f(e.name,e.state);n>=0&&(G.length=0,G.push(n))};We(),G.length||G.push(Math.max(0,f(`Loudoun`,`VA`)));let X=()=>Array.from({length:V},(e,n)=>Number(t.options[`weight${n}`]??1)),Ge=()=>{let e=new Uint32Array(s);for(let t of G)e[t]=1;N.write(e),Fe=ke.write(De(o,G)),Y=!0,$.markStale()},Ke=()=>{let e=t.options;R.write(me({resultCount:e.resultCount,direction:e.direction,excludeReference:e.excludeReference,weights:X()},V)),_e.write(Float32Array.of(e.resultCount,0,0,0)),Y=!0,$.markStale()},Z=e=>q&&q.ranks[e]!==W?q.ranks[e]+1:NaN,qe=()=>q?K>=0?K:q.topIds[0]??-1:-1,Je=e=>G.reduce((t,n)=>t+m[e][n],0)/G.length,Ye=()=>G.length>2?`${G.length} references`:G.map(u).join(` and `),Xe=()=>{if(!q)return;let e=t.options,n=t.ground();J=ue({direction:e.direction,matchCount:e.resultCount,rankedCount:q.rankedCount,ground:n});let r=[0,0,0,0,0],i=0,a=new Set(G);for(let t=0;t<s;t++){let n=q.ranks[t];n===W?a.has(t)||i++:r[ce(n,e.resultCount)]++}let o={table:J,counts:r,missingCount:i,rankedCount:q.rankedCount,referenceName:Ye(),referenceCount:G.length,groundIsDark:n===`dark`};t.setLegendData(`similarPlaces`,o)},Ze=(e,n)=>{let r=t.options;if(n<0||G.includes(n)||!e.length){t.setChart(`profile`,null),t.setChart(`contribution`,null),t.setReadout(`selectedMatch`,null),t.setReadout(`topAttribute`,null);return}let i=r.standardization,a=k[i],o=z(a,G),s=e.reduce((e,t)=>e+t.contribution,0),c=u(G[0]),l=Math.max(3,Math.ceil(Math.max(...e.map(e=>Math.abs(o[e.attribute])+Math.abs(e.delta)))));t.setChart(`profile`,{kind:`dumbbell`,title:`${u(n)} against ${G.length>1?`the reference mean`:c}`,rows:e.map((e,t)=>({label:A[e.attribute].label,a:o[e.attribute],b:a[e.attribute][n],highlight:t===0})),aLabel:G.length>1?`Reference mean`:c,bLabel:u(n),xLabel:i===`zscore`?`Standard deviations from the national mean`:`Percentile (0 lowest, 1 highest)`,xDomain:i===`zscore`?[-l,l]:[0,1],height:Math.max(120,e.length*15+44),description:`Dumbbell chart: standardised values of ${u(n)} and the reference for each attribute with weight, sorted by their share of the distance.`}),t.setChart(`contribution`,{kind:`bars`,title:`Share of the squared distance`,values:e.map(e=>s>0?100*e.contribution/s:0),labels:e.map(e=>A[e.attribute].label),horizontal:!0,highlight:[0],xLabel:`Percent of the squared distance`,table:!1,description:`Bars: each weighted attribute's share of the squared distance between ${u(n)} and the reference.`});let d=e[0];t.setReadout(`selectedMatch`,`${u(n)}, rank ${E(Z(n))}`),t.setReadout(`topAttribute`,`${A[d.attribute].label}, ${s>0?Math.round(100*d.contribution/s):0}% of the squared distance`)},Qe=e=>{let n=t.options,r=[];if(!q||!e){t.setAnnotations(`similar-places`,null);return}if(G.slice(0,2).forEach(e=>{r.push({kind:`point`,coordinate:p(e),text:u(e),marker:`ring`,rank:`subject`,tone:`signal`,priority:10})}),n.showMatches&&q.topIds.slice(0,U).forEach((e,t)=>{r.push({kind:`marker`,coordinate:p(e),number:t+1,text:`${t+1}. ${u(e)}`})}),n.showArcs&&G.length===1&&q.topIds.length&&r.push({kind:`ring`,coordinate:p(G[0]),radiusMeters:e.medianKilometers*1e3,geodesic:!0,text:w(`median match {distance:distance}`,{distance:e.medianKilometers*1e3}),tone:`muted`,priority:3}),n.showNeighbours&&G.length===1&&e.neighbours.slice(0,Me).forEach(({row:e})=>{r.push({kind:`point`,coordinate:p(e),text:u(e),detail:w(`rank {rank:integer}`,{rank:Z(e)}),marker:`dot`,rank:`context`,priority:4})}),n.direction===`most`&&!n.showNeighbours&&n.showArcs){let e=-1,t=0;for(let n of q.topIds){let r=Math.min(...G.map(e=>B(p(e),p(n))));r>t&&(t=r,e=n)}e>=0&&r.push({kind:`note`,coordinate:p(e),title:w(`{distance:distance} away`,{distance:t*1e3}),text:`${u(e)}, the farthest match`,priority:6})}if(n.direction===`least`){let e=z(k[n.standardization],G);q.topIds.slice(0,3).forEach(t=>{let i=be(k[n.standardization],e,X(),t)[0];if(!i)return;let a=A[i.attribute];r.push({kind:`note`,coordinate:p(t),title:u(t),text:w(`{label}: {value} against {reference}`,{label:a.label,value:j(m[i.attribute][t]),reference:j(Je(i.attribute))}),priority:6})})}if(n.outlineSelected&&e.inspected>=0&&e.contributions.length){let t=e.contributions.reduce((e,t)=>e+t.contribution,0),n=e.contributions[0];t>0&&!G.includes(e.inspected)&&r.push({kind:`note`,coordinate:p(e.inspected),title:w(`{share:percent} of the distance`,{share:n.contribution/t}),text:`${A[n.attribute].label} in ${u(e.inspected)}`,priority:6})}t.setAnnotations(`similar-places`,r.length?r:null)},Q=()=>{if(!q)return;let e=t.options,n=X(),r=q.topIds,i=r.length,a=new Set(G),s=e.excludeReference?a:new Set,l=k[e.standardization],f=z(l,G),m=ye(l,f,n,{direction:e.direction,excluded:s}),h=Array.from(m.order.slice(0,i)),g=ye(k.zscore,z(k.zscore,G),de(V),{direction:e.direction,excluded:s}),_=Ce(r,Array.from(g.order.slice(0,i))),v=e=>Math.min(...G.map(t=>B(p(t),p(e)))),y=xe(r.map(v)),b=Se(c,G,Math.max(i,1)),x=b.map(({row:e})=>Z(e)).filter(e=>Number.isFinite(e)),S=Ce(b.map(({row:e})=>e),r),C=re(q.distances);t.setReadout(`reference`,Ye()),t.setReadout(`ranked`,q.rankedCount),t.setReadout(`matchList`,r.map((e,t)=>`${t+1}. ${u(e)}, ${D(v(e)*1e3)}`).join(`
`)||`-`),t.setReadout(`medianKm`,i?D(y*1e3):`-`),t.setReadout(`neighbourKm`,b.length?D(xe(b.map(e=>e.kilometers))*1e3):`-`),t.setReadout(`neighbourRank`,x.length?E(Math.round(xe(x))):`-`),t.setReadout(`neighbourHits`,`${S} of ${b.length}`),t.setReadout(`neighbourList`,b.slice(0,Me).map(({row:e,kilometers:t})=>{let n=Z(e);return`${u(e)}, ${D(t*1e3)}: rank ${Number.isFinite(n)?E(n):`none`}`}).join(`
`)||`-`),t.setReadout(`persist`,`${_} of ${i}`),t.setReadout(`topDistance`,i?q.distances[r[0]].toFixed(2):`-`),t.setReadout(`farSouth`,`${r.filter(e=>fe.has(d(e))).length} of ${i}`);let w=we(k.zscore,n);t.setReadout(`strongestPair`,w?`${A[w.a].label} and ${A[w.b].label.toLowerCase()}, r = ${w.r.toFixed(2)}`:`-`),t.setReadout(`cpuCheck`,`${Ce(r,h)} of ${i} matches agree with the CPU ranking`),t.setReadout(`spread`,`nearest ${ie(C,0).toFixed(2)}, median ${ie(C,.5).toFixed(2)}, farthest ${ie(C,1).toFixed(2)}`),Pe=Oe.write(De(o,r.slice(0,U))),Ie=G.length===1&&i?Ne.write(Ae(o,p(G[0]),r.slice(0,U).map(p))):0;let T=qe();Le=T>=0?o.setSelection(T):0;let O=T>=0&&!a.has(T)?be(l,f,n,T):[];Xe(),Ze(O,T),Qe({contributions:O,inspected:T,medianKilometers:y,neighbours:b}),t.requestLayers()},$=new y(i,`similar-places`,[{buffer:P,size:s*4},{buffer:F,size:s*4},{buffer:I,size:H*4},{buffer:L,size:4}],e=>{let t=ae(e),n=t.words(s).slice(),r=t.floats(s).slice(),i=t.words(H),a=t.words(1)[0],o=0;for(let e=0;e<s;e++)n[e]!==W&&o++;q={ranks:n,distances:r,topIds:Array.from(i.subarray(0,Math.min(a,H))),rankedCount:o},Q()});t.setFurniture({title:{sample:`${E(s)} counties of the contiguous US (Alaska and Hawaii not in the data)`}}),t.setCost({records:s}),Ge(),Ke();let $e=()=>{let e=t.options,n=t.ground(),r=x(C[n].ink),i=x(C[n].halo),a=[o.origin[0],o.origin[1],0],c=[new g({id:`similar-fill`,coordinateOrigin:a,triangles:o.buffers.triangles,features:o.buffers.triangleFeatures,vertexCount:o.buffers.vertexCount,opacity:1,values:he,valueFormat:`uint32`,colormap:`category`,palette:J.colors.map(e=>[e[0],e[1],e[2],e[3]??255]),noDataValue:4294967295,noDataColor:S[n],highlightClasses:Ue}),new h({id:`similar-county-hairlines`,coordinateOrigin:a,segments:o.buffers.outline,instanceCount:o.buffers.outlineCount,widthPixels:s>2e3?.4:.5,color:ne(n)})];if(o.stateLines){let e=te(n);c.push(new h({id:`similar-state-lines`,coordinateOrigin:a,segments:o.stateLines.buffer,instanceCount:o.stateLines.segmentCount,widthPixels:e.widthPixels,color:e.color,outlineColor:e.casing,outlineWidthPixels:(e.casingPixels-e.widthPixels)/2}))}return Pe>0&&e.showMatches&&c.push(new h({id:`similar-match-outlines`,coordinateOrigin:a,segments:Oe.buffer,instanceCount:Pe,widthPixels:1.2,color:r})),Ie>0&&e.showArcs&&c.push(new h({id:`similar-arcs`,coordinateOrigin:a,segments:Ne.buffer,instanceCount:Ie,widthPixels:1,color:[r[0],r[1],r[2],115],dashArray:[5,4]})),Le>0&&(e.outlineSelected||K>=0)&&c.push(new h({id:`similar-inspected`,coordinateOrigin:a,segments:o.selectionBuffer,instanceCount:Le,widthPixels:2.5,color:r,outlineColor:i,outlineWidthPixels:1})),Fe>0&&c.push(new h({id:`similar-reference`,coordinateOrigin:a,segments:ke.buffer,instanceCount:Fe,widthPixels:2.5,color:r,outlineColor:i,outlineWidthPixels:1})),c},et=e=>{if(!q)return null;let n=t.options,r=o.getRings(e),i=p(e),a=G.includes(e),s=k[n.standardization],c=z(s,G),l=[];if(a){let t=[...Array(V).keys()].filter(e=>Number(n[`weight${e}`]??1)>0).sort((e,t)=>Math.abs(c[t])-Math.abs(c[e])).slice(0,3);for(let n of t)l.push({label:A[n].label,value:j(m[n][e]),unit:A[n].unit});return{title:u(e),subtitle:`Reference county`,rows:l,anchor:i,highlight:{kind:`polygon`,rings:r}}}let d=Z(e);if(!Number.isFinite(d))return{title:u(e),subtitle:`County`,note:`Not ranked: an attribute is missing`,anchor:i,highlight:{kind:`polygon`,rings:r}};let f=ce(d-1,n.resultCount),h=J.colors[f],g=Math.min(...G.map(e=>B(p(e),i)));l.push({label:n.direction===`most`?`Similarity rank`:`Rank from the least similar`,value:E(d),unit:`of ${E(q.rankedCount)}`,swatch:[h[0],h[1],h[2],h[3]??255],emphasis:!0},{label:`Distance in attribute space`,value:q.distances[e].toFixed(2),unit:n.standardization===`zscore`?`standard deviations`:`percentile units`},{label:`From the reference`,value:D(g*1e3)});let _=X();for(let t of be(s,c,_,e).slice(0,3)){let n=A[t.attribute];l.push({label:n.label,value:j(m[t.attribute][e]),unit:`${n.unit}; reference ${j(Je(t.attribute))}`})}return{title:u(e),subtitle:`County, ${T(d)} in the ranking`,rows:l,anchor:i,highlight:{kind:`polygon`,rings:r}}};return{getCompiledGraphs:()=>[...ze.values()].map(e=>e.compiled),setOption(e){e===`standardization`?He.request({build:()=>Be(t.options.standardization),commit:e=>{Ve=e,Y=!0,$.markStale(),t.requestLayers()}}):e===`preset`?(We(),K=-1,Ge(),Ke()):e===`click`||(e===`showMatches`||e===`showArcs`||e===`showNeighbours`||e===`outlineSelected`?Q():Ke()),t.requestLayers()},encode(e,t){(Y||t.frameIndex<2)&&(Ve.compiled.encode(e,{parameters:void 0}),Y=!1,$.markStale()),t.frameIndex>=2&&$.flush(e)},getLayers:$e,onGroundChange(){Xe(),t.requestLayers()},onLegendFilter(e,n){Ue=n?[...n]:null,t.requestLayers()},getTooltip(e){if(!e.coordinate)return null;let t=o.locator.find(e.coordinate);return t?et(t.index):null},onClick(e){if(!e.coordinate)return!1;let n=o.locator.find(e.coordinate);if(!n)return!1;let r=n.index,i=t.options.click;if(i===`inspect`)return K=r===K?-1:r,Q(),!0;let a=G.indexOf(r);return i===`add`&&a>=0?G.length>1&&G.splice(a,1):i===`add`?G.length<je&&G.push(r):(G.length=0,G.push(r)),K=-1,t.setOptions({preset:`custom`}),Ge(),t.requestLayers(),!0},destroy(){$.stop(),t.setAnnotations(`similar-places`,null),t.setChart(`profile`,null),t.setChart(`contribution`,null),i.destroy()}}}export{Ne as createSimilarPlaces};