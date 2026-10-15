import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{t as e}from"./status-BwNARr9H.js";import{E as t,h as n}from"./analysis-frame-Blu0tYdw.js";import{a as r,b as i,d as a,f as o,i as s,n as c,p as l,r as u,s as d,t as f,u as p}from"./pair-statistics-parameters-CO--_nmZ.js";import{c as m,d as h,f as g,i as _,l as v,n as y,o as b,p as x,r as S,s as C,u as w}from"./permutation-inference-kernels-D-xhEewi.js";import{N as T,_ as E,d as D,h as O,j as k,t as A,v as j,w as M,y as N}from"./resources-CwvhEOBe.js";import{o as P,s as F}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{r as I,s as ee}from"./spatial-weights-D4bd9AQ6.js";import{t as te}from"./gpu-spatial-lag-Q5OutSKD.js";import{t as ne}from"./philox-random-CHwyk0dd.js";import{a as re,i as L,n as ie,r as R,t as z}from"./gpu-global-spatial-statistics-Bnj2kmfn.js";import{t as ae}from"./permutation-random-C7SSHCbB.js";import{t as oe}from"./permutation-shuffle-B9oRDb5C.js";import{U as se,c as ce,o as B}from"./index-Bjk18V2I.js";import{t as V}from"./pair-statistics-fixed-point-unit-Byg9Sbw4.js";import{t as H}from"./graph-buffers-B4_80RlX.js";import{t as le}from"./graph-swap-41Ru7LVu.js";import{t as ue}from"./summary-reader-CsgWxENf.js";import{t as de}from"./mode-kernels-Dxm2GWvI.js";import"./anchors-D9E9ucGL.js";import{g as fe,o as pe}from"./gazetteer-OhisynFF.js";import{C as me,E as U,S as he,a as ge,b as _e,l as ve,x as ye}from"./class-table-BVEo3-Vs.js";import{a as be}from"./hue-registry-4FjIJ9xz.js";import{a as xe,i as Se,n as W,r as Ce}from"./live-text-83XBWyr1.js";import{i as we}from"./picking-DRmnefrA.js";import{i as Te}from"./projection-notes-BUOXX2bT.js";import{r as Ee,t as De}from"./segments-BtINkq-i.js";import"./b6-polygons-Bf2djBQx.js";import{f as Oe,l as ke}from"./hot-spots.style-DcEoMUmX.js";import{a as Ae,i as je,n as Me,r as Ne,t as G}from"./b4-format-Cgp0VQWI.js";import{n as Pe,r as Fe,t as Ie}from"./b4-weights-core-mAiwYc53.js";import{n as Le,r as Re}from"./b4-geography-Dd13GyrB.js";import{t as ze}from"./b4-layers-BaIKQJXr.js";import{a as Be,c as Ve,i as He,l as Ue,n as We,o as Ge,r as Ke,s as qe,t as Je}from"./global-autocorrelation.style-DpjR4NU4.js";var K=`GPUGlobalPermutationTest`,q=256,Ye=2048,J=256,Y=256,Xe=128,Ze=64,Qe={base:4,greater:5,lesser:6,stoppedAt:7,done:8,length:12};function $e(e,t){let n=Math.ceil(e/Ye);return Math.min(t,Math.max(Math.ceil(Xe/n),Math.ceil(t/Ze),1))}var et=Math.floor((2**32-1)/J),tt={observed:0,pseudoPValue:1,exceedances:2,permutationCount:3,simulatedMean:4,simulatedStandardDeviation:5,zSimulated:6,pZSimulated:7,minimum:8,maximum:9,islandCount:10,analysisCount:11,length:12},nt=class{id;props;constructor(e){let t=e.id??`global-permutation-test`;if(this.id=t,this.props=e,m({...e,id:t,operation:K}),![`moran`,`geary`,`getisOrdG`,`bivariateMoran`,`leesL`].includes(e.statistic))throw Error(`${t} unknown statistic ${e.statistic}`);if(x(t,e.stopAfterExceedances,g(t,e.alternative),e.maximumPermutations),ee(t,e.islands),(e.statistic===`bivariateMoran`||e.statistic===`leesL`)&&!e.secondValues)throw Error(`${t} ${e.statistic} requires secondValues`);j(t,`permutations x row blocks`,(e.maximumPermutations+1)*Math.ceil(e.values.length/Ye),et);let{output:n}=e;if(N(n.summary,[`float32`],`${t} output.summary`),D(t,`output.summary length`,n.summary.length,tt.length),n.referenceDistribution&&(N(n.referenceDistribution,[`float32`],`${t} output.referenceDistribution`),n.referenceDistribution.length!==e.maximumPermutations))throw Error(`${t} referenceDistribution length must equal maximumPermutations`);n.histogram&&(k(n.histogram,`${t} output.histogram`),D(t,`output.histogram length`,n.histogram.length)),O(t,[e.output.summary,e.output.referenceDistribution,e.output.histogram],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.values,e.secondValues,e.mask,e.parameters,e.weights.status?.overflow])}getParameterValues(e){return v(e,{maximumPermutations:this.props.maximumPermutations})}getInferenceMetadata(e){this.getParameterValues(e);let t=ee(this.id,this.props.islands)===`keep`;return{alternative:g(this.id,this.props.alternative),permutationCount:e.permutationCount,seed:e.seed,multipleTesting:`none`,testedRowsDefinition:`one global statistic over the analysis rows (nonzero mask, finite values; islands, `+(t?`rows without a positive-weight neighbor, are kept in n)`:`rows without a positive-weight neighbor, are excluded)`)}}getCommandNodes(e){let{id:t,props:n}=this,{weights:r,parameters:i,statistic:a,maximumPermutations:o}=n;E(t,e,[r.offsets,r.neighbors,r.weights,n.values,n.secondValues,n.mask,i,r.status?.overflow,n.output.summary,n.output.referenceDistribution,n.output.histogram]);let s=n.values.length,c=a===`leesL`,l=a===`bivariateMoran`||c,u=g(t,n.alternative),d=w(u),f=b(e,{...n,secondValues:l?n.secondValues:void 0,islands:n.islands,id:`${t}-inputs`,operation:K,centerX:a!==`getisOrdG`}),p=f.nodes,{rowPositions:m,compactX:v,compactY:x,totals:T,islandCount:D}=f,O=f.weights,k=Math.ceil(s/Ye),A=o+1,j=M(e,`${t}-partials`,`float32`,A*k),N=M(e,`${t}-pair-sums`,`float32`,A),P=l?`compactY`:`compactX`,ee=c?`let permutedNeighbor = getPermutedPosition(neighborPosition, permutation, count);
        lag += weight * compactX[compactXOffset + permutedNeighbor];
        lagY += weight * compactY[compactYOffset + permutedNeighbor];`:a===`geary`?`let difference = focus - neighborValue;
        lag += weight * difference * difference;`:`lag += weight * neighborValue;`,te=c?`partialSum += lag * lagY;`:a===`geary`?`partialSum += lag;`:`partialSum += focus * lag;`,re=n.stopAfterExceedances,L=re!==void 0,ie=L?M(e,`${t}-batch-state`,`uint32`,Qe.length):void 0,R=ie??i,z=[{name:`offsets`,view:r.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`weights`,view:O,type:`f32`,access:`read`},{name:`rowPositions`,view:m,type:`u32`,access:`read`},{name:`compactX`,view:v,type:`f32`,access:`read`},...l?[{name:`compactY`,view:x,type:`f32`,access:`read`}]:[],{name:`parameters`,view:R,type:`u32`,access:`read`},{name:`partials`,view:j,type:`f32`,access:`read_write`}],se=t=>F(e,{id:t.id,operation:K,variant:t.variant,bindings:z,workgroupSize:J,invocationCount:t.invocationCount,guardIndex:!1,...t.gate,declarations:`const ROWS: u32 = ${s}u;
const CAPACITY: u32 = ${r.neighbors.length}u;
const INVALID_POSITION: u32 = ${S}u;
const BLOCK_ROWS: u32 = ${Ye}u;
const BLOCK_COUNT: u32 = ${k}u;
const WORKGROUP_SIZE: u32 = ${J}u;
var<workgroup> partialSums: array<f32, ${J}>;
${C(o)}
${ae}
${oe(ne.permutationShuffle.id)}
fn getPermutedPosition(position: u32, permutation: u32, count: u32) -> u32 {
  if (permutation == 0u) {
    return position;
  }
  return getSwapOrNotPermutationIndex(position, count);
}`,body:`let workgroup = index / WORKGROUP_SIZE;
  let permutation = ${t.permutationWGSL};
  let block = workgroup % BLOCK_COUNT;
  let count = rowPositions[rowPositionsOffset + ROWS];
  let runsPermutation = index < INVOCATION_COUNT && permutation <= readValidPermutationCount();
  if (runsPermutation && permutation > 0u) {
    loadSwapOrNotRounds(readSeedKey(), permutation, count, localInvocationIndex, WORKGROUP_SIZE);
  }
  workgroupBarrier();
  var partialSum = 0.0;
  if (runsPermutation) {
    let rowEnd = min((block + 1u) * BLOCK_ROWS, ROWS);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < rowEnd; row += WORKGROUP_SIZE) {
      let position = rowPositions[rowPositionsOffset + row];
      if (position == INVALID_POSITION) {
        continue;
      }
      // Lee's L has no focus term: it is the product of the two lags.
      let focus = ${c?`0.0`:`compactX[compactXOffset + ${l?`position`:`getPermutedPosition(position, permutation, count)`}]`};
      var lag = 0.0;
      var lagY = 0.0;
      ${I(`rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION`,`let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
        let neighborValue = ${P}[${P}Offset +
          getPermutedPosition(neighborPosition, permutation, count)];
        ${ee}`,`row`)}
      ${te}
    }
  }
  partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (index < INVOCATION_COUNT && localInvocationIndex == 0u${L?` && permutation <= MAXIMUM_PERMUTATIONS`:``}) {
    partials[partialsOffset + permutation * BLOCK_COUNT + block] = partialSums[0];
  }`}),ce=t=>F(e,{id:t.id,operation:K,variant:t.variant,bindings:[{name:`partials`,view:j,type:`f32`,access:`read`},{name:`pairSums`,view:N,type:`f32`,access:`read_write`},...t.permutationWGSL?[{name:`parameters`,view:R,type:`u32`,access:`read`}]:[]],invocationCount:t.invocationCount,...t.gate,declarations:`const BLOCK_COUNT: u32 = ${k}u;${t.permutationWGSL?`\n${C(o)}`:``}`,body:t.permutationWGSL?`let permutation = ${t.permutationWGSL};
  if (permutation > readValidPermutationCount()) {
    return;
  }
  var sum = 0.0;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += partials[partialsOffset + permutation * BLOCK_COUNT + block];
  }
  pairSums[pairSumsOffset + permutation] = sum;`:`var sum = 0.0;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += partials[partialsOffset + index * BLOCK_COUNT + block];
  }
  pairSums[pairSumsOffset + index] = sum;`});!L||!ie?p.push(se({id:`${t}-pairs`,variant:`pairs-${a}`,invocationCount:A*k*J,permutationWGSL:`workgroup / BLOCK_COUNT`}),ce({id:`${t}-pair-sums`,variant:`pair-sums`,invocationCount:A})):p.push(...this.getSequentialBatchNodes(e,{batchState:ie,createPairsNode:se,createPairSumsNode:ce,pairSums:N,blockCount:k,alternative:u,stopAfter:re}));let B=_,V={moran:`count / (totals[totalsOffset + ${B.s0}u] * totals[totalsOffset + ${B.sumSquares}u])`,geary:`(count - 1.0) / (2.0 * totals[totalsOffset + ${B.s0}u] * totals[totalsOffset + ${B.sumSquares}u])`,getisOrdG:`1.0 / (count * (count - 1.0) * totals[totalsOffset + ${B.mean}u] * totals[totalsOffset + ${B.mean}u] - totals[totalsOffset + ${B.sumSquares}u])`,bivariateMoran:`count / (totals[totalsOffset + ${B.s0}u] * sqrt(totals[totalsOffset + ${B.sumSquares}u] * totals[totalsOffset + ${B.sumSquaresY}u]))`,leesL:`count / (totals[totalsOffset + ${B.s0Squared}u] * sqrt(totals[totalsOffset + ${B.sumSquares}u] * totals[totalsOffset + ${B.sumSquaresY}u]))`}[a],H=tt,le=n.output.histogram?.length??0,ue=[{name:`pairSums`,view:N,type:`f32`,access:`read`},{name:`totals`,view:T,type:`f32`,access:`read`},{name:`rowPositions`,view:m,type:`u32`,access:`read`},{name:`islandCount`,view:D,type:`u32`,access:`read`},{name:`parameters`,view:R,type:`u32`,access:`read`},{name:`results`,view:n.output.summary,type:`f32`,access:`read_write`}];return n.output.referenceDistribution&&ue.push({name:`referenceDistribution`,view:n.output.referenceDistribution,type:`f32`,access:`read_write`}),n.output.histogram&&ue.push({name:`histogram`,view:n.output.histogram,type:`atomic<u32>`,access:`read_write`}),p.push(F(e,{id:`${t}-finalize`,operation:K,variant:`finalize`,bindings:ue,workgroupSize:q,invocationCount:q,guardIndex:!1,declarations:`${C(o)}
${y}
const ROWS: u32 = ${s}u;
const BINS: u32 = ${le}u;
const WORKGROUP_SIZE: u32 = ${q}u;
var<workgroup> scratchFloat: array<f32, ${q}>;
var<workgroup> scratchCount: array<u32, ${q}>;

// Each reduction ends with a barrier so the scratch can be reused; all threads get the result.
fn reduceFloat(local: u32, value: f32, mode: u32) -> f32 {
  scratchFloat[local] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      let other = scratchFloat[local + stride];
      scratchFloat[local] = select(select(scratchFloat[local] + other, max(scratchFloat[local], other), mode == 2u), min(scratchFloat[local], other), mode == 1u);
    }
    workgroupBarrier();
  }
  let result = scratchFloat[0];
  workgroupBarrier();
  return result;
}

fn reduceCount(local: u32, value: u32) -> u32 {
  scratchCount[local] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      scratchCount[local] += scratchCount[local + stride];
    }
    workgroupBarrier();
  }
  let result = scratchCount[0];
  workgroupBarrier();
  return result;
}`,body:`let local = localInvocationIndex;
  // An invalid requested count is never clamped: no permutation is summed and the outputs are NaN.
  ${L?`// A stopped run used the first L permutations; the others the requested count.
  let stoppedAt = parameters[parametersOffset + ${Qe.stoppedAt}u];
  let permutations = select(readValidPermutationCount(), stoppedAt, stoppedAt != 0u);`:`let permutations = readValidPermutationCount();`}
  let count = totals[totalsOffset + ${B.count}u];
  let scale = ${V};
  let observedSum = pairSums[pairSumsOffset];
  let observed = scale * observedSum;
  var greaterPart = 0u;
  var lesserPart = 0u;
  var totalPart = 0.0;
  var minimumPart = 3.0e38;
  var maximumPart = -3.0e38;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let pairSum = pairSums[pairSumsOffset + permutation];
    let simulated = scale * pairSum;
    greaterPart += select(0u, 1u, pairSum >= observedSum);
    lesserPart += select(0u, 1u, pairSum <= observedSum);
    totalPart += simulated;
    minimumPart = min(minimumPart, simulated);
    maximumPart = max(maximumPart, simulated);
    ${n.output.referenceDistribution?`referenceDistribution[referenceDistributionOffset + permutation - 1u] = simulated;`:``}
  }
  ${n.output.referenceDistribution?`for (var permutation = permutations + 1u + local; permutation <= MAXIMUM_PERMUTATIONS; permutation += WORKGROUP_SIZE) {
    referenceDistribution[referenceDistributionOffset + permutation - 1u] = getQuietNaN(permutation);
  }`:``}
  let greater = reduceCount(local, greaterPart);
  let lesser = reduceCount(local, lesserPart);
  let total = reduceFloat(local, totalPart, 0u);
  let minimum = reduceFloat(local, minimumPart, 1u);
  let maximum = reduceFloat(local, maximumPart, 2u);
  let mean = select(getQuietNaN(2u), total / f32(permutations), permutations > 0u);
  var squaresPart = 0.0;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let deviation = scale * pairSums[pairSumsOffset + permutation] - mean;
    squaresPart += deviation * deviation;
  }
  let squares = reduceFloat(local, squaresPart, 0u);
  let standardDeviation = select(getQuietNaN(3u), sqrt(squares / f32(permutations)), permutations > 0u);
  ${u===`folded`?`var foldedPart = 0u;
  let observedDistance = abs(observed - mean);
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    foldedPart += select(0u, 1u, abs(scale * pairSums[pairSumsOffset + permutation] - mean) >= observedDistance);
  }
  let folded = reduceCount(local, foldedPart);`:``}
  ${n.output.histogram?`for (var bin = local; bin < BINS; bin += WORKGROUP_SIZE) {
    atomicStore(&histogram[histogramOffset + bin], 0u);
  }
  storageBarrier();
  workgroupBarrier();
  let range = maximum - minimum;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let simulated = scale * pairSums[pairSumsOffset + permutation];
    var bin = 0u;
    if (range > 0.0 && isFiniteFloat(range)) {
      bin = min(u32(max(floor((simulated - minimum) / range * f32(BINS)), 0.0)), BINS - 1u);
    }
    atomicAdd(&histogram[histogramOffset + bin], 1u);
  }`:``}
  if (local == 0u) {
    let larger = ${h(u)};
    let zSimulated = (observed - mean) / standardDeviation;
    let validZ = standardDeviation > 0.0 && isFiniteFloat(zSimulated);
    // Tested: a valid permutation count and an observed statistic that exists (n >= 2, not truncated).
    let tested = permutations > 0u && isFiniteFloat(observed);
    results[resultsOffset + ${H.observed}u] = select(getQuietNaN(0u), observed, isFiniteFloat(observed));
    results[resultsOffset + ${H.pseudoPValue}u] = select(getQuietNaN(1u), ${L?`select(f32(min(${d}u * (larger + 1u), permutations + 1u)) / f32(permutations + 1u), f32(min(${d}u * ${re}u, permutations)) / f32(permutations), stoppedAt != 0u)`:`f32(min(${d}u * (larger + 1u), permutations + 1u)) / f32(permutations + 1u)`}, tested);
    results[resultsOffset + ${H.exceedances}u] = select(getQuietNaN(1u), f32(larger), tested);
    results[resultsOffset + ${H.permutationCount}u] = select(getQuietNaN(1u), f32(permutations), permutations > 0u);
    results[resultsOffset + ${H.simulatedMean}u] = mean;
    results[resultsOffset + ${H.simulatedStandardDeviation}u] = standardDeviation;
    results[resultsOffset + ${H.zSimulated}u] = select(getQuietNaN(1u), zSimulated, validZ && tested);
    results[resultsOffset + ${H.pZSimulated}u] = select(getQuietNaN(1u), 0.5 * getTwoSidedPValue(zSimulated), validZ && tested);
    results[resultsOffset + ${H.minimum}u] = select(getQuietNaN(4u), minimum, permutations > 0u);
    results[resultsOffset + ${H.maximum}u] = select(getQuietNaN(4u), maximum, permutations > 0u);
    results[resultsOffset + ${H.islandCount}u] = f32(islandCount[islandCountOffset]);
    results[resultsOffset + ${H.analysisCount}u] = f32(rowPositions[rowPositionsOffset + ROWS]);
  }`})),p}getSequentialBatchNodes(e,t){let{id:n,props:r}=this,{batchState:i,pairSums:a,blockCount:o,alternative:s,stopAfter:c}=t,{maximumPermutations:l,parameters:u}=r,d=r.values.length,f=$e(d,l),p=Math.ceil(l/f),m=Math.ceil(f/Y),g=Qe,_=f*o*J,v=P(e,{id:`${n}-batch-dispatch`,operation:K,shapes:[[_,J],[f]]}),y=[F(e,{id:`${n}-sequential-init`,operation:K,variant:`sequential-init`,bindings:[{name:`parameters`,view:u,type:`u32`,access:`read`},{name:`state`,view:i,type:`u32`,access:`read_write`},v.binding],invocationCount:1,declarations:C(l),body:`for (var word = 0u; word < 4u; word++) {
    state[stateOffset + word] = parameters[parametersOffset + word];
  }
  for (var word = ${g.base}u; word < ${g.length}u; word++) {
    state[stateOffset + word] = 0u;
  }
  if (isPermutationCountValid()) {
    ${v.armWGSL}
  } else {
    state[stateOffset + ${g.done}u] = 1u;
    ${v.disarmWGSL}
  }`}),t.createPairsNode({id:`${n}-pairs-observed`,variant:`pairs-${r.statistic}-observed`,invocationCount:o*J,permutationWGSL:`0u`}),t.createPairSumsNode({id:`${n}-pair-sums-observed`,variant:`pair-sums`,invocationCount:1})],b=`const BATCH_SIZE: u32 = ${f}u;
const CHUNK_LENGTH: u32 = ${m}u;
const STOP_AFTER_EXCEEDANCES: u32 = ${c}u;
const WORKGROUP_SIZE: u32 = ${Y}u;
var<workgroup> chunkGreater: array<u32, ${Y}>;
var<workgroup> chunkLesser: array<u32, ${Y}>;
fn getExceedanceCount(greater: u32, lesser: u32) -> u32 {
  return ${h(s)};
}`;for(let o=0;o<p;o++){let s=v.getGate(0),c=v.getGate(1);y.push(t.createPairsNode({id:`${n}-batch-${o}-pairs`,variant:`pairs-${r.statistic}-batch`,invocationCount:_,permutationWGSL:`parameters[parametersOffset + ${g.base}u] + 1u + workgroup / BLOCK_COUNT`,gate:{...s,condition:{...s.condition,id:`${n}-batch-${o}-pairs-gate`}}}),t.createPairSumsNode({id:`${n}-batch-${o}-pair-sums`,variant:`pair-sums-batch`,invocationCount:f,permutationWGSL:`parameters[parametersOffset + ${g.base}u] + 1u + index`,gate:{...c,condition:{...c.condition,id:`${n}-batch-${o}-sums-gate`}}}),F(e,{id:`${n}-batch-${o}-scan`,operation:K,variant:`sequential-scan`,bindings:[{name:`pairSums`,view:a,type:`f32`,access:`read`},{name:`state`,view:i,type:`u32`,access:`read_write`},v.binding],workgroupSize:Y,invocationCount:Y,guardIndex:!1,declarations:b,body:`let local = localInvocationIndex;
  let count = state[stateOffset + 2u];
  let base = state[stateOffset + ${g.base}u];
  let isRunning = state[stateOffset + ${g.done}u] == 0u;
  let observedSum = pairSums[pairSumsOffset];
  var greaterPart = 0u;
  var lesserPart = 0u;
  if (isRunning) {
    for (var item = 0u; item < CHUNK_LENGTH; item++) {
      let offset = local * CHUNK_LENGTH + item;
      let permutation = base + 1u + offset;
      if (offset < BATCH_SIZE && permutation <= count) {
        let pairSum = pairSums[pairSumsOffset + permutation];
        greaterPart += select(0u, 1u, pairSum >= observedSum);
        lesserPart += select(0u, 1u, pairSum <= observedSum);
      }
    }
  }
  chunkGreater[local] = greaterPart;
  chunkLesser[local] = lesserPart;
  workgroupBarrier();
  if (local == 0u && isRunning) {
    var greater = state[stateOffset + ${g.greater}u];
    var lesser = state[stateOffset + ${g.lesser}u];
    var stoppedAt = 0u;
    for (var chunk = 0u; chunk < WORKGROUP_SIZE && stoppedAt == 0u; chunk++) {
      let greaterAfter = greater + chunkGreater[chunk];
      let lesserAfter = lesser + chunkLesser[chunk];
      if (getExceedanceCount(greaterAfter, lesserAfter) >= STOP_AFTER_EXCEEDANCES) {
        for (var item = 0u; item < CHUNK_LENGTH; item++) {
          let offset = chunk * CHUNK_LENGTH + item;
          let permutation = base + 1u + offset;
          if (offset < BATCH_SIZE && permutation <= count) {
            let pairSum = pairSums[pairSumsOffset + permutation];
            greater += select(0u, 1u, pairSum >= observedSum);
            lesser += select(0u, 1u, pairSum <= observedSum);
            if (getExceedanceCount(greater, lesser) >= STOP_AFTER_EXCEEDANCES) {
              stoppedAt = permutation;
              break;
            }
          }
        }
      } else {
        greater = greaterAfter;
        lesser = lesserAfter;
      }
    }
    let nextBase = min(base + BATCH_SIZE, count);
    state[stateOffset + ${g.greater}u] = greater;
    state[stateOffset + ${g.lesser}u] = lesser;
    state[stateOffset + ${g.base}u] = nextBase;
    if (stoppedAt != 0u || nextBase >= count) {
      state[stateOffset + ${g.stoppedAt}u] = stoppedAt;
      state[stateOffset + ${g.done}u] = 1u;
      ${v.disarmWGSL}
    }
  }`}))}return y}},rt=4294967295;function it(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUSpatialCorrelogram target must hold 8 elements`);f(`GPUSpatialCorrelogram`,e.bounds,e.maximumDistance);let n=e.varianceAssumption??`randomization`;if(n!==`normality`&&n!==`randomization`)throw Error(`GPUSpatialCorrelogram varianceAssumption must be normality or randomization`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,n===`randomization`?1:0,0,0]),t}var X=`GPUSpatialCorrelogram`,at=Object.freeze({...n,measure:`distance`}),ot=64,st=16777216,ct=8,lt=1e-4,ut=4,dt=class{id;props;spatialContext;studyWindow;bandMode;constructor(n){this.id=n.id??`spatial-correlogram`,this.props=n;let r=this.id;this.spatialContext=t(r,n.spatialContext,at),this.studyWindow=i(r,n.studyWindow,this.spatialContext),o(r,{...n,parameterLength:8});let a=n.bandMode??`cumulative`;if(a!==`cumulative`&&a!==`annulus`)throw Error(`${r} bandMode must be cumulative or annulus`);this.bandMode=a;let{bandCount:c,output:l}=n;D(r,`bandCount`,c,1,ot),s(r,c,ct),l.status?.overflow&&e(r,{overflow:l.status?.overflow});for(let[e,t]of[[`output.moransI`,l.moransI],[`output.zScores`,l.zScores],[`output.pValues`,l.pValues],[`output.expectedI`,l.expectedI],[`output.varianceI`,l.varianceI]])if(t&&(N(t,[`float32`],`${r} ${e}`),t.length<c))throw Error(`${r} ${e} must hold bandCount rows`);if(l.pairCounts&&(k(l.pairCounts,`${r} output.pairCounts`),l.pairCounts.length<c))throw Error(`${r} output.pairCounts must hold bandCount rows`);if(l.peakBands&&(k(l.peakBands,`${r} output.peakBands`),l.peakBands.length<2))throw Error(`${r} output.peakBands must hold 2 rows`);if(l.statistics&&(N(l.statistics,[`float32`],`${r} output.statistics`),l.statistics.length<5))throw Error(`${r} output.statistics must hold 5 rows`);O(r,[l.moransI,l.zScores,l.pValues,l.expectedI,l.varianceI,l.pairCounts,l.peakBands,l.statistics,l.status?.overflow],[n.positions,n.values,n.parameters,n.mask])}getCommandNodes(e){let{id:t,props:n,bandMode:i}=this,{positions:o,parameters:s,gridSize:f,bandCount:m,output:h}=n;l(t,e.device,o.length),E(t,e,[o,n.values,s,n.mask,h.moransI,h.zScores,h.pValues,h.expectedI,h.varianceI,h.pairCounts,h.peakBands,h.statistics,h.status?.overflow]);let g=o.length,_=i===`cumulative`,v=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);v&&E(t,e,[v.edges,v.summary,v.slabs.params,v.slabs.slabOffsets,v.slabs.slabEntries,v.sampleCells,v.sampleCellCount,v.packed]);let y=v?a(e,{id:t,operation:X,parameters:s,parameterLength:s.length,window:v}):void 0,b=y?.parameters??s,x=d(e,{id:t,operation:X,positions:o,parameters:b,gridSize:f,values:n.values,mask:n.mask,studyWindow:v}),S=x.statistics,C=[...y?.nodes??[],...x.nodes],w=V(e,{id:t,operation:X,values:n.values,mask:n.mask,statistics:S});C.push(...w.nodes);let T=w.unitVariance,D=`const MOMENTS: u32 = ${g}u;
const BAND_COUNT: u32 = ${m}u;
const PRODUCT_UNITS_PER_VARIANCE: f32 = ${st}.0;
const LARGE_TERM_UNIT: f32 = ${2**12}.0;

// Fixed-point units per unit of product, from the typical magnitude (the variance), not the maximum.
fn getProductScale() -> f32 {
  let unitVariance = unitVarianceView[unitVarianceViewOffset];
  return select(0.0, PRODUCT_UNITS_PER_VARIANCE / unitVariance, unitVariance > 0.0);
}`,O=M(e,`${t}-fourth-terms`,`float32`,g),k=M(e,`${t}-fourth-total`,`float32`,1);C.push(F(e,{id:`${t}-fourth-terms`,operation:X,variant:`fourth-terms`,bindings:[{name:`statistics`,view:S,type:`f32`,access:`read`},{name:`fourthTerms`,view:O,type:`f32`,access:`read_write`}],invocationCount:g,declarations:r,body:`let centered = statistics[statisticsOffset + index];
  let square = centered * centered;
  fourthTerms[fourthTermsOffset + index] = select(0.0, square * square, isFiniteFloat(centered));`}),...p(e,{id:`${t}-fourth-sum`,operation:X,input:O,output:k}));let A=c(e,{id:`${t}-histogram`,operation:X,positions:o,parameters:b,gridSize:f,sortedRows:x.sortedRows,cellOffsets:x.cellOffsets,slotCount:m,channelCount:ct,pairOrder:`ordered`,extraBindings:[{name:`statistics`,view:S,type:`f32`,access:`read`},{name:`unitVarianceView`,view:T,type:`f32`,access:`read`}],declarations:D,focusPrologue:`let productScale = getProductScale();
    let focusValue = statistics[statisticsOffset + focus];
    var degree: array<u32, ${m}>;`,pairAction:`let band = min(u32(pairDistance / lattice.maximumDistance * f32(BAND_COUNT)), BAND_COUNT - 1u);
            degree[band] += 1u;
            let product = focusValue * statistics[statisticsOffset + neighbor];
            accumulate(band, 0u, 1u);
            let positive = product >= 0.0;
            accumulateScaled(band, select(2u, 1u, positive), select(7u, 6u, positive), abs(product) * productScale);`,focusEpilogue:`var runningDegree = 0u;
    for (var band = 0u; band < BAND_COUNT; band++) {
      ${_?`runningDegree += degree[band];`:`runningDegree = degree[band];`}
      let high = runningDegree >> 12u;
      let low = runningDegree & 0xfffu;
      accumulate(band, 3u, high * high);
      accumulate(band, 4u, 2u * high * low);
      accumulate(band, 5u, low * low);
    }`});C.push(...A.nodes);let j=M(e,`${t}-band-results`,`float32`,m*ut),N=M(e,`${t}-band-counts`,`uint32`,m);C.push(F(e,{id:`${t}-finish`,operation:X,variant:_?`finish-cumulative`:`finish-annulus`,bindings:[{name:`accumulators`,view:A.accumulators,type:`u32`,access:`read`},{name:`statistics`,view:S,type:`f32`,access:`read`},{name:`unitVarianceView`,view:T,type:`f32`,access:`read`},{name:`parameters`,view:b,type:`f32`,access:`read`},{name:`fourthTotal`,view:k,type:`f32`,access:`read`},{name:`bandResults`,view:j,type:`f32`,access:`read_write`},{name:`bandCounts`,view:N,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${r}
${D}
const CHANNEL_COUNT: u32 = ${ct}u;
const VARIANCE_CANCELLATION: f32 = ${lt};
const FIRST_BAND_OFFSET: u32 = ${_?0:1}u;

// Exact 64-bit sum of one channel over annuli [first, last], as (low, high) words.
fn sumChannel(first: u32, last: u32, channel: u32) -> vec2<u32> {
  var low = 0u;
  var high = 0u;
  for (var band = first; band <= last; band++) {
    let word = (band * CHANNEL_COUNT + channel) * 2u;
    let addend = accumulators[accumulatorsOffset + word];
    let previous = low;
    low += addend;
    high += accumulators[accumulatorsOffset + word + 1u] + select(0u, 1u, low < previous);
  }
  return vec2<u32>(low, high);
}

fn toFloat(sum: vec2<u32>) -> f32 {
  return f32(sum.y) * 4294967296.0 + f32(sum.x);
}`,body:`let firstBand = select(0u, index, FIRST_BAND_OFFSET == 1u);
  let count = f32(statistics[statisticsOffset + MOMENTS]);
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 2u];
  let productScale = getProductScale();
  let orderedPairs = sumChannel(firstBand, index, 0u);
  // Unordered pairs = ordered / 2, saturated to 32 bits.
  bandCounts[bandCountsOffset + index] =
    select(0xffffffffu, (orderedPairs.y << 31u) | (orderedPairs.x >> 1u), orderedPairs.y < 2u);
  let weightSum = toFloat(orderedPairs);
  // The large channels hold products in units of LARGE_TERM_UNIT fixed-point units.
  let positiveSum = toFloat(sumChannel(firstBand, index, 1u)) + toFloat(sumChannel(firstBand, index, 6u)) * LARGE_TERM_UNIT;
  let negativeSum = toFloat(sumChannel(firstBand, index, 2u)) + toFloat(sumChannel(firstBand, index, 7u)) * LARGE_TERM_UNIT;
  let productSum = select(0.0, (positiveSum - negativeSum) / productScale, productScale > 0.0);
  // Squared neighbor counts are per band (already cumulative when the bands are).
  let degreeSquares = toFloat(sumChannel(index, index, 3u)) * 16777216.0 +
    toFloat(sumChannel(index, index, 4u)) * 4096.0 + toFloat(sumChannel(index, index, 5u));
  let nan = getQuietNaN(index);
  var moransI = nan;
  var expected = nan;
  var variance = nan;
  var zScore = nan;
  let randomization = parameters[parametersOffset + 5u] != 0.0;
  if (count >= 2.0) {
    expected = -1.0 / (count - 1.0);
  }
  if (weightSum > 0.0 && count >= 3.0 && sumOfSquares > 0.0) {
    moransI = count / weightSum * productSum / sumOfSquares;
    let s1 = 2.0 * weightSum;
    let s2 = 4.0 * degreeSquares;
    let weightSquare = weightSum * weightSum;
    let countSquare = count * count;
    // Magnitude of the summed terms: a variance below VARIANCE_CANCELLATION of it is f32
    // cancellation noise (for example a band holding every pair, whose true variance is 0).
    var magnitude = 0.0;
    if (!randomization) {
      let firstTerm = countSquare / (countSquare - 1.0) * (s1 / weightSquare);
      let secondTerm = count / (countSquare - 1.0) * (s2 / weightSquare);
      let thirdTerm = 3.0 / (countSquare - 1.0);
      variance = firstTerm - secondTerm + thirdTerm - expected * expected;
      magnitude = firstTerm + secondTerm + thirdTerm + expected * expected;
    } else if (count >= 4.0) {
      let kurtosis = count * fourthTotal[fourthTotalOffset] / (sumOfSquares * sumOfSquares);
      let denominator = (count - 1.0) * (count - 2.0) * (count - 3.0);
      let firstTerm = count * ((countSquare - 3.0 * count + 3.0) * (s1 / weightSquare) -
        count * (s2 / weightSquare) + 3.0);
      let secondTerm = kurtosis * ((countSquare - count) * (s1 / weightSquare) -
        2.0 * count * (s2 / weightSquare) + 6.0);
      variance = (firstTerm - secondTerm) / denominator - expected * expected;
      magnitude = (abs(firstTerm) + abs(secondTerm)) / denominator + expected * expected;
    }
    if (isFiniteFloat(variance) && variance > VARIANCE_CANCELLATION * magnitude) {
      zScore = (moransI - expected) / sqrt(variance);
    } else {
      variance = nan;
    }
  }
  let base = index * ${ut}u;
  bandResults[bandResultsOffset + base] = moransI;
  bandResults[bandResultsOffset + base + 1u] = expected;
  bandResults[bandResultsOffset + base + 2u] = variance;
  bandResults[bandResultsOffset + base + 3u] = zScore;`}));let P=[{name:`bandResults`,view:j,type:`f32`,access:`read`},{name:`moransI`,view:h.moransI,type:`f32`,access:`read_write`}],I=[`moransI[moransIOffset + index] = bandResults[bandResultsOffset + base];`];for(let[e,t,n]of[[`expectedI`,h.expectedI,1],[`varianceI`,h.varianceI,2],[`zScores`,h.zScores,3]])t&&(P.push({name:e,view:t,type:`f32`,access:`read_write`}),I.push(`${e}[${e}Offset + index] = bandResults[bandResultsOffset + base + ${n}u];`));return h.pValues&&(P.push({name:`pValues`,view:h.pValues,type:`f32`,access:`read_write`}),I.push(`let zScore = bandResults[bandResultsOffset + base + 3u];
  pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(zScore), isFiniteFloat(zScore));`)),h.pairCounts&&(P.push({name:`bandCounts`,view:N,type:`u32`,access:`read`},{name:`pairCounts`,view:h.pairCounts,type:`u32`,access:`read_write`}),I.push(`pairCounts[pairCountsOffset + index] = bandCounts[bandCountsOffset + index];`)),C.push(F(e,{id:`${t}-publish`,operation:X,variant:`publish`,bindings:P,invocationCount:m,declarations:`${r}
${ft}`,body:`let base = index * ${ut}u;
  ${I.join(`
  `)}`})),h.status?.overflow&&C.push(F(e,{id:`${t}-overflow`,operation:X,variant:`overflow`,bindings:[{name:`accumulators`,view:A.accumulators,type:`u32`,access:`read`},{name:`bandCounts`,view:N,type:`u32`,access:`read`},{name:`overflow`,view:h.status?.overflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:u(`accumulators`,ct,``,m),body:`var flagged = readPairOverflow();
  for (var band = 0u; band < ${m}u; band++) {
    flagged = flagged || bandCounts[bandCountsOffset + band] == 0xffffffffu;
  }
  ${_?`for (var channelIndex = 0u; channelIndex < 4u; channelIndex++) {
    let channel = select(channelIndex + 4u, channelIndex + 1u, channelIndex < 2u);
    var low = 0u;
    var high = 0u;
    for (var band = 0u; band < ${m}u; band++) {
      let word = (band * ${ct}u + channel) * 2u;
      let previousLow = low;
      low += accumulators[accumulatorsOffset + word];
      let withLow = high + select(0u, 1u, low < previousLow);
      flagged = flagged || withLow < high;
      let nextHigh = withLow + accumulators[accumulatorsOffset + word + 1u];
      flagged = flagged || nextHigh < withLow;
      high = nextHigh;
    }
  }`:``}
  overflow[overflowOffset] = select(0u, 1u, flagged);`})),h.peakBands&&C.push(F(e,{id:`${t}-peaks`,operation:X,variant:`peaks`,bindings:[{name:`bandResults`,view:j,type:`f32`,access:`read`},{name:`peakBands`,view:h.peakBands,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${r}
const BAND_COUNT: u32 = ${m}u;
const NO_BAND: u32 = ${rt}u;

fn readZ(band: u32) -> f32 {
  return bandResults[bandResultsOffset + band * ${ut}u + 3u];
}`,body:`var firstPeak = NO_BAND;
  var maximumBand = NO_BAND;
  var maximumZ = 0.0;
  for (var band = 0u; band < BAND_COUNT; band++) {
    let z = readZ(band);
    if (!isFiniteFloat(z)) {
      continue;
    }
    if (maximumBand == NO_BAND || z > maximumZ) {
      maximumBand = band;
      maximumZ = z;
    }
    if (firstPeak == NO_BAND && band > 0u && band + 1u < BAND_COUNT) {
      let previous = readZ(band - 1u);
      let next = readZ(band + 1u);
      if (isFiniteFloat(previous) && isFiniteFloat(next) && z > previous && z > next) {
        firstPeak = band;
      }
    }
  }
  peakBands[peakBandsOffset] = firstPeak;
  peakBands[peakBandsOffset + 1u] = maximumBand;`})),h.statistics&&C.push(F(e,{id:`${t}-statistics`,operation:X,variant:`statistics`,bindings:[{name:`statistics`,view:S,type:`f32`,access:`read`},{name:`fourthTotal`,view:k,type:`f32`,access:`read`},{name:`statisticsOut`,view:h.statistics,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const MOMENTS: u32 = ${g}u;`,body:`let count = statistics[statisticsOffset + MOMENTS];
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 2u];
  statisticsOut[statisticsOutOffset] = count;
  statisticsOut[statisticsOutOffset + 1u] = statistics[statisticsOffset + MOMENTS + 1u];
  statisticsOut[statisticsOutOffset + 2u] = sumOfSquares / count;
  statisticsOut[statisticsOutOffset + 3u] = sumOfSquares;
  statisticsOut[statisticsOutOffset + 4u] = count * fourthTotal[fourthTotalOffset] / (sumOfSquares * sumOfSquares);`})),C}},ft=`
fn getTwoSidedPValue(z: f32) -> f32 {
  let x = abs(z) * 0.70710678118654752;
  let t = 1.0 / (1.0 + 0.5 * x);
  let exponent = -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))));
  return min(t * exp(exponent), 1.0);
}
`,pt=4294967295;function mt(e,t,n,r,i){let a=Math.sqrt(i),o=[];for(let t=0;t<e.length;t++)Number.isFinite(e[t])&&n[t+1]>n[t]&&o.push(t);let s=new Float32Array(o.length),c=new Float32Array(o.length),l=new Uint8Array(o.length),u=[0,0,0,0],d=0,f=0;return o.forEach((n,i)=>{let o=(e[n]-r)/a,p=(t[n]-r)/a;s[i]=o,c[i]=p;let m=e[n]>=r,h=t[n]>=r,g=m?h?0:3:h?1:2;l[i]=g,u[g]++,d+=o*p,f+=o*o}),{rows:Uint32Array.from(o),x:s,y:c,quadrant:l,counts:u,slope:f>0?d/f:NaN}}function ht(e,t){let n=[];for(let t=0;t<e.length;t++)Number.isFinite(e[t])&&n.push(t);let r=n.map(t=>e[t]),i=se(t*2654435761);for(let e=r.length-1;e>0;e--){let t=Math.floor(i()*(e+1));[r[e],r[t]]=[r[t],r[e]]}let a=Float32Array.from(e);return n.forEach((e,t)=>{a[e]=r[t]}),a}function gt(e){return U(e,3)}function _t(e,t,n,r,i){let a=new Uint32Array(e.length);for(let o=0;o<e.length;o++){if(!Number.isFinite(e[o])||!Number.isFinite(t[o])){a[o]=pt;continue}let s=Math.min(2,he(e[o],n)),c=Math.min(2,he(t[o],r));a[o]=(i?2-c:c)*3+s}return a}function vt(e,t,n,r,i=48){let a=n-t,o=Array.from(e);if(!(a>0)||!Number.isFinite(r))return{values:o,domain:[t,n]};let s=.04*Math.max(a,Math.abs(r-t),Math.abs(r-n)),c=Math.min(t,r-s),l=Math.max(n,r+s);if(c===t&&l===n)return{values:o,domain:[t,n]};let u=a/o.length,d=Array(i).fill(0);return o.forEach((e,n)=>{let r=t+(n+.5)*u,a=Math.min(i-1,Math.max(0,Math.floor((r-c)/(l-c)*i)));d[a]+=e}),{values:d,domain:[c,l]}}function yt(e,t){let n=0,r=e.length;for(;n<r;){let i=n+r>>1;e[i]<=t?n=i+1:r=i}return e.length?n/e.length:NaN}function bt(e){return Number.isFinite(e)?String(Number(e.toPrecision(2))):`n/a`}var xt=999,St=24,Z=32,Ct=50,Q=R.length,wt=tt.length,Tt=300,Et={"us-counties":[-87.65,41.84],"chicago-tracts":[-87.63,41.88]},Dt={"us-counties":[-124.8,24.4,-66.9,49.4],"chicago-tracts":[-87.94,41.644,-87.524,42.023]},$=[`queen`,`rook`,`knn`,`band`],Ot=[`High-High`,`Low-High`,`Low-Low`,`High-Low`],kt=`
  let included = mask[maskOffset + index] != 0u;
  let value = x[xOffset + index];
  let lagged = lag[lagOffset + index];
  if (!included) {
    quadrant[quadrantOffset + index] = 0u;
    joinClass[joinClassOffset + index] = 0xffffffffu;
    var nanBits = 0x7fc00000u;
    binary[binaryOffset + index] = bitcast<f32>(nanBits);
    return;
  }
  // Moran scatterplot quadrant of (x - mean, lag - mean): 1 HH, 2 LH, 3 LL, 4 HL.
  let mean = summary[summaryOffset + ${L.mean}u];
  let high = value >= mean;
  let neighborsHigh = lagged >= mean;
  var quadrantValue = 3u;
  if (high && neighborsHigh) { quadrantValue = 1u; }
  else if (high) { quadrantValue = 4u; }
  else if (neighborsHigh) { quadrantValue = 2u; }
  quadrant[quadrantOffset + index] = quadrantValue;
  let black = value > threshold[thresholdOffset];
  joinClass[joinClassOffset + index] = select(0u, 1u, black);
  binary[binaryOffset + index] = select(0.0, 1.0, black);`,At=`
  var isolated = true;
  let begin = min(rowOffsets[rowOffsetsOffset + index], ROW_SLOTS);
  let end = min(rowOffsets[rowOffsetsOffset + index + 1u], ROW_SLOTS);
  for (var slot = begin; slot < end; slot++) {
    let neighbor = rowNeighbors[rowNeighborsOffset + slot];
    if (neighbor < ROW_COUNT && neighbor != index && rowWeights[rowWeightsOffset + slot] > 0.0 &&
        mask[maskOffset + neighbor] != 0u) {
      isolated = false;
      break;
    }
  }
  if (isolated || mask[maskOffset + index] == 0u) {
    quadrant[quadrantOffset + index] = 0u;
    var nanBits = 0x7fc00000u;
    lag[lagOffset + index] = bitcast<f32>(nanBits);
  }`,jt=new Map;async function Mt(e){let{device:t}=e,n=!1,r=null,i=0,a=new le,o=0,s=!0,c=!0,l=!0,u=null,d=t=>{let n=jt.get(t);return n||(n=Re(t,e.datasets,e.signal),n.catch(()=>jt.delete(t)),jt.set(t,n)),n},f=()=>{let t=e.options;return{source:t.source,k:t.k,snapTolerance:0,bandFactor:t.bandFactor,knnCapFactor:0,weightKind:`binary`,kernel:`triangular`,power:1,distanceFloor:0,rowStandardize:!1,transform:t.transform,transformKernel:`bisquare`,bandwidthFactor:0,doubleSum:`one`}},p=e=>({...f(),source:e,transform:`row`}),m=e=>e.getStats(Fe(f())),h=t=>e.options.maxDistanceFactor*t.medianSpacing,g=async n=>{let r=new A(t,`global-${n.id}`),i=n.count,a=(e,t)=>r.createBuffer(e,t),o=Ie({device:t,resources:r,id:`global-core`,geography:n}),s=e.options,c={globalResults:a(`global-results`,Q*4),bivariateResults:a(`bivariate-results`,Q*4),joinResults:a(`join-results`,Q*4),shuffledResults:a(`shuffled-results`,Q*4),joinCounts:a(`join-counts`,16),permutationResults:a(`permutation-results`,wt*4),histogram:a(`permutation-histogram`,St*4),moransI:a(`correlogram-i`,Z*4),zScores:a(`correlogram-z`,Z*4),pValues:a(`correlogram-p`,Z*4),expectedI:a(`correlogram-expected`,Z*4),pairCounts:a(`correlogram-pairs`,Z*4),peaks:a(`correlogram-peaks`,8),correlogramOverflow:a(`correlogram-overflow`,4),correlogramStatistics:a(`correlogram-statistics`,20),binary:a(`binary-values`,i*4)},l=a(`x`,Float32Array.from(n.getVariable(s.variable))),u=a(`y`,Float32Array.from(n.getVariable(s.secondVariable))),d=a(`mask-x`,new Uint32Array(i)),p=a(`mask-y`,new Uint32Array(i)),m=a(`mask-xy`,new Uint32Array(i)),h=a(`join-threshold`,new Float32Array(1)),g=a(`lag`,i*4),_=a(`quadrant`,i*4),v=a(`join-class`,i*4),y=a(`shuffled-x`,i*4),b=a(`bivariate-class`,i*4),x=r.createParameterBuffer(`permutation-parameters`,`uint32`,4),S=r.createParameterBuffer(`correlogram-parameters`,`float32`,8),C=Math.max(0,n.pick(...Et[n.id])),w=n.id===`us-counties`?e.datasets.get(`us-states`).geojson:null,E=w?Ee(we(w).flatMap(({polygon:e})=>e),De(n.origin)):null,D=new Map,O=async e=>{let a=new T(t,{id:`global-statistics-${n.id}-${e?`row`:`raw`}`}),s={spatialContext:n.analysisFrame.spatialContext,offsets:H(a,`offsets`,o.csr.offsets,`uint32`,i+1),neighbors:H(a,`neighbors`,o.csr.neighbors,`uint32`,o.slots),weights:H(a,`weights`,o.csr.weights,`float32`,o.slots),information:Pe(i,e)},f=H(a,`x`,l,`float32`,i),p=H(a,`y`,u,`float32`,i),b=H(a,`mask-x`,d,`uint32`,i),x=H(a,`mask-xy`,m,`uint32`,i),S=H(a,`global-results`,c.globalResults,`float32`,Q);a.add(new z({id:`global-moran-geary-g`,weights:s,values:f,mask:b,statistics:[`moran`,`geary`,`getisOrdG`],output:{summary:S}})),a.add(new z({id:`global-bivariate`,weights:s,values:f,secondValues:p,mask:x,statistics:[`bivariateMoran`],output:{summary:H(a,`bivariate-results`,c.bivariateResults,`float32`,Q)}})),a.add(new z({id:`global-shuffled`,weights:s,values:H(a,`shuffled-x`,y,`float32`,i),mask:b,statistics:[`moran`],output:{summary:H(a,`shuffled-results`,c.shuffledResults,`float32`,Q)}}));let C=H(a,`lag`,g,`float32`,i);a.add(new te({id:`spatial-lag`,values:f,mask:b,weights:s,normalize:!0,output:C}));let w=H(a,`binary`,c.binary,`float32`,i),E=H(a,`quadrant`,_,`uint32`,i);return de(a,{id:`global-classes`,invocationCount:i,bindings:[{name:`x`,view:f,type:`f32`,access:`read`},{name:`lag`,view:C,type:`f32`,access:`read`},{name:`mask`,view:b,type:`u32`,access:`read`},{name:`summary`,view:S,type:`f32`,access:`read`},{name:`threshold`,view:H(a,`threshold`,h,`float32`,1),type:`f32`,access:`read`},{name:`quadrant`,view:E,type:`u32`,access:`read_write`},{name:`joinClass`,view:H(a,`join-class`,v,`uint32`,i),type:`u32`,access:`read_write`},{name:`binary`,view:w,type:`f32`,access:`read_write`}],body:kt}),de(a,{id:`global-islands`,invocationCount:i,bindings:[{name:`rowOffsets`,view:s.offsets,type:`u32`,access:`read`},{name:`rowNeighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`rowWeights`,view:s.weights,type:`f32`,access:`read`},{name:`mask`,view:b,type:`u32`,access:`read`},{name:`quadrant`,view:E,type:`u32`,access:`read_write`},{name:`lag`,view:C,type:`f32`,access:`read_write`}],declarations:`const ROW_COUNT: u32 = ${i}u;
  const ROW_SLOTS: u32 = ${o.slots}u;`,body:At}),a.add(new z({id:`global-join-count`,weights:s,values:w,mask:b,statistics:[`joinCount`],output:{summary:H(a,`join-results`,c.joinResults,`float32`,Q),joinCounts:H(a,`join-counts`,c.joinCounts,`uint32`,3)}})),r.track(await a.compileAsync())},k=e=>D.get(e),j=async e=>{let t=D.get(e);return t||(t=await O(e),D.set(e,t)),t},M=await j(Fe(f())),N=new ue(r,`global-${n.id}`,[{buffer:c.globalResults,size:Q*4},{buffer:c.bivariateResults,size:Q*4},{buffer:c.joinResults,size:Q*4},{buffer:c.joinCounts,size:16},{buffer:c.permutationResults,size:wt*4},{buffer:c.histogram,size:St*4},{buffer:c.moransI,size:Z*4},{buffer:c.zScores,size:Z*4},{buffer:c.pValues,size:Z*4},{buffer:c.pairCounts,size:Z*4},{buffer:c.peaks,size:8},{buffer:c.correlogramStatistics,size:20},{buffer:c.shuffledResults,size:Q*4},{buffer:g,size:i*4},{buffer:o.csr.offsets,size:(i+1)*4},{buffer:c.correlogramOverflow,size:4}],e=>Ae(`global snapshot`,()=>q(e)));return{geography:n,projection:n.analysisFrame,resources:r,core:o,buffers:ze(r,n,`global`),stateSegments:E,stateBuffer:E?.length?a(`state-lines`,E):null,x:l,y:u,maskX:d,maskY:p,maskXY:m,threshold:h,lag:g,quadrant:_,joinClass:v,shuffled:y,bivariateClass:b,bivariateClasses:new Uint32Array(i).fill(pt),focusRow:C,statsCompiled:M,getStats:k,prepareStats:j,statsGraphs:()=>[...D.values()],permutation:new Map,correlogram:new Map,permutationParameters:x,correlogramParameters:S,buffersByName:c,reader:N,snapshot:null,classes:new Map,sortedX:new Float32Array,bivariateCounts:Array(9).fill(0),sweep:null}},_=(e,t)=>{let n=e.classes.get(t);if(!n){let r=e.geography.getVariable(t);n={breaks:_e(r,5,`quantile`),extent:me(r)},e.classes.set(t,n)}return n},y=(t,n)=>{let r=_(t,n);return Ue(Le(n),t.geography.id,r.breaks,r.extent,e.ground())},b=()=>{let t=r;if(!t)return;let n=e.ground(),i=e.options,a=t.geography.getVariable(i.variable),o=t.geography.getVariable(i.secondVariable),s=y(t,i.variable),c=y(t,i.secondVariable);e.setLegendData(`valueTable`,s),e.setLegendData(`secondTable`,c),e.setLegendData(`valueCounts`,ye(a,s.breaks)),e.setLegendData(`secondCounts`,ye(o,c.breaks)),e.setLegendData(`bivariateColors`,Be(n)),e.setLegendData(`quadrantColors`,Ge(n)),e.setLegendData(`noDataColor`,be[n])},x=t=>{let n=t.geography.getVariable(e.options.variable);t.shuffled.write(ht(n,e.options.seed))},S=t=>{let n=e.options,r=t.geography.getVariable(n.variable),i=t.geography.getVariable(n.secondVariable);t.x.write(Float32Array.from(r)),t.y.write(Float32Array.from(i));let a=e=>Number.isFinite(e);t.maskX.write(Uint32Array.from(r,e=>a(e)?1:0)),t.maskY.write(Uint32Array.from(i,e=>a(e)?1:0)),t.maskXY.write(Uint32Array.from(r,(e,t)=>a(e)&&a(i[t])?1:0)),t.threshold.write(Float32Array.of(je(r,n.joinPercentile)));let o=_t(r,i,gt(r),gt(i),n.secondVariable===`income`);t.bivariateClass.write(o),t.bivariateClasses=o,t.bivariateCounts=Array(9).fill(0);for(let e of o)e!==4294967295&&t.bivariateCounts[e]++;t.sortedX=Float32Array.from(r.filter(e=>a(e))).sort(),x(t),b()},C=()=>{let t=r;if(!t)return;let n=e.options;t.core.writeParameters(f()),t.permutationParameters.write(v({seed:n.seed,permutationCount:n.permutations}));let i=t.geography.bounds;t.correlogramParameters.write(it({bounds:[i[0]-1e3,i[1]-1e3,i[2]+1e3,i[3]+1e3],maximumDistance:h(t.geography),varianceAssumption:n.varianceAssumption}))},w=()=>{let t=e.options;return t.statistic===`joinCount`?null:`${t.statistic}-${t.alternative}-${Fe(f())?`row`:`raw`}`},E=async n=>{let r=w();if(!r||n.permutation.has(r))return!1;let i=e.options,a=i.statistic,o=n.geography.count,s=new T(t,{id:`global-permutation-${r}`}),c=a===`bivariateMoran`;return s.add(new nt({id:`global-permutation`,weights:{spatialContext:n.geography.analysisFrame.spatialContext,offsets:H(s,`offsets`,n.core.csr.offsets,`uint32`,o+1),neighbors:H(s,`neighbors`,n.core.csr.neighbors,`uint32`,n.core.slots),weights:H(s,`weights`,n.core.csr.weights,`float32`,n.core.slots),information:Pe(o,Fe(f()))},values:H(s,`x`,n.x,`float32`,o),secondValues:c?H(s,`y`,n.y,`float32`,o):void 0,mask:H(s,`mask`,c?n.maskXY:n.maskX,`uint32`,o),statistic:a,alternative:i.alternative,parameters:n.permutationParameters.importToGraph(s),maximumPermutations:xt,output:{summary:H(s,`results`,n.buffersByName.permutationResults,`float32`,wt),histogram:H(s,`histogram`,n.buffersByName.histogram,`uint32`,St)}})),n.permutation.set(r,n.resources.track(await s.compileAsync())),!0},D=()=>`${e.options.bandCount}-${e.options.bandMode}`,O=async n=>{let r=D();if(n.correlogram.has(r))return!1;let i=e.options,a=n.geography.count,o=new T(t,{id:`global-correlogram-${r}`}),s=(e,t,n,r)=>H(o,e,t,n,r);return o.add(new dt({spatialContext:n.geography.analysisFrame.spatialContext,id:`correlogram`,positions:H(o,`positions`,n.core.positions,`float32x2`,a),values:s(`x`,n.x,`float32`,a),mask:s(`mask`,n.maskX,`uint32`,a),parameters:n.correlogramParameters.importToGraph(o),gridSize:[128,128],bandCount:i.bandCount,bandMode:i.bandMode,output:{moransI:s(`moransI`,n.buffersByName.moransI,`float32`,i.bandCount),zScores:s(`zScores`,n.buffersByName.zScores,`float32`,i.bandCount),pValues:s(`pValues`,n.buffersByName.pValues,`float32`,i.bandCount),expectedI:s(`expectedI`,n.buffersByName.expectedI,`float32`,i.bandCount),pairCounts:s(`pairCounts`,n.buffersByName.pairCounts,`uint32`,i.bandCount),status:{overflow:s(`overflow`,n.buffersByName.correlogramOverflow,`uint32`,1)},peakBands:s(`peaks`,n.buffersByName.peaks,`uint32`,2),statistics:s(`statistics`,n.buffersByName.correlogramStatistics,`float32`,5)}})),n.correlogram.set(r,n.resources.track(await o.compileAsync())),!0},k=async e=>{if(e.sweep)return e.sweep;let{resources:n,geography:r}=e,i=r.count,a=Ie({device:t,resources:n,id:`sweep-core`,geography:r}),o=n.createBuffer(`sweep-results`,Q*4),s=new T(t,{id:`sweep-statistics-${r.id}`});s.add(new z({id:`sweep-moran`,weights:{spatialContext:e.geography.analysisFrame.spatialContext,offsets:H(s,`offsets`,a.csr.offsets,`uint32`,i+1),neighbors:H(s,`neighbors`,a.csr.neighbors,`uint32`,a.slots),weights:H(s,`weights`,a.csr.weights,`float32`,a.slots),information:Pe(i,!0)},values:H(s,`x`,e.x,`float32`,i),mask:H(s,`mask`,e.maskX,`uint32`,i),statistics:[`moran`],output:{summary:H(s,`results`,o,`float32`,Q)}}));let c={core:a,graph:n.track(await s.compileAsync()),reader:new ue(n,`sweep-${r.id}`,[{buffer:o,size:Q*4}],e=>Ae(`sweep result`,()=>F(e))),values:$.map(()=>NaN),index:0,state:`idle`,encoded:!1};return e.sweep=c,c},j=async e=>{let t=await k(e);for(let e of $)await t.core.prepare(p(e))},M=t=>{let n=t.sweep;!e.options.showRuleSweep||!n||(n.values=$.map(()=>NaN),n.index=0,n.state=`running`,n.encoded=!1,e.setChart(`iByW`,null),e.setReadout(`iRange`,null))},N=()=>{let t=e.options;return[`Queen`,`Rook`,`${t.k} nearest`,`Band, ${t.bandFactor} x spacing`]},P=t=>{let n=e.options,r=t.values.filter(e=>Number.isFinite(e));r.length<$.length||(e.setChart(`iByW`,{kind:`bars`,horizontal:!0,title:`Moran's I under each neighbour rule`,values:t.values,labels:N(),highlight:[$.indexOf(n.source)],formatX:e=>e.toFixed(2),description:`Horizontal bars of row-standardised Moran's I under queen, rook, k nearest and distance-band neighbours; the rule currently chosen is highlighted.`,table:!1}),e.setReadout(`iRange`,`${G(Math.min(...r),2)} to ${G(Math.max(...r),2)}`))},F=t=>{let n=r,i=n?.sweep;if(!n||!i||i.state!==`running`)return;let a=new Float32Array(t,0,Q);i.values[i.index]=a[R.moran+re.statistic],i.index++,i.encoded=!1,i.index>=$.length&&(i.state=`done`,P(i),he()),e.requestLayers()},I=(e,t)=>e.geography.getCentroidLngLat(t),ee=(e,t)=>{let{geography:n,projection:r}=e,i=[];for(let e=n.featureRingOffsets[t];e<n.featureRingOffsets[t+1];e++){let t=[];for(let i=n.contiguityRingOffsets[e];i<n.contiguityRingOffsets[e+1];i++)t.push(r.unproject(n.analysisContiguityVertices[i*2],n.analysisContiguityVertices[i*2+1]));i.push(t)}return i},ne=e=>e.peaks[0]===4294967295?e.peaks[1]===4294967295?{band:-1,isFirstPeak:!1}:{band:e.peaks[1],isFirstPeak:!1}:{band:e.peaks[0],isFirstPeak:!0},ae=()=>{let t=r,n=t?.snapshot;if(!t||!n){e.setChart(`moranScatter`,null);return}let i=e.options,{scatter:a}=n,o=R,s=re,c=n.global[o.moran+s.statistic],l=i.transform===`row`,u=1;for(let e=0;e<a.x.length;e++)u=Math.max(u,Math.abs(a.x[e]),Math.abs(a.y[e]));let d=Math.ceil(u);e.setChart(`moranScatter`,{kind:`scatter`,title:`Moran scatterplot`,x:a.x,y:a.y,colorIndex:a.quadrant,palette:Ve(e.ground()),xLabel:`${Le(i.variable).label} (standard score)`,yLabel:`Neighbour average (standard score)`,xDomain:[-d,d],yDomain:[-d,d],radius:1.5,quadrants:{x:0,y:0,labels:Ot},fit:{slope:l?c:a.slope,intercept:0,label:l?`slope = I = ${G(c,2)}`:`slope = ${G(a.slope,2)} (lag is row standardised)`},onBrush:t=>{let n=r;if(!n||!t?.length){e.setHighlight(null);return}let i=Math.max(1,Math.ceil(t.length/Tt)),o=[];for(let e=0;e<t.length;e+=i)o.push({kind:`polygon`,rings:ee(n,a.rows[t[e]])});e.setHighlight(o)},description:`Scatterplot of every county: its standardised value across, the standardised average of its neighbours up. The cloud rises from lower left to upper right; the fitted line has slope equal to Moran I when the weights are row standardised.`,table:!1})},oe=()=>{let t=r,n=t?.snapshot,i=e.options;if(!t||!n||i.statistic===`joinCount`){e.setChart(`nullDistribution`,null);return}let a=tt,o=n.permutation,s=o[a.observed],{values:c,domain:l}=vt(n.histogram,o[a.minimum],o[a.maximum],s),u={moran:`Moran's I`,geary:`Geary's C`,getisOrdG:`General G`,bivariateMoran:`Bivariate Moran's I`,joinCount:`Join counts`};e.setChart(`nullDistribution`,{kind:`histogram`,title:`${u[i.statistic]} of shuffled maps`,values:c,xDomain:l,xLabel:u[i.statistic],yLabel:`Shuffles`,formatX:e=>e.toFixed(3),now:s,nowLabel:`observed`,description:`Histogram of ${u[i.statistic]} over the shuffles of the county values, with the observed value marked at ${G(s,3)}.`,table:!1})},se=()=>{let t=r,n=t?.snapshot;if(!t||!n){e.setChart(`correlogram`,null);return}let i=e.options.bandCount,a=h(t.geography),o=Array.from({length:i},(e,t)=>(t+1)*a/i).map(e=>e/1e3),s=Array.from(n.zScores.subarray(0,i)),c=s.filter(e=>Number.isFinite(e)),l=ne(n);e.setChart(`correlogram`,{kind:`line`,title:`Moran's I by distance`,xLabel:`Distance (km at the focus county)`,yLabel:`Moran's I`,series:[{label:`Moran's I`,x:o,y:Array.from(n.moransI.subarray(0,i))},{label:`z-score`,x:o,y:s,axis:`y2`,dashed:!0,color:1}],y2Domain:c.length?[Math.min(0,...c),Math.max(...c)*1.05]:[0,1],y2Label:`z-score`,markers:l.band>=0?[{x:o[l.band],label:l.isFirstPeak?`First z peak`:`Strongest tested band`}]:[],formatX:e=>e<10?e.toFixed(1):e.toFixed(0),description:`Line chart of global Moran's I for neighbours within each distance, falling as the distance grows, with the z-score on a second axis and a marker at the first local z peak, or the strongest tested band if no local peak is detected.`,table:!1})},V=()=>{let t=r;if(!t)return;let n=e.options,i=t.geography,a=i.id===`us-counties`,o=a?`${W(i.count)} counties of the contiguous US (Alaska and Hawaii not in the data)`:`${W(i.count)} census tracts, ACS 2018-2022 via CDC SVI 2022`,s=I(t,t.focusRow)[1],c=t.snapshot,l=c?ne(c):{band:-1,isFirstPeak:!1},u=n.showBands&&c&&l.band>=0?[(l.band+1)*h(i)/n.bandCount]:void 0;e.setFurniture({title:{sample:o},scaleBar:n.showBands?{latitude:s,minZoom:a?3.5:void 0,ticks:u}:a?{latitude:37,minZoom:3.5}:{units:`metric`},caveat:a?Te():``})},U=()=>{let t=r,n=t?.snapshot,i=e.options;if(!t)return;let a=t.geography.id===`us-counties`;if(e.setAnnotations(`geography`,a?null:pe(fe,[`lake-michigan`,`loop`],{loop:{minZoom:9}})),a&&i.display===`value`&&!i.showShuffled&&!i.showBands){let n=Le(i.variable),r=t.geography.getVariable(i.variable),a=-1,o=-1;for(let e=0;e<r.length;e++)Number.isFinite(r[e])&&((a<0||r[e]>r[a])&&(a=e),(o<0||r[e]<r[o])&&(o=e));let s=(e,i,a)=>({kind:`note`,id:`extreme-${i}`,coordinate:I(t,e),title:`${He(n,r[e])} ${n.unit}`,text:`${i}: ${t.geography.getName(e)}, ${t.geography.getGroupName(e)}`,tone:a,priority:5});e.setAnnotations(`extremes`,a>=0&&o>=0?[s(a,`Highest`,`accent`),s(o,`Lowest`,`ink`)]:null)}else e.setAnnotations(`extremes`,null);if(n&&i.showBands){let r=I(t,t.focusRow),a=h(t.geography),o=ne(n),s=[{kind:`point`,id:`focus`,coordinate:r,text:t.geography.getName(t.focusRow),marker:`ring`,tone:`signal`,rank:`subject`,priority:9},{kind:`ring`,id:`band-longest`,coordinate:r,radiusMeters:a,text:Ce(a),dashed:!0,geodesic:!0}];if(o.band>=0){let e=(o.band+1)*h(t.geography)/i.bandCount;s.push({kind:`ring`,id:`band-peak`,coordinate:r,radiusMeters:e,text:`${o.isFirstPeak?`z peak`:`strongest`} ${Ce(e)}`,tone:`accent`,geodesic:!0})}e.setAnnotations(`rings`,s)}else e.setAnnotations(`rings`,null)},he=()=>{let t=r;if(!t)return;let n=t.snapshot?.global[R.summary+L.count],i=t.sweep;e.setCost({records:n??t.geography.count,passes:(m(t)??t.statsCompiled).stats.nodeOrder.length,note:i&&i.state===`done`?`${i.core.getGraphs().length} weights graphs compiled once, reused for 4 rules`:void 0})},K=()=>{let t=r,n=t?.snapshot;if(!t||!n)return;let i=e.options,a=R,o=re,s=L,c=ie,l=tt,u=n.global,d=(e,t)=>({statistic:t[e+o.statistic],expected:t[e+o.expected],zRand:t[e+o.zRandomization],pRand:t[e+o.pRandomization]});e.setReadout(`places`,W(u[a.summary+s.count]));let f=d(a.moran,u);e.setReadout(`moran`,G(f.statistic,2)),e.setReadout(`moranZ`,G(f.zRand,1)),e.setReadout(`expected`,G(f.expected,4));let{counts:p,rows:m}={counts:n.scatter.counts,rows:n.scatter.rows.length};e.setReadout(`quadrantShare`,m?xe((p[0]+p[2])/m):null);let g=d(a.geary,u);e.setReadout(`geary`,`C = ${G(g.statistic,3)} (E = 1), z ${G(g.zRand,1)}, p ${Me(g.pRand)}`);let _=d(a.getisOrdG,u);e.setReadout(`getis`,`G = ${G(_.statistic,5)} (E = ${G(_.expected,5)}), z ${G(_.zRand,1)}, p ${Me(_.pRand)}`);let v=d(a.bivariateMoran,n.bivariate);e.setReadout(`bivariate`,G(v.statistic,2)),e.setReadout(`darkCorner`,W(t.bivariateCounts[8]));let y=n.join;e.setReadout(`joins`,`BB ${W(n.joinCounts[0]/2)} (E ${G(y[a.joinCount+c.expectedBlackBlack],0)}), BW ${W(n.joinCounts[1]/2)} (E ${G(y[a.joinCount+c.expectedBlackWhite],0)}), WW ${W(n.joinCounts[2]/2)}`),e.setReadout(`joinTest`,`BB z ${G(y[a.joinCount+c.zBlackBlack],1)} p ${Me(y[a.joinCount+c.pBlackBlack])}; BW z ${G(y[a.joinCount+c.zBlackWhite],1)} p ${Me(y[a.joinCount+c.pBlackWhite])}`);let b=n.permutation;if(i.statistic===`joinCount`)for(let t of[`pSim`,`pFloor`,`permutations`,`exceedances`,`zSim`])e.setReadout(t,`no permutation test for join counts`);else{let t=b[l.permutationCount];e.setReadout(`pSim`,bt(b[l.pseudoPValue])),e.setReadout(`pFloor`,bt(Ne(i.alternative,t))),e.setReadout(`permutations`,W(t)),e.setReadout(`exceedances`,W(b[l.exceedances])),e.setReadout(`zSim`,G(b[l.zSimulated],1))}e.setReadout(`shuffledI`,G(n.shuffled[a.moran+o.statistic],2));let x=h(t.geography),S=ne(n);e.setReadout(`peak`,S.band>=0?`${S.isFirstPeak?`first local z peak`:`strongest tested band (no local peak)`} at ${Ce((S.band+1)*x/i.bandCount)}`:`no defined band`),e.setReadout(`maxDistance`,Ce(x)),e.setReadout(`bandSpan`,`${Ce(x)} in ${i.bandCount} ${i.bandMode} bands${S.band>=0&&!S.isFirstPeak?`; no z peak, strongest band shown`:``}${n.correlogramOverflowed?`; pair counts saturated (overflow), treat with care`:``}`),e.setReadout(`distanceNote`,`metres of ${t.geography.analysisFrame.name}, within ${(Math.ceil(t.geography.analysisFrame.maxScaleError*1e6)/1e4).toFixed(4)}% of ground distance`),ae(),oe(),se(),V(),U(),he()},q=t=>{let n=r;if(!n)return;let i=n.geography.count,a=0,o=e=>{let n=new Float32Array(t,a,e);return a+=e*4,n},s=e=>{let n=new Uint32Array(t,a,e);return a+=e*4,n},c=o(Q),l=o(Q),u=o(Q),d=s(4),f=o(wt),p=s(St),m=o(Z),h=o(Z),g=o(Z),v=s(Z),y=s(2),b=o(5),x=o(Q),S=o(i),C=s(i+1),w=s(1)[0]!==0,T=L,E=R,D=e.options,O=mt(n.geography.getVariable(D.variable),S,C,c[E.summary+T.mean],c[E.summary+T.variance]);n.snapshot={global:c,bivariate:l,join:u,joinCounts:d,permutation:f,histogram:p,moransI:m,zScores:h,pValues:g,pairCounts:v,peaks:y,correlogramOverflowed:w,statistics:b,shuffled:x,lag:S,offsets:C,scatter:O};let k=c[E.summary+T.islandCount];e.setLegendData(`quadrantCounts`,[...O.counts,k]),e.setLegendData(`lagCounts`,ye(Array.from(O.rows,e=>S[e]),_(n,D.variable).breaks)),K(),e.requestLayers()},Ye=async t=>{let a=++i;e.setStatus(`Loading...`);let f=await d(t);if(a!==i||n)return;let p=r;r=null,e.requestLayers(),p&&(p.reader.stop(),p.sweep?.reader.stop(),setTimeout(()=>p.resources.destroy(),200));let m=await g(f),h;do h=o,await J(m);while(h!==o&&a===i&&!n);if(a!==i||n){m.resources.destroy();return}r=m,u=null,S(m),C(),M(m),s=!0,c=!0,l=!0,V(),U(),e.setStatus(``),e.fitBounds(Dt[t],{transitionMs:1200}),e.requestLayers()},J=async t=>{await t.prepareStats(Fe(f())),await t.core.prepare(f()),await E(t),await O(t),e.options.showRuleSweep&&await j(t)},Y=await d(e.options.geography);if(e.signal.aborted)throw Error(`aborted`);r=await g(Y),S(r),await J(r),C(),M(r),V(),U(),e.setStatus(``);let Xe=e=>r&&e.coordinate?r.geography.pick(e.coordinate[0],e.coordinate[1]):-1,Ze=(t,n)=>({values:t,valueFormat:`float32`,colormap:`greys`,...ve(n),noDataColor:be[e.ground()],highlightClasses:u}),Qe=(e,t,n)=>({values:e,valueFormat:`uint32`,colormap:`category`,palette:t,noDataValue:pt,noDataColor:n,highlightClasses:u}),$e=(t,n)=>{let r=e.options,i=t.geography,a=e.ground(),o=Le(r.variable),s=Le(r.secondVariable),c=i.getVariable(r.variable)[n],l=i.getVariable(r.secondVariable)[n],u=y(t,r.variable),d=y(t,r.secondVariable),f=t.snapshot,p=f?.lag[n],m=f?f.offsets[n+1]===f.offsets[n]:!1,h=(e,t)=>{let n=ge(e,t);return n>=0?e.colors[n]:void 0},g=-1;if(f&&!m&&Number.isFinite(c)){let e=f.global[L.mean],t=c>=e,n=(p??0)>=e;g=t?n?0:3:n?1:2}let _=Ge(a),v=g<0?void 0:[_.highHigh,_.lowHigh,_.lowLow,_.highLow][g],b={label:o.label,value:He(o,c),unit:o.unit,swatch:h(u,c),emphasis:r.display===`value`},x=f&&!m&&Number.isFinite(p)?{label:`Neighbour average`,value:He(o,p),unit:o.unit,swatch:h(u,p),emphasis:r.display===`lag`}:null,S={label:`Moran quadrant`,value:g>=0?Ot[g]:m?`No neighbours`:`No data`,swatch:v,emphasis:r.display===`quadrant`},C={label:s.label,value:He(s,l),unit:s.unit,swatch:h(d,l),emphasis:r.display===`second`},w={label:`Rank`,value:Number.isFinite(c)?`${Se(yt(t.sortedX,c)*100)} percentile`:`-`,unit:`of ${i.unitPlural}`};if(r.display===`lag`&&x)return[x,w,b,S];if(r.display===`quadrant`)return[S,w,b,...x?[x]:[]];if(r.display===`second`)return[C,w,b];if(r.display===`bivariate`){let e=t.bivariateClasses[n],i=[`Low`,`Medium`,`High`],c=r.secondVariable===`income`,l=e===4294967295?``:i[c?2-Math.floor(e/3):Math.floor(e/3)];return[{label:`${o.label} with ${s.label.toLowerCase()}`,value:e===4294967295?`no data`:`${i[e%3].toLowerCase()} x, ${l.toLowerCase()} y`,swatch:e===4294967295?void 0:Be(a)[e],emphasis:!0},w,b,C]}if(r.display===`binary`){let e=Number.isFinite(c)&&c>je(i.getVariable(r.variable),r.joinPercentile);return[{label:`Join-count colour`,value:e?`Black`:`White`,swatch:e?Je:Ke,emphasis:!0},w,b]}return[{...b,emphasis:!0},w,...x?[x]:[],S]},et=[],rt=(t,n)=>{if(n===`variable`&&M(t),(n===`showRuleSweep`||n===`k`||n===`bandFactor`)&&M(t),[`display`,`showBands`,`showShuffled`,`showOutlines`,`variable`].includes(n)&&(V(),U()),[`display`,`showBands`,`showShuffled`,`showOutlines`,`showRuleSweep`].includes(n)){if((n===`showRuleSweep`||n===`showBands`)&&V(),t.snapshot&&ae(),n===`showRuleSweep`){let e=t.sweep;e?.state===`done`&&P(e),he()}return e.requestLayers(),!1}return n===`source`&&t.sweep?.state===`done`&&P(t.sweep),!0};return{getCompiledGraphs(){let e=r;return e?[...e.core.getGraphs(),...e.statsGraphs(),...e.permutation.values(),...e.correlogram.values(),...e.sweep?[...e.sweep.core.getGraphs(),e.sweep.graph]:[]]:[]},setOption(t,n){if(o++,t===`geography`){Ye(n);return}let i=r;if(i){if([`display`,`variable`,`secondVariable`,`showShuffled`].includes(t)&&(u=null),(t===`variable`||t===`secondVariable`||t===`joinPercentile`)&&S(i),t===`seed`&&x(i),[`source`,`k`,`bandFactor`,`transform`].includes(t)&&(c=!0),[`display`,`showBands`,`showShuffled`,`showOutlines`].includes(t)){rt(i,t);return}et.push(t),a.request({build:()=>J(i),commit:()=>{if(r!==i)return;let t=et.splice(0),n=!1;for(let e of t)n=rt(i,e)||n;n&&(C(),s=!0,l=!0,e.requestLayers())}})}},onAction(t){if(t!==`shuffle`)return;let n=e.options.seed%Ct+1;e.setOptions({seed:n},{notify:!0})},onGroundChange(){b(),ae(),e.requestLayers()},onLegendFilter(t,n){u=n===null?null:t===`quadrant-classes`?n.map(e=>We[e]??0):[...n],e.requestLayers()},encode(e,t){let n=r;if(!n||a.pending)return;if(s||t.frameIndex<2){(c||t.frameIndex<2)&&(n.core.encode(e,f()),c=!1),m(n)?.encode(e,{parameters:void 0});let r=w();r&&n.permutation.get(r)?.encode(e,{parameters:void 0}),n.correlogram.get(D())?.encode(e,{parameters:void 0}),s=!1,l=!0}l&&t.frameIndex>=1&&!n.reader.isPending&&(l=!1,n.reader.request(e));let i=n.sweep;if(i&&i.state===`running`&&t.frameIndex>=1)if(!i.encoded&&!i.reader.isPending){let t=p($[i.index]);i.core.writeParameters(t),i.core.encode(e,t),i.graph.encode(e,{parameters:void 0}),i.encoded=!0,i.reader.request(e)}else i.encoded&&i.reader.flush(e)},getLayers(){let t=r;if(!t)return[];let n=e.options,i=e.ground(),{geography:a,buffers:o}=t,s=[a.origin[0],a.origin[1],0],c=be[i],l=y(t,n.variable),u=y(t,n.secondVariable),d=Be(i),f=(e,t,n)=>new B({id:e,coordinateOrigin:s,triangles:o.triangles,features:o.features,vertexCount:o.triangleVertexCount,...t,compareSide:n,opacity:1}),p=[];if(n.showShuffled&&n.display===`value`?(p.push(f(`global-fill-shuffled`,Ze(t.shuffled,l),`a`)),p.push(f(`global-fill`,Ze(t.x,l),`b`))):n.display===`quadrant`?p.push(f(`global-fill`,Qe(t.quadrant,qe(i),c))):n.display===`bivariate`?p.push(f(`global-fill`,Qe(t.bivariateClass,d,c))):n.display===`binary`?p.push(f(`global-fill`,Qe(t.joinClass,[Ke,Je],c))):n.display===`second`?p.push(f(`global-fill`,Ze(t.y,u))):n.display===`lag`?p.push(f(`global-fill`,Ze(t.lag,l))):p.push(f(`global-fill`,Ze(t.x,l))),n.showOutlines&&(p.push(new ce({id:`global-outline`,coordinateOrigin:s,segments:o.outline,instanceCount:o.outlineSegmentCount,widthPixels:a.count>2e3?.4:.5,color:ke(i)})),t.stateBuffer&&t.stateSegments)){let e=Oe(i);p.push(new ce({id:`global-state-lines`,coordinateOrigin:s,segments:t.stateBuffer,instanceCount:t.stateSegments.length/4,widthPixels:e.widthPixels,color:e.color,outlineColor:e.casing,outlineWidthPixels:(e.casingPixels-e.widthPixels)/2}))}return p},getTooltip(e){let t=r,n=Xe(e);if(!t||n<0)return null;let i=t.geography;return{title:i.getName(n),subtitle:i.getGroupName(n),rows:$e(t,n),anchor:I(t,n),highlight:{kind:`polygon`,rings:ee(t,n)}}},onClick(t){let n=r,i=Xe(t);return!n||i<0?!1:(n.focusRow=i,U(),V(),n.snapshot&&se(),e.requestLayers(),!0)},destroy(){n=!0,i++,a.destroy();let e=r;r=null,e&&(e.reader.stop(),e.sweep?.reader.stop(),e.resources.destroy())}}}export{Mt as createGlobalAutocorrelation};