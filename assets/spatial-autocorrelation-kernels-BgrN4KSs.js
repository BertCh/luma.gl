import{t as e}from"./parameter-schema-DB2zG6AH.js";import{d as t,j as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{s as a}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as o,c as s}from"./wgsl-common-BNRVzgd8.js";import{c,r as l,s as u}from"./spatial-weights-D4bd9AQ6.js";import{n as d,t as f}from"./float32-values-C3OXugIo.js";var p=e({id:`spatial-autocorrelation`,format:`float32`,wordLength:8,fields:[{name:`significanceLevel`,format:`float32`,wordOffset:0,defaultValue:.05,minimum:Number.MIN_VALUE,maximum:1,dynamic:!0},{name:`useFixedMoments`,format:`float32`,wordOffset:1,defaultValue:0,minimum:0,maximum:1,dynamic:!0},{name:`fixedCount`,format:`float32`,wordOffset:2,defaultValue:0,minimum:0,dynamic:!0},{name:`fixedMean`,format:`float32`,wordOffset:3,defaultValue:0,dynamic:!0},{name:`fixedVariance`,format:`float32`,wordOffset:4,defaultValue:0,minimum:0,dynamic:!0}]}),m=[1.6448536269514722,1.959963984540054,2.5758293035489004],h=[.1,.05,.01],g={NOT_SIGNIFICANT:0,HIGH_HIGH:1,LOW_HIGH:2,LOW_LOW:3,HIGH_LOW:4};function _(e={},n=new Float32Array(8)){t(`getGPUSpatialAutocorrelationParameterValues`,`target length`,n.length,8);let r=e.significanceLevel??.05;if(!(r>0&&r<1))throw Error(`Spatial autocorrelation significanceLevel must be in (0, 1)`);let i=e.fixedMoments;if(i&&(!Number.isFinite(i.count)||!Number.isFinite(i.mean)||!Number.isFinite(i.variance)||i.count<2||i.variance<=0))throw Error(`Spatial autocorrelation fixedMoments are invalid`);return n.set([r,i?1:0,i?.count??0,i?.mean??0,i?.variance??0,0,0,0]),n}var v=4096,y=256,b=0,x=1,S=2;function C(e,n){let r=n??1;return t(e,`minimumNeighbors`,r),r}function w(e){let{id:r}=e,a=c(r,e.weights);if(d(e.values,`${r} values`),i(e.parameters,[`float32`],`${r} parameters`),e.islands!==void 0&&u(r,e.islands),e.weights.status?.overflow&&n(e.weights.status.overflow,`${r} weights.status.overflow`),t(r,`weights row count`,a,1,2**24-1),e.values.length!==a)throw Error(`${r} values length must equal the weights row count`);if(t(r,`parameters length`,e.parameters.length,8),e.mask&&(n(e.mask,`${r} mask`),e.mask.length!==a))throw Error(`${r} mask length must equal the weights row count`);return e.globalStatistics&&(i(e.globalStatistics,[`float32`],`${r} output.globalStatistics`),t(r,`output.globalStatistics length`,e.globalStatistics.length,5)),a}var T=`
${s}

// A runtime operand keeps the quiet-NaN bit pattern out of constant evaluation.
${o}

// Two-sided normal p-value erfc(|z| / sqrt(2)) with the Numerical Recipes erfcc Chebyshev fit
// (fractional error below 1.2e-7 in exact arithmetic), so tiny p-values keep relative accuracy.
fn getTwoSidedPValue(z: f32) -> f32 {
  let x = abs(z) * 0.70710678118654752;
  let t = 1.0 / (1.0 + 0.5 * x);
  let exponent = -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))));
  return min(t * exp(exponent), 1.0);
}
`;function E(){return`
fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

${T}
`}function D(e,t){return`const MOMENTS: u32 = ${e}u;
const ROWS: u32 = ${e}u;
const CAPACITY: u32 = ${t}u;`}function O(e){return l(`isFiniteFloat(statistics[statisticsOffset + neighbor])`,e)}function k(e,t){let{id:n,operation:i,parameters:o,mask:s,weights:c}=t,u=f(e,`${n}-values`,t.values),d=u.view,p=d.length,m=Math.ceil(p/v),h=E(),g=[...u.nodes],_=t.islands!==void 0,C=t.islands===`keep`,w=c.status?.overflow,D=r(e,`${n}-value-partials`,`float32`,m),O=r(e,`${n}-count-partials`,`uint32`,m),k=_?r(e,`${n}-island-partials`,`uint32`,m):void 0,A=r(e,`${n}-deviation-partials`,`float32`,m),j=r(e,`${n}-square-partials`,`float32`,m),M=r(e,`${n}-statistics`,`float32`,p+5),N=_?r(e,`${n}-row-states`,`uint32`,p):void 0,P=s?[{name:`mask`,view:s,type:`u32`,access:`read`}]:[],F=`fn isRawRowValid(row: u32) -> bool {
  return ${s?`mask[maskOffset + row] != 0u && `:``}isFiniteFloat(values[valuesOffset + row]);
}`,I=N?[{name:`rowStates`,view:N,type:`u32`,access:`read`}]:P,L=`${T}
const MOMENTS: u32 = ${p}u;
const ROW_COUNT: u32 = ${p}u;
const BLOCK_ROWS: u32 = ${v}u;
const WORKGROUP_SIZE: u32 = ${y}u;
${N?`fn isRowValid(row: u32) -> bool {
  let state = rowStates[rowStatesOffset + row];
  return state == ${x}u || (state == ${S}u && ${C});
}`:F.replace(`isRawRowValid`,`isRowValid`)}`;return N&&g.push(a(e,{id:`${n}-row-states`,operation:i,variant:`row-states`,bindings:[{name:`offsets`,view:c.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:c.neighbors,type:`u32`,access:`read`},{name:`weights`,view:c.weights,type:`f32`,access:`read`},{name:`values`,view:d,type:`f32`,access:`read`},...P,{name:`rowStates`,view:N,type:`u32`,access:`read_write`}],invocationCount:p,declarations:`${T}
const ROWS: u32 = ${p}u;
const CAPACITY: u32 = ${c.neighbors.length}u;
${F}`,body:`if (!isRawRowValid(index)) {
    rowStates[rowStatesOffset + index] = ${b}u;
    return;
  }
  var hasNeighbor = false;
  ${l(`isRawRowValid(neighbor)`,`hasNeighbor = true; break;`)}
  rowStates[rowStatesOffset + index] = select(${S}u, ${x}u, hasNeighbor);`})),g.push(a(e,{id:`${n}-moments-blocks`,operation:i,variant:`moments-blocks`,bindings:[{name:`values`,view:d,type:`f32`,access:`read`},...I,{name:`valuePartials`,view:D,type:`f32`,access:`read_write`},{name:`countPartials`,view:O,type:`u32`,access:`read_write`},...k?[{name:`islandPartials`,view:k,type:`u32`,access:`read_write`}]:[]],workgroupSize:y,invocationCount:m*y,guardIndex:!1,declarations:`${L}
var<workgroup> partialSums: array<f32, ${y}>;
var<workgroup> partialCounts: array<u32, ${y}>;
var<workgroup> partialIslands: array<u32, ${y}>;`,body:`let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  var count = 0u;
  var islands = 0u;
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      if (isRowValid(row)) {
        sum += values[valuesOffset + row];
        count++;
      }
      ${N?`islands += select(0u, 1u, rowStates[rowStatesOffset + row] == ${S}u);`:``}
    }
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  partialIslands[localInvocationIndex] = islands;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
      partialIslands[localInvocationIndex] += partialIslands[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    valuePartials[valuePartialsOffset + block] = partialSums[0];
    countPartials[countPartialsOffset + block] = partialCounts[0];
    ${k?`islandPartials[islandPartialsOffset + block] = partialIslands[0];`:``}
  }`}),a(e,{id:`${n}-moments-mean`,operation:i,variant:`moments-mean`,bindings:[{name:`valuePartials`,view:D,type:`f32`,access:`read`},{name:`countPartials`,view:O,type:`u32`,access:`read`},...k?[{name:`islandPartials`,view:k,type:`u32`,access:`read`}]:[],{name:`statistics`,view:M,type:`f32`,access:`read_write`}],workgroupSize:y,invocationCount:y,guardIndex:!1,declarations:`${T}
const MOMENTS: u32 = ${p}u;
const BLOCK_COUNT: u32 = ${m}u;
const WORKGROUP_SIZE: u32 = ${y}u;
var<workgroup> partialSums: array<f32, ${y}>;
var<workgroup> partialCounts: array<u32, ${y}>;
var<workgroup> partialIslands: array<u32, ${y}>;`,body:`var sum = 0.0;
  var count = 0u;
  var islands = 0u;
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    sum += valuePartials[valuePartialsOffset + block];
    count += countPartials[countPartialsOffset + block];
    ${k?`islands += islandPartials[islandPartialsOffset + block];`:``}
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  partialIslands[localInvocationIndex] = islands;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
      partialIslands[localInvocationIndex] += partialIslands[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    // The count is exact in u32 and converted to f32 only here. The mean is provisional: the
    // deviation pass corrects it.
    let total = f32(partialCounts[0]);
    statistics[statisticsOffset + MOMENTS] = total;
    statistics[statisticsOffset + MOMENTS + 1u] = select(getQuietNaN(index), partialSums[0] / total, total >= 1.0);
    statistics[statisticsOffset + MOMENTS + 4u] = f32(partialIslands[0]);
  }`}),a(e,{id:`${n}-moments-deviations`,operation:i,variant:`moments-deviations`,bindings:[{name:`values`,view:d,type:`f32`,access:`read`},...I,{name:`statistics`,view:M,type:`f32`,access:`read`},{name:`deviationPartials`,view:A,type:`f32`,access:`read_write`}],workgroupSize:y,invocationCount:m*y,guardIndex:!1,declarations:`${L}
var<workgroup> partialSums: array<f32, ${y}>;`,body:`let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  if (isInRange) {
    let provisionalMean = statistics[statisticsOffset + MOMENTS + 1u];
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      if (isRowValid(row)) {
        sum += values[valuesOffset + row] - provisionalMean;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    deviationPartials[deviationPartialsOffset + block] = partialSums[0];
  }`}),a(e,{id:`${n}-moments-correct`,operation:i,variant:`moments-correct`,bindings:[{name:`deviationPartials`,view:A,type:`f32`,access:`read`},{name:`parameters`,view:o,type:`f32`,access:`read`},{name:`statistics`,view:M,type:`f32`,access:`read_write`},...w?[{name:`overflow`,view:w,type:`u32`,access:`read`}]:[]],workgroupSize:y,invocationCount:y,guardIndex:!1,declarations:`${h}
const MOMENTS: u32 = ${p}u;
const BLOCK_COUNT: u32 = ${m}u;
const WORKGROUP_SIZE: u32 = ${y}u;
var<workgroup> partialSums: array<f32, ${y}>;`,body:`var sum = 0.0;
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    sum += deviationPartials[deviationPartialsOffset + block];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    var total = statistics[statisticsOffset + MOMENTS];
    var mean = statistics[statisticsOffset + MOMENTS + 1u];
    if (total >= 1.0) {
      mean += partialSums[0] / total;
    }
    if (readParameter(1u) != 0.0) {
      total = readParameter(2u);
      mean = readParameter(3u);
    }
    ${w?`// Truncated weights: no statistic from them is trustworthy, so every output becomes NaN.
    if (overflow[overflowOffset] != 0u) {
      mean = getQuietNaN(index);
    }`:``}
    statistics[statisticsOffset + MOMENTS] = total;
    statistics[statisticsOffset + MOMENTS + 1u] = mean;
  }`}),a(e,{id:`${n}-center`,operation:i,variant:`center`,bindings:[{name:`values`,view:d,type:`f32`,access:`read`},...I,{name:`statistics`,view:M,type:`f32`,access:`read_write`},{name:`squarePartials`,view:j,type:`f32`,access:`read_write`}],workgroupSize:y,invocationCount:m*y,guardIndex:!1,declarations:`${L}
var<workgroup> partialSums: array<f32, ${y}>;`,body:`let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  if (isInRange) {
    let mean = statistics[statisticsOffset + MOMENTS + 1u];
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      let valid = isRowValid(row);
      let centered = values[valuesOffset + row] - mean;
      // Rows outside the analysis store quiet NaN, which every later kernel reads as "not a focus
      // row and not a neighbor".
      statistics[statisticsOffset + row] = select(getQuietNaN(row), centered, valid);
      sum += select(0.0, centered * centered, valid);
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    squarePartials[squarePartialsOffset + block] = partialSums[0];
  }`}),a(e,{id:`${n}-moments-variance`,operation:i,variant:`moments-variance`,bindings:[{name:`squarePartials`,view:j,type:`f32`,access:`read`},{name:`parameters`,view:o,type:`f32`,access:`read`},{name:`statistics`,view:M,type:`f32`,access:`read_write`},...t.globalStatistics?[{name:`globalStatistics`,view:t.globalStatistics,type:`f32`,access:`read_write`}]:[]],workgroupSize:y,invocationCount:y,guardIndex:!1,declarations:`${h}
const MOMENTS: u32 = ${p}u;
const BLOCK_COUNT: u32 = ${m}u;
const WORKGROUP_SIZE: u32 = ${y}u;
var<workgroup> partialSums: array<f32, ${y}>;`,body:`var sum = 0.0;
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    sum += squarePartials[squarePartialsOffset + block];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let count = statistics[statisticsOffset + MOMENTS];
    let mean = statistics[statisticsOffset + MOMENTS + 1u];
    var sumOfSquares = partialSums[0];
    var variance = select(getQuietNaN(index), sumOfSquares / count, count >= 1.0);
    if (readParameter(1u) != 0.0) {
      variance = readParameter(4u);
      sumOfSquares = variance * count;
    }
    statistics[statisticsOffset + MOMENTS + 2u] = variance;
    statistics[statisticsOffset + MOMENTS + 3u] = sumOfSquares;
    ${t.globalStatistics?`globalStatistics[globalStatisticsOffset] = count;
    globalStatistics[globalStatisticsOffset + 1u] = mean;
    globalStatistics[globalStatisticsOffset + 2u] = variance;
    globalStatistics[globalStatisticsOffset + 3u] = sqrt(variance);
    globalStatistics[globalStatisticsOffset + 4u] = statistics[statisticsOffset + MOMENTS + 4u];`:``}
  }`})),{nodes:g,statistics:M}}export{O as a,m as c,p as d,_ as f,E as i,h as l,D as n,C as o,k as r,w as s,T as t,g as u};