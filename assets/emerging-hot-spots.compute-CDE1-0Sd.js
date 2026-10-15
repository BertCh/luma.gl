import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{E as e}from"./analysis-frame-Blu0tYdw.js";import{N as t,_ as n,d as r,h as i,j as a,n as o,t as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{l as u,n as d,s as f}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as p}from"./wgsl-common-BNRVzgd8.js";import{c as m}from"./spatial-weights-D4bd9AQ6.js";import{r as h,t as g}from"./gpu-multiple-testing-adjustment-C7k5NYt0.js";import{t as _}from"./spatial-autocorrelation-kernels-BgrN4KSs.js";import{n as v,t as ee}from"./gpu-neighbor-search-Fct8W8hB.js";import{t as y}from"./gpu-color-scale-BxTDjZQ3.js";import{r as b,s as x,t as S}from"./recipe-utils-CqprOQld.js";import{i as C,n as w,r as T,t as E}from"./b13-nature-events-CKVFzbno.js";import{t as D}from"./gpu-group-statistics-MBJHmgtx.js";import{a as O,s as k}from"./index-Bjk18V2I.js";import{n as A,t as j}from"./gpu-temporal-reduction-uwWQ39vO.js";import{t as M}from"./graph-buffers-B4_80RlX.js";import{t as N}from"./graph-swap-41Ru7LVu.js";import{t as te}from"./summary-reader-CsgWxENf.js";import{t as P}from"./playback-BqFO3Eom.js";import{i as F,n as I,r as L,t as R}from"./b13-categories-BgESQBzu.js";var ne={NO_PATTERN:0,NEW_HOT:1,CONSECUTIVE_HOT:2,INTENSIFYING_HOT:3,PERSISTENT_HOT:4,DIMINISHING_HOT:5,SPORADIC_HOT:6,OSCILLATING_HOT:7,HISTORICAL_HOT:8,NEW_COLD:9,CONSECUTIVE_COLD:10,INTENSIFYING_COLD:11,PERSISTENT_COLD:12,DIMINISHING_COLD:13,SPORADIC_COLD:14,OSCILLATING_COLD:15,HISTORICAL_COLD:16,UNDETERMINED:17},z={"0.9":1.6448536269514722,"0.95":1.959963984540054,"0.99":2.5758293035489004};function B(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUEmergingHotSpots target is too short`);let{temporalWindow:n}=e,r=e.radius??0;if(!Number.isFinite(r)||r<0)throw Error(`GPUEmergingHotSpots radius must be a finite number >= 0`);if(!Number.isInteger(n)||n<0)throw Error(`GPUEmergingHotSpots temporalWindow must be an integer >= 0`);let i=e.confidenceLevel??.95;if(!(i>0&&i<1))throw Error(`GPUEmergingHotSpots confidenceLevel must be in (0, 1)`);let a=e.multipleTesting??`benjamini-hochberg`,o=e.criticalZ;if(o!==void 0){if(a!==`none`)throw Error(`GPUEmergingHotSpots criticalZ requires multipleTesting 'none'`);if(!Number.isFinite(o)||o<=0)throw Error(`GPUEmergingHotSpots criticalZ must be a finite number > 0`)}else if(o=z[String(i)],o===void 0){if(a===`none`)throw Error(`GPUEmergingHotSpots confidenceLevel must be 0.9, 0.95 or 0.99`);o=0}let s=e.minimumTrendSliceCount??10;if(!Number.isInteger(s)||s<0)throw Error(`GPUEmergingHotSpots minimumTrendSliceCount must be an integer >= 0`);let c=e.trendSignificanceLevel??1-i;if(!(c>0&&c<1))throw Error(`GPUEmergingHotSpots trendSignificanceLevel must be in (0, 1)`);let l=e.persistentFraction??.9;if(!(l>.5&&l<=1))throw Error(`GPUEmergingHotSpots persistentFraction must be in (0.5, 1]`);return t.set([r,n,o,c,l,s,1-i,0]),t}var V=`GPUEmergingHotSpots`,H=4096,U=256,W=256,G=64,re=64,ie=2**31-1,ae=class{id;props;multipleTesting;finalSlice;constructor(e){this.id=e.id??`emerging-hot-spots`,this.props=e;let t=this.id,{output:n}=e;if(this.multipleTesting=h(t,e.multipleTesting??`benjamini-hochberg`),this.finalSlice=e.finalSlice??`require`,this.finalSlice!==`require`&&this.finalSlice!==`last-valid`)throw Error(`${t} finalSlice must be 'require' or 'last-valid'`);let{gridWidth:o,gridHeight:s,sliceCount:c}=e,u=o!==void 0||s!==void 0;if(u===(e.weights!==void 0))throw Error(`${t} needs exactly one of gridWidth and gridHeight, or weights`);let d=e.selfWeight??1;if(!Number.isFinite(d)||d<0)throw Error(`${t} selfWeight must be a finite number >= 0`);u&&(r(t,`gridWidth`,o??NaN),r(t,`gridHeight`,s??NaN)),r(t,`sliceCount`,c,1,256);let f=e.weights?m(t,e.weights):o*s;if(f>=2**24)throw Error(`${t} must have fewer than 2^24 cells`);let p=f*c;if(p>ie)throw Error(`${t} cellCount * sliceCount must be below 2^31`);if(r(t,`maximumRadius`,e.maximumRadius??4,0,32),l(e.values,[`float32`,`uint32`],`${t} values`),e.values.length!==p)throw Error(`${t} values length must equal cellCount * sliceCount`);if(l(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<8)throw Error(`${t} parameters is too short`);if(e.mask&&(a(e.mask,`${t} mask`),e.mask.length!==f))throw Error(`${t} mask length must equal the cell count`);if(l(n.giZScores,[`float32`],`${t} giZScores`),n.giZScores.length!==p)throw Error(`${t} giZScores length must equal values length`);for(let[e,r]of[[`pValues`,n.pValues],[`adjustedPValues`,n.adjustedPValues]])if(r&&(l(r,[`float32`],`${t} ${e}`),r.length!==p))throw Error(`${t} ${e} length must equal values length`);if((this.multipleTesting!==`none`||n.pValues||n.adjustedPValues)&&p>=16777216)throw Error(`${t} multiple testing needs fewer than 2^24 bins`);if(n.lastValidSlice&&(a(n.lastValidSlice,`${t} lastValidSlice`),n.lastValidSlice.length!==f))throw Error(`${t} lastValidSlice length must equal the cell count`);l(n.trendZ,[`float32`],`${t} trendZ`),l(n.trendP,[`float32`],`${t} trendP`),l(n.trendS,[`sint32`],`${t} trendS`),a(n.category,`${t} category`),a(n.hotSliceCount,`${t} hotSliceCount`),a(n.coldSliceCount,`${t} coldSliceCount`);for(let[e,r]of[[`trendZ`,n.trendZ],[`trendP`,n.trendP],[`trendS`,n.trendS],[`category`,n.category],[`hotSliceCount`,n.hotSliceCount],[`coldSliceCount`,n.coldSliceCount]])if(r.length!==f)throw Error(`${t} ${e} length must equal the cell count`);if(n.globalStatistics&&(l(n.globalStatistics,[`float32`],`${t} globalStatistics`),n.globalStatistics.length<4))throw Error(`${t} globalStatistics is too short`);i(t,[n.giZScores,n.pValues,n.adjustedPValues,n.lastValidSlice,n.trendZ,n.trendP,n.trendS,n.category,n.hotSliceCount,n.coldSliceCount,n.globalStatistics],[e.values,e.parameters,e.mask,e.weights?.offsets,e.weights?.neighbors,e.weights?.weights])}getParameterValues(e,t){return B(this._resolveSettings(e),t)}getInferenceMetadata(e){let t=this._resolveSettings(e);return B(t),{alternative:`two-sided`,permutationCount:0,multipleTesting:this.multipleTesting,significanceLevel:1-(t.confidenceLevel??.95),testedRowsDefinition:`valid space-time bins`}}_resolveSettings(e){if(e.multipleTesting!==void 0&&e.multipleTesting!==this.multipleTesting)throw Error(`${this.id} settings.multipleTesting must match the node`);return{...e,multipleTesting:this.multipleTesting}}getCommandNodes(e){let{id:t,props:r}=this,{values:i,mask:a,parameters:o,weights:s,sliceCount:l,output:d}=r,{giZScores:m}=d,h=r.gridWidth??1,v=r.gridHeight??1,ee=r.selfWeight??1;n(t,e,[i,a,o,s?.offsets,s?.neighbors,s?.weights,m,d.trendZ,d.trendP,d.pValues,d.adjustedPValues,d.lastValidSlice,d.trendS,d.category,d.hotSliceCount,d.coldSliceCount,d.globalStatistics]);let{finalSlice:y,multipleTesting:b}=this,x=b!==`none`,S=x||d.pValues!==void 0||d.adjustedPValues!==void 0,C=r.maximumRadius??4,w=s?s.offsets.length-1:h*v,T=w*l;if(T>Math.floor(e.device.limits.maxStorageBufferBindingSize/4))throw Error(`${t} cube exceeds maxStorageBufferBindingSize`);let E=l>=re,D=Math.ceil(T/H),O=i.format===`uint32`?`u32`:`f32`,k=c(e,`${t}-sum-partials`,`float32`,D),A=c(e,`${t}-count-partials`,`uint32`,D),j=c(e,`${t}-square-partials`,`float32`,D),M=d.globalStatistics??c(e,`${t}-statistics`,`float32`,4),N=`
const GRID_WIDTH: u32 = ${h}u;
const GRID_HEIGHT: u32 = ${v}u;
const CELL_COUNT: u32 = ${w}u;
const SLICE_COUNT: u32 = ${l}u;
const BIN_COUNT: u32 = ${T}u;
const BIN_BLOCK: u32 = ${H}u;
const MAXIMUM_RADIUS: f32 = ${C}.0;
const SELF_WEIGHT: f32 = ${u(ee)};
${_}

fn readValue(bin: u32) -> f32 {
  return f32(values[valuesOffset + bin]);
}

fn isBinValid(bin: u32, cell: u32) -> bool {
  ${a?`if (mask[maskOffset + cell] == 0u) { return false; }`:``}
  return isFiniteFloat(readValue(bin));
}
`,te={name:`values`,view:i,type:O,access:`read`},P=a?[{name:`mask`,view:a,type:`u32`,access:`read`}]:[],F={name:`parameters`,view:o,type:`f32`,access:`read`},I=e=>({name:`statistics`,view:M,type:`f32`,access:e}),L=s?[{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`weights`,view:s.weights,type:`f32`,access:`read`}]:[],R=Math.max(1,Math.floor(W/l)),ne=`let group = index / ${W}u;
  let local = localInvocationIndex;
  let cellInGroup = local / SLICE_COUNT;
  let slice = local % SLICE_COUNT;
  let cell = group * CELLS_PER_GROUP + cellInGroup;
  let isActive = index < INVOCATION_COUNT && cellInGroup < CELLS_PER_GROUP && cell < CELL_COUNT;
  let rawRadius = ${s?`0.0`:`parameters[parametersOffset]`};
  let rawWindow = parameters[parametersOffset + 1u];
  let count = statistics[statisticsOffset];
  let mean = statistics[statisticsOffset + 1u];
  let variance = statistics[statisticsOffset + 2u];
  let deviation = statistics[statisticsOffset + 3u];
  let parametersValid = isFiniteFloat(rawRadius) && isFiniteFloat(rawWindow) && rawRadius >= 0.0 && rawWindow >= 0.0;
  let radius = min(rawRadius, MAXIMUM_RADIUS);
  let radiusSquared = radius * radius;
  let reach = i32(floor(radius));
  instantWeighted[local] = 0.0;
  instantWeight[local] = 0.0;
  instantSquare[local] = 0.0;
  if (isActive && parametersValid) {
    ${s?`var weighted = 0.0;
    var weightTotal = 0.0;
    var squareTotal = 0.0;
    let bin = cell * SLICE_COUNT + slice;
    // The focal cell has weight SELF_WEIGHT; its CSR row never lists the cell itself.
    if (SELF_WEIGHT != 0.0 && isBinValid(bin, cell)) {
      weighted += SELF_WEIGHT * (readValue(bin) - mean);
      weightTotal += SELF_WEIGHT;
      squareTotal += SELF_WEIGHT * SELF_WEIGHT;
    }
    for (var slot = offsets[offsetsOffset + cell]; slot < offsets[offsetsOffset + cell + 1u]; slot++) {
      let neighborCell = neighbors[neighborsOffset + slot];
      if (neighborCell >= CELL_COUNT || neighborCell == cell) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      let neighborBin = neighborCell * SLICE_COUNT + slice;
      if (isBinValid(neighborBin, neighborCell)) {
        weighted += weight * (readValue(neighborBin) - mean);
        weightTotal += weight;
        squareTotal += weight * weight;
      }
    }
    instantWeighted[local] = weighted;
    instantWeight[local] = weightTotal;
    instantSquare[local] = squareTotal;`:`var weighted = 0.0;
    var neighborCount = 0u;
    let column = i32(cell % GRID_WIDTH);
    let row = i32(cell / GRID_WIDTH);
    for (var deltaRow = -reach; deltaRow <= reach; deltaRow++) {
      let neighborRow = row + deltaRow;
      if (neighborRow < 0 || neighborRow >= i32(GRID_HEIGHT)) {
        continue;
      }
      for (var deltaColumn = -reach; deltaColumn <= reach; deltaColumn++) {
        let neighborColumn = column + deltaColumn;
        if (neighborColumn < 0 || neighborColumn >= i32(GRID_WIDTH) ||
            f32(deltaRow * deltaRow + deltaColumn * deltaColumn) > radiusSquared) {
          continue;
        }
        let neighborCell = u32(neighborRow) * GRID_WIDTH + u32(neighborColumn);
        let neighborBin = neighborCell * SLICE_COUNT + slice;
        if (isBinValid(neighborBin, neighborCell)) {
          weighted += readValue(neighborBin) - mean;
          neighborCount++;
        }
      }
    }
    instantWeighted[local] = weighted;
    instantWeight[local] = f32(neighborCount);
    instantSquare[local] = 0.0;`}
  }
  // Every invocation reaches the barrier.
  workgroupBarrier();
  if (isActive) {
    var zScore = getQuietNaN(index);
    if (parametersValid && isBinValid(cell * SLICE_COUNT + slice, cell)) {
      let windowSize = u32(min(floor(rawWindow), f32(SLICE_COUNT - 1u)));
      let firstSlice = slice - min(windowSize, slice);
      ${s?`var weightedSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    for (var windowSlice = firstSlice; windowSlice <= slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      weightedSum += instantWeighted[slot];
      weightSum += instantWeight[slot];
      squareSum += instantSquare[slot];
    }`:`var weightedSum = 0.0;
    var weightSum = 0.0;
    for (var windowSlice = firstSlice; windowSlice <= slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      weightedSum += instantWeighted[slot];
      weightSum += instantWeight[slot];
    }
    let squareSum = weightSum;`}
      ${s?`let spread = (count * squareSum - weightSum * weightSum) / (count - 1.0);`:`let spread = weightSum * (count - weightSum) / (count - 1.0);`}
      if (count >= 2.0 && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        zScore = weightedSum / (deviation * sqrt(spread));
      }
    }
    giZScores[giZScoresOffset + cell * SLICE_COUNT + slice] = zScore;
  }`,z=d.pValues??(S?c(e,`${t}-p-values`,`float32`,T):void 0),B=d.adjustedPValues??(S?c(e,`${t}-adjusted-p-values`,`float32`,T):void 0),ie=()=>[f(e,{id:`${t}-p-values`,operation:V,variant:`two-sided-p-values`,bindings:[{name:`giZScores`,view:m,type:`f32`,access:`read`},{name:`pValues`,view:z,type:`f32`,access:`read_write`}],invocationCount:T,declarations:`${_}
${p}`,body:`let zScore = giZScores[giZScoresOffset + index];
  // erfc(|z| / sqrt(2)); NaN z (missing bin) stays NaN and is excluded from the family.
  pValues[pValuesOffset + index] = select(
    getQuietNaN(index),
    min(erfcValue(abs(zScore) * 0.7071067811865476), 1.0),
    isFiniteFloat(zScore)
  );`}),...new g({id:`${t}-adjustment`,pValues:z,method:b,output:{adjustedPValues:B}}).getCommandNodes(e)],ae=[];return ae.push(f(e,{id:`${t}-block-sums`,operation:V,variant:`block-sums`,bindings:[te,...P,{name:`sumPartials`,view:k,type:`f32`,access:`read_write`},{name:`countPartials`,view:A,type:`u32`,access:`read_write`}],workgroupSize:U,invocationCount:D*U,guardIndex:!1,declarations:`${N}
var<workgroup> partialSums: array<f32, ${U}>;
var<workgroup> partialCounts: array<u32, ${U}>;`,body:`let block = index / ${U}u;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  var count = 0u;
  if (isInRange) {
    let begin = block * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${U}u) {
      if (isBinValid(bin, bin / SLICE_COUNT)) {
        sum += readValue(bin);
        count++;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${U/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    sumPartials[sumPartialsOffset + block] = partialSums[0];
    countPartials[countPartialsOffset + block] = partialCounts[0];
  }`}),f(e,{id:`${t}-mean`,operation:V,variant:`mean`,bindings:[{name:`sumPartials`,view:k,type:`f32`,access:`read`},{name:`countPartials`,view:A,type:`u32`,access:`read`},I(`read_write`)],invocationCount:1,declarations:`const BLOCK_COUNT: u32 = ${D}u;`,body:`var sum = 0.0;
  var count = 0u;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += sumPartials[sumPartialsOffset + block];
    count += countPartials[countPartialsOffset + block];
  }
  statistics[statisticsOffset] = f32(count);
  statistics[statisticsOffset + 1u] = select(0.0, sum / f32(count), count > 0u);
  statistics[statisticsOffset + 2u] = 0.0;
  statistics[statisticsOffset + 3u] = 0.0;`}),f(e,{id:`${t}-block-squares`,operation:V,variant:`block-squares`,bindings:[te,...P,I(`read`),{name:`squarePartials`,view:j,type:`f32`,access:`read_write`}],workgroupSize:U,invocationCount:D*U,guardIndex:!1,declarations:`${N}
var<workgroup> partialSums: array<f32, ${U}>;`,body:`let block = index / ${U}u;
  let isInRange = index < INVOCATION_COUNT;
  let mean = statistics[statisticsOffset + 1u];
  var sum = 0.0;
  if (isInRange) {
    let begin = block * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${U}u) {
      if (isBinValid(bin, bin / SLICE_COUNT)) {
        let centered = readValue(bin) - mean;
        sum += centered * centered;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${U/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    squarePartials[squarePartialsOffset + block] = partialSums[0];
  }`}),f(e,{id:`${t}-deviation`,operation:V,variant:`deviation`,bindings:[{name:`squarePartials`,view:j,type:`f32`,access:`read`},I(`read_write`)],invocationCount:1,declarations:`const BLOCK_COUNT: u32 = ${D}u;`,body:`var sum = 0.0;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += squarePartials[squarePartialsOffset + block];
  }
  let count = statistics[statisticsOffset];
  let variance = select(0.0, sum / count, count > 0.0);
  statistics[statisticsOffset + 2u] = variance;
  statistics[statisticsOffset + 3u] = sqrt(variance);`}),f(e,{id:`${t}-gi-star`,operation:V,variant:s?`gi-star-weights`:`gi-star`,bindings:[te,...P,F,...L,I(`read`),{name:`giZScores`,view:m,type:`f32`,access:`read_write`}],workgroupSize:W,invocationCount:Math.ceil(w/R)*W,guardIndex:!1,declarations:`${N}
const CELLS_PER_GROUP: u32 = ${R}u;
var<workgroup> instantWeighted: array<f32, ${W}>;
var<workgroup> instantWeight: array<f32, ${W}>;
var<workgroup> instantSquare: array<f32, ${W}>;`,body:ne}),...S?ie():[],f(e,{id:`${t}-slice-counts`,operation:V,variant:`slice-counts`,bindings:[F,{name:`giZScores`,view:m,type:`f32`,access:`read`},...x?[{name:`adjustedPValues`,view:B,type:`f32`,access:`read`}]:[],{name:`hotSliceCount`,view:d.hotSliceCount,type:`u32`,access:`read_write`},{name:`coldSliceCount`,view:d.coldSliceCount,type:`u32`,access:`read_write`},...d.lastValidSlice?[{name:`lastValidSlice`,view:d.lastValidSlice,type:`u32`,access:`read_write`}]:[]],invocationCount:w,declarations:`const SLICE_COUNT: u32 = ${l}u;
${_}
${K(x)}`,body:`let base = index * SLICE_COUNT;
  ${oe({valueExpression:`giZScores[giZScoresOffset + base + slice]`,adjustedExpression:`adjustedPValues[adjustedPValuesOffset + base + slice]`,usesAdjustment:x,collectSeries:!1})}
  hotSliceCount[hotSliceCountOffset + index] = hot;
  coldSliceCount[coldSliceCountOffset + index] = cold;
  ${d.lastValidSlice?`lastValidSlice[lastValidSliceOffset + index] = lastValid;`:``}`}),f(e,{id:`${t}-mann-kendall`,operation:V,variant:E?`mann-kendall-cooperative`:`mann-kendall-classify`,bindings:[F,{name:`giZScores`,view:m,type:`f32`,access:`read`},...x?[{name:`adjustedPValues`,view:B,type:`f32`,access:`read`}]:[],{name:`trendZ`,view:d.trendZ,type:`f32`,access:`read_write`},{name:`trendP`,view:d.trendP,type:`f32`,access:`read_write`},{name:`trendS`,view:d.trendS,type:`i32`,access:`read_write`},{name:`category`,view:d.category,type:`u32`,access:`read_write`}],workgroupSize:E?G:void 0,invocationCount:E?w*G:w,guardIndex:!E,declarations:E?`const CELL_COUNT: u32 = ${w}u;
const SLICE_COUNT: u32 = ${l}u;
const MANN_KENDALL_LANES: u32 = ${G}u;
var<workgroup> partialStatistics: array<i32, ${G}>;
var<workgroup> partialTieTerms: array<u32, ${G}>;
var<workgroup> sharedSeries: array<f32, ${l}>;
var<workgroup> tieCandidates: array<u32, ${l}>;
var<workgroup> equalBeforeBits: array<atomic<u32>, ${Math.ceil(l/32)}>;
${_}
${K(x)}`:`const SLICE_COUNT: u32 = ${l}u;
${_}
${K(x)}`,body:E?`let cell = index / MANN_KENDALL_LANES;
  let lane = localInvocationIndex;
  let isActiveCell = cell < CELL_COUNT;
  let base = cell * SLICE_COUNT;
  for (var slice = lane; slice < SLICE_COUNT; slice += MANN_KENDALL_LANES) {
    var value = getQuietNaN(index);
    if (isActiveCell) {
      value = giZScores[giZScoresOffset + base + slice];
    }
    sharedSeries[slice] = value;
    tieCandidates[slice] = 0u;
  }
  for (var word = lane; word < ${Math.ceil(l/32)}u; word += MANN_KENDALL_LANES) {
    atomicStore(&equalBeforeBits[word], 0u);
  }
  workgroupBarrier();
  var laneStatistic = 0;
  for (var first = lane; first < SLICE_COUNT; first += MANN_KENDALL_LANES) {
    let firstValue = sharedSeries[first];
    if (isFiniteFloat(firstValue)) {
      var groupSize = 1u;
      for (var second = first + 1u; second < SLICE_COUNT; second++) {
        let secondValue = sharedSeries[second];
        if (isFiniteFloat(secondValue)) {
          laneStatistic += select(select(0, -1, secondValue < firstValue), 1, secondValue > firstValue);
          if (secondValue == firstValue) {
            groupSize++;
            atomicOr(&equalBeforeBits[second / 32u], 1u << (second % 32u));
          }
        }
      }
      if (groupSize > 1u) {
        tieCandidates[first] = groupSize * (groupSize - 1u) * (2u * groupSize + 5u);
      }
    }
  }
  partialStatistics[lane] = laneStatistic;
  workgroupBarrier();
  var laneTieTerm = 0u;
  for (var first = lane; first < SLICE_COUNT; first += MANN_KENDALL_LANES) {
    let hasEqualBefore = (atomicLoad(&equalBeforeBits[first / 32u]) & (1u << (first % 32u))) != 0u;
    if (!hasEqualBefore) {
      laneTieTerm += tieCandidates[first];
    }
  }
  partialTieTerms[lane] = laneTieTerm;
  workgroupBarrier();
  for (var stride = MANN_KENDALL_LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      partialStatistics[lane] += partialStatistics[lane + stride];
      partialTieTerms[lane] += partialTieTerms[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && isActiveCell) {
  let statistic = partialStatistics[0];
  let tieTerm = partialTieTerms[0];
  ${oe({valueExpression:`sharedSeries[slice]`,adjustedExpression:`adjustedPValues[adjustedPValuesOffset + base + slice]`,usesAdjustment:x,collectSeries:!1})}
  ${se({cellName:`cell`,finalSlice:y})}
  }`:ce({sliceCount:l,usesAdjustment:x,finalSlice:y})})),ae}};function K(e){return e?`fn getSliceState(z: f32, adjustedP: f32, level: f32) -> u32 {
  if (adjustedP <= level) {
    if (z > 0.0) {
      return 1u;
    }
    if (z < 0.0) {
      return 2u;
    }
  }
  return 0u;
}`:`fn getSliceState(z: f32, criticalZ: f32) -> u32 {
  if (z >= criticalZ) {
    return 1u;
  }
  if (z <= -criticalZ) {
    return 2u;
  }
  return 0u;
}`}function oe(e){let t=e.usesAdjustment?`getSliceState(value, ${e.adjustedExpression}, significanceLevel)`:`getSliceState(value, criticalZ)`;return`let criticalZ = parameters[parametersOffset + 2u];
  let significanceLevel = parameters[parametersOffset + 6u];
  var valid = 0u;
  var hot = 0u;
  var cold = 0u;
  var trailingHot = 0u;
  var trailingCold = 0u;
  var finalState = 0u;
  var lastValid = 0xffffffffu;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = ${e.valueExpression};
    if (isFiniteFloat(value)) {
      ${e.collectSeries?`series[valid] = value;`:``}
      valid++;
      lastValid = slice;
      let state = ${t};
      if (state == 1u) {
        hot++;
        trailingHot++;
        trailingCold = 0u;
        finalState = 1u;
      } else if (state == 2u) {
        cold++;
        trailingCold++;
        trailingHot = 0u;
        finalState = 2u;
      } else {
        trailingHot = 0u;
        trailingCold = 0u;
        finalState = 0u;
      }
    }
  }`}function se(e){let{cellName:t}=e;return`let trendLevel = parameters[parametersOffset + 3u];
  let fraction = parameters[parametersOffset + 4u];
  let minimumTrendSliceCount = parameters[parametersOffset + 5u];
  var numerator = 0u;
  if (valid >= 2u) {
    numerator = valid * (valid - 1u) * (2u * valid + 5u) - tieTerm;
  }
  var trendZScore = 0.0;
  var trendPValue = 1.0;
  if (numerator > 0u) {
    let deviation = sqrt(f32(numerator) / 18.0);
    if (statistic > 0) {
      trendZScore = f32(statistic - 1) / deviation;
    } else if (statistic < 0) {
      trendZScore = f32(statistic + 1) / deviation;
    }
    trendPValue = getTwoSidedPValue(trendZScore);
  }
  // Too few finite slices for a Mann-Kendall decision: no trend z or p, and no trend category.
  let isTrendKnown = f32(valid) >= minimumTrendSliceCount;
  if (!isTrendKnown) {
    trendZScore = getQuietNaN(${t});
    trendPValue = getQuietNaN(${t});
  }
  trendZ[trendZOffset + ${t}] = trendZScore;
  trendP[trendPOffset + ${t}] = trendPValue;
  trendS[trendSOffset + ${t}] = statistic;
  let threshold = fraction * f32(valid);
  let hotPersistent = f32(hot) >= threshold;
  let coldPersistent = f32(cold) >= threshold;
  let trendSignificant = trendPValue <= trendLevel;
  let trendUp = trendSignificant && trendZScore > 0.0;
  let trendDown = trendSignificant && trendZScore < 0.0;
  var result = 0u;
  if (valid > 0u) {
    if (finalState == 1u) {
      if (hot == 1u) {
        result = 1u;
      } else if (trailingHot >= 2u && trailingHot == hot && !hotPersistent) {
        result = 2u;
      } else if (hotPersistent) {
        result = select(17u, select(select(4u, 5u, trendDown), 3u, trendUp), isTrendKnown);
      } else {
        result = select(7u, 6u, cold == 0u);
      }
    } else if (finalState == 2u) {
      if (cold == 1u) {
        result = 9u;
      } else if (trailingCold >= 2u && trailingCold == cold && !coldPersistent) {
        result = 10u;
      } else if (coldPersistent) {
        result = select(17u, select(select(12u, 13u, trendUp), 11u, trendDown), isTrendKnown);
      } else {
        result = select(15u, 14u, hot == 0u);
      }
    } else if (hot > 0u && hotPersistent) {
      result = 8u;
    } else if (cold > 0u && coldPersistent) {
      result = 16u;
    } else if (hot > 0u && cold == 0u) {
      result = 6u;
    } else if (cold > 0u && hot == 0u) {
      result = 14u;
    }
    ${e.finalSlice===`require`?`// A missing final slice makes the pattern undecidable instead of silently shifting it.
    if (lastValid != SLICE_COUNT - 1u) {
      result = 17u;
    }`:``}
  }
  category[categoryOffset + ${t}] = result;`}function ce(e){let{sliceCount:t}=e;return`let base = index * SLICE_COUNT;
  var series: array<f32, SLICE_COUNT>;
  var hasEqualBefore: array<u32, ${Math.ceil(t/32)}>;
  ${oe({valueExpression:`giZScores[giZScoresOffset + base + slice]`,adjustedExpression:`adjustedPValues[adjustedPValuesOffset + base + slice]`,usesAdjustment:e.usesAdjustment,collectSeries:!0})}
  var statistic = 0;
  var tieTerm = 0u;
  for (var first = 0u; first < valid; first++) {
    let firstValue = series[first];
    var groupSize = 1u;
    for (var second = first + 1u; second < valid; second++) {
      let secondValue = series[second];
      statistic += select(select(0, -1, secondValue < firstValue), 1, secondValue > firstValue);
      if (secondValue == firstValue) {
        groupSize++;
        hasEqualBefore[second / 32u] |= 1u << (second % 32u);
      }
    }
    let isFirstMember = (hasEqualBefore[first / 32u] & (1u << (first % 32u))) == 0u;
    if (isFirstMember && groupSize > 1u) {
      tieTerm += groupSize * (groupSize - 1u) * (2u * groupSize + 5u);
    }
  }
  ${se({cellName:`index`,finalSlice:e.finalSlice})}`}var q=`addSpaceTimeHotSpotsRecipe`;function le(e,t,n){let r=t<=2?e-1:e,i=Math.trunc((r>=0?r:r-399)/400),a=r-i*400,o=(t+9)%12,s=Math.trunc((153*o+2)/5)+n-1,c=a*365+Math.trunc(a/4)-Math.trunc(a/100)+s;return i*146097+c-719468}function J(e,t){let{year:n,month:r=1,day:i=1,hour:a=0}=t;switch(e){case`year`:return n;case`quarter`:return n*4+Math.trunc((r-1)/3);case`month`:return n*12+r-1;case`week`:case`day`:return le(n,r,i);case`hour`:return le(n,r,i)*24+a}}function Y(e){return(e.interval??1)*(e.step===`week`?7:1)}function ue(t,n){b(q,n.utcOffsets===void 0||n.timeZone===void 0,`pass utcOffsets or timeZone, not both`);let r=n.id??`space-time-hot-spots`,i=new S(t),a=n.outputs??{},o=n.scratch??{},{slices:s,cells:c}=n,l=c.kind===`lattice`?e(r,c.spatialContext):c.weights.spatialContext,p=n.timestamps.length,m=s.count;b(q,!(`field`in s),`slices.field is no longer accepted: cyclic calendar fields cannot carry a trend, use slices.step with an origin`),b(q,Number.isInteger(m)&&m>=1,`slices.count must be a positive integer`);let h=s.interval??1;b(q,Number.isInteger(h)&&h>=1,`slices.interval must be a positive integer`),b(q,[`year`,`quarter`,`month`,`week`,`day`,`hour`].includes(s.step),`slices.step must be year, quarter, month, week, day or hour`);let{origin:g}=s;for(let[e,t,n,r]of[[`year`,g.year,-1e6,1e6],[`month`,g.month??1,1,12],[`day`,g.day??1,1,31],[`hour`,g.hour??0,0,23]])b(q,Number.isInteger(t)&&t>=n&&t<=r,`slices.origin.${e} must be an integer in ${n}..${r}`);let _=c.kind===`ids`?c.cellCount:c.width*c.height,v=_*m;b(q,v<4294967294,`cellCount * sliceCount is too large for a u32 key`);let ee=s.step!==`year`,C=s.step===`week`||s.step===`day`||s.step===`hour`,w=s.step===`hour`,E=x(t,`${r}-calendar-year`,`sint32`,p,o.calendarYear),O=ee?x(t,`${r}-calendar-month`,`uint32`,p,o.calendarMonth):void 0,k=C?x(t,`${r}-calendar-day-of-month`,`uint32`,p,o.calendarDayOfMonth):void 0,A=w?x(t,`${r}-calendar-hour`,`uint32`,p,o.calendarHour):void 0;i.add(new T({id:`${r}-calendar`,timestamps:n.timestamps,mask:n.mask,utcOffsets:n.utcOffsets,timeZone:n.timeZone,parameters:n.calendarParameters,output:{year:E,...O?{month:O}:{},...k?{dayOfMonth:k}:{},...A?{hour:A}:{}}}));let j=c.kind===`ids`?[{name:`cellIds`,view:c.cellIds,type:`u32`,access:`read`}]:[{name:`eventPositions`,view:c.positions,type:`f32`,access:`read`}],M=c.kind===`ids`?`let cell = cellIds[cellIdsOffset + index];
  let cellValid = cell < ${_}u;`:`let px = eventPositions[eventPositionsOffset + 2u * index];
  let py = eventPositions[eventPositionsOffset + 2u * index + 1u];
  let fx = floor((px - ${u(c.bounds[0])}) / ${u((c.bounds[2]-c.bounds[0])/c.width)});
  let fy = floor((py - ${u(c.bounds[1])}) / ${u((c.bounds[3]-c.bounds[1])/c.height)});
  let cellValid = fx >= 0.0 && fy >= 0.0 && fx < ${c.width}.0 && fy < ${c.height}.0;
  let cell = select(0u, u32(fy) * ${c.width}u + u32(fx), cellValid);`,N=Y(s),te=J(s.step,g),P={year:`year`,quarter:`year * 4 + (month - 1) / 3`,month:`year * 12 + month - 1`,week:`daysFromCivil(year, month, day)`,day:`daysFromCivil(year, month, day)`,hour:`daysFromCivil(year, month, day) * 24 + hour`}[s.step],F=`${M}
  let year = calendarYear[calendarYearOffset + index];
  ${O?`let month = i32(calendarMonth[calendarMonthOffset + index]);`:``}
  ${k?`let day = i32(calendarDay[calendarDayOffset + index]);`:``}
  ${A?`let hour = i32(calendarHour[calendarHourOffset + index]);`:``}
  let isCalendarValid = year != bitcast<i32>(0x80000000u);
  let stepOffset = select(-1, ${P} - (${te}), isCalendarValid);
  let slice = stepOffset / ${N};
  let sliceValid = stepOffset >= 0 && slice < ${m};
  let valid = cellValid && sliceValid;`,I=[{name:`calendarYear`,view:E,type:`i32`,access:`read`},...O?[{name:`calendarMonth`,view:O,type:`u32`,access:`read`}]:[],...k?[{name:`calendarDay`,view:k,type:`u32`,access:`read`}]:[],...A?[{name:`calendarHour`,view:A,type:`u32`,access:`read`}]:[]],L=`fn daysFromCivil(year: i32, month: i32, day: i32) -> i32 {
  let adjustedYear = select(year, year - 1, month <= 2);
  let era = select(adjustedYear, adjustedYear - 399, adjustedYear < 0) / 400;
  let yearOfEra = adjustedYear - era * 400;
  let shiftedMonth = (month + 9) % 12;
  let dayOfYear = (153 * shiftedMonth + 2) / 5 + day - 1;
  let dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}`,R=x(t,`${r}-skipped-count`,`uint32`,1,a.status?.skippedEventCount);t.add(d(t,{id:`${r}-skipped-count-clear`,operation:`GPUSpaceTimeHotSpotsCube`,view:R,type:`u32`,value:`0u`})),t.add(f(t,{id:`${r}-skipped-count`,operation:`GPUSpaceTimeHotSpotsCube`,variant:`slice-range`,bindings:[...I,...j,{name:`skippedOutput`,view:R,type:`atomic<u32>`,access:`read_write`}],invocationCount:p,declarations:L,body:`${F}
  if (cellValid && isCalendarValid && !sliceValid) {
    atomicAdd(&skippedOutput[skippedOutputOffset], 1u);
  }`}));let z=x(t,`${r}-cube`,`uint32`,v,a.cube);t.add(d(t,{id:`${r}-cube-clear`,operation:`GPUSpaceTimeHotSpotsCube`,view:z,type:`u32`,value:`0u`}));let B=o.keys!==void 0||o.keyMask!==void 0||o.groupKeys!==void 0||o.groupCounts!==void 0||o.groupCount!==void 0||a.status?.groupOverflow!==void 0,V={calendarYear:E,...O?{calendarMonth:O}:{},...k?{calendarDayOfMonth:k}:{},...A?{calendarHour:A}:{}},H={stages:[{stage:`slice-range`,status:{invalidCount:R}}]};if(!B)t.add(f(t,{id:`${r}-cube-count`,operation:`GPUSpaceTimeHotSpotsCube`,variant:`atomic-count`,bindings:[...I,...j,{name:`cubeOutput`,view:z,type:`atomic<u32>`,access:`read_write`}],invocationCount:p,declarations:L,body:`${F}
  if (valid) {
    atomicAdd(&cubeOutput[cubeOutputOffset + cell * ${m}u + u32(slice)], 1u);
  }`}));else{let e=x(t,`${r}-keys`,`uint32`,p,o.keys),n=x(t,`${r}-key-mask`,`uint32`,p,o.keyMask);t.add(f(t,{id:`${r}-keys`,operation:`GPUSpaceTimeHotSpotsKeys`,bindings:[...I,...j,{name:`keyOutput`,view:e,type:`u32`,access:`read_write`},{name:`keyMaskOutput`,view:n,type:`u32`,access:`read_write`}],invocationCount:p,declarations:L,body:`${F}
  keyOutput[keyOutputOffset + index] = select(0xffffffffu, cell * ${m}u + u32(slice), valid);
  keyMaskOutput[keyMaskOutputOffset + index] = select(0u, 1u, valid);`}));let s=x(t,`${r}-group-keys`,`uint32`,v,o.groupKeys),c=x(t,`${r}-group-counts`,`uint32`,v,o.groupCounts),l=x(t,`${r}-group-count`,`uint32`,1,o.groupCount),u=x(t,`${r}-group-overflow`,`uint32`,1,a.status?.groupOverflow),d=x(t,`${r}-group-required-count`,`uint32`,1,a.status?.groupRequiredCount);i.add(new D({id:`${r}-counts`,keys:e,mask:n,columns:[],output:{keys:s,counts:c,status:{count:l,overflow:u,requiredCount:d}}})),t.add(f(t,{id:`${r}-cube-scatter`,operation:`GPUSpaceTimeHotSpotsCube`,variant:`scatter`,bindings:[{name:`groupKeys`,view:s,type:`u32`,access:`read`},{name:`groupCounts`,view:c,type:`u32`,access:`read`},{name:`groupCountScalar`,view:l,type:`u32`,access:`read`},{name:`cubeOutput`,view:z,type:`u32`,access:`read_write`}],invocationCount:Math.min(v,p),body:`if (index < groupCountScalar[groupCountScalarOffset]) {
    let key = groupKeys[groupKeysOffset + index];
    if (key < ${v}u) {
      cubeOutput[cubeOutputOffset + key] = groupCounts[groupCountsOffset + index];
    }
  }`})),V={...V,keys:e,keyMask:n,groupKeys:s,groupCounts:c,groupCount:l},H={stages:[...H.stages,{stage:`cube-counts`,status:{count:l,overflow:u,requiredCount:d}}]}}b(q,n.parameters.length>=8,`parameters needs 8 values`);let U=n.parameters.importToGraph(t,`${r}-parameters`),W={giZScores:x(t,`${r}-gi-z`,`float32`,v,a.giZScores),trendZ:x(t,`${r}-trend-z`,`float32`,_,a.trendZ),trendP:x(t,`${r}-trend-p`,`float32`,_,a.trendP),trendS:x(t,`${r}-trend-s`,`sint32`,_,a.trendS),category:x(t,`${r}-category`,`uint32`,_,a.category),hotSliceCount:x(t,`${r}-hot-slices`,`uint32`,_,a.hotSliceCount),coldSliceCount:x(t,`${r}-cold-slices`,`uint32`,_,a.coldSliceCount)},G={...a.pValues?{pValues:a.pValues}:{},...a.adjustedPValues?{adjustedPValues:a.adjustedPValues}:{},...a.lastValidSlice?{lastValidSlice:a.lastValidSlice}:{}},re=i.add(new ae({id:`${r}-emerging`,values:z,...c.kind===`ids`?{weights:c.weights,selfWeight:c.selfWeight}:{gridWidth:c.width,gridHeight:c.height,maximumRadius:c.maximumRadius},sliceCount:m,mask:n.cellMask,parameters:U,multipleTesting:n.multipleTesting,finalSlice:n.finalSlice,output:{...W,...G}})),ie=e=>{let t=re.getParameterValues(e),r=re.getInferenceMetadata(e);return n.parameters.write(t),r},K={contributors:i.contributors,spatialContext:l,cellCount:_,sliceCount:m,cube:z,...W,...G,outputs:{cube:z,...W,...G},intermediates:V,status:H,inference:ie(n.settings),setSettings:e=>{K.inference=ie(e)}};if(n.color){let{color:e}=n,s=e.maximumPaletteCount??Object.keys(ne).length,c=x(t,`${r}-colors`,`uint32`,_,a.colors);i.add(new y({id:`${r}-color-scale`,values:W.category,domain:x(t,`${r}-color-domain`,`float32`,2,o.colorDomain),palette:e.palette,parameters:e.parameters,maximumDomainCount:2,maximumPaletteCount:s,output:{colors:c}})),K.colors=c,K.outputs.colors=c}return K}var de=`
fn emergingCategoryColor(category: u32) -> vec4<f32> {
  let colors = array<vec3<f32>, ${I.length}>(
    ${I.map(e=>`vec3<f32>(${e[0].toFixed(1)}, ${e[1].toFixed(1)}, ${e[2].toFixed(1)})`).join(`, `)}
  );
  if (category == 0u || category >= ${I.length}u) {
    return vec4<f32>(0.0);
  }
  return vec4<f32>(colors[category] / 255.0, 0.9);
}
`,fe=`return spatialAnalysisStyle.palette[raw % size];`,pe=class extends k{static layerName=`EmergingCategoryRasterLayer`;getShaderSource(){let e=super.getShaderSource();if(!e.includes(fe))throw Error(`EmergingCategoryRasterLayer: the shared style shader changed; update it`);return`${e.replace(fe,`return emergingCategoryColor(raw);`)}\n${de}`}},X=500,me=4,he=7*86400,ge=117*86400,_e=121*86400,Z=[`Jan`,`Feb`,`Mar`,`Apr`,`May`,`Jun`,`Jul`,`Aug`,`Sep`,`Oct`,`Nov`,`Dec`],Q=5,ve={months:{sliceCount:F.months,slices:{step:`month`,origin:{year:2023,month:1},count:12}},days:{sliceCount:F.days,slices:{step:`day`,interval:Q,origin:{year:2023,month:1,day:1},count:73}},weeks:{sliceCount:F.weeks,slices:null}};function ye(e,t){if(e===`months`)return`${Z[t]??`?`} 2023`;if(e===`days`){let e=new Date(Date.UTC(2023,0,1+t*Q)),n=new Date(Date.UTC(2023,0,Q+t*Q)),r=e=>`${Z[e.getUTCMonth()]} ${e.getUTCDate()}`;return`Days ${t*Q+1}-${(t+1)*Q}: ${r(e)} to ${r(n)}`}let n=new Date(Date.UTC(2023,0,1+t*7)),r=new Date(Date.UTC(2023,0,7+t*7)),i=e=>`${Z[e.getUTCMonth()]} ${e.getUTCDate()}`;return`Week ${t+1}: ${i(n)} to ${i(r)}`}async function be(e){let{device:n}=e,r=w(e.datasets.get(`chicago-nature`)),i=r.origin,a=e.datasets.get(`chicago-nature`).getProjection(i),c=r.count,l=1/0,u=1/0,d=-1/0,f=-1/0;for(let e=0;e<c;e++){let t=r.positions[e*2],n=r.positions[e*2+1];l=Math.min(l,t),d=Math.max(d,t),u=Math.min(u,n),f=Math.max(f,n)}let p=Math.ceil((d-l)/X+1e-6),m=Math.ceil((f-u)/X+1e-6),h=p*m,g=[l,u,l+p*X,u+m*X],_=new Uint32Array(c),y=new Uint32Array(h);for(let e=0;e<c;e++){let t=Math.min(p-1,Math.floor((r.positions[e*2]-l)/X)),n=Math.min(m-1,Math.floor((r.positions[e*2+1]-u)/X))*p+t;_[e]=n,y[n]++}let b=new Float32Array(h*2);for(let e=0;e<m;e++)for(let t=0;t<p;t++){let n=e*p+t;b[n*2]=l+(t+.5)*X,b[n*2+1]=u+(e+.5)*X}let x=new s(n,`emerging`),S=x.track(new N),T=x.createBuffer(`positions`,r.positions),D=x.createBuffer(`time-words`,r.timeWords),F=x.createBuffer(`seconds`,r.seconds),ne=x.createBuffer(`ones`,new Float32Array(c).fill(1)),z=x.createBuffer(`cell-ids`,_),B=x.createBuffer(`cell-centers`,b),V=x.createBuffer(`event-mask`,c*4),H=x.createBuffer(`cell-mask`,h*4),U=x.createBuffer(`slice-index`,h*4),W=h*(Math.ceil(Math.PI*me**2)+8),G=x.createBuffer(`offsets`,(h+1)*4),re=x.createBuffer(`neighbors`,W*4),ie=x.createBuffer(`weights`,W*4),K=x.createBuffer(`distances`,W*4),oe=x.createParameterBuffer(`calendar-parameters`,`sint32`,2,C(0,0)),se=x.createParameterBuffer(`reduction-parameters`,`float32`,2,A(0,he)),ce=x.createParameterBuffer(`hot-spot-parameters`,`float32`,8),q=x.createParameterBuffer(`search-parameters`,`float32`,12),le=!1,J=!0,Y=0,de=0,fe=0,Z=null,be=!1,$=null,xe=null,Se=null,Ce=null,we=null,Te=null,Ee=1,De=new Map,Oe=(e,t,n,r,i)=>M(e,t,n,r,i);async function ke(e,r,i){let o=ve[e],s=o.sliceCount,l=h*s,u=`${e}-${r}-${i}`,d=(e,t)=>x.createBuffer(`${u}-${e}`,t),f=d(`counts`,l*4),_=d(`gi-z`,l*4),v=d(`trend-z`,h*4),y=d(`trend-p`,h*4),b=d(`trend-s`,h*4),S=d(`category`,h*4),C=d(`hot-slices`,h*4),w=d(`cold-slices`,h*4),E=d(`overflow`,4),O=new t(n,{id:`emerging-${u}`}),k=new Map,A=(e,t,n,r)=>{let i=k.get(e);return i||(i=Oe(O,`${u}-${e}`,t,n,r),k.set(e,i)),i},M;r===`weights`&&(M={spatialContext:a.spatialContext,offsets:A(`offsets`,G,`uint32`,h+1),neighbors:A(`neighbors`,re,`uint32`,W),weights:A(`weights`,ie,`float32`,W),distances:A(`distances`,K,`float32`,W),status:{overflow:A(`overflow`,E,`uint32`,1)}},O.add(new ee({spatialContext:a.spatialContext,id:`${u}-search`,mode:`radius`,positions:A(`cell-centers`,B,`float32x2`,h),mask:A(`cell-mask`,H,`uint32`,h),parameters:q.importToGraph(O),gridSize:[64,64],output:M})));let N=null,P,I,L=Fe();if(e===`weeks`){let e=d(`occupied-ids`,l*4);N=d(`occupied-count`,4);let t=d(`occupied-overflow`,4);O.add(new j({id:`${u}-reduction`,cellIds:A(`cell-ids`,z,`uint32`,c),timestamps:A(`seconds`,F,`float32`,c),values:A(`ones`,ne,`float32`,c),mask:A(`event-mask`,V,`uint32`,c),parameters:se.importToGraph(O),cellCount:h,bucketCount:s,output:{counts:A(`counts`,f,`uint32`,l),minimums:A(`min`,d(`min`,l*4),`float32`,l),maximums:A(`max`,d(`max`,l*4),`float32`,l),first:A(`first`,d(`first`,l*4),`float32`,l),last:A(`last`,d(`last`,l*4),`float32`,l),occupiedSlots:{ids:A(`occupied-ids`,e,`uint32`,l),status:{count:A(`occupied-count`,N,`uint32`,1),overflow:A(`occupied-overflow`,t,`uint32`,1)}}}}));let n=new ae({id:`${u}-emerging`,values:A(`counts`,f,`uint32`,l),...M?{weights:M}:{gridWidth:p,gridHeight:m,maximumRadius:me},sliceCount:s,mask:A(`cell-mask`,H,`uint32`,h),parameters:ce.importToGraph(O),multipleTesting:i,output:{giZScores:A(`gi-z`,_,`float32`,l),trendZ:A(`trend-z`,v,`float32`,h),trendP:A(`trend-p`,y,`float32`,h),trendS:A(`trend-s`,b,`sint32`,h),category:A(`category`,S,`uint32`,h),hotSliceCount:A(`hot-slices`,C,`uint32`,h),coldSliceCount:A(`cold-slices`,w,`uint32`,h)}});O.add(n),I=e=>(ce.write(n.getParameterValues(e)),n.getInferenceMetadata(e)),P=I(L)}else{let e=ue(O,{id:`${u}-recipe`,timestamps:A(`time-words`,D,`uint32x2`,c),mask:A(`event-mask`,V,`uint32`,c),calendarParameters:oe.importToGraph(O),slices:o.slices,multipleTesting:i,cells:M?{kind:`ids`,cellIds:A(`cell-ids`,z,`uint32`,c),cellCount:h,weights:M,selfWeight:1}:{kind:`lattice`,positions:A(`positions`,T,`float32x2`,c),width:p,height:m,bounds:g,maximumRadius:me},cellMask:A(`cell-mask`,H,`uint32`,h),settings:L,parameters:ce,outputs:{cube:A(`counts`,f,`uint32`,l),giZScores:A(`gi-z`,_,`float32`,l),trendZ:A(`trend-z`,v,`float32`,h),trendP:A(`trend-p`,y,`float32`,h),trendS:A(`trend-s`,b,`sint32`,h),category:A(`category`,S,`uint32`,h),hotSliceCount:A(`hot-slices`,C,`uint32`,h),coldSliceCount:A(`cold-slices`,w,`uint32`,h)}});I=t=>(e.setSettings(t),e.inference),P=e.inference}let R=x.track(await O.compileAsync()),U=new te(x,u,[{buffer:S,size:h*4},{buffer:v,size:h*4},{buffer:C,size:h*4},{buffer:w,size:h*4},{buffer:f,size:l*4}],e=>{if(le||Z?.counts!==f)return;let t=h*4;$=new Uint32Array(e,0,h),xe=new Float32Array(e,t,h),Se=new Uint32Array(e,t*2,h),Ce=new Uint32Array(e,t*3,h),we=new Uint32Array(e,t*4,l),ze()});return{compiled:R,sliceCount:s,counts:f,giZ:_,trendZ:v,trendP:y,category:S,hotSlices:C,coldSlices:w,overflow:E,occupiedCount:N,inference:P,applySettings:I,reader:U}}async function Ae(){let{cube:t,neighborhood:n,multipleTesting:r}=e.options,i=`${t}-${n}-${r}`,a=De.get(i);return a||(a=await ke(t,n,r),De.set(i,a)),a}function je(){let t=new Uint32Array(c);fe=E(r,e.options.groupType,t);let n=0;for(let e=0;e<c;e++)t[e]&&r.seconds[e]>=ge&&r.seconds[e]<_e&&n++;V.write(t),e.setReadout(`events`,`${o(fe)} of ${o(c)}`),e.setReadout(`challengePulse`,`${o(n)} records (${(100*n/Math.max(1,fe)).toFixed(1)}%)`),J=!0}function Me(){let t=new Uint32Array(h);de=0;for(let n=0;n<h;n++){let r=y[n]>=e.options.minimumEvents?1:0;t[n]=r,de+=r}H.write(t),e.setReadout(`cells`,`${o(de)} of ${o(h)}`),Ne(),J=!0}function Ne(){let t=Z?.sliceCount??ve[e.options.cube].sliceCount;e.setReadout(`tests`,`${o(de*t)} Gi* bins (${R[e.options.multipleTesting]})`)}function Pe(){e.setLegendData(`inference`,Z?.inference)}function Fe(){let t=e.options,n=Number(t.confidence);return{radius:t.radius,temporalWindow:t.temporalWindow,confidenceLevel:n,trendSignificanceLevel:t.tieTrend?1-n:t.trendLevel,persistentFraction:t.persistentFraction}}function Ie(){let t=e.options;Z&&(Z.inference=Z.applySettings(Fe()),Pe()),q.write(v({bounds:[g[0]-X,g[1]-X,g[2]+X,g[3]+X],radius:t.radius*X,weightKind:t.weightKind===`inverse-distance`?`inverseDistance`:t.weightKind===`kernel`?`kernel`:`binary`,power:1,distanceFloor:X/2,kernel:`triangular`})),J=!0}function Le(t){let n=Z?.sliceCount??1,r=Math.min(t,n-1),i=new Uint32Array(h);for(let e=0;e<h;e++)i[e]=e*n+r;U.write(i),Y=r,e.setReadout(`sliceLabel`,ye(e.options.cube,r)),Re()}function Re(){if(!Te){e.setChart(`sliceTimeline`,null);return}let t=Te.map((e,t)=>t+1);e.setChart(`sliceTimeline`,{kind:`timeline`,title:`Observations by time slice`,x:t,y:Te,mode:`area`,playhead:Y+1,xLabel:`time slice`,yLabel:`observations`,formatX:t=>ye(e.options.cube,Math.round(t)-1),link:{option:`slice`,label:t=>ye(e.options.cube,Math.round(t)-1)},description:`Records in every time slice, summed across the active study-area cells. Dates use the source Chicago local wall clock; the playhead is the slice shown on the map.`})}function ze(){if(!$||!we||!Z)return;let t=Array(L.length).fill(0);for(let e=0;e<h;e++)$[e]<t.length&&t[$[e]]++;let n=(e,n)=>t.slice(e,n+1).reduce((e,t)=>e+t,0);e.setReadout(`hot`,n(1,8)),e.setReadout(`cold`,n(9,16)),e.setReadout(`undetermined`,t[17]),e.setReadout(`new`,t[1]),e.setReadout(`intensifying`,t[3]),e.setReadout(`persistent`,t[4]),e.setReadout(`diminishing`,t[5]),e.setReadout(`sporadic`,t[6]),e.setChart(`categoryProfile`,{kind:`bars`,title:`Cells by emerging hot-spot category`,values:t.slice(1),labels:L.slice(1),colors:I.slice(1),horizontal:!0,table:!1,description:`Counts of active cells in each of the eight hot and eight cold emerging-pattern categories, plus cells whose category cannot be decided (final slice missing, or too few slices for a trend).`});let r=Z.sliceCount,i=Array(r).fill(0),a=0;Ee=1;for(let t=0;t<h;t++)if(!(y[t]<e.options.minimumEvents))for(let e=0;e<r;e++){let n=we[t*r+e];i[e]+=n,a+=n,Ee=Math.max(Ee,n)}Te=i,Re(),e.setTimelineData({domain:[1,r],histogram:i,events:e.options.cube===`weeks`?[{at:17.7,label:`City Nature Challenge, 28 Apr–1 May`}]:e.options.cube===`months`?[{at:4,label:`City Nature Challenge, 28 Apr–1 May`}]:[{at:117/Q+1,label:`City Nature Challenge, 28 Apr–1 May`}]}),e.setReadout(`cubeEvents`,`${o(a)} observations in ${o(de*r)} active bins`);let s=0;for(let e=1;e<r;e++)i[e]>i[s]&&(s=e);e.setReadout(`busiest`,`${ye(e.options.cube,s)} (${o(i[s])})`),e.setLegendExtent(`count`,[0,Ee]),e.requestLayers()}je(),Me(),Ie(),Z=await Ae(),Pe();let Be=P(e,{time:`slice`,play:`play`},{range:[1,Z.sliceCount],step:1,rate:1/.45});return Ne(),Le(e.options.slice-1),e.setReadout(`grid`,`${p} x ${m} cells of ${X} m`),{getCompiledGraphs:()=>Z?[Z.compiled]:[],setOption(t,n,r){t===`cube`||t===`neighborhood`||t===`multipleTesting`?(t===`cube`&&(be=!0),S.request({build:Ae,commit:t=>{Z=t,Be.setRange(1,t.sliceCount),Ne(),Ie(),be&&(be=!1,Te=null,e.setChart(`sliceTimeline`,null)),Le(e.options.slice-1),$=null,J=!0,e.requestLayers()}})):t===`groupType`?je():t===`minimumEvents`?Me():t===`slice`?Le(r.slice-1):t===`radius`||t===`weightKind`||t===`temporalWindow`||t===`confidence`||t===`tieTrend`||t===`trendLevel`||t===`persistentFraction`?Ie():e.requestLayers()},onThemeChange(){e.requestLayers()},onCompareChange(){e.requestLayers()},getTooltip({coordinate:t}){if(!t||!$||!we||!Z)return null;let[n,r]=e.datasets.get(`chicago-nature`).getProjection(i).project(t[0],t[1]),a=Math.floor((n-l)/X),s=Math.floor((r-u)/X);if(a<0||s<0||a>=p||s>=m)return null;let c=s*p+a;if(!Ve(c))return null;let d=Z.sliceCount,f=0;for(let e=0;e<d;e++)f+=we[c*d+e];let h=we[c*d+Y];return[L[$[c]]??`Unknown`,`${o(f)} observations in 2023, ${o(h)} in ${ye(e.options.cube,Y)}`,`Hot in ${Se?.[c]??0} and cold in ${Ce?.[c]??0} of ${d} slices`,`Mann-Kendall trend z = ${(xe?.[c]??0).toFixed(2)}`].join(`
`)},encode(e,t){if(!Z)return;let n=Math.round(Be.advance(t))-1;Y!==Math.min(n,Z.sliceCount-1)&&Le(n),(J||t.frameIndex<3)&&(Z.compiled.encode(e,{parameters:void 0}),J=!1,Z.reader.request(e)),Z.reader.flush(e)},getLayers(){if(!Z)return[];let t=e.options,n=[i[0],i[1],0],r=e.theme()===`dark`,a=[];t.showEvents&&a.push(new O({id:`emerging-events`,coordinateOrigin:n,positions:T,instanceCount:c,values:V,valueFormat:`uint32`,colormap:`mask`,color:r?[190,215,255,70]:[40,60,110,70],noDataColor:[0,0,0,0],radiusPixels:1}));let o={coordinateOrigin:n,gridSize:[p,m],bounds:g,opacity:t.opacity};switch(t.mapView){case`category`:e.getCompare()&&a.push(new k({...o,id:`emerging-category-compare-events`,values:Z.counts,valueIndices:U,valueFormat:`uint32`,colormap:`magma`,valueRange:[0,Ee],sqrtScale:!0,discardAtOrBelow:0,color:[255,255,255,255],compareSide:`a`})),a.push(new pe({...o,id:`emerging-category`,values:Z.category,valueFormat:`uint32`,colormap:`category`,color:[255,255,255,255],...e.getCompare()?{compareSide:`b`}:{}}));break;case`gi`:a.push(new k({...o,id:`emerging-gi`,values:Z.giZ,valueIndices:U,valueFormat:`float32`,colormap:`diverging`,valueRange:[-4,4],color:[255,255,255,255]}));break;case`trend`:a.push(new k({...o,id:`emerging-trend`,values:Z.trendZ,valueFormat:`float32`,colormap:`diverging`,valueRange:[-4,4],color:[255,255,255,255]}));break;case`hot-count`:case`cold-count`:a.push(new k({...o,id:`emerging-${t.mapView}`,values:t.mapView===`hot-count`?Z.hotSlices:Z.coldSlices,valueFormat:`uint32`,colormap:t.mapView===`hot-count`?`inferno`:`cividis`,valueRange:[0,Z.sliceCount],discardAtOrBelow:0,color:[255,255,255,255]}));break;case`events`:a.push(new k({...o,id:`emerging-events-cube`,values:Z.counts,valueIndices:U,valueFormat:`uint32`,colormap:`magma`,valueRange:[0,Ee],sqrtScale:!0,discardAtOrBelow:0,color:[255,255,255,255]}));break}return a},destroy(){le=!0;for(let e of De.values())e.reader.stop();x.destroy()}};function Ve(t){return y[t]>=e.options.minimumEvents}}export{be as createEmergingHotSpots};