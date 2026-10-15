import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{E as t,h as n,i as r,t as i}from"./analysis-frame-Blu0tYdw.js";import{l as a,s as o}from"./permutation-inference-kernels-D-xhEewi.js";import{_ as ee,a as s,b as c,c as l,d as u,f as d,l as f,m as p,n as m,o as te,r as h,s as g,t as ne,u as _}from"./gpu-geographically-weighted-regression-BYSqsAQL.js";import{N as v,_ as y,d as b,h as x,t as S,w as re,y as C}from"./resources-CwvhEOBe.js";import{s as w}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{r as T,s as E}from"./wgsl-common-BNRVzgd8.js";import{t as D}from"./philox-random-CHwyk0dd.js";import{t as ie}from"./permutation-random-C7SSHCbB.js";import{t as ae}from"./permutation-shuffle-B9oRDb5C.js";import{c as O}from"./index-Bjk18V2I.js";import{t as k}from"./graph-buffers-B4_80RlX.js";import{t as A}from"./graph-swap-41Ru7LVu.js";import{_ as oe,c as j,f as se,g as ce,h as M,l as N,m as le,n as P,o as ue,p as F,r as I,s as L,t as de,y as fe}from"./b6-polygons-Bf2djBQx.js";import{n as pe,t as R}from"./b6-fill-layer-CwuOmgDO.js";function me(e){let{summary:t}=e;if(t.length<12)throw Error(`GWR summary must hold 12 values`);let n=te,r=t[n.HAS_VALID_CANDIDATE]>.5,i=t[n.TRACE_OF_HAT],a=c({kind:`geographically-weighted-regression`,coefficients:[],rowCount:t[n.OBSERVATION_COUNT],residualSumOfSquares:t[n.RESIDUAL_SUM_OF_SQUARES],standardErrorSigmaSquared:`degrees-of-freedom`,statusCode:r?0:1,rSquared:t[n.R_SQUARED],logLikelihood:r?t[n.LOG_LIKELIHOOD]:null,parameterCount:i,informationCriterionParameterCount:i+1,correctedAkaikeInformationCriterion:r?t[n.AICC]:null}),o=e.selectedBandwidth;return{...a,effectiveParameterCount:i,singularRowCount:t[n.SINGULAR_ROW_COUNT],includedRowCount:t[n.INCLUDED_ROW_COUNT],adjustedAlpha:t[n.ADJUSTED_ALPHA],criticalT:t[n.CRITICAL_T],selectedBandwidth:o?{index:o[0],value:o[1]}:null}}var he={observedStandardDeviation:0,pseudoPValue:1,exceedances:2,simulatedMean:3,simulatedStandardDeviation:4,zSimulated:5},z={permutationCount:0,locationCount:1,observedFitCount:2,failedFitCount:3,length:4},ge=Object.freeze({...n,measure:`distance`}),B=`GPUGeographicallyWeightedRegressionNonstationarityTest`,V=256,_e=1048576,ve=16777216,ye=4096,be=8,H=64,xe=class{id;props;rowCount;coefficientCount;maximumBandwidthCount;maximumNeighborCount;tileRowCount;tileCount;indexGridSize;constructor(e){let n=e.id??`gwr-nonstationarity-test`;this.id=n,this.props=e;let{predictorCount:r,output:i,maximumPermutations:a}=e;if(t(n,e.spatialContext,ge),b(n,`predictorCount`,r,1,7),b(n,`maximumPermutations`,a,1,2**20),this.coefficientCount=r+1,this.maximumBandwidthCount=e.maximumBandwidthCount??32,b(n,`maximumBandwidthCount`,this.maximumBandwidthCount,1,32),this.maximumNeighborCount=e.maximumNeighborCount??128,b(n,`maximumNeighborCount`,this.maximumNeighborCount,1,128),C(e.positions,[`float32x2`],`${n} positions`),this.rowCount=e.positions.length,this.rowCount<2||this.rowCount>_e)throw Error(`${n} needs between 2 and ${_e} rows`);if(e.indexGridSize===!1)this.indexGridSize=null;else if(e.indexGridSize){let[t,r]=e.indexGridSize;if(!Number.isInteger(t)||!Number.isInteger(r)||t<1||r<1||t*r>=4294967295)throw Error(`${n} indexGridSize must be two positive integers with columns * rows < 2^32 - 1`);this.indexGridSize=[t,r]}else if(this.rowCount>=1024){let e=Math.min(h,Math.max(1,Math.ceil(Math.sqrt(this.rowCount/8))));this.indexGridSize=[e,e]}else this.indexGridSize=null;if(C(e.predictors,[`float32`],`${n} predictors`),e.predictors.length!==this.rowCount*r)throw Error(`${n} predictors length must equal rows * predictorCount`);if(C(e.response,[`float32`],`${n} response`),e.response.length!==this.rowCount)throw Error(`${n} response length must equal the row count`);if(e.mask&&(C(e.mask,[`uint32`],`${n} mask`),e.mask.length!==this.rowCount))throw Error(`${n} mask length must equal the row count`);C(e.bandwidthParameters,[`float32`],`${n} bandwidthParameters`);let o=g(this.maximumBandwidthCount);if(e.bandwidthParameters.length<o)throw Error(`${n} bandwidthParameters must hold ${o} float32 values`);if(C(e.selectedBandwidth,[`float32`],`${n} selectedBandwidth`),e.selectedBandwidth.length<2)throw Error(`${n} selectedBandwidth must hold 2 float32 values`);if(C(e.coefficients,[`float32`],`${n} coefficients`),e.coefficients.length<this.rowCount*this.coefficientCount)throw Error(`${n} coefficients must hold rows * (predictorCount + 1) values`);if(C(e.parameters,[`uint32`],`${n} parameters`),e.parameters.length<4)throw Error(`${n} parameters must hold 4 uint32`);let ee=this.coefficientCount*6;if(C(i.table,[`float32`],`${n} output.table`),i.table.length<ee)throw Error(`${n} output.table must hold ${ee} float32 values`);if(C(i.summary,[`float32`],`${n} output.summary`),i.summary.length<z.length)throw Error(`${n} output.summary must hold ${z.length} float32 values`);if(i.standardDeviations){C(i.standardDeviations,[`float32`],`${n} output.standardDeviations`);let e=(a+1)*this.coefficientCount;if(i.standardDeviations.length<e)throw Error(`${n} output.standardDeviations must hold ${e} float32 values`)}let s=2*this.coefficientCount+1,c=a+1;if(e.tileRowCount!==void 0)b(n,`tileRowCount`,e.tileRowCount),this.tileRowCount=e.tileRowCount;else{let e=Math.max(1,Math.min(ye,Math.floor(ve/(c*s))));this.tileRowCount=Math.max(be,Math.ceil(this.rowCount/e))}if(this.tileCount=Math.ceil(this.rowCount/this.tileRowCount),this.tileCount*c*s>ve*4)throw Error(`${n} tileRowCount is too small for the row and permutation counts`);if(this.tileCount*a>=2**31)throw Error(`${n} tileCount * maximumPermutations must stay below 2^31`);x(n,[i.table,i.summary,i.standardDeviations],[e.positions,e.predictors,e.response,e.mask,e.bandwidthParameters,e.selectedBandwidth,e.coefficients,e.parameters])}getCommandNodes(e){let{id:t,props:n,rowCount:r,coefficientCount:i,maximumBandwidthCount:a,maximumNeighborCount:ee,tileRowCount:s,tileCount:c,indexGridSize:l}=this,{output:d,predictorCount:te,maximumPermutations:h,mask:g}=n,ne=Math.ceil(c/H);y(t,e,[n.positions,n.predictors,n.response,g,n.bandwidthParameters,n.selectedBandwidth,n.coefficients,n.parameters,d.table,d.summary,d.standardDeviations]);let v=te+4,b=2*i+1,x=Math.ceil(r/V),S=(n,r,i)=>re(e,`${t}-${n}`,r,Math.max(i,1)),C=S(`row-valid`,`uint32`,r),T=S(`tile-counts`,`uint32`,x),E=S(`tile-offsets`,`uint32`,x+1),O=S(`rows`,`float32`,r*v+1),k=l?re(e,`${t}-compact-positions`,`float32x2`,Math.max(r,1)):null,A=l&&k?f(e,{id:t,operation:B,positions:k,rowCount:r,gridSize:l}):null,oe=S(`shifts`,`float32`,i),j=S(`partial`,`float32`,(h+1)*c*b),se=S(`fit-counts`,`float32`,h+1),ce=d.standardDeviations??S(`standard-deviations`,`float32`,(h+1)*i),M=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),N=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),le=A?[M(`cellOffsets`,A.cellOffsets,`u32`),M(`sortedIds`,A.sortedIds,`u32`)]:[],P=`const ROW_COUNT: u32 = ${r}u;
const PREDICTOR_COUNT: u32 = ${te}u;
const P: u32 = ${i}u;
const ROW_STRIDE: u32 = ${v}u;
const PARTIAL_STRIDE: u32 = ${b}u;
const TILE_ROWS: u32 = ${s}u;
const TILE_COUNT: u32 = ${c}u;
const SCAN_TILE_ROWS: u32 = ${V}u;
const SCAN_TILE_COUNT: u32 = ${x}u;
const LADDER: u32 = ${a}u;
const MAXIMUM_NEIGHBORS: u32 = ${ee}u;
const HEADER: u32 = 4u;
const SENTINEL: f32 = 3.0e38;
const ADAPTIVE_FACTOR: f32 = ${m};
${Se}`,ue=`${P}
${o(h)}`,F=l?_(l):``,I=r*v;return[w(e,{id:`${t}-validate`,operation:B,variant:`validate`,bindings:[M(`positions`,n.positions,`f32`),M(`predictors`,n.predictors,`f32`),M(`response`,n.response,`f32`),...g?[M(`rowMask`,g,`u32`)]:[],N(`rowValid`,C,`u32`)],invocationCount:r,declarations:P,body:`var isValid = true;
  ${g?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  if (!isFiniteBits(positions[positionsOffset + 2u * index]) ||
    !isFiniteBits(positions[positionsOffset + 2u * index + 1u]) ||
    !isFiniteBits(response[responseOffset + index])) {
    isValid = false;
  }
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + index * PREDICTOR_COUNT + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);`}),w(e,{id:`${t}-tile-counts`,operation:B,variant:`tile-counts`,bindings:[M(`rowValid`,C,`u32`),N(`tileCounts`,T,`u32`)],invocationCount:x,declarations:P,body:`let firstRow = index * SCAN_TILE_ROWS;
  let endRow = min(firstRow + SCAN_TILE_ROWS, ROW_COUNT);
  var count = 0u;
  for (var row = firstRow; row < endRow; row++) {
    count += select(0u, 1u, rowValid[rowValidOffset + row] != 0u);
  }
  tileCounts[tileCountsOffset + index] = count;`}),w(e,{id:`${t}-tile-offsets`,operation:B,variant:`tile-offsets`,bindings:[M(`tileCounts`,T,`u32`),N(`tileOffsets`,E,`u32`)],invocationCount:x+1,declarations:P,body:`var offset = 0u;
  for (var tile = 0u; tile < index; tile++) {
    offset += tileCounts[tileCountsOffset + tile];
  }
  tileOffsets[tileOffsetsOffset + index] = offset;`}),w(e,{id:`${t}-compact`,operation:B,variant:`compact`,bindings:[M(`positions`,n.positions,`f32`),M(`predictors`,n.predictors,`f32`),M(`response`,n.response,`f32`),M(`rowValid`,C,`u32`),M(`tileOffsets`,E,`u32`),N(`rows`,O,`f32`)],invocationCount:r,declarations:P,body:`if (index == 0u) {
    rows[rowsOffset + ${I}u] = f32(tileOffsets[tileOffsetsOffset + SCAN_TILE_COUNT]);
  }
  if (rowValid[rowValidOffset + index] == 0u) {
    return;
  }
  let tile = index / SCAN_TILE_ROWS;
  var rank = tileOffsets[tileOffsetsOffset + tile];
  for (var row = tile * SCAN_TILE_ROWS; row < index; row++) {
    rank += select(0u, 1u, rowValid[rowValidOffset + row] != 0u);
  }
  let base = rowsOffset + rank * ROW_STRIDE;
  rows[base] = positions[positionsOffset + 2u * index];
  rows[base + 1u] = positions[positionsOffset + 2u * index + 1u];
  rows[base + 2u] = 0.0;
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    rows[base + 3u + column] = predictors[predictorsOffset + index * PREDICTOR_COUNT + column];
  }
  rows[base + 3u + PREDICTOR_COUNT] = response[responseOffset + index];`}),...A&&k?[w(e,{id:`${t}-compact-positions`,operation:B,variant:`compact-positions`,bindings:[M(`rows`,O,`f32`),N(`compactPositions`,k,`f32`)],invocationCount:r,declarations:P,body:`let count = u32(rows[rowsOffset + ${I}u]);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  let included = index < count;
  compactPositions[compactPositionsOffset + 2u * index] =
    select(invalid, rows[rowsOffset + index * ROW_STRIDE], included);
  compactPositions[compactPositionsOffset + 2u * index + 1u] =
    select(invalid, rows[rowsOffset + index * ROW_STRIDE + 1u], included);`}),...A.nodes]:[],w(e,{id:`${t}-bandwidths`,operation:B,variant:`bandwidths`,bindings:[N(`rows`,O,`f32`),M(`params`,n.bandwidthParameters,`f32`),M(`selection`,n.selectedBandwidth,`f32`),...le],invocationCount:r,declarations:`${P}
${F}
${Te()}
${l?u():``}`,body:`let count = u32(rows[rowsOffset + ${I}u]);
  if (index >= count) {
    return;
  }
  // Slot 1 holds the selected ladder value, also for refined candidates outside the caller ladder.
  let value = selection[selectionOffset + 1u];
  var bandwidth = -1.0;
  if (params[paramsOffset + 1u] < 0.5) {
    bandwidth = select(-1.0, value, value > 0.0);
  } else if (value >= 2.0 && value <= f32(MAXIMUM_NEIGHBORS)) {
    let neighborCount = u32(floor(value + 0.5));
    var list: array<f32, MAXIMUM_NEIGHBORS>;
    ${l?`selectNeighborDistances(index, neighborCount, &list);`:`for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
      list[slot] = SENTINEL;
    }
    let origin = getPosition(index);
    for (var row = 0u; row < count; row++) {
      let delta = getPosition(row) - origin;
      let squared = dot(delta, delta);
      if (squared < list[neighborCount - 1u]) {
        var slot = neighborCount - 1u;
        loop {
          if (slot == 0u || list[slot - 1u] <= squared) {
            break;
          }
          list[slot] = list[slot - 1u];
          slot = slot - 1u;
        }
        list[slot] = squared;
      }
    }`}
    let squared = list[neighborCount - 1u];
    if (squared < SENTINEL) {
      bandwidth = sqrt(squared) * ADAPTIVE_FACTOR;
    }
  }
  rows[rowsOffset + index * ROW_STRIDE + 2u] = bandwidth;`}),w(e,{id:`${t}-anchor`,operation:B,variant:`anchor`,bindings:[M(`coefficients`,n.coefficients,`f32`),N(`shifts`,oe,`f32`)],invocationCount:1,declarations:`${P}
${Ce}`,body:`for (var column = 0u; column < P; column++) {
    shifts[shiftsOffset + column] = 0.0;
  }
  for (var row = 0u; row < ROW_COUNT; row++) {
    if (isFiniteRow(row)) {
      for (var column = 0u; column < P; column++) {
        shifts[shiftsOffset + column] = coefficients[coefficientsOffset + row * P + column];
      }
      break;
    }
  }`}),w(e,{id:`${t}-observed-tiles`,operation:B,variant:`observed-tiles`,bindings:[M(`coefficients`,n.coefficients,`f32`),M(`shifts`,oe,`f32`),N(`partial`,j,`f32`)],invocationCount:c,declarations:`${P}
${Ce}`,body:`var sums: array<f32, ${b}>;
  let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  for (var row = firstRow; row < endRow; row++) {
    if (!isFiniteRow(row)) {
      continue;
    }
    sums[0] += 1.0;
    for (var column = 0u; column < P; column++) {
      let deviation = coefficients[coefficientsOffset + row * P + column] - shifts[shiftsOffset + column];
      sums[1u + column] += deviation;
      sums[1u + P + column] += deviation * deviation;
    }
  }
  let base = partialOffset + index * PARTIAL_STRIDE;
  for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
    partial[base + slot] = sums[slot];
  }`}),w(e,{id:`${t}-permuted-tiles`,operation:B,variant:`permuted-tiles`,bindings:[M(`rows`,O,`f32`),M(`params`,n.bandwidthParameters,`f32`),M(`parameters`,n.parameters,`u32`),M(`shifts`,oe,`f32`),N(`partial`,j,`f32`),...le],workgroupSize:H,guardIndex:!1,invocationCount:ne*H*h,declarations:`${ue}
${F}
const TILE_GROUP_COUNT: u32 = ${ne}u;
const PERMUTED_WORKGROUP_SIZE: u32 = ${H}u;
${ie}
${ae(D.geographicallyWeightedRegressionNonstationarityShuffle.id)}
${p(i)}
${we(i,!!l)}`,body:`let workgroup = index / PERMUTED_WORKGROUP_SIZE;
  let permutation = workgroup / TILE_GROUP_COUNT + 1u;
  let tile = (workgroup % TILE_GROUP_COUNT) * PERMUTED_WORKGROUP_SIZE + localInvocationIndex;
  let count = u32(rows[rowsOffset + ${I}u]);
  let runsPermutation = permutation <= readPermutationCount();
  if (runsPermutation) {
    loadSwapOrNotRounds(readSeedKey(), permutation, count, localInvocationIndex, PERMUTED_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  var sums: array<f32, ${b}>;
  if (runsPermutation && tile < TILE_COUNT) {
    let firstRow = tile * TILE_ROWS;
    let endRow = min(firstRow + TILE_ROWS, count);
    for (var location = firstRow; location < endRow; location++) {
      let bandwidth = rows[rowsOffset + location * ROW_STRIDE + 2u];
      if (!(bandwidth > 0.0)) {
        continue;
      }
      var estimate: array<f32, ${i}>;
      if (!fitPermuted(location, bandwidth, count, &estimate)) {
        continue;
      }
      sums[0] += 1.0;
      for (var column = 0u; column < P; column++) {
        let deviation = estimate[column] - shifts[shiftsOffset + column];
        sums[1u + column] += deviation;
        sums[1u + P + column] += deviation * deviation;
      }
    }
    let base = partialOffset + (permutation * TILE_COUNT + tile) * PARTIAL_STRIDE;
    for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
      partial[base + slot] = sums[slot];
    }
  }`}),w(e,{id:`${t}-merge`,operation:B,variant:`merge`,bindings:[M(`partial`,j,`f32`),M(`parameters`,n.parameters,`u32`),N(`fitCounts`,se,`f32`),N(`standardDeviations`,ce,`f32`)],invocationCount:h+1,declarations:ue,body:`let nan = getNaN();
  var sums: array<f32, ${b}>;
  if (index <= readPermutationCount()) {
    for (var tile = 0u; tile < TILE_COUNT; tile++) {
      let base = partialOffset + (index * TILE_COUNT + tile) * PARTIAL_STRIDE;
      for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
        sums[slot] += partial[base + slot];
      }
    }
  } else {
    sums[0] = -1.0;
  }
  fitCounts[fitCountsOffset + index] = select(nan, sums[0], sums[0] >= 0.0);
  for (var column = 0u; column < P; column++) {
    var spread = nan;
    if (sums[0] > 0.0) {
      let mean = sums[1u + column] / sums[0];
      spread = sqrt(max(sums[1u + P + column] / sums[0] - mean * mean, 0.0));
    }
    standardDeviations[standardDeviationsOffset + index * P + column] = spread;
  }`}),w(e,{id:`${t}-finish`,operation:B,variant:`finish`,bindings:[M(`standardDeviations`,ce,`f32`),M(`fitCounts`,se,`f32`),M(`parameters`,n.parameters,`u32`),M(`rows`,O,`f32`),N(`table`,d.table,`f32`),N(`summary`,d.summary,`f32`)],invocationCount:i,declarations:ue,body:`let nan = getNaN();
  let permutationCount = readPermutationCount();
  let observed = standardDeviations[standardDeviationsOffset + index];
  var greater = 0u;
  var validCount = 0.0;
  var total = 0.0;
  for (var run = 1u; run <= permutationCount; run++) {
    let value = standardDeviations[standardDeviationsOffset + run * P + index];
    if (isFiniteBits(value)) {
      validCount += 1.0;
      total += value;
      if (isFiniteBits(observed) && value >= observed) {
        greater += 1u;
      }
    }
  }
  let mean = select(nan, total / validCount, validCount > 0.0);
  var squares = 0.0;
  for (var run = 1u; run <= permutationCount; run++) {
    let value = standardDeviations[standardDeviationsOffset + run * P + index];
    if (isFiniteBits(value)) {
      squares += (value - mean) * (value - mean);
    }
  }
  let simulatedDeviation = select(nan, sqrt(squares / validCount), validCount > 0.0);
  let base = tableOffset + index * 6u;
  table[base] = observed;
  table[base + 1u] = select(nan, f32(greater + 1u) / f32(permutationCount + 1u), isFiniteBits(observed));
  table[base + 2u] = select(nan, f32(greater), isFiniteBits(observed));
  table[base + 3u] = mean;
  table[base + 4u] = simulatedDeviation;
  table[base + 5u] = select(nan, (observed - mean) / simulatedDeviation, simulatedDeviation > 0.0);
  if (index == 0u) {
    let count = rows[rowsOffset + ${I}u];
    var failed = 0.0;
    for (var run = 1u; run <= permutationCount; run++) {
      failed += count - fitCounts[fitCountsOffset + run];
    }
    summary[summaryOffset + ${z.permutationCount}u] = f32(permutationCount);
    summary[summaryOffset + ${z.locationCount}u] = count;
    summary[summaryOffset + ${z.observedFitCount}u] = fitCounts[fitCountsOffset];
    summary[summaryOffset + ${z.failedFitCount}u] = failed;
  }`})]}},Se=`
${T}

${E}
`,Ce=`
fn isFiniteRow(row: u32) -> bool {
  for (var column = 0u; column < P; column++) {
    if (!isFiniteBits(coefficients[coefficientsOffset + row * P + column])) {
      return false;
    }
  }
  return true;
}
`;function we(e,t){let n=e;return`
fn getWeight(distance: f32, bandwidth: f32) -> f32 {
  let ratio = distance / bandwidth;
  if (params[paramsOffset] > 0.5) {
    if (distance >= bandwidth) {
      return 0.0;
    }
    let t = 1.0 - ratio * ratio;
    return t * t;
  }
  return exp(-0.5 * ratio * ratio);
}

fn fitPermuted(
  location: u32,
  bandwidth: f32,
  count: u32,
  estimate: ptr<function, array<f32, ${n}>>
) -> bool {
  var a: array<f32, ${n*n}>;
  var b: array<f32, ${n}>;
  var z: array<f32, ${n}>;
  var scale: array<f32, ${n}>;
  let origin = vec2f(rows[rowsOffset + location * ROW_STRIDE], rows[rowsOffset + location * ROW_STRIDE + 1u]);
  let focalBase = rowsOffset + getSwapOrNotPermutationIndex(location, count) * ROW_STRIDE;
  z[0] = 1.0;
  ${t?`// Bisquare weights vanish beyond the bandwidth, so only the covering grid cells are visited.
  let useGrid = params[paramsOffset] > 0.5 && bandwidth > 0.0 && isFiniteBits(bandwidth);
  let range = select(vec4u(0u, 0u, 0u, 0u), getGridRange(origin, bandwidth), useGrid);
  let gridRowCount = select(1u, range.w - range.z + 1u, useGrid);
  for (var gridRowStep = 0u; gridRowStep < gridRowCount; gridRowStep++) {
    var firstSlot = 0u;
    var endSlot = count;
    if (useGrid) {
      let cellRow = (range.z + gridRowStep) * INDEX_WIDTH;
      firstSlot = cellOffsets[cellOffsetsOffset + cellRow + range.x];
      endSlot = cellOffsets[cellOffsetsOffset + cellRow + range.y + 1u];
    }
    for (var slot = firstSlot; slot < endSlot; slot++) {
      var row = slot;
      if (useGrid) {
        row = sortedIds[sortedIdsOffset + slot];
      }
      let rowBase = rowsOffset + row * ROW_STRIDE;
      let delta = vec2f(rows[rowBase], rows[rowBase + 1u]) - origin;
      let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
      if (!(weight > 0.0)) {
        continue;
      }
      let dataBase = rowsOffset + getSwapOrNotPermutationIndex(row, count) * ROW_STRIDE;
      for (var column = 0u; column < PREDICTOR_COUNT; column++) {
        z[column + 1u] = rows[dataBase + 3u + column] - rows[focalBase + 3u + column];
      }
      let y = rows[dataBase + 3u + PREDICTOR_COUNT];
      for (var r = 0u; r < P; r++) {
        let weighted = weight * z[r];
        b[r] = b[r] + weighted * y;
        for (var c = 0u; c <= r; c++) {
          a[r * P + c] = a[r * P + c] + weighted * z[c];
        }
      }
    }
  }`:`for (var row = 0u; row < count; row++) {
    let rowBase = rowsOffset + row * ROW_STRIDE;
    let delta = vec2f(rows[rowBase], rows[rowBase + 1u]) - origin;
    let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let dataBase = rowsOffset + getSwapOrNotPermutationIndex(row, count) * ROW_STRIDE;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = rows[dataBase + 3u + column] - rows[focalBase + 3u + column];
    }
    let y = rows[dataBase + 3u + PREDICTOR_COUNT];
    for (var r = 0u; r < P; r++) {
      let weighted = weight * z[r];
      b[r] = b[r] + weighted * y;
      for (var c = 0u; c <= r; c++) {
        a[r * P + c] = a[r * P + c] + weighted * z[c];
      }
    }
  }`}
  for (var r = 0u; r < P; r++) {
    let diagonal = a[r * P + r];
    if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) {
      return false;
    }
    scale[r] = 1.0 / sqrt(diagonal);
  }
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c <= r; c++) {
      a[r * P + c] = a[r * P + c] * scale[r] * scale[c];
    }
    b[r] = b[r] * scale[r];
  }
  if (!choleskyFactor_${n}(&a)) {
    return false;
  }
  choleskySolve_${n}(&a, &b);
  var intercept = b[0] * scale[0];
  for (var column = 1u; column < P; column++) {
    let slope = b[column] * scale[column];
    (*estimate)[column] = slope;
    intercept = intercept - slope * rows[focalBase + 2u + column];
  }
  (*estimate)[0] = intercept;
  var finite = true;
  for (var column = 0u; column < P; column++) {
    finite = finite && isFiniteBits((*estimate)[column]);
  }
  return finite;
}
`}function Te(){return`
fn getPosition(row: u32) -> vec2f {
  return vec2f(rows[rowsOffset + row * ROW_STRIDE], rows[rowsOffset + row * ROW_STRIDE + 1u]);
}
`}var Ee={intercept:0,income:1,density:2,age65:3,education:4},De=[`ln median household income`,`ln population density`,`Age 65+ share`,`No high-school diploma share`],U=De.length,W=U+1,G=8,Oe=128,ke=99,Ae=[24,36,48,64,80,96,112,128],je=[15e4,25e4,35e4,5e5,7e5,1e6,14e5,2e6],Me=30,Ne=new Set([1,5,10,11,12,13,21,22,24,28,37,40,45,47,48,51,54]);async function Pe(t){let n=t.datasets.get(`us-counties`),o=t.datasets.get(`us-states`),{device:c}=t,u=new S(c,`local-relationships`),f=oe(n),p=f.featureCount,m=e=>n.column(e),h=m(`places_diabetes_ageAdj`),_=m(`medianHouseholdIncome`),y=m(`popDensity`),b=m(`age65plus`),x=m(`noHighSchool`),re=m(`rucc2023`),C=n.column(`stateFips`),w=[];for(let e=0;e<p;e++)Number.isFinite(h[e])&&_[e]>0&&y[e]>0&&Number.isFinite(b[e])&&Number.isFinite(x[e])&&w.push(e);let T=w.length,E=new Uint32Array(p).fill(de);w.forEach((e,t)=>{E[e]=t});let D=(e,t)=>{let n=Float32Array.from(w,n=>t(e[n])),{mean:r,deviation:i}=se(n);return Float32Array.from(n,e=>(e-r)/i)},ie=[D(_,Math.log),D(y,Math.log),D(b,e=>e),D(x,e=>e)],ae=new Float32Array(T*U);for(let e=0;e<T;e++)for(let t=0;t<U;t++)ae[e*U+t]=ie[t][e];let ge=Float32Array.from(w,e=>h[e]),B=n.column(`centroid`),V=new Float32Array(T*2);w.forEach((e,t)=>{V[t*2]=B[e*2],V[t*2+1]=B[e*2+1]});let _e=ce(V,M),ve=Uint32Array.from(w,e=>re[e]<=3?1:0),ye=e=>Uint32Array.from(w,(t,n)=>e===`all`?1:e===`metro`?ve[n]:e===`nonmetro`?1-ve[n]:Ne.has(C[t])?1:0),be=(n.geojson?.features??[]).map(e=>e.properties),H=fe(f,E),Se=fe(f),Ce=u.createBuffer(`triangles`,H.corners),we=u.createBuffer(`owners`,H.owners),Te=u.createBuffer(`every-triangles`,Se.corners),Pe=u.createBuffer(`every-owners`,Se.owners),Fe=I(f),Ie=u.createBuffer(`county-outline`,Fe),Le=o.geojson?P(o.geojson):new Float32Array(4),Re=u.createBuffer(`state-outline`,Le),ze=ue(f),Be=u.createBuffer(`positions`,_e),Ve=u.createBuffer(`predictors`,ae),He=u.createBuffer(`response`,ge),Ue=u.createBuffer(`mask`,ye(t.options.subset)),We=u.createBuffer(`coefficients`,T*W*4),Ge=u.createBuffer(`local-r2`,T*4),Ke=u.createBuffer(`residuals`,T*4),qe=u.createBuffer(`hat-diagonal`,T*4),Je=u.createBuffer(`local-status`,T*4),Ye=u.createBuffer(`condition-number`,T*4),Xe=u.createBuffer(`bandwidth-scores`,G*4),Ze=u.createBuffer(`selected-bandwidth`,8),Qe=u.createBuffer(`summary`,48),$e=u.createParameterBuffer(`gwr-parameters`,`float32`,g(G)),et=u.createBuffer(`ols-coefficients`,W*4),tt=u.createBuffer(`ols-standard-errors`,W*4),nt=u.createBuffer(`ols-t-statistics`,W*4),rt=u.createBuffer(`ols-summary`,88),it=u.createBuffer(`ols-status`,4),at=u.createBuffer(`ols-residuals`,T*4),ot=u.createBuffer(`ols-fitted`,T*4),st=W*6,ct=u.createBuffer(`test-table`,st*4),lt=u.createBuffer(`test-summary`,z.length*4),ut=u.createParameterBuffer(`permutation-parameters`,`uint32`,4),dt=new Map,ft=async e=>{let t=dt.get(e);if(t)return t;let n=new v(c,{id:`gwr-${e?`indexed`:`scan`}`});return n.add(new ne({spatialContext:r(i),id:`gwr`,positions:k(n,`positions`,Be,`float32x2`,T),predictors:k(n,`predictors`,Ve,`float32`,T*U),predictorCount:U,response:k(n,`response`,He,`float32`,T),mask:k(n,`mask`,Ue,`uint32`,T),parameters:$e.importToGraph(n),maximumBandwidthCount:G,maximumNeighborCount:Oe,indexGridSize:e?void 0:!1,output:{coefficients:k(n,`coefficients`,We,`float32`,T*W),localR2:k(n,`local-r2`,Ge,`float32`,T),residuals:k(n,`residuals`,Ke,`float32`,T),hatDiagonal:k(n,`hat-diagonal`,qe,`float32`,T),localStatus:k(n,`local-status`,Je,`uint32`,T),localConditionNumber:k(n,`condition-number`,Ye,`float32`,T),bandwidthScores:k(n,`bandwidth-scores`,Xe,`float32`,G),selectedBandwidth:k(n,`selected-bandwidth`,Ze,`float32`,2),summary:k(n,`summary`,Qe,`float32`,12)}})),t=u.track(await n.compileAsync()),dt.set(e,t),t},K=new v(c,{id:`gwr-ols-baseline`});K.add(new d({id:`ols`,predictors:k(K,`predictors`,Ve,`float32`,T*U),response:k(K,`response`,He,`float32`,T),mask:k(K,`mask`,Ue,`uint32`,T),predictorCount:U,output:{coefficients:k(K,`ols-coefficients`,et,`float32`,W),standardErrors:k(K,`ols-standard-errors`,tt,`float32`,W),tStatistics:k(K,`ols-t-statistics`,nt,`float32`,W),summary:k(K,`ols-summary`,rt,`float32`,22),status:{code:k(K,`ols-status`,it,`uint32`,1)},residuals:k(K,`ols-residuals`,at,`float32`,T),fitted:k(K,`ols-fitted`,ot,`float32`,T)}}));let pt=u.track(await K.compileAsync()),q=new v(c,{id:`gwr-nonstationarity-test`}),mt=null;try{q.add(new xe({spatialContext:r(i),id:`gwr-test`,positions:k(q,`positions`,Be,`float32x2`,T),predictors:k(q,`predictors`,Ve,`float32`,T*U),predictorCount:U,response:k(q,`response`,He,`float32`,T),mask:k(q,`mask`,Ue,`uint32`,T),bandwidthParameters:$e.importToGraph(q),selectedBandwidth:k(q,`selected-bandwidth`,Ze,`float32`,2),coefficients:k(q,`coefficients`,We,`float32`,T*W),parameters:ut.importToGraph(q),maximumPermutations:ke,maximumBandwidthCount:G,maximumNeighborCount:Oe,output:{table:k(q,`test-table`,ct,`float32`,st),summary:k(q,`test-summary`,lt,`float32`,z.length)}}))}catch(e){mt=e instanceof Error?e.message:String(e)}let J=mt?null:u.track(await q.compileAsync()),ht=(c.limits.maxStorageBuffersPerShaderStage??8)>=10,Y=t.options.spatialIndex&&ht,X=!0,gt=!0,_t=!1,vt=!1,Z=0,Q=[-1,1],yt=[0,1],$=null,bt=()=>t.options.mode===`adaptive`?Ae:je,xt=()=>{let{kernel:e,mode:n,bandwidth:r}=t.options,i=bt();$e.write(l({kernel:e,bandwidthMode:n,bandwidths:r<=0?i:[i[Math.min(r,G)-1]]},G)),X=!0,vt=!1},St=()=>{ut.write(a({seed:t.options.seed,permutationCount:t.options.permutations})),vt=!1};xt(),St(),await ft(Y);let Ct=u.track(new A);t.setReadout(`index`,Y?`grid-indexed (fixed bisquare visits nearby cells only)`:`full scan`),t.setReadout(`rows`,`${L(T)} of ${L(p)} counties`),t.setReadout(`test`,`press “Run the Monte Carlo test”`),t.setStatus(`${L(T)} counties`);let wt=()=>{if(!$)return;let{map:e}=t.options,n=Ee[e];if(n!==void 0){let e=new Float32Array(T);for(let t=0;t<T;t++)e[t]=$.coefficients[t*W+n];Z=t.options.center===`global`?$.olsCoefficients[n]:0;let r=F(e,.03)-Z,i=F(e,.97)-Z,a=Math.max(1e-6,Math.abs(r),Math.abs(i));Q=[Z-a,Z+a],t.setLegendExtent(`coefficient`,Q),t.setReadout(`spread`,`${j(F(e,.05),2)} to ${j(F(e,.95),2)} (national OLS ${j($.olsCoefficients[n],2)})`)}else (e===`residuals`||e===`olsResiduals`)&&(Q=le(e===`residuals`?$.residuals:$.olsResiduals,.03,.97),t.setLegendExtent(`residual`,Q))},Tt=pe(u,`gwr`,[{name:`summary`,buffer:Qe,bytes:48},{name:`selected`,buffer:Ze,bytes:8},{name:`scores`,buffer:Xe,bytes:G*4},{name:`coefficients`,buffer:We,bytes:T*W*4},{name:`localR2`,buffer:Ge,bytes:T*4},{name:`residuals`,buffer:Ke,bytes:T*4},{name:`status`,buffer:Je,bytes:T*4},{name:`condition`,buffer:Ye,bytes:T*4},{name:`hat`,buffer:qe,bytes:T*4},{name:`olsCoefficients`,buffer:et,bytes:W*4},{name:`olsStatus`,buffer:it,bytes:4},{name:`olsStandardErrors`,buffer:tt,bytes:W*4},{name:`olsTStatistics`,buffer:nt,bytes:W*4},{name:`olsSummary`,buffer:rt,bytes:88},{name:`olsResiduals`,buffer:at,bytes:T*4},{name:`testTable`,buffer:ct,bytes:st*4},{name:`testSummary`,buffer:lt,bytes:z.length*4}],e=>{let n=e(`summary`).f32,r=e(`selected`).f32,i=e(`scores`).f32,a=e(`coefficients`).f32,o=e(`status`).u32,c=e(`condition`).f32,l=e(`olsSummary`).f32,u=e(`olsCoefficients`).f32,d=ee({coefficients:u,standardErrors:e(`olsStandardErrors`).f32,tStatistics:e(`olsTStatistics`).f32,summary:l,status:e(`olsStatus`).u32},{predictorNames:[...De]}),f=me({summary:n,selectedBandwidth:r}),p=te;$={coefficients:a,localR2:e(`localR2`).f32,residuals:e(`residuals`).f32,olsResiduals:e(`olsResiduals`).f32,olsCoefficients:u,selected:r[1],mode:t.options.mode};let m=0,h=0;for(let e=0;e<T;e++)o[e]===s.SINGULAR?m++:o[e]===s.EXCLUDED&&h++;let g=n[p.HAS_VALID_CANDIDATE]!==0,ne=t.options.mode===`adaptive`?`nearest counties`:`km`,_=t.options.mode===`adaptive`?r[1]:r[1]/1e3;t.setReadout(`bandwidth`,g?`${j(_,0)} ${ne} (${r[0]<G?`ladder step ${t.options.bandwidth>0?t.options.bandwidth:r[0]+1}`:`refined between ladder steps`})`:`no valid candidate (all singular)`);let v=t.options.bandwidth<=0?G:1;t.setReadout(`scores`,`${v===1?`Single manual candidate: `:`Initial ladder: `}`+Array.from(i).slice(0,v).map(e=>Number.isFinite(e)?e.toFixed(0):`invalid`).join(` / `));let y=f.correctedAkaikeInformationCriterion??NaN,b=d.correctedAkaikeInformationCriterion??NaN;t.setReadout(`aicc`,`GWR ${j(y,0)} vs OLS ${j(b,0)} (${y<b?`GWR better by`:`OLS better by`} ${j(Math.abs(b-y),0)})`),t.setReadout(`rSquared`,`GWR ${j(f.rSquared)} vs OLS ${j(d.rSquared)}`),t.setReadout(`trace`,`${j(f.effectiveParameterCount,1)} (OLS: ${W})`),t.setReadout(`significance`,`a local coefficient counts as significant when |t| exceeds ${j(f.criticalT,2)} (alpha 0.05 adjusted to ${j(f.adjustedAlpha,4)} for ${j(f.effectiveParameterCount,1)} effective parameters)`),t.setReadout(`singular`,`${L(m)} singular, ${L(h)} excluded of ${L(T)}`);let x=Array.from(c).filter(Number.isFinite),S=x.filter(e=>e>Me).length;t.setReadout(`globalCondition`,d.conditioning?`${j(d.conditioning.conditionNumber,1)}${d.conditioning.illConditioned?` (float32 precision warning)`:``}; largest VIF ${j(d.conditioning.maximumVarianceInflationFactor,1)}`:`n/a`),t.setReadout(`condition`,x.length?`median ${j(F(x,.5),1)}, max ${j(F(x,1),0)}, ${L(S)} above ${Me}`:`n/a`),t.setReadout(`globalCoefficients`,De.map((e,t)=>`${e} ${j(d.coefficients[t+1].estimate,2)} (se ${j(d.coefficients[t+1].standardError,2)}, ${N(d.coefficients[t+1].pValue)})`).join(`; `));let re=e(`hat`).f32,C=Array.from(re).filter(Number.isFinite);yt=[0,Math.max(1e-6,F(C,.98))];let w=e(`testSummary`).f32,E=e(`testTable`).f32,D=w[z.permutationCount];if(vt&&D>0){let e=[`Intercept`,...De].map((e,t)=>{let n=t*6;return`${e}: ${N(E[n+he.pseudoPValue])} (spread ${j(E[n+he.observedStandardDeviation],2)} vs ${j(E[n+he.simulatedMean],2)} by chance)`});t.setReadout(`test`,`${D} permutations; ${L(w[z.failedFitCount])} singular refits`),e.forEach((e,n)=>t.setReadout(`test${n}`,e))}wt(),t.requestLayers()}),Et=()=>t.theme()===`dark`;return{getCompiledGraphs:()=>[dt.get(Y),pt,...J?[J]:[]],setOption(e){if(e===`map`||e===`center`||e===`borders`){wt(),t.requestLayers();return}if(e===`subset`){Ue.write(ye(t.options.subset)),gt=!0,xt();return}if(e===`spatialIndex`){let e=t.options.spatialIndex&&ht,n=t.options.spatialIndex&&!ht;Ct.request({build:()=>ft(e),commit:()=>{Y=e,t.setReadout(`index`,n?`unavailable: this device allows ${c.limits.maxStorageBuffersPerShaderStage} storage buffers per stage, the indexed kernel needs 10`:Y?`grid-indexed (fixed bisquare visits nearby cells only)`:`full scan`),X=!0,t.requestLayers()}});return}if(e===`permutations`||e===`seed`){St();return}xt()},onAction(e){if(e===`runTest`){if(!J){t.setReadout(`test`,`unavailable on this device: ${mt}`);return}_t=!0,t.setReadout(`test`,`running…`)}},encode(e,t){t.frameIndex<2&&(X=!0,gt=!0),X||gt||_t?(gt&&=(pt.encode(e,{parameters:void 0}),!1),X&&=(dt.get(Y).encode(e,{parameters:void 0}),!1),_t&&J&&(J.encode(e,{parameters:void 0}),_t=!1,vt=!0),Tt.request(e)):Tt.flush(e)},getLayers(){let{map:n}=t.options,r=Et(),i=[new R({id:`counties-base`,triangles:Te,owners:Pe,triangleCount:Se.triangleCount,colormap:`uniform`,color:r?[88,92,102,170]:[182,184,190,170]})],a={triangles:Ce,owners:we,triangleCount:H.triangleCount,opacity:.9,valueFormat:`float32`,color:[255,255,255,255],noDataColor:[0,0,0,0]},o=Ee[n];return o===void 0?n===`localR2`?i.push(new R({...a,id:`counties-r2`,values:Ge,colormap:`blues`,valueRange:[0,1]})):n===`condition`?i.push(new R({...a,id:`counties-condition`,values:Ye,colormap:`magma`,valueRange:[1,Me]})):n===`influence`?i.push(new R({...a,id:`counties-hat`,values:qe,colormap:`inferno`,valueRange:yt})):i.push(new R({...a,id:`counties-${n}`,values:n===`residuals`?Ke:at,colormap:`diverging`,valueRange:Q})):i.push(new R({...a,id:`counties-coefficient`,values:We,valueStride:W,valueOffset:o,colormap:`diverging`,valueRange:Q})),i.push(new O({id:`counties-outline`,coordinateSystem:e.LNGLAT,segments:Ie,instanceCount:Fe.length/4,widthPixels:.5,color:r?[225,228,238,28]:[40,44,56,40]})),t.options.borders&&i.push(new O({id:`states-outline`,coordinateSystem:e.LNGLAT,segments:Re,instanceCount:Le.length/4,widthPixels:1.2,color:r?[235,238,248,150]:[30,34,46,160]})),i},onThemeChange(){t.requestLayers()},getTooltip(e){if(!e.coordinate||!$)return null;let t=ze(e.coordinate[0],e.coordinate[1]);if(t<0)return null;let n=be[t],r=E[t],i=`${n?.name??`County`}, ${n?.state??``}`;if(r===4294967295)return`${i}\n(not in the model: missing data)`;let a=e=>$.coefficients[r*W+e];return[i,`Diabetes ${h[t].toFixed(1)}%, income $${L(_[t])}`,`Local effect of income: ${Number.isFinite(a(1))?a(1).toFixed(2):`n/a`} (national ${$.olsCoefficients[1].toFixed(2)})`,`Local R²: ${Number.isFinite($.localR2[r])?$.localR2[r].toFixed(2):`n/a`}`].join(`
`)},destroy(){Tt.stop(),u.destroy()}}}export{Pe as createLocalRelationships};