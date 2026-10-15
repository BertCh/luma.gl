import{_ as e,d as t,h as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{t as a}from"./gpu-sort-Bl3Dztdb.js";import{o,s}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as c,t as l}from"./wgsl-common-BNRVzgd8.js";import{c as u}from"./spatial-weights-D4bd9AQ6.js";import{t as d}from"./philox-random-CHwyk0dd.js";import{t as f}from"./permutation-random-C7SSHCbB.js";import{t as p}from"./permutation-shuffle-B9oRDb5C.js";var m={observed:0,pairCount:1,timeClosePairs:2,expected:3,permutationMean:4,permutationVariance:5,greaterCount:6,lesserCount:7,pseudoPGreater:8,pseudoPLesser:9,pseudoPTwoSided:10,zScore:11};function h(e){let{seed:n,permutationCount:r}=e;if(!Number.isSafeInteger(n)||n<0)throw Error(`space-time seed must be a non-negative safe integer`);t(`space-time`,`permutationCount`,r,1,2**31-1);let i=e.timeThreshold??0;if(Number.isNaN(i)||i<0)throw Error(`space-time timeThreshold must be non-negative`);let a=new Uint32Array(4);return a[0]=n%2**32,a[1]=Math.floor(n/2**32),a[2]=r,a[3]=new Uint32Array(new Float32Array([i]).buffer)[0],a}function g(e,t){if(e<=0)return 1;if(!(t>0))return 0;let n=-t,r=0;for(let i=0;i<e;i++)r+=Math.exp(n),n+=Math.log(t)-Math.log(i+1);return Math.max(0,Math.min(1,1-r))}function _(e,t,n){return h(t),{alternative:`greater`,permutationCount:Math.min(t.permutationCount,n),seed:t.seed,multipleTesting:`none`,testedRowsDefinition:e===`knox`?`all unordered event pairs listed in the spatial pair list, counted once`:`all unordered event pairs listed in the distance band, counted once`}}var v=64,y={cursor:0,exceedances:1,stopIndex:2,shiftDistance:4,shiftTime:5};function b(e){let t=64;for(;Math.ceil(e/t)>256;)t*=2;return t}function x(e,n,r,a){if(n===void 0){if(a)throw Error(`${e} output.permutationCount needs stopAfterExceedances`);return}if(t(e,`stopAfterExceedances`,n,1,r),!a)throw Error(`${e} stopAfterExceedances needs output.permutationCount`);if(i(a,[`uint32`],`${e} output.permutationCount`),a.length<1)throw Error(`${e} output.permutationCount needs 1 row`)}function S(e,t){let{id:n,operation:r,state:i,statistics:a,parameters:c,common:l}=t,u=b(t.slotCount),d=Math.ceil(t.slotCount/u),f=o(e,{id:`${n}-stop-dispatch`,operation:r,shapes:[[u*t.groupsPerSlot*t.workgroupSize,t.workgroupSize],[u,v]]}),p=e=>({name:`state`,view:i,type:`u32`,access:e}),m=t.statisticType===`u32`?`0u`:`0.0`,h=(e,t)=>{let{condition:n,extraResources:r}=f.getGate(t);return{id:e,condition:{...n,id:`${e}-gate`},extraResources:r}},g=y,_=[s(e,{id:`${n}-stop-reset`,operation:r,variant:`stop-reset`,bindings:[p(`read_write`),f.binding],invocationCount:1,body:`state[stateOffset + ${g.cursor}u] = 0u;
  state[stateOffset + ${g.exceedances}u] = 0u;
  state[stateOffset + ${g.stopIndex}u] = 0u;
  ${f.armWGSL}`})];for(let i=0;i<d;i++)_.push(t.createPartialsNode(h(`${n}-batch-${i}-partials`,0)),t.createReduceNode(h(`${n}-batch-${i}-reduce`,1)),s(e,{id:`${n}-batch-${i}-scan`,operation:r,variant:`stop-scan`,bindings:[{name:`parameters`,view:c,type:`u32`,access:`read`},{name:`statistics`,view:a,type:t.statisticType,access:`read`},p(`read_write`),f.binding],invocationCount:1,declarations:`${l}
const BATCH_SIZE: u32 = ${u}u;
const STOP_AFTER: u32 = ${t.stopAfterExceedances}u;`,body:`let permutations = readPermutationCount();
  let cursor = state[stateOffset + ${g.cursor}u];
  if (state[stateOffset + ${g.stopIndex}u] != 0u || cursor * BATCH_SIZE > permutations) {
    return;
  }
  let observed = statistics[statisticsOffset];
  var exceedances = state[stateOffset + ${g.exceedances}u];
  var stopIndex = 0u;
  for (var offset = 0u; offset < BATCH_SIZE; offset++) {
    let slot = cursor * BATCH_SIZE + offset;
    if (slot > permutations) {
      break;
    }
    if (slot > 0u && statistics[statisticsOffset + slot] >= observed) {
      exceedances++;
      if (exceedances == STOP_AFTER) {
        stopIndex = slot;
        break;
      }
    }
  }
  state[stateOffset + ${g.exceedances}u] = exceedances;
  state[stateOffset + ${g.stopIndex}u] = stopIndex;
  state[stateOffset + ${g.cursor}u] = cursor + 1u;
  if (stopIndex != 0u || (cursor + 1u) * BATCH_SIZE > permutations) {
    ${f.disarmWGSL}
  }`}));return _.push(s(e,{id:`${n}-stop-finalize`,operation:r,variant:`stop-finalize`,bindings:[{name:`parameters`,view:c,type:`u32`,access:`read`},p(`read`),{name:`statistics`,view:a,type:t.statisticType,access:`read_write`}],invocationCount:a.length,declarations:l,body:`let stopIndex = state[stateOffset + ${g.stopIndex}u];
  let used = select(readPermutationCount(), stopIndex, (stopIndex != 0u));
  if (index > used) {
    statistics[statisticsOffset + index] = ${m};
  }`})),_}function C(e,t){return r(e,`${t}-stop-state`,`uint32`,8)}var w=2**16,T=2**24-1;function E(e){return Math.ceil(e/64)}function D(e,t=256){let n=Math.max(1,Math.min(256,t,e));return[n,Math.ceil(e/n)]}function O(e,t){return s(e,{id:`${t.id}-pair-mask`,operation:t.operation,variant:`pair-mask`,bindings:[{name:`offsets`,view:t.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:t.neighbors,type:`u32`,access:`read`},{name:`pairMask`,view:t.pairMask,type:`u32`,access:`read_write`}],invocationCount:t.rows,declarations:`const ROWS: u32 = ${t.rows}u;`,body:`let start = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var entry = start; entry < end; entry++) {
    let other = neighbors[neighborsOffset + entry];
    var counted = other < ROWS && other != index;
    if (counted) {
      for (var earlier = start; earlier < entry; earlier++) {
        if (neighbors[neighborsOffset + earlier] == other) {
          counted = false;
          break;
        }
      }
    }
    if (counted && other < index) {
      for (var mirror = offsets[offsetsOffset + other]; mirror < offsets[offsetsOffset + other + 1u]; mirror++) {
        if (neighbors[neighborsOffset + mirror] == index) {
          counted = false;
          break;
        }
      }
    }
    pairMask[pairMaskOffset + entry] = select(0u, 1u, counted);
  }`})}function k(e,t,n,r){return`
${f}
const ROWS: u32 = ${e}u;
const BLOCKS: u32 = ${n}u;
const ROWS_PER_BLOCK: u32 = ${r}u;
const MAXIMUM_PERMUTATIONS: u32 = ${t}u;
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutationCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn readTimeThreshold() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
`}function A(e,t){return`
${p(t)}
const SPACE_TIME_WORKGROUP_SIZE: u32 = 64u;
const GROUPS_PER_SLOT: u32 = ${E(e)}u;
fn getPermutedRow(row: u32, slot: u32) -> u32 {
  if (slot == 0u) {
    return row;
  }
  return getSwapOrNotPermutationIndex(row, ROWS);
}
`}var j=`let workgroup = index / SPACE_TIME_WORKGROUP_SIZE;
  let slot = workgroup / GROUPS_PER_SLOT;
  let block = (workgroup - slot * GROUPS_PER_SLOT) * SPACE_TIME_WORKGROUP_SIZE + localInvocationIndex;
  let permutations = readPermutationCount();
  if (slot > 0u && slot <= permutations) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  let runsSlot = slot <= permutations && block < BLOCKS;`,M=`let workgroup = index / SPACE_TIME_WORKGROUP_SIZE;
  let localSlot = workgroup / GROUPS_PER_SLOT;
  let slot = state[stateOffset] * BATCH_SIZE + localSlot;
  let block = (workgroup - localSlot * GROUPS_PER_SLOT) * SPACE_TIME_WORKGROUP_SIZE + localInvocationIndex;
  let permutations = readPermutationCount();
  if (slot > 0u && slot <= permutations) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  let runsSlot = slot <= permutations && block < BLOCKS;`;function N(e,t){let n=m,r=t!==void 0,i=t=>e===`u32`?`f32(${t})`:t;return`
  ${r?`let stopIndex = state[stateOffset + ${y.stopIndex}u];
  let permutations = select(readPermutationCount(), stopIndex, (stopIndex != 0u));
  permutationsUsed[permutationsUsedOffset] = permutations;`:`let permutations = readPermutationCount();`}
  let observed = statistics[statisticsOffset];
  var greater = 0u;
  var lesser = 0u;
  var sumDeviation = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let value = statistics[statisticsOffset + slot];
    if (value >= observed) { greater++; }
    if (value <= observed) { lesser++; }
    sumDeviation += ${e===`u32`?`f32(i32(value) - i32(observed))`:`value - observed`};
  }
  let permutationCount = f32(permutations);
  let meanDeviation = sumDeviation / permutationCount;
  var sumSquares = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let value = statistics[statisticsOffset + slot];
    let deviation = ${e===`u32`?`f32(i32(value) - i32(observed))`:`value - observed`} - meanDeviation;
    sumSquares += deviation * deviation;
  }
  let variance = select(0.0, sumSquares / (permutationCount - 1.0), permutations > 1u);
  let mean = ${i(`observed`)} + meanDeviation;
  let pGreater = ${r?`select(f32(greater + 1u) / (permutationCount + 1.0), min(1.0, ${t}.0 / f32(max(stopIndex, 1u))), (stopIndex != 0u))`:`f32(greater + 1u) / (permutationCount + 1.0)`};
  let pLesser = f32(lesser + 1u) / (permutationCount + 1.0);
  summary[summaryOffset + ${n.observed}u] = ${i(`observed`)};
  summary[summaryOffset + ${n.pairCount}u] = observedPairs;
  summary[summaryOffset + ${n.timeClosePairs}u] = timeClosePairs;
  summary[summaryOffset + ${n.expected}u] = expected;
  summary[summaryOffset + ${n.permutationMean}u] = mean;
  summary[summaryOffset + ${n.permutationVariance}u] = variance;
  summary[summaryOffset + ${n.greaterCount}u] = f32(greater);
  summary[summaryOffset + ${n.lesserCount}u] = f32(lesser);
  summary[summaryOffset + ${n.pseudoPGreater}u] = pGreater;
  summary[summaryOffset + ${n.pseudoPLesser}u] = pLesser;
  summary[summaryOffset + ${n.pseudoPTwoSided}u] = min(1.0, 2.0 * min(pGreater, pLesser));
  let zScore = (${i(`observed`)} - mean) / sqrt(variance);
  summary[summaryOffset + ${n.zScore}u] = select(bitcast<f32>(parameters[parametersOffset + 3u] | 0x7fc00000u), zScore, variance > 0.0);
`}var P=`GPUKnoxTest`,F=class{id;props;rows;constructor(e){let r=e.id??`knox-test`;if(this.id=r,this.props=e,this.rows=u(r,e.pairs,`pairs`),i(e.times,[`float32`],`${r} times`),i(e.parameters,[`uint32`],`${r} parameters`),i(e.output.statistics,[`uint32`],`${r} output.statistics`),i(e.output.summary,[`float32`],`${r} output.summary`),this.rows<2)throw Error(`${r} needs at least two rows`);if(this.rows>T)throw Error(`${r} supports at most ${T} rows`);if(e.times.length!==this.rows)throw Error(`${r} times needs one row per pairs row`);if(e.parameters.length<4)throw Error(`${r} parameters needs 4 words`);if(t(r,`maximumPermutations`,e.maximumPermutations,1,w),e.output.statistics.length<e.maximumPermutations+1)throw Error(`${r} output.statistics needs maximumPermutations + 1 rows`);if(x(r,e.stopAfterExceedances,e.maximumPermutations,e.output.permutationCount),e.output.summary.length<12)throw Error(`${r} output.summary needs 12 rows`);if(n(r,[e.output.statistics,e.output.summary,e.output.permutationCount],[e.times,e.parameters,e.pairs.offsets,e.pairs.neighbors,e.pairs.weights]),e.output.statistics.buffer===e.output.summary.buffer||e.output.permutationCount&&(e.output.permutationCount.buffer===e.output.statistics.buffer||e.output.permutationCount.buffer===e.output.summary.buffer))throw Error(`${r} outputs must use separate buffers`)}getInferenceMetadata(e){return _(`knox`,e,this.props.maximumPermutations)}getCommandNodes(t){let{id:n,props:i,rows:o}=this,{pairs:u,times:f,parameters:p,maximumPermutations:m}=i,{statistics:h,summary:g}=i.output;e(n,t,[u.offsets,u.neighbors,u.weights,f,p,h,g,...i.output.permutationCount?[i.output.permutationCount]:[]]);let[_,v]=D(o),y=m+1,x=E(_),w=k(o,m,_,v),T=(e,i)=>r(t,`${n}-${e}`,`uint32`,i),F=i.stopAfterExceedances,I=F!==void 0,L=b(y),R=I?L:y,z=I?C(t,n):void 0,B=z?[{name:`state`,view:z,type:`u32`,access:`read`}]:[],V=I?`\nconst BATCH_SIZE: u32 = ${L}u;`:``,H=T(`sort-keys`,o),U=T(`sort-indices`,o),W=T(`sorted-keys`,o),G=T(`sorted-indices`,o),K=T(`time-close`,2),q=T(`partials`,R*_),J=T(`pair-partials`,_),Y=T(`pair-mask`,u.neighbors.length),X=`
${c}${l}`,Z=e=>s(t,{id:e?.id??`${n}-partials`,operation:P,variant:`partials`,bindings:[{name:`offsets`,view:u.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`},{name:`pairMask`,view:Y,type:`u32`,access:`read`},{name:`times`,view:f,type:`f32`,access:`read`},{name:`parameters`,view:p,type:`u32`,access:`read`},...B,{name:`partials`,view:q,type:`u32`,access:`read_write`},{name:`pairPartials`,view:J,type:`u32`,access:`read_write`}],condition:e?.condition,extraResources:e?.extraResources,workgroupSize:64,invocationCount:R*x*64,guardIndex:!1,declarations:`${w}
${A(_,d.knoxTestShuffle.id)}
const SLOT_COUNT: u32 = ${R}u;${V}`,body:`${I?M:j}
  var count = 0u;
  var pairCount = 0u;
  if (runsSlot) {
    let threshold = readTimeThreshold();
    for (var row = block; row < ROWS; row += BLOCKS) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (pairMask[pairMaskOffset + entry] != 0u) {
          pairCount++;
          let otherTime = times[timesOffset + getPermutedRow(other, slot)];
          if (abs(time - otherTime) <= threshold) {
            count++;
          }
        }
      }
    }
  }
  if (block < BLOCKS) {
    partials[partialsOffset + block * SLOT_COUNT + ${I?`localSlot`:`slot`}] = count;
  }
  if (slot == 0u && block < BLOCKS) {
    pairPartials[pairPartialsOffset + block] = pairCount;
  }`}),Q=e=>s(t,{id:e?.id??`${n}-reduce`,operation:P,variant:`reduce`,bindings:[{name:`parameters`,view:p,type:`u32`,access:`read`},...B,{name:`partials`,view:q,type:`u32`,access:`read`},{name:`statistics`,view:h,type:`u32`,access:`read_write`}],condition:e?.condition,extraResources:e?.extraResources,...I?{workgroupSize:64,invocationCount:L}:{invocationCount:h.length},declarations:`${w}
const SLOT_COUNT: u32 = ${R}u;${V}`,body:I?`let slot = state[stateOffset] * BATCH_SIZE + index;
  if (slot <= readPermutationCount()) {
    var total = 0u;
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + block * SLOT_COUNT + index];
    }
    statistics[statisticsOffset + slot] = total;
  }`:`var total = 0u;
  if (index <= readPermutationCount()) {
    for (var block = 0u; block < BLOCKS; block++) {
      total += partials[partialsOffset + block * SLOT_COUNT + index];
    }
  }
  statistics[statisticsOffset + index] = total;`}),$=I&&z?S(t,{id:n,operation:P,stopAfterExceedances:F,slotCount:y,groupsPerSlot:x,workgroupSize:64,statisticType:`u32`,parameters:p,statistics:h,state:z,common:w,createPartialsNode:Z,createReduceNode:Q}):[];return[s(t,{id:`${n}-keys`,operation:P,variant:`keys`,bindings:[{name:`times`,view:f,type:`f32`,access:`read`},{name:`sortKeys`,view:H,type:`u32`,access:`read_write`},{name:`sortIndices`,view:U,type:`u32`,access:`read_write`}],invocationCount:o,declarations:X,body:`sortKeys[sortKeysOffset + index] = getOrderedKey(times[timesOffset + index]);
  sortIndices[sortIndicesOffset + index] = index;`}),...new a({id:`${n}-sort`,keys:H,values:U,outputKeys:W,outputValues:G}).getCommandNodes(t),s(t,{id:`${n}-clear-close`,operation:P,variant:`clear-close`,bindings:[{name:`timeClose`,view:K,type:`u32`,access:`read_write`}],invocationCount:2,body:`timeClose[timeCloseOffset + index] = 0u;`}),s(t,{id:`${n}-count-close`,operation:P,variant:`count-close`,bindings:[{name:`sortedKeys`,view:W,type:`u32`,access:`read`},{name:`parameters`,view:p,type:`u32`,access:`read`},{name:`timeClose`,view:K,type:`atomic<u32>`,access:`read_write`}],invocationCount:o,workgroupSize:256,guardIndex:!1,declarations:`${X}
const ROWS: u32 = ${o}u;
var<workgroup> groupLow: atomic<u32>;
var<workgroup> groupCarry: atomic<u32>;`,body:`if (index < ROWS) {
    let threshold = bitcast<f32>(parameters[parametersOffset + 3u]);
    let time = decodeOrderedKey(sortedKeys[sortedKeysOffset + index]);
    var low = index + 1u;
    var high = ROWS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (abs(decodeOrderedKey(sortedKeys[sortedKeysOffset + middle]) - time) <= threshold) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let close = low - (index + 1u);
    if (close > 0u) {
      let previous = atomicAdd(&groupLow, close);
      if (previous + close < previous) {
        atomicAdd(&groupCarry, 1u);
      }
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let groupTotal = atomicLoad(&groupLow);
    var carries = atomicLoad(&groupCarry);
    if (groupTotal > 0u) {
      let previous = atomicAdd(&timeClose[timeCloseOffset], groupTotal);
      if (previous + groupTotal < previous) {
        carries++;
      }
    }
    if (carries > 0u) {
      atomicAdd(&timeClose[timeCloseOffset + 1u], carries);
    }
  }`}),O(t,{id:n,operation:P,rows:o,offsets:u.offsets,neighbors:u.neighbors,pairMask:Y}),...I?$:[Z(),Q()],s(t,{id:`${n}-summary`,operation:P,variant:`summary`,bindings:[{name:`parameters`,view:p,type:`u32`,access:`read`},{name:`statistics`,view:h,type:`u32`,access:`read`},{name:`pairPartials`,view:J,type:`u32`,access:`read`},{name:`timeClose`,view:K,type:`u32`,access:`read`},{name:`summary`,view:g,type:`f32`,access:`read_write`},...B,...i.output.permutationCount?[{name:`permutationsUsed`,view:i.output.permutationCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:w,body:`var pairTotal = 0u;
  for (var block = 0u; block < BLOCKS; block++) {
    pairTotal += pairPartials[pairPartialsOffset + block];
  }
  let observedPairs = f32(pairTotal);
  let timeClosePairs = f32(timeClose[timeCloseOffset + 1u]) * 4294967296.0 + f32(timeClose[timeCloseOffset]);
  let allPairs = f32(ROWS) * f32(ROWS - 1u) * 0.5;
  let expected = observedPairs * timeClosePairs / allPairs;
  ${N(`u32`,F)}`})]}};export{m as _,j as a,g as b,k as c,N as d,y as f,x as g,b as h,T as i,E as l,S as m,M as n,O as o,C as p,w as r,D as s,F as t,A as u,_ as v,h as y};