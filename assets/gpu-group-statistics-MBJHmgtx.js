import{_ as e,h as t,l as n,v as r,w as i,y as a}from"./resources-CwvhEOBe.js";import{t as o}from"./gpu-scan-BuGhTLDQ.js";import{t as s}from"./gpu-sort-Bl3Dztdb.js";import{r as c,s as l}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as u,n as d,r as f}from"./cell-key-wgsl-BQM8miC7.js";import{t as p}from"./order-statistic-wgsl-w2Gvl3O6.js";var m=4294967295;function h(e,t,n=`u32`){return{name:e,view:t,type:n,access:`read`}}function g(e,t,n=`u32`){return{name:e,view:t,type:n,access:`read_write`}}function _(e,t){return{name:e,view:t,type:`atomic<u32>`,access:`read_write`}}var v=`
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}

fn getValueKey(rawBits: u32) -> u32 {
  if (!isFiniteBits(rawBits)) {
    return 0xffffffffu;
  }
  var bits = rawBits;
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`;function y(e,t){return l(e,{id:t.id,operation:t.operation,variant:`gather`,bindings:[h(`indices`,t.indices),h(`source`,t.source),g(`gathered`,t.target)],invocationCount:t.target.length,body:`gathered[gatheredOffset + index] = source[sourceOffset + indices[indicesOffset + index]];`})}var b=256;function x(e){return`
const SAMPLE_VARIANCE: bool = ${e===`sample`};

fn getMomentM2(count: u32, sums: vec4f) -> f32 {
  let n = f32(count);
  let delta = sums.x / n;
  var m2 = sums.y - sums.x * delta;
  if (m2 <= 1.9073486e-6 * sums.y) {
    m2 = 0.0;
  }
  return max(m2, 0.0);
}

fn getMomentVariance(count: u32, m2: f32) -> f32 {
  if (SAMPLE_VARIANCE) {
    return select(getNaN(), m2 / f32(count - 1u), count >= 2u);
  }
  return m2 / f32(count);
}

fn getMomentSkewness(count: u32, sums: vec4f, m2: f32) -> f32 {
  if (m2 == 0.0) {
    return getNaN();
  }
  let n = f32(count);
  let delta = sums.x / n;
  let m3 = sums.z - 3.0 * delta * sums.y + 2.0 * n * delta * delta * delta;
  let scale = m2 / n;
  return (m3 / n) / (scale * sqrt(scale));
}

fn getMomentKurtosis(count: u32, sums: vec4f, m2: f32) -> f32 {
  if (m2 == 0.0) {
    return getNaN();
  }
  let n = f32(count);
  let delta = sums.x / n;
  let delta2 = delta * delta;
  let m4 = sums.w - 4.0 * delta * sums.z + 6.0 * delta2 * sums.y - 3.0 * n * delta2 * delta2;
  let scale = m2 / n;
  return (m4 / n) / (scale * scale) - 3.0;
}
`}var S=256,C=256;function w(e,t){let{id:n,operation:r,structure:a,valueBits:o,needs:s,output:c}=t,{capacity:u,rowCount:p}=a,y=[],w=(t,r)=>i(e,`${n}-${t}`,`uint32`,r),T=(t,r)=>i(e,`${n}-${t}`,`float32`,r),E=w(`finite-counts`,u),D=s.sums||s.moments||s.zScore?i(e,`${n}-sums`,`uint32x2`,u):void 0,O=s.minimum?w(`minimum-keys`,u):void 0,k=s.maximum?w(`maximum-keys`,u):void 0,A=s.moments||s.zScore,j=!!c.means||A,M=A?T(`means`,u):void 0;y.push(l(e,{id:`${n}-clear`,operation:r,variant:`clear`,bindings:[g(`finiteCounts`,E),...D?[g(`sums`,D)]:[],...O?[g(`minimumKeys`,O)]:[],...k?[g(`maximumKeys`,k)]:[]],invocationCount:u,body:`finiteCounts[finiteCountsOffset + index] = 0u;
  ${D?`sums[sumsOffset + 2u * index] = 0u;
  sums[sumsOffset + 2u * index + 1u] = 0u;`:``}
  ${O?`minimumKeys[minimumKeysOffset + index] = 0xffffffffu;`:``}
  ${k?`maximumKeys[maximumKeysOffset + index] = 0u;`:``}`}));let N=u<=S;if(y.push(l(e,{id:`${n}-accumulate`,operation:r,variant:`accumulate`,bindings:[h(`rowGroups`,a.rowGroups),h(`valueBits`,o),_(`finiteCounts`,E),...D?[_(`sums`,D)]:[],...O?[_(`minimumKeys`,O)]:[],...k?[_(`maximumKeys`,k)]:[]],invocationCount:p,...N?{guardIndex:!1,workgroupSize:C,declarations:`${f(t.sumScale)}
${v}
const GROUP_CAPACITY: u32 = ${u}u;
var<workgroup> localCounts: array<atomic<u32>, ${u}>;
${D?`var<workgroup> localSums: array<atomic<u32>, ${2*u}>;`:``}
${O?`var<workgroup> localMinimums: array<atomic<u32>, ${u}>;`:``}
${k?`var<workgroup> localMaximums: array<atomic<u32>, ${u}>;`:``}`,body:`for (var slot = localInvocationIndex; slot < GROUP_CAPACITY; slot += ${C}u) {
    atomicStore(&localCounts[slot], 0u);
    ${D?`atomicStore(&localSums[2u * slot], 0u);
    atomicStore(&localSums[2u * slot + 1u], 0u);`:``}
    ${O?`atomicStore(&localMinimums[slot], 0xffffffffu);`:``}
    ${k?`atomicStore(&localMaximums[slot], 0u);`:``}
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let group = rowGroups[rowGroupsOffset + index];
    let bits = valueBits[valueBitsOffset + index];
    if (group != ${m}u && isFiniteBits(bits)) {
      atomicAdd(&localCounts[group], 1u);
      ${D?`let value = cellScaleValue(bitcast<f32>(bits));
      let previous = atomicAdd(&localSums[2u * group], value.y);
      let high = value.x + select(0u, 1u, previous + value.y < previous);
      if (high != 0u) {
        atomicAdd(&localSums[2u * group + 1u], high);
      }`:``}
      ${O?`atomicMin(&localMinimums[group], getValueKey(bits));`:``}
      ${k?`atomicMax(&localMaximums[group], getValueKey(bits));`:``}
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < GROUP_CAPACITY; slot += ${C}u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&finiteCounts[finiteCountsOffset + slot], count);
      ${D?`let localLow = atomicLoad(&localSums[2u * slot]);
      let previous = atomicAdd(&sums[sumsOffset + 2u * slot], localLow);
      let high = atomicLoad(&localSums[2u * slot + 1u]) + select(0u, 1u, previous + localLow < previous);
      if (high != 0u) {
        atomicAdd(&sums[sumsOffset + 2u * slot + 1u], high);
      }`:``}
      ${O?`atomicMin(&minimumKeys[minimumKeysOffset + slot], atomicLoad(&localMinimums[slot]));`:``}
      ${k?`atomicMax(&maximumKeys[maximumKeysOffset + slot], atomicLoad(&localMaximums[slot]));`:``}
    }
  }`}:{declarations:`${f(t.sumScale)}
${v}`,body:`let group = rowGroups[rowGroupsOffset + index];
  if (group == ${m}u) {
    return;
  }
  let bits = valueBits[valueBitsOffset + index];
  if (!isFiniteBits(bits)) {
    return;
  }
  atomicAdd(&finiteCounts[finiteCountsOffset + group], 1u);
  ${D?`let value = cellScaleValue(bitcast<f32>(bits));
  let previous = atomicAdd(&sums[sumsOffset + 2u * group], value.y);
  let high = value.x + select(0u, 1u, previous + value.y < previous);
  if (high != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * group + 1u], high);
  }`:``}
  ${O?`atomicMin(&minimumKeys[minimumKeysOffset + group], getValueKey(bits));`:``}
  ${k?`atomicMax(&maximumKeys[maximumKeysOffset + group], getValueKey(bits));`:``}`}})),(c.counts||D)&&y.push(l(e,{id:`${n}-finish-sums`,operation:r,variant:`finish-sums`,bindings:[h(`finiteCounts`,E),...D?[h(`sums`,D)]:[],...c.counts?[g(`countsOut`,c.counts)]:[],...c.sums?[g(`sumsOut`,c.sums)]:[],...c.sumValues?[g(`sumValuesOut`,c.sumValues,`f32`)]:[],...c.means?[g(`meansOut`,c.means,`f32`)]:[],...M?[g(`meanScratch`,M,`f32`)]:[]],invocationCount:u,declarations:`${f(t.sumScale)}
${d}`,body:`let count = finiteCounts[finiteCountsOffset + index];
  ${c.counts?`countsOut[countsOutOffset + index] = count;`:``}
  ${D?`let low = sums[sumsOffset + 2u * index];
  let high = sums[sumsOffset + 2u * index + 1u];
  ${c.sums?`sumsOut[sumsOutOffset + 2u * index] = low;
  sumsOut[sumsOutOffset + 2u * index + 1u] = high;`:``}
  let total = cellI64ToF32(vec2u(high, low));
  ${c.sumValues?`sumValuesOut[sumValuesOutOffset + index] = total / SUM_SCALE;`:``}
  ${j?`let mean = select(getNaN(), total / (SUM_SCALE * f32(count)), count > 0u);`:``}
  ${c.means?`meansOut[meansOutOffset + index] = mean;`:``}
  ${M?`meanScratch[meanScratchOffset + index] = mean;`:``}`:``}`})),(O||k)&&y.push(l(e,{id:`${n}-finish-extremes`,operation:r,variant:`finish-extremes`,bindings:[h(`finiteCounts`,E),...O?[h(`minimumKeys`,O)]:[],...k?[h(`maximumKeys`,k)]:[],...c.minimums?[g(`minimumsOut`,c.minimums,`f32`)]:[],...c.maximums?[g(`maximumsOut`,c.maximums,`f32`)]:[]],invocationCount:u,declarations:d,body:`let isEmpty = finiteCounts[finiteCountsOffset + index] == 0u;
  ${O&&c.minimums?`minimumsOut[minimumsOutOffset + index] = select(decodeOrderedKey(minimumKeys[minimumKeysOffset + index]), getNaN(), isEmpty);`:``}
  ${k&&c.maximums?`maximumsOut[maximumsOutOffset + index] = select(decodeOrderedKey(maximumKeys[maximumKeysOffset + index]), getNaN(), isEmpty);`:``}`})),!A||!M)return{nodes:y,finiteCounts:E};let P=T(`moment-sums`,4*u);return y.push(l(e,{id:`${n}-moments`,operation:r,variant:`moments`,bindings:[h(`offsets`,a.offsets),h(`permutation`,a.permutation),h(`valueBits`,o),h(`means`,M,`f32`),h(`finiteCounts`,E),g(`momentSums`,P,`f32`)],workgroupSize:b,invocationCount:u*b,guardIndex:!1,declarations:`var<workgroup> partials: array<f32, ${4*b}>;
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}`,body:`let group = index / ${b}u;
  let isInRange = index < INVOCATION_COUNT;
  var s1 = 0.0;
  var s2 = 0.0;
  var s3 = 0.0;
  var s4 = 0.0;
  if (isInRange && finiteCounts[finiteCountsOffset + group] > 0u) {
    let begin = offsets[offsetsOffset + group];
    let end = offsets[offsetsOffset + group + 1u];
    let mean = means[meansOffset + group];
    for (var position = begin + localInvocationIndex; position < end; position += ${b}u) {
      let bits = valueBits[valueBitsOffset + permutation[permutationOffset + position]];
      if (isFiniteBits(bits)) {
        let d = bitcast<f32>(bits) - mean;
        let d2 = d * d;
        s1 += d;
        s2 += d2;
        s3 += d2 * d;
        s4 += d2 * d2;
      }
    }
  }
  partials[4u * localInvocationIndex] = s1;
  partials[4u * localInvocationIndex + 1u] = s2;
  partials[4u * localInvocationIndex + 2u] = s3;
  partials[4u * localInvocationIndex + 3u] = s4;
  workgroupBarrier();
  for (var stride = ${b/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      for (var component = 0u; component < 4u; component++) {
        partials[4u * localInvocationIndex + component] += partials[4u * (localInvocationIndex + stride) + component];
      }
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    for (var component = 0u; component < 4u; component++) {
      momentSums[momentSumsOffset + 4u * group + component] = partials[component];
    }
  }`})),(c.variances||c.standardDeviations||c.skewness||c.kurtosis)&&y.push(l(e,{id:`${n}-finish-moments`,operation:r,variant:`finish-moments`,bindings:[h(`finiteCounts`,E),h(`momentSums`,P,`f32`),...c.variances?[g(`variancesOut`,c.variances,`f32`)]:[],...c.standardDeviations?[g(`deviationsOut`,c.standardDeviations,`f32`)]:[],...c.skewness?[g(`skewnessOut`,c.skewness,`f32`)]:[],...c.kurtosis?[g(`kurtosisOut`,c.kurtosis,`f32`)]:[]],invocationCount:u,declarations:`${d}
${x(t.variance)}`,body:`let count = finiteCounts[finiteCountsOffset + index];
  let nan = getNaN();
  var variance = nan;
  var skewness = nan;
  var kurtosis = nan;
  if (count > 0u) {
    let sums = vec4f(
      momentSums[momentSumsOffset + 4u * index],
      momentSums[momentSumsOffset + 4u * index + 1u],
      momentSums[momentSumsOffset + 4u * index + 2u],
      momentSums[momentSumsOffset + 4u * index + 3u]
    );
    let m2 = getMomentM2(count, sums);
    variance = getMomentVariance(count, m2);
    skewness = getMomentSkewness(count, sums, m2);
    kurtosis = getMomentKurtosis(count, sums, m2);
  }
  ${c.variances?`variancesOut[variancesOutOffset + index] = variance;`:``}
  ${c.standardDeviations?`deviationsOut[deviationsOutOffset + index] = select(nan, sqrt(variance), count > 0u && (!SAMPLE_VARIANCE || count >= 2u));`:``}
  ${c.skewness?`skewnessOut[skewnessOutOffset + index] = skewness;`:``}
  ${c.kurtosis?`kurtosisOut[kurtosisOutOffset + index] = kurtosis;`:``}`})),c.zScores&&y.push(l(e,{id:`${n}-z-scores`,operation:r,variant:`z-scores`,bindings:[h(`rowGroups`,a.rowGroups),h(`valueBits`,o),h(`means`,M,`f32`),h(`finiteCounts`,E),h(`momentSums`,P,`f32`),g(`zScoresOut`,c.zScores,`f32`)],invocationCount:p,declarations:`${d}
${x(t.variance)}
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}`,body:`let group = rowGroups[rowGroupsOffset + index];
  let bits = valueBits[valueBitsOffset + index];
  var z = getNaN();
  if (group != ${m}u && isFiniteBits(bits)) {
    let count = finiteCounts[finiteCountsOffset + group];
    let sums = vec4f(
      momentSums[momentSumsOffset + 4u * group],
      momentSums[momentSumsOffset + 4u * group + 1u],
      momentSums[momentSumsOffset + 4u * group + 2u],
      momentSums[momentSumsOffset + 4u * group + 3u]
    );
    let m2 = getMomentM2(count, sums);
    let variance = select(0.0, getMomentVariance(count, m2), count >= 2u || !SAMPLE_VARIANCE);
    z = select((bitcast<f32>(bits) - means[meansOffset + group]) / sqrt(variance), 0.0, m2 == 0.0 || !(variance > 0.0));
  }
  zScoresOut[zScoresOutOffset + index] = z;`})),{nodes:y,finiteCounts:E}}function T(e,t){let{id:n,operation:r,keyLow:a,keyHigh:o,rowIds:c}=t,l=a.length,u=[],d=t=>i(e,`${n}-${t}`,`uint32`,l),f=t.initialRows??c,p=a;t.initialRows&&(p=d(`low-by-rows`),u.push(y(e,{id:`${n}-gather-low`,operation:r,indices:t.initialRows,source:a,target:p})));let m=d(`rows-by-low`),h=d(`sorted-low-scratch`);if(u.push(...new s({id:`${n}-sort-low`,keys:p,values:f,outputKeys:h,outputValues:m,keyBits:t.lowKeyBits??32}).getCommandNodes(e)),f=m,o){let t=d(`high-by-rows`);u.push(y(e,{id:`${n}-gather-high`,operation:r,indices:f,source:o,target:t}));let i=d(`rows-by-high`);u.push(...new s({id:`${n}-sort-high`,keys:t,values:f,outputKeys:d(`sorted-high-scratch`),outputValues:i,keyBits:32}).getCommandNodes(e)),f=i}let g=o||t.initialRows?void 0:h;return{nodes:u,permutation:f,sortedKeyLow:g}}function E(e,t){let{id:n,operation:r,output:a}=t,s=t.keys.length,u=a.counts.length,d=t.keys.format===`uint32x2`,f=t.keyCount,p=f!==void 0,_=p?f:4294967295,v=p?Math.max(1,32-Math.clz32(f)):32,b=(t,r)=>i(e,`${n}-${t}`,`uint32`,r),x=[],S=b(`key-low`,s),C=d?b(`key-high`,s):void 0,w=b(`row-ids`,s);x.push(l(e,{id:`${n}-prepare`,operation:r,variant:`prepare`,bindings:[h(`keys`,t.keys),...t.mask?[h(`mask`,t.mask)]:[],g(`keyLow`,S),...C?[g(`keyHigh`,C)]:[],g(`rowIds`,w)],invocationCount:s,body:`${d?`var low = keys[keysOffset + 2u * index];
  var high = keys[keysOffset + 2u * index + 1u];
  var isValid = !(low == 0xffffffffu && high == 0xffffffffu);`:`var low = keys[keysOffset + index];
  var isValid = low != 0xffffffffu;`}
  ${t.mask?`if (mask[maskOffset + index] == 0u) {
    isValid = false;
  }`:``}
  ${p?`if (low >= ${f}u) {\n    isValid = false;\n  }`:``}
  if (!isValid) {
    low = ${_}u;
    ${d?`high = 0xffffffffu;`:``}
  }
  keyLow[keyLowOffset + index] = low;
  ${d?`keyHigh[keyHighOffset + index] = high;`:``}
  rowIds[rowIdsOffset + index] = index;`}));let E=T(e,{id:`${n}-order`,operation:r,keyLow:S,keyHigh:C,rowIds:w,lowKeyBits:v});x.push(...E.nodes);let D=E.permutation,O=E.sortedKeyLow;O||(O=b(`sorted-low`,s),x.push(y(e,{id:`${n}-gather-sorted-low`,operation:r,indices:D,source:S,target:O})));let k=d?b(`sorted-high`,s):void 0;k&&C&&x.push(y(e,{id:`${n}-gather-sorted-high`,operation:r,indices:D,source:C,target:k}));let A=()=>[h(`sortedLow`,O),...k?[h(`sortedHigh`,k)]:[]],j=e=>d?`vec2u(sortedHigh[sortedHighOffset + ${e}], sortedLow[sortedLowOffset + ${e}])`:`vec2u(0u, sortedLow[sortedLowOffset + ${e}])`,M=`const TWO_WORDS: bool = ${d};
fn isInvalidKey(key: vec2u) -> bool {
  return key.y == ${_}u && (!TWO_WORDS || key.x == 0xffffffffu);
}`,N=b(`heads`,s),P=b(`valid-count`,1);x.push(l(e,{id:`${n}-heads`,operation:r,variant:`heads`,bindings:[...A(),g(`heads`,N),g(`validCount`,P)],invocationCount:s,declarations:`const ROW_COUNT: u32 = ${s}u;
${M}`,body:`let key = ${j(`index`)};
  let isValid = !isInvalidKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${j(`index - 1u`)});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isInvalidKey(${j(`index + 1u`)}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`}));let F=b(`offsets`,u+1),I=b(`group-total`,1),L=b(`position-groups`,s);if(p)x.push(l(e,{id:`${n}-dense-offsets`,operation:r,variant:`dense-offsets`,bindings:[...A(),h(`validCount`,P),g(`offsets`,F),g(`groupTotal`,I)],invocationCount:u+1,body:`var low = 0u;
  var high = validCount[validCountOffset];
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedLow[sortedLowOffset + middle] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  offsets[offsetsOffset + index] = low;
  if (index == 0u) {
    groupTotal[groupTotalOffset] = ${u}u;
  }`}),l(e,{id:`${n}-dense-position-groups`,operation:r,variant:`dense-position-groups`,bindings:[...A(),h(`validCount`,P),g(`positionGroups`,L)],invocationCount:s,body:`positionGroups[positionGroupsOffset + index] =
    select(0xffffffffu, sortedLow[sortedLowOffset + index], index < validCount[validCountOffset]);`}));else{let t=b(`group-indices`,s);x.push(...new o({id:`${n}-scan`,input:N,output:t,mode:`exclusive`}).getCommandNodes(e)),x.push(l(e,{id:`${n}-offsets`,operation:r,variant:`offsets`,bindings:[h(`heads`,N),h(`groupIndices`,t),h(`validCount`,P),g(`offsets`,F),g(`groupTotal`,I),g(`positionGroups`,L)],invocationCount:s,declarations:`const ROW_COUNT: u32 = ${s}u;
const CAPACITY: u32 = ${u}u;`,body:`let isHead = heads[headsOffset + index] != 0u;
  let group = groupIndices[groupIndicesOffset + index] + select(0u, 1u, isHead) - 1u;
  let isValid = index < validCount[validCountOffset];
  positionGroups[positionGroupsOffset + index] = select(0xffffffffu, group, isValid && group < CAPACITY);
  if (isHead && group <= CAPACITY) {
    offsets[offsetsOffset + group] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let total = groupIndices[groupIndicesOffset + last] + heads[headsOffset + last];
    groupTotal[groupTotalOffset] = total;
    if (total <= CAPACITY) {
      offsets[offsetsOffset + total] = validCount[validCountOffset];
    }
  }`}))}let R=b(`row-groups`,s);return x.push(l(e,{id:`${n}-row-groups`,operation:r,variant:`row-groups`,bindings:[h(`permutation`,D),h(`positionGroups`,L),g(`rowGroups`,R)],invocationCount:s,body:`rowGroups[rowGroupsOffset + permutation[permutationOffset + index]] =
    positionGroups[positionGroupsOffset + index];`})),p?x.push(l(e,{id:`${n}-groups`,operation:r,variant:`dense-groups`,bindings:[h(`offsets`,F),g(`keysOut`,a.keys),g(`countsOut`,a.counts)],invocationCount:u,body:`keysOut[keysOutOffset + index] = index;
  countsOut[countsOutOffset + index] = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];`})):x.push(l(e,{id:`${n}-groups`,operation:r,variant:`groups`,bindings:[h(`offsets`,F),h(`groupTotal`,I),...A(),g(`keysOut`,a.keys),g(`countsOut`,a.counts)],invocationCount:u,declarations:`const CAPACITY: u32 = ${u}u;
const GROUP_NONE: u32 = ${m}u;`,body:`if (index >= min(groupTotal[groupTotalOffset], CAPACITY)) {
    ${d?`keysOut[keysOutOffset + 2u * index] = 0xffffffffu;
    keysOut[keysOutOffset + 2u * index + 1u] = 0xffffffffu;`:`keysOut[keysOutOffset + index] = 0xffffffffu;`}
    countsOut[countsOutOffset + index] = 0u;
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let key = ${j(`begin`)};
  ${d?`keysOut[keysOutOffset + 2u * index] = key.y;
  keysOut[keysOutOffset + 2u * index + 1u] = key.x;`:`keysOut[keysOutOffset + index] = key.y;`}
  countsOut[countsOutOffset + index] = offsets[offsetsOffset + index + 1u] - begin;`})),x.push(c(e,{id:`${n}-publish`,operation:r,requiredCount:I,output:{ids:a.counts,status:a.status}})),{nodes:x,structure:{rowCount:s,capacity:u,twoWords:d,permutation:D,keyLow:S,lowKeyBits:v,keyHigh:C,rowIds:w,offsets:F,groupTotal:I,positionGroups:L,rowGroups:R}}}var D=`
${p}

fn getQuantile(begin: u32, count: u32, p: f32) -> f32 {
  if (count == 0u || !isOrderStatisticProbability(p)) {
    return getNaN();
  }
  // Counts stay below 2^31, so this is a runtime zero that blocks multiply-add fusion.
  let zero = count >> 31u;
  let h = roundOrderStatisticStep(f32(count - 1u) * p, zero);
  let floorH = floor(h);
  let lowIndex = min(u32(floorH), count - 1u);
  let low = decodeOrderedKey(sortedKeys[sortedKeysOffset + begin + lowIndex]);
  if (lowIndex + 1u >= count) {
    return low;
  }
  let high = decodeOrderedKey(sortedKeys[sortedKeysOffset + begin + lowIndex + 1u]);
  return interpolateOrderStatistic(low, high, h - floorH, zero);
}
`;function O(e,t){let{id:n,operation:r,structure:a,valueBits:c,finiteCounts:u,output:f}=t,{rowCount:p,capacity:m}=a,b=[],x=(t,r=p)=>i(e,`${n}-${t}`,`uint32`,r),S=x(`value-keys`);b.push(l(e,{id:`${n}-value-keys`,operation:r,variant:`value-keys`,bindings:[h(`valueBits`,c),g(`valueKeys`,S)],invocationCount:p,declarations:v,body:`valueKeys[valueKeysOffset + index] = getValueKey(valueBits[valueBitsOffset + index]);`}));let C=x(`rows-by-value`);b.push(...new s({id:`${n}-sort-value`,keys:S,values:a.rowIds,outputKeys:x(`sorted-value-scratch`),outputValues:C,keyBits:32}).getCommandNodes(e));let w=T(e,{id:`${n}-order`,operation:r,keyLow:a.keyLow,keyHigh:a.keyHigh,rowIds:a.rowIds,initialRows:C,lowKeyBits:a.lowKeyBits});b.push(...w.nodes);let E=x(`sorted-keys`);b.push(y(e,{id:`${n}-gather-sorted-keys`,operation:r,indices:w.permutation,source:S,target:E}));let O=`${d}
${v}
${D}`;if(f.medians&&b.push(l(e,{id:`${n}-medians`,operation:r,variant:`medians`,bindings:[h(`finiteCounts`,u),h(`offsets`,a.offsets),h(`sortedKeys`,E),g(`mediansOut`,f.medians,`f32`)],invocationCount:m,declarations:O,body:`let count = finiteCounts[finiteCountsOffset + index];
  var begin = 0u;
  if (count > 0u) {
    begin = offsets[offsetsOffset + index];
  }
  mediansOut[mediansOutOffset + index] = getQuantile(begin, count, 0.5);`})),f.percentiles&&t.percentiles){let i=t.percentiles.length;b.push(l(e,{id:`${n}-percentiles`,operation:r,variant:`percentiles`,bindings:[h(`finiteCounts`,u),h(`offsets`,a.offsets),h(`sortedKeys`,E),h(`fractions`,t.percentiles,`f32`),g(`percentilesOut`,f.percentiles,`f32`)],invocationCount:m*i,declarations:`${O}
const PERCENTILE_COUNT: u32 = ${i}u;`,body:`let group = index / PERCENTILE_COUNT;
  let count = finiteCounts[finiteCountsOffset + group];
  var begin = 0u;
  if (count > 0u) {
    begin = offsets[offsetsOffset + group];
  }
  percentilesOut[percentilesOutOffset + index] =
    getQuantile(begin, count, fractions[fractionsOffset + index % PERCENTILE_COUNT]);`}))}let k=!!f.modes,A=!!f.uniqueCounts;if(!k&&!A)return b;let j=x(`run-heads`);b.push(l(e,{id:`${n}-run-heads`,operation:r,variant:`run-heads`,bindings:[h(`sortedKeys`,E),h(`positionGroups`,a.positionGroups),h(`offsets`,a.offsets),g(`heads`,j)],invocationCount:p,body:`let group = positionGroups[positionGroupsOffset + index];
  let key = sortedKeys[sortedKeysOffset + index];
  var isHead = group != 0xffffffffu && key != 0xffffffffu;
  if (isHead && index != offsets[offsetsOffset + group]) {
    isHead = key != sortedKeys[sortedKeysOffset + index - 1u];
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);`}));let M=x(`unique-counts`,m),N=k?x(`maximum-run-lengths`,m):void 0,P=k?x(`mode-runs`,m):void 0;if(b.push(l(e,{id:`${n}-run-clear`,operation:r,variant:`run-clear`,bindings:[g(`uniqueCounts`,M),...N?[g(`maximumLengths`,N)]:[],...P?[g(`modeRuns`,P)]:[]],invocationCount:m,body:`uniqueCounts[uniqueCountsOffset + index] = 0u;
  ${N?`maximumLengths[maximumLengthsOffset + index] = 0u;`:``}
  ${P?`modeRuns[modeRunsOffset + index] = 0xffffffffu;`:``}`})),!k)b.push(l(e,{id:`${n}-count-runs`,operation:r,variant:`count-runs`,bindings:[h(`heads`,j),h(`positionGroups`,a.positionGroups),_(`uniqueCounts`,M)],invocationCount:p,body:`if (heads[headsOffset + index] != 0u) {
    atomicAdd(&uniqueCounts[uniqueCountsOffset + positionGroups[positionGroupsOffset + index]], 1u);
  }`}));else if(N&&P){let t=x(`run-indices`);b.push(...new o({id:`${n}-run-scan`,input:j,output:t,mode:`exclusive`}).getCommandNodes(e));let i=x(`run-starts`);b.push(l(e,{id:`${n}-run-starts`,operation:r,variant:`run-starts`,bindings:[h(`heads`,j),h(`runIndices`,t),g(`runStarts`,i)],invocationCount:p,body:`if (heads[headsOffset + index] != 0u) {
    runStarts[runStartsOffset + runIndices[runIndicesOffset + index]] = index;
  }`}));let s=()=>[h(`heads`,j),h(`runIndices`,t),h(`runStarts`,i),h(`positionGroups`,a.positionGroups),h(`offsets`,a.offsets),h(`finiteCounts`,u)],c=`const ROW_COUNT: u32 = ${p}u;`,v=`let run = runIndices[runIndicesOffset + index];
  let group = positionGroups[positionGroupsOffset + index];
  let totalRuns = runIndices[runIndicesOffset + ROW_COUNT - 1u] + heads[headsOffset + ROW_COUNT - 1u];
  var nextStart = ROW_COUNT;
  if (run + 1u < totalRuns) {
    nextStart = runStarts[runStartsOffset + run + 1u];
  }
  let runEnd = min(nextStart, offsets[offsetsOffset + group] + finiteCounts[finiteCountsOffset + group]);
  let runLength = runEnd - index;`;b.push(l(e,{id:`${n}-run-lengths`,operation:r,variant:`run-lengths`,bindings:[...s(),_(`maximumLengths`,N),...A?[_(`uniqueCounts`,M)]:[]],invocationCount:p,declarations:c,body:`if (heads[headsOffset + index] == 0u) {
    return;
  }
  ${v}
  atomicMax(&maximumLengths[maximumLengthsOffset + group], runLength);
  ${A?`atomicAdd(&uniqueCounts[uniqueCountsOffset + group], 1u);`:``}`})),b.push(l(e,{id:`${n}-mode-runs`,operation:r,variant:`mode-runs`,bindings:[...s(),_(`maximumLengths`,N),_(`modeRuns`,P)],invocationCount:p,declarations:c,body:`if (heads[headsOffset + index] == 0u) {
    return;
  }
  ${v}
  if (runLength == atomicLoad(&maximumLengths[maximumLengthsOffset + group])) {
    atomicMin(&modeRuns[modeRunsOffset + group], run);
  }`})),b.push(l(e,{id:`${n}-finish-modes`,operation:r,variant:`finish-modes`,bindings:[h(`finiteCounts`,u),h(`modeRuns`,P),h(`runStarts`,i),h(`sortedKeys`,E),g(`modesOut`,f.modes,`f32`)],invocationCount:m,declarations:d,body:`var mode = getNaN();
  if (finiteCounts[finiteCountsOffset + index] > 0u) {
    mode = decodeOrderedKey(
      sortedKeys[sortedKeysOffset + runStarts[runStartsOffset + modeRuns[modeRunsOffset + index]]]
    );
  }
  modesOut[modesOutOffset + index] = mode;`}))}return f.uniqueCounts&&b.push(l(e,{id:`${n}-finish-unique`,operation:r,variant:`finish-unique`,bindings:[h(`uniqueCounts`,M),g(`uniqueOut`,f.uniqueCounts)],invocationCount:m,body:`uniqueOut[uniqueOutOffset + index] = uniqueCounts[uniqueCountsOffset + index];`})),b}var k=`GPUGroupStatistics`,A=4;function j(e){return n(e,16)}function M(e){return n(e,4)}var N={count:[`counts`],sum:[`sums`,`sumValues`],mean:[`means`],minimum:[`minimums`],maximum:[`maximums`],variance:[`variances`],standardDeviation:[`standardDeviations`],skewness:[`skewness`],kurtosis:[`kurtosis`],median:[`medians`],percentiles:[`percentiles`],mode:[`modes`],uniqueCount:[`uniqueCounts`],zScore:[`zScores`]},P=[`sumValues`,`means`,`minimums`,`maximums`,`variances`,`standardDeviations`,`skewness`,`kurtosis`,`medians`,`percentiles`,`modes`,`zScores`],F=class{id;props;variance;sumScale;constructor(e){this.id=e.id??`group-statistics`,this.props=e;let n=this.id;if(this.variance=e.variance??`sample`,this.variance!==`sample`&&this.variance!==`population`)throw Error(`${n} variance must be 'sample' or 'population'`);this.sumScale=e.sumScale??65536,u(n,this.sumScale);let{keys:r,mask:i,columns:o,output:s}=e;if(o.length>A)throw Error(`${n} supports at most ${A} columns`);a(r,[`uint32`,`uint32x2`],`${n} keys`);let c=r.length;if(c<1)throw Error(`${n} keys must hold at least one row`);if(i&&(a(i,[`uint32`],`${n} mask`),i.length!==c))throw Error(`${n} mask must have the same length as keys`);if(e.keyCount!==void 0){if(!Number.isInteger(e.keyCount)||e.keyCount<1||e.keyCount>=4294967295)throw Error(`${n} keyCount must be an integer in [1, 2^32 - 2]`);if(r.format!==`uint32`)throw Error(`${n} keyCount needs uint32 keys`);if(s.keys.length!==e.keyCount)throw Error(`${n} output.keys must hold exactly keyCount rows`)}a(s.keys,[r.format],`${n} output.keys`);let l=s.keys.length;if(l<1)throw Error(`${n} output.keys must hold at least one row`);if(a(s.counts,[`uint32`],`${n} output.counts`),s.counts.length!==l)throw Error(`${n} output.counts must have the same length as output.keys`);for(let e of[`count`,`overflow`,`requiredCount`]){let t=s.status[e];if(t&&(a(t,[`uint32`],`${n} output.status.${e}`),t.length<1))throw Error(`${n} output.status.${e} must contain one uint32 row`)}let d=0;for(let[t,r]of o.entries()){let i=`columns[${t}]`;if(a(r.values,[`float32`],`${n} ${i}.values`),r.values.length!==c)throw Error(`${n} ${i}.values must have the same length as keys`);let o=new Set(r.statistics);for(let e of o)if(!(e in N))throw Error(`${n} ${i} has unknown statistic '${e}'`);for(let e of o){let t=N[e];if(!t.some(e=>r.output[e]))throw Error(`${n} ${i} statistic '${e}' needs output.${t.join(` or output.`)}`)}for(let[e,t]of Object.entries(r.output)){if(!t)continue;let r=Object.entries(N).find(([,t])=>t.includes(e));if(!r)throw Error(`${n} ${i} has unknown output.${e}`);if(!o.has(r[0]))throw Error(`${n} ${i} output.${e} requires statistic '${r[0]}'`)}let s=r.output;for(let e of[`counts`,`uniqueCounts`]){let t=s[e];t&&a(t,[`uint32`],`${n} ${i}.output.${e}`)}s.sums&&a(s.sums,[`uint32x2`],`${n} ${i}.output.sums`);for(let e of P){let t=s[e];t&&a(t,[`float32`],`${n} ${i}.output.${e}`)}let u=[[`zScores`,c]];for(let e of[`counts`,`sums`,`sumValues`,`means`,`minimums`,`maximums`,`variances`,`standardDeviations`,`skewness`,`kurtosis`,`medians`,`modes`,`uniqueCounts`])u.push([e,l]);for(let[e,t]of u){let r=s[e];if(r&&r.length!==t)throw Error(`${n} ${i}.output.${e} must have ${e===`zScores`?`the same length as keys`:`the same length as output.keys`}`)}if(o.has(`percentiles`)){if(d++,!e.percentiles)throw Error(`${n} ${i} statistic 'percentiles' needs the percentiles view`);let t=s.percentiles;if(t&&t.length!==l*e.percentiles.length)throw Error(`${n} ${i}.output.percentiles must hold output.keys.length * percentiles.length rows`)}}if(e.percentiles){if(a(e.percentiles,[`float32`],`${n} percentiles`),e.percentiles.length<1)throw Error(`${n} percentiles must hold at least one fraction`);if(d===0)throw Error(`${n} percentiles is only used by the 'percentiles' statistic`)}t(n,this.getOutputViews(),this.getInputViews())}getInputViews(){let{props:e}=this;return[e.keys,e.mask,e.percentiles,...e.columns.map(e=>e.values)]}getOutputViews(){let{output:e}=this.props;return[e.keys,e.counts,e.status.count,e.status.overflow,e.status.requiredCount,...this.props.columns.flatMap(e=>Object.values(e.output))]}getCommandNodes(t){let{id:n,props:a}=this;e(n,t,[...this.getInputViews(),...this.getOutputViews()]);let o=a.output.keys.length;r(n,`output.keys rows`,o,j(t.device)),a.percentiles&&r(n,`output.keys rows x percentiles`,o*a.percentiles.length,M(t.device));let{nodes:s,structure:c}=E(t,{id:n,operation:k,keys:a.keys,mask:a.mask,keyCount:a.keyCount,output:a.output}),u=[...s];for(let[e,r]of a.columns.entries()){let o=`${n}-c${e}`,s=new Set(r.statistics),d=r.output,f=i(t,`${o}-value-bits`,`uint32`,c.rowCount);u.push(l(t,{id:`${o}-value-bits`,operation:k,variant:`value-bits`,bindings:[h(`values`,r.values,`f32`),g(`valueBits`,f)],invocationCount:c.rowCount,body:`valueBits[valueBitsOffset + index] = bitcast<u32>(values[valuesOffset + index]);`}));let p=[`variance`,`standardDeviation`,`skewness`,`kurtosis`].some(e=>s.has(e)),m=w(t,{id:o,operation:k,structure:c,valueBits:f,needs:{sums:s.has(`sum`)||s.has(`mean`),minimum:s.has(`minimum`),maximum:s.has(`maximum`),moments:p,zScore:s.has(`zScore`)},variance:this.variance,sumScale:this.sumScale,output:d});u.push(...m.nodes),(s.has(`median`)||s.has(`percentiles`)||s.has(`mode`)||s.has(`uniqueCount`))&&u.push(...O(t,{id:o,operation:k,structure:c,valueBits:f,finiteCounts:m.finiteCounts,percentiles:a.percentiles,output:d}))}return u}};export{F as t};