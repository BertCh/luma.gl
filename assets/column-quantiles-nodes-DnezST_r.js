import{_ as e,d as t,w as n}from"./resources-CwvhEOBe.js";import{s as r}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as i}from"./column-classification-shared-Cj6aSTEo.js";import{t as a}from"./order-statistic-wgsl-w2Gvl3O6.js";var o={lower:0,higher:1,nearest:2,linear:3,midpoint:4};function s(e){return 4+e}function c(e,n){let r=e.quantileCount??e.quantiles.length;if(t(`GPUColumnQuantiles`,`quantileCount`,r,1,64),e.quantiles.length>r)throw Error(`GPUColumnQuantiles probabilities exceed quantileCount`);let i=s(r),a=n??new Float32Array(i);if(a.length<i)throw Error(`GPUColumnQuantiles parameter target must hold ${i} elements`);let c=e.interpolation??`linear`,l=o[c];if(l===void 0)throw Error(`GPUColumnQuantiles unknown interpolation ${String(c)}`);let[u,d]=e.filterRange??[0,1];a[0]=l,a[1]=u,a[2]=d,a[3]=0;for(let t=0;t<r;t++)a[4+t]=t<e.quantiles.length?e.quantiles[t]:NaN;return a}var l=256,u=256,d=4,f=8,p=16,m=256;function h(t,o){let{id:s,operation:c,values:h,mask:g,parameters:_,quantileCount:v,output:y,gate:b}=o;e(s,t,[h,g,_,b?.view,y.quantiles,y.validCount,y.outputMask,y.filterBounds]);let x=h.length,S=!!(y.outputMask||y.filterBounds),C=2*v+(S?2:0),w=n(t,`${s}-histogram`,`uint32`,C*u),T=4*C+2,E=n(t,`${s}-state`,`uint32`,T),D=Math.min(m,Math.max(1,Math.ceil(x/(l*p)))),O=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),k=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),A=b?[O(`gate`,b.view,`u32`)]:[],j=b?`gate[gateOffset] == ${b.value>>>0}u`:`true`,M=`
const QUANTILE_COUNT: u32 = ${v}u;
const TARGET_COUNT: u32 = ${C}u;
const HEADER_LENGTH: u32 = 4u;
const STATE_PREFIX: u32 = 0u;
const STATE_RANK: u32 = ${C}u;
const STATE_SLOT: u32 = ${2*C}u;
const STATE_ACTIVE_PREFIX: u32 = ${3*C}u;
const STATE_ACTIVE_COUNT: u32 = ${4*C}u;
const STATE_VALID_COUNT: u32 = ${4*C+1}u;
${i}
${a}

fn isMissingValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
`,N=`
// Round half to even; the fraction h - floor(h) is exact in f32.
fn roundHalfEven(h: f32) -> f32 {
  let whole = floor(h);
  let fraction = h - whole;
  if (fraction < 0.5) {
    return whole;
  }
  if (fraction > 0.5) {
    return whole + 1.0;
  }
  return select(whole, whole + 1.0, (u32(whole) & 1u) == 1u);
}

// The product is correctly rounded in WGSL, so the CPU oracle matches it with Math.fround.
fn getQuantileHeight(validCount: u32, p: f32) -> f32 {
  return f32(validCount - 1u) * p;
}

fn getQuantileRanks(validCount: u32, p: f32) -> vec2<u32> {
  let h = getQuantileHeight(validCount, p);
  let lower = floor(h);
  let upper = ceil(h);
  let code = u32(params[paramsOffset]);
  var first = lower;
  var second = lower;
  if (code == 1u) {
    first = upper;
    second = upper;
  } else if (code == 2u) {
    first = roundHalfEven(h);
    second = first;
  } else if (code == 3u) {
    second = min(lower + 1.0, f32(validCount - 1u));
  } else if (code == 4u) {
    second = upper;
  }
  return vec2<u32>(u32(first), u32(second));
}

fn getTargetRank(targetIndex: u32, validCount: u32) -> u32 {
  if (validCount == 0u) {
    return 0u;
  }
  if (targetIndex < 2u * QUANTILE_COUNT) {
    let p = params[paramsOffset + HEADER_LENGTH + targetIndex / 2u];
    if (!isOrderStatisticProbability(p)) {
      return 0u;
    }
    let ranks = getQuantileRanks(validCount, p);
    return select(ranks.x, ranks.y, (targetIndex & 1u) == 1u);
  }
  let lastIndex = f32(validCount - 1u);
  if (targetIndex == 2u * QUANTILE_COUNT) {
    let lowerFraction = params[paramsOffset + 1u];
    let fraction = select(clamp(lowerFraction, 0.0, 1.0), 0.0, isNanBits(lowerFraction));
    return u32(clamp(floor(f32(validCount) * fraction), 0.0, lastIndex));
  }
  let upperFraction = params[paramsOffset + 2u];
  let fraction = select(clamp(upperFraction, 0.0, 1.0), 1.0, isNanBits(upperFraction));
  return u32(clamp(ceil(f32(validCount) * fraction) - 1.0, 0.0, lastIndex));
}
`,P=[];P.push(r(t,{id:`${s}-init`,operation:c,variant:`quantile-init`,bindings:[k(`histogram`,w),...A],invocationCount:w.length,body:`if (${j}) {
    histogram[histogramOffset + index] = 0u;
  }`}));for(let e=0;e<d;e++){let n=24-8*e,i=f*u,a=e===0?`let slot = 0u;`:`let prefixValue = key >> ${n+8}u;
      var low = 0u;
      var high = activeCount;
      while (low < high) {
        let middle = (low + high) / 2u;
        if (state[stateOffset + STATE_ACTIVE_PREFIX + middle] < prefixValue) {
          low = middle + 1u;
        } else {
          high = middle;
        }
      }
      if (low >= activeCount || state[stateOffset + STATE_ACTIVE_PREFIX + low] != prefixValue) {
        continue;
      }
      let slot = low;`;P.push(r(t,{id:`${s}-histogram-${e}`,operation:c,variant:`quantile-histogram-${e}`,bindings:[O(`values`,h,`f32`),...g?[O(`rowMask`,g,`u32`)]:[],O(`state`,E,`u32`),k(`histogram`,w,`atomic<u32>`),...A],invocationCount:D*l,workgroupSize:l,guardIndex:!1,declarations:`${M}
const ROW_COUNT: u32 = ${x}u;
const LOCAL_SLOT_COUNT: u32 = ${f}u;
const LOCAL_BIN_COUNT: u32 = ${i}u;
var<workgroup> localHistogram: array<atomic<u32>, ${i}>;`,body:`for (var bin = localInvocationIndex; bin < LOCAL_BIN_COUNT; bin = bin + ${l}u) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  let isEnabled = ${j};
  if (isEnabled) {
    let activeCount = ${e===0?`1u`:`state[stateOffset + STATE_ACTIVE_COUNT]`};
    for (var row = index; row < ROW_COUNT; row = row + INVOCATION_COUNT) {
      let value = values[valuesOffset + row];
      if (isMissingValue(value)) {
        continue;
      }
      ${g?`if (rowMask[rowMaskOffset + row] == 0u) {
        continue;
      }`:``}
      let key = getOrderStatisticKey(value);
      ${a}
      let digit = (key >> ${n}u) & 255u;
      if (slot < LOCAL_SLOT_COUNT) {
        atomicAdd(&localHistogram[slot * 256u + digit], 1u);
      } else {
        atomicAdd(&histogram[histogramOffset + slot * 256u + digit], 1u);
      }
    }
  }
  workgroupBarrier();
  if (isEnabled) {
    for (var bin = localInvocationIndex; bin < LOCAL_BIN_COUNT; bin = bin + ${l}u) {
      let count = atomicLoad(&localHistogram[bin]);
      if (count != 0u && bin < ${C*u}u) {
        atomicAdd(&histogram[histogramOffset + bin], count);
      }
    }
  }`}));let o=e===0,p=e===d-1;P.push(r(t,{id:`${s}-select-${e}`,operation:c,variant:`quantile-select-${e}`,bindings:[k(`histogram`,w),k(`state`,E),O(`params`,_,`f32`),...A],invocationCount:l,workgroupSize:l,guardIndex:!1,declarations:`${M}
${N}
var<workgroup> sharedPrefix: array<u32, ${C}>;
var<workgroup> sharedFirst: array<u32, ${C}>;
var<workgroup> sharedValidCount: u32;`,body:`let isEnabled = ${j};
  let targetIndex = localInvocationIndex;
  ${o?`if (targetIndex == 0u) {
    var total = 0u;
    if (isEnabled) {
      for (var bin = 0u; bin < 256u; bin = bin + 1u) {
        total = total + histogram[histogramOffset + bin];
      }
      state[stateOffset + STATE_VALID_COUNT] = total;
    }
    sharedValidCount = total;
  }
  workgroupBarrier();
  let validCount = sharedValidCount;`:``}
  if (isEnabled && targetIndex < TARGET_COUNT) {
    var prefix = ${o?`0u`:`state[stateOffset + STATE_PREFIX + targetIndex]`};
    var remaining = ${o?`getTargetRank(targetIndex, validCount)`:`state[stateOffset + STATE_RANK + targetIndex]`};
    let slot = ${o?`0u`:`state[stateOffset + STATE_SLOT + targetIndex]`};
    var digit = 255u;
    for (var bin = 0u; bin < 256u; bin = bin + 1u) {
      let count = histogram[histogramOffset + slot * 256u + bin];
      if (remaining < count) {
        digit = bin;
        break;
      }
      remaining = remaining - count;
    }
    prefix = (prefix << 8u) | digit;
    sharedPrefix[targetIndex] = prefix;
    state[stateOffset + STATE_PREFIX + targetIndex] = prefix;
    state[stateOffset + STATE_RANK + targetIndex] = remaining;
  }
  // The histogram of this pass is consumed: clear it for the next pass.
  workgroupBarrier();
  if (isEnabled) {
    for (var cell = targetIndex; cell < ${C*u}u; cell = cell + ${l}u) {
      histogram[histogramOffset + cell] = 0u;
    }
  }
  ${p?``:`if (isEnabled && targetIndex < TARGET_COUNT) {
    var first = 1u;
    for (var other = 0u; other < targetIndex; other = other + 1u) {
      if (sharedPrefix[other] == sharedPrefix[targetIndex]) {
        first = 0u;
        break;
      }
    }
    sharedFirst[targetIndex] = first;
  }
  workgroupBarrier();
  if (isEnabled && targetIndex < TARGET_COUNT) {
    let prefix = sharedPrefix[targetIndex];
    var slot = 0u;
    for (var other = 0u; other < TARGET_COUNT; other = other + 1u) {
      if (sharedFirst[other] == 1u && sharedPrefix[other] < prefix) {
        slot = slot + 1u;
      }
    }
    state[stateOffset + STATE_SLOT + targetIndex] = slot;
    if (sharedFirst[targetIndex] == 1u) {
      state[stateOffset + STATE_ACTIVE_PREFIX + slot] = prefix;
    }
    if (targetIndex == 0u) {
      var distinct = 0u;
      for (var other = 0u; other < TARGET_COUNT; other = other + 1u) {
        distinct = distinct + sharedFirst[other];
      }
      state[stateOffset + STATE_ACTIVE_COUNT] = distinct;
    }
  }`}`}))}let F=[O(`state`,E,`u32`),O(`params`,_,`f32`),k(`quantilesOut`,y.quantiles,`f32`),k(`validCountOut`,y.validCount,`u32`),...y.filterBounds?[k(`filterBoundsOut`,y.filterBounds,`f32`)]:[],...A];return P.push(r(t,{id:`${s}-finish`,operation:c,variant:`quantile-finish`,bindings:F,invocationCount:v+1,declarations:`${M}
${N}`,body:`if (!(${j})) {
    return;
  }
  let validCount = state[stateOffset + STATE_VALID_COUNT];
  if (index < QUANTILE_COUNT) {
    var result = getNaN();
    let p = params[paramsOffset + HEADER_LENGTH + index];
    if (validCount > 0u && isOrderStatisticProbability(p)) {
      let first = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * index]);
      let second = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * index + 1u]);
      result = first;
      let code = u32(params[paramsOffset]);
      // validCount is below 2^24, so this is a runtime zero that blocks multiply-add fusion.
      let zero = validCount >> 31u;
      if (code == 3u) {
        let h = getQuantileHeight(validCount, p);
        result = interpolateOrderStatistic(first, second, h - floor(h), zero);
      } else if (code == 4u) {
        result = getOrderStatisticMidpoint(first, second, zero);
      }
    }
    quantilesOut[quantilesOutOffset + index] = result;
  } else {
    validCountOut[validCountOutOffset] = validCount;
    ${y.filterBounds?`let hasRows = validCount > 0u;
    let lowerBound = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT]);
    let upperBound = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT + 1u]);
    filterBoundsOut[filterBoundsOutOffset] = select(getNaN(), lowerBound, hasRows);
    filterBoundsOut[filterBoundsOutOffset + 1u] = select(getNaN(), upperBound, hasRows);`:``}
  }`})),y.outputMask&&P.push(r(t,{id:`${s}-filter-mask`,operation:c,variant:`quantile-filter-mask`,bindings:[O(`values`,h,`f32`),...g?[O(`rowMask`,g,`u32`)]:[],O(`state`,E,`u32`),k(`filterMaskOut`,y.outputMask,`u32`),...A],invocationCount:x,declarations:M,body:`if (!(${j})) {
    return;
  }
  var inside = 0u;
  let value = values[valuesOffset + index];
  if (state[stateOffset + STATE_VALID_COUNT] > 0u && !isMissingValue(value)) {
    ${g?`if (rowMask[rowMaskOffset + index] != 0u) {`:`{`}
      let key = getOrderStatisticKey(value);
      let lowerKey = state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT];
      let upperKey = state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT + 1u];
      inside = select(0u, 1u, key >= lowerKey && key <= upperKey);
    }
  }
  filterMaskOut[filterMaskOutOffset + index] = inside;`})),P}export{s as n,c as r,h as t};