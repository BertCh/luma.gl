import{_ as e,d as t,h as n,l as r,v as i,w as a,y as o}from"./resources-CwvhEOBe.js";import{l as s,n as c,s as l}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as u}from"./wgsl-common-BNRVzgd8.js";import{t as d}from"./column-classification-shared-Cj6aSTEo.js";function f(e){return r(e,4)}var p=4294967295,m={count:0,nullCount:1,minimum:2,maximum:3,sum:4,mean:5,variance:6,sampleVariance:7,standardDeviation:8,distinctEstimate:9,overflowCount:10};function h(e){return e*2}var g=256*8,_=7,v=4,y=4,b=`
struct Moments {
  count: f32,
  nullCount: f32,
  finiteCount: f32,
  mean: f32,
  m2: f32,
  sum: f32,
  origin: f32,
}

fn emptyMoments() -> Moments {
  return Moments(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

fn mergeMoments(left: Moments, right: Moments) -> Moments {
  var result = emptyMoments();
  result.count = left.count + right.count;
  result.nullCount = left.nullCount + right.nullCount;
  result.sum = left.sum + right.sum;
  if (right.finiteCount == 0.0) {
    result.finiteCount = left.finiteCount;
    result.origin = left.origin;
    result.mean = left.mean;
    result.m2 = left.m2;
  } else if (left.finiteCount == 0.0) {
    result.finiteCount = right.finiteCount;
    result.origin = right.origin;
    result.mean = right.mean;
    result.m2 = right.m2;
  } else {
    let count = left.finiteCount + right.finiteCount;
    let delta = (right.origin - left.origin) + (right.mean - left.mean);
    result.origin = left.origin;
    result.finiteCount = count;
    result.mean = left.mean + delta * (right.finiteCount / count);
    result.m2 = left.m2 + right.m2 + delta * delta * (left.finiteCount * right.finiteCount / count);
  }
  return result;
}

var<workgroup> sharedMoments: array<Moments, 256>;

fn reduceSharedMoments(local: u32) {
  for (var stride = ${256/2}u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedMoments[local] = mergeMoments(sharedMoments[local], sharedMoments[local + stride]);
    }
  }
}
`,x=`
fn getBinIndex(value: f32, minimum: f32, width: f32) -> u32 {
  let difference = value - minimum;
  var bin = floor(difference / width);
  bin = clamp(bin, 0.0, f32(BIN_COUNT - 1u));
  if (difference < bin * width) {
    bin = bin - 1.0;
  } else if (difference >= (bin + 1.0) * width) {
    bin = bin + 1.0;
  }
  return u32(clamp(bin, 0.0, f32(BIN_COUNT - 1u)));
}
`,ee=`
fn hashBits(bits: u32) -> u32 {
  var h = bits;
  h = h ^ (h >> 16u);
  h = h * 0x85ebca6bu;
  h = h ^ (h >> 13u);
  h = h * 0xc2b2ae35u;
  h = h ^ (h >> 16u);
  return h;
}
`,S=1024;function C(e){let{maximumSlotCount:t,slotCount:n,targetName:r,targetBase:i,slotBody:a}=e,o=`const NO_SLOT: u32 = 0xffffffffu;
fn getSlot(index: u32, column: u32) -> u32 {
  ${a}
}`,s=`let column = columnCursor[columnCursorOffset];`,c=`${r}[${r}Offset + ${i} + `;return t>S||t<1?{workgroupSize:256,declarations:`${e.declarations}\n${o}`,body:`${s}
  let slot = getSlot(index, column);
  if (slot != NO_SLOT) {
    atomicAdd(&${c}slot], 1u);
  }`}:{declarations:`${e.declarations}
${o}
var<workgroup> localCounts: array<atomic<u32>, ${t}>;`,body:`${s}
  let slotCount = ${n};
  for (var slot = localInvocationIndex; slot < slotCount; slot += 256u) {
    atomicStore(&localCounts[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let slot = getSlot(index, column);
    if (slot != NO_SLOT) {
      atomicAdd(&localCounts[slot], 1u);
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < slotCount; slot += 256u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&${c}slot], count);
    }
  }`,guardIndex:!1,workgroupSize:256}}function w(e){return e===16?.673:e===32?.697:e===64?.709:.7213/(1+1.079/e)}function T(e){return e.reduce((e,t)=>e+(t.kind===`category`?t.categoryCount:0),0)}var E=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),D=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`});function O(e){let t=t=>`array<u32, ${e.length}>(${t.map(e=>`${e}u`).join(`, `)})`;return`const COLUMN_COUNT: u32 = ${e.length}u;
var<private> columnCategoryCounts: array<u32, ${e.length}> = ${t(e.map(e=>e.categoryCount))};
var<private> columnCategoryBases: array<u32, ${e.length}> = ${t(e.map(e=>e.categoryBase))};`}var k=`let column = columnCursor[columnCursorOffset];`;function A(e,t){let{id:n,operation:r,columns:i,rowCount:o,mask:f,parameters:m,output:h}=t,{histogramBinCount:S,hyperLogLogPrecision:w,topCategoryCount:A}=t,P=i.length,F=2**w,I=Math.ceil(o/g),L=T(i),R=!!h.histograms,z=!!(h.topCategories&&h.topCategoryCounts),te=i.some(e=>e.kind===`category`)&&(R||z),B=f?[E(`rowMask`,f,`u32`)]:[],ne=e=>e.kind===`numeric`?`f32`:`u32`,V=(t,r)=>a(e,`${n}-${t}`,`uint32`,r),H=(t,r)=>a(e,`${n}-${t}`,`float32`,r),U=H(`partials`,P*I*_),W=H(`records`,P*_),G=V(`keys`,P*v),K=H(`domains`,P*y),q=h.hyperLogLogRegisters??V(`registers`,P*F),J=te?V(`category-counts`,L):void 0,Y=V(`column-cursor`,1),X=O(i),Z=E(`columnCursor`,Y,`u32`),Q=[];Q.push(l(e,{id:`${n}-init`,operation:r,variant:`init`,bindings:[D(`keys`,G),D(`columnCursor`,Y)],invocationCount:P*v,body:`keys[keysOffset + index] = select(0xffffffffu, 0u, (index & 1u) == 1u);
  if (index == 0u) {
    columnCursor[columnCursorOffset] = 0u;
  }`}));let $=(t,i,a,o)=>Q.push(c(e,{id:`${n}-${t}`,operation:r,view:i,type:`u32`,value:o,componentCount:a}));$(`clear-registers`,q,P*F,`0u`),h.histograms&&$(`clear-histograms`,h.histograms,P*S,`0u`),J&&$(`clear-category-counts`,J,L,`0u`),z&&($(`clear-top-categories`,h.topCategories,P*A,`0xffffffffu`),$(`clear-top-category-counts`,h.topCategoryCounts,P*A,`0u`));for(let[t,a]of i.entries())Q.push(j(e,{id:`${n}-column-${t}-moments-tile`,operation:r,column:a,rowCount:o,tileCount:I,maskBinding:B,partials:U,keys:G,columnCursor:Y,columnConstants:X})),Q.push(l(e,{id:`${n}-column-${t}-advance`,operation:r,variant:`column-advance`,bindings:[D(`columnCursor`,Y)],invocationCount:1,declarations:`const COLUMN_COUNT: u32 = ${P}u;`,body:`columnCursor[columnCursorOffset] = (columnCursor[columnCursorOffset] + 1u) % COLUMN_COUNT;`}));Q.push(l(e,{id:`${n}-moments-merge`,operation:r,variant:`moments-merge`,bindings:[E(`partials`,U,`f32`),D(`records`,W,`f32`)],invocationCount:P*256,workgroupSize:256,guardIndex:!1,declarations:`const TILE_COUNT: u32 = ${I}u;
const STEP_COUNT: u32 = ${Math.ceil(I/256)}u;
${b}`,body:`let local = localInvocationIndex;
  let column = workgroupIndex;
  var accumulated = emptyMoments();
  for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
    let tile = step * 256u + local;
    if (tile < TILE_COUNT) {
      let base = partialsOffset + (column * TILE_COUNT + tile) * ${_}u;
      accumulated = mergeMoments(
        accumulated,
        Moments(partials[base], partials[base + 1u], partials[base + 2u], partials[base + 3u], partials[base + 4u], partials[base + 5u], partials[base + 6u])
      );
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = recordsOffset + column * ${_}u;
    records[base] = record.count;
    records[base + 1u] = record.nullCount;
    records[base + 2u] = record.finiteCount;
    records[base + 3u] = record.mean;
    records[base + 4u] = record.m2;
    records[base + 5u] = record.sum;
    records[base + 6u] = record.origin;
  }`})),R&&Q.push(l(e,{id:`${n}-domain`,operation:r,variant:`domain`,bindings:[E(`keys`,G,`u32`),...m?[E(`params`,m,`f32`)]:[],D(`domains`,K,`f32`)],invocationCount:P,declarations:`const INVERSE_BIN_COUNT: f32 = ${s(1/S)};
${d}
${u}`,body:`let column = index;
  let keyBase = keysOffset + column * ${v}u;
  var lo = getNaN();
  var hi = getNaN();
  ${m?`lo = params[paramsOffset + column * 2u];
  hi = params[paramsOffset + column * 2u + 1u];`:``}
  let finiteMinimumKey = keys[keyBase + 2u];
  let finiteMaximumKey = keys[keyBase + 3u];
  if (isNanBits(lo)) {
    lo = select(getNaN(), decodeOrderedKey(finiteMinimumKey), finiteMinimumKey != 0xffffffffu);
  }
  if (isNanBits(hi)) {
    hi = select(getNaN(), decodeOrderedKey(finiteMaximumKey), finiteMaximumKey != 0u);
  }
  var mode = 0.0;
  var width = 0.0;
  if (isFiniteBits(lo) && isFiniteBits(hi) && lo <= hi) {
    if (lo == hi) {
      mode = 2.0;
    } else {
      let range = hi - lo;
      width = range * INVERSE_BIN_COUNT;
      if (isFiniteBits(range) && isFiniteBits(width) && width > 0.0) {
        mode = 1.0;
      }
    }
  }
  let base = domainsOffset + column * ${y}u;
  domains[base] = lo;
  domains[base + 1u] = hi;
  domains[base + 2u] = width;
  domains[base + 3u] = mode;`}));let re=Math.max(0,...i.map(e=>e.kind===`category`?e.categoryCount:0));for(let[t,a]of i.entries()){let i=`${n}-column-${t}`,s=a.kind===`numeric`,c=ne(a);R&&s&&Q.push(l(e,{id:`${i}-histogram`,operation:r,variant:`histogram`,bindings:[E(`values`,a.values,`f32`),...B,E(`domains`,K,`f32`),D(`histogram`,h.histograms,`atomic<u32>`),Z],invocationCount:o,...C({maximumSlotCount:S,slotCount:`${S}u`,targetName:`histogram`,targetBase:`column * BIN_COUNT`,declarations:`const BIN_COUNT: u32 = ${S}u;
${x}`,slotBody:`${f?`if (rowMask[rowMaskOffset + index] == 0u) {
    return NO_SLOT;
  }`:``}
  let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return NO_SLOT;
  }
  let base = domainsOffset + column * ${y}u;
  let mode = domains[base + 3u];
  let lo = domains[base];
  if (mode == 0.0 || value < lo || value > domains[base + 1u]) {
    return NO_SLOT;
  }
  var bin = 0u;
  if (mode == 1.0) {
    bin = getBinIndex(value, lo, domains[base + 2u]);
  }
  return bin;`})})),Q.push(l(e,{id:`${i}-hyperloglog`,operation:r,variant:`hyperloglog`,bindings:[E(`values`,a.values,c),...B,D(`registers`,q,`atomic<u32>`),Z],invocationCount:o,declarations:`const PRECISION: u32 = ${w}u;
const REGISTER_COUNT: u32 = ${F}u;
${ee}`,body:`${k}
  // Equals getColumnProfileHashSeed(column).
  let seed = (column + 1u) * 0x9e3779b9u;
  ${f?`if (rowMask[rowMaskOffset + index] == 0u) {
    return;
  }`:``}
  ${s?`var bits = bitcast<u32>(values[valuesOffset + index]);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return;
  }
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }`:`let bits = values[valuesOffset + index];
  if (bits == ${p}u) {
    return;
  }`}
  let hash = hashBits(bits ^ seed);
  let registerIndex = hash >> (32u - PRECISION);
  let rank = min(countLeadingZeros(hash << PRECISION), 32u - PRECISION) + 1u;
  atomicMax(&registers[registersOffset + column * REGISTER_COUNT + registerIndex], rank);`})),J&&!s&&Q.push(l(e,{id:`${i}-category-counts`,operation:r,variant:`category-counts`,bindings:[E(`values`,a.values,`u32`),...B,D(`categoryCounts`,J,`atomic<u32>`),Z],invocationCount:o,...C({maximumSlotCount:re,slotCount:`columnCategoryCounts[column]`,targetName:`categoryCounts`,targetBase:`columnCategoryBases[column]`,declarations:X,slotBody:`${f?`if (rowMask[rowMaskOffset + index] == 0u) {
    return NO_SLOT;
  }`:``}
  let code = values[valuesOffset + index];
  if (code < columnCategoryCounts[column]) {
    return code;
  }
  return NO_SLOT;`})})),z&&J&&!s&&Q.push(M(e,{id:`${i}-top-categories`,operation:r,topCategoryCount:A,columnConstants:X,columnCursor:Y,categoryCounts:J,topCategories:h.topCategories,topCategoryCounts:h.topCategoryCounts})),Q.push(N(e,{id:`${i}-finish`,operation:r,column:a,columnConstants:X,columnCursor:Y,registerCount:F,hyperLogLogPrecision:w,histogramBinCount:S,records:W,keys:G,registers:q,categoryCounts:J,output:h}))}return Q}function j(e,t){let{column:n,rowCount:r,tileCount:i,maskBinding:a}=t,o=n.kind===`numeric`,s=a.length>0,c=o?`let bits = bitcast<u32>(value);
      if ((bits & 0x7fffffffu) > 0x7f800000u) {
        accumulated.nullCount = accumulated.nullCount + 1.0;
      } else {
        accumulated.count = accumulated.count + 1.0;
        let key = getOrderedKey(value);
        minimumKey = min(minimumKey, key);
        maximumKey = max(maximumKey, key);
        if ((bits & 0x7fffffffu) != 0x7f800000u) {
          finiteMinimumKey = min(finiteMinimumKey, key);
          finiteMaximumKey = max(finiteMaximumKey, key);
          if (accumulated.finiteCount == 0.0) {
            accumulated.origin = value;
          }
          let shiftedValue = value - accumulated.origin;
          let count = accumulated.finiteCount + 1.0;
          let delta = shiftedValue - accumulated.mean;
          accumulated.finiteCount = count;
          accumulated.mean = accumulated.mean + delta / count;
          accumulated.m2 = accumulated.m2 + delta * (shiftedValue - accumulated.mean);
          accumulated.sum = accumulated.sum + value;
        }
      }`:`if (value == ${p}u) {
        accumulated.nullCount = accumulated.nullCount + 1.0;
      } else {
        accumulated.count = accumulated.count + 1.0;
        if (value >= categoryCount) {
          accumulated.finiteCount = accumulated.finiteCount + 1.0;
        }
        minimumKey = min(minimumKey, value);
        maximumKey = max(maximumKey, value);
      }`;return l(e,{id:t.id,operation:t.operation,variant:`moments-tile`,bindings:[E(`values`,n.values,o?`f32`:`u32`),...a,D(`partials`,t.partials,`f32`),D(`keys`,t.keys,`atomic<u32>`),E(`columnCursor`,t.columnCursor,`u32`)],invocationCount:i*256,workgroupSize:256,guardIndex:!1,declarations:`const ROW_COUNT: u32 = ${r}u;
const TILE_COUNT: u32 = ${i}u;
const TILE_SIZE: u32 = ${g}u;
const ROWS_PER_INVOCATION: u32 = 8u;
${t.columnConstants}
${d}
${b}
var<workgroup> sharedKeys: array<atomic<u32>, ${v}>;`,body:`let local = localInvocationIndex;
  let tile = workgroupIndex;
  ${k}
  let categoryCount = columnCategoryCounts[column];
  if (local < ${v}u) {
    atomicStore(&sharedKeys[local], select(0xffffffffu, 0u, (local & 1u) == 1u));
  }
  workgroupBarrier();
  var accumulated = emptyMoments();
  var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  var finiteMinimumKey = 0xffffffffu;
  var finiteMaximumKey = 0u;
  for (var step = 0u; step < ROWS_PER_INVOCATION; step = step + 1u) {
    // Uniform trip count: rows past the end are skipped by a predicate, never by a break.
    let row = tile * TILE_SIZE + step * 256u + local;
    if (row < ROW_COUNT${s?` && rowMask[rowMaskOffset + row] != 0u`:``}) {
      let value = values[valuesOffset + row];
      ${c}
    }
  }
  sharedMoments[local] = accumulated;
  atomicMin(&sharedKeys[0], minimumKey);
  atomicMax(&sharedKeys[1], maximumKey);
  atomicMin(&sharedKeys[2], finiteMinimumKey);
  atomicMax(&sharedKeys[3], finiteMaximumKey);
  reduceSharedMoments(local);
  workgroupBarrier();
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = partialsOffset + (column * TILE_COUNT + tile) * ${_}u;
    partials[base] = record.count;
    partials[base + 1u] = record.nullCount;
    partials[base + 2u] = record.finiteCount;
    partials[base + 3u] = record.mean;
    partials[base + 4u] = record.m2;
    partials[base + 5u] = record.sum;
    partials[base + 6u] = record.origin;
    let keyBase = keysOffset + column * ${v}u;
    atomicMin(&keys[keyBase], atomicLoad(&sharedKeys[0]));
    atomicMax(&keys[keyBase + 1u], atomicLoad(&sharedKeys[1]));
    atomicMin(&keys[keyBase + 2u], atomicLoad(&sharedKeys[2]));
    atomicMax(&keys[keyBase + 3u], atomicLoad(&sharedKeys[3]));
  }`})}function M(e,t){let{topCategoryCount:n}=t;return l(e,{id:t.id,operation:t.operation,variant:`top-categories`,bindings:[E(`categoryCounts`,t.categoryCounts,`u32`),D(`topCategories`,t.topCategories),D(`topCategoryCounts`,t.topCategoryCounts),E(`columnCursor`,t.columnCursor,`u32`)],invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`${t.columnConstants}
const TOP_COUNT: u32 = ${n}u;
var<workgroup> bestCounts: array<u32, 256>;
var<workgroup> bestCodes: array<u32, 256>;`,body:`let local = localInvocationIndex;
  ${k}
  let categoryCount = columnCategoryCounts[column];
  let base = columnCategoryBases[column];
  let stepCount = (categoryCount + 255u) / 256u;
  var previousCount = 0u;
  var previousCode = 0u;
  var hasPrevious = false;
  for (var rank = 0u; rank < TOP_COUNT; rank = rank + 1u) {
    var bestCount = 0u;
    var bestCode = 0xffffffffu;
    for (var step = 0u; step < stepCount; step = step + 1u) {
      let code = step * 256u + local;
      if (code < categoryCount) {
        let count = categoryCounts[categoryCountsOffset + base + code];
        // Strictly after the previous pick in (count descending, code ascending) order.
        let isCandidate = count > 0u && (!hasPrevious || count < previousCount || (count == previousCount && code > previousCode));
        if (isCandidate && count > bestCount) {
          bestCount = count;
          bestCode = code;
        }
      }
    }
    bestCounts[local] = bestCount;
    bestCodes[local] = bestCode;
    for (var stride = ${256/2}u; stride > 0u; stride = stride >> 1u) {
      workgroupBarrier();
      if (local < stride) {
        let otherCount = bestCounts[local + stride];
        let otherCode = bestCodes[local + stride];
        if (otherCount > bestCounts[local] || (otherCount == bestCounts[local] && otherCode < bestCodes[local])) {
          bestCounts[local] = otherCount;
          bestCodes[local] = otherCode;
        }
      }
    }
    workgroupBarrier();
    let pickedCount = bestCounts[0];
    let pickedCode = bestCodes[0];
    if (local == 0u) {
      topCategories[topCategoriesOffset + column * TOP_COUNT + rank] = select(0xffffffffu, pickedCode, pickedCount > 0u);
      topCategoryCounts[topCategoryCountsOffset + column * TOP_COUNT + rank] = pickedCount;
    }
    previousCount = pickedCount;
    previousCode = pickedCode;
    hasPrevious = true;
    workgroupBarrier();
  }`})}function N(e,t){let{column:n,registerCount:r,histogramBinCount:i,output:a}=t,o=n.kind===`numeric`,c=w(r)*r*r,u=!o&&a.histograms&&t.categoryCounts,f=m,p=[E(`records`,t.records,`f32`),E(`keys`,t.keys,`u32`),E(`registers`,t.registers,`u32`),D(`statistics`,a.statistics,`f32`)];a.counts&&p.push(D(`countsOut`,a.counts)),u&&p.push(E(`categoryCounts`,t.categoryCounts,`u32`),D(`histogram`,a.histograms)),p.push(D(`columnCursor`,t.columnCursor));let h=o?`let hasValues = count > 0.0;
    let hasFinite = finiteCount > 0.0;
    let variance = select(nan, m2 / finiteCount, hasFinite);
    statistics[statisticsBase + ${f.minimum}u] = select(nan, decodeOrderedKey(minimumKey), hasValues);
    statistics[statisticsBase + ${f.maximum}u] = select(nan, decodeOrderedKey(maximumKey), hasValues);
    statistics[statisticsBase + ${f.sum}u] = select(nan, sum, hasFinite);
    statistics[statisticsBase + ${f.mean}u] = select(nan, mean, hasFinite);
    statistics[statisticsBase + ${f.variance}u] = variance;
    statistics[statisticsBase + ${f.sampleVariance}u] = select(nan, m2 / (finiteCount - 1.0), finiteCount >= 2.0);
    statistics[statisticsBase + ${f.standardDeviation}u] = select(nan, sqrt(variance), hasFinite);
    statistics[statisticsBase + ${f.overflowCount}u] = 0.0;`:`let hasValues = count > 0.0;
    statistics[statisticsBase + ${f.minimum}u] = select(nan, f32(minimumKey), hasValues);
    statistics[statisticsBase + ${f.maximum}u] = select(nan, f32(maximumKey), hasValues);
    statistics[statisticsBase + ${f.sum}u] = nan;
    statistics[statisticsBase + ${f.mean}u] = nan;
    statistics[statisticsBase + ${f.variance}u] = nan;
    statistics[statisticsBase + ${f.sampleVariance}u] = nan;
    statistics[statisticsBase + ${f.standardDeviation}u] = nan;
    statistics[statisticsBase + ${f.overflowCount}u] = finiteCount;`;return l(e,{id:t.id,operation:t.operation,variant:`finish`,bindings:p,invocationCount:256,workgroupSize:256,guardIndex:!1,declarations:`${t.columnConstants}
const REGISTER_COUNT: u32 = ${r}u;
const REGISTER_STEP_COUNT: u32 = ${Math.ceil(r/256)}u;
const ALPHA_TIMES_SQUARE: f32 = ${s(c)};
const BIN_COUNT: u32 = ${i}u;
${d}
var<workgroup> sharedColumn: u32;
var<workgroup> sharedInverseSums: array<f32, 256>;
var<workgroup> sharedZeroCounts: array<u32, 256>;`,body:`let local = localInvocationIndex;
  if (local == 0u) {
    sharedColumn = columnCursor[columnCursorOffset];
  }
  let column = workgroupUniformLoad(&sharedColumn);
  var inverseSum = 0.0;
  var zeroCount = 0u;
  for (var step = 0u; step < REGISTER_STEP_COUNT; step = step + 1u) {
    let registerIndex = step * 256u + local;
    if (registerIndex < REGISTER_COUNT) {
      let rank = registers[registersOffset + column * REGISTER_COUNT + registerIndex];
      // 2^-rank is exact: rank is at most ${33-t.hyperLogLogPrecision}.
      inverseSum = inverseSum + bitcast<f32>((127u - rank) << 23u);
      zeroCount = zeroCount + select(0u, 1u, rank == 0u);
    }
  }
  sharedInverseSums[local] = inverseSum;
  sharedZeroCounts[local] = zeroCount;
  for (var stride = ${256/2}u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedInverseSums[local] = sharedInverseSums[local] + sharedInverseSums[local + stride];
      sharedZeroCounts[local] = sharedZeroCounts[local] + sharedZeroCounts[local + stride];
    }
  }
  ${u?`for (var bin = local; bin < BIN_COUNT; bin = bin + 256u) {
    histogram[histogramOffset + column * BIN_COUNT + bin] = select(0u, categoryCounts[categoryCountsOffset + columnCategoryBases[column] + bin], bin < columnCategoryCounts[column]);
  }`:``}
  if (local == 0u) {
    let recordBase = recordsOffset + column * ${_}u;
    let count = records[recordBase];
    let nullCount = records[recordBase + 1u];
    let finiteCount = records[recordBase + 2u];
    let mean = records[recordBase + 6u] + records[recordBase + 3u];
    let m2 = records[recordBase + 4u];
    let sum = records[recordBase + 5u];
    let keyBase = keysOffset + column * ${v}u;
    let minimumKey = keys[keyBase];
    let maximumKey = keys[keyBase + 1u];
    let nan = getNaN();
    let statisticsBase = statisticsOffset + column * 11u;
    statistics[statisticsBase + ${f.count}u] = count;
    statistics[statisticsBase + ${f.nullCount}u] = nullCount;
    ${h}
    let registerCountFloat = f32(REGISTER_COUNT);
    var estimate = ALPHA_TIMES_SQUARE / sharedInverseSums[0];
    let zeros = sharedZeroCounts[0];
    if (zeros == REGISTER_COUNT) {
      estimate = 0.0;
    } else if (estimate <= 2.5 * registerCountFloat && zeros > 0u) {
      // Linear counting m * ln(m / zeros). For few occupied registers the ratio is close to 1 and
      // log() cancels, so use the series of ln(1 + u) with the exact integer u = (m - zeros) / zeros.
      let occupied = f32(REGISTER_COUNT - zeros);
      let ratio = occupied / f32(zeros);
      if (ratio < 0.0625) {
        let r2 = ratio * ratio;
        estimate = registerCountFloat * (ratio - r2 * (0.5 - ratio * (0.33333334 - ratio * (0.25 - ratio * 0.2))));
      } else {
        estimate = registerCountFloat * log(registerCountFloat / f32(zeros));
      }
    }
    statistics[statisticsBase + ${f.distinctEstimate}u] = estimate;
    ${a.counts?`countsOut[countsOutOffset + column * 2u] = u32(count);
    countsOut[countsOutOffset + column * 2u + 1u] = u32(nullCount);`:``}
    columnCursor[columnCursorOffset] = (column + 1u) % COLUMN_COUNT;
  }`})}var P=`GPUColumnProfile`,F=2**24,I=2**20,L=12,R=10,z=class{id;props;rowCount;hyperLogLogPrecision;topCategoryCount;columns;constructor(e){this.id=e.id??`column-profile`,this.props=e;let r=this.id,{output:i}=e,a=e.columns.length;t(r,`column count`,a,1,16);let{histogramBinCount:s}=e;t(r,`histogramBinCount`,s,1,I),this.hyperLogLogPrecision=e.hyperLogLogPrecision??L,t(r,`hyperLogLogPrecision`,this.hyperLogLogPrecision,4,16),this.topCategoryCount=e.topCategoryCount??R,t(r,`topCategoryCount`,this.topCategoryCount,1,64);let c=e.columns[0].values.length;if(c<1)throw Error(`${r} needs at least one row`);if(c>F)throw Error(`${r} supports at most ${F} rows`);this.rowCount=c;let l=0;if(this.columns=e.columns.map((e,n)=>{let i=`columns[${n}]`;if(e.values.length!==c)throw Error(`${r} ${i} length must equal columns[0] length`);if(e.kind===`category`){o(e.values,[`uint32`],`${r} ${i}.values`),t(r,`${i}.categoryCount`,e.categoryCount,1,F);let n=l;return l+=e.categoryCount,{kind:`category`,values:e.values,categoryCount:e.categoryCount,categoryBase:n}}if(e.kind!==void 0&&e.kind!==`numeric`)throw Error(`${r} ${i}.kind must be 'numeric' or 'category'`);return o(e.values,[`float32`],`${r} ${i}.values`),{kind:`numeric`,values:e.values,categoryCount:0,categoryBase:0}}),e.mask&&(o(e.mask,[`uint32`],`${r} mask`),e.mask.length!==c))throw Error(`${r} mask length must equal the column length`);if(e.parameters){o(e.parameters,[`float32`],`${r} parameters`);let t=h(a);if(e.parameters.length<t)throw Error(`${r} parameters must hold ${t} float32 values`)}let u=(e,t,n)=>{if(t.length<n)throw Error(`${r} output.${e} must hold ${n} rows`)};o(i.statistics,[`float32`],`${r} output.statistics`),u(`statistics`,i.statistics,a*11);for(let[e,t,n]of[[`counts`,i.counts,a*2],[`histograms`,i.histograms,a*s],[`hyperLogLogRegisters`,i.hyperLogLogRegisters,a*2**this.hyperLogLogPrecision],[`topCategories`,i.topCategories,a*this.topCategoryCount],[`topCategoryCounts`,i.topCategoryCounts,a*this.topCategoryCount]])t&&(o(t,[`uint32`],`${r} output.${e}`),u(e,t,n));if(!!i.topCategories!=!!i.topCategoryCounts)throw Error(`${r} output.topCategories and output.topCategoryCounts go together`);n(r,[i.statistics,i.counts,i.histograms,i.hyperLogLogRegisters,i.topCategories,i.topCategoryCounts],[...e.columns.map(e=>e.values),e.mask,e.parameters])}getCommandNodes(t){let{props:n,id:r}=this,{output:a}=n;return e(r,t,[...n.columns.map(e=>e.values),n.mask,n.parameters,a.statistics,a.counts,a.histograms,a.hyperLogLogRegisters,a.topCategories,a.topCategoryCounts]),i(r,`category columns categoryCount total`,this.columns.reduce((e,t)=>e+t.categoryCount,0),f(t.device)),A(t,{id:r,operation:P,columns:this.columns,rowCount:this.rowCount,mask:n.mask,parameters:n.parameters,histogramBinCount:n.histogramBinCount,hyperLogLogPrecision:this.hyperLogLogPrecision,topCategoryCount:this.topCategoryCount,output:a})}};export{m as n,z as t};