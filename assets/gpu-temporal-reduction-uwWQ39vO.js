import{_ as e,d as t,f as n,h as r,w as i,y as a}from"./resources-CwvhEOBe.js";import{t as o}from"./gpu-visibility-workflow-CuWho9Ae.js";import{a as s,r as c,s as l}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as u,r as d,t as f,u as p}from"./wgsl-common-BNRVzgd8.js";import{t as m}from"./time-words-C-gLAZQw.js";function h(e,t){let n=t.timestamps.format===`uint32x2`,r=t.timestamps.length,i=n?`${m}
override BUCKET_COUNT: u32;
override ROW_COUNT: u32;

fn getRowTime(row: u32) -> vec2<u32> {
  return vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
}

// Same exact division as the bucket kernel: high word first, then the low word bit by bit.
fn isQuotientAtLeastBucketCount(difference: vec2<u32>, width: u32) -> bool {
  let quotientHigh = difference.y / width;
  var remainder = difference.y % width;
  var quotientLow = 0u;
  for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
    let carry = remainder >> 31u;
    remainder = (remainder << 1u) | ((difference.x >> u32(bit)) & 1u);
    if (carry != 0u || remainder >= width) {
      remainder = remainder - width;
      quotientLow = quotientLow | (1u << u32(bit));
    }
  }
  return quotientHigh != 0u || quotientLow >= BUCKET_COUNT;
}

fn isRowBelowRange(row: u32, origin: vec2<u32>) -> bool {
  return timeWordsCompare(getRowTime(row), origin) < 0i;
}

// Only called for rows at or after the lower bound, which are not before the origin.
fn isRowAboveRange(row: u32, origin: vec2<u32>, width: u32) -> bool {
  return isQuotientAtLeastBucketCount(timeWordsSubtract(getRowTime(row), origin), width);
}`:`${p}
override BUCKET_COUNT: u32;
override ROW_COUNT: u32;

fn isRowBelowRange(row: u32, origin: f32) -> bool {
  return rowTimes[rowTimesOffset + row] - origin < 0.0;
}

fn isRowAboveRange(row: u32, origin: f32, limit: f32) -> bool {
  return !(rowTimes[rowTimesOffset + row] - origin < limit);
}`,a=n?`let origin = vec2<u32>(params[paramsOffset], params[paramsOffset + 1u]);
  let width = params[paramsOffset + 2u];
  let isValid = width != 0u;`:`let origin = params[paramsOffset];
  let width = params[paramsOffset + 1u];
  let isValid = !isNanBits(origin) && !isNanBits(width) && width > 0.0 &&
    (bitcast<u32>(width) & 0x7fffffffu) != 0x7f800000u;
  let limit = f32(BUCKET_COUNT) * width;`,o=n?`isRowAboveRange(middle, origin, width)`:`isRowAboveRange(middle, origin, limit)`;return l(e,{id:t.id,operation:`GPUTemporalReduction`,variant:n?`range-search-words`:`range-search`,bindings:[{name:`rowTimes`,view:t.timestamps,type:n?`u32`:`f32`,access:`read`},{name:`params`,view:t.parameters,type:n?`u32`:`f32`,access:`read`},{name:`rowSlots`,view:t.rowSlots,type:`u32`,access:`read_write`}],invocationCount:1,workgroupSize:32,declarations:i,constants:{BUCKET_COUNT:t.bucketCount,ROW_COUNT:r},body:`
  ${a}
  var firstRow = 0u;
  var endRow = 0u;
  if (isValid) {
    // Lower bound: the first row that is not below the range.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (isRowBelowRange(middle, origin)) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    firstRow = low;
    // Upper bound: the first row above the range, searched from the lower bound.
    high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (${o}) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    endRow = low;
  }
  rowSlots[rowSlotsOffset] = firstRow;
  rowSlots[rowSlotsOffset + 1u] = endRow - firstRow;`})}function g(e,t,n=new Float32Array(2)){if(n.length<2)throw Error(`Temporal reduction parameter target must hold 2 elements`);return n[0]=e,n[1]=t,n}var _=`GPUTemporalReduction`,v=256,y=`0xffffffffu`,b=256,x=4294967294,S=2**20,C=class{id;props;slotCount;constructor(e){this.id=e.id??`temporal-reduction`,this.props=e;let i=this.id,{cellCount:o,bucketCount:s,output:c}=e;if(t(i,`cellCount`,o),t(i,`bucketCount`,s),this.slotCount=o*s,this.slotCount>x)throw Error(`${i} cellCount * bucketCount must not exceed ${x}`);let l=e.values.length;if(l<1)throw Error(`${i} needs at least one row`);let u=e.timestamps.format===`uint32x2`;a(e.cellIds,[`uint32`],`${i} cellIds`),a(e.timestamps,[u?`uint32x2`:`float32`],`${i} timestamps`),a(e.values,[`float32`],`${i} values`),e.mask&&a(e.mask,[`uint32`],`${i} mask`);for(let[t,n]of[[`cellIds`,e.cellIds],[`timestamps`,e.timestamps],[`mask`,e.mask]])if(n&&n.length!==l)throw Error(`${i} ${t} length must equal values length`);if(u){if(a(e.parameters,[`uint32`],`${i} parameters`),e.parameters.length<4)throw Error(`${i} parameters must hold 4 uint32 values`)}else{if(a(e.parameters,[`float32`],`${i} parameters`),e.parameters.length<2)throw Error(`${i} parameters must hold 2 float32 values`);if(s>S)throw Error(`${i} float32 times support at most ${S} buckets`)}a(c.counts,[`uint32`],`${i} output.counts`);for(let e of[`minimums`,`maximums`,`first`,`last`])a(c[e],[`float32`],`${i} output.${e}`);for(let e of[`counts`,`minimums`,`maximums`,`first`,`last`])if(c[e].length<this.slotCount)throw Error(`${i} output.${e} must hold cellCount * bucketCount rows`);n(i,c.occupiedSlots),r(i,[c.counts,c.minimums,c.maximums,c.first,c.last,c.occupiedSlots.ids,c.occupiedSlots.status.count,c.occupiedSlots.status.overflow,c.occupiedSlots.status.requiredCount],[e.cellIds,e.timestamps,e.values,e.mask,e.parameters])}getCommandNodes(t){let{props:n,id:r,slotCount:a}=this,{output:u,cellCount:d,bucketCount:f}=n;e(r,t,[n.cellIds,n.timestamps,n.values,n.mask,n.parameters,u.counts,u.minimums,u.maximums,u.first,u.last,u.occupiedSlots.ids,u.occupiedSlots.status.count,u.occupiedSlots.status.overflow,u.occupiedSlots.status.requiredCount]);let p=n.values.length,g=n.timestamps.format===`uint32x2`,x=(e,n)=>i(t,`${r}-${e}`,`uint32`,n),S=n.sorted===!0,C=S?`2u + `:``,D=x(`row-slots`,p+(S?2:0)),O=x(`row-time-keys`,p),k=S?x(`row-value-keys`,p):void 0,A=x(`min-keys`,a),j=x(`max-keys`,a),M=x(`first-time-keys`,a),N=x(`last-time-keys`,a),P=x(`first-rows`,a),F=x(`last-rows`,a),I=x(`occupied`,a),{counts:L}=u,R=k?`rowValueKeys[rowValueKeysOffset + index]`:`getOrderedKey(values[valuesOffset + index])`,z=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),B=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),V=[],H;S&&(V.push(h(t,{id:`${r}-range-search`,timestamps:n.timestamps,parameters:n.parameters,rowSlots:D,bucketCount:f})),H=s(t,{id:`${r}-range-dispatch`,operation:_,count:D,countRow:1,maximumItemCount:p,workgroupSize:b}),V.push(H.updateNode)),V.push(l(t,{id:`${r}-init`,operation:_,variant:`init`,bindings:[B(`counts`,L),B(`minKeys`,A),B(`maxKeys`,j),B(`firstTimeKeys`,M),B(`lastTimeKeys`,N),B(`firstRows`,P),B(`lastRows`,F)],invocationCount:a,body:`counts[countsOffset + index] = 0u;
  minKeys[minKeysOffset + index] = 0xffffffffu;
  maxKeys[maxKeysOffset + index] = 0u;
  firstTimeKeys[firstTimeKeysOffset + index] = 0xffffffffu;
  lastTimeKeys[lastTimeKeysOffset + index] = 0u;
  firstRows[firstRowsOffset + index] = 0xffffffffu;
  lastRows[lastRowsOffset + index] = 0xffffffffu;`})),V.push(l(t,{id:`${r}-classify`,operation:_,variant:g?`classify-words`:`classify`,bindings:[z(`cellIds`,n.cellIds,`u32`),z(`timestamps`,n.timestamps,g?`u32`:`f32`),z(`values`,n.values,`f32`),...n.mask?[z(`rowMask`,n.mask,`u32`)]:[],z(`params`,n.parameters,g?`u32`:`f32`),B(`rowSlots`,D),B(`rowTimeKeys`,O),...k?[B(`rowValueKeys`,k)]:[]],invocationCount:p,...S?{workgroupSize:b,condition:H.condition}:{},declarations:`const CELL_COUNT: u32 = ${d}u;
const BUCKET_COUNT: u32 = ${f}u;
${w}
${g?m:``}`,body:`${S?`let rangeFirstRow = rowSlots[rowSlotsOffset];
  let rangeRowCount = rowSlots[rowSlotsOffset + 1u];
  rowSlots[rowSlotsOffset + 2u + index] = ${y};
  if (index >= rangeRowCount) {
    return;
  }
  let row = rangeFirstRow + index;`:`rowSlots[rowSlotsOffset + index] = ${y};
  let row = index;`}
  let cell = cellIds[cellIdsOffset + row];
  let value = values[valuesOffset + row];
  if (cell >= CELL_COUNT || isNanBits(value)) {
    return;
  }
  ${n.mask?`if (rowMask[rowMaskOffset + row] == 0u) {
    return;
  }`:``}
  var bucket: u32;
  var timeKey: u32;
  ${g?E:T}
  rowSlots[rowSlotsOffset + ${C}index] = cell * BUCKET_COUNT + bucket;
  rowTimeKeys[rowTimeKeysOffset + index] = timeKey;${k?`
  rowValueKeys[rowValueKeysOffset + index] = getOrderedKey(value);`:``}`})),V.push(l(t,{id:`${r}-extremes`,operation:_,variant:`extremes`,bindings:[z(`rowSlots`,D,`u32`),z(`rowTimeKeys`,O,`u32`),k?z(`rowValueKeys`,k,`u32`):z(`values`,n.values,`f32`),B(`counts`,L,`atomic<u32>`),B(`minKeys`,A,`atomic<u32>`),B(`maxKeys`,j,`atomic<u32>`),B(`firstTimeKeys`,M,`atomic<u32>`),B(`lastTimeKeys`,N,`atomic<u32>`)],invocationCount:p,...S?{condition:H.condition}:{},...a<=v?{guardIndex:!1,declarations:`${w}
const SLOT_COUNT: u32 = ${a}u;
var<workgroup> localCounts: array<atomic<u32>, ${a}>;
var<workgroup> localMinimums: array<atomic<u32>, ${a}>;
var<workgroup> localMaximums: array<atomic<u32>, ${a}>;
var<workgroup> localFirstTimes: array<atomic<u32>, ${a}>;
var<workgroup> localLastTimes: array<atomic<u32>, ${a}>;`,body:`for (var slot = localInvocationIndex; slot < SLOT_COUNT; slot += 256u) {
    atomicStore(&localCounts[slot], 0u);
    atomicStore(&localMinimums[slot], 0xffffffffu);
    atomicStore(&localMaximums[slot], 0u);
    atomicStore(&localFirstTimes[slot], 0xffffffffu);
    atomicStore(&localLastTimes[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let slot = rowSlots[rowSlotsOffset + ${C}index];
    if (slot != ${y}) {
      let valueKey = ${R};
      let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
      atomicAdd(&localCounts[slot], 1u);
      atomicMin(&localMinimums[slot], valueKey);
      atomicMax(&localMaximums[slot], valueKey);
      atomicMin(&localFirstTimes[slot], timeKey);
      atomicMax(&localLastTimes[slot], timeKey);
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < SLOT_COUNT; slot += 256u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&counts[countsOffset + slot], count);
      atomicMin(&minKeys[minKeysOffset + slot], atomicLoad(&localMinimums[slot]));
      atomicMax(&maxKeys[maxKeysOffset + slot], atomicLoad(&localMaximums[slot]));
      atomicMin(&firstTimeKeys[firstTimeKeysOffset + slot], atomicLoad(&localFirstTimes[slot]));
      atomicMax(&lastTimeKeys[lastTimeKeysOffset + slot], atomicLoad(&localLastTimes[slot]));
    }
  }`}:{declarations:w,...S?{workgroupSize:b}:{},body:`let slot = rowSlots[rowSlotsOffset + ${C}index];
  if (slot == ${y}) {
    return;
  }
  let valueKey = ${R};
  let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
  atomicAdd(&counts[countsOffset + slot], 1u);
  atomicMin(&minKeys[minKeysOffset + slot], valueKey);
  atomicMax(&maxKeys[maxKeysOffset + slot], valueKey);
  atomicMin(&firstTimeKeys[firstTimeKeysOffset + slot], timeKey);
  atomicMax(&lastTimeKeys[lastTimeKeysOffset + slot], timeKey);`}})),V.push(l(t,{id:`${r}-tie-rows`,operation:_,variant:`tie-rows`,bindings:[z(`rowSlots`,D,`u32`),z(`rowTimeKeys`,O,`u32`),z(`firstTimeKeys`,M,`u32`),z(`lastTimeKeys`,N,`u32`),B(`firstRows`,P,`atomic<u32>`),B(`lastRows`,F,`atomic<u32>`)],invocationCount:p,...S?{workgroupSize:b,condition:H.condition}:{},body:`let slot = rowSlots[rowSlotsOffset + ${C}index];
  if (slot == ${y}) {
    return;
  }
  let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
  ${S?`let row = rowSlots[rowSlotsOffset] + index;`:`let row = index;`}
  if (timeKey == firstTimeKeys[firstTimeKeysOffset + slot]) {
    atomicMin(&firstRows[firstRowsOffset + slot], row);
  }
  if (timeKey == lastTimeKeys[lastTimeKeysOffset + slot]) {
    atomicMin(&lastRows[lastRowsOffset + slot], row);
  }`})),V.push(l(t,{id:`${r}-finish-values`,operation:_,variant:`finish-values`,bindings:[z(`values`,n.values,`f32`),z(`firstRows`,P,`u32`),z(`lastRows`,F,`u32`),z(`counts`,L,`u32`),B(`firstOut`,u.first,`f32`),B(`lastOut`,u.last,`f32`),B(`occupied`,I)],invocationCount:a,declarations:w,body:`let count = counts[countsOffset + index];
  let isOccupied = count != 0u;
  occupied[occupiedOffset + index] = select(0u, 1u, isOccupied);
  let nan = getNaN();
  firstOut[firstOutOffset + index] = select(nan, values[valuesOffset + firstRows[firstRowsOffset + index]], isOccupied);
  lastOut[lastOutOffset + index] = select(nan, values[valuesOffset + lastRows[lastRowsOffset + index]], isOccupied);`})),V.push(l(t,{id:`${r}-finish-extrema`,operation:_,variant:`finish-extrema`,bindings:[z(`minKeys`,A,`u32`),z(`maxKeys`,j,`u32`),z(`counts`,L,`u32`),B(`minOut`,u.minimums,`f32`),B(`maxOut`,u.maximums,`f32`)],invocationCount:a,declarations:w,body:`let isOccupied = counts[countsOffset + index] != 0u;
  let nan = getNaN();
  minOut[minOutOffset + index] = select(nan, decodeOrderedKey(minKeys[minKeysOffset + index]), isOccupied);
  maxOut[maxOutOffset + index] = select(nan, decodeOrderedKey(maxKeys[maxKeysOffset + index]), isOccupied);`}));let U=u.occupiedSlots.ids.length>=a,W=x(`total`,1),G=U?u.occupiedSlots.ids:x(`compact-ids`,a);return V.push(...new o({id:`${r}-compact`,predicates:[{kind:`selection`,mask:I}],output:G,count:W}).getCommandNodes(t)),V.push(c(t,{id:`${r}-publish`,operation:_,requiredCount:W,compactIds:U?void 0:G,output:u.occupiedSlots})),V}},w=`
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${d}

${p}

${u}

${f}
`,T=`let time = timestamps[timestampsOffset + row];
  let origin = params[paramsOffset];
  let width = params[paramsOffset + 1u];
  if (isNanBits(time) || isNanBits(origin) || isNanBits(width)) {
    return;
  }
  if (!(width > 0.0) || (bitcast<u32>(width) & 0x7fffffffu) == 0x7f800000u) {
    return;
  }
  let difference = time - origin;
  var bucketFloat = floor(difference / width);
  if (isNanBits(bucketFloat) || bucketFloat < -1.0 || bucketFloat > f32(BUCKET_COUNT) + 1.0) {
    return;
  }
  // Correct the estimate with correctly rounded products: one step down, then one step up.
  if (difference < bucketFloat * width) {
    bucketFloat = bucketFloat - 1.0;
  } else if (difference >= (bucketFloat + 1.0) * width) {
    bucketFloat = bucketFloat + 1.0;
  }
  if (bucketFloat < 0.0 || bucketFloat >= f32(BUCKET_COUNT)) {
    return;
  }
  bucket = u32(bucketFloat);
  timeKey = getOrderedKey(select(time, 0.0, time == 0.0));`,E=`let origin = vec2<u32>(params[paramsOffset], params[paramsOffset + 1u]);
  let width = params[paramsOffset + 2u];
  if (width == 0u) {
    return;
  }
  let time = vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
  // Exact signed order first: the sign of a wrapped difference is wrong when |time - origin| >=
  // 2^63. Past the check, time - origin is a non-negative value below 2^64 (unsigned words).
  if (timeWordsCompare(time, origin) < 0i) {
    return;
  }
  let difference = timeWordsSubtract(time, origin);
  // Exact 64-bit by 32-bit division: high word first, then the low word bit by bit.
  let quotientHigh = difference.y / width;
  var remainder = difference.y % width;
  var quotientLow = 0u;
  for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
    let carry = remainder >> 31u;
    remainder = (remainder << 1u) | ((difference.x >> u32(bit)) & 1u);
    if (carry != 0u || remainder >= width) {
      remainder = remainder - width;
      quotientLow = quotientLow | (1u << u32(bit));
    }
  }
  if (quotientHigh != 0u || quotientLow >= BUCKET_COUNT) {
    return;
  }
  bucket = quotientLow;
  timeKey = remainder;`;export{g as n,C as t};