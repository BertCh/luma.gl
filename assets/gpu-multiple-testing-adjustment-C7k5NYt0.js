import{_ as e,d as t,h as n,j as r,w as i,y as a}from"./resources-CwvhEOBe.js";import{t as o}from"./gpu-sort-Bl3Dztdb.js";import{n as s,s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{r as l}from"./wgsl-common-BNRVzgd8.js";var u=`GPUMultipleTestingAdjustment`,d=256,f=1065353217,p=30,m=1<<24,h=[`none`,`bonferroni`,`holm`,`benjamini-hochberg`,`benjamini-yekutieli`];function g(e,t){let n=t??`none`;if(!h.includes(n))throw Error(`${e} method must be one of ${h.join(`, `)}, got ${String(t)}`);return n}function _(e,n){let r=n??0;return t(e,`minimumFamilySize`,r,0,m-1),r}var v=class{id;props;constructor(e){let i=e.id??`multiple-testing-adjustment`;this.id=i,this.props=e,g(i,e.method),_(i,e.minimumFamilySize),a(e.pValues,[`float32`],`${i} pValues`),a(e.output.adjustedPValues,[`float32`],`${i} output.adjustedPValues`);let o=e.pValues.length;if(t(i,`pValues length`,o,1,m-1),e.output.adjustedPValues.length!==o)throw Error(`${i} adjustedPValues length must equal the pValues length`);e.output.testedCount&&(r(e.output.testedCount,`${i} output.testedCount`),t(i,`output.testedCount length`,e.output.testedCount.length)),n(i,[e.output.adjustedPValues,e.output.testedCount],[e.pValues])}getCommandNodes(t){let{id:n,props:r}=this;e(n,t,[r.pValues,r.output.adjustedPValues,r.output.testedCount]);let a=g(n,r.method),m=_(n,r.minimumFamilySize),h=r.pValues.length,v=[],y=`${l}
// A p-value is tested when it is in [0, 1]: +0..1.0 as positive bit patterns, or -0.
fn isTestedBits(bits: u32) -> bool {
  return bits <= 0x3f800000u || bits == 0x80000000u;
}`;if(a===`none`&&!r.output.testedCount)return v.push(c(t,{id:`${n}-apply`,operation:u,variant:`none`,bindings:[{name:`pValues`,view:r.pValues,type:`f32`,access:`read`},{name:`adjusted`,view:r.output.adjustedPValues,type:`f32`,access:`read_write`}],invocationCount:h,declarations:y,body:`let p = pValues[pValuesOffset + index];
  adjusted[adjustedOffset + index] = select(getNaN(), p + 0.0, isTestedBits(bitcast<u32>(p)));`})),v;let b=a===`holm`||a===`benjamini-hochberg`||a===`benjamini-yekutieli`,x=i(t,`${n}-counters`,`uint32`,1),S=i(t,`${n}-family`,`uint32`,2),C=i(t,`${n}-family-scale`,`float32`,1),w=b?i(t,`${n}-keys`,`uint32`,h):void 0,T=b?i(t,`${n}-row-ids`,`uint32`,h):void 0,E=[{name:`pValues`,view:r.pValues,type:`f32`,access:`read`},{name:`counters`,view:x,type:`atomic<u32>`,access:`read_write`}];w&&T&&E.push({name:`keys`,view:w,type:`u32`,access:`read_write`},{name:`rowIds`,view:T,type:`u32`,access:`read_write`}),v.push(s(t,{id:`${n}-counters-clear`,operation:u,view:x,type:`u32`,value:`0u`}),c(t,{id:`${n}-count`,operation:u,variant:w?`count-keys`:`count`,bindings:E,invocationCount:h,declarations:`${y}
const UNTESTED_KEY: u32 = ${f}u;`,body:`let bits = bitcast<u32>(pValues[pValuesOffset + index]);
  let tested = isTestedBits(bits);
  if (tested) {
    atomicAdd(&counters[countersOffset], 1u);
  }
  ${w?`keys[keysOffset + index] = select(UNTESTED_KEY, bits & 0x7fffffffu, tested);
  rowIds[rowIdsOffset + index] = index;`:``}`}));let D=[{name:`counters`,view:x,type:`u32`,access:`read`},{name:`family`,view:S,type:`u32`,access:`read_write`},{name:`familyScale`,view:C,type:`f32`,access:`read_write`}];if(r.output.testedCount&&D.push({name:`testedCount`,view:r.output.testedCount,type:`u32`,access:`read_write`}),v.push(c(t,{id:`${n}-family`,operation:u,variant:`family`,bindings:D,invocationCount:1,declarations:`const MINIMUM_FAMILY_SIZE: u32 = ${m}u;
const EULER_MASCHERONI: f32 = 0.5772157;
// Harmonic number c(m) = sum_{i=1..m} 1/i. Exact fixed-order sum (smallest terms first) for small
// m; asymptotic series ln m + gamma + 1/(2m) - 1/(12 m^2) (error < 1/(120 m^4)) beyond.
fn getHarmonicNumber(m: u32) -> f32 {
  if (m <= 64u) {
    var sum = 0.0;
    for (var i = m; i >= 1u; i--) {
      sum += 1.0 / f32(i);
    }
    return sum;
  }
  let x = f32(m);
  return log(x) + EULER_MASCHERONI + 0.5 / x - 1.0 / (12.0 * x * x);
}`,body:`let tested = counters[countersOffset];
  let m = max(tested, MINIMUM_FAMILY_SIZE);
  family[familyOffset] = tested;
  family[familyOffset + 1u] = m;
  familyScale[familyScaleOffset] = ${a===`benjamini-yekutieli`?`getHarmonicNumber(m)`:`1.0`};
  ${r.output.testedCount?`testedCount[testedCountOffset] = m;`:``}`})),a===`none`||a===`bonferroni`)return v.push(c(t,{id:`${n}-apply`,operation:u,variant:a,bindings:[{name:`pValues`,view:r.pValues,type:`f32`,access:`read`},{name:`family`,view:S,type:`u32`,access:`read`},{name:`adjusted`,view:r.output.adjustedPValues,type:`f32`,access:`read_write`}],invocationCount:h,declarations:y,body:`let p = pValues[pValuesOffset + index];
  let m = f32(family[familyOffset + 1u]);
  let value = ${a===`none`?`p + 0.0`:`min(p * m, 1.0)`};
  adjusted[adjustedOffset + index] = select(getNaN(), value, isTestedBits(bitcast<u32>(p)));`})),v;let O=i(t,`${n}-sorted-keys`,`uint32`,h),k=i(t,`${n}-sorted-rows`,`uint32`,h),A=Math.ceil(h/d),j=Math.ceil(A/d),M=i(t,`${n}-scanned`,`float32`,h),N=i(t,`${n}-block-totals`,`float32`,A),P=a===`holm`,F=!P,I=`const ROWS: u32 = ${h}u;
const BLOCK_COUNT: u32 = ${A}u;
const BLOCK_CHUNK: u32 = ${j}u;
const REVERSE: bool = ${F};
const IDENTITY: f32 = ${P?`0.0`:`3.0e38`};
var<workgroup> scanValues: array<f32, ${d}>;

fn combine(left: f32, right: f32) -> f32 {
  return ${P?`max(left, right)`:`min(left, right)`};
}

// Raw step value of sorted rank r (rank k = r + 1) given tested count, family size m and c(m);
// IDENTITY beyond the tested rows.
fn getStepValue(r: u32, key: u32, tested: u32, m: u32, scale: f32) -> f32 {
  if (r >= tested) {
    return IDENTITY;
  }
  let p = bitcast<f32>(key);
  ${P?`return f32(m - r) * p;`:`return ((p * f32(m)) / f32(r + 1u)) * scale;`}
}

// Hillis-Steele inclusive scan of scanValues across the workgroup. Every invocation must call it.
fn scanWorkgroup(localIndex: u32) {
  workgroupBarrier();
  for (var offset = 1u; offset < ${d}u; offset = offset * 2u) {
    var other = IDENTITY;
    if (localIndex >= offset) {
      other = scanValues[localIndex - offset];
    }
    let combined = combine(scanValues[localIndex], other);
    workgroupBarrier();
    scanValues[localIndex] = combined;
    workgroupBarrier();
  }
}`;return v.push(...new o({id:`${n}-sort`,keys:w,values:T,outputKeys:O,outputValues:k,keyBits:p}).getCommandNodes(t),c(t,{id:`${n}-block-scan`,operation:u,variant:`block-scan`,bindings:[{name:`sortedKeys`,view:O,type:`u32`,access:`read`},{name:`family`,view:S,type:`u32`,access:`read`},{name:`familyScale`,view:C,type:`f32`,access:`read`},{name:`scanned`,view:M,type:`f32`,access:`read_write`},{name:`blockTotals`,view:N,type:`f32`,access:`read_write`}],invocationCount:h,workgroupSize:d,guardIndex:!1,declarations:I,body:`// Scan position j; the sorted rank it reads is r.
  let j = index;
  var value = IDENTITY;
  if (j < ROWS) {
    let r = select(j, ROWS - 1u - j, REVERSE);
    value = getStepValue(r, sortedKeys[sortedKeysOffset + r], family[familyOffset], family[familyOffset + 1u], familyScale[familyScaleOffset]);
  }
  scanValues[localInvocationIndex] = value;
  scanWorkgroup(localInvocationIndex);
  if (j < ROWS) {
    scanned[scannedOffset + j] = scanValues[localInvocationIndex];
  }
  if (localInvocationIndex == ${d-1}u) {
    blockTotals[blockTotalsOffset + workgroupIndex] = scanValues[localInvocationIndex];
  }`})),A>1&&v.push(c(t,{id:`${n}-total-scan`,operation:u,variant:`total-scan`,bindings:[{name:`blockTotals`,view:N,type:`f32`,access:`read_write`}],invocationCount:d,workgroupSize:d,guardIndex:!1,declarations:I,body:`let begin = min(localInvocationIndex * BLOCK_CHUNK, BLOCK_COUNT);
  let end = min(begin + BLOCK_CHUNK, BLOCK_COUNT);
  var running = IDENTITY;
  for (var block = begin; block < end; block++) {
    running = combine(running, blockTotals[blockTotalsOffset + block]);
    blockTotals[blockTotalsOffset + block] = running;
  }
  scanValues[localInvocationIndex] = running;
  scanWorkgroup(localInvocationIndex);
  var prefix = IDENTITY;
  if (localInvocationIndex > 0u) {
    prefix = scanValues[localInvocationIndex - 1u];
  }
  for (var block = begin; block < end; block++) {
    blockTotals[blockTotalsOffset + block] = combine(blockTotals[blockTotalsOffset + block], prefix);
  }`})),v.push(c(t,{id:`${n}-scatter`,operation:u,variant:`scatter`,bindings:[{name:`sortedRows`,view:k,type:`u32`,access:`read`},{name:`family`,view:S,type:`u32`,access:`read`},{name:`scanned`,view:M,type:`f32`,access:`read`},{name:`blockTotals`,view:N,type:`f32`,access:`read`},{name:`adjusted`,view:r.output.adjustedPValues,type:`f32`,access:`read_write`}],invocationCount:h,declarations:`${l}
const ROWS: u32 = ${h}u;
const REVERSE: bool = ${F};
fn combine(left: f32, right: f32) -> f32 {
  return ${P?`max(left, right)`:`min(left, right)`};
}`,body:`let j = select(index, ROWS - 1u - index, REVERSE);
  var value = scanned[scannedOffset + j];
  let block = j / ${d}u;
  if (block > 0u) {
    value = combine(value, blockTotals[blockTotalsOffset + block - 1u]);
  }
  adjusted[adjustedOffset + sortedRows[sortedRowsOffset + index]] =
    select(getNaN(), min(value, 1.0), index < family[familyOffset]);`})),v}};export{m as n,g as r,v as t};