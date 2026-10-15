import{mt as e}from"./src-CNCKFW2b.js";import{_ as t,d as n,h as r,w as i,x as a,y as o}from"./resources-CwvhEOBe.js";import{t as s}from"./gpu-sort-Bl3Dztdb.js";import{c,l,s as u}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as d}from"./column-classification-shared-Cj6aSTEo.js";import{t as f}from"./column-quantiles-nodes-DnezST_r.js";var p=256*8,m={method:0,classCount:1,layerCursor:2,refinementCursor:3,refinementMoved:4,refinementDone:5},h={minimumKey:0,maximumKey:1,validCount:2,finiteCount:3},g={done:0,breakCount:1,thresholdKey:2,previousCount:3},_=`
struct Moments {
  count: f32,
  mean: f32,
  m2: f32,
  minimum: f32,
  maximum: f32,
}

fn mergeMoments(left: Moments, right: Moments) -> Moments {
  if (right.count == 0.0) {
    return left;
  }
  if (left.count == 0.0) {
    return right;
  }
  let count = left.count + right.count;
  let delta = right.mean - left.mean;
  let mean = left.mean + delta * (right.count / count);
  let m2 = left.m2 + right.m2 + delta * delta * (left.count * right.count / count);
  return Moments(count, mean, m2, min(left.minimum, right.minimum), max(left.maximum, right.maximum));
}

fn emptyMoments() -> Moments {
  return Moments(0.0, 0.0, 0.0, 0.0, 0.0);
}

var<workgroup> sharedMoments: array<Moments, 256>;
var<workgroup> gateShared: u32;

fn reduceSharedMoments(local: u32) {
  for (var stride = ${256/2}u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedMoments[local] = mergeMoments(sharedMoments[local], sharedMoments[local + stride]);
    }
  }
}
`,v=3;function y(e,t,n){return{condition:{id:`${e}-gate`,source:`gpu`,mode:`indirect`,buffer:t.buffer,byteOffset:t.byteOffset+n*v*4},extraResources:[{buffer:t,usage:`indirect`}]}}function b(e,t){let n=a(t.operation,Math.max(t.invocationCount,1),t.workgroupSize,e.device.limits.maxComputeWorkgroupsPerDimension),r=t.slot*v;return`dispatch[dispatchOffset + ${r}u] = select(0u, ${n.x}u, ${t.activeExpression});
  dispatch[dispatchOffset + ${r+1}u] = ${n.y}u;
  dispatch[dispatchOffset + ${r+2}u] = ${n.z}u;`}function x(e,t){let{id:n,operation:r,values:a,mask:o,state:s,partials:c,moments:l,slot:f,gate:h,indirectGates:v}=t,y=a.length,b=Math.ceil(y/p),x=h.kind===`head-tail`?h.headTail:void 0,S=h.kind===`head-tail`?`state[stateOffset + ${m.method}u] == ${h.methodCode}u && headTail[headTailOffset + ${g.done}u] == 0u`:`state[stateOffset + ${m.method}u] == ${h.methodCode}u`,C=i(e,`${n}-origin`,`float32`,1),w=[{name:`values`,view:a,type:`f32`,access:`read`},...o?[{name:`rowMask`,view:o,type:`u32`,access:`read`}]:[],{name:`state`,view:s,type:`u32`,access:`read`},...x?[{name:`headTail`,view:x,type:`u32`,access:`read`}]:[]],T=u(e,{id:`${n}-origin`,operation:r,variant:`moments-origin`,bindings:[...w,{name:`origin`,view:C,type:`f32`,access:`read_write`}],invocationCount:256,workgroupSize:256,guardIndex:!1,...v?.merge,declarations:`const ROW_COUNT: u32 = ${y}u;
${d}
var<workgroup> firstRows: array<u32, 256>;
var<workgroup> gateShared: u32;`,body:`let local = localInvocationIndex;
  if (local == 0u) {
    gateShared = select(0u, 1u, ${S});
  }
  if (workgroupUniformLoad(&gateShared) == 0u) {
    return;
  }
  ${x?`let thresholdKey = headTail[headTailOffset + ${g.thresholdKey}u];`:``}
  var firstRow = ROW_COUNT;
  for (var row = local; row < ROW_COUNT; row += 256u) {
    let value = values[valuesOffset + row];
    var include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
    ${o?`include = include && rowMask[rowMaskOffset + row] != 0u;`:``}
    ${x?`include = include && getOrderedKey(value) > thresholdKey;`:``}
    if (include) {
      firstRow = row;
      break;
    }
  }
  firstRows[local] = firstRow;
  for (var stride = ${256/2}u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (local < stride) {
      firstRows[local] = min(firstRows[local], firstRows[local + stride]);
    }
  }
  if (local == 0u) {
    var pivot = 0.0;
    if (firstRows[0] < ROW_COUNT) {
      pivot = values[valuesOffset + firstRows[0]];
    }
    origin[originOffset] = pivot;
  }`}),E=[{name:`values`,view:a,type:`f32`,access:`read`},...o?[{name:`rowMask`,view:o,type:`u32`,access:`read`}]:[],{name:`state`,view:s,type:`u32`,access:`read`},...x?[{name:`headTail`,view:x,type:`u32`,access:`read`}]:[],{name:`origin`,view:C,type:`f32`,access:`read`},{name:`partials`,view:c,type:`f32`,access:`read_write`}],D=u(e,{id:`${n}-tile`,operation:r,variant:`moments-tile`,bindings:E,invocationCount:b*256,workgroupSize:256,guardIndex:!1,...v?.tile,declarations:`const ROW_COUNT: u32 = ${y}u;
const TILE_SIZE: u32 = ${p}u;
const ROWS_PER_INVOCATION: u32 = 8u;
${d}
${_}
var<workgroup> centeredValueBits: array<u32, ${p}>;`,body:`let local = localInvocationIndex;
  if (local == 0u) {
    gateShared = select(0u, 1u, ${S});
  }
  if (workgroupUniformLoad(&gateShared) == 0u) {
    return;
  }
  let tile = workgroupIndex;
  ${x?`let thresholdKey = headTail[headTailOffset + ${g.thresholdKey}u];`:``}
  let pivot = origin[originOffset];
  for (var step = 0u; step < ROWS_PER_INVOCATION; step = step + 1u) {
    // Uniform trip count: rows past the end are skipped by a predicate, never by a break, so
    // the barriers after the loop stay in uniform control flow.
    let row = min(tile * TILE_SIZE + step * 256u + local, ROW_COUNT - 1u);
    let inRange = tile * TILE_SIZE + step * 256u + local < ROW_COUNT;
    let value = values[valuesOffset + row];
    var include = inRange && (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
    ${o?`include = include && rowMask[rowMaskOffset + row] != 0u;`:``}
    ${x?`include = include && getOrderedKey(value) > thresholdKey;`:``}
    centeredValueBits[step * 256u + local] = select(0x7fc00000u, bitcast<u32>(value - pivot), include);
  }
  workgroupBarrier();
  var accumulated = emptyMoments();
  for (var step = 0u; step < ROWS_PER_INVOCATION; step += 1u) {
    // Read another invocation's materialized subtraction, so the optimizer cannot reassociate
    // (value - pivot) - mean into value - (pivot + mean) at the original large offset.
    let tileRow = step * 256u + (local ^ ${256/2}u);
    let shifted = bitcast<f32>(centeredValueBits[tileRow]);
    // Only the NaN sentinel excludes a row; an overflowing finite difference still counts.
    if ((bitcast<u32>(shifted) & 0x7fffffffu) <= 0x7f800000u) {
      let value = values[valuesOffset + tile * TILE_SIZE + tileRow];
      accumulated = mergeMoments(accumulated, Moments(1.0, shifted, 0.0, value, value));
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = partialsOffset + tile * 6u;
    partials[base] = record.count;
    partials[base + 1u] = record.mean;
    partials[base + 2u] = record.m2;
    partials[base + 3u] = record.minimum;
    partials[base + 4u] = record.maximum;
    partials[base + 5u] = select(0.0, pivot, record.count > 0.0);
  }`}),O=[{name:`state`,view:s,type:`u32`,access:`read`},...x?[{name:`headTail`,view:x,type:`u32`,access:`read`}]:[],{name:`origin`,view:C,type:`f32`,access:`read`},{name:`partials`,view:c,type:`f32`,access:`read`},{name:`moments`,view:l,type:`f32`,access:`read_write`}];return[T,D,u(e,{id:`${n}-merge`,operation:r,variant:`moments-merge`,bindings:O,invocationCount:256,workgroupSize:256,guardIndex:!1,...v?.merge,declarations:`const TILE_COUNT: u32 = ${b}u;
const STEP_COUNT: u32 = ${Math.ceil(b/256)}u;
${_}`,body:`let local = localInvocationIndex;
  if (local == 0u) {
    gateShared = select(0u, 1u, ${S});
  }
  if (workgroupUniformLoad(&gateShared) == 0u) {
    return;
  }
  var accumulated = emptyMoments();
  for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
    let tile = step * 256u + local;
    if (tile < TILE_COUNT) {
      let base = partialsOffset + tile * 6u;
      accumulated = mergeMoments(
        accumulated,
        Moments(partials[base], partials[base + 1u], partials[base + 2u], partials[base + 3u], partials[base + 4u])
      );
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = momentsOffset + ${f*6}u;
    moments[base] = record.count;
    moments[base + 1u] = select(0.0, origin[originOffset] + record.mean, record.count > 0.0);
    moments[base + 2u] = record.m2;
    moments[base + 3u] = record.minimum;
    moments[base + 4u] = record.maximum;
    moments[base + 5u] = 0.0;
  }`})]}function S(e){return Math.ceil(e/p)}var C=`
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
`,w=1024,T=`
fn addWide(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x + right.x;
  let carry = select(0u, 1u, low < left.x);
  return vec2<u32>(low, left.y + right.y + carry);
}

fn subtractWide(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x - right.x;
  let borrow = select(0u, 1u, left.x < right.x);
  return vec2<u32>(low, left.y - right.y - borrow);
}

// Full 32 x 32 -> 64 bit product from 16-bit limbs; no partial product exceeds 2^32.
fn multiplyWide(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowProduct = leftLow * rightLow;
  let crossA = leftLow * rightHigh;
  let crossB = leftHigh * rightLow;
  let highProduct = leftHigh * rightHigh;
  let cross = crossA + crossB;
  let crossCarry = select(0u, 1u, cross < crossA);
  let low = lowProduct + ((cross & 0xffffu) << 16u);
  let lowCarry = select(0u, 1u, low < lowProduct);
  return vec2<u32>(low, highProduct + (cross >> 16u) + (crossCarry << 16u) + lowCarry);
}

// Product modulo 2^64.
fn multiplyWideLow(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let product = multiplyWide(left.x, right.x);
  return vec2<u32>(product.x, product.y + left.x * right.y + left.y * right.x);
}

fn wideToUnsignedFloat(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}

// A small negative number has a huge low word in two's complement, so convert the magnitude.
fn wideToSignedFloat(value: vec2<u32>) -> f32 {
  if (bitcast<i32>(value.y) < 0) {
    return -wideToUnsignedFloat(subtractWide(vec2<u32>(0u, 0u), value));
  }
  return wideToUnsignedFloat(value);
}

fn getClassCost(count: u32, sum: vec2<u32>, squares: vec2<u32>) -> f32 {
  if (count == 0u) {
    return 0.0;
  }
  let countFloat = f32(count);
  let center = i32(round(wideToSignedFloat(sum) / countFloat));
  let centerBits = bitcast<u32>(center);
  let centerWide = vec2<u32>(centerBits, select(0u, 0xffffffffu, (center < 0)));
  let centeredSum = bitcast<i32>(sum.x - centerBits * count);
  let crossTerm = multiplyWideLow(centerWide, sum);
  let centerSquared = multiplyWideLow(centerWide, centerWide);
  let centeredSquares = addWide(
    subtractWide(squares, addWide(crossTerm, crossTerm)),
    multiplyWideLow(centerSquared, vec2<u32>(count, 0u))
  );
  let centeredSumFloat = f32(centeredSum);
  let correction = centeredSumFloat * centeredSumFloat / countFloat;
  return max(wideToUnsignedFloat(centeredSquares) - correction, 0.0);
}
`,E={"equal-interval":0,quantile:1,"standard-deviation":2,"head-tail":3,"box-plot":4,"maximum-breaks":5,"natural-breaks":6,custom:7},D=Object.keys(E);function O(e){return 8+e+1}function k(e,t,r=new Float32Array(O(t))){let i=O(t);if(r.length<i)throw Error(`GPUClassBreaks parameter target must hold ${i} elements`);let{method:a}=e,o=E[a];if(o===void 0)throw Error(`GPUClassBreaks unknown method ${String(a)}`);let s=e.classCount??5;if(a===`box-plot`)s=6;else if(a===`custom`){let t=e.customEdges?.length??0;if(t<2)throw Error(`GPUClassBreaks custom needs at least two edges`);s=t-1}n(`GPUClassBreaks`,`classCount`,s,1,t);for(let[t,n]of[[`standardDeviationInterval`,e.standardDeviationInterval],[`headTailRatio`,e.headTailRatio],[`boxPlotHinge`,e.boxPlotHinge]])if(n!==void 0&&!Number.isFinite(n))throw Error(`GPUClassBreaks ${t} must be finite`);if(a===`custom`&&Array.from(e.customEdges).some(Number.isNaN))throw Error(`GPUClassBreaks custom edges must not be NaN`);if(r.fill(NaN,0,i),r[0]=o,r[1]=s,r[2]=e.standardDeviationInterval??1,r[3]=e.headTailRatio??.4,r[4]=e.boxPlotHinge??1.5,r[5]=0,r[6]=0,r[7]=0,a===`quantile`)for(let e=1;e<s;e++)r[8+e-1]=e/s;else if(a===`box-plot`)r[8]=.25,r[9]=.5,r[10]=.75;else if(a===`custom`){let t=e.customEdges;for(let e=0;e<t.length;e++)r[8+e]=t[e]}return r}var A=E,j=`3.0e38`,M=`0xffffffffu`,N=99,P=16,F=256;function I(e,t,n,r,i){return`{
    let previousLow = atomicAdd(&${e}[${t}], ${r});
    let carry = select(0u, 1u, previousLow + ${r} < previousLow);
    let highPart = ${i} + carry;
    if (highPart != 0u) {
      atomicAdd(&${e}[${n}], highPart);
    }
  }`}function L(t,n){let{operation:r,values:a,mask:o,maximumClassCount:s,binCount:c}=n,f=`${n.id}-natural-breaks`,p=a.length,g=(e,n)=>i(t,`${f}-${e}`,`uint32`,Math.max(n,1)),_=(e,n)=>i(t,`${f}-${e}`,`float32`,Math.max(n,1)),v=g(`histogram`,5*c),x=g(`prefix`,5*(c+1)),S=_(`totals`,s*c),w=g(`starts`,s*c),E={name:`state`,view:n.state,type:`u32`,access:`read`},D=o?[{name:`rowMask`,view:o,type:`u32`,access:`read`}]:[],O=o?`rowMask[rowMaskOffset + index] != 0u`:`true`,k=`state[stateOffset + ${m.method}u] != ${A[`natural-breaks`]}u`,M=`${d}
const BIN_COUNT: u32 = ${c}u;
const INVERSE_BIN_COUNT: f32 = ${l(1/c)};
${C}`,N=`let minimum = decodeOrderedKey(extremes[extremesOffset + ${h.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${h.maximumKey}u]);
  let width = (maximum - minimum) * INVERSE_BIN_COUNT;`,P=[],L=l=>{let d=`${f}${l}`,g=[];g.push(u(t,{id:`${d}-clear`,operation:r,variant:`natural-breaks-clear`,bindings:[E,{name:`histogram`,view:v,type:`u32`,access:`read_write`}],invocationCount:5*c,body:`if (${k}) {
      return;
    }
    histogram[histogramOffset + index] = 0u;`})),g.push(u(t,{id:`${d}-histogram`,operation:r,variant:`natural-breaks-histogram`,bindings:[{name:`values`,view:a,type:`f32`,access:`read`},...D,E,{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`histogram`,view:v,type:`atomic<u32>`,access:`read_write`}],invocationCount:p,declarations:`${M}
  ${T}`,body:`if (${k}) {
      return;
    }
    let value = values[valuesOffset + index];
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
      return;
    }
    ${o?`if (rowMask[rowMaskOffset + index] == 0u) {
    return;
  }`:``}
    // Rows outside [minimum, maximum] belong to other classes when the stage runs over one.
    let key = getOrderedKey(value);
    if (key < extremes[extremesOffset + ${h.minimumKey}u] || key > extremes[extremesOffset + ${h.maximumKey}u]) {
      return;
    }
    ${N}
    var bin = 0u;
    var offset = 0u;
    if (width > 0.0) {
      bin = getBinIndex(value, minimum, width);
      // Position inside the bin in 1/256 bins.
      let remainder = (value - minimum) - f32(bin) * width;
      offset = u32(clamp(floor(remainder * (256.0 / width)), 0.0, 255.0));
    }
    let position = bin * 256u + offset;
    let square = multiplyWide(position, position);
    atomicAdd(&histogram[histogramOffset + bin], 1u);
    ${I(`histogram`,`histogramOffset + ${c}u + bin`,`histogramOffset + ${2*c}u + bin`,`position`,`0u`)}
    ${I(`histogram`,`histogramOffset + ${3*c}u + bin`,`histogramOffset + ${4*c}u + bin`,`square.x`,`square.y`)}`})),g.push(u(t,{id:`${d}-prefix`,operation:r,variant:`natural-breaks-prefix`,bindings:[E,{name:`histogram`,view:v,type:`u32`,access:`read`},{name:`prefix`,view:x,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const BIN_COUNT: u32 = ${c}u;
  const PREFIX_STRIDE: u32 = ${c+1}u;
  ${T}`,body:`if (${k}) {
      return;
    }
    var count = 0u;
    var sum = vec2<u32>(0u, 0u);
    var squares = vec2<u32>(0u, 0u);
    for (var bin = 0u; bin < BIN_COUNT; bin = bin + 1u) {
      prefix[prefixOffset + bin] = count;
      prefix[prefixOffset + PREFIX_STRIDE + bin] = sum.x;
      prefix[prefixOffset + 2u * PREFIX_STRIDE + bin] = sum.y;
      prefix[prefixOffset + 3u * PREFIX_STRIDE + bin] = squares.x;
      prefix[prefixOffset + 4u * PREFIX_STRIDE + bin] = squares.y;
      count = count + histogram[histogramOffset + bin];
      sum = addWide(sum, vec2<u32>(histogram[histogramOffset + BIN_COUNT + bin], histogram[histogramOffset + 2u * BIN_COUNT + bin]));
      squares = addWide(squares, vec2<u32>(histogram[histogramOffset + 3u * BIN_COUNT + bin], histogram[histogramOffset + 4u * BIN_COUNT + bin]));
    }
    prefix[prefixOffset + BIN_COUNT] = count;
    prefix[prefixOffset + PREFIX_STRIDE + BIN_COUNT] = sum.x;
    prefix[prefixOffset + 2u * PREFIX_STRIDE + BIN_COUNT] = sum.y;
    prefix[prefixOffset + 3u * PREFIX_STRIDE + BIN_COUNT] = squares.x;
    prefix[prefixOffset + 4u * PREFIX_STRIDE + BIN_COUNT] = squares.y;`}));let _=`const BIN_COUNT: u32 = ${c}u;
  const PREFIX_STRIDE: u32 = ${c+1}u;
  ${T}

  fn getRangeCost(first: u32, last: u32) -> f32 {
    let end = last + 1u;
    let count = prefix[prefixOffset + end] - prefix[prefixOffset + first];
    let sum = subtractWide(
      vec2<u32>(prefix[prefixOffset + PREFIX_STRIDE + end], prefix[prefixOffset + 2u * PREFIX_STRIDE + end]),
      vec2<u32>(prefix[prefixOffset + PREFIX_STRIDE + first], prefix[prefixOffset + 2u * PREFIX_STRIDE + first])
    );
    let squares = subtractWide(
      vec2<u32>(prefix[prefixOffset + 3u * PREFIX_STRIDE + end], prefix[prefixOffset + 4u * PREFIX_STRIDE + end]),
      vec2<u32>(prefix[prefixOffset + 3u * PREFIX_STRIDE + first], prefix[prefixOffset + 4u * PREFIX_STRIDE + first])
    );
    return getClassCost(count, sum, squares);
  }`,C={name:`prefix`,view:x,type:`u32`,access:`read`},O={name:`totals`,view:S,type:`f32`,access:`read_write`},P={name:`starts`,view:w,type:`u32`,access:`read_write`};g.push(u(t,{id:`${d}-layer-0`,operation:r,variant:`natural-breaks-layer`,bindings:[E,C,O,P],invocationCount:c,declarations:_,body:`if (${k}) {
      return;
    }
    totals[totalsOffset + index] = getRangeCost(0u, index);
    starts[startsOffset + index] = 0u;`}));let L=i(t,`${d}-layer-dispatch`,`uint32`,3,e.STORAGE|e.INDIRECT),R=`state[stateOffset + ${m.layerCursor}u]`,z=`state[stateOffset + ${m.method}u] == ${A[`natural-breaks`]}u && layer < state[stateOffset + ${m.classCount}u]`;for(let e=1;e<s;e++){let i=`${d}-layer-${e}`;g.push(u(t,{id:`${i}-advance`,operation:r,variant:`natural-breaks-layer-advance`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`dispatch`,view:L,type:`u32`,access:`read_write`}],invocationCount:1,body:`let layer = ${R} + 1u;
    ${R} = layer;
    ${b(t,{operation:r,slot:0,invocationCount:F,workgroupSize:F,activeExpression:`(${z})`})}`}),u(t,{id:i,operation:r,variant:`natural-breaks-layer`,bindings:[E,C,O,P],invocationCount:F,workgroupSize:F,guardIndex:!1,...y(i,L,0),declarations:`${_}
  const WORKGROUP_SIZE: u32 = ${F}u;
  // Best start of the last class for every solved end bin (opt_c). 4 B per bin: 8 KiB at 2048 bins.
  var<workgroup> optimum: array<u32, ${c}>;
  var<workgroup> candidateCost: array<f32, ${F}>;
  var<workgroup> candidateStart: array<u32, ${F}>;
  var<workgroup> sharedLayer: u32;

  // Candidates for the last class of a mid at 1-based position 'position' (end bin 'end'): from the
  // optimum of the solved neighbour on the left to that of the solved neighbour on the right.
  fn getSearchLow(layer: u32, position: u32, stride: u32, end: u32) -> u32 {
    if (position == stride) {
      return layer;
    }
    return max(optimum[end - stride], layer);
  }

  fn getSearchHigh(count: u32, position: u32, stride: u32, end: u32, low: u32) -> u32 {
    var high = BIN_COUNT - 1u;
    if (position + stride <= count) {
      high = optimum[end + stride];
    }
    // Rounding in the costs can break monotonicity by one step; never search an empty range.
    return min(end, max(high, low));
  }

  fn getCandidateCost(layer: u32, start: u32, end: u32) -> f32 {
    return totals[totalsOffset + (layer - 1u) * BIN_COUNT + start - 1u] + getRangeCost(start, end);
  }`,body:`let local = localInvocationIndex;
    if (local == 0u) {
      sharedLayer = ${R};
    }
    let layer = workgroupUniformLoad(&sharedLayer);
    if (layer >= BIN_COUNT) {
      return;
    }
    // Ends layer..BIN_COUNT-1 are positions 1..count of a perfect tree with 2^levels - 1 mids.
    let count = BIN_COUNT - layer;
    var levels = 0u;
    while ((1u << levels) - 1u < count) {
      levels = levels + 1u;
    }
    for (var level = 0u; level < levels; level = level + 1u) {
      let stride = 1u << (levels - 1u - level);
      let nodeCount = 1u << level;
      let teamSize = max(WORKGROUP_SIZE >> level, 1u);
      let teamCount = WORKGROUP_SIZE / teamSize;
      let team = local / teamSize;
      let lane = local % teamSize;
      if (teamSize == 1u) {
        // Deep levels: a thread per node, no cooperation and so no barrier in the loop.
        for (var node = team; node < nodeCount; node = node + teamCount) {
          let position = (2u * node + 1u) * stride;
          if (position <= count) {
            let end = layer + position - 1u;
            let low = getSearchLow(layer, position, stride, end);
            let high = getSearchHigh(count, position, stride, end, low);
            var bestCost = ${j};
            var bestStart = low;
            for (var start = low; start <= high; start = start + 1u) {
              let cost = getCandidateCost(layer, start, end);
              if (cost < bestCost) {
                bestCost = cost;
                bestStart = start;
              }
            }
            optimum[end] = bestStart;
            totals[totalsOffset + layer * BIN_COUNT + end] = bestCost;
            starts[startsOffset + layer * BIN_COUNT + end] = bestStart;
          }
        }
      } else {
        // Shallow levels: a team of lanes per node scans its range together and the lanes reduce
        // to the lowest start of least cost.
        let position = (2u * team + 1u) * stride;
        let valid = position <= count;
        let end = layer + position - 1u;
        var bestCost = ${j};
        var bestStart = 0xffffffffu;
        if (valid) {
          let low = getSearchLow(layer, position, stride, end);
          let high = getSearchHigh(count, position, stride, end, low);
          for (var start = low + lane; start <= high; start = start + teamSize) {
            let cost = getCandidateCost(layer, start, end);
            if (cost < bestCost) {
              bestCost = cost;
              bestStart = start;
            }
          }
        }
        candidateCost[local] = bestCost;
        candidateStart[local] = bestStart;
        for (var reduction = teamSize >> 1u; reduction > 0u; reduction = reduction >> 1u) {
          workgroupBarrier();
          if (lane < reduction) {
            let otherCost = candidateCost[local + reduction];
            let otherStart = candidateStart[local + reduction];
            if (otherCost < candidateCost[local] || (otherCost == candidateCost[local] && otherStart < candidateStart[local])) {
              candidateCost[local] = otherCost;
              candidateStart[local] = otherStart;
            }
          }
        }
        if (valid && lane == 0u) {
          optimum[end] = candidateStart[local];
          totals[totalsOffset + layer * BIN_COUNT + end] = candidateCost[local];
          starts[startsOffset + layer * BIN_COUNT + end] = candidateStart[local];
        }
      }
      // The next level reads the optima of this one.
      workgroupBarrier();
    }`}))}return g};return P.push(...L(``)),P.push(...R(t,n,{prefix:f,u32:g,f32:_,starts:w,getCoarseNodes:L,gate:k,binDeclarations:M,extremesRead:N,maskBinding:D,maskTest:O})),P}function R(t,n,r){let{operation:a,values:o,maximumClassCount:s,binCount:f,edges:p,edgeState:g}=n,{prefix:_,gate:v,binDeclarations:x,extremesRead:S,maskBinding:C,maskTest:E}=r,D=`min(state[stateOffset + ${m.classCount}u], BIN_COUNT)`,O=`const BIN_COUNT: u32 = ${f}u;`,k=o.length,F=w,L=F+2,R=Math.max(s-1,1),z=`state[stateOffset + ${m.method}u]`,B={name:`state`,view:n.state,type:`u32`,access:`read`},V=[],H=r.u32(`extents`,3*s),U=r.f32(`bulk-edges`,s+1),ee=r.u32(`backup`,8),te=r.u32(`break-state`,3*(s+1)),W=r.f32(`plan`,3*(s+1)),ne=r.u32(`cells`,R*L*7),G=i(t,`${_}-refinement-dispatch`,`uint32`,18,e.STORAGE|e.INDIRECT),re=R*L*7,K=c(k),q=c(R),J=c(re),Y=`const MAXIMUM_CLASS_COUNT: u32 = ${s}u;
${O}`,X=`const HALF_WIDTH_BASE: u32 = 0u;
const MOVEMENT_BASE: u32 = ${s+1}u;
const LEVEL_BASE: u32 = ${2*(s+1)}u;`,Z=`const CELL_COUNT: u32 = ${F}u;
const ELEMENT_COUNT: u32 = ${L}u;
const ELEMENT_STRIDE: u32 = 7u;
const NO_MINIMUM: u32 = ${M};`,ie=2**18;if(V.push(u(t,{id:`${_}-backtrack`,operation:a,variant:`natural-breaks-backtrack`,bindings:[B,{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`starts`,view:r.starts,type:`u32`,access:`read`},{name:`edges`,view:p,type:`f32`,access:`read_write`},{name:`edgeState`,view:g,type:`u32`,access:`read_write`},{name:`extents`,view:H,type:`u32`,access:`read_write`},{name:`dispatch`,view:G,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${x}
const MAXIMUM_CLASS_COUNT: u32 = ${s}u;`,body:`let classCount = ${D};
  let isActive = !(${v}) && classCount >= 2u && extremes[extremesOffset + ${h.minimumKey}u] < extremes[extremesOffset + ${h.maximumKey}u];
  ${b(t,{operation:a,slot:0,invocationCount:k,workgroupSize:K,activeExpression:`isActive`})}
  if (${v}) {
    return;
  }
  ${S}
  var end = BIN_COUNT - 1u;
  for (var layer = classCount - 1u; layer > 0u; layer = layer - 1u) {
    let start = starts[startsOffset + layer * BIN_COUNT + end];
    // The class starting at bin 'start' begins at that bin's lower edge.
    edges[edgesOffset + layer] = minimum + f32(start) * width;
    end = start - 1u;
  }
  edgeState[edgeStateOffset] = classCount;
  for (var index = 0u; index < MAXIMUM_CLASS_COUNT; index = index + 1u) {
    extents[extentsOffset + index] = 0xffffffffu;
    extents[extentsOffset + MAXIMUM_CLASS_COUNT + index] = 0u;
    extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index] = 0u;
  }`})),s<2)return V;let ae=(e,n)=>u(t,{id:e,operation:a,variant:`natural-breaks-extents`,bindings:[{name:`values`,view:o,type:`f32`,access:`read`},...C,B,{name:`edges`,view:p,type:`f32`,access:`read`},{name:`extents`,view:H,type:`atomic<u32>`,access:`read_write`}],invocationCount:k,workgroupSize:K,...y(e,G,n),declarations:`${d}
${Y}`,body:`let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u || !(${E})) {
    return;
  }
  let classCount = ${D};
  let key = getOrderedKey(value);
  // The class is the number of cuts e[1..classCount-1] at or below the key.
  var low = 1u;
  var high = classCount;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(edges[edgesOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  atomicMin(&extents[extentsOffset + low - 1u], key);
  atomicMax(&extents[extentsOffset + MAXIMUM_CLASS_COUNT + low - 1u], key);
  atomicAdd(&extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + low - 1u], 1u);`}),oe=e=>u(t,{id:e,operation:a,variant:`natural-breaks-snap`,bindings:[B,{name:`extents`,view:H,type:`u32`,access:`read`},{name:`edges`,view:p,type:`f32`,access:`read_write`},{name:`breakState`,view:te,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${d}
${Y}
${X}`,body:`if (${v}) {
    return;
  }
  let classCount = ${D};
  for (var cut = 1u; cut < classCount; cut = cut + 1u) {
    let smallest = extents[extentsOffset + cut];
    if (smallest != 0xffffffffu) {
      edges[edgesOffset + cut] = decodeOrderedKey(smallest);
    }
  }
  for (var index = 0u; index <= MAXIMUM_CLASS_COUNT; index = index + 1u) {
    breakState[breakStateOffset + HALF_WIDTH_BASE + index] = 0u;
    breakState[breakStateOffset + MOVEMENT_BASE + index] = bitcast<u32>(${j});
    breakState[breakStateOffset + LEVEL_BASE + index] = 0u;
  }
  // No break lies beside the first and last class.
  breakState[breakStateOffset + MOVEMENT_BASE] = 0u;
  breakState[breakStateOffset + MOVEMENT_BASE + classCount] = 0u;`});V.push(ae(`${_}-extents`,0)),V.push(oe(`${_}-snap`)),V.push(u(t,{id:`${_}-bulk-switch`,operation:a,variant:`natural-breaks-bulk-switch`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`extremes`,view:n.extremes,type:`u32`,access:`read_write`},{name:`extents`,view:H,type:`u32`,access:`read`},{name:`backup`,view:ee,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${Y}
const NO_MINIMUM: u32 = ${M};`,body:`let classCount = ${D};
  let minimumKey = extremes[extremesOffset + ${h.minimumKey}u];
  let maximumKey = extremes[extremesOffset + ${h.maximumKey}u];
  backup[backupOffset] = ${z};
  backup[backupOffset + 1u] = state[stateOffset + ${m.classCount}u];
  backup[backupOffset + 2u] = minimumKey;
  backup[backupOffset + 3u] = maximumKey;
  backup[backupOffset + 4u] = 0u;
  backup[backupOffset + 5u] = 0u;
  backup[backupOffset + 6u] = 0u;
  var bulk = 0u;
  var bulkRows = 0u;
  var nonEmpty = 0u;
  for (var index = 0u; index < classCount; index = index + 1u) {
    if (extents[extentsOffset + index] != NO_MINIMUM) {
      nonEmpty = nonEmpty + 1u;
      let rows = extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index];
      if (rows > bulkRows) {
        bulkRows = rows;
        bulk = index;
      }
    }
  }
  let emptyCount = classCount - nonEmpty;
  let bulkLow = extents[extentsOffset + bulk];
  let bulkHigh = extents[extentsOffset + MAXIMUM_CLASS_COUNT + bulk];
  let isActive = ${z} == ${A[`natural-breaks`]}u && classCount >= 2u && minimumKey < maximumKey
    && emptyCount > 0u && bulkLow < bulkHigh;
  if (isActive) {
    backup[backupOffset + 4u] = bulk;
    backup[backupOffset + 5u] = 1u;
    backup[backupOffset + 6u] = emptyCount;
    extremes[extremesOffset + ${h.minimumKey}u] = bulkLow;
    extremes[extremesOffset + ${h.maximumKey}u] = bulkHigh;
    state[stateOffset + ${m.classCount}u] = min(1u + emptyCount, BIN_COUNT);
    ${`state[stateOffset + ${m.layerCursor}u]`} = 0u;
  } else {
    // Every node of the second stage returns at once.
    ${z} = 0xffffffffu;
  }`})),V.push(...r.getCoarseNodes(`-bulk`)),V.push(u(t,{id:`${_}-bulk-backtrack`,operation:a,variant:`natural-breaks-bulk-backtrack`,bindings:[B,{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`starts`,view:r.starts,type:`u32`,access:`read`},{name:`bulkEdges`,view:U,type:`f32`,access:`read_write`}],invocationCount:1,declarations:x,body:`if (${v}) {
    return;
  }
  let classCount = ${D};
  ${S}
  var end = BIN_COUNT - 1u;
  for (var layer = classCount - 1u; layer > 0u; layer = layer - 1u) {
    let start = starts[startsOffset + layer * BIN_COUNT + end];
    bulkEdges[bulkEdgesOffset + layer] = minimum + f32(start) * width;
    end = start - 1u;
  }`})),V.push(u(t,{id:`${_}-bulk-assemble`,operation:a,variant:`natural-breaks-bulk-assemble`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`extremes`,view:n.extremes,type:`u32`,access:`read_write`},{name:`extents`,view:H,type:`u32`,access:`read_write`},{name:`edges`,view:p,type:`f32`,access:`read_write`},{name:`bulkEdges`,view:U,type:`f32`,access:`read`},{name:`backup`,view:ee,type:`u32`,access:`read`},{name:`dispatch`,view:G,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${d}
${Y}
const NO_MINIMUM: u32 = ${M};`,body:`let isActive = backup[backupOffset + 5u] == 1u;
  ${b(t,{operation:a,slot:5,invocationCount:k,workgroupSize:K,activeExpression:`isActive`})}
  ${z} = backup[backupOffset];
  state[stateOffset + ${m.classCount}u] = backup[backupOffset + 1u];
  extremes[extremesOffset + ${h.minimumKey}u] = backup[backupOffset + 2u];
  extremes[extremesOffset + ${h.maximumKey}u] = backup[backupOffset + 3u];
  if (!isActive) {
    return;
  }
  let classCount = min(backup[backupOffset + 1u], BIN_COUNT);
  let bulk = backup[backupOffset + 4u];
  let parts = min(1u + backup[backupOffset + 6u], BIN_COUNT);
  var emitted = 0u;
  var isFirst = true;
  for (var index = 0u; index < classCount; index = index + 1u) {
    if (extents[extentsOffset + index] == NO_MINIMUM) {
      continue;
    }
    // The smallest value of the class starts it; the first start is the column minimum.
    if (isFirst) {
      isFirst = false;
    } else {
      emitted = emitted + 1u;
      edges[edgesOffset + emitted] = decodeOrderedKey(extents[extentsOffset + index]);
    }
    if (index == bulk) {
      for (var part = 1u; part < parts; part = part + 1u) {
        emitted = emitted + 1u;
        edges[edgesOffset + emitted] = bulkEdges[bulkEdgesOffset + part];
      }
    }
  }
  for (var index = 0u; index < MAXIMUM_CLASS_COUNT; index = index + 1u) {
    extents[extentsOffset + index] = NO_MINIMUM;
    extents[extentsOffset + MAXIMUM_CLASS_COUNT + index] = 0u;
    extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index] = 0u;
  }`})),V.push(ae(`${_}-extents-bulk`,5)),V.push(oe(`${_}-snap-bulk`));let se=`state[stateOffset + ${m.refinementCursor}u]`,Q=`(1u - (${`(${se} - 1u)`} & 1u))`,$=F/2,ce=2/F;for(let e=0;e<P;e++){let r=`${_}-refine-${e}`;V.push(u(t,{id:`${r}-advance`,operation:a,variant:`natural-breaks-refine-advance`,bindings:[{name:`state`,view:n.state,type:`u32`,access:`read_write`},{name:`extremes`,view:n.extremes,type:`u32`,access:`read`},{name:`breakState`,view:te,type:`u32`,access:`read`},{name:`dispatch`,view:G,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${O}
${X}`,body:`let cursor = ${se};
  let classCount = ${D};
  var isActive = ${z} == ${A[`natural-breaks`]}u && classCount >= 2u
    && extremes[extremesOffset + ${h.minimumKey}u] < extremes[extremesOffset + ${h.maximumKey}u]
    && state[stateOffset + ${m.refinementDone}u] == 0u
    && cursor < ${P}u;
  if (isActive && cursor > 0u && (cursor & 1u) == 0u) {
    var zoomed = true;
    for (var cut = 1u; cut < classCount; cut = cut + 1u) {
      if (breakState[breakStateOffset + LEVEL_BASE + cut] < 2u) {
        zoomed = false;
      }
    }
    if (state[stateOffset + ${m.refinementMoved}u] == 0u && zoomed) {
      state[stateOffset + ${m.refinementDone}u] = 1u;
      isActive = false;
    }
    state[stateOffset + ${m.refinementMoved}u] = 0u;
  }
  ${se} = cursor + 1u;
  ${[{slot:1,invocationCount:R,workgroupSize:q},{slot:2,invocationCount:re,workgroupSize:J},{slot:3,invocationCount:k,workgroupSize:K},{slot:4,invocationCount:R,workgroupSize:q}].map(e=>b(t,{operation:a,...e,activeExpression:`isActive`})).join(`
  `)}`})),V.push(u(t,{id:`${r}-plan`,operation:a,variant:`natural-breaks-refine-plan`,bindings:[B,{name:`edges`,view:p,type:`f32`,access:`read`},{name:`extents`,view:H,type:`u32`,access:`read`},{name:`breakState`,view:te,type:`u32`,access:`read_write`},{name:`plan`,view:W,type:`f32`,access:`read_write`}],invocationCount:R,workgroupSize:q,...y(`${r}-plan`,G,1),declarations:`${d}
${Y}
${X}`,body:`let cut = index + 1u;
  if (cut >= ${D}) {
    return;
  }
  plan[planOffset + 3u * cut + 1u] = 0.0;
  if ((cut & 1u) != ${Q}) {
    return;
  }
  // The data extent of the two classes beside the cut.
  let lowKey = min(extents[extentsOffset + cut - 1u], extents[extentsOffset + cut]);
  let highKey = max(extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut - 1u], extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut]);
  if (lowKey == 0xffffffffu) {
    breakState[breakStateOffset + LEVEL_BASE + cut] = ${N}u;
    return;
  }
  let lowValue = decodeOrderedKey(lowKey);
  let highValue = decodeOrderedKey(highKey);
  // A cut beside an empty class lies outside the data; centre the window inside the extent.
  let anchor = min(max(edges[edgesOffset + cut], lowValue), highValue);
  let span = max(anchor - lowValue, highValue - anchor);
  if (!(span > 0.0)) {
    breakState[breakStateOffset + LEVEL_BASE + cut] = ${N}u;
    return;
  }
  // The whole extent until the break has been refined; then the zoomed window, widened to follow
  // neighbours that moved since the break's own last update.
  var halfWidth = span;
  let stored = bitcast<f32>(breakState[breakStateOffset + HALF_WIDTH_BASE + cut]);
  if (stored > 0.0) {
    let shift = max(
      bitcast<f32>(breakState[breakStateOffset + MOVEMENT_BASE + cut - 1u]),
      bitcast<f32>(breakState[breakStateOffset + MOVEMENT_BASE + cut + 1u])
    );
    halfWidth = min(max(stored, 2.0 * min(shift, span)), span);
  }
  plan[planOffset + 3u * cut] = anchor;
  plan[planOffset + 3u * cut + 1u] = halfWidth * ${l(ce)};
  plan[planOffset + 3u * cut + 2u] = ${l(ie)} / span;`})),V.push(u(t,{id:`${r}-clear`,operation:a,variant:`natural-breaks-refine-clear`,bindings:[{name:`cells`,view:ne,type:`u32`,access:`read_write`}],invocationCount:re,workgroupSize:J,...y(`${r}-clear`,G,2),declarations:Z,body:`// Words: count, sum, squares (low and high), smallest key, largest key.
  let word = index % ELEMENT_STRIDE;
  cells[cellsOffset + index] = select(0u, NO_MINIMUM, word == 5u);`})),V.push(u(t,{id:`${r}-accumulate`,operation:a,variant:`natural-breaks-refine-accumulate`,bindings:[{name:`values`,view:o,type:`f32`,access:`read`},...C,B,{name:`edges`,view:p,type:`f32`,access:`read`},{name:`plan`,view:W,type:`f32`,access:`read`},{name:`cells`,view:ne,type:`atomic<u32>`,access:`read_write`}],invocationCount:k,workgroupSize:K,...y(`${r}-accumulate`,G,3),declarations:`${d}
${T}
${O}
${Z}
const QUANTIZATION: f32 = ${l(ie)};`,body:`let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u || !(${E})) {
    return;
  }
  let classCount = ${D};
  let key = getOrderedKey(value);
  // The class of the row: the number of cuts e[1..classCount-1] at or below its key.
  var low = 1u;
  var high = classCount;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(edges[edgesOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let rowClass = low - 1u;
  // The one break of this half-round beside the class: below it (rowClass) or above it.
  let wanted = ${Q};
  var cut = 0u;
  if (rowClass >= 1u && (rowClass & 1u) == wanted) {
    cut = rowClass;
  } else if (rowClass + 1u < classCount && ((rowClass + 1u) & 1u) == wanted) {
    cut = rowClass + 1u;
  }
  if (cut == 0u) {
    return;
  }
  let cellWidth = plan[planOffset + 3u * cut + 1u];
  if (!(cellWidth > 0.0)) {
    return;
  }
  let offset = value - plan[planOffset + 3u * cut];
  // Cell of the row, with the current boundary at cell ${$}: rows of the class on the
  // right of the cut never fall left of it and the other way round.
  var cell = i32(floor(clamp(offset / cellWidth, -${2*F}.0, ${2*F}.0))) + ${$};
  if (rowClass == cut) {
    cell = max(cell, ${$});
  } else {
    cell = min(cell, ${$-1});
  }
  // Element 0 holds the rows left of the window, 1..CELL_COUNT the cells, the last the rows right.
  var element = u32(cell) + 1u;
  if (cell < 0) {
    element = 0u;
  } else if (cell >= ${F}) {
    element = ELEMENT_COUNT - 1u;
  }
  let quantized = i32(round(clamp(offset * plan[planOffset + 3u * cut + 2u], -QUANTIZATION, QUANTIZATION)));
  let magnitude = u32(abs(quantized));
  let square = multiplyWide(magnitude, magnitude);
  let base = cellsOffset + ((cut - 1u) * ELEMENT_COUNT + element) * ELEMENT_STRIDE;
  atomicAdd(&cells[base], 1u);
  ${I(`cells`,`base + 1u`,`base + 2u`,`bitcast<u32>(quantized)`,`select(0u, 0xffffffffu, (quantized < 0))`)}
  ${I(`cells`,`base + 3u`,`base + 4u`,`square.x`,`square.y`)}
  atomicMin(&cells[base + 5u], key);
  atomicMax(&cells[base + 6u], key);`})),V.push(u(t,{id:`${r}-solve`,operation:a,variant:`natural-breaks-refine-solve`,bindings:[{name:`state`,view:n.state,type:`atomic<u32>`,access:`read_write`},{name:`edges`,view:p,type:`f32`,access:`read_write`},{name:`extents`,view:H,type:`u32`,access:`read_write`},{name:`breakState`,view:te,type:`u32`,access:`read_write`},{name:`plan`,view:W,type:`f32`,access:`read`},{name:`cells`,view:ne,type:`u32`,access:`read`}],invocationCount:R,workgroupSize:q,...y(`${r}-solve`,G,4),declarations:`${d}
${T}
${Y}
${X}
${Z}`,body:`let cut = index + 1u;
  if (cut >= min(atomicLoad(&state[stateOffset + ${m.classCount}u]), BIN_COUNT)) {
    return;
  }
  let cellWidth = plan[planOffset + 3u * cut + 1u];
  if (!(cellWidth > 0.0)) {
    return;
  }
  let base = cellsOffset + (cut - 1u) * ELEMENT_COUNT * ELEMENT_STRIDE;
  var totalCount = 0u;
  var totalSum = vec2<u32>(0u, 0u);
  var totalSquares = vec2<u32>(0u, 0u);
  for (var element = 0u; element < ELEMENT_COUNT; element = element + 1u) {
    let at = base + element * ELEMENT_STRIDE;
    totalCount = totalCount + cells[at];
    totalSum = addWide(totalSum, vec2<u32>(cells[at + 1u], cells[at + 2u]));
    totalSquares = addWide(totalSquares, vec2<u32>(cells[at + 3u], cells[at + 4u]));
  }
  // Cost of every boundary 0..CELL_COUNT: elements 0..boundary are left of it. The current cut is
  // boundary ${$}; another boundary has to be strictly cheaper to replace it.
  var leftCount = cells[base];
  var leftSum = vec2<u32>(cells[base + 1u], cells[base + 2u]);
  var leftSquares = vec2<u32>(cells[base + 3u], cells[base + 4u]);
  var bestCost = ${j};
  var bestBoundary = ${$}u;
  var currentCost = ${j};
  var currentLeftCount = 0u;
  var cost = ${j};
  for (var boundary = 0u; boundary <= CELL_COUNT; boundary = boundary + 1u) {
    // An empty cell leaves the partition, and so the cost, as it was one boundary before.
    var isChanged = boundary == 0u || boundary == ${$}u;
    if (boundary > 0u) {
      let at = base + boundary * ELEMENT_STRIDE;
      isChanged = isChanged || cells[at] != 0u;
      leftCount = leftCount + cells[at];
      leftSum = addWide(leftSum, vec2<u32>(cells[at + 1u], cells[at + 2u]));
      leftSquares = addWide(leftSquares, vec2<u32>(cells[at + 3u], cells[at + 4u]));
    }
    if (isChanged) {
      cost = getClassCost(leftCount, leftSum, leftSquares)
        + getClassCost(totalCount - leftCount, subtractWide(totalSum, leftSum), subtractWide(totalSquares, leftSquares));
    }
    if (boundary == ${$}u) {
      currentCost = cost;
      currentLeftCount = leftCount;
    }
    if (cost < bestCost) {
      bestCost = cost;
      bestBoundary = boundary;
    }
  }
  if (currentCost <= bestCost) {
    bestBoundary = ${$}u;
  }
  // Extents of the two new classes: elements 0..bestBoundary on the left, the rest on the right.
  var leftMinimum = NO_MINIMUM;
  var leftMaximum = 0u;
  var rightMinimum = NO_MINIMUM;
  var rightMaximum = 0u;
  for (var element = 0u; element < ELEMENT_COUNT; element = element + 1u) {
    let at = base + element * ELEMENT_STRIDE;
    if (cells[at] != 0u) {
      if (element <= bestBoundary) {
        leftMinimum = min(leftMinimum, cells[at + 5u]);
        leftMaximum = max(leftMaximum, cells[at + 6u]);
      } else {
        rightMinimum = min(rightMinimum, cells[at + 5u]);
        rightMaximum = max(rightMaximum, cells[at + 6u]);
      }
    }
  }
  extents[extentsOffset + cut - 1u] = leftMinimum;
  extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut - 1u] = leftMaximum;
  extents[extentsOffset + cut] = rightMinimum;
  extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut] = rightMaximum;
  // The cut is the smallest value on its right: an actual data value.
  var movement = 0.0;
  if (rightMinimum != NO_MINIMUM) {
    let newCut = decodeOrderedKey(rightMinimum);
    movement = abs(newCut - edges[edgesOffset + cut]);
    edges[edgesOffset + cut] = newCut;
    if (movement > 0.0) {
      atomicOr(&state[stateOffset + ${m.refinementMoved}u], 1u);
    }
  }
  breakState[breakStateOffset + MOVEMENT_BASE + cut] = bitcast<u32>(movement);
  // Zoom around the chosen boundary, unless a class beside the break was empty (the window must
  // cover the extent again) or the optimum sits at the window edge (the cut may continue).
  let degenerate = currentLeftCount == 0u || totalCount - currentLeftCount == 0u;
  let atEdge = abs(i32(bestBoundary) - ${$}) >= ${$-4};
  let halfWidth = cellWidth * ${l($)};
  var nextHalfWidth = halfWidth * ${l(8/F)};
  var level = breakState[breakStateOffset + LEVEL_BASE + cut] + 1u;
  if (degenerate) {
    nextHalfWidth = 0.0;
    level = 0u;
  } else if (atEdge) {
    nextHalfWidth = halfWidth;
    level = level - 1u;
  }
  breakState[breakStateOffset + HALF_WIDTH_BASE + cut] = bitcast<u32>(nextHalfWidth);
  breakState[breakStateOffset + LEVEL_BASE + cut] = level;`}))}return V.push(u(t,{id:`${_}-finalize`,operation:a,variant:`natural-breaks-finalize`,bindings:[B,{name:`extents`,view:H,type:`u32`,access:`read`},{name:`edges`,view:p,type:`f32`,access:`read_write`},{name:`edgeState`,view:g,type:`u32`,access:`read_write`}],invocationCount:1,declarations:Y,body:`if (${v}) {
    return;
  }
  let classCount = edgeState[edgeStateOffset];
  var kept = 0u;
  for (var cut = 1u; cut < classCount; cut = cut + 1u) {
    if (extents[extentsOffset + cut] != 0xffffffffu) {
      kept = kept + 1u;
      edges[edgesOffset + kept] = edges[edgesOffset + cut];
    }
  }
  edgeState[edgeStateOffset] = kept + 1u;`})),V}var z=`GPUClassBreaks`,B=64,V=2**24,H=2048,U=2048,ee=3,te=4294967295,W=E,ne=4,G=class{id;props;methods;naturalBreaksBinCount;quantileCount;constructor(e){this.id=e.id??`class-breaks`,this.props=e;let t=this.id,{maximumClassCount:i,output:a}=e;if(n(t,`maximumClassCount`,i,1,B),this.methods=[...new Set(e.methods??D)],this.methods.length===0)throw Error(`${t} needs at least one method`);for(let e of this.methods)if(W[e]===void 0)throw Error(`${t} unknown method ${String(e)}`);if(this.methods.includes(`box-plot`)&&i<6)throw Error(`${t} box-plot needs maximumClassCount of at least 6`);this.naturalBreaksBinCount=e.naturalBreaksBinCount??H,n(t,`naturalBreaksBinCount`,this.naturalBreaksBinCount,2,U),this.quantileCount=Math.max(i-1,3),o(e.values,[`float32`,`uint32`,`sint32`],`${t} values`);let s=e.values.length;if(n(t,`values length`,s,1,V),e.mask&&(o(e.mask,[`uint32`],`${t} mask`),e.mask.length!==s))throw Error(`${t} mask length must equal values length`);o(e.parameters,[`float32`],`${t} parameters`);let c=O(i);if(e.parameters.length<c)throw Error(`${t} parameters must hold ${c} float32 values`);if(o(a.breaks,[`float32`],`${t} output.breaks`),a.breaks.length<i+1)throw Error(`${t} output.breaks must hold maximumClassCount + 1 rows`);if(o(a.classCount,[`uint32`],`${t} output.classCount`),a.classCount.length<1)throw Error(`${t} output.classCount must hold one row`);if(a.classCounts&&(o(a.classCounts,[`uint32`],`${t} output.classCounts`),a.classCounts.length<i))throw Error(`${t} output.classCounts must hold maximumClassCount rows`);r(t,[a.breaks,a.classCount,a.classCounts],[e.values,e.mask,e.parameters])}getCommandNodes(n){let{id:r,props:a,methods:o,quantileCount:s}=this,{mask:c,parameters:l,output:p,maximumClassCount:_}=a;t(r,n,[a.values,c,l,p.breaks,p.classCount,p.classCounts]);let v=a.values.length,C=a.values.format===`float32`?void 0:a.values.format,w=C?i(n,`${r}-float-values`,`float32`,v):a.values,T=e=>o.includes(e),E=T(`quantile`)||T(`box-plot`),D=T(`standard-deviation`),O=T(`head-tail`),k=_-1,A=(e,t)=>i(n,`${r}-${e}`,`uint32`,t),j=(e,t)=>i(n,`${r}-${e}`,`float32`,t),M=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),N=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),P=c?[M(`rowMask`,c,`u32`)]:[],F=c?`rowMask[rowMaskOffset + index] != 0u`:`true`,I=A(`state`,6),R=A(`extremes`,4),B=j(`edges`,_+1),V=A(`edge-state`,1),H=E?A(`quantile-gate`,1):void 0,U=E?j(`quantile-parameters`,4+s):void 0,G=E?j(`quantile-values`,s):void 0,K=D||O,q=(O?1:0)+1,J=K?j(`moments`,q*6):void 0,Y=K?j(`moment-partials`,S(v)*6):void 0,X=O?A(`head-tail`,ne+Math.max(k,1)):void 0,Z=O?i(n,`${r}-head-tail-dispatch`,`uint32`,6,e.STORAGE|e.INDIRECT):void 0,ie=e=>[b(n,{operation:z,slot:0,invocationCount:S(v)*256,workgroupSize:256,activeExpression:e}),b(n,{operation:z,slot:1,invocationCount:256,workgroupSize:256,activeExpression:e})].join(`
  `),ae=o.reduce((e,t)=>e|1<<W[t],0),oe=`state[stateOffset + ${m.method}u]`,se=`state[stateOffset + ${m.classCount}u]`,Q=[];if(C){let e=C===`uint32`?`u32`:`i32`;Q.push(u(n,{id:`${r}-integer-values`,operation:z,variant:`integer-values`,bindings:[{name:`integers`,view:a.values,type:e,access:`read`},{name:`floats`,view:w,type:`f32`,access:`read_write`}],invocationCount:v,body:`floats[floatsOffset + index] = f32(integers[integersOffset + index]);`}))}let $=[M(`params`,l,`f32`),N(`state`,I),N(`extremes`,R),N(`edgeState`,V),...H&&U?[N(`quantileGate`,H),N(`quantileParameters`,U,`f32`)]:[],...X?[N(`headTail`,X)]:[],...Z?[N(`dispatch`,Z)]:[]];if(Q.push(u(n,{id:`${r}-prepare`,operation:z,variant:`prepare`,bindings:$,invocationCount:1,declarations:`const COMPILED_METHODS: u32 = ${ae}u;
const MAXIMUM_CLASS_COUNT: u32 = ${_}u;
const QUANTILE_COUNT: u32 = ${s}u;
const HEADER: u32 = 8u;`,body:`let methodValue = params[paramsOffset];
  var method = ${te}u;
  if (methodValue >= 0.0 && methodValue < 32.0 && floor(methodValue) == methodValue) {
    let code = u32(methodValue);
    if ((COMPILED_METHODS & (1u << code)) != 0u) {
      method = code;
    }
  }
  let classValue = params[paramsOffset + 1u];
  var classCount = 1u;
  if (classValue >= 1.0) {
    classCount = u32(min(classValue, f32(MAXIMUM_CLASS_COUNT)));
  }
  if (method == ${W[`box-plot`]}u) {
    classCount = 6u;
  }
  state[stateOffset + ${m.method}u] = method;
  state[stateOffset + ${m.classCount}u] = classCount;
  state[stateOffset + ${m.layerCursor}u] = 0u;
  state[stateOffset + ${m.refinementCursor}u] = 0u;
  state[stateOffset + ${m.refinementMoved}u] = 0u;
  state[stateOffset + ${m.refinementDone}u] = 0u;
  extremes[extremesOffset + ${h.minimumKey}u] = 0xffffffffu;
  extremes[extremesOffset + ${h.maximumKey}u] = 0u;
  extremes[extremesOffset + ${h.validCount}u] = 0u;
  extremes[extremesOffset + ${h.finiteCount}u] = 0u;
  edgeState[edgeStateOffset] = 0u;
  ${H?`quantileGate[quantileGateOffset] = select(0u, 1u, method == ${W.quantile}u || method == ${W[`box-plot`]}u);
  quantileParameters[quantileParametersOffset] = ${ee}.0;
  quantileParameters[quantileParametersOffset + 1u] = 0.0;
  quantileParameters[quantileParametersOffset + 2u] = 1.0;
  quantileParameters[quantileParametersOffset + 3u] = 0.0;
  for (var slot = 0u; slot < QUANTILE_COUNT; slot = slot + 1u) {
    quantileParameters[quantileParametersOffset + 4u + slot] = params[paramsOffset + HEADER + slot];
  }`:``}
  ${X?`headTail[headTailOffset + ${g.done}u] = 0u;
  headTail[headTailOffset + ${g.breakCount}u] = 0u;
  headTail[headTailOffset + ${g.thresholdKey}u] = 0u;
  headTail[headTailOffset + ${g.previousCount}u] = 0u;
  ${ie(`method == ${W[`head-tail`]}u`)}`:``}`})),Q.push(u(n,{id:`${r}-extremes`,operation:z,variant:`extremes`,bindings:[M(`values`,w,`f32`),...P,N(`extremes`,R,`atomic<u32>`)],invocationCount:v,declarations:d,body:`let value = values[valuesOffset + index];
  if (isNanBits(value) || !(${F})) {
    return;
  }
  atomicAdd(&extremes[extremesOffset + ${h.validCount}u], 1u);
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return;
  }
  let key = getOrderedKey(value);
  atomicAdd(&extremes[extremesOffset + ${h.finiteCount}u], 1u);
  atomicMin(&extremes[extremesOffset + ${h.minimumKey}u], key);
  atomicMax(&extremes[extremesOffset + ${h.maximumKey}u], key);`})),H&&U&&G&&Q.push(...f(n,{id:`${r}-quantiles`,operation:z,values:w,mask:c,parameters:U,quantileCount:s,output:{quantiles:G,validCount:A(`quantile-valid-count`,1)},gate:{view:H,value:1}})),J&&Y&&D&&Q.push(...x(n,{id:`${r}-moments`,operation:z,values:w,mask:c,state:I,partials:Y,moments:J,slot:q-1,gate:{kind:`method`,methodCode:W[`standard-deviation`]}})),J&&Y&&X&&Z)for(let e=0;e<k;e++){let t=`${r}-head-tail-${e}`;Q.push(...x(n,{id:t,operation:z,values:w,mask:c,state:I,partials:Y,moments:J,slot:0,gate:{kind:`head-tail`,methodCode:W[`head-tail`],headTail:X},indirectGates:{tile:y(`${t}-tile`,Z,0),merge:y(`${t}-merge`,Z,1)}})),Q.push(u(n,{id:`${t}-split`,operation:z,variant:`head-tail-split`,bindings:[M(`params`,l,`f32`),M(`state`,I,`u32`),M(`moments`,J,`f32`),N(`headTail`,X),N(`dispatch`,Z)],invocationCount:1,declarations:d,body:re(oe,se,ie(`false`))}))}T(`maximum-breaks`)&&Q.push(...this.getMaximumBreaksNodes(n,{values:w,state:I,extremes:R,edges:B,edgeState:V})),T(`natural-breaks`)&&Q.push(...L(n,{id:r,operation:z,values:w,mask:c,state:I,extremes:R,edges:B,edgeState:V,maximumClassCount:_,binCount:this.naturalBreaksBinCount}));let ce=[M(`params`,l,`f32`),M(`state`,I,`u32`),M(`extremes`,R,`u32`),...G?[M(`quantileValues`,G,`f32`)]:[],...J?[M(`moments`,J,`f32`)]:[],...X?[M(`headTail`,X,`u32`)]:[],N(`edges`,B,`f32`)];return Q.push(u(n,{id:`${r}-finish`,operation:z,variant:`finish`,bindings:[...ce,N(`edgeState`,V)],invocationCount:1,declarations:`${d}
const HEADER: u32 = 8u;`,body:this.getFinishBody({methodRead:oe,classCountRead:se,hasQuantiles:!!G,momentsSlot:D?q-1:-1,hasHeadTail:!!X})})),Q.push(u(n,{id:`${r}-publish`,operation:z,variant:`publish`,bindings:[M(`state`,I,`u32`),M(`extremes`,R,`u32`),M(`edges`,B,`f32`),M(`edgeState`,V,`u32`),N(`breaksOut`,p.breaks,`f32`),N(`classCountOut`,p.classCount)],invocationCount:1,declarations:`${d}
const MAXIMUM_CLASS_COUNT: u32 = ${_}u;`,body:`let method = ${oe};
  let finiteCount = extremes[extremesOffset + ${h.finiteCount}u];
  let minimum = decodeOrderedKey(extremes[extremesOffset + ${h.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${h.maximumKey}u]);
  var classes = edgeState[edgeStateOffset];
  let isCustom = method == ${W.custom}u;
  if (method == ${te}u) {
    classes = 0u;
  } else if (!isCustom && finiteCount == 0u) {
    classes = 0u;
  } else if (!isCustom && minimum == maximum) {
    classes = 1u;
  }
  classes = min(classes, MAXIMUM_CLASS_COUNT);
  // A NaN edge (a hand-written parameter view with a NaN probability, hinge, interval or custom
  // edge) makes the classification undefined: report 0 classes and NaN edges, the no-data result,
  // instead of clamping NaN to an end edge and assigning a silent wrong class.
  for (var edge = 0u; edge <= classes; edge = edge + 1u) {
    let isComputedEnd = !isCustom && (edge == 0u || edge == classes);
    if (!isComputedEnd && isNanBits(edges[edgesOffset + edge])) {
      classes = 0u;
    }
  }
  let nan = getNaN();
  for (var edge = 0u; edge <= MAXIMUM_CLASS_COUNT; edge = edge + 1u) {
    var value = nan;
    if (classes > 0u && edge <= classes) {
      if (isCustom) {
        value = edges[edgesOffset + edge];
      } else if (edge == 0u) {
        value = minimum;
      } else if (edge == classes) {
        value = maximum;
      } else {
        value = clamp(edges[edgesOffset + edge], minimum, maximum);
      }
    }
    breaksOut[breaksOutOffset + edge] = value;
  }
  classCountOut[classCountOutOffset] = classes;`})),p.classCounts&&(Q.push(u(n,{id:`${r}-class-counts-clear`,operation:z,variant:`class-counts-clear`,bindings:[N(`classCounts`,p.classCounts)],invocationCount:_,body:`classCounts[classCountsOffset + index] = 0u;`})),Q.push(u(n,{id:`${r}-class-counts`,operation:z,variant:`class-counts`,bindings:[M(`values`,w,`f32`),...P,M(`breaks`,p.breaks,`f32`),M(`classCountIn`,p.classCount,`u32`),N(`classCounts`,p.classCounts,`atomic<u32>`)],invocationCount:v,declarations:d,body:`let value = values[valuesOffset + index];
  let classes = classCountIn[classCountInOffset];
  if (classes == 0u || isNanBits(value) || !(${F})) {
    return;
  }
  let key = getOrderedKey(value);
  // Binary search for the number of inner edges e[1..classes-1] whose key is at or below key.
  var low = 1u;
  var high = classes;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(breaks[breaksOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  atomicAdd(&classCounts[classCountsOffset + low - 1u], 1u);`}))),Q}getFinishBody(e){let{methodRead:t,classCountRead:n,hasQuantiles:r,momentsSlot:i,hasHeadTail:a}=e,o=`extremes[extremesOffset + ${h.minimumKey}u]`,s=`extremes[extremesOffset + ${h.maximumKey}u]`,c=i*6;return`let method = ${t};
  let classCount = ${n};
  let minimum = decodeOrderedKey(${o});
  let maximum = decodeOrderedKey(${s});
  if (method == ${W[`equal-interval`]}u) {
    let width = (maximum - minimum) / f32(classCount);
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = minimum + width * f32(edge);
    }
    edgeState[edgeStateOffset] = classCount;
  }
  ${r?`if (method == ${W.quantile}u) {
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = quantileValues[quantileValuesOffset + edge - 1u];
    }
    edgeState[edgeStateOffset] = classCount;
  }
  if (method == ${W[`box-plot`]}u) {
    let hinge = params[paramsOffset + 4u];
    let lowerQuartile = quantileValues[quantileValuesOffset];
    let median = quantileValues[quantileValuesOffset + 1u];
    let upperQuartile = quantileValues[quantileValuesOffset + 2u];
    let spread = upperQuartile - lowerQuartile;
    edges[edgesOffset + 1u] = lowerQuartile - hinge * spread;
    edges[edgesOffset + 2u] = lowerQuartile;
    edges[edgesOffset + 3u] = median;
    edges[edgesOffset + 4u] = upperQuartile;
    edges[edgesOffset + 5u] = upperQuartile + hinge * spread;
    edgeState[edgeStateOffset] = 6u;
  }`:``}
  ${i>=0?`if (method == ${W[`standard-deviation`]}u) {
    let count = moments[momentsOffset + ${c}u];
    let mean = moments[momentsOffset + ${c+1}u];
    // Sample standard deviation (n - 1), as mapclassify StdMean and R classInt 'sd'.
    let deviation = sqrt(moments[momentsOffset + ${c+2}u] / max(count - 1.0, 1.0));
    let interval = params[paramsOffset + 2u] * deviation;
    let center = f32(classCount - 2u) * 0.5;
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = mean + (f32(edge - 1u) - center) * interval;
    }
    edgeState[edgeStateOffset] = classCount;
  }`:``}
  ${a?`if (method == ${W[`head-tail`]}u) {
    let breakCount = headTail[headTailOffset + ${g.breakCount}u];
    for (var edge = 1u; edge <= breakCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = bitcast<f32>(
        headTail[headTailOffset + ${ne-1}u + edge]
      );
    }
    edgeState[edgeStateOffset] = breakCount + 1u;
  }`:``}
  if (method == ${W.custom}u) {
    for (var edge = 0u; edge <= classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = params[paramsOffset + HEADER + edge];
    }
    edgeState[edgeStateOffset] = classCount;
  }`}getMaximumBreaksNodes(e,t){let{id:n,props:r}=this,{values:a}=t,{mask:o,maximumClassCount:c}=r,l=a.length,f=`${n}-maximum-breaks`,p=t=>i(e,`${f}-${t}`,`uint32`,l),g=p(`keys`),_=p(`rows`),v=p(`sorted-keys`),y=p(`sorted-rows`),b=p(`gap-keys`),x=p(`gap-positions`),S=p(`sorted-gap-keys`),C=p(`sorted-gap-positions`),w=`state[stateOffset + ${m.method}u] != ${W[`maximum-breaks`]}u`,T=[];return T.push(u(e,{id:`${f}-keys`,operation:z,variant:`maximum-breaks-keys`,bindings:[{name:`values`,view:a,type:`f32`,access:`read`},...o?[{name:`rowMask`,view:o,type:`u32`,access:`read`}]:[],{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`keys`,view:g,type:`u32`,access:`read_write`},{name:`rows`,view:_,type:`u32`,access:`read_write`}],invocationCount:l,declarations:d,body:`if (${w}) {
    return;
  }
  let value = values[valuesOffset + index];
  var include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  ${o?`include = include && rowMask[rowMaskOffset + index] != 0u;`:``}
  // Excluded rows take the NaN key 0xffffffff, which no finite value has, and sort last.
  keys[keysOffset + index] = select(0xffffffffu, getOrderedKey(value), include);
  rows[rowsOffset + index] = index;`})),T.push(...new s({id:`${f}-sort`,keys:g,values:_,outputKeys:v,outputValues:y,algorithm:`radix`}).getCommandNodes(e)),T.push(u(e,{id:`${f}-gaps`,operation:z,variant:`maximum-breaks-gaps`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`extremes`,view:t.extremes,type:`u32`,access:`read`},{name:`sortedKeys`,view:v,type:`u32`,access:`read`},{name:`gapKeys`,view:b,type:`u32`,access:`read_write`},{name:`gapPositions`,view:x,type:`u32`,access:`read_write`}],invocationCount:l,declarations:d,body:`if (${w}) {
    return;
  }
  let finiteCount = extremes[extremesOffset + ${h.finiteCount}u];
  var gapKey = 0xffffffffu;
  if (index > 0u && index < finiteCount) {
    let lowerKey = sortedKeys[sortedKeysOffset + index - 1u];
    let upperKey = sortedKeys[sortedKeysOffset + index];
    if (upperKey != lowerKey) {
      let gap = decodeOrderedKey(upperKey) - decodeOrderedKey(lowerKey);
      // Inverted keys sort the largest gap first; the stable sort keeps lower positions first.
      gapKey = ~getOrderedKey(gap);
    }
  }
  gapKeys[gapKeysOffset + index] = gapKey;
  gapPositions[gapPositionsOffset + index] = index;`})),T.push(...new s({id:`${f}-gap-sort`,keys:b,values:x,outputKeys:S,outputValues:C,algorithm:`radix`}).getCommandNodes(e)),T.push(u(e,{id:`${f}-finish`,operation:z,variant:`maximum-breaks-finish`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`sortedKeys`,view:v,type:`u32`,access:`read`},{name:`sortedGapKeys`,view:S,type:`u32`,access:`read`},{name:`sortedGapPositions`,view:C,type:`u32`,access:`read`},{name:`edges`,view:t.edges,type:`f32`,access:`read_write`},{name:`edgeState`,view:t.edgeState,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${d}
const ROW_COUNT: u32 = ${l}u;
const MAXIMUM_CLASS_COUNT: u32 = ${c}u;`,body:`if (${w}) {
    return;
  }
  let classCount = state[stateOffset + ${m.classCount}u];
  var midpoints: array<f32, ${c}>;
  var found = 0u;
  for (var rank = 0u; rank + 1u < classCount && rank < ROW_COUNT; rank = rank + 1u) {
    if (sortedGapKeys[sortedGapKeysOffset + rank] == 0xffffffffu) {
      break;
    }
    let position = sortedGapPositions[sortedGapPositionsOffset + rank];
    let lower = decodeOrderedKey(sortedKeys[sortedKeysOffset + position - 1u]);
    let upper = decodeOrderedKey(sortedKeys[sortedKeysOffset + position]);
    // (upper - lower) * 0.5 is exact, so a fused multiply-add rounds the same way.
    let midpoint = lower + (upper - lower) * 0.5;
    // Insertion sort keeps the midpoints ascending.
    var slot = found;
    while (slot > 0u && midpoints[slot - 1u] > midpoint) {
      midpoints[slot] = midpoints[slot - 1u];
      slot = slot - 1u;
    }
    midpoints[slot] = midpoint;
    found = found + 1u;
  }
  for (var edge = 0u; edge < found; edge = edge + 1u) {
    edges[edgesOffset + edge + 1u] = midpoints[edge];
  }
  edgeState[edgeStateOffset] = found + 1u;`})),T}};function re(e,t,n){let r=e=>`headTail[headTailOffset + ${g[e]}u]`,i=r(`done`);return`if (${e} != ${W[`head-tail`]}u || ${i} != 0u) {
    return;
  }
  let round = ${r(`breakCount`)};
  let count = moments[momentsOffset];
  let mean = moments[momentsOffset + 1u];
  let minimum = moments[momentsOffset + 3u];
  let maximum = moments[momentsOffset + 4u];
  var ratio = params[paramsOffset + 3u];
  if (!(ratio >= 0.0)) {
    ratio = 0.4;
  }
  // At most k - 1 breaks; stop on an empty or single-valued head.
  var stop = round + 1u >= ${t} || count < 1.0 || !(minimum < maximum);
  if (!stop && round > 0u) {
    let previousCount = bitcast<f32>(${r(`previousCount`)});
    // The head kept from the previous split must be a minority (the 40% rule).
    stop = count > ratio * previousCount || count <= 1.0;
  }
  if (stop) {
    ${i} = 1u;
    ${n}
    return;
  }
  headTail[headTailOffset + ${ne}u + round] = bitcast<u32>(mean);
  ${r(`breakCount`)} = round + 1u;
  ${r(`thresholdKey`)} = getOrderedKey(mean);
  ${r(`previousCount`)} = bitcast<u32>(count);`}export{k as a,O as i,D as n,E as r,G as t};