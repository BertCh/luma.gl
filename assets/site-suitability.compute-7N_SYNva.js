import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{M as e,N as t,R as n,_ as r,n as i,t as a,w as o}from"./resources-CwvhEOBe.js";import{l as s,s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as l,l as u,r as d,t as f}from"./wgsl-common-BNRVzgd8.js";import{a as p}from"./raster-utils-DhdnkkOa.js";import{i as m,s as h}from"./raster-band-utils-CA-sh8xt.js";import{a as g,i as _,n as v,o as y,r as b,s as x,t as S}from"./raster-algebra-utils-C7qGzplC.js";import{a as ee,i as te,n as C,o as ne,r as w,s as re,t as ie}from"./gpu-raster-arithmetic-Dk0SdDgu.js";import{n as ae}from"./gpu-raster-grid-BOSvDY5A.js";import{t as oe}from"./gpu-raster-zonal-statistics-BWDSXs3o.js";import{U as se,a as ce,r as le}from"./index-Bjk18V2I.js";import{t as ue}from"./summary-reader-CsgWxENf.js";import{n as de,t as fe}from"./gpu-terrain-derivatives-BZk8xB3m.js";import{n as pe,t as me}from"./b16-colors-Jr7XN5JQ.js";import{c as he,i as T,n as E,o as ge,s as _e,t as D}from"./b16-common-3eXVh1Vw.js";var O=8,ve=8;function k(e){return O+ve*e}function ye(e,t=new Float32Array(k(e.layers.length)),n={}){let r=k(e.layers.length);m(`getGPUWeightedOverlayParameterValues`,t,r),t.fill(0,0,r),t[0]=e.normalizeWeights?1:0,t[1]=e.noDataPolicy===`ignore`?1:0;for(let[r,i]of e.layers.entries()){let e=i.inputMinimum??0,a=i.inputMaximum??1,o=i.breakCount??0;for(let[t,n]of[[`weight`,i.weight],[`inputMinimum`,e],[`inputMaximum`,a]])if(!Number.isFinite(n))throw Error(`getGPUWeightedOverlayParameterValues layer ${r} ${t} must be finite`);if(!Number.isSafeInteger(o)||o<0)throw Error(`getGPUWeightedOverlayParameterValues layer ${r} breakCount must be a non-negative integer`);if(i.mode===`table`&&n.hasTables===!1)throw Error(`getGPUWeightedOverlayParameterValues layer ${r} is a table layer but has no remap tables`);let s=O+r*ve,c=Math.fround(a)===Math.fround(e);t[s]=i.weight,t[s+1]=i.mode===`table`?1:0,t[s+2]=e,t[s+3]=c?0:1/(Math.fround(a)-Math.fround(e)),t[s+4]=i.invert?1:0,t[s+5]=o,t[s+6]=i.closed===`right`?1:0,t[s+7]=c?1:0}return t}var A=`GPUWeightedOverlay`,j=256,be=class{id;props;maximumBreakCount;constructor(e){this.id=e.id??`weighted-overlay`,this.props=e;let{id:t}=this,{output:n,layerCount:r,cellCount:i}=e;if(_(t,`layerCount`,r,1,16),_(t,`cellCount`,i,0,4294967295),r*i>4294967295)throw Error(`${t} layerCount * cellCount must fit in uint32`);if(y(t,e.noDataValue),x(t,`stack`,e.stack,`float32`,r*i),x(t,`parameters`,e.parameters,`float32`,k(r)),!!e.remapBreaks!=!!e.remapValues)throw Error(`${t} remapBreaks and remapValues must be provided together`);if(this.maximumBreakCount=e.remapBreaks?e.maximumBreakCount??0:0,e.remapBreaks&&(_(t,`maximumBreakCount`,this.maximumBreakCount,1,65536),x(t,`remapBreaks`,e.remapBreaks,`float32`,r*this.maximumBreakCount),x(t,`remapValues`,e.remapValues,`float32`,r*(this.maximumBreakCount+1))),!n?.score)throw Error(`${t} needs output.score`);x(t,`output.score`,n.score,`float32`,i),x(t,`output.outputValidity`,n.outputValidity,`uint32`,i),x(t,`output.scoreRange`,n.scoreRange,`float32`,2),b(t,[n.score,n.outputValidity,n.scoreRange],[e.stack,e.parameters,e.remapBreaks,e.remapValues])}getCommandNodes(e){let{id:t,props:n,maximumBreakCount:r}=this,{output:i,layerCount:a,cellCount:s}=n;g(t,e,[n.stack,n.parameters,n.remapBreaks,n.remapValues,i.score,i.outputValidity,i.scoreRange]);let u=[],p=i.scoreRange?o(e,`${t}-range-keys`,`uint32`,2):void 0;p&&u.push(c(e,{id:`${t}-clear-range`,operation:A,variant:`clear-range`,bindings:[{name:`keys`,view:p,type:`u32`,access:`read_write`}],invocationCount:1,body:`keys[keysOffset] = 0xffffffffu;
  keys[keysOffset + 1u] = 0u;`}));let m=!!n.remapBreaks,h=[{name:`stack`,view:n.stack,type:`f32`,access:`read`},{name:`params`,view:n.parameters,type:`f32`,access:`read`}];return m&&h.push({name:`remapBreaks`,view:n.remapBreaks,type:`f32`,access:`read`},{name:`remapValues`,view:n.remapValues,type:`f32`,access:`read`}),h.push({name:`scoreOut`,view:i.score,type:`f32`,access:`read_write`}),i.outputValidity&&h.push({name:`validityOut`,view:i.outputValidity,type:`u32`,access:`read_write`}),p&&h.push({name:`rangeKeys`,view:p,type:`atomic<u32>`,access:`read_write`}),u.push(c(e,{id:`${t}-score`,operation:A,variant:`score`,bindings:h,invocationCount:s,guardIndex:!p,workgroupSize:p?j:void 0,declarations:`const LAYER_COUNT: u32 = ${a}u;
const CELL_COUNT: u32 = ${s}u;
const MAXIMUM_BREAK_COUNT: u32 = ${r}u;
${p?`var<workgroup> tileMinimum: array<u32, ${j}>;
var<workgroup> tileMaximum: array<u32, ${j}>;`:``}
${v(n.noDataValue)}
${m?S(`countBreaksBelow`,`remapBreaks`):``}
${l}`,body:`${p?`var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  if (index < CELL_COUNT) {`:``}
  let normalizeWeights = params[paramsOffset] != 0.0;
  let ignoreNoData = params[paramsOffset + 1u] != 0.0;
  var sum = 0.0;
  var weightSum = 0.0;
  var usedCount = 0u;
  var isDefined = true;
  for (var layer = 0u; layer < LAYER_COUNT; layer++) {
    let base = paramsOffset + 8u + layer * 8u;
    let weight = params[base];
    let value = stack[stackOffset + layer * CELL_COUNT + index];
    if (isNoDataValue(value)) {
      if (!ignoreNoData) {
        isDefined = false;
      }
      continue;
    }
    var remapped = 0.0;
    if (params[base + 1u] == 0.0) {
      let inputMinimum = params[base + 2u];
      if (params[base + 7u] != 0.0) {
        remapped = select(0.0, 1.0, value >= inputMinimum);
      } else {
        remapped = clamp((value - inputMinimum) * params[base + 3u], 0.0, 1.0);
      }
      if (params[base + 4u] != 0.0) {
        remapped = 1.0 - remapped;
      }
    } else {
      ${m?`let breakCountValue = params[base + 5u];
      let breakCount = min(u32(clamp(select(0.0, breakCountValue, isFiniteValue(breakCountValue)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
      let classIndex = countBreaksBelow(layer * MAXIMUM_BREAK_COUNT, breakCount, value, params[base + 6u] != 0.0);
      remapped = remapValues[remapValuesOffset + layer * (MAXIMUM_BREAK_COUNT + 1u) + classIndex];`:`remapped = getNaN();`}
    }
    if (isNaNValue(remapped)) {
      // Restricted class: the cell is excluded whatever the nodata policy.
      isDefined = false;
    }
    sum = sum + weight * remapped;
    weightSum = weightSum + abs(weight);
    usedCount = usedCount + 1u;
  }
  isDefined = isDefined && usedCount > 0u;
  var score = sum;
  if (normalizeWeights) {
    isDefined = isDefined && weightSum > 0.0;
    score = sum / weightSum;
  }
  score = select(getNaN(), score, isDefined && !isNaNValue(score));
  let isScoreDefined = !isNaNValue(score);
  scoreOut[scoreOutOffset + index] = score;
  ${i.outputValidity?`validityOut[validityOutOffset + index] = select(0u, 1u, isScoreDefined);`:``}
  ${p?`if (isScoreDefined) {
    let key = getOrderedKey(score);
    minimumKey = key;
    maximumKey = key;
  }
  }
  // Combine the workgroup's keys in a tree; one pair of global atomics per workgroup, not per cell.
  tileMinimum[localInvocationIndex] = minimumKey;
  tileMaximum[localInvocationIndex] = maximumKey;
  workgroupBarrier();
  for (var stride = ${j/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      tileMinimum[localInvocationIndex] = min(tileMinimum[localInvocationIndex], tileMinimum[localInvocationIndex + stride]);
      tileMaximum[localInvocationIndex] = max(tileMaximum[localInvocationIndex], tileMaximum[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && tileMinimum[0] <= tileMaximum[0]) {
    atomicMin(&rangeKeys[rangeKeysOffset], tileMinimum[0]);
    atomicMax(&rangeKeys[rangeKeysOffset + 1u], tileMaximum[0]);
  }`:``}`})),p&&i.scoreRange&&u.push(c(e,{id:`${t}-decode-range`,operation:A,variant:`decode-range`,bindings:[{name:`keys`,view:p,type:`u32`,access:`read`},{name:`rangeOut`,view:i.scoreRange,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${d}
${f}`,body:`let minimumKey = keys[keysOffset];
  let maximumKey = keys[keysOffset + 1u];
  let isEmpty = minimumKey > maximumKey;
  rangeOut[rangeOutOffset] = select(decodeOrderedKey(minimumKey), getNaN(), isEmpty);
  rangeOut[rangeOutOffset + 1u] = select(decodeOrderedKey(maximumKey), getNaN(), isEmpty);`})),u}},M=`GPURasterCellStatistics`,N=64,xe=class{id;props;constructor(e){this.id=e.id??`raster-cell-statistics`,this.props=e;let{id:t}=this,{layerCount:n,cellCount:r,output:i}=e;if(_(t,`layerCount`,n,1,N),_(t,`cellCount`,r,0,4294967295),n*r>4294967295)throw Error(`${t} layerCount * cellCount must fit in uint32`);y(t,e.noDataValue),x(t,`stack`,e.stack,`float32`,n*r),x(t,`parameters`,e.parameters,`float32`,4);let a=P(i);if(a.length===0)throw Error(`${t} needs at least one output`);for(let[e,n]of a)x(t,`output.${e}`,n,e===`counts`||e===`variety`?`uint32`:`float32`,r);b(t,a.map(([,e])=>e),[e.stack,e.parameters])}getCommandNodes(e){let{id:t,props:n}=this,{output:r,layerCount:i,cellCount:a}=n;g(t,e,[n.stack,n.parameters,...P(r).map(([,e])=>e)]);let o=[{name:`stack`,view:n.stack,type:`f32`,access:`read`},{name:`params`,view:n.parameters,type:`f32`,access:`read`}],s=(e,t,n=`f32`)=>t?[{name:e,view:t,type:n,access:`read_write`}]:[],l=`const LAYER_COUNT: u32 = ${i}u;
const CELL_COUNT: u32 = ${a}u;
${v(n.noDataValue)}
fn readLayer(layer: u32, cell: u32, value: ptr<function, f32>) -> bool {
  let sample = stack[stackOffset + layer * CELL_COUNT + cell];
  *value = sample;
  return !isNoDataValue(sample);
}`,u=`let propagateNoData = params[paramsOffset] != 0.0;
  let minimumCountValue = params[paramsOffset + 1u];
  let minimumCount = max(u32(clamp(select(1.0, minimumCountValue, isFiniteValue(minimumCountValue)), 0.0, 4294967040.0)), 1u);
  var count = 0u;`,d=`let isDefined = count >= minimumCount && (!propagateNoData || count == LAYER_COUNT);
  let nan = getNaN();`,f=e=>`for (var layer = 0u; layer < LAYER_COUNT; layer++) {
    var value = 0.0;
    if (!readLayer(layer, index, &value)) {
      continue;
    }
    ${e}
  }`,p=[];return(r.sum||r.mean||r.standardDeviation||r.counts)&&p.push(c(e,{id:`${t}-moments`,operation:M,variant:`moments`,bindings:[...o,...s(`sumOut`,r.sum),...s(`meanOut`,r.mean),...s(`deviationOut`,r.standardDeviation),...s(`countOut`,r.counts,`u32`)],invocationCount:a,declarations:l,body:`${u}
  var sum = 0.0;
  ${f(`sum = sum + value;
    count += 1u;`)}
  ${d}
  let mean = sum / f32(max(count, 1u));
  ${r.sum?`sumOut[sumOutOffset + index] = select(nan, sum, isDefined);`:``}
  ${r.mean?`meanOut[meanOutOffset + index] = select(nan, mean, isDefined);`:``}
  ${r.counts?`countOut[countOutOffset + index] = count;`:``}
  ${r.standardDeviation?`var squaredDeviationSum = 0.0;
  ${f(`let deviation = value - mean;
    squaredDeviationSum = squaredDeviationSum + deviation * deviation;`)}
  deviationOut[deviationOutOffset + index] = select(nan, sqrt(squaredDeviationSum / f32(max(count, 1u))), isDefined);`:``}`})),(r.minimum||r.maximum||r.range)&&p.push(c(e,{id:`${t}-extremes`,operation:M,variant:`extremes`,bindings:[...o,...s(`minimumOut`,r.minimum),...s(`maximumOut`,r.maximum),...s(`rangeOut`,r.range)],invocationCount:a,declarations:l,body:`${u}
  var minimum = 0.0;
  var maximum = 0.0;
  ${f(`if (count == 0u) {
      minimum = value;
      maximum = value;
    } else {
      minimum = min(minimum, value);
      maximum = max(maximum, value);
    }
    count += 1u;`)}
  ${d}
  ${r.minimum?`minimumOut[minimumOutOffset + index] = select(nan, minimum, isDefined);`:``}
  ${r.maximum?`maximumOut[maximumOutOffset + index] = select(nan, maximum, isDefined);`:``}
  ${r.range?`rangeOut[rangeOutOffset + index] = select(nan, maximum - minimum, isDefined);`:``}`})),(r.majority||r.minority||r.variety)&&p.push(c(e,{id:`${t}-frequencies`,operation:M,variant:`frequencies`,bindings:[...o,...s(`majorityOut`,r.majority),...s(`minorityOut`,r.minority),...s(`varietyOut`,r.variety,`u32`)],invocationCount:a,workgroupSize:64,declarations:l,body:`${u}
  var gathered: array<f32, ${i}>;
  ${f(`gathered[count] = value;
    count += 1u;`)}
  ${d}
  var majority = 0.0;
  var majorityCount = 0u;
  var minority = 0.0;
  var minorityCount = 0xffffffffu;
  var variety = 0u;
  for (var position = 0u; position < count; position++) {
    let value = gathered[position];
    var frequency = 0u;
    var isFirst = true;
    for (var other = 0u; other < count; other++) {
      if (gathered[other] == value) {
        frequency += 1u;
        if (other < position) {
          isFirst = false;
        }
      }
    }
    if (!isFirst) {
      continue;
    }
    variety += 1u;
    if (frequency > majorityCount || (frequency == majorityCount && value < majority)) {
      majority = value;
      majorityCount = frequency;
    }
    if (frequency < minorityCount || (frequency == minorityCount && value < minority)) {
      minority = value;
      minorityCount = frequency;
    }
  }
  ${r.majority?`majorityOut[majorityOutOffset + index] = select(nan, majority, isDefined);`:``}
  ${r.minority?`minorityOut[minorityOutOffset + index] = select(nan, minority, isDefined);`:``}
  ${r.variety?`varietyOut[varietyOutOffset + index] = select(0u, variety, isDefined);`:``}`})),p}};function P(e){return[`minimum`,`maximum`,`range`,`sum`,`mean`,`standardDeviation`,`majority`,`minority`,`variety`,`counts`].flatMap(t=>{let n=e[t];return n?[[t,n]]:[]})}var F={nearest:0,bilinear:1,bicubic:2};function Se(e,t,n,r,i){m(e,i,12),h(e,n);let{width:a,height:o}=n,[s,c,l,u,d,f]=n.affine;if(c!==0||u!==0)throw Error(`${e} needs an axis-aligned grid`);if(!(s>0)||!(Math.abs(d)>0))throw Error(`${e} grid cell size must be finite and positive`);let g=n.pixelInterpretation===`point`?-.5:0,_=l+s*g,v=f+d*g,y=_,b=_+s*a,x=v+d*o,S=Math.min(v,x),ee=Math.max(v,x),te=t.method??`bilinear`;if(!(te in F))throw Error(`${e} method ${te} is not supported`);let C=t.noDataPolicy??`strict`;if(C!==`strict`&&C!==`renormalize`)throw Error(`${e} noDataPolicy ${C} is not supported`);let[ne,w]=p(e,t.localOrigin);return i.fill(0,0,12),i[0]=y-ne,i[1]=S-w,i[2]=b-ne,i[3]=ee-w,i[4]=Math.fround(s),i[5]=Math.fround(Math.abs(d)),i[6]=1/s,i[7]=1/d,i[8]=F[te],i[9]=C===`renormalize`?1:0,i[10]=r,i[11]=v-w,i}function Ce(e,t,n=new Float32Array(12)){return Se(`getGPURasterSamplingParameterValues`,e,t,0,n)}function I(t,r,i,a,o){if(!i)throw Error(`${t} needs ${r}`);if(i instanceof n)throw Error(`${t} ${r} must be a single packed view, not a chunked vector`);if(e(i,[a],`${t} ${r}`),i.length<o)throw Error(`${t} ${r} must hold at least ${o} rows`)}function L(e,t,n){let r=h(e,t);if(r>2147483647)throw Error(`${e} grid width * height must not exceed 2^31 - 1`);if(n!==void 0&&!Number.isFinite(n))throw Error(`${e} noDataValue must be finite (NaN cells are always nodata)`);return r}function R(e){let{width:t,height:n,noDataValue:r,hasValidity:i}=e;return`
const RASTER_WIDTH: i32 = ${t};
const RASTER_HEIGHT: i32 = ${n};
${d}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
${u}

// Cell value with every nodata convention folded into NaN.
fn fetchCell(column: i32, row: i32) -> f32 {
  let cell = u32(row * RASTER_WIDTH + column);
  var value = raster[rasterOffset + cell];
  ${i?`if (validity[validityOffset + cell] == 0u) { value = getNaN(); }`:``}
  ${r===void 0?``:`if (value == ${s(r)}) { value = getNaN(); }`}
  return value;
}
fn fetchClamped(column: i32, row: i32) -> f32 {
  return fetchCell(clamp(column, 0, RASTER_WIDTH - 1), clamp(row, 0, RASTER_HEIGHT - 1));
}

fn getCubicWeights(t: f32) -> vec4<f32> {
  return vec4<f32>(
    t * (-0.5 + t * (1.0 - 0.5 * t)),
    1.0 + t * t * (-2.5 + 1.5 * t),
    t * (0.5 + t * (2.0 - 1.5 * t)),
    t * t * (-0.5 + 0.5 * t)
  );
}

// A sample participates only when both of its axis weights are nonzero.
fn sampleBilinear(iu: i32, iv: i32, fx: f32, fy: f32, renormalize: bool) -> f32 {
  var wx = array<f32, 2>(1.0 - fx, fx);
  var wy = array<f32, 2>(1.0 - fy, fy);
  var sum = 0.0;
  var weightSum = 0.0;
  var hasNoData = false;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      if (wx[i] != 0.0 && wy[j] != 0.0) {
        let value = fetchClamped(iu + i, iv + j);
        let weight = wx[i] * wy[j];
        if (isNaNValue(value)) {
          hasNoData = true;
        } else {
          sum += weight * value;
          weightSum += weight;
        }
      }
    }
  }
  if (!hasNoData) {
    return sum;
  }
  if (renormalize && weightSum > 0.0) {
    return sum / weightSum;
  }
  return getNaN();
}

fn sampleBicubic(iu: i32, iv: i32, fx: f32, fy: f32, renormalize: bool) -> f32 {
  let cx = getCubicWeights(fx);
  let cy = getCubicWeights(fy);
  var wx = array<f32, 4>(cx.x, cx.y, cx.z, cx.w);
  var wy = array<f32, 4>(cy.x, cy.y, cy.z, cy.w);
  var cells = array<f32, 16>();
  var hasNoData = false;
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      if (wx[i] != 0.0 && wy[j] != 0.0) {
        let value = fetchClamped(iu - 1 + i, iv - 1 + j);
        cells[j * 4 + i] = value;
        if (isNaNValue(value)) {
          hasNoData = true;
        }
      }
    }
  }
  if (hasNoData) {
    return sampleBilinear(iu, iv, fx, fy, renormalize);
  }
  var result = 0.0;
  for (var j = 0; j < 4; j++) {
    if (wy[j] != 0.0) {
      var row = 0.0;
      for (var i = 0; i < 4; i++) {
        if (wx[i] != 0.0) {
          row += wx[i] * cells[j * 4 + i];
        }
      }
      result += wy[j] * row;
    }
  }
  return result;
}

fn sampleRaster(x: f32, y: f32) -> f32 {
  if (isNaNValue(x) || isNaNValue(y)) {
    return getNaN();
  }
  let minX = params[paramsOffset];
  let minY = params[paramsOffset + 1u];
  if (x < minX || x > params[paramsOffset + 2u] || y < minY || y > params[paramsOffset + 3u]) {
    return getNaN();
  }
  let method = u32(params[paramsOffset + 8u]);
  let renormalize = params[paramsOffset + 9u] != 0.0;
  // The row coordinate counts from the outer edge of row 0 with a signed step, so it is correct
  // for rows that advance toward larger or smaller y.
  let tx = (x - minX) * params[paramsOffset + 6u];
  let ty = (y - params[paramsOffset + 11u]) * params[paramsOffset + 7u];
  if (method == 0u) {
    return fetchClamped(i32(floor(tx)), i32(floor(ty)));
  }
  let u = tx - 0.5;
  let v = ty - 0.5;
  let u0 = floor(u);
  let v0 = floor(v);
  if (method == 1u) {
    return sampleBilinear(i32(u0), i32(v0), u - u0, v - v0, renormalize);
  }
  return sampleBicubic(i32(u0), i32(v0), u - u0, v - v0, renormalize);
}`}var z=`GPURasterSampling`,we=class{id;props;constructor(e){this.id=e.id??`raster-sampling`,this.props=e;let{id:t}=this,{output:n}=e,r=L(t,e.grid,e.noDataValue);if(I(t,`values`,e.values,`float32`,r),e.validity&&I(t,`validity`,e.validity,`uint32`,r),I(t,`positions`,e.positions,`float32x2`,0),e.pointCount&&I(t,`pointCount`,e.pointCount,`uint32`,1),I(t,`parameters`,e.parameters,`float32`,12),!n?.values)throw Error(`${t} needs output.values`);let i=e.positions.length;I(t,`output.values`,n.values,`float32`,i),n.outputValidity&&I(t,`output.outputValidity`,n.outputValidity,`uint32`,i);let a=[n.values,n.outputValidity].filter(e=>e!==void 0),o=[e.values,e.validity,e.positions,e.pointCount,e.parameters];if(new Set(a.map(e=>e.buffer)).size!==a.length)throw Error(`${t} outputs must not share buffers`);let s=new Set(o.filter(e=>e!==void 0).map(e=>e.buffer));if(a.some(e=>s.has(e.buffer)))throw Error(`${t} outputs must not share buffers with inputs`)}getCommandNodes(e){let{id:t,props:n}=this,{output:i}=n;r(t,e,[n.values,n.validity,n.positions,n.pointCount,n.parameters,i.values,i.outputValidity]);let a=[{name:`raster`,view:n.values,type:`f32`,access:`read`}];return n.validity&&a.push({name:`validity`,view:n.validity,type:`u32`,access:`read`}),a.push({name:`positions`,view:n.positions,type:`f32`,access:`read`}),n.pointCount&&a.push({name:`pointCount`,view:n.pointCount,type:`u32`,access:`read`}),a.push({name:`params`,view:n.parameters,type:`f32`,access:`read`},{name:`valuesOut`,view:i.values,type:`f32`,access:`read_write`}),i.outputValidity&&a.push({name:`validityOut`,view:i.outputValidity,type:`u32`,access:`read_write`}),[c(e,{id:`${t}-sample`,operation:z,variant:`sample`,bindings:a,invocationCount:n.positions.length,declarations:`${R({width:n.grid.width,height:n.grid.height,noDataValue:n.noDataValue,hasValidity:!!n.validity})}
const POINT_CAPACITY: u32 = ${n.positions.length}u;`,body:`let activeCount = ${n.pointCount?`min(pointCount[pointCountOffset], POINT_CAPACITY)`:`POINT_CAPACITY`};
  var result = getNaN();
  if (index < activeCount) {
    result = sampleRaster(
      positions[positionsOffset + 2u * index],
      positions[positionsOffset + 2u * index + 1u]
    );
  }
  valuesOut[valuesOutOffset + index] = result;
  ${i.outputValidity?`validityOut[validityOutOffset + index] = select(1u, 0u, isNaNValue(result));`:``}`})]}},B=5,Te=[`burn severity`,`slope`,`land cover`,`proximity`,`greenness`],V=[15,25,35,45,55,65,75,85],H=V.length,U=8,W=200,Ee=350,De=[587,538];async function Oe(e){let n=e.datasets.get(`dixie-fire`),{device:r}=e;e.setStatus(`Loading Sentinel-2 bands and the DEM`);let o=await _e(n,e.signal);e.signal.throwIfAborted();let s=he(n),{width:c,height:l,cellCount:u,frame:d}=o,f=new a(r,`suit`),p=ae({width:c,height:l,cellSize:d.cellSize,origin:[-d.halfWidth,d.halfHeight]}),m=new Float32Array(u);for(let e=0;e<l;e++)for(let t=0;t<c;t++)m[e*c+t]=d.cellSize*Math.hypot(t-De[0],e-De[1]);let h=new Uint32Array(u);for(let e=0;e<u;e++)h[e]=Math.max(0,Math.round(o.worldCoverClasses[e]/10)-1);let g=se(20210713),_=Math.floor(2*d.halfWidth/W)-1,v=_*_,y=new Float32Array(v*2),b=new Float32Array(v*2);for(let e=0;e<_;e++)for(let t=0;t<_;t++){let n=e*_+t,r=-d.halfWidth+W*(t+1)+(g()-.5)*W*.8,i=-d.halfHeight+W*(e+1)+(g()-.5)*W*.8;y[n*2]=r,y[n*2+1]=i;let[a,o]=d.localToMeters(r,i);b[n*2]=a,b[n*2+1]=o}let x=(e,t=u)=>f.createBuffer(e,t*4),S={redAfter:f.createBuffer(`red-after`,o.redAfter),nirBefore:f.createBuffer(`nir-before`,o.nirBefore),swirBefore:f.createBuffer(`swir-before`,o.swirBefore),nirAfter:f.createBuffer(`nir-after`,o.nirAfter),swirAfter:f.createBuffer(`swir-after`,o.swirAfter),elevation:f.createBuffer(`elevation`,o.elevation),cover:f.createBuffer(`cover-code`,o.worldCover),distance:f.createBuffer(`distance`,m)},O={nbrBefore:x(`nbr-before`),nbrAfter:x(`nbr-after`),dnbr:x(`dnbr`),ndviAfter:x(`ndvi-after`),slope:x(`slope`),burnedFlag:x(`burned-flag`)},ve=x(`raw-stack`,u*B),A=x(`normalized-stack`,u*B),j=Array.from({length:B},(e,t)=>x(`normalized-${t}`)),M={mean:x(`stat-mean`),spread:x(`stat-spread`),minimum:x(`stat-minimum`),maximum:x(`stat-maximum`),range:x(`stat-range`),count:x(`stat-count`)},N=x(`score`),P=f.createBuffer(`score-range`,8),F=f.createParameterBuffer(`site-count`,`uint32`,1),Se=f.createBuffer(`site-sampling-positions`,y),I=f.createBuffer(`site-display-positions`,b),L=x(`site-values`,v),R=f.createBuffer(`zones`,h),z={sorted:{score:ke(f,`sorted-score`),burned:ke(f,`sorted-burned`)},atomic:{score:ke(f,`atomic-score`)}},Oe=Array.from({length:4},(e,t)=>f.createParameterBuffer(`normalizer-${t}`,`float32`,8)),Ae=f.createParameterBuffer(`cover-breaks`,`float32`,H),je=f.createParameterBuffer(`cover-values`,`float32`,H+1),Me=f.createParameterBuffer(`cover-reclassify`,`float32`,4),Ne=f.createParameterBuffer(`statistics-parameters`,`float32`,4),Pe=f.createParameterBuffer(`overlay-parameters`,`float32`,k(B)),Fe=f.createParameterBuffer(`overlay-breaks`,`float32`,B*H),Ie=f.createParameterBuffer(`overlay-values`,`float32`,B*(H+1)),Le=f.createParameterBuffer(`sampling-parameters`,`float32`,12),Re=f.createParameterBuffer(`burned-parameters`,`float32`,8),ze=f.createParameterBuffer(`terrain-settings`,`float32`,8),Be=(e,t)=>({id:e,format:`float32`,storage:{kind:`buffer`,values:t}}),G=new t(r,{id:`suit-criteria`});{let e=E(G),t=(t,n)=>e(t,n,`float32`,u),n=f.createParameterBuffer(`index-parameters`,`float32`,8);n.write(w({operation:`normalizedDifference`,scaleA:1e-4,scaleB:1e-4}));let r=f.createParameterBuffer(`difference-parameters`,`float32`,8);r.write(w({operation:`subtract`}));let i=n.importToGraph(G);for(let[e,n,r,a]of[[`nbr-before`,S.nirBefore,S.swirBefore,O.nbrBefore],[`nbr-after`,S.nirAfter,S.swirAfter,O.nbrAfter],[`ndvi-after`,S.nirAfter,S.redAfter,O.ndviAfter]])G.add(new ie({id:e,cellCount:u,a:t(`${e}-a`,n),b:t(`${e}-b`,r),noDataValue:0,parameters:i,output:{values:t(`${e}-out`,a)}}));G.add(new ie({id:`dnbr`,cellCount:u,a:t(`dnbr-a`,O.nbrBefore),b:t(`dnbr-b`,O.nbrAfter),parameters:r.importToGraph(G),output:{values:t(`dnbr-out`,O.dnbr)}})),Re.write(ee({comparison:`>`,threshold:.27,constantA:1,constantB:0})),G.add(new C({id:`burned`,cellCount:u,conditionValues:t(`burned-condition`,O.dnbr),parameters:Re.importToGraph(G),output:{values:t(`burned-out`,O.burnedFlag)}})),ze.write(de({},p)),G.add(new fe({id:`slope`,grid:p,elevation:Be(`elevation`,t(`elevation`,S.elevation)),parameters:ze.importToGraph(G),output:{slope:t(`slope-out`,O.slope)},cellSizeMode:`uniform`,rowDirection:`south`}))}let Ve=f.track(await G.compileAsync()),K=new t(r,{id:`suit-normalize`});{let e=E(K),t=(t,n)=>e(t,n,`float32`,u),n=[O.dnbr,O.slope,null,S.distance,O.ndviAfter],r=0;for(let e=0;e<B;e++){let i=n[e];i&&K.add(new ie({id:`normalize-${Te[e].replace(` `,`-`)}`,cellCount:u,a:t(`source-${e}`,i),parameters:Oe[r++].importToGraph(K),output:{values:t(`normalized-${e}`,j[e])}}))}K.add(new ne({id:`cover-scores`,values:t(`cover`,S.cover),breaks:Ae.importToGraph(K),classValues:je.importToGraph(K),parameters:Me.importToGraph(K),output:{reclassified:t(`normalized-2`,j[2])}}))}let He=f.track(await K.compileAsync()),q=new t(r,{id:`suit-statistics`});{let e=E(q),t=(t,n)=>e(t,n,`float32`,u);q.add(new xe({id:`criteria-statistics`,stack:e(`normalized-stack`,A,`float32`,u*B),layerCount:B,cellCount:u,parameters:Ne.importToGraph(q),output:{mean:t(`mean`,M.mean),standardDeviation:t(`spread`,M.spread),minimum:t(`minimum`,M.minimum),maximum:t(`maximum`,M.maximum),range:t(`range`,M.range),counts:e(`count`,M.count,`uint32`,u)}}))}let Ue=f.track(await q.compileAsync()),J=new t(r,{id:`suit-overlay`});{let e=E(J),t=(t,n,r=u)=>e(t,n,`float32`,r);J.add(new be({id:`overlay`,stack:t(`raw-stack`,ve,u*B),layerCount:B,cellCount:u,parameters:Pe.importToGraph(J),remapBreaks:Fe.importToGraph(J),remapValues:Ie.importToGraph(J),maximumBreakCount:H,output:{score:t(`score`,N),scoreRange:t(`score-range`,P,2)}})),J.add(new we({id:`sites`,grid:p,values:t(`sampled-score`,N),positions:e(`site-positions`,Se,`float32x2`,v),pointCount:F.importToGraph(J),parameters:Le.importToGraph(J),output:{values:t(`site-values`,L,v)}}))}let We=f.track(await J.compileAsync()),Ge=async(e,n,i)=>{let a=new t(r,{id:e}),o=E(a);for(let e of i){let t=(t,n,r)=>o(`${e.name}-${t}`,n,r,U);a.add(new oe({id:`zonal-${n}-${e.name}`,grid:p,zones:o(`zones`,R,`uint32`,u),values:Be(e.name,o(`${e.name}-values`,e.values,`float32`,u)),zoneCapacity:U,sumOrder:n,output:{cellCounts:t(`cell-counts`,e.outputs.cellCounts,`uint32`),valueCounts:t(`value-counts`,e.outputs.valueCounts,`uint32`),sums:t(`sums`,e.outputs.sums,`float32`),means:t(`means`,e.outputs.means,`float32`),minimums:t(`minimums`,e.outputs.minimums,`float32`),maximums:t(`maximums`,e.outputs.maximums,`float32`),status:{overflow:o(`${e.name}-overflow`,e.outputs.overflow,`uint32`,1)}}}))}return f.track(await a.compileAsync())},Y={sorted:await Ge(`suit-zonal-sorted`,`sorted`,[{name:`score`,values:N,outputs:z.sorted.score},{name:`burned`,values:O.burnedFlag,outputs:z.sorted.burned}]),atomic:await Ge(`suit-zonal-atomic`,`atomic`,[{name:`score`,values:N,outputs:z.atomic.score}])};{let e=r.createCommandEncoder({id:`suit-criteria-encoder`});Ve.encode(e,{parameters:void 0}),[O.dnbr,O.slope,S.cover,S.distance,O.ndviAfter].forEach((t,n)=>{e.copyBufferToBuffer({sourceBuffer:t,destinationBuffer:ve,destinationOffset:n*u*4,size:u*4})}),r.submit(e.finish())}let X=!1,Ke=!0,qe=performance.now(),Je=!0,Ye=!0,Z=null,Xe=null,Q=null,Ze=!1,Qe=()=>{Ke=!0,qe=performance.now()},$e=e=>{let t=e.restrictBuiltWater?NaN:0;return[e.coverTree,e.coverShrub,e.coverGrass,e.coverCrop,e.restrictBuiltWater?t:e.coverCrop*.5,e.coverBare,0,t,e.coverGrass]},et=(e,t)=>{let n=Math.min(e[0],e[1]),r=Math.max(e[0],e[1]),i=Math.max(r-n,1e-6),a=1/i,o=-n/i;return t?{scaleA:-a,offsetA:1-o}:{scaleA:a,offsetA:o}},tt=e=>({severity:$(e.severityRange),slope:$(e.slopeRange),proximity:$(e.proximityRange),greenness:$(e.greennessRange)}),nt=()=>{let t=e.options,n=tt(t);[et(n.severity,!1),et(n.slope,!1),et(n.proximity,!0),et(n.greenness,!0)].forEach((e,t)=>{Oe[t].write(w({operation:`add`,...e,constantB:0,clampMinimum:0,clampMaximum:1}))});let r=$e(t);Ae.write(Float32Array.from(V)),je.write(Float32Array.from(r)),Me.write(re({breakCount:H})),Ne.write(te({noDataPolicy:t.statisticsPolicy,minimumValidCount:t.minimumValid}));let i=[t.weightSeverity,t.weightSlope,t.weightCover,t.weightProximity,t.weightGreenness];Pe.write(ye({normalizeWeights:t.normalizeWeights,noDataPolicy:t.noDataPolicy,layers:[{weight:i[0],inputMinimum:n.severity[0],inputMaximum:n.severity[1]},{weight:i[1],inputMinimum:n.slope[0],inputMaximum:n.slope[1]},{weight:i[2],mode:`table`,breakCount:H,closed:`left`},{weight:i[3],inputMinimum:n.proximity[0],inputMaximum:n.proximity[1],invert:!0},{weight:i[4],inputMinimum:n.greenness[0],inputMaximum:n.greenness[1],invert:!0}]}));let a=new Float32Array(B*H),o=new Float32Array(B*(H+1));a.set(V,2*H),o.set(r,2*(H+1)),Fe.write(a),Ie.write(o),Le.write(Ce({method:t.sampleMethod,noDataPolicy:t.sampleNoData},p)),F.write(Uint32Array.of(Math.min(t.siteCount,v)))},rt=null,it=()=>{if(rt===e.options.ignoreTrees)return;if(rt=e.options.ignoreTrees,!e.options.ignoreTrees){R.write(h);return}let t=new Uint32Array(h);for(let e=0;e<t.length;e++)t[e]===0&&(t[e]=4294967295);R.write(t)},at=new ue(f,`suit-criteria`,[{buffer:O.dnbr,size:u*4},{buffer:O.slope,size:u*4},{buffer:O.ndviAfter,size:u*4}],e=>{X||(Z={dnbr:new Float32Array(e,0,u),slope:new Float32Array(e,u*4,u),ndviAfter:new Float32Array(e,u*8,u)})}),ot=new ue(f,`suit-score`,[{buffer:N,size:u*4}],e=>{X||(Xe=new Float32Array(e))}),st=e=>[{buffer:e.cellCounts,size:U*4},{buffer:e.valueCounts,size:U*4},{buffer:e.means,size:U*4},{buffer:e.minimums,size:U*4},{buffer:e.maximums,size:U*4},{buffer:e.overflow,size:4}],ct=new ue(f,`suit-summary`,[{buffer:P,size:8},{buffer:L,size:v*4},...st(z.sorted.score),...st(z.sorted.burned),...st(z.atomic.score)],e=>{if(X)return;let t=0,n=(n,r)=>{let i=new n(e,t,r);return t+=r*4,i},r=n(Float32Array,2);Q=n(Float32Array,v);let i=()=>({cellCounts:n(Uint32Array,U),valueCounts:n(Uint32Array,U),means:n(Float32Array,U),minimums:n(Float32Array,U),maximums:n(Float32Array,U),overflow:n(Uint32Array,1)[0]}),a=i(),o=i(),s=i();ut([r[0],r[1]],a,o,s)}),lt=[{id:`zoneTree`,zone:0,reference:`Tree cover`},{id:`zoneGrass`,zone:2,reference:`Grassland`},{id:`zoneCrop`,zone:3,reference:`Cropland`},{id:`zoneBuilt`,zone:4,reference:`Built-up`},{id:`zoneBare`,zone:5,reference:`Bare / sparse vegetation`},{id:`zoneWater`,zone:7,reference:`Permanent water bodies`}];function ut(t,n,r,i){let a=e.options.zonalOrder===`atomic`,o=a?i:n;if(a){let t=0;for(let e=0;e<U;e++){let r=Math.abs(i.means[e]-n.means[e]);Number.isFinite(r)&&(t=Math.max(t,r))}e.setReadout(`zonalSums`,`atomic vs sorted, largest difference of a zone mean: ${t.toExponential(1)}`)}else e.setReadout(`zonalSums`,`sorted: bitwise reproducible between runs`);e.setLegendExtent(`score`,t),e.setReadout(`scoreRange`,Number.isFinite(t[0])?`${T(t[0],3)} to ${T(t[1],3)}`:`no defined cells`);for(let t of lt){let n=o.cellCounts[t.zone];if(n===0){e.setReadout(t.id,`no cells in this window`);continue}let i=ge(n/u,n/u<.01?2:1),a=o.means[t.zone],c=r.means[t.zone],l=s.burnedByCover[t.reference??``],d=Number.isFinite(a)?`score ${T(a,2)} (${T(o.minimums[t.zone],2)} to ${T(o.maximums[t.zone],2)})`:`score restricted`,f=`burned ${ge(c,1)}${l===void 0?``:` (CPU ${ge(l,1)})`}`;e.setReadout(t.id,`${i} of area · ${d} · ${f}`)}e.setReadout(`zonalOverflow`,o.overflow?`yes: a zone id at or above the capacity of ${U} (herbaceous wetland, 1 pixel)`:`no`),Q&&ft()}function $(e){return[Math.min(e[0],e[1]),Math.max(e[0],e[1])]}function dt(e,t){if(!Z)return NaN;let n=tt(t),r=[Z.dnbr[e],Z.slope[e],o.worldCover[e],m[e],Z.ndviAfter[e]],i=[t.weightSeverity,t.weightSlope,t.weightCover,t.weightProximity,t.weightGreenness],a=$e(t),s=0,c=0;for(let e=0;e<B;e++){let o=r[e],l;if(e===2){let e=0;for(let t of V)t<=o&&e++;if(l=a[e],Number.isNaN(l))return NaN}else{if(Number.isNaN(o)){if(t.noDataPolicy===`propagate`)return NaN;continue}let[r,i]=[n.severity,n.slope,null,n.proximity,n.greenness][e];l=Math.min(1,Math.max(0,(o-r)/(i-r))),(e===3||e===4)&&(l=1-l)}s+=i[e]*l,c+=Math.abs(i[e])}return t.normalizeWeights?c>0?s/c:NaN:s}function ft(){if(!Q)return;let t=e.options,n=Math.min(t.siteCount,v),r=0,a=0,o=0;for(let e=0;e<n;e++){let n=Q[e];Number.isFinite(n)&&(r++,o+=n,n>t.siteThreshold&&a++)}if(e.setReadout(`sites`,`${i(a)} of ${i(r)} sites above ${t.siteThreshold.toFixed(2)} · mean ${T(r?o/r:NaN,3)} · ${i(n-r)} without data`),t.sampleMethod===`nearest`&&Z){let r=0,a=0;for(let e=0;e<n;e++){let n=y[e*2],i=y[e*2+1],o=Math.floor((n+d.halfWidth)/d.cellSize),s=Math.floor((d.halfHeight-i)/d.cellSize);if(o<0||s<0||o>=c||s>=l)continue;let u=dt(s*c+o,t),f=Q[e];if(Number.isNaN(u)&&Number.isNaN(f)){a++;continue}if(Number.isNaN(u)!==Number.isNaN(f)){r=Math.max(r,1),a++;continue}r=Math.max(r,Math.abs(u-f)),a++}e.setReadout(`parity`,`${i(a)} sites: GPU sample vs CPU recomputation, largest difference ${r.toExponential(1)}`)}else e.setReadout(`parity`,`shown for the nearest method (exact cell lookup)`)}async function pt(){if(!(Ze||X)){Ze=!0;try{let t={parameters:void 0,completionBuffer:z.sorted.score.overflow},n=await le(r,Y.sorted,t),i=await le(r,Y.atomic,t);if(X)return;e.setReadout(`zonalTiming`,`sorted (2 columns) ${n.milliseconds.toFixed(2)} ms · atomic (1 column) ${i.milliseconds.toFixed(2)} ms · ${n.method===`gpu-timestamps`?`GPU timestamps`:`wall clock`}`)}catch{}finally{Ze=!1}}}e.setReadout(`sites`,`sampling...`),e.setReadout(`zonalTiming`,`press the button`),e.setStatus(``);let mt=e=>[`c-severity`,`c-slope`,`c-cover`,`c-proximity`,`c-greenness`].indexOf(e);return{getCompiledGraphs:()=>[Ve,He,Ue,We,Y.sorted,Y.atomic],setOption(){Qe(),Je=!0,Ye=!0,e.requestLayers()},onAction(e){e===`measure`&&pt()},encode(t,n){let r=e.options;(Ke||n.frameIndex<2)&&(it(),nt(),He.encode(t,{parameters:void 0}),j.forEach((e,n)=>{t.copyBufferToBuffer({sourceBuffer:e,destinationBuffer:A,destinationOffset:n*u*4,size:u*4})}),Ue.encode(t,{parameters:void 0}),We.encode(t,{parameters:void 0}),Y.sorted.encode(t,{parameters:void 0}),r.zonalOrder===`atomic`&&Y.atomic.encode(t,{parameters:void 0}),Ke=!1,Je=!0,Ye=!0),n.frameIndex===3&&!Z&&at.request(t),performance.now()-qe>Ee&&n.frameIndex>4&&(Je&&!ct.isPending&&(ct.request(t),Je=!1),Ye&&!ot.isPending&&(ot.request(t),Ye=!1)),at.flush(t),ct.flush(t),ot.flush(t)},getLayers(){let t=e.options,n={id:`suit-${t.layer}`,frame:d,opacity:t.opacity,noDataColor:[0,0,0,0]},r=[],i=mt(t.layer);if(t.layer===`score`)r.push(new D({...n,values:N,valueFormat:`float32`,colormap:t.ramp,extent:P,discardAtOrBelow:t.minimumScore>0?t.minimumScore:void 0}));else if(t.layer===`cover`)r.push(new D({...n,values:R,valueFormat:`uint32`,colormap:`category`,palette:me}));else if(i>=0)r.push(new D({...n,id:`suit-criterion-${i}`,values:j[i],valueFormat:`float32`,colormap:t.ramp,valueRange:[0,1]}));else{let e={meanCriteria:M.mean,spread:M.spread,weakest:M.minimum,strongest:M.maximum,range:M.range,validLayers:M.count}[t.layer];r.push(new D({...n,values:e,valueFormat:t.layer===`validLayers`?`uint32`:`float32`,colormap:t.ramp,valueRange:t.layer===`validLayers`?[0,B]:[0,t.layer===`spread`?.5:1]}))}return t.showSites&&r.push(new ce({id:`suit-sites`,coordinateOrigin:[d.origin[0],d.origin[1],0],positions:I,instanceCount:Math.min(t.siteCount,v),values:L,valueFormat:`float32`,colormap:`cividis`,valueRange:[0,1],discardAtOrBelow:t.siteThreshold,radiusPixels:t.siteSize,noDataColor:[0,0,0,0]})),r},getTooltip(t){if(!t.coordinate)return null;let[n,r]=d.lngLatToCell(t.coordinate[0],t.coordinate[1]),i=Math.floor(n),a=Math.floor(r);if(i<0||a<0||i>=c||a>=l)return null;let s=a*c+i,u=e.options,f=[],p=Xe?Xe[s]:NaN;f.push(Number.isFinite(p)?`Suitability ${T(p,3)}`:`Suitability: restricted or no data`),Z&&f.push(`dNBR ${T(Z.dnbr[s],2)} · slope ${Math.round(Z.slope[s])}° · NDVI after ${T(Z.ndviAfter[s],2)}`);let g=pe[h[s]]??`Other`;return f.push(`${g} · ${(m[s]/1e3).toFixed(1)} km from Greenville · ${Math.round(o.elevation[s]).toLocaleString(`en-US`)} m`),u.layer.startsWith(`c-`)&&Z&&f.push(`Rescaled criterion layer shown`),f.join(`
`)},onThemeChange(){e.requestLayers()},destroy(){X=!0,at.stop(),ot.stop(),ct.stop(),f.destroy()}}}function ke(e,t){let n=(n,r=U)=>e.createBuffer(`${t}-${n}`,r*4);return{cellCounts:n(`cell-counts`),valueCounts:n(`value-counts`),sums:n(`sums`),means:n(`means`),minimums:n(`minimums`),maximums:n(`maximums`),overflow:n(`overflow`,1)}}export{Oe as createSiteSuitability};