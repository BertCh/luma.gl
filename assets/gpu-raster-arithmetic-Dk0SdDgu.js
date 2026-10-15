import{n as e,s as t}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as n}from"./raster-band-utils-CA-sh8xt.js";import{a as r,i,n as a,o,r as s,s as c,t as l}from"./raster-algebra-utils-C7qGzplC.js";function u(e,t=new Float32Array(4)){if(n(`getGPURasterReclassifyParameterValues`,t,4),!Number.isSafeInteger(e.breakCount)||e.breakCount<0)throw Error(`getGPURasterReclassifyParameterValues breakCount must be a non-negative integer`);return t[0]=e.breakCount,t[1]=e.closed===`right`?1:0,t[2]=0,t[3]=0,t}var d=`GPURasterReclassify`,f=256,p=class{id;props;rowCount;maximumBreakCount;constructor(e){this.id=e.id??`raster-reclassify`,this.props=e;let{id:t}=this,{output:n}=e;if(c(t,`values`,e.values,`float32`,0),this.rowCount=e.rowCount??e.values.length,i(t,`rowCount`,this.rowCount,0,e.values.length),o(t,e.noDataValue),c(t,`validity`,e.validity,`uint32`,this.rowCount),c(t,`breaks`,e.breaks,`float32`,1),this.maximumBreakCount=e.breaks.length,c(t,`classValues`,e.classValues,`float32`,this.maximumBreakCount+1),c(t,`parameters`,e.parameters,`float32`,4),!n.classes&&!n.reclassified&&!n.classCounts)throw Error(`${t} needs at least one output`);if(n.reclassified&&!e.classValues)throw Error(`${t} output.reclassified requires classValues`);c(t,`output.classes`,n.classes,`uint32`,this.rowCount),c(t,`output.reclassified`,n.reclassified,`float32`,this.rowCount),c(t,`output.classCounts`,n.classCounts,`uint32`,this.maximumBreakCount+1),s(t,[n.classes,n.reclassified,n.classCounts],[e.values,e.validity,e.breaks,e.classValues,e.parameters])}getCommandNodes(n){let{id:i,props:o,rowCount:s,maximumBreakCount:c}=this,{output:u}=o;r(i,n,[o.values,o.validity,o.breaks,o.classValues,o.parameters,u.classes,u.reclassified,u.classCounts]);let p=[];u.classCounts&&p.push(e(n,{id:`${i}-clear-counts`,operation:d,view:u.classCounts,type:`u32`,value:`0u`,componentCount:c+1}));let m=!!u.classCounts&&(c+1)*4<=n.device.limits.maxComputeWorkgroupStorageSize,h=[{name:`source`,view:o.values,type:`f32`,access:`read`},{name:`breaks`,view:o.breaks,type:`f32`,access:`read`},{name:`params`,view:o.parameters,type:`f32`,access:`read`}];return o.validity&&h.push({name:`validity`,view:o.validity,type:`u32`,access:`read`}),o.classValues&&h.push({name:`classValues`,view:o.classValues,type:`f32`,access:`read`}),u.classes&&h.push({name:`classesOut`,view:u.classes,type:`u32`,access:`read_write`}),u.reclassified&&h.push({name:`reclassifiedOut`,view:u.reclassified,type:`f32`,access:`read_write`}),u.classCounts&&h.push({name:`countsOut`,view:u.classCounts,type:`atomic<u32>`,access:`read_write`}),p.push(t(n,{id:`${i}-classify`,operation:d,variant:`classify`,bindings:h,invocationCount:s,guardIndex:!m,workgroupSize:m?f:void 0,declarations:`const MAXIMUM_BREAK_COUNT: u32 = ${c}u;
${m?`var<workgroup> localCounts: array<atomic<u32>, ${c+1}>;`:``}
const NO_DATA_CLASS: u32 = 0xffffffffu;
${a(o.noDataValue)}
${l(`countBreaksBelow`,`breaks`)}`,body:`${m?`for (var slot = localInvocationIndex; slot <= MAXIMUM_BREAK_COUNT; slot += ${f}u) {
    atomicStore(&localCounts[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {`:``}
  let value = source[sourceOffset + index];
  var isValid = !isNoDataValue(value);
  ${o.validity?`isValid = isValid && validity[validityOffset + index] != 0u;`:``}
  let breakCountValue = params[paramsOffset];
  let breakCount = min(u32(clamp(select(0.0, breakCountValue, isFiniteValue(breakCountValue)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
  let closedRight = params[paramsOffset + 1u] != 0.0;
  var classIndex = NO_DATA_CLASS;
  if (isValid) {
    classIndex = countBreaksBelow(0u, breakCount, value, closedRight);
  }
  ${u.classes?`classesOut[classesOutOffset + index] = classIndex;`:``}
  ${u.reclassified?`reclassifiedOut[reclassifiedOutOffset + index] = select(getNaN(), classValues[classValuesOffset + min(classIndex, MAXIMUM_BREAK_COUNT)], isValid);`:``}
  ${u.classCounts?m?`if (isValid) {
    atomicAdd(&localCounts[classIndex], 1u);
  }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot <= MAXIMUM_BREAK_COUNT; slot += ${f}u) {
    let partial = atomicLoad(&localCounts[slot]);
    if (partial != 0u) {
      atomicAdd(&countsOut[countsOutOffset + slot], partial);
    }
  }`:`if (isValid) {
    atomicAdd(&countsOut[countsOutOffset + classIndex], 1u);
  }`:``}`})),p}};function m(e={},t=new Float32Array(4)){n(`getGPURasterCellStatisticsParameterValues`,t,4);let r=e.minimumValidCount??1;if(!Number.isSafeInteger(r)||r<0)throw Error(`getGPURasterCellStatisticsParameterValues minimumValidCount must be a non-negative integer`);return t[0]=e.noDataPolicy===`propagate`?1:0,t[1]=r,t[2]=0,t[3]=0,t}var h={"<":0,"<=":1,">":2,">=":3,"==":4,"!=":5,between:6};function g(e={},t=new Float32Array(8)){n(`getGPURasterConditionalParameterValues`,t,8);let r=e.comparison??`>`;if(!(r in h))throw Error(`getGPURasterConditionalParameterValues comparison ${r} is not supported`);let i=e.threshold??0;return t.fill(0,0,8),t[0]=h[r],t[1]=i,t[2]=e.upperThreshold??i,t[3]=e.constantA??1,t[4]=e.constantB??0,t}var _={add:0,subtract:1,multiply:2,divide:3,minimum:4,maximum:5,power:6,absoluteDifference:7,normalizedDifference:8,absolute:16,squareRoot:17,logarithm:18,exponential:19,floor:20,ceil:21,round:22};function v(e,t=new Float32Array(8)){n(`getGPURasterArithmeticParameterValues`,t,8);let r=_[e.operation];if(r===void 0)throw Error(`getGPURasterArithmeticParameterValues operation ${e.operation} is not supported`);return t[0]=r,t[1]=e.scaleA??1,t[2]=e.offsetA??0,t[3]=e.scaleB??1,t[4]=e.offsetB??0,t[5]=e.constantB??0,t[6]=e.clampMinimum??-1/0,t[7]=e.clampMaximum??1/0,t}var y=`GPURasterConditional`,b=class{id;props;constructor(e){this.id=e.id??`raster-conditional`,this.props=e;let{id:t}=this,{cellCount:n,output:r}=e;if(i(t,`cellCount`,n,0,4294967295),o(t,e.noDataValue),!!e.mask==!!e.conditionValues)throw Error(`${t} needs exactly one of mask and conditionValues`);if(c(t,`mask`,e.mask,`uint32`,n),c(t,`conditionValues`,e.conditionValues,`float32`,n),c(t,`a`,e.a,`float32`,n),c(t,`b`,e.b,`float32`,n),c(t,`parameters`,e.parameters,`float32`,8),!r?.values)throw Error(`${t} needs output.values`);c(t,`output.values`,r.values,`float32`,n),c(t,`output.outputMask`,r.outputMask,`uint32`,n),s(t,[r.values,r.outputMask],[e.mask,e.conditionValues,e.a,e.b,e.parameters])}getCommandNodes(e){let{id:n,props:i}=this,{output:o}=i;r(n,e,[i.mask,i.conditionValues,i.a,i.b,i.parameters,o.values,o.outputMask]);let s=[{name:`params`,view:i.parameters,type:`f32`,access:`read`}];return i.mask&&s.push({name:`conditionMask`,view:i.mask,type:`u32`,access:`read`}),i.conditionValues&&s.push({name:`conditionValues`,view:i.conditionValues,type:`f32`,access:`read`}),i.a&&s.push({name:`aValues`,view:i.a,type:`f32`,access:`read`}),i.b&&s.push({name:`bValues`,view:i.b,type:`f32`,access:`read`}),s.push({name:`valuesOut`,view:o.values,type:`f32`,access:`read_write`}),o.outputMask&&s.push({name:`maskOut`,view:o.outputMask,type:`u32`,access:`read_write`}),[t(e,{id:`${n}-select`,operation:y,variant:`select`,bindings:s,invocationCount:i.cellCount,declarations:a(i.noDataValue),body:`var conditionDefined = true;
  var condition = false;
  ${i.mask?`condition = conditionMask[conditionMaskOffset + index] != 0u;`:`let conditionValue = conditionValues[conditionValuesOffset + index];
  conditionDefined = !isNoDataValue(conditionValue);
  let comparison = u32(params[paramsOffset]);
  let threshold = params[paramsOffset + 1u];
  let upperThreshold = params[paramsOffset + 2u];
  switch comparison {
    case 0u: { condition = conditionValue < threshold; }
    case 1u: { condition = conditionValue <= threshold; }
    case 2u: { condition = conditionValue > threshold; }
    case 3u: { condition = conditionValue >= threshold; }
    case 4u: { condition = conditionValue == threshold; }
    case 5u: { condition = conditionValue != threshold; }
    default: { condition = conditionValue >= threshold && conditionValue <= upperThreshold; }
  }`}
  let aValue = ${i.a?`aValues[aValuesOffset + index]`:`params[paramsOffset + 3u]`};
  let bValue = ${i.b?`bValues[bValuesOffset + index]`:`params[paramsOffset + 4u]`};
  let chosen = select(bValue, aValue, condition);
  let isDefined = conditionDefined && !isNoDataValue(chosen);
  valuesOut[valuesOutOffset + index] = select(getNaN(), chosen, isDefined);
  ${o.outputMask?`maskOut[maskOutOffset + index] = select(0u, 1u, conditionDefined && condition);`:``}`})]}},x=`GPURasterArithmetic`,S=class{id;props;constructor(e){this.id=e.id??`raster-arithmetic`,this.props=e;let{id:t}=this,{cellCount:n,output:r}=e;if(i(t,`cellCount`,n,0,4294967295),o(t,e.noDataValue),!e.a)throw Error(`${t} needs a`);if(c(t,`a`,e.a,`float32`,n),c(t,`b`,e.b,`float32`,n),c(t,`parameters`,e.parameters,`float32`,8),!r?.values)throw Error(`${t} needs output.values`);c(t,`output.values`,r.values,`float32`,n),s(t,[r.values],[e.a,e.b,e.parameters])}getCommandNodes(e){let{id:n,props:i}=this;r(n,e,[i.a,i.b,i.parameters,i.output.values]);let o=[{name:`aValues`,view:i.a,type:`f32`,access:`read`},...i.b?[{name:`bValues`,view:i.b,type:`f32`,access:`read`}]:[],{name:`params`,view:i.parameters,type:`f32`,access:`read`},{name:`valuesOut`,view:i.output.values,type:`f32`,access:`read_write`}];return[t(e,{id:`${n}-evaluate`,operation:x,variant:`evaluate`,bindings:o,invocationCount:i.cellCount,declarations:`${a(i.noDataValue)}
fn getPower(base: f32, exponent: f32) -> f32 {
  if (base > 0.0) {
    return pow(base, exponent);
  }
  if (base == 0.0 && exponent > 0.0) {
    return 0.0;
  }
  return getNaN();
}`,body:`let operation = u32(params[paramsOffset]);
  let aRaw = aValues[aValuesOffset + index];
  let bRaw = ${i.b?`bValues[bValuesOffset + index]`:`params[paramsOffset + 5u]`};
  let isUnary = operation >= 16u;
  let isDefined = !isNoDataValue(aRaw) && (isUnary || !isNoDataValue(bRaw));
  let a = aRaw * params[paramsOffset + 1u] + params[paramsOffset + 2u];
  let b = bRaw * params[paramsOffset + 3u] + params[paramsOffset + 4u];
  let nan = getNaN();
  var result = nan;
  switch operation {
    case 0u: { result = a + b; }
    case 1u: { result = a - b; }
    case 2u: { result = a * b; }
    case 3u: { result = select(a / b, nan, b == 0.0); }
    case 4u: { result = min(a, b); }
    case 5u: { result = max(a, b); }
    case 6u: { result = getPower(a, b); }
    case 7u: { result = abs(a - b); }
    case 8u: {
      let total = a + b;
      result = select((a - b) / total, nan, total == 0.0);
    }
    case 16u: { result = abs(a); }
    case 17u: { result = select(sqrt(a), nan, a < 0.0); }
    case 18u: { result = select(log(a), nan, a <= 0.0); }
    case 19u: { result = exp(a); }
    case 20u: { result = floor(a); }
    case 21u: { result = ceil(a); }
    case 22u: { result = round(a); }
    default: { result = nan; }
  }
  if (isDefined && !isNaNValue(result)) {
    result = min(max(result, params[paramsOffset + 6u]), params[paramsOffset + 7u]);
  } else {
    result = nan;
  }
  valuesOut[valuesOutOffset + index] = result;`})]}};export{g as a,m as i,b as n,p as o,v as r,u as s,S as t};