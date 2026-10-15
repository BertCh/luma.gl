import{M as e,R as t,_ as n,f as r,g as i,h as a,j as o,s,w as c}from"./resources-CwvhEOBe.js";import{t as l}from"./gpu-visibility-workflow-CuWho9Ae.js";import{t as u}from"./gpu-group-aggregation-DrNKSCi8.js";import{i as d,n as f,r as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as h}from"./time-words-C-gLAZQw.js";import{n as g,t as _}from"./time-window-classify-node-Dq1W0FVm.js";function v(e,t){let n=t.timestamps.length,r=t.timestamps.format===`uint32x2`,i=[{name:`params`,view:t.window,type:r?`u32`:`f32`,access:`read`}];n>0&&(i.push({name:`rowTimes`,view:t.timestamps,type:r?`u32`:`f32`,access:`read`}),t.timestampsLow&&i.push({name:`rowTimesLow`,view:t.timestampsLow,type:`f32`,access:`read`})),i.push({name:`rangeOut`,view:t.range,type:`u32`,access:`read_write`},{name:`totalOut`,view:t.total,type:`u32`,access:`read_write`}),t.firstInstance&&i.push({name:`firstInstanceOut`,view:t.firstInstance,type:`u32`,access:`read_write`});let a=r?`${h}
override ROW_COUNT: u32;

fn isRowBeforeStart(row: u32, windowStart: vec2<u32>, windowStartFraction: f32) -> bool {
  let time = vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
  return !isTimeWordsAtLeast(time, 0.0, windowStart, windowStartFraction);
}

fn isRowAfterEnd(row: u32, windowEnd: vec2<u32>, windowEndFraction: f32) -> bool {
  let time = vec2<u32>(rowTimes[rowTimesOffset + 2u * row], rowTimes[rowTimesOffset + 2u * row + 1u]);
  return !isTimeWordsAtLeast(windowEnd, windowEndFraction, time, 0.0);
}`:`${_}
override ROW_COUNT: u32;

fn getRowLow(row: u32) -> f32 {
  return ${t.timestampsLow?`rowTimesLow[rowTimesLowOffset + row]`:`0.0`};
}

fn isRowBeforeStart(row: u32, windowStartHigh: f32, windowStartLow: f32) -> bool {
  return !isTimeAtLeast(rowTimes[rowTimesOffset + row], getRowLow(row), windowStartHigh, windowStartLow);
}

fn isRowAfterEnd(row: u32, windowEndHigh: f32, windowEndLow: f32) -> bool {
  return !isTimeAtLeast(windowEndHigh, windowEndLow, rowTimes[rowTimesOffset + row], getRowLow(row));
}`,o=r?`windowStart, windowStartFraction`:`windowStartHigh, windowStartLow`,s=r?`windowEnd, windowEndFraction`:`windowEndHigh, windowEndLow`;return m(e,{id:t.id,operation:`GPUTimeWindowFilter`,variant:r?`sorted-range-words`:`sorted-range`,bindings:i,invocationCount:1,workgroupSize:32,declarations:n===0?void 0:a,constants:n===0?void 0:{ROW_COUNT:n},body:n===0?`rangeOut[rangeOutOffset] = 0u;
  rangeOut[rangeOutOffset + 1u] = 0u;
  totalOut[totalOutOffset] = 0u;
  ${t.firstInstance?`firstInstanceOut[firstInstanceOutOffset] = 0u;`:``}`:`
  var isWindowValid = false;
  ${r?`
  let windowStart = vec2<u32>(params[paramsOffset + 0u], params[paramsOffset + 1u]);
  let windowEnd = vec2<u32>(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let windowStartFraction = bitcast<f32>(params[paramsOffset + 4u]);
  let windowEndFraction = bitcast<f32>(params[paramsOffset + 5u]);
  isWindowValid = isTimeWordsAtLeast(windowEnd, windowEndFraction, windowStart, windowStartFraction);`:`
  let windowStartHigh = params[paramsOffset + 0u];
  let windowStartLow = params[paramsOffset + 1u];
  let windowEndHigh = params[paramsOffset + 2u];
  let windowEndLow = params[paramsOffset + 3u];
  isWindowValid = isTimeAtLeast(windowEndHigh, windowEndLow, windowStartHigh, windowStartLow);`}
  var firstRow = 0u;
  var endRow = 0u;
  if (isWindowValid) {
    // Lower bound: the first row that is not before the window start.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (isRowBeforeStart(middle, ${o})) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    firstRow = low;
    // Upper bound: the first row after the window end, searched from the lower bound.
    high = ROW_COUNT;
    loop {
      if (low >= high) { break; }
      let middle = low + (high - low) / 2u;
      if (isRowAfterEnd(middle, ${s})) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    endRow = low;
  }
  let rowCount = endRow - firstRow;
  rangeOut[rangeOutOffset] = firstRow;
  rangeOut[rangeOutOffset + 1u] = rowCount;
  totalOut[totalOutOffset] = rowCount;
  ${t.firstInstance?`firstInstanceOut[firstInstanceOutOffset] = firstRow;`:``}`})}function y(e,t){let n=[{name:`range`,view:t.range,type:`u32`,access:`read`},{name:`idsOut`,view:t.ids,type:`u32`,access:`read_write`}];return t.sourceIds&&n.push({name:`sourceIds`,view:t.sourceIds,type:`u32`,access:`read`}),m(e,{id:t.id,operation:`GPUTimeWindowFilter`,variant:`sorted-ids`,bindings:n,invocationCount:t.ids.length,body:`let firstRow = range[rangeOffset];
  // The invocation count is the capacity.
  let count = min(range[rangeOffset + 1u], INVOCATION_COUNT);
  if (index >= count) {
    return;
  }
  idsOut[idsOutOffset + index] = ${t.sourceIds?`sourceIds[sourceIdsOffset + firstRow + index]`:`firstRow + index`};`})}function b(e,t){let n=t.timestamps.format===`uint32x2`,r=[{name:`rowTimes`,view:t.timestamps,type:n?`u32`:`f32`,access:`read`}];return t.timestampsLow&&r.push({name:`rowTimesLow`,view:t.timestampsLow,type:`f32`,access:`read`}),r.push({name:`violations`,view:t.violations,type:`atomic<u32>`,access:`read_write`}),m(e,{id:t.id,operation:`GPUTimeWindowFilter`,variant:n?`sorted-order-check-words`:`sorted-order-check`,bindings:r,invocationCount:Math.max(t.timestamps.length-1,1),declarations:n?h:_,body:n?`if (index + 1u >= ${t.timestamps.length}u) {
    return;
  }
  let previous = vec2<u32>(rowTimes[rowTimesOffset + 2u * index], rowTimes[rowTimesOffset + 2u * index + 1u]);
  let next = vec2<u32>(rowTimes[rowTimesOffset + 2u * index + 2u], rowTimes[rowTimesOffset + 2u * index + 3u]);
  if (!isTimeWordsAtLeast(next, 0.0, previous, 0.0)) {
    atomicAdd(&violations[violationsOffset], 1u);
  }`:`if (index + 1u >= ${t.timestamps.length}u) {
    return;
  }
  ${t.timestampsLow?`let previousLow = rowTimesLow[rowTimesLowOffset + index];
  let nextLow = rowTimesLow[rowTimesLowOffset + index + 1u];`:`let previousLow = 0.0;
  let nextLow = 0.0;`}
  if (!isTimeAtLeast(rowTimes[rowTimesOffset + index + 1u], nextLow, rowTimes[rowTimesOffset + index], previousLow)) {
    atomicAdd(&violations[violationsOffset], 1u);
  }`})}var x=class{id;props;constructor(t){this.id=t.id??`time-window-filter`,this.props=t;let n=this.id,c=t.timestamps.length,{output:l}=t,u=t.timestamps.format===`uint32x2`;if(t.endTimestamps&&t.endTimestamps.format===`uint32x2`!==u)throw Error(`${n} endTimestamps must have the same format as timestamps`);if(u&&(t.timestampsLow||t.endTimestampsLow))throw Error(`${n} timestampsLow and endTimestampsLow require float32 timestamps`);for(let[r,a]of[[`timestamps`,t.timestamps],[`timestampsLow`,t.timestampsLow],[`endTimestamps`,t.endTimestamps],[`endTimestampsLow`,t.endTimestampsLow],[`output.fadeWeights`,l.fadeWeights]]){let o=u&&(r===`timestamps`||r===`endTimestamps`);for(let t of a?s(a):[])e(t,[o?`uint32x2`:`float32`],`${n} ${r}`);r!==`timestamps`&&i(n,r,t.timestamps,a)}for(let t of l.clipFractions?s(l.clipFractions):[])e(t,[`float32x2`],`${n} output.clipFractions`);i(n,`output.clipFractions`,t.timestamps,l.clipFractions);for(let e of l.outputMask?s(l.outputMask):[])o(e,`${n} output.outputMask`);if(i(n,`output.outputMask`,t.timestamps,l.outputMask),u){if(t.window.format!==`uint32`)throw Error(`${n} uint32x2 word timestamps require a packed uint32 word window`);if(e(t.window,[`uint32`],`${n} window`),t.window.length<8)throw Error(`${n} window must hold 8 uint32 values`)}else{if(t.window.format!==`float32`)throw Error(`${n} float32 timestamps require a float32 window`);if(e(t.window,[`float32`],`${n} window`),t.window.length<8)throw Error(`${n} window must hold 8 float32 values`)}for(let[e,r]of[[`sourceIds`,t.sourceIds],[`trackIds`,t.trackIds],...(t.additionalPredicates??[]).map((e,t)=>[`additionalPredicates[${t}]`,e.mask])])if(r&&r.length!==c)throw Error(`${n} ${e} length must equal timestamps length`);if(t.endTimestampsLow&&!t.endTimestamps)throw Error(`${n} endTimestampsLow requires endTimestamps`);if(l.clipFractions&&!t.endTimestamps)throw Error(`${n} output.clipFractions requires endTimestamps`);if(!!t.trackIds!=!!l.trackVisibleCounts)throw Error(`${n} trackIds and output.trackVisibleCounts must be given together`);if(l.trackVisibleCounts&&(o(l.trackVisibleCounts,`${n} output.trackVisibleCounts`),l.trackVisibleCounts.length<1))throw Error(`${n} output.trackVisibleCounts must contain at least one track`);if(r(n,t.output),this.validateSorted(),l.drawInstanceCount&&(o(l.drawInstanceCount,`${n} output.drawInstanceCount`),l.drawInstanceCount.length<1))throw Error(`${n} output.drawInstanceCount must contain one uint32 row`);a(n,[t.output.ids,t.output.status.count,t.output.status.overflow,t.output.status.requiredCount,l.outputMask,l.fadeWeights,l.clipFractions,l.trackVisibleCounts,l.drawInstanceCount,l.rowRange,l.drawFirstInstance,l.orderViolations],[t.timestamps,t.timestampsLow,t.endTimestamps,t.endTimestampsLow,t.window,t.sourceIds,t.trackIds,...(t.additionalPredicates??[]).map(e=>e.mask)])}validateSorted(){let{props:e,id:n}=this,{output:r}=e,i=[[`output.rowRange`,r.rowRange],[`output.drawFirstInstance`,r.drawFirstInstance],[`validateOrder`,e.validateOrder]];if(!e.sorted){for(let[e,t]of i)if(t)throw Error(`${n} ${e} requires sorted`);if(r.orderViolations)throw Error(`${n} output.orderViolations requires sorted and validateOrder`);return}if(e.endTimestamps)throw Error(`${n} sorted requires instant timestamps`);if(e.additionalPredicates?.length||e.trackIds)throw Error(`${n} sorted does not support additionalPredicates or trackIds`);for(let[r,i]of[[`timestamps`,e.timestamps],[`timestampsLow`,e.timestampsLow],[`sourceIds`,e.sourceIds]])if(i instanceof t)throw Error(`${n} sorted requires a single packed ${r} view`);if(r.rowRange&&(o(r.rowRange,`${n} output.rowRange`),r.rowRange.length<2))throw Error(`${n} output.rowRange must contain two uint32 rows`);if(r.drawFirstInstance&&(o(r.drawFirstInstance,`${n} output.drawFirstInstance`),r.drawFirstInstance.length<1))throw Error(`${n} output.drawFirstInstance must contain one uint32 row`);if(!!e.validateOrder!=!!r.orderViolations)throw Error(`${n} validateOrder and output.orderViolations must be given together`);if(r.orderViolations&&(o(r.orderViolations,`${n} output.orderViolations`),r.orderViolations.length<1))throw Error(`${n} output.orderViolations must contain one uint32 row`)}getCommandNodes(e){let{props:t,id:r}=this,{output:i,trackIds:a}=t,{outputMask:o,trackVisibleCounts:s,drawInstanceCount:f}=i,m=t.additionalPredicates??[];n(r,e,[t.timestamps,t.timestampsLow,t.endTimestamps,t.endTimestampsLow,t.window,t.sourceIds,o,i.fadeWeights,i.clipFractions,a,s,f,i.rowRange,i.drawFirstInstance,i.orderViolations,i.ids,i.status.count,i.status.overflow,i.status.requiredCount,...m.map(e=>e.mask)]);let h=t.timestamps.length;if(t.sorted)return this.getSortedCommandNodes(e);let _=m.length>0,v=[],y=!_&&o?o:d(e,`${r}-time-mask`,t.timestamps);v.push(...g(e,{id:`${r}-classify`,timestamps:t.timestamps,timestampsLow:t.timestampsLow,endTimestamps:t.endTimestamps,endTimestampsLow:t.endTimestampsLow,window:t.window,mask:y,fadeWeights:i.fadeWeights,clipFractions:i.clipFractions}));let b=y;_&&(b=o??(s?d(e,`${r}-mask`,t.timestamps):void 0));let x=h>0&&i.ids.length>=h,S=c(e,`${r}-total`,`uint32`,1),C=x?i.ids:c(e,`${r}-compact-ids`,`uint32`,h);return v.push(...new l({id:`${r}-visibility`,predicates:[{kind:`time-range`,mask:y},...m],output:C,count:S,outputMask:b,sourceIds:t.sourceIds}).getCommandNodes(e)),a&&s&&b&&v.push(...new u({id:`${r}-track-counts`,keys:a,mask:b,output:s}).getCommandNodes(e)),v.push(p(e,{id:`${r}-publish`,operation:`GPUTimeWindowFilter`,requiredCount:S,compactIds:x?void 0:C,output:i,extraCounts:f?[f]:[]})),v}getSortedCommandNodes(e){let{props:t,id:n}=this,{output:r}=t,i=t.timestamps,a=t.timestampsLow,o=[];r.orderViolations&&(o.push(f(e,{id:`${n}-order-fill`,operation:`GPUTimeWindowFilter`,view:r.orderViolations,type:`u32`,value:`0u`,componentCount:1})),i.length>=2&&o.push(b(e,{id:`${n}-order-check`,timestamps:i,timestampsLow:a,violations:r.orderViolations})));let s=r.rowRange??c(e,`${n}-range`,`uint32`,2),l=c(e,`${n}-total`,`uint32`,1);return o.push(v(e,{id:`${n}-range-search`,timestamps:i,timestampsLow:a,window:t.window,range:s,total:l,firstInstance:r.drawFirstInstance})),(r.outputMask||r.fadeWeights)&&o.push(...g(e,{id:`${n}-classify`,timestamps:t.timestamps,timestampsLow:t.timestampsLow,window:t.window,mask:r.outputMask??d(e,`${n}-time-mask`,i),fadeWeights:r.fadeWeights})),r.ids.length>0&&i.length>0&&o.push(y(e,{id:`${n}-ids`,range:s,ids:r.ids,sourceIds:t.sourceIds})),o.push(p(e,{id:`${n}-publish`,operation:`GPUTimeWindowFilter`,requiredCount:l,output:r,extraCounts:r.drawInstanceCount?[r.drawInstanceCount]:[]})),o}};export{x as t};