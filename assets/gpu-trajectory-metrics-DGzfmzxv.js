import{E as e,S as t,p as n}from"./analysis-frame-Blu0tYdw.js";import{n as r,t as i}from"./parameter-schema-DB2zG6AH.js";import{_ as a,f as o,h as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{t as u}from"./gpu-segmented-reduction-BOmwUgZN.js";import{t as d}from"./gpu-compaction-Cpx1OCRM.js";import{t as f}from"./gpu-group-aggregation-DrNKSCi8.js";import{l as p,r as m,s as h}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as g}from"./geodesic-wgsl-DS0CJZ42.js";import{t as _}from"./time-words-C-gLAZQw.js";var v=`GPUTrajectoryMetrics`,y=8;function b(e){return e.timestamps.format===`uint32x2`?`words`:e.timestampsLow?`double-single`:`float32`}function x(e){let t=[{name:`timestamps`,view:e.timestamps,type:e.timestamps.format===`uint32x2`?`u32`:`f32`,access:`read`}];return e.timestampsLow&&t.push({name:`timestampsLow`,view:e.timestampsLow,type:`f32`,access:`read`}),t}function S(e){switch(e){case`words`:return`${_}
fn getTimeWords(row: u32) -> vec2<u32> {
  return vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
}

fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return timeWordsDifferenceToF32(getTimeWords(a), getTimeWords(b));
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  return timeWordsCompare(getTimeWords(a), getTimeWords(b)) >= 0i;
}
`;case`double-single`:return`
fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return (timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b]) +
    (timestampsLow[timestampsLowOffset + a] - timestampsLow[timestampsLowOffset + b]);
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  let aHigh = timestamps[timestampsOffset + a];
  let bHigh = timestamps[timestampsOffset + b];
  return aHigh > bHigh ||
    (aHigh == bHigh && timestampsLow[timestampsLowOffset + a] >= timestampsLow[timestampsLowOffset + b]);
}
`;default:return`
fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  return timestamps[timestampsOffset + a] >= timestamps[timestampsOffset + b];
}
`}}function C(e){return`
const TRACK_COUNT: u32 = ${e}u;
const NO_TRACK: u32 = 0xffffffffu;

fn findTrack(row: u32) -> u32 {
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NO_TRACK;
  }
  return low - 1u;
}`}function w(e=!0,t){let n=`${t===void 0?`
fn getStepDistance(row: u32) -> f32 {
  let deltaX = positions[positionsOffset + 2u * row] - positions[positionsOffset + 2u * (row - 1u)];
  let deltaY = positions[positionsOffset + 2u * row + 1u] - positions[positionsOffset + 2u * (row - 1u) + 1u];
  return sqrt(deltaX * deltaX + deltaY * deltaY);
}
`:`${g}
fn getStepPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getStepDistance(row: u32) -> f32 {
  return geodesicHaversineDistance(getStepPosition(row - 1u), getStepPosition(row), ${p(t)});
}
`}

fn getStepDuration(row: u32) -> f32 {
  return timeDifferenceRows(row, row - 1u);
}
`;return e?`${n}
fn isSlowStep(row: u32) -> bool {
  let stepDuration = getStepDuration(row);
  let stepDistance = getStepDistance(row);
  return isTimeNonDecreasing(row, row - 1u) &&
    (stepDistance == 0.0 || stepDistance < parameters[parametersOffset] * stepDuration);
}`:n}function T(e,t){let n=!!(t.runStartFlags&&t.runEndFlags&&t.parameters);if(3+(t.timestampsLow?1:0)+(n?3:0)+(t.stepDistances?1:0)+(t.stepSpeeds?1:0)<=y||!n)return[E(e,t)];let{runStartFlags:r,runEndFlags:i,parameters:a,...o}=t,s=[];return(t.stepDistances||t.stepSpeeds)&&s.push(E(e,o)),s.push(E(e,{id:`${t.id}-runs`,positions:t.positions,timestamps:t.timestamps,timestampsLow:t.timestampsLow,trackOffsets:t.trackOffsets,parameters:a,geodesicRadius:t.geodesicRadius,runStartFlags:r,runEndFlags:i})),s}function E(e,t){let n=t.trackOffsets.length-1,r=!!(t.runStartFlags&&t.runEndFlags&&t.parameters),i=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...x(t),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}];return r&&i.push({name:`parameters`,view:t.parameters,type:`f32`,access:`read`}),t.stepDistances&&i.push({name:`stepDistances`,view:t.stepDistances,type:`f32`,access:`read_write`}),t.stepSpeeds&&i.push({name:`stepSpeeds`,view:t.stepSpeeds,type:`f32`,access:`read_write`}),r&&i.push({name:`runStartFlags`,view:t.runStartFlags,type:`u32`,access:`read_write`},{name:`runEndFlags`,view:t.runEndFlags,type:`u32`,access:`read_write`}),h(e,{id:t.id,operation:v,variant:`steps`,bindings:i,invocationCount:t.positions.length,declarations:`${C(n)}
${S(b(t))}
${w(r,t.geodesicRadius)}`,body:`
  var stepDistance = 0.0;
  var stepSpeed = 0.0;
  var runStart = 0u;
  var runEnd = 0u;
  let track = findTrack(index);
  if (track != NO_TRACK) {
    let trackStart = trackOffsets[trackOffsetsOffset + track];
    let trackEnd = trackOffsets[trackOffsetsOffset + track + 1u];
    if (index > trackStart) {
      stepDistance = getStepDistance(index);
      let stepDuration = getStepDuration(index);
      if (stepDuration > 0.0) {
        stepSpeed = stepDistance / stepDuration;
      }
      ${r?`if (isSlowStep(index)) {
        let previousSlow = index - 1u > trackStart && isSlowStep(index - 1u);
        let nextSlow = index + 1u < trackEnd && isSlowStep(index + 1u);
        runStart = select(1u, 0u, previousSlow);
        runEnd = select(1u, 0u, nextSlow);
      }`:``}
    }
  }
  ${t.stepDistances?`stepDistances[stepDistancesOffset + index] = stepDistance;`:``}
  ${t.stepSpeeds?`stepSpeeds[stepSpeedsOffset + index] = stepSpeed;`:``}
  ${r?`runStartFlags[runStartFlagsOffset + index] = runStart;
  runEndFlags[runEndFlagsOffset + index] = runEnd;`:``}`})}function D(e,t){let n=t.trackOffsets.length-1,r=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...x(t),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}];for(let[e,n]of[[`stepSpeeds`,t.stepSpeeds],[`stepHeadings`,t.stepHeadings],[`stepAccelerations`,t.stepAccelerations]])n&&r.push({name:e,view:n,type:`f32`,access:`read_write`});return h(e,{id:t.id,operation:v,variant:`step-columns`,bindings:r,invocationCount:t.positions.length,declarations:`${C(n)}
${S(b(t))}
${w(!1,t.geodesicRadius)}

fn getStepSpeed(row: u32) -> f32 {
  let duration = getStepDuration(row);
  if (duration > 0.0) {
    return getStepDistance(row) / duration;
  }
  return 0.0;
}`,body:`
  var speed = 0.0;
  var heading = 0.0;
  var acceleration = 0.0;
  let track = findTrack(index);
  if (track != NO_TRACK) {
    let trackStart = trackOffsets[trackOffsetsOffset + track];
    if (index > trackStart) {
      speed = getStepSpeed(index);
      ${t.geodesicRadius===void 0?`let deltaX = positions[positionsOffset + 2u * index] - positions[positionsOffset + 2u * (index - 1u)];
      let deltaY = positions[positionsOffset + 2u * index + 1u] - positions[positionsOffset + 2u * (index - 1u) + 1u];
      if (deltaX != 0.0 || deltaY != 0.0) {
        heading = atan2(deltaY, deltaX);
      }`:`if (getStepDistance(index) > 0.0) {
        // The planar convention (counterclockwise from east) applied to the initial bearing
        // (clockwise from north): atan2(cos(bearing), sin(bearing)) = pi / 2 - bearing, wrapped.
        let bearing = geodesicInitialBearingDegrees(getStepPosition(index - 1u), getStepPosition(index)) * GEODESIC_DEGREES_TO_RADIANS;
        heading = atan2(cos(bearing), sin(bearing));
      }`}
      let duration = getStepDuration(index);
      if (index - 1u > trackStart && duration > 0.0) {
        acceleration = (speed - getStepSpeed(index - 1u)) / duration;
      }
    }
  }
  ${t.stepSpeeds?`stepSpeeds[stepSpeedsOffset + index] = speed;`:``}
  ${t.stepHeadings?`stepHeadings[stepHeadingsOffset + index] = heading;`:``}
  ${t.stepAccelerations?`stepAccelerations[stepAccelerationsOffset + index] = acceleration;`:``}`})}function O(e,t){let n=t.trackOffsets.length-1,r=[...x(t),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}];return t.averageSpeeds&&t.trackLengths&&r.push({name:`trackLengths`,view:t.trackLengths,type:`f32`,access:`read`}),t.trackDurations&&r.push({name:`trackDurations`,view:t.trackDurations,type:`f32`,access:`read_write`}),t.averageSpeeds&&r.push({name:`averageSpeeds`,view:t.averageSpeeds,type:`f32`,access:`read_write`}),h(e,{id:t.id,operation:v,variant:`finalize`,bindings:r,invocationCount:n,declarations:S(b(t)),body:`
  let trackStart = trackOffsets[trackOffsetsOffset + index];
  let trackEnd = trackOffsets[trackOffsetsOffset + index + 1u];
  var trackDuration = 0.0;
  if (trackEnd >= trackStart + 2u) {
    trackDuration = timeDifferenceRows(trackEnd - 1u, trackStart);
  }
  ${t.trackDurations?`trackDurations[trackDurationsOffset + index] = trackDuration;`:``}
  ${t.averageSpeeds?`var averageSpeed = 0.0;
  if (trackDuration > 0.0) {
    averageSpeed = trackLengths[trackLengthsOffset + index] / trackDuration;
  }
  averageSpeeds[averageSpeedsOffset + index] = averageSpeed;`:``}`})}function k(e,t){return h(e,{id:t.id,operation:v,variant:`row-ids`,bindings:[{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`},{name:`runIndices`,view:t.runIndices,type:`u32`,access:`read_write`}],invocationCount:Math.max(t.rowCount,t.runCapacity),declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const RUN_CAPACITY: u32 = ${t.runCapacity}u;`,body:`
  if (index < ROW_COUNT) {
    rowIds[rowIdsOffset + index] = index;
  }
  if (index < RUN_CAPACITY) {
    runIndices[runIndicesOffset + index] = index;
  }`})}function A(e,t){let n=x(t),r=S(b(t)),i=[{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runEnds`,view:t.runEnds,type:`u32`,access:`read`},{name:`runCountIn`,view:t.runCount,type:`u32`,access:`read`}],a=[{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}],o=[{name:`qualifyFlags`,view:t.qualifyFlags,type:`u32`,access:`read_write`},{name:`runTracks`,view:t.runTracks,type:`u32`,access:`read_write`}],s=[{name:`parameters`,view:t.parameters,type:`f32`,access:`read`}],c=`
  var flag = 0u;
  if (index < runCountIn[runCountInOffset]) {
    let firstStepRow = runStarts[runStartsOffset + index];
    let lastStepRow = runEnds[runEndsOffset + index];
    if (timeDifferenceRows(lastStepRow, firstStepRow - 1u) >= parameters[parametersOffset + 1u]) {
      flag = 1u;
    }
  }`,l={operation:v,invocationCount:t.runCapacity},u=C(t.trackOffsets.length-1),d=[...i,...n,...s,...a,...o];return d.length<=y?[h(e,{...l,id:t.id,variant:`qualify`,bindings:d,declarations:`${u}\n${r}`,body:`${c}
  var track = NO_TRACK;
  if (flag == 1u) {
    track = findTrack(runStarts[runStartsOffset + index]);
  }
  qualifyFlags[qualifyFlagsOffset + index] = flag;
  runTracks[runTracksOffset + index] = track;`})]:[h(e,{...l,id:t.id,variant:`qualify`,bindings:[...i,...n,...s,o[0]],declarations:r,body:`${c}
  qualifyFlags[qualifyFlagsOffset + index] = flag;`}),h(e,{...l,id:`${t.id}-tracks`,variant:`qualify-tracks`,bindings:[i[0],i[2],...a,{name:`qualifyFlags`,view:t.qualifyFlags,type:`u32`,access:`read`},o[1]],declarations:u,body:`
  var track = NO_TRACK;
  if (index < runCountIn[runCountInOffset] && qualifyFlags[qualifyFlagsOffset + index] == 1u) {
    track = findTrack(runStarts[runStartsOffset + index]);
  }
  runTracks[runTracksOffset + index] = track;`})]}function j(e,t){return h(e,{id:t.id,operation:v,variant:`centroid-offsets`,bindings:[{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runEnds`,view:t.runEnds,type:`u32`,access:`read`},{name:`runCountIn`,view:t.runCount,type:`u32`,access:`read`},{name:`centroidOffsets`,view:t.centroidOffsets,type:`u32`,access:`read_write`}],invocationCount:t.runCapacity,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const RUN_CAPACITY: u32 = ${t.runCapacity}u;`,body:`
  var segmentStart = ROW_COUNT;
  var segmentEnd = ROW_COUNT;
  if (index < runCountIn[runCountInOffset]) {
    segmentStart = runStarts[runStartsOffset + index] - 1u;
    segmentEnd = runEnds[runEndsOffset + index] + 1u;
  }
  centroidOffsets[centroidOffsetsOffset + 2u * index] = segmentStart;
  centroidOffsets[centroidOffsetsOffset + 2u * index + 1u] = segmentEnd;
  if (index == 0u) {
    centroidOffsets[centroidOffsetsOffset + 2u * RUN_CAPACITY] = ROW_COUNT;
  }`})}function M(e,t){return h(e,{id:t.id,operation:v,variant:`split-positions`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`xs`,view:t.xs,type:`f32`,access:`read_write`},{name:`ys`,view:t.ys,type:`f32`,access:`read_write`}],invocationCount:t.positions.length,body:`
  xs[xsOffset + index] = positions[positionsOffset + 2u * index];
  ys[ysOffset + index] = positions[positionsOffset + 2u * index + 1u];`})}function N(e,t){let n=[],r=[{name:`stopTotalIn`,view:t.stopTotal,type:`u32`,access:`read`},{name:`qualifiedRuns`,view:t.qualifiedRuns,type:`u32`,access:`read`},{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runEnds`,view:t.runEnds,type:`u32`,access:`read`}],i=`const CAPACITY: u32 = ${t.capacity}u;`,a=`
  let count = min(stopTotalIn[stopTotalInOffset], CAPACITY);
  let isStop = index < count;
  var run = 0u;
  if (isStop) {
    run = qualifiedRuns[qualifiedRunsOffset + index];
  }`,o=[{name:`stopTotalIn`,view:t.stopTotal,type:`u32`,access:`read`},{name:`qualifiedRuns`,view:t.qualifiedRuns,type:`u32`,access:`read`},{name:`runTracks`,view:t.runTracks,type:`u32`,access:`read`},{name:`idsOut`,view:t.ids,type:`u32`,access:`read_write`}];return t.startRows&&o.push({name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`}),t.endRows&&o.push({name:`runEnds`,view:t.runEnds,type:`u32`,access:`read`}),t.startRows&&o.push({name:`startRowsOut`,view:t.startRows,type:`u32`,access:`read_write`}),t.endRows&&o.push({name:`endRowsOut`,view:t.endRows,type:`u32`,access:`read_write`}),n.push(h(e,{id:t.id,operation:v,variant:`stop-gather`,bindings:o,invocationCount:t.capacity,declarations:i,body:`${a}
  idsOut[idsOutOffset + index] = select(0xffffffffu, runTracks[runTracksOffset + run], isStop);
  ${t.startRows?`startRowsOut[startRowsOutOffset + index] = select(0xffffffffu, runStarts[runStartsOffset + run] - 1u, isStop);`:``}
  ${t.endRows?`endRowsOut[endRowsOutOffset + index] = select(0xffffffffu, runEnds[runEndsOffset + run], isStop);`:``}`})),t.durations&&n.push(h(e,{id:`${t.id}-durations`,operation:v,variant:`stop-durations`,bindings:[...r,...x(t),{name:`durationsOut`,view:t.durations,type:`f32`,access:`read_write`}],invocationCount:t.capacity,declarations:`${i}
${S(b(t))}`,body:`${a}
  var stopDuration = 0.0;
  if (isStop) {
    stopDuration = timeDifferenceRows(runEnds[runEndsOffset + run], runStarts[runStartsOffset + run] - 1u);
  }
  durationsOut[durationsOutOffset + index] = stopDuration;`})),t.centroids&&t.sumsX&&t.sumsY&&n.push(h(e,{id:`${t.id}-centroids`,operation:v,variant:`stop-centroids`,bindings:[...r,{name:`sumsX`,view:t.sumsX,type:`f32`,access:`read`},{name:`sumsY`,view:t.sumsY,type:`f32`,access:`read`},{name:`centroidsOut`,view:t.centroids,type:`f32`,access:`read_write`}],invocationCount:t.capacity,declarations:i,body:`${a}
  var centroid = vec2<f32>(0.0, 0.0);
  if (isStop) {
    let rowCount = f32(runEnds[runEndsOffset + run] - runStarts[runStartsOffset + run] + 2u);
    centroid = vec2<f32>(sumsX[sumsXOffset + 2u * run], sumsY[sumsYOffset + 2u * run]) / rowCount;
  }
  centroidsOut[centroidsOutOffset + 2u * index] = centroid.x;
  centroidsOut[centroidsOutOffset + 2u * index + 1u] = centroid.y;`})),n}function P(e,t,n){let r=t.output.length;if(r<=n)return new u({...t}).getCommandNodes(e);let i=n>=64?Math.floor(n/64)*64:n,a=[];for(let n=0,o=0;n<r;n+=i,o++){let s=Math.min(i,r-n),c=e.createDataView(t.segmentOffsets.buffer,{format:`uint32`,length:s+1,byteOffset:t.segmentOffsets.byteOffset+n*4}),l=e.createDataView(t.output.buffer,{format:`float32`,length:s,byteOffset:t.output.byteOffset+n*4});a.push(...new u({id:`${t.id}-chunk-${o}`,input:t.input,segmentOffsets:c,output:l,operation:t.operation}).getCommandNodes(e))}return a}var F=i({id:`trajectory-metrics`,format:`float32`,wordLength:4,fields:[{name:`stopSpeedThreshold`,format:`float32`,wordOffset:0,defaultValue:0,minimum:0,units:`spatial-context-units-per-time-unit`,dynamic:!0},{name:`stopMinimumDuration`,format:`float32`,wordOffset:1,defaultValue:0,minimum:0,units:`time-units`,dynamic:!0}]});function I(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Trajectory metrics target must hold 4 elements`);return t.set(r(F,e)),t}var L=Object.freeze({...n,measure:`distance`}),R=class{id;props;spatialContext;constructor(t){this.id=t.id??`trajectory-metrics`,this.props=t;let n=this.id;if(this.spatialContext=e(n,t.spatialContext,L),l(t.positions,[`float32x2`],`${n} positions`),l(t.timestamps,[`float32`,`uint32x2`],`${n} timestamps`),t.timestampsLow){if(t.timestamps.format!==`float32`)throw Error(`${n} timestampsLow requires float32 timestamps, not Int64 words`);if(l(t.timestampsLow,[`float32`],`${n} timestampsLow`),t.timestampsLow.length!==t.positions.length)throw Error(`${n} timestampsLow length must equal positions length`)}if(l(t.trackOffsets,[`uint32`],`${n} trackOffsets`),t.timestamps.length!==t.positions.length)throw Error(`${n} timestamps length must equal positions length`);if(t.trackOffsets.length<2)throw Error(`${n} trackOffsets must contain at least two rows`);let r=t.trackOffsets.length-1,{output:i}=t;for(let[e,t]of[[`trackLengths`,i.trackLengths],[`trackDurations`,i.trackDurations],[`averageSpeeds`,i.averageSpeeds],[`maximumSpeeds`,i.maximumSpeeds]])if(t&&(l(t,[`float32`],`${n} ${e}`),t.length!==r))throw Error(`${n} ${e} length must equal the track count`);for(let[e,r]of[[`stepSpeeds`,i.stepSpeeds],[`stepHeadings`,i.stepHeadings],[`stepAccelerations`,i.stepAccelerations]])if(r&&(l(r,[`float32`],`${n} ${e}`),r.length!==t.positions.length))throw Error(`${n} ${e} length must equal positions length`);if(i.trackStopCounts&&(l(i.trackStopCounts,[`uint32`],`${n} trackStopCounts`),i.trackStopCounts.length!==r))throw Error(`${n} trackStopCounts length must equal the track count`);let{stops:a}=i;if(a){if(o(n,a),a.drawInstanceCount&&(l(a.drawInstanceCount,[`uint32`],`${n} stops.drawInstanceCount`),a.drawInstanceCount.length<1))throw Error(`${n} stops.drawInstanceCount must contain one uint32 row`);let e=a.ids.length;for(let[t,r,i]of[[`startRows`,a.startRows,`uint32`],[`endRows`,a.endRows,`uint32`],[`centroids`,a.centroids,`float32x2`],[`durations`,a.durations,`float32`]])if(r&&(l(r,[i],`${n} stops.${t}`),r.length!==e))throw Error(`${n} stops.${t} length must equal the stop capacity`)}if(t.parameters&&(l(t.parameters,[`float32`],`${n} parameters`),t.parameters.length<4))throw Error(`${n} parameters must hold 4 float32 values`);if((a||i.trackStopCounts)&&!t.parameters)throw Error(`${n} parameters are required for stops and trackStopCounts`);if(!(i.trackLengths||i.trackDurations||i.averageSpeeds||i.maximumSpeeds||i.trackStopCounts||i.stepSpeeds||i.stepHeadings||i.stepAccelerations||a))throw Error(`${n} requires at least one output`);s(n,[i.trackLengths,i.trackDurations,i.averageSpeeds,i.maximumSpeeds,i.stepSpeeds,i.stepHeadings,i.stepAccelerations,i.trackStopCounts,a?.ids,a?.status.count,a?.status.overflow,a?.status.requiredCount,a?.drawInstanceCount,a?.startRows,a?.endRows,a?.centroids,a?.durations],[t.positions,t.timestamps,t.timestampsLow,t.trackOffsets,t.parameters])}getCommandNodes(e){let{props:n,id:r}=this,{output:i}=n,{stops:o}=i;a(r,e,[n.positions,n.timestamps,n.timestampsLow,n.trackOffsets,n.parameters,i.trackLengths,i.trackDurations,i.averageSpeeds,i.maximumSpeeds,i.stepSpeeds,i.stepHeadings,i.stepAccelerations,i.trackStopCounts,o?.ids,o?.status.count,o?.status.overflow,o?.status.requiredCount,o?.drawInstanceCount,o?.startRows,o?.endRows,o?.centroids,o?.durations]);let s=n.positions.length,l=n.trackOffsets.length-1,u=e.device.limits.maxComputeWorkgroupsPerDimension,p=!!(o||i.trackStopCounts),h=[],g=t(this.spatialContext),_=i.trackLengths||i.averageSpeeds?c(e,`${r}-step-distances`,`float32`,s):void 0,v=i.maximumSpeeds?c(e,`${r}-step-speeds`,`float32`,s):void 0,y=Math.max(1,Math.ceil(s/2)),b=p?c(e,`${r}-run-start-flags`,`uint32`,s):void 0,x=p?c(e,`${r}-run-end-flags`,`uint32`,s):void 0;h.push(...T(e,{id:`${r}-steps`,positions:n.positions,timestamps:n.timestamps,timestampsLow:n.timestampsLow,trackOffsets:n.trackOffsets,parameters:p?n.parameters:void 0,geodesicRadius:g,stepDistances:_,stepSpeeds:v,runStartFlags:b,runEndFlags:x})),(i.stepSpeeds||i.stepHeadings||i.stepAccelerations)&&h.push(D(e,{id:`${r}-step-columns`,positions:n.positions,timestamps:n.timestamps,timestampsLow:n.timestampsLow,trackOffsets:n.trackOffsets,geodesicRadius:g,stepSpeeds:i.stepSpeeds,stepHeadings:i.stepHeadings,stepAccelerations:i.stepAccelerations}));let S=i.trackLengths??(i.averageSpeeds?c(e,`${r}-lengths`,`float32`,l):void 0);if(_&&S&&h.push(...P(e,{id:`${r}-track-lengths`,input:_,segmentOffsets:n.trackOffsets,output:S,operation:`sum`},u)),v&&i.maximumSpeeds&&h.push(...P(e,{id:`${r}-maximum-speeds`,input:v,segmentOffsets:n.trackOffsets,output:i.maximumSpeeds,operation:`max`},u)),(i.trackDurations||i.averageSpeeds)&&h.push(O(e,{id:`${r}-track-finalize`,timestamps:n.timestamps,timestampsLow:n.timestampsLow,trackOffsets:n.trackOffsets,trackLengths:S,trackDurations:i.trackDurations,averageSpeeds:i.averageSpeeds})),!p||!b||!x||!n.parameters)return h;let C=c(e,`${r}-row-ids`,`uint32`,s),w=c(e,`${r}-run-indices`,`uint32`,y),E=c(e,`${r}-run-starts`,`uint32`,Math.max(s,1)),F=c(e,`${r}-run-ends`,`uint32`,Math.max(s,1)),I=c(e,`${r}-run-start-count`,`uint32`,1),L=c(e,`${r}-run-end-count`,`uint32`,1),R=c(e,`${r}-qualify-flags`,`uint32`,y),z=c(e,`${r}-run-tracks`,`uint32`,y);h.push(k(e,{id:`${r}-row-ids`,rowIds:C,rowCount:s,runIndices:w,runCapacity:y}),...new d({id:`${r}-run-starts`,input:C,flags:b,output:E,count:I}).getCommandNodes(e),...new d({id:`${r}-run-ends`,input:C,flags:x,output:F,count:L}).getCommandNodes(e),...A(e,{id:`${r}-run-qualify`,timestamps:n.timestamps,timestampsLow:n.timestampsLow,trackOffsets:n.trackOffsets,parameters:n.parameters,runStarts:E,runEnds:F,runCount:I,qualifyFlags:R,runTracks:z,runCapacity:y}));let B;if(o?.centroids){let t=c(e,`${r}-centroid-offsets`,`uint32`,2*y+1),i=c(e,`${r}-xs`,`float32`,s),a=c(e,`${r}-ys`,`float32`,s),o=c(e,`${r}-sums-x`,`float32`,2*y),l=c(e,`${r}-sums-y`,`float32`,2*y);h.push(j(e,{id:`${r}-centroid-offsets`,runStarts:E,runEnds:F,runCount:I,centroidOffsets:t,runCapacity:y,rowCount:s}),M(e,{id:`${r}-split-positions`,positions:n.positions,xs:i,ys:a}),...P(e,{id:`${r}-centroid-sums-x`,input:i,segmentOffsets:t,output:o,operation:`sum`},u),...P(e,{id:`${r}-centroid-sums-y`,input:a,segmentOffsets:t,output:l,operation:`sum`},u)),B={sumsX:o,sumsY:l}}if(o){let t=c(e,`${r}-qualified-runs`,`uint32`,y),a=c(e,`${r}-stop-total`,`uint32`,1);h.push(...new d({id:`${r}-stop-indices`,input:w,flags:R,output:t,count:a}).getCommandNodes(e),...N(e,{id:`${r}-stop-gather`,capacity:o.ids.length,stopTotal:a,qualifiedRuns:t,runStarts:E,runEnds:F,runTracks:z,timestamps:n.timestamps,timestampsLow:n.timestampsLow,sumsX:B?.sumsX,sumsY:B?.sumsY,ids:o.ids,startRows:o.startRows,endRows:o.endRows,centroids:o.centroids,durations:o.durations})),i.trackStopCounts&&h.push(...new f({id:`${r}-stop-counts`,keys:z,output:i.trackStopCounts}).getCommandNodes(e)),h.push(m(e,{id:`${r}-publish`,operation:`GPUTrajectoryMetrics`,requiredCount:a,output:o,extraCounts:o.drawInstanceCount?[o.drawInstanceCount]:[]}))}else i.trackStopCounts&&h.push(...new f({id:`${r}-stop-counts`,keys:z,output:i.trackStopCounts}).getCommandNodes(e));return h}};export{I as n,R as t};