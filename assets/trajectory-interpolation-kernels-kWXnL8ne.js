import{l as e,s as t}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as n}from"./geodesic-wgsl-DS0CJZ42.js";import{t as r}from"./time-words-C-gLAZQw.js";var i={empty:0,active:1,beforeStart:2,afterEnd:3,gap:4};function a(e,t=new Float32Array(4)){if(o(t),typeof e.playhead==`bigint`)throw Error(`Float32 playheads must be numbers, not bigints`);let n=s(e);if(!Number.isFinite(e.playhead))throw Error(`Trajectory playhead must be finite`);return t.set([e.playhead,n,0,0]),t}function o(e){if(e.length<4)throw Error(`Trajectory playhead target must hold 4 elements`)}function s(e){let t=e.maximumGap??0;if(!Number.isFinite(t)||t<0)throw Error(`Playhead maximumGap must be finite and non-negative`);return t}var c=`GPUTrajectoryPlayhead`,l=`GPUTrajectoryResample`,u=`
const NO_ROW: u32 = 0xffffffffu;
const STATUS_EMPTY: u32 = ${i.empty}u;
const STATUS_ACTIVE: u32 = ${i.active}u;
const STATUS_BEFORE_START: u32 = ${i.beforeStart}u;
const STATUS_AFTER_END: u32 = ${i.afterEnd}u;
const STATUS_GAP: u32 = ${i.gap}u;
`;function d(t){return t===void 0?`
fn interpolatePosition(a: vec2<f32>, b: vec2<f32>, fraction: f32) -> vec2<f32> {
  return select(a + (b - a) * fraction, b, fraction >= 1.0);
}

fn getSegmentLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return length(b - a);
}

fn getSegmentHeading(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let delta = b - a;
  return select(0.0, atan2(delta.y, delta.x), delta.x != 0.0 || delta.y != 0.0);
}
`:`${n}
fn interpolatePosition(a: vec2<f32>, b: vec2<f32>, fraction: f32) -> vec2<f32> {
  if (fraction >= 1.0) {
    // b shifted by whole turns to continue the longitude of a (exactly b without a shift).
    return vec2<f32>(b.x - 360.0 * round((b.x - a.x) / 360.0), b.y);
  }
  if (fraction <= 0.0) {
    return a;
  }
  return geodesicInterpolate(a, b, geodesicCentralAngle(a, b), fraction);
}

fn getSegmentLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return geodesicHaversineDistance(a, b, ${e(t)});
}

fn getSegmentHeading(a: vec2<f32>, b: vec2<f32>) -> f32 {
  // Planar convention (counterclockwise from east) of the initial bearing: pi / 2 - bearing, wrapped.
  let bearing = geodesicInitialBearingDegrees(a, b) * GEODESIC_DEGREES_TO_RADIANS;
  return select(0.0, atan2(cos(bearing), sin(bearing)), geodesicCentralAngle(a, b) > 0.0);
}
`}function f(e){return{name:`timestamps`,view:e,type:e.format===`uint32x2`?`u32`:`f32`,access:`read`}}function p(e){return e?`${r}
fn getTimeWords(row: u32) -> vec2<u32> {
  return vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
}

fn getRowTimeDifference(a: u32, b: u32) -> f32 {
  return timeWordsDifferenceToF32(getTimeWords(a), getTimeWords(b));
}
`:`
fn getRowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}
`}function m(e){return e?`
fn getPlayheadWords() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}

fn getPlayheadFraction() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 2u]);
}

fn getPlayheadMaxGap() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}

fn isRowAfterPlayhead(row: u32) -> bool {
  return !isTimeWordsAtLeast(getPlayheadWords(), getPlayheadFraction(), getTimeWords(row), 0.0);
}

fn isRowBeforePlayhead(row: u32) -> bool {
  return !isTimeWordsAtLeast(getTimeWords(row), 0.0, getPlayheadWords(), getPlayheadFraction());
}

fn getPlayheadElapsed(row: u32) -> f32 {
  return timeWordsDifference(getPlayheadWords(), getPlayheadFraction(), getTimeWords(row), 0.0);
}
`:`
fn getPlayheadMaxGap() -> f32 {
  return parameters[parametersOffset + 1u];
}

fn isRowAfterPlayhead(row: u32) -> bool {
  return timestamps[timestampsOffset + row] > parameters[parametersOffset];
}

fn isRowBeforePlayhead(row: u32) -> bool {
  return timestamps[timestampsOffset + row] < parameters[parametersOffset];
}

fn getPlayheadElapsed(row: u32) -> f32 {
  return parameters[parametersOffset] - timestamps[timestampsOffset + row];
}
`}function h(e){return e?`
fn getClockOffset(firstRow: u32) -> f32 {
  let start = vec2<u32>(clockParameters[clockParametersOffset], clockParameters[clockParametersOffset + 1u]);
  return timeWordsDifference(start, bitcast<f32>(clockParameters[clockParametersOffset + 2u]), getTimeWords(firstRow), 0.0);
}

fn getClockStep() -> f32 {
  return bitcast<f32>(clockParameters[clockParametersOffset + 3u]);
}
`:`
fn getClockOffset(firstRow: u32) -> f32 {
  return clockParameters[clockParametersOffset] - timestamps[timestampsOffset + firstRow];
}

fn getClockStep() -> f32 {
  return clockParameters[clockParametersOffset + 1u];
}
`}var g=`
  let trackStart = min(trackOffsets[trackOffsetsOffset + track], ROW_COUNT);
  let trackEnd = clamp(trackOffsets[trackOffsetsOffset + track + 1u], trackStart, ROW_COUNT);`;function _(e,n){let r=n.timestamps.format===`uint32x2`;return t(e,{id:n.id,operation:c,variant:r?`bracket-words`:`bracket`,bindings:[f(n.timestamps),{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:r?`u32`:`f32`,access:`read`},{name:`bracket`,view:n.bracket,type:`u32`,access:`read_write`}],invocationCount:n.trackCount,declarations:`const ROW_COUNT: u32 = ${n.rowCount}u;
const BRACKET_STRIDE: u32 = 5u;
${u}
${p(r)}
${m(r)}`,body:`let track = index;
  ${g}
  var status = STATUS_EMPTY;
  var segmentRow = NO_ROW;
  var nextRow = NO_ROW;
  var fraction = 0.0;
  var duration = 0.0;
  if (trackEnd > trackStart) {
    let lastRow = trackEnd - 1u;
    let lastSegmentRow = select(trackStart, lastRow - 1u, lastRow > trackStart);
    if (isRowAfterPlayhead(trackStart)) {
      status = STATUS_BEFORE_START;
      segmentRow = trackStart;
      nextRow = min(trackStart + 1u, lastRow);
    } else if (isRowBeforePlayhead(lastRow)) {
      status = STATUS_AFTER_END;
      segmentRow = lastSegmentRow;
      nextRow = lastRow;
      fraction = 1.0;
    } else {
      // t[trackStart] <= playhead <= t[lastRow]: first row in (trackStart, trackEnd) after it.
      var low = trackStart + 1u;
      var high = trackEnd;
      loop {
        if (low >= high) {
          break;
        }
        let middle = (low + high) / 2u;
        if (isRowAfterPlayhead(middle)) {
          high = middle;
        } else {
          low = middle + 1u;
        }
      }
      status = STATUS_ACTIVE;
      if (low == trackEnd) {
        // The playhead equals the last sample time.
        segmentRow = lastSegmentRow;
        nextRow = lastRow;
        fraction = 1.0;
      } else {
        // t[low - 1] <= playhead < t[low], so the interval is strictly positive.
        segmentRow = low - 1u;
        nextRow = low;
        let interval = getRowTimeDifference(nextRow, segmentRow);
        fraction = clamp(getPlayheadElapsed(segmentRow) / interval, 0.0, 1.0);
        let maxGap = getPlayheadMaxGap();
        if (maxGap > 0.0 && interval > maxGap && isRowBeforePlayhead(segmentRow)) {
          status = STATUS_GAP;
          fraction = 0.0;
        }
      }
    }
    duration = getRowTimeDifference(nextRow, segmentRow);
  }
  let base = bracketOffset + BRACKET_STRIDE * track;
  bracket[base] = segmentRow;
  bracket[base + 1u] = nextRow;
  bracket[base + 2u] = status;
  bracket[base + 3u] = bitcast<u32>(fraction);
  bracket[base + 4u] = bitcast<u32>(duration);`})}function v(e,n){let r=[{name:`bracket`,view:n.bracket,type:`u32`,access:`read`},{name:`positions`,view:n.positions,type:`f32`,access:`read`}],i=(e,t,n)=>{t&&r.push({name:e,view:t,type:`f32`,access:n})},a=!!(n.elevations&&n.currentElevations);return i(`elevations`,a?n.elevations:void 0,`read`),i(`currentPositions`,n.currentPositions,`read_write`),i(`currentElevations`,a?n.currentElevations:void 0,`read_write`),i(`headings`,n.headings,`read_write`),i(`speeds`,n.speeds,`read_write`),t(e,{id:n.id,operation:c,variant:`geometry`,bindings:r,invocationCount:n.trackCount,declarations:`const BRACKET_STRIDE: u32 = 5u;
${u}
${d(n.geodesicRadius)}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`let base = bracketOffset + BRACKET_STRIDE * index;
  let segmentRow = bracket[base];
  let nextRow = bracket[base + 1u];
  let status = bracket[base + 2u];
  let fraction = bitcast<f32>(bracket[base + 3u]);
  let duration = bitcast<f32>(bracket[base + 4u]);
  var position = vec2<f32>(0.0, 0.0);
  var elevation = 0.0;
  var heading = 0.0;
  var speed = 0.0;
  if (status != STATUS_EMPTY) {
    let startPosition = getPosition(segmentRow);
    let endPosition = getPosition(nextRow);
    position = interpolatePosition(startPosition, endPosition, fraction);
    ${a?`let startElevation = elevations[elevationsOffset + segmentRow];
    let endElevation = elevations[elevationsOffset + nextRow];
    elevation = select(
      startElevation + (endElevation - startElevation) * fraction,
      endElevation,
      fraction >= 1.0
    );`:``}
    heading = getSegmentHeading(startPosition, endPosition);
    if (status == STATUS_ACTIVE && duration > 0.0) {
      speed = getSegmentLength(startPosition, endPosition) / duration;
    }
  }
  ${n.currentPositions?`currentPositions[currentPositionsOffset + 2u * index] = position.x;
  currentPositions[currentPositionsOffset + 2u * index + 1u] = position.y;`:``}
  ${a?`currentElevations[currentElevationsOffset + index] = elevation;`:``}
  ${n.headings?`headings[headingsOffset + index] = heading;`:``}
  ${n.speeds?`speeds[speedsOffset + index] = speed;`:``}
  _ = elevation;`})}function y(e,n){let r=[{name:`bracket`,view:n.bracket,type:`u32`,access:`read`}],i=[];for(let[e,t,a,o]of[[`status`,n.status,`u32`,`trackStatus`],[`segmentRows`,n.segmentRows,`u32`,`trackSegmentRow`],[`segmentFractions`,n.segmentFractions,`f32`,`trackFraction`],[`activeFlags`,n.activeFlags,`u32`,`select(0u, 1u, trackStatus == STATUS_ACTIVE)`],[`trackIndices`,n.trackIndices,`u32`,`index`]])t&&(r.push({name:e,view:t,type:a,access:`read_write`}),i.push(`${e}[${e}Offset + index] = ${o};`));return t(e,{id:n.id,operation:c,variant:`columns`,bindings:r,invocationCount:n.trackCount,declarations:`const BRACKET_STRIDE: u32 = 5u;
${u}`,body:`let base = bracketOffset + BRACKET_STRIDE * index;
  let trackSegmentRow = bracket[base];
  let trackStatus = bracket[base + 2u];
  let trackFraction = bitcast<f32>(bracket[base + 3u]);
  _ = trackSegmentRow;
  _ = trackFraction;
  _ = trackStatus;
  ${i.join(`
  `)}`})}function b(e,n){return t(e,{id:n.id,operation:l,variant:`arc-length`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`},{name:`cumulativeLengths`,view:n.cumulativeLengths,type:`f32`,access:`read_write`}],invocationCount:n.trackCount,declarations:`const ROW_COUNT: u32 = ${n.rowCount}u;
${d(n.geodesicRadius)}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`let track = index;
  ${g}
  if (trackEnd > trackStart) {
    var total = 0.0;
    cumulativeLengths[cumulativeLengthsOffset + trackStart] = 0.0;
    for (var row = trackStart + 1u; row < trackEnd; row++) {
      total = total + getSegmentLength(getPosition(row - 1u), getPosition(row));
      cumulativeLengths[cumulativeLengthsOffset + row] = total;
    }
  }`})}function x(e,n){let r=!!n.cumulativeLengths,i=!!n.clock,a=n.timestamps,o=a?.format===`uint32x2`,s=!!(n.elevations&&n.sampleElevations),c=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`trackOffsets`,view:n.trackOffsets,type:`u32`,access:`read`}];a&&c.push(f(a)),n.clock&&c.push({name:`clockParameters`,view:n.clock,type:o?`u32`:`f32`,access:`read`}),n.cumulativeLengths&&c.push({name:`cumulativeLengths`,view:n.cumulativeLengths,type:`f32`,access:`read`}),s&&n.elevations&&n.sampleElevations&&c.push({name:`elevations`,view:n.elevations,type:`f32`,access:`read`},{name:`sampleElevations`,view:n.sampleElevations,type:`f32`,access:`read_write`}),c.push({name:`samples`,view:n.samples,type:`f32`,access:`read_write`}),n.sampleTimes&&c.push({name:`sampleTimes`,view:n.sampleTimes,type:`f32`,access:`read_write`});let u=r?`fn getProgress(row: u32, firstRow: u32) -> f32 {
  _ = firstRow;
  return cumulativeLengths[cumulativeLengthsOffset + row];
}`:`fn getProgress(row: u32, firstRow: u32) -> f32 {
  return getRowTimeDifference(row, firstRow);
}`,m=n.sampleTimes?i?`time = f32(sampleIndex) * clockStep;`:r?`let startTime = getRowTimeDifference(row0, trackStart);
    let endTime = getRowTimeDifference(row1, trackStart);
    time = select(startTime + (endTime - startTime) * fraction, endTime, fraction >= 1.0);`:`time = targetProgress;`:``;return t(e,{id:n.id,operation:l,variant:i?o?`clock-words`:`clock`:r?`arc-length`:`time`,bindings:c,invocationCount:n.trackCount*n.sampleCount,declarations:`const ROW_COUNT: u32 = ${n.rowCount}u;
const SAMPLE_COUNT: u32 = ${n.sampleCount}u;
${a?p(o):``}
${i?h(o):``}
${d(n.geodesicRadius)}
${u}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`let track = index / SAMPLE_COUNT;
  let sampleIndex = index - track * SAMPLE_COUNT;
  ${g}
  var position = vec2<f32>(0.0, 0.0);
  var elevation = 0.0;
  var time = 0.0;
  ${i?`let clockStep = getClockStep();
  var isAbsent = trackEnd <= trackStart;`:``}
  if (trackEnd > trackStart) {
    let lastRow = trackEnd - 1u;
    let total = getProgress(lastRow, trackStart);
    var targetProgress = 0.0;
    ${i?`targetProgress = getClockOffset(trackStart) + f32(sampleIndex) * clockStep;
    isAbsent = !(targetProgress >= 0.0 && targetProgress <= total);`:``}
    if (${i?`false`:`SAMPLE_COUNT > 1u`}) {
      targetProgress = select(
        total * (f32(sampleIndex) / f32(SAMPLE_COUNT - 1u)),
        total,
        sampleIndex == SAMPLE_COUNT - 1u
      );
    }
    var low = trackStart + 1u;
    var high = trackEnd;
    loop {
      if (low >= high) {
        break;
      }
      let middle = (low + high) / 2u;
      if (getProgress(middle, trackStart) > targetProgress) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    // Past the last row: the end of the last segment, so a great-circle track keeps its longitude
    // continuous across the antimeridian (planar: exactly the last row).
    var row0 = select(lastRow, lastRow - 1u, lastRow > trackStart);
    var row1 = lastRow;
    var fraction = 1.0;
    if (low < trackEnd) {
      row0 = low - 1u;
      row1 = low;
      let startProgress = getProgress(row0, trackStart);
      fraction = clamp(
        (targetProgress - startProgress) / (getProgress(row1, trackStart) - startProgress),
        0.0,
        1.0
      );
    }
    let startPosition = getPosition(row0);
    let endPosition = getPosition(row1);
    position = interpolatePosition(startPosition, endPosition, fraction);
    ${s?`let startElevation = elevations[elevationsOffset + row0];
    let endElevation = elevations[elevationsOffset + row1];
    elevation = select(
      startElevation + (endElevation - startElevation) * fraction,
      endElevation,
      fraction >= 1.0
    );`:``}
    ${m}
  }
  ${i?`if (isAbsent) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    position = vec2<f32>(bitcast<f32>(0x7fc00000u | (index & 0u)));
    elevation = position.x;
  }`:``}
  samples[samplesOffset + 2u * index] = position.x;
  samples[samplesOffset + 2u * index + 1u] = position.y;
  ${s?`sampleElevations[sampleElevationsOffset + index] = elevation;`:``}
  ${n.sampleTimes?`sampleTimes[sampleTimesOffset + index] = time;`:``}
  _ = elevation;
  _ = time;`})}export{x as a,v as i,_ as n,i as o,y as r,a as s,b as t};