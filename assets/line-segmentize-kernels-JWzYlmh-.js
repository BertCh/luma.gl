import{J as e}from"./src-CNCKFW2b.js";import{l as t,s as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as r}from"./lon-lat-ring-wgsl-C5XyUe9j.js";import{t as i}from"./geodesic-wgsl-DS0CJZ42.js";function a(e){return`
const BALANCED_ROWS: u32 = ${e}u;
const BALANCED_LANES: u32 = 128u;
const BALANCED_NONE: u32 = 0xffffffffu;
var<workgroup> balancedWindow: array<u32, 2>;

// Last row in [low, high] whose output range starts at or before the slot.
fn balancedLastAtMost(slot: u32, lowRow: u32, highRow: u32) -> u32 {
  var low = lowRow;
  var high = highRow;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (balancedStart(middle) <= slot) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  return low;
}

fn findBalancedOwner(slot: u32, firstSlot: u32, lane: u32) -> u32 {
  if (lane == 0u) {
    let lowRow = balancedLastAtMost(firstSlot, 0u, BALANCED_ROWS - 1u);
    balancedWindow[0] = lowRow;
    balancedWindow[1] = balancedLastAtMost(firstSlot + (BALANCED_LANES - 1u), lowRow, BALANCED_ROWS - 1u);
  }
  let windowLow = workgroupUniformLoad(&balancedWindow[0]);
  let windowHigh = workgroupUniformLoad(&balancedWindow[1]);
  let total = balancedStart(BALANCED_ROWS - 1u) + balancedCount(BALANCED_ROWS - 1u);
  if (slot >= total) {
    return BALANCED_NONE;
  }
  return balancedLastAtMost(slot, windowLow, windowHigh);
}
`}var o=`
const NO_LINE: u32 = 0xffffffffu;

fn findLine(row: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > LINE_COUNT) {
    return NO_LINE;
  }
  return low - 1u;
}
`;function s(e,n){let r=e===`spherical`?`return geodesicCentralAngle(a, b) * RADIUS;`:`return length(b - a);`;return`
${e===`spherical`?i:``}
const RADIUS: f32 = ${t(n)};

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getSegmentLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  ${r}
}
`}var c=64;function l(t,i){e(!(i.rowTurns&&i.rowShifts));let a=i.rowTurns??i.rowShifts,o=i.rowTurns?`i32`:`f32`,l=i.rowTurns?`0i`:`0.0`,u=i.rowTurns?`-i32(round((current.x - previous.x) / 360.0))`:`-360.0 * round((current.x - previous.x) / 360.0)`,d=i.polygonEdges?`select(${u}, ${l}, lonLatIsPoleEdge(previous, current))`:u,f=i.rowTurns?`turnCarry[0]`:`prefixCarry[2]`,p=i.lineOffsets.length-1,m=[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:i.lineOffsets,type:`u32`,access:`read`}];i.rowMeasures&&m.push({name:`rowMeasures`,view:i.rowMeasures,type:`f32`,access:`read_write`}),a&&m.push({name:`rowShifts`,view:a,type:o,access:`read_write`});let{rowMeasures:h}=i,g=a;return n(t,{id:i.id,operation:i.operation,variant:`line-prefix`,bindings:m,invocationCount:Math.ceil(p/c)*c,workgroupSize:c,guardIndex:!1,declarations:`const LINE_COUNT: u32 = ${p}u;
const ROW_COUNT: u32 = ${i.positions.length}u;
const PREFIX_LANES: u32 = ${c}u;
var<workgroup> largeBits: array<atomic<u32>, 2>;
var<workgroup> largeBitsCopy: array<u32, 2>;
var<workgroup> largeStart: array<u32, ${c}>;
var<workgroup> largeEnd: array<u32, ${c}>;
var<workgroup> prefixCarry: array<f32, 3>;
var<workgroup> tileLengths: array<f32, ${c}>;
var<workgroup> tileShifts: array<${o}, ${c}>;
${i.rowTurns?`var<workgroup> turnCarry: array<i32, 1>;`:``}
${s(i.coordinateSystem,i.radius)}
${i.polygonEdges?r:``}`,body:`let lane = localInvocationIndex;
  let lineBase = workgroupIndex * PREFIX_LANES;
  let ownLine = lineBase + lane;
  // Phase 1: this lane's own line, serially when small; larger lines are queued for phase 2.
  if (ownLine < LINE_COUNT) {
    let first = min(lineOffsets[lineOffsetsOffset + ownLine], ROW_COUNT);
    let last = min(max(lineOffsets[lineOffsetsOffset + ownLine + 1u], first), ROW_COUNT);
    if (last > first) {
      if (last - first <= PREFIX_LANES) {
        var sum = 0.0;
        var compensation = 0.0;
        var shift = ${l};
        var previous = getPosition(first);
        ${h?`rowMeasures[rowMeasuresOffset + first] = 0.0;`:``}
        ${g?`rowShifts[rowShiftsOffset + first] = ${l};`:``}
        for (var row = first + 1u; row < last; row++) {
          let current = getPosition(row);
          ${h?`let segmentLength = getSegmentLength(previous, current);
          let nextSum = sum + segmentLength;
          if (abs(sum) >= abs(segmentLength)) {
            compensation = compensation + ((sum - nextSum) + segmentLength);
          } else {
            compensation = compensation + ((segmentLength - nextSum) + sum);
          }
          sum = nextSum;
          rowMeasures[rowMeasuresOffset + row] = sum + compensation;`:``}
          ${g?`shift = shift + ${d};
          rowShifts[rowShiftsOffset + row] = shift;`:``}
          previous = current;
        }
      } else {
        largeStart[lane] = first;
        largeEnd[lane] = last;
        atomicOr(&largeBits[lane / 32u], 1u << (lane % 32u));
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    largeBitsCopy[0] = atomicLoad(&largeBits[0]);
    largeBitsCopy[1] = atomicLoad(&largeBits[1]);
  }
  // Phase 2: every barrier below is in workgroup-uniform control flow (the queue is read through
  // workgroupUniformLoad), and the queued lines are visited in ascending order.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&largeBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let lineStart = workgroupUniformLoad(&largeStart[slot]);
      let lineEnd = workgroupUniformLoad(&largeEnd[slot]);
      if (lane == 0u) {
        ${h?`rowMeasures[rowMeasuresOffset + lineStart] = 0.0;`:``}
        ${g?`rowShifts[rowShiftsOffset + lineStart] = ${l};`:``}
        prefixCarry[0] = 0.0;
        prefixCarry[1] = 0.0;
        ${f} = ${l};
      }
      for (var tileStart = lineStart + 1u; tileStart < lineEnd; tileStart += PREFIX_LANES) {
        let row = tileStart + lane;
        var segmentLength = 0.0;
        var segmentShift = ${l};
        if (row < lineEnd) {
          let previous = getPosition(row - 1u);
          let current = getPosition(row);
          ${h?`segmentLength = getSegmentLength(previous, current);`:``}
          ${g?`segmentShift = ${d};`:``}
        }
        tileLengths[lane] = segmentLength;
        tileShifts[lane] = segmentShift;
        workgroupBarrier();
        for (var offset = 1u; offset < PREFIX_LANES; offset = offset << 1u) {
          var addedLength = 0.0;
          var addedShift = ${l};
          if (lane >= offset) {
            addedLength = tileLengths[lane - offset];
            addedShift = tileShifts[lane - offset];
          }
          workgroupBarrier();
          tileLengths[lane] += addedLength;
          tileShifts[lane] += addedShift;
          workgroupBarrier();
        }
        if (row < lineEnd) {
          ${h?`rowMeasures[rowMeasuresOffset + row] = (prefixCarry[0] + tileLengths[lane]) + prefixCarry[1];`:``}
          ${g?`rowShifts[rowShiftsOffset + row] = ${f} + tileShifts[lane];`:``}
        }
        workgroupBarrier();
        if (lane == 0u) {
          // Neumaier-compensated carry of the tile total into the running measure.
          let total = tileLengths[PREFIX_LANES - 1u];
          let sum = prefixCarry[0];
          let nextSum = sum + total;
          if (abs(sum) >= abs(total)) {
            prefixCarry[1] += (sum - nextSum) + total;
          } else {
            prefixCarry[1] += (total - nextSum) + sum;
          }
          prefixCarry[0] = nextSum;
          ${f} += tileShifts[PREFIX_LANES - 1u];
        }
        workgroupBarrier();
      }
    }
  }`})}function u(t,r){let i=r.parameterIndex??0;e(Number.isInteger(i)&&i>=0&&i<r.parameters.length);let a=(r.interpolation??(r.coordinateSystem===`spherical`?`great-circle`:`planar`))===`great-circle`,c=r.lineOffsets.length-1;return n(t,{id:r.id,operation:r.operation,variant:`count`,bindings:[{name:`positions`,view:r.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:r.lineOffsets,type:`u32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`counts`,view:r.counts,type:`u32`,access:`read_write`},...r.approximation?[{name:`approximation`,view:r.approximation,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:r.positions.length,declarations:`const LINE_COUNT: u32 = ${c}u;
const MAXIMUM_PIECES: u32 = ${r.maximumPieceCount}u;
${o}
${s(a||r.polygonEdges?`spherical`:r.coordinateSystem,r.radius)}

fn getPieceExtent(start: vec2<f32>, end: vec2<f32>) -> f32 {
  ${r.pieceCountModel===`degree-step`?`let rawLongitude = end.x - start.x;
  let longitudeStep = rawLongitude - 360.0 * round(rawLongitude / 360.0);
  return max(abs(longitudeStep), ${a||r.polygonEdges?`geodesicCentralAngle(start, end) * GEODESIC_RADIANS_TO_DEGREES`:`abs(end.y - start.y)`});`:`return getSegmentLength(start, end);`}
}

fn getPieceCount(segmentLength: f32, maximumLength: f32) -> u32 {
  if (!(maximumLength > 0.0) || !(segmentLength > 0.0)) {
    return 1u;
  }
  let requestedPieces = ceil(segmentLength / maximumLength);
  ${r.approximation?`if (requestedPieces > f32(MAXIMUM_PIECES)) {
    atomicStore(&approximation[approximationOffset], 1u);
  }`:``}
  let pieces = min(requestedPieces, f32(MAXIMUM_PIECES));
  return clamp(u32(pieces), 1u, MAXIMUM_PIECES);
}`,body:`let line = findLine(index);
  var count = 0u;
  if (line != NO_LINE) {
    let first = lineOffsets[lineOffsetsOffset + line];
    let last = min(lineOffsets[lineOffsetsOffset + line + 1u], ${r.positions.length}u);
    ${r.closedLines?`if (last > first + 1u && index + 1u == last &&
        all(bitcast<vec2<u32>>(getPosition(index)) == bitcast<vec2<u32>>(getPosition(first)))) {
      counts[countsOffset + index] = 0u;
      return;
    }`:``}
    if (index + 1u >= last ${r.closedLines?`&& last <= first + 1u`:``}) {
      count = 1u;
    } else {
      count = getPieceCount(
        getPieceExtent(getPosition(index), getPosition(${r.closedLines?`select(index + 1u, first, index + 1u >= last)`:`index + 1u`})),
        parameters[parametersOffset + ${i}u]
      );
    }
  }
  counts[countsOffset + index] = count;`})}function d(t,i){e(!i.closedLines||i.lineOffsets),e(!i.outputTurns||i.outputTurns.length>=i.outputPositions.length);let c=i.interpolation??(i.coordinateSystem===`spherical`?`great-circle`:`planar`),l=c===`great-circle`,u=c!==`planar`,d=!!(i.wrapOutputLongitudes||i.outputTurns),f=i.rowTurns??i.rowShifts,p=[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`counts`,view:i.counts,type:`u32`,access:`read`},{name:`starts`,view:i.starts,type:`u32`,access:`read`},{name:`outputPositions`,view:i.outputPositions,type:`f32`,access:`read_write`}];i.closedLines&&i.lineOffsets&&p.push({name:`lineOffsets`,view:i.lineOffsets,type:`u32`,access:`read`}),i.outputTurns&&p.push({name:`outputTurns`,view:i.outputTurns,type:`i32`,access:`read_write`}),u&&f&&p.push({name:`rowShifts`,view:f,type:i.rowTurns?`i32`:`f32`,access:`read`}),i.outputMeasures&&i.rowMeasures&&p.push({name:`rowMeasures`,view:i.rowMeasures,type:`f32`,access:`read`},{name:`outputMeasures`,view:i.outputMeasures,type:`f32`,access:`read_write`}),i.outputSourceRows&&p.push({name:`outputSourceRows`,view:i.outputSourceRows,type:`u32`,access:`read_write`});let m=!!(i.outputMeasures&&i.rowMeasures),h=l?`geodesicInterpolate(start, end, angle, fraction)`:c===`linear-degrees`?`vec2<f32>(start.x + (end.x - start.x - 360.0 * round((end.x - start.x) / 360.0)) * fraction, mix(start.y, end.y, fraction))`:`mix(start, end, fraction)`,g=i.polygonEdges?`interpolatePolygonEdge(start, end, fraction)`:h;return n(t,{id:i.id,operation:i.operation,variant:`emit`,bindings:p,invocationCount:i.outputPositions.length,workgroupSize:128,guardIndex:!1,declarations:`const CAPACITY: u32 = ${i.outputPositions.length}u;
${s(l?`spherical`:i.coordinateSystem,i.radius)}
${i.polygonEdges?`${r}
fn interpolatePolygonEdge(start: vec2<f32>, end: vec2<f32>, fraction: f32) -> vec2<f32> {
  if (lonLatIsPoleEdge(start, end)) { return mix(start, end, fraction); }
  ${l?`let angle = geodesicCentralAngle(start, end);`:``}
  return ${h};
}`:``}
${i.closedLines?`const LINE_COUNT: u32 = ${i.lineOffsets.length-1}u;
${o}`:``}
${a(i.positions.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32, prefixTurn: i32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  var longitude = vertex.x;
  var turn = prefixTurn;
  ${d?`// Preserve canonical input endpoints exactly, including signed zero and both seam signs.
  if (longitude < -180.0 || longitude > 180.0) {
    let localTurn = i32(floor((longitude + 180.0) / 360.0));
    longitude -= 360.0 * f32(localTurn);
    turn += localTurn;
  }`:``}
  outputPositions[outputPositionsOffset + 2u * outputRow] = longitude;
  ${i.outputTurns?`outputTurns[outputTurnsOffset + outputRow] = turn;`:``}
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${m?`outputMeasures[outputMeasuresOffset + outputRow] = measure;`:``}
  ${i.outputSourceRows?`outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;`:``}
}`,body:`let row = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (row == BALANCED_NONE) {
    return;
  }
  let count = counts[countsOffset + row];
  let piece = index - starts[startsOffset + row];
  var start = getPosition(row);
  let prefixTurn = ${u&&f?i.rowTurns?`rowShifts[rowShiftsOffset + row]`:`i32(rowShifts[rowShiftsOffset + row] / 360.0)`:`0i`};
  ${!d&&u&&f?i.rowTurns?`start.x += 360.0 * f32(prefixTurn);`:`start.x = start.x + rowShifts[rowShiftsOffset + row];`:``}
  let measure = ${m?`rowMeasures[rowMeasuresOffset + row]`:`0.0`};
  if (piece == 0u) {
    writeVertex(index, start, measure, row, prefixTurn);
    return;
  }
  ${i.closedLines?`let line = findLine(row);
  let first = lineOffsets[lineOffsetsOffset + line];
  let last = min(lineOffsets[lineOffsetsOffset + line + 1u], ${i.positions.length}u);
  let next = select(row + 1u, first, row + 1u >= last);`:`let next = row + 1u;`}
  let end = getPosition(next);
  let segmentLength = getSegmentLength(start, end);
  ${l?`let angle = geodesicCentralAngle(start, end);`:``}
  let fraction = f32(piece) / f32(count);
  writeVertex(index, ${g}, measure + segmentLength * fraction, row, prefixTurn);`})}function f(e,t){return n(e,{id:t.id,operation:t.operation,variant:`source-rows`,bindings:[{name:`counts`,view:t.counts,type:`u32`,access:`read`},{name:`starts`,view:t.starts,type:`u32`,access:`read`},{name:`outputSourceRows`,view:t.outputSourceRows,type:`u32`,access:`read_write`}],invocationCount:t.outputSourceRows.length,workgroupSize:128,guardIndex:!1,declarations:`${a(t.counts.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }`,body:`let row = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (index < ${t.outputSourceRows.length}u && row != BALANCED_NONE) {
    outputSourceRows[outputSourceRowsOffset + index] = row;
  }`})}var p=`
fn getSource(row: u32) -> vec2<f32> {
  return vec2<f32>(sources[sourcesOffset + 2u * row], sources[sourcesOffset + 2u * row + 1u]);
}

fn getTarget(row: u32) -> vec2<f32> {
  return vec2<f32>(targets[targetsOffset + 2u * row], targets[targetsOffset + 2u * row + 1u]);
}
`;function m(e,r){return n(e,{id:r.id,operation:r.operation,variant:`count`,bindings:[{name:`sources`,view:r.sources,type:`f32`,access:`read`},{name:`targets`,view:r.targets,type:`f32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`counts`,view:r.counts,type:`u32`,access:`read_write`}],invocationCount:r.sources.length,declarations:`${i}
const RADIUS: f32 = ${t(r.radius)};
const MAXIMUM_SEGMENTS: u32 = ${r.maximumSegmentCount}u;
${p}`,body:`let distance = geodesicCentralAngle(getSource(index), getTarget(index)) * RADIUS;
  let maximumLength = parameters[parametersOffset];
  let minimumSegments = parameters[parametersOffset + 1u];
  var segments = 1.0;
  if (maximumLength > 0.0 && distance > 0.0) {
    segments = ceil(min(distance / maximumLength, f32(MAXIMUM_SEGMENTS)));
  }
  if (minimumSegments > segments) {
    segments = min(round(minimumSegments), f32(MAXIMUM_SEGMENTS));
  }
  counts[countsOffset + index] = clamp(u32(segments), 1u, MAXIMUM_SEGMENTS) + 1u;`})}function h(e,r){let o=[{name:`sources`,view:r.sources,type:`f32`,access:`read`},{name:`targets`,view:r.targets,type:`f32`,access:`read`},{name:`counts`,view:r.counts,type:`u32`,access:`read`},{name:`starts`,view:r.starts,type:`u32`,access:`read`},{name:`outputPositions`,view:r.outputPositions,type:`f32`,access:`read_write`}];return r.outputMeasures&&o.push({name:`outputMeasures`,view:r.outputMeasures,type:`f32`,access:`read_write`}),r.outputSourceRows&&o.push({name:`outputSourceRows`,view:r.outputSourceRows,type:`u32`,access:`read_write`}),n(e,{id:r.id,operation:r.operation,variant:`emit`,bindings:o,invocationCount:r.outputPositions.length,workgroupSize:128,guardIndex:!1,declarations:`${i}
const RADIUS: f32 = ${t(r.radius)};
const CAPACITY: u32 = ${r.outputPositions.length}u;
${p}
${a(r.sources.length)}
fn balancedStart(row: u32) -> u32 { return starts[startsOffset + row]; }
fn balancedCount(row: u32) -> u32 { return counts[countsOffset + row]; }

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = vertex.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${r.outputMeasures?`outputMeasures[outputMeasuresOffset + outputRow] = measure;`:``}
  ${r.outputSourceRows?`outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;`:``}
}`,body:`let pair = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (pair == BALANCED_NONE) {
    return;
  }
  let segments = counts[countsOffset + pair] - 1u;
  let vertex = index - starts[startsOffset + pair];
  let source = getSource(pair);
  if (vertex == 0u) {
    writeVertex(index, source, 0.0, pair);
    return;
  }
  let destination = getTarget(pair);
  // Measure and interpolate from the original endpoints. Reconstructing an unwrapped
  // target first rounds away short antimeridian separations even with a stable angle.
  let angle = geodesicCentralAngle(source, destination);
  if (vertex == segments) {
    let unwrapped = vec2<f32>(source.x + geodesicLongitudeDelta(source.x, destination.x), destination.y);
    writeVertex(index, unwrapped, angle * RADIUS, pair);
    return;
  }
  let fraction = f32(vertex) / f32(segments);
  writeVertex(
    index,
    geodesicInterpolate(source, destination, angle, fraction),
    angle * RADIUS * fraction,
    pair
  );`})}function g(e,t){let r=[];t.lineOffsets&&r.push({name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`}),r.push({name:`counts`,view:t.counts,type:`u32`,access:`read`},{name:`starts`,view:t.starts,type:`u32`,access:`read`},{name:`outputLineOffsets`,view:t.outputLineOffsets,type:`u32`,access:`read_write`},{name:`countOut`,view:t.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:t.overflow,type:`u32`,access:`read_write`}),t.requiredCount&&r.push({name:`totalOut`,view:t.requiredCount,type:`u32`,access:`read_write`}),t.lineCountOutput&&r.push({name:`lineCountOut`,view:t.lineCountOutput,type:`u32`,access:`read_write`});let i=t.lineOffsets?`min(lineOffsets[lineOffsetsOffset + index], ROW_COUNT)`:`min(index, ROW_COUNT)`;return n(e,{id:t.id,operation:t.operation,variant:`publish`,bindings:r,invocationCount:t.lineCount+1,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const CAPACITY: u32 = ${t.capacity}u;
const LINE_COUNT: u32 = ${t.lineCount}u;`,body:`let total = starts[startsOffset + ROW_COUNT - 1u] + counts[countsOffset + ROW_COUNT - 1u];
  let row = ${i};
  var start = total;
  if (row < ROW_COUNT) {
    start = starts[startsOffset + row];
  }
  outputLineOffsets[outputLineOffsetsOffset + index] = min(start, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${t.requiredCount?`totalOut[totalOutOffset] = total;`:``}
    ${t.lineCountOutput?`lineCountOut[lineCountOutOffset] = LINE_COUNT;`:``}
  }`})}function _(e,t){return n(e,{id:t.id,operation:t.operation,variant:`source-line-indexs`,bindings:[{name:`sourceLineIndices`,view:t.sourceLineIndices,type:`u32`,access:`read_write`}],invocationCount:t.sourceLineIndices.length,body:`sourceLineIndices[sourceLineIndicesOffset + index] = index;`})}export{g as a,d as c,_ as i,f as l,m as n,l as o,h as r,u as s,o as t,a as u};