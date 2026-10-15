import{s as e}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t}from"./line-segmentize-kernels-JWzYlmh-.js";var n=4294967295,r=`
${t}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getClip(row: u32) -> vec3<f32> {
  let line = findLine(row);
  if (line == NO_LINE || row + 1u >= min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT)) {
    return vec3<f32>(0.0);
  }
  let a = getPosition(row);
  let d = getPosition(row + 1u) - a;
  let lower = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let upper = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  var t0 = 0.0;
  var t1 = 1.0;
  var p = array<f32, 4>(-d.x, d.x, -d.y, d.y);
  var q = array<f32, 4>(a.x - lower.x, upper.x - a.x, a.y - lower.y, upper.y - a.y);
  for (var side = 0u; side < 4u; side++) {
    if (p[side] == 0.0) {
      if (q[side] < 0.0) {
        return vec3<f32>(0.0);
      }
    } else {
      let r = q[side] / p[side];
      if (p[side] < 0.0) {
        if (r > t1) {
          return vec3<f32>(0.0);
        }
        t0 = max(t0, r);
      } else {
        if (r < t0) {
          return vec3<f32>(0.0);
        }
        t1 = min(t1, r);
      }
    }
  }
  // A single touching point is not a segment, except a genuinely zero-length input segment.
  if (t1 < t0 || (t1 == t0 && (d.x != 0.0 || d.y != 0.0))) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(t0, t1, 1.0);
}

fn getLineStart(row: u32) -> u32 {
  return lineOffsets[lineOffsetsOffset + findLine(row)];
}

fn isContinuation(row: u32, clip: vec3<f32>) -> bool {
  if (clip.z == 0.0 || row == 0u || row <= getLineStart(row)) {
    return false;
  }
  let previous = getClip(row - 1u);
  return previous.z != 0.0 && previous.y == 1.0 && clip.x == 0.0;
}

fn getClippedStart(row: u32, clip: vec3<f32>) -> vec2<f32> {
  let a = getPosition(row);
  return select(a + (getPosition(row + 1u) - a) * clip.x, a, clip.x == 0.0);
}

fn getClippedEnd(row: u32, clip: vec3<f32>) -> vec2<f32> {
  let a = getPosition(row);
  let b = getPosition(row + 1u);
  return select(a + (b - a) * clip.y, b, clip.y == 1.0);
}
`;function i(e){return`const LINE_COUNT: u32 = ${e.lineOffsets.length-1}u;
const ROW_COUNT: u32 = ${e.positions.length}u;
${r}`}function a(e){return[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:e.lineOffsets,type:`u32`,access:`read`},{name:`parameters`,view:e.parameters,type:`f32`,access:`read`}]}function o(t,n){return e(t,{id:n.id,operation:n.operation,variant:`line-count`,bindings:[...a(n),{name:`counts`,view:n.counts,type:`u32`,access:`read_write`},{name:`startFlags`,view:n.startFlags,type:`u32`,access:`read_write`}],invocationCount:n.positions.length,declarations:i(n),body:`let clip = getClip(index);
  let continues = isContinuation(index, clip);
  counts[countsOffset + index] = select(0u, select(2u, 1u, continues), clip.z != 0.0);
  startFlags[startFlagsOffset + index] = select(0u, 1u, clip.z != 0.0 && !continues);`})}function s(t,n){let r=n.outputPositions.length,o=n.outputLineOffsets.length-1,s=[...a(n),{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`lineIndices`,view:n.lineIndices,type:`u32`,access:`read`},{name:`outPositions`,view:n.outputPositions,type:`f32`,access:`read_write`},{name:`outLineOffsets`,view:n.outputLineOffsets,type:`u32`,access:`read_write`}];return n.outputSourceLineIndices&&s.push({name:`outSourceLineIndices`,view:n.outputSourceLineIndices,type:`u32`,access:`read_write`}),e(t,{id:n.id,operation:n.operation,variant:`line-emit`,bindings:s,invocationCount:n.positions.length,declarations:`${i(n)}
const CAPACITY: u32 = ${r}u;
const LINE_CAPACITY: u32 = ${o}u;
fn writeVertex(slot: u32, p: vec2<f32>) {
  if (slot < CAPACITY) {
    outPositions[outPositionsOffset + 2u * slot] = p.x;
    outPositions[outPositionsOffset + 2u * slot + 1u] = p.y;
  }
}`,body:`let clip = getClip(index);
  if (clip.z == 0.0) {
    return;
  }
  let continues = isContinuation(index, clip);
  var slot = starts[startsOffset + index];
  if (!continues) {
    let lineIndex = lineIndices[lineIndicesOffset + index];
    if (lineIndex < LINE_CAPACITY) {
      outLineOffsets[outLineOffsetsOffset + lineIndex] = min(slot, CAPACITY);
      ${n.outputSourceLineIndices?`outSourceLineIndices[outSourceLineIndicesOffset + lineIndex] = findLine(index);`:``}
    }
    writeVertex(slot, getClippedStart(index, clip));
    slot += 1u;
  }
  writeVertex(slot, getClippedEnd(index, clip));`})}function c(t,n){let r=[{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`counts`,view:n.counts,type:`u32`,access:`read`},{name:`lineIndices`,view:n.lineIndices,type:`u32`,access:`read`},{name:`startFlags`,view:n.startFlags,type:`u32`,access:`read`},{name:`totals`,view:n.totals,type:`u32`,access:`read_write`},{name:`countOut`,view:n.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:n.overflow,type:`u32`,access:`read_write`}];n.requiredCount&&r.push({name:`totalOut`,view:n.requiredCount,type:`u32`,access:`read_write`});let i=n.starts.length-1;return e(t,{id:n.id,operation:n.operation,variant:`line-publish`,bindings:r,invocationCount:1,body:`let vertexTotal = starts[startsOffset + ${i}u] + counts[countsOffset + ${i}u];
  let lineTotal = lineIndices[lineIndicesOffset + ${i}u] + startFlags[startFlagsOffset + ${i}u];
  let lineClamped = min(lineTotal, ${n.lineCapacity}u);
  totals[totalsOffset] = min(vertexTotal, ${n.capacity}u);
  totals[totalsOffset + 1u] = lineClamped;
  countOut[countOutOffset] = min(vertexTotal, ${n.capacity}u);
  overflowOut[overflowOutOffset] = select(0u, 1u, vertexTotal > ${n.capacity}u || lineTotal > ${n.lineCapacity}u);
  ${n.requiredCount?`totalOut[totalOutOffset] = vertexTotal;`:``}`})}function l(t,r){let i=[{name:`totals`,view:r.totals,type:`u32`,access:`read`},{name:`lineOffsets`,view:r.lineOffsets,type:`u32`,access:`read_write`}];return r.lineCount&&i.push({name:`lineCountOut`,view:r.lineCount,type:`u32`,access:`read_write`}),r.sourceLineIndices&&i.push({name:`sourceLineIndices`,view:r.sourceLineIndices,type:`u32`,access:`read_write`}),e(t,{id:r.id,operation:r.operation,variant:`line-tail`,bindings:i,invocationCount:r.lineOffsets.length,body:`${r.lineCount?`if (index == 0u) {
    lineCountOut[lineCountOutOffset] = totals[totalsOffset + 1u];
  }`:``}
  if (index >= totals[totalsOffset + 1u]) {
    lineOffsets[lineOffsetsOffset + index] = totals[totalsOffset];
    ${r.sourceLineIndices?`if (index < ${r.sourceLineIndices.length}u) {\n      sourceLineIndices[sourceLineIndicesOffset + index] = ${n}u;\n    }`:``}
  }`})}var u=`
${t}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

const BOUND_SLOTS = array<u32, 4>(0u, 2u, 1u, 3u);

fn getBound() -> f32 {
  // left x >= minX, right x <= maxX, bottom y >= minY, top y <= maxY
  return parameters[parametersOffset + BOUND_SLOTS[BOUNDARY]];
}

fn isInside(p: vec2<f32>) -> bool {
  let bound = getBound();
  switch (BOUNDARY) {
    case 0u: { return p.x >= bound; }
    case 1u: { return p.x <= bound; }
    case 2u: { return p.y >= bound; }
    default: { return p.y <= bound; }
  }
}

fn getIntersection(s: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
  let bound = getBound();
  if (BOUNDARY < 2u) {
    let t = (bound - s.x) / (e.x - s.x);
    return vec2<f32>(bound, s.y + t * (e.y - s.y));
  }
  let t = (bound - s.y) / (e.y - s.y);
  return vec2<f32>(s.x + t * (e.x - s.x), bound);
}

// Previous vertex of the ring that owns \`row\`, or false when the row is outside every ring.
fn getRingPrevious(row: u32) -> vec3<f32> {
  let ring = findLine(row);
  if (ring == NO_LINE) {
    return vec3<f32>(0.0);
  }
  let start = lineOffsets[lineOffsetsOffset + ring];
  let end = min(lineOffsets[lineOffsetsOffset + ring + 1u], INPUT_ROWS);
  let previous = getPosition(select(row - 1u, end - 1u, row == start));
  return vec3<f32>(previous, 1.0);
}
`;function d(e){return`const BOUNDARY: u32 = ${e.boundary}u;
const LINE_COUNT: u32 = ${e.ringOffsets.length-1}u;
const INPUT_ROWS: u32 = ${e.positions.length}u;
${e.positionTurns?u.replace(`positions[positionsOffset + 2u * row],`,`positions[positionsOffset + 2u * row] + 360.0 * f32(positionTurns[positionTurnsOffset + row] - ${e.longitudeCopy??0}i),`):u}`}function f(e){return[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:e.ringOffsets,type:`u32`,access:`read`},{name:`parameters`,view:e.parameters,type:`f32`,access:`read`},...e.positionTurns?[{name:`positionTurns`,view:e.positionTurns,type:`i32`,access:`read`}]:[]]}function p(t,n){return e(t,{id:n.id,operation:n.operation,variant:`polygon-count`,bindings:[...f(n),{name:`counts`,view:n.counts,type:`u32`,access:`read_write`}],invocationCount:n.positions.length,declarations:d(n),body:`let previous = getRingPrevious(index);
  var count = 0u;
  if (previous.z != 0.0) {
    let endInside = isInside(getPosition(index));
    let startInside = isInside(previous.xy);
    count = select(select(0u, 1u, startInside), select(2u, 1u, startInside), endInside);
  }
  counts[countsOffset + index] = count;`})}function m(t,n){let r=n.outputPositions.length;return e(t,{id:n.id,operation:n.operation,variant:`polygon-emit`,bindings:[...f(n),{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`outPositions`,view:n.outputPositions,type:`f32`,access:`read_write`},...n.outputTurns?[{name:`outputTurns`,view:n.outputTurns,type:`i32`,access:`read_write`}]:[]],invocationCount:n.positions.length,declarations:`${d(n)}
const CAPACITY: u32 = ${r}u;
fn writeVertex(slot: u32, p: vec2<f32>) {
  if (slot < CAPACITY) {
    ${n.outputTurns?`let turn = select(i32(floor((p.x + 180.0) / 360.0)), 0i, abs(p.x) <= 180.0);
    outPositions[outPositionsOffset + 2u * slot] = p.x - 360.0 * f32(turn);
    outputTurns[outputTurnsOffset + slot] = turn;`:`outPositions[outPositionsOffset + 2u * slot] = p.x;`}
    outPositions[outPositionsOffset + 2u * slot + 1u] = p.y;
  }
}
fn writeKeptVertex(slot: u32, row: u32, position: vec2<f32>) {
  ${n.positionTurns&&n.outputTurns?`if (slot < CAPACITY) {
    // Copy the original bits; reconstructing then wrapping a kept vertex loses precision.
    outPositions[outPositionsOffset + 2u * slot] = positions[positionsOffset + 2u * row];
    outPositions[outPositionsOffset + 2u * slot + 1u] = positions[positionsOffset + 2u * row + 1u];
    outputTurns[outputTurnsOffset + slot] = positionTurns[positionTurnsOffset + row] - ${n.longitudeCopy??0}i;
  }`:`writeVertex(slot, position);`}
}`,body:`let previous = getRingPrevious(index);
  if (previous.z == 0.0) {
    return;
  }
  let end = getPosition(index);
  let endInside = isInside(end);
  let startInside = isInside(previous.xy);
  var slot = starts[startsOffset + index];
  if (endInside) {
    if (!startInside) {
      writeVertex(slot, getIntersection(previous.xy, end));
      slot += 1u;
    }
    writeKeptVertex(slot, index, end);
  } else if (startInside) {
    writeVertex(slot, getIntersection(previous.xy, end));
  }`})}function h(t,n){let r=n.starts.length;return e(t,{id:n.id,operation:n.operation,variant:`polygon-offsets`,bindings:[{name:`inputOffsets`,view:n.inputOffsets,type:`u32`,access:`read`},{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`counts`,view:n.counts,type:`u32`,access:`read`},{name:`outputOffsets`,view:n.outputOffsets,type:`u32`,access:`read_write`},{name:`stageFlags`,view:n.stageFlags,type:`u32`,access:`read_write`}],invocationCount:n.outputOffsets.length,declarations:`const ROWS: u32 = ${r}u;
const CAPACITY: u32 = ${n.capacity}u;`,body:`let total = starts[startsOffset + ROWS - 1u] + counts[countsOffset + ROWS - 1u];
  let row = inputOffsets[inputOffsetsOffset + index];
  let mapped = select(total, starts[startsOffset + min(row, ROWS - 1u)], row < ROWS);
  outputOffsets[outputOffsetsOffset + index] = min(mapped, CAPACITY);
  if (index == 0u) {
    stageFlags[stageFlagsOffset + ${n.stage}u] = total;
  }`})}function g(t,n){let r=[{name:`stageFlags`,view:n.stageFlags,type:`u32`,access:`read`},{name:`countOut`,view:n.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:n.overflow,type:`u32`,access:`read_write`}];return n.requiredCount&&r.push({name:`totalOut`,view:n.requiredCount,type:`u32`,access:`read_write`}),n.lineCount&&r.push({name:`lineCountOut`,view:n.lineCount,type:`u32`,access:`read_write`}),e(t,{id:n.id,operation:n.operation,variant:`polygon-publish`,bindings:r,invocationCount:1,body:`let finalTotal = stageFlags[stageFlagsOffset + 3u];
  var overflow = false;
  for (var stage = 0u; stage < 4u; stage++) {
    overflow = overflow || stageFlags[stageFlagsOffset + stage] > ${n.capacity}u;
  }
  countOut[countOutOffset] = min(finalTotal, ${n.capacity}u);
  overflowOut[overflowOutOffset] = select(0u, 1u, overflow);
  ${n.requiredCount?`totalOut[totalOutOffset] = stageFlags[stageFlagsOffset + ${n.requiredCountIndex??3}u];`:``}
  ${n.lineCount?`lineCountOut[lineCountOutOffset] = ${n.ringCount}u;`:``}`})}export{g as a,h as c,l as i,s as n,p as o,c as r,m as s,o as t};