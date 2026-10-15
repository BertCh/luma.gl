import{J as e}from"./src-CNCKFW2b.js";import{E as t,T as n}from"./analysis-frame-Blu0tYdw.js";import{_ as r,b as i,c as a,d as o,f as s,h as c,w as l,y as u}from"./resources-CwvhEOBe.js";import{t as d}from"./gpu-sort-Bl3Dztdb.js";import{t as f}from"./gpu-compaction-Cpx1OCRM.js";import{r as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{d as h}from"./wgsl-common-BNRVzgd8.js";import{t as g}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{t as _}from"./exact-geometry-wgsl-DOqxBQzc.js";import{t as ee}from"./sorted-segment-offsets-HVRgs63A.js";import{t as v}from"./load-balanced-for-each-OquID6hm.js";import{n as te,p as ne,t as y}from"./spatial-join-passes-C0ZwcuNs.js";import{t as b}from"./time-words-C-gLAZQw.js";var x=`GPUZoneEvents`,S=`0x80000000u`,C=`
const NO_TRACK: u32 = 0xffffffffu;
fn findTrack(row: u32) -> u32 {
  if (row < trackOffsets[trackOffsetsOffset] || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NO_TRACK;
  }
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] > row) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}`;function w(e){return e?`${b}
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  let wordsA = vec2<u32>(timestamps[timestampsOffset + 2u * a], timestamps[timestampsOffset + 2u * a + 1u]);
  let wordsB = vec2<u32>(timestamps[timestampsOffset + 2u * b], timestamps[timestampsOffset + 2u * b + 1u]);
  return timeWordsDifferenceToF32(wordsA, wordsB);
}`:`
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}`}function T(e){return`const TRACK_COUNT: u32 = ${e.trackCount}u;
const ZONE_COUNT: u32 = ${e.zoneCount}u;
const EDGE_COUNT: u32 = ${e.edgeCount}u;
const CANDIDATE_CAPACITY: u32 = ${e.candidateCapacity}u;
const MAXIMUM_EVENTS_PER_TRACK: u32 = ${e.maximumEventsPerTrack}u;`}function E(e,t,n){return`fn ${e}(value: u32) -> u32 {
  if (value < ${t}[${t}Offset] || value >= ${t}[${t}Offset + ${n}u]) { return NO_INTERVAL; }
  var low = 0u;
  var high = ${n}u;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (${t}[${t}Offset + middle] > value) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}`}function re(e,t){let{zones:n}=t;return m(e,{id:t.id,operation:x,variant:`edges`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:n.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:n.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:n.ringOffsets,type:`u32`,access:`read`},{name:`edgeStarts`,view:t.edgeStarts,type:`f32`,access:`read_write`},{name:`edgeEnds`,view:t.edgeEnds,type:`f32`,access:`read_write`},{name:`edgeZones`,view:t.edgeZones,type:`u32`,access:`read_write`}],invocationCount:n.positions.length,declarations:`const NO_INTERVAL: u32 = 0xffffffffu;
${E(`findRing`,`ringOffsets`,n.ringOffsets.length-1)}
${E(`findPolygon`,`polygonOffsets`,n.polygonOffsets.length-1)}
${E(`findFeature`,`featureOffsets`,n.featureOffsets.length-1)}`,body:`let vertex = index;
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let notANumber = bitcast<f32>(0x7fc00000u | (index & 0u));
  var zone = NO_INTERVAL;
  var start = vec2f(notANumber, notANumber);
  var end = vec2f(notANumber, notANumber);
  let ring = findRing(vertex);
  if (ring != NO_INTERVAL) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    let polygon = findPolygon(ring);
    var feature = NO_INTERVAL;
    if (polygon != NO_INTERVAL) { feature = findFeature(polygon); }
    if (ringEnd - ringStart >= 3u && feature != NO_INTERVAL) {
      let next = select(vertex + 1u, ringStart, vertex + 1u >= ringEnd);
      let a = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
      let b = vec2f(positions[positionsOffset + next * 2u], positions[positionsOffset + next * 2u + 1u]);
      let isRepeat = bitcast<u32>(a.x) == bitcast<u32>(b.x) && bitcast<u32>(a.y) == bitcast<u32>(b.y);
      if (!isRepeat) {
        zone = feature;
        start = a;
        end = b;
      }
    }
  }
  edgeStarts[edgeStartsOffset + vertex * 2u] = start.x;
  edgeStarts[edgeStartsOffset + vertex * 2u + 1u] = start.y;
  edgeEnds[edgeEndsOffset + vertex * 2u] = end.x;
  edgeEnds[edgeEndsOffset + vertex * 2u + 1u] = end.y;
  edgeZones[edgeZonesOffset + vertex] = zone;`})}function ie(e,t){return m(e,{id:t.id,operation:x,variant:`clear`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read_write`},{name:`initialInside`,view:t.initialInside,type:`u32`,access:`read_write`}],invocationCount:Math.max(t.cellCount,4),body:`if (index < 4u) { state[stateOffset + index] = 0u; }
  if (index < ${t.cellCount}u) { initialInside[initialInsideOffset + index] = 0u; }`})}function ae(e,t){let{bvh:n,shape:r}=t;return m(e,{id:t.id,operation:x,variant:`probe`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`nodeMinima`,view:n.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:n.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:n.leafIds,type:`u32`,access:`read`},{name:`state`,view:t.state,type:`atomic<u32>`,access:`read_write`},{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read_write`}],invocationCount:r.rowCount,declarations:`${y}
${T(r)}
const INTERNAL_NODE_COUNT: u32 = ${n.internalNodeCount}u;
${C}
${h}`,body:`let row = index;
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  let isFirst = row == trackOffsets[trackOffsetsOffset + track];
  let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  if (!isFiniteValue(current.x) || !isFiniteValue(current.y)) { return; }
  var queryMinimum = current;
  var queryMaximum = vec2f(FLOAT32_MAXIMUM, current.y);
  if (!isFirst) {
    let previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
    if (!isFiniteValue(previous.x) || !isFiniteValue(previous.y)) { return; }
    queryMinimum = min(previous, current);
    queryMaximum = max(previous, current);
  }
  let rowWord = select(row, row | ${S}, isFirst);
  var node = 0u;
  loop {
    if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
      if (node < INTERNAL_NODE_COUNT) {
        node = node * 2u + 1u;
        continue;
      }
      let edgeRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (edgeRow < EDGE_COUNT) {
        let slot = atomicAdd(&state[stateOffset], 1u);
        if (slot < CANDIDATE_CAPACITY) {
          candidatePairs[candidatePairsOffset + slot * 2u] = rowWord;
          candidatePairs[candidatePairsOffset + slot * 2u + 1u] = edgeRow;
        }
      }
    }
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    if (node == 0u) { break; }
    node = node + 1u;
  }`})}function oe(e,t){return m(e,{id:t.id,operation:x,variant:`crossing`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read`},{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`edgeStarts`,view:t.edgeStarts,type:`f32`,access:`read`},{name:`edgeEnds`,view:t.edgeEnds,type:`f32`,access:`read`},{name:`edgeZones`,view:t.edgeZones,type:`u32`,access:`read`},{name:`candidateKinds`,view:t.candidateKinds,type:`u32`,access:`read_write`},{name:`candidateParameters`,view:t.candidateParameters,type:`f32`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,declarations:`${y}
${T(t.shape)}
${g}
${_}
fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }`,body:`candidateKinds[candidateKindsOffset + index] = 0u;
  candidateParameters[candidateParametersOffset + index] = 0.0;
  if (index >= min(state[stateOffset], CANDIDATE_CAPACITY)) { return; }
  let rowWord = candidatePairs[candidatePairsOffset + index * 2u];
  let edge = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (edgeZones[edgeZonesOffset + edge] >= ZONE_COUNT) { return; }
  let a = vec2f(edgeStarts[edgeStartsOffset + edge * 2u], edgeStarts[edgeStartsOffset + edge * 2u + 1u]);
  let b = vec2f(edgeEnds[edgeEndsOffset + edge * 2u], edgeEnds[edgeEndsOffset + edge * 2u + 1u]);
  if (!isFiniteValue(a.x) || !isFiniteValue(a.y) || !isFiniteValue(b.x) || !isFiniteValue(b.y)) { return; }
  let isRay = (rowWord & ${S}) != 0u;
  let row = rowWord & 0x7fffffffu;
  let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  // Straddle rule of the perturbed point (see the boundary rule above): a.y >= y differs from b.y >= y.
  if (isRay && ((a.y >= current.y) == (b.y >= current.y))) { return; }
  var previous = current;
  if (!isRay) {
    previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
  }
  // One batch of exact signs for both candidate kinds keeps a single orientSign call site:
  // the ray needs orient(a, b, current); the segment test all four.
  let exact = exactOrientSigns(
    select(4u, 1u, isRay),
    array<vec2f, 4>(a, a, previous, previous),
    array<vec2f, 4>(b, b, current, current),
    array<vec2f, 4>(current, previous, a, b)
  );
  if (isRay) {
    let side = sosPointSide(exact.signs.x, a, b);
    if (select((side < 0), (side > 0), (b.y > a.y))) { candidateKinds[candidateKindsOffset + index] = 2u; }
    return;
  }
  if (sosPointSide(exact.signs.y, a, b) == sosPointSide(exact.signs.x, a, b)) { return; }
  if (sosSegmentSide(exact.signs.z, previous, current) == sosSegmentSide(exact.signs.w, previous, current)) { return; }
  // Only the decision above is exact. The parameter places the event in time.
  let direction = current - previous;
  let edgeDirection = b - a;
  let offset = a - previous;
  let denominator = cross2(direction, edgeDirection);
  var along = 0.0;
  if (denominator != 0.0) {
    along = cross2(offset, edgeDirection) / denominator;
  } else {
    // Parallel track and edge: the perturbed crossing lies at the edge endpoint nearest the track.
    let length2 = dot(direction, direction);
    if (length2 > 0.0) { along = dot(offset, direction) / length2; }
  }
  along = select(0.0, clamp(along, 0.0, 1.0), isFiniteValue(along));
  candidateKinds[candidateKindsOffset + index] = 1u;
  candidateParameters[candidateParametersOffset + index] = along;`})}function se(e,t){return m(e,{id:t.id,operation:x,variant:`initial-state`,bindings:[{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read`},{name:`candidateKinds`,view:t.candidateKinds,type:`u32`,access:`read`},{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`edgeZones`,view:t.edgeZones,type:`u32`,access:`read`},{name:`initialInside`,view:t.initialInside,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,declarations:`${T(t.shape)}
${C}`,body:`if (candidateKinds[candidateKindsOffset + index] != 2u) { return; }
  let row = candidatePairs[candidatePairsOffset + index * 2u] & 0x7fffffffu;
  let edge = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  atomicXor(&initialInside[initialInsideOffset + track * ZONE_COUNT + edgeZones[edgeZonesOffset + edge]], 1u);`})}function ce(e,t){return m(e,{id:t.id,operation:x,variant:t.isWordMode?`event-time-words`:`event-time`,bindings:[{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read`},{name:`candidateKinds`,view:t.candidateKinds,type:`u32`,access:`read`},{name:`candidateParameters`,view:t.candidateParameters,type:`f32`,access:`read`},{name:`timestamps`,view:t.timestamps,type:t.isWordMode?`u32`:`f32`,access:`read`},{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`candidateTracks`,view:t.candidateTracks,type:`u32`,access:`read_write`},{name:`candidateTimes`,view:t.candidateTimes,type:`f32`,access:`read_write`},{name:`state`,view:t.state,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,declarations:`${T(t.shape)}
${C}
${w(t.isWordMode)}`,body:`candidateTracks[candidateTracksOffset + index] = TRACK_COUNT;
  candidateTimes[candidateTimesOffset + index] = 0.0;
  if (candidateKinds[candidateKindsOffset + index] != 1u) { return; }
  let row = candidatePairs[candidatePairsOffset + index * 2u];
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  let firstRow = trackOffsets[trackOffsetsOffset + track];
  let base = rowTimeDifference(row - 1u, firstRow);
  let span = rowTimeDifference(row, row - 1u);
  let eventTime = max(base + candidateParameters[candidateParametersOffset + index] * span, 0.0);
  candidateTracks[candidateTracksOffset + index] = track;
  candidateTimes[candidateTimesOffset + index] = eventTime;
  atomicAdd(&state[stateOffset + 1u], 1u);`})}function D(e,t){let n=[],r;return t.variant===`edge`?(n.push({name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read`},{name:`candidateKinds`,view:t.candidateKinds,type:`u32`,access:`read`},{name:`identity`,view:t.identity,type:`u32`,access:`read_write`}),r=`identity[identityOffset + index] = index;
  let isEvent = candidateKinds[candidateKindsOffset + index] == 1u;
  keys[keysOffset + index] = select(0u, candidatePairs[candidatePairsOffset + index * 2u + 1u], isEvent);`):t.variant===`time`?(n.push({name:`order`,view:t.order,type:`u32`,access:`read`},{name:`candidateTimes`,view:t.candidateTimes,type:`f32`,access:`read`}),r=`keys[keysOffset + index] = bitcast<u32>(candidateTimes[candidateTimesOffset + order[orderOffset + index]]);`):(n.push({name:`order`,view:t.order,type:`u32`,access:`read`},{name:`candidateTracks`,view:t.candidateTracks,type:`u32`,access:`read`}),r=`keys[keysOffset + index] = candidateTracks[candidateTracksOffset + order[orderOffset + index]];`),n.push({name:`keys`,view:t.keys,type:`u32`,access:`read_write`}),m(e,{id:t.id,operation:x,variant:`sort-key-${t.variant}`,bindings:n,invocationCount:t.shape.candidateCapacity,body:r})}function le(e,t){return m(e,{id:t.id,operation:x,variant:`sorted-columns`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`order`,view:t.order,type:`u32`,access:`read`},{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read`},{name:`edgeZones`,view:t.edgeZones,type:`u32`,access:`read`},{name:`candidateTimes`,view:t.candidateTimes,type:`f32`,access:`read`},{name:`sortedZones`,view:t.sortedZones,type:`u32`,access:`read_write`},{name:`sortedTimes`,view:t.sortedTimes,type:`f32`,access:`read_write`},{name:`sortedRows`,view:t.sortedRows,type:`u32`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,body:`if (index >= state[stateOffset + 1u]) {
    sortedZones[sortedZonesOffset + index] = 0xffffffffu;
    sortedTimes[sortedTimesOffset + index] = 0.0;
    sortedRows[sortedRowsOffset + index] = 0xffffffffu;
    return;
  }
  let slot = order[orderOffset + index];
  let edge = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  sortedZones[sortedZonesOffset + index] = edgeZones[edgeZonesOffset + edge];
  sortedTimes[sortedTimesOffset + index] = candidateTimes[candidateTimesOffset + slot];
  sortedRows[sortedRowsOffset + index] = candidatePairs[candidatePairsOffset + slot * 2u];`})}function ue(e,t){return m(e,{id:t.id,operation:x,variant:`cell-keys`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`sortedTracks`,view:t.sortedTracks,type:`u32`,access:`read`},{name:`sortedZones`,view:t.sortedZones,type:`u32`,access:`read`},{name:`cellKeys`,view:t.cellKeys,type:`u32`,access:`read_write`},{name:`identity`,view:t.identity,type:`u32`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,declarations:T(t.shape),body:`identity[identityOffset + index] = index;
  var key = TRACK_COUNT * ZONE_COUNT;
  if (index < state[stateOffset + 1u]) {
    key = sortedTracks[sortedTracksOffset + index] * ZONE_COUNT + sortedZones[sortedZonesOffset + index];
  }
  cellKeys[cellKeysOffset + index] = key;`})}function de(e,t){return m(e,{id:t.id,operation:x,variant:`cell-events`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`cellOrder`,view:t.cellOrder,type:`u32`,access:`read`},{name:`sortedCellKeys`,view:t.sortedCellKeys,type:`u32`,access:`read`},{name:`cellOffsets`,view:t.cellOffsets,type:`u32`,access:`read`},{name:`initialInside`,view:t.initialInside,type:`u32`,access:`read`},{name:`sortedTimes`,view:t.sortedTimes,type:`f32`,access:`read`},{name:`eventTypes`,view:t.eventTypes,type:`u32`,access:`read_write`},{name:`cellTimes`,view:t.cellTimes,type:`f32`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,body:`if (index >= state[stateOffset + 1u]) {
    cellTimes[cellTimesOffset + index] = 0.0;
    return;
  }
  let cell = sortedCellKeys[sortedCellKeysOffset + index];
  let rank = index - cellOffsets[cellOffsetsOffset + cell];
  let after = (initialInside[initialInsideOffset + cell] & 1u) ^ ((rank + 1u) & 1u);
  let slot = cellOrder[cellOrderOffset + index];
  eventTypes[eventTypesOffset + slot] = select(1u, 0u, after == 1u);
  cellTimes[cellTimesOffset + index] = sortedTimes[sortedTimesOffset + slot];`})}var fe=`
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> f32 {
  let rank = item - loadBalancedOffsets[loadBalancedOffsetsOffset + row];
  let seed = initialInside[initialInsideOffset + row] & 1u;
  let after = seed ^ ((rank + 1u) & 1u);
  if (after == 1u) {
    return 0.0;
  }
  var enterTime = 0.0;
  if (rank > 0u) {
    enterTime = cellTimes[cellTimesOffset + item - 1u];
  }
  return cellTimes[cellTimesOffset + item] - enterTime;
}`;function pe(e,t){return m(e,{id:t.id,operation:x,variant:`event-ranks`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`sortedTracks`,view:t.sortedTracks,type:`u32`,access:`read`},{name:`eventRanks`,view:t.eventRanks,type:`u32`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,body:`let eventTotal = state[stateOffset + 1u];
  if (index >= eventTotal) {
    eventRanks[eventRanksOffset + index] = 0xffffffffu;
    return;
  }
  let track = sortedTracks[sortedTracksOffset + index];
  var low = 0u;
  var high = index;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (sortedTracks[sortedTracksOffset + middle] < track) { low = middle + 1u; } else { high = middle; }
  }
  eventRanks[eventRanksOffset + index] = index - low;`})}function me(e,t){return m(e,{id:t.id,operation:x,variant:`track-counts`,bindings:[{name:`state`,view:t.state,type:`atomic<u32>`,access:`read_write`},{name:`sortedTracks`,view:t.sortedTracks,type:`u32`,access:`read`},{name:`trackEventCounts`,view:t.trackEventCounts,type:`u32`,access:`read_write`}],invocationCount:t.shape.trackCount,declarations:`${T(t.shape)}
// First event position whose track is at least (or, with strict, above) the given track.
fn findTrackBound(track: u32, eventTotal: u32, strict: bool) -> u32 {
  var low = 0u;
  var high = eventTotal;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    let value = sortedTracks[sortedTracksOffset + middle];
    if (value < track || (strict && value == track)) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,body:`let eventTotal = atomicLoad(&state[stateOffset + 1u]);
  let count = findTrackBound(index, eventTotal, true) - findTrackBound(index, eventTotal, false);
  trackEventCounts[trackEventCountsOffset + index] = count;
  if (count > MAXIMUM_EVENTS_PER_TRACK) { atomicOr(&state[stateOffset + 2u], 1u); }`})}function he(e,t){return m(e,{id:t.id,operation:x,variant:t.isWordMode?`cell-finalize-words`:`cell-finalize`,bindings:[{name:`timestamps`,view:t.timestamps,type:t.isWordMode?`u32`:`f32`,access:`read`},{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`cellOffsets`,view:t.cellOffsets,type:`u32`,access:`read`},{name:`initialInside`,view:t.initialInside,type:`u32`,access:`read`},{name:`cellTimes`,view:t.cellTimes,type:`f32`,access:`read`},{name:`dwellTimes`,view:t.dwellTimes,type:`f32`,access:`read_write`},{name:`visitCounts`,view:t.visitCounts,type:`u32`,access:`read_write`}],invocationCount:t.shape.trackCount*t.shape.zoneCount,declarations:`${T(t.shape)}
${w(t.isWordMode)}`,body:`let start = cellOffsets[cellOffsetsOffset + index];
  let eventCount = cellOffsets[cellOffsetsOffset + index + 1u] - start;
  let seed = initialInside[initialInsideOffset + index] & 1u;
  let enters = select((eventCount + 1u) / 2u, eventCount / 2u, seed == 1u);
  visitCounts[visitCountsOffset + index] = seed + enters;
  if ((seed ^ (eventCount & 1u)) == 0u) { return; }
  var enterTime = 0.0;
  if (eventCount > 0u) { enterTime = cellTimes[cellTimesOffset + start + eventCount - 1u]; }
  let track = index / ZONE_COUNT;
  let firstRow = trackOffsets[trackOffsetsOffset + track];
  let lastRow = trackOffsets[trackOffsetsOffset + track + 1u] - 1u;
  let duration = rowTimeDifference(lastRow, firstRow);
  dwellTimes[dwellTimesOffset + index] = dwellTimes[dwellTimesOffset + index] + (duration - enterTime);`})}function ge(e,t){return m(e,{id:t.id,operation:x,variant:t.isWordMode?`cell-spans-words`:`cell-spans`,bindings:[{name:`timestamps`,view:t.timestamps,type:t.isWordMode?`u32`:`f32`,access:`read`},{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`cellOffsets`,view:t.cellOffsets,type:`u32`,access:`read`},{name:`initialInside`,view:t.initialInside,type:`u32`,access:`read`},{name:`cellTimes`,view:t.cellTimes,type:`f32`,access:`read`},{name:`firstEnterTimes`,view:t.firstEnterTimes,type:`f32`,access:`read_write`},{name:`lastExitTimes`,view:t.lastExitTimes,type:`f32`,access:`read_write`}],invocationCount:t.shape.trackCount*t.shape.zoneCount,declarations:`${T(t.shape)}
${w(t.isWordMode)}`,body:`let start = cellOffsets[cellOffsetsOffset + index];
  let eventCount = cellOffsets[cellOffsetsOffset + index + 1u] - start;
  let seed = initialInside[initialInsideOffset + index] & 1u;
  var firstEnter = -1.0;
  if (seed == 1u) {
    firstEnter = 0.0;
  } else if (eventCount > 0u) {
    firstEnter = cellTimes[cellTimesOffset + start];
  }
  firstEnterTimes[firstEnterTimesOffset + index] = firstEnter;
  var lastExit = 0.0;
  if ((seed ^ (eventCount & 1u)) == 1u) {
    let track = index / ZONE_COUNT;
    let firstRow = trackOffsets[trackOffsetsOffset + track];
    let lastRow = trackOffsets[trackOffsetsOffset + track + 1u] - 1u;
    lastExit = rowTimeDifference(lastRow, firstRow);
  } else if (eventCount > 0u) {
    lastExit = cellTimes[cellTimesOffset + start + eventCount - 1u];
  }
  lastExitTimes[lastExitTimesOffset + index] = lastExit;`})}function _e(e,t){return m(e,{id:t.id,operation:x,variant:`keep-flags`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`eventRanks`,view:t.eventRanks,type:`u32`,access:`read`},{name:`keepFlags`,view:t.keepFlags,type:`u32`,access:`read_write`},{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,declarations:T(t.shape),body:`rowIds[rowIdsOffset + index] = index;
  let isKept = index < state[stateOffset + 1u] && eventRanks[eventRanksOffset + index] < MAXIMUM_EVENTS_PER_TRACK;
  keepFlags[keepFlagsOffset + index] = select(0u, 1u, isKept);`})}function ve(e,t){let n=[{name:`keptTotal`,view:t.keptTotal,type:`u32`,access:`read`},{name:`keptRows`,view:t.keptRows,type:`u32`,access:`read`}],r=[];for(let[e,i]of t.columns.entries()){n.push({name:`source${e}`,view:i.source,type:`u32`,access:`read`},{name:`destination${e}`,view:i.destination,type:`u32`,access:`read_write`});let t=i.stride??1;for(let n=0;n<t;n++)r.push(`destination${e}[destination${e}Offset + index * ${t}u + ${n}u] = select(${i.sentinel}, source${e}[source${e}Offset + row * ${t}u + ${n}u], isLive);`)}return m(e,{id:t.id,operation:x,variant:`gather`,bindings:n,invocationCount:t.capacity,body:`let isLive = index < keptTotal[keptTotalOffset];
  let row = select(0u, keptRows[keptRowsOffset + index], isLive);
  ${r.join(`
  `)}`})}function ye(e,t){return m(e,{id:t.id,operation:x,variant:`overflow`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`flag`,view:t.flag,type:`u32`,access:`read_write`}],invocationCount:1,declarations:T(t.shape),body:`let candidateOverflow = state[stateOffset] > CANDIDATE_CAPACITY;
  let trackOverflow = state[stateOffset + 2u] != 0u;
  flag[flagOffset] = select(0u, 1u, candidateOverflow || trackOverflow);`})}function be(e,t){let n=[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`keptTotal`,view:t.keptTotal,type:`u32`,access:`read`}],r=[],i=[[`candidateCount`,t.candidateCount,`state[stateOffset]`],[`candidateOverflow`,t.candidateOverflow,`select(0u, 1u, state[stateOffset] > CANDIDATE_CAPACITY)`],[`trackOverflow`,t.trackOverflow,`select(0u, 1u, state[stateOffset + 2u] != 0u)`],[`eventOverflow`,t.eventOverflow,`select(0u, 1u, keptTotal[keptTotalOffset] > EVENT_CAPACITY)`]];for(let[e,t,a]of i)t&&(n.push({name:e,view:t,type:`u32`,access:`read_write`}),r.push(`${e}[${e}Offset] = ${a};`));return m(e,{id:t.id,operation:x,variant:`diagnostics`,bindings:n,invocationCount:1,declarations:`${T(t.shape)}
const EVENT_CAPACITY: u32 = ${t.eventCapacity}u;`,body:r.join(`
  `)})}function xe(e,t){return m(e,{id:t.id,operation:x,variant:`sorted-positions`,bindings:[{name:`state`,view:t.state,type:`u32`,access:`read`},{name:`order`,view:t.order,type:`u32`,access:`read`},{name:`candidatePairs`,view:t.candidatePairs,type:`u32`,access:`read`},{name:`candidateParameters`,view:t.candidateParameters,type:`f32`,access:`read`},{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`sortedPositions`,view:t.sortedPositions,type:`f32`,access:`read_write`}],invocationCount:t.shape.candidateCapacity,body:`var crossing = vec2f(0.0, 0.0);
  if (index < state[stateOffset + 1u]) {
    let slot = order[orderOffset + index];
    let row = candidatePairs[candidatePairsOffset + slot * 2u];
    let along = candidateParameters[candidateParametersOffset + slot];
    let previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
    let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
    crossing = previous + along * (current - previous);
  }
  sortedPositions[sortedPositionsOffset + index * 2u] = crossing.x;
  sortedPositions[sortedPositionsOffset + index * 2u + 1u] = crossing.y;`})}function O(e,t){return m(e,{id:t.id,operation:x,variant:`table-flags`,bindings:[{name:`visitCounts`,view:t.visitCounts,type:`u32`,access:`read`},{name:`flags`,view:t.flags,type:`u32`,access:`read_write`},{name:`cellIds`,view:t.cellIds,type:`u32`,access:`read_write`}],invocationCount:t.cellCount,body:`cellIds[cellIdsOffset + index] = index;
  flags[flagsOffset + index] = select(0u, 1u, visitCounts[visitCountsOffset + index] > 0u);`})}function k(e){return e.kind===`value`?2:1}function Se(t,n){let r=[{name:`keptTotal`,view:n.keptTotal,type:`u32`,access:`read`},{name:`keptCells`,view:n.keptCells,type:`u32`,access:`read`}],i=[];for(let[e,t]of n.columns.entries()){let n=t.destination.format===`float32`?`f32`:`u32`,a=n===`f32`?`0.0`:`0xffffffffu`;r.push({name:`destination${e}`,view:t.destination,type:n,access:`read_write`});let o;t.kind===`value`&&t.source?(r.push({name:`source${e}`,view:t.source,type:n,access:`read`}),o=`source${e}[source${e}Offset + cell]`):o=t.kind===`track`?`cell / ZONE_COUNT`:`cell % ZONE_COUNT`,i.push(`destination${e}[destination${e}Offset + index] = select(${a}, ${o}, isLive);`)}return e(r.length<=8),m(t,{id:n.id,operation:x,variant:`table-gather`,bindings:r,invocationCount:n.capacity,declarations:`const ZONE_COUNT: u32 = ${n.zoneCount}u;`,body:`let isLive = index < keptTotal[keptTotalOffset];
  let cell = select(0u, keptCells[keptCellsOffset + index], isLive);
  ${i.join(`
  `)}`})}var A=`GPUZoneEvents`,Ce={enter:0,exit:1},j=class{id;props;spatialContext;trackCount;zoneCount;edgeCount;leafCapacity;constructor(e){this.id=e.id??`zone-events`,this.props=e;let{id:r}=this;this.spatialContext=t(r,e.spatialContext),n(this.id,`zones`,e.zonesSpatialContext,this.spatialContext);let{tracks:l,zones:d,output:f}=e;if(u(l.positions,[`float32x2`],`${r} tracks.positions`),u(l.timestamps,[`float32`,`uint32x2`],`${r} tracks.timestamps`),u(l.trackOffsets,[`uint32`],`${r} tracks.trackOffsets`),u(d.positions,[`float32x2`],`${r} zones.positions`),u(d.featureOffsets,[`uint32`],`${r} zones.featureOffsets`),u(d.polygonOffsets,[`uint32`],`${r} zones.polygonOffsets`),u(d.ringOffsets,[`uint32`],`${r} zones.ringOffsets`),l.timestamps.length!==l.positions.length)throw Error(`${r} timestamps length must equal positions length`);if(l.trackOffsets.length<2)throw Error(`${r} trackOffsets must contain at least two rows`);if(d.featureOffsets.length<2)throw Error(`${r} zones need at least one feature`);if(d.polygonOffsets.length<1||d.ringOffsets.length<1)throw Error(`${r} zones polygonOffsets and ringOffsets must not be empty`);if(this.trackCount=l.trackOffsets.length-1,this.zoneCount=d.featureOffsets.length-1,this.edgeCount=d.positions.length,this.edgeCount<1)throw Error(`${r} zones need at least one vertex`);if(o(r,`candidateCapacity`,e.candidateCapacity),o(r,`trackEventCapacity`,e.trackEventCapacity),l.positions.length>=2147483648)throw Error(`${r} supports fewer than 2^31 rows`);let p=this.trackCount*this.zoneCount;if(p>4294967295)throw Error(`${r} trackCount * zoneCount must fit in 32 bits`);for(let[e,t,n]of[[`dwellTimes`,f.dwellTimes,`float32`],[`visitCounts`,f.visitCounts,`uint32`]])if(t&&(u(t,[n],`${r} ${e}`),t.length!==p))throw Error(`${r} ${e} length must equal trackCount * zoneCount`);if(f.trackEventCounts&&(u(f.trackEventCounts,[`uint32`],`${r} trackEventCounts`),f.trackEventCounts.length!==this.trackCount))throw Error(`${r} trackEventCounts length must equal the track count`);let{events:m}=f;s(r,m);let h=m.ids.length;for(let[e,t,n]of[[`eventZones`,m.eventZones,`uint32`],[`eventTypes`,m.eventTypes,`uint32`],[`eventTimes`,m.eventTimes,`float32`],[`eventRows`,m.eventRows,`uint32`]])if(t&&(u(t,[n],`${r} events.${e}`),t.length!==h))throw Error(`${r} events.${e} length must equal the capacity`);if(m.eventPositions&&(u(m.eventPositions,[`float32x2`],`${r} events.eventPositions`),m.eventPositions.length!==h))throw Error(`${r} events.eventPositions length must equal the capacity`);let{visitTable:g}=f;if(g){s(`${r} visitTable`,g);let e=g.ids.length;for(let[t,n,i]of[[`zones`,g.zones,`uint32`],[`visits`,g.visits,`uint32`],[`dwellTimes`,g.dwellTimes,`float32`],[`firstEnterTimes`,g.firstEnterTimes,`float32`],[`lastExitTimes`,g.lastExitTimes,`float32`]])if(n&&(u(n,[i],`${r} visitTable.${t}`),n.length!==e))throw Error(`${r} visitTable.${t} length must equal the capacity`)}for(let[e,t]of Object.entries(f.status??{}))if(t&&(u(t,[`uint32`],`${r} status.${e}`),t.length!==1))throw Error(`${r} status.${e} must have one row`);this.leafCapacity=e.leafCapacity??a(this.edgeCount),i(r,`leafCapacity`,this.leafCapacity),c(r,[m.ids,m.status.count,m.status.overflow,m.status.requiredCount,m.eventZones,m.eventTypes,m.eventTimes,m.eventRows,m.eventPositions,...this.getTableViews(),f.dwellTimes,f.visitCounts,f.trackEventCounts,...Object.values(f.status??{})],[l.positions,l.timestamps,l.trackOffsets,d.positions,d.featureOffsets,d.polygonOffsets,d.ringOffsets])}getCommandNodes(e){let{id:t,props:n,trackCount:i,edgeCount:a,leafCapacity:o}=this,{tracks:s,zones:c,output:u}=n,{events:m}=u;r(t,e,[s.positions,s.timestamps,s.trackOffsets,c.positions,c.featureOffsets,c.polygonOffsets,c.ringOffsets,m.ids,m.status.count,m.status.overflow,m.status.requiredCount,m.eventZones,m.eventTypes,m.eventTimes,m.eventRows,m.eventPositions,...this.getTableViews(),u.dwellTimes,u.visitCounts,u.trackEventCounts,...Object.values(u.status??{})]);let h=s.positions.length,g=n.candidateCapacity,_=i*this.zoneCount,y=s.timestamps.format===`uint32x2`,b={rowCount:h,trackCount:i,edgeCount:a,zoneCount:this.zoneCount,candidateCapacity:g,maximumEventsPerTrack:n.trackEventCapacity},x=(n,r)=>l(e,`${t}-${n}`,`uint32`,Math.max(r,1)),S=(n,r)=>l(e,`${t}-${n}`,`float32`,Math.max(r,1)),C=[],w=l(e,`${t}-edge-starts`,`float32x2`,a),T=l(e,`${t}-edge-ends`,`float32x2`,a),E=x(`edge-zones`,a);C.push(re(e,{id:`${t}-edges`,zones:c,edgeStarts:w,edgeEnds:T,edgeZones:E}));let O=l(e,`${t}-edge-minima`,`float32x2`,a),k=l(e,`${t}-edge-maxima`,`float32x2`,a);C.push(te(e,{id:`${t}-bounds`,operation:A,featureCount:a,source:{kind:`segments`,starts:w,ends:T},minima:O,maxima:k}));let{bvh:Se,nodes:Ce}=ne(e,t,A,O,k,o,n.spatialSort??!1);C.push(...Ce);let j=x(`state`,4),N=x(`initial-inside`,_),P=l(e,`${t}-candidate-pairs`,`uint32x2`,g),F=x(`candidate-kinds`,g),I=S(`candidate-parameters`,g),we=x(`candidate-tracks`,g),L=S(`candidate-times`,g);C.push(ie(e,{id:`${t}-clear`,state:j,initialInside:N,cellCount:_}),ae(e,{id:`${t}-probe`,shape:b,positions:s.positions,trackOffsets:s.trackOffsets,bvh:Se,state:j,candidatePairs:P}),oe(e,{id:`${t}-crossing`,shape:b,positions:s.positions,edgeStarts:w,edgeEnds:T,edgeZones:E,state:j,candidatePairs:P,candidateKinds:F,candidateParameters:I}),se(e,{id:`${t}-initial-state`,shape:b,trackOffsets:s.trackOffsets,edgeZones:E,candidatePairs:P,candidateKinds:F,initialInside:N}),ce(e,{id:`${t}-event-time`,shape:b,isWordMode:y,timestamps:s.timestamps,trackOffsets:s.trackOffsets,state:j,candidatePairs:P,candidateKinds:F,candidateParameters:I,candidateTracks:we,candidateTimes:L}));let Te=x(`identity`,g),Ee=x(`edge-keys`,g),De=x(`edge-sorted-keys`,g),R=x(`edge-order`,g),Oe=x(`time-keys`,g),ke=x(`time-sorted-keys`,g),z=x(`time-order`,g),Ae=x(`track-keys`,g),B=x(`sorted-tracks`,g),V=x(`order`,g);C.push(D(e,{id:`${t}-edge-keys`,shape:b,variant:`edge`,candidatePairs:P,candidateKinds:F,keys:Ee,identity:Te}),...new d({id:`${t}-edge-sort`,keys:Ee,values:Te,outputKeys:De,outputValues:R,keyBits:M(a-1)}).getCommandNodes(e),D(e,{id:`${t}-time-keys`,shape:b,variant:`time`,order:R,candidateTimes:L,keys:Oe}),...new d({id:`${t}-time-sort`,keys:Oe,values:R,outputKeys:ke,outputValues:z,keyBits:32}).getCommandNodes(e),D(e,{id:`${t}-track-keys`,shape:b,variant:`track`,order:z,candidateTracks:we,keys:Ae}),...new d({id:`${t}-track-sort`,keys:Ae,values:z,outputKeys:B,outputValues:V,keyBits:M(i)}).getCommandNodes(e));let H=x(`sorted-zones`,g),U=S(`sorted-times`,g),je=x(`sorted-rows`,g),Me=x(`event-types`,g),Ne=x(`event-ranks`,g),{visitTable:W}=u,Pe=!!(W?.firstEnterTimes||W?.lastExitTimes),G=Pe?S(`first-enter-times`,_):void 0,K=Pe?S(`last-exit-times`,_):void 0,q=u.dwellTimes??S(`dwell-times`,_),Fe=u.visitCounts??x(`visit-counts`,_),Ie=u.trackEventCounts??x(`track-event-counts`,i),Le=x(`cell-keys`,g),Re=x(`cell-identity`,g),J=x(`sorted-cell-keys`,g),ze=x(`cell-order`,g),Y=x(`cell-offsets`,_+1),X=S(`cell-times`,g);C.push(le(e,{id:`${t}-sorted-columns`,shape:b,state:j,order:V,candidatePairs:P,edgeZones:E,candidateTimes:L,sortedZones:H,sortedTimes:U,sortedRows:je}),pe(e,{id:`${t}-event-ranks`,shape:b,state:j,sortedTracks:B,eventRanks:Ne}),me(e,{id:`${t}-track-counts`,shape:b,state:j,sortedTracks:B,trackEventCounts:Ie}),ue(e,{id:`${t}-cell-keys`,shape:b,state:j,sortedTracks:B,sortedZones:H,cellKeys:Le,identity:Re}),...new d({id:`${t}-cell-sort`,keys:Le,values:Re,outputKeys:J,outputValues:ze,keyBits:M(_)}).getCommandNodes(e),ee(e,{id:`${t}-cell-offsets`,operation:A,variant:`cell-offsets`,segmentCount:_,sortedKeys:J,segmentOffsets:Y}),de(e,{id:`${t}-cell-events`,shape:b,state:j,cellOrder:ze,sortedCellKeys:J,cellOffsets:Y,initialInside:N,sortedTimes:U,eventTypes:Me,cellTimes:X}),...v(e,{id:`${t}-dwell`,operation:A,variant:`dwell-sum`,offsets:Y,itemCapacity:g,accumulatorType:`f32`,output:q,bindings:[{name:`initialInside`,view:N,type:`u32`,access:`read`},{name:`cellTimes`,view:X,type:`f32`,access:`read`}],declarations:fe}),he(e,{id:`${t}-cell-finalize`,shape:b,isWordMode:y,timestamps:s.timestamps,trackOffsets:s.trackOffsets,cellOffsets:Y,initialInside:N,cellTimes:X,dwellTimes:q,visitCounts:Fe})),G&&K&&C.push(ge(e,{id:`${t}-cell-spans`,shape:b,isWordMode:y,timestamps:s.timestamps,trackOffsets:s.trackOffsets,cellOffsets:Y,initialInside:N,cellTimes:X,firstEnterTimes:G,lastExitTimes:K}));let Z;m.eventPositions&&(Z=l(e,`${t}-sorted-positions`,`float32x2`,Math.max(g,1)),C.push(xe(e,{id:`${t}-sorted-positions`,shape:b,state:j,order:V,candidatePairs:P,candidateParameters:I,positions:s.positions,sortedPositions:Z})));let Be=x(`keep-flags`,g),Ve=x(`row-ids`,g),He=x(`kept-rows`,g),Q=x(`kept-total`,1),Ue=x(`overflow-flag`,1);C.push(_e(e,{id:`${t}-keep-flags`,shape:b,state:j,eventRanks:Ne,keepFlags:Be,rowIds:Ve}),...new f({id:`${t}-kept`,input:Ve,flags:Be,output:He,count:Q}).getCommandNodes(e),ye(e,{id:`${t}-overflow`,shape:b,state:j,flag:Ue})),u.status&&Object.values(u.status).some(Boolean)&&C.push(be(e,{id:`${t}-diagnostics`,shape:b,eventCapacity:m.ids.length,state:j,keptTotal:Q,...u.status}));let $=[{source:B,destination:m.ids,sentinel:`0xffffffffu`}];m.eventZones&&$.push({source:H,destination:m.eventZones,sentinel:`0xffffffffu`}),m.eventRows&&$.push({source:je,destination:m.eventRows,sentinel:`0xffffffffu`}),m.eventTypes&&$.push({source:Me,destination:m.eventTypes,sentinel:`0u`}),m.eventTimes&&$.push({source:U,destination:m.eventTimes,sentinel:`0u`}),m.eventPositions&&Z&&$.push({source:Z,destination:m.eventPositions,sentinel:`0u`,stride:2});for(let n=0,r=0;n<$.length;n+=3,r++)C.push(ve(e,{id:`${t}-gather-${r}`,capacity:m.ids.length,keptTotal:Q,keptRows:He,columns:$.slice(n,n+3)}));return C.push(p(e,{id:`${t}-publish`,operation:A,requiredCount:Q,output:m,overflowSources:[Ue]})),W&&C.push(...this.getTableNodes(e,{cellCount:_,visitCounts:Fe,dwellTimes:q,firstEnterTimes:G,lastExitTimes:K})),C}getTableViews(){let e=this.props.output.visitTable;return e?[e.ids,e.status.count,e.status.overflow,e.status.requiredCount,e.zones,e.visits,e.dwellTimes,e.firstEnterTimes,e.lastExitTimes]:[]}getTableNodes(e,t){let{id:n}=this,r=this.props.output.visitTable,{cellCount:i}=t,a=r.ids.length,o=(t,r)=>l(e,`${n}-${t}`,`uint32`,Math.max(r,1)),s=o(`table-flags`,i),c=o(`table-cell-ids`,i),u=o(`table-kept-cells`,i),d=o(`table-kept-total`,1),m=[O(e,{id:`${n}-table-flags`,cellCount:i,visitCounts:t.visitCounts,flags:s,cellIds:c}),...new f({id:`${n}-table-cells`,input:c,flags:s,output:u,count:d}).getCommandNodes(e)],h=[{kind:`track`,destination:r.ids}];r.zones&&h.push({kind:`zone`,destination:r.zones}),r.visits&&h.push({kind:`value`,source:t.visitCounts,destination:r.visits}),r.dwellTimes&&h.push({kind:`value`,source:t.dwellTimes,destination:r.dwellTimes}),r.firstEnterTimes&&t.firstEnterTimes&&h.push({kind:`value`,source:t.firstEnterTimes,destination:r.firstEnterTimes}),r.lastExitTimes&&t.lastExitTimes&&h.push({kind:`value`,source:t.lastExitTimes,destination:r.lastExitTimes});let g=[],_=2,ee=0,v=()=>{g.length>0&&(m.push(Se(e,{id:`${n}-table-gather-${ee++}`,capacity:a,zoneCount:this.zoneCount,keptTotal:d,keptCells:u,columns:g})),g=[],_=2)};for(let e of h){let t=k(e);_+t>8&&v(),g.push(e),_+=t}return v(),m.push(p(e,{id:`${n}-table-publish`,operation:A,requiredCount:d,output:r})),m}};function M(e){return Math.max(1,32-Math.clz32(Math.max(e,1)))}export{Ce as n,j as t};