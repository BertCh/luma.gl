import{_ as e,h as t,i as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{n as a,s as o}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as s}from"./raster-utils-DhdnkkOa.js";import{i as c,s as l,u}from"./raster-band-utils-CA-sh8xt.js";import{t as d}from"./raster-relaxation-BJ40xgro.js";import{t as f}from"./gpu-raster-buffer-to-texture-BroAAmI3.js";var p=4294967295,m=32768;function h(e,t,n=new Float32Array(8)){c(`getGPUDistanceFieldParameterValues`,n,8);let[r,i,a,o,l,u]=t.affine;if(i!==0||o!==0)throw Error(`getGPUDistanceFieldParameterValues needs an axis-aligned grid`);let d=Math.fround(Math.abs(r)),f=Math.fround(Math.abs(l));if(!(Number.isFinite(d)&&d>0&&Number.isFinite(f)&&f>0))throw Error(`getGPUDistanceFieldParameterValues cell size must be finite and positive`);let p=t.pixelInterpretation===`point`?-.5:0,[m,h]=s(`Distance field`,e.localOrigin);return n.fill(0,0,8),n[0]=a+r*p-m,n[1]=u+l*p-h,n[2]=d,n[3]=f,n[4]=1/Math.fround(r),n[5]=1/Math.fround(l),n[6]=e.maximumDistance??1/0,n}var g=`GPUDistanceField`,_=`
fn multiplyWide(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowLow = leftLow * rightLow;
  let lowHigh = leftLow * rightHigh;
  let highLow = leftHigh * rightLow;
  let middle = (lowLow >> 16u) + (lowHigh & 0xffffu) + (highLow & 0xffffu);
  let low = (lowLow & 0xffffu) | (middle << 16u);
  let high = leftHigh * rightHigh + (lowHigh >> 16u) + (highLow >> 16u) + (middle >> 16u);
  return vec2<u32>(low, high);
}

/** Whether (candidate + ulp / 2)^2 > mantissa * 2^exponent, for a positive normal candidate. */
fn isUpperMidpointAbove(candidate: f32, mantissa: u32, exponent: i32) -> bool {
  let bits = bitcast<u32>(candidate);
  let candidateExponent = i32((bits >> 23u) & 0xffu);
  let odd = (((bits & 0x7fffffu) | 0x800000u) << 1u) | 1u;
  // Midpoint^2 = odd^2 * 2^(2 * candidateExponent - 302); compare odd^2 with mantissa * 2^shift.
  let square = multiplyWide(odd, odd);
  let shift = exponent - 2 * candidateExponent + 302;
  if (shift < 0) { return true; }
  if (shift >= 64) { return false; }
  var scaled: vec2<u32>;
  if (shift == 0) {
    scaled = vec2<u32>(mantissa, 0u);
  } else if (shift < 32) {
    scaled = vec2<u32>(mantissa << u32(shift), mantissa >> u32(32 - shift));
  } else {
    let highShift = u32(shift - 32);
    if (highShift > 0u && (mantissa >> (32u - highShift)) != 0u) { return false; }
    scaled = vec2<u32>(0u, mantissa << highShift);
  }
  return square.y > scaled.y || (square.y == scaled.y && square.x > scaled.x);
}

fn getRoundedSquareRoot(mantissa: u32, exponent: i32) -> f32 {
  if (mantissa == 0u) { return 0.0; }
  var root = sqrt(f32(mantissa) * exp2(f32(exponent)));
  for (var step = 0u; step < 8u; step++) {
    if (!isUpperMidpointAbove(root, mantissa, exponent)) {
      root = bitcast<f32>(bitcast<u32>(root) + 1u);
    } else if (isUpperMidpointAbove(bitcast<f32>(bitcast<u32>(root) - 1u), mantissa, exponent)) {
      root = bitcast<f32>(bitcast<u32>(root) - 1u);
    } else {
      break;
    }
  }
  return root;
}`,v=64;function y(e,t){return`
const GRID_WIDTH: u32 = ${e.width}u;
const GRID_HEIGHT: u32 = ${e.height}u;
const NONE: u32 = ${p}u;

fn getAbsoluteDifference(left: u32, right: u32) -> u32 {
  return select(right - left, left - right, left > right);
}
${t?`
fn isIsotropic() -> bool {
  return settings[settingsOffset + 2u] == settings[settingsOffset + 3u];
}

fn getAnisotropicKey(dx: u32, dy: u32) -> f32 {
  let groundX = settings[settingsOffset + 2u] * f32(dx);
  let groundY = settings[settingsOffset + 3u] * f32(dy);
  return groundX * groundX + groundY * groundY;
}

fn isCloser(dxA: u32, dyA: u32, idA: u32, dxB: u32, dyB: u32, idB: u32) -> bool {
  if (isIsotropic()) {
    let keyA = dxA * dxA + dyA * dyA;
    let keyB = dxB * dxB + dyB * dyB;
    if (keyA != keyB) { return keyA < keyB; }
  } else {
    let keyA = getAnisotropicKey(dxA, dyA);
    let keyB = getAnisotropicKey(dxB, dyB);
    if (keyA != keyB) { return keyA < keyB; }
  }
  return idA < idB;
}`:``}`}function b(e,t){let{id:n,grid:r}=t,i=r.width*r.height,a=[{name:`cellSeeds`,view:t.cellSeeds,type:`u32`,access:`read_write`}];t.seedMask&&a.unshift({name:`seedMask`,view:t.seedMask,type:`u32`,access:`read`});let s=[o(e,{id:`${n}-seed-clear`,operation:g,variant:`seed-clear`,bindings:a,invocationCount:i,body:t.seedMask?`cellSeeds[cellSeedsOffset + index] = seedMask[seedMaskOffset + index] - 1u;`:`cellSeeds[cellSeedsOffset + index] = ${p}u;`})],c=t.seedPositions;if(c&&c.length>0){let i=[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`positions`,view:c,type:`f32`,access:`read`}];t.seedIds&&i.push({name:`seedIds`,view:t.seedIds,type:`u32`,access:`read`}),t.seedCount&&i.push({name:`seedCount`,view:t.seedCount,type:`u32`,access:`read`}),i.push({name:`cellSeedBits`,view:t.cellSeeds,type:`atomic<u32>`,access:`read_write`}),s.push(o(e,{id:`${n}-seed-scatter`,operation:g,variant:`seed-scatter`,bindings:i,invocationCount:c.length,declarations:y(r,!1),body:`${t.seedCount?`if (index >= seedCount[seedCountOffset]) { return; }`:``}
  let seedId = ${t.seedIds?`seedIds[seedIdsOffset + index]`:`index`};
  if (seedId == NONE) { return; }
  let column = floor((positions[positionsOffset + index * 2u] - settings[settingsOffset]) * settings[settingsOffset + 4u]);
  let row = floor((positions[positionsOffset + index * 2u + 1u] - settings[settingsOffset + 1u]) * settings[settingsOffset + 5u]);
  // Negated comparisons also reject NaN and infinite coordinates.
  if (!(column >= 0.0 && column < f32(GRID_WIDTH) && row >= 0.0 && row < f32(GRID_HEIGHT))) { return; }
  atomicMin(&cellSeedBits[cellSeedBitsOffset + u32(row) * GRID_WIDTH + u32(column)], seedId);`}))}return s}function x(e){let t=Math.ceil(e.height/32),n=Math.ceil(e.width/32);return{columnBandCount:t,rowBandCount:n,mergeRoundCount:n>1?Math.ceil(Math.log2(n)):0}}function S(e,t){let{grid:n,scratch:r}=t,{columnBandCount:i}=x(n),a=`${y(n,!1)}
const BAND_SIZE: u32 = 32u;
const COLUMN_BAND_COUNT: u32 = ${i}u;`,s=i*n.width;return[o(e,{id:`${t.id}-column-bands`,operation:g,variant:`exact-column-bands`,bindings:[{name:`cellSeeds`,view:t.cellSeeds,type:`u32`,access:`read`},{name:`bandSeeds`,view:r.bandSeeds,type:`u32`,access:`read_write`}],invocationCount:s,declarations:a,body:`let column = index % GRID_WIDTH;
  let rowStart = (index / GRID_WIDTH) * BAND_SIZE;
  let rowEnd = min(rowStart + BAND_SIZE, GRID_HEIGHT);
  var topRow = NONE;
  var topId = NONE;
  var bottomRow = NONE;
  var bottomId = NONE;
  for (var row = rowStart; row < rowEnd; row++) {
    let seedId = cellSeeds[cellSeedsOffset + row * GRID_WIDTH + column];
    if (seedId != NONE) {
      if (topRow == NONE) {
        topRow = row;
        topId = seedId;
      }
      bottomRow = row;
      bottomId = seedId;
    }
  }
  let base = bandSeedsOffset + 4u * index;
  bandSeeds[base] = topRow;
  bandSeeds[base + 1u] = topId;
  bandSeeds[base + 2u] = bottomRow;
  bandSeeds[base + 3u] = bottomId;`}),o(e,{id:`${t.id}-column-carry`,operation:g,variant:`exact-column-carry`,bindings:[{name:`bandSeeds`,view:r.bandSeeds,type:`u32`,access:`read`},{name:`bandCarries`,view:r.bandCarries,type:`u32`,access:`read_write`}],invocationCount:n.width,workgroupSize:v,declarations:a,body:`let column = index;
  var lastRow = NONE;
  var lastId = NONE;
  for (var band = 0u; band < COLUMN_BAND_COUNT; band++) {
    let base = 4u * (band * GRID_WIDTH + column);
    bandCarries[bandCarriesOffset + base] = lastRow;
    bandCarries[bandCarriesOffset + base + 1u] = lastId;
    let bottomRow = bandSeeds[bandSeedsOffset + base + 2u];
    if (bottomRow != NONE) {
      lastRow = bottomRow;
      lastId = bandSeeds[bandSeedsOffset + base + 3u];
    }
  }
  lastRow = NONE;
  lastId = NONE;
  for (var step = 0u; step < COLUMN_BAND_COUNT; step++) {
    let base = 4u * ((COLUMN_BAND_COUNT - 1u - step) * GRID_WIDTH + column);
    bandCarries[bandCarriesOffset + base + 2u] = lastRow;
    bandCarries[bandCarriesOffset + base + 3u] = lastId;
    let topRow = bandSeeds[bandSeedsOffset + base];
    if (topRow != NONE) {
      lastRow = topRow;
      lastId = bandSeeds[bandSeedsOffset + base + 1u];
    }
  }`}),o(e,{id:`${t.id}-column-fill`,operation:g,variant:`exact-column-fill`,bindings:[{name:`cellSeeds`,view:t.cellSeeds,type:`u32`,access:`read`},{name:`bandCarries`,view:r.bandCarries,type:`u32`,access:`read`},{name:`columnRows`,view:t.columnRows,type:`u32`,access:`read_write`},{name:`columnIds`,view:t.columnIds,type:`u32`,access:`read_write`}],invocationCount:s,declarations:a,body:`let column = index % GRID_WIDTH;
  let rowStart = (index / GRID_WIDTH) * BAND_SIZE;
  let rowEnd = min(rowStart + BAND_SIZE, GRID_HEIGHT);
  let base = bandCarriesOffset + 4u * index;
  var lastRow = bandCarries[base];
  var lastId = bandCarries[base + 1u];
  for (var row = rowStart; row < rowEnd; row++) {
    let cell = row * GRID_WIDTH + column;
    let seedId = cellSeeds[cellSeedsOffset + cell];
    if (seedId != NONE) {
      lastRow = row;
      lastId = seedId;
    }
    columnRows[columnRowsOffset + cell] = lastRow;
    columnIds[columnIdsOffset + cell] = lastId;
  }
  lastRow = bandCarries[base + 2u];
  lastId = bandCarries[base + 3u];
  for (var row = rowEnd; row > rowStart; row--) {
    let cell = (row - 1u) * GRID_WIDTH + column;
    let seedId = cellSeeds[cellSeedsOffset + cell];
    if (seedId != NONE) {
      lastRow = row - 1u;
      lastId = seedId;
    }
    if (lastRow == NONE) { continue; }
    let forwardRow = columnRows[columnRowsOffset + cell];
    var useBackward = forwardRow == NONE;
    if (!useBackward) {
      let forwardDistance = row - 1u - forwardRow;
      let backwardDistance = lastRow - (row - 1u);
      useBackward = backwardDistance < forwardDistance ||
        (backwardDistance == forwardDistance && lastId < columnIds[columnIdsOffset + cell]);
    }
    if (useBackward) {
      columnRows[columnRowsOffset + cell] = lastRow;
      columnIds[columnIdsOffset + cell] = lastId;
    }
  }`})]}var C=`
/** Whether the candidate in column q beats the candidate in column v at cell x of this row. */
fn beatsAt(q: u32, qDy: u32, qId: u32, v: u32, vDy: u32, vId: u32, x: u32) -> bool {
  return isCloser(getAbsoluteDifference(x, q), qDy, qId, getAbsoluteDifference(x, v), vDy, vId);
}

fn getEntrySlot(column: u32, row: u32) -> u32 {
  return column * GRID_HEIGHT + row;
}

/**
 * Felzenszwalb-Huttenlocher insertion of the candidate in column q into the envelope whose tail is
 * 'tail': pops every tail entry q beats at its start (marking it off the envelope) and returns
 * vec2(new tail, start of q), start GRID_WIDTH when q never wins. The start is the first cell
 * after the tail's start where q wins, found by binary search with isCloser, so ties break exactly
 * on the smallest seed ID.
 */
fn insertEntry(q: u32, row: u32, tailIn: u32) -> vec2<u32> {
  let rowBase = row * GRID_WIDTH;
  let qDy = getAbsoluteDifference(columnRows[columnRowsOffset + rowBase + q], row);
  let qId = columnIds[columnIdsOffset + rowBase + q];
  var tail = tailIn;
  loop {
    if (tail == NONE) { return vec2<u32>(NONE, 0u); }
    let tailSlot = getEntrySlot(tail, row);
    let vStart = entryStarts[entryStartsOffset + tailSlot];
    let vDy = getAbsoluteDifference(columnRows[columnRowsOffset + rowBase + tail], row);
    let vId = columnIds[columnIdsOffset + rowBase + tail];
    if (beatsAt(q, qDy, qId, tail, vDy, vId, vStart)) {
      entryStarts[entryStartsOffset + tailSlot] = NONE;
      tail = entryPrevious[entryPreviousOffset + tailSlot];
      continue;
    }
    // First cell in (vStart, GRID_WIDTH) where q wins, or GRID_WIDTH when q never wins.
    var low = vStart + 1u;
    var high = GRID_WIDTH;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (beatsAt(q, qDy, qId, tail, vDy, vId, middle)) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    return vec2<u32>(tail, low);
  }
}`;function w(e,t){let{grid:n,scratch:r}=t,{rowBandCount:i,mergeRoundCount:s}=x(n),c=n.width*n.height,l=i*n.height,u=`${y(n,!0)}
const BAND_SIZE: u32 = 32u;
const ROW_BAND_COUNT: u32 = ${i}u;`,f=[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`columnRows`,view:t.columnRows,type:`u32`,access:`read`},{name:`columnIds`,view:t.columnIds,type:`u32`,access:`read`},{name:`entryNext`,view:r.entryNext,type:`u32`,access:`read_write`},{name:`entryPrevious`,view:r.entryPrevious,type:`u32`,access:`read_write`},{name:`entryStarts`,view:r.entryStarts,type:`u32`,access:`read_write`},{name:`blockEnds`,view:r.blockEnds,type:`u32`,access:`read_write`}],m=[o(e,{id:`${t.id}-row-bands`,operation:g,variant:`exact-row-bands`,bindings:f,invocationCount:l,declarations:`${u}\n${C}`,body:`let row = index % GRID_HEIGHT;
  let band = index / GRID_HEIGHT;
  let rowBase = row * GRID_WIDTH;
  let columnStart = band * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var head = NONE;
  var tail = NONE;
  for (var q = columnStart; q < columnEnd; q++) {
    let slot = getEntrySlot(q, row);
    entryStarts[entryStartsOffset + slot] = NONE;
    if (columnRows[columnRowsOffset + rowBase + q] == NONE) { continue; }
    let inserted = insertEntry(q, row, tail);
    tail = inserted.x;
    if (inserted.y < GRID_WIDTH) {
      entryStarts[entryStartsOffset + slot] = inserted.y;
      entryPrevious[entryPreviousOffset + slot] = tail;
      entryNext[entryNextOffset + slot] = NONE;
      if (tail == NONE) {
        head = q;
      } else {
        entryNext[entryNextOffset + getEntrySlot(tail, row)] = q;
      }
      tail = q;
    }
  }
  blockEnds[blockEndsOffset + 2u * index] = head;
  blockEnds[blockEndsOffset + 2u * index + 1u] = tail;`})];if(s>0){let r=d(e,{id:`${t.id}-row-merge-rounds`,operation:g,groupSize:s});m.push(r.writeNode);let a=Math.ceil(i/2)*n.height;for(let n=0;n<s;n++)m.push(o(e,{id:`${t.id}-row-merge-${n}`,operation:g,variant:`exact-row-merge`,bindings:[...f,{name:`mergeRound`,view:r.getSlot(n),type:`u32`,access:`read`}],invocationCount:a,declarations:`${u}\n${C}`,body:`let blockBands = 1u << mergeRound[mergeRoundOffset];
  let row = index % GRID_HEIGHT;
  let leftBand = (index / GRID_HEIGHT) * 2u * blockBands;
  let rightBand = leftBand + blockBands;
  if (rightBand >= ROW_BAND_COUNT) { return; }
  let leftEnds = blockEndsOffset + 2u * (leftBand * GRID_HEIGHT + row);
  let rightEnds = blockEndsOffset + 2u * (rightBand * GRID_HEIGHT + row);
  let leftHead = blockEnds[leftEnds];
  let rightTail = blockEnds[rightEnds + 1u];
  var tail = blockEnds[leftEnds + 1u];
  var current = blockEnds[rightEnds];
  var mergedHead = NONE;
  var reachedLocalState = false;
  loop {
    if (current == NONE) { break; }
    let slot = getEntrySlot(current, row);
    let localPrevious = entryPrevious[entryPreviousOffset + slot];
    let localStart = entryStarts[entryStartsOffset + slot];
    let localNext = entryNext[entryNextOffset + slot];
    let inserted = insertEntry(current, row, tail);
    tail = inserted.x;
    if (inserted.y >= GRID_WIDTH) {
      entryStarts[entryStartsOffset + slot] = NONE;
      current = localNext;
      continue;
    }
    entryStarts[entryStartsOffset + slot] = inserted.y;
    entryPrevious[entryPreviousOffset + slot] = tail;
    if (tail == NONE) {
      mergedHead = current;
    } else {
      entryNext[entryNextOffset + getEntrySlot(tail, row)] = current;
    }
    if (tail == localPrevious && inserted.y == localStart) {
      reachedLocalState = true;
      break;
    }
    tail = current;
    current = localNext;
  }
  // The left head survives unless the merge popped the whole left envelope.
  if (leftHead != NONE && entryStarts[entryStartsOffset + getEntrySlot(leftHead, row)] != NONE) {
    mergedHead = leftHead;
  }
  blockEnds[leftEnds] = mergedHead;
  blockEnds[leftEnds + 1u] = select(tail, rightTail, reachedLocalState);`}))}let h=`const GRID_WIDTH: u32 = ${n.width}u;
const GRID_HEIGHT: u32 = ${n.height}u;
const NONE: u32 = ${p}u;
const BAND_SIZE: u32 = 32u;
const ROW_BAND_COUNT: u32 = ${i}u;`;return m.push(a(e,{id:`${t.id}-row-marker-clear`,operation:g,view:r.startMarkers,type:`u32`,value:`0u`}),o(e,{id:`${t.id}-row-start-markers`,operation:g,variant:`exact-row-start-markers`,bindings:[{name:`entryStarts`,view:r.entryStarts,type:`u32`,access:`read`},{name:`startMarkers`,view:r.startMarkers,type:`u32`,access:`read_write`}],invocationCount:c,declarations:h,body:`let start = entryStarts[entryStartsOffset + index];
  if (start == NONE) { return; }
  let column = index / GRID_HEIGHT;
  let row = index % GRID_HEIGHT;
  startMarkers[startMarkersOffset + row * GRID_WIDTH + start] = column + 1u;`}),o(e,{id:`${t.id}-row-marker-bands`,operation:g,variant:`exact-row-marker-bands`,bindings:[{name:`startMarkers`,view:r.startMarkers,type:`u32`,access:`read`},{name:`bandMarkers`,view:r.bandMarkers,type:`u32`,access:`read_write`}],invocationCount:l,declarations:h,body:`let row = index % GRID_HEIGHT;
  let columnStart = (index / GRID_HEIGHT) * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var maximum = 0u;
  for (var x = columnStart; x < columnEnd; x++) {
    maximum = max(maximum, startMarkers[startMarkersOffset + row * GRID_WIDTH + x]);
  }
  bandMarkers[bandMarkersOffset + index] = maximum;`}),o(e,{id:`${t.id}-row-marker-carry`,operation:g,variant:`exact-row-marker-carry`,bindings:[{name:`bandMarkers`,view:r.bandMarkers,type:`u32`,access:`read_write`}],invocationCount:n.height,workgroupSize:v,declarations:h,body:`var running = 0u;
  for (var band = 0u; band < ROW_BAND_COUNT; band++) {
    let slot = bandMarkersOffset + band * GRID_HEIGHT + index;
    let bandMaximum = bandMarkers[slot];
    bandMarkers[slot] = running;
    running = max(running, bandMaximum);
  }`}),o(e,{id:`${t.id}-row-color`,operation:g,variant:`exact-row-color`,bindings:[{name:`columnRows`,view:t.columnRows,type:`u32`,access:`read`},{name:`columnIds`,view:t.columnIds,type:`u32`,access:`read`},{name:`startMarkers`,view:r.startMarkers,type:`u32`,access:`read`},{name:`bandMarkers`,view:r.bandMarkers,type:`u32`,access:`read`},{name:`nearestCells`,view:t.nearestCells,type:`u32`,access:`read_write`},{name:`nearestIds`,view:t.nearestIds,type:`u32`,access:`read_write`}],invocationCount:l,declarations:h,body:`let row = index % GRID_HEIGHT;
  let rowBase = row * GRID_WIDTH;
  let columnStart = (index / GRID_HEIGHT) * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var running = bandMarkers[bandMarkersOffset + index];
  for (var x = columnStart; x < columnEnd; x++) {
    let cell = rowBase + x;
    running = max(running, startMarkers[startMarkersOffset + cell]);
    if (running == 0u) {
      nearestCells[nearestCellsOffset + cell] = NONE;
      nearestIds[nearestIdsOffset + cell] = NONE;
      continue;
    }
    let column = running - 1u;
    nearestCells[nearestCellsOffset + cell] = columnRows[columnRowsOffset + rowBase + column] * GRID_WIDTH + column;
    nearestIds[nearestIdsOffset + cell] = columnIds[columnIdsOffset + rowBase + column];
  }`})),m}function T(e,t){let n=[],r=1;for(;r<Math.max(e.width,e.height);)r*=2;for(let e=r/2;e>=1;e/=2)n.push(e);return t===2?n.push(2,1):t===1&&n.push(1),n}function E(e,t){let{id:n,grid:i,buffers:a}=t,s=i.width*i.height,c=[o(e,{id:`${n}-jump-flood-initialize`,operation:g,variant:`jump-flood-initialize`,bindings:[{name:`cellSeeds`,view:t.cellSeeds,type:`u32`,access:`read`},{name:`targetCells`,view:a[0].cells,type:`u32`,access:`read_write`},{name:`targetIds`,view:a[0].ids,type:`u32`,access:`read_write`}],invocationCount:s,body:`let seedId = cellSeeds[cellSeedsOffset + index];
  targetCells[targetCellsOffset + index] = select(${p}u, index, seedId != ${p}u);
  targetIds[targetIdsOffset + index] = seedId;`})],l=T(i,t.refinementIterationCount);if(l.length===0)return{nodes:c,result:a[0]};let u=r(e,`${n}-jump-flood-step-words`,`uint32`,l.length*64);c.push(o(e,{id:`${n}-jump-flood-steps`,operation:g,variant:`jump-flood-steps`,bindings:[{name:`stepWords`,view:u,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const STEPS = array<u32, ${l.length}>(${l.map(e=>`${e}u`).join(`, `)});`,body:`for (var stepIndex = 0u; stepIndex < ${l.length}u; stepIndex++) {
    stepWords[stepWordsOffset + stepIndex * 64u] = STEPS[stepIndex];
  }`}));let d=a[0];for(let[r]of l.entries()){let l=d===a[0]?a[1]:a[0];c.push(o(e,{id:`${n}-jump-flood-${r}`,operation:g,variant:`jump-flood-step`,bindings:[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`sourceCells`,view:d.cells,type:`u32`,access:`read`},{name:`sourceIds`,view:d.ids,type:`u32`,access:`read`},{name:`targetCells`,view:l.cells,type:`u32`,access:`read_write`},{name:`targetIds`,view:l.ids,type:`u32`,access:`read_write`},{name:`jumpStep`,view:e.createDataView(u.buffer,{format:`uint32`,length:1,byteOffset:u.byteOffset+r*64*4}),type:`u32`,access:`read`}],invocationCount:s,declarations:y(i,!0),body:`let step = i32(jumpStep[jumpStepOffset]);
  let x = index % GRID_WIDTH;
  let y = index / GRID_WIDTH;
  var bestCell = sourceCells[sourceCellsOffset + index];
  var bestId = sourceIds[sourceIdsOffset + index];
  for (var offsetY = -1; offsetY <= 1; offsetY++) {
    for (var offsetX = -1; offsetX <= 1; offsetX++) {
      let neighborX = i32(x) + offsetX * step;
      let neighborY = i32(y) + offsetY * step;
      if ((offsetX == 0 && offsetY == 0) || neighborX < 0 || neighborY < 0 ||
          neighborX >= i32(GRID_WIDTH) || neighborY >= i32(GRID_HEIGHT)) { continue; }
      let neighbor = u32(neighborY) * GRID_WIDTH + u32(neighborX);
      let candidateCell = sourceCells[sourceCellsOffset + neighbor];
      if (candidateCell == NONE) { continue; }
      let candidateId = sourceIds[sourceIdsOffset + neighbor];
      let candidateDx = getAbsoluteDifference(x, candidateCell % GRID_WIDTH);
      let candidateDy = getAbsoluteDifference(y, candidateCell / GRID_WIDTH);
      if (bestCell == NONE || isCloser(
        candidateDx, candidateDy, candidateId,
        getAbsoluteDifference(x, bestCell % GRID_WIDTH), getAbsoluteDifference(y, bestCell / GRID_WIDTH), bestId
      )) {
        bestCell = candidateCell;
        bestId = candidateId;
      }
    }
  }
  targetCells[targetCellsOffset + index] = bestCell;
  targetIds[targetIdsOffset + index] = bestId;`})),d=l}return{nodes:c,result:d}}function D(e,t){let{grid:n}=t,r=[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`nearestCells`,view:t.nearestCells,type:`u32`,access:`read`},{name:`nearestIds`,view:t.nearestIds,type:`u32`,access:`read`},{name:`distanceBits`,view:t.distances,type:`u32`,access:`read_write`}];return t.allocation&&r.push({name:`allocation`,view:t.allocation,type:`u32`,access:`read_write`}),t.nearestCellsOutput&&r.push({name:`nearestCellsOutput`,view:t.nearestCellsOutput,type:`u32`,access:`read_write`}),t.withinDistance&&r.push({name:`withinDistance`,view:t.withinDistance,type:`u32`,access:`read_write`}),o(e,{id:`${t.id}-finalize`,operation:g,variant:`finalize`,bindings:r,invocationCount:n.width*n.height,declarations:`${y(n,!0)}
${_}`,body:`var nearestCell = nearestCells[nearestCellsOffset + index];
  var seedId = nearestIds[nearestIdsOffset + index];
  var bits = 0x7f800000u;
  if (nearestCell != NONE) {
    let dx = getAbsoluteDifference(index % GRID_WIDTH, nearestCell % GRID_WIDTH);
    let dy = getAbsoluteDifference(index / GRID_WIDTH, nearestCell / GRID_WIDTH);
    var distance: f32;
    if (isIsotropic()) {
      // Exact integer key, so the only roundings are the square root and the scale.
      distance = settings[settingsOffset + 2u] * getRoundedSquareRoot(dx * dx + dy * dy, 0);
    } else {
      let key = getAnisotropicKey(dx, dy);
      let keyBits = bitcast<u32>(key);
      let keyExponent = (keyBits >> 23u) & 0xffu;
      if (keyExponent == 0u || keyExponent == 0xffu) {
        distance = sqrt(key);
      } else {
        distance = getRoundedSquareRoot((keyBits & 0x7fffffu) | 0x800000u, i32(keyExponent) - 150);
      }
    }
    if (distance > settings[settingsOffset + 6u]) {
      nearestCell = NONE;
      seedId = NONE;
    } else {
      bits = bitcast<u32>(distance);
    }
  }
  if (nearestCell == NONE) { seedId = NONE; }
  distanceBits[distanceBitsOffset + index] = bits;
  ${t.allocation?`allocation[allocationOffset + index] = seedId;`:``}
  ${t.nearestCellsOutput?`nearestCellsOutput[nearestCellsOutputOffset + index] = nearestCell;`:``}
  ${t.withinDistance?`withinDistance[withinDistanceOffset + index] = select(0u, 1u, nearestCell != NONE);`:``}`})}var O=class{id;props;mode;jumpFloodRefinementIterationCount;cellCount;constructor(e){this.id=e.id??`distance-field`,this.props=e,this.mode=e.mode??`exact`,this.jumpFloodRefinementIterationCount=e.jumpFloodRefinementIterationCount??1;let{id:n}=this,{output:r}=e;this.cellCount=l(n,e.grid);let{width:a,height:o}=e.grid;for(let[e,t]of[[`width`,a],[`height`,o]])if(t>32768)throw Error(`${n} grid ${e} must be an integer from 1 to ${m}`);if(![`exact`,`jump-flood`].includes(this.mode))throw Error(`${n} mode must be exact or jump-flood`);if(![0,1,2].includes(this.jumpFloodRefinementIterationCount))throw Error(`${n} jumpFloodRefinementIterationCount must be 0, 1, or 2`);if(u(n,e.parameters,8),!e.seedPositions&&!e.seedMask)throw Error(`${n} requires seedPositions or seedMask`);if(e.seedPositions&&i(e.seedPositions,[`float32x2`],`${n} seedPositions`),(e.seedIds||e.seedCount)&&!e.seedPositions)throw Error(`${n} seedIds and seedCount require seedPositions`);for(let[t,a]of[[`seedIds`,e.seedIds],[`seedCount`,e.seedCount],[`seedMask`,e.seedMask],[`output.allocation`,r.allocation],[`output.nearestCells`,r.nearestCells],[`output.withinDistance`,r.withinDistance]])a&&i(a,[`uint32`],`${n} ${t}`);if(i(r.distances,[`float32`],`${n} output.distances`),e.seedIds&&e.seedIds.length!==e.seedPositions?.length)throw Error(`${n} seedIds length must equal seedPositions length`);if(e.seedCount&&e.seedCount.length!==1)throw Error(`${n} seedCount must contain exactly one row`);for(let[t,i]of[[`seedMask`,e.seedMask],[`output.distances`,r.distances],[`output.allocation`,r.allocation],[`output.nearestCells`,r.nearestCells],[`output.withinDistance`,r.withinDistance]])if(i&&i.length!==this.cellCount)throw Error(`${n} ${t} must contain one row per cell`);if(r.texture&&(r.texture.format!==`r32float`||r.texture.width!==a||r.texture.height!==o))throw Error(`${n} output.texture must be r32float with a width x height extent`);let s=[r.distances,r.allocation,r.nearestCells,r.withinDistance];t(n,s,[e.parameters,e.seedPositions,e.seedIds,e.seedCount,e.seedMask]);let c=s.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(c).size!==c.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(t){let{id:i,props:a,cellCount:o}=this,{output:s,parameters:c}=a,l={width:a.grid.width,height:a.grid.height};if(e(i,t,[c,a.seedPositions,a.seedIds,a.seedCount,a.seedMask,s.distances,s.allocation,s.nearestCells,s.withinDistance]),s.texture&&s.texture.texture.graph!==t)throw Error(`${i} views must belong to the target graph`);let u=e=>r(t,`${i}-${e}`,`uint32`,o),d=u(`cell-seeds`),p=b(t,{id:i,grid:l,settings:c,cellSeeds:d,seedMask:a.seedMask,seedPositions:a.seedPositions,seedIds:a.seedIds,seedCount:a.seedCount}),m;if(this.mode===`exact`){let e=u(`column-rows`),n=u(`column-ids`);m={cells:u(`nearest-cells`),ids:u(`nearest-ids`)};let a=x(l),o=(e,n)=>r(t,`${i}-${e}`,`uint32`,n);p.push(...S(t,{id:i,grid:l,cellSeeds:d,columnRows:e,columnIds:n,scratch:{bandSeeds:o(`column-band-seeds`,4*a.columnBandCount*l.width),bandCarries:o(`column-band-carries`,4*a.columnBandCount*l.width)}}),...w(t,{id:i,grid:l,settings:c,columnRows:e,columnIds:n,scratch:{entryNext:u(`entry-next`),entryPrevious:u(`entry-previous`),entryStarts:u(`entry-starts`),blockEnds:o(`block-ends`,2*a.rowBandCount*l.height),startMarkers:u(`start-markers`),bandMarkers:o(`band-markers`,a.rowBandCount*l.height)},nearestCells:m.cells,nearestIds:m.ids}))}else{let e=E(t,{id:i,grid:l,settings:c,cellSeeds:d,buffers:[{cells:u(`jump-flood-cells-0`),ids:u(`jump-flood-ids-0`)},{cells:u(`jump-flood-cells-1`),ids:u(`jump-flood-ids-1`)}],refinementIterationCount:this.jumpFloodRefinementIterationCount});p.push(...e.nodes),m=e.result}if(p.push(D(t,{id:i,grid:l,settings:c,nearestCells:m.cells,nearestIds:m.ids,distances:s.distances,allocation:s.allocation,nearestCellsOutput:s.nearestCells,withinDistance:s.withinDistance})),s.texture){let e=s.texture;p.push(...n(t,()=>new f({id:`${i}-texture`,input:{id:`${i}-texture-band`,format:`float32`,storage:{kind:`buffer`,values:s.distances}},output:e}).addToGraph(t)))}return p}};export{p as n,h as r,O as t};