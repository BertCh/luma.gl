import{D as e,E as t,S as n,p as r}from"./analysis-frame-Blu0tYdw.js";import{_ as i,d as a,f as o,h as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{t as u}from"./gpu-compaction-Cpx1OCRM.js";import{r as d,s as f}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as p}from"./geodesic-wgsl-DS0CJZ42.js";import{d as m,i as h,o as g,r as _,s as v}from"./raster-relaxation-BJ40xgro.js";var y=`GPULineSimplification`,b=2139095040,x=`
const LINE_MAXIMUM_FINITE: f32 = bitcast<f32>(0x7f7fffffu);

// Runtime zero: the kernel stores \`leftAnchor >> 31u\`, a row index loaded from memory (rows are
// below 2^31), which the compiler cannot fold, unlike thread IDs or \`arrayLength\` whose ranges it
// knows. ORing it into a
// product's bits keeps the compiler from contracting \`a * b + c\` or \`a * b > c\` into a fused
// multiply-add (an unrounded product), which Metal otherwise does, changing results by a few ULP.
var<private> lineOpaqueZero: u32;

fn lineProduct(left: f32, right: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(left * right) | lineOpaqueZero);
}

fn lineNextUp(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) + 1u);
}

fn lineNextDown(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) - 1u);
}

fn lineDivideFloor(numerator: f32, denominator: f32) -> f32 {
  if (!(numerator > 0.0)) {
    return 0.0;
  }
  var quotient = numerator / denominator;
  if (!(quotient >= 0.0)) {
    quotient = 0.0;
  }
  quotient = min(quotient, LINE_MAXIMUM_FINITE);
  for (var step = 0u; step < 16u; step++) {
    if (quotient > 0.0 && lineProduct(quotient, denominator) > numerator) {
      quotient = lineNextDown(quotient);
    } else {
      break;
    }
  }
  for (var step = 0u; step < 16u; step++) {
    let next = lineNextUp(quotient);
    if (lineProduct(next, denominator) <= numerator) {
      quotient = next;
    } else {
      break;
    }
  }
  return quotient;
}

fn lineSqrtFloor(value: f32) -> f32 {
  if (!(value > 0.0)) {
    return 0.0;
  }
  var root = sqrt(value);
  if (!(root >= 0.0)) {
    root = 0.0;
  }
  root = min(root, LINE_MAXIMUM_FINITE);
  for (var step = 0u; step < 16u; step++) {
    if (root > 0.0 && lineProduct(root, root) > value) {
      root = lineNextDown(root);
    } else {
      break;
    }
  }
  for (var step = 0u; step < 16u; step++) {
    let next = lineNextUp(root);
    if (lineProduct(next, next) <= value) {
      root = next;
    } else {
      break;
    }
  }
  return root;
}

fn linePointDistance(px: f32, py: f32, ax: f32, ay: f32) -> f32 {
  let deltaX = px - ax;
  let deltaY = py - ay;
  return lineSqrtFloor(lineProduct(deltaX, deltaX) + lineProduct(deltaY, deltaY));
}

fn lineSegmentDistance(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
  let segmentX = bx - ax;
  let segmentY = by - ay;
  let offsetX = px - ax;
  let offsetY = py - ay;
  let lengthSquared = lineProduct(segmentX, segmentX) + lineProduct(segmentY, segmentY);
  let projection = lineProduct(offsetX, segmentX) + lineProduct(offsetY, segmentY);
  if (lengthSquared <= 0.0 || projection <= 0.0) {
    return linePointDistance(px, py, ax, ay);
  }
  if (projection >= lengthSquared) {
    return linePointDistance(px, py, bx, by);
  }
  let segmentLength = lineSqrtFloor(lengthSquared);
  if (segmentLength <= 0.0) {
    return linePointDistance(px, py, ax, ay);
  }
  let cross = abs(lineProduct(segmentX, offsetY) - lineProduct(segmentY, offsetX));
  return lineDivideFloor(cross, segmentLength);
}

fn lineTimeRatioDistance(
  px: f32, py: f32, pt: f32, ax: f32, ay: f32, at: f32, bx: f32, by: f32, bt: f32
) -> f32 {
  let duration = bt - at;
  if (!(duration > 0.0)) {
    return linePointDistance(px, py, ax, ay);
  }
  let elapsed = pt - at;
  var ratio = lineDivideFloor(abs(elapsed), duration);
  if (elapsed < 0.0) {
    ratio = -ratio;
  }
  let expectedX = ax + lineProduct(ratio, bx - ax);
  let expectedY = ay + lineProduct(ratio, by - ay);
  return linePointDistance(px, py, expectedX, expectedY);
}
`;function S(e){return`${p}
const LINE_SPHERE_RADIUS: f32 = ${e};

fn lineGeodesicSegmentDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  let fraction = geodesicSegmentFraction(point, start, end);
  return geodesicSegmentDistance(point, start, end, fraction, LINE_SPHERE_RADIUS);
}

// Great-circle distance to the chord position linearly interpolated (longitude difference wrapped)
// at the timestamp of the point.
fn lineGeodesicTimeRatioDistance(
  point: vec2<f32>, pointTime: f32, start: vec2<f32>, startTime: f32, end: vec2<f32>, endTime: f32
) -> f32 {
  let duration = endTime - startTime;
  if (!(duration > 0.0)) {
    return geodesicHaversineDistance(point, start, LINE_SPHERE_RADIUS);
  }
  let ratio = (pointTime - startTime) / duration;
  let expected = start + vec2<f32>(
    geodesicWrapLongitudeDelta(end.x - start.x),
    end.y - start.y
  ) * ratio;
  return geodesicHaversineDistance(point, expected, LINE_SPHERE_RADIUS);
}
`}var C=e=>`let leftAnchor = leftAnchors[leftAnchorsOffset + ${e}];
  let rightAnchor = rightAnchors[rightAnchorsOffset + ${e}];
  if (!(leftAnchor < ${e} && ${e} < rightAnchor)) {
    return;
  }`;function w(e,t){let n=t.lineOffsets.length-1;return f(e,{id:t.id,operation:y,variant:`importance-init`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`leftAnchors`,view:t.scratch.leftAnchors,type:`u32`,access:`read_write`},{name:`rightAnchors`,view:t.scratch.rightAnchors,type:`u32`,access:`read_write`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,declarations:`const LINE_COUNT: u32 = ${n}u;
const ENDPOINT_IMPORTANCE: u32 = ${b}u;`,body:`let firstRow = lineOffsets[lineOffsetsOffset];
  let endRow = lineOffsets[lineOffsetsOffset + LINE_COUNT];
  if (index < firstRow || index >= endRow) {
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    importance[importanceOffset + index] = 0u;
    return;
  }
  // Last line whose start is <= index (empty lines share a start with the next line).
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let line = low - 1u;
  let lineStart = lineOffsets[lineOffsetsOffset + line];
  let lineLast = lineOffsets[lineOffsetsOffset + line + 1u] - 1u;
  if (index == lineStart || index >= lineLast) {
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;
    return;
  }
  leftAnchors[leftAnchorsOffset + index] = lineStart;
  rightAnchors[rightAnchorsOffset + index] = lineLast;
  importance[importanceOffset + index] = 0u;`})}function T(e,t){let{scratch:n,gate:r}=t,i=t.finishSpanLimit??0,a=t.importance.length,o=[{name:`leftAnchors`,view:n.leftAnchors,type:`u32`,access:`read`},{name:`rightAnchors`,view:n.rightAnchors,type:`u32`,access:`read`}],s={name:`bestKeys`,view:n.bestKeys,type:`atomic<u32>`,access:`read_write`},c={name:`bestRows`,view:n.bestRows,type:`atomic<u32>`,access:`read_write`},l={operation:y,invocationCount:a,condition:r.condition,extraResources:r.extraResources},u=t.metric===`time-ratio`,d=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...o,{name:`distanceKeys`,view:n.distanceKeys,type:`u32`,access:`read_write`},s];u&&t.timestamps&&d.push({name:`timestamps`,view:t.timestamps,type:`f32`,access:`read`});let p=(e,t)=>`positions[positionsOffset + 2u * ${e} + ${t}u]`,m=e=>`timestamps[timestampsOffset + ${e}]`,h=e=>`vec2<f32>(${p(e,0)}, ${p(e,1)})`,g=t.sphereRadius===void 0?x:`${x}${S(t.sphereRadius)}`,_=t.sphereRadius===void 0?t.metric:`${t.metric}-sphere`,v=(e,n,r)=>t.sphereRadius===void 0?u?`lineTimeRatioDistance(${p(e,0)}, ${p(e,1)}, ${m(e)},
      ${p(n,0)}, ${p(n,1)}, ${m(n)},
      ${p(r,0)}, ${p(r,1)}, ${m(r)})`:`lineSegmentDistance(${p(e,0)}, ${p(e,1)},
      ${p(n,0)}, ${p(n,1)},
      ${p(r,0)}, ${p(r,1)})`:u?`lineGeodesicTimeRatioDistance(${h(e)}, ${m(e)}, ${h(n)}, ${m(n)}, ${h(r)}, ${m(r)})`:`lineGeodesicSegmentDistance(${h(e)}, ${h(n)}, ${h(r)})`,b=v(`index`,`leftAnchor`,`rightAnchor`);return[i>0?f(e,{...l,id:`${t.id}-reset`,variant:`importance-reset-finish-${_}`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`leftAnchors`,view:n.leftAnchors,type:`atomic<u32>`,access:`read_write`},{name:`rightAnchors`,view:n.rightAnchors,type:`atomic<u32>`,access:`read_write`},s,c,{name:`importance`,view:t.importance,type:`u32`,access:`read_write`},...u&&t.timestamps?[{name:`timestamps`,view:t.timestamps,type:`f32`,access:`read`}]:[]],declarations:`${g}
const FINISH_SPAN_LIMIT: u32 = ${i}u;

// Distance key of a row against the chord (left, right), as in the distance node.
fn getDistanceKey(row: u32, leftAnchor: u32, rightAnchor: u32) -> u32 {
  let distance = ${v(`row`,`leftAnchor`,`rightAnchor`)};
  return select(bitcast<u32>(distance), 0u, distance <= 0.0);
}

// Decides every interior row of the open interval (first, last) with the recursion the rounds
// would run, depth first from one lane. Importance is min(key, parent) where parent is the
// importance of the split that created the interval; both children of a split row inherit that
// row's importance as parent because it never exceeds either anchor's.
fn finishInterval(first: u32, last: u32) {
  lineOpaqueZero = first >> 31u;
  var stackLeft: array<u32, ${i}>;
  var stackRight: array<u32, ${i}>;
  var stackParent: array<u32, ${i}>;
  stackLeft[0] = first;
  stackRight[0] = last;
  stackParent[0] = min(importance[importanceOffset + first], importance[importanceOffset + last]);
  var depth = 1u;
  while (depth > 0u) {
    depth--;
    let left = stackLeft[depth];
    let right = stackRight[depth];
    let parent = stackParent[depth];
    var bestKey = 0u;
    var bestRow = 0xffffffffu;
    for (var row = left + 1u; row < right; row++) {
      let key = getDistanceKey(row, left, right);
      // Strict comparison keeps the smallest row on ties.
      if (key > bestKey) {
        bestKey = key;
        bestRow = row;
      }
    }
    if (bestKey == 0u) {
      for (var row = left + 1u; row < right; row++) {
        importance[importanceOffset + row] = 0u;
        atomicStore(&leftAnchors[leftAnchorsOffset + row], row);
        atomicStore(&rightAnchors[rightAnchorsOffset + row], row);
      }
      continue;
    }
    let splitImportance = min(bestKey, parent);
    importance[importanceOffset + bestRow] = splitImportance;
    atomicStore(&leftAnchors[leftAnchorsOffset + bestRow], bestRow);
    atomicStore(&rightAnchors[rightAnchorsOffset + bestRow], bestRow);
    if (bestRow > left + 1u) {
      stackLeft[depth] = left;
      stackRight[depth] = bestRow;
      stackParent[depth] = splitImportance;
      depth++;
    }
    if (right > bestRow + 1u) {
      stackLeft[depth] = bestRow;
      stackRight[depth] = right;
      stackParent[depth] = splitImportance;
      depth++;
    }
  }
}`,body:`let leftAnchor = atomicLoad(&leftAnchors[leftAnchorsOffset + index]);
  let rightAnchor = atomicLoad(&rightAnchors[rightAnchorsOffset + index]);
  if (!(leftAnchor < index && index < rightAnchor)) {
    return;
  }
  if (rightAnchor - leftAnchor - 1u <= FINISH_SPAN_LIMIT) {
    // The first interior row finishes the whole interval; its siblings have nothing to reset.
    if (index == leftAnchor + 1u) {
      finishInterval(leftAnchor, rightAnchor);
    }
    return;
  }
  atomicStore(&bestKeys[bestKeysOffset + leftAnchor], 0u);
  atomicStore(&bestRows[bestRowsOffset + leftAnchor], 0xffffffffu);`}):f(e,{...l,id:`${t.id}-reset`,variant:`importance-reset`,bindings:[...o,s,c],body:`${C(`index`)}
  atomicStore(&bestKeys[bestKeysOffset + leftAnchor], 0u);
  atomicStore(&bestRows[bestRowsOffset + leftAnchor], 0xffffffffu);`}),f(e,{...l,id:`${t.id}-distance`,variant:`importance-distance-${_}`,bindings:d,declarations:g,body:`${C(`index`)}
  lineOpaqueZero = leftAnchor >> 31u;
  let distance = ${b};
  // Canonical +0 so equal distances always have equal bits.
  let key = select(bitcast<u32>(distance), 0u, distance <= 0.0);
  distanceKeys[distanceKeysOffset + index] = key;
  atomicMax(&bestKeys[bestKeysOffset + leftAnchor], key);`}),f(e,{...l,id:`${t.id}-argmax`,variant:`importance-argmax`,bindings:[...o,{name:`distanceKeys`,view:n.distanceKeys,type:`u32`,access:`read`},s,c],body:`${C(`index`)}
  if (distanceKeys[distanceKeysOffset + index] == atomicLoad(&bestKeys[bestKeysOffset + leftAnchor])) {
    atomicMin(&bestRows[bestRowsOffset + leftAnchor], index);
  }`}),f(e,{...l,id:`${t.id}-split`,variant:`importance-split`,bindings:[{name:`leftAnchors`,view:n.leftAnchors,type:`u32`,access:`read_write`},{name:`rightAnchors`,view:n.rightAnchors,type:`u32`,access:`read_write`},{name:`distanceKeys`,view:n.distanceKeys,type:`u32`,access:`read`},s,c,{name:`importance`,view:t.importance,type:`u32`,access:`read_write`},{name:`status`,view:t.status,type:`atomic<u32>`,access:`read_write`}],body:`${C(`index`)}
  if (atomicLoad(&bestKeys[bestKeysOffset + leftAnchor]) == 0u) {
    importance[importanceOffset + index] = 0u;
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    return;
  }
  let splitRow = atomicLoad(&bestRows[bestRowsOffset + leftAnchor]);
  if (index == splitRow) {
    // Non-negative f32 bits order like their values, so u32 min is f32 min.
    let parentImportance = min(
      importance[importanceOffset + leftAnchor],
      importance[importanceOffset + rightAnchor]
    );
    importance[importanceOffset + index] = min(distanceKeys[distanceKeysOffset + index], parentImportance);
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    return;
  }
  if (index < splitRow) {
    rightAnchors[rightAnchorsOffset + index] = splitRow;
  } else {
    leftAnchors[leftAnchorsOffset + index] = splitRow;
  }
  atomicStore(&status[statusOffset], 1u);`})]}function E(e,t){return f(e,{id:t.id,operation:y,variant:`importance-unresolved`,bindings:[{name:`leftAnchors`,view:t.scratch.leftAnchors,type:`u32`,access:`read`},{name:`rightAnchors`,view:t.scratch.rightAnchors,type:`u32`,access:`read`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,body:`${C(`index`)}
  importance[importanceOffset + index] = min(
    importance[importanceOffset + leftAnchor],
    importance[importanceOffset + rightAnchor]
  );`})}function D(e,t){let n=t.lineOffsets.length-1;return f(e,{id:t.id,operation:y,variant:`keep-mask`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`importance`,view:t.importance,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`outputMask`,view:t.outputMask,type:`u32`,access:`read_write`},{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,declarations:`const LINE_COUNT: u32 = ${n}u;`,body:`rowIds[rowIdsOffset + index] = index;
  let tolerance = parameters[parametersOffset];
  let inLine = index >= lineOffsets[lineOffsetsOffset] &&
    index < lineOffsets[lineOffsetsOffset + LINE_COUNT];
  let keep = inLine && importance[importanceOffset + index] > tolerance;
  outputMask[outputMaskOffset + index] = select(0u, 1u, keep);`})}function O(e,t){let n=t.lineOffsets.length-1,r=[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`compactRows`,view:t.compactRows,type:`u32`,access:`read`},{name:`requiredCount`,view:t.requiredCount,type:`u32`,access:`read`}];return t.lineCounts&&r.push({name:`lineCounts`,view:t.lineCounts,type:`u32`,access:`read_write`}),t.lineStarts&&r.push({name:`lineStarts`,view:t.lineStarts,type:`u32`,access:`read_write`}),f(e,{id:t.id,operation:y,variant:`kept-ranges`,bindings:r,invocationCount:n,declarations:`const COMPACT_CAPACITY: u32 = ${t.compactRows.length}u;

// In-kernel lower bound on data-dependent keys; GPUGallopingSearch is a batch node and would
// add a pass and a buffer.
fn lowerBoundKeptRow(row: u32, count: u32) -> u32 {
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (compactRows[compactRowsOffset + middle] < row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,body:`let keptCount = min(requiredCount[requiredCountOffset], COMPACT_CAPACITY);
  let lineStart = lineOffsets[lineOffsetsOffset + index];
  let lineEnd = max(lineOffsets[lineOffsetsOffset + index + 1u], lineStart);
  let first = lowerBoundKeptRow(lineStart, keptCount);
  let last = lowerBoundKeptRow(lineEnd, keptCount);
  ${t.lineCounts?`lineCounts[lineCountsOffset + index] = last - first;`:``}
  ${t.lineStarts?`lineStarts[lineStartsOffset + index] = first;`:``}`})}function k(t,n=new Float32Array(4)){if(n.length<4)throw Error(`Line simplification target must hold 4 elements`);let{tolerance:r}=t;if(!Number.isFinite(r)||r<0)throw Error(`Line simplification tolerance must be finite and non-negative`);return(t.method??`douglas-peucker`)===`douglas-peucker`&&e(`GPULineSimplification`,t.spatialContext,r),n.set([r,0,0,0]),n}var A=`GPULineSimplification`,j=`let previousRow = previousRows[previousRowsOffset + index];
  let nextRow = nextRows[nextRowsOffset + index];
  if (!(previousRow < index && index < nextRow)) {
    return;
  }`;function M(e,t){let n=t.lineOffsets.length-1,{scratch:r}=t;return f(e,{id:t.id,operation:A,variant:`visvalingam-init`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`previousRows`,view:r.previousRows,type:`u32`,access:`read_write`},{name:`nextRows`,view:r.nextRows,type:`u32`,access:`read_write`},{name:`areaKeys`,view:r.areaKeys,type:`u32`,access:`read_write`},{name:`floorKeys`,view:r.floorKeys,type:`u32`,access:`read_write`},{name:`pendingFlags`,view:r.pendingFlags,type:`u32`,access:`read_write`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,declarations:`const LINE_COUNT: u32 = ${n}u;
const ENDPOINT_IMPORTANCE: u32 = ${b}u;`,body:`areaKeys[areaKeysOffset + index] = 0u;
  floorKeys[floorKeysOffset + index] = 0u;
  pendingFlags[pendingFlagsOffset + index] = 0u;
  previousRows[previousRowsOffset + index] = index;
  nextRows[nextRowsOffset + index] = index;
  importance[importanceOffset + index] = 0u;
  let firstRow = lineOffsets[lineOffsetsOffset];
  let endRow = lineOffsets[lineOffsetsOffset + LINE_COUNT];
  if (index < firstRow || index >= endRow) {
    return;
  }
  // Last line whose start is <= index (empty lines share a start with the next line).
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let line = low - 1u;
  let lineStart = lineOffsets[lineOffsetsOffset + line];
  let lineLast = lineOffsets[lineOffsetsOffset + line + 1u] - 1u;
  if (index == lineStart || index >= lineLast) {
    importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;
    return;
  }
  previousRows[previousRowsOffset + index] = index - 1u;
  nextRows[nextRowsOffset + index] = index + 1u;`})}function N(e,t){let{scratch:n,gate:r,neighborhoodRadius:i}=t,a={operation:A,invocationCount:t.importance.length,condition:r.condition,extraResources:r.extraResources},o=e=>[{name:`previousRows`,view:n.previousRows,type:`u32`,access:e},{name:`nextRows`,view:n.nextRows,type:`u32`,access:e}],s={name:`floorKeys`,view:n.floorKeys,type:`atomic<u32>`,access:`read_write`},c=(e,t)=>`positions[positionsOffset + 2u * ${e} + ${t}u]`;return[f(e,{...a,id:`${t.id}-area`,variant:`visvalingam-area`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...o(`read`),{name:`areaKeys`,view:n.areaKeys,type:`u32`,access:`read_write`}],declarations:x,body:`${j}
  lineOpaqueZero = previousRow >> 31u;
  let originX = ${c(`previousRow`,0)};
  let originY = ${c(`previousRow`,1)};
  let toMiddleX = ${c(`index`,0)} - originX;
  let toMiddleY = ${c(`index`,1)} - originY;
  let toNextX = ${c(`nextRow`,0)} - originX;
  let toNextY = ${c(`nextRow`,1)} - originY;
  let cross = lineProduct(toMiddleX, toNextY) - lineProduct(toMiddleY, toNextX);
  // abs() makes the key a canonical non-negative f32 whose bits order like its value.
  areaKeys[areaKeysOffset + index] = bitcast<u32>(abs(cross) * 0.5);`}),f(e,{...a,id:`${t.id}-mark`,variant:`visvalingam-mark-${i}`,bindings:[...o(`read`),{name:`areaKeys`,view:n.areaKeys,type:`u32`,access:`read`},s,{name:`pendingFlags`,view:n.pendingFlags,type:`u32`,access:`read_write`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`},{name:`status`,view:t.status,type:`atomic<u32>`,access:`read_write`}],declarations:`const NEIGHBORHOOD_RADIUS: u32 = ${i}u;

// True when a surviving row within the neighborhood orders before (key, row).
fn hasSmallerNeighbor(row: u32, key: u32, forward: bool) -> bool {
  var cursor = row;
  for (var step = 0u; step < NEIGHBORHOOD_RADIUS; step++) {
    cursor = select(previousRows[previousRowsOffset + cursor], nextRows[nextRowsOffset + cursor], forward);
    // Endpoints and rows outside a line are not undecided interior rows: stop the walk.
    if (!(previousRows[previousRowsOffset + cursor] < cursor &&
        cursor < nextRows[nextRowsOffset + cursor])) {
      return false;
    }
    let neighborKey = areaKeys[areaKeysOffset + cursor];
    if (neighborKey < key || (neighborKey == key && cursor < row)) {
      return true;
    }
  }
  return false;
}`,body:`${j}
  let key = areaKeys[areaKeysOffset + index];
  if (hasSmallerNeighbor(index, key, false) || hasSmallerNeighbor(index, key, true)) {
    atomicStore(&status[statusOffset], 1u);
    return;
  }
  importance[importanceOffset + index] = max(key, atomicLoad(&floorKeys[floorKeysOffset + index]));
  pendingFlags[pendingFlagsOffset + index] = 1u;`}),f(e,{...a,id:`${t.id}-unlink`,variant:`visvalingam-unlink`,bindings:[...o(`read_write`),s,{name:`pendingFlags`,view:n.pendingFlags,type:`u32`,access:`read_write`},{name:`importance`,view:t.importance,type:`u32`,access:`read`}],body:`if (pendingFlags[pendingFlagsOffset + index] == 0u) {
    return;
  }
  let previousRow = previousRows[previousRowsOffset + index];
  let nextRow = nextRows[nextRowsOffset + index];
  let effectiveArea = importance[importanceOffset + index];
  nextRows[nextRowsOffset + previousRow] = nextRow;
  previousRows[previousRowsOffset + nextRow] = previousRow;
  atomicMax(&floorKeys[floorKeysOffset + previousRow], effectiveArea);
  atomicMax(&floorKeys[floorKeysOffset + nextRow], effectiveArea);
  previousRows[previousRowsOffset + index] = index;
  nextRows[nextRowsOffset + index] = index;
  pendingFlags[pendingFlagsOffset + index] = 0u;`})]}function P(e,t){return f(e,{id:t.id,operation:A,variant:`visvalingam-unresolved`,bindings:[{name:`previousRows`,view:t.scratch.previousRows,type:`u32`,access:`read`},{name:`nextRows`,view:t.scratch.nextRows,type:`u32`,access:`read`},{name:`importance`,view:t.importance,type:`u32`,access:`read_write`}],invocationCount:t.importance.length,declarations:`const ENDPOINT_IMPORTANCE: u32 = ${b}u;`,body:`${j}
  importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;`})}var F=`GPULineSimplification`,I=Object.freeze({"douglas-peucker":Object.freeze({...r,measure:`distance`}),visvalingam:Object.freeze({...r,measure:`area`})});function L(e){let{output:t}=e;return{selection:t?.ids?t:void 0,status:t?.status}}var R=1024,z=class{id;props;spatialContext;sphereRadius;method;neighborhoodRadius;metric;maximumIterations;finishSpanLimit;computeImportance;constructor(e){this.id=e.id??`line-simplification`,this.method=e.method??`douglas-peucker`,this.spatialContext=t(this.id,e.spatialContext,this.method===`visvalingam`?I.visvalingam:I[`douglas-peucker`]),this.sphereRadius=n(this.spatialContext),this.props=e,this.metric=e.metric??`segment`,this.neighborhoodRadius=e.neighborhoodRadius??3,this.maximumIterations=e.maximumIterations??(this.method===`visvalingam`?256:64),this.computeImportance=e.computeImportance??!0,this.finishSpanLimit=e.finishSpanLimit??32,a(this.id,`finishSpanLimit`,this.finishSpanLimit,0,128);let{id:r,metric:i,maximumIterations:c,method:u,neighborhoodRadius:d}=this,{selection:f,status:p}=L(e);if(u!==`douglas-peucker`&&u!==`visvalingam`)throw Error(`${r} method must be 'douglas-peucker' or 'visvalingam'`);if(u===`visvalingam`){if(this.sphereRadius!==void 0)throw Error(`${r} method 'visvalingam' is planar only; project positions to a planar frame first`);if(i!==`segment`)throw Error(`${r} method 'visvalingam' does not support metric '${i}'`);a(r,`neighborhoodRadius`,d,1,8)}if(i!==`segment`&&i!==`time-ratio`)throw Error(`${r} metric must be 'segment' or 'time-ratio'`);a(r,`maximumIterations`,c,1,R),l(e.positions,[`float32x2`],`${r} positions`);let m=e.positions.length;if(m<1)throw Error(`${r} needs at least one position`);if(l(e.lineOffsets,[`uint32`],`${r} lineOffsets`),e.lineOffsets.length<2)throw Error(`${r} lineOffsets needs two rows`);let h=e.lineOffsets.length-1;if(l(e.importance,[`float32`],`${r} importance`),e.importance.length!==m)throw Error(`${r} importance length must equal positions length`);if(i===`time-ratio`&&this.computeImportance){if(!e.timestamps)throw Error(`${r} metric 'time-ratio' requires timestamps`);if(l(e.timestamps,[`float32`],`${r} timestamps`),e.timestamps.length!==m)throw Error(`${r} timestamps length must equal positions length`)}if((p?.converged||p?.iterationCount)&&!this.computeImportance)throw Error(`${r} output.status.converged and iterationCount require computeImportance`);for(let[e,t]of[[`converged`,p?.converged],[`iterationCount`,p?.iterationCount]])if(t&&(l(t,[`uint32`],`${r} output.status.${e}`),t.length<1))throw Error(`${r} output.status.${e} needs one row`);if(f){if(!e.parameters)throw Error(`${r} parameters are required to select rows`);if(l(e.parameters,[`float32`],`${r} parameters`),e.parameters.length<4)throw Error(`${r} parameters must hold 4 float32 values`);if(o(r,f),f.outputMask&&(l(f.outputMask,[`uint32`],`${r} output.outputMask`),f.outputMask.length!==m))throw Error(`${r} output.outputMask length must equal positions length`);for(let[e,t]of[[`lineCounts`,f.lineCounts],[`lineStarts`,f.lineStarts]])if(t&&(l(t,[`uint32`],`${r} output.${e}`),t.length!==h))throw Error(`${r} output.${e} length must equal the line count`)}if(!this.computeImportance&&!f)throw Error(`${r} requires computeImportance or selected output`);let g=[e.positions,e.lineOffsets,e.timestamps,e.parameters];this.computeImportance||g.push(e.importance),s(r,[this.computeImportance?e.importance:void 0,p?.converged,p?.iterationCount,f?.ids,f?.status.count,f?.status.overflow,f?.status.requiredCount,f?.outputMask,f?.lineCounts,f?.lineStarts],g)}getCommandNodes(e){let{props:t,id:n}=this,{selection:r,status:a}=L(t);i(n,e,[t.positions,t.lineOffsets,t.timestamps,t.importance,t.parameters,a?.converged,a?.iterationCount,r?.ids,r?.status.count,r?.status.overflow,r?.status.requiredCount,r?.outputMask,r?.lineCounts,r?.lineStarts]);let o=[];return this.computeImportance&&o.push(...this._getImportanceNodes(e)),r&&t.parameters&&o.push(...this._getSelectionNodes(e,r,t.parameters)),o}_getVisvalingamNodes(e){let{props:t,id:n,maximumIterations:r}=this,i=t.positions.length,a={previousRows:c(e,`${n}-previous-rows`,`uint32`,i),nextRows:c(e,`${n}-next-rows`,`uint32`,i),areaKeys:c(e,`${n}-area-keys`,`uint32`,i),floorKeys:c(e,`${n}-floor-keys`,`uint32`,i),pendingFlags:c(e,`${n}-pending-flags`,`uint32`,i)},o=v(e,`${n}-rounds`,F,i,n),s=[g(e,{id:`${n}-rounds-reset`,operation:F,state:o}),M(e,{id:`${n}-init`,lineOffsets:t.lineOffsets,scratch:a,importance:t.importance})];for(let i=0;i<r;i++){let c=`${n}-round-${i}`;s.push(...N(e,{id:c,neighborhoodRadius:this.neighborhoodRadius,positions:t.positions,scratch:a,importance:t.importance,status:o.status,gate:m(o,c)}),h(e,{id:`${c}-gate`,operation:F,state:o,maxIterations:r}))}s.push(P(e,{id:`${n}-unresolved`,scratch:a,importance:t.importance}));let l=t.output?.status;return(l?.converged||l?.iterationCount)&&s.push(_(e,{id:`${n}-status`,operation:F,state:o,converged:l.converged,iterationCount:l.iterationCount})),s}_getImportanceNodes(e){if(this.method===`visvalingam`)return this._getVisvalingamNodes(e);let{props:t,id:n,maximumIterations:r}=this,i=t.positions.length,a={leftAnchors:c(e,`${n}-left-anchors`,`uint32`,i),rightAnchors:c(e,`${n}-right-anchors`,`uint32`,i),distanceKeys:c(e,`${n}-distance-keys`,`uint32`,i),bestKeys:c(e,`${n}-best-keys`,`uint32`,i),bestRows:c(e,`${n}-best-rows`,`uint32`,i)},o=v(e,`${n}-rounds`,F,i,n),s=[g(e,{id:`${n}-rounds-reset`,operation:F,state:o}),w(e,{id:`${n}-init`,lineOffsets:t.lineOffsets,scratch:a,importance:t.importance})];for(let i=0;i<r;i++){let c=`${n}-round-${i}`;s.push(...T(e,{id:c,metric:this.metric,positions:t.positions,timestamps:t.timestamps,scratch:a,importance:t.importance,status:o.status,gate:m(o,c),finishSpanLimit:this.finishSpanLimit,sphereRadius:this.sphereRadius}),h(e,{id:`${c}-gate`,operation:F,state:o,maxIterations:r}))}s.push(E(e,{id:`${n}-unresolved`,scratch:a,importance:t.importance}));let l=t.output?.status;return(l?.converged||l?.iterationCount)&&s.push(_(e,{id:`${n}-status`,operation:F,state:o,converged:l.converged,iterationCount:l.iterationCount})),s}_getSelectionNodes(e,t,n){let{props:r,id:i}=this,a=r.positions.length,o=t.outputMask??c(e,`${i}-keep-mask`,`uint32`,a),s=c(e,`${i}-row-ids`,`uint32`,a),l=c(e,`${i}-compact-rows`,`uint32`,a),f=c(e,`${i}-kept-total`,`uint32`,1),p=[D(e,{id:`${i}-keep-mask`,lineOffsets:r.lineOffsets,importance:r.importance,parameters:n,outputMask:o,rowIds:s}),...new u({id:`${i}-compaction`,input:s,flags:o,output:l,count:f}).getCommandNodes(e),d(e,{id:`${i}-publish`,operation:F,requiredCount:f,compactIds:l,output:t})];return(t.lineCounts||t.lineStarts)&&p.push(O(e,{id:`${i}-line-ranges`,lineOffsets:r.lineOffsets,compactRows:l,requiredCount:f,lineCounts:t.lineCounts,lineStarts:t.lineStarts})),p}};export{k as n,z as t};