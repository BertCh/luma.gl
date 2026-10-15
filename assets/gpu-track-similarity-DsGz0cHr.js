import{E as e,S as t,p as n}from"./analysis-frame-Blu0tYdw.js";import{_ as r,d as i,h as a,w as o,y as s}from"./resources-CwvhEOBe.js";import{s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as l}from"./geodesic-wgsl-DS0CJZ42.js";var u=4294967295,d=`
fn coupledMetric(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  if (swapped) {
    return pointMetric(densifiedPointB(startB, countB, rowLong), densifiedPointA(startA, countA, rowShort));
  }
  return pointMetric(densifiedPointA(startA, countA, rowLong), densifiedPointB(startB, countB, rowShort));
}`,f=`let infinity = 3.402823466e+38;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  let swapped = denseB > denseA;
  if (lane == 0u) {
    sharedLongCount = max(denseA, denseB);
    sharedShortCount = min(denseA, denseB);
  }
  // Barrier-bearing loops need bounds the compiler can prove uniform across the workgroup.
  let longCount = workgroupUniformLoad(&sharedLongCount);
  let shortCount = workgroupUniformLoad(&sharedShortCount);
  let stripCount = (longCount + LANES - 1u) / LANES;`;function p(e,t,n){return`const WORKGROUP_SIZE: u32 = ${e}u;
const LANES: u32 = ${e}u;
const BAND_RADIUS: u32 = ${n}u;
const NO_BAND: u32 = ${u}u;
var<workgroup> olderDiagonal: array<f32, ${e}>;
var<workgroup> previousDiagonal: array<f32, ${e}>;
var<workgroup> boundaryRow: array<f32, ${t}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${d}
// Sakoe-Chiba band of tslearn's sakoe_chiba_mask(sz1, sz2, radius), written in long/short roles
// (tslearn's two branches are the same rule with the roles swapped): long index l is in the band
// of short index s when s - radius <= l <= s + (longCount - shortCount) + radius.
fn isInBand(rowLong: u32, rowShort: u32, longCount: u32, shortCount: u32) -> bool {
  if (BAND_RADIUS == NO_BAND) {
    return true;
  }
  return rowLong + BAND_RADIUS >= rowShort && rowLong <= rowShort + (longCount - shortCount) + BAND_RADIUS;
}`}function m(e){return`${e}
  ${f}
  var finalValue = infinity;
  for (var strip = 0u; strip < stripCount; strip++) {
    let firstRow = strip * LANES;
    let rows = min(LANES, longCount - firstRow);
    let diagonalCount = rows + shortCount - 1u;
    olderDiagonal[lane] = infinity;
    previousDiagonal[lane] = infinity;
    workgroupBarrier();
    for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
      var value = infinity;
      var column = 0u;
      let isActive = lane < rows && diagonal >= lane && diagonal - lane < shortCount;
      if (isActive) {
        column = diagonal - lane;
        let rowLong = firstRow + lane;
        if (isInBand(rowLong, column, longCount, shortCount)) {
          let cost = finishMetric(coupledMetric(swapped, startA, countA, startB, countB, rowLong, column));
          var up = infinity;
          var upLeft = infinity;
          var left = infinity;
          if (lane > 0u) {
            up = previousDiagonal[lane - 1u];
            upLeft = olderDiagonal[lane - 1u];
          } else if (strip > 0u) {
            up = boundaryRow[column];
            if (column > 0u) { upLeft = boundaryRow[column - 1u]; }
          }
          if (column > 0u) { left = previousDiagonal[lane]; }
          let best = min(min(up, upLeft), left);
          if (rowLong == 0u && column == 0u) {
            value = cost;
          } else if (best < infinity) {
            value = min(best + cost, infinity);
          }
        }
        if (rowLong == longCount - 1u && column == shortCount - 1u) { finalValue = value; }
      }
      workgroupBarrier();
      olderDiagonal[lane] = previousDiagonal[lane];
      previousDiagonal[lane] = value;
      if (isActive && lane == LANES - 1u) { boundaryRow[column] = value; }
      workgroupBarrier();
    }
  }
  if (!skipped && longCount > 0u && lane == (longCount - 1u) % LANES) {
    distances[distancesOffset + pair] = finalValue;
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`}function h(e,t,n){return`const WORKGROUP_SIZE: u32 = ${e}u;
const LANES: u32 = ${e}u;
var<workgroup> olderDiagonal: array<f32, ${e}>;
var<workgroup> previousDiagonal: array<f32, ${e}>;
var<workgroup> boundaryRow: array<f32, ${t}>;
const PLANAR_BOUNDS: bool = ${!n};
var<workgroup> partials: array<vec4f, ${e}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
var<workgroup> sharedDecision: u32;
var<workgroup> sharedReachable: u32;
var<workgroup> reachableCounter: atomic<u32>;
var<workgroup> boundaryReachableCounter: atomic<u32>;
${d}
// Epsilon in the internal metric space of pointMetric (squared planar distance, or the context's length unit).
fn getEpsilonMetric(epsilon: f32) -> f32 {
  ${n?`return epsilon;`:`return epsilon * epsilon;`}
}
// Short index coupled with long index rowLong by a fixed monotone coupling: proportional
// (floor(l * (S - 1) / (L - 1))) when that product fits 32 bits, otherwise the diagonal min(l, S - 1).
// Each step advances the short index by at most 1 (S <= L), so it is a valid discrete coupling and
// its largest coupled distance is an upper bound of the Frechet distance.
fn getCoupledShortRow(rowLong: u32, longCount: u32, shortCount: u32) -> u32 {
  if (longCount <= 1u || shortCount <= 1u) {
    return 0u;
  }
  if (longCount - 1u <= 0xffffffffu / (shortCount - 1u)) {
    return rowLong * (shortCount - 1u) / (longCount - 1u);
  }
  return min(rowLong, shortCount - 1u);
}`}function g(e){return`${e}
  ${f}
  let epsilon = frechetEpsilon[frechetEpsilonOffset];
  let isEpsilonUsable = (bitcast<u32>(epsilon) & 0x7f800000u) != 0x7f800000u && epsilon >= 0.0;
  let limit = getEpsilonMetric(epsilon);
  // Reject endpoints before reading the rest of either track or reducing a coupling bound.
  if (lane == 0u) {
    sharedDecision = 2u;
    if (!skipped && isEpsilonUsable && longCount > 0u) {
      let lower = max(
        coupledMetric(swapped, startA, countA, startB, countB, 0u, 0u),
        coupledMetric(swapped, startA, countA, startB, countB, longCount - 1u, shortCount - 1u)
      );
      if (!(lower > limit)) { sharedDecision = 0u; }
    }
  }
  var decision = workgroupUniformLoad(&sharedDecision);
  if (decision == 2u) {
    if (lane == 0u && pair < PAIR_COUNT) { frechetWithin[frechetWithinOffset + pair] = 0u; }
    return;
  }
  // A fixed coupling accepts in linear work. Do not pay for box reductions on accepted pairs.
  var coupled = 0.0;
  for (var rowLong = lane; rowLong < longCount; rowLong += LANES) {
    let rowShort = getCoupledShortRow(rowLong, longCount, shortCount);
    coupled = max(coupled, coupledMetric(swapped, startA, countA, startB, countB, rowLong, rowShort));
  }
  partials[lane] = vec4f(coupled);
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  if (lane == 0u && partials[0].x <= limit) { sharedDecision = 1u; }
  decision = workgroupUniformLoad(&sharedDecision);
  if (decision == 0u && PLANAR_BOUNDS) {
    // Every vertex must be matched. Differences between corresponding extrema of the two
    // boxes lower-bound the maximum distance in any coupling, even when the boxes overlap.
    // Use densified samples too: these are exactly the coordinates used by the free-space DP.
    var minima = vec4f(infinity);
    var maxima = vec4f(-infinity);
    for (var vertex = lane; vertex < denseA; vertex += LANES) {
      let point = densifiedPointA(startA, countA, vertex);
      minima = vec4f(min(minima.xy, point), minima.zw);
      maxima = vec4f(max(maxima.xy, point), maxima.zw);
    }
    for (var vertex = lane; vertex < denseB; vertex += LANES) {
      let point = densifiedPointB(startB, countB, vertex);
      minima = vec4f(minima.xy, min(minima.zw, point));
      maxima = vec4f(maxima.xy, max(maxima.zw, point));
    }
    partials[lane] = minima;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) { partials[lane] = min(partials[lane], partials[lane + stride]); }
      workgroupBarrier();
    }
    let minimumGap = abs(partials[0].xy - partials[0].zw);
    workgroupBarrier();
    partials[lane] = maxima;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let maximumGap = abs(partials[0].xy - partials[0].zw);
      let gaps = max(minimumGap, maximumGap);
      let lower = max(gaps.x, gaps.y);
      // Widen the threshold to keep f32 rounding at equality in the exact DP path.
      if (lower * lower > limit * 1.000001) { sharedDecision = 2u; }
    }
    decision = workgroupUniformLoad(&sharedDecision);
  }
  // 2. Free-space reachability, 1.0 reachable and 0.0 not.
  if (decision == 0u) {
    var finalValue = 0.0;
    var previousBoundaryCount = 0u;
    for (var strip = 0u; strip < stripCount; strip++) {
      let firstRow = strip * LANES;
      let rows = min(LANES, longCount - firstRow);
      let diagonalCount = rows + shortCount - 1u;
      olderDiagonal[lane] = 0.0;
      previousDiagonal[lane] = 0.0;
      if (lane == 0u) {
        atomicStore(&reachableCounter, 0u);
        atomicStore(&boundaryReachableCounter, 0u);
      }
      workgroupBarrier();
      var lastWrittenColumn = 0u;
      var hasWritten = false;
      // Consecutive anti-diagonals without a reachable cell (meaningful in lane 0 only).
      var deadDiagonals = 0u;
      for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
        var value = 0.0;
        var column = 0u;
        let isActive = lane < rows && diagonal >= lane && diagonal - lane < shortCount;
        if (isActive) {
          column = diagonal - lane;
          let rowLong = firstRow + lane;
          {
            var predecessor = rowLong == 0u && column == 0u;
            if (lane > 0u) {
              predecessor = predecessor || previousDiagonal[lane - 1u] > 0.0 || olderDiagonal[lane - 1u] > 0.0;
            } else if (strip > 0u) {
              predecessor = predecessor || boundaryRow[column] > 0.0;
              if (column > 0u) { predecessor = predecessor || boundaryRow[column - 1u] > 0.0; }
            }
            if (column > 0u) { predecessor = predecessor || previousDiagonal[lane] > 0.0; }
            if (predecessor && coupledMetric(swapped, startA, countA, startB, countB, rowLong, column) <= limit) {
              value = 1.0;
              atomicAdd(&reachableCounter, 1u);
            }
          }
          if (rowLong == longCount - 1u && column == shortCount - 1u) { finalValue = value; }
        }
        workgroupBarrier();
        olderDiagonal[lane] = previousDiagonal[lane];
        previousDiagonal[lane] = value;
        if (isActive && lane == LANES - 1u) {
          boundaryRow[column] = value;
          if (value > 0.0) { atomicAdd(&boundaryReachableCounter, 1u); }
        }
        if (lane == 0u) {
          // A cell's predecessors lie on the two previous anti-diagonals (up and left on the
          // last, up-left on the one before) or in the carried row (lane 0, from column
          // diagonal + 1 on). Two dead anti-diagonals and no reachable carried cell end the strip.
          deadDiagonals = select(0u, deadDiagonals + 1u, atomicLoad(&reachableCounter) == 0u);
          let isDead = deadDiagonals >= 2u &&
            (strip == 0u || previousBoundaryCount == 0u || diagonal + 1u >= shortCount);
          sharedReachable = select(1u, 0u, isDead);
          atomicStore(&reachableCounter, 0u);
        }
        lastWrittenColumn = select(lastWrittenColumn, diagonal - (LANES - 1u), diagonal >= LANES - 1u);
        hasWritten = hasWritten || diagonal >= LANES - 1u;
        let isAlive = workgroupUniformLoad(&sharedReachable);
        if (isAlive == 0u) { break; }
      }
      workgroupBarrier();
      // Columns of the carried row this strip never reached are unreachable.
      if (rows == LANES) {
        let firstUnwritten = select(0u, lastWrittenColumn + 1u, hasWritten);
        for (var column = firstUnwritten + lane; column < shortCount; column += LANES) {
          boundaryRow[column] = 0.0;
        }
      }
      if (lane == 0u) {
        sharedReachable = atomicLoad(&boundaryReachableCounter);
      }
      previousBoundaryCount = workgroupUniformLoad(&sharedReachable);
      // A full strip whose last row has no reachable cell cuts every monotone path.
      if (rows == LANES && strip + 1u < stripCount && previousBoundaryCount == 0u) { break; }
    }
    if (lane == (longCount - 1u) % LANES && finalValue > 0.0) {
      sharedDecision = 1u;
    }
    decision = workgroupUniformLoad(&sharedDecision);
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    frechetWithin[frechetWithinOffset + pair] = select(0u, 1u, decision == 1u);
  }`}var _=Object.freeze({...n,measure:`distance`}),v=`GPUTrackSimilarity`,y=64,b=256,x=Math.floor(4294967295/b),S=2048,C=4096,w=2**30;function T(e){return`// Metric value (squared distance, or the context's length unit on the sphere) from a point to polyline ${e}; may stop early once it is at most worst.
fn distanceToPolyline${e}(point: vec2<f32>, start: u32, count: u32, worst: f32, hint: ptr<function, u32>) -> f32 {
  if (count == 1u) {
    return pointMetric(point, load${e}(start));
  }
  let segments = count - 1u;
  var nearest = 3.402823466e+38;
  var first = *hint;
  if (first >= segments) { first = 0u; }
  for (var step = 0u; step < segments; step++) {
    var j = first + step;
    if (j >= segments) { j -= segments; }
    let squared = pointSegmentMetric(point, load${e}(start + j), load${e}(start + j + 1u));
    if (squared < nearest) {
      nearest = squared;
      *hint = j;
      if (nearest <= worst) { break; }
    }
  }
  return nearest;
}
// Cooperative minimum over the segments of polyline ${e}: every lane takes a stride of segments.
fn cooperativeDistanceToPolyline${e}(point: vec2<f32>, start: u32, count: u32, lane: u32) -> f32 {
  var nearest = 3.402823466e+38;
  if (count == 1u) {
    return pointMetric(point, load${e}(start));
  }
  for (var j = lane; j + 1u < count; j += WORKGROUP_SIZE) {
    nearest = min(nearest, pointSegmentMetric(point, load${e}(start + j), load${e}(start + j + 1u)));
  }
  return nearest;
}`}function E(e,t){return`{
    let pointCount = uniformDense${e};
    if (pointCount * 8u <= WORKGROUP_SIZE) {
      for (var i = 0u; i < pointCount; i++) {
        let point = densifiedPoint${e}(start${e}, count${e}, i);
        partials[lane] = cooperativeDistanceToPolyline${t}(point, start${t}, count${t}, lane);
        workgroupBarrier();
        for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
          if (lane < stride) { partials[lane] = min(partials[lane], partials[lane + stride]); }
          workgroupBarrier();
        }
        worst = max(worst, partials[0]);
        workgroupBarrier();
      }
    } else {
      let chunk = (pointCount + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
      let chunkStart = min(lane * chunk, pointCount);
      let chunkEnd = min(chunkStart + chunk, pointCount);
      var hint = 0u;
      for (var i = chunkStart; i < chunkEnd; i++) {
        let nearest = distanceToPolyline${t}(densifiedPoint${e}(start${e}, count${e}, i), start${t}, count${t}, worst, &hint);
        worst = max(worst, nearest);
      }
    }
  }`}var D={emptyTrack:1,frechetCapExceeded:2,invalidPair:4},O=class{id;props;spatialContext;sphereRadius;maximumFrechetVertexCount;densifySubdivisions;constructor(n){this.id=n.id??`track-similarity`,this.props=n;let{id:r}=this;this.spatialContext=e(r,n.spatialContext,_),this.sphereRadius=t(this.spatialContext);let{linesA:o,linesB:c,output:l}=n;if(s(o.positions,[`float32x2`],`${r} linesA.positions`),s(o.lineOffsets,[`uint32`],`${r} linesA.lineOffsets`),o.lineOffsets.length<2)throw Error(`${r} linesA.lineOffsets needs two rows`);if(c&&(s(c.positions,[`float32x2`],`${r} linesB.positions`),s(c.lineOffsets,[`uint32`],`${r} linesB.lineOffsets`),c.lineOffsets.length<2))throw Error(`${r} linesB.lineOffsets needs two rows`);if(s(n.pairA,[`uint32`],`${r} pairA`),s(n.pairB,[`uint32`],`${r} pairB`),n.pairA.length<1||n.pairB.length!==n.pairA.length)throw Error(`${r} pairA and pairB must have the same positive length`);if(i(r,`pairA.length`,n.pairA.length,1,x),n.activePairCount&&(s(n.activePairCount,[`uint32`],`${r} activePairCount`),n.activePairCount.length<1))throw Error(`${r} activePairCount must contain one uint32 row`);if(!l.hausdorff&&!l.frechet&&!l.maximumDistance&&!l.frechetWithin&&!l.dynamicTimeWarping)throw Error(`${r} needs hausdorff, frechet, frechetWithin, dynamicTimeWarping or maximumDistance output`);if(!!l.frechetWithin!=!!n.frechetEpsilon)throw Error(`${r} output.frechetWithin and frechetEpsilon go together`);if(n.frechetEpsilon&&(s(n.frechetEpsilon,[`float32`],`${r} frechetEpsilon`),n.frechetEpsilon.length<1))throw Error(`${r} frechetEpsilon must contain one float32 row`);n.dynamicTimeWarpingBandRadius!==void 0&&i(r,`dynamicTimeWarpingBandRadius`,n.dynamicTimeWarpingBandRadius,0,w);for(let[e,t,i]of[[`hausdorff`,l.hausdorff,`float32`],[`frechet`,l.frechet,`float32`],[`frechetWithin`,l.frechetWithin,`uint32`],[`dynamicTimeWarping`,l.dynamicTimeWarping,`float32`],[`maximumDistance`,l.maximumDistance,`float32`],[`pairStatus`,l.pairStatus,`uint32`]])if(t&&(s(t,[i],`${r} ${e}`),t.length!==n.pairA.length))throw Error(`${r} ${e} length must equal the pair capacity`);let u=n.densify??0;if(!(u>=0&&u<=1))throw Error(`${r} densify must be a fraction from 0 to 1`);if(this.densifySubdivisions=u===0?1:Math.round(1/u),this.densifySubdivisions>C)throw Error(`${r} densify must be at least 1 / ${C}`);this.maximumFrechetVertexCount=n.maximumFrechetVertexCount??256,i(r,`maximumFrechetVertexCount`,this.maximumFrechetVertexCount,1,S),a(r,[l.hausdorff,l.frechet,l.frechetWithin,l.dynamicTimeWarping,l.maximumDistance,l.pairStatus],[o.positions,o.lineOffsets,c?.positions,c?.lineOffsets,n.pairA,n.pairB,n.activePairCount,n.frechetEpsilon])}getCommandNodes(e){let{id:t,props:n}=this,{linesA:i,linesB:a,output:s}=n;r(t,e,[i.positions,i.lineOffsets,a?.positions,a?.lineOffsets,n.pairA,n.pairB,n.activePairCount,n.frechetEpsilon,s.hausdorff,s.frechet,s.frechetWithin,s.dynamicTimeWarping,s.maximumDistance,s.pairStatus]);let u=a?.positions??i.positions,d=a?.lineOffsets??i.lineOffsets,f=n.pairA.length,_=[],x=`const PAIR_COUNT: u32 = ${f}u;
const TRACK_COUNT_A: u32 = ${i.lineOffsets.length-1}u;
const TRACK_COUNT_B: u32 = ${d.length-1}u;
const FRECHET_CAP: u32 = ${this.maximumFrechetVertexCount}u;
const DENSIFY: u32 = ${this.densifySubdivisions}u;
const STATUS_EMPTY: u32 = ${D.emptyTrack}u;
const STATUS_TOO_LONG: u32 = ${D.frechetCapExceeded}u;
const STATUS_INVALID: u32 = ${D.invalidPair}u;`,S=s.pairStatus??o(e,`${t}-status`,`uint32`,f),C=[{name:`pairA`,view:n.pairA,type:`u32`,access:`read`},{name:`pairB`,view:n.pairB,type:`u32`,access:`read`},{name:`offsetsA`,view:i.lineOffsets,type:`u32`,access:`read`},{name:`offsetsB`,view:d,type:`u32`,access:`read`}];n.activePairCount&&C.push({name:`activePairCount`,view:n.activePairCount,type:`u32`,access:`read`}),C.push({name:`status`,view:S,type:`u32`,access:`read_write`}),_.push(c(e,{id:`${t}-status`,operation:v,variant:`status`,bindings:C,invocationCount:f,declarations:x,body:`let a = pairA[pairAOffset + index];
  let b = pairB[pairBOffset + index];
  var flags = 0u;
  ${n.activePairCount?`if (index >= activePairCount[activePairCountOffset]) { flags = STATUS_INVALID; }`:``}
  if (a >= TRACK_COUNT_A || b >= TRACK_COUNT_B) {
    flags = STATUS_INVALID;
  } else if (flags == 0u) {
    let countA = offsetsA[offsetsAOffset + a + 1u] - offsetsA[offsetsAOffset + a];
    let countB = offsetsB[offsetsBOffset + b + 1u] - offsetsB[offsetsBOffset + b];
    if (countA == 0u || countB == 0u) { flags = STATUS_EMPTY; }
    else if (min((countA - 1u) * DENSIFY, (countB - 1u) * DENSIFY) + 1u > FRECHET_CAP) { flags = STATUS_TOO_LONG; }
  }
  status[statusOffset + index] = flags;`}));let w=e=>[{name:`pairA`,view:n.pairA,type:`u32`,access:`read`},{name:`pairB`,view:n.pairB,type:`u32`,access:`read`},{name:`positionsA`,view:i.positions,type:`f32`,access:`read`},{name:`offsetsA`,view:i.lineOffsets,type:`u32`,access:`read`},{name:`positionsB`,view:u,type:`f32`,access:`read`},{name:`offsetsB`,view:d,type:`u32`,access:`read`},{name:`status`,view:S,type:`u32`,access:`read`},{name:`distances`,view:e,type:`f32`,access:`read_write`}],O=e=>`let pair = index / WORKGROUP_SIZE;
  let lane = localInvocationIndex;
  let flags = status[statusOffset + min(pair, PAIR_COUNT - 1u)];
  let skipped = pair >= PAIR_COUNT || (flags & ${e}) != 0u;
  var startA = 0u;
  var countA = 0u;
  var startB = 0u;
  var countB = 0u;
  if (!skipped) {
    let a = pairA[pairAOffset + pair];
    let b = pairB[pairBOffset + pair];
    startA = offsetsA[offsetsAOffset + a];
    countA = offsetsA[offsetsAOffset + a + 1u] - startA;
    startB = offsetsB[offsetsBOffset + b];
    countB = offsetsB[offsetsBOffset + b + 1u] - startB;
  }
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let notANumber = bitcast<f32>(0x7fc00000u | (index & 0u));`,k=`fn loadA(row: u32) -> vec2<f32> {
  return vec2<f32>(positionsA[positionsAOffset + row * 2u], positionsA[positionsAOffset + row * 2u + 1u]);
}
fn loadB(row: u32) -> vec2<f32> {
  return vec2<f32>(positionsB[positionsBOffset + row * 2u], positionsB[positionsBOffset + row * 2u + 1u]);
}
fn densifiedCount(count: u32) -> u32 {
  return select(0u, (count - 1u) * DENSIFY + 1u, count > 0u);
}
fn densifiedPointA(start: u32, count: u32, k: u32) -> vec2<f32> {
  let segment = k / DENSIFY;
  let subStep = k - segment * DENSIFY;
  if (subStep == 0u || segment + 1u >= count) { return loadA(start + min(segment, count - 1u)); }
  return interpolatePoint(loadA(start + segment), loadA(start + segment + 1u), f32(subStep) / f32(DENSIFY));
}
fn densifiedPointB(start: u32, count: u32, k: u32) -> vec2<f32> {
  let segment = k / DENSIFY;
  let subStep = k - segment * DENSIFY;
  if (subStep == 0u || segment + 1u >= count) { return loadB(start + min(segment, count - 1u)); }
  return interpolatePoint(loadB(start + segment), loadB(start + segment + 1u), f32(subStep) / f32(DENSIFY));
}
${this.sphereRadius===void 0?`fn pointMetric(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let offset = a - b;
  return dot(offset, offset);
}
fn finishMetric(value: f32) -> f32 {
  return sqrt(value);
}
fn interpolatePoint(from0: vec2<f32>, to0: vec2<f32>, fraction: f32) -> vec2<f32> {
  return from0 + (to0 - from0) * fraction;
}
fn pointSegmentMetric(point: vec2<f32>, from0: vec2<f32>, to0: vec2<f32>) -> f32 {
  let edge = to0 - from0;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) { t = clamp(dot(point - from0, edge) / lengthSquared, 0.0, 1.0); }
  let offset = point - (from0 + edge * t);
  return dot(offset, offset);
}`:`${l}
const SPHERE_RADIUS: f32 = ${this.sphereRadius};
fn pointMetric(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return geodesicHaversineDistance(a, b, SPHERE_RADIUS);
}
fn finishMetric(value: f32) -> f32 {
  return value;
}
// Linear in longitude/latitude with the longitude difference wrapped across the antimeridian.
fn interpolatePoint(from0: vec2<f32>, to0: vec2<f32>, fraction: f32) -> vec2<f32> {
  return from0 + vec2<f32>(geodesicWrapLongitudeDelta(to0.x - from0.x), to0.y - from0.y) * fraction;
}
fn pointSegmentMetric(point: vec2<f32>, from0: vec2<f32>, to0: vec2<f32>) -> f32 {
  let t = geodesicSegmentFraction(point, from0, to0);
  return geodesicSegmentDistance(point, from0, to0, t, SPHERE_RADIUS);
}`}`;if(s.hausdorff&&_.push(c(e,{id:`${t}-hausdorff`,operation:v,variant:`hausdorff`,bindings:w(s.hausdorff),workgroupSize:y,invocationCount:f*y,guardIndex:!1,declarations:`${x}
const WORKGROUP_SIZE: u32 = ${y}u;
var<workgroup> partials: array<f32, ${y}>;
var<workgroup> sharedDenseA: u32;
var<workgroup> sharedDenseB: u32;
${k}
${T(`A`)}
${T(`B`)}`,body:`${O(`(STATUS_EMPTY | STATUS_INVALID)`)}
  var worst = 0.0;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  if (lane == 0u) {
    sharedDenseA = denseA;
    sharedDenseB = denseB;
  }
  workgroupBarrier();
  let uniformDenseA = workgroupUniformLoad(&sharedDenseA);
  let uniformDenseB = workgroupUniformLoad(&sharedDenseB);
  ${E(`A`,`B`)}
  ${E(`B`,`A`)}
  partials[lane] = worst;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = select(finishMetric(partials[0]), notANumber, skipped);
  }`})),s.maximumDistance&&_.push(c(e,{id:`${t}-max-distance`,operation:v,variant:`max-distance`,bindings:w(s.maximumDistance),workgroupSize:y,invocationCount:f*y,guardIndex:!1,declarations:`${x}
const WORKGROUP_SIZE: u32 = ${y}u;
var<workgroup> partials: array<f32, ${y}>;
${k}`,body:`${O(`(STATUS_EMPTY | STATUS_INVALID)`)}
  var farthest = 0.0;
  // Lanes stride over the longer track and scan the shorter one, so a short track next to a long
  // one does not leave most lanes idle. The maximum is order independent.
  if (countA >= countB) {
    for (var i = lane; i < countA; i += WORKGROUP_SIZE) {
      let point = loadA(startA + i);
      for (var j = 0u; j < countB; j++) {
        farthest = max(farthest, pointMetric(point, loadB(startB + j)));
      }
    }
  } else {
    for (var j = lane; j < countB; j += WORKGROUP_SIZE) {
      let point = loadB(startB + j);
      for (var i = 0u; i < countA; i++) {
        farthest = max(farthest, pointMetric(point, loadA(startA + i)));
      }
    }
  }
  partials[lane] = farthest;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = select(finishMetric(partials[0]), notANumber, skipped);
  }`})),s.frechet&&_.push(c(e,{id:`${t}-frechet`,operation:v,variant:`frechet`,bindings:w(s.frechet),workgroupSize:b,invocationCount:f*b,guardIndex:!1,declarations:`${x}
const WORKGROUP_SIZE: u32 = ${b}u;
const LANES: u32 = ${b}u;
var<workgroup> olderDiagonal: array<f32, ${b}>;
var<workgroup> previousDiagonal: array<f32, ${b}>;
var<workgroup> boundaryRow: array<f32, ${this.maximumFrechetVertexCount}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${k}
// Metric value between densified vertex rowLong of the longer track and rowShort of the other.
fn coupledSquared(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  if (swapped) {
    return pointMetric(densifiedPointB(startB, countB, rowLong), densifiedPointA(startA, countA, rowShort));
  }
  return pointMetric(densifiedPointA(startA, countA, rowLong), densifiedPointB(startB, countB, rowShort));
}`,body:`${O(`(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)`)}
  let infinity = 3.402823466e+38;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  let swapped = denseB > denseA;
  if (lane == 0u) {
    sharedLongCount = max(denseA, denseB);
    sharedShortCount = min(denseA, denseB);
  }
  // Barrier-bearing loops need bounds the compiler can prove uniform across the workgroup.
  let longCount = workgroupUniformLoad(&sharedLongCount);
  let shortCount = workgroupUniformLoad(&sharedShortCount);
  var finalValue = infinity;
  let stripCount = (longCount + LANES - 1u) / LANES;
  for (var strip = 0u; strip < stripCount; strip++) {
    let firstRow = strip * LANES;
    let rows = min(LANES, longCount - firstRow);
    let diagonalCount = rows + shortCount - 1u;
    olderDiagonal[lane] = infinity;
    previousDiagonal[lane] = infinity;
    workgroupBarrier();
    for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
      var value = infinity;
      var column = 0u;
      let isActive = lane < rows && diagonal >= lane && diagonal - lane < shortCount;
      if (isActive) {
        column = diagonal - lane;
        let cost = coupledSquared(swapped, startA, countA, startB, countB, firstRow + lane, column);
        let hasUp = lane > 0u || strip > 0u;
        let hasLeft = column > 0u;
        var up = infinity;
        var upLeft = infinity;
        var left = infinity;
        if (lane > 0u) {
          up = previousDiagonal[lane - 1u];
          upLeft = olderDiagonal[lane - 1u];
        } else if (strip > 0u) {
          up = boundaryRow[column];
          if (hasLeft) { upLeft = boundaryRow[column - 1u]; }
        }
        if (hasLeft) { left = previousDiagonal[lane]; }
        if (!hasUp && !hasLeft) {
          value = cost;
        } else if (!hasUp) {
          value = max(left, cost);
        } else if (!hasLeft) {
          value = max(up, cost);
        } else {
          value = max(min(min(up, upLeft), left), cost);
        }
        if (firstRow + lane == longCount - 1u && column == shortCount - 1u) { finalValue = value; }
      }
      workgroupBarrier();
      olderDiagonal[lane] = previousDiagonal[lane];
      previousDiagonal[lane] = value;
      if (isActive && lane == LANES - 1u) { boundaryRow[column] = value; }
      workgroupBarrier();
    }
  }
  if (!skipped && longCount > 0u && lane == (longCount - 1u) % LANES) {
    distances[distancesOffset + pair] = finishMetric(finalValue);
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`})),s.dynamicTimeWarping&&_.push(c(e,{id:`${t}-dynamic-time-warping`,operation:v,variant:`dynamic-time-warping`,bindings:w(s.dynamicTimeWarping),workgroupSize:b,invocationCount:f*b,guardIndex:!1,declarations:`${x}
${p(b,this.maximumFrechetVertexCount,n.dynamicTimeWarpingBandRadius??4294967295)}
${k}`,body:m(O(`(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)`))})),s.frechetWithin&&n.frechetEpsilon){let r=o(e,`${t}-pair-ranges`,`uint32`,f*4);_.push(c(e,{id:`${t}-pair-ranges`,operation:v,variant:`pair-ranges`,bindings:[{name:`pairA`,view:n.pairA,type:`u32`,access:`read`},{name:`pairB`,view:n.pairB,type:`u32`,access:`read`},{name:`offsetsA`,view:i.lineOffsets,type:`u32`,access:`read`},{name:`offsetsB`,view:d,type:`u32`,access:`read`},{name:`status`,view:S,type:`u32`,access:`read`},{name:`pairRanges`,view:r,type:`u32`,access:`read_write`}],invocationCount:f,declarations:x,body:`var range = vec4<u32>(0u);
  if (status[statusOffset + index] == 0u) {
    let a = pairA[pairAOffset + index];
    let b = pairB[pairBOffset + index];
    let startA = offsetsA[offsetsAOffset + a];
    let startB = offsetsB[offsetsBOffset + b];
    range = vec4<u32>(startA, offsetsA[offsetsAOffset + a + 1u] - startA,
      startB, offsetsB[offsetsBOffset + b + 1u] - startB);
  }
  pairRanges[pairRangesOffset + index * 4u] = range.x;
  pairRanges[pairRangesOffset + index * 4u + 1u] = range.y;
  pairRanges[pairRangesOffset + index * 4u + 2u] = range.z;
  pairRanges[pairRangesOffset + index * 4u + 3u] = range.w;`}),c(e,{id:`${t}-frechet-decision`,operation:v,variant:`frechet-decision`,bindings:[{name:`pairRanges`,view:r,type:`u32`,access:`read`},{name:`positionsA`,view:i.positions,type:`f32`,access:`read`},{name:`positionsB`,view:u,type:`f32`,access:`read`},{name:`status`,view:S,type:`u32`,access:`read`},{name:`frechetEpsilon`,view:n.frechetEpsilon,type:`f32`,access:`read`},{name:`frechetWithin`,view:s.frechetWithin,type:`u32`,access:`read_write`}],workgroupSize:b,invocationCount:f*b,guardIndex:!1,declarations:`${x}
${h(b,this.maximumFrechetVertexCount,this.sphereRadius!==void 0)}
${k}`,body:g(`let pair = index / WORKGROUP_SIZE;
  let lane = localInvocationIndex;
  let flags = status[statusOffset + min(pair, PAIR_COUNT - 1u)];
  let skipped = pair >= PAIR_COUNT ||
    (flags & (STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)) != 0u;
  let rangeBase = pairRangesOffset + min(pair, PAIR_COUNT - 1u) * 4u;
  let startA = pairRanges[rangeBase];
  let countA = select(pairRanges[rangeBase + 1u], 0u, skipped);
  let startB = pairRanges[rangeBase + 2u];
  let countB = select(pairRanges[rangeBase + 3u], 0u, skipped);`)}))}return _}};export{O as t};