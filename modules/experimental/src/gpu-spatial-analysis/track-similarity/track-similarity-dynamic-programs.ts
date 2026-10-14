// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL of the anti-diagonal dynamic programs of `GPUTrackSimilarity` beyond the Frechet distance:
 * the Frechet decision (free-space reachability against `epsilon`) and banded dynamic time
 * warping. Both use the layout of the Frechet distance kernel: one 256-lane workgroup per pair, the
 * longer (densified) track along the lanes in strips of 256 rows, the shorter track (capped) along
 * the anti-diagonals, and the last row of a strip carried to the next strip in workgroup memory.
 *
 * Both expect the shared track helpers of the contributor (`densifiedCount`, `densifiedPointA`,
 * `densifiedPointB`, `pointMetric`, `finishMetric`) and its status constants in scope.
 *
 * @internal
 */

/** Value of `BAND_RADIUS` for an unbanded dynamic time warping. @internal */
export const TRACK_SIMILARITY_NO_BAND = 0xffffffff;

/** WGSL: metric value between densified vertex `rowLong` of the longer track and `rowShort` of the other. */
const COUPLED_METRIC_WGSL = /* wgsl */ `
fn coupledMetric(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  if (swapped) {
    return pointMetric(densifiedPointB(startB, countB, rowLong), densifiedPointA(startA, countA, rowShort));
  }
  return pointMetric(densifiedPointA(startA, countA, rowLong), densifiedPointB(startB, countB, rowShort));
}`;

/** WGSL statements shared by both programs: track lengths and roles, uniform across the workgroup. */
const TRACK_ROLES_WGSL = /* wgsl */ `let infinity = 3.402823466e+38;
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
  let stripCount = (longCount + LANES - 1u) / LANES;`;

/**
 * Declarations of the banded dynamic time warping kernel. `bandRadius` is the Sakoe-Chiba radius
 * in densified vertices, or {@link TRACK_SIMILARITY_NO_BAND}. @internal
 */
export function getDynamicTimeWarpingDeclarations(
  lanes: number,
  shortCapacity: number,
  bandRadius: number
): string {
  return `const WORKGROUP_SIZE: u32 = ${lanes}u;
const LANES: u32 = ${lanes}u;
const BAND_RADIUS: u32 = ${bandRadius}u;
const NO_BAND: u32 = ${TRACK_SIMILARITY_NO_BAND}u;
var<workgroup> olderDiagonal: array<f32, ${lanes}>;
var<workgroup> previousDiagonal: array<f32, ${lanes}>;
var<workgroup> boundaryRow: array<f32, ${shortCapacity}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${COUPLED_METRIC_WGSL}
// Sakoe-Chiba band of tslearn's sakoe_chiba_mask(sz1, sz2, radius), written in long/short roles
// (tslearn's two branches are the same rule with the roles swapped): long index l is in the band
// of short index s when s - radius <= l <= s + (longCount - shortCount) + radius.
fn isInBand(rowLong: u32, rowShort: u32, longCount: u32, shortCount: u32) -> bool {
  if (BAND_RADIUS == NO_BAND) {
    return true;
  }
  return rowLong + BAND_RADIUS >= rowShort && rowLong <= rowShort + (longCount - shortCount) + BAND_RADIUS;
}`;
}

/**
 * Body of the banded dynamic time warping kernel: the cost of the cheapest monotone warping path,
 * the sum of its point distances (`finishMetric` of each coupled metric), with `cost + min(up,
 * up-left, left)` per cell. Cells outside the band are never evaluated and act as +infinity. The
 * caller's prologue defines `pair`, `lane`, `skipped`, the track ranges and `notANumber`. @internal
 */
export function getDynamicTimeWarpingBody(prologue: string): string {
  return `${prologue}
  ${TRACK_ROLES_WGSL}
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
  }`;
}

/** Declarations of the Frechet decision kernel. @internal */
export function getFrechetDecisionDeclarations(
  lanes: number,
  shortCapacity: number,
  isGreatCircle: boolean
): string {
  return `const WORKGROUP_SIZE: u32 = ${lanes}u;
const LANES: u32 = ${lanes}u;
var<workgroup> olderDiagonal: array<f32, ${lanes}>;
var<workgroup> previousDiagonal: array<f32, ${lanes}>;
var<workgroup> boundaryRow: array<f32, ${shortCapacity}>;
const PLANAR_BOUNDS: bool = ${!isGreatCircle};
var<workgroup> partials: array<vec4f, ${lanes}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
var<workgroup> sharedDecision: u32;
var<workgroup> sharedReachable: u32;
var<workgroup> reachableCounter: atomic<u32>;
var<workgroup> boundaryReachableCounter: atomic<u32>;
${COUPLED_METRIC_WGSL}
// Epsilon in the internal metric space of pointMetric (squared planar distance, or the context's length unit).
fn getEpsilonMetric(epsilon: f32) -> f32 {
  ${isGreatCircle ? 'return epsilon;' : 'return epsilon * epsilon;'}
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
}`;
}

/**
 * Body of the Frechet decision kernel: writes 1 when the discrete Frechet distance is at most
 * `epsilon`, else 0 (also 0 for skipped pairs and an unusable epsilon).
 *
 * 1. Bound pruning, `O(L + S)`: the Frechet distance is at least the larger endpoint distance
 *    (lower bound, above epsilon decides 0) and at most the largest coupled distance of a fixed
 *    monotone coupling (upper bound, at most epsilon decides 1). Endpoints are checked first;
 *    undecided planar pairs also use coordinate-extrema box lower bounds before the DP.
 * 2. Otherwise the free-space dynamic program: a cell is reachable when its coupled distance is
 *    at most epsilon and a predecessor (up, up-left, left) is reachable. A strip stops early once
 *    an anti-diagonal holds no reachable cell and no later column of the carried row is reachable,
 *    and the pair stops as 0 once a strip's last row holds no reachable cell (no monotone path can
 *    cross it). The early exits are workgroup-uniform (`workgroupUniformLoad`), so barriers stay in
 *    uniform control flow.
 *
 * The caller's prologue defines `pair`, `lane`, `skipped` and the track ranges. @internal
 */
export function getFrechetDecisionBody(prologue: string): string {
  return `${prologue}
  ${TRACK_ROLES_WGSL}
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
  }`;
}
