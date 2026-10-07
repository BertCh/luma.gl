// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  forEachEdge,
  getSpatialPredicateWGSL,
  type SpatialDistanceParameter,
  type SpatialPredicateSide
} from './spatial-predicate-wgsl';

/** Invocations per pair of the workgroup `dwithin` kernel. @internal */
export const SPATIAL_DWITHIN_WORKGROUP_SIZE = 64;

/**
 * Returns module-scope WGSL declaring `dwithinWorkgroupPair(l, r, valid, lane) -> bool`: the
 * `dwithin` test of one pair split across the lanes of a workgroup. It follows the same rules as
 * the single-invocation kernel (`pairMatches` of {@link getSpatialPredicateWGSL}): true when the
 * closed geometries intersect or are at most `distance` apart.
 *
 * Lanes take every 64th edge of the left feature and stop at the first right edge within reach,
 * skipping edges whose bounding box is farther than `distance` from the right feature (the skip
 * threshold has a 0.1% margin so rounding never turns a match into a miss). Every lane of the
 * workgroup must call it with the same arguments from uniform control flow (it contains a
 * barrier); only lane 0 receives the merged result.
 *
 * @internal
 */
export function getSpatialDwithinWorkgroupWGSL(
  left: SpatialPredicateSide,
  right: SpatialPredicateSide,
  distance: number | SpatialDistanceParameter
): string {
  const size = SPATIAL_DWITHIN_WORKGROUP_SIZE;
  return `${getSpatialPredicateWGSL(left, right, 'dwithin', distance)}
var<private> relateLane: u32 = 0u;
var<private> relateStride: u32 = 1u;
var<workgroup> dwithinMatched: atomic<u32>;
// Squared gap between the bounding boxes of segments a-b and c-d (zero when they overlap).
fn boxGapSq(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  let gap = max(max(min(a, b) - max(c, d), min(c, d) - max(a, b)), vec2f(0.0));
  return dot(gap, gap);
}
fn dwithinWorkgroupPair(l: u32, r: u32, valid: bool, lane: u32) -> bool {
  relateLane = lane;
  relateStride = ${size}u;
  let limitSq = distanceLimitSq();
  let reachSq = limitSq * 1.001 + 1e-30;
  if (valid && limitSq >= 0.0 && !(leftFirst(l).z == 0.0 || rightFirst(r).z == 0.0)) {
    // Bounding box of the right feature.
    var rightLow = vec2f(FLOAT32_MAXIMUM);
    var rightHigh = vec2f(-FLOAT32_MAXIMUM);
    ${forEachEdge(
      right,
      'r',
      'c',
      'd',
      `rightLow = min(rightLow, min(c, d));
    rightHigh = max(rightHigh, max(c, d));`
    )}
    var matched = false;
    ${forEachEdge(
      left,
      'l',
      'a',
      'b',
      `if (matched) { break; }
    if (boxGapSq(a, b, rightLow, rightHigh) > reachSq) { continue; }
    ${forEachEdge(
      right,
      'r',
      'c',
      'd',
      `if (boxGapSq(a, b, c, d) > reachSq) { continue; }
      if (segmentDistanceSq(a, b, c, d) <= limitSq) { matched = true; break; }`
    )}`,
      true
    )}
    // Containment without an edge within reach (one polygon inside the other, or a point in one).
    if (lane == 0u && !matched) {
      ${right.kind === 'polygons' ? 'if (rightLocate(leftFirst(l).xy, r) != 0u) { matched = true; }' : ''}
      ${left.kind === 'polygons' ? 'if (leftLocate(rightFirst(r).xy, l) != 0u) { matched = true; }' : ''}
    }
    if (matched) { atomicOr(&dwithinMatched, 1u); }
  }
  workgroupBarrier();
  return atomicLoad(&dwithinMatched) != 0u;
}`;
}
