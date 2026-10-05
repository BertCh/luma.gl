// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getRasterExtremaPyramidWGSL,
  type GPURasterExtremaPyramidLayout
} from '../../gpu-raster/raster-pyramid/index';
import {TERRAIN_WGSL_HELPERS} from './terrain-analysis-utils';

/** Marching strategy of the sight-line contributors. */
export type GPUTerrainSightLineTraversal = 'march' | 'pyramid';

/** Relative slack of the pyramid skip bound: 2^-17, more than 100 float32 ULP. @internal */
const TERRAIN_SIGHT_LINE_SKIP_EPSILON = 7.62939453125e-6;

/** Options for {@link getTerrainSightLineWGSL}. @internal */
export type TerrainSightLineWGSLOptions = {
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** `'pyramid'` additionally needs a `pyramid` storage binding holding a combined min-max pyramid. */
  traversal: GPUTerrainSightLineTraversal;
  /** Pyramid layout; required for `'pyramid'`. */
  layout?: GPURasterExtremaPyramidLayout;
  /** When true no early exit happens, so `maxS` and the clearance are exact. */
  clearance: boolean;
};

/**
 * Generates the module-scope WGSL of the tolerance-aware sight-line model shared by
 * `GPUTerrainViewshed`, `GPUTerrainLineOfSight` and `GPUTerrainCumulativeViewshed`.
 *
 * The consuming kernel must bind `elevationValues` (f32), `elevationValidity` (u32) and, for
 * `'pyramid'`, `pyramid` (f32, `2 * layout.length` values: maxima then minima). It gets the
 * constants `HIDDEN`, `VISIBLE`, `OUT_OF_RANGE`, `NO_DATA`, `MARGINAL`, `isValidPixel`,
 * `getElevation`, `sampleElevation` and `traceSightLine`.
 *
 * Sample lattice and expressions are the legacy viewshed ones (`k = 1 .. n-1` at
 * `observer + delta * (f32(k) / f32(n))`, bilinear `sampleElevation`, slope
 * `(h - c*s*s - zo) / s`). Both traversals evaluate this identical per-sample expression.
 *
 * ### Why `'pyramid'` returns bit-identical results to `'march'`
 *
 * The pyramid walk only omits samples that provably cannot change the output; every sample it
 * does evaluate is computed by the same code as in the march. Let `threshold` be the value a
 * sample slope `S` must exceed to matter: `maxS` when the exact maximum is output (clearance), else
 * `T + beta` once `maxS > T - beta` and `T - beta` before (the code only changes when `maxS`
 * crosses `T - beta` or `T + beta`; the early-exit state `maxS > T - beta` is therefore the same in
 * both traversals).
 *
 * 1. Membership. For sample indices `i <= k <= j` the ray position per axis is `o + d * f(k)`
 *    with `f(k) = f32(k) / f32(n)`. Division, multiplication and addition by a fixed value are
 *    monotone under round-to-nearest, so each axis is a monotone function of `k`, and so are the
 *    clamp and `floor`. If the base pixels of samples `i` and `j` (recomputed with the identical
 *    code) are in the same S x S block, every `k` between has its base pixel in that block (a
 *    monotone coordinate cannot leave an interval and come back). The `'bilinear'` pyramid footprint
 *    contains every corner a bilinear sample with such a base reads, so the sample value lies in
 *    `[Hmin, Hmax]` up to the rounding of `mix` (a few ULP). Samples with an invalid corner are
 *    ignored by the march anyway. The candidate `j` is the analytic block exit; it is verified by
 *    recomputing its position and decremented up to three times, so an inexact exit estimate
 *    only costs skipping power, never correctness. Distances `s_k = D * f(k)` are monotone too,
 *    so `s_i <= s_k <= s_j`.
 * 2. Value bound. With `N_k = (h - zo) - c*s_k*s_k` and `S_k = N_k / s_k`: `c*s*s` is monotone in
 *    `s` (either sign of `c`), so `N_k <= (Hmax - zo) - min(c*s_i*s_i, c*s_j*s_j)`. Adding
 *    `EPS * (M + |zo| + max|c*s*s|)` with `EPS = 2^-17` covers the roundings of every float32
 *    operation (each is at most 2^-24 of an operand magnitude, so five or six of them are well under
 *    2^-17), including FMA fusion and fast-math reassociation. Dividing by `s_k in [s_i, s_j]`
 *    gives an upper bound at `s_i` for a non-negative numerator and at `s_j` for a negative one;
 *    the `(1 +- EPS)` factors cover the division rounding. If that bound is `<= threshold`, no skipped
 *    sample can exceed the threshold, so neither the early exit, the code nor `maxS` (when it is
 *    output) changes. Cells without a valid pixel can only contain ignored samples.
 * 3. Every pyramid level is an independent valid skip from the same sample `i`; the walk tries the
 *    level that worked last (descending until one works), then climbs while the bound holds, and
 *    resumes at the largest skippable `j + 1`. The search order only affects speed, never the result.
 *
 * Remaining assumption: a compiler may evaluate the position expression in two textual copies with
 * different FMA contraction; the effect is at most one ULP of position at a block boundary, where
 * the sample still reads only pixels inside the footprint.
 *
 * @internal
 */
export function getTerrainSightLineWGSL(options: TerrainSightLineWGSLOptions): string {
  const {width, height, traversal, layout, clearance} = options;
  if (traversal === 'pyramid' && !layout) {
    throw new Error('terrain sight line traversal pyramid requires a pyramid layout');
  }
  const pyramid = traversal === 'pyramid' ? getRasterExtremaPyramidWGSL(layout!) : '';
  const skipFunction =
    traversal === 'pyramid'
      ? /* wgsl */ `
// Tries to skip samples \`first\` .. last at one pyramid level; returns last + 1, or 0 when the block
// does not contain the ray long enough or its bound exceeds the threshold.
fn trySkipLevel(
  level: u32,
  observer: vec2<f32>,
  delta: vec2<f32>,
  targetDistance: f32,
  first: u32,
  count: u32,
  baseFirst: vec2<u32>,
  distanceFirst: f32,
  eye: f32,
  curvature: f32,
  threshold: f32
) -> u32 {
  let gridMaximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let blockSize = pyramidLevelBlockSize(level);
  let cell = baseFirst / vec2<u32>(blockSize);
  let origin = vec2<f32>(cell * vec2<u32>(blockSize));
  let size = f32(blockSize);
  var exitFraction = 2.0;
  if (delta.x > 0.0) {
    exitFraction = min(exitFraction, (origin.x + size - observer.x) / delta.x);
  } else if (delta.x < 0.0) {
    exitFraction = min(exitFraction, (origin.x - observer.x) / delta.x);
  }
  if (delta.y > 0.0) {
    exitFraction = min(exitFraction, (origin.y + size - observer.y) / delta.y);
  } else if (delta.y < 0.0) {
    exitFraction = min(exitFraction, (origin.y - observer.y) / delta.y);
  }
  var last = u32(clamp(floor(exitFraction * f32(count)), f32(first), f32(count - 1u)));
  var inside = false;
  for (var attempt = 0u; attempt < 4u; attempt++) {
    let baseLast = vec2<u32>(floor(clamp(rayPosition(observer, delta, last, count), vec2<f32>(0.0), gridMaximum)));
    if (all(baseLast / vec2<u32>(blockSize) == cell)) {
      inside = true;
      break;
    }
    if (last <= first) {
      break;
    }
    last = last - 1u;
  }
  if (!inside) {
    return 0u;
  }
  let cellIndex = pyramidLevelIndex(level, baseFirst.x, baseFirst.y);
  let heightMaximum = pyramid[pyramidOffset + cellIndex];
  if (heightMaximum <= PYRAMID_EMPTY_MAXIMUM) {
    return last + 1u;
  }
  let heightMinimum = pyramid[pyramidOffset + PYRAMID_MINIMUM_OFFSET + cellIndex];
  let distanceLast = rayDistance(targetDistance, last, count);
  let magnitude = max(abs(heightMaximum), abs(heightMinimum));
  let dropFirst = curvature * distanceFirst * distanceFirst;
  let dropLast = curvature * distanceLast * distanceLast;
  let dropLow = min(dropFirst, dropLast);
  let dropAbsolute = max(abs(dropFirst), abs(dropLast));
  let numerator = (heightMaximum - eye) - dropLow +
    SKIP_EPSILON * (magnitude + abs(eye) + dropAbsolute);
  let upper = select(
    (numerator / distanceLast) * (1.0 - SKIP_EPSILON),
    (numerator / distanceFirst) * (1.0 + SKIP_EPSILON),
    numerator >= 0.0
  );
  return select(0u, last + 1u, upper <= threshold);
}
// Returns (index of the first sample that must still be evaluated, coarsest level that skipped).
// Every level is an independent valid skip, so the search starts at the level that worked last,
// descends until one works, then climbs while coarser levels also work.
fn skipSamples(
  observer: vec2<f32>,
  delta: vec2<f32>,
  targetDistance: f32,
  first: u32,
  count: u32,
  eye: f32,
  curvature: f32,
  threshold: f32,
  startLevel: u32
) -> vec2<u32> {
  let gridMaximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let baseFirst = vec2<u32>(floor(clamp(rayPosition(observer, delta, first, count), vec2<f32>(0.0), gridMaximum)));
  let distanceFirst = rayDistance(targetDistance, first, count);
  var level = min(startLevel, PYRAMID_LEVEL_COUNT - 1u);
  var next = trySkipLevel(level, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
  while (next == 0u && level > 0u) {
    level = level - 1u;
    next = trySkipLevel(level, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
  }
  if (next == 0u) {
    return vec2<u32>(first, 0u);
  }
  while (level + 1u < PYRAMID_LEVEL_COUNT) {
    let coarser = trySkipLevel(level + 1u, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
    if (coarser == 0u) {
      break;
    }
    next = max(next, coarser);
    level = level + 1u;
  }
  return vec2<u32>(next, level);
}`
      : '';
  const thresholdStatement =
    traversal === 'pyramid'
      ? `
    let threshold = ${clearance ? 'maxSlope' : 'select(lowLimit, highLimit, maxSlope > lowLimit)'};
    let skipped = skipSamples(observer, delta, targetDistance, sampleIndex, stepCount, eye, curvature, threshold, skipLevel);
    if (skipped.x > sampleIndex) {
      sampleIndex = skipped.x;
      skipLevel = skipped.y;
      continue;
    }`
      : '';
  return `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const HIDDEN: u32 = 0u;
const VISIBLE: u32 = 1u;
const OUT_OF_RANGE: u32 = 2u;
const NO_DATA: u32 = 3u;
const MARGINAL: u32 = 4u;
const MAXIMUM_FLOAT: f32 = 3.4028234663852886e38;
const SKIP_EPSILON: f32 = ${TERRAIN_SIGHT_LINE_SKIP_EPSILON};
${TERRAIN_WGSL_HELPERS}
${pyramid}
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation at a clamped pixel-center position; .y is 0 when any corner is invalid.
fn sampleElevation(position: vec2<f32>) -> vec2<f32> {
  let maximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let clamped = clamp(position, vec2<f32>(0.0), maximum);
  let base = vec2<u32>(floor(clamped));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = clamped - floor(clamped);
  if (!isValidPixel(base.x, base.y) || !isValidPixel(next.x, base.y) ||
      !isValidPixel(base.x, next.y) || !isValidPixel(next.x, next.y)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(getElevation(base.x, base.y), getElevation(next.x, base.y), fraction.x);
  let bottom = mix(getElevation(base.x, next.y), getElevation(next.x, next.y), fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}
fn rayPosition(observer: vec2<f32>, delta: vec2<f32>, sampleIndex: u32, count: u32) -> vec2<f32> {
  let fraction = f32(sampleIndex) / f32(count);
  return observer + delta * fraction;
}
fn rayDistance(targetDistance: f32, sampleIndex: u32, count: u32) -> f32 {
  let fraction = f32(sampleIndex) / f32(count);
  return targetDistance * fraction;
}
struct SightLineResult {
  code: u32,
  clearance: f32
}
${skipFunction}
// Observer and target are pixel-center positions; eye is the observer elevation (ground + height);
// targetTop is the target elevation (ground + height) before the curvature drop;
// tolerance is [toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction].
fn traceSightLine(
  observer: vec2<f32>,
  targetPosition: vec2<f32>,
  eye: f32,
  targetTop: f32,
  cellSize: vec2<f32>,
  curvature: f32,
  maxDistance: f32,
  tolerance: vec4<f32>
) -> SightLineResult {
  let delta = targetPosition - observer;
  let targetDistance = length(delta * cellSize);
  if (maxDistance > 0.0 && targetDistance > maxDistance) {
    return SightLineResult(OUT_OF_RANGE, 0.0);
  }
  if (targetDistance == 0.0) {
    return SightLineResult(VISIBLE, MAXIMUM_FLOAT);
  }
  let targetElevation = targetTop - curvature * targetDistance * targetDistance;
  let targetSlope = (targetElevation - eye) / targetDistance;
  let band = (tolerance.x + tolerance.y * targetDistance / 1000.0) / targetDistance;
  let highLimit = targetSlope + band;
  let lowLimit = targetSlope - band;
  let stopDistance = targetDistance - max(tolerance.z, tolerance.w * targetDistance);
  let stepCount = u32(ceil(max(abs(delta.x), abs(delta.y))));
  var maxSlope = -MAXIMUM_FLOAT;
  var tested = false;
  var sampleIndex = 1u;
  var skipLevel = 0u;
  loop {
    if (sampleIndex >= stepCount) { break; }
    if (rayDistance(targetDistance, sampleIndex, stepCount) > stopDistance) { break; }${thresholdStatement}
    let current = sampleIndex;
    sampleIndex = sampleIndex + 1u;
    let elevationSample = sampleElevation(rayPosition(observer, delta, current, stepCount));
    // Invalid samples never block.
    if (elevationSample.y == 0.0) { continue; }
    let sampleDistance = rayDistance(targetDistance, current, stepCount);
    let sampleSlope =
      (elevationSample.x - curvature * sampleDistance * sampleDistance - eye) / sampleDistance;
    tested = true;
    if (sampleSlope > maxSlope) {
      maxSlope = sampleSlope;${
        clearance
          ? ''
          : `
      if (maxSlope > highLimit) {
        return SightLineResult(HIDDEN, 0.0);
      }`
      }
    }
  }
  var code = MARGINAL;
  if (maxSlope > highLimit) {
    code = HIDDEN;
  } else if (maxSlope <= lowLimit) {
    code = VISIBLE;
  }
  return SightLineResult(code, select(MAXIMUM_FLOAT, (targetSlope - maxSlope) * targetDistance, tested));
}`;
}
