// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Largest lattice coordinate of the snap-rounding lattice: `2^22 + 1`. Lattice coordinates are
 * integers in `[0, 2^22 + 1]`, stored as integer-valued `f32`.
 *
 * @internal
 */
export const LATTICE_MAXIMUM_COORDINATE = 2 ** 22 + 1;

/**
 * Exact integer helpers for the snap-rounding lattice, in WGSL.
 *
 * Everything works on signed integers with a magnitude of up to 80 bits, kept as `array<u32, 6>`:
 * word 0 is the sign (an `i32` bit pattern, 0 for zero), words 1 to 5 the 16-bit limbs of the
 * magnitude, least significant first (an array, not a struct: the CPU interpreter of the executed
 * specs cannot construct structs that contain arrays). The crossing terms are `array<u32, 12>`:
 * the denominator in words 0 to 5 and the numerator in words 6 to 11. The inputs are lattice coordinates (magnitude at most `2^22 + 1`)
 * and half-integers (doubled to integers), so every product below stays under `2^70`.
 *
 * - `latticeCrossingTerms(a, b, c, d)`: the numerator `N = cross(c - a, d - c)` and denominator
 *   `D = cross(b - a, d - c)` of the crossing parameter `t = N / D` of segments `ab` and `cd`
 *   (both below `2^46` in magnitude).
 * - `latticeRoundedCrossing(a, b, terms)`: the crossing point of `ab` and `cd` rounded half up to
 *   the lattice, per axis the largest integer `k` with `(2 (k - a) - 1) D <= 2 N (b - a)` (`D > 0`
 *   after flipping signs), found by binary search with an exact comparison of the two products.
 * - `latticeSegmentMeetsPixel(a, b, center)`: whether segment `ab`, known to meet the closed unit
 *   square around the lattice point `center`, meets the half-open square
 *   `[center - 1/2, center + 1/2)` on both axes (closed left and bottom, as GEOS `HotPixel`).
 *
 * One call site rule: each of `latticeCrossingTerms`, `latticeRoundedCrossing` and
 * `latticeSegmentMeetsPixel` reaches `latticeBigMultiply` from exactly one call site (a loop), so
 * a kernel that calls ONE of them inlines the multiplication once. Do not call two of them from one
 * kernel.
 *
 * @internal
 */
export const EXACT_LATTICE_WGSL = /* wgsl */ `
fn latticeSignOf(value: i32) -> i32 {
  return select(select(0, -1, value < 0), 1, value > 0);
}

fn latticeBigSign(value: array<u32, 6>) -> i32 { return bitcast<i32>(value[0]); }

fn latticeBigZero() -> array<u32, 6> {
  return array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
}

// |value| < 2^31.
fn latticeBigFromI32(value: i32) -> array<u32, 6> {
  let magnitude = u32(abs(value));
  return array<u32, 6>(bitcast<u32>(latticeSignOf(value)), magnitude & 0xffffu, magnitude >> 16u, 0u, 0u, 0u);
}

fn latticeBigNegate(value: array<u32, 6>) -> array<u32, 6> {
  var result = value;
  result[0] = bitcast<u32>(-bitcast<i32>(value[0]));
  return result;
}

// Compares magnitudes (words 1 to 5): -1, 0 or 1.
fn latticeCompareMagnitude(first: array<u32, 6>, second: array<u32, 6>) -> i32 {
  // Parameters are copied to variables before indexing them with a loop index.
  var firstLimbs = first;
  var secondLimbs = second;
  for (var slot = 5u; slot > 0u; slot--) {
    if (firstLimbs[slot] != secondLimbs[slot]) { return select(-1, 1, firstLimbs[slot] > secondLimbs[slot]); }
  }
  return 0;
}

// Compares signed values: -1, 0 or 1.
fn latticeBigCompare(first: array<u32, 6>, second: array<u32, 6>) -> i32 {
  let firstSign = latticeBigSign(first);
  let secondSign = latticeBigSign(second);
  if (firstSign != secondSign) { return select(-1, 1, firstSign > secondSign); }
  if (firstSign == 0) { return 0; }
  let order = latticeCompareMagnitude(first, second);
  return select(-order, order, firstSign > 0);
}

fn latticeBigAdd(first: array<u32, 6>, second: array<u32, 6>) -> array<u32, 6> {
  let firstSign = latticeBigSign(first);
  let secondSign = latticeBigSign(second);
  if (firstSign == 0) { return second; }
  if (secondSign == 0) { return first; }
  var result = latticeBigZero();
  var firstLimbs = first;
  var secondLimbs = second;
  if (firstSign == secondSign) {
    var carry = 0u;
    for (var slot = 1u; slot < 6u; slot++) {
      let total = firstLimbs[slot] + secondLimbs[slot] + carry;
      result[slot] = total & 0xffffu;
      carry = total >> 16u;
    }
    result[0] = first[0];
    return result;
  }
  let order = latticeCompareMagnitude(first, second);
  if (order == 0) { return result; }
  var larger = firstLimbs;
  var smaller = secondLimbs;
  if (order < 0) {
    larger = secondLimbs;
    smaller = firstLimbs;
  }
  var borrow = 0u;
  for (var slot = 1u; slot < 6u; slot++) {
    let difference = larger[slot] + 0x10000u - smaller[slot] - borrow;
    result[slot] = difference & 0xffffu;
    borrow = 1u - (difference >> 16u);
  }
  result[0] = larger[0];
  return result;
}

// value * factor for |factor| < 2^31 (the product must stay below 2^80).
fn latticeBigMultiply(value: array<u32, 6>, factor: i32) -> array<u32, 6> {
  var result = latticeBigZero();
  if (latticeBigSign(value) == 0 || factor == 0) { return result; }
  let magnitude = u32(abs(factor));
  var factorLimbs = array<u32, 2>(magnitude & 0xffffu, magnitude >> 16u);
  var valueLimbs = value;
  for (var factorSlot = 0u; factorSlot < 2u; factorSlot++) {
    var carry = 0u;
    for (var slot = 1u; slot + factorSlot < 6u; slot++) {
      let total = result[slot + factorSlot] + valueLimbs[slot] * factorLimbs[factorSlot] + carry;
      result[slot + factorSlot] = total & 0xffffu;
      carry = total >> 16u;
    }
  }
  result[0] = bitcast<u32>(latticeBigSign(value) * latticeSignOf(factor));
  return result;
}

// Denominator D = cross(b - a, d - c) (words 0 to 5) and numerator N = cross(c - a, d - c) (words
// 6 to 11) of the crossing parameter of segments ab and cd. Coordinates are lattice integers
// (magnitude <= 2^22 + 1).
fn latticeCrossingTerms(a: vec2i, b: vec2i, c: vec2i, d: vec2i) -> array<u32, 12> {
  var left = array<i32, 4>(b.x - a.x, b.y - a.y, c.x - a.x, c.y - a.y);
  var right = array<i32, 4>(d.y - c.y, d.x - c.x, d.y - c.y, d.x - c.x);
  var denominator = latticeBigZero();
  var numerator = latticeBigZero();
  for (var term = 0u; term < 4u; term++) {
    var product = latticeBigMultiply(latticeBigFromI32(left[term]), right[term]);
    if ((term & 1u) == 1u) { product = latticeBigNegate(product); }
    if (term < 2u) { denominator = latticeBigAdd(denominator, product); } else { numerator = latticeBigAdd(numerator, product); }
  }
  var terms = array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  for (var slot = 0u; slot < 6u; slot++) {
    terms[slot] = denominator[slot];
    terms[slot + 6u] = numerator[slot];
  }
  return terms;
}

// The crossing point of ab and cd rounded half up to the lattice: per axis the largest integer k
// in the range of the segment with (2 (k - a) - 1) D <= 2 N (b - a), D > 0. Requires D != 0 and a
// crossing on ab (the exact point lies within the range of the segment).
fn latticeRoundedCrossing(a: vec2i, b: vec2i, terms: array<u32, 12>) -> vec2i {
  var denominator = latticeBigZero();
  var numerator = latticeBigZero();
  var termWords = terms;
  for (var slot = 0u; slot < 6u; slot++) {
    denominator[slot] = termWords[slot];
    numerator[slot] = termWords[slot + 6u];
  }
  if (latticeBigSign(denominator) < 0) {
    denominator = latticeBigNegate(denominator);
    numerator = latticeBigNegate(numerator);
  }
  var result = vec2i(0, 0);
  for (var axis = 0u; axis < 2u; axis++) {
    let origin = select(a.y, a.x, axis == 0u);
    let endCoordinate = select(b.y, b.x, axis == 0u);
    var low = min(origin, endCoordinate);
    var high = max(origin, endCoordinate);
    var right = latticeBigZero();
    var middle = low;
    // Step 0 forms 2 N (b - a); every later step tests one candidate. The at most 23 candidate
    // steps of a range below 2^23 fit the 24 steps.
    for (var step = 0u; step < 25u; step++) {
      var multiplicand = denominator;
      var factor = 0i;
      if (step == 0u) {
        multiplicand = numerator;
        factor = 2 * (endCoordinate - origin);
      } else {
        if (low >= high) { break; }
        middle = low + (high - low + 1) / 2;
        factor = 2 * (middle - origin) - 1;
      }
      let product = latticeBigMultiply(multiplicand, factor);
      if (step == 0u) {
        right = product;
      } else if (latticeBigCompare(product, right) <= 0) {
        low = middle;
      } else {
        high = middle - 1;
      }
    }
    if (axis == 0u) { result.x = low; } else { result.y = low; }
  }
  return result;
}

// Signs of cross(b - a, corner - a) for the four corners of the unit square around center, in the
// order bottom-left, top-left, top-right, bottom-right (coordinates doubled so corners are integers).
fn latticeCornerSigns(a: vec2i, b: vec2i, center: vec2i) -> vec4i {
  let edge = b - a;
  var signs = array<i32, 4>(0, 0, 0, 0);
  for (var corner = 0u; corner < 4u; corner++) {
    let cornerX = 2 * center.x + select(1, -1, corner < 2u);
    let cornerY = 2 * center.y + select(-1, 1, corner == 1u || corner == 2u);
    let relativeX = cornerX - 2 * a.x;
    let relativeY = cornerY - 2 * a.y;
    var difference = latticeBigZero();
    for (var term = 0u; term < 2u; term++) {
      var product = latticeBigMultiply(latticeBigFromI32(select(edge.x, edge.y, term == 1u)), select(relativeY, relativeX, term == 1u));
      if (term == 1u) { product = latticeBigNegate(product); }
      difference = latticeBigAdd(difference, product);
    }
    signs[corner] = latticeBigSign(difference);
  }
  return vec4i(signs[0], signs[1], signs[2], signs[3]);
}

// Whether segment ab meets the half-open pixel [center - 1/2, center + 1/2) in x and y, for a
// segment with lattice endpoints that meets the closed square. Only a segment that touches the
// closed square in exactly one top-left, top-right or bottom-right corner misses the half-open
// square (an endpoint is a lattice point, so the segment passes through such a corner and the
// square lies entirely on one side of its line).
fn latticeSegmentMeetsPixel(a: vec2i, b: vec2i, center: vec2i) -> bool {
  let signs = latticeCornerSigns(a, b, center);
  var values = array<i32, 4>(signs.x, signs.y, signs.z, signs.w);
  for (var corner = 1u; corner < 4u; corner++) {
    if (values[corner] != 0) { continue; }
    let cornerX = 2 * center.x + select(1, -1, corner < 2u);
    let cornerY = 2 * center.y + select(-1, 1, corner == 1u || corner == 2u);
    let onSegment = 2 * min(a.x, b.x) <= cornerX && cornerX <= 2 * max(a.x, b.x) &&
      2 * min(a.y, b.y) <= cornerY && cornerY <= 2 * max(a.y, b.y);
    if (!onSegment) { continue; }
    var hasPositive = false;
    var hasNegative = false;
    for (var other = 0u; other < 4u; other++) {
      if (other == corner) { continue; }
      hasPositive = hasPositive || values[other] > 0;
      hasNegative = hasNegative || values[other] < 0;
    }
    if (!(hasPositive && hasNegative)) { return false; }
  }
  return true;
}
`;
