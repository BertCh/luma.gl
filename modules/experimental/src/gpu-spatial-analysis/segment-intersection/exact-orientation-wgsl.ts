// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact 2D orientation predicate in WGSL, shared by the segment, relate and point-in-polygon kernels.
 *
 * `orientSign(a, b, c)` returns the sign of `cross(b - a, c - a)` as `-1`, `0` or `1`, or `2` when
 * the sign cannot be certified. A floating-point filter with a conservative error bound decides
 * clear cases. Otherwise the determinant is evaluated exactly in integer arithmetic: it is the sum
 * of six products of raw coordinates, each product of two 24-bit significands is exact in 48 bits,
 * and the six signed terms are aligned to the smallest exponent and added into two 256-bit
 * unsigned accumulators (positive and negative terms) that are then compared. Integer arithmetic is
 * used because float error-free transforms (`twoSum`, `fma` residuals) are removed by shader
 * compilers that reassociate floating-point expressions.
 *
 * The result is `2` for non-finite coordinates and when the terms span more than 200 binary
 * orders of magnitude (the accumulators are 256 bits wide). Subnormal coordinates are not
 * supported: devices may flush them to zero.
 *
 * @internal
 */
export const EXACT_ORIENTATION_WGSL = /* wgsl */ `
const ORIENT_MAXIMUM_SPAN: i32 = 200;
const ORIENT_FLOAT_MAXIMUM: f32 = 3.402823466e+38;

fn orientIsFinite(value: f32) -> bool { return value == value && abs(value) <= ORIENT_FLOAT_MAXIMUM; }

// Exact 24-bit by 24-bit product as (low word, high word).
fn multiply24(first: u32, second: u32) -> vec2u {
  let firstHigh = first >> 12u;
  let firstLow = first & 0xfffu;
  let secondHigh = second >> 12u;
  let secondLow = second & 0xfffu;
  let highProduct = firstHigh * secondHigh;
  let middle = firstHigh * secondLow + firstLow * secondHigh;
  let lowProduct = firstLow * secondLow;
  let partial = lowProduct + ((middle & 0xfffffu) << 12u);
  let carryOne = select(0u, 1u, partial < lowProduct);
  let low = partial + (highProduct << 24u);
  let carryTwo = select(0u, 1u, low < partial);
  return vec2u(low, (highProduct >> 8u) + (middle >> 20u) + carryOne + carryTwo);
}

// Adds (high * 2^32 + low) * 2^shift to a 256-bit little-endian accumulator. high < 2^16.
fn accumulateShifted(accumulator: ptr<function, array<u32, 8>>, low: u32, high: u32, shift: u32) {
  let firstWord = shift / 32u;
  let bit = shift % 32u;
  var word0 = low << bit;
  var word1 = high << bit;
  var word2 = 0u;
  if (bit > 0u) {
    word1 = word1 | (low >> (32u - bit));
    word2 = high >> (32u - bit);
  }
  var carry = 0u;
  for (var slot = firstWord; slot < 8u; slot++) {
    let relative = slot - firstWord;
    var addend = 0u;
    if (relative == 0u) { addend = word0; } else if (relative == 1u) { addend = word1; } else if (relative == 2u) { addend = word2; } else if (carry == 0u) { break; }
    let partial = (*accumulator)[slot] + addend;
    let carryOne = select(0u, 1u, partial < addend);
    let total = partial + carry;
    let carryTwo = select(0u, 1u, total < partial);
    (*accumulator)[slot] = total;
    carry = carryOne | carryTwo;
  }
}

fn orientExact(a: vec2f, b: vec2f, c: vec2f) -> i32 {
  var xBits = array<u32, 6>(
    bitcast<u32>(a.x), bitcast<u32>(a.y) ^ 0x80000000u, bitcast<u32>(b.x),
    bitcast<u32>(b.y) ^ 0x80000000u, bitcast<u32>(c.x), bitcast<u32>(c.y) ^ 0x80000000u);
  var yBits = array<u32, 6>(
    bitcast<u32>(b.y), bitcast<u32>(b.x), bitcast<u32>(c.y),
    bitcast<u32>(c.x), bitcast<u32>(a.y), bitcast<u32>(a.x));
  var exponents: array<i32, 6>;
  var lows: array<u32, 6>;
  var highs: array<u32, 6>;
  var negatives: array<bool, 6>;
  var present: array<bool, 6>;
  var minimumExponent = 100000;
  var maximumExponent = -100000;
  for (var term = 0u; term < 6u; term++) {
    let xMagnitude = xBits[term] & 0x7fffffffu;
    let yMagnitude = yBits[term] & 0x7fffffffu;
    let xExponentBits = xMagnitude >> 23u;
    let yExponentBits = yMagnitude >> 23u;
    let xFraction = xMagnitude & 0x7fffffu;
    let yFraction = yMagnitude & 0x7fffffu;
    if (xMagnitude == 0u || yMagnitude == 0u) { continue; }
    let xSignificand = select(0x800000u | xFraction, xFraction, xExponentBits == 0u);
    let ySignificand = select(0x800000u | yFraction, yFraction, yExponentBits == 0u);
    let xExponent = select(i32(xExponentBits) - 150, -149, xExponentBits == 0u);
    let yExponent = select(i32(yExponentBits) - 150, -149, yExponentBits == 0u);
    let product = multiply24(xSignificand, ySignificand);
    present[term] = true;
    lows[term] = product.x;
    highs[term] = product.y;
    exponents[term] = xExponent + yExponent;
    negatives[term] = ((xBits[term] ^ yBits[term]) >> 31u) == 1u;
    minimumExponent = min(minimumExponent, exponents[term]);
    maximumExponent = max(maximumExponent, exponents[term]);
  }
  if (minimumExponent > maximumExponent) { return 0; }
  if (maximumExponent - minimumExponent > ORIENT_MAXIMUM_SPAN) { return 2; }
  var positive: array<u32, 8>;
  var negative: array<u32, 8>;
  for (var term = 0u; term < 6u; term++) {
    if (!present[term]) { continue; }
    let shift = u32(exponents[term] - minimumExponent);
    if (negatives[term]) {
      accumulateShifted(&negative, lows[term], highs[term], shift);
    } else {
      accumulateShifted(&positive, lows[term], highs[term], shift);
    }
  }
  for (var slot = 8u; slot > 0u; slot--) {
    let plus = positive[slot - 1u];
    let minus = negative[slot - 1u];
    if (plus != minus) { return select(-1, 1, plus > minus); }
  }
  return 0;
}

fn orientSign(a: vec2f, b: vec2f, c: vec2f) -> i32 {
  if (!(orientIsFinite(a.x) && orientIsFinite(a.y) && orientIsFinite(b.x) &&
        orientIsFinite(b.y) && orientIsFinite(c.x) && orientIsFinite(c.y))) {
    return 2;
  }
  if ((a.x == c.x || b.y == c.y) && (a.y == c.y || b.x == c.x)) { return 0; }
  let detLeft = (a.x - c.x) * (b.y - c.y);
  let detRight = (a.y - c.y) * (b.x - c.x);
  let det = detLeft - detRight;
  // Shewchuk's f32 filter bound is about 1.8e-7 of the magnitude sum; keep a safety margin.
  let bound = 3.0e-7 * (abs(detLeft) + abs(detRight));
  if (det > bound) { return 1; }
  if (-det > bound) { return -1; }
  return orientExact(a, b, c);
}
`;
