// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Small WGSL helper declarations shared by the GPU contributor kernels.
 *
 * Each constant declares exactly one function, so a kernel interpolates only what it calls.
 * Declaring a function twice in one shader fails compilation, so kernels that concatenate several
 * declaration strings must include each constant once.
 */

/**
 * `getNaN()`: a quiet NaN. WGSL rejects NaN constants, so it is built from a runtime bit pattern.
 * @internal
 */
export const GET_NAN_WGSL = /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
`;

/** `getQuietNaN()`: same as {@link GET_NAN_WGSL} under the name the raster and terrain kernels use. @internal */
export const GET_QUIET_NAN_WGSL = /* wgsl */ `
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
`;

/**
 * `getQuietNaN(seed)`: a quiet NaN that depends on a runtime operand (`seed & 0u` is always zero),
 * which keeps the bit pattern out of constant evaluation. @internal
 */
export const GET_QUIET_NAN_SEEDED_WGSL = /* wgsl */ `
fn getQuietNaN(seed: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (seed & 0u)); }
`;

/** `isNanBits(x)`: NaN test on the bit pattern, immune to compiler folding of `x != x`. @internal */
export const IS_NAN_BITS_WGSL = /* wgsl */ `
fn isNanBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}
`;

/** `isFiniteBits(x)`: finite test on the exponent bits. @internal */
export const IS_FINITE_BITS_WGSL = /* wgsl */ `
fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
`;

/** `isFiniteValue(value)`: finite test on the exponent bits. @internal */
export const IS_FINITE_VALUE_WGSL = /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`;

/** `isFiniteFloat(value)`: same as {@link IS_FINITE_VALUE_WGSL} under the name the older kernels use. @internal */
export const IS_FINITE_FLOAT_WGSL = /* wgsl */ `
fn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`;

/**
 * `getOrderedKey(x)`: order-preserving `u32` key of an `f32`, so unsigned comparison orders values
 * numerically (`-0` sorts below `+0`). @internal
 */
export const GET_ORDERED_KEY_WGSL = /* wgsl */ `
fn getOrderedKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`;

/** `decodeOrderedKey(key)`: inverse of {@link GET_ORDERED_KEY_WGSL}. @internal */
export const DECODE_ORDERED_KEY_WGSL = /* wgsl */ `
fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}
`;

/** `spreadBits(value)`: spreads the low 16 bits of `value` to even bit positions (Morton order). @internal */
export const SPREAD_BITS_WGSL = /* wgsl */ `
fn spreadBits(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
`;

/**
 * `nodeOverlaps(node, queryMinimum, queryMaximum)`: closed-box overlap of a BVH node against a
 * query box. Requires `nodeMinima` and `nodeMaxima` bindings. @internal
 */
export const NODE_OVERLAPS_WGSL = /* wgsl */ `
fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}
`;

/** `erfcValue(x)`: complementary error function, Numerical Recipes Chebyshev fit (relative error under 1.2e-7). @internal */
export const ERFC_VALUE_WGSL = /* wgsl */ `
fn erfcValue(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return select(2.0 - r, r, x >= 0.0);
}
`;
