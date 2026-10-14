// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../utils/wgsl-kernel-nodes';
import {DECODE_ORDERED_KEY_WGSL, GET_NAN_WGSL, GET_ORDERED_KEY_WGSL} from '../utils/wgsl-common';

/** Default fixed-point scale of cell-table sums: 16 fractional bits. */
export const GPU_CELL_DEFAULT_SUM_SCALE = 65536;

/** Validates a fixed-point sum scale. @internal */
export function validateSumScale(id: string, sumScale: number): void {
  if (!Number.isFinite(sumScale) || sumScale <= 0 || Math.fround(sumScale) !== sumScale) {
    throw new Error(`${id} sumScale must be a positive finite float32 value`);
  }
}

/**
 * Order-preserving u32 encoding of f32 (`-0` below `+0`) and a NaN builder.
 *
 * @internal
 */
export const ORDERED_KEY_WGSL = /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${GET_NAN_WGSL}

${GET_ORDERED_KEY_WGSL}

${DECODE_ORDERED_KEY_WGSL}
`;

/** Largest f32 below 2^62: scaled values saturate here so 64-bit sums of 2^31 rows cannot wrap far. */
export const CELL_MAXIMUM_SCALED_VALUE = Math.fround(2 ** 62 - 2 ** 38);

/** Signed 64-bit fixed-point helpers on `vec2u(high, low)`. @internal */
export function getFixedPointWGSL(sumScale: number): string {
  return /* wgsl */ `
const SUM_SCALE: f32 = ${getWGSLFloatLiteral(sumScale)};
const MAXIMUM_SCALED_VALUE: f32 = ${getWGSLFloatLiteral(CELL_MAXIMUM_SCALED_VALUE)};

// Not dggs_u64_add and friends: pulling in the dggs module costs compile time (see cell-keys.ts).
fn cellAddI64(left: vec2u, right: vec2u) -> vec2u {
  let low = left.y + right.y;
  return vec2u(left.x + right.x + select(0u, 1u, low < left.y), low);
}

fn cellNegateI64(value: vec2u) -> vec2u {
  let low = ~value.y + 1u;
  return vec2u(~value.x + select(0u, 1u, low == 0u), low);
}

/** roundHalfEven(value * SUM_SCALE), saturated, as a two's-complement 64-bit integer. */
fn cellScaleValue(value: f32) -> vec2u {
  let scaled = clamp(round(value * SUM_SCALE), -MAXIMUM_SCALED_VALUE, MAXIMUM_SCALED_VALUE);
  let magnitude = abs(scaled);
  let highFloat = floor(magnitude * 2.3283064365386963e-10);
  let lowFloat = magnitude - highFloat * 4294967296.0;
  let result = vec2u(u32(highFloat), u32(lowFloat));
  return select(result, cellNegateI64(result), scaled < 0.0);
}

fn cellI64ToF32(value: vec2u) -> f32 {
  let isNegative = (value.x & 0x80000000u) != 0u;
  let magnitude = select(value, cellNegateI64(value), isNegative);
  let result = f32(magnitude.x) * 4294967296.0 + f32(magnitude.y);
  return select(result, -result, isNegative);
}
`;
}
