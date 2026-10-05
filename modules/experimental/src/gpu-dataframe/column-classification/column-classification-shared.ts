// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL helpers shared by the column-classification recipes: a NaN constant, a NaN test by bits,
 * and the order-preserving `u32` encoding of `f32` values (`-0` sorts below `+0`).
 *
 * Unsigned comparison of the keys orders values exactly as a numeric sort does, so selection,
 * minimum, and maximum run on integer keys and are bitwise exact.
 *
 * @internal
 */
export const COLUMN_ORDERED_KEY_WGSL = /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isNanBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn getOrderedKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}
`;

const scratchFloat = new Float32Array(1);
const scratchBits = new Uint32Array(scratchFloat.buffer);

/**
 * Order-preserving `u32` key of a non-NaN `f32` value, identical to the WGSL `getOrderedKey`.
 *
 * @internal
 */
export function getOrderedFloat32Key(value: number): number {
  scratchFloat[0] = value;
  const bits = scratchBits[0];
  return ((bits & 0x80000000) !== 0 ? ~bits : bits ^ 0x80000000) >>> 0;
}

/**
 * Inverse of {@link getOrderedFloat32Key}, identical to the WGSL `decodeOrderedKey`.
 *
 * @internal
 */
export function decodeOrderedFloat32Key(key: number): number {
  scratchBits[0] = ((key & 0x80000000) !== 0 ? key ^ 0x80000000 : ~key) >>> 0;
  return scratchFloat[0];
}
