// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  DECODE_ORDERED_KEY_WGSL,
  GET_NAN_WGSL,
  GET_ORDERED_KEY_WGSL,
  IS_NAN_BITS_WGSL
} from '../../utils/wgsl-common';

/**
 * WGSL helpers shared by the column-classification contributors: a NaN constant, a NaN test by bits,
 * and the order-preserving `u32` encoding of `f32` values (`-0` sorts below `+0`).
 *
 * Unsigned comparison of the keys orders values exactly as a numeric sort does, so selection,
 * minimum, and maximum run on integer keys and are bitwise exact.
 *
 * @internal
 */
export const COLUMN_ORDERED_KEY_WGSL = /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${GET_NAN_WGSL}

${IS_NAN_BITS_WGSL}

${GET_ORDERED_KEY_WGSL}

${DECODE_ORDERED_KEY_WGSL}
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
