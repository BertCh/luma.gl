// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact Int64 time on a GPU that only has 32-bit numbers.
 *
 * A time is an absolute signed 64-bit integer, typically Arrow `Int64` or `Timestamp` epoch
 * milliseconds, stored as its two little-endian `uint32` words `(low, high)`: one `uint32x2` row,
 * which for an Arrow leaf is the leaf's own bytes ({@link getInt64TimeWords}). Per-frame instants
 * such as a playhead or a window edge use the same words plus an f32 fraction in `[0, 1)` for
 * sub-unit motion ({@link splitTimeWords}). Kernels subtract two times with a borrow, which is exact,
 * and only then convert the difference to f32 ({@link TIME_WORDS_WGSL}). There is no epoch to
 * choose and nothing to rebase when the playhead moves days or years.
 *
 * Precision: comparisons are exact for every Int64 value. A converted difference is exact below
 * 2^24 units and rounded to f32 above (relative error below 2^-23, two roundings).
 */

import type {GraphDataView, GraphVectorView} from '@luma.gl/gpgpu/gpu-core';

/**
 * Int64 word timestamps: one packed `uint32x2` row `(low, high)` per row, a single view or an
 * ordered vector of chunks, for example an Arrow `Int64` or `Timestamp` column uploaded as is.
 */
export type GPUInt64TimeWordRows = GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;

/** Number of uint32 elements in a word time-window parameter buffer. */
export const GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH = 8;

const INT64_MINIMUM = -(2n ** 63n);
const INT64_MAXIMUM = 2n ** 63n - 1n;
const IS_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** One instant split into Int64 words and an f32 fraction, `value = high * 2^32 + low + fraction`. */
export type GPUTimeWords = {
  /** Low word of the integer part, as an unsigned 32-bit value. */
  low: number;
  /** High word of the integer part in two's complement, as an unsigned 32-bit value. */
  high: number;
  /** f32 fraction in `[0, 1)`. Always 0 for a `bigint` instant. */
  fraction: number;
};

/**
 * CPU description of one per-frame time window over Int64 word timestamps.
 *
 * Instants use the unit of the timestamps, for example epoch milliseconds. A `number` may carry a
 * fraction (a smooth playhead); a `bigint` is an exact integer.
 */
export type GPUTimeWordWindow = {
  /** Inclusive window start. */
  start: number | bigint;
  /** Inclusive window end. A window with `end < start` accepts nothing. */
  end: number | bigint;
  /** Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. */
  startFadeDuration?: number;
  /** Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. */
  endFadeDuration?: number;
};

/**
 * Splits one instant into Int64 words and an f32 fraction.
 *
 * A `number` is split into `floor(value)` and `Math.fround(value - floor(value))`; when the fraction
 * rounds up to 1 the integer part is incremented instead, so the fraction stays in `[0, 1)`.
 *
 * @throws If a `number` is not finite or its integer part is not a safe integer, or a `bigint` is
 * outside the signed 64-bit range.
 */
export function splitTimeWords(value: number | bigint): GPUTimeWords {
  let integer: bigint;
  let fraction = 0;
  if (typeof value === 'bigint') {
    integer = value;
  } else {
    if (!Number.isFinite(value)) {
      throw new Error('Time words require a finite instant');
    }
    let whole = Math.floor(value);
    fraction = Math.fround(value - whole);
    if (fraction >= 1) {
      whole += 1;
      fraction = 0;
    }
    if (!Number.isSafeInteger(whole)) {
      throw new Error('Time words require an instant whose integer part is a safe integer');
    }
    integer = BigInt(whole);
  }
  if (integer < INT64_MINIMUM || integer > INT64_MAXIMUM) {
    throw new Error('Time words require an instant inside the signed 64-bit range');
  }
  const unsigned = BigInt.asUintN(64, integer);
  return {
    low: Number(unsigned & 0xffffffffn),
    high: Number(unsigned >> 32n),
    fraction
  };
}

/** Joins Int64 words back into a `bigint`, the inverse of the integer part of {@link splitTimeWords}. */
export function joinTimeWords(low: number, high: number): bigint {
  return BigInt.asIntN(64, (BigInt(high >>> 0) << 32n) | BigInt(low >>> 0));
}

/**
 * Returns the `(low, high)` words of Int64 values as a zero-copy `Uint32Array` over the same bytes,
 * two words per row, ready to upload as a `uint32x2` column.
 *
 * @throws On a big-endian platform, where the in-memory words would be swapped.
 */
export function getInt64TimeWords(values: BigInt64Array | BigUint64Array): Uint32Array {
  if (!IS_LITTLE_ENDIAN) {
    throw new Error('Int64 time words require a little-endian platform');
  }
  return new Uint32Array(values.buffer, values.byteOffset, values.length * 2);
}

/**
 * Packs a word time window into the 8-element `uint32` layout read by word-time kernels.
 *
 * Layout: `[startLow, startHigh, endLow, endHigh, startFraction, endFraction, startFadeDuration,
 * endFadeDuration]`, where the last four are f32 bit patterns. Write it into a
 * `GPUParameterBuffer` with `format: 'uint32'` between encodings.
 *
 * @param window Window to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If an instant cannot be split, a fade duration is negative or not finite, or `target` is
 * too short.
 */
export function getGPUTimeWindowWordParameterValues(
  window: GPUTimeWordWindow,
  target: Uint32Array = new Uint32Array(GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH) {
    throw new Error(
      `Time word window target must hold ${GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH} elements`
    );
  }
  const startFadeDuration = window.startFadeDuration ?? 0;
  const endFadeDuration = window.endFadeDuration ?? 0;
  for (const value of [startFadeDuration, endFadeDuration]) {
    if (!Number.isFinite(value) || value < 0) {
      throw new Error('Time window fade durations must be finite and non-negative');
    }
  }
  const start = splitTimeWords(window.start);
  const end = splitTimeWords(window.end);
  const floats = new Float32Array([
    start.fraction,
    end.fraction,
    startFadeDuration,
    endFadeDuration
  ]);
  target.set([start.low, start.high, end.low, end.high]);
  target.set(new Uint32Array(floats.buffer), 4);
  return target;
}

/**
 * WGSL helpers for Int64 word times stored as `vec2<u32>(low, high)`.
 *
 * - `timeWordsSubtract(a, b)`: exact two's-complement `a - b` with a borrow.
 * - `timeWordsIsNegative(d)`: sign of a difference.
 * - `timeWordsToF32(d)`: signed difference rounded to f32 (exact below 2^24).
 * - `isTimeWordsAtLeast(a, aFraction, b, bFraction)`: exact `a + aFraction >= b + bFraction` for
 *   fractions in `[0, 1)`. The fractions only matter when the integer parts are equal.
 * - `timeWordsDifference(a, aFraction, b, bFraction)`: `(a + aFraction) - (b + bFraction)` in f32.
 */
export const TIME_WORDS_WGSL = /* wgsl */ `
fn timeWordsSubtract(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let borrow = select(0u, 1u, a.x < b.x);
  return vec2<u32>(a.x - b.x, a.y - b.y - borrow);
}

fn timeWordsIsNegative(difference: vec2<u32>) -> bool {
  return (difference.y & 0x80000000u) != 0u;
}

fn timeWordsToF32(difference: vec2<u32>) -> f32 {
  let negative = timeWordsIsNegative(difference);
  var low = difference.x;
  var high = difference.y;
  if (negative) {
    low = ~difference.x + 1u;
    high = ~difference.y + select(0u, 1u, low == 0u);
  }
  let magnitude = f32(high) * 4294967296.0 + f32(low);
  return select(magnitude, -magnitude, negative);
}

fn isTimeWordsAtLeast(a: vec2<u32>, aFraction: f32, b: vec2<u32>, bFraction: f32) -> bool {
  let difference = timeWordsSubtract(a, b);
  if (difference.x == 0u && difference.y == 0u) {
    return aFraction >= bFraction;
  }
  return !timeWordsIsNegative(difference);
}

fn timeWordsDifference(a: vec2<u32>, aFraction: f32, b: vec2<u32>, bFraction: f32) -> f32 {
  return timeWordsToF32(timeWordsSubtract(a, b)) + (aFraction - bFraction);
}
`;
