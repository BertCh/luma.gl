// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUTimeWindowFilterWordParameterValues,
  getInt64TimeWords,
  GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,
  joinTimeWords,
  splitTimeWords
} from '../../../src/gpu-dataframe/time-window-filter';
import {TIME_WORDS_WGSL} from '../../../src/gpu-dataframe/time-window-filter/time-words';

it('splitTimeWords and joinTimeWords round-trip Int64 values', () => {
  const values = [
    0n,
    1n,
    -1n,
    0xffff_ffffn,
    0x1_0000_0000n,
    0x1_ffff_fff0n,
    1_700_000_000_000n,
    -1_700_000_000_000n,
    -(2n ** 63n),
    2n ** 63n - 1n
  ];
  for (const value of values) {
    const words = splitTimeWords(value);
    expect(words.fraction).toBe(0);
    expect(joinTimeWords(words.low, words.high)).toBe(value);
  }
  expect(splitTimeWords(-1n)).toEqual({
    low: 0xffff_ffff,
    high: 0xffff_ffff,
    fraction: 0
  });
  expect(splitTimeWords(0x2_0000_0001n)).toEqual({
    low: 1,
    high: 2,
    fraction: 0
  });
});

it('splitTimeWords splits numbers into floor and fraction', () => {
  expect(splitTimeWords(1_700_000_000_000.5)).toEqual({
    low: Number(1_700_000_000_000n & 0xffff_ffffn),
    high: Number(1_700_000_000_000n >> 32n),
    fraction: 0.5
  });
  // Negative fractions floor toward -infinity.
  const negative = splitTimeWords(-2.25);
  expect(joinTimeWords(negative.low, negative.high)).toBe(-3n);
  expect(negative.fraction).toBe(0.75);
  // A fraction that rounds to 1 in f32 bumps the integer instead.
  const bumped = splitTimeWords(-1e-9);
  expect(joinTimeWords(bumped.low, bumped.high)).toBe(0n);
  expect(bumped.fraction).toBe(0);
  const wrapped = splitTimeWords(0xffff_ffff + 1 - 1e-9);
  expect(wrapped).toEqual({low: 0, high: 1, fraction: 0});
});

it('splitTimeWords rejects values it cannot represent', () => {
  expect(() => splitTimeWords(NaN)).toThrow();
  expect(() => splitTimeWords(Infinity)).toThrow();
  expect(() => splitTimeWords(2 ** 60)).toThrow(/safe integer/);
  expect(() => splitTimeWords(2n ** 63n)).toThrow(/64-bit/);
  expect(() => splitTimeWords(-(2n ** 63n) - 1n)).toThrow(/64-bit/);
});

it('getInt64TimeWords is a zero-copy view of the same bytes', () => {
  const values = BigInt64Array.from([1n, -2n, 0x1_0000_0003n]);
  const words = getInt64TimeWords(values);
  expect(words.buffer).toBe(values.buffer);
  expect(Array.from(words)).toEqual([1, 0, 0xffff_fffe, 0xffff_ffff, 3, 1]);

  const backing = new BigInt64Array(4);
  backing.set([9n, 7n, 0x2_0000_0000n, 5n]);
  const subarray = backing.subarray(1, 3);
  const subWords = getInt64TimeWords(subarray);
  expect(subWords.buffer).toBe(backing.buffer);
  expect(subWords.byteOffset).toBe(8);
  expect(Array.from(subWords)).toEqual([7, 0, 0, 2]);

  expect(Array.from(getInt64TimeWords(BigUint64Array.from([2n ** 64n - 1n])))).toEqual([
    0xffff_ffff, 0xffff_ffff
  ]);
});

it('getGPUTimeWindowFilterWordParameterValues packs words, fractions, and fades', () => {
  const values = getGPUTimeWindowFilterWordParameterValues({
    start: 0x1_0000_0002n,
    end: 5_000_000_000.5,
    startFadeDuration: 2,
    endFadeDuration: 4
  });
  expect(values).toBeInstanceOf(Uint32Array);
  expect(values.length).toBe(GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH);
  expect(Array.from(values.subarray(0, 4))).toEqual([2, 1, 5_000_000_000 % 2 ** 32, 1]);
  const floats = new Float32Array(values.buffer, values.byteOffset, 8);
  expect(Array.from(floats.subarray(4))).toEqual([0, 0.5, 2, 4]);

  const target = new Uint32Array(10);
  expect(getGPUTimeWindowFilterWordParameterValues({start: 1, end: 2}, target)).toBe(target);
});

it('getGPUTimeWindowFilterWordParameterValues validates its input', () => {
  expect(() =>
    getGPUTimeWindowFilterWordParameterValues({start: 0, end: 1}, new Uint32Array(7))
  ).toThrow(/must hold 8/);
  expect(() =>
    getGPUTimeWindowFilterWordParameterValues({
      start: 0,
      end: 1,
      startFadeDuration: -1
    })
  ).toThrow();
  expect(() =>
    getGPUTimeWindowFilterWordParameterValues({
      start: 0,
      end: 1,
      endFadeDuration: NaN
    })
  ).toThrow();
  expect(() => getGPUTimeWindowFilterWordParameterValues({start: NaN, end: 1})).toThrow();
});

// Review B-07: ordering by the sign of the wrapped difference is wrong when |a - b| >= 2^63, so an
// INT64_MIN/INT64_MAX "unbounded" window rejected pre-1970 rows. The functions below port the WGSL
// statement for statement in uint32 arithmetic (Math.imul-free: only xor, compare and borrow
// subtraction), and are checked against BigInt on every pair of the extreme values. The GPU spec in
// gpu-time-window-filter-int64.spec.ts runs the real kernel with the same window edges.
type Words = [low: number, high: number];

function toWords(value: bigint): Words {
  const {low, high} = splitTimeWords(value);
  return [low, high];
}

function subtractWords(a: Words, b: Words): Words {
  const borrow = a[0] < b[0] ? 1 : 0;
  return [(a[0] - b[0]) >>> 0, (a[1] - b[1] - borrow) >>> 0];
}

function compareWords(a: Words, b: Words): number {
  const aHigh = (a[1] ^ 0x80000000) >>> 0;
  const bHigh = (b[1] ^ 0x80000000) >>> 0;
  if (aHigh !== bHigh) {
    return aHigh > bHigh ? 1 : -1;
  }
  if (a[0] !== b[0]) {
    return a[0] > b[0] ? 1 : -1;
  }
  return 0;
}

function differenceToF32(a: Words, b: Words): number {
  const isForward = compareWords(a, b) >= 0;
  const magnitude = isForward ? subtractWords(a, b) : subtractWords(b, a);
  const value = Math.fround(Math.fround(magnitude[1]) * 4294967296 + Math.fround(magnitude[0]));
  return isForward ? value : -value;
}

/** The pre-fix rule: `!timeWordsIsNegative(a - b)`. */
function isAtLeastBySign(a: Words, b: Words): boolean {
  return (subtractWords(a, b)[1] & 0x80000000) === 0;
}

const EXTREMES = [
  -(2n ** 63n),
  -(2n ** 63n) + 1n,
  -(2n ** 62n),
  -1_700_000_000_000n,
  -1n,
  0n,
  1n,
  1_700_000_000_000n,
  2n ** 62n,
  2n ** 63n - 2n,
  2n ** 63n - 1n
];

it('TIME_WORDS_WGSL orders Int64 words lexicographically, not by a wrapped difference', () => {
  expect(TIME_WORDS_WGSL).toMatch(/fn timeWordsCompare\(a: vec2<u32>, b: vec2<u32>\) -> i32/);
  const atLeast = TIME_WORDS_WGSL.slice(TIME_WORDS_WGSL.indexOf('fn isTimeWordsAtLeast'));
  expect(atLeast.slice(0, atLeast.indexOf('\n}'))).toContain('timeWordsCompare(a, b)');
  const difference = TIME_WORDS_WGSL.slice(TIME_WORDS_WGSL.indexOf('fn timeWordsDifference('));
  expect(difference.slice(0, difference.indexOf('\n}'))).toContain('timeWordsDifferenceToF32');
});

it('time-word comparison is exact for every pair of extreme Int64 values', () => {
  let signFailures = 0;
  for (const a of EXTREMES) {
    for (const b of EXTREMES) {
      const expected = a > b ? 1 : a < b ? -1 : 0;
      expect(compareWords(toWords(a), toWords(b))).toBe(expected);
      if (isAtLeastBySign(toWords(a), toWords(b)) !== a >= b) {
        signFailures++;
      }
      // Correct sign and f32 rounding of the exact difference (two roundings, as in WGSL).
      const exact = Number(a - b);
      const actual = differenceToF32(toWords(a), toWords(b));
      expect(Math.sign(actual)).toBe(Math.sign(exact));
      expect(Math.abs(actual - exact)).toBeLessThanOrEqual(Math.abs(exact) * 2 ** -22);
    }
  }
  // The review's cases under the old rule: atLeast(MAX, -1), atLeast(0, MIN), atLeast(MAX, MIN)
  // were all false.
  const maximum = toWords(2n ** 63n - 1n);
  const minimum = toWords(-(2n ** 63n));
  expect(isAtLeastBySign(maximum, toWords(-1n))).toBe(false);
  expect(isAtLeastBySign(toWords(0n), minimum)).toBe(false);
  expect(isAtLeastBySign(maximum, minimum)).toBe(false);
  expect(signFailures).toBeGreaterThan(0);
  expect(compareWords(maximum, toWords(-1n))).toBe(1);
  expect(compareWords(toWords(0n), minimum)).toBe(1);
  expect(compareWords(maximum, minimum)).toBe(1);
});
