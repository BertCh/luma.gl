// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUTimeWindowWordParameterValues,
  getInt64TimeWords,
  GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH,
  joinTimeWords,
  splitTimeWords
} from '../../../src/map-graphs/time-window-filter';

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

it('getGPUTimeWindowWordParameterValues packs words, fractions, and fades', () => {
  const values = getGPUTimeWindowWordParameterValues({
    start: 0x1_0000_0002n,
    end: 5_000_000_000.5,
    startFadeDuration: 2,
    endFadeDuration: 4
  });
  expect(values).toBeInstanceOf(Uint32Array);
  expect(values.length).toBe(GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH);
  expect(Array.from(values.subarray(0, 4))).toEqual([2, 1, 5_000_000_000 % 2 ** 32, 1]);
  const floats = new Float32Array(values.buffer, values.byteOffset, 8);
  expect(Array.from(floats.subarray(4))).toEqual([0, 0.5, 2, 4]);

  const target = new Uint32Array(10);
  expect(getGPUTimeWindowWordParameterValues({start: 1, end: 2}, target)).toBe(target);
});

it('getGPUTimeWindowWordParameterValues validates its input', () => {
  expect(() => getGPUTimeWindowWordParameterValues({start: 0, end: 1}, new Uint32Array(7))).toThrow(
    /must hold 8/
  );
  expect(() =>
    getGPUTimeWindowWordParameterValues({
      start: 0,
      end: 1,
      startFadeDuration: -1
    })
  ).toThrow();
  expect(() =>
    getGPUTimeWindowWordParameterValues({
      start: 0,
      end: 1,
      endFadeDuration: NaN
    })
  ).toThrow();
  expect(() => getGPUTimeWindowWordParameterValues({start: NaN, end: 1})).toThrow();
});
