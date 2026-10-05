// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Deterministic xorshift in [0, 1). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** Distance in f32 units in the last place, treating `-0` and `+0` as equal. */
export function getUlpDistance(left: number, right: number): number {
  if (left === right) {
    return 0;
  }
  const toOrdered = (value: number) => {
    const bits = new Int32Array(new Float32Array([value]).buffer)[0];
    return bits < 0 ? -(bits & 0x7fffffff) : bits;
  };
  return Math.abs(toOrdered(left) - toOrdered(right));
}

/** Expects equal NaN patterns and values within `maximumUlp` (0 = bit-exact up to signed zero). */
export function expectFloatArraysClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  maximumUlp: number,
  label: string
): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const gpu = actual[index];
    const cpu = expected[index];
    if (Number.isNaN(cpu) || Number.isNaN(gpu)) {
      expect(Number.isNaN(gpu), `${label}[${index}] nodata (gpu ${gpu}, cpu ${cpu})`).toBe(
        Number.isNaN(cpu)
      );
    } else {
      expect(
        getUlpDistance(gpu, cpu),
        `${label}[${index}] gpu ${gpu} cpu ${cpu}`
      ).toBeLessThanOrEqual(maximumUlp);
    }
  }
}
