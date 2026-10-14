// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';

const U32 = 2 ** 32;

/** CPU twin of the spike-repair WGSL `getWideProduct` (u32 arithmetic, factor below 2^16). */
function getWideProduct(value: number, factor: number): [number, number] {
  const low = (value % 65536) * factor;
  const high = Math.floor(value / 65536) * factor;
  const lowWord = (low + (high % 65536) * 65536) % U32;
  const carry = lowWord < low ? 1 : 0;
  return [lowWord, Math.floor(high / 65536) + carry];
}

function toBigInt([low, high]: [number, number]): bigint {
  return (BigInt(high) << 32n) + BigInt(low);
}

it('review C-15: the spike-repair percentage tests are exact 64-bit products', () => {
  const values = [0, 1, 99, 65535, 65536, 42_949_673, 42_949_672, 2 ** 31, U32 - 1];
  for (const value of values) {
    for (const factor of [0, 1, 51, 80, 100, 65535]) {
      expect(toBigInt(getWideProduct(value, factor)), `${value} * ${factor}`).toBe(
        BigInt(value) * BigInt(factor)
      );
    }
  }
  // The old u32 test `size * 100 <= validCount * 10` wrapped: a 50M-pixel component of a 100M
  // valid grid (50%) passed a 10% cap because 5e9 mod 2^32 = 705032704 < 1e9.
  const size = 50_000_000;
  const validCount = 100_000_000;
  expect((size * 100) % U32 <= (validCount * 10) % U32).toBe(true);
  expect(toBigInt(getWideProduct(size, 100)) <= toBigInt(getWideProduct(validCount, 10))).toBe(
    false
  );
});
