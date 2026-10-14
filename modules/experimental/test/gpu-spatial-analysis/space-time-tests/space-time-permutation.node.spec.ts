// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getPermutationSeedKey} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {fillShufflePermutation} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-shuffle';
import {
  getSpaceTimeGroupsPerSlot,
  getSpaceTimeShuffleWGSL,
  SPACE_TIME_SLOT_PROLOGUE_WGSL,
  SPACE_TIME_WORKGROUP_SIZE
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-kernels';
import {getPermutedTimes} from './space-time-oracle';

// The Knox and Mantel oracle (and so the GPU kernels it must equal bit for bit) permute times with
// the swap-or-not shuffle: permuted time of row r under slot s is times[fillShufflePermutation(s)[r]].
describe('space-time permutations use the swap-or-not shuffle', () => {
  it('getPermutedTimes equals fillShufflePermutation applied to the times', () => {
    const times = Float32Array.from({length: 37}, (_, row) => row * 1.5 + 0.25);
    const seed = 2 ** 33 + 12345;
    for (const slot of [1, 2, 9]) {
      const permutation = new Uint32Array(times.length);
      fillShufflePermutation(permutation, times.length, getPermutationSeedKey(seed), slot);
      expect(Array.from(getPermutedTimes(times, slot, seed))).toEqual(
        Array.from(permutation, index => times[index])
      );
    }
    expect(getPermutedTimes(times, 0, seed)).toBe(times);
  });

  // Uniformity at n = 4: each of the 4 positions receives each source with probability 1/4. The
  // chi-square critical value 27.877 is scipy.stats.chi2.isf(1e-3, 9) (9 dof for the 4 x 4 table).
  it('places each source uniformly over the destinations on a small domain', () => {
    const count = 4;
    const draws = 40000;
    const table = new Float64Array(count * count);
    const permutation = new Uint32Array(count);
    for (let slot = 1; slot <= draws; slot++) {
      fillShufflePermutation(permutation, count, getPermutationSeedKey(7), slot);
      for (let row = 0; row < count; row++) table[row * count + permutation[row]]++;
    }
    const expected = draws / count;
    let chiSquare = 0;
    for (const observed of table) chiSquare += (observed - expected) ** 2 / expected;
    expect(chiSquare).toBeLessThan(27.87716487125657);
  });

  it('assigns one permutation per workgroup and uses the shared shuffle WGSL', () => {
    expect(getSpaceTimeGroupsPerSlot(256)).toBe(256 / SPACE_TIME_WORKGROUP_SIZE);
    expect(getSpaceTimeGroupsPerSlot(1)).toBe(1);
    const wgsl = getSpaceTimeShuffleWGSL(256);
    expect(wgsl).toContain('getSwapOrNotPermutationIndex');
    expect(wgsl).not.toContain('Feistel');
    expect(SPACE_TIME_SLOT_PROLOGUE_WGSL).toContain('workgroupBarrier()');
  });
});
