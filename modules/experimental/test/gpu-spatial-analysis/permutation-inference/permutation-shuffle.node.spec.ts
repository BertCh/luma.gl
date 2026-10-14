// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  fillShufflePermutation,
  getSwapOrNotPermutationIndex,
  getSwapOrNotRoundCount,
  getSwapOrNotRounds,
  PERMUTATION_SHUFFLE_MAXIMUM_ROUNDS
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-shuffle';
import {
  getFeistelPermutationIndex,
  getFeistelRoundKeys
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';
import {createWeights, type CPUSpatialWeights} from './permutation-oracle';

describe('swap-or-not shuffle', () => {
  it('uses 6 ceil(log2 n) + 6 rounds and fits the WGSL storage', () => {
    expect(getSwapOrNotRoundCount(1)).toBe(0);
    expect(getSwapOrNotRoundCount(2)).toBe(12);
    expect(getSwapOrNotRoundCount(100)).toBe(6 * 7 + 6);
    expect(getSwapOrNotRoundCount(1024)).toBe(6 * 10 + 6);
    expect(getSwapOrNotRoundCount(1025)).toBe(6 * 11 + 6);
    // The permutation tests accept fewer than 2^24 rows.
    expect(getSwapOrNotRoundCount(2 ** 24 - 1)).toBeLessThanOrEqual(
      PERMUTATION_SHUFFLE_MAXIMUM_ROUNDS
    );
  });

  it('is a bijection of [0, n), deterministic, and differs per permutation and key', () => {
    for (const count of [1, 2, 3, 4, 5, 17, 64, 65, 1000, 4097]) {
      const output = new Uint32Array(count);
      fillShufflePermutation(output, count, [11, 0], 3);
      expect(new Set(output).size).toBe(count);
      expect(Math.max(...output)).toBe(count - 1);
      const again = new Uint32Array(count);
      fillShufflePermutation(again, count, [11, 0], 3);
      expect(again).toEqual(output);
    }
    const a = new Uint32Array(50);
    const b = new Uint32Array(50);
    const c = new Uint32Array(50);
    fillShufflePermutation(a, 50, [1, 0], 1);
    fillShufflePermutation(b, 50, [1, 0], 2);
    fillShufflePermutation(c, 50, [2, 0], 1);
    expect(a).not.toEqual(b);
    expect(a).not.toEqual(c);
    const rounds = getSwapOrNotRounds([1, 0], 1, 50);
    expect(getSwapOrNotPermutationIndex(7, 50, rounds)).toBe(a[7]);
  });
});

// Oracle: fix/work/A/oracle-u2-vi-rand-pin.py (esda 2.10.0 / libpysal 4.15.0): Moran(x, lat2W(10, 10),
// transformation='O') for x_i = (((i * 37) % 100) + 1) ** 2 / 100.
const ESDA_EXPECTED_I = -0.010101010101010102;
const ESDA_VARIANCE_RANDOMIZATION = 0.0053893219325306324;
// scipy.stats.chi2.isf(1e-3, df): the 0.1 percent critical values used below.
const CHI_SQUARE_CRITICAL_DF9 = 27.87716487125657;
const CHI_SQUARE_CRITICAL_DF99 = 148.23035916510173;

function createRookLattice(side: number): CPUSpatialWeights {
  const lists: [number, number][][] = [];
  for (let row = 0; row < side; row++) {
    for (let column = 0; column < side; column++) {
      const neighbors: number[] = [];
      if (row > 0) neighbors.push((row - 1) * side + column);
      if (row < side - 1) neighbors.push((row + 1) * side + column);
      if (column > 0) neighbors.push(row * side + column - 1);
      if (column < side - 1) neighbors.push(row * side + column + 1);
      lists.push(neighbors.map((neighbor): [number, number] => [neighbor, 1]));
    }
  }
  return createWeights(lists);
}

/** Randomization moments of Moran's I (Cliff and Ord; esda `VI_rand`) for binary-or-general weights. */
function getRandomizationMoments(weights: CPUSpatialWeights, z: Float64Array) {
  const n = z.length;
  const entries = new Map<number, number>();
  const rowSums = new Float64Array(n);
  const columnSums = new Float64Array(n);
  let s0 = 0;
  for (let i = 0; i < n; i++) {
    for (let slot = weights.offsets[i]; slot < weights.offsets[i + 1]; slot++) {
      const j = weights.neighbors[slot];
      const weight = weights.weights[slot];
      s0 += weight;
      rowSums[i] += weight;
      columnSums[j] += weight;
      entries.set(i * n + j, weight);
    }
  }
  let s1 = 0;
  for (const [key, weight] of entries) {
    const i = Math.floor(key / n);
    const j = key % n;
    const reverse = entries.get(j * n + i);
    // Each ordered pair contributes (w_ij + w_ji)^2 / 2; a pair listed once contributes w^2 / 2 from each side.
    s1 += 0.5 * (weight + (reverse ?? 0)) ** 2;
    if (reverse === undefined) {
      s1 += 0.5 * weight * weight;
    }
  }
  let s2 = 0;
  for (let i = 0; i < n; i++) {
    s2 += (rowSums[i] + columnSums[i]) ** 2;
  }
  let m2 = 0;
  let m4 = 0;
  for (const value of z) {
    m2 += value * value;
    m4 += value ** 4;
  }
  const kurtosis = (n * m4) / (m2 * m2);
  const expected = -1 / (n - 1);
  const a = n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0 * s0);
  const b = kurtosis * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0 * s0);
  const variance = (a - b) / ((n - 1) * (n - 2) * (n - 3) * s0 * s0) - expected * expected;
  return {s0, sumSquares: m2, expected, variance};
}

describe('global shuffle uniformity (swap-or-not)', () => {
  const side = 10;
  const n = side * side;
  const weights = createRookLattice(side);
  const x = Float64Array.from({length: n}, (_, i) => (((i * 37) % 100) + 1) ** 2 / 100);
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const z = x.map(value => value - mean);

  it('reproduces the esda randomization moments of Moran I (oracle pin)', () => {
    const moments = getRandomizationMoments(weights, z);
    expect(moments.expected).toBeCloseTo(ESDA_EXPECTED_I, 14);
    expect(moments.variance).toBeCloseTo(ESDA_VARIANCE_RANDOMIZATION, 12);
  });

  // The simulated mean and variance of Moran I under R keyed shuffles must match the exact
  // randomization moments. With R = 4000 independent draws the variance ratio has standard error
  // sqrt((kurtosis - 1) / R) (about sqrt(2 / R) = 0.022, using the sample kurtosis of I); the
  // mean has standard error sqrt(variance / R). Both are checked at 4 standard errors, and the
  // joint placement of (pi(0), pi(1)), a rook neighbor pair, with a chi-square test at the 0.1
  // percent level. The seed is fixed, so the outcome is deterministic.
  it('matches the randomization moments and places a neighbor pair uniformly', () => {
    const permutations = 4000;
    const moments = getRandomizationMoments(weights, z);
    const scale = n / (moments.s0 * moments.sumSquares);
    const permutation = new Uint32Array(n);
    const bins = 10;
    const marginal = new Float64Array(bins);
    const joint = new Float64Array(bins * bins);
    const statistics: number[] = [];
    for (let p = 1; p <= permutations; p++) {
      fillShufflePermutation(permutation, n, [20240601, 0], p);
      let sum = 0;
      for (let i = 0; i < n; i++) {
        let lag = 0;
        for (let slot = weights.offsets[i]; slot < weights.offsets[i + 1]; slot++) {
          lag += weights.weights[slot] * z[permutation[weights.neighbors[slot]]];
        }
        sum += z[permutation[i]] * lag;
      }
      statistics.push(scale * sum);
      const binFirst = Math.floor((permutation[0] * bins) / n);
      const binSecond = Math.floor((permutation[1] * bins) / n);
      marginal[binFirst]++;
      joint[binFirst * bins + binSecond]++;
    }
    const simulatedMean = statistics.reduce((a, b) => a + b, 0) / permutations;
    const deviations = statistics.map(value => value - simulatedMean);
    const variance = deviations.reduce((a, b) => a + b * b, 0) / permutations;
    const kurtosis =
      deviations.reduce((a, b) => a + b ** 4, 0) / permutations / (variance * variance);
    const varianceStandardError = Math.sqrt((kurtosis - 1) / permutations);
    const meanStandardError = Math.sqrt(moments.variance / permutations);
    expect(Math.abs(variance / moments.variance - 1)).toBeLessThan(4 * varianceStandardError);
    expect(Math.abs(simulatedMean - moments.expected)).toBeLessThan(4 * meanStandardError);

    // Ten bins of ten positions: P(pi(0) in a) = 1/10; P(pi(0) in a, pi(1) in b) = (100 - 10 [a == b]) / (100 * 99).
    let marginalChi = 0;
    for (const count of marginal) {
      marginalChi += (count - permutations / bins) ** 2 / (permutations / bins);
    }
    expect(marginalChi).toBeLessThan(CHI_SQUARE_CRITICAL_DF9);
    let jointChi = 0;
    for (let a = 0; a < bins; a++) {
      for (let b = 0; b < bins; b++) {
        const expected = ((100 - (a === b ? 10 : 0)) / (n * (n - 1))) * permutations;
        jointChi += (joint[a * bins + b] - expected) ** 2 / expected;
      }
    }
    expect(jointChi).toBeLessThan(CHI_SQUARE_CRITICAL_DF99);
  });

  // Why the four-round Feistel network was replaced: on a ten-element domain, pi(0) is visibly
  // non-uniform with it (chi-square about 650 at 400000 draws, df 9; fix/work/A/u2/joint-feistel.json
  // shows the same for domains up to 50 and the joint placement of pairs), while the swap-or-not
  // shuffle passes at the 0.1 percent level (fix/work/A/u2/joint-swap.json).
  it('is uniform on a small domain where the Feistel network is not', () => {
    const count = 10;
    const draws = 400000;
    const feistelCounts = new Float64Array(count);
    const swapCounts = new Float64Array(count);
    const permutation = new Uint32Array(count);
    for (let p = 1; p <= draws; p++) {
      const keys = getFeistelRoundKeys([1, 0], p);
      feistelCounts[getFeistelPermutationIndex(0, count, keys)]++;
      fillShufflePermutation(permutation, count, [1, 0], p);
      swapCounts[permutation[0]]++;
    }
    const chiSquare = (counts: Float64Array) =>
      counts.reduce((total, value) => total + (value - draws / count) ** 2 / (draws / count), 0);
    expect(chiSquare(feistelCounts)).toBeGreaterThan(CHI_SQUARE_CRITICAL_DF9);
    expect(chiSquare(swapCounts)).toBeLessThan(CHI_SQUARE_CRITICAL_DF9);
  });
});
