// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fillShufflePermutation} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-shuffle';
import {getPermutationSeedKey} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';

/** Pair list: a CSR with optional distances; each unordered pair counts once if listed in either direction. */
export type OraclePairs = {
  offsets: ArrayLike<number>;
  neighbors: ArrayLike<number>;
  distances?: ArrayLike<number>;
};

/**
 * Unique unordered pairs of a CSR in either direction, independent of the GPU mask kernel: rows
 * ascend, so a pair listed from its lower row takes that entry's distance, otherwise the higher
 * row's. Self entries, out-of-range IDs and duplicates are dropped.
 */
export function getUniquePairs(pairs: OraclePairs, rows: number) {
  const seen = new Set<number>();
  const first: number[] = [];
  const second: number[] = [];
  const distances: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let entry = pairs.offsets[row]; entry < pairs.offsets[row + 1]; entry++) {
      const other = pairs.neighbors[entry];
      if (other === row || other >= rows) continue;
      const key = Math.min(row, other) * rows + Math.max(row, other);
      if (seen.has(key)) continue;
      seen.add(key);
      first.push(row);
      second.push(other);
      if (pairs.distances) distances.push(pairs.distances[entry]);
    }
  }
  return {first, second, distances};
}

/**
 * Returns the time of every row under permutation `slot` (0 is identity, `s >= 1` is draw `s` of
 * the swap-or-not shuffle, the CPU twin of the GPU kernels).
 */
export function getPermutedTimes(times: Float32Array, slot: number, seed: number): Float32Array {
  if (slot === 0) return times;
  const permutation = new Uint32Array(times.length);
  fillShufflePermutation(permutation, times.length, getPermutationSeedKey(seed), slot);
  return Float32Array.from(times, (_, row) => times[permutation[row]]);
}

/** Brute-force Knox counts for the identity and every permutation. */
export function computeKnoxOracle(
  pairs: OraclePairs,
  times: Float32Array,
  seed: number,
  permutations: number,
  threshold: number
) {
  const rows = times.length;
  const statistics: number[] = [];
  const unique = getUniquePairs(pairs, rows);
  const pairCount = unique.first.length;
  for (let slot = 0; slot <= permutations; slot++) {
    const permuted = getPermutedTimes(times, slot, seed);
    let count = 0;
    for (let index = 0; index < pairCount; index++) {
      const difference = Math.fround(
        permuted[unique.first[index]] - permuted[unique.second[index]]
      );
      if (Math.abs(difference) <= threshold) count++;
    }
    statistics.push(count);
  }
  let timeClose = 0;
  for (let first = 0; first < rows; first++) {
    for (let second = first + 1; second < rows; second++) {
      if (Math.abs(Math.fround(times[first] - times[second])) <= threshold) timeClose++;
    }
  }
  const expected = (pairCount * timeClose) / ((rows * (rows - 1)) / 2);
  const permuted = statistics.slice(1);
  const greater = permuted.filter(value => value >= statistics[0]).length;
  const lesser = permuted.filter(value => value <= statistics[0]).length;
  return {statistics, pairCount, timeClose, expected, greater, lesser};
}

/** Brute-force Mantel correlation (double precision) for the identity and every permutation. */
export function computeMantelOracle(
  pairs: OraclePairs,
  times: Float32Array,
  seed: number,
  permutations: number
): number[] {
  const rows = times.length;
  const results: number[] = [];
  const unique = getUniquePairs(pairs, rows);
  for (let slot = 0; slot <= permutations; slot++) {
    const permuted = getPermutedTimes(times, slot, seed);
    const xs = unique.distances;
    const ys = unique.first.map((row, index) =>
      Math.abs(permuted[row] - permuted[unique.second[index]])
    );
    const mean = (values: number[]) =>
      values.reduce((sum, value) => sum + value, 0) / values.length;
    const meanX = mean(xs);
    const meanY = mean(ys);
    let covariance = 0;
    let varianceX = 0;
    let varianceY = 0;
    for (let index = 0; index < xs.length; index++) {
      covariance += (xs[index] - meanX) * (ys[index] - meanY);
      varianceX += (xs[index] - meanX) ** 2;
      varianceY += (ys[index] - meanY) ** 2;
    }
    results.push(covariance / Math.sqrt(varianceX * varianceY));
  }
  return results;
}
