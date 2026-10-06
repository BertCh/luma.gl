// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getFeistelPermutationIndex,
  getFeistelRoundKeys
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';
import {getPermutationSeedKey} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';

/** Pair list: entries `j > i` of a CSR with optional distances. */
export type OraclePairs = {
  offsets: ArrayLike<number>;
  neighbors: ArrayLike<number>;
  distances?: ArrayLike<number>;
};

/** Returns the time of every row under permutation `slot` (0 is identity, `s >= 1` is permutation `s - 1`). */
export function getPermutedTimes(times: Float32Array, slot: number, seed: number): Float32Array {
  if (slot === 0) return times;
  const keys = getFeistelRoundKeys(getPermutationSeedKey(seed), slot - 1);
  return Float32Array.from(
    times,
    (_, row) => times[getFeistelPermutationIndex(row, times.length, keys)]
  );
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
  let pairCount = 0;
  for (let slot = 0; slot <= permutations; slot++) {
    const permuted = getPermutedTimes(times, slot, seed);
    let count = 0;
    let spatial = 0;
    for (let row = 0; row < rows; row++) {
      for (let entry = pairs.offsets[row]; entry < pairs.offsets[row + 1]; entry++) {
        const other = pairs.neighbors[entry];
        if (other > row) {
          spatial++;
          if (Math.abs(Math.fround(permuted[row] - permuted[other])) <= threshold) count++;
        }
      }
    }
    statistics.push(count);
    pairCount = spatial;
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
  for (let slot = 0; slot <= permutations; slot++) {
    const permuted = getPermutedTimes(times, slot, seed);
    const xs: number[] = [];
    const ys: number[] = [];
    for (let row = 0; row < rows; row++) {
      for (let entry = pairs.offsets[row]; entry < pairs.offsets[row + 1]; entry++) {
        const other = pairs.neighbors[entry];
        if (other > row) {
          xs.push(pairs.distances![entry]);
          ys.push(Math.abs(permuted[row] - permuted[other]));
        }
      }
    }
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
