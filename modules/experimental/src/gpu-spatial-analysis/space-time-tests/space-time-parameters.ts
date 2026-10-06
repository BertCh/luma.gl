// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of uint32 elements in a space-time test parameter view. */
export const GPU_SPACE_TIME_PARAMETER_LENGTH = 4;

/** Number of float32 elements in the `summary` output of the space-time tests. */
export const GPU_SPACE_TIME_SUMMARY_LENGTH = 12;

/**
 * Indices of the `summary` output of `GPUKnoxTest` and `GPUMantelTest`. A summary reports the
 * statistic of the observed times and the Monte Carlo reference distribution of the `P`
 * Feistel-permuted time assignments.
 */
export const GPU_SPACE_TIME_SUMMARY = {
  /** Observed statistic: the Knox count, or Mantel's `r`. */
  observed: 0,
  /** Spatial pairs `i < j` of the pair list (the pairs the statistic runs over). */
  pairCount: 1,
  /** Knox: all pairs of rows within the time threshold of each other (0 for Mantel). */
  timeClosePairs: 2,
  /** Knox: `pairCount * timeClosePairs / (n (n - 1) / 2)`, the permutation mean (0 for Mantel). */
  expected: 3,
  /** Mean of the permuted statistics. */
  permutationMean: 4,
  /** Sample variance (`P - 1` denominator) of the permuted statistics. */
  permutationVariance: 5,
  /** Permutations with statistic `>=` observed. */
  greaterCount: 6,
  /** Permutations with statistic `<=` observed. */
  lesserCount: 7,
  /** `(greaterCount + 1) / (P + 1)`: pseudo p-value for positive space-time interaction. */
  pseudoPGreater: 8,
  /** `(lesserCount + 1) / (P + 1)`. */
  pseudoPLesser: 9,
  /** `min(1, 2 min(pseudoPGreater, pseudoPLesser))`. */
  pseudoPTwoSided: 10,
  /** `(observed - permutationMean) / sqrt(permutationVariance)`, NaN when the variance is 0. */
  zScore: 11
} as const;

/** Per-frame parameters of `GPUKnoxTest` and `GPUMantelTest`. */
export type GPUSpaceTimeParameters = {
  /** Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. */
  seed: number;
  /** Number of time permutations `P`, at most the contributor's `maximumPermutations`. */
  permutations: number;
  /**
   * Knox only: two rows are close in time when `|t_i - t_j| <= timeThreshold` (f32). Ignored by
   * `GPUMantelTest`.
   */
  timeThreshold?: number;
};

/**
 * Packs {@link GPUSpaceTimeParameters} into the uint32 layout read by the kernels:
 * `[seedLow, seedHigh, permutations, float32 bits of timeThreshold]`.
 */
export function getGPUSpaceTimeParameterValues(parameters: GPUSpaceTimeParameters): Uint32Array {
  const {seed, permutations} = parameters;
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new Error('space-time seed must be a non-negative safe integer');
  }
  if (!Number.isInteger(permutations) || permutations < 1 || permutations >= 2 ** 31) {
    throw new Error('space-time permutations must be a positive integer below 2^31');
  }
  const timeThreshold = parameters.timeThreshold ?? 0;
  if (Number.isNaN(timeThreshold) || timeThreshold < 0) {
    throw new Error('space-time timeThreshold must be non-negative');
  }
  const values = new Uint32Array(GPU_SPACE_TIME_PARAMETER_LENGTH);
  values[0] = seed % 2 ** 32;
  values[1] = Math.floor(seed / 2 ** 32);
  values[2] = permutations;
  values[3] = new Uint32Array(new Float32Array([timeThreshold]).buffer)[0];
  return values;
}

/**
 * Upper-tail p-value `P(X >= observed)` of a Poisson distribution with mean `expected`, the
 * classic Knox (1964) test. CPU helper for the `observed` and `expected` of `GPUKnoxTest`'s
 * `summary`. Sums the probability mass function in double precision.
 *
 * @param observed Observed Knox count.
 * @param expected Expected count under no interaction.
 */
export function getKnoxPoissonPValue(observed: number, expected: number): number {
  if (observed <= 0) {
    return 1;
  }
  if (!(expected > 0)) {
    return 0;
  }
  // 1 - P(X <= observed - 1), summed in log space from the mode for stability.
  let logTerm = -expected;
  let lowerTail = 0;
  for (let count = 0; count < observed; count++) {
    lowerTail += Math.exp(logTerm);
    logTerm += Math.log(expected) - Math.log(count + 1);
  }
  return Math.max(0, Math.min(1, 1 - lowerTail));
}
