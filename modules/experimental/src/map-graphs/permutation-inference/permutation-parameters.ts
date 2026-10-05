// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of uint32 elements in a permutation-test parameter view. */
export const GPU_PERMUTATION_PARAMETER_LENGTH = 4;

/** Per-frame parameters of `GPULocalPermutationTest` and `GPUGlobalPermutationTest`. */
export type GPUPermutationParameters = {
  /**
   * Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and
   * `permutations`, independent of the dispatch shape.
   */
  seed: number;
  /** Number of permutations `P`, at most the recipe's `maximumPermutations`. */
  permutations: number;
  /** Significance level for the local `significant` mask. Defaults to 0.05. Ignored by the global test. */
  significanceLevel?: number;
};

/**
 * Packs {@link GPUPermutationParameters} into the uint32 layout read by the kernels:
 * `[seedLow, seedHigh, permutations, float32 bits of significanceLevel]`.
 */
export function getGPUPermutationParameterValues(
  parameters: GPUPermutationParameters
): Uint32Array {
  const {seed, permutations} = parameters;
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new Error('permutation seed must be a non-negative safe integer');
  }
  if (!Number.isInteger(permutations) || permutations < 1 || permutations >= 2 ** 31) {
    throw new Error('permutations must be a positive integer below 2^31');
  }
  const values = new Uint32Array(GPU_PERMUTATION_PARAMETER_LENGTH);
  values[0] = seed % 2 ** 32;
  values[1] = Math.floor(seed / 2 ** 32);
  values[2] = permutations;
  values[3] = new Uint32Array(new Float32Array([parameters.significanceLevel ?? 0.05]).buffer)[0];
  return values;
}

/** Splits a seed into the two Philox key words, as the kernels do. @internal */
export function getPermutationSeedKey(seed: number): [number, number] {
  return [seed % 2 ** 32, Math.floor(seed / 2 ** 32)];
}
