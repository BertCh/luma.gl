// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUInferenceAlternative,
  GPUMultipleTestingMethod,
  GPUPermutationMetadata
} from '../contracts/index';
import {resolvePermutationAlternative} from './permutation-alternative';

/** Number of uint32 elements in a permutation-test parameter view. */
export const GPU_PERMUTATION_PARAMETER_LENGTH = 4;

/** Per-frame parameters of `GPULocalPermutationTest` and `GPUGlobalPermutationTest`. */
export type GPUPermutationParameters = {
  /**
   * Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and
   * `permutations`, independent of the dispatch shape.
   */
  seed: number;
  /**
   * Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is
   * never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are
   * not tested and `overflow` is set; global: `results.permutations` is NaN). Validate on the CPU
   * with the contributor's `getParameterValues` or the `maximumPermutations` option of
   * {@link getGPUPermutationParameterValues}.
   */
  permutations: number;
  /** Significance level for the local `significant` mask. Defaults to 0.05. Ignored by the global test. */
  significanceLevel?: number;
};

/**
 * Packs {@link GPUPermutationParameters} into the uint32 layout read by the kernels:
 * `[seedLow, seedHigh, permutations, float32 bits of significanceLevel]`.
 *
 * @param parameters Seed, permutation count and significance level.
 * @param options.maximumPermutations The contributor's compile-time maximum. When given, a larger
 * `permutations` throws; the kernels never clamp it.
 */
export function getGPUPermutationParameterValues(
  parameters: GPUPermutationParameters,
  options: {maximumPermutations?: number} = {}
): Uint32Array {
  const {seed, permutations} = parameters;
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new Error('permutation seed must be a non-negative safe integer');
  }
  if (!Number.isInteger(permutations) || permutations < 1 || permutations >= 2 ** 31) {
    throw new Error('permutations must be a positive integer below 2^31');
  }
  if (options.maximumPermutations !== undefined && permutations > options.maximumPermutations) {
    throw new Error(
      `permutations (${permutations}) exceeds maximumPermutations (${options.maximumPermutations})`
    );
  }
  const level = parameters.significanceLevel ?? 0.05;
  if (!(level > 0 && level < 1)) {
    throw new Error('significanceLevel must be in (0, 1)');
  }
  const values = new Uint32Array(GPU_PERMUTATION_PARAMETER_LENGTH);
  values[0] = seed % 2 ** 32;
  values[1] = Math.floor(seed / 2 ** 32);
  values[2] = permutations;
  values[3] = new Uint32Array(new Float32Array([level]).buffer)[0];
  return values;
}

/**
 * Returns the serializable metadata shared by every permutation contributor. The default
 * `alternative` is `'two-sided'` (`'less'` is reported as `'lesser'`).
 */
export function getGPUPermutationMetadata(
  parameters: GPUPermutationParameters,
  options: {
    alternative?: GPUInferenceAlternative;
    multipleTesting?: GPUMultipleTestingMethod;
    /** When given, `permutations` above it throws instead of being reported as run. */
    maximumPermutations?: number;
  } = {}
): GPUPermutationMetadata {
  // Packing performs the canonical seed and count validation.
  getGPUPermutationParameterValues(parameters, {maximumPermutations: options.maximumPermutations});
  return {
    seed: parameters.seed,
    permutationCount: parameters.permutations,
    alternative: resolvePermutationAlternative('getGPUPermutationMetadata', options.alternative),
    multipleTesting: options.multipleTesting ?? 'none',
    includeObserved: true
  };
}

/** Splits a seed into the two Philox key words, as the kernels do. @internal */
export function getPermutationSeedKey(seed: number): [number, number] {
  return [seed % 2 ** 32, Math.floor(seed / 2 ** 32)];
}
