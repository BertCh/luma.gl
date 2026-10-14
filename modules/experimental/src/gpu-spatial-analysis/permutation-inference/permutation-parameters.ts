// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUInferenceAlternative,
  GPUMultipleTestingMethod,
  GPUPermutationMetadata
} from '../../analysis-core/contracts/index';
import {validateCapacity} from '../../utils/gpu-contributor-utils';
import {resolvePermutationAlternative} from './permutation-alternative';

/** Number of uint32 elements in a permutation-test parameter view. */
export const GPU_PERMUTATION_PARAMETER_LENGTH = 4;

/**
 * Per-frame settings of `GPULocalPermutationTest` and `GPUGlobalPermutationTest`, packed into their
 * `parameters` view by {@link getGPUPermutationParameterValues} and described by their
 * `getInferenceMetadata(settings)`.
 */
export type GPUPermutationSettings = {
  /**
   * Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and
   * `permutationCount`, independent of the dispatch shape. Per frame.
   */
  seed: number;
  /**
   * Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is
   * never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are
   * not tested and `output.status.overflow` is set; global: the `permutationCount` field of
   * `output.summary` is NaN). Validate on the CPU with the contributor's `getParameterValues` or
   * the `maximumPermutations` option of {@link getGPUPermutationParameterValues}.
   */
  permutationCount: number;
  /** Significance level for the local `significant` mask. Defaults to 0.05. Ignored by the global test. */
  significanceLevel?: number;
};

/**
 * Packs {@link GPUPermutationSettings} into the uint32 layout read by the kernels:
 * `[seedLow, seedHigh, permutationCount, float32 bits of significanceLevel]`.
 *
 * @param settings Seed, permutation count and significance level.
 * @param options.maximumPermutations The contributor's compile-time maximum. When given, a larger
 * `permutationCount` throws; the kernels never clamp it.
 */
export function getGPUPermutationParameterValues(
  settings: GPUPermutationSettings,
  options: {maximumPermutations?: number} = {}
): Uint32Array {
  const {seed, permutationCount} = settings;
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new Error('permutation seed must be a non-negative safe integer');
  }
  validateCapacity(
    'getGPUPermutationParameterValues',
    'permutationCount',
    permutationCount,
    1,
    2 ** 31 - 1
  );
  if (options.maximumPermutations !== undefined && permutationCount > options.maximumPermutations) {
    throw new Error('permutationCount exceeds maximumPermutations');
  }
  const level = settings.significanceLevel ?? 0.05;
  if (!(level > 0 && level < 1)) {
    throw new Error('significanceLevel must be in (0, 1)');
  }
  const values = new Uint32Array(GPU_PERMUTATION_PARAMETER_LENGTH);
  values[0] = seed % 2 ** 32;
  values[1] = Math.floor(seed / 2 ** 32);
  values[2] = permutationCount;
  values[3] = new Uint32Array(new Float32Array([level]).buffer)[0];
  return values;
}

/**
 * Returns the serializable metadata shared by every permutation contributor. The default
 * `alternative` is `'two-sided'`.
 */
export function getGPUPermutationMetadata(
  settings: GPUPermutationSettings,
  options: {
    alternative?: GPUInferenceAlternative;
    multipleTesting?: GPUMultipleTestingMethod;
    /** When given, `permutationCount` above it throws instead of being reported as run. */
    maximumPermutations?: number;
  } = {}
): GPUPermutationMetadata {
  // Packing performs the canonical seed and count validation.
  getGPUPermutationParameterValues(settings, {maximumPermutations: options.maximumPermutations});
  return {
    seed: settings.seed,
    permutationCount: settings.permutationCount,
    alternative: resolvePermutationAlternative('getGPUPermutationMetadata', options.alternative),
    multipleTesting: options.multipleTesting ?? 'none',
    includeObserved: true
  };
}

/** Splits a seed into the two Philox key words, as the kernels do. @internal */
export function getPermutationSeedKey(seed: number): [number, number] {
  return [seed % 2 ** 32, Math.floor(seed / 2 ** 32)];
}
