// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of uint32 elements in a {@link GPUNetworkKFunction} parameter view. */
export const GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUNetworkKFunction}. */
export type GPUNetworkKFunctionSettings = {
  /** Random seed of the simulated patterns, an integer in `[0, 2^53)`. */
  seed: number;
  /**
   * Number of simulated patterns evaluated, at most the compile-time `simulationCount`. The
   * remaining patterns are skipped (their K values and counts are 0).
   */
  activeSimulations: number;
};

/**
 * Packs per-frame {@link GPUNetworkKFunction} settings.
 *
 * Layout (uint32): `[seedLow, seedHigh, activeSimulations, 0]`.
 *
 * @param settings Seed and active simulation count.
 * @param target Optional destination of at least 4 elements.
 * @throws If the target is too short or a value is invalid.
 */
export function getGPUNetworkKFunctionParameterValues(
  settings: GPUNetworkKFunctionSettings,
  target: Uint32Array = new Uint32Array(GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH) {
    throw new Error(
      `K function parameter target must hold ${GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH} elements`
    );
  }
  const {seed, activeSimulations} = settings;
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new Error('K function seed must be a non-negative safe integer');
  }
  if (!Number.isSafeInteger(activeSimulations) || activeSimulations < 0) {
    throw new Error('K function activeSimulations must be a non-negative integer');
  }
  target[0] = seed % 2 ** 32;
  target[1] = Math.floor(seed / 2 ** 32);
  target[2] = activeSimulations;
  target[3] = 0;
  return target;
}
