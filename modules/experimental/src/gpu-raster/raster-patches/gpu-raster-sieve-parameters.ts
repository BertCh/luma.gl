// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateParameterTarget} from '../raster-band-utils';

/** Number of uint32 elements in a sieve parameter buffer. */
export const GPU_RASTER_SIEVE_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame settings packed for `GPURasterSieve`. */
export type GPURasterSieveSettings = {
  /** Patches with fewer pixels than this are sieved. Integer from 1 through 2^32 - 1. */
  minimumPixels: number;
};

/**
 * Packs sieve parameters into the 4-element uint32 layout read by `GPURasterSieve`.
 *
 * Layout: `[minimumPixels, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between
 * encodings to change the threshold without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If `minimumPixels` is not an integer in `[1, 2^32 - 1]`, or `target` is too short.
 */
export function getGPURasterSieveParameterValues(
  settings: GPURasterSieveSettings,
  target: Uint32Array = new Uint32Array(GPU_RASTER_SIEVE_PARAMETER_LENGTH)
): Uint32Array {
  validateParameterTarget(
    'getGPURasterSieveParameterValues',
    target,
    GPU_RASTER_SIEVE_PARAMETER_LENGTH
  );
  const {minimumPixels} = settings;
  if (!Number.isInteger(minimumPixels) || minimumPixels < 1 || minimumPixels > 0xffffffff) {
    throw new Error(
      'getGPURasterSieveParameterValues minimumPixels must be an integer in [1, 2^32 - 1]'
    );
  }
  target.set([minimumPixels, 0, 0, 0]);
  return target;
}
