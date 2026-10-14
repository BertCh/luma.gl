// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePairStatisticsBounds,
  type GPUPairStatisticsBounds
} from './pair-statistics-parameters';

/** Number of float32 elements in a `GPURipley` parameter buffer. */
export const GPU_RIPLEY_PARAMETER_LENGTH = 8;

/**
 * Parameter slot holding the Monte Carlo seed read by `GPURipleyEnvelope` (ignored by `GPURipley`).
 * The slot holds the bits of a uint32, not a float32 value.
 */
export const GPU_RIPLEY_SEED_SLOT = 6;

/** Number of float32 rows of the `statistics` summary of `GPURipleyEnvelope`. */
export const GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH = 6;

/**
 * Edge-correction codes stored in the `GPURipley` parameter buffer (slot 5). The GPU switches on
 * the stored code, so the mode can change every frame without recompiling.
 */
export const GPU_RIPLEY_EDGE_CORRECTION = {
  /** Uncorrected estimator. */
  none: 0,
  /** Border (reduced-sample) correction. */
  border: 1,
  /** Ripley (1977) isotropic correction for rectangle or polygon windows. */
  isotropic: 2
} as const;

/** Edge-correction mode name of `GPURipley`. */
export type GPURipleyEdgeCorrection = keyof typeof GPU_RIPLEY_EDGE_CORRECTION;

/**
 * CPU description of the per-frame parameters of `GPURipley`.
 *
 * Every field can change between encodings without rebuilding or recompiling the graph.
 */
export type GPURipleySettings = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` observation window. Rows outside it, or with a non-finite
   * coordinate, are excluded. Its area is the study area of the estimators.
   * Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots.
   */
  bounds: GPUPairStatisticsBounds;
  /**
   * Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. Keep it well below
   * the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but
   * its variance grows and the isotropic weight cap biases it low once a circle is mostly outside.
   */
  maximumDistance: number;
  /** Edge-correction mode. Defaults to `'isotropic'`. */
  edgeCorrection?: GPURipleyEdgeCorrection;
  /**
   * Monte Carlo seed of `GPURipleyEnvelope`, a uint32. `GPURipley` ignores it. Defaults to 0. The
   * same seed reproduces the same simulated patterns exactly.
   */
  seed?: number;
};

/**
 * Packs `GPURipley` parameters into the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, edgeCorrectionCode, seed, 0]`, where `seed` is
 * stored as uint32 bits.
 *
 * @param@param settings Settings to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, the bounds are inverted, `maximumDistance <= 0`, the mode is
 * unknown, `seed` is not a uint32, or `target` is too short.
 */
export function getGPURipleyParameterValues(
  settings: GPURipleySettings,
  target: Float32Array = new Float32Array(GPU_RIPLEY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_RIPLEY_PARAMETER_LENGTH) {
    throw new Error(`GPURipley target must hold ${GPU_RIPLEY_PARAMETER_LENGTH} elements`);
  }
  validatePairStatisticsBounds('GPURipley', settings.bounds, settings.maximumDistance);
  const edgeCorrection = settings.edgeCorrection ?? 'isotropic';
  if (!Object.prototype.hasOwnProperty.call(GPU_RIPLEY_EDGE_CORRECTION, edgeCorrection)) {
    throw new Error(`GPURipley edgeCorrection must be 'none', 'border' or 'isotropic'`);
  }
  const seed = settings.seed ?? 0;
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('GPURipley seed must be a uint32');
  }
  target.set([
    ...(settings.bounds === 'study-window' ? [NaN, NaN, NaN, NaN] : settings.bounds),
    settings.maximumDistance,
    GPU_RIPLEY_EDGE_CORRECTION[edgeCorrection],
    0,
    0
  ]);
  // The seed travels as uint32 bits: a float32 value would cap it at 2^24.
  new Uint32Array(target.buffer, target.byteOffset, GPU_RIPLEY_PARAMETER_LENGTH)[
    GPU_RIPLEY_SEED_SLOT
  ] = seed;
  return target;
}
