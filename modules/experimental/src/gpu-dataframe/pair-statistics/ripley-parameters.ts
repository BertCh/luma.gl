// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePairStatisticsBounds} from './pair-statistics-parameters';

/** Number of float32 elements in a `GPURipley` parameter buffer. */
export const GPU_RIPLEY_PARAMETER_LENGTH = 8;

/** Parameter slot holding the Monte Carlo seed read by `GPURipleyEnvelope` (ignored by `GPURipley`). */
export const GPU_RIPLEY_SEED_SLOT = 6;

/** Number of float32 rows of the `statistics` summary of `GPURipleyEnvelope`. */
export const GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH = 6;

/** Largest Monte Carlo seed: it travels as an exactly representable float32 integer. */
export const GPU_RIPLEY_MAXIMUM_SEED = 2 ** 24 - 1;

/**
 * Edge-correction codes stored in the `GPURipley` parameter buffer (slot 5). The GPU switches on
 * the stored code, so the mode can change every frame without recompiling.
 */
export const GPU_RIPLEY_EDGE_CORRECTION = {
  /** Uncorrected estimator. */
  none: 0,
  /** Border (reduced-sample) correction. */
  border: 1,
  /** Ripley (1977) isotropic correction for a rectangular window. */
  isotropic: 2
} as const;

/** Edge-correction mode name of `GPURipley`. */
export type GPURipleyEdgeCorrection = keyof typeof GPU_RIPLEY_EDGE_CORRECTION;

/**
 * CPU description of the per-frame parameters of `GPURipley`.
 *
 * Every field can change between encodings without rebuilding or recompiling the graph.
 */
export type GPURipleyParameters = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` observation window. Rows outside it, or with a non-finite
   * coordinate, are excluded. Its area is the study area of the estimators.
   */
  bounds: readonly [number, number, number, number];
  /**
   * Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. Keep it well below
   * the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but
   * its variance grows and the isotropic weight cap biases it low once a circle is mostly outside.
   */
  maximumDistance: number;
  /** Edge-correction mode. Defaults to `'isotropic'`. */
  edgeCorrection?: GPURipleyEdgeCorrection;
  /**
   * Monte Carlo seed of `GPURipleyEnvelope`, an integer in `[0, 2^24)`. `GPURipley` ignores it.
   * Defaults to 0. The same seed reproduces the same simulated patterns exactly.
   */
  seed?: number;
};

/**
 * Packs `GPURipley` parameters into the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, edgeCorrectionCode, seed, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, the bounds are inverted, `maximumDistance <= 0`, the mode is
 * unknown, `seed` is not an integer in `[0, 2^24)`, or `target` is too short.
 */
export function getGPURipleyParameterValues(
  parameters: GPURipleyParameters,
  target: Float32Array = new Float32Array(GPU_RIPLEY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_RIPLEY_PARAMETER_LENGTH) {
    throw new Error(`Ripley target must hold ${GPU_RIPLEY_PARAMETER_LENGTH} elements`);
  }
  validatePairStatisticsBounds('Ripley', parameters.bounds, parameters.maximumDistance);
  const edgeCorrection = parameters.edgeCorrection ?? 'isotropic';
  if (!Object.prototype.hasOwnProperty.call(GPU_RIPLEY_EDGE_CORRECTION, edgeCorrection)) {
    throw new Error(`Ripley edgeCorrection must be 'none', 'border' or 'isotropic'`);
  }
  const seed = parameters.seed ?? 0;
  if (!Number.isInteger(seed) || seed < 0 || seed > GPU_RIPLEY_MAXIMUM_SEED) {
    throw new Error(`Ripley seed must be an integer in [0, 2^24)`);
  }
  target.set([
    ...parameters.bounds,
    parameters.maximumDistance,
    GPU_RIPLEY_EDGE_CORRECTION[edgeCorrection],
    seed,
    0
  ]);
  return target;
}
