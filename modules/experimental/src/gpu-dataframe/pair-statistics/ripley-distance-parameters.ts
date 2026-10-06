// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePairStatisticsBounds} from './pair-statistics-parameters';

/** Number of float32 elements in a `GPURipleyDistanceFunctions` parameter buffer. */
export const GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH = 8;

/** Edge-correction mode of `GPURipleyDistanceFunctions`: none or the border (reduced sample) method. */
export type GPURipleyDistanceEdgeCorrection = 'none' | 'border';

/**
 * CPU description of the per-frame parameters of `GPURipleyDistanceFunctions`.
 *
 * Every field can change between encodings without rebuilding or recompiling the graph.
 */
export type GPURipleyDistanceParameters = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` observation window. Events outside it, or with a
   * non-finite coordinate, are excluded. The reference lattice of F covers this rectangle.
   */
  bounds: readonly [number, number, number, number];
  /** Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. */
  maximumDistance: number;
  /** `'border'` (default) divides by the points farther from the window edge than `r`. */
  edgeCorrection?: GPURipleyDistanceEdgeCorrection;
};

/**
 * Packs `GPURipleyDistanceFunctions` parameters into the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, edgeCorrectionCode, 0, 0]`, where the code is 0 for
 * `'none'` and 1 for `'border'`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, the bounds are inverted, `maximumDistance <= 0`, the mode is
 * unknown, or `target` is too short.
 */
export function getGPURipleyDistanceParameterValues(
  parameters: GPURipleyDistanceParameters,
  target: Float32Array = new Float32Array(GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH) {
    throw new Error(
      `Ripley distance functions target must hold ${GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH} elements`
    );
  }
  validatePairStatisticsBounds(
    'Ripley distance functions',
    parameters.bounds,
    parameters.maximumDistance
  );
  const edgeCorrection = parameters.edgeCorrection ?? 'border';
  if (edgeCorrection !== 'none' && edgeCorrection !== 'border') {
    throw new Error(`Ripley distance functions edgeCorrection must be 'none' or 'border'`);
  }
  target.set([
    ...parameters.bounds,
    parameters.maximumDistance,
    edgeCorrection === 'border' ? 1 : 0,
    0,
    0
  ]);
  return target;
}
