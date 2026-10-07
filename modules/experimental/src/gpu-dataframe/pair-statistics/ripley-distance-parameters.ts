// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePairStatisticsBounds} from './pair-statistics-parameters';

/** Number of float32 elements in a `GPURipleyDistanceFunctions` parameter buffer. */
export const GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH = 8;

/**
 * Edge-correction mode of `GPURipleyDistanceFunctions`: none, the border (reduced sample, spatstat
 * `"rs"`) method, the Kaplan-Meier estimator (`"km"`) or the Hanisch estimator (`"han"` for G, the
 * Chiu-Stoyan weighting `"cs"` for F).
 */
export type GPURipleyDistanceEdgeCorrection = 'none' | 'border' | 'kaplan-meier' | 'hanisch';

/** Float32 code of each {@link GPURipleyDistanceEdgeCorrection} in the parameter layout. */
const EDGE_CORRECTION_CODES: Record<GPURipleyDistanceEdgeCorrection, number> = {
  none: 0,
  border: 1,
  'kaplan-meier': 2,
  hanisch: 3
};

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
  /**
   * `'border'` (default) divides by the points farther from the window edge than `r`;
   * `'kaplan-meier'` and `'hanisch'` use every point (see `GPURipleyDistanceFunctions`).
   */
  edgeCorrection?: GPURipleyDistanceEdgeCorrection;
};

/**
 * Packs `GPURipleyDistanceFunctions` parameters into the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, edgeCorrectionCode, 0, 0]`, where the code is 0 for
 * `'none'`, 1 for `'border'`, 2 for `'kaplan-meier'` and 3 for `'hanisch'`.
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
  if (!Object.prototype.hasOwnProperty.call(EDGE_CORRECTION_CODES, edgeCorrection)) {
    throw new Error(
      `Ripley distance functions edgeCorrection must be 'none', 'border', 'kaplan-meier' or 'hanisch'`
    );
  }
  target.set([
    ...parameters.bounds,
    parameters.maximumDistance,
    EDGE_CORRECTION_CODES[edgeCorrection],
    0,
    0
  ]);
  return target;
}
