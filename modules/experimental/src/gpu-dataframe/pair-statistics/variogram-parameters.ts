// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePairStatisticsBounds} from './pair-statistics-parameters';

/** Number of float32 elements in a `GPUVariogram` parameter buffer. */
export const GPU_VARIOGRAM_PARAMETER_LENGTH = 8;

/** Number of float32 rows written to the optional `GPUVariogram` `statistics` output. */
export const GPU_VARIOGRAM_STATISTICS_LENGTH = 5;

/**
 * CPU description of the per-frame parameters of `GPUVariogram`.
 *
 * Distances use the planar units of the positions. Every field can change between encodings
 * without rebuilding or recompiling the graph.
 */
export type GPUVariogramParameters = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` extent. Rows outside it, or with a non-finite coordinate
   * or value, are excluded. The extent also sizes the neighbor-search lattice.
   */
  bounds: readonly [number, number, number, number];
  /** Largest lag: pairs farther apart are ignored. Lag bins split `[0, maximumDistance]`. */
  maximumDistance: number;
  /**
   * Start of the first direction sector, in radians counterclockwise from +x. Sector `s` holds
   * pairs whose undirected direction (taken modulo pi) minus this offset, again modulo pi, lies in
   * `[s * pi / directionCount, (s + 1) * pi / directionCount)`. Ignored when `directionCount` is 1.
   * Defaults to 0.
   */
  azimuthOffset?: number;
};

/**
 * Packs `GPUVariogram` parameters into the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, azimuthOffset, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, the bounds are inverted, `maximumDistance <= 0`, or `target`
 * is too short.
 */
export function getGPUVariogramParameterValues(
  parameters: GPUVariogramParameters,
  target: Float32Array = new Float32Array(GPU_VARIOGRAM_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_VARIOGRAM_PARAMETER_LENGTH) {
    throw new Error(`Variogram target must hold ${GPU_VARIOGRAM_PARAMETER_LENGTH} elements`);
  }
  const azimuthOffset = parameters.azimuthOffset ?? 0;
  validatePairStatisticsBounds('Variogram', parameters.bounds, parameters.maximumDistance);
  if (!Number.isFinite(azimuthOffset)) {
    throw new Error('Variogram azimuthOffset must be finite');
  }
  target.set([...parameters.bounds, parameters.maximumDistance, azimuthOffset, 0, 0]);
  return target;
}
