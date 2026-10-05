// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePairStatisticsBounds} from './pair-statistics-parameters';

/** Number of float32 elements in a `GPUPointPatternIndices` parameter buffer. */
export const GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH = 8;

/** Number of float32 rows of the `clarkEvans` summary of `GPUPointPatternIndices`. */
export const GPU_CLARK_EVANS_LENGTH = 6;

/** Number of float32 rows of the `quadratStatistics` summary of `GPUPointPatternIndices`. */
export const GPU_QUADRAT_STATISTICS_LENGTH = 6;

/** Largest quadrat count (`columns * rows`) of `GPUPointPatternIndices`. */
export const GPU_QUADRAT_MAXIMUM_COUNT = 65536;

/**
 * CPU description of the per-frame parameters of `GPUPointPatternIndices`.
 *
 * Every field can change between encodings without rebuilding or recompiling the graph.
 */
export type GPUPointPatternIndicesParameters = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` window. Rows outside it, or with a non-finite coordinate,
   * are excluded. Its area is the study area of the Clark-Evans index and the quadrats tile it.
   */
  bounds: readonly [number, number, number, number];
  /**
   * Size hint of the neighbor-search cells (cells are at least this wide). Results never depend on
   * it, only speed: a value near the expected nearest-neighbor distance is fastest.
   */
  maximumDistance: number;
};

/**
 * Packs `GPUPointPatternIndices` parameters into the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, 0, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, the bounds are inverted, `maximumDistance <= 0`, or `target`
 * is too short.
 */
export function getGPUPointPatternIndicesParameterValues(
  parameters: GPUPointPatternIndicesParameters,
  target: Float32Array = new Float32Array(GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH) {
    throw new Error(
      `Point pattern target must hold ${GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH} elements`
    );
  }
  validatePairStatisticsBounds('Point pattern', parameters.bounds, parameters.maximumDistance);
  target.set([...parameters.bounds, parameters.maximumDistance, 0, 0, 0]);
  return target;
}
