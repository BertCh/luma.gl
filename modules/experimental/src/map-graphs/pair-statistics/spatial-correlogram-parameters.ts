// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePairStatisticsBounds} from './pair-statistics-parameters';

/** Number of float32 elements in a `GPUSpatialCorrelogram` parameter buffer. */
export const GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH = 8;

/** Number of float32 rows written to the optional `GPUSpatialCorrelogram` `statistics` output. */
export const GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH = 5;

/** Sentinel written to `peakBands` when no band qualifies. */
export const GPU_SPATIAL_CORRELOGRAM_NO_BAND = 0xffffffff;

/**
 * Null hypothesis of the analytic Moran's I variance:
 * - `'normality'`: values are independent draws from a normal distribution.
 * - `'randomization'`: the observed values are randomly permuted over the locations (uses the
 *   sample kurtosis; needs `n >= 4`). esda's default `z_rand` / `p_rand`.
 */
export type GPUSpatialCorrelogramVarianceAssumption = 'normality' | 'randomization';

/**
 * CPU description of the per-frame parameters of `GPUSpatialCorrelogram`.
 *
 * Every field can change between encodings without rebuilding or recompiling the graph.
 */
export type GPUSpatialCorrelogramParameters = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` extent. Rows outside it, or with a non-finite coordinate
   * or value, are excluded. The extent also sizes the neighbor-search lattice.
   */
  bounds: readonly [number, number, number, number];
  /** Upper distance of the last band. Band `b` ends at `maximumDistance * (b + 1) / bandCount`. */
  maximumDistance: number;
  /** Variance assumption of the z-scores and p-values. Defaults to `'randomization'`. */
  varianceAssumption?: GPUSpatialCorrelogramVarianceAssumption;
};

/**
 * Packs `GPUSpatialCorrelogram` parameters into the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, randomization (0 or 1), 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, the bounds are inverted, `maximumDistance <= 0`, the variance
 * assumption is unknown, or `target` is too short.
 */
export function getGPUSpatialCorrelogramParameterValues(
  parameters: GPUSpatialCorrelogramParameters,
  target: Float32Array = new Float32Array(GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH) {
    throw new Error(
      `Spatial correlogram target must hold ${GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH} elements`
    );
  }
  validatePairStatisticsBounds(
    'Spatial correlogram',
    parameters.bounds,
    parameters.maximumDistance
  );
  const assumption = parameters.varianceAssumption ?? 'randomization';
  if (assumption !== 'normality' && assumption !== 'randomization') {
    throw new Error('Spatial correlogram varianceAssumption must be normality or randomization');
  }
  target.set([
    ...parameters.bounds,
    parameters.maximumDistance,
    assumption === 'randomization' ? 1 : 0,
    0,
    0
  ]);
  return target;
}
