// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a spatial-autocorrelation parameter buffer. */
export const GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH = 12;

/** Number of float32 rows written to an optional `globalStatistics` output. */
export const GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH = 4;

/**
 * Two-sided standard normal critical values of the Getis-Ord Gi* confidence bins, in bin order
 * `1, 2, 3`: 90%, 95% and 99% confidence (`p <= 0.10, 0.05, 0.01`).
 */
export const GPU_HOT_SPOT_CRITICAL_Z_SCORES = [
  1.6448536269514722, 1.959963984540054, 2.5758293035489004
] as const;

/** Significance levels of the Getis-Ord Gi* confidence bins `1, 2, 3`. */
export const GPU_HOT_SPOT_SIGNIFICANCE_LEVELS = [0.1, 0.05, 0.01] as const;

/** Local Moran quadrant codes written by `GPULocalMoran` (esda numbering). */
export const GPU_LOCAL_MORAN_QUADRANT = {
  /** Not significant, invalid, or undefined. */
  NOT_SIGNIFICANT: 0,
  /** High value surrounded by high values (hot spot). */
  HIGH_HIGH: 1,
  /** Low value surrounded by high values (outlier). */
  LOW_HIGH: 2,
  /** Low value surrounded by low values (cold spot). */
  LOW_LOW: 3,
  /** High value surrounded by low values (outlier). */
  HIGH_LOW: 4
} as const;

/**
 * Global moments that replace the moments computed from the current rows.
 *
 * Use them to pin the reference distribution, for example to moments measured once over the
 * full dataset, so that z-scores do not change when a viewport mask changes the row set.
 */
export type GPUSpatialAutocorrelationFixedMoments = {
  /** Population size `n`, at least 2. */
  count: number;
  /** Population mean. */
  mean: number;
  /** Population variance (denominator `n`), positive. */
  variance: number;
};

/**
 * CPU description of the per-frame parameters of `GPUHotSpotAnalysis` and `GPULocalMoran`.
 *
 * Distances use the same planar units as the positions. Every field can change between encodings
 * without rebuilding or recompiling the graph.
 */
export type GPUSpatialAutocorrelationParameters = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` extent of the data. Rows outside it, or with a non-finite
   * coordinate, are excluded. Bounds size the active cell lattice; a tighter extent is faster.
   */
  bounds: readonly [number, number, number, number];
  /**
   * Distance band. Row `j` is a neighbor of row `i` when `dx * dx + dy * dy <= radius * radius`
   * (evaluated in f32). Positive.
   */
  radius: number;
  /**
   * Two-sided significance level of `GPULocalMoran` quadrants, in `(0, 1)`. Defaults to `0.05`.
   * `GPUHotSpotAnalysis` ignores it and always reports 90/95/99% bins.
   */
  significanceLevel?: number;
  /**
   * Weight transform used by the local Moran's I value: `'binary'` (`w = 1`) or `'row'`
   * (`w = 1 / neighborCount`, esda's default). Defaults to `'row'`. z-scores, p-values and
   * quadrants are invariant to it; only `localI` changes. Gi* always uses binary weights.
   */
  weightTransform?: 'binary' | 'row';
  /** Optional moments that replace the moments of the current rows. */
  fixedMoments?: GPUSpatialAutocorrelationFixedMoments;
};

/**
 * Packs spatial-autocorrelation parameters into the 12-element float32 layout read by
 * `GPUHotSpotAnalysis` and `GPULocalMoran`.
 *
 * Layout: `[minX, minY, maxX, maxY, radius, significanceLevel, rowStandardized,
 * useFixedMoments, fixedCount, fixedMean, fixedVariance, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 12 elements. A new array is returned when omitted.
 * @throws If a value is not finite, `radius <= 0`, `significanceLevel` is outside `(0, 1)`, fixed
 * moments are invalid, or `target` is too short.
 */
export function getGPUSpatialAutocorrelationParameterValues(
  parameters: GPUSpatialAutocorrelationParameters,
  target: Float32Array = new Float32Array(GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH) {
    throw new Error(
      `Spatial autocorrelation target must hold ${GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} elements`
    );
  }
  const significanceLevel = parameters.significanceLevel ?? 0.05;
  for (const value of [...parameters.bounds, parameters.radius, significanceLevel]) {
    if (!Number.isFinite(value)) {
      throw new Error(
        'Spatial autocorrelation bounds, radius and significanceLevel must be finite'
      );
    }
  }
  if (parameters.radius <= 0) {
    throw new Error('Spatial autocorrelation radius must be positive');
  }
  if (significanceLevel <= 0 || significanceLevel >= 1) {
    throw new Error('Spatial autocorrelation significanceLevel must be in (0, 1)');
  }
  const weightTransform = parameters.weightTransform ?? 'row';
  if (weightTransform !== 'binary' && weightTransform !== 'row') {
    throw new Error('Spatial autocorrelation weightTransform must be binary or row');
  }
  const fixedMoments = parameters.fixedMoments;
  if (fixedMoments) {
    if (
      !Number.isFinite(fixedMoments.count) ||
      !Number.isFinite(fixedMoments.mean) ||
      !Number.isFinite(fixedMoments.variance) ||
      fixedMoments.count < 2 ||
      fixedMoments.variance <= 0
    ) {
      throw new Error(
        'Spatial autocorrelation fixedMoments need a finite count >= 2, mean and positive variance'
      );
    }
  }
  target.set([
    ...parameters.bounds,
    parameters.radius,
    significanceLevel,
    weightTransform === 'row' ? 1 : 0,
    fixedMoments ? 1 : 0,
    fixedMoments?.count ?? 0,
    fixedMoments?.mean ?? 0,
    fixedMoments?.variance ?? 0,
    0
  ]);
  return target;
}
