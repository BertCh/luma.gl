// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUInverseDistanceWeighting} parameter view. */
export const GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH = 9;

/** Number of float32 elements in a {@link GPUFocalStatistics} parameter view. */
export const GPU_FOCAL_STATISTICS_PARAMETER_LENGTH = 4;

/**
 * Weight taper of {@link GPUInverseDistanceWeighting}: `'none'` is `w = 1 / d^p` inside the
 * search radius (a hard cutoff); `'modified-shepard'` is `w = ((R - d)+ / (R d))^p` with `R` the
 * search radius (Franke and Nielson 1980), which falls to zero at `R`.
 */
export type GPUInverseDistanceWeightingTaper = 'none' | 'modified-shepard';

/** Per-frame settings of {@link GPUInverseDistanceWeighting}. */
export type GPUInverseDistanceWeightingSettings = {
  /** Output raster extent `[minX, minY, maxX, maxY]` in position units. Cell centers are sampled. */
  extent: readonly [number, number, number, number];
  /**
   * Search radius in position units. `Infinity` searches the whole index domain, except with the
   * `'modified-shepard'` taper, which needs a finite radius.
   */
  searchRadius: number;
  /** Distance power `p` in `w = 1 / d^p`. Zero gives an unweighted mean. Defaults to 2. */
  power?: number;
  /** Weight taper at the search radius. Defaults to `'none'` (hard cutoff). */
  taper?: GPUInverseDistanceWeightingTaper;
  /**
   * Nearest-neighbor limit `k`; zero (default) uses every sample within the radius. Values above
   * the contributor's compile-time `maximumNeighborCount` are clamped to it.
   */
  neighborCount?: number;
  /** Cells with fewer contributing samples (and no exact hit) are nodata. Defaults to 1. */
  minimumNeighborCount?: number;
};

/**
 * Packs per-frame {@link GPUInverseDistanceWeighting} parameters.
 *
 * Layout (float32): `[minX, minY, maxX, maxY, searchRadius, power, neighborCount,
 * minimumNeighborCount, taper (0 none, 1 modified-shepard)]`.
 *
 * @param settings Extent, search radius, power, taper, and neighbor limits.
 * @param target Optional destination of at least 9 elements.
 * @throws If the target is too short, a count is not a non-negative integer, the taper is unknown,
 *   or the `'modified-shepard'` taper is requested with a non-finite radius.
 */
export function getGPUInverseDistanceWeightingParameterValues(
  settings: GPUInverseDistanceWeightingSettings,
  target: Float32Array = new Float32Array(GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH) {
    throw new Error(
      `Inverse distance weighting parameter target must hold ${GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH} elements`
    );
  }
  const neighborCount = settings.neighborCount ?? 0;
  const minimumNeighborCount = settings.minimumNeighborCount ?? 1;
  for (const [name, count] of [
    ['neighborCount', neighborCount],
    ['minimumNeighborCount', minimumNeighborCount]
  ] as const) {
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`Inverse distance weighting ${name} must be a non-negative integer`);
    }
  }
  const taper = settings.taper ?? 'none';
  if (taper !== 'none' && taper !== 'modified-shepard') {
    throw new Error("Inverse distance weighting taper must be 'none' or 'modified-shepard'");
  }
  if (taper === 'modified-shepard' && !Number.isFinite(settings.searchRadius)) {
    throw new Error('Inverse distance weighting taper needs a finite searchRadius');
  }
  target.set(settings.extent, 0);
  target[4] = settings.searchRadius;
  target[5] = settings.power ?? 2;
  target[6] = neighborCount;
  target[7] = minimumNeighborCount;
  target[8] = taper === 'modified-shepard' ? 1 : 0;
  return target;
}

/** Focal window shape. */
export type GPUFocalStatisticsShape = 'square' | 'circle';

/** Per-frame settings of {@link GPUFocalStatistics}. */
export type GPUFocalStatisticsSettings = {
  /**
   * Window radius in cells. A square window covers `|dx|, |dy| <= floor(radius)`; a circle covers
   * offsets with `dx^2 + dy^2 <= radius^2` inside that square. Clamped to the contributor's
   * compile-time `maximumRadius`.
   */
  radius: number;
  /** Window shape. Defaults to `'square'`. */
  shape?: GPUFocalStatisticsShape;
  /** Cells whose window has fewer valid cells are nodata. Defaults to 1. */
  minimumCount?: number;
  /** When true, a nodata center cell yields nodata statistics. Defaults to false. */
  propagateCenterNoData?: boolean;
};

/**
 * Packs per-frame {@link GPUFocalStatistics} parameters.
 *
 * Layout (float32): `[radius, shape (0 square, 1 circle), minimumCount, propagateCenterNoData]`.
 *
 * @param settings Radius, shape, and nodata policy.
 * @param target Optional destination of at least 4 elements.
 * @throws If the target is too short or `minimumCount` is not a non-negative integer.
 */
export function getGPUFocalStatisticsParameterValues(
  settings: GPUFocalStatisticsSettings,
  target: Float32Array = new Float32Array(GPU_FOCAL_STATISTICS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_FOCAL_STATISTICS_PARAMETER_LENGTH) {
    throw new Error(
      `Focal statistics parameter target must hold ${GPU_FOCAL_STATISTICS_PARAMETER_LENGTH} elements`
    );
  }
  const minimumCount = settings.minimumCount ?? 1;
  if (!Number.isSafeInteger(minimumCount) || minimumCount < 0) {
    throw new Error('Focal statistics minimumCount must be a non-negative integer');
  }
  const shape = settings.shape ?? 'square';
  if (shape !== 'square' && shape !== 'circle') {
    throw new Error("Focal statistics shape must be 'square' or 'circle'");
  }
  target[0] = settings.radius;
  target[1] = shape === 'circle' ? 1 : 0;
  target[2] = minimumCount;
  target[3] = settings.propagateCenterNoData ? 1 : 0;
  return target;
}
