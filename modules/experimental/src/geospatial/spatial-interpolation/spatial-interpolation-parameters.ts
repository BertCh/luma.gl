// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUInverseDistanceWeighting} parameter view. */
export const GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH = 8;

/** Number of float32 elements in a {@link GPUFocalStatistics} parameter view. */
export const GPU_FOCAL_STATISTICS_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUInverseDistanceWeighting}. */
export type GPUInverseDistanceWeightingSettings = {
  /** Output raster extent `[minX, minY, maxX, maxY]` in sample units. Cell centers are sampled. */
  extent: readonly [number, number, number, number];
  /** Search radius in sample units. `Infinity` searches the whole index domain. */
  searchRadius: number;
  /** Distance power `p` in `w = 1 / d^p`. Zero gives an unweighted mean. Defaults to 2. */
  power?: number;
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
 * minimumNeighborCount]`.
 *
 * @param settings Extent, search radius, power, and neighbor limits.
 * @param target Optional destination of at least 8 elements.
 * @throws If the target is too short or a count is not a non-negative integer.
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
  target.set(settings.extent, 0);
  target[4] = settings.searchRadius;
  target[5] = settings.power ?? 2;
  target[6] = neighborCount;
  target[7] = minimumNeighborCount;
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
