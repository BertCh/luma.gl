// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPULocalOutlierFactor} parameter view. */
export const GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH = 4;

/** scikit-learn's `LocalOutlierFactor` adds this to the mean reachability distance. */
export const GPU_LOCAL_OUTLIER_FACTOR_DEFAULT_DENSITY_FLOOR = 1e-10;

/** Per-frame settings of {@link GPULocalOutlierFactor}, packed by {@link getGPULocalOutlierFactorParameterValues}. */
export type GPULocalOutlierFactorSettings = {
  /**
   * Rows with `lof > threshold` are flagged in the `outlier` mask and counted. There is no default:
   * LOF has no null distribution, so no cut is calibrated (Breunig et al. 2000 give none; values
   * near 1 are as dense as their neighbors, well above 1 locally sparse, and the scale depends on
   * `k` and the data). Choose it from the score distribution, for example a top fraction. When
   * omitted the packed threshold is NaN, no row is flagged, the `outlier` mask is all 0 and
   * `outlierCount` is 0; the `lof` scores are unaffected. The flag is exploratory, not a
   * significance statement.
   */
  threshold?: number;
  /**
   * Added to each row's mean reachability distance before inverting it, so a cluster of
   * coincident points (mean reachability 0) gets a large finite density `1 / densityFloor`
   * instead of infinity. Must be positive and finite. Defaults to scikit-learn's `1e-10`.
   * Raise it (for example to the coordinate resolution) to make duplicate-heavy data less
   * extreme. Its unit is the length unit of `neighbors.spatialContext`; declare the context
   * of this value through `GPULocalOutlierFactorProps.spatialContext` to check compatibility.
   */
  densityFloor?: number;
};

/**
 * Packs {@link GPULocalOutlierFactorSettings} into the float32 layout read by the kernels.
 *
 * Layout: `[threshold (NaN when omitted: nothing is flagged), densityFloor, 0, 0]`.
 */
export function getGPULocalOutlierFactorParameterValues(
  settings: GPULocalOutlierFactorSettings = {}
): Float32Array {
  const threshold = settings.threshold ?? Number.NaN;
  const densityFloor = settings.densityFloor ?? GPU_LOCAL_OUTLIER_FACTOR_DEFAULT_DENSITY_FLOOR;
  if (settings.threshold !== undefined && Number.isNaN(settings.threshold)) {
    throw new Error('GPULocalOutlierFactor threshold must not be NaN');
  }
  if (!Number.isFinite(densityFloor) || !(Math.fround(densityFloor) > 0)) {
    throw new Error('GPULocalOutlierFactor densityFloor must be a positive finite number');
  }
  const values = new Float32Array(GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH);
  values[0] = threshold;
  values[1] = densityFloor;
  return values;
}
