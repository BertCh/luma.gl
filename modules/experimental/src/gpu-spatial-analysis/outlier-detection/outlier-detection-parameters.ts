// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPULocalOutlierFactor} parameter view. */
export const GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH = 4;

/** scikit-learn's `LocalOutlierFactor` adds this to the mean reachability distance. */
export const GPU_LOCAL_OUTLIER_FACTOR_DEFAULT_DENSITY_FLOOR = 1e-10;

/** Per-frame parameters of {@link GPULocalOutlierFactor}, packed by {@link getGPULocalOutlierFactorParameterValues}. */
export type GPULocalOutlierFactorParameters = {
  /**
   * Rows with `lof > threshold` are flagged in the `outlier` mask and counted. Defaults to 1.5,
   * a common cut: scores near 1 are inliers, scores well above 1 are locally sparse.
   */
  threshold?: number;
  /**
   * Added to each row's mean reachability distance before inverting it, so a cluster of
   * coincident points (mean reachability 0) gets a large finite density `1 / densityFloor`
   * instead of infinity. Must be positive and finite. Defaults to scikit-learn's `1e-10`.
   * Raise it (for example to the coordinate resolution) to make duplicate-heavy data less
   * extreme.
   */
  densityFloor?: number;
};

/**
 * Packs {@link GPULocalOutlierFactorParameters} into the float32 layout read by the kernels.
 *
 * Layout: `[threshold, densityFloor, 0, 0]`.
 */
export function getGPULocalOutlierFactorParameterValues(
  parameters: GPULocalOutlierFactorParameters = {}
): Float32Array {
  const threshold = parameters.threshold ?? 1.5;
  const densityFloor = parameters.densityFloor ?? GPU_LOCAL_OUTLIER_FACTOR_DEFAULT_DENSITY_FLOOR;
  if (Number.isNaN(threshold)) {
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
