// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a change-detection parameter view. */
export const GPU_CHANGE_DETECTION_PARAMETER_LENGTH = 5;

/** Per-frame parameters of {@link GPUChangeDetection}. */
export type GPUChangeDetectionParameters = {
  /** Slice index of the "before" slice for two-slice comparisons. Out of range makes them NaN. */
  beforeSlice: number;
  /** Slice index of the "after" slice for two-slice comparisons. Out of range makes them NaN. */
  afterSlice: number;
  /** Denominator guard of the log ratio. Defaults to `1e-6`. */
  epsilon?: number;
  /** Two-sided significance level for the `significance` class. Defaults to `0.05`. */
  alpha?: number;
  /**
   * First slice of the "after" group of the Welch t-test: slices `[0, splitSlice)` are "before"
   * and `[splitSlice, sliceCount)` are "after". Must be in `[1, sliceCount - 1]`. Defaults to 1.
   */
  splitSlice?: number;
};

/**
 * Packs change-detection parameters.
 *
 * Layout (float32): `[beforeSlice, afterSlice, epsilon, alpha, splitSlice]`. Slice indices are
 * small integers and are exact in float32.
 *
 * @param parameters Values to pack.
 * @param target Optional destination of at least 5 elements.
 */
export function getGPUChangeDetectionParameterValues(
  parameters: GPUChangeDetectionParameters,
  target: Float32Array = new Float32Array(GPU_CHANGE_DETECTION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_CHANGE_DETECTION_PARAMETER_LENGTH) {
    throw new Error(
      `Change detection parameter target must hold ${GPU_CHANGE_DETECTION_PARAMETER_LENGTH} elements`
    );
  }
  target[0] = parameters.beforeSlice;
  target[1] = parameters.afterSlice;
  target[2] = parameters.epsilon ?? 1e-6;
  target[3] = parameters.alpha ?? 0.05;
  target[4] = parameters.splitSlice ?? 1;
  return target;
}
