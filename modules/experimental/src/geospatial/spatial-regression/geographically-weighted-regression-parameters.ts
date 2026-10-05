// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Largest predictor count (columns of `predictors`, without the intercept) of `GPUGeographicallyWeightedRegression`. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT = 7;

/** Largest compile-time bandwidth ladder length. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH = 32;

/** Largest compile-time `k` of adaptive (k-th nearest neighbour) bandwidths. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT = 128;

/** Largest row count: every location scans every row, so the cost grows with `rows^2 * ladder`. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT = 65536;

/** Header float32 slots before the ladder in the parameter view. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH = 4;

/**
 * Multiplier applied to the k-th nearest neighbour distance of adaptive bandwidths so that the
 * k-th neighbour keeps a small positive bisquare weight (mgwr uses 1.0000001; f32 needs more).
 */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR = 1.00001;

/** Smallest residual variance `RSS / n` used inside the AICc logarithm. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE = 1e-30;

/** Kernel codes stored in parameter slot 0. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_KERNEL = {
  /** `w = exp(-0.5 (d / h)^2)`. */
  gaussian: 0,
  /** `w = (1 - (d / h)^2)^2` for `d < h`, else 0. */
  bisquare: 1
} as const;

/** Bandwidth mode codes stored in parameter slot 1. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_BANDWIDTH_MODE = {
  /** Ladder values are distances `h` in position units. */
  fixed: 0,
  /** Ladder values are neighbour counts `k`; `h` is the distance to the k-th nearest row. */
  adaptive: 1
} as const;

/** Values of the `localStatus` output. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS = {
  /** The local system was solved. */
  OK: 0,
  /** The local weighted system was singular, ill-conditioned, or the bandwidth was invalid. */
  SINGULAR: 1,
  /** The row is masked out or has a non-finite position, predictor or response. */
  EXCLUDED: 2
} as const;

/** Float32 slots of the `summary` output. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY = {
  /** Residual sum of squares at the selected bandwidth. */
  RESIDUAL_SUM_OF_SQUARES: 0,
  /** Trace of the hat matrix, `sum(S_ii)`. */
  TRACE_OF_HAT: 1,
  /** Corrected Akaike information criterion of the selected bandwidth. */
  AICC: 2,
  /** `1 - RSS / TSS` over the included rows (global, not the mean of local R^2). */
  R_SQUARED: 3,
  /** Number of included rows `n`. */
  OBSERVATION_COUNT: 4,
  /** 1 when at least one ladder candidate had a finite AICc, otherwise 0. */
  HAS_VALID_CANDIDATE: 5
} as const;

/** Number of float32 elements in the `summary` output. */
export const GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH = 6;

/** Kernel names accepted by {@link getGPUGeographicallyWeightedRegressionParameterValues}. */
export type GPUGeographicallyWeightedRegressionKernel =
  keyof typeof GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_KERNEL;

/** Bandwidth mode names accepted by {@link getGPUGeographicallyWeightedRegressionParameterValues}. */
export type GPUGeographicallyWeightedRegressionBandwidthMode =
  keyof typeof GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_BANDWIDTH_MODE;

/** CPU description of the per-frame parameters of `GPUGeographicallyWeightedRegression`. */
export type GPUGeographicallyWeightedRegressionSettings = {
  /** Weight kernel. Defaults to `'bisquare'`. */
  kernel?: GPUGeographicallyWeightedRegressionKernel;
  /** Meaning of the ladder values. Defaults to `'fixed'`. */
  bandwidthMode?: GPUGeographicallyWeightedRegressionBandwidthMode;
  /**
   * Candidate bandwidths: distances (`'fixed'`) or neighbour counts `k` (`'adaptive'`, rounded to
   * integers). One value is a fixed bandwidth; several are searched by AICc.
   */
  bandwidths: ArrayLike<number>;
};

/**
 * Returns the float32 length of a parameter view for a ladder capacity.
 *
 * @param maximumBandwidthCount Compile-time ladder length of the contributor.
 */
export function getGPUGeographicallyWeightedRegressionParameterLength(
  maximumBandwidthCount: number = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH
): number {
  return GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH + maximumBandwidthCount;
}

/**
 * Packs per-frame `GPUGeographicallyWeightedRegression` parameters.
 *
 * Layout (float32): `[kernel, bandwidthMode, candidateCount, 0, ladder...]`. Every value can change
 * between encodings without rebuilding the graph.
 *
 * @param settings Kernel, bandwidth mode and ladder.
 * @param maximumBandwidthCount Ladder capacity of the contributor (its `maximumBandwidthCount`).
 * @param target Optional destination of at least the parameter length.
 * @throws If the ladder is empty, longer than the capacity, or has a non-finite value, or a name is
 * not recognized, or `target` is too short.
 */
export function getGPUGeographicallyWeightedRegressionParameterValues(
  settings: GPUGeographicallyWeightedRegressionSettings,
  maximumBandwidthCount: number = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  target: Float32Array = new Float32Array(
    getGPUGeographicallyWeightedRegressionParameterLength(maximumBandwidthCount)
  )
): Float32Array {
  const length = getGPUGeographicallyWeightedRegressionParameterLength(maximumBandwidthCount);
  if (target.length < length) {
    throw new Error(`GWR parameter target must hold ${length} elements`);
  }
  const {bandwidths} = settings;
  if (bandwidths.length < 1 || bandwidths.length > maximumBandwidthCount) {
    throw new Error(`GWR bandwidths must hold 1 to ${maximumBandwidthCount} values`);
  }
  const kernel = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_KERNEL[settings.kernel ?? 'bisquare'];
  const mode =
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_BANDWIDTH_MODE[settings.bandwidthMode ?? 'fixed'];
  if (kernel === undefined || mode === undefined) {
    throw new Error('GWR kernel or bandwidth mode is not recognized');
  }
  target.fill(0, 0, length);
  target[0] = kernel;
  target[1] = mode;
  target[2] = bandwidths.length;
  for (let index = 0; index < bandwidths.length; index++) {
    if (!Number.isFinite(bandwidths[index])) {
      throw new Error('GWR bandwidths must be finite');
    }
    target[GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH + index] =
      bandwidths[index];
  }
  return target;
}
