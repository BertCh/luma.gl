// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Largest indicator (column) count supported by `GPUCompositeScore`. */
export const GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT = 16;

/** Number of float32 elements in a `GPUCompositeScore` parameter view. */
export const GPU_COMPOSITE_SCORE_PARAMETER_LENGTH =
  4 + 2 * GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT;

/** Float32 values per indicator in the `columnStatistics` output: `[min, max, mean, std]`. */
export const GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE = 4;

/** Float32 slots of the `principalComponentSummary` output. */
export const GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY = {
  /** Largest eigenvalue of the correlation matrix (Rayleigh quotient of the final vector). */
  EIGENVALUE: 0,
  /** Eigenvalue divided by the trace (the number of non-constant indicators). */
  EXPLAINED_VARIANCE_RATIO: 1,
  /** Norm of `C v - lambda v` for the final unit vector `v`; small when converged. */
  RESIDUAL: 2
} as const;

/** Number of float32 elements in the `principalComponentSummary` output. */
export const GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY_LENGTH = 3;

/** Fixed number of power-iteration steps of the first principal component. */
export const GPU_COMPOSITE_SCORE_POWER_ITERATIONS = 64;

/** Scaler codes stored in parameter slot 0. */
export const GPU_COMPOSITE_SCORE_SCALER = {
  /** `(x - min) / (max - min)`, 0 for a constant column. */
  'min-max': 0,
  /** `(x - mean) / std` with the population standard deviation, 0 for a constant column. */
  'z-score': 1,
  /** Percentile rank `averageRank / (n - 1)` in `[0, 1]`, ties averaged. Needs `enableRank`. */
  rank: 2
} as const;

/** Aggregation codes stored in parameter slot 1. */
export const GPU_COMPOSITE_SCORE_AGGREGATION = {
  /** `sum(w * s) / sum(|w|)`. */
  'weighted-sum': 0,
  /** `exp(sum(w * ln(s + epsilon)) / sum(w))` over non-negative weights. */
  'weighted-geometric-mean': 1,
  /** Projection on the first principal component of the correlation matrix. Needs `enablePrincipalComponent`. */
  'principal-component': 2
} as const;

/** Scaler names accepted by {@link getGPUCompositeScoreParameterValues}. */
export type GPUCompositeScoreScaler = keyof typeof GPU_COMPOSITE_SCORE_SCALER;

/** Aggregation names accepted by {@link getGPUCompositeScoreParameterValues}. */
export type GPUCompositeScoreAggregation = keyof typeof GPU_COMPOSITE_SCORE_AGGREGATION;

/** CPU description of the per-frame parameters of `GPUCompositeScore`. */
export type GPUCompositeScoreSettings = {
  /** Per-indicator scaler. Defaults to `'min-max'`. */
  scaler?: GPUCompositeScoreScaler;
  /** How scaled indicators combine into the score. Defaults to `'weighted-sum'`. */
  aggregation?: GPUCompositeScoreAggregation;
  /** One weight per indicator (at most 16). Missing trailing weights are 0. */
  weights: ArrayLike<number>;
  /**
   * One direction per indicator: a negative value means "higher is worse" and flips the scaled
   * value (`1 - s` for min-max and rank, `-s` for z-scores). Defaults to all `+1`.
   */
  directions?: ArrayLike<number>;
  /** Positive shift added before the logarithm of the geometric mean. Defaults to `1e-6`. */
  epsilon?: number;
};

/**
 * Packs per-frame `GPUCompositeScore` parameters.
 *
 * Layout (float32): `[scaler, aggregation, epsilon, 0, weights[16], directions[16]]`, where each
 * direction is stored as `+1` or `-1`. Every value can change between encodings without rebuilding
 * the graph.
 *
 * @param settings Scaler, aggregation, weights and directions.
 * @param target Optional destination of at least {@link GPU_COMPOSITE_SCORE_PARAMETER_LENGTH} elements.
 * @throws If more than 16 weights or directions are given, a weight is not finite, epsilon is not
 * positive, or `target` is too short.
 */
export function getGPUCompositeScoreParameterValues(
  settings: GPUCompositeScoreSettings,
  target: Float32Array = new Float32Array(GPU_COMPOSITE_SCORE_PARAMETER_LENGTH)
): Float32Array {
  const maximum = GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT;
  if (target.length < GPU_COMPOSITE_SCORE_PARAMETER_LENGTH) {
    throw new Error(
      `Composite score parameter target must hold ${GPU_COMPOSITE_SCORE_PARAMETER_LENGTH} elements`
    );
  }
  const {weights, directions} = settings;
  if (weights.length > maximum || (directions && directions.length > maximum)) {
    throw new Error(`Composite score supports at most ${maximum} weights and directions`);
  }
  const epsilon = settings.epsilon ?? 1e-6;
  if (!(epsilon > 0) || !Number.isFinite(epsilon)) {
    throw new Error('Composite score epsilon must be finite and positive');
  }
  const scaler = GPU_COMPOSITE_SCORE_SCALER[settings.scaler ?? 'min-max'];
  const aggregation = GPU_COMPOSITE_SCORE_AGGREGATION[settings.aggregation ?? 'weighted-sum'];
  if (scaler === undefined || aggregation === undefined) {
    throw new Error('Composite score scaler or aggregation is not recognized');
  }
  target.fill(0, 0, GPU_COMPOSITE_SCORE_PARAMETER_LENGTH);
  target[0] = scaler;
  target[1] = aggregation;
  target[2] = epsilon;
  for (let column = 0; column < maximum; column++) {
    const weight = column < weights.length ? weights[column] : 0;
    if (!Number.isFinite(weight)) {
      throw new Error('Composite score weights must be finite');
    }
    target[4 + column] = weight;
    const direction = directions && column < directions.length ? directions[column] : 1;
    target[4 + maximum + column] = direction < 0 ? -1 : 1;
  }
  return target;
}
