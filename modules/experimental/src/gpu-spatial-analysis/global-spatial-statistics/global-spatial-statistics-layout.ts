// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A global statistic computed by `GPUGlobalSpatialStatistics`. */
export type GPUGlobalSpatialStatistic =
  | 'moran'
  | 'geary'
  | 'getisOrdG'
  | 'bivariateMoran'
  | 'joinCount';

/**
 * Offsets of the blocks in the `results` view of `GPUGlobalSpatialStatistics`. The layout is fixed
 * whatever statistics are requested; blocks of statistics that were not requested hold quiet NaN.
 */
export const GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT = {
  /** `GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY` fields. */
  summary: 0,
  /** `GPU_GLOBAL_SPATIAL_STATISTIC_FIELD` fields of global Moran's I. */
  moran: 8,
  /** `GPU_GLOBAL_SPATIAL_STATISTIC_FIELD` fields of Geary's C. */
  geary: 16,
  /** `GPU_GLOBAL_SPATIAL_STATISTIC_FIELD` fields of Getis-Ord General G. */
  getisOrdG: 24,
  /** `GPU_GLOBAL_SPATIAL_STATISTIC_FIELD` fields of bivariate Moran's I. */
  bivariateMoran: 32,
  /** `GPU_GLOBAL_JOIN_COUNT_FIELD` fields of the join counts. */
  joinCount: 40,
  /** Number of float32 values in `results`. */
  length: 52
} as const;

/** Fields of the summary block of `results`. */
export const GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY = {
  /** Number of included rows `n`. */
  count: 0,
  /** `S0 = sum_ij w_ij` over included pairs. */
  s0: 1,
  /** `S1 = 1/2 sum_ij (w_ij + w_ji)^2`. */
  s1: 2,
  /** `S2 = sum_i (w_i. + w_.i)^2`. */
  s2: 3,
  /** Mean of `values` over included rows. */
  mean: 4,
  /** Population variance (`ddof = 0`) of `values` over included rows. */
  variance: 5,
  /** Number of included rows whose value is nonzero (black, for join counts). */
  blackCount: 6,
  /** Number of included rows with no included neighbor (out or in). */
  islandCount: 7
} as const;

/** Fields of the Moran, Geary, General G and bivariate Moran blocks of `results`. */
export const GPU_GLOBAL_SPATIAL_STATISTIC_FIELD = {
  /** The statistic. */
  statistic: 0,
  /** Its expectation under the null. */
  expected: 1,
  /** Variance under the normality assumption (NaN where not defined). */
  varianceNormality: 2,
  /** `(statistic - expected) / sqrt(varianceNormality)`. */
  zNormality: 3,
  /** Two-sided normal p-value of `zNormality`. */
  pNormality: 4,
  /** Variance under randomization (all permutations of the values over the included rows). */
  varianceRandomization: 5,
  /** `(statistic - expected) / sqrt(varianceRandomization)`. */
  zRandomization: 6,
  /** Two-sided normal p-value of `zRandomization`. */
  pRandomization: 7
} as const;

/** Fields of the join-count block of `results` (randomization, i.e. nonfree sampling). */
export const GPU_GLOBAL_JOIN_COUNT_FIELD = {
  /** Black-black joins `BB = 1/2 sum_ij b_ij x_i x_j`. */
  blackBlack: 0,
  /** Black-white joins `BW = 1/2 sum_ij b_ij (x_i - x_j)^2`. */
  blackWhite: 1,
  /** White-white joins. */
  whiteWhite: 2,
  /** Total joins `J = S0 / 2` of the binarized weights. */
  joins: 3,
  expectedBlackBlack: 4,
  varianceBlackBlack: 5,
  zBlackBlack: 6,
  pBlackBlack: 7,
  expectedBlackWhite: 8,
  varianceBlackWhite: 9,
  zBlackWhite: 10,
  pBlackWhite: 11
} as const;
