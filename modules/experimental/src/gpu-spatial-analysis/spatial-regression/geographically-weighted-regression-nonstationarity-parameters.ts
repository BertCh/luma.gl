// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Float32 values per coefficient row of the `table` output. */
export const GPU_GWR_NONSTATIONARITY_TABLE_STRIDE = 6;

/** Fields of one coefficient row of the `table` output (intercept first, then each predictor). */
export const GPU_GWR_NONSTATIONARITY_TABLE = {
  /** Standard deviation of the observed local estimates over the fitted locations (mgwr `init_sd`). */
  observedStandardDeviation: 0,
  /**
   * Pseudo p-value `(g + 1) / (P + 1)`, `g` the number of permutations whose standard deviation is
   * at least the observed one: the probability of this much spatial variation by chance.
   */
  pseudoPValue: 1,
  /** Exceedance count `g`, an exact integer. */
  exceedances: 2,
  /** Mean of the permuted standard deviations. */
  simulatedMean: 3,
  /** Population standard deviation of the permuted standard deviations. */
  simulatedStandardDeviation: 4,
  /** `(observed - simulatedMean) / simulatedStandardDeviation`. */
  zSimulated: 5
} as const;

/** Float32 slots of the `summary` output. */
export const GPU_GWR_NONSTATIONARITY_SUMMARY = {
  /** Permutations `P` run. */
  permutations: 0,
  /** Included locations `m` (unmasked, finite). */
  locationCount: 1,
  /** Locations with a finite observed fit (they enter the observed standard deviations). */
  observedFitCount: 2,
  /** Permuted local fits that failed (singular or invalid bandwidth), summed over permutations. */
  failedFitCount: 3,
  /** Number of float32 values in `summary`. */
  length: 4
} as const;
