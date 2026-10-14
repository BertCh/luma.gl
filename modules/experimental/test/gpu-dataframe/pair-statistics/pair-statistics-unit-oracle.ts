// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  PAIR_STATISTICS_INTERQUARTILE_RANGE_PER_SIGMA,
  PAIR_STATISTICS_MINIMUM_UNIT_FRACTION,
  PAIR_STATISTICS_MINIMUM_VARIANCE,
  PAIR_STATISTICS_ZERO_SPREAD_UNIT_FRACTION
} from '../../../src/gpu-dataframe/pair-statistics/pair-statistics-fixed-point-unit';

// Moved from shipped src (a CPU oracle there breaks the GPU-only policy, as D-21 and E-17).
/**
 * CPU mirror of the unit variance kernel for oracles: the robust variance from the `linear`
 * interpolated quartiles of the finite `values`, clamped to `[variance * 2^-32, variance]`.
 */
export function getPairStatisticsUnitVarianceOnCPU(values: ArrayLike<number>): number {
  const finite = Array.from(values)
    .filter(value => Number.isFinite(value))
    .sort((a, b) => a - b);
  const count = finite.length;
  if (count === 0) {
    return 0;
  }
  const mean = finite.reduce((sum, value) => sum + value, 0) / count;
  const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;
  if (!(variance > PAIR_STATISTICS_MINIMUM_VARIANCE)) {
    return 0;
  }
  const quantile = (p: number) => {
    const height = (count - 1) * p;
    const lower = Math.floor(height);
    const upper = Math.min(lower + 1, count - 1);
    return finite[lower] + (finite[upper] - finite[lower]) * (height - lower);
  };
  const spread = (quantile(0.75) - quantile(0.25)) / PAIR_STATISTICS_INTERQUARTILE_RANGE_PER_SIGMA;
  const robustVariance =
    spread > 0 ? spread * spread : variance * PAIR_STATISTICS_ZERO_SPREAD_UNIT_FRACTION;
  return Math.min(
    Math.max(robustVariance, variance * PAIR_STATISTICS_MINIMUM_UNIT_FRACTION),
    variance
  );
}
