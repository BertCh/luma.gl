// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Float64 CPU oracle of the `GPUKernelDensity` bandwidth rule (moved out of the shipped source,
 * E-17). The GPU computes the same quantity with `bandwidthRule`; this reference is only a test
 * oracle, validated against `scipy.stats.gaussian_kde` by `kernel-density.node.spec.ts` and the
 * committed fixtures `oracles/geometry-kernel-density.py`.
 */

/** Per-axis and isotropic bandwidth of the normal-reference rule in two dimensions. */
export type BandwidthOracle = {
  /** Bandwidth along x, `factor * standard deviation of x`. */
  x: number;
  /** Bandwidth along y. */
  y: number;
  /** `factor * sqrt((variance x + variance y) / 2)`, the single bandwidth of the radial kernels. */
  isotropic: number;
  /** `n_eff^(-1/6)`: scipy's `scotts_factor` and `silverman_factor` for d = 2. */
  factor: number;
  /** `(sum w)^2 / sum w^2`. */
  effectiveSampleSize: number;
};

/**
 * Scott/Silverman bandwidth of interleaved `[x, y, ...]` positions with unbiased weighted
 * covariance (`numpy.cov(aweights = w)`), skipping rows with a non-finite position or a
 * non-finite or negative weight. Two passes (mean, then centered sums), as the GPU does.
 *
 * @throws If fewer than two rows are included or the effective sample size is not above one.
 */
export function getBandwidthOracle(
  positions: ArrayLike<number>,
  weights?: ArrayLike<number>
): BandwidthOracle {
  const rowCount = Math.floor(positions.length / 2);
  const rows: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (
      Number.isFinite(positions[2 * row]) &&
      Number.isFinite(positions[2 * row + 1]) &&
      (!weights || (Number.isFinite(weights[row]) && weights[row] >= 0))
    ) {
      rows.push(row);
    }
  }
  const weightOf = (row: number) => (weights ? weights[row] : 1);
  let weightSum = 0;
  let weightSquareSum = 0;
  for (const row of rows) {
    weightSum += weightOf(row);
    weightSquareSum += weightOf(row) ** 2;
  }
  const effectiveSampleSize = (weightSum * weightSum) / weightSquareSum;
  if (rows.length < 2 || !(weightSum > 0) || !(effectiveSampleSize > 1)) {
    throw new Error(
      'Bandwidth rules need at least two rows and an effective sample size above one'
    );
  }
  let meanX = 0;
  let meanY = 0;
  for (const row of rows) {
    meanX += (weightOf(row) * positions[2 * row]) / weightSum;
    meanY += (weightOf(row) * positions[2 * row + 1]) / weightSum;
  }
  let spreadX = 0;
  let spreadY = 0;
  for (const row of rows) {
    spreadX += (weightOf(row) * (positions[2 * row] - meanX) ** 2) / weightSum;
    spreadY += (weightOf(row) * (positions[2 * row + 1] - meanY) ** 2) / weightSum;
  }
  const correction = 1 - weightSquareSum / (weightSum * weightSum);
  const varianceX = spreadX / correction;
  const varianceY = spreadY / correction;
  const factor = effectiveSampleSize ** (-1 / 6);
  return {
    x: factor * Math.sqrt(varianceX),
    y: factor * Math.sqrt(varianceY),
    isotropic: factor * Math.sqrt((varianceX + varianceY) / 2),
    factor,
    effectiveSampleSize
  };
}
