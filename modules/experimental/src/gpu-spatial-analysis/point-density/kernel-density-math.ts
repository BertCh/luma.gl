// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Radially symmetric 2D kernel of {@link GPUKernelDensity}. */
export type GPUKernelDensityKernel = 'gaussian' | 'quartic' | 'epanechnikov';

/** Window of the edge correction, `[minX, minY, maxX, maxY]` in position units. */
export type GPUKernelDensityWindow = readonly [number, number, number, number];

/** Gaussian kernels are truncated at this many bandwidths; the lost mass is `exp(-12.5)`, 3.7e-6. */
export const GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION = 5;

/**
 * Radius, in position units, beyond which the kernel is treated as zero: `bandwidth` for the
 * compact `'quartic'` and `'epanechnikov'` kernels, `5 * bandwidth` for `'gaussian'`.
 */
export function getGPUKernelDensitySupportRadius(
  kernel: GPUKernelDensityKernel,
  bandwidth: number
): number {
  return kernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION * bandwidth : bandwidth;
}

/**
 * Kernel value `K_h(d) = K(d / h) / h^2` at distance `d`, normalised so that it integrates to one
 * over the plane (Silverman 1986, Density Estimation, Table 3.1 for d = 2):
 *
 * - gaussian: `exp(-t / 2) / (2 pi h^2)` with `t = d^2 / h^2`
 * - epanechnikov: `2 / pi (1 - t) / h^2` for `t < 1`
 * - quartic (biweight): `3 / pi (1 - t)^2 / h^2` for `t < 1`
 *
 * Zero beyond {@link getGPUKernelDensitySupportRadius}.
 */
export function evaluateGPUKernelDensityKernel(
  kernel: GPUKernelDensityKernel,
  bandwidth: number,
  distance: number
): number {
  const t = (distance * distance) / (bandwidth * bandwidth);
  if (distance >= getGPUKernelDensitySupportRadius(kernel, bandwidth)) {
    return 0;
  }
  if (kernel === 'gaussian') {
    return Math.exp(-0.5 * t) / (2 * Math.PI * bandwidth * bandwidth);
  }
  if (!(t < 1)) {
    return 0;
  }
  return kernel === 'epanechnikov'
    ? (2 * (1 - t)) / (Math.PI * bandwidth * bandwidth)
    : (3 * (1 - t) * (1 - t)) / (Math.PI * bandwidth * bandwidth);
}

/** Complementary error function, Numerical Recipes Chebyshev fit (relative error under 1.2e-7). */
function getComplementaryErrorFunction(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    );
  return x >= 0 ? r : 2 - r;
}

/** Normal probability mass between `low` and `high`, accurate in either tail. */
function getNormalMass(low: number, high: number): number {
  const getUpperTail = (z: number) => 0.5 * getComplementaryErrorFunction(z * Math.SQRT1_2);
  if (low > 0) {
    return getUpperTail(low) - getUpperTail(high);
  }
  if (high < 0) {
    return getUpperTail(-high) - getUpperTail(-low);
  }
  return 1 - getUpperTail(-low) - getUpperTail(high);
}

/** Integral of `(1 - t^2)^(n / 2)` over `[0, x]` for odd `n` in `{1, 3, 5}`, `0 <= x <= 1`. */
function getUnitDiskChordIntegral(order: 1 | 3 | 5, x: number): number {
  const s = Math.sqrt(Math.max(1 - x * x, 0));
  const first = 0.5 * (x * s + Math.asin(Math.min(x, 1)));
  if (order === 1) {
    return first;
  }
  const third = (x * s * s * s) / 4 + 0.75 * first;
  return order === 3 ? third : (x * s ** 5) / 6 + (5 / 6) * third;
}

/**
 * Mass of the unit-bandwidth polynomial kernel `(p + 1) / pi (1 - r^2)^p` over the quadrant
 * rectangle `[0, a] x [0, b]` (`p` = 1 epanechnikov, 2 quartic).
 */
function getPolynomialQuadrantMass(power: 1 | 2, a: number, b: number): number {
  const width = Math.min(a, 1);
  const height = Math.min(b, 1);
  if (!(width > 0) || !(height > 0)) {
    return 0;
  }
  // Rows x < kink lie fully below the disk edge at height `height`; beyond it the chord ends first.
  const kink = Math.sqrt(1 - height * height);
  const bulkEnd = Math.min(width, kink);
  let total: number;
  if (power === 1) {
    total = height * (bulkEnd - bulkEnd ** 3 / 3) - (height ** 3 * bulkEnd) / 3;
  } else {
    total =
      height * (bulkEnd - (2 * bulkEnd ** 3) / 3 + bulkEnd ** 5 / 5) -
      ((2 * height ** 3) / 3) * (bulkEnd - bulkEnd ** 3 / 3) +
      (height ** 5 * bulkEnd) / 5;
  }
  if (width > kink) {
    total +=
      power === 1
        ? (2 / 3) * (getUnitDiskChordIntegral(3, width) - getUnitDiskChordIntegral(3, kink))
        : (8 / 15) * (getUnitDiskChordIntegral(5, width) - getUnitDiskChordIntegral(5, kink));
  }
  return ((power + 1) / Math.PI) * total;
}

/**
 * Kernel mass inside a rectangular window, `e(u) = integral over the window of K_h(u - v) dv`,
 * the quantity that edge correction divides by. It is 1 when the whole kernel lies inside the
 * window, and decreases toward 0 as `point` moves outside.
 *
 * Gaussian: the product of normal-CDF differences (untruncated; the 5-bandwidth truncation of the
 * estimator loses less than 4e-6). Quartic and epanechnikov: exact. By symmetry the mass of the
 * window is an inclusion-exclusion over four signed quadrant rectangles, and each quadrant mass is
 * a closed-form integral of the polynomial kernel over disk-clipped rows (an asin term from the
 * chord integral `(1 - x^2)^(n / 2)`), with no numerical quadrature. The erfc fit limits the gaussian to
 * about 1e-7 relative error. This function is the float64 reference of the WGSL used on the GPU.
 *
 * @param kernel Kernel.
 * @param bandwidth Bandwidth `h` in position units.
 * @param point Query position `u`.
 * @param window Window `[minX, minY, maxX, maxY]`.
 */
export function getGPUKernelDensityWindowMass(
  kernel: GPUKernelDensityKernel,
  bandwidth: number,
  point: readonly [number, number],
  window: GPUKernelDensityWindow
): number {
  const x0 = (window[0] - point[0]) / bandwidth;
  const y0 = (window[1] - point[1]) / bandwidth;
  const x1 = (window[2] - point[0]) / bandwidth;
  const y1 = (window[3] - point[1]) / bandwidth;
  if (kernel === 'gaussian') {
    return getNormalMass(x0, x1) * getNormalMass(y0, y1);
  }
  const power = kernel === 'epanechnikov' ? 1 : 2;
  const getSignedMass = (a: number, b: number) =>
    Math.sign(a) * Math.sign(b) * getPolynomialQuadrantMass(power, Math.abs(a), Math.abs(b));
  return (
    getSignedMass(x1, y1) - getSignedMass(x0, y1) - getSignedMass(x1, y0) + getSignedMass(x0, y0)
  );
}

/** Per-axis and isotropic bandwidth from a rule of thumb. */
export type GPUKernelDensityBandwidth = {
  /** Bandwidth along x, `factor * standard deviation of x`. */
  x: number;
  /** Bandwidth along y. */
  y: number;
  /**
   * Single bandwidth for the radially symmetric kernels of {@link GPUKernelDensity}:
   * `factor * sqrt((variance x + variance y) / 2)` (pooled standard deviation).
   */
  isotropic: number;
  /** Rule factor: `n_eff^(-1/6)`, the shared two-dimensional value of Scott's and Silverman's rules. */
  factor: number;
  /** Effective sample size `(sum w)^2 / sum w^2` (the row count without weights). */
  effectiveSampleSize: number;
};

function getRuleOfThumbBandwidth(
  positions: ArrayLike<number>,
  weights?: ArrayLike<number>
): GPUKernelDensityBandwidth {
  const rowCount = Math.floor(positions.length / 2);
  const isIncluded = (row: number): boolean =>
    Number.isFinite(positions[2 * row]) &&
    Number.isFinite(positions[2 * row + 1]) &&
    (!weights || (Number.isFinite(weights[row]) && weights[row] >= 0));
  let weightSum = 0;
  let weightSquareSum = 0;
  let includedCount = 0;
  for (let row = 0; row < rowCount; row++) {
    if (isIncluded(row)) {
      const weight = weights ? weights[row] : 1;
      weightSum += weight;
      weightSquareSum += weight * weight;
      includedCount++;
    }
  }
  if (includedCount < 2 || !(weightSum > 0)) {
    throw new Error('Bandwidth rules need at least two included positions with positive weight');
  }
  const effectiveSampleSize = (weightSum * weightSum) / weightSquareSum;
  if (!(effectiveSampleSize > 1)) {
    throw new Error('Bandwidth rules need an effective sample size above one');
  }
  let meanX = 0;
  let meanY = 0;
  for (let row = 0; row < rowCount; row++) {
    if (isIncluded(row)) {
      const weight = weights ? weights[row] : 1;
      meanX += (weight * positions[2 * row]) / weightSum;
      meanY += (weight * positions[2 * row + 1]) / weightSum;
    }
  }
  let spreadX = 0;
  let spreadY = 0;
  for (let row = 0; row < rowCount; row++) {
    if (isIncluded(row)) {
      const weight = weights ? weights[row] : 1;
      spreadX += (weight * (positions[2 * row] - meanX) ** 2) / weightSum;
      spreadY += (weight * (positions[2 * row + 1] - meanY) ** 2) / weightSum;
    }
  }
  // Unbiased weighted covariance, numpy.cov(aweights=w): divide by 1 - sum(w_n^2).
  const correction = 1 - weightSquareSum / (weightSum * weightSum);
  const varianceX = spreadX / correction;
  const varianceY = spreadY / correction;
  // scipy: scotts_factor = n_eff^(-1/(d+4)); silverman_factor = (n_eff * (d + 2) / 4)^(-1/(d+4)).
  // For d = 2 the two rules coincide.
  const factor = effectiveSampleSize ** (-1 / 6);
  return {
    x: factor * Math.sqrt(varianceX),
    y: factor * Math.sqrt(varianceY),
    isotropic: factor * Math.sqrt((varianceX + varianceY) / 2),
    factor,
    effectiveSampleSize
  };
}

/**
 * Scott's (1992, Multivariate Density Estimation, eq. 6.42) rule of thumb for a 2D normal-reference
 * bandwidth, `h_i = n_eff^(-1/6) * sigma_i`, matching `scipy.stats.gaussian_kde(bw_method='scott')`
 * (`sigma_i` from the unbiased weighted covariance). Rows with a non-finite position, or a
 * non-finite or negative weight, are skipped, as the estimator does. `isotropic` is the pooled
 * single bandwidth for the radial kernels. A normal-reference rule oversmooths clustered or
 * multimodal data: treat it as a starting point, not an optimum.
 *
 * @param positions Interleaved `[x0, y0, x1, y1, ...]` read-back positions.
 * @param weights Optional per-row weights.
 * @throws If fewer than two rows are included or the effective sample size is not above one.
 */
export function getScottBandwidth(
  positions: ArrayLike<number>,
  weights?: ArrayLike<number>
): GPUKernelDensityBandwidth {
  return getRuleOfThumbBandwidth(positions, weights);
}

/**
 * Silverman's (1986, Density Estimation, eq. 4.14) d-dimensional rule of thumb,
 * `h_i = (4 / (n_eff * (d + 2)))^(1 / (d + 4)) * sigma_i`, matching
 * `scipy.stats.gaussian_kde(bw_method='silverman')`. For d = 2 the factor `n_eff^(-1/6)` equals
 * Scott's, so both rules return the same bandwidths in two dimensions. Inputs, skipped rows, and
 * errors are those of {@link getScottBandwidth}.
 *
 * @param positions Interleaved `[x0, y0, x1, y1, ...]` read-back positions.
 * @param weights Optional per-row weights.
 */
export function getSilvermanBandwidth(
  positions: ArrayLike<number>,
  weights?: ArrayLike<number>
): GPUKernelDensityBandwidth {
  return getRuleOfThumbBandwidth(positions, weights);
}
