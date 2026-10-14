// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Radially symmetric 2D kernel of {@link GPUKernelDensity}. */
export type GPUKernelDensityKernel = 'gaussian' | 'quartic' | 'epanechnikov';

/**
 * Normal-reference bandwidth rule computed on the GPU by {@link GPUKernelDensity}. Both rules of
 * `scipy.stats.gaussian_kde` reduce to the same factor `n_eff^(-1/6)` in two dimensions.
 */
export type GPUKernelDensityBandwidthRule = 'scott' | 'silverman';

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

/**
 * Evaluation method of {@link GPUKernelDensity}. `'exact'` gathers the points inside the kernel
 * support for every cell; `'binned'` bins the points onto the grid and blurs the bins with an
 * almost-gaussian filter whose cost does not depend on the bandwidth; `'auto'` picks one on the
 * GPU every encoding from the bandwidth.
 */
export type GPUKernelDensityMethod = 'exact' | 'binned' | 'auto';

/**
 * `'auto'` evaluates exactly while the gaussian support radius (`5 h`) is at most this many cells
 * (the larger of the two axes) and with the binned blur above it.
 */
export const GPU_KERNEL_DENSITY_AUTO_SUPPORT_CELLS = 32;

/** Box blurs per axis of the binned method. Together with one two-sided exponential pass. */
export const GPU_KERNEL_DENSITY_BINNED_BOX_PASS_COUNT = 3;

/**
 * Share of the blur variance (`(h / cell)^2` per axis) carried by the two-sided exponential pass
 * of the binned method; the three box blurs carry the rest equally. Boxes have a negative excess
 * kurtosis (`-6/5` each) and the exponential a positive one (`+3`), so this share cancels the
 * fourth cumulant of the cascade: `3 * 1.2 * v_box^2 = 3 * v_exp^2` gives `v_exp = 1.095 v_box`, so
 * `v_exp / (3 v_box + v_exp) = 0.267`; 0.32 is the share that minimised the worst kernel error
 * over bandwidths of 3 to 200 cells in a sweep (0.30 to 0.34 are within 15% of it).
 */
export const GPU_KERNEL_DENSITY_BINNED_EXPONENTIAL_VARIANCE_SHARE = 0.32;

/**
 * Largest kernel-shape error of the binned filter relative to the gaussian kernel peak, for
 * bandwidths of at least {@link GPU_KERNEL_DENSITY_BINNED_MINIMUM_BANDWIDTH_CELLS} cells. One axis
 * measures at most 0.0076 (sup norm of the 1D filter minus the sampled gaussian, divided by the
 * gaussian peak); the product of two axes then differs from the 2D gaussian by at most
 * `2 * 0.0076 + 0.0076^2`, which is rounded up to 0.016.
 */
export const GPU_KERNEL_DENSITY_BINNED_SHAPE_ERROR = 0.016;

/**
 * Margin, in bandwidths (`h / cell` cells each), that the binned grid needs on every side of the
 * cells whose values you read. The blur is a cascade of separate passes and each pass drops the
 * mass that leaves the array, whereas the exact kernel would let it return; the box passes spread
 * at most `3 * 0.48 h = 1.43 h` and the exponential tail is shorter than the remaining margin.
 * Measured against the same filter on a margin of `7 h`, the difference in units of `W K_h(0)` is
 * 8e-4 at `1.5 h`, 3e-5 at `2 h` and below 1e-11 at `2.5 h` (bandwidths of 3 to 24 cells, points
 * on the grid corners and edges); with less margin the estimate near the edges drops, by 27% of
 * `W K_h(0)` with none. `GPUKernelDensity` runs the binned path only at `2.5 h` or more.
 */
export const GPU_KERNEL_DENSITY_BINNED_MARGIN_BANDWIDTHS = 2.5;

/** Smallest `h / cell` for which {@link getGPUKernelDensityBinnedErrorBound} holds. */
export const GPU_KERNEL_DENSITY_BINNED_MINIMUM_BANDWIDTH_CELLS = 3;

/** float32 unit roundoff, `2^-24`. */
const FLOAT32_UNIT_ROUNDOFF = 2 ** -24;

/**
 * Bound on `|binned - exact|` of one cell in units of `W * K_h(0)`, where `W` is the sum of the
 * included weights and `K_h(0) = 1 / (2 pi h^2)` is the gaussian peak, for `method: 'binned'`
 * with every point inside the binned region (see `binnedPadding`). Because the kernel difference
 * of one point is at most the bound times the peak, the sum over points is at most `W` times it.
 * The terms add:
 *
 * - linear (cloud-in-cell) binning: bilinear interpolation of `K_h(u - x)` between cell centers
 *   errs by at most `(cellX^2 / 8) max|d2K/dx2| + (cellY^2 / 8) max|d2K/dy2|` and `max|d2K/dx2| =
 *   K_h(0) / h^2`, so `(r^2 / 4) K_h(0)` for `r = max(cellX, cellY) / h`;
 * - the filter shape: {@link GPU_KERNEL_DENSITY_BINNED_SHAPE_ERROR};
 * - the 5-bandwidth truncation of the exact estimate, `exp(-12.5)`, 3.7e-6 (rounded to 4e-6);
 * - float32 filtering, worst case: with `s = 1 / r` the box passes carry running-sum chains of at
 *   most `2R + 3` plus the refresh interval terms (`R = 0.476 s`) and the exponential passes
 *   `1 / (1 - a) = 0.4 s` terms of roundoff, `u * (13 s + 210)` in total for `u = 2^-24`, an
 *   upper bound that real data stays far below (errors do not align);
 * - the cascade's dropped mass at the array edge, below 1e-5 once the margin is at least
 *   {@link GPU_KERNEL_DENSITY_BINNED_MARGIN_BANDWIDTHS} bandwidths.
 *
 * Valid for `h / cell >= 3` (smaller bandwidths have a much larger shape error: use `'exact'`).
 *
 * @param cellOverBandwidth `r`, the larger cell size over the bandwidth.
 */
export function getGPUKernelDensityBinnedErrorBound(cellOverBandwidth: number): number {
  const r = cellOverBandwidth;
  return (
    (r * r) / 4 +
    GPU_KERNEL_DENSITY_BINNED_SHAPE_ERROR +
    1.4e-5 +
    FLOAT32_UNIT_ROUNDOFF * (13 / r + 210)
  );
}
