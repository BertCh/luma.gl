// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Screen-space antialiasing for projected Gaussians, shared by the CPU and GPU projection paths.
 *
 * Base 3D Gaussian splatting dilates every projected covariance by a small isotropic screen-space
 * filter so a sub-pixel Gaussian still covers a pixel. Dilation alone brightens what it widens:
 * the Gaussian keeps its peak value while its footprint grows, so total emitted energy rises, and
 * the error grows as the Gaussian shrinks relative to a pixel - exactly what happens when a
 * geospatial camera zooms out. Mip-Splatting's compensation restores the lost normalization by
 * scaling opacity with the square root of the determinant ratio between the original and the
 * dilated covariance.
 *
 * @see {@link https://niujinshuchong.github.io/mip-splatting/ | Mip-Splatting}
 * @see {@link https://lzhnb.github.io/project-pages/analytic-splatting/ | Analytic-Splatting}
 */

/** Screen-space antialiasing applied to a dilated projected covariance. */
export type SplatAntialiasingMode =
  /** Dilate the covariance without compensating opacity, matching base 3DGS. */
  | 'none'
  /** Dilate and rescale opacity by `sqrt(det(original) / det(dilated))`. */
  | 'mip-splatting';

/** Per-fragment Gaussian evaluation used to resolve coverage inside the support quad. */
export type SplatFragmentKernel =
  /** Evaluate the Gaussian at the pixel center, matching base 3DGS. */
  | 'gaussian'
  /** Integrate the Gaussian over the pixel footprint with a logistic CDF approximation. */
  | 'analytic';

/**
 * Default added screen-space filter variance in square pixels.
 *
 * The reference 3DGS rasterizer adds `0.3` px^2 of *variance*. Expressing the same filter as a
 * standard deviation gives `sqrt(0.3) ~= 0.5477` px, so a renderer that squares a `0.3` px
 * standard deviation under-filters sub-pixel Gaussians by a factor of `0.3 / 0.09 ~= 3.3`.
 */
export const DEFAULT_SPLAT_SCREEN_FILTER_VARIANCE = 0.3;

/** Smallest determinant treated as invertible when compensating a dilated covariance. */
const MINIMUM_COVARIANCE_DETERMINANT = 1e-12;

/** Opacity below which a Gaussian contributes less than one 8-bit color step at its peak. */
const MINIMUM_VISIBLE_ALPHA = 1 / 255;

/**
 * Resolves the added screen-space filter variance from either spelling of the property.
 *
 * `kernel2DSize` names a standard deviation and is retained for compatibility;
 * `screenSpaceFilterVariance` names the variance the reference rasterizer actually specifies and
 * takes precedence when both are supplied.
 */
export function getSplatScreenFilterVariance(props: {
  screenSpaceFilterVariance?: number;
  kernel2DSize?: number;
}): number {
  if (props.screenSpaceFilterVariance !== undefined) {
    return Math.max(props.screenSpaceFilterVariance, 0);
  }
  if (props.kernel2DSize !== undefined) {
    const standardDeviation = Math.max(props.kernel2DSize, 0);
    return standardDeviation * standardDeviation;
  }
  return DEFAULT_SPLAT_SCREEN_FILTER_VARIANCE;
}

/**
 * Returns Mip-Splatting's opacity compensation for one dilated 2D covariance.
 *
 * `covariance00`, `covariance01` and `covariance11` describe the covariance *before* dilation;
 * `addedVariance` is the isotropic variance added to both diagonal terms. The off-diagonal term is
 * unchanged by an isotropic filter, so only the diagonal enters the dilated determinant.
 *
 * @returns A factor in `(0, 1]` to multiply into opacity. Returns `1` when the filter is disabled
 * or the pre-dilation covariance is degenerate, which leaves base 3DGS behavior untouched.
 */
export function getSplatDilationCompensation(
  covariance00: number,
  covariance01: number,
  covariance11: number,
  addedVariance: number
): number {
  if (!(addedVariance > 0)) {
    return 1;
  }
  const originalDeterminant = covariance00 * covariance11 - covariance01 * covariance01;
  const dilatedDeterminant =
    (covariance00 + addedVariance) * (covariance11 + addedVariance) - covariance01 * covariance01;
  if (
    !Number.isFinite(originalDeterminant) ||
    !Number.isFinite(dilatedDeterminant) ||
    originalDeterminant <= MINIMUM_COVARIANCE_DETERMINANT ||
    dilatedDeterminant <= MINIMUM_COVARIANCE_DETERMINANT
  ) {
    return 1;
  }
  return Math.min(Math.sqrt(originalDeterminant / dilatedDeterminant), 1);
}

/**
 * Returns the energy-conserving opacity compensation for a uniformly rescaled covariance.
 *
 * Clamping a projected Gaussian to a maximum screen size scales both axes by `axisScale`, which
 * scales the covariance by `axisScale^2` and its determinant by `axisScale^4`. Without
 * compensation the clamp silently removes energy; with it, a shrunk Gaussian raises its peak so
 * the integral is preserved, up to full opacity.
 *
 * @returns A factor of at least `1`. Callers still clamp the resulting opacity to `1`.
 */
export function getSplatClampCompensation(axisScale: number): number {
  if (!(axisScale > 0) || axisScale >= 1) {
    return 1;
  }
  return 1 / (axisScale * axisScale);
}

/**
 * Returns the support radius, in standard deviations, at which a Gaussian falls below 1/255.
 *
 * A fixed 3-sigma quad is correct only for fully opaque Gaussians. Solving
 * `alpha * exp(-r^2 / 2) = 1/255` gives `r = sqrt(2 * ln(255 * alpha))`, which shrinks the quad
 * for the low-opacity Gaussians that dominate a trained scene. `KHR_gaussian_splatting` mandates
 * 3 sigma as the cutoff, so the result is also clamped to `maximumRadius`.
 *
 * @returns `0` when the Gaussian cannot reach one 8-bit color step anywhere.
 */
export function getSplatSupportRadius(alpha: number, maximumRadius: number): number {
  if (!(alpha > MINIMUM_VISIBLE_ALPHA) || !(maximumRadius > 0)) {
    return 0;
  }
  return Math.min(Math.sqrt(2 * Math.log(255 * alpha)), maximumRadius);
}

/**
 * Logistic approximation of the standard normal CDF used by Analytic-Splatting.
 *
 * `S(x) = 1 / (1 + exp(-1.6x - 0.07x^3))` was fit against the Gaussian CDF. It is a fit, not an
 * identity: its derivative overshoots the standard normal density by 0.27% at the mean, and the
 * error stays under 0.3% out to four standard deviations. The constants are kept as published
 * rather than renormalized, because they were fit to minimize error over the integral rather than
 * at the peak.
 */
export function getSplatLogisticCdf(value: number): number {
  return 1 / (1 + Math.exp(-1.6 * value - 0.07 * value * value * value));
}

/**
 * Returns the mean of `exp(-t^2 / 2)` over a one-dimensional pixel footprint.
 *
 * Averaging rather than sampling at the center is the whole of Analytic-Splatting: the result
 * converges to `exp(-center^2 / 2)` as the footprint shrinks, so a Gaussian that is large
 * relative to a pixel is unaffected while a sub-pixel Gaussian stops aliasing.
 *
 * @param center Pixel center offset from the Gaussian mean, in standard deviations.
 * @param halfWidth Half the pixel footprint along this axis, in standard deviations.
 */
export function getSplatPixelIntegral(center: number, halfWidth: number): number {
  if (!(halfWidth > 0)) {
    return Math.exp(-0.5 * center * center);
  }
  const difference =
    getSplatLogisticCdf(center + halfWidth) - getSplatLogisticCdf(center - halfWidth);
  return (Math.sqrt(2 * Math.PI) * difference) / (2 * halfWidth);
}

/**
 * Shared WGSL helpers for dilation compensation, support radius, and the pixel integral.
 *
 * Kept in one string so the projection, render, picking and compatibility shaders cannot drift
 * from each other or from the CPU implementations above.
 *
 * @internal
 */
export const SPLAT_ANTIALIASING_WGSL = /* wgsl */ `\
const SPLAT_MINIMUM_COVARIANCE_DETERMINANT: f32 = 1e-12;
const SPLAT_MINIMUM_VISIBLE_ALPHA: f32 = 0.00392156862745098;
const SPLAT_SQRT_TWO_PI: f32 = 2.5066282746310002;

/** Mip-Splatting opacity compensation for an isotropically dilated 2D covariance. */
fn getSplatDilationCompensation(
  covariance00: f32,
  covariance01: f32,
  covariance11: f32,
  addedVariance: f32
) -> f32 {
  if (addedVariance <= 0.0) {
    return 1.0;
  }
  let originalDeterminant = covariance00 * covariance11 - covariance01 * covariance01;
  let dilatedDeterminant =
    (covariance00 + addedVariance) * (covariance11 + addedVariance) - covariance01 * covariance01;
  if (
    originalDeterminant <= SPLAT_MINIMUM_COVARIANCE_DETERMINANT ||
    dilatedDeterminant <= SPLAT_MINIMUM_COVARIANCE_DETERMINANT
  ) {
    return 1.0;
  }
  return min(sqrt(originalDeterminant / dilatedDeterminant), 1.0);
}

/** Energy-conserving compensation for a covariance uniformly rescaled by \`axisScale\`. */
fn getSplatClampCompensation(axisScale: f32) -> f32 {
  if (axisScale <= 0.0 || axisScale >= 1.0) {
    return 1.0;
  }
  return 1.0 / (axisScale * axisScale);
}

/** Standard deviations at which a Gaussian of this opacity falls below one 8-bit color step. */
fn getSplatSupportRadius(alpha: f32, maximumRadius: f32) -> f32 {
  if (alpha <= SPLAT_MINIMUM_VISIBLE_ALPHA || maximumRadius <= 0.0) {
    return 0.0;
  }
  return min(sqrt(2.0 * log(255.0 * alpha)), maximumRadius);
}

/** Logistic approximation of the standard normal CDF, fit by Analytic-Splatting. */
fn getSplatLogisticCdf(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-1.6 * value - 0.07 * value * value * value));
}

/** Mean of \`exp(-t^2 / 2)\` over one pixel footprint, in standard deviations. */
fn getSplatPixelIntegral(center: f32, halfWidth: f32) -> f32 {
  if (halfWidth <= 0.0) {
    return exp(-0.5 * center * center);
  }
  let difference = getSplatLogisticCdf(center + halfWidth) - getSplatLogisticCdf(center - halfWidth);
  return (SPLAT_SQRT_TWO_PI * difference) / (2.0 * halfWidth);
}
`;
