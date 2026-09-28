// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  DEFAULT_SPLAT_SCREEN_FILTER_VARIANCE,
  getSplatClampCompensation,
  getSplatDilationCompensation,
  getSplatLogisticCdf,
  getSplatPixelIntegral,
  getSplatScreenFilterVariance,
  getSplatSupportRadius
} from '../src/splat-antialiasing';

it('the screen-space filter defaults to the reference rasterizer variance, not its square', () => {
  expect(
    getSplatScreenFilterVariance({}),
    'the reference rasterizer adds 0.3 px^2 of variance'
  ).toBe(DEFAULT_SPLAT_SCREEN_FILTER_VARIANCE);
  expect(
    getSplatScreenFilterVariance({kernel2DSize: 0.3}),
    'the deprecated spelling names a standard deviation and is squared'
  ).toBeCloseTo(0.09, 6);
  expect(
    getSplatScreenFilterVariance({}) / getSplatScreenFilterVariance({kernel2DSize: 0.3}),
    'which under-filters sub-pixel Gaussians by a factor of about 3.3'
  ).toBeCloseTo(3.333, 2);
  expect(
    getSplatScreenFilterVariance({screenSpaceFilterVariance: 1, kernel2DSize: 0.3}),
    'the variance spelling wins when both are supplied'
  ).toBe(1);
  expect(getSplatScreenFilterVariance({kernel2DSize: -1}), 'negative widths clamp to zero').toBe(0);
});

it('dilation compensation shrinks opacity by exactly the determinant ratio', () => {
  // An isotropic Gaussian of variance s dilated by f has determinant (s+f)^2 rather than s^2.
  const variance = 4;
  const addedVariance = 0.3;
  const compensation = getSplatDilationCompensation(variance, 0, variance, addedVariance);

  expect(compensation, 'rho = sqrt(det original / det dilated)').toBeCloseTo(
    variance / (variance + addedVariance),
    6
  );
  expect(compensation < 1, 'dilation always removes opacity, never adds it').toBe(true);
  expect(
    getSplatDilationCompensation(variance, 0, variance, 0),
    'no filter means no compensation'
  ).toBe(1);
  expect(
    getSplatDilationCompensation(0, 0, 0, 0.3),
    'a degenerate covariance falls back to base 3DGS rather than vanishing'
  ).toBe(1);
});

it('a sub-pixel Gaussian is compensated far more than a large one', () => {
  const addedVariance = DEFAULT_SPLAT_SCREEN_FILTER_VARIANCE;
  const subPixel = getSplatDilationCompensation(0.01, 0, 0.01, addedVariance);
  const large = getSplatDilationCompensation(100, 0, 100, addedVariance);

  expect(subPixel < 0.1, 'a sub-pixel Gaussian loses almost all of its dilated energy').toBe(true);
  expect(large > 0.99, 'a Gaussian far larger than a pixel is untouched').toBe(true);
});

it('the screen-size clamp conserves integrated energy instead of distorting silently', () => {
  expect(getSplatClampCompensation(0.5), 'halving both axes quarters the area').toBe(4);
  expect(getSplatClampCompensation(1), 'an unclamped Gaussian is unchanged').toBe(1);
  expect(getSplatClampCompensation(2), 'the clamp only ever shrinks').toBe(1);
});

it('the support radius follows opacity down to one 8-bit color step', () => {
  expect(getSplatSupportRadius(1, 3), 'a fully opaque Gaussian reaches the 3-sigma cutoff').toBe(3);
  expect(
    getSplatSupportRadius(0.1, 3),
    'a tenth-opacity Gaussian needs only about 2.5 sigma'
  ).toBeCloseTo(Math.sqrt(2 * Math.log(25.5)), 6);
  expect(
    getSplatSupportRadius(1 / 255, 3),
    'a Gaussian that never reaches one color step has no support at all'
  ).toBe(0);
  // The cutoff binds only below alpha = exp(4.5) / 255 ~= 0.353, which is still the large
  // majority of Gaussians in a trained scene.
  expect(getSplatSupportRadius(0.36, 3), 'just above the crossover the 3-sigma cutoff binds').toBe(
    3
  );
  expect(
    getSplatSupportRadius(0.34, 3) < 3,
    'just below it the quad starts shrinking with opacity'
  ).toBe(true);
});

it('the logistic CDF approximates the Gaussian integral closely enough to replace it', () => {
  expect(getSplatLogisticCdf(0), 'the standard normal CDF is a half at the mean').toBeCloseTo(
    0.5,
    6
  );
  expect(getSplatLogisticCdf(3) > 0.998, 'and saturates by three sigma').toBe(true);
  expect(getSplatLogisticCdf(-3) < 0.002, 'symmetrically in both directions').toBe(true);
});

it('the pixel integral converges to the point sample as the footprint shrinks', () => {
  // The logistic CDF is a fit, not an identity, so the limit reproduces the Gaussian to within
  // the fit's own error: about 0.27% at the peak and under 0.6% anywhere inside three sigma.
  let worstError = 0;
  for (let center = 0; center <= 4; center += 0.01) {
    const approximated = getSplatPixelIntegral(center, 1e-5);
    worstError = Math.max(worstError, Math.abs(approximated - Math.exp(-0.5 * center * center)));
  }
  expect(worstError < 0.006, 'the fit stays within 0.6% of the Gaussian it replaces').toBe(true);
  expect(
    getSplatPixelIntegral(0, 1e-5),
    "the peak carries the fit's small positive bias"
  ).toBeCloseTo(1, 2);
});

it('the pixel integral is what stops a sub-pixel Gaussian aliasing', () => {
  // A Gaussian one tenth of a pixel wide is sampled at 10 sigma per pixel step: point sampling
  // either hits its peak or misses it entirely, which is exactly the aliasing to remove.
  const halfWidth = 5;
  const atCenter = getSplatPixelIntegral(0, halfWidth);
  const offCenter = getSplatPixelIntegral(4, halfWidth);

  expect(atCenter < 0.3, 'averaging over the pixel spreads the peak out').toBe(true);
  expect(
    offCenter > Math.exp(-0.5 * 16),
    'and lifts a sample that point sampling would have missed'
  ).toBe(true);
  expect(
    Math.abs(atCenter - offCenter) < Math.abs(1 - Math.exp(-0.5 * 16)),
    'so neighboring pixels differ far less than their point samples do'
  ).toBe(true);
});
