// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Natural log of the gamma function at `degreesOfFreedom / 2` for a positive integer
 * `degreesOfFreedom`, by the exact half-integer recurrence `Gamma(a + 1) = a Gamma(a)` starting
 * from `Gamma(1) = 1` or `Gamma(1/2) = sqrt(pi)`.
 */
export function getLogGammaOfHalfInteger(degreesOfFreedom: number): number {
  if (!Number.isInteger(degreesOfFreedom) || degreesOfFreedom < 1) {
    throw new Error('Chi-square degrees of freedom must be a positive integer');
  }
  const target = degreesOfFreedom / 2;
  let argument = degreesOfFreedom % 2 === 0 ? 1 : 0.5;
  let logGamma = degreesOfFreedom % 2 === 0 ? 0 : 0.5 * Math.log(Math.PI);
  while (argument < target) {
    logGamma += Math.log(argument);
    argument += 1;
  }
  return logGamma;
}

/** Fixed iteration cap shared by the series and the continued fraction, GPU and CPU. */
export const CHI_SQUARE_MAXIMUM_ITERATIONS = 200;

/**
 * Survival function `P(X > value)` of a chi-square distribution with `degreesOfFreedom` degrees of
 * freedom, computed as the regularized upper incomplete gamma `Q(df / 2, value / 2)`.
 *
 * Uses the power series for `x < a + 1` and the modified Lentz continued fraction otherwise, each
 * capped at `CHI_SQUARE_MAXIMUM_ITERATIONS` iterations: the same algorithm as the WGSL in
 * `GPUOrdinaryLeastSquares`, in float64. Returns NaN for a NaN value and 1 for `value <= 0`.
 */
export function getChiSquareSurvival(value: number, degreesOfFreedom: number): number {
  const logGamma = getLogGammaOfHalfInteger(degreesOfFreedom);
  if (Number.isNaN(value)) {
    return NaN;
  }
  if (!(value > 0)) {
    return 1;
  }
  const a = degreesOfFreedom / 2;
  const x = value / 2;
  const logPrefactor = -x + a * Math.log(x) - logGamma;
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let iteration = 1; iteration <= CHI_SQUARE_MAXIMUM_ITERATIONS; iteration++) {
      term *= x / (a + iteration);
      sum += term;
      if (term < sum * 1e-15) {
        break;
      }
    }
    return Math.min(1, Math.max(0, 1 - sum * Math.exp(logPrefactor)));
  }
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let iteration = 1; iteration <= CHI_SQUARE_MAXIMUM_ITERATIONS; iteration++) {
    const an = -iteration * (iteration - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (Math.abs(c) < tiny) {
      c = tiny;
    }
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) {
      break;
    }
  }
  return Math.min(1, Math.max(0, Math.exp(logPrefactor) * h));
}
