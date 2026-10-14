// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Student t two-sided tail probability and critical value, as WGSL for the GWR significance mask
 * and as a float64 CPU mirror with the identical algorithm.
 *
 * The two-sided tail of a t variate with `df` degrees of freedom is the regularized incomplete
 * beta `I_x(df / 2, 1 / 2)` with `x = df / (df + t^2)`. It is evaluated with the Lentz continued
 * fraction (Numerical Recipes `betacf`), on the side of the symmetry relation where the fraction
 * converges quickly. The beta-function prefactor needs `Gamma(a + 1/2) / Gamma(a)`; it is formed
 * from a recurrence up to `a >= 10` and the asymptotic series there, so no `lgamma` difference of
 * large numbers appears (a plain `lgamma(a + 1/2) - lgamma(a)` would lose all float32 digits for
 * `df` in the thousands). `x^a` uses `log1p(t^2 / df)` for the same reason. The critical value
 * inverts the tail by doubling and bisection (the tail is monotone in `t`).
 *
 * Degrees of freedom above `GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM` are clamped: the quantile
 * differs from the normal one by about `(z^3 + z) / (4 df)` there, below 2e-5, which is under
 * float32 resolution.
 */

/** Largest degrees of freedom used; larger values are clamped (see above). */
export const GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM = 200000;

/** Continued-fraction iteration cap; the fraction needs about `sqrt(df)` iterations. */
const MAXIMUM_ITERATIONS = 3000;

/** Bisection steps; the bracket is at most a factor 2 wide before them. */
const BISECTION_COUNT = 40;

/**
 * Returns WGSL declaring `gwrStudentTwoSidedTail(t, df)` and `gwrStudentCriticalValue(tail, df)`.
 *
 * - `gwrStudentTwoSidedTail(t: f32, df: f32) -> f32` is `P(|T| >= t)`.
 * - `gwrStudentCriticalValue(tail: f32, df: f32) -> f32` is the `t >= 0` whose two-sided tail is
 *   `tail`: 0 for `tail >= 1`, NaN for a non-positive or non-finite `tail` or `df < 1`, and
 *   `scipy.stats.t.ppf(1 - tail / 2, df)` otherwise.
 *
 * The declaration uses only `log`, `exp`, `sqrt` and arithmetic, plus `getNaN()` (`GET_NAN_WGSL`),
 * which the including kernel must declare.
 */
export function getStudentTWGSL(): string {
  return /* wgsl */ `
const GWR_T_EPSILON: f32 = 3.0e-7;
const GWR_T_MINIMUM: f32 = 1.0e-30;
const GWR_T_SQRT_PI: f32 = 1.7724538509055159;
const GWR_T_MAXIMUM_DF: f32 = ${GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM}.0;

// log(1 + z) without the rounding of 1 + z (z small and positive).
fn gwrLog1p(z: f32) -> f32 {
  let u = 1.0 + z;
  if (u == 1.0) {
    return z;
  }
  return log(u) * z / (u - 1.0);
}

// Gamma(a + 1/2) / Gamma(a) for a > 0: recurrence up to a >= 10, then the asymptotic series.
fn gwrGammaHalfRatio(a: f32) -> f32 {
  var s = a;
  var factor = 1.0;
  for (var step = 0u; step < 12u; step++) {
    if (s >= 10.0) {
      break;
    }
    factor = factor * s / (s + 0.5);
    s = s + 1.0;
  }
  let inverse = 1.0 / s;
  let series = 1.0 - 0.125 * inverse +
    inverse * inverse * (1.0 / 128.0 + inverse * (5.0 / 1024.0 -
      inverse * (21.0 / 32768.0 + inverse * (399.0 / 262144.0 - inverse * (869.0 / 4194304.0)))));
  return factor * sqrt(s) * series;
}

// Lentz continued fraction of the incomplete beta function (Numerical Recipes betacf).
fn gwrBetaContinuedFraction(a: f32, b: f32, x: f32) -> f32 {
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  var c = 1.0;
  var d = 1.0 - qab * x / qap;
  if (abs(d) < GWR_T_MINIMUM) {
    d = GWR_T_MINIMUM;
  }
  d = 1.0 / d;
  var h = d;
  for (var m = 1u; m <= ${MAXIMUM_ITERATIONS}u; m++) {
    let fm = f32(m);
    let m2 = 2.0 * fm;
    var aa = fm * (b - fm) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (abs(d) < GWR_T_MINIMUM) {
      d = GWR_T_MINIMUM;
    }
    c = 1.0 + aa / c;
    if (abs(c) < GWR_T_MINIMUM) {
      c = GWR_T_MINIMUM;
    }
    d = 1.0 / d;
    h = h * d * c;
    aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (abs(d) < GWR_T_MINIMUM) {
      d = GWR_T_MINIMUM;
    }
    c = 1.0 + aa / c;
    if (abs(c) < GWR_T_MINIMUM) {
      c = GWR_T_MINIMUM;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < GWR_T_EPSILON) {
      break;
    }
  }
  return h;
}

// P(|T| >= t) for a Student t variate with df degrees of freedom.
fn gwrStudentTwoSidedTail(t: f32, degreesOfFreedom: f32) -> f32 {
  if (!(t > 0.0)) {
    return 1.0;
  }
  let df = min(degreesOfFreedom, GWR_T_MAXIMUM_DF);
  let a = 0.5 * df;
  let ratio = t * t / df;
  let x = 1.0 / (1.0 + ratio);
  let y = ratio / (1.0 + ratio);
  let front = exp(-a * gwrLog1p(ratio)) * sqrt(y) * gwrGammaHalfRatio(a) / GWR_T_SQRT_PI;
  if (x < (a + 1.0) / (a + 2.5)) {
    return front * gwrBetaContinuedFraction(a, 0.5, x) / a;
  }
  return 1.0 - 2.0 * front * gwrBetaContinuedFraction(0.5, a, y);
}

// The t >= 0 with P(|T| >= t) = tail.
fn gwrStudentCriticalValue(tail: f32, degreesOfFreedom: f32) -> f32 {
  if (!(tail > 0.0) || !(degreesOfFreedom >= 1.0) || tail > 3.0e38) {
    return getNaN();
  }
  if (tail >= 1.0) {
    return 0.0;
  }
  var low = 0.0;
  var high = 1.0;
  for (var step = 0u; step < 60u; step++) {
    if (gwrStudentTwoSidedTail(high, degreesOfFreedom) < tail) {
      break;
    }
    low = high;
    high = high * 2.0;
  }
  for (var step = 0u; step < ${BISECTION_COUNT}u; step++) {
    let middle = 0.5 * (low + high);
    if (gwrStudentTwoSidedTail(middle, degreesOfFreedom) < tail) {
      high = middle;
    } else {
      low = middle;
    }
  }
  return 0.5 * (low + high);
}
`;
}

function getGammaHalfRatio(a: number): number {
  let s = a;
  let factor = 1;
  for (let step = 0; step < 12 && s < 10; step++) {
    factor *= s / (s + 0.5);
    s += 1;
  }
  const inverse = 1 / s;
  const series =
    1 -
    0.125 * inverse +
    inverse *
      inverse *
      (1 / 128 +
        inverse *
          (5 / 1024 -
            inverse * (21 / 32768 + inverse * (399 / 262144 - inverse * (869 / 4194304)))));
  return factor * Math.sqrt(s) * series;
}

function getBetaContinuedFraction(a: number, b: number, x: number): number {
  const minimum = 1e-300;
  const epsilon = 1e-15;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < minimum) d = minimum;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIMUM_ITERATIONS; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < minimum) d = minimum;
    c = 1 + aa / c;
    if (Math.abs(c) < minimum) c = minimum;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < minimum) d = minimum;
    c = 1 + aa / c;
    if (Math.abs(c) < minimum) c = minimum;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < epsilon) {
      break;
    }
  }
  return h;
}

/**
 * Float64 CPU mirror of `gwrStudentTwoSidedTail`: `P(|T| >= t)` for `df` degrees of freedom
 * (clamped to `GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM`).
 */
export function getStudentTTwoSidedTail(t: number, degreesOfFreedom: number): number {
  if (!(t > 0)) {
    return 1;
  }
  const df = Math.min(degreesOfFreedom, GWR_STUDENT_T_MAXIMUM_DEGREES_OF_FREEDOM);
  const a = 0.5 * df;
  const ratio = (t * t) / df;
  const x = 1 / (1 + ratio);
  const y = ratio / (1 + ratio);
  const front =
    (Math.exp(-a * Math.log1p(ratio)) * Math.sqrt(y) * getGammaHalfRatio(a)) / Math.sqrt(Math.PI);
  if (x < (a + 1) / (a + 2.5)) {
    return (front * getBetaContinuedFraction(a, 0.5, x)) / a;
  }
  return 1 - 2 * front * getBetaContinuedFraction(0.5, a, y);
}

/**
 * Float64 CPU mirror of `gwrStudentCriticalValue`: the `t >= 0` with two-sided tail `tail`, that
 * is `scipy.stats.t.ppf(1 - tail / 2, df)`. Returns 0 for `tail >= 1` and NaN for a non-positive
 * or non-finite `tail` or `df < 1`.
 */
export function getStudentTCriticalValue(tail: number, degreesOfFreedom: number): number {
  if (!(tail > 0) || !(degreesOfFreedom >= 1) || !Number.isFinite(tail)) {
    return Number.NaN;
  }
  if (tail >= 1) {
    return 0;
  }
  let low = 0;
  let high = 1;
  for (let step = 0; step < 60; step++) {
    if (getStudentTTwoSidedTail(high, degreesOfFreedom) < tail) {
      break;
    }
    low = high;
    high *= 2;
  }
  for (let step = 0; step < 100; step++) {
    const middle = 0.5 * (low + high);
    if (getStudentTTwoSidedTail(middle, degreesOfFreedom) < tail) {
      high = middle;
    } else {
      low = middle;
    }
  }
  return 0.5 * (low + high);
}
