// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Fixed continued-fraction iteration cap of the incomplete beta function (early exit allowed). */
export const BETA_CONTINUED_FRACTION_ITERATIONS = 64;

/** Convergence threshold of the continued fraction, `|delta - 1|`. */
export const BETA_CONTINUED_FRACTION_TOLERANCE = 3e-7;

/** NaN helpers shared by every change-detection kernel. */
export const NAN_WGSL = /* wgsl */ `
fn quietNan() -> f32 {
  // A runtime value: WGSL rejects constant-expression NaN.
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn isNanValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}
`;

/**
 * Student-t two-sided p-value via the regularized incomplete beta function.
 *
 * `p = I_x(df / 2, 1 / 2)` with `x = df / (df + t^2)` (Numerical Recipes `betai` / `betacf`),
 * with a Stirling-series log-gamma (argument shifted up to at least 8 in a fixed 8-step loop).
 * The CPU oracle in the tests mirrors these formulas in float64.
 */
export const STUDENT_T_WGSL = /* wgsl */ `
fn logGammaPositive(argument: f32) -> f32 {
  var x = argument;
  var shiftProduct = 1.0;
  for (var step = 0u; step < 8u; step++) {
    if (x < 8.0) {
      shiftProduct *= x;
      x += 1.0;
    }
  }
  let inverse = 1.0 / x;
  let inverseSquared = inverse * inverse;
  let series = inverse * (1.0 / 12.0 - inverseSquared * (1.0 / 360.0 - inverseSquared * (1.0 / 1260.0)));
  return (x - 0.5) * log(x) - x + 0.9189385332 + series - log(shiftProduct);
}

fn betaContinuedFraction(a: f32, b: f32, x: f32) -> f32 {
  let tiny = 1e-30;
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  var c = 1.0;
  var d = 1.0 - qab * x / qap;
  if (abs(d) < tiny) { d = tiny; }
  d = 1.0 / d;
  var h = d;
  for (var m = 1u; m <= ${BETA_CONTINUED_FRACTION_ITERATIONS}u; m++) {
    let mf = f32(m);
    let m2 = 2.0 * mf;
    var aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (abs(d) < tiny) { d = tiny; }
    c = 1.0 + aa / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (abs(d) < tiny) { d = tiny; }
    c = 1.0 + aa / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < ${BETA_CONTINUED_FRACTION_TOLERANCE}) {
      break;
    }
  }
  return h;
}

fn studentTTwoSidedP(t: f32, degreesOfFreedom: f32) -> f32 {
  let t2 = t * t;
  if (t2 == 0.0) {
    return 1.0;
  }
  if (t2 > 1e30) {
    return 0.0;
  }
  let a = 0.5 * degreesOfFreedom;
  let b = 0.5;
  let y = t2 / (degreesOfFreedom + t2);
  let x = degreesOfFreedom / (degreesOfFreedom + t2);
  let logX = -log(1.0 + t2 / degreesOfFreedom);
  let logFront = a * logX + b * log(y)
    - (logGammaPositive(a) + logGammaPositive(b) - logGammaPositive(a + b));
  var incomplete: f32;
  if (x < (a + 1.0) / (a + b + 2.0)) {
    incomplete = exp(logFront) * betaContinuedFraction(a, b, x) / a;
  } else {
    incomplete = 1.0 - exp(logFront) * betaContinuedFraction(b, a, y) / b;
  }
  return clamp(incomplete, 0.0, 1.0);
}
`;

/**
 * Complementary error function (Numerical Recipes `erfcc`, fractional error below 1.2e-7).
 * The two-sided normal p-value of z is `erfc(|z| / sqrt(2))`.
 */
export const ERFC_WGSL = /* wgsl */ `
fn complementaryError(value: f32) -> f32 {
  let z = abs(value);
  let t = 1.0 / (1.0 + 0.5 * z);
  let polynomial = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
    + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
    + t * (-0.82215223 + t * 0.17087277))))))));
  let result = t * exp(polynomial);
  return select(2.0 - result, result, value >= 0.0);
}
`;
