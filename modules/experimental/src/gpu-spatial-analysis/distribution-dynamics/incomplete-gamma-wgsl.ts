// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Iteration cap shared by the series and the continued fraction of {@link GPU_UPPER_INCOMPLETE_GAMMA_WGSL}. */
export const GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS = 500;

/**
 * WGSL for the chi-square survival function `P(X > value)` for any positive (integer or real)
 * degrees of freedom, as the regularized upper incomplete gamma `Q(dof / 2, value / 2)`.
 *
 * Declares `getLogGammaValue(z: f32) -> f32` (Stirling series after an upward shift, `z > 0`) and
 * `getChiSquareSurvivalValue(value: f32, degreesOfFreedom: f32) -> f32`. The survival function
 * uses the power series for `x < a + 1` and a modified Lentz continued fraction otherwise, each
 * capped at {@link GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS}. It returns 1 for `value <= 0`
 * and a NaN for a NaN value or `degreesOfFreedom <= 0`.
 *
 * Accuracy (f32, checked against `scipy.stats.chi2.sf` for 1 to 500 degrees of freedom): relative
 * error below 6e-5 for p-values above 1e-6 and absolute error below 1.5e-5 for p-values above
 * 1e-3. The error is dominated by f32 cancellation in `-x + a ln x - lnGamma(a)`, so it grows
 * roughly with `6e-8 * max(x, a ln x)`; p-values below the f32 normal range flush to 0.
 *
 * The declaration requires `getNaN()` and `isNanBits(x)` (`GET_NAN_WGSL`, `IS_NAN_BITS_WGSL` of
 * `utils/wgsl-common.ts`) to be declared once in the same module. It is kept here because it
 * belongs in `utils/wgsl-common.ts` next to those helpers; move it there when a second consumer
 * appears.
 *
 * @internal
 */
export const GPU_UPPER_INCOMPLETE_GAMMA_WGSL = /* wgsl */ `
fn getLogGammaValue(z: f32) -> f32 {
  var shifted = z;
  var product = 1.0;
  for (var step = 0u; step < 8u; step++) {
    if (shifted >= 8.0) {
      break;
    }
    product = product * shifted;
    shifted = shifted + 1.0;
  }
  let inverse = 1.0 / shifted;
  let inverseSquared = inverse * inverse;
  let series = inverse * (1.0 / 12.0 - inverseSquared * (1.0 / 360.0 - inverseSquared * (1.0 / 1260.0)));
  return (shifted - 0.5) * log(shifted) - shifted + 0.9189385332 + series - log(product);
}

fn getChiSquareSurvivalValue(value: f32, degreesOfFreedom: f32) -> f32 {
  if (isNanBits(value) || !(degreesOfFreedom > 0.0)) {
    return getNaN();
  }
  if (!(value > 0.0)) {
    return 1.0;
  }
  let a = degreesOfFreedom * 0.5;
  let x = value * 0.5;
  let logPrefactor = -x + a * log(x) - getLogGammaValue(a);
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= ${GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS}u; iteration++) {
      term = term * x / (a + f32(iteration));
      sum = sum + term;
      if (term < sum * 1.0e-7) {
        break;
      }
    }
    return clamp(1.0 - sum * exp(logPrefactor), 0.0, 1.0);
  }
  let tiny = 1.0e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1u; iteration <= ${GPU_UPPER_INCOMPLETE_GAMMA_MAXIMUM_ITERATIONS}u; iteration++) {
    let stepIndex = f32(iteration);
    let an = -stepIndex * (stepIndex - a);
    b = b + 2.0;
    d = an * d + b;
    if (abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < 2.0e-7) {
      break;
    }
  }
  return clamp(exp(logPrefactor) * h, 0.0, 1.0);
}
`;
