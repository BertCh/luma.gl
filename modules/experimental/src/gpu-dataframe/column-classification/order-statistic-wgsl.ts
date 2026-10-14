// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL of the one order-statistic policy shared by `GPUColumnQuantiles`, `GPUClassBreaks`
 * (`'quantile'` and `'box-plot'`) and the `GPUGroupStatistics` medians and percentiles.
 *
 * Policy (finding B-14; the tables in `test/gpu-dataframe/dataframe-edge-policy.ts` pin it):
 * - The sample is the finite values of the selected rows. NaN and +-Infinity are missing (rule 8:
 *   a kernel that interpolates treats non-finite as missing), so the result equals
 *   `numpy.quantile(x[numpy.isfinite(x)], p)`. numpy and pandas keep infinities; this is the one
 *   documented deviation.
 * - `-0` and `+0` are one value (numpy orders them as equal); a result that is zero is `+0`.
 * - A probability that is NaN or outside `[0, 1]` gives NaN (numpy raises; a GPU kernel cannot).
 * - An empty sample gives NaN.
 * - `'linear'` is R-7 (numpy, pandas and d3 default): `h = fround((n - 1) * p)`, then
 *   `x[floor h] + (x[floor h + 1] - x[floor h]) * (h - floor h)`, every step rounded to f32 with
 *   no fused multiply-add, so a CPU `Math.fround` mirror is bit exact. When the difference
 *   overflows (values of opposite sign near the f32 limit) the convex form
 *   `x[lo] * (1 - t) + x[hi] * t` is used instead, which stays finite.
 *
 * Functions: `isOrderStatisticProbability(p)`, `getOrderStatisticKey(value)` (canonical `-0`,
 * order-preserving u32; only for finite values), `interpolateOrderStatistic(low, high, fraction,
 * zero)` and `getOrderStatisticMidpoint(low, high, zero)`. `zero` must be a runtime `0u` the
 * compiler cannot see (for example `count >> 31u` with `count < 2^31`); it blocks multiply-add
 * fusion.
 *
 * @internal
 */
export const ORDER_STATISTIC_WGSL = /* wgsl */ `
fn isOrderStatisticProbability(p: f32) -> bool {
  // Ordered comparisons with NaN are false, but a compiler may fold them; test the bits.
  let bits = bitcast<u32>(p);
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    return false;
  }
  return p >= 0.0 && p <= 1.0;
}

fn getOrderStatisticKey(value: f32) -> u32 {
  var bits = bitcast<u32>(value);
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

// Identity the compiler cannot fold: the bit operation on a runtime zero stops multiply-add
// fusion, so every f32 step rounds exactly like a Math.fround mirror.
fn roundOrderStatisticStep(value: f32, zero: u32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) | zero);
}

fn interpolateOrderStatistic(low: f32, high: f32, fraction: f32, zero: u32) -> f32 {
  if (fraction == 0.0 || low == high) {
    return low;
  }
  let difference = roundOrderStatisticStep(high - low, zero);
  if ((bitcast<u32>(difference) & 0x7f800000u) != 0x7f800000u) {
    return low + roundOrderStatisticStep(difference * fraction, zero);
  }
  // high - low overflowed; a convex combination of two finite values cannot.
  return roundOrderStatisticStep(low * (1.0 - fraction), zero) +
    roundOrderStatisticStep(high * fraction, zero);
}

fn getOrderStatisticMidpoint(low: f32, high: f32, zero: u32) -> f32 {
  if (low == high) {
    return low;
  }
  let sum = roundOrderStatisticStep(low + high, zero);
  if ((bitcast<u32>(sum) & 0x7f800000u) != 0x7f800000u) {
    return sum * 0.5;
  }
  return roundOrderStatisticStep(low * 0.5, zero) + roundOrderStatisticStep(high * 0.5, zero);
}
`;
