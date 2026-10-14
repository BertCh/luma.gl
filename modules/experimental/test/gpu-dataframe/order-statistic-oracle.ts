// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU mirror of the one order-statistic policy (`ORDER_STATISTIC_WGSL`) shared by
 * `GPUColumnQuantiles`, `GPUClassBreaks` and the `GPUGroupStatistics` medians and percentiles.
 * Every step is rounded to f32 with `Math.fround`, exactly as the kernels round it (they block
 * multiply-add fusion), so results are bit exact. The policy rows live in
 * `dataframe-edge-policy.ts`.
 */

const fround = Math.fround;

/** Whether `p` is a valid probability: finite and in `[0, 1]`. NaN and out-of-range give NaN. */
export function isOrderStatisticProbability(p: number): boolean {
  const value = fround(p);
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

/**
 * The order-statistic sample: finite unmasked values rounded to f32, `-0` canonicalised to `+0`,
 * sorted ascending. NaN and +-Infinity are missing.
 */
export function getOrderStatisticSample(
  values: ArrayLike<number>,
  mask?: ArrayLike<number>
): number[] {
  const sample: number[] = [];
  for (let row = 0; row < values.length; row++) {
    const value = fround(values[row]);
    if (Number.isFinite(value) && (!mask || mask[row] !== 0)) {
      sample.push(value === 0 ? 0 : value);
    }
  }
  return sample.sort((left, right) => left - right);
}

/** `low + (high - low) * fraction` in f32 steps, with the convex form when the difference overflows. */
export function interpolateOrderStatistic(low: number, high: number, fraction: number): number {
  if (fraction === 0 || low === high) {
    return low;
  }
  const difference = fround(high - low);
  if (Number.isFinite(difference)) {
    return fround(low + fround(difference * fraction));
  }
  return fround(fround(low * fround(1 - fraction)) + fround(high * fraction));
}

/** `(low + high) / 2` in f32 steps, halving first when the sum overflows. */
export function getOrderStatisticMidpoint(low: number, high: number): number {
  if (low === high) {
    return low;
  }
  const sum = fround(low + high);
  if (Number.isFinite(sum)) {
    return fround(sum * 0.5);
  }
  return fround(fround(low * 0.5) + fround(high * 0.5));
}

/**
 * R-7 linear quantile (numpy, pandas and d3 default) of an ascending order-statistic sample:
 * `h = fround(fround(n - 1) * p)`, then {@link interpolateOrderStatistic}. NaN for an empty
 * sample or an invalid probability.
 */
export function getLinearOrderStatistic(sorted: readonly number[], p: number): number {
  const count = sorted.length;
  if (count === 0 || !isOrderStatisticProbability(p)) {
    return NaN;
  }
  const height = fround(fround(count - 1) * fround(p));
  const lowIndex = Math.min(Math.floor(height), count - 1);
  const low = sorted[lowIndex];
  if (lowIndex + 1 >= count) {
    return low;
  }
  return interpolateOrderStatistic(low, sorted[lowIndex + 1], fround(height - Math.floor(height)));
}
