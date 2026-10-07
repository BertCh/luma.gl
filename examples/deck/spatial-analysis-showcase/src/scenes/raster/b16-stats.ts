// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Small CPU reference implementations of the trend statistics the `GPUChangeDetection` contributor
 * computes per cell (Mann-Kendall with tie correction, Theil-Sen, Welch t-test). They exist to
 * show GPU agreement on a sample of pixels; they run in float64 on the same float32 inputs.
 */

/** Result of the CPU trend statistics for one time series. */
export type TrendReference = {
  validCount: number;
  mannKendallS: number;
  mannKendallZ: number;
  mannKendallP: number;
  senSlope: number;
  tStatistic: number;
  tPValue: number;
};

/** Complementary error function (Numerical Recipes Chebyshev fit, fractional error below 1.2e-7). */
export function erfc(value: number): number {
  const z = Math.abs(value);
  const t = 1 / (1 + 0.5 * z);
  const answer =
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
  return value >= 0 ? answer : 2 - answer;
}

function logGamma(x: number): number {
  const coefficients = [
    76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
    0.1208650973866179e-2, -0.5395239384953e-5
  ];
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let series = 1.000000000190015;
  for (const coefficient of coefficients) series += coefficient / ++y;
  return -tmp + Math.log((2.5066282746310005 * series) / x);
}

function betaContinuedFraction(a: number, b: number, x: number): number {
  const tiny = 1e-30;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 200; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-12) break;
  }
  return h;
}

/** Regularized incomplete beta function `I_x(a, b)`. */
export function incompleteBeta(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(
    logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x)
  );
  if (x < (a + 1) / (a + b + 2)) return (front * betaContinuedFraction(a, b, x)) / a;
  return 1 - (front * betaContinuedFraction(b, a, 1 - x)) / b;
}

/**
 * Trend statistics of one series. `series[i]` is the value of slice `i` (NaN is missing). The
 * Welch groups are slices `[0, splitSlice)` and `[splitSlice, length)`.
 */
export function computeTrendReference(
  series: ArrayLike<number>,
  splitSlice: number
): TrendReference {
  const length = series.length;
  const indices: number[] = [];
  const values: number[] = [];
  for (let slice = 0; slice < length; slice++) {
    if (Number.isFinite(series[slice])) {
      indices.push(slice);
      values.push(series[slice]);
    }
  }
  const n = values.length;
  let s = 0;
  const slopes: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      s += Math.sign(values[j] - values[i]);
      slopes.push((values[j] - values[i]) / (indices[j] - indices[i]));
    }
  }
  const tieCounts = new Map<number, number>();
  for (const value of values) tieCounts.set(value, (tieCounts.get(value) ?? 0) + 1);
  let tieTerm = 0;
  for (const count of tieCounts.values()) tieTerm += count * (count - 1) * (2 * count + 5);
  const variance = (n * (n - 1) * (2 * n + 5) - tieTerm) / 18;
  let z = Number.NaN;
  let p = Number.NaN;
  if (n >= 2) {
    if (variance > 0) {
      z = (s - Math.sign(s)) / Math.sqrt(variance);
      p = erfc(Math.abs(z) / Math.SQRT2);
    } else {
      z = 0;
      p = 1;
    }
  }
  let sen = Number.NaN;
  if (slopes.length > 0) {
    slopes.sort((a, b) => a - b);
    const middle = slopes.length >> 1;
    sen = slopes.length % 2 ? slopes[middle] : (slopes[middle - 1] + slopes[middle]) / 2;
  }
  const before = values.filter((_, index) => indices[index] < splitSlice);
  const after = values.filter((_, index) => indices[index] >= splitSlice);
  let t = Number.NaN;
  let tp = Number.NaN;
  if (before.length >= 2 && after.length >= 2) {
    const mean = (list: number[]) => list.reduce((a, b) => a + b, 0) / list.length;
    const sampleVariance = (list: number[], m: number) =>
      list.reduce((a, b) => a + (b - m) * (b - m), 0) / (list.length - 1);
    const meanBefore = mean(before);
    const meanAfter = mean(after);
    const varianceBefore = sampleVariance(before, meanBefore) / before.length;
    const varianceAfter = sampleVariance(after, meanAfter) / after.length;
    const standardError = Math.sqrt(varianceBefore + varianceAfter);
    if (standardError > 0) {
      t = (meanAfter - meanBefore) / standardError;
      const df =
        (varianceBefore + varianceAfter) ** 2 /
        (varianceBefore ** 2 / (before.length - 1) + varianceAfter ** 2 / (after.length - 1));
      tp = incompleteBeta(df / 2, 0.5, df / (df + t * t));
    }
  }
  return {
    validCount: n,
    mannKendallS: s,
    mannKendallZ: z,
    mannKendallP: p,
    senSlope: sen,
    tStatistic: t,
    tPValue: tp
  };
}
