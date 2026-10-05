// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Input of {@link detectChangeOnCPU}. */
export type ChangeDetectionCPUInput = {
  slices: Float32Array;
  mask?: Uint32Array;
  cellCount: number;
  sliceCount: number;
  bandCount?: number;
  beforeSlice: number;
  afterSlice: number;
  epsilon?: number;
  alpha?: number;
  splitSlice?: number;
  significanceSource?: 't-test' | 'mann-kendall';
};

/** Per-cell results of {@link detectChangeOnCPU}; NaN or the documented defaults when undefined. */
export type ChangeDetectionCPUResult = {
  difference: Float64Array;
  logRatio: Float64Array;
  percentChange: Float64Array;
  tStatistic: Float64Array;
  tDegreesOfFreedom: Float64Array;
  tPValue: Float64Array;
  senSlope: Float64Array;
  mannKendallS: Int32Array;
  mannKendallZ: Float64Array;
  mannKendallP: Float64Array;
  changeMagnitude: Float64Array;
  /** Angle for 2 bands, sign-pattern code (0xffffffff when undefined) for more. */
  changeDirection: Float64Array;
  significance: Uint32Array;
};

function logGammaPositive(argument: number): number {
  let x = argument;
  let shiftProduct = 1;
  for (let step = 0; step < 8; step++) {
    if (x < 8) {
      shiftProduct *= x;
      x += 1;
    }
  }
  const inverse = 1 / x;
  const inverseSquared = inverse * inverse;
  const series = inverse * (1 / 12 - inverseSquared * (1 / 360 - inverseSquared * (1 / 1260)));
  return (x - 0.5) * Math.log(x) - x + 0.9189385332 + series - Math.log(shiftProduct);
}

function betaContinuedFraction(a: number, b: number, x: number): number {
  const tiny = 1e-30;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 64; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 3e-7) break;
  }
  return h;
}

/** Two-sided Student-t p-value; mirrors the WGSL incomplete-beta formulation in float64. */
export function studentTTwoSidedPOnCPU(t: number, degreesOfFreedom: number): number {
  const t2 = t * t;
  if (t2 === 0) return 1;
  if (t2 > 1e30) return 0;
  const a = 0.5 * degreesOfFreedom;
  const b = 0.5;
  const y = t2 / (degreesOfFreedom + t2);
  const x = degreesOfFreedom / (degreesOfFreedom + t2);
  const logX = -Math.log(1 + t2 / degreesOfFreedom);
  const logFront =
    a * logX +
    b * Math.log(y) -
    (logGammaPositive(a) + logGammaPositive(b) - logGammaPositive(a + b));
  const incomplete =
    x < (a + 1) / (a + b + 2)
      ? (Math.exp(logFront) * betaContinuedFraction(a, b, x)) / a
      : 1 - (Math.exp(logFront) * betaContinuedFraction(b, a, y)) / b;
  return Math.min(Math.max(incomplete, 0), 1);
}

/** Complementary error function (Numerical Contributors erfcc); mirrors the WGSL helper. */
export function complementaryErrorOnCPU(value: number): number {
  const z = Math.abs(value);
  const t = 1 / (1 + 0.5 * z);
  const polynomial =
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
                            t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
  const result = t * Math.exp(polynomial);
  return value >= 0 ? result : 2 - result;
}

/**
 * CPU oracle for GPUChangeDetection. Float32 slice values are read exactly; statistics are
 * computed in float64 with the same algorithms as the kernels, so GPU results agree to float32
 * tolerance. Sen slope uses a full sort and picks the exact median.
 */
export function detectChangeOnCPU(input: ChangeDetectionCPUInput): ChangeDetectionCPUResult {
  const {cellCount, sliceCount, slices} = input;
  const bandCount = input.bandCount ?? 1;
  const epsilon = input.epsilon ?? 1e-6;
  const alpha = input.alpha ?? 0.05;
  const split = input.splitSlice ?? 1;
  const source = input.significanceSource ?? 't-test';
  const bandRows = cellCount * bandCount;
  const result: ChangeDetectionCPUResult = {
    difference: new Float64Array(bandRows).fill(NaN),
    logRatio: new Float64Array(bandRows).fill(NaN),
    percentChange: new Float64Array(bandRows).fill(NaN),
    tStatistic: new Float64Array(cellCount).fill(NaN),
    tDegreesOfFreedom: new Float64Array(cellCount).fill(NaN),
    tPValue: new Float64Array(cellCount).fill(NaN),
    senSlope: new Float64Array(cellCount).fill(NaN),
    mannKendallS: new Int32Array(cellCount),
    mannKendallZ: new Float64Array(cellCount).fill(NaN),
    mannKendallP: new Float64Array(cellCount).fill(NaN),
    changeMagnitude: new Float64Array(cellCount).fill(NaN),
    changeDirection: new Float64Array(cellCount).fill(bandCount === 2 ? NaN : 0xffffffff),
    significance: new Uint32Array(cellCount)
  };
  const valueAt = (cell: number, slice: number, band = 0) =>
    slices[(cell * sliceCount + slice) * bandCount + band];
  const beforeValid = input.beforeSlice >= 0 && input.beforeSlice < sliceCount;
  const afterValid = input.afterSlice >= 0 && input.afterSlice < sliceCount;
  for (let cell = 0; cell < cellCount; cell++) {
    if (input.mask && input.mask[cell] === 0) {
      continue;
    }
    if (beforeValid && afterValid) {
      let squaredNorm = 0;
      let code = 0;
      const differences: number[] = [];
      let complete = true;
      for (let band = 0; band < bandCount; band++) {
        const before = valueAt(cell, input.beforeSlice, band);
        const after = valueAt(cell, input.afterSlice, band);
        if (Number.isNaN(before) || Number.isNaN(after)) {
          complete = false;
          continue;
        }
        const row = cell * bandCount + band;
        result.difference[row] = after - before;
        if (after > -epsilon && before > -epsilon) {
          result.logRatio[row] = Math.log((after + epsilon) / (before + epsilon));
        }
        if (before !== 0) {
          result.percentChange[row] = (100 * (after - before)) / Math.abs(before);
        }
        const difference = after - before;
        differences.push(difference);
        squaredNorm += difference * difference;
        if (difference > 0) code |= 1 << band;
        else if (difference < 0) code |= 1 << (16 + band);
      }
      if (bandCount >= 2 && complete) {
        result.changeMagnitude[cell] = Math.sqrt(squaredNorm);
        result.changeDirection[cell] =
          bandCount === 2
            ? differences[0] !== 0 || differences[1] !== 0
              ? Math.atan2(differences[1], differences[0])
              : 0
            : code >>> 0;
      }
    }
    if (bandCount !== 1) {
      continue;
    }
    const values: number[] = [];
    for (let slice = 0; slice < sliceCount; slice++) {
      values.push(valueAt(cell, slice));
    }
    // Welch t-test.
    if (split >= 1 && split <= sliceCount - 1) {
      const group1 = values.slice(0, split).filter(value => !Number.isNaN(value));
      const group2 = values.slice(split).filter(value => !Number.isNaN(value));
      if (group1.length >= 2 && group2.length >= 2) {
        const mean = (group: number[]) =>
          group.reduce((sum, value) => sum + value, 0) / group.length;
        const mean1 = mean(group1);
        const mean2 = mean(group2);
        const squares = (group: number[], center: number) =>
          group.reduce((sum, value) => sum + (value - center) * (value - center), 0);
        const errorSquared1 = squares(group1, mean1) / (group1.length - 1) / group1.length;
        const errorSquared2 = squares(group2, mean2) / (group2.length - 1) / group2.length;
        const errorSquared = errorSquared1 + errorSquared2;
        const denominator =
          (errorSquared1 * errorSquared1) / (group1.length - 1) +
          (errorSquared2 * errorSquared2) / (group2.length - 1);
        if (errorSquared > 0 && denominator > 0) {
          const t = (mean2 - mean1) / Math.sqrt(errorSquared);
          const degreesOfFreedom = (errorSquared * errorSquared) / denominator;
          const pValue = studentTTwoSidedPOnCPU(t, degreesOfFreedom);
          result.tStatistic[cell] = t;
          result.tDegreesOfFreedom[cell] = degreesOfFreedom;
          result.tPValue[cell] = pValue;
          if (source === 't-test' && pValue < alpha) {
            result.significance[cell] = t > 0 ? 1 : 2;
          }
        }
      }
    }
    // Sen slope and Mann-Kendall over the valid slices.
    const valid = values
      .map((value, slice) => ({value, slice}))
      .filter(entry => !Number.isNaN(entry.value));
    if (valid.length >= 2) {
      const slopes: number[] = [];
      let score = 0;
      for (let i = 0; i < valid.length; i++) {
        for (let j = i + 1; j < valid.length; j++) {
          slopes.push((valid[j].value - valid[i].value) / (valid[j].slice - valid[i].slice));
          score += Math.sign(valid[j].value - valid[i].value);
        }
      }
      slopes.sort((left, right) => left - right);
      const middle = slopes.length >> 1;
      result.senSlope[cell] =
        slopes.length % 2 === 1 ? slopes[middle] : 0.5 * (slopes[middle - 1] + slopes[middle]);
      const n = valid.length;
      const groups = new Map<number, number>();
      for (const {value} of valid) {
        groups.set(value, (groups.get(value) ?? 0) + 1);
      }
      let tieTerm = 0;
      for (const size of groups.values()) {
        tieTerm += size * (size - 1) * (2 * size + 5);
      }
      const varianceNumerator = n * (n - 1) * (2 * n + 5) - tieTerm;
      let z = 0;
      let pValue = 1;
      if (varianceNumerator > 0) {
        z = (score - Math.sign(score)) / Math.sqrt(varianceNumerator / 18);
        pValue = complementaryErrorOnCPU(Math.abs(z) * Math.SQRT1_2);
      }
      result.mannKendallS[cell] = score;
      result.mannKendallZ[cell] = z;
      result.mannKendallP[cell] = pValue;
      if (source === 'mann-kendall' && pValue < alpha && score !== 0) {
        result.significance[cell] = score > 0 ? 1 : 2;
      }
    }
  }
  return result;
}
