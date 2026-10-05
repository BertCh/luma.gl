// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_COMPOSITE_SCORE_AGGREGATION,
  GPU_COMPOSITE_SCORE_POWER_ITERATIONS,
  GPU_COMPOSITE_SCORE_SCALER,
  type GPUCompositeScoreSettings
} from '../../../src/gpu-dataframe/composite-indicators/composite-score-parameters';

/** Inputs of the float64 composite-score oracle. */
export type CompositeScoreOracleInput = {
  indicators: Float32Array;
  indicatorCount: number;
  mask?: Uint32Array;
  settings: GPUCompositeScoreSettings;
};

/** Float64 results of {@link computeCompositeScoreOnCPU}. */
export type CompositeScoreOracleResult = {
  score: Float64Array;
  scaled: Float64Array;
  columnStatistics: Float64Array;
  loadings: Float64Array;
  /** `[eigenvalue, explainedVarianceRatio]`. */
  principalComponent: [number, number];
};

/**
 * Float64 reference of `GPUCompositeScore`: same inclusion rule, scalers, aggregations and power
 * iteration (start vector, step count, sign rule) as the WGSL kernels.
 */
export function computeCompositeScoreOnCPU(
  input: CompositeScoreOracleInput
): CompositeScoreOracleResult {
  const {indicators, indicatorCount: d, mask, settings} = input;
  const rows = indicators.length / d;
  const scaler = GPU_COMPOSITE_SCORE_SCALER[settings.scaler ?? 'min-max'];
  const aggregation = GPU_COMPOSITE_SCORE_AGGREGATION[settings.aggregation ?? 'weighted-sum'];
  const epsilon = Math.fround(settings.epsilon ?? 1e-6);
  const weights = Array.from({length: d}, (_, c) =>
    Math.fround(c < settings.weights.length ? settings.weights[c] : 0)
  );
  const reversed = Array.from({length: d}, (_, c) =>
    settings.directions && c < settings.directions.length ? settings.directions[c] < 0 : false
  );
  const valid: number[] = [];
  for (let row = 0; row < rows; row++) {
    if (mask && mask[row] === 0) {
      continue;
    }
    let ok = true;
    for (let c = 0; c < d; c++) {
      ok = ok && Number.isFinite(indicators[row * d + c]);
    }
    if (ok) {
      valid.push(row);
    }
  }
  const n = valid.length;
  const statistics = new Float64Array(d * 4).fill(NaN);
  const means = new Float64Array(d);
  for (let c = 0; c < d; c++) {
    if (n === 0) {
      continue;
    }
    let minimum = Infinity;
    let maximum = -Infinity;
    let sum = 0;
    for (const row of valid) {
      const value = indicators[row * d + c];
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      sum += value;
    }
    means[c] = sum / n;
    let squares = 0;
    for (const row of valid) {
      squares += (indicators[row * d + c] - means[c]) ** 2;
    }
    statistics.set([minimum, maximum, means[c], Math.sqrt(squares / n)], c * 4);
  }
  const covariance = new Float64Array(d * d);
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < d; b++) {
      let sum = 0;
      for (const row of valid) {
        sum += (indicators[row * d + a] - means[a]) * (indicators[row * d + b] - means[b]);
      }
      covariance[a * d + b] = n > 0 ? sum / n : NaN;
    }
  }
  const ranks = new Float64Array(rows * d);
  for (let c = 0; c < d; c++) {
    const sorted = valid.map(row => indicators[row * d + c]).sort((x, y) => x - y);
    for (const row of valid) {
      const value = indicators[row * d + c];
      const first = lowerBound(sorted, value);
      const last = lowerBound(sorted, value, true) - 1;
      ranks[row * d + c] = n > 1 ? (first + last) / 2 / (n - 1) : 0.5;
    }
  }
  const {loadings, eigenvalue, ratio} = getPrincipalComponent(covariance, d, reversed);
  const isPrincipal = aggregation === 2;
  const score = new Float64Array(rows).fill(NaN);
  const scaled = new Float64Array(rows * d).fill(NaN);
  for (const row of valid) {
    let weightedSum = 0;
    let weightTotal = 0;
    let logSum = 0;
    let positiveTotal = 0;
    let projection = 0;
    for (let c = 0; c < d; c++) {
      const value = indicators[row * d + c];
      const [minimum, maximum, mean, deviation] = statistics.subarray(c * 4, c * 4 + 4);
      let s: number;
      if (isPrincipal || scaler === 1) {
        s = deviation > 0 ? (value - mean) / deviation : 0;
        s = reversed[c] ? -s : s;
      } else {
        s =
          scaler === 0
            ? maximum - minimum > 0
              ? (value - minimum) / (maximum - minimum)
              : 0
            : ranks[row * d + c];
        s = reversed[c] ? 1 - s : s;
      }
      scaled[row * d + c] = s;
      weightedSum += weights[c] * s;
      weightTotal += Math.abs(weights[c]);
      if (weights[c] > 0) {
        logSum += weights[c] * (s + epsilon > 0 ? Math.log(s + epsilon) : NaN);
        positiveTotal += weights[c];
      }
      projection += loadings[c] * s;
    }
    score[row] =
      aggregation === 0
        ? weightTotal > 0
          ? weightedSum / weightTotal
          : NaN
        : aggregation === 1
          ? positiveTotal > 0
            ? Math.exp(logSum / positiveTotal)
            : NaN
          : projection;
  }
  return {
    score,
    scaled,
    columnStatistics: statistics,
    loadings,
    principalComponent: [eigenvalue, ratio]
  };
}

/**
 * Power iteration on the correlation matrix of direction-adjusted z-scores, mirroring the WGSL
 * (start vector `1 / sqrt(d)`, fixed step count, non-negative sum sign rule).
 */
function getPrincipalComponent(
  covariance: Float64Array,
  d: number,
  reversed: boolean[]
): {loadings: Float64Array; eigenvalue: number; ratio: number} {
  const deviations = Array.from({length: d}, (_, c) =>
    covariance[c * d + c] > 0 ? Math.sqrt(covariance[c * d + c]) : 0
  );
  const nonConstant = deviations.filter(value => value > 0).length;
  const correlation = new Float64Array(d * d);
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < d; b++) {
      const scale = deviations[a] * deviations[b];
      const sign = (reversed[a] ? -1 : 1) * (reversed[b] ? -1 : 1);
      correlation[a * d + b] = scale > 0 ? (sign * covariance[a * d + b]) / scale : 0;
    }
  }
  let vector = new Float64Array(d).fill(1 / Math.sqrt(d));
  for (let iteration = 0; iteration < GPU_COMPOSITE_SCORE_POWER_ITERATIONS; iteration++) {
    const product = multiply(correlation, vector, d);
    const norm = Math.hypot(...product);
    if (!(norm > 0)) {
      break;
    }
    vector = product.map(value => value / norm);
  }
  const total = vector.reduce((sum, value) => sum + value, 0);
  let largest = 0;
  for (let c = 1; c < d; c++) {
    if (Math.abs(vector[c]) > Math.abs(vector[largest])) {
      largest = c;
    }
  }
  if (!(total > 0 || (total === 0 && vector[largest] >= 0))) {
    vector = vector.map(value => -value);
  }
  const product = multiply(correlation, vector, d);
  const eigenvalue = vector.reduce((sum, value, c) => sum + value * product[c], 0);
  return {
    loadings: vector,
    eigenvalue: nonConstant > 0 ? eigenvalue : NaN,
    ratio: nonConstant > 0 ? eigenvalue / nonConstant : NaN
  };
}

/** Cyclic Jacobi eigen-decomposition of a symmetric matrix; returns the largest eigenpair. */
export function getLargestEigenpairByJacobi(
  matrix: Float64Array,
  d: number
): {value: number; vector: Float64Array} {
  const a = Float64Array.from(matrix);
  const v = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    v[i * d + i] = 1;
  }
  for (let sweep = 0; sweep < 100; sweep++) {
    let offDiagonal = 0;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        offDiagonal += a[p * d + q] ** 2;
      }
    }
    if (offDiagonal < 1e-22) {
      break;
    }
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        if (Math.abs(a[p * d + q]) < 1e-300) {
          continue;
        }
        const theta = (a[q * d + q] - a[p * d + p]) / (2 * a[p * d + q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < d; k++) {
          const akp = a[k * d + p];
          const akq = a[k * d + q];
          a[k * d + p] = c * akp - s * akq;
          a[k * d + q] = s * akp + c * akq;
        }
        for (let k = 0; k < d; k++) {
          const apk = a[p * d + k];
          const aqk = a[q * d + k];
          a[p * d + k] = c * apk - s * aqk;
          a[q * d + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < d; k++) {
          const vkp = v[k * d + p];
          const vkq = v[k * d + q];
          v[k * d + p] = c * vkp - s * vkq;
          v[k * d + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < d; i++) {
    if (a[i * d + i] > a[best * d + best]) {
      best = i;
    }
  }
  const vector = Float64Array.from({length: d}, (_, k) => v[k * d + best]);
  return {value: a[best * d + best], vector};
}

function multiply(matrix: Float64Array, vector: Float64Array, d: number): Float64Array {
  const product = new Float64Array(d);
  for (let row = 0; row < d; row++) {
    let sum = 0;
    for (let column = 0; column < d; column++) {
      sum += matrix[row * d + column] * vector[column];
    }
    product[row] = sum;
  }
  return product;
}

function lowerBound(sorted: number[], value: number, inclusive = false): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (inclusive ? sorted[middle] <= value : sorted[middle] < value) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}
