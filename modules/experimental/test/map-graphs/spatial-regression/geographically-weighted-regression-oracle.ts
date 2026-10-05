// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/map-graphs/spatial-regression/geographically-weighted-regression-parameters';

/** Inputs of {@link computeGeographicallyWeightedRegressionOnCPU}. */
export type GeographicallyWeightedRegressionOracleInput = {
  /** Interleaved `x, y` positions. */
  positions: Float32Array;
  /** Row-major `row * predictorCount + column`. */
  predictors: Float32Array;
  predictorCount: number;
  response: Float32Array;
  mask?: Uint32Array;
  settings: GPUGeographicallyWeightedRegressionSettings;
  /** Largest adaptive `k`. Defaults to 128. */
  maximumNeighborCount?: number;
};

/** Float64 results mirroring the GPU outputs. */
export type GeographicallyWeightedRegressionOracleResult = {
  coefficients: number[];
  localR2: number[];
  fitted: number[];
  residuals: number[];
  hatDiagonal: number[];
  localStatus: number[];
  /** One AICc per ladder value, NaN for invalid candidates. */
  bandwidthScores: number[];
  selectedIndex: number;
  selectedValue: number;
  residualSumOfSquares: number;
  traceOfHat: number;
  aicc: number;
  rSquared: number;
  observationCount: number;
  hasValidCandidate: boolean;
};

type LocalFit = {
  ok: boolean;
  /** Centred coefficients: `beta[0]` is the fitted value, the rest are slopes. */
  beta: number[];
  hat: number;
  weightSum: number;
  weightedResponse: number;
};

/** Gaussian or bisquare weight of distance `d` at bandwidth `h`. */
export function getGeographicallyWeightedRegressionWeight(
  kernel: 'gaussian' | 'bisquare',
  distance: number,
  bandwidth: number
): number {
  const ratio = distance / bandwidth;
  if (kernel === 'bisquare') {
    if (distance >= bandwidth) {
      return 0;
    }
    return (1 - ratio * ratio) ** 2;
  }
  return Math.exp(-0.5 * ratio * ratio);
}

/** Solves the symmetric positive definite system by Cholesky in float64; undefined when singular. */
function solveSymmetric(
  matrix: number[][],
  rightHandSide: number[]
): {solution: number[]; inverse00: number} | undefined {
  const p = rightHandSide.length;
  const lower = matrix.map(row => row.slice());
  for (let column = 0; column < p; column++) {
    let diagonal = lower[column][column];
    for (let k = 0; k < column; k++) {
      diagonal -= lower[column][k] ** 2;
    }
    if (!(diagonal > 1e-12 * matrix[column][column])) {
      return undefined;
    }
    const pivot = Math.sqrt(diagonal);
    lower[column][column] = pivot;
    for (let row = column + 1; row < p; row++) {
      let value = lower[row][column];
      for (let k = 0; k < column; k++) {
        value -= lower[row][k] * lower[column][k];
      }
      lower[row][column] = value / pivot;
    }
  }
  const solve = (input: number[]): number[] => {
    const x = input.slice();
    for (let row = 0; row < p; row++) {
      for (let k = 0; k < row; k++) {
        x[row] -= lower[row][k] * x[k];
      }
      x[row] /= lower[row][row];
    }
    for (let row = p - 1; row >= 0; row--) {
      for (let k = row + 1; k < p; k++) {
        x[row] -= lower[k][row] * x[k];
      }
      x[row] /= lower[row][row];
    }
    return x;
  };
  const unit = new Array<number>(p).fill(0);
  unit[0] = 1;
  return {solution: solve(rightHandSide), inverse00: solve(unit)[0]};
}

/**
 * Float64 reference for `GPUGeographicallyWeightedRegression`: the same definitions (focal
 * centring, kernels, adaptive k-th distance including the row itself times 1.00001, hat diagonal,
 * AICc with `n ln(RSS/n)`, singular-candidate invalidation, argmin to the lowest index) evaluated
 * with plain loops.
 */
export function computeGeographicallyWeightedRegressionOnCPU(
  input: GeographicallyWeightedRegressionOracleInput
): GeographicallyWeightedRegressionOracleResult {
  const {positions, predictors, predictorCount, response, mask, settings} = input;
  const p = predictorCount + 1;
  const rowCount = response.length;
  const kernel = settings.kernel ?? 'bisquare';
  const isAdaptive = settings.bandwidthMode === 'adaptive';
  const maximumK =
    input.maximumNeighborCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT;
  const ladder = Array.from(settings.bandwidths);

  const valid: boolean[] = [];
  for (let row = 0; row < rowCount; row++) {
    let isValid = mask ? mask[row] !== 0 : true;
    isValid &&= Number.isFinite(positions[2 * row]) && Number.isFinite(positions[2 * row + 1]);
    isValid &&= Number.isFinite(response[row]);
    for (let column = 0; column < predictorCount; column++) {
      isValid &&= Number.isFinite(predictors[row * predictorCount + column]);
    }
    valid.push(isValid);
  }
  const distance = (a: number, b: number) =>
    Math.hypot(positions[2 * a] - positions[2 * b], positions[2 * a + 1] - positions[2 * b + 1]);

  const getBandwidth = (row: number, value: number): number => {
    if (!isAdaptive) {
      return value > 0 ? value : -1;
    }
    if (!(value >= 2) || value > maximumK) {
      return -1;
    }
    const k = Math.floor(value + 0.5);
    const distances: number[] = [];
    for (let other = 0; other < rowCount; other++) {
      if (valid[other]) {
        distances.push(distance(row, other));
      }
    }
    distances.sort((left, right) => left - right);
    if (k > distances.length) {
      return -1;
    }
    return distances[k - 1] * GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR;
  };

  const fitLocation = (row: number, bandwidth: number): LocalFit => {
    const a = Array.from({length: p}, () => new Array<number>(p).fill(0));
    const b = new Array<number>(p).fill(0);
    const z = new Array<number>(p).fill(0);
    z[0] = 1;
    for (let other = 0; other < rowCount; other++) {
      if (!valid[other]) {
        continue;
      }
      const weight = getGeographicallyWeightedRegressionWeight(
        kernel,
        distance(row, other),
        bandwidth
      );
      if (!(weight > 0)) {
        continue;
      }
      for (let column = 0; column < predictorCount; column++) {
        z[column + 1] =
          predictors[other * predictorCount + column] - predictors[row * predictorCount + column];
      }
      for (let r = 0; r < p; r++) {
        b[r] += weight * z[r] * response[other];
        for (let c = 0; c <= r; c++) {
          a[r][c] += weight * z[r] * z[c];
        }
      }
    }
    for (let r = 0; r < p; r++) {
      for (let c = r + 1; c < p; c++) {
        a[r][c] = a[c][r];
      }
    }
    const fit: LocalFit = {
      ok: false,
      beta: new Array<number>(p).fill(0),
      hat: 0,
      weightSum: a[0][0],
      weightedResponse: b[0]
    };
    if (!a.every((line, r) => line[r] > 0)) {
      return fit;
    }
    const solved = solveSymmetric(a, b);
    if (!solved) {
      return fit;
    }
    fit.ok = true;
    fit.beta = solved.solution;
    fit.hat = solved.inverse00;
    return fit;
  };

  const nan = Number.NaN;
  const includedCount = valid.filter(Boolean).length;
  let total = 0;
  let mean = 0;
  for (let row = 0; row < rowCount; row++) {
    if (valid[row]) {
      mean += response[row];
    }
  }
  mean /= Math.max(includedCount, 1);
  for (let row = 0; row < rowCount; row++) {
    if (valid[row]) {
      total += (response[row] - mean) ** 2;
    }
  }

  const bandwidthScores: number[] = [];
  const candidateSums: {residualSum: number; traceSum: number}[] = [];
  for (const value of ladder) {
    let residualSum = 0;
    let traceSum = 0;
    let failed = false;
    for (let row = 0; row < rowCount && !failed; row++) {
      if (!valid[row]) {
        continue;
      }
      const bandwidth = getBandwidth(row, value);
      const fit = bandwidth > 0 ? fitLocation(row, bandwidth) : undefined;
      if (!fit || !fit.ok) {
        failed = true;
        break;
      }
      residualSum += (response[row] - fit.beta[0]) ** 2;
      traceSum += fit.hat;
    }
    const denominator = includedCount - 2 - traceSum;
    if (failed || includedCount === 0 || !(denominator > 0)) {
      bandwidthScores.push(nan);
      candidateSums.push({residualSum: nan, traceSum: nan});
      continue;
    }
    const variance = Math.max(
      residualSum / includedCount,
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE
    );
    bandwidthScores.push(
      includedCount * Math.log(variance) +
        includedCount * Math.log(2 * Math.PI) +
        (includedCount * (includedCount + traceSum)) / denominator
    );
    candidateSums.push({residualSum, traceSum});
  }
  let selectedIndex = 0;
  let hasValidCandidate = false;
  bandwidthScores.forEach((score, index) => {
    if (Number.isFinite(score) && (!hasValidCandidate || score < bandwidthScores[selectedIndex])) {
      hasValidCandidate = true;
      selectedIndex = index;
    }
  });
  const selectedValue = ladder[selectedIndex];

  const coefficients: number[] = [];
  const localR2: number[] = [];
  const fitted: number[] = [];
  const residuals: number[] = [];
  const hatDiagonal: number[] = [];
  const localStatus: number[] = [];
  const STATUS = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS;
  for (let row = 0; row < rowCount; row++) {
    let status: number = STATUS.SINGULAR;
    let r2 = nan;
    let fittedValue = nan;
    let residual = nan;
    let hat = nan;
    const rowCoefficients = new Array<number>(p).fill(nan);
    if (!valid[row]) {
      status = STATUS.EXCLUDED;
    } else {
      const bandwidth = getBandwidth(row, selectedValue);
      const fit = bandwidth > 0 ? fitLocation(row, bandwidth) : undefined;
      if (fit && fit.ok) {
        status = STATUS.OK;
        let intercept = fit.beta[0];
        for (let column = 1; column < p; column++) {
          rowCoefficients[column] = fit.beta[column];
          intercept -= fit.beta[column] * predictors[row * predictorCount + column - 1];
        }
        rowCoefficients[0] = intercept;
        fittedValue = fit.beta[0];
        residual = response[row] - fittedValue;
        hat = fit.hat;
        const weightedMean = fit.weightedResponse / fit.weightSum;
        let residualSum = 0;
        let totalSum = 0;
        for (let other = 0; other < rowCount; other++) {
          if (!valid[other]) {
            continue;
          }
          const weight = getGeographicallyWeightedRegressionWeight(
            kernel,
            distance(row, other),
            bandwidth
          );
          if (!(weight > 0)) {
            continue;
          }
          let prediction = fit.beta[0];
          for (let column = 0; column < predictorCount; column++) {
            prediction +=
              fit.beta[column + 1] *
              (predictors[other * predictorCount + column] -
                predictors[row * predictorCount + column]);
          }
          residualSum += weight * (response[other] - prediction) ** 2;
          totalSum += weight * (response[other] - weightedMean) ** 2;
        }
        r2 = totalSum > 0 ? 1 - residualSum / totalSum : nan;
      }
    }
    coefficients.push(...rowCoefficients);
    localR2.push(r2);
    fitted.push(fittedValue);
    residuals.push(residual);
    hatDiagonal.push(hat);
    localStatus.push(status);
  }
  const selectedSums = hasValidCandidate ? candidateSums[selectedIndex] : undefined;
  return {
    coefficients,
    localR2,
    fitted,
    residuals,
    hatDiagonal,
    localStatus,
    bandwidthScores,
    selectedIndex,
    selectedValue,
    residualSumOfSquares: selectedSums ? selectedSums.residualSum : nan,
    traceOfHat: selectedSums ? selectedSums.traceSum : nan,
    aicc: hasValidCandidate ? bandwidthScores[selectedIndex] : nan,
    rSquared: selectedSums && total > 0 ? 1 - selectedSums.residualSum / total : nan,
    observationCount: includedCount,
    hasValidCandidate
  };
}
