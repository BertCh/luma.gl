// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getChiSquareSurvival} from './ordinary-least-squares-statistics';
import {getSymmetricEigenvalueRangeOnCPU} from './spatial-regression-solve';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BISECTION_STEPS,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SCAN_STEPS,
  getSpatialLogDeterminantDerivativeFromTraces,
  getSpatialMaximumLikelihoodSeriesLimit,
  getSpatialMaximumLikelihoodTraceOne,
  getSpatialMaximumLikelihoodTraceTwo
} from './spatial-maximum-likelihood-parameters';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  getSpatialLogDeterminantFromTraces,
  getSpatialLogDeterminantTruncationBound,
  getSpatialWeightsTracesOnCPU,
  getSpatialWeightsTracesProbeValue,
  type GPUSpatialWeightsTracesMode,
  type SpatialWeightsCSR
} from './spatial-weights-traces-parameters';

const LN_TWO_PI = Math.log(2 * Math.PI);

/** Options of {@link fitSpatialLagMaximumLikelihoodOnCPU} and the error counterpart. */
export type SpatialMaximumLikelihoodCPUOptions = {
  /** Series length `Q` (at least 2). Defaults to 128. */
  maximumPower?: number;
  /** `'exact'` (default) or `'stochastic'` traces, see `GPUSpatialWeightsTraces`. */
  traceMode?: GPUSpatialWeightsTracesMode;
  /** Hutchinson probe count. Defaults to 64. */
  probeCount?: number;
  /** Philox seed of the stochastic probes. Defaults to 0. */
  seed?: number;
};

/** Result of the grid scan and bisection of {@link searchSpatialMaximumLikelihoodParameter}. */
export type SpatialMaximumLikelihoodSearchResult = {
  /** Maximizing parameter, NaN when no candidate exists (non-finite derivative everywhere). */
  parameter: number;
  /** Local maxima found: bracketed interior stationary points plus boundary maxima. */
  localMaximumCount: number;
  /** True when the best candidate is an end of the interval. */
  atBoundary: boolean;
};

/**
 * Float64 mirror of the GPU parameter search. The derivative of the concentrated log-likelihood is
 * evaluated on `scanSteps + 1` equally spaced points of `[-bound, bound]`; every sign change from
 * positive to non-positive brackets a local maximum that is refined by `bisectionSteps` bisection
 * steps on the derivative, and an end point is a candidate when the derivative points outward
 * there. The candidate with the largest objective wins (the first on ties).
 */
export function searchSpatialMaximumLikelihoodParameter(
  evaluateObjective: (parameter: number) => number,
  evaluateDerivative: (parameter: number) => number,
  bound: number,
  scanSteps = GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SCAN_STEPS,
  bisectionSteps = GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BISECTION_STEPS
): SpatialMaximumLikelihoodSearchResult {
  let best = NaN;
  let bestObjective = -Infinity;
  let bestAtBoundary = false;
  let localMaximumCount = 0;
  const consider = (parameter: number, atBoundary: boolean) => {
    const objective = evaluateObjective(parameter);
    localMaximumCount++;
    if (objective > bestObjective) {
      best = parameter;
      bestObjective = objective;
      bestAtBoundary = atBoundary;
    }
  };
  const step = (2 * bound) / scanSteps;
  let previousParameter = -bound;
  let previous = evaluateDerivative(previousParameter);
  if (previous <= 0) {
    consider(-bound, true);
  }
  for (let index = 1; index <= scanSteps; index++) {
    const parameter = index === scanSteps ? bound : -bound + index * step;
    const current = evaluateDerivative(parameter);
    if (previous > 0 && current <= 0) {
      let low = previousParameter;
      let high = parameter;
      for (let iteration = 0; iteration < bisectionSteps; iteration++) {
        const middle = 0.5 * (low + high);
        if (evaluateDerivative(middle) > 0) {
          low = middle;
        } else {
          high = middle;
        }
      }
      consider(0.5 * (low + high), false);
    }
    previous = current;
    previousParameter = parameter;
  }
  if (previous > 0) {
    consider(bound, true);
  }
  return {parameter: best, localMaximumCount, atBoundary: bestAtBoundary};
}

/** Largest absolute row sum `||W||_inf` of CSR weights (slots with neighbor `>= rows` dropped). */
export function getSpatialWeightsInfinityNorm(weights: SpatialWeightsCSR): number {
  const rows = weights.offsets.length - 1;
  let largest = 0;
  for (let row = 0; row < rows; row++) {
    let sum = 0;
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      if (weights.neighbors[slot] < rows) {
        sum += Math.abs(weights.weights[slot]);
      }
    }
    largest = Math.max(largest, sum);
  }
  return largest;
}

/** `W v` for CSR weights (slots with neighbor `>= rows` dropped). */
export function multiplySpatialWeights(
  weights: SpatialWeightsCSR,
  vector: ArrayLike<number>
): Float64Array {
  const rows = weights.offsets.length - 1;
  const result = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    let sum = 0;
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      const neighbor = weights.neighbors[slot];
      if (neighbor < rows) {
        sum += weights.weights[slot] * vector[neighbor];
      }
    }
    result[row] = sum;
  }
  return result;
}

/**
 * Truncated `(I - rho W)^-1 z = sum_{m=0}^{Q-1} rho^m W^m z` by Horner's rule
 * (`u <- z + rho W u`, `Q - 1` times), the series the GPU chain evaluates.
 */
export function applySpatialInverseSeries(
  weights: SpatialWeightsCSR,
  rho: number,
  vector: ArrayLike<number>,
  maximumPower: number
): Float64Array {
  let current: Float64Array = Float64Array.from(vector);
  for (let step = 1; step < maximumPower; step++) {
    const next = multiplySpatialWeights(weights, current);
    for (let row = 0; row < next.length; row++) {
      next[row] = vector[row] + rho * next[row];
    }
    current = next;
  }
  return current;
}

/**
 * `tr((W (I - rho W)^-1)' W (I - rho W)^-1) = ||W (I - rho W)^-1||_F^2` (the `tr3` of spreg) from
 * probes `z`: `E ||W A z||^2` with `A z` the truncated series of {@link applySpatialInverseSeries}.
 * `'exact'` uses the unit vectors (no sampling error); `'stochastic'` averages `probeCount`
 * Rademacher probes, the same stream as `GPUSpatialWeightsTraces`.
 */
export function getSpatialFrobeniusTraceOnCPU(
  weights: SpatialWeightsCSR,
  rho: number,
  maximumPower: number,
  options: SpatialMaximumLikelihoodCPUOptions = {}
): number {
  const rows = weights.offsets.length - 1;
  const stochastic = options.traceMode === 'stochastic';
  const probeTotal = stochastic
    ? (options.probeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT)
    : rows;
  const seed = (options.seed ?? 0) >>> 0;
  let total = 0;
  for (let probe = 0; probe < probeTotal; probe++) {
    const z = new Float64Array(rows);
    if (stochastic) {
      for (let row = 0; row < rows; row++) {
        z[row] = getSpatialWeightsTracesProbeValue(seed, probe, row);
      }
    } else {
      z[probe] = 1;
    }
    const wa = multiplySpatialWeights(
      weights,
      applySpatialInverseSeries(weights, rho, z, maximumPower)
    );
    for (let row = 0; row < rows; row++) {
      total += wa[row] * wa[row];
    }
  }
  return stochastic ? total / probeTotal : total;
}

/** Inverts a symmetric positive definite matrix by Cholesky; null if a pivot is not positive. */
export function invertSymmetricPositiveDefinite(matrix: number[][]): number[][] | null {
  const size = matrix.length;
  const factor = matrix.map(row => row.slice());
  for (let column = 0; column < size; column++) {
    let diagonal = factor[column][column];
    for (let k = 0; k < column; k++) {
      diagonal -= factor[column][k] ** 2;
    }
    // A pivot below 1e-12 of the original diagonal is rounding noise of a singular matrix.
    if (!(diagonal > 1e-12 * matrix[column][column]) || !Number.isFinite(diagonal)) {
      return null;
    }
    const pivot = Math.sqrt(diagonal);
    factor[column][column] = pivot;
    for (let row = column + 1; row < size; row++) {
      let value = factor[row][column];
      for (let k = 0; k < column; k++) {
        value -= factor[row][k] * factor[column][k];
      }
      factor[row][column] = value / pivot;
    }
  }
  const inverse: number[][] = [];
  for (let target = 0; target < size; target++) {
    const solution = new Array<number>(size).fill(0);
    solution[target] = 1;
    for (let row = 0; row < size; row++) {
      let value = solution[row];
      for (let k = 0; k < row; k++) {
        value -= factor[row][k] * solution[k];
      }
      solution[row] = value / factor[row][row];
    }
    for (let row = size - 1; row >= 0; row--) {
      let value = solution[row];
      for (let k = row + 1; k < size; k++) {
        value -= factor[k][row] * solution[k];
      }
      solution[row] = value / factor[row][row];
    }
    inverse.push(solution);
  }
  return inverse;
}

function solveNormalEquations(matrix: number[][], rightHandSide: number[]): number[] | null {
  const inverse = invertSymmetricPositiveDefinite(matrix);
  if (!inverse) {
    return null;
  }
  return inverse.map(row =>
    row.reduce((sum, value, column) => sum + value * rightHandSide[column], 0)
  );
}

/** Least squares of `response` on `[1, predictors]`; returns coefficients and residuals. */
function fitOrdinaryLeastSquares(
  predictors: ArrayLike<number>,
  response: ArrayLike<number>,
  predictorCount: number
): {coefficients: number[]; residuals: Float64Array} | null {
  const rows = response.length;
  const width = predictorCount + 1;
  const gram = Array.from({length: width}, () => new Array<number>(width).fill(0));
  const moment = new Array<number>(width).fill(0);
  const design = (row: number, column: number) =>
    column === 0 ? 1 : predictors[row * predictorCount + column - 1];
  for (let row = 0; row < rows; row++) {
    for (let a = 0; a < width; a++) {
      moment[a] += design(row, a) * response[row];
      for (let b = 0; b < width; b++) {
        gram[a][b] += design(row, a) * design(row, b);
      }
    }
  }
  const coefficients = solveNormalEquations(gram, moment);
  if (!coefficients) {
    return null;
  }
  const residuals = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    let fitted = 0;
    for (let a = 0; a < width; a++) {
      fitted += design(row, a) * coefficients[a];
    }
    residuals[row] = response[row] - fitted;
  }
  return {coefficients, residuals};
}

/** Standard normal two-sided p-value of `z` (float64, via the chi-square survival of `z^2`). */
export function getNormalTwoSidedPValue(z: number): number {
  return Number.isNaN(z) ? NaN : getChiSquareSurvival(z * z, 1);
}

/** Conditioning of a unit-diagonal-scaled predictor matrix, see `GPURegressionConditioning`. */
export type SpatialMaximumLikelihoodConditioning = {
  conditionNumber: number;
  correlationConditionNumber: number;
  maximumVarianceInflationFactor: number;
};

/** Float64 fit of the spatial lag model, see {@link fitSpatialLagMaximumLikelihoodOnCPU}. */
export type SpatialLagMaximumLikelihoodCPUResult = {
  /** Intercept, predictors, then `rho`. */
  coefficients: number[];
  standardErrors: number[];
  zStatistics: number[];
  pValues: number[];
  rho: number;
  sigmaSquared: number;
  residualSumOfSquares: number;
  /** Full Gaussian log-likelihood (spreg `logll`). */
  logLikelihood: number;
  akaikeInformationCriterion: number;
  schwarzCriterion: number;
  pseudoRSquared: number;
  pseudoRSquaredSpatial: number;
  logDeterminant: number;
  /** `[tr1, tr2, tr3]` of `W (I - rho W)^-1`. */
  traces: [number, number, number];
  bound: number;
  seriesLimited: boolean;
  truncationBound: number;
  localMaximumCount: number;
  atBoundary: boolean;
  residuals: Float64Array;
  conditioning: SpatialMaximumLikelihoodConditioning;
};

function getParameterBound(
  infinityNorm: number,
  maximumPower: number
): {bound: number; seriesLimited: boolean} {
  const limit = getSpatialMaximumLikelihoodSeriesLimit(maximumPower);
  const stationarity = GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION;
  return {bound: Math.min(stationarity, limit) / infinityNorm, seriesLimited: limit < stationarity};
}

function getColumnMeans(
  predictors: ArrayLike<number>,
  rows: number,
  predictorCount: number
): number[] {
  const means = new Array<number>(predictorCount).fill(0);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < predictorCount; column++) {
      means[column] += predictors[row * predictorCount + column] / rows;
    }
  }
  return means;
}

function getPearsonSquared(left: ArrayLike<number>, right: ArrayLike<number>): number {
  const rows = left.length;
  let meanLeft = 0;
  let meanRight = 0;
  for (let row = 0; row < rows; row++) {
    meanLeft += left[row] / rows;
    meanRight += right[row] / rows;
  }
  let cross = 0;
  let varianceLeft = 0;
  let varianceRight = 0;
  for (let row = 0; row < rows; row++) {
    cross += (left[row] - meanLeft) * (right[row] - meanRight);
    varianceLeft += (left[row] - meanLeft) ** 2;
    varianceRight += (right[row] - meanRight) ** 2;
  }
  return (cross * cross) / (varianceLeft * varianceRight);
}

/** Maps a centered-parameterization covariance to the raw intercept: `b0 = a0 - m'b`. */
function getInterceptVariance(inverse: number[][], means: number[]): number {
  const k = means.length;
  let variance = inverse[0][0];
  for (let j = 0; j < k; j++) {
    variance -= 2 * means[j] * inverse[0][1 + j];
    for (let l = 0; l < k; l++) {
      variance += means[j] * means[l] * inverse[1 + j][1 + l];
    }
  }
  return variance;
}

/**
 * Float64 mirror of `GPUSpatialLagMaximumLikelihood`, step for step: OLS residuals `e0`, `e1` of
 * `y` and `W y` on `[1, X]`, the concentrated log-likelihood scanned and refined by
 * {@link searchSpatialMaximumLikelihoodParameter} within `min(0.99, seriesLimit) / ||W||_inf`,
 * the log-determinant by the truncated trace series, and the spreg information matrix with
 * `tr1`, `tr2` from the traces and `tr3` from the probe pass. The `sigma2` row of the information
 * matrix is eliminated by its Schur complement (`tr2 + tr3 + (W Xb~)'(W Xb~) / s2 - 2 tr1^2 / n`
 * for the `rho` entry), which leaves the coefficient block of spreg's inverse unchanged.
 * Weights are CSR; returns null when the design is singular.
 */
export function fitSpatialLagMaximumLikelihoodOnCPU(
  weights: SpatialWeightsCSR,
  predictors: ArrayLike<number>,
  response: ArrayLike<number>,
  predictorCount: number,
  options: SpatialMaximumLikelihoodCPUOptions = {}
): SpatialLagMaximumLikelihoodCPUResult | null {
  const maximumPower = options.maximumPower ?? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER;
  const rows = response.length;
  const width = predictorCount + 1;
  const lagged = multiplySpatialWeights(weights, response);
  const first = fitOrdinaryLeastSquares(predictors, response, predictorCount);
  const second = fitOrdinaryLeastSquares(predictors, lagged, predictorCount);
  if (!first || !second) {
    return null;
  }
  const e0 = first.residuals;
  const e1 = second.residuals;
  let a = 0;
  let b = 0;
  let c = 0;
  for (let row = 0; row < rows; row++) {
    a += e0[row] * e0[row];
    b += e0[row] * e1[row];
    c += e1[row] * e1[row];
  }
  const traceOptions = {
    mode: options.traceMode,
    probeCount: options.probeCount,
    seed: options.seed
  };
  const {powerTraces} = getSpatialWeightsTracesOnCPU(weights, maximumPower, traceOptions);
  const infinityNorm = getSpatialWeightsInfinityNorm(weights);
  const {bound, seriesLimited} = getParameterBound(infinityNorm, maximumPower);
  const sumOfSquares = (rho: number) => a - 2 * rho * b + rho * rho * c;
  const search = searchSpatialMaximumLikelihoodParameter(
    rho =>
      -0.5 * rows * Math.log(sumOfSquares(rho) / rows) +
      getSpatialLogDeterminantFromTraces(rho, powerTraces, infinityNorm),
    rho =>
      (-0.5 * rows * (-2 * b + 2 * rho * c)) / sumOfSquares(rho) +
      getSpatialLogDeterminantDerivativeFromTraces(rho, powerTraces, infinityNorm),
    bound
  );
  const rho = search.parameter;
  const beta = first.coefficients.map((value, index) => value - rho * second.coefficients[index]);
  const residuals = new Float64Array(rows);
  const reduced = new Float64Array(rows);
  const predicted = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    residuals[row] = e0[row] - rho * e1[row];
    predicted[row] = response[row] - residuals[row];
    reduced[row] = predicted[row] - rho * lagged[row];
  }
  const residualSumOfSquares = residuals.reduce((sum, value) => sum + value * value, 0);
  const sigmaSquared = residualSumOfSquares / rows;
  const logDeterminant = getSpatialLogDeterminantFromTraces(rho, powerTraces, infinityNorm);
  const logLikelihood =
    -0.5 * rows * Math.log(sigmaSquared) + logDeterminant - 0.5 * rows * (LN_TWO_PI + 1);
  const spatialPrediction = applySpatialInverseSeries(weights, rho, reduced, maximumPower);
  const weightedPrediction = multiplySpatialWeights(weights, spatialPrediction);
  const traceOne = getSpatialMaximumLikelihoodTraceOne(rho, powerTraces, infinityNorm);
  const traceTwo = getSpatialMaximumLikelihoodTraceTwo(rho, powerTraces, infinityNorm);
  const traceThree = getSpatialFrobeniusTraceOnCPU(weights, rho, maximumPower, options);
  const means = getColumnMeans(predictors, rows, predictorCount);
  const size = width + 1;
  const information = Array.from({length: size}, () => new Array<number>(size).fill(0));
  const centered = (row: number, column: number) =>
    column === 0 ? 1 : predictors[row * predictorCount + column - 1] - means[column - 1];
  for (let row = 0; row < rows; row++) {
    for (let p = 0; p < width; p++) {
      for (let q = 0; q < width; q++) {
        information[p][q] += (centered(row, p) * centered(row, q)) / sigmaSquared;
      }
      information[p][width] += (centered(row, p) * weightedPrediction[row]) / sigmaSquared;
    }
    information[width][width] += (weightedPrediction[row] * weightedPrediction[row]) / sigmaSquared;
  }
  for (let p = 0; p < width; p++) {
    information[width][p] = information[p][width];
  }
  information[width][width] += traceTwo + traceThree - (2 * traceOne * traceOne) / rows;
  const inverse = invertSymmetricPositiveDefinite(information);
  const standardErrors = new Array<number>(width + 1).fill(NaN);
  if (inverse) {
    standardErrors[0] = Math.sqrt(getInterceptVariance(inverse, means));
    for (let p = 1; p <= width; p++) {
      standardErrors[p] = Math.sqrt(inverse[p][p]);
    }
  }
  const coefficients = [...beta, rho];
  const zStatistics = coefficients.map((value, index) => value / standardErrors[index]);
  const parameterCount = width + 1;
  return {
    coefficients,
    standardErrors,
    zStatistics,
    pValues: zStatistics.map(getNormalTwoSidedPValue),
    rho,
    sigmaSquared,
    residualSumOfSquares,
    logLikelihood,
    akaikeInformationCriterion: 2 * parameterCount - 2 * logLikelihood,
    schwarzCriterion: parameterCount * Math.log(rows) - 2 * logLikelihood,
    pseudoRSquared: getPearsonSquared(response, predicted),
    pseudoRSquaredSpatial: getPearsonSquared(response, spatialPrediction),
    logDeterminant,
    traces: [traceOne, traceTwo, traceThree],
    bound,
    seriesLimited,
    truncationBound: getSpatialLogDeterminantTruncationBound(rho, maximumPower, rows, infinityNorm),
    localMaximumCount: search.localMaximumCount,
    atBoundary: search.atBoundary,
    residuals,
    conditioning: getDesignConditioning(predictors, rows, predictorCount)
  };
}

/**
 * Conditioning of the design `[1, X]`: Belsley condition number of the raw Gram matrix scaled to
 * unit diagonal; correlation condition number and largest variance inflation factor of the
 * centered predictor correlation matrix.
 */
export function getDesignConditioning(
  predictors: ArrayLike<number>,
  rows: number,
  predictorCount: number
): SpatialMaximumLikelihoodConditioning {
  const width = predictorCount + 1;
  const raw = Array.from({length: width}, () => new Array<number>(width).fill(0));
  const column = (row: number, index: number) =>
    index === 0 ? 1 : predictors[row * predictorCount + index - 1];
  for (let row = 0; row < rows; row++) {
    for (let p = 0; p < width; p++) {
      for (let q = 0; q < width; q++) {
        raw[p][q] += column(row, p) * column(row, q);
      }
    }
  }
  return getConditioningFromRawGram(raw);
}

/** Conditioning from a raw Gram matrix whose first row and column belong to the constant. */
export function getConditioningFromRawGram(raw: number[][]): SpatialMaximumLikelihoodConditioning {
  const width = raw.length;
  const scaled = raw.map((row, p) =>
    row.map((value, q) => value / Math.sqrt(raw[p][p] * raw[q][q]))
  );
  const [smallestRaw, largestRaw] = getSymmetricEigenvalueRangeOnCPU(scaled.flat(), width);
  // Correlation matrix of the predictors: centered cross-products from the raw Gram.
  const count = raw[0][0];
  const centeredGram = Array.from({length: width - 1}, (_, p) =>
    Array.from(
      {length: width - 1},
      (_, q) => raw[p + 1][q + 1] - (raw[0][p + 1] * raw[0][q + 1]) / count
    )
  );
  const correlation = centeredGram.map((row, p) =>
    row.map((value, q) => value / Math.sqrt(centeredGram[p][p] * centeredGram[q][q]))
  );
  const [smallest, largest] = getSymmetricEigenvalueRangeOnCPU(correlation.flat(), width - 1);
  const inverse = invertSymmetricPositiveDefinite(correlation);
  return {
    conditionNumber: Math.sqrt(largestRaw / smallestRaw),
    correlationConditionNumber: largest / smallest,
    maximumVarianceInflationFactor: inverse
      ? Math.max(...inverse.map((row, index) => row[index]))
      : Infinity
  };
}

/** Float64 fit of the spatial error model, see {@link fitSpatialErrorMaximumLikelihoodOnCPU}. */
export type SpatialErrorMaximumLikelihoodCPUResult = {
  /** Intercept, predictors, then `lambda`. */
  coefficients: number[];
  standardErrors: number[];
  zStatistics: number[];
  pValues: number[];
  lambda: number;
  /** `e'e / n` of the filtered residuals. */
  sigmaSquared: number;
  /** `u'u` of the unfiltered residuals `y - X b`. */
  residualSumOfSquares: number;
  logLikelihood: number;
  akaikeInformationCriterion: number;
  schwarzCriterion: number;
  pseudoRSquared: number;
  logDeterminant: number;
  traces: [number, number, number];
  bound: number;
  seriesLimited: boolean;
  truncationBound: number;
  localMaximumCount: number;
  atBoundary: boolean;
  residuals: Float64Array;
  conditioning: SpatialMaximumLikelihoodConditioning;
};

/**
 * Float64 mirror of `GPUSpatialErrorMaximumLikelihood`. Columns `Z = [1, X - mean, y - mean]` are
 * filtered as `Z - lambda W Z`, so the filtered Gram matrix is exactly quadratic in `lambda`,
 * `G0 - lambda (G1 + G1') + lambda^2 G2` with `G0 = Z'Z`, `G1 = Z'WZ`, `G2 = (WZ)'(WZ)`. The
 * concentrated likelihood and its derivative (envelope theorem) need one small solve per
 * `lambda`. Coefficient covariance is `sigma2 (Xs'Xs)^-1` and the variance of `lambda`
 * `1 / (tr2 + tr3 - 2 tr1^2 / n)` (the Schur complement of spreg's 2x2 information matrix), with
 * the traces of `W (I - lambda W)^-1`. Returns null when the filtered design is singular.
 */
export function fitSpatialErrorMaximumLikelihoodOnCPU(
  weights: SpatialWeightsCSR,
  predictors: ArrayLike<number>,
  response: ArrayLike<number>,
  predictorCount: number,
  options: SpatialMaximumLikelihoodCPUOptions = {}
): SpatialErrorMaximumLikelihoodCPUResult | null {
  const maximumPower = options.maximumPower ?? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER;
  const rows = response.length;
  const width = predictorCount + 1;
  const columns = predictorCount + 2;
  const means = getColumnMeans(predictors, rows, predictorCount);
  let responseMean = 0;
  for (let row = 0; row < rows; row++) {
    responseMean += response[row] / rows;
  }
  const z = Array.from({length: columns}, () => new Float64Array(rows));
  for (let row = 0; row < rows; row++) {
    z[0][row] = 1;
    for (let j = 0; j < predictorCount; j++) {
      z[1 + j][row] = predictors[row * predictorCount + j] - means[j];
    }
    z[columns - 1][row] = response[row] - responseMean;
  }
  const wz = z.map(column => multiplySpatialWeights(weights, column));
  const dot = (left: Float64Array, right: Float64Array) => {
    let sum = 0;
    for (let row = 0; row < rows; row++) {
      sum += left[row] * right[row];
    }
    return sum;
  };
  const g0 = z.map(left => z.map(right => dot(left, right)));
  const g1 = z.map((left, p) => z.map((_, q) => dot(left, wz[q]) + dot(z[q], wz[p])));
  const g2 = wz.map(left => wz.map(right => dot(left, right)));
  const {powerTraces} = getSpatialWeightsTracesOnCPU(weights, maximumPower, {
    mode: options.traceMode,
    probeCount: options.probeCount,
    seed: options.seed
  });
  const infinityNorm = getSpatialWeightsInfinityNorm(weights);
  const {bound, seriesLimited} = getParameterBound(infinityNorm, maximumPower);
  const filteredSystem = (lambda: number) => {
    const gram = g0.map((row, p) =>
      row.map((value, q) => value - lambda * g1[p][q] + lambda * lambda * g2[p][q])
    );
    const derivative = g0.map((row, p) => row.map((_, q) => -g1[p][q] + 2 * lambda * g2[p][q]));
    const matrix = gram.slice(0, width).map(row => row.slice(0, width));
    const rightHandSide = gram.slice(0, width).map(row => row[width]);
    const solution = solveNormalEquations(matrix, rightHandSide);
    return {gram, derivative, matrix, rightHandSide, solution};
  };
  const evaluate = (lambda: number) => {
    const system = filteredSystem(lambda);
    if (!system.solution) {
      return {errorSumOfSquares: NaN, derivative: NaN, solution: null, system};
    }
    const beta = system.solution;
    let errorSumOfSquares = system.gram[columns - 1][columns - 1];
    let derivative = system.derivative[columns - 1][columns - 1];
    for (let p = 0; p < width; p++) {
      errorSumOfSquares -= 2 * beta[p] * system.rightHandSide[p];
      derivative -= 2 * beta[p] * system.derivative[p][width];
      for (let q = 0; q < width; q++) {
        errorSumOfSquares += beta[p] * system.matrix[p][q] * beta[q];
        derivative += beta[p] * system.derivative[p][q] * beta[q];
      }
    }
    return {errorSumOfSquares, derivative, solution: beta, system};
  };
  const search = searchSpatialMaximumLikelihoodParameter(
    lambda =>
      -0.5 * rows * Math.log(evaluate(lambda).errorSumOfSquares / rows) +
      getSpatialLogDeterminantFromTraces(lambda, powerTraces, infinityNorm),
    lambda => {
      const {errorSumOfSquares, derivative} = evaluate(lambda);
      return (
        (-0.5 * rows * derivative) / errorSumOfSquares +
        getSpatialLogDeterminantDerivativeFromTraces(lambda, powerTraces, infinityNorm)
      );
    },
    bound
  );
  const lambda = search.parameter;
  const final = evaluate(lambda);
  if (!final.solution) {
    return null;
  }
  const filtered = final.solution;
  const slopes = filtered.slice(1);
  let intercept = filtered[0] + responseMean;
  for (let j = 0; j < predictorCount; j++) {
    intercept -= means[j] * slopes[j];
  }
  const sigmaSquared = final.errorSumOfSquares / rows;
  const inverse = invertSymmetricPositiveDefinite(final.system.matrix);
  const standardErrors = new Array<number>(width + 1).fill(NaN);
  if (inverse) {
    const scaled = inverse.map(row => row.map(value => value * sigmaSquared));
    standardErrors[0] = Math.sqrt(getInterceptVariance(scaled, means));
    for (let p = 1; p < width; p++) {
      standardErrors[p] = Math.sqrt(scaled[p][p]);
    }
  }
  const traceOne = getSpatialMaximumLikelihoodTraceOne(lambda, powerTraces, infinityNorm);
  const traceTwo = getSpatialMaximumLikelihoodTraceTwo(lambda, powerTraces, infinityNorm);
  const traceThree = getSpatialFrobeniusTraceOnCPU(weights, lambda, maximumPower, options);
  standardErrors[width] = Math.sqrt(1 / (traceTwo + traceThree - (2 * traceOne * traceOne) / rows));
  const residuals = new Float64Array(rows);
  const predicted = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    let fitted = intercept;
    for (let j = 0; j < predictorCount; j++) {
      fitted += predictors[row * predictorCount + j] * slopes[j];
    }
    predicted[row] = fitted;
    residuals[row] = response[row] - fitted;
  }
  const coefficients = [intercept, ...slopes, lambda];
  const zStatistics = coefficients.map((value, index) => value / standardErrors[index]);
  const logDeterminant = getSpatialLogDeterminantFromTraces(lambda, powerTraces, infinityNorm);
  const logLikelihood =
    -0.5 * rows * Math.log(sigmaSquared) + logDeterminant - 0.5 * rows * (LN_TWO_PI + 1);
  // spreg ML_Error counts the intercept and predictors only (`k = x.shape[1]`).
  const criterionCount = width;
  // Raw filtered Gram: columns (c, Xc_s + m c) are the transform T = [[1, m'], [0, I]] of the
  // centered filtered columns, so Gram_raw = T' A T.
  const transform = (column: number) =>
    Array.from({length: width}, (_, index) =>
      index === column ? 1 : index === 0 && column > 0 ? means[column - 1] : 0
    );
  const raw = final.system.matrix.map((row, p) =>
    row.map((_, q) => {
      const left = transform(p);
      const right = transform(q);
      let sum = 0;
      for (let a = 0; a < width; a++) {
        for (let b = 0; b < width; b++) {
          sum += left[a] * final.system.matrix[a][b] * right[b];
        }
      }
      return sum;
    })
  );
  return {
    coefficients,
    standardErrors,
    zStatistics,
    pValues: zStatistics.map(getNormalTwoSidedPValue),
    lambda,
    sigmaSquared,
    residualSumOfSquares: residuals.reduce((sum, value) => sum + value * value, 0),
    logLikelihood,
    akaikeInformationCriterion: 2 * criterionCount - 2 * logLikelihood,
    schwarzCriterion: criterionCount * Math.log(rows) - 2 * logLikelihood,
    pseudoRSquared: getPearsonSquared(response, predicted),
    logDeterminant,
    traces: [traceOne, traceTwo, traceThree],
    bound,
    seriesLimited,
    truncationBound: getSpatialLogDeterminantTruncationBound(
      lambda,
      maximumPower,
      rows,
      infinityNorm
    ),
    localMaximumCount: search.localMaximumCount,
    atBoundary: search.atBoundary,
    residuals,
    conditioning: getConditioningFromFilteredMatrix(final.system.matrix, raw)
  };
}

/** Conditioning of the filtered regression from its centered normal matrix `A` and raw Gram. */
function getConditioningFromFilteredMatrix(
  centeredMatrix: number[][],
  raw: number[][]
): SpatialMaximumLikelihoodConditioning {
  const width = raw.length;
  const scaledRaw = raw.map((row, p) =>
    row.map((value, q) => value / Math.sqrt(raw[p][p] * raw[q][q]))
  );
  const [smallestRaw, largestRaw] = getSymmetricEigenvalueRangeOnCPU(scaledRaw.flat(), width);
  const scaled = centeredMatrix.map((row, p) =>
    row.map((value, q) => value / Math.sqrt(centeredMatrix[p][p] * centeredMatrix[q][q]))
  );
  const [smallest, largest] = getSymmetricEigenvalueRangeOnCPU(scaled.flat(), width);
  const inverse = invertSymmetricPositiveDefinite(scaled);
  return {
    conditionNumber: Math.sqrt(largestRaw / smallestRaw),
    correlationConditionNumber: largest / smallest,
    maximumVarianceInflationFactor: inverse
      ? Math.max(...inverse.slice(1).map((row, index) => row[index + 1]))
      : Infinity
  };
}
