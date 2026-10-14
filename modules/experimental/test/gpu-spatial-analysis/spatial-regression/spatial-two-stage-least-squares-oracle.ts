// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getChiSquareSurvival} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {getSymmetricEigenvalueRangeOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-regression-solve';
import {
  createDiagnosticsRandom,
  createLatticeWeights,
  createNearestNeighborScene,
  invertMatrix,
  multiply,
  toDenseWeights,
  transpose,
  type DiagnosticsOracleWeights
} from './spatial-regression-diagnostics-oracle';

/** Result of {@link fitSpatialTwoStageLeastSquaresOnCPU}; table rows are intercept, X, rho. */
export type TwoStageOracleResult = {
  coefficients: number[];
  standardErrors: number[];
  zStatistics: number[];
  pValues: number[];
  sigmaSquared: number;
  residualSumOfSquares: number;
  pseudoRSquared: number;
  moranI: number;
  anselinKelejian: number;
  anselinKelejianPValue: number;
  residuals: number[];
  /** Diagnostics written to the GPU summary, in float64; see the `..._SUMMARY_*` slot docs. */
  diagnostics: TwoStageOracleDiagnostics;
};

/** Conditioning, first-stage strength and `rho` admissibility of a two-stage fit. */
export type TwoStageOracleDiagnostics = {
  conditionNumber: number;
  correlationConditionNumber: number;
  maximumVarianceInflationFactor: number;
  instrumentCorrelationConditionNumber: number;
  firstStageF: number;
  firstStagePartialRSquared: number;
  weightsInfinityNorm: number;
  rhoOutsideAdmissibleRange: boolean;
};

function getUnitDiagonal(matrix: number[][]): number[][] {
  return matrix.map((row, i) =>
    row.map((value, j) => value / Math.sqrt(matrix[i][i] * matrix[j][j]))
  );
}

function getEigenvalueRatio(matrix: number[][]): number {
  const size = matrix.length;
  const [smallest, largest] = getSymmetricEigenvalueRangeOnCPU(matrix.flat(), size);
  return largest / smallest;
}

function getCenteredColumns(matrix: number[][]): number[][] {
  const means = matrix[0].map(
    (_, j) => matrix.reduce((sum, row) => sum + row[j], 0) / matrix.length
  );
  return matrix.map(row => row.map((value, j) => value - means[j]));
}

/** Correlation matrix of the columns of `matrix` (centered, unit diagonal). */
function getCorrelation(matrix: number[][]): number[][] {
  const centered = getCenteredColumns(matrix);
  return getUnitDiagonal(multiply(transpose(centered), centered));
}

/**
 * Float64 oracle of spreg `GM_Lag` with `w_lags = instrumentOrder`, from the matrix definitions:
 * `Z = [1, X, Wy]`, `H = [1, X, WX]` (plus `W^2 X` for order 2), `delta = (Z'PZ)^-1 Z'Py`, `sigma2 = u'u / n` (spreg `sig2n`),
 * `cov = sigma2 (Z'PZ)^-1`. Coefficients are reordered to (intercept, X, rho).
 */
export function fitSpatialTwoStageLeastSquaresOnCPU(
  weights: DiagnosticsOracleWeights,
  predictors: Float32Array,
  response: Float32Array,
  predictorCount: number,
  instrumentOrder: 1 | 2 = 1
): TwoStageOracleResult {
  const n = response.length;
  const k = predictorCount;
  const w = toDenseWeights(weights);
  const y = Array.from(response, value => [value]);
  const x = Array.from({length: n}, (_, row) =>
    Array.from({length: k}, (_, column) => predictors[row * k + column])
  );
  const wy = multiply(w, y);
  const wx = multiply(w, x);
  const z = x.map((row, i) => [1, ...row, wy[i][0]]);
  const w2x = instrumentOrder === 2 ? multiply(w, wx) : wx.map(() => []);
  const h = x.map((row, i) => [1, ...row, ...wx[i], ...w2x[i]]);
  const ht = transpose(h);
  const zt = transpose(z);
  const projector = multiply(multiply(h, invertMatrix(multiply(ht, h))), ht);
  const normal = multiply(multiply(zt, projector), z);
  const normalInverse = invertMatrix(normal);
  const delta = multiply(multiply(normalInverse, multiply(zt, projector)), y);
  const fittedStructural = multiply(z, delta);
  const u = y.map((value, i) => value[0] - fittedStructural[i][0]);
  const residualSumOfSquares = u.reduce((sum, value) => sum + value * value, 0);
  const sigmaSquared = residualSumOfSquares / n;
  const standardErrors = delta.map((_, i) => Math.sqrt(sigmaSquared * normalInverse[i][i]));
  const coefficients = delta.map(row => row[0]);
  const zStatistics = coefficients.map((value, i) => value / standardErrors[i]);
  const predicted = y.map((value, i) => value[0] - u[i]);
  const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
  const meanY = mean(y.map(row => row[0]));
  const meanPredicted = mean(predicted);
  let covariance = 0;
  let varianceY = 0;
  let variancePredicted = 0;
  for (let i = 0; i < n; i++) {
    covariance += (y[i][0] - meanY) * (predicted[i] - meanPredicted);
    varianceY += (y[i][0] - meanY) ** 2;
    variancePredicted += (predicted[i] - meanPredicted) ** 2;
  }
  // Anselin-Kelejian (spreg akTest, case 'gen').
  const wu = multiply(
    w,
    u.map(value => [value])
  );
  const s0 = w.reduce((sum, row) => sum + row.reduce((a, b) => a + b, 0), 0);
  const wt = transpose(w);
  const traceT =
    multiply(wt, w).reduce((sum, row, i) => sum + row[i], 0) +
    multiply(w, w).reduce((sum, row, i) => sum + row[i], 0);
  const utwu = u.reduce((sum, value, i) => sum + value * wu[i][0], 0);
  const moranI = (n * utwu) / (s0 * residualSumOfSquares);
  const etwz = multiply([u], multiply(w, z));
  const quadratic = multiply(multiply(etwz, normalInverse), transpose(etwz))[0][0];
  const phiSquared = (traceT + (4 / sigmaSquared) * quadratic) / ((s0 / n) ** 2 * n);
  const anselinKelejian = (n * moranI * moranI) / phiSquared;
  // Diagnostics. First stage of W y on the instruments: classical joint F and partial R squared of
  // the excluded instruments; conditioning of [1, P Z] (Belsley), of Z'PZ and of the instruments.
  const wyColumn = wy;
  const sumOfSquares = (design: number[][]) => {
    const coefficientsOls = multiply(
      invertMatrix(multiply(transpose(design), design)),
      multiply(transpose(design), wyColumn)
    );
    const fitted = multiply(design, coefficientsOls);
    return wyColumn.reduce((sum, row, i) => sum + (row[0] - fitted[i][0]) ** 2, 0);
  };
  const rssFull = sumOfSquares(h);
  const rssRestricted = sumOfSquares(x.map(row => [1, ...row]));
  const instrumentCount = h[0].length - 1;
  const explainedGain = rssRestricted - rssFull;
  const projectedRegressors = multiply(projector, z);
  const secondStagePredictors = projectedRegressors.map(row => row.slice(1));
  const weightsInfinityNorm = Math.max(
    ...w.map(row => row.reduce((sum, value) => sum + Math.abs(value), 0))
  );
  const diagnostics: TwoStageOracleDiagnostics = {
    conditionNumber: Math.sqrt(
      getEigenvalueRatio(
        getUnitDiagonal(multiply(transpose(projectedRegressors), projectedRegressors))
      )
    ),
    correlationConditionNumber: getEigenvalueRatio(getCorrelation(secondStagePredictors)),
    maximumVarianceInflationFactor: Math.max(
      ...invertMatrix(getCorrelation(secondStagePredictors)).map((row, i) => row[i])
    ),
    instrumentCorrelationConditionNumber: getEigenvalueRatio(
      getCorrelation(h.map(row => row.slice(1)))
    ),
    firstStageF: explainedGain / (instrumentCount - k) / (rssFull / (n - instrumentCount - 1)),
    firstStagePartialRSquared: explainedGain / rssRestricted,
    weightsInfinityNorm,
    rhoOutsideAdmissibleRange:
      Math.abs(coefficients[coefficients.length - 1]) * weightsInfinityNorm >= 1
  };
  return {
    diagnostics,
    moranI,
    anselinKelejian,
    anselinKelejianPValue: getChiSquareSurvival(anselinKelejian, 1),
    coefficients,
    standardErrors,
    zStatistics,
    pValues: zStatistics.map(value => getChiSquareSurvival(value * value, 1)),
    sigmaSquared,
    residualSumOfSquares,
    pseudoRSquared: covariance ** 2 / (varianceY * variancePredicted),
    residuals: u
  };
}

/**
 * Lattice scene from the spatial lag model `y = (I - rho W)^-1 (1 + 2 x1 - x2 + eps)` on rook
 * weights, so the recovered `rho` can be sanity-checked against the generating value.
 */
export function createLagScene(
  side: number,
  seed: number,
  rho: number,
  rowStandardized: boolean,
  predictorCount = 2
): {
  weights: DiagnosticsOracleWeights;
  predictors: Float32Array;
  response: Float32Array;
  predictorCount: number;
} {
  const random = createDiagnosticsRandom(seed);
  const n = side * side;
  const weights = createLatticeWeights(side, rowStandardized);
  const w = toDenseWeights(weights);
  const predictors = new Float32Array(n * predictorCount);
  const rightHandSide = Array.from({length: n}, () => [0]);
  for (let row = 0; row < n; row++) {
    predictors[row * predictorCount] = 10 * random() + 3;
    predictors[row * predictorCount + 1] = random() * 4 - 2;
    // Extra predictors (beyond the first two) enter with coefficient 0.5; the first two draws and
    // the noise draw keep their order, so scenes with two predictors are unchanged.
    let extra = 0;
    for (let column = 2; column < predictorCount; column++) {
      predictors[row * predictorCount + column] = random() * 4 - 2;
      extra += 0.5 * predictors[row * predictorCount + column];
    }
    rightHandSide[row][0] =
      1 +
      2 * predictors[row * predictorCount] -
      predictors[row * predictorCount + 1] +
      extra +
      (random() * 2 - 1) * 2;
  }
  const system = w.map((row, i) => row.map((value, j) => (i === j ? 1 : 0) - rho * value));
  const solved = multiply(invertMatrix(system), rightHandSide);
  return {
    weights,
    predictors,
    response: Float32Array.from(solved, row => row[0]),
    predictorCount
  };
}

/**
 * Weak-instrument scene: kNN weights (`k = 3`, row-standardized, directed) of `count` random
 * points, independent uniform predictors and a response that barely depends on them
 * (`y = 0.03 x1 + noise`), so `W X` explains almost none of `W y` and the first stage is weak.
 */
export function createWeakInstrumentScene(
  count: number,
  seed: number
): {
  weights: DiagnosticsOracleWeights;
  predictors: Float32Array;
  response: Float32Array;
  predictorCount: number;
} {
  const base = createNearestNeighborScene(count, 3, seed, 0, false);
  const random = createDiagnosticsRandom(seed + 1000);
  const predictors = new Float32Array(count * 2);
  const response = new Float32Array(count);
  for (let row = 0; row < count; row++) {
    predictors[row * 2] = random() * 10;
    predictors[row * 2 + 1] = random() * 4 - 2;
    response[row] = 0.03 * predictors[row * 2] + (random() * 2 - 1) * 2;
  }
  return {weights: base.weights, predictors, response, predictorCount: 2};
}

/**
 * Ill-conditioned scene for the Belsley condition number: `createLagScene` with the first two
 * predictors replaced by `x1 = 1000 + x1` and `x2 = x1 + 0.3 x2` (large common mean, correlation
 * about 0.995), so the unit-column `[1, P Z]` design has a condition number in the thousands. The
 * raw Gram route (`sqrt(lambdaMax / lambdaMin)`) cannot resolve that in float32.
 */
export function createIllConditionedLagScene(
  side: number,
  seed: number,
  rho: number
): ReturnType<typeof createLagScene> {
  const scene = createLagScene(side, seed, rho, true);
  const predictors = Float32Array.from(scene.predictors);
  for (let row = 0; row < scene.response.length; row++) {
    const first = 1000 + predictors[row * 2];
    predictors[row * 2] = first;
    predictors[row * 2 + 1] = first + 0.3 * predictors[row * 2 + 1];
  }
  return {...scene, predictors};
}
