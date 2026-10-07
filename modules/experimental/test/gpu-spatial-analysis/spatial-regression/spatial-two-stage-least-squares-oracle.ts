// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getChiSquareSurvival} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {
  createDiagnosticsRandom,
  createLatticeWeights,
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
};

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
  return {
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
  rowStandardized: boolean
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
  const predictors = new Float32Array(n * 2);
  const rightHandSide = Array.from({length: n}, () => [0]);
  for (let row = 0; row < n; row++) {
    predictors[row * 2] = 10 * random() + 3;
    predictors[row * 2 + 1] = random() * 4 - 2;
    rightHandSide[row][0] =
      1 + 2 * predictors[row * 2] - predictors[row * 2 + 1] + (random() * 2 - 1) * 2;
  }
  const system = w.map((row, i) => row.map((value, j) => (i === j ? 1 : 0) - rho * value));
  const solved = multiply(invertMatrix(system), rightHandSide);
  return {
    weights,
    predictors,
    response: Float32Array.from(solved, row => row[0]),
    predictorCount: 2
  };
}
