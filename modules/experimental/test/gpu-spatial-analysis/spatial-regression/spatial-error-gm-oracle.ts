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

/** Result of {@link fitSpatialErrorGMOnCPU}; table rows are intercept, X (no row for lambda). */
export type ErrorGMOracleResult = {
  coefficients: number[];
  standardErrors: number[];
  zStatistics: number[];
  pValues: number[];
  lambda: number;
  sigmaSquared: number;
  residualSumOfSquares: number;
  pseudoRSquared: number;
  residuals: number[];
  /** Moments `[u'u, u'Wu, (Wu)'Wu, u'WWu, (Wu)'WWu, (WWu)'WWu, tr W'W]` of the OLS residuals. */
  moments: number[];
};

/**
 * Float64 oracle of spreg 1.9.1 `GM_Error`, from the published formulas (Kelejian and Prucha
 * 1998, 1999; spreg `error_sp.BaseGM_Error` and `_momentsGM_Error`) with dense algebra. The oracle is
 * checked against real spreg 1.9.1 output in the node spec.
 *
 * `lambda` minimizes `|G [l, l^2, s]' - g|^2` over `l` in `[-0.99, 0.99]` and `s >= 0` (spreg's
 * bounds). Here `s` is eliminated by least squares per `l`, clamped at zero, and `l` is found by a 20001-point grid
 * followed by golden-section refinement (a different search than the GPU's stationary-point
 * bisection, over the same objective).
 */
export function fitSpatialErrorGMOnCPU(
  weights: DiagnosticsOracleWeights,
  predictors: Float32Array,
  response: Float32Array,
  predictorCount: number
): ErrorGMOracleResult {
  const n = response.length;
  const k = predictorCount;
  const w = toDenseWeights(weights);
  const y = Array.from(response, value => [value]);
  const x = Array.from({length: n}, (_, row) => [
    1,
    ...Array.from({length: k}, (_, column) => predictors[row * k + column])
  ]);
  const xt = transpose(x);
  const olsBeta = multiply(invertMatrix(multiply(xt, x)), multiply(xt, y));
  const olsFit = multiply(x, olsBeta);
  const u = y.map((value, i) => [value[0] - olsFit[i][0]]);
  const wu = multiply(w, u);
  const wwu = multiply(w, wu);
  const dot = (left: number[][], right: number[][]) =>
    left.reduce((sum, row, i) => sum + row[0] * right[i][0], 0);
  const wtw = multiply(transpose(w), w);
  const traceWtW = wtw.reduce((sum, row, i) => sum + row[i], 0);
  const u2 = dot(u, u);
  const uwu = dot(u, wu);
  const wu2 = dot(wu, wu);
  const uwwu = dot(u, wwu);
  const wuwwu = dot(wu, wwu);
  const wwu2 = dot(wwu, wwu);
  const gMatrix = [
    [2 * uwu, -wu2, 1],
    [2 * wuwwu, -wwu2, traceWtW / n],
    [uwwu + wu2, -wuwwu, 0]
  ].map(row => row.map(value => value / n));
  const gVector = [u2, wu2, uwu].map(value => value / n);
  const bound = 0.99;
  const objective = (lambda: number) => {
    const free = [0, 1, 2].map(
      i => gVector[i] - gMatrix[i][0] * lambda - gMatrix[i][1] * lambda ** 2
    );
    const column = [gMatrix[0][2], gMatrix[1][2], gMatrix[2][2]];
    // sigma2 >= 0 as in spreg's bounds: least squares in sigma2, clamped at zero.
    const sigma = Math.max(
      free.reduce((sum, value, i) => sum + value * column[i], 0) /
        column.reduce((sum, value) => sum + value * value, 0),
      0
    );
    return free.reduce((sum, value, i) => sum + (value - column[i] * sigma) ** 2, 0);
  };
  const gridSteps = 20000;
  const gridPoint = (step: number) => bound * (-1 + (2 * step) / gridSteps);
  let best = 0;
  for (let step = 1; step <= gridSteps; step++) {
    if (objective(gridPoint(step)) < objective(gridPoint(best))) {
      best = step;
    }
  }
  let low = gridPoint(Math.max(0, best - 1));
  let high = gridPoint(Math.min(gridSteps, best + 1));
  const golden = (Math.sqrt(5) - 1) / 2;
  for (let iteration = 0; iteration < 100; iteration++) {
    const left = high - golden * (high - low);
    const right = low + golden * (high - low);
    if (objective(left) < objective(right)) {
      high = right;
    } else {
      low = left;
    }
  }
  const lambda = (low + high) / 2;
  // Spatially filtered regression, the constant column included (spreg get_spFilter).
  const filter = (matrix: number[][]) => {
    const lagged = multiply(w, matrix);
    return matrix.map((row, i) => row.map((value, j) => value - lambda * lagged[i][j]));
  };
  const xs = filter(x);
  const ys = filter(y);
  const xst = transpose(xs);
  const normalInverse = invertMatrix(multiply(xst, xs));
  const beta = multiply(normalInverse, multiply(xst, ys));
  const filteredFit = multiply(xs, beta);
  const filteredSquares = ys.reduce((sum, row, i) => sum + (row[0] - filteredFit[i][0]) ** 2, 0);
  const sigmaSquared = filteredSquares / n;
  const fit = multiply(x, beta);
  const residuals = y.map((row, i) => row[0] - fit[i][0]);
  const standardErrors = beta.map((_, i) => Math.sqrt(sigmaSquared * normalInverse[i][i]));
  const coefficients = beta.map(row => row[0]);
  const zStatistics = coefficients.map((value, i) => value / standardErrors[i]);
  const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
  const meanY = mean(y.map(row => row[0]));
  const meanFit = mean(fit.map(row => row[0]));
  let covariance = 0;
  let varianceY = 0;
  let varianceFit = 0;
  for (let i = 0; i < n; i++) {
    covariance += (y[i][0] - meanY) * (fit[i][0] - meanFit);
    varianceY += (y[i][0] - meanY) ** 2;
    varianceFit += (fit[i][0] - meanFit) ** 2;
  }
  return {
    coefficients,
    standardErrors,
    zStatistics,
    pValues: zStatistics.map(value => getChiSquareSurvival(value * value, 1)),
    lambda,
    sigmaSquared,
    residualSumOfSquares: residuals.reduce((sum, value) => sum + value * value, 0),
    pseudoRSquared: covariance ** 2 / (varianceY * varianceFit),
    residuals,
    moments: [u2, uwu, wu2, uwwu, wuwwu, wwu2, traceWtW]
  };
}

/** Which weights a {@link createErrorScene} uses. */
export type ErrorSceneWeights = 'lattice-row-standardized' | 'lattice-binary' | 'knn';

/**
 * Scene from the spatial error model `y = 1 + 2 x1 - x2 + (I - lambda W)^-1 eps`, so the
 * recovered `lambda` can be sanity-checked against the generating value. `knn` weights are
 * row-standardized `k = 4` nearest neighbors of random points with an asymmetric sparsity pattern.
 */
export function createErrorScene(
  count: number,
  seed: number,
  lambda: number,
  kind: ErrorSceneWeights
): {
  weights: DiagnosticsOracleWeights;
  predictors: Float32Array;
  response: Float32Array;
  predictorCount: number;
} {
  const random = createDiagnosticsRandom(seed);
  let weights: DiagnosticsOracleWeights;
  let n = count;
  if (kind === 'knn') {
    const xs = Array.from({length: n}, () => random() * 10);
    const ys = Array.from({length: n}, () => random() * 10);
    const offsets = [0];
    const neighbors: number[] = [];
    const values: number[] = [];
    for (let row = 0; row < n; row++) {
      const order = Array.from({length: n}, (_, index) => index)
        .filter(index => index !== row)
        .sort(
          (a, b) =>
            (xs[a] - xs[row]) ** 2 +
              (ys[a] - ys[row]) ** 2 -
              (xs[b] - xs[row]) ** 2 -
              (ys[b] - ys[row]) ** 2 || a - b
        );
      for (const neighbor of order.slice(0, 4).sort((a, b) => a - b)) {
        neighbors.push(neighbor);
        values.push(0.25);
      }
      offsets.push(neighbors.length);
    }
    weights = {
      offsets: Uint32Array.from(offsets),
      neighbors: Uint32Array.from(neighbors),
      weights: Float32Array.from(values)
    };
  } else {
    const side = Math.round(Math.sqrt(count));
    n = side * side;
    weights = createLatticeWeights(side, kind === 'lattice-row-standardized');
  }
  const w = toDenseWeights(weights);
  const predictors = new Float32Array(n * 2);
  const mean = Array.from({length: n}, () => [0]);
  const noise = Array.from({length: n}, () => [0]);
  for (let row = 0; row < n; row++) {
    predictors[row * 2] = 10 * random() + 3;
    predictors[row * 2 + 1] = random() * 4 - 2;
    mean[row][0] = 1 + 2 * predictors[row * 2] - predictors[row * 2 + 1];
    noise[row][0] = (random() * 2 - 1) * 2;
  }
  // Binary weights have row sums up to 4, so scale lambda to keep I - lambda W invertible.
  const effectiveLambda = kind === 'lattice-binary' ? lambda / 4 : lambda;
  const system = w.map((row, i) =>
    row.map((value, j) => (i === j ? 1 : 0) - effectiveLambda * value)
  );
  const correlated = multiply(invertMatrix(system), noise);
  return {
    weights,
    predictors,
    response: Float32Array.from(mean, (row, i) => row[0] + correlated[i][0]),
    predictorCount: 2
  };
}
