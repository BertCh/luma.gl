// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getChiSquareSurvival} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {getSymmetricEigenvalueRangeOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-regression-solve';
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
 * `lambda` minimizes `|G [l, l^2, s]' - g|^2` over `l` in `[-b, b]`, `b = 0.99 / ||W||_inf` (spreg's 0.99
 * for row-standardized weights), and `s >= 0` (spreg's
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
  // 0.99 / ||W||_inf, as the GPU: spreg's 0.99 for a row-standardized W.
  const bound =
    0.99 / Math.max(...w.map(row => row.reduce((sum, value) => sum + Math.abs(value), 0)));
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

/** Result of {@link fitSpatialErrorGMHomOnCPU}. */
export type ErrorGMHomOracleResult = ErrorGMOracleResult & {
  /** Standard error, z statistic and two-sided normal p-value of `lambda`. */
  lambdaStandardError: number;
  lambdaZStatistic: number;
  lambdaPValue: number;
  /** First-step moment estimate (spreg `lambda1`); `lambda` is the weighted second step. */
  lambda1: number;
  /** `lambda` bound, `0.99 / ||W||_inf`. */
  lambdaBound: number;
  /** Moment matrices of the second step: `G` row-major 2 x 2 and `g`. */
  momentMatrix: number[];
  momentVector: number[];
  /** `psi` at `lambda1` and at `lambda`, row-major 2 x 2. */
  psiAtLambda1: number[];
  psiAtLambda: number[];
};

/** Traces of `W` that the homoskedastic variance needs, computed here exactly with dense algebra. */
export type ErrorGMHomTraces = {
  /** `tr(W)`, `tr(W^2)`. */
  powerTraces: [number, number];
  /** `[tr(W'W), tr(W'WW), tr((W'W)^2), sum_i c_i^2]` with `c_i = sum_l w_li^2`. */
  gramTraces: [number, number, number, number];
};

/** Exact power and Gram traces of a dense `W`, the layout of `GPUSpatialWeightsTraces`. */
export function getErrorGMHomTraces(w: number[][]): ErrorGMHomTraces {
  const trace = (matrix: number[][]) => matrix.reduce((sum, row, i) => sum + row[i], 0);
  const wtw = multiply(transpose(w), w);
  const columnSquares = w[0].map((_, column) =>
    w.reduce((sum, row) => sum + row[column] * row[column], 0)
  );
  return {
    powerTraces: [trace(w), trace(multiply(w, w))],
    gramTraces: [
      trace(wtw),
      trace(multiply(wtw, w)),
      trace(multiply(wtw, wtw)),
      columnSquares.reduce((sum, value) => sum + value * value, 0)
    ]
  };
}

/**
 * Float64 oracle of spreg 1.9.1 `GM_Error_Hom` (`A1 = 'hom_sc'`, `max_iter = 1`; Drukker, Egger and
 * Prucha 2013, Anselin 2011), written with the closed forms the GPU kernels use instead of the
 * `n x n` matrices `A1`, `A2`. With `t = tr(W'W)`, `s = 1 + (t / n)^2`, `A1 = (W'W - (t / n) I) / s`
 * and `A2 = (W + W') / 2`:
 * - `G = [[2 ((Wu)'WWu - (t/n) u'Wu) / s, -((WWu)'WWu - (t/n) (Wu)'Wu) / s],
 *   [u'WWu + (Wu)'Wu, -(Wu)'WWu]] / n` and `g = [((Wu)'Wu - (t/n) u'u) / s, u'Wu] / n`;
 * - `tr(A1 A1) = (tr((W'W)^2) - t^2 / n) / s^2`,
 *   `tr(A1 2 A2) = (2 tr(W'WW) - 2 (t/n) tr(W)) / s`, `tr(2 A2 A2) = tr(W^2) + t`,
 *   `diag(A1)'diag(A1) = (sum c_i^2 - t^2 / n) / s^2`.
 * The steps are spreg's: OLS residuals give `lambda1` (unweighted `|G p - g|^2`); FGLS at `lambda1`
 * gives the reported betas and `u1 = y - X b1`; `psi` at `lambda1` weights the second minimization
 * `(G p - g)' psi^-1 (G p - g)` for `lambda`; `Var(b) = sigma2 (Xs'Xs)^-1` with `Xs`, `u_s` filtered
 * at `lambda` and `Var(lambda) = (j' psi^-1 j)^-1 / n`, `j = G [1, 2 lambda]'`, `psi` at `lambda`.
 * Both minimizations are global (20001-point grid plus golden section) on
 * `[-0.99 / ||W||_inf, 0.99 / ||W||_inf]`; spreg runs a local L-BFGS-B on `[-0.99, 0.99]`.
 * `traces` defaults to the exact traces of `W`.
 */
export function fitSpatialErrorGMHomOnCPU(
  weights: DiagnosticsOracleWeights,
  predictors: Float32Array,
  response: Float32Array,
  predictorCount: number,
  traces?: ErrorGMHomTraces
): ErrorGMHomOracleResult {
  const n = response.length;
  const k = predictorCount;
  const w = toDenseWeights(weights);
  const exactTraces = traces ?? getErrorGMHomTraces(w);
  const y = Array.from(response, value => [value]);
  const x = Array.from({length: n}, (_, row) => [
    1,
    ...Array.from({length: k}, (_, column) => predictors[row * k + column])
  ]);
  const xt = transpose(x);
  const dot = (left: number[][], right: number[][]) =>
    left.reduce((sum, row, i) => sum + row[0] * right[i][0], 0);
  const lagOf = (vector: number[][]) => multiply(w, vector);
  const [traceW, traceWW] = exactTraces.powerTraces;
  const [t, traceWtWW, traceWtWSquared, columnSquares] = exactTraces.gramTraces;
  const scale = 1 + (t / n) ** 2;
  const bound =
    0.99 / Math.max(...w.map(row => row.reduce((sum, value) => sum + Math.abs(value), 0)));

  const getMoments = (u: number[][]) => {
    const wu = lagOf(u);
    const wwu = lagOf(wu);
    const u2 = dot(u, u);
    const uwu = dot(u, wu);
    const wu2 = dot(wu, wu);
    const uwwu = dot(u, wwu);
    const wuwwu = dot(wu, wwu);
    const wwu2 = dot(wwu, wwu);
    const matrix = [
      [(2 * (wuwwu - (t / n) * uwu)) / scale, -(wwu2 - (t / n) * wu2) / scale],
      [uwwu + wu2, -wuwwu]
    ].map(row => row.map(value => value / n));
    const vector = [(wu2 - (t / n) * u2) / scale, uwu].map(value => value / n);
    return {matrix, vector, wu};
  };
  const getPsi = (u: number[][], wu: number[][], lambda: number) => {
    const filtered = u.map((row, i) => row[0] - lambda * wu[i][0]);
    const sigma = filtered.reduce((sum, value) => sum + value * value, 0) / n;
    const mu3 = filtered.reduce((sum, value) => sum + value ** 3, 0) / n;
    const mu4 = filtered.reduce((sum, value) => sum + value ** 4, 0) / n;
    const trace11 = (traceWtWSquared - (t * t) / n) / scale ** 2;
    const trace12 = (2 * traceWtWW - (2 * t * traceW) / n) / scale;
    const trace22 = traceWW + t;
    const vectorNorm = (columnSquares - (t * t) / n) / scale ** 2;
    const psi11 = 2 * sigma ** 2 * trace11 + (mu4 - 3 * sigma ** 2) * vectorNorm;
    const psi12 = sigma ** 2 * trace12;
    const psi22 = sigma ** 2 * trace22;
    return {
      psi: [
        [psi11 / n, psi12 / n],
        [psi12 / n, psi22 / n]
      ],
      sigma,
      mu3,
      filtered
    };
  };
  const inverse2 = (m: number[][]) => {
    const determinant = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    return [
      [m[1][1] / determinant, -m[0][1] / determinant],
      [-m[1][0] / determinant, m[0][0] / determinant]
    ];
  };
  const minimize = (objective: (lambda: number) => number) => {
    const gridSteps = 20000;
    const gridPoint = (step: number) => bound * (-1 + (2 * step) / gridSteps);
    let best = 0;
    let bestValue = objective(gridPoint(0));
    for (let step = 1; step <= gridSteps; step++) {
      const value = objective(gridPoint(step));
      if (value < bestValue) {
        best = step;
        bestValue = value;
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
    return (low + high) / 2;
  };
  const getResidual = (moments: {matrix: number[][]; vector: number[]}, lambda: number) =>
    [0, 1].map(
      i => moments.matrix[i][0] * lambda + moments.matrix[i][1] * lambda ** 2 - moments.vector[i]
    );

  // Step 1: OLS residuals and the unweighted moment solve.
  const olsBeta = multiply(invertMatrix(multiply(xt, x)), multiply(xt, y));
  const olsFit = multiply(x, olsBeta);
  const u0 = y.map((value, i) => [value[0] - olsFit[i][0]]);
  const first = getMoments(u0);
  const lambda1 = minimize(lambda => {
    const residual = getResidual(first, lambda);
    return residual[0] ** 2 + residual[1] ** 2;
  });
  // Step 2a: FGLS at lambda1 (reported betas), residuals u1 = y - X b1.
  const filter = (matrix: number[][], lambda: number) => {
    const lagged = multiply(w, matrix);
    return matrix.map((row, i) => row.map((value, j) => value - lambda * lagged[i][j]));
  };
  const xs1 = filter(x, lambda1);
  const ys1 = filter(y, lambda1);
  const xs1t = transpose(xs1);
  const beta = multiply(invertMatrix(multiply(xs1t, xs1)), multiply(xs1t, ys1));
  const fit = multiply(x, beta);
  const u1 = y.map((value, i) => [value[0] - fit[i][0]]);
  // Step 2b: weighted moment solve.
  const second = getMoments(u1);
  const psiAtLambda1 = getPsi(u1, second.wu, lambda1).psi;
  const weight = inverse2(psiAtLambda1);
  const lambda = minimize(lambda => {
    const residual = getResidual(second, lambda);
    return (
      residual[0] * (weight[0][0] * residual[0] + weight[0][1] * residual[1]) +
      residual[1] * (weight[1][0] * residual[0] + weight[1][1] * residual[1])
    );
  });
  // Variance at lambda.
  const final = getPsi(u1, second.wu, lambda);
  const xs2 = filter(x, lambda);
  const normalInverse = invertMatrix(multiply(transpose(xs2), xs2));
  const standardErrors = beta.map((_, i) => Math.sqrt(final.sigma * normalInverse[i][i]));
  const coefficients = beta.map(row => row[0]);
  const zStatistics = coefficients.map((value, i) => value / standardErrors[i]);
  const j = [0, 1].map(i => second.matrix[i][0] + 2 * lambda * second.matrix[i][1]);
  const psiInverse = inverse2(final.psi);
  const quadratic =
    j[0] * (psiInverse[0][0] * j[0] + psiInverse[0][1] * j[1]) +
    j[1] * (psiInverse[1][0] * j[0] + psiInverse[1][1] * j[1]);
  const lambdaStandardError = Math.sqrt(1 / quadratic / n);
  const lambdaZStatistic = lambda / lambdaStandardError;
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
    sigmaSquared: final.sigma,
    residualSumOfSquares: u1.reduce((sum, row) => sum + row[0] * row[0], 0),
    pseudoRSquared: covariance ** 2 / (varianceY * varianceFit),
    residuals: u1.map(row => row[0]),
    moments: [],
    lambdaStandardError,
    lambdaZStatistic,
    lambdaPValue: getChiSquareSurvival(lambdaZStatistic ** 2, 1),
    lambda1,
    lambdaBound: bound,
    momentMatrix: second.matrix.flat(),
    momentVector: second.vector,
    psiAtLambda1: psiAtLambda1.flat(),
    psiAtLambda: final.psi.flat()
  };
}

/** Conditioning of the filtered regression of `GPUSpatialErrorGM`. */
export type ErrorGMConditioning = {
  conditionNumber: number;
  correlationConditionNumber: number;
  maximumVarianceInflationFactor: number;
};

/**
 * Float64 mirror of the conditioning the GPU finish kernel reports for the filtered regression at
 * `lambda`: the filtered columns are `[(I - lambda W) 1, (I - lambda W)(x_j - mean_j)]`, their
 * Gram matrix `C` is the centered-model moment matrix, the Belsley condition number is that of the
 * raw filtered `[1, X]` Gram matrix REBUILT from `C` and the means (`raw_jl = C_jl + m_l C_0j +
 * m_j C_0l + m_j m_l C_00`), the correlation condition number is that of the unit-diagonal `C`, and
 * the variance inflation factor is the largest diagonal entry of its inverse over the predictor
 * columns.
 */
export function getErrorGMConditioningOnCPU(
  weights: DiagnosticsOracleWeights,
  predictors: Float32Array,
  predictorCount: number,
  lambda: number
): ErrorGMConditioning {
  const k = predictorCount;
  const q = k + 1;
  const n = predictors.length / k;
  const w = toDenseWeights(weights);
  const means = Array.from(
    {length: k},
    (_, column) =>
      Array.from({length: n}, (_, row) => predictors[row * k + column]).reduce((a, b) => a + b, 0) /
      n
  );
  const centered = Array.from({length: n}, (_, row) => [
    1,
    ...Array.from({length: k}, (_, column) => predictors[row * k + column] - means[column])
  ]);
  const lagged = multiply(w, centered);
  // The constant column is filtered but not centered; the others are centered then filtered.
  const filtered = centered.map((row, i) => row.map((value, j) => value - lambda * lagged[i][j]));
  const moments = multiply(transpose(filtered), filtered);
  const raw = Array.from({length: q}, () => new Array<number>(q).fill(0));
  raw[0][0] = moments[0][0];
  for (let j = 0; j < k; j++) {
    raw[0][1 + j] = raw[1 + j][0] = moments[0][1 + j] + means[j] * moments[0][0];
    for (let l = 0; l < k; l++) {
      raw[1 + j][1 + l] =
        moments[1 + j][1 + l] +
        means[l] * moments[0][1 + j] +
        means[j] * moments[0][1 + l] +
        means[j] * means[l] * moments[0][0];
    }
  }
  const unit = (matrix: number[][]) =>
    matrix.map((row, i) => row.map((value, j) => value / Math.sqrt(matrix[i][i] * matrix[j][j])));
  const ratio = (matrix: number[][]) => {
    const [smallest, largest] = getSymmetricEigenvalueRangeOnCPU(matrix.flat(), q);
    return largest / smallest;
  };
  const correlation = unit(moments);
  const inverse = invertMatrix(correlation);
  return {
    conditionNumber: Math.sqrt(ratio(unit(raw))),
    correlationConditionNumber: ratio(correlation),
    maximumVarianceInflationFactor: Math.max(...inverse.map((row, i) => row[i]).slice(1))
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
  kind: ErrorSceneWeights,
  predictorCount = 2
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
  const predictors = new Float32Array(n * predictorCount);
  const mean = Array.from({length: n}, () => [0]);
  const noise = Array.from({length: n}, () => [0]);
  for (let row = 0; row < n; row++) {
    predictors[row * predictorCount] = 10 * random() + 3;
    predictors[row * predictorCount + 1] = random() * 4 - 2;
    // Extra predictors (beyond the first two) enter with coefficient 0.5; scenes with two
    // predictors keep their random stream.
    let extra = 0;
    for (let column = 2; column < predictorCount; column++) {
      predictors[row * predictorCount + column] = random() * 4 - 2;
      extra += 0.5 * predictors[row * predictorCount + column];
    }
    mean[row][0] =
      1 + 2 * predictors[row * predictorCount] - predictors[row * predictorCount + 1] + extra;
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
    predictorCount
  };
}
