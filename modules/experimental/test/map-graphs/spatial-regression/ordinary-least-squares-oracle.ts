// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '../../../src/map-graphs/spatial-regression/ordinary-least-squares-parameters';
import {getChiSquareSurvival} from '../../../src/map-graphs/spatial-regression/ordinary-least-squares-statistics';

/** Input of {@link fitOrdinaryLeastSquaresOnCPU}. */
export type OrdinaryLeastSquaresOracleInput = {
  /** Row-major `row * predictorCount + column`. */
  predictors: Float32Array;
  response: Float32Array;
  mask?: Uint32Array;
  predictorCount: number;
  ridgeLambda?: number;
};

/** Result of {@link fitOrdinaryLeastSquaresOnCPU}, same layouts as the GPU outputs. */
export type OrdinaryLeastSquaresOracleResult = {
  status: number;
  rowCount: number;
  coefficients: Float64Array;
  standardErrors: Float64Array;
  tStatistics: Float64Array;
  summary: Float64Array;
  residuals: Float64Array;
  fitted: Float64Array;
};

type CoreFit = {
  status: number;
  rowCount: number;
  means: number[];
  slopes: number[];
  scale: number[];
  correlationInverse: number[][];
  moments: number[][];
};

function factorCholesky(matrix: number[][]): number[][] | null {
  const size = matrix.length;
  const lower = matrix.map(row => row.map(() => 0));
  for (let column = 0; column < size; column++) {
    let diagonal = matrix[column][column];
    for (let k = 0; k < column; k++) {
      diagonal -= lower[column][k] * lower[column][k];
    }
    if (!(diagonal > 0)) {
      return null;
    }
    const pivot = Math.sqrt(diagonal);
    lower[column][column] = pivot;
    for (let row = column + 1; row < size; row++) {
      let value = matrix[row][column];
      for (let k = 0; k < column; k++) {
        value -= lower[row][k] * lower[column][k];
      }
      lower[row][column] = value / pivot;
    }
  }
  return lower;
}

function solveCholesky(lower: number[][], rightHandSide: number[]): number[] {
  const size = lower.length;
  const solution = rightHandSide.slice();
  for (let row = 0; row < size; row++) {
    for (let k = 0; k < row; k++) {
      solution[row] -= lower[row][k] * solution[k];
    }
    solution[row] /= lower[row][row];
  }
  for (let row = size - 1; row >= 0; row--) {
    for (let k = row + 1; k < size; k++) {
      solution[row] -= lower[k][row] * solution[k];
    }
    solution[row] /= lower[row][row];
  }
  return solution;
}

/** Float64 centered-moment fit of the used rows; mirrors the GPU scaling and singularity rule. */
function fitCore(
  rows: {x: number[]; y: number}[],
  predictorCount: number,
  ridgeLambda: number
): CoreFit {
  const k = predictorCount;
  const n = rows.length;
  const empty = (status: number): CoreFit => ({
    status,
    rowCount: n,
    means: [],
    slopes: [],
    scale: [],
    correlationInverse: [],
    moments: []
  });
  if (n <= k + 1) {
    return empty(2);
  }
  const d = k + 1;
  const means = new Array<number>(d).fill(0);
  for (const row of rows) {
    for (let j = 0; j < k; j++) {
      means[j] += row.x[j];
    }
    means[k] += row.y;
  }
  for (let j = 0; j < d; j++) {
    means[j] /= n;
  }
  const moments = Array.from({length: d}, () => new Array<number>(d).fill(0));
  for (const row of rows) {
    const centered = [...row.x.map((value, j) => value - means[j]), row.y - means[k]];
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < d; b++) {
        moments[a][b] += centered[a] * centered[b];
      }
    }
  }
  const scale: number[] = [];
  for (let j = 0; j < k; j++) {
    if (!(moments[j][j] > 0)) {
      return empty(1);
    }
    scale.push(Math.sqrt(moments[j][j]));
  }
  const correlation = Array.from({length: k}, (_, a) =>
    Array.from({length: k}, (_, b) => {
      const entry = moments[a][b] / (scale[a] * scale[b]);
      return a === b ? entry + ridgeLambda / (scale[a] * scale[a]) : entry;
    })
  );
  const lower = factorCholesky(correlation);
  if (
    !lower ||
    lower.some((row, index) => !(row[index] >= GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE))
  ) {
    return empty(1);
  }
  const rightHandSide = scale.map((value, j) => moments[j][k] / value);
  const gamma = solveCholesky(lower, rightHandSide);
  const correlationInverse = Array.from({length: k}, (_, column) =>
    solveCholesky(
      lower,
      Array.from({length: k}, (_, row) => (row === column ? 1 : 0))
    )
  );
  return {
    status: 0,
    rowCount: n,
    means,
    slopes: gamma.map((value, j) => value / scale[j]),
    scale,
    correlationInverse,
    moments
  };
}

/**
 * Float64 oracle for `GPUOrdinaryLeastSquares`: centered moments, scaled Cholesky solve, the same
 * pivot tolerance, statistics, Jarque-Bera, and Koenker Breusch-Pagan (regression of `e^2` on the
 * predictors without ridge). Predictor and response values are the exact float32 inputs.
 */
export function fitOrdinaryLeastSquaresOnCPU(
  input: OrdinaryLeastSquaresOracleInput
): OrdinaryLeastSquaresOracleResult {
  const k = input.predictorCount;
  const rowCount = input.response.length;
  const lambda = input.ridgeLambda !== undefined && input.ridgeLambda > 0 ? input.ridgeLambda : 0;
  const usedIndices: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (input.mask && input.mask[row] === 0) {
      continue;
    }
    let finite = Number.isFinite(input.response[row]);
    for (let j = 0; j < k; j++) {
      finite = finite && Number.isFinite(input.predictors[row * k + j]);
    }
    if (finite) {
      usedIndices.push(row);
    }
  }
  const rows = usedIndices.map(row => ({
    x: Array.from({length: k}, (_, j) => input.predictors[row * k + j]),
    y: input.response[row]
  }));
  const fit = fitCore(rows, k, lambda);
  const result: OrdinaryLeastSquaresOracleResult = {
    status: fit.status,
    rowCount: rows.length,
    coefficients: new Float64Array(k + 1).fill(NaN),
    standardErrors: new Float64Array(k + 1).fill(NaN),
    tStatistics: new Float64Array(k + 1).fill(NaN),
    summary: new Float64Array(GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH).fill(NaN),
    residuals: new Float64Array(rowCount).fill(NaN),
    fitted: new Float64Array(rowCount).fill(NaN)
  };
  result.summary[0] = rows.length;
  if (fit.status !== 0) {
    return result;
  }
  const n = rows.length;
  const p = k + 1;
  const freedom = n - p;
  const squared: number[] = [];
  let sumSquares = 0;
  let sumCubes = 0;
  let sumFourth = 0;
  rows.forEach((row, index) => {
    let fit_ = 0;
    for (let j = 0; j < k; j++) {
      fit_ += fit.slopes[j] * (row.x[j] - fit.means[j]);
    }
    const residual = row.y - fit.means[k] - fit_;
    result.residuals[usedIndices[index]] = residual;
    result.fitted[usedIndices[index]] = fit.means[k] + fit_;
    squared.push(residual * residual);
    sumSquares += residual * residual;
    sumCubes += residual ** 3;
    sumFourth += residual ** 4;
  });
  const totalSumSquares = fit.moments[k][k];
  let intercept = fit.means[k];
  for (let j = 0; j < k; j++) {
    intercept -= fit.slopes[j] * fit.means[j];
  }
  result.coefficients[0] = intercept;
  fit.slopes.forEach((value, j) => (result.coefficients[1 + j] = value));
  const sigmaSquared = sumSquares / freedom;
  const u = fit.means.slice(0, k).map((mean, j) => mean / fit.scale[j]);
  let quadratic = 0;
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      quadratic += u[a] * fit.correlationInverse[a][b] * u[b];
    }
  }
  result.standardErrors[0] = Math.sqrt(sigmaSquared * (1 / n + quadratic));
  for (let j = 0; j < k; j++) {
    result.standardErrors[1 + j] = Math.sqrt(
      (sigmaSquared * fit.correlationInverse[j][j]) / fit.scale[j] ** 2
    );
  }
  for (let j = 0; j < p; j++) {
    result.tStatistics[j] =
      result.standardErrors[j] > 0 ? result.coefficients[j] / result.standardErrors[j] : NaN;
  }
  const rSquared = totalSumSquares > 0 ? 1 - sumSquares / totalSumSquares : NaN;
  const logLikelihood =
    -0.5 * n * (Math.log(2 * Math.PI) + Math.log(Math.max(sumSquares / n, 1.17549435e-38)) + 1);
  const m2 = sumSquares / n;
  const skewness = sumCubes / n / m2 ** 1.5;
  const kurtosis = sumFourth / n / (m2 * m2);
  const jarqueBera = (n / 6) * (skewness ** 2 + (kurtosis - 3) ** 2 / 4);
  // Koenker Breusch-Pagan: n R2 of e^2 ~ X (no ridge).
  let breuschPagan = NaN;
  let breuschPaganP = NaN;
  const breuschPaganFit = fitCore(
    rows.map((row, index) => ({x: row.x, y: squared[index]})),
    k,
    0
  );
  if (breuschPaganFit.status === 0 && breuschPaganFit.moments[k][k] > 0) {
    let explained = 0;
    for (let j = 0; j < k; j++) {
      explained += breuschPaganFit.slopes[j] * breuschPaganFit.moments[j][k];
    }
    const bpR2 = Math.min(1, Math.max(0, explained / breuschPaganFit.moments[k][k]));
    breuschPagan = n * bpR2;
    breuschPaganP = getChiSquareSurvival(breuschPagan, k);
  }
  const summary = result.summary;
  summary[1] = rSquared;
  summary[2] = 1 - ((1 - rSquared) * (n - 1)) / freedom;
  summary[3] = sigmaSquared;
  summary[4] = logLikelihood;
  summary[5] = 2 * p - 2 * logLikelihood;
  summary[6] = p * Math.log(n) - 2 * logLikelihood;
  summary[7] = jarqueBera;
  summary[8] = Math.exp(-jarqueBera / 2);
  summary[9] = breuschPagan;
  summary[10] = breuschPaganP;
  summary[11] = sumSquares;
  summary[12] = totalSumSquares;
  summary[13] = skewness;
  summary[14] = kurtosis;
  summary[15] = lambda;
  return result;
}

/** Seeded LCG in [0, 1) (Numerical Recipes constants). */
export function createOrdinaryLeastSquaresRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** One synthetic regression scene. */
export type OrdinaryLeastSquaresScene = {
  predictors: Float32Array;
  response: Float32Array;
  mask?: Uint32Array;
  predictorCount: number;
};

/**
 * Generates `y = intercept + sum(b x) + noise` with predictors offset from zero (so centering
 * matters) and noise from a sum of four uniforms. `noiseScale = 0` gives exact linear data. Row
 * `5 * i + 3` is masked when `withMask`, and a few rows get NaN in the predictors or response.
 */
export function createOrdinaryLeastSquaresScene(
  seed: number,
  rowCount: number,
  intercept: number,
  slopes: readonly number[],
  noiseScale: number,
  options: {withMask?: boolean; withNonFinite?: boolean} = {}
): OrdinaryLeastSquaresScene {
  const random = createOrdinaryLeastSquaresRandom(seed);
  const k = slopes.length;
  const predictors = new Float32Array(rowCount * k);
  const response = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    let value = intercept;
    for (let j = 0; j < k; j++) {
      const x = 40 + 10 * random() + j * 3;
      predictors[row * k + j] = x;
      value += slopes[j] * x;
    }
    const noise = (random() + random() + random() + random() - 2) * noiseScale;
    response[row] = value + noise;
  }
  const scene: OrdinaryLeastSquaresScene = {
    predictors,
    response,
    predictorCount: k
  };
  if (options.withMask) {
    scene.mask = Uint32Array.from({length: rowCount}, (_, row) => (row % 5 === 3 ? 0 : 1));
  }
  if (options.withNonFinite) {
    response[1] = NaN;
    predictors[7 * k] = Infinity;
    predictors[(rowCount - 2) * k + (k - 1)] = NaN;
  }
  return scene;
}
