// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getChiSquareSurvival} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';

/** CSR weights in test form. */
export type DiagnosticsOracleWeights = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** One chi-square style test: statistic, degrees of freedom, p-value. */
export type DiagnosticsOracleTest = {statistic: number; degreesOfFreedom: number; pValue: number};

/** Result of {@link computeSpatialRegressionDiagnosticsOnCPU}. */
export type DiagnosticsOracleResult = {
  lmLag: DiagnosticsOracleTest;
  lmError: DiagnosticsOracleTest;
  robustLmLag: DiagnosticsOracleTest;
  robustLmError: DiagnosticsOracleTest;
  lmSarma: DiagnosticsOracleTest;
  sigmaSquared: number;
  trace: number;
  information: number;
  weightsSum: number;
  moranI: number;
  moranExpectation: number;
  moranVariance: number;
  moranZ: number;
  moranPValue: number;
  residuals: Float64Array;
};

type Matrix = number[][];

function zeros(rows: number, columns: number): Matrix {
  return Array.from({length: rows}, () => new Array<number>(columns).fill(0));
}

export function multiply(left: Matrix, right: Matrix): Matrix {
  const result = zeros(left.length, right[0].length);
  for (let i = 0; i < left.length; i++) {
    for (let k = 0; k < right.length; k++) {
      const value = left[i][k];
      if (value !== 0) {
        for (let j = 0; j < right[0].length; j++) {
          result[i][j] += value * right[k][j];
        }
      }
    }
  }
  return result;
}

export function transpose(matrix: Matrix): Matrix {
  const result = zeros(matrix[0].length, matrix.length);
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[0].length; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  return result;
}

function trace(matrix: Matrix): number {
  let sum = 0;
  for (let i = 0; i < matrix.length; i++) {
    sum += matrix[i][i];
  }
  return sum;
}

/** Gauss-Jordan inverse with partial pivoting. */
export function invertMatrix(matrix: Matrix): Matrix {
  const size = matrix.length;
  const work = matrix.map((row, i) => [
    ...row,
    ...Array.from({length: size}, (_, j) => +(i === j))
  ]);
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(work[row][column]) > Math.abs(work[pivot][column])) {
        pivot = row;
      }
    }
    [work[column], work[pivot]] = [work[pivot], work[column]];
    const divisor = work[column][column];
    for (let j = 0; j < 2 * size; j++) {
      work[column][j] /= divisor;
    }
    for (let row = 0; row < size; row++) {
      if (row !== column) {
        const factor = work[row][column];
        for (let j = 0; j < 2 * size; j++) {
          work[row][j] -= factor * work[column][j];
        }
      }
    }
  }
  return work.map(row => row.slice(size));
}

/** Builds the dense `n x n` matrix of a CSR weights structure. */
export function toDenseWeights(weights: DiagnosticsOracleWeights): Matrix {
  const rows = weights.offsets.length - 1;
  const dense = zeros(rows, rows);
  for (let row = 0; row < rows; row++) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      dense[row][weights.neighbors[slot]] = weights.weights[slot];
    }
  }
  return dense;
}

/** Builds rook-contiguity CSR weights of a `side x side` lattice; optionally row-standardized. */
export function createLatticeWeights(
  side: number,
  rowStandardized: boolean
): DiagnosticsOracleWeights {
  const offsets = [0];
  const neighbors: number[] = [];
  const values: number[] = [];
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      const row = y * side + x;
      const list: number[] = [];
      if (y > 0) list.push(row - side);
      if (x > 0) list.push(row - 1);
      if (x < side - 1) list.push(row + 1);
      if (y < side - 1) list.push(row + side);
      for (const neighbor of list) {
        neighbors.push(neighbor);
        values.push(rowStandardized ? 1 / list.length : 1);
      }
      offsets.push(neighbors.length);
    }
  }
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(values)
  };
}

/**
 * Float64 oracle written from the textbook matrix definitions (Anselin 1988, Anselin et al. 1996,
 * Cliff and Ord 1981; spreg `diagnostics_sp.LMtests` and `MoranRes`) with dense `n x n` algebra:
 * `M = I - Z (Z'Z)^-1 Z'`, `T = tr(W'W + WW)`, `J` from `W Z b`, and the Moran variance from
 * `tr(MWMW')`, `tr(MWMW)`, `tr(MW)` directly. It shares no algebra with the GPU's Gram expansion.
 */
export function computeSpatialRegressionDiagnosticsOnCPU(
  weights: DiagnosticsOracleWeights,
  predictors: Float32Array,
  response: Float32Array,
  predictorCount: number
): DiagnosticsOracleResult {
  const n = response.length;
  const p = predictorCount + 1;
  const w = toDenseWeights(weights);
  const z = zeros(n, p);
  for (let row = 0; row < n; row++) {
    z[row][0] = 1;
    for (let column = 0; column < predictorCount; column++) {
      z[row][column + 1] = predictors[row * predictorCount + column];
    }
  }
  const y = Array.from(response, value => [value]);
  const zt = transpose(z);
  const inverse = invertMatrix(multiply(zt, z));
  const projector = multiply(multiply(z, inverse), zt);
  const annihilator = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      annihilator[i][j] = (i === j ? 1 : 0) - projector[i][j];
    }
  }
  const coefficients = multiply(multiply(inverse, zt), y);
  const fitted = multiply(z, coefficients);
  const e = y.map((value, i) => [value[0] - fitted[i][0]]);
  const et = transpose(e);
  const wt = transpose(w);
  const sigmaSquared = multiply(et, e)[0][0] / n;
  const weightsSum = w.reduce((sum, row) => sum + row.reduce((a, b) => a + b, 0), 0);
  const traceT = trace(multiply(wt, w)) + trace(multiply(w, w));
  const errorGradient = multiply(multiply(et, w), e)[0][0] / sigmaSquared;
  const wxb = multiply(w, fitted);
  const lagGradient = multiply(multiply(et, w), y)[0][0] / sigmaSquared;
  const information =
    multiply(multiply(transpose(wxb), annihilator), wxb)[0][0] / sigmaSquared + traceT;
  const test = (statistic: number, degreesOfFreedom: number): DiagnosticsOracleTest => ({
    statistic,
    degreesOfFreedom,
    pValue: getChiSquareSurvival(statistic, degreesOfFreedom)
  });
  const lmError = errorGradient ** 2 / traceT;
  const lmLag = lagGradient ** 2 / information;
  const robustLmError =
    (errorGradient - (traceT * lagGradient) / information) ** 2 /
    (traceT - (traceT * traceT) / information);
  const robustLmLag = (lagGradient - errorGradient) ** 2 / (information - traceT);
  const scale = n / weightsSum;
  const moranI = (scale * multiply(multiply(et, w), e)[0][0]) / multiply(et, e)[0][0];
  const mw = multiply(annihilator, w);
  const degrees = n - p;
  const moranExpectation = (scale * trace(mw)) / degrees;
  const moranVariance =
    (scale ** 2 *
      (trace(multiply(multiply(mw, annihilator), wt)) + trace(multiply(mw, mw)) + trace(mw) ** 2)) /
      (degrees * (degrees + 2)) -
    moranExpectation ** 2;
  const moranZ = (moranI - moranExpectation) / Math.sqrt(moranVariance);
  return {
    lmLag: test(lmLag, 1),
    lmError: test(lmError, 1),
    robustLmLag: test(robustLmLag, 1),
    robustLmError: test(robustLmError, 1),
    lmSarma: test(lmLag + robustLmError, 2),
    sigmaSquared,
    trace: traceT,
    information,
    weightsSum,
    moranI,
    moranExpectation,
    moranVariance,
    moranZ,
    moranPValue: getChiSquareSurvival(moranZ * moranZ, 1),
    residuals: Float64Array.from(e, value => value[0])
  };
}

/** Deterministic xorshift generator in `[0, 1)`. */
export function createDiagnosticsRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/**
 * Lattice regression scene with spatially dependent errors: `u = eps + smoothing passes of W eps`,
 * `y = 1 + 2 x1 - x2 + u`. A larger `dependence` raises the error-dependence tests.
 */
export function createDiagnosticsScene(
  side: number,
  seed: number,
  dependence: number,
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
  const predictors = new Float32Array(n * 2);
  const noise = new Float64Array(n);
  for (let row = 0; row < n; row++) {
    predictors[row * 2] = 10 * random() + 3;
    predictors[row * 2 + 1] = random() * 4 - 2;
    noise[row] = random() * 2 - 1;
  }
  let errors = Float64Array.from(noise);
  for (let pass = 0; pass < 6; pass++) {
    const next = Float64Array.from(noise);
    for (let row = 0; row < n; row++) {
      let lag = 0;
      let count = 0;
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        lag += errors[weights.neighbors[slot]];
        count++;
      }
      next[row] += (dependence * lag) / count;
    }
    errors = next;
  }
  const response = new Float32Array(n);
  for (let row = 0; row < n; row++) {
    response[row] = 1 + 2 * predictors[row * 2] - predictors[row * 2 + 1] + errors[row];
  }
  return {weights, predictors, response, predictorCount: 2};
}

/**
 * Scene on random points with symmetrized k-nearest-neighbor weights (union of the directed kNN
 * pattern, then row-standardized, so the values are asymmetric but the pattern is symmetric) and
 * spatially dependent errors from a smoothing pass over those weights. With `symmetrize = false`
 * the plain directed kNN pattern is kept (row-standardized), whose pattern is asymmetric.
 */
export function createNearestNeighborScene(
  count: number,
  neighborCount: number,
  seed: number,
  dependence: number,
  symmetrize = true
): {
  weights: DiagnosticsOracleWeights;
  predictors: Float32Array;
  response: Float32Array;
  predictorCount: number;
} {
  const random = createDiagnosticsRandom(seed);
  const xs = Array.from({length: count}, () => random() * 10);
  const ys = Array.from({length: count}, () => random() * 10);
  const sets = Array.from({length: count}, () => new Set<number>());
  for (let row = 0; row < count; row++) {
    const order = Array.from({length: count}, (_, index) => index)
      .filter(index => index !== row)
      .sort(
        (a, b) =>
          (xs[a] - xs[row]) ** 2 +
            (ys[a] - ys[row]) ** 2 -
            (xs[b] - xs[row]) ** 2 -
            (ys[b] - ys[row]) ** 2 || a - b
      );
    for (const neighbor of order.slice(0, neighborCount)) {
      sets[row].add(neighbor);
      if (symmetrize) sets[neighbor].add(row);
    }
  }
  const offsets = [0];
  const neighbors: number[] = [];
  const values: number[] = [];
  for (let row = 0; row < count; row++) {
    const list = [...sets[row]].sort((a, b) => a - b);
    for (const neighbor of list) {
      neighbors.push(neighbor);
      values.push(1 / list.length);
    }
    offsets.push(neighbors.length);
  }
  const weights = {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(values)
  };
  const predictors = new Float32Array(count * 2);
  const noise = new Float64Array(count);
  for (let row = 0; row < count; row++) {
    predictors[row * 2] = 10 * random() + 3;
    predictors[row * 2 + 1] = random() * 4 - 2;
    noise[row] = random() * 2 - 1;
  }
  let errors = Float64Array.from(noise);
  for (let pass = 0; pass < 6; pass++) {
    const next = Float64Array.from(noise);
    for (let row = 0; row < count; row++) {
      let lag = 0;
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        lag += errors[neighbors[slot]] / (offsets[row + 1] - offsets[row]);
      }
      next[row] += dependence * lag;
    }
    errors = next;
  }
  const response = new Float32Array(count);
  for (let row = 0; row < count; row++) {
    response[row] = 1 + 2 * predictors[row * 2] - predictors[row * 2 + 1] + errors[row];
  }
  return {weights, predictors, response, predictorCount: 2};
}
