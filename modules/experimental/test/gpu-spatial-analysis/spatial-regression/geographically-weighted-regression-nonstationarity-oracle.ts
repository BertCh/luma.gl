// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUGeographicallyWeightedRegressionSettings} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getPermutationSeedKey} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {
  getFeistelPermutationIndex,
  getFeistelRoundKeys
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';

/** Scene of the non-stationarity specs. */
export type NonstationarityScene = {
  positions: Float32Array;
  predictors: Float32Array;
  predictorCount: number;
  response: Float32Array;
};

function createRandom(seed: number): () => number {
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
 * Jittered `side x side` grid on `[0, 10]^2` with two predictors and a response whose first slope
 * varies strongly west to east, whose second slope is constant and whose intercept varies mildly,
 * so the first slope must test as non-stationary and the second must not.
 */
export function createNonstationarityScene(side: number, seed: number): NonstationarityScene {
  const random = createRandom(seed);
  const rows = side * side;
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const x = ((row % side) + 0.2 + 0.6 * random()) * (10 / side);
    const y = (Math.floor(row / side) + 0.2 + 0.6 * random()) * (10 / side);
    positions[2 * row] = x;
    positions[2 * row + 1] = y;
    const x1 = random() * 4 - 2 + 1;
    const x2 = random() * 4 - 2;
    predictors[2 * row] = x1;
    predictors[2 * row + 1] = x2;
    const slope1 = -2 + 0.8 * Math.fround(x);
    const intercept = 1 + 0.1 * Math.fround(y);
    response[row] =
      intercept + slope1 * Math.fround(x1) + 0.5 * Math.fround(x2) + 0.6 * (random() - 0.5);
  }
  return {positions, predictors, predictorCount: 2, response};
}

/**
 * The relabelling of permutation `permutation` (1-based): location `j` receives the observation of
 * row `result[j]`, for `count` included rows (the keyed Feistel bijection of the permutation
 * contributors).
 */
export function getNonstationarityAssignment(
  count: number,
  seed: number,
  permutation: number
): number[] {
  const roundKeys = getFeistelRoundKeys(getPermutationSeedKey(seed), permutation);
  return Array.from({length: count}, (_, location) =>
    getFeistelPermutationIndex(location, count, roundKeys)
  );
}

/** Population standard deviation of each coefficient column over rows whose fit succeeded. */
export function getLocalEstimateStandardDeviations(
  coefficients: readonly number[],
  coefficientCount: number
): number[] {
  const rows = coefficients.length / coefficientCount;
  const sums = new Array<number>(coefficientCount).fill(0);
  const squares = new Array<number>(coefficientCount).fill(0);
  let count = 0;
  for (let row = 0; row < rows; row++) {
    const values = coefficients.slice(row * coefficientCount, (row + 1) * coefficientCount);
    if (!values.every(Number.isFinite)) {
      continue;
    }
    count++;
    values.forEach((value, column) => {
      sums[column] += value;
      squares[column] += value * value;
    });
  }
  return sums.map((sum, column) =>
    count > 0 ? Math.sqrt(Math.max(squares[column] / count - (sum / count) ** 2, 0)) : Number.NaN
  );
}

/** Result of {@link computeNonstationarityOnCPU}. */
export type NonstationarityOracleResult = {
  /** Row 0 is the observed run, rows `1..P` the permutations; each holds one value per coefficient. */
  standardDeviations: number[][];
  /** Per coefficient: `(g + 1) / (P + 1)` with `g = #{sd_p >= sd_observed}`. */
  pseudoPValues: number[];
  exceedances: number[];
};

/**
 * Float64 reference of `GPUGeographicallyWeightedRegressionNonstationarityTest`: every permutation
 * physically rearranges `(X, y)` among the locations by the Feistel assignment and refits with the
 * float64 `GPUGeographicallyWeightedRegression` oracle at the same fixed bandwidth, then compares
 * the standard deviations of the local estimates.
 */
export function computeNonstationarityOnCPU(
  scene: NonstationarityScene,
  settings: GPUGeographicallyWeightedRegressionSettings,
  permutations: number,
  seed: number
): NonstationarityOracleResult {
  const {positions, predictors, predictorCount, response} = scene;
  const rows = response.length;
  const coefficientCount = predictorCount + 1;
  const run = (assignment: readonly number[]) => {
    const permutedPredictors = new Float32Array(predictors.length);
    const permutedResponse = new Float32Array(rows);
    for (let location = 0; location < rows; location++) {
      const source = assignment[location];
      for (let column = 0; column < predictorCount; column++) {
        permutedPredictors[location * predictorCount + column] =
          predictors[source * predictorCount + column];
      }
      permutedResponse[location] = response[source];
    }
    const result = computeGeographicallyWeightedRegressionOnCPU({
      positions,
      predictors: permutedPredictors,
      predictorCount,
      response: permutedResponse,
      settings
    });
    return getLocalEstimateStandardDeviations(result.coefficients, coefficientCount);
  };
  const identity = Array.from({length: rows}, (_, row) => row);
  const standardDeviations = [run(identity)];
  for (let permutation = 1; permutation <= permutations; permutation++) {
    standardDeviations.push(run(getNonstationarityAssignment(rows, seed, permutation)));
  }
  const exceedances = standardDeviations[0].map(
    (observed, column) => standardDeviations.slice(1).filter(row => row[column] >= observed).length
  );
  return {
    standardDeviations,
    exceedances,
    pseudoPValues: exceedances.map(count => (count + 1) / (permutations + 1))
  };
}
