// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  type GPUGlobalSpatialStatisticsProps
} from '../../../src/map-graphs/global-spatial-statistics';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {
  computeGlobalSpatialStatisticsOracle,
  createGridWeights,
  createSeededRandom,
  createWeights,
  getDenseMatrix,
  getGearyMoments,
  getMoranMoments,
  getPermutationMoments,
  type CPUSpatialWeights
} from './global-spatial-statistics-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUGlobalSpatialStatisticsProps> = {}
): GPUGlobalSpatialStatisticsProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    weights: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 40),
      weights: view('float32', 40)
    },
    values: view('float32', 10),
    statistics: ['moran'],
    results: view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length),
    ...overrides
  };
}

/** Mirrors the WGSL centered-form randomization variance of x'Wx (zero diagonal). */
function getCenteredQuadraticFormVariance(matrix: number[][], values: number[]): number {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const z = values.map(value => value - mean);
  const power = (k: number) => z.reduce((total, value) => total + value ** k, 0);
  let s0 = 0;
  let s1 = 0;
  const degrees = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      s0 += matrix[i][j];
      s1 += 0.5 * (matrix[i][j] + matrix[j][i]) ** 2;
      degrees[i] += matrix[i][j];
      degrees[j] += matrix[i][j];
    }
  }
  const s2 = degrees.reduce((total, d) => total + d * d, 0);
  const meanDegree = degrees.reduce((a, b) => a + b, 0) / n;
  const degreeDeviation = degrees.reduce((total, d) => total + (d - meanDegree) ** 2, 0);
  const kurtosis = (n * power(4)) / power(2) ** 2;
  const moranVariance = getMoranMoments(n, s0, s1, s2, kurtosis)[2];
  const scale = (s0 * power(2)) / n;
  return (
    (mean * mean * power(2) * degreeDeviation) / (n - 1) -
    (2 * mean * power(3) * degreeDeviation) / ((n - 1) * (n - 2)) +
    moranVariance * scale * scale
  );
}

function createRandomWeights(n: number, seed: number, density: number): CPUSpatialWeights {
  const random = createSeededRandom(seed);
  const lists = Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j): [number, number] => [j, random()]).filter(
      ([j]) => j !== i && random() < density
    )
  );
  return createWeights(lists);
}

function quadraticForm(matrix: number[][], left: number[], right: number[]): number {
  let total = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) {
      total += matrix[i][j] * left[i] * right[j];
    }
  }
  return total;
}

function expectClose(actual: number, expected: number, label: string): void {
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    1e-9 * Math.max(1, Math.abs(expected))
  );
}

it('analytic randomization moments equal exact permutation moments on tiny asymmetric weights', () => {
  for (const seed of [1, 2, 3]) {
    const n = 7;
    const weights = createRandomWeights(n, seed, 0.5);
    const matrix = getDenseMatrix(weights);
    const random = createSeededRandom(seed + 100);
    const values = Array.from({length: n}, () => Math.round(random() * 9) + 1);
    const secondValues = Array.from({length: n}, () => random() * 4 - 2);
    const oracle = computeGlobalSpatialStatisticsOracle({
      weights,
      values: Float32Array.from(values),
      secondValues: Float32Array.from(secondValues)
    });
    const label = `seed ${seed}`;
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const moran = (permuted: number[]) => {
      const z = permuted.map(value => value - mean);
      return ((n / oracle.s0) * quadraticForm(matrix, z, z)) / z.reduce((t, v) => t + v * v, 0);
    };
    const [moranMean, moranVariance] = getPermutationMoments(values, moran);
    expectClose(oracle.moran.expected, moranMean, `${label} EI`);
    expectClose(oracle.moran.varianceRandomization, moranVariance, `${label} VI_rand`);

    const geary = (permuted: number[]) => {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          sum += matrix[i][j] * (permuted[i] - permuted[j]) ** 2;
        }
      }
      const squares = permuted.reduce((t, v) => t + (v - mean) ** 2, 0);
      return ((n - 1) * sum) / (2 * oracle.s0 * squares);
    };
    const [gearyMean, gearyVariance] = getPermutationMoments(values, geary);
    expectClose(gearyMean, 1, `${label} EC`);
    expectClose(oracle.geary.varianceRandomization, gearyVariance, `${label} VC_rand`);

    const sum = values.reduce((a, b) => a + b, 0);
    const sumSquares = values.reduce((a, b) => a + b * b, 0);
    const generalG = (permuted: number[]) =>
      quadraticForm(matrix, permuted, permuted) / (sum * sum - sumSquares);
    const [gMean, gVariance] = getPermutationMoments(values, generalG);
    expectClose(oracle.getisOrdG.expected, gMean, `${label} EG`);
    expectClose(oracle.getisOrdG.varianceRandomization, gVariance, `${label} VG (esda)`);
    expectClose(
      getCenteredQuadraticFormVariance(matrix, values) / (sum * sum - sumSquares) ** 2,
      gVariance,
      `${label} VG (centered)`
    );

    const meanY = secondValues.reduce((a, b) => a + b, 0) / n;
    const zx = values.map(value => value - mean);
    const zxSquares = zx.reduce((t, v) => t + v * v, 0);
    const bivariate = (permuted: number[]) => {
      const zy = permuted.map(value => value - meanY);
      return (
        ((n / oracle.s0) * quadraticForm(matrix, zx, zy)) /
        Math.sqrt(zxSquares * zy.reduce((t, v) => t + v * v, 0))
      );
    };
    const [bivariateMean, bivariateVariance] = getPermutationMoments(secondValues, bivariate);
    expectClose(bivariateMean, 0, `${label} E BV`);
    expectClose(oracle.bivariateMoran.varianceRandomization, bivariateVariance, `${label} V BV`);
  }
});

it('Cliff-Ord join-count moments equal exact permutation moments and the centered forms', () => {
  for (const seed of [4, 5]) {
    const n = 8;
    const weights = createRandomWeights(n, seed, 0.45);
    const matrix = getDenseMatrix(weights);
    const binary = matrix.map(row => row.map(w => (w > 0 ? 1 : 0)));
    const values = [1, 0, 1, 1, 0, 0, 1, 0];
    const oracle = computeGlobalSpatialStatisticsOracle({
      weights,
      values: Float32Array.from(values)
    });
    const blackBlack = (permuted: number[]) => 0.5 * quadraticForm(binary, permuted, permuted);
    const blackWhite = (permuted: number[]) => {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          sum += binary[i][j] * (permuted[i] - permuted[j]) ** 2;
        }
      }
      return 0.5 * sum;
    };
    const [bbMean, bbVariance] = getPermutationMoments(values, blackBlack);
    const [bwMean, bwVariance] = getPermutationMoments(values, blackWhite);
    const join = oracle.joinCount;
    expectClose(join.expectedBlackBlack, bbMean, `seed ${seed} E BB`);
    expectClose(join.varianceBlackBlack, bbVariance, `seed ${seed} V BB`);
    expectClose(join.expectedBlackWhite, bwMean, `seed ${seed} E BW`);
    expectClose(join.varianceBlackWhite, bwVariance, `seed ${seed} V BW`);
    // The GPU evaluates BB through the quadratic form and BW through Geary's moments.
    expectClose(
      0.25 * getCenteredQuadraticFormVariance(binary, values),
      bbVariance,
      `seed ${seed} V BB centered`
    );
    const p = 4 / n;
    const sumSquares = n * p * (1 - p);
    const kurtosis = (n * (n * p * (1 - p) * (p ** 3 + (1 - p) ** 3))) / sumSquares ** 2;
    let s0 = 0;
    let s1 = 0;
    const degrees = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        s0 += binary[i][j];
        s1 += 0.5 * (binary[i][j] + binary[j][i]) ** 2;
        degrees[i] += binary[i][j];
        degrees[j] += binary[i][j];
      }
    }
    const s2 = degrees.reduce((t, d) => t + d * d, 0);
    const scale = (s0 * sumSquares) / (n - 1);
    expectClose(
      getGearyMoments(n, s0, s1, s2, kurtosis)[1] * scale * scale,
      bwVariance,
      `seed ${seed} V BW Geary`
    );
  }
});

it('the oracle gives I = -1 and BB = 0 on a rook checkerboard', () => {
  const weights = createGridWeights(6, 6);
  const values = Float32Array.from(
    {length: 36},
    (_, index) => (Math.floor(index / 6) + (index % 6)) % 2
  );
  const oracle = computeGlobalSpatialStatisticsOracle({weights, values});
  expectClose(oracle.moran.statistic, -1, 'I');
  expect(oracle.joinCount.blackBlack).toBe(0);
  expect(oracle.joinCount.whiteWhite).toBe(0);
  expect(oracle.joinCount.blackWhite).toBe(oracle.joinCount.joins);
});

it('GPUGlobalSpatialStatistics validates its props and emits deterministic node IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'global-statistics-validation'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `bad-${serial++}`, format, length);
  expect(() => new GPUGlobalSpatialStatistics(createProps(graph))).not.toThrow();
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {values: view('float32', 9)}))
  ).toThrow(/values length/);
  expect(() => new GPUGlobalSpatialStatistics(createProps(graph, {statistics: []}))).toThrow(
    /at least one/
  );
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {statistics: ['bivariateMoran']}))
  ).toThrow(/secondValues/);
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {statistics: ['nope' as 'moran']}))
  ).toThrow(/unknown/);
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {results: view('float32', 8)}))
  ).toThrow(/results/);
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {mask: view('uint32', 3)}))
  ).toThrow(/mask/);
  const values = view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
  expect(
    () =>
      new GPUGlobalSpatialStatistics(
        createProps(graph, {
          weights: {
            offsets: view('uint32', values.length + 1),
            neighbors: view('uint32', 4),
            weights: view('float32', 4)
          },
          values,
          results: values
        })
      )
  ).toThrow(/must not share/);
  const recipe = new GPUGlobalSpatialStatistics({...createProps(graph), id: 'g'});
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('g-include');
  expect(ids.at(-1)).toBe('g-finalize');
  expect(new Set(ids).size).toBe(ids.length);
});
