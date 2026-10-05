// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_GLOBAL_PERMUTATION_RESULT} from '../../../src/map-graphs/permutation-inference';
import {
  createGlobalPermutationHarness,
  type GlobalPermutationReadback
} from './permutation-harness';
import {
  computeGlobalPermutationOracle,
  createKnnWeights,
  createSeededRandom,
  type GlobalPermutationOracle
} from './permutation-oracle';

const R = GPU_GLOBAL_PERMUTATION_RESULT;

function close(actual: number, expected: number, absolute: number, relative: number): boolean {
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected);
}

/**
 * The GPU sums in f32 and the oracle in f64 over the same permutations, so statistics agree to
 * f32 rounding and the exceedance count to within the simulated values that lie that close to the
 * observed one.
 */
function expectMatchesOracle(
  result: GlobalPermutationReadback,
  oracle: GlobalPermutationOracle,
  permutations: number,
  label: string
): void {
  const tolerance = 1e-4 * Math.max(...oracle.simulated.map(Math.abs), Math.abs(oracle.observed));
  expect(
    close(result.results[R.observed], oracle.observed, tolerance, 1e-4),
    `${label} observed`
  ).toBe(true);
  for (let index = 0; index < permutations; index++) {
    if (!close(result.referenceDistribution[index], oracle.simulated[index], tolerance, 1e-4)) {
      throw new Error(
        `${label} permutation ${index + 1}: ${result.referenceDistribution[index]} != ${oracle.simulated[index]}`
      );
    }
  }
  const nearTies = oracle.simulated.filter(
    value => Math.abs(value - oracle.observed) <= 2 * tolerance
  ).length;
  expect(
    Math.abs(result.results[R.exceedances] - oracle.exceedances),
    `${label} exceedances`
  ).toBeLessThanOrEqual(nearTies);
  expect(result.results[R.permutations]).toBe(permutations);
  expect(
    close(
      result.results[R.pseudoPValue],
      (result.results[R.exceedances] + 1) / (permutations + 1),
      1e-7,
      1e-6
    )
  ).toBe(true);
  expect(
    close(result.results[R.simulatedMean], oracle.simulatedMean, tolerance, 1e-4),
    `${label} mean`
  ).toBe(true);
  expect(
    close(
      result.results[R.simulatedStandardDeviation],
      oracle.simulatedStandardDeviation,
      tolerance,
      1e-3
    ),
    `${label} std`
  ).toBe(true);
  expect(close(result.results[R.zSimulated], oracle.zSimulated, 1e-2, 1e-3), `${label} z`).toBe(
    true
  );
  expect(
    result.histogram.reduce((a, b) => a + b, 0),
    `${label} histogram total`
  ).toBe(permutations);
  expect(result.results[R.minimum]).toBe(
    Math.min(...result.referenceDistribution.slice(0, permutations))
  );
  expect(result.results[R.maximum]).toBe(
    Math.max(...result.referenceDistribution.slice(0, permutations))
  );
}

function createScene(count: number, seed: number) {
  const random = createSeededRandom(seed);
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const values = Float32Array.from(
    {length: count},
    (_, index) => 3 + Math.sin(positions[index * 2] / 12) + random() * 0.8
  );
  const secondValues = Float32Array.from(
    {length: count},
    (_, index) => Math.sin(positions[index * 2] / 12) + random() * 0.5
  );
  return {positions, values, secondValues};
}

it('GPUGlobalPermutationTest matches the oracle for Moran, Geary, General G and bivariate Moran', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values, secondValues} = createScene(900, 1);
  const weights = createKnnWeights(positions, 6, true);
  for (const statistic of ['moran', 'geary', 'getisOrdG', 'bivariateMoran'] as const) {
    const parameters = {seed: 9, permutations: 199};
    const harness = createGlobalPermutationHarness(device, {
      scene: {weights, values, secondValues},
      statistic,
      parameters,
      maximumPermutations: 255
    });
    const result = await harness.run();
    const oracle = computeGlobalPermutationOracle({
      weights,
      values,
      secondValues,
      statistic,
      ...parameters
    });
    expectMatchesOracle(result, oracle, 199, statistic);
    // Strong spatial structure: the observed statistic is outside every permutation.
    expect(result.results[R.pseudoPValue], statistic).toBeCloseTo(1 / 200, 6);
    for (let index = 199; index < 255; index++) {
      expect(result.referenceDistribution[index]).toBeNaN();
    }
    harness.destroy();
  }
});

it('GPUGlobalPermutationTest on random values gives a reference distribution centred on -1/(n-1)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createSeededRandom(5);
  const count = 3000;
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const values = Float32Array.from({length: count}, () => random());
  const weights = createKnnWeights(positions, 8, false);
  const mask = Uint32Array.from({length: count}, () => (random() < 0.9 ? 1 : 0));
  values[3] = NaN;
  const parameters = {seed: 123, permutations: 999};
  const harness = createGlobalPermutationHarness(device, {
    scene: {weights, values, mask},
    statistic: 'moran',
    parameters,
    maximumPermutations: 999,
    histogramBins: 20
  });
  const result = await harness.run();
  const oracle = computeGlobalPermutationOracle({
    weights,
    values,
    mask,
    statistic: 'moran',
    ...parameters
  });
  expectMatchesOracle(result, oracle, 999, 'random');
  const included = Array.from(mask).filter(
    (flag, row) => flag && Number.isFinite(values[row])
  ).length;
  expect(Math.abs(result.results[R.simulatedMean] + 1 / (included - 1))).toBeLessThan(
    (4 * result.results[R.simulatedStandardDeviation]) / Math.sqrt(999)
  );
  expect(result.results[R.pseudoPValue]).toBeGreaterThan(0.01);
  expect(harness.buildCount).toBe(1);
  harness.destroy();
});

it('GPUGlobalPermutationTest is bitwise reproducible per seed and follows seed, P and values per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createScene(1500, 2);
  const weights = createKnnWeights(positions, 5, true);
  const harness = createGlobalPermutationHarness(device, {
    scene: {weights, values},
    statistic: 'geary',
    parameters: {seed: 1, permutations: 99},
    maximumPermutations: 499
  });
  const first = await harness.run();
  const second = await harness.run();
  expect(second.resultBits).toEqual(first.resultBits);
  expect(second.referenceDistribution).toEqual(first.referenceDistribution);
  const reseeded = await harness.run({seed: 2, permutations: 99});
  expect(reseeded.referenceDistribution).not.toEqual(first.referenceDistribution);
  expect(reseeded.results[R.observed]).toBe(first.results[R.observed]);
  const shuffled = values.map((_, index) => values[(index * 7919) % values.length]);
  const parameters = {seed: 4, permutations: 499};
  const fresh = await harness.run(parameters, {values: shuffled});
  expectMatchesOracle(
    fresh,
    computeGlobalPermutationOracle({weights, values: shuffled, statistic: 'geary', ...parameters}),
    499,
    'shuffled'
  );
  expect(harness.buildCount).toBe(1);
  harness.destroy();
});
