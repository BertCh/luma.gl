// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_LOCAL_PERMUTATION_NOT_TESTED} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {createLocalPermutationHarness, type LocalPermutationReadback} from './permutation-harness';
import {
  computeLocalPermutationOracle,
  createKnnWeights,
  createSeededRandom,
  createWeights,
  type CPUSpatialWeights,
  type LocalPermutationOracle
} from './permutation-oracle';

function createClusteredScene(count: number, seed: number) {
  const random = createSeededRandom(seed);
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const values = Float32Array.from(
    {length: count},
    (_, index) =>
      2 + Math.sin(positions[index * 2] / 15) + Math.cos(positions[index * 2 + 1] / 20) + random()
  );
  return {positions, values};
}

/**
 * Exceedance counts match the f32-emulating oracle exactly except where a simulated term lies
 * within f32 rounding of the observed one (the GPU mean and fused multiply-adds may differ by one
 * ulp); such rows differ by at most one and are rare.
 */
function expectMatchesOracle(
  result: LocalPermutationReadback,
  oracle: LocalPermutationOracle,
  permutations: number,
  label: string,
  falseDiscoveryRate = false
): void {
  const rows = oracle.exceedances.length;
  let mismatched = 0;
  for (let row = 0; row < rows; row++) {
    const expected = oracle.exceedances[row];
    const actual = result.exceedances[row];
    if (expected === GPU_LOCAL_PERMUTATION_NOT_TESTED) {
      expect(actual, `${label} row ${row} not tested`).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
      expect(result.pseudoPValues[row]).toBeNaN();
      expect(result.observed[row]).toBeNaN();
      expect(result.significant[row]).toBe(0);
      continue;
    }
    if (actual !== expected) {
      mismatched++;
      expect(Math.abs(actual - expected), `${label} row ${row}`).toBeLessThanOrEqual(1);
      continue;
    }
    expect(Math.abs(result.pseudoPValues[row] - (actual + 1) / (permutations + 1))).toBeLessThan(
      1e-6
    );
    const observed = oracle.observed[row];
    expect(
      Math.abs(result.observed[row] - observed),
      `${label} observed ${row}`
    ).toBeLessThanOrEqual(2e-4 + 2e-4 * Math.abs(observed));
  }
  expect(mismatched, `${label} mismatched rows`).toBeLessThanOrEqual(Math.max(1, rows * 0.005));
  if (mismatched === 0) {
    expect(result.significant, `${label} significant`).toEqual(
      falseDiscoveryRate ? oracle.significantFalseDiscoveryRate : oracle.significant
    );
  }
}

it('GPULocalPermutationTest matches the oracle for local Moran, Geary, G and G* on kNN weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createClusteredScene(700, 1);
  for (const [statistic, weights] of [
    ['localMoran', createKnnWeights(positions, 6, true)],
    ['localGeary', createKnnWeights(positions, 6, true)],
    ['localG', createKnnWeights(positions, 5, false)],
    ['localGStar', createKnnWeights(positions, 8, false)]
  ] as const) {
    const parameters = {seed: 42, permutations: 199, significanceLevel: 0.05};
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values},
      statistic,
      parameters
    });
    const result = await harness.run();
    const oracle = computeLocalPermutationOracle({
      weights,
      values,
      statistic,
      maximumNeighbors: 32,
      ...parameters
    });
    expectMatchesOracle(result, oracle, 199, statistic);
    // The clustered field makes many rows significant.
    expect(result.significant.filter(Boolean).length, statistic).toBeGreaterThan(50);
    harness.destroy();
  }
});

it('GPULocalPermutationTest handles masks, NaN values, islands, overflow and false discovery rate', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createClusteredScene(500, 2);
  const knn = createKnnWeights(positions, 5, true);
  // Row 0 gets 12 neighbors (overflow with maximumNeighbors 8); row 1 becomes an island.
  const lists = Array.from({length: 500}, (_, row) => {
    if (row === 0) {
      return Array.from({length: 12}, (_, index): [number, number] => [index + 10, 1 / 12]);
    }
    if (row === 1) {
      return [];
    }
    return Array.from(
      knn.neighbors.subarray(knn.offsets[row], knn.offsets[row + 1]),
      (neighbor, slot): [number, number] => [neighbor, knn.weights[knn.offsets[row] + slot]]
    ).filter(([neighbor]) => neighbor !== 1);
  });
  const weights: CPUSpatialWeights = createWeights(lists);
  const random = createSeededRandom(3);
  const mask = Uint32Array.from({length: 500}, () => (random() < 0.9 ? 1 : 0));
  mask[0] = 1;
  mask[1] = 1;
  values[7] = NaN;
  for (const falseDiscoveryRate of [false, true]) {
    const parameters = {seed: 7, permutations: 99, significanceLevel: 0.1};
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values, mask},
      statistic: 'localMoran',
      parameters,
      maximumNeighbors: 8,
      falseDiscoveryRate
    });
    const result = await harness.run();
    const oracle = computeLocalPermutationOracle({
      weights,
      values,
      mask,
      statistic: 'localMoran',
      maximumNeighbors: 8,
      ...parameters
    });
    expect(result.overflow).toBe(1);
    expect(result.exceedances[0]).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
    expect(result.exceedances[1]).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
    expectMatchesOracle(result, oracle, 99, `fdr ${falseDiscoveryRate}`, falseDiscoveryRate);
    if (falseDiscoveryRate) {
      const plain = oracle.significant.filter(Boolean).length;
      expect(result.significant.filter(Boolean).length).toBeLessThanOrEqual(plain);
    }
    harness.destroy();
  }
});

it('GPULocalPermutationTest is reproducible per seed and follows per-frame parameters without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createClusteredScene(2000, 4);
  const weights = createKnnWeights(positions, 8, true);
  const harness = createLocalPermutationHarness(device, {
    scene: {weights, values},
    statistic: 'localMoran',
    parameters: {seed: 1, permutations: 99}
  });
  const first = await harness.run();
  const second = await harness.run();
  expect(second.exceedances).toEqual(first.exceedances);
  expect(second.pseudoPValues).toEqual(first.pseudoPValues);
  const reseeded = await harness.run({seed: 2, permutations: 99});
  expect(reseeded.exceedances).not.toEqual(first.exceedances);
  const parameters = {seed: 3, permutations: 499, significanceLevel: 0.01};
  const more = await harness.run(parameters);
  const oracle = computeLocalPermutationOracle({
    weights,
    values,
    statistic: 'localMoran',
    maximumNeighbors: 32,
    ...parameters
  });
  expectMatchesOracle(more, oracle, 499, 'P 499');
  // New values between submissions.
  const shuffled = values.map((_, index) => values[(index * 7919) % values.length]);
  const fresh = await harness.run(parameters, {values: shuffled});
  expectMatchesOracle(
    fresh,
    computeLocalPermutationOracle({
      weights,
      values: shuffled,
      statistic: 'localMoran',
      maximumNeighbors: 32,
      ...parameters
    }),
    499,
    'shuffled'
  );
  expect(harness.buildCount).toBe(1);
  harness.destroy();
});
