// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUPermutationParameterValues,
  GPU_LOCAL_PERMUTATION_NOT_TESTED
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import type {
  GPUInferenceAlternative,
  GPUMultipleTestingMethod
} from '../../../src/gpu-spatial-analysis/contracts';
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
  options: {alternative?: GPUInferenceAlternative; multipleTesting?: GPUMultipleTestingMethod} = {}
): void {
  const multiplier = (options.alternative ?? 'two-sided') === 'two-sided' ? 2 : 1;
  const rows = oracle.exceedances.length;
  let mismatched = 0;
  for (let row = 0; row < rows; row++) {
    const expected = oracle.exceedances[row];
    const actual = result.exceedances[row];
    if (expected === GPU_LOCAL_PERMUTATION_NOT_TESTED) {
      expect(actual, `${label} row ${row} not tested`).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
      expect(result.pseudoPValues[row]).toBeNaN();
      expect(result.adjustedPValues[row]).toBeNaN();
      if (Number.isNaN(oracle.observed[row])) {
        expect(result.observed[row]).toBeNaN();
      } else {
        // A kept island: the formula value, never tested.
        expect(Math.abs(result.observed[row] - oracle.observed[row])).toBeLessThanOrEqual(
          2e-4 + 2e-4 * Math.abs(oracle.observed[row])
        );
      }
      expect(result.significant[row]).toBe(0);
      continue;
    }
    if (actual !== expected) {
      mismatched++;
      expect(Math.abs(actual - expected), `${label} row ${row}`).toBeLessThanOrEqual(1);
      continue;
    }
    expect(
      Math.abs(result.pseudoPValues[row] - oracle.pseudoPValues[row]),
      `${label} p ${row}`
    ).toBeLessThan(1e-6);
    expect(
      Math.abs(
        result.pseudoPValues[row] -
          Math.min(multiplier * (actual + 1), permutations + 1) / (permutations + 1)
      )
    ).toBeLessThan(1e-6);
    const observed = oracle.observed[row];
    expect(
      Math.abs(result.observed[row] - observed),
      `${label} observed ${row}`
    ).toBeLessThanOrEqual(2e-4 + 2e-4 * Math.abs(observed));
  }
  expect(mismatched, `${label} mismatched rows`).toBeLessThanOrEqual(Math.max(1, rows * 0.005));
  if (mismatched === 0) {
    if ((options.multipleTesting ?? 'none') !== 'none') {
      for (let row = 0; row < rows; row++) {
        if (!Number.isNaN(oracle.adjustedPValues[row])) {
          expect(
            Math.abs(result.adjustedPValues[row] - oracle.adjustedPValues[row]),
            `${label} adjusted ${row}`
          ).toBeLessThan(1e-5);
        }
      }
    }
    expect(result.significant, `${label} significant`).toEqual(oracle.significant);
  }
}

it('GPULocalPermutationTest matches the oracle for local Moran, Geary, G and G* on kNN weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createClusteredScene(700, 1);
  for (const [statistic, weights, rowStandardized] of [
    ['localMoran', createKnnWeights(positions, 6, true), true],
    ['localGeary', createKnnWeights(positions, 6, true), true],
    ['localG', createKnnWeights(positions, 5, false), false],
    ['localGStar', createKnnWeights(positions, 8, false), false]
  ] as const) {
    const parameters = {seed: 42, permutations: 199, significanceLevel: 0.05};
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values, rowStandardized},
      statistic,
      parameters
    });
    const result = await harness.run();
    const oracle = computeLocalPermutationOracle({
      weights,
      values,
      statistic,
      maximumNeighbors: 32,
      rowStandardized,
      ...parameters
    });
    expectMatchesOracle(result, oracle, 199, statistic);
    expect(result.islandCount).toBe(0);
    // The clustered field makes many rows significant.
    expect(result.significant.filter(Boolean).length, statistic).toBeGreaterThan(50);
    harness.destroy();
  }
});

function createIslandScene() {
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
  return {weights, values, mask};
}

it('GPULocalPermutationTest handles masks, NaN values, islands, overflow and every multiple-testing method', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {weights, values, mask} = createIslandScene();
  const methods: GPUMultipleTestingMethod[] = [
    'none',
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ];
  for (const multipleTesting of methods) {
    const parameters = {seed: 7, permutations: 99, significanceLevel: 0.1};
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values, mask, rowStandardized: true},
      statistic: 'localMoran',
      parameters,
      maximumNeighbors: 8,
      multipleTesting
    });
    const result = await harness.run();
    const oracle = computeLocalPermutationOracle({
      weights,
      values,
      mask,
      statistic: 'localMoran',
      maximumNeighbors: 8,
      rowStandardized: true,
      multipleTesting,
      ...parameters
    });
    expect(result.overflow).toBe(1);
    expect(result.exceedances[0]).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
    expect(result.exceedances[1]).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
    // Row 1 is an island under the default 'exclude': NaN, not in n.
    expect(result.islandCount).toBe(oracle.islandCount);
    expect(result.islandCount).toBeGreaterThanOrEqual(1);
    expect(result.observed[1]).toBeNaN();
    expectMatchesOracle(result, oracle, 99, `multiple testing ${multipleTesting}`, {
      multipleTesting
    });
    if (multipleTesting !== 'none') {
      const unadjusted = result.pseudoPValues.filter(p => p <= 0.1).length;
      expect(result.significant.filter(Boolean).length).toBeLessThanOrEqual(unadjusted);
    }
    harness.destroy();
  }
});

it("GPULocalPermutationTest islands 'keep' retains islands in n but never tests them", async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {weights, values, mask} = createIslandScene();
  const parameters = {seed: 7, permutations: 99, significanceLevel: 0.1};
  const run = async (islands: 'exclude' | 'keep') => {
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values, mask, rowStandardized: true},
      statistic: 'localMoran',
      parameters,
      maximumNeighbors: 8,
      islands
    });
    const result = await harness.run();
    harness.destroy();
    return result;
  };
  const kept = await run('keep');
  const excluded = await run('exclude');
  const oracle = computeLocalPermutationOracle({
    weights,
    values,
    mask,
    statistic: 'localMoran',
    maximumNeighbors: 8,
    rowStandardized: true,
    islands: 'keep',
    ...parameters
  });
  expectMatchesOracle(kept, oracle, 99, 'keep');
  // The kept island has local I = 0 (zero lag) and is not tested.
  expect(kept.exceedances[1]).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
  expect(kept.observed[1]).toBe(0);
  expect(kept.pseudoPValues[1]).toBeNaN();
  expect(kept.islandCount).toBe(excluded.islandCount);
  // Dropping the island changes n and the mean, hence other rows' statistics.
  expect(kept.observed.some((value, row) => value !== excluded.observed[row] && row > 1)).toBe(
    true
  );
});

it('GPULocalPermutationTest does not let ties or zero-weight slots make rows look significant', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const count = 300;
  const random = createSeededRandom(21);
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  // Zero-inflated binary data: most neighborhoods are all zeros, so observed == every simulated term.
  const values = Float32Array.from({length: count}, () => (random() < 0.04 ? 1 : 0));
  const knn = createKnnWeights(positions, 5, false);
  // Row 5 lists four neighbors, two with zero weight (not neighbors); row 6 lists only zero weights.
  const lists = Array.from({length: count}, (_, row): [number, number][] => {
    const entries = Array.from(
      knn.neighbors.subarray(knn.offsets[row], knn.offsets[row + 1]),
      (neighbor): [number, number] => [neighbor, 1]
    );
    if (row === 5) {
      return entries
        .slice(0, 4)
        .map(([neighbor], slot): [number, number] => [neighbor, slot < 2 ? 0 : 1]);
    }
    if (row === 6) {
      return entries.slice(0, 3).map(([neighbor]): [number, number] => [neighbor, 0]);
    }
    return entries;
  });
  const weights = createWeights(lists);
  const permutations = 99;
  const parameters = {seed: 5, permutations, significanceLevel: 0.05};
  for (const alternative of ['directed', 'two-sided'] as const) {
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values},
      statistic: 'localG',
      parameters,
      alternative,
      maximumPermutations: 255
    });
    const result = await harness.run();
    harness.destroy();
    const oracle = computeLocalPermutationOracle({
      weights,
      values,
      statistic: 'localG',
      maximumNeighbors: 32,
      alternative,
      ...parameters
    });
    expectMatchesOracle(result, oracle, permutations, alternative, {alternative});
    // Row 6 has no positive-weight neighbor: an island, never tested (before: p = 1/(R+1)).
    expect(result.exceedances[6], alternative).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
    expect(result.exceedances[5], alternative).not.toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
    // Rows whose observed lag is 0 tie most simulations: never the smallest attainable p.
    const tiedRows = result.observed.filter(
      (value, row) => value === 0 && result.exceedances[row] !== GPU_LOCAL_PERMUTATION_NOT_TESTED
    ).length;
    expect(tiedRows, alternative).toBeGreaterThan(100);
    for (let row = 0; row < count; row++) {
      if (result.observed[row] === 0 && result.pseudoPValues[row] === result.pseudoPValues[row]) {
        expect(result.pseudoPValues[row], `${alternative} row ${row}`).toBeGreaterThan(0.2);
      }
    }
  }
});

it('GPULocalPermutationTest re-standardizes row-standardized weights over the neighbors left after masking', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createClusteredScene(400, 6);
  const weights = createKnnWeights(positions, 6, true);
  const random = createSeededRandom(8);
  const mask = Uint32Array.from({length: 400}, () => (random() < 0.8 ? 1 : 0));
  const parameters = {seed: 3, permutations: 99, significanceLevel: 0.05};
  const run = async (rowStandardized: boolean) => {
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values, mask, rowStandardized},
      statistic: 'localMoran',
      parameters
    });
    const result = await harness.run();
    harness.destroy();
    return result;
  };
  const standardized = await run(true);
  const asGiven = await run(false);
  const oracle = computeLocalPermutationOracle({
    weights,
    values,
    mask,
    statistic: 'localMoran',
    maximumNeighbors: 32,
    rowStandardized: true,
    ...parameters
  });
  expectMatchesOracle(standardized, oracle, 99, 'standardized');
  // With a masked neighbor the row's weights summed to less than 1 as given; re-standardizing
  // rescales the lag, so the observed local I differs for exactly those rows.
  let rescaled = 0;
  for (let row = 0; row < 400; row++) {
    const lost = Array.from(
      weights.neighbors.subarray(weights.offsets[row], weights.offsets[row + 1])
    ).filter(neighbor => mask[neighbor] === 0).length;
    if (mask[row] && lost > 0 && lost < 6 && values[row] === values[row]) {
      rescaled++;
      expect(standardized.observed[row], `row ${row}`).not.toBe(asGiven.observed[row]);
    }
  }
  expect(rescaled).toBeGreaterThan(20);
});

it('GPULocalPermutationTest never clamps an invalid permutation count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createClusteredScene(300, 9);
  const weights = createKnnWeights(positions, 6, true);
  const harness = createLocalPermutationHarness(device, {
    scene: {weights, values},
    statistic: 'localMoran',
    parameters: {seed: 1, permutations: 99},
    maximumPermutations: 99
  });
  const valid = await harness.run();
  expect(valid.overflow).toBe(0);
  expect(valid.exceedances.some(count => count !== GPU_LOCAL_PERMUTATION_NOT_TESTED)).toBe(true);
  for (const requested of [0, 100, 1000]) {
    // Raw words: [seedLow, seedHigh, permutations, float32 bits of the level].
    const words = getGPUPermutationParameterValues({seed: 1, permutations: 99});
    words[2] = requested;
    const result = await harness.run(words);
    expect(result.overflow, `P=${requested}`).toBe(1);
    expect(result.exceedances.every(count => count === GPU_LOCAL_PERMUTATION_NOT_TESTED)).toBe(
      true
    );
    expect(result.pseudoPValues.every(Number.isNaN), `P=${requested}`).toBe(true);
    expect(result.significant.every(flag => flag === 0)).toBe(true);
  }
  harness.destroy();
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
