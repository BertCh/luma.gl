// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUPermutationMetadata,
  GPU_GLOBAL_PERMUTATION_RESULT,
  GPU_LOCAL_PERMUTATION_NOT_TESTED
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {createGlobalPermutationHarness, createLocalPermutationHarness} from './permutation-harness';
import {
  computeGlobalPermutationOracle,
  computeLocalPermutationOracle,
  createKnnWeights,
  createSeededRandom,
  getOraclePValue,
  getOracleExceedance,
  type OracleAlternative
} from './permutation-oracle';

const R = GPU_GLOBAL_PERMUTATION_RESULT;
const ALTERNATIVES: OracleAlternative[] = ['directed', 'two-sided', 'greater', 'lesser'];

it('publishes canonical reproducibility metadata', () => {
  expect(
    getGPUPermutationMetadata(
      {seed: 9, permutations: 199},
      {alternative: 'two-sided', multipleTesting: 'benjamini-hochberg'}
    )
  ).toEqual({
    seed: 9,
    permutationCount: 199,
    alternative: 'two-sided',
    multipleTesting: 'benjamini-hochberg',
    includeObserved: true
  });
});

function createScene(count: number, seed: number, sign: number) {
  const random = createSeededRandom(seed);
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const values = Float32Array.from(
    {length: count},
    (_, index) => 3 + sign * (Math.sin(positions[index * 2] / 12) + random() * 0.8)
  );
  return {positions, values};
}

it('GPULocalPermutationTest alternatives match the oracle and relate as esda defines', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createScene(500, 3, 1);
  const weights = createKnnWeights(positions, 6, true);
  const permutations = 199;
  const parameters = {seed: 11, permutations, significanceLevel: 0.05};
  const byAlternative = new Map<string, Awaited<ReturnType<typeof runLocal>>>();
  async function runLocal(alternative: OracleAlternative) {
    const harness = createLocalPermutationHarness(device!, {
      scene: {weights, values},
      statistic: 'localMoran',
      parameters,
      alternative,
      maximumPermutations: 255
    });
    const result = await harness.run();
    harness.destroy();
    return result;
  }
  for (const alternative of ALTERNATIVES) {
    const result = await runLocal(alternative);
    const oracle = computeLocalPermutationOracle({
      weights,
      values,
      statistic: 'localMoran',
      maximumNeighbors: 32,
      alternative,
      ...parameters
    });
    byAlternative.set(alternative, result);
    let mismatched = 0;
    for (let row = 0; row < values.length; row++) {
      if (oracle.exceedances[row] === GPU_LOCAL_PERMUTATION_NOT_TESTED) {
        continue;
      }
      if (result.exceedances[row] !== oracle.exceedances[row]) {
        mismatched++;
        expect(Math.abs(result.exceedances[row] - oracle.exceedances[row])).toBeLessThanOrEqual(1);
      } else {
        expect(
          Math.abs(result.pseudoPValues[row] - oracle.pseudoPValues[row]),
          `${alternative} p ${row}`
        ).toBeLessThan(1e-6);
      }
    }
    expect(mismatched, alternative).toBeLessThanOrEqual(3);
    expect(result.significant.filter(Boolean).length, alternative).toBeGreaterThan(0);
  }
  // Derived relations: two-sided = min(2 min(g, l), ...) vs. directed and greater/lesser.
  const directed = byAlternative.get('directed')!;
  const twoSided = byAlternative.get('two-sided')!;
  const greater = byAlternative.get('greater')!;
  const lesser = byAlternative.get('lesser')!;
  let strict = 0;
  for (let row = 0; row < values.length; row++) {
    if (directed.exceedances[row] === GPU_LOCAL_PERMUTATION_NOT_TESTED) {
      continue;
    }
    // 'greater' and 'lesser' tails together cover the permutations (ties count in both).
    expect(greater.exceedances[row] + lesser.exceedances[row]).toBeGreaterThanOrEqual(permutations);
    expect(twoSided.pseudoPValues[row]).toBeLessThanOrEqual(1);
    expect(twoSided.pseudoPValues[row]).toBeGreaterThanOrEqual(directed.pseudoPValues[row] - 1e-6);
    expect(twoSided.exceedances[row]).toBe(
      Math.min(greater.exceedances[row], lesser.exceedances[row])
    );
    expect(
      Math.abs(
        getOraclePValue(twoSided.exceedances[row], 2, permutations) - twoSided.pseudoPValues[row]
      )
    ).toBeLessThan(1e-6);
    if (twoSided.pseudoPValues[row] > directed.pseudoPValues[row] + 1e-6) {
      strict++;
    }
  }
  expect(strict).toBeGreaterThan(100);
});

it('GPUGlobalPermutationTest alternatives follow esda tail semantics', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const permutations = 199;
  for (const sign of [1, -1]) {
    // Positive autocorrelation in the smooth field; the greater tail is rejected, not the lesser.
    const {positions, values} = createScene(900, 1, sign);
    const weights = createKnnWeights(positions, 6, true);
    const parameters = {seed: 9, permutations};
    const p: Record<string, number> = {};
    for (const alternative of ALTERNATIVES) {
      const harness = createGlobalPermutationHarness(device, {
        scene: {weights, values},
        statistic: 'moran',
        parameters,
        alternative,
        maximumPermutations: 255
      });
      const result = await harness.run();
      const oracle = computeGlobalPermutationOracle({
        weights,
        values,
        statistic: 'moran',
        alternative,
        ...parameters
      });
      expect(
        Math.abs(result.results[R.exceedances] - oracle.exceedances),
        alternative
      ).toBeLessThanOrEqual(1);
      expect(result.results[R.pseudoPValue]).toBeCloseTo(
        getOraclePValue(
          result.results[R.exceedances],
          alternative === 'two-sided' ? 2 : 1,
          permutations
        ),
        6
      );
      p[alternative] = result.results[R.pseudoPValue];
      harness.destroy();
    }
    expect(p['greater'], 'greater').toBeCloseTo(1 / 200, 6);
    expect(p['lesser'], 'lesser').toBeGreaterThan(0.9);
    expect(p['directed'], 'directed').toBeCloseTo(1 / 200, 6);
    expect(p['two-sided'], 'two-sided').toBeCloseTo(2 / 200, 6);
  }
});

it('folded tail counts distance from the simulated mean (esda calculate_significance)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const permutations = 199;
  const {positions, values} = createScene(500, 3, 1);
  const weights = createKnnWeights(positions, 6, true);
  const parameters = {seed: 11, permutations, significanceLevel: 0.05};

  // Global: folded exceedance equals the oracle (+-1 for f32 boundary ties) and p = (f + 1) / (R + 1).
  const globalHarness = createGlobalPermutationHarness(device, {
    scene: {weights, values},
    statistic: 'moran',
    parameters: {seed: 9, permutations},
    alternative: 'folded',
    maximumPermutations: 255
  });
  const globalResult = await globalHarness.run();
  globalHarness.destroy();
  const globalOracle = computeGlobalPermutationOracle({
    weights,
    values,
    statistic: 'moran',
    alternative: 'folded',
    seed: 9,
    permutations
  });
  expect(globalResult.results[R.exceedances]).toBeLessThanOrEqual(permutations);
  expect(
    Math.abs(globalResult.results[R.exceedances] - globalOracle.exceedances)
  ).toBeLessThanOrEqual(1);
  expect(globalResult.results[R.pseudoPValue]).toBeCloseTo(
    getOraclePValue(globalResult.results[R.exceedances], 1, permutations),
    6
  );
  // Strongly autocorrelated field: observed is far from the simulated mean, so only the +1 remains.
  expect(globalResult.results[R.pseudoPValue]).toBeCloseTo(1 / 200, 6);

  // Local: per-row folded counts match the oracle and are not all trivial.
  const harness = createLocalPermutationHarness(device, {
    scene: {weights, values},
    statistic: 'localMoran',
    parameters,
    alternative: 'folded',
    maximumPermutations: 255
  });
  const result = await harness.run();
  harness.destroy();
  const oracle = computeLocalPermutationOracle({
    weights,
    values,
    statistic: 'localMoran',
    maximumNeighbors: 32,
    alternative: 'folded',
    ...parameters
  });
  let tested = 0;
  let mismatched = 0;
  let small = 0;
  for (let row = 0; row < values.length; row++) {
    if (oracle.exceedances[row] === GPU_LOCAL_PERMUTATION_NOT_TESTED) {
      continue;
    }
    tested++;
    const difference = Math.abs(result.exceedances[row] - oracle.exceedances[row]);
    expect(difference, `folded ${row}`).toBeLessThanOrEqual(1);
    mismatched += difference > 0 ? 1 : 0;
    small += oracle.exceedances[row] < 10 ? 1 : 0;
  }
  expect(tested).toBeGreaterThan(400);
  expect(mismatched).toBeLessThanOrEqual(3);
  expect(small).toBeGreaterThan(20);
  expect(small).toBeLessThan(tested);
});

// Pinned from esda 2.10.0: calculate_significance(observed, reference, alternative='folded') on this
// fixed simulated array (reference excludes the observed value; mean is the reference mean).
it('folded p-value formula matches esda 2.10.0 pinned values', () => {
  const reference = [
    0.5, -0.2, 0.1, 0.9, -0.4, 0.3, 0.0, 0.7, -0.6, 0.2, 0.4, -0.1, 0.15, 0.35, -0.3, 0.8, -0.05,
    0.25, 0.6, -0.7
  ];
  const mean = reference.reduce((a, b) => a + b, 0) / reference.length;
  const pinned: [number, number][] = [
    [0.9, 3 / 21],
    [-0.4, 7 / 21],
    [0.05, 18 / 21],
    [1.5, 1 / 21]
  ];
  for (const [observed, expected] of pinned) {
    const folded = reference.filter(
      value => Math.abs(value - mean) >= Math.abs(observed - mean)
    ).length;
    const {count, multiplier} = getOracleExceedance('folded', 0, 0, reference.length, folded);
    expect(getOraclePValue(count, multiplier, reference.length)).toBeCloseTo(expected, 12);
  }
});
