// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUPermutationParameterValues,
  GPU_GLOBAL_PERMUTATION_RESULT
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  createGlobalPermutationHarness,
  type GlobalPermutationReadback
} from './permutation-harness';
import {
  computeGlobalPermutationOracle,
  createKnnWeights,
  createSeededRandom,
  createWeights,
  type CPUSpatialWeights,
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
  label: string,
  alternative: 'two-sided' | 'directed' | 'greater' | 'lesser' | 'folded' = 'two-sided'
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
      Math.min(
        (alternative === 'two-sided' ? 2 : 1) * (result.results[R.exceedances] + 1),
        permutations + 1
      ) /
        (permutations + 1),
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
    // Two-sided (default): 2 (0 + 1) / (P + 1).
    expect(result.results[R.pseudoPValue], statistic).toBeCloseTo(2 / 200, 6);
    expect(result.results[R.islandCount]).toBe(0);
    expect(result.results[R.analysisCount]).toBe(900);
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

function createLatticeWeights(side: number, rowStandardize: boolean): CPUSpatialWeights {
  const lists: [number, number][][] = [];
  for (let row = 0; row < side; row++) {
    for (let column = 0; column < side; column++) {
      const neighbors: number[] = [];
      if (row > 0) neighbors.push((row - 1) * side + column);
      if (row < side - 1) neighbors.push((row + 1) * side + column);
      if (column > 0) neighbors.push(row * side + column - 1);
      if (column < side - 1) neighbors.push(row * side + column + 1);
      lists.push(
        neighbors.map((neighbor): [number, number] => [
          neighbor,
          rowStandardize ? 1 / neighbors.length : 1
        ])
      );
    }
  }
  return createWeights(lists);
}

it('GPUGlobalPermutationTest handles islands, masks and re-standardization like the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const side = 20;
  const count = side * side;
  const random = createSeededRandom(31);
  const lattice = createLatticeWeights(side, true);
  // Rows 0 and 1 lose every neighbor (islands, and nobody lists them as neighbor either).
  const lists = Array.from({length: count}, (_, row): [number, number][] => {
    if (row === 0 || row === 1) {
      return [];
    }
    return Array.from(
      lattice.neighbors.subarray(lattice.offsets[row], lattice.offsets[row + 1]),
      (neighbor, slot): [number, number] => [neighbor, lattice.weights[lattice.offsets[row] + slot]]
    ).filter(([neighbor]) => neighbor > 1);
  });
  const weights = createWeights(lists);
  const values = Float32Array.from({length: count}, () => random() * 3);
  const mask = Uint32Array.from({length: count}, () => (random() < 0.9 ? 1 : 0));
  mask[0] = 1;
  mask[1] = 1;
  const parameters = {seed: 17, permutations: 199};
  const analysisCounts: Record<string, number> = {};
  for (const islands of ['exclude', 'keep'] as const) {
    const harness = createGlobalPermutationHarness(device, {
      scene: {weights, values, mask, rowStandardized: true},
      statistic: 'moran',
      parameters,
      islands,
      maximumPermutations: 255
    });
    const result = await harness.run();
    harness.destroy();
    const oracle = computeGlobalPermutationOracle({
      weights,
      values,
      mask,
      statistic: 'moran',
      islands,
      rowStandardized: true,
      ...parameters
    });
    expectMatchesOracle(result, oracle, 199, `islands ${islands}`);
    expect(result.results[R.islandCount]).toBe(oracle.islandCount);
    expect(result.results[R.islandCount]).toBeGreaterThanOrEqual(2);
    expect(result.results[R.analysisCount]).toBe(oracle.analysisCount);
    analysisCounts[islands] = result.results[R.analysisCount];
  }
  // Kept islands stay in n; excluded ones do not.
  expect(analysisCounts['keep']).toBeGreaterThan(analysisCounts['exclude']);
});

it('GPUGlobalPermutationTest reports p = 1 when every permutation ties the observed statistic', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const side = 12;
  const weights = createLatticeWeights(side, false);
  const values = new Float32Array(side * side).fill(4);
  for (const alternative of ['directed', 'two-sided'] as const) {
    const harness = createGlobalPermutationHarness(device, {
      scene: {weights, values},
      statistic: 'getisOrdG',
      parameters: {seed: 2, permutations: 99},
      alternative,
      maximumPermutations: 99
    });
    const result = await harness.run();
    harness.destroy();
    // Every relabeling of constant data is the identity: g = l = P, so p = 1 (not 1 / (P + 1)).
    expect(result.results[R.pseudoPValue], alternative).toBe(1);
  }
});

it('GPUGlobalPermutationTest never clamps an invalid permutation count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values} = createScene(300, 3);
  const weights = createKnnWeights(positions, 5, true);
  const harness = createGlobalPermutationHarness(device, {
    scene: {weights, values},
    statistic: 'moran',
    parameters: {seed: 1, permutations: 49},
    maximumPermutations: 49
  });
  const valid = await harness.run();
  expect(valid.results[R.permutations]).toBe(49);
  for (const requested of [0, 50, 5000]) {
    const words = getGPUPermutationParameterValues({seed: 1, permutations: 49});
    words[2] = requested;
    const result = await harness.run(words);
    expect(result.results[R.permutations], `P=${requested}`).toBeNaN();
    expect(result.results[R.pseudoPValue], `P=${requested}`).toBeNaN();
    expect(result.results[R.exceedances]).toBeNaN();
    expect(result.referenceDistribution.every(Number.isNaN)).toBe(true);
    // The observed statistic is still reported.
    expect(result.results[R.observed]).toBeCloseTo(valid.results[R.observed], 6);
  }
  harness.destroy();
});
