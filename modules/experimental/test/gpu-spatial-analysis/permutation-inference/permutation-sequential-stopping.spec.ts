// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_LOCAL_PERMUTATION_NOT_TESTED
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  createGlobalPermutationHarness,
  createLocalPermutationHarness,
  type GlobalPermutationReadback
} from './permutation-harness';
import {
  computeLocalPermutationOracle,
  createKnnWeights,
  createSeededRandom
} from './permutation-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const R = GPU_GLOBAL_PERMUTATION_SUMMARY;
const f32 = Math.fround;

/** WGSL division permits 2.5 ULP error; the small integer operands here are exact f32 values.
 * Compare to the exact rational, using ULP = 2^(floor(log2(p)) - 23) for positive normal p.
 * https://www.w3.org/TR/WGSL/#floating-point-accuracy
 */
function expectProbability(actual: number, numerator: number, denominator: number): void {
  const expected = numerator / denominator;
  const tolerance = 2.5 * 2 ** (Math.floor(Math.log2(expected)) - 23);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

function createScene(count: number, seed: number, clusters = false) {
  const random = createSeededRandom(seed);
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const values = Float32Array.from({length: count}, (_, index) =>
    clusters
      ? // Two far-apart spatial clusters: every interior row has an extreme local statistic.
        (positions[index * 2] < 50 ? 0 : 100) + random()
      : 2 + Math.sin(positions[index * 2] / 15) + Math.cos(positions[index * 2 + 1] / 20) + random()
  );
  return {positions, values};
}

/**
 * Local test, stopping on. The GPU equals the CPU oracle of the sequential rule exactly except for
 * rows where a simulated term lies within f32 rounding of the observed one (at most 0.5% of rows,
 * as in `gpu-local-permutation-test.spec.ts`), and every stopped row obeys the rule from its own
 * outputs: `exceedances == h`, `p == min(m h, L) / L` in f32, `h <= L <= P`.
 */
it('GPULocalPermutationTest with stopAfterExceedances equals the sequential oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values} = createScene(700, 3);
  const weights = createKnnWeights(positions, 6, true);
  const permutations = 499;
  const target = 8;
  for (const alternative of ['two-sided', 'greater', 'lesser', 'directed'] as const) {
    const parameters = {seed: 17, permutationCount: permutations, significanceLevel: 0.05};
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values, rowStandardized: true},
      statistic: 'localMoran',
      parameters,
      maximumPermutations: 999,
      alternative,
      stopAfterExceedances: target
    });
    const result = await harness.run();
    const oracle = computeLocalPermutationOracle({
      weights,
      values,
      statistic: 'localMoran',
      ...parameters,
      maximumNeighbors: 32,
      alternative,
      rowStandardized: true,
      stopAfterExceedances: target
    });
    const multiplier = alternative === 'two-sided' ? 2 : 1;
    let mismatched = 0;
    let stopped = 0;
    for (let row = 0; row < values.length; row++) {
      const used = result.permutationCounts[row];
      if (oracle.exceedances[row] === GPU_LOCAL_PERMUTATION_NOT_TESTED) {
        expect(used).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
        expect(result.exceedances[row]).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
        continue;
      }
      expect(used).toBeGreaterThanOrEqual(1);
      expect(used).toBeLessThanOrEqual(permutations);
      if (result.exceedances[row] === target) {
        stopped++;
        expect(used).toBeGreaterThanOrEqual(target);
        expectProbability(result.pseudoPValues[row], Math.min(multiplier * target, used), used);
      } else {
        expect(used).toBe(permutations);
        expectProbability(
          result.pseudoPValues[row],
          Math.min(multiplier * (result.exceedances[row] + 1), permutations + 1),
          permutations + 1
        );
      }
      if (
        used !== oracle.permutationCounts[row] ||
        result.exceedances[row] !== oracle.exceedances[row]
      ) {
        mismatched++;
      }
    }
    expect(stopped, `${alternative} stopped rows`).toBeGreaterThan(0);
    expect(mismatched, `${alternative} rows that differ from the oracle`).toBeLessThanOrEqual(
      Math.max(1, values.length * 0.005)
    );
    harness.destroy();
  }
});

it('GPULocalPermutationTest with h > P equals the unstopped run bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values} = createScene(500, 4);
  const weights = createKnnWeights(positions, 6, true);
  const parameters = {seed: 5, permutationCount: 199, significanceLevel: 0.05};
  const scene = {weights, values, rowStandardized: true};
  const plain = createLocalPermutationHarness(device, {
    scene,
    statistic: 'localGeary',
    parameters,
    maximumPermutations: 255
  });
  // A target above the permutation count never stops (the maximum is 255, the count 199).
  const sequential = createLocalPermutationHarness(device, {
    scene,
    statistic: 'localGeary',
    parameters,
    maximumPermutations: 255,
    stopAfterExceedances: 255
  });
  const expected = await plain.run();
  const actual = await sequential.run();
  expect(actual.exceedances).toEqual(expected.exceedances);
  expect(actual.pseudoPValues.map(f32)).toEqual(expected.pseudoPValues.map(f32));
  expect(actual.significant).toEqual(expected.significant);
  expect(actual.classes).toEqual(expected.classes);
  expect(
    actual.permutationCounts.every(
      (used, row) =>
        used === (expected.exceedances[row] === GPU_LOCAL_PERMUTATION_NOT_TESTED ? used : 199)
    )
  ).toBe(true);
  plain.destroy();
  sequential.destroy();
});

it('GPULocalPermutationTest: ties stop every row at L = h, extreme rows never stop', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values: clustered} = createScene(600, 8, true);
  const weights = createKnnWeights(positions, 4, true);
  const permutations = 299;
  const parameters = {seed: 2, permutationCount: permutations, significanceLevel: 0.05};
  // Constant values: every simulated term ties the observed one, so g = l = L after L permutations
  // and every tested row stops at exactly L = h with p = min(2 h, h) / h = 1.
  const ties = createLocalPermutationHarness(device, {
    scene: {weights, values: new Float32Array(600).fill(3), rowStandardized: true},
    statistic: 'localMoran',
    parameters,
    maximumPermutations: 999,
    stopAfterExceedances: 12
  });
  const tied = await ties.run();
  for (let row = 0; row < 600; row++) {
    expect(tied.permutationCounts[row], `row ${row}`).toBe(12);
    expect(tied.exceedances[row]).toBe(12);
    expect(tied.pseudoPValues[row]).toBe(1);
  }
  // 'greater' on two far-apart clusters: the interior rows have an observed local Moran that few
  // simulations reach, so most never get 40 exceedances, run all P and keep (M + 1) / (P + 1).
  const extreme = createLocalPermutationHarness(device, {
    scene: {weights, values: clustered, rowStandardized: true},
    statistic: 'localMoran',
    parameters,
    maximumPermutations: 999,
    alternative: 'greater',
    stopAfterExceedances: 40
  });
  const result = await extreme.run();
  let neverStopped = 0;
  for (let row = 0; row < 600; row++) {
    if (result.exceedances[row] < 40) {
      neverStopped++;
      expect(result.permutationCounts[row]).toBe(permutations);
      expectProbability(result.pseudoPValues[row], result.exceedances[row] + 1, permutations + 1);
    }
  }
  expect(neverStopped).toBeGreaterThan(300);
  ties.destroy();
  extreme.destroy();
});

/**
 * Index of the first permutation at which the exceedance count of `alternative` reaches `target`
 * in the unstopped run, from its reference distribution, or `permutations` when it never does.
 */
function findStoppingIndex(
  unstopped: GlobalPermutationReadback,
  permutations: number,
  target: number,
  alternative: 'two-sided' | 'greater' | 'lesser'
): {permutations: number; stopped: boolean} {
  let greater = 0;
  let lesser = 0;
  const observed = unstopped.results[R.observed];
  for (let index = 0; index < permutations; index++) {
    const simulated = unstopped.referenceDistribution[index];
    greater += simulated >= observed ? 1 : 0;
    lesser += simulated <= observed ? 1 : 0;
    const count =
      alternative === 'greater'
        ? greater
        : alternative === 'lesser'
          ? lesser
          : Math.min(greater, lesser);
    if (count >= target) {
      return {permutations: index + 1, stopped: true};
    }
  }
  return {permutations, stopped: false};
}

it('GPUGlobalPermutationTest with stopAfterExceedances equals the unstopped run and its in-order scan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values} = createScene(900, 6);
  const secondValues = Float32Array.from(
    values,
    (value, index) => value + positions[index * 2] / 50
  );
  const weights = createKnnWeights(positions, 6, true);
  const permutations = 399;
  const target = 6;
  for (const statistic of ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'leesL'] as const) {
    for (const alternative of ['two-sided', 'greater', 'lesser'] as const) {
      const parameters = {seed: 31, permutationCount: permutations};
      const options = {
        scene: {weights, values, secondValues, rowStandardized: true},
        statistic,
        parameters,
        maximumPermutations: 511,
        alternative
      };
      const plain = createGlobalPermutationHarness(device, options);
      const sequential = createGlobalPermutationHarness(device, {
        ...options,
        stopAfterExceedances: target
      });
      const expected = await plain.run();
      const actual = await sequential.run();
      const label = `${statistic} ${alternative}`;
      const {permutations: used, stopped} = findStoppingIndex(
        expected,
        permutations,
        target,
        alternative
      );
      expect(actual.results[R.permutationCount], `${label} L`).toBe(used);
      expect(actual.results[R.observed]).toBe(expected.results[R.observed]);
      // The first L permutations are the unstopped run's, bit for bit; the rest are NaN.
      for (let index = 0; index < 511; index++) {
        if (index < used) {
          expect(actual.referenceDistribution[index], `${label} ref ${index}`).toBe(
            expected.referenceDistribution[index]
          );
        } else {
          expect(actual.referenceDistribution[index]).toBeNaN();
        }
      }
      const multiplier = alternative === 'two-sided' ? 2 : 1;
      if (stopped) {
        expect(actual.results[R.exceedances]).toBe(target);
        expect(actual.results[R.pseudoPValue]).toBe(
          f32(Math.min(multiplier * target, used)) / f32(used)
        );
      } else {
        expect(actual.results[R.exceedances]).toBe(expected.results[R.exceedances]);
        expect(actual.resultBits[R.pseudoPValue]).toBe(expected.resultBits[R.pseudoPValue]);
      }
      expect(actual.histogram.reduce((a, b) => a + b, 0)).toBe(used);
      plain.destroy();
      sequential.destroy();
    }
  }
});

it('GPUGlobalPermutationTest with h > P equals the unstopped run bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values} = createScene(600, 12);
  const weights = createKnnWeights(positions, 5, true);
  const options = {
    scene: {weights, values, rowStandardized: true},
    statistic: 'moran' as const,
    parameters: {seed: 4, permutationCount: 255},
    maximumPermutations: 255
  };
  const plain = createGlobalPermutationHarness(device, options);
  const sequential = createGlobalPermutationHarness(device, {
    ...options,
    stopAfterExceedances: 255
  });
  const expected = await plain.run();
  const actual = await sequential.run();
  // Summary rows can differ only in fields that are NaN in both: compare bits.
  expect(actual.resultBits).toEqual(expected.resultBits);
  expect(actual.referenceDistribution.map(f32)).toEqual(expected.referenceDistribution.map(f32));
  expect(actual.histogram).toEqual(expected.histogram);
  plain.destroy();
  sequential.destroy();
});

it('GPUGlobalPermutationTest: a clustered scene never stops under greater and stops at L = h under lesser', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values} = createScene(800, 9, true);
  const weights = createKnnWeights(positions, 5, true);
  const permutations = 299;
  const target = 10;
  const run = async (alternative: 'greater' | 'lesser') => {
    const harness = createGlobalPermutationHarness(device, {
      scene: {weights, values, rowStandardized: true},
      statistic: 'moran',
      parameters: {seed: 8, permutationCount: permutations},
      maximumPermutations: 511,
      alternative,
      stopAfterExceedances: target
    });
    const result = await harness.run();
    harness.destroy();
    return result;
  };
  const greater = await run('greater');
  expect(greater.results[R.permutationCount]).toBe(permutations);
  expect(greater.results[R.exceedances]).toBe(0);
  expect(greater.results[R.pseudoPValue]).toBe(f32(1 / (permutations + 1)));
  const lesser = await run('lesser');
  expect(lesser.results[R.permutationCount]).toBe(target);
  expect(lesser.results[R.exceedances]).toBe(target);
  expect(lesser.results[R.pseudoPValue]).toBe(1);
});

/**
 * A/B on one GPU: the same 10k-row lattice-like scene (P = 999) with stopping off and h = 10,
 * same run, logging both timings. Local test (a LISA map) first, then the global test. No timing
 * assertion: the correctness checks above are the gate; the log is the evidence.
 */
it('A/B: sequential stopping versus the full permutation run (10k rows, P = 999, h = 10)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values} = createScene(10000, 21);
  const weights = createKnnWeights(positions, 6, true);
  const parameters = {seed: 1, permutationCount: 999, significanceLevel: 0.05};
  const time = async <Result>(run: () => Promise<Result>) => {
    await run();
    const start = performance.now();
    const result = await run();
    return {result, milliseconds: performance.now() - start};
  };
  const localOptions = {
    scene: {weights, values, rowStandardized: true},
    statistic: 'localMoran' as const,
    parameters,
    maximumPermutations: 999
  };
  const localOld = createLocalPermutationHarness(device, localOptions);
  const localNew = createLocalPermutationHarness(device, {
    ...localOptions,
    stopAfterExceedances: 10
  });
  const oldLocal = await time(() => localOld.run());
  const newLocal = await time(() => localNew.run());
  const savedLocal =
    newLocal.result.permutationCounts.reduce((total, used) => total + used, 0) / (10000 * 999);
  console.log(
    `A/B local LISA map: old ${oldLocal.milliseconds.toFixed(1)} ms, new ${newLocal.milliseconds.toFixed(1)} ms, ` +
      `mean fraction of permutations used ${savedLocal.toFixed(3)}`
  );
  const globalOptions = {
    scene: {weights, values, rowStandardized: true},
    statistic: 'moran' as const,
    parameters: {seed: 1, permutationCount: 999},
    maximumPermutations: 999
  };
  const globalOld = createGlobalPermutationHarness(device, globalOptions);
  const globalNew = createGlobalPermutationHarness(device, {
    ...globalOptions,
    stopAfterExceedances: 10
  });
  const oldGlobal = await time(() => globalOld.run());
  const newGlobal = await time(() => globalNew.run());
  console.log(
    `A/B global Moran: old ${oldGlobal.milliseconds.toFixed(1)} ms, new ${newGlobal.milliseconds.toFixed(1)} ms, ` +
      `L = ${newGlobal.result.results[R.permutationCount]}`
  );
  expect(newGlobal.result.results[R.permutationCount]).toBeLessThanOrEqual(999);
  localOld.destroy();
  localNew.destroy();
  globalOld.destroy();
  globalNew.destroy();
});
