// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_GLOBAL_PERMUTATION_RESULT,
  GPU_LOCAL_PERMUTATION_NOT_TESTED,
  type GPULocalJoinCountCase,
  type GPULocalPermutationStatistic
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import type {GPUInferenceAlternative} from '../../../src/gpu-spatial-analysis/contracts';
import {createGlobalPermutationHarness, createLocalPermutationHarness} from './permutation-harness';
import {
  computeGlobalPermutationOracle,
  computeLocalPermutationOracle,
  createWeights,
  type CPUSpatialWeights
} from './permutation-oracle';
import {NEW_STATISTICS_REFERENCE as REFERENCE} from './new-statistics-reference';

// Oracle: fix/work/A/oracle-u7-new-statistics.py (esda 2.10.0 / libpysal 4.15.0). Statistics are
// pinned against esda; exceedance counts against the TypeScript twin of the Philox draws
// (permutation-oracle.ts); p-values against esda p_sim within a Monte Carlo bound at R = 9999.

const PERMUTATIONS = REFERENCE.permutations;
const SEED = 20240607;
/** Largest plausible difference of two independent p-value estimates at R = 9999 (five sigma). */
const P_VALUE_BOUND = 5 * Math.sqrt((2 * 0.25) / PERMUTATIONS);
/** esda's interpolated two-sided p-value adds a small definitional difference (see the node spec). */
const TWO_SIDED_P_VALUE_BOUND = P_VALUE_BOUND + 0.015;
const R = GPU_GLOBAL_PERMUTATION_RESULT;

const weights: CPUSpatialWeights = createWeights(
  REFERENCE.neighbors.map(neighbors =>
    neighbors.map((neighbor): [number, number] => [neighbor, 1 / REFERENCE.k])
  )
);
const x = Float32Array.from(REFERENCE.x);
const y = Float32Array.from(REFERENCE.y);
const binaryX = Float32Array.from(REFERENCE.bx);
const binaryZ = Float32Array.from(REFERENCE.bz);
const binaryComplement = Float32Array.from(REFERENCE.bv);

type Case = {
  label: string;
  statistic: GPULocalPermutationStatistic;
  joinCase?: GPULocalJoinCountCase;
  values: Float32Array;
  secondValues?: Float32Array;
  alternative: GPUInferenceAlternative;
  /** esda statistic. */
  statisticReference: readonly number[];
  /** esda p_sim (NaN as null). */
  pReference?: readonly (number | null)[];
  pBound: number;
};

const CASES: Case[] = [
  {
    label: 'local Geary',
    statistic: 'localGeary',
    values: x,
    alternative: 'two-sided',
    statisticReference: REFERENCE.gearyLocal,
    pReference: REFERENCE.gearyPSim,
    pBound: TWO_SIDED_P_VALUE_BOUND
  },
  {
    label: 'local bivariate Moran',
    statistic: 'localBivariateMoran',
    values: x,
    secondValues: y,
    alternative: 'two-sided',
    statisticReference: REFERENCE.bivariateMoranI,
    pReference: REFERENCE.bivariateMoranPSim,
    pBound: TWO_SIDED_P_VALUE_BOUND
  },
  {
    label: 'local join count',
    statistic: 'localJoinCount',
    values: binaryX,
    alternative: 'greater',
    statisticReference: REFERENCE.joinCount,
    pReference: REFERENCE.joinCountPSim,
    pBound: P_VALUE_BOUND
  },
  {
    label: 'bivariate join count CLC',
    statistic: 'localBivariateJoinCount',
    joinCase: 'CLC',
    values: binaryX,
    secondValues: binaryZ,
    alternative: 'greater',
    statisticReference: REFERENCE.joinCountCLC,
    pReference: REFERENCE.joinCountCLCPSim,
    pBound: P_VALUE_BOUND
  },
  {
    // esda simulates a different statistic than it observes for BJC, so only the statistic is
    // pinned against esda; the p-values follow the consistent numpy null of the oracle script.
    label: 'bivariate join count BJC',
    statistic: 'localBivariateJoinCount',
    joinCase: 'BJC',
    values: binaryX,
    secondValues: binaryComplement,
    alternative: 'greater',
    statisticReference: REFERENCE.joinCountBJC,
    pReference: REFERENCE.joinCountBJCConsistentPSim,
    pBound: 5 * Math.sqrt(0.25 / PERMUTATIONS + 0.25 / 99999)
  },
  {
    label: 'local Lee L',
    statistic: 'localLeeL',
    values: x,
    secondValues: y,
    alternative: 'directed',
    statisticReference: REFERENCE.leeLocal,
    pReference: REFERENCE.leeLocalPSim,
    pBound: P_VALUE_BOUND
  }
];

it('GPULocalPermutationTest new statistics match esda statistics, the twin counts and esda p_sim', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const parameters = {seed: SEED, permutations: PERMUTATIONS, significanceLevel: 0.05};
  for (const item of CASES) {
    const scene = {
      weights,
      rowStandardized: true,
      values: item.values,
      secondValues: item.secondValues
    };
    const harness = createLocalPermutationHarness(device, {
      scene,
      statistic: item.statistic,
      joinCase: item.joinCase,
      alternative: item.alternative,
      islands: 'keep',
      parameters,
      maximumPermutations: PERMUTATIONS,
      maximumNeighbors: 8
    });
    const result = await harness.run();
    const oracle = computeLocalPermutationOracle({
      weights,
      rowStandardized: true,
      values: item.values,
      secondValues: item.secondValues,
      statistic: item.statistic,
      joinCase: item.joinCase,
      alternative: item.alternative,
      islands: 'keep',
      maximumNeighbors: 8,
      ...parameters
    });
    expect(result.overflow, `${item.label} overflow`).toBe(0);
    let mismatched = 0;
    for (let row = 0; row < REFERENCE.n; row++) {
      // Statistic against esda (f32 inputs, sums and divisions).
      const reference = item.statisticReference[row];
      expect(
        Math.abs(result.observed[row] - reference),
        `${item.label} observed row ${row}`
      ).toBeLessThanOrEqual(1e-4 + 1e-4 * Math.abs(reference));
      // Counts against the twin of the Philox draws; near ties of the f32 terms may move one count.
      const expected = oracle.exceedances[row];
      if (expected === GPU_LOCAL_PERMUTATION_NOT_TESTED) {
        expect(result.exceedances[row], `${item.label} row ${row} not tested`).toBe(
          GPU_LOCAL_PERMUTATION_NOT_TESTED
        );
        expect(result.pseudoPValues[row]).toBeNaN();
        continue;
      }
      if (result.exceedances[row] !== expected) {
        mismatched++;
        expect(
          Math.abs(result.exceedances[row] - expected),
          `${item.label} exceedances row ${row}`
        ).toBeLessThanOrEqual(1);
      }
      // p-values against esda (null: esda NaN, our not-tested row; checked above).
      const pReference = item.pReference?.[row];
      if (pReference !== null && pReference !== undefined) {
        expect(
          Math.abs(result.pseudoPValues[row] - pReference),
          `${item.label} p row ${row}`
        ).toBeLessThanOrEqual(item.pBound);
      }
    }
    expect(mismatched, `${item.label} mismatched rows`).toBeLessThanOrEqual(2);
    if (item.pReference) {
      item.pReference.forEach((value, row) => {
        // A row esda leaves NaN (zero join count) is not tested here either.
        if (value === null) {
          expect(result.exceedances[row], `${item.label} untested row ${row}`).toBe(
            GPU_LOCAL_PERMUTATION_NOT_TESTED
          );
        }
      });
    }
    harness.destroy();
  }
});

it('GPULocalPermutationTest local Geary classes and bivariate Moran quadrants', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const parameters = {seed: SEED, permutations: 999, significanceLevel: 0.05};
  for (const multipleTesting of ['none', 'benjamini-hochberg'] as const) {
    const options = {
      scene: {weights, rowStandardized: true, values: x},
      statistic: 'localGeary' as const,
      alternative: 'two-sided' as const,
      islands: 'keep' as const,
      multipleTesting,
      parameters,
      maximumPermutations: 999,
      maximumNeighbors: 8
    };
    const harness = createLocalPermutationHarness(device, options);
    const result = await harness.run();
    const oracle = computeLocalPermutationOracle({
      weights,
      rowStandardized: true,
      values: x,
      statistic: 'localGeary',
      alternative: 'two-sided',
      islands: 'keep',
      multipleTesting,
      maximumNeighbors: 8,
      ...parameters
    });
    let different = 0;
    for (let row = 0; row < REFERENCE.n; row++) {
      different += result.classes[row] !== oracle.classes[row] ? 1 : 0;
    }
    // Same draws, same rule: a class can differ only where a count moved by an f32 tie.
    expect(different, `classes (${multipleTesting})`).toBeLessThanOrEqual(2);
    expect(result.classes.some(code => code === 1 || code === 2)).toBe(true);
    harness.destroy();
  }
  // Quadrants are esda Moran_Local_BV.q (PySAL scheme).
  const harness = createLocalPermutationHarness(device, {
    scene: {weights, rowStandardized: true, values: x, secondValues: y},
    statistic: 'localBivariateMoran',
    islands: 'keep',
    parameters,
    maximumPermutations: 999,
    maximumNeighbors: 8
  });
  const result = await harness.run();
  expect(result.quadrants).toEqual(REFERENCE.bivariateMoranQuadrants);
  harness.destroy();
});

it('GPULocalPermutationTest bivariate statistics exclude rows with a missing second value', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const missing = Float32Array.from(y);
  missing[4] = NaN;
  const mask = Uint32Array.from({length: REFERENCE.n}, (_, row) => (row === 9 ? 0 : 1));
  const parameters = {seed: SEED, permutations: 199, significanceLevel: 0.05};
  const harness = createLocalPermutationHarness(device, {
    scene: {weights, rowStandardized: true, values: x, secondValues: missing, mask},
    statistic: 'localBivariateMoran',
    parameters,
    maximumPermutations: 199,
    maximumNeighbors: 8
  });
  const result = await harness.run();
  const oracle = computeLocalPermutationOracle({
    weights,
    rowStandardized: true,
    values: x,
    secondValues: missing,
    mask,
    statistic: 'localBivariateMoran',
    maximumNeighbors: 8,
    ...parameters
  });
  for (const row of [4, 9]) {
    expect(result.observed[row]).toBeNaN();
    expect(result.exceedances[row]).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
    expect(result.quadrants[row]).toBe(0);
  }
  let mismatched = 0;
  for (let row = 0; row < REFERENCE.n; row++) {
    if (result.exceedances[row] !== oracle.exceedances[row]) {
      mismatched++;
    }
    if (!Number.isNaN(oracle.observed[row])) {
      expect(Math.abs(result.observed[row] - oracle.observed[row])).toBeLessThanOrEqual(
        1e-4 + 1e-4 * Math.abs(oracle.observed[row])
      );
    }
  }
  expect(mismatched).toBeLessThanOrEqual(2);
  harness.destroy();
});

it('GPUGlobalPermutationTest Lee L matches esda Spatial_Pearson, the twin and its p-value', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const parameters = {seed: SEED, permutations: PERMUTATIONS};
  const harness = createGlobalPermutationHarness(device, {
    scene: {weights, rowStandardized: true, values: x, secondValues: y},
    statistic: 'leesL',
    alternative: 'directed',
    islands: 'keep',
    parameters,
    maximumPermutations: PERMUTATIONS
  });
  const result = await harness.run();
  const oracle = computeGlobalPermutationOracle({
    weights,
    rowStandardized: true,
    values: x,
    secondValues: y,
    statistic: 'leesL',
    alternative: 'directed',
    islands: 'keep',
    ...parameters
  });
  expect(Math.abs(result.results[R.observed] - REFERENCE.leeGlobal)).toBeLessThan(1e-4);
  expect(Math.abs(result.results[R.observed] - oracle.observed)).toBeLessThan(1e-4);
  for (let index = 0; index < PERMUTATIONS; index++) {
    if (Math.abs(result.referenceDistribution[index] - oracle.simulated[index]) > 1e-4) {
      throw new Error(`Lee L permutation ${index + 1} differs from the twin`);
    }
  }
  const tolerance = 2e-4;
  const nearTies = oracle.simulated.filter(
    value => Math.abs(value - oracle.observed) <= tolerance
  ).length;
  expect(Math.abs(result.results[R.exceedances] - oracle.exceedances)).toBeLessThanOrEqual(
    nearTies
  );
  expect(Math.abs(result.results[R.pseudoPValue] - REFERENCE.leeGlobalPSim)).toBeLessThanOrEqual(
    P_VALUE_BOUND
  );
  expect(result.results[R.analysisCount]).toBe(REFERENCE.n);
  harness.destroy();
});
