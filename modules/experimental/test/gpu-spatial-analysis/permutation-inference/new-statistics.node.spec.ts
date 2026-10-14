// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGlobalPermutationTest,
  GPULocalPermutationTest,
  GPU_GLOBAL_PERMUTATION_RESULT,
  GPU_LOCAL_GEARY_CLASS,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  type GPULocalPermutationTestProps
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  classifyLocalGeary,
  computeGlobalPermutationOracle,
  computeLocalPermutationOracle,
  createWeights,
  type CPUSpatialWeights
} from './permutation-oracle';
import {NEW_STATISTICS_REFERENCE as REFERENCE} from './new-statistics-reference';

// Oracle: fix/work/A/oracle-u7-new-statistics.py (esda 2.10.0 / libpysal 4.15.0). The statistics
// are pinned against esda; the p-values against the TypeScript twin of our Philox draws, plus a
// Monte Carlo bound against esda's p_sim (different random streams cannot agree draw by draw).

const PERMUTATIONS = REFERENCE.permutations;
/**
 * Two independent estimates of one pseudo p-value at R = 9999: the standard deviation of each is at
 * most sqrt(0.25 / R) = 0.005, so their difference has at most 0.0071. Five of those, 0.036.
 */
const P_VALUE_BOUND = 5 * Math.sqrt((2 * 0.25) / PERMUTATIONS);
/**
 * esda's 'two-sided' p-value interpolates percentiles of the reference distribution instead of
 * counting 2 min(g, l), which shifts continuous-statistic p-values by up to about 0.01 on top of
 * the Monte Carlo error (nine seeds of the twin give a largest difference of 0.015 to 0.037).
 */
const TWO_SIDED_P_VALUE_BOUND = P_VALUE_BOUND + 0.015;
const SEED = 20240607;

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

function expectClose(actual: readonly number[], expected: readonly number[], label: string): void {
  expect(actual.length).toBe(expected.length);
  for (const [row, value] of expected.entries()) {
    // f32 inputs and accumulation: a few ulp relative to the larger of the value and 1.
    expect(Math.abs(actual[row] - value), `${label} row ${row}`).toBeLessThanOrEqual(
      2e-6 * Math.max(1, Math.abs(value))
    );
  }
}

function expectPValuesNear(
  actual: readonly number[],
  expected: readonly (number | null)[],
  label: string,
  bound = P_VALUE_BOUND
): number {
  let compared = 0;
  for (const [row, value] of expected.entries()) {
    if (value === null || Number.isNaN(actual[row])) {
      // esda NaN (zero join count) is exactly our not-tested row.
      expect(value === null, `${label} row ${row} tested`).toBe(Number.isNaN(actual[row]));
      continue;
    }
    expect(Math.abs(actual[row] - value), `${label} row ${row}`).toBeLessThanOrEqual(bound);
    compared++;
  }
  return compared;
}

const common = {
  weights,
  seed: SEED,
  permutations: PERMUTATIONS,
  significanceLevel: 0.05,
  maximumNeighbors: 8,
  islands: 'keep' as const,
  rowStandardized: true
};

it('local Geary matches esda Geary_Local and the twin p-values agree with esda p_sim', () => {
  const oracle = computeLocalPermutationOracle({
    ...common,
    values: x,
    statistic: 'localGeary',
    alternative: 'two-sided'
  });
  expectClose(oracle.observed, REFERENCE.gearyLocal, 'Geary_Local.localG');
  const compared = expectPValuesNear(
    oracle.pseudoPValues,
    REFERENCE.gearyPSim,
    'Geary p',
    TWO_SIDED_P_VALUE_BOUND
  );
  expect(compared).toBe(REFERENCE.n);
});

it('local Geary classes follow the esda Geary_Local label rule', () => {
  const mean = REFERENCE.x.reduce((total, value) => total + value, 0) / REFERENCE.n;
  const classes = classifyLocalGeary({
    observed: REFERENCE.gearyLocal,
    centeredValues: REFERENCE.x.map(value => value - mean),
    pValues: REFERENCE.gearyInjectedP.map(value => (value === null ? NaN : value)),
    level: 0.05
  });
  // esda NaN labels are code 0 (undefined); 1 high-high, 2 low-low, 3 negative association, 4 ns.
  expect(classes).toEqual(REFERENCE.gearyLabels.map(label => label ?? 0));
  const counts = [0, 1, 2, 3, 4].map(code => classes.filter(value => value === code).length);
  // The fixture exercises every class.
  expect(
    counts.every(count => count > 0),
    `class counts ${counts}`
  ).toBe(true);
  expect(GPU_LOCAL_GEARY_CLASS).toEqual({
    undefined: 0,
    highHigh: 1,
    lowLow: 2,
    negativeAssociation: 3,
    notSignificant: 4
  });
});

it('local bivariate Moran matches esda Moran_Local_BV', () => {
  const oracle = computeLocalPermutationOracle({
    ...common,
    values: x,
    secondValues: y,
    statistic: 'localBivariateMoran',
    alternative: 'two-sided'
  });
  expectClose(oracle.observed, REFERENCE.bivariateMoranI, 'Moran_Local_BV.Is');
  expect(oracle.quadrants).toEqual(REFERENCE.bivariateMoranQuadrants);
  expect(
    expectPValuesNear(
      oracle.pseudoPValues,
      REFERENCE.bivariateMoranPSim,
      'BV p',
      TWO_SIDED_P_VALUE_BOUND
    )
  ).toBe(REFERENCE.n);
});

it('local join counts match esda Join_Counts_Local and the p-values agree with p_sim', () => {
  const oracle = computeLocalPermutationOracle({
    ...common,
    values: binaryX,
    statistic: 'localJoinCount',
    alternative: 'greater'
  });
  expect(oracle.observed).toEqual(REFERENCE.joinCount);
  const compared = expectPValuesNear(oracle.pseudoPValues, REFERENCE.joinCountPSim, 'LJC p');
  expect(compared).toBe(REFERENCE.joinCount.filter(count => count > 0).length);
  expect(compared).toBeGreaterThan(10);
});

it('bivariate join counts: CLC matches esda, BJC statistic matches and p-values follow the consistent null', () => {
  const clc = computeLocalPermutationOracle({
    ...common,
    values: binaryX,
    secondValues: binaryZ,
    statistic: 'localBivariateJoinCount',
    joinCase: 'CLC',
    alternative: 'greater'
  });
  expect(clc.observed).toEqual(REFERENCE.joinCountCLC);
  expect(expectPValuesNear(clc.pseudoPValues, REFERENCE.joinCountCLCPSim, 'CLC p')).toBe(
    REFERENCE.joinCountCLC.filter(count => count > 0).length
  );
  const bjc = computeLocalPermutationOracle({
    ...common,
    values: binaryX,
    secondValues: binaryComplement,
    statistic: 'localBivariateJoinCount',
    joinCase: 'BJC',
    alternative: 'greater'
  });
  expect(bjc.observed).toEqual(REFERENCE.joinCountBJC);
  // esda simulates x_i * sum(z_j) for BJC, a different statistic from the one it observes (it ignores
  // z_i = 0 and x_j = 0). We simulate the observed statistic; the reference below does the same in
  // numpy (99999 draws), and differs from esda's own p_sim.
  const bound = 5 * Math.sqrt(0.25 / PERMUTATIONS + 0.25 / 99999);
  let tested = 0;
  let esdaDisagrees = 0;
  for (const [row, consistent] of REFERENCE.joinCountBJCConsistentPSim.entries()) {
    if (consistent === null) {
      expect(bjc.pseudoPValues[row]).toBeNaN();
      continue;
    }
    tested++;
    expect(Math.abs(bjc.pseudoPValues[row] - consistent), `BJC row ${row}`).toBeLessThanOrEqual(
      bound
    );
    const esda = REFERENCE.joinCountBJCPSim[row]!;
    esdaDisagrees += Math.abs(esda - consistent) > 0.05 ? 1 : 0;
  }
  expect(tested).toBe(REFERENCE.joinCountBJC.filter(count => count > 0).length);
  expect(esdaDisagrees).toBeGreaterThan(0);
});

it('local Lee L matches esda Spatial_Pearson_Local and the p-values agree with its directed p', () => {
  const oracle = computeLocalPermutationOracle({
    ...common,
    values: x,
    secondValues: y,
    statistic: 'localLeeL',
    alternative: 'directed'
  });
  expectClose(oracle.observed, REFERENCE.leeLocal, 'Spatial_Pearson_Local.associations_');
  expect(expectPValuesNear(oracle.pseudoPValues, REFERENCE.leeLocalPSim, 'Lee local p')).toBe(
    REFERENCE.n
  );
});

it('global Lee L matches esda Spatial_Pearson and the p-value agrees with significance_', () => {
  const oracle = computeGlobalPermutationOracle({
    weights,
    values: x,
    secondValues: y,
    statistic: 'leesL',
    seed: SEED,
    permutations: PERMUTATIONS,
    alternative: 'directed',
    islands: 'keep',
    rowStandardized: true
  });
  expect(Math.abs(oracle.observed - REFERENCE.leeGlobal)).toBeLessThan(1e-6);
  expect(Math.abs(oracle.pseudoPValue - REFERENCE.leeGlobalPSim)).toBeLessThanOrEqual(
    P_VALUE_BOUND
  );
  // The mean of the local values is the global L for row-standardized weights (Lee 2001).
  const meanLocal = REFERENCE.leeLocal.reduce((total, value) => total + value, 0) / REFERENCE.n;
  expect(Math.abs(meanLocal - REFERENCE.leeGlobal)).toBeLessThan(1e-9);
});

it('GPULocalPermutationTest validates the new statistics and stays within eight storage buffers', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'new-statistics-validation'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const create = (overrides: Partial<GPULocalPermutationTestProps> = {}) =>
    new GPULocalPermutationTest({
      weights: {
        offsets: view('uint32', 11),
        neighbors: view('uint32', 40),
        weights: view('float32', 40)
      },
      values: view('float32', 10),
      parameters: view('uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
      maximumPermutations: 999,
      statistic: 'localGeary',
      exceedances: view('uint32', 10),
      pseudoPValues: view('float32', 10),
      overflow: view('uint32', 1),
      ...overrides
    });
  const second = () => view('float32', 10);
  expect(() => create({statistic: 'localBivariateMoran'})).toThrow(/requires secondValues/);
  expect(() => create({statistic: 'localLeeL'})).toThrow(/requires secondValues/);
  expect(() => create({statistic: 'localBivariateJoinCount', secondValues: second()})).toThrow(
    /joinCase/
  );
  expect(() =>
    create({statistic: 'localBivariateJoinCount', secondValues: second(), joinCase: 'CLC'})
  ).not.toThrow();
  expect(() => create({joinCase: 'BJC'})).toThrow(/joinCase/);
  expect(() => create({secondValues: second()})).toThrow(/does not take secondValues/);
  expect(() => create({statistic: 'localMoran', classes: view('uint32', 10)})).toThrow(/classes/);
  expect(() => create({statistic: 'localGeary', quadrants: view('uint32', 10)})).toThrow(
    /quadrants/
  );
  expect(() => create({classes: view('uint32', 9)})).toThrow(/classes length/);
  // Building the nodes checks the storage-buffer limit of every kernel.
  const build = (id: string, overrides: Partial<GPULocalPermutationTestProps>) =>
    create({id, ...overrides})
      .getCommandNodes(graph)
      .map(node => node.id);
  const bivariate = build('bvm', {
    statistic: 'localBivariateMoran',
    secondValues: second(),
    quadrants: view('uint32', 10),
    observed: view('float32', 10),
    significant: view('uint32', 10)
  });
  expect(bivariate).toContain('bvm-pairs');
  const geary = build('geary', {
    classes: view('uint32', 10),
    multipleTesting: 'holm',
    adjustedPValues: view('float32', 10)
  });
  expect(geary).toContain('geary-geary-classes');
  expect(geary.indexOf('geary-geary-classes')).toBeGreaterThan(
    geary.findIndex(id => id.startsWith('geary-multiple-testing'))
  );
  expect(build('geary-plain', {classes: view('uint32', 10)})).toContain(
    'geary-plain-geary-classes'
  );
  for (const joinCase of ['BJC', 'CLC'] as const) {
    build(`join-${joinCase}`, {
      statistic: 'localBivariateJoinCount',
      joinCase,
      secondValues: second()
    });
  }
  build('join', {statistic: 'localJoinCount'});
  build('lee', {statistic: 'localLeeL', secondValues: second()});
});

it('GPUGlobalPermutationTest accepts leesL only with secondValues', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'lee-validation'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `lee-view-${serial++}`, format, length);
  const create = (secondValues?: ReturnType<typeof view<'float32'>>) =>
    new GPUGlobalPermutationTest({
      weights: {
        offsets: view('uint32', 11),
        neighbors: view('uint32', 40),
        weights: view('float32', 40)
      },
      values: view('float32', 10),
      secondValues,
      parameters: view('uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
      maximumPermutations: 999,
      statistic: 'leesL',
      results: view('float32', GPU_GLOBAL_PERMUTATION_RESULT.length)
    });
  expect(() => create()).toThrow(/leesL requires secondValues/);
  const ids = create(view('float32', 10))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('global-permutation-test-pairs');
});
