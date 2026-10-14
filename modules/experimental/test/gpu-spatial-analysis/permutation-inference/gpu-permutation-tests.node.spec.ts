// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGlobalPermutationTest,
  GPULocalPermutationTest,
  GPU_GLOBAL_PERMUTATION_RESULT,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  type GPUGlobalPermutationTestProps,
  type GPULocalPermutationTestProps
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeGlobalPermutationOracle,
  computeLocalPermutationOracle,
  createKnnWeights,
  createSeededRandom
} from './permutation-oracle';

let serial = 0;

function createViews(graph: GPUCommandGraph) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    view,
    common: {
      weights: {
        offsets: view('uint32', 11),
        neighbors: view('uint32', 40),
        weights: view('float32', 40)
      },
      values: view('float32', 10),
      parameters: view('uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
      maximumPermutations: 999
    }
  };
}

it('GPULocalPermutationTest validates its props and emits deterministic node IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'local-permutation-validation'});
  const {view, common} = createViews(graph);
  const create = (overrides: Partial<GPULocalPermutationTestProps> = {}) =>
    new GPULocalPermutationTest({
      ...common,
      statistic: 'localMoran',
      exceedances: view('uint32', 10),
      pseudoPValues: view('float32', 10),
      overflow: view('uint32', 1),
      ...overrides
    });
  expect(() => create()).not.toThrow();
  expect(() => create({maximumNeighbors: 65})).toThrow(/maximumNeighbors/);
  expect(() => create({maximumPermutations: 0})).toThrow(/maximumPermutations/);
  expect(() => create({statistic: 'nope' as 'localG'})).toThrow(/unknown/);
  expect(() => create({multipleTesting: 'holm'})).toThrow(/requires significant/);
  expect(() => create({falseDiscoveryRate: true})).toThrow(/requires significant/);
  expect(() => create({islands: 'drop' as 'keep'})).toThrow(/islands/);
  expect(() => create({multipleTesting: 'sidak' as 'holm'})).toThrow(/method must be/);
  // The deprecated alias may not contradict an explicit method.
  expect(() =>
    create({
      significant: view('uint32', 10),
      falseDiscoveryRate: true,
      multipleTesting: 'bonferroni'
    })
  ).toThrow(/falseDiscoveryRate/);
  expect(() =>
    create({
      significant: view('uint32', 10),
      falseDiscoveryRate: true,
      multipleTesting: 'benjamini-hochberg'
    })
  ).not.toThrow();
  expect(() => create({pseudoPValues: view('float32', 9)})).toThrow(/pseudoPValues length/);
  expect(() => create({parameters: view('uint32', 2)})).toThrow(/parameters/);
  expect(() => create({exceedances: common.weights.offsets})).toThrow();
  const contributor = create({
    id: 'local',
    significant: view('uint32', 10),
    multipleTesting: 'benjamini-hochberg',
    adjustedPValues: view('float32', 10),
    islandCount: view('uint32', 1),
    observed: view('float32', 10)
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('local-permute');
  expect(ids.some(id => id.startsWith('local-multiple-testing'))).toBe(true);
  expect(ids).toContain('local-inputs-island-clear');
  expect(ids.at(-1)).toBe('local-significant');
  // Without a correction the exact integer comparison is fused into the classify kernel.
  const plain = create({id: 'plain', significant: view('uint32', 10)});
  expect(
    plain
      .getCommandNodes(graph)
      .map(node => node.id)
      .at(-1)
  ).toBe('plain-classify');
  expect(new Set(ids).size).toBe(ids.length);
});

it('GPUGlobalPermutationTest validates its props and emits deterministic node IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'global-permutation-validation'
  });
  const {view, common} = createViews(graph);
  const create = (overrides: Partial<GPUGlobalPermutationTestProps> = {}) =>
    new GPUGlobalPermutationTest({
      ...common,
      statistic: 'moran',
      results: view('float32', GPU_GLOBAL_PERMUTATION_RESULT.length),
      ...overrides
    });
  expect(() => create()).not.toThrow();
  expect(() => create({statistic: 'bivariateMoran'})).toThrow(/secondValues/);
  expect(() => create({results: view('float32', 4)})).toThrow(/results/);
  expect(() => create({referenceDistribution: view('float32', 10)})).toThrow(
    /referenceDistribution/
  );
  expect(() => create({values: view('float32', 3)})).toThrow(/values length/);
  const contributor = create({
    id: 'global',
    histogram: view('uint32', 8),
    referenceDistribution: view('float32', 999)
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('global-pairs');
  expect(ids.at(-1)).toBe('global-finalize');
  expect(new Set(ids).size).toBe(ids.length);
});

it('permutation oracles behave as nulls on random values and detect clustered values', () => {
  const random = createSeededRandom(17);
  const count = 400;
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const weights = createKnnWeights(positions, 6, true);
  const noise = Float32Array.from({length: count}, () => random());
  const local = computeLocalPermutationOracle({
    weights,
    values: noise,
    statistic: 'localMoran',
    seed: 3,
    permutations: 199,
    significanceLevel: 0.05,
    maximumNeighbors: 32
  });
  // Two-sided pseudo p-values are super-uniform under the null, so about 5% are at most 0.05
  // (the legacy 'directed' tail gave about 10%); allow a generous band.
  const small = local.pseudoPValues.filter(p => p <= 0.05).length / count;
  expect(small).toBeGreaterThan(0.01);
  expect(small).toBeLessThan(0.1);
  // Benjamini-Hochberg removes almost all of these null discoveries.
  const corrected = computeLocalPermutationOracle({
    weights,
    values: noise,
    statistic: 'localMoran',
    seed: 3,
    permutations: 199,
    significanceLevel: 0.05,
    maximumNeighbors: 32,
    multipleTesting: 'benjamini-hochberg'
  });
  expect(corrected.significant.filter(Boolean).length).toBeLessThan(
    local.significant.filter(Boolean).length
  );

  const clustered = Float32Array.from(
    {length: count},
    (_, index) => positions[index * 2] / 10 + random()
  );
  const global = computeGlobalPermutationOracle({
    weights,
    values: clustered,
    statistic: 'moran',
    seed: 5,
    permutations: 99
  });
  expect(global.pseudoPValue).toBeCloseTo(2 / 100, 10);
  expect(global.zSimulated).toBeGreaterThan(5);
  const nullGlobal = computeGlobalPermutationOracle({
    weights,
    values: noise,
    statistic: 'moran',
    seed: 5,
    permutations: 499
  });
  expect(Math.abs(nullGlobal.simulatedMean + 1 / (count - 1))).toBeLessThan(0.01);
});

it('both tests validate permutation counts and describe their inference', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'permutation-metadata'});
  const {view, common} = createViews(graph);
  const local = new GPULocalPermutationTest({
    ...common,
    statistic: 'localMoran',
    maximumPermutations: 99,
    maximumNeighbors: 8,
    significant: view('uint32', 10),
    multipleTesting: 'benjamini-yekutieli',
    exceedances: view('uint32', 10),
    pseudoPValues: view('float32', 10),
    overflow: view('uint32', 1)
  });
  expect(local.getParameterValues({seed: 5, permutations: 99})[2]).toBe(99);
  expect(() => local.getParameterValues({seed: 5, permutations: 100})).toThrow(
    /maximumPermutations/
  );
  const metadata = local.getInferenceMetadata({seed: 5, permutations: 49, significanceLevel: 0.01});
  expect(metadata).toMatchObject({
    alternative: 'two-sided',
    permutationCount: 49,
    seed: 5,
    multipleTesting: 'benjamini-yekutieli',
    significanceLevel: 0.01
  });
  expect(metadata.testedRowsDefinition).toMatch(/1 to 8 positive-weight neighbors/);
  expect(metadata.testedRowsDefinition).toMatch(/excluded/);

  const global = new GPUGlobalPermutationTest({
    ...common,
    statistic: 'moran',
    alternative: 'less',
    islands: 'keep',
    maximumPermutations: 99,
    results: view('float32', GPU_GLOBAL_PERMUTATION_RESULT.length)
  });
  expect(() => global.getParameterValues({seed: 5, permutations: 1000})).toThrow(
    /maximumPermutations/
  );
  const globalMetadata = global.getInferenceMetadata({seed: 5, permutations: 99});
  expect(globalMetadata.alternative).toBe('lesser');
  expect(globalMetadata.multipleTesting).toBe('none');
  expect(globalMetadata.testedRowsDefinition).toMatch(/kept in n/);
  expect(GPU_GLOBAL_PERMUTATION_RESULT.islandCount).toBe(10);
  expect(GPU_GLOBAL_PERMUTATION_RESULT.analysisCount).toBe(11);
  expect(GPU_GLOBAL_PERMUTATION_RESULT.length).toBe(12);
});
