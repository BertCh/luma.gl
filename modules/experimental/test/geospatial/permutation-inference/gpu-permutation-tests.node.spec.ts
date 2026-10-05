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
} from '../../../src/geospatial/permutation-inference';
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
  expect(() => create({falseDiscoveryRate: true})).toThrow(/requires significant/);
  expect(() => create({pseudoPValues: view('float32', 9)})).toThrow(/pseudoPValues length/);
  expect(() => create({parameters: view('uint32', 2)})).toThrow(/parameters/);
  expect(() => create({exceedances: common.weights.offsets})).toThrow();
  const recipe = create({
    id: 'local',
    significant: view('uint32', 10),
    falseDiscoveryRate: true,
    observed: view('float32', 10)
  });
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('local-permute');
  expect(ids).toContain('local-fdr-threshold');
  expect(ids.at(-1)).toBe('local-classify');
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
  const recipe = create({
    id: 'global',
    histogram: view('uint32', 8),
    referenceDistribution: view('float32', 999)
  });
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
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
  // Folded pseudo p-values are uniform on {1..P/2+1}/(P+1) under the null, so about 10% are
  // at most 0.05; allow a generous band.
  const small = local.pseudoPValues.filter(p => p <= 0.05).length / count;
  expect(small).toBeGreaterThan(0.03);
  expect(small).toBeLessThan(0.2);
  // Benjamini-Hochberg removes almost all of these null discoveries.
  expect(local.significantFalseDiscoveryRate.filter(Boolean).length).toBeLessThan(
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
  expect(global.pseudoPValue).toBeCloseTo(1 / 100, 10);
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
