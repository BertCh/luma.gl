// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGlobalPermutationTest,
  GPULocalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  type GPUGlobalPermutationStatistic,
  type GPULocalPermutationStatistic
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {getGPUGlobalPermutationBatchSize} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test';
import {
  getPermutationPValue,
  getSequentialPermutationPValue
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-alternative';
import {
  countDistinctPipelines,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  computeGlobalPermutationOracle,
  computeLocalPermutationOracle,
  createKnnWeights,
  createSeededRandom
} from './permutation-oracle';

let serial = 0;

function createViews(graph: GPUCommandGraph, maximumPermutations = 999) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `sequential-view-${serial++}`, format, length);
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
      maximumPermutations
    }
  };
}

/**
 * The kernels this feature adds or changes. The shared front-end and column-sum kernels are excluded:
 * `wgsl_reflect` does not parse some of them, which is independent of sequential stopping.
 */
function getOwnNodes<Node extends {id: string}>(nodes: readonly Node[]): Node[] {
  return nodes.filter(node =>
    /-(permute|classify|split-counts|geary-classes|sequential-init|pairs|pairs-observed|pair-sums|pair-sums-observed|finalize|batch-\d+-(pairs|pair-sums|scan))$/.test(
      node.id
    )
  );
}

function createLocal(
  graph: GPUCommandGraph,
  overrides: {
    statistic?: GPULocalPermutationStatistic;
    stopAfterExceedances?: number;
    maximumPermutations?: number;
    withCounts?: boolean;
    alternative?: 'two-sided' | 'greater' | 'lesser' | 'directed' | 'folded';
    significant?: boolean;
    classes?: boolean;
  } = {}
) {
  const {view, common} = createViews(graph, overrides.maximumPermutations);
  const bivariate = overrides.statistic === 'localBivariateMoran';
  return new GPULocalPermutationTest({
    ...common,
    id: `local-${serial++}`,
    statistic: overrides.statistic ?? 'localMoran',
    secondValues: bivariate ? view('float32', 10) : undefined,
    alternative: overrides.alternative,
    stopAfterExceedances: overrides.stopAfterExceedances,
    output: {
      exceedances: view('uint32', 10),
      pseudoPValues: view('float32', 10),
      permutationCounts: overrides.withCounts ? view('uint32', 10) : undefined,
      significant: overrides.significant ? view('uint32', 10) : undefined,
      classes: overrides.classes ? view('uint32', 10) : undefined,
      status: {overflow: view('uint32', 1)}
    }
  });
}

function createGlobal(
  graph: GPUCommandGraph,
  overrides: {
    statistic?: GPUGlobalPermutationStatistic;
    stopAfterExceedances?: number;
    maximumPermutations?: number;
    alternative?: 'two-sided' | 'greater' | 'lesser' | 'directed' | 'folded';
  } = {}
) {
  const {view, common} = createViews(graph, overrides.maximumPermutations);
  const statistic = overrides.statistic ?? 'moran';
  const bivariate = statistic === 'bivariateMoran' || statistic === 'leesL';
  return new GPUGlobalPermutationTest({
    ...common,
    id: `global-${serial++}`,
    statistic,
    secondValues: bivariate ? view('float32', 10) : undefined,
    alternative: overrides.alternative,
    stopAfterExceedances: overrides.stopAfterExceedances,
    output: {summary: view('float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length)}
  });
}

it('stopAfterExceedances validates its range, the folded alternative and permutationCounts', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'sequential-validation'});
  for (const create of [
    (stopAfterExceedances: number, alternative?: 'folded') =>
      createLocal(graph, {stopAfterExceedances, alternative, maximumPermutations: 99}),
    (stopAfterExceedances: number, alternative?: 'folded') =>
      createGlobal(graph, {stopAfterExceedances, alternative, maximumPermutations: 99})
  ]) {
    expect(() => create(1)).not.toThrow();
    expect(() => create(99)).not.toThrow();
    expect(() => create(0)).toThrow(/stopAfterExceedances must be an integer in \[1, 99\]/);
    expect(() => create(100)).toThrow(/stopAfterExceedances/);
    expect(() => create(1.5)).toThrow(/stopAfterExceedances/);
    expect(() => create(5, 'folded')).toThrow(/does not support the folded alternative/);
  }
  expect(() => createLocal(graph, {withCounts: true})).toThrow(
    /permutationCounts requires stopAfterExceedances/
  );
  expect(() => createLocal(graph, {withCounts: true, stopAfterExceedances: 5})).not.toThrow();
});

it('the sequential rule stops at the first exceedance count h and keeps the standard p otherwise', () => {
  const flags = (pattern: string) =>
    [...pattern].map(character => ({greater: character === 'g', lesser: character === 'l'}));
  // Greater: exceedances at permutations 2, 4 and 5; h = 2 stops at L = 4 with p = 2 / 4.
  expect(getSequentialPermutationPValue('greater', flags('.g.g.g'), 2)).toEqual({
    permutations: 4,
    exceedances: 2,
    stopped: true,
    pValue: 0.5
  });
  // Two-sided counts min(greater, lesser) and doubles: needs both tails; g and l alternate.
  const alternating = getSequentialPermutationPValue('two-sided', flags('glglgl'), 2);
  expect(alternating).toMatchObject({permutations: 4, exceedances: 2, stopped: true});
  expect(alternating.pValue).toBe(1);
  // Never reaching h: the standard (M + 1) / (P + 1).
  const unstopped = getSequentialPermutationPValue('greater', flags('.g....'), 3);
  expect(unstopped).toMatchObject({permutations: 6, exceedances: 1, stopped: false});
  expect(unstopped.pValue).toBe(getPermutationPValue('greater', {greater: 1, lesser: 0}, 6).pValue);
  // Reaching h exactly at L = P is a stop: p = h / P, not (h + 1) / (P + 1).
  expect(getSequentialPermutationPValue('greater', flags('....gg'), 2)).toMatchObject({
    permutations: 6,
    stopped: true,
    pValue: 2 / 6
  });
  expect(() => getSequentialPermutationPValue('folded', [], 1)).toThrow();
});

it('the local oracle stops each row at its first h-th exceedance and h > P equals the full run', () => {
  const random = createSeededRandom(23);
  const count = 120;
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const weights = createKnnWeights(positions, 5, true);
  const values = Float32Array.from({length: count}, (_, index) => positions[index * 2] + random());
  const base = {
    weights,
    values,
    statistic: 'localMoran' as const,
    seed: 11,
    permutationCount: 99,
    significanceLevel: 0.05,
    maximumNeighbors: 32
  };
  const full = computeLocalPermutationOracle(base);
  const never = computeLocalPermutationOracle({...base, stopAfterExceedances: 100});
  expect(never.exceedances).toEqual(full.exceedances);
  expect(never.pseudoPValues).toEqual(full.pseudoPValues);
  expect(never.permutationCounts.every(used => used === 99)).toBe(true);
  const stopped = computeLocalPermutationOracle({...base, stopAfterExceedances: 5});
  let stoppedRows = 0;
  for (let row = 0; row < count; row++) {
    const used = stopped.permutationCounts[row];
    expect(used).toBeLessThanOrEqual(99);
    if (stopped.exceedances[row] === 5) {
      stoppedRows++;
      expect(stopped.pseudoPValues[row]).toBeCloseTo(Math.min(2 * 5, used) / used, 6);
    } else {
      // Not stopped: all permutations ran and the standard p applies.
      expect(used).toBe(99);
      expect(stopped.pseudoPValues[row]).toBe(full.pseudoPValues[row]);
    }
  }
  // Noise-like rows are not significant, so a good share stops early.
  expect(stoppedRows).toBeGreaterThan(count / 4);
  expect(stopped.permutationCounts.some(used => used < 99)).toBe(true);
});

it('the global oracle supports the sequential rule through the shared flags', () => {
  const random = createSeededRandom(5);
  const count = 100;
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const weights = createKnnWeights(positions, 5, true);
  const values = Float32Array.from({length: count}, () => random());
  const oracle = computeGlobalPermutationOracle({
    weights,
    values,
    statistic: 'moran',
    seed: 3,
    permutationCount: 99
  });
  const flags = oracle.pairSums.slice(1).map(sum => ({
    greater: sum >= oracle.pairSums[0],
    lesser: sum <= oracle.pairSums[0]
  }));
  const sequential = getSequentialPermutationPValue('two-sided', flags, 5);
  expect(sequential.permutations).toBeLessThanOrEqual(99);
  if (!sequential.stopped) {
    expect(sequential.pValue).toBe(oracle.pseudoPValue);
  }
});

it('sequential local kernels parse and keep the off path free of stopping code', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'sequential-local-wgsl'});
  for (const statistic of [
    'localMoran',
    'localGeary',
    'localBivariateMoran'
  ] as const satisfies GPULocalPermutationStatistic[]) {
    const on = createLocal(graph, {
      statistic,
      stopAfterExceedances: 10,
      withCounts: true,
      significant: true,
      classes: statistic === 'localGeary'
    });
    const sources = getParsedKernelSources(getOwnNodes(on.getCommandNodes(graph)));
    const permute = [...sources].find(([id]) => id.endsWith('-permute'))![1];
    expect(permute).toContain('STOP_AFTER_EXCEEDANCES');
    expect(permute).toContain('break;');
    const classify = [...sources].find(([id]) => id.endsWith('-classify'))![1];
    expect(classify).toContain('permutationCounts');
    expect([...sources.keys()].some(id => id.endsWith('-split-counts'))).toBe(true);
    const off = createLocal(graph, {statistic, classes: statistic === 'localGeary'});
    const offSources = getParsedKernelSources(getOwnNodes(off.getCommandNodes(graph)));
    expect([...offSources.values()].some(source => source.includes('STOP_AFTER'))).toBe(false);
    expect([...offSources.keys()].some(id => id.endsWith('-split-counts'))).toBe(false);
  }
});

it('sequential global kernels parse for every statistic and the off path keeps one dispatch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'sequential-global-wgsl'});
  for (const statistic of [
    'moran',
    'geary',
    'getisOrdG',
    'bivariateMoran',
    'leesL'
  ] as const satisfies GPUGlobalPermutationStatistic[]) {
    const nodes = createGlobal(graph, {
      statistic,
      stopAfterExceedances: 10,
      maximumPermutations: 999
    }).getCommandNodes(graph);
    // `wgsl_reflect` cannot parse the pair kernels (`workgroup % BLOCK_COUNT`, a `block` variable;
    // both unchanged by this feature), so they are excluded from the parse and compared as text.
    const parseable = getOwnNodes(nodes).filter(node => !/-pairs|-pair-sums/.test(node.id));
    getParsedKernelSources(parseable);
    const sources = new Map(
      getGPUCommandNodeKernelSources(nodes).map(kernel => [kernel.id, kernel.source])
    );
    const batchSize = getGPUGlobalPermutationBatchSize(10, 999);
    const batches = Math.ceil(999 / batchSize);
    expect([...sources.keys()].filter(id => /-batch-\d+-scan$/.test(id))).toHaveLength(batches);
    // Every batch shares one shader per kernel: the batch base is a state word (rule 12).
    for (const suffix of ['pairs', 'pair-sums', 'scan']) {
      const distinct = new Set(
        [...sources]
          .filter(([id]) => new RegExp(`-batch-\\d+-${suffix}$`).test(id))
          .map(([, s]) => s)
      );
      expect(distinct.size, suffix).toBe(1);
    }
    const off = createGlobal(graph, {statistic}).getCommandNodes(graph);
    const offIds = off.map(node => node.id);
    expect(offIds.some(id => id.includes('-batch-'))).toBe(false);
    expect(offIds.filter(id => id.endsWith('-pairs'))).toHaveLength(1);
    const offSources = getGPUCommandNodeKernelSources(off).map(kernel => kernel.source);
    expect(offSources.some(source => source.includes('STOP_AFTER'))).toBe(false);
  }
});

it('rule 12: the number of distinct pipelines does not grow with the number of batches', () => {
  // Rows are few, so a batch is 128 permutations while maximumPermutations <= 8192.
  const pipelines = (maximumPermutations: number) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `batches-${serial++}`});
    return countDistinctPipelines(
      createGlobal(graph, {stopAfterExceedances: 10, maximumPermutations}).getCommandNodes(graph)
    );
  };
  expect(getGPUGlobalPermutationBatchSize(10, 256)).toBe(128);
  expect(getGPUGlobalPermutationBatchSize(10, 384)).toBe(128);
  const twoBatches = pipelines(256);
  const threeBatches = pipelines(384);
  expect(threeBatches).toBe(twoBatches);
  expect(pipelines(1024)).toBe(twoBatches);
});

it('the batch size occupies the GPU, caps the batch count and never exceeds maximumPermutations', () => {
  expect(getGPUGlobalPermutationBatchSize(10, 999)).toBe(128);
  expect(getGPUGlobalPermutationBatchSize(10, 50)).toBe(50);
  // 20000 rows are 10 row blocks: 13 permutations fill 130 workgroups.
  expect(getGPUGlobalPermutationBatchSize(20000, 999)).toBe(16);
  // A million permutations are at most 64 batches.
  expect(Math.ceil(2 ** 20 / getGPUGlobalPermutationBatchSize(10, 2 ** 20))).toBeLessThanOrEqual(
    64
  );
});
