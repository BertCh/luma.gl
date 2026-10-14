// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUGlobalPermutationTest,
  GPULocalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  type GPUGlobalPermutationStatistic,
  type GPULocalPermutationStatistic
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 100;
const SLOTS = 8;
let serial = 0;

type Resource = {buffer: {byteLength?: number; length?: number; rowByteLength?: number}};

/** Bytes of every buffer or view bound by the nodes. */
function getBoundBytes(nodes: readonly unknown[]): {id: string; bytes: number}[] {
  const result: {id: string; bytes: number}[] = [];
  for (const node of nodes as {
    id: string;
    resources?: Resource[];
    workload?: {resources?: Resource[]};
  }[]) {
    for (const resource of node.resources ?? node.workload?.resources ?? []) {
      const buffer = resource.buffer;
      const bytes =
        buffer.byteLength ??
        (buffer.length ?? 0) * (buffer.rowByteLength ?? Uint32Array.BYTES_PER_ELEMENT);
      result.push({id: node.id, bytes});
    }
  }
  return result;
}

function expectCoreLimits(nodes: readonly unknown[], device = createNullWebGPUDevice()): void {
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(0);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), `${id} private bytes`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  const bound = getBoundBytes(nodes);
  expect(bound.length).toBeGreaterThan(0);
  for (const {id, bytes} of bound) {
    expect(bytes, `${id} bound bytes`).toBeLessThanOrEqual(
      device.limits.maxStorageBufferBindingSize
    );
  }
}

function createCommon(graph: GPUCommandGraph, maximumPermutations: number) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `permutation-limits-${serial++}`, format, length);
  return {
    view,
    common: {
      weights: {
        offsets: view('uint32', ROWS + 1),
        neighbors: view('uint32', ROWS * SLOTS),
        weights: view('float32', ROWS * SLOTS)
      },
      values: view('float32', ROWS),
      secondValues: view('float32', ROWS),
      mask: view('uint32', ROWS),
      parameters: view('uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
      maximumPermutations
    }
  };
}

const LOCAL_STATISTICS: GPULocalPermutationStatistic[] = [
  'localMoran',
  'localGeary',
  'localG',
  'localGStar',
  'localBivariateMoran',
  'localLeeL',
  'localJoinCount',
  'localBivariateJoinCount'
];

function buildLocal(
  statistic: GPULocalPermutationStatistic,
  maximumPermutations: number,
  overrides: {
    joinCase?: 'BJC' | 'CLC';
    multipleTesting?: 'bonferroni' | 'holm' | 'benjamini-hochberg' | 'benjamini-yekutieli';
    maximumNeighbors?: number;
    islands?: 'keep' | 'exclude';
  } = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `limits-local-${serial++}`});
  const {view, common} = createCommon(graph, maximumPermutations);
  const bivariate = ['localBivariateMoran', 'localLeeL', 'localBivariateJoinCount'].includes(
    statistic
  );
  const contributor = new GPULocalPermutationTest({
    ...common,
    id: `local-${statistic}`,
    statistic,
    secondValues: bivariate ? common.secondValues : undefined,
    joinCase: statistic === 'localBivariateJoinCount' ? (overrides.joinCase ?? 'BJC') : undefined,
    maximumNeighbors: overrides.maximumNeighbors ?? GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS,
    islands: overrides.islands ?? 'keep',
    alternative: 'two-sided',
    multipleTesting: overrides.multipleTesting ?? 'benjamini-hochberg',
    output: {
      exceedances: view('uint32', ROWS),
      pseudoPValues: view('float32', ROWS),
      observed: view('float32', ROWS),
      significant: view('uint32', ROWS),
      adjustedPValues: view('float32', ROWS),
      islandCount: view('uint32', 1),
      status: {overflow: view('uint32', 1)},
      classes: statistic === 'localGeary' ? view('uint32', ROWS) : undefined,
      quadrants: statistic === 'localBivariateMoran' ? view('uint32', ROWS) : undefined
    }
  });
  return {nodes: contributor.getCommandNodes(graph), device};
}

describe('GPULocalPermutationTest at core limits', () => {
  for (const statistic of LOCAL_STATISTICS) {
    it(`${statistic} at 64 neighbors, 999 permutations and every output`, () => {
      const {nodes, device} = buildLocal(statistic, 999);
      expectCoreLimits(nodes, device);
    });
  }

  for (const joinCase of ['BJC', 'CLC'] as const) {
    it(`localBivariateJoinCount ${joinCase}`, () => {
      const {nodes, device} = buildLocal('localBivariateJoinCount', 999, {joinCase});
      expectCoreLimits(nodes, device);
    });
  }

  for (const multipleTesting of [
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    it(`localMoran with ${multipleTesting}`, () => {
      const {nodes, device} = buildLocal('localMoran', 999, {multipleTesting});
      expectCoreLimits(nodes, device);
    });
  }

  it('documented maximum of 2^20 permutations', () => {
    for (const statistic of LOCAL_STATISTICS) {
      const {nodes, device} = buildLocal(statistic, 2 ** 20);
      expectCoreLimits(nodes, device);
    }
  });
});

const GLOBAL_STATISTICS: GPUGlobalPermutationStatistic[] = [
  'moran',
  'geary',
  'getisOrdG',
  'bivariateMoran',
  'leesL'
];

function buildGlobal(
  statistic: GPUGlobalPermutationStatistic,
  maximumPermutations: number,
  islands: 'keep' | 'exclude' = 'keep'
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `limits-global-${serial++}`});
  const {view, common} = createCommon(graph, maximumPermutations);
  const needsSecond = statistic === 'bivariateMoran' || statistic === 'leesL';
  const contributor = new GPUGlobalPermutationTest({
    ...common,
    id: `global-${statistic}`,
    statistic,
    secondValues: needsSecond ? common.secondValues : undefined,
    alternative: 'two-sided',
    islands,
    output: {
      summary: view('float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length),
      referenceDistribution: view('float32', maximumPermutations),
      histogram: view('uint32', 64)
    }
  });
  return {nodes: contributor.getCommandNodes(graph), device};
}

describe('GPUGlobalPermutationTest at core limits', () => {
  for (const statistic of GLOBAL_STATISTICS) {
    it(`${statistic} at 999 permutations and every output`, () => {
      const {nodes, device} = buildGlobal(statistic, 999);
      expectCoreLimits(nodes, device);
    });
    it(`${statistic} at the documented maximum of 2^20 permutations`, () => {
      const {nodes, device} = buildGlobal(statistic, 2 ** 20, 'exclude');
      expectCoreLimits(nodes, device);
    });
  }
});
