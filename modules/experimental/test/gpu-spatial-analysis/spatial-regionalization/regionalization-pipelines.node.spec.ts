// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  getGPUAZPMaximumTableWordCount,
  GPU_MAX_P_STATUS,
  GPUAZPRegions,
  GPUMaxPRegions,
  GPUSkaterRegions,
  GPUWardRegions
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 6;
const CAPACITY = 16;

type NodeRecord = {id: string; condition?: {mode: string}};

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `pipelines-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', ROWS + 1),
    neighbors: view('uint32', CAPACITY),
    weights: view('float32', CAPACITY)
  };
  return {graph, view, weights};
}

function buildSkater(maximumRegionCount: number): readonly NodeRecord[] {
  const {graph, view, weights} = setup();
  return new GPUSkaterRegions({
    id: 'skater',
    weights,
    treeEdgeFlags: view('uint32', CAPACITY),
    componentLabels: view('uint32', ROWS),
    values: view('float32', ROWS * 2),
    columnCount: 2,
    maximumRegionCount,
    parameters: view('uint32', 2),
    output: {
      labels: view('uint32', ROWS),
      regionCount: view('uint32', 1),
      cutEdges: view('uint32', maximumRegionCount - 1),
      cutGains: view('float32', maximumRegionCount - 1)
    }
  }).getCommandNodes(graph) as readonly NodeRecord[];
}

it('GPUSkaterRegions: one pipeline per kernel at any step count (rule 12), gated steps (rule 13)', () => {
  expect(getGPUCommandNodePipelineKeys(buildSkater(5)).size).toBe(
    getGPUCommandNodePipelineKeys(buildSkater(4)).size
  );
  const nodes = buildSkater(4);
  for (const stage of ['candidates', 'best-gain', 'best-edge', 'relabel', 'update-ancestors']) {
    expect(nodes.find(node => node.id === `skater-step-1-${stage}`)?.condition?.mode, stage).toBe(
      'indirect'
    );
  }
  // The one-thread kernels write the cursor and the gate, so they stay ungated.
  for (const stage of ['reset-best', 'apply', 'deactivate']) {
    expect(
      nodes.find(node => node.id === `skater-step-1-${stage}`)?.condition,
      stage
    ).toBeUndefined();
  }
  for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(() => new WgslReflect(source), id).not.toThrow();
  }
});

it('GPUAZPRegions, GPUMaxPRegions and GPUWardRegions keep one pipeline per kernel across iterations (rule 12)', () => {
  const azp = (maximumIterations: number) => {
    const {graph, view, weights} = setup();
    return new GPUAZPRegions({
      weights,
      values: view('float32', ROWS),
      initialLabels: view('uint32', ROWS),
      regionCapacity: 2,
      maximumIterations,
      parameters: view('uint32', 1),
      output: {
        labels: view('uint32', ROWS),
        objective: view('float32', 1),
        status: view('uint32', 4)
      }
    }).getCommandNodes(graph);
  };
  expect(getGPUCommandNodePipelineKeys(azp(3)).size).toBe(
    getGPUCommandNodePipelineKeys(azp(2)).size
  );
  const maxP = (maximumIterations: number, maximumGrowthSteps: number = 3) => {
    const {graph, view, weights} = setup();
    return new GPUMaxPRegions({
      weights,
      values: view('float32', ROWS),
      extensiveValues: view('float32', ROWS),
      threshold: view('float32', 1),
      parameters: view('uint32', 1),
      maximumIterations,
      maximumGrowthSteps,
      output: {
        labels: view('uint32', ROWS),
        objective: view('float32', 1),
        status: view('uint32', GPU_MAX_P_STATUS.length)
      }
    }).getCommandNodes(graph);
  };
  expect(getGPUCommandNodePipelineKeys(maxP(3)).size).toBe(
    getGPUCommandNodePipelineKeys(maxP(2)).size
  );
  // N and N + 1 construction rounds share every pipeline (rule 12).
  expect(getGPUCommandNodePipelineKeys(maxP(0, 4)).size).toBe(
    getGPUCommandNodePipelineKeys(maxP(0, 3)).size
  );
  const ward = (maximumIterations: number) => {
    const {graph, view, weights} = setup();
    return new GPUWardRegions({
      weights,
      values: view('float32', ROWS),
      targetRegionCount: 2,
      maximumIterations,
      parameters: view('uint32', 1),
      output: {
        labels: view('uint32', ROWS),
        objective: view('float32', 1),
        status: view('uint32', 4)
      }
    }).getCommandNodes(graph);
  };
  expect(getGPUCommandNodePipelineKeys(ward(3)).size).toBe(
    getGPUCommandNodePipelineKeys(ward(2)).size
  );
});

it('GPUAZPRegions gates its iterations (rule 13) and parses as WGSL', () => {
  const {graph, view, weights} = setup();
  const nodes = new GPUAZPRegions({
    id: 'azp',
    weights,
    values: view('float32', ROWS),
    initialLabels: view('uint32', ROWS),
    regionCapacity: 2,
    maximumIterations: 3,
    parameters: view('uint32', 1),
    output: {labels: view('uint32', ROWS), objective: view('float32', 1), status: view('uint32', 4)}
  }).getCommandNodes(graph) as readonly NodeRecord[];
  expect(nodes.find(node => node.id === 'azp-iteration-1-candidates')?.condition?.mode).toBe(
    'indirect'
  );
  for (const stage of ['claim-gain', 'claim-key', 'select', 'apply']) {
    expect(nodes.find(node => node.id === `azp-iteration-1-${stage}`)?.condition?.mode, stage).toBe(
      'indirect'
    );
  }
  // The contiguity check is multi-workgroup and gated by its own dispatch gate (rule 13).
  for (const stage of ['hook', 'apply', 'jump-0']) {
    expect(
      nodes.find(node => node.id === `azp-contiguity-1-${stage}`)?.condition?.mode,
      stage
    ).toBe('indirect');
  }
  expect(nodes.find(node => node.id === 'azp-contiguity-1-finish')?.condition).toBeUndefined();
  expect(nodes.find(node => node.id === 'azp-initial-contiguity')?.condition).toBeUndefined();
  // The round-finish kernel writes the gate, so it stays ungated.
  expect(nodes.find(node => node.id === 'azp-iteration-1-finish')?.condition).toBeUndefined();
  for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(() => new WgslReflect(source), id).not.toThrow();
  }
});

it('GPUMaxPRegions has no row cap: 262,144 rows build, and scratch beyond one binding is rejected', () => {
  const build = (rows: number, columnCount: number, startCount: number) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `max-p-cap-${serial++}`, format, length);
    return new GPUMaxPRegions({
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows),
        information: {rowCount: rows, transform: 'raw', symmetric: true}
      },
      values: view('float32', rows * columnCount),
      columnCount,
      extensiveValues: view('float32', rows),
      threshold: view('float32', 1),
      startCount,
      maximumGrowthSteps: 2,
      parameters: view('uint32', 1),
      output: {
        labels: view('uint32', rows),
        objective: view('float32', 1),
        status: view('uint32', GPU_MAX_P_STATUS.length)
      }
    }).getCommandNodes(graph);
  };
  // 8 times the former cap, with the default 16 starts and one column.
  const nodes = build(262144, 1, 16);
  expect(nodes.length).toBeGreaterThan(10);
  for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
    expect(getPrivateMemoryBytes(kernel.source), kernel.id).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
  }
  // The region table (columnCount + 3 words per start and row) must fit one 128 MiB binding:
  // 16 starts x 262,144 rows x 11 words x 4 bytes is 184 MB.
  expect(() => build(262144, 8, 16)).toThrow(/scratch exceeds one storage binding/);
  expect(() => build(262144, 8, 11)).not.toThrow();
});

it('GPUAZPRegions has no row cap and keeps private memory independent of the row count (rule 16)', () => {
  const device = createNullWebGPUDevice();
  const maximumTableWordCount = getGPUAZPMaximumTableWordCount(device);
  const build = (rows: number, columnCount: number, regionCapacity: number) => {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `private-${serial++}`, format, length);
    return new GPUAZPRegions({
      id: 'azp-cap',
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows),
        information: {rowCount: rows, transform: 'raw', symmetric: true}
      },
      values: view('float32', rows * columnCount),
      columnCount,
      initialLabels: view('uint32', rows),
      regionCapacity,
      maximumIterations: 1,
      parameters: view('uint32', 1),
      output: {
        labels: view('uint32', rows),
        objective: view('float32', 1),
        status: view('uint32', 4)
      }
    }).getCommandNodes(graph);
  };
  // 262,144 rows (8 times the former cap) and the widest table that fits one binding.
  const sizes = [262144, Math.floor(maximumTableWordCount / 8)];
  const privateBytes = sizes.map(rows =>
    getGPUCommandNodeKernelSources(build(rows, 8, 64)).map(kernel => {
      expect(kernel.source, kernel.id).not.toMatch(/var<private> reached:/);
      expect(getPrivateMemoryBytes(kernel.source), kernel.id).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      return getPrivateMemoryBytes(kernel.source);
    })
  );
  // The kernel list grows with the row count (reduction levels), the private memory does not.
  expect(Math.max(...privateBytes[1])).toBe(Math.max(...privateBytes[0]));
  const candidates = getGPUCommandNodeKernelSources(build(262144, 8, 64)).find(({id}) =>
    id.endsWith('-iteration-0-candidates')
  );
  // Neighborhood rows, hash table, stack, visited bits and the count: 2,068 bytes.
  expect(candidates && getPrivateMemoryBytes(candidates.source)).toBe(2068);
  // One word past the binding limit is refused with the documented maximum.
  expect(() => build(Math.floor(maximumTableWordCount / 8) + 1, 8, 64)).toThrow(
    new RegExp(`must be at most ${maximumTableWordCount}$`)
  );
});
