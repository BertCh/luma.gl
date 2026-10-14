// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPU_WARD_ROUND_COUNT_STATUS_WORD,
  GPU_WARD_STATUS,
  GPUWardRegions
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {findDisconnectedLabels} from './large-lattice-invariants';
import {canonicalizePartition, evaluatePartition} from './spatial-regionalization-oracle';
import {
  createCompleteAdjacency,
  createLatticeAdjacency,
  getCSR,
  runWardReference,
  WARD_FIXTURE
} from './ward-reference';

const ROWS = 6;
const CAPACITY = 16;

type NodeRecord = {id: string; condition?: {mode: string}};

function buildWard(maximumIterations: number, statusLength: number = GPU_WARD_STATUS.length) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `ward-mutual-${serial++}`, format, length);
  return new GPUWardRegions({
    id: 'ward',
    weights: {
      offsets: view('uint32', ROWS + 1),
      neighbors: view('uint32', CAPACITY),
      weights: view('float32', CAPACITY)
    },
    values: view('float32', ROWS * 2),
    columnCount: 2,
    targetRegionCount: 2,
    maximumIterations,
    mergeSchedule: 'mutual-nearest',
    parameters: view('uint32', 1),
    output: {
      labels: view('uint32', ROWS),
      objective: view('float32', 1),
      status: view('uint32', statusLength)
    }
  }).getCommandNodes(graph) as readonly NodeRecord[];
}

it('GPUWardRegions mutual-nearest keeps one pipeline per kernel at any round count (rule 12)', () => {
  expect(getGPUCommandNodePipelineKeys(buildWard(4)).size).toBe(
    getGPUCommandNodePipelineKeys(buildWard(3)).size
  );
  // The round budget is an override of the finalize kernel, so it is not WGSL text either.
  const sources = (rounds: number) =>
    new Set(getGPUCommandNodeKernelSources(buildWard(rounds)).map(kernel => kernel.source));
  expect(sources(4)).toEqual(sources(3));
  // The per-round kernel count is constant: 10 kernels per round.
  expect(
    getGPUCommandNodeKernelSources(buildWard(4)).length -
      getGPUCommandNodeKernelSources(buildWard(3)).length
  ).toBe(10);
});

it('GPUWardRegions mutual-nearest gates its rounds (rule 13) and stays within core limits', () => {
  const nodes = buildWard(3);
  const ungated = new Set(['finish']);
  for (const stage of [
    'reset-nearest',
    'nearest-candidates',
    'nearest-cost',
    'nearest-key',
    'nearest-region',
    'select-pairs',
    'merge-regions',
    'relabel',
    'sum-merges',
    'finish'
  ]) {
    const node = nodes.find(entry => entry.id === `ward-round-1-${stage}`);
    expect(node, stage).toBeDefined();
    // The finish kernel writes the gate, so it stays ungated.
    expect(node?.condition?.mode, stage).toBe(ungated.has(stage) ? undefined : 'indirect');
  }
  for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(getPrivateMemoryBytes(source), id).toBeLessThanOrEqual(CORE_PRIVATE_MEMORY_LIMIT_BYTES);
    expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
});

it('GPUWardRegions mutual-nearest writes the optional round-count status word only when present', () => {
  expect(GPU_WARD_ROUND_COUNT_STATUS_WORD).toBe(GPU_WARD_STATUS.length);
  const initialize = (statusLength: number) =>
    getGPUCommandNodeKernelSources(buildWard(2, statusLength)).find(kernel =>
      kernel.id.endsWith('initialize-status')
    )!.source;
  expect(initialize(5)).toContain('status[statusOffset + 4u] = 0u;');
  expect(initialize(4)).not.toContain('status[statusOffset + 4u]');
});

it('GPUWardRegions rejects an unknown mergeSchedule', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `ward-schedule-${serial++}`, format, length);
  expect(
    () =>
      new GPUWardRegions({
        weights: {
          offsets: view('uint32', ROWS + 1),
          neighbors: view('uint32', CAPACITY),
          weights: view('float32', CAPACITY)
        },
        values: view('float32', ROWS),
        targetRegionCount: 2,
        maximumIterations: 2,
        mergeSchedule: 'parallel' as 'sequential',
        parameters: view('uint32', 1),
        output: {
          labels: view('uint32', ROWS),
          objective: view('float32', 1),
          status: view('uint32', 4)
        }
      })
  ).toThrow(/mergeSchedule/);
});

it('the CPU mirrors equal scipy on the separated complete-adjacency scene for both schedules', () => {
  const scene = WARD_FIXTURE.separated;
  const adjacency = createCompleteAdjacency(scene.rowCount);
  for (const cut of scene.cuts) {
    for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
      const result = runWardReference({
        values: scene.values,
        columnCount: scene.columnCount,
        adjacency,
        targetRegionCount: cut.targetRegionCount,
        mergeSchedule
      });
      expect(result.labels, `${mergeSchedule} k=${cut.targetRegionCount}`).toEqual(cut.labels);
      expect(result.objective).toBeCloseTo(cut.objective, 8);
      expect(result.mergeCount).toBe(scene.rowCount - cut.targetRegionCount);
    }
  }
  // 27 sequential merges need 27 rounds; the mutual schedule halves the regions per round.
  const rounds = runWardReference({
    values: scene.values,
    columnCount: scene.columnCount,
    adjacency,
    targetRegionCount: 1,
    mergeSchedule: 'mutual-nearest'
  }).roundCount;
  expect(rounds).toBeLessThanOrEqual(2 * Math.ceil(Math.log2(scene.rowCount)));
});

it('the sequential CPU mirror equals scipy (random complete) and scikit-learn (constrained lattice)', () => {
  const random = WARD_FIXTURE.random;
  const lattice = WARD_FIXTURE.lattice;
  for (const [scene, adjacency] of [
    [random, createCompleteAdjacency(random.rowCount)],
    [lattice, createLatticeAdjacency(lattice.width!, lattice.width!)]
  ] as const) {
    const cut = scene.cuts[0];
    const result = runWardReference({
      values: scene.values,
      columnCount: scene.columnCount,
      adjacency,
      targetRegionCount: cut.targetRegionCount,
      mergeSchedule: 'sequential'
    });
    expect(result.labels).toEqual(cut.labels);
    expect(Math.abs(result.objective - cut.objective)).toBeLessThan(1e-8 * cut.objective);
  }
});

it('mutual-nearest on the constrained lattice is a contiguous partition with an exact objective', () => {
  const scene = WARD_FIXTURE.lattice;
  const adjacency = createLatticeAdjacency(scene.width!, scene.width!);
  const {offsets, neighbors} = getCSR(adjacency);
  const targetRegionCount = scene.cuts[0].targetRegionCount;
  const result = runWardReference({
    values: scene.values,
    columnCount: scene.columnCount,
    adjacency,
    targetRegionCount,
    mergeSchedule: 'mutual-nearest'
  });
  expect(new Set(result.labels).size).toBe(targetRegionCount);
  expect(findDisconnectedLabels(offsets, neighbors, result.labels)).toEqual([]);
  const exact = evaluatePartition(scene.values, scene.columnCount, result.labels).withinSsd;
  expect(Math.abs(result.objective - exact)).toBeLessThan(1e-9 * exact);
  expect(result.roundCount).toBeLessThan(result.mergeCount / 3);
});

it('documents a complete-adjacency case where the schedules differ at a non-natural cut', () => {
  // Ward's linkage is reducible on a complete graph, so both schedules build the same dendrogram;
  // but a round also merges costly reciprocal pairs before cheaper pairs that only exist after
  // other merges of the same round, so a cut that is not natural can differ. Search a fixed
  // pseudo-random family for the first difference; the search itself documents that it exists.
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  let found = false;
  for (let trial = 0; trial < 400 && !found; trial++) {
    const rowCount = 12;
    const values = Array.from({length: rowCount}, () => random() * 10);
    const adjacency = createCompleteAdjacency(rowCount);
    for (const targetRegionCount of [2, 3, 4, 5, 6, 7]) {
      const sequential = runWardReference({
        values,
        columnCount: 1,
        adjacency,
        targetRegionCount,
        mergeSchedule: 'sequential'
      });
      const mutual = runWardReference({
        values,
        columnCount: 1,
        adjacency,
        targetRegionCount,
        mergeSchedule: 'mutual-nearest'
      });
      // Both stay valid partitions of the same size, each with its exact objective.
      expect(new Set(mutual.labels).size).toBe(targetRegionCount);
      expect(mutual.objective).toBeCloseTo(
        evaluatePartition(values, 1, mutual.labels).withinSsd,
        8
      );
      if (
        canonicalizePartition(sequential.labels).join() !==
        canonicalizePartition(mutual.labels).join()
      ) {
        found = true;
        break;
      }
    }
  }
  expect(found).toBe(true);
});
