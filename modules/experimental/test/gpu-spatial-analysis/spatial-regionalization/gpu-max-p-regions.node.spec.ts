// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPU_MAX_P_STATUS,
  GPUMaxPRegions
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {getGPUMaxPRoundCounts} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
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
import {
  createRandomValues,
  createRookLattice,
  findDisconnectedLabels
} from './large-lattice-invariants';
import {
  constructMaxPStartMirror,
  createSymmetricAdjacencyMirror,
  getExpectedRegionSizeMirror,
  getRegionSums,
  MAX_P_TIE_INVERSE,
  MAX_P_TIE_MULTIPLIER,
  runMaxPRegionsMirror
} from './max-p-construction-mirror';
import {canonicalizePartition} from './spatial-regionalization-oracle';

const NONE = 0xffffffff;

type NodeRecord = {id: string; condition?: {mode: string}};

function buildNodes(
  rows: number,
  options: {startCount?: number; maximumGrowthSteps?: number; maximumIterations?: number} = {}
): readonly NodeRecord[] {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `max-p-${serial++}`, format, length);
  return new GPUMaxPRegions({
    id: 'max-p',
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', 4 * rows),
      weights: view('float32', 4 * rows)
    },
    values: view('float32', rows * 2),
    columnCount: 2,
    extensiveValues: view('float32', rows),
    threshold: view('float32', 1),
    parameters: view('uint32', 1),
    ...options,
    output: {
      labels: view('uint32', rows),
      objective: view('float32', 1),
      status: view('uint32', GPU_MAX_P_STATUS.length)
    }
  }).getCommandNodes(graph) as readonly NodeRecord[];
}

it('GPUMaxPRegions builds the same pipelines at N and N + 1 rounds (rule 12)', () => {
  // The round index never reaches WGSL text or constants: it is a GPU state word.
  expect(getGPUCommandNodePipelineKeys(buildNodes(20, {maximumGrowthSteps: 5})).size).toBe(
    getGPUCommandNodePipelineKeys(buildNodes(20, {maximumGrowthSteps: 4})).size
  );
  const four = buildNodes(20, {maximumGrowthSteps: 4});
  const five = buildNodes(20, {maximumGrowthSteps: 5});
  // Nine nodes per growth round (control, reset, seed mark and commit, two propose, two claim,
  // commit) and three per enclave round.
  expect(five.length - four.length).toBe(9 + 3);
});

it('GPUMaxPRegions gates its rounds (rule 13); the control nodes that write the gate stay ungated', () => {
  const nodes = buildNodes(20, {maximumGrowthSteps: 3});
  const find = (id: string) => nodes.find(node => node.id === id);
  for (const stage of [
    'reset',
    'seed-mark',
    'seed-commit',
    'propose-1',
    'propose-2',
    'claim-1',
    'claim-2',
    'commit'
  ]) {
    expect(find(`max-p-round-1-${stage}`)?.condition?.mode, stage).toBe('indirect');
  }
  for (const stage of ['propose', 'commit']) {
    expect(find(`max-p-enclave-1-${stage}`)?.condition?.mode, stage).toBe('indirect');
  }
  expect(find('max-p-round-1-control')?.condition).toBeUndefined();
  expect(find('max-p-enclave-1-control')?.condition).toBeUndefined();
});

it('GPUMaxPRegions kernels parse as WGSL and fit the core device limits', () => {
  for (const options of [{maximumGrowthSteps: 2}, {maximumGrowthSteps: 2, startCount: 64}]) {
    const nodes = buildNodes(20, options);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(kernels.length).toBeGreaterThan(10);
    for (const {id, source} of kernels) {
      expect(() => new WgslReflect(source), id).not.toThrow();
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source), id).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
      expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
        CORE_WORKGROUP_STORAGE_LIMIT_BYTES
      );
    }
  }
});

it('GPUMaxPRegions round counts and tie order match the mirror', () => {
  expect(getGPUMaxPRoundCounts(1000, 256)).toEqual({growthRounds: 256, enclaveRounds: 256});
  expect(getGPUMaxPRoundCounts(3, 256)).toEqual({growthRounds: 8, enclaveRounds: 4});
  expect(getGPUMaxPRoundCounts(1000, 7)).toEqual({growthRounds: 7, enclaveRounds: 7});
  // The tie order `(x * multiplier) ^ salt` is a bijection: the inverse recovers every value.
  for (const value of [0, 1, 2, 12345, 262143, 0x7fffffff, 0xfffffffe]) {
    const tie = (Math.imul(value, MAX_P_TIE_MULTIPLIER) ^ 0x5bd1e995) >>> 0;
    expect(Math.imul((tie ^ 0x5bd1e995) >>> 0, MAX_P_TIE_INVERSE) >>> 0).toBe(value);
  }
  // Expected region size: ceil(threshold / mean extensive).
  expect(getExpectedRegionSizeMirror(Float32Array.from([1, 2, 3, 4]), 10)).toBe(4);
  expect(getExpectedRegionSizeMirror(Float32Array.from([2, 2, 2, 2]), 0)).toBe(1);
  expect(getExpectedRegionSizeMirror(Float32Array.from([0, 0, 0]), 5)).toBe(1);
});

it('the max-p mirror splits a path into the two natural regions', () => {
  // Four rows on a path: values 0, 1, 10, 11 and a threshold of two rows.
  const result = runMaxPRegionsMirror({
    offsets: [0, 1, 3, 5, 6],
    neighbors: [1, 0, 2, 1, 3, 2],
    values: [0, 1, 10, 11],
    extensive: [1, 1, 1, 1],
    threshold: 2,
    standardize: false
  });
  expect(canonicalizePartition(Array.from(result.labels))).toEqual([0, 0, 1, 1]);
  expect(result.regionCount).toBe(2);
  expect(result.infeasibleRegionCount).toBe(0);
});

function checkPartition(
  lattice: {offsets: Uint32Array; neighbors: Uint32Array},
  extensive: ArrayLike<number>,
  threshold: number,
  result: ReturnType<typeof runMaxPRegionsMirror>
) {
  const {labels} = result;
  // Dense labels, every row labelled.
  const regions = new Set(labels);
  expect(regions.size).toBe(result.regionCount);
  expect(Math.max(...labels)).toBe(result.regionCount - 1);
  expect(labels.every(label => label !== NONE)).toBe(true);
  // Contiguity and the threshold.
  expect(findDisconnectedLabels(lattice.offsets, lattice.neighbors, labels)).toEqual([]);
  const sums = getRegionSums(labels, extensive);
  let belowThreshold = 0;
  for (const sum of sums.values()) if (sum < threshold) belowThreshold++;
  expect(belowThreshold).toBe(result.infeasibleRegionCount);
}

it('the max-p mirror builds contiguous regions that all reach the threshold on rook lattices', () => {
  for (const [width, threshold, seed] of [
    [12, 12, 0],
    [20, 30, 3],
    [30, 12, 7],
    [30, 60, 1]
  ]) {
    const lattice = createRookLattice(width, width);
    const rowCount = width * width;
    const extensive = Float32Array.from(createRandomValues(rowCount, 202 + seed, 5), value =>
      Math.floor(value + 1)
    );
    const result = runMaxPRegionsMirror({
      ...lattice,
      values: createRandomValues(rowCount, 101 + seed),
      extensive,
      threshold,
      seed,
      startCount: 4
    });
    checkPartition(lattice, extensive, threshold, result);
    expect(result.infeasibleRegionCount, `${width}x${width} floor ${threshold}`).toBe(0);
    expect(result.regionCount).toBeGreaterThan(1);
    // Deterministic for a given seed.
    const again = runMaxPRegionsMirror({
      ...lattice,
      values: createRandomValues(rowCount, 101 + seed),
      extensive,
      threshold,
      seed,
      startCount: 4
    });
    expect(Array.from(again.labels)).toEqual(Array.from(result.labels));
  }
});

it('the max-p mirror reports an island below the threshold and keeps every row labelled', () => {
  // A 6-row path plus a separate pair whose extensive sum is below the threshold.
  const offsets = [0, 1, 3, 5, 7, 9, 10, 11, 12];
  const neighbors = [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 7, 6];
  const result = runMaxPRegionsMirror({
    offsets,
    neighbors,
    values: [0, 0.1, 0.2, 5, 5.1, 5.2, 9, 9.1],
    extensive: [2, 2, 2, 2, 2, 2, 1, 1],
    threshold: 4,
    standardize: false,
    startCount: 3
  });
  expect(result.labels.every(label => label < result.regionCount)).toBe(true);
  // Rows 6 and 7 cannot reach 4: one under-threshold region, the path splits in whole regions.
  expect(result.infeasibleRegionCount).toBe(1);
  expect(result.labels[6]).toBe(result.labels[7]);
  expect(result.labels[6]).not.toBe(result.labels[0]);
});

it('the max-p mirror degrades to reported under-threshold regions when the rounds run out', () => {
  const lattice = createRookLattice(10, 10);
  const extensive = Float32Array.from({length: 100}, () => 1);
  // Regions need 30 rows but only 5 rounds exist: nothing completes.
  const starved = runMaxPRegionsMirror({
    ...lattice,
    values: createRandomValues(100, 5),
    extensive,
    threshold: 30,
    maximumGrowthSteps: 5,
    startCount: 2
  });
  expect(starved.infeasibleRegionCount).toBeGreaterThan(0);
  expect(starved.labels.every(label => label < starved.regionCount)).toBe(true);
  // With enough rounds the same input completes.
  const roomy = runMaxPRegionsMirror({
    ...lattice,
    values: createRandomValues(100, 5),
    extensive,
    threshold: 30,
    maximumGrowthSteps: 256,
    startCount: 2
  });
  expect(roomy.infeasibleRegionCount).toBe(0);
});

it('the max-p mirror selects the start by fewest under-threshold regions, then most regions', () => {
  const lattice = createRookLattice(15, 15);
  const adjacency = createSymmetricAdjacencyMirror(lattice.offsets, lattice.neighbors);
  const extensive = Float32Array.from(createRandomValues(225, 9, 5), value =>
    Math.floor(value + 1)
  );
  const props = {
    offsets: adjacency.offsets,
    neighbors: adjacency.neighbors,
    values: Float32Array.from(createRandomValues(225, 4)),
    columnCount: 1,
    extensive,
    threshold: 15,
    seed: 2,
    startCount: 8,
    maximumGrowthSteps: 256
  };
  const result = runMaxPRegionsMirror({
    ...lattice,
    values: props.values,
    extensive,
    threshold: 15,
    seed: 2,
    startCount: 8
  });
  const starts = Array.from({length: 8}, (_, start) => constructMaxPStartMirror(props, start));
  const best = starts[result.selectedStart];
  for (const start of starts) {
    expect(start.infeasibleRegionCount).toBeGreaterThanOrEqual(best.infeasibleRegionCount);
    if (start.infeasibleRegionCount === best.infeasibleRegionCount) {
      expect(start.regionCount).toBeLessThanOrEqual(best.regionCount);
    }
  }
  // Starts draw independent seed orders: the region counts differ across starts.
  expect(new Set(starts.map(start => start.regionCount)).size).toBeGreaterThan(1);
});

it('Max-P materializes means before distance ranking to prevent reciprocal FMA contraction', () => {
  // Metal can contract x - sum / n to fma(-sum, reciprocal(n), x). Even integer inputs
  // then lose exact zero distances: reciprocal(3) = 0.3333333432674408 in f32.
  const reciprocal = Math.fround(1 / 3);
  const contractedDifference = Math.fround(1 - 3 * reciprocal);
  const storedMean = Math.fround(3 * reciprocal);
  expect(contractedDifference).toBe(-(2 ** -25));
  expect(Math.fround(1 - storedMean)).toBe(0);
  const contractedKey =
    new Uint32Array(
      Float32Array.of(Math.fround(contractedDifference * contractedDifference)).buffer
    )[0] >>> 6;
  expect(contractedKey).toBe(10092544);
  const sources = getGPUCommandNodeKernelSources(buildNodes(20, {maximumGrowthSteps: 2}));
  const commit = sources.find(kernel => kernel.id === 'max-p-round-0-commit')!.source;
  expect(commit).toContain('regionSums[sumIndex] = sum;');
  expect(commit).toContain(
    'regionData[regionBase + REGION_HEADER + column] = sum / (count + 1.0);'
  );
  for (const {source} of sources.filter(kernel => kernel.source.includes('fn getDistance'))) {
    expect(source).toContain('let mean = regionData[regionBase + REGION_HEADER + column];');
    expect(source).not.toContain('REGION_HEADER + column] / count');
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }
});
