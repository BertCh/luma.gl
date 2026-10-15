// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Written, not yet run: needs a WebGPU device (`yarn test-headless`/browser project).

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {Buffer, Device} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {GPULocalPermutationReplay} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-replay';
import {
  getGPUPermutationParameterValues,
  GPULocalPermutationTest,
  type GPULocalPermutationStatistic
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const ROWS = 12;
const PERMUTATIONS = 99;

type Alternative = 'two-sided' | 'greater' | 'lesser';

/** Runs the real test and the replay of one row on a 12-row ring lattice (two neighbors per row). */
async function runReplay(
  device: Device,
  options: {
    statistic: GPULocalPermutationStatistic;
    alternative: Alternative;
    row: number;
    stopAfterExceedances?: number;
  }
) {
  const offsets = Uint32Array.from({length: ROWS + 1}, (_, index) => index * 2);
  const neighbors = Uint32Array.from(
    {length: ROWS * 2},
    (_, index) => (Math.floor(index / 2) + (index % 2 === 0 ? ROWS - 1 : 1)) % ROWS
  );
  const weights = new Float32Array(ROWS * 2).fill(0.5);
  const values = Float32Array.from(
    {length: ROWS},
    (_, index) => Math.sin(index * 1.7) * 3 + index / 4
  );
  const parameters = getGPUPermutationParameterValues({
    seed: 11,
    permutationCount: PERMUTATIONS,
    significanceLevel: 0.05
  });
  const buffers = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, weights),
    values: createInputBuffer(device, values),
    parameters: createInputBuffer(device, Uint32Array.from(parameters)),
    exceedances: createOutputBuffer(device, ROWS),
    permutationCounts: createOutputBuffer(device, ROWS),
    pseudoPValues: createOutputBuffer(device, ROWS),
    overflow: createOutputBuffer(device, 1),
    row: createInputBuffer(device, Uint32Array.of(options.row)),
    nullDistribution: createOutputBuffer(device, PERMUTATIONS),
    replayInfo: createOutputBuffer(device, 3)
  };
  const graph = new GPUCommandGraph(device, {id: 'replay-spec', recording: {}});
  const importView = <Format extends 'uint32' | 'float32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);
  const test = new GPULocalPermutationTest({
    id: 'replay-spec-test',
    weights: {
      offsets: importView('offsets', buffers.offsets, 'uint32', ROWS + 1),
      neighbors: importView('neighbors', buffers.neighbors, 'uint32', ROWS * 2),
      weights: importView('weights', buffers.weights, 'float32', ROWS * 2)
    },
    values: importView('values', buffers.values, 'float32', ROWS),
    parameters: importView('parameters', buffers.parameters, 'uint32', parameters.length),
    maximumPermutations: PERMUTATIONS,
    maximumNeighbors: 2,
    statistic: options.statistic,
    alternative: options.alternative,
    stopAfterExceedances: options.stopAfterExceedances,
    output: {
      exceedances: importView('exceedances', buffers.exceedances, 'uint32', ROWS),
      permutationCounts:
        options.stopAfterExceedances === undefined
          ? undefined
          : importView('permutation-counts', buffers.permutationCounts, 'uint32', ROWS),
      pseudoPValues: importView('pseudo-p', buffers.pseudoPValues, 'float32', ROWS),
      status: {overflow: importView('overflow', buffers.overflow, 'uint32', 1)}
    }
  });
  const replay = new GPULocalPermutationReplay({
    test,
    row: importView('row', buffers.row, 'uint32', 1),
    output: {
      nullDistribution: importView(
        'null-distribution',
        buffers.nullDistribution,
        'float32',
        PERMUTATIONS
      ),
      replayInfo: importView('replay-info', buffers.replayInfo, 'float32', 3)
    }
  });
  graph.add(test);
  graph.add(replay);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const exceedances = await readUint32(buffers.exceedances, ROWS);
  const permutationCounts =
    options.stopAfterExceedances === undefined
      ? undefined
      : await readUint32(buffers.permutationCounts, ROWS);
  const nullDistribution = await readFloat32(buffers.nullDistribution, PERMUTATIONS);
  const info = await readFloat32(buffers.replayInfo, 3);
  return {exceedances, permutationCounts, nullDistribution, observed: info[0], count: info[1]};
}

/** The exceedance count of a replayed prefix, by the test's alternative. */
function countExceedances(
  nullDistribution: readonly number[],
  observed: number,
  alternative: Alternative,
  used: number
): number {
  let greater = 0;
  let lesser = 0;
  for (let permutation = 0; permutation < used; permutation++) {
    greater += nullDistribution[permutation] >= observed ? 1 : 0;
    lesser += nullDistribution[permutation] <= observed ? 1 : 0;
  }
  return alternative === 'greater'
    ? greater
    : alternative === 'lesser'
      ? lesser
      : Math.min(greater, lesser);
}

for (const statistic of ['localMoran', 'localGeary'] as const) {
  it(`${statistic} replay reproduces the test's exceedance count for every row`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const row of [0, 5, 11]) {
      const result = await runReplay(device, {statistic, alternative: 'two-sided', row});
      expect(result.count).toBe(PERMUTATIONS);
      expect(
        countExceedances(result.nullDistribution, result.observed, 'two-sided', PERMUTATIONS)
      ).toBe(result.exceedances[row]);
    }
  });
}

it('a sequential-stopping replay prefix matches the stopping index of the test', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const target = 5;
  const row = 3;
  const result = await runReplay(device, {
    statistic: 'localMoran',
    alternative: 'greater',
    row,
    stopAfterExceedances: target
  });
  const used = result.permutationCounts![row];
  // The replay runs all P draws; its first `used` terms are the draws the test consumed.
  const stoppingCount = countExceedances(result.nullDistribution, result.observed, 'greater', used);
  expect(stoppingCount).toBe(result.exceedances[row]);
  if (used < PERMUTATIONS) {
    expect(stoppingCount).toBe(target);
    expect(countExceedances(result.nullDistribution, result.observed, 'greater', used - 1)).toBe(
      target - 1
    );
  }
});
