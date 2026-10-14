// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPUKnoxTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-knox-test';
import {GPUMantelTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test';
import {
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {
  getSpaceTimeStoppingBatchSize,
  SPACE_TIME_MAXIMUM_STOPPING_BATCHES,
  SPACE_TIME_STOPPING_BATCH_SIZE
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-stopping';
import {
  countDistinctPipelines,
  createNullWebGPUDevice,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getGPUCommandNodeKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes,
  recordComputePipelines
} from '../../utils/gpu-contributor-test-utils';
import {computeSequentialStop, computeKnoxOracle, computeMantelOracle} from './space-time-oracle';
import {buildBand, createStoppingEvents} from './space-time-scenes';

const ROWS = 300;
let serial = 0;

type Options = {stopAfterExceedances?: number; permutationCount?: boolean};

function build(kind: 'knox' | 'mantel', maximumPermutations: number, options: Options = {}) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `space-time-stop-${serial++}`, format, length);
  const common = {
    pairs: {
      offsets: view('uint32', ROWS + 1),
      neighbors: view('uint32', 4 * ROWS),
      weights: view('float32', 4 * ROWS),
      distances: view('float32', 4 * ROWS)
    },
    times: view('float32', ROWS),
    parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
    maximumPermutations,
    ...(options.stopAfterExceedances === undefined
      ? {}
      : {stopAfterExceedances: options.stopAfterExceedances})
  };
  const summary = view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  const permutationCount = options.permutationCount ? view('uint32', 1) : undefined;
  const create = () =>
    kind === 'knox'
      ? new GPUKnoxTest({
          ...common,
          output: {statistics: view('uint32', maximumPermutations + 1), summary, permutationCount}
        })
      : new GPUMantelTest({
          ...common,
          output: {statistics: view('float32', maximumPermutations + 1), summary, permutationCount}
        });
  return {device, graph, create};
}

function getNodes(kind: 'knox' | 'mantel', maximumPermutations: number, options: Options) {
  const {device, graph, create} = build(kind, maximumPermutations, options);
  const nodes = create().getCommandNodes(graph);
  return {nodes, device};
}

for (const kind of ['knox', 'mantel'] as const) {
  describe(`${kind} stopAfterExceedances`, () => {
    it('validates the exceedance count and the permutationCount output', () => {
      for (const stopAfterExceedances of [0, 1.5, -1, 100, Number.NaN]) {
        expect(() =>
          build(kind, 99, {stopAfterExceedances, permutationCount: true}).create()
        ).toThrow(/stopAfterExceedances must be an integer in \[1, 99\]/);
      }
      expect(() => build(kind, 99, {stopAfterExceedances: 5}).create()).toThrow(
        /stopAfterExceedances needs output.permutationCount/
      );
      expect(() => build(kind, 99, {permutationCount: true}).create()).toThrow(
        /output.permutationCount needs stopAfterExceedances/
      );
      expect(() =>
        build(kind, 99, {stopAfterExceedances: 99, permutationCount: true}).create()
      ).not.toThrow();
    });

    it('keeps the off path free of batch nodes and state', () => {
      const {nodes} = getNodes(kind, 99, {});
      const ids = nodes.map(node => node.id);
      expect(ids.some(id => id.includes('-batch-') || id.includes('-stop-'))).toBe(false);
      expect(ids.filter(id => id.endsWith('-partials'))).toHaveLength(1);
    });

    it('builds the same pipelines for N and N + 1 batches and one pipeline per batch kernel', () => {
      const batch = SPACE_TIME_STOPPING_BATCH_SIZE;
      // slotCount = maximumPermutations + 1: 3 batches, then 4.
      const three = getNodes(kind, 3 * batch - 1, {
        stopAfterExceedances: 5,
        permutationCount: true
      });
      const four = getNodes(kind, 3 * batch, {stopAfterExceedances: 5, permutationCount: true});
      expect(four.nodes.length - three.nodes.length).toBe(3);
      expect(countDistinctPipelines(four.nodes)).toBe(countDistinctPipelines(three.nodes));
      // Within one build, every batch shares a module per role (rule 12): the cursor is a GPU word.
      const pipelines = recordComputePipelines(four.nodes);
      for (const role of ['partials', 'reduce', 'scan']) {
        const keys = new Set(
          pipelines
            .filter(({id}) => new RegExp(`-batch-\\d+-${role}$`).test(id))
            .map(({key}) => key)
        );
        expect(keys.size, role).toBe(1);
      }
      for (const {source} of pipelines) {
        expect(source).not.toMatch(/\bcursor\s*=\s*\d+u/);
      }
    });

    it('batch kernels fit the core limits and keep the barrier in uniform flow', () => {
      const {nodes} = getNodes(kind, 200, {stopAfterExceedances: 10, permutationCount: true});
      const sources = getGPUCommandNodeKernelSources(nodes);
      expect(sources.length).toBeGreaterThan(0);
      for (const {id, source} of sources) {
        expect(getStorageBindingCount(source), `${id} bindings`).toBeLessThanOrEqual(8);
        expect(getPrivateMemoryBytes(source), id).toBeLessThanOrEqual(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES
        );
        expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
          CORE_WORKGROUP_STORAGE_LIMIT_BYTES
        );
      }
      const partials = sources.find(({id}) => id.endsWith('-batch-0-partials'))!;
      expect(partials.source).toContain('state[stateOffset] * BATCH_SIZE + localSlot');
      const body = partials.source.slice(partials.source.indexOf('fn main('));
      expect(body).toMatch(/\n {2}workgroupBarrier\(\);/);
      const prologue = body.slice(
        body.indexOf('let workgroup = '),
        body.indexOf('workgroupBarrier()')
      );
      expect(prologue).not.toContain('return');
      // The scan is the only writer of the cursor and the dispatch gate.
      const scan = sources.find(({id}) => id.endsWith('-batch-0-scan'))!;
      expect(scan.source).toContain('STOP_AFTER: u32 = 10u');
      expect(scan.source).toContain('exceedances == STOP_AFTER');
      expect(sources.find(({id}) => id.endsWith('-stop-reset'))).toBeDefined();
      expect(sources.find(({id}) => id.endsWith('-stop-finalize'))).toBeDefined();
    });
  });
}

describe('stopping batch size', () => {
  it('is 64 up to 256 batches, then doubles', () => {
    expect(getSpaceTimeStoppingBatchSize(1000)).toBe(64);
    expect(getSpaceTimeStoppingBatchSize(64 * SPACE_TIME_MAXIMUM_STOPPING_BATCHES)).toBe(64);
    expect(getSpaceTimeStoppingBatchSize(64 * SPACE_TIME_MAXIMUM_STOPPING_BATCHES + 1)).toBe(128);
    expect(getSpaceTimeStoppingBatchSize(65537)).toBe(512);
  });
});

// The sequential rule: stops at the first L with e(L) = h and reports min(1, h / L); with fewer
// than h exceedances it is the usual (e + 1) / (P + 1).
describe('sequential stopping oracle', () => {
  const statistics = [10, 3, 12, 10, 1, 11, 10, 2];
  it('stops at the first index with h exceedances', () => {
    // Exceedances (>= 10) at slots 2, 3, 5, 6.
    expect(computeSequentialStop(statistics, 7, 1)).toEqual({
      stopped: true,
      permutationsUsed: 2,
      greater: 1,
      pValue: 1 / 2
    });
    expect(computeSequentialStop(statistics, 7, 3)).toEqual({
      stopped: true,
      permutationsUsed: 5,
      greater: 3,
      pValue: 3 / 5
    });
  });
  it('falls back to the standard p-value when the rule never fires', () => {
    expect(computeSequentialStop(statistics, 7, 5)).toEqual({
      stopped: false,
      permutationsUsed: 7,
      greater: 4,
      pValue: 5 / 8
    });
    // A per-frame count below the stopping slot ignores later permutations.
    expect(computeSequentialStop(statistics, 4, 3).stopped).toBe(false);
  });
  it('caps the p-value at 1', () => {
    expect(computeSequentialStop([0, 1, 1], 2, 1).pValue).toBe(1);
  });
});

it('the headless stopping fixture exercises early, later-batch and no stops', () => {
  const {positions, times} = createStoppingEvents();
  for (const kind of ['knox', 'mantel'] as const) {
    const pairs = buildBand(positions, kind === 'knox' ? 9 : 14);
    const statistics =
      kind === 'knox'
        ? computeKnoxOracle(pairs, times, 7, 150, 3).statistics
        : computeMantelOracle(pairs, times, 7, 150);
    const lengths = [1, 2, 5, 20, 60, 191].map(
      target => computeSequentialStop(statistics, 150, target).permutationsUsed
    );
    expect(lengths, kind).toEqual(
      kind === 'knox' ? [1, 7, 16, 65, 150, 150] : [1, 2, 5, 22, 65, 150]
    );
  }
});
