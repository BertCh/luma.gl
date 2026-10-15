// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPULocalPermutationReplay} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-replay';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createReplayTest} from './local-permutation-test-fixture';

function createReplay(
  overrides: {nullDistributionLength?: number; rowLength?: number} = {},
  configuration: Parameters<typeof createReplayTest>[1] = {}
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {recording: {}});
  const test = createReplayTest(graph, configuration);
  const testNodes = test.getCommandNodes(graph);
  const replay = new GPULocalPermutationReplay({
    test,
    row: createTransientView(graph, 'row', 'uint32', overrides.rowLength ?? 1),
    output: {
      nullDistribution: createTransientView(
        graph,
        'null-distribution',
        'float32',
        overrides.nullDistributionLength ?? 99
      ),
      replayInfo: createTransientView(graph, 'replay-info', 'float32', 3)
    }
  });
  return {graph, test, testNodes, replay};
}

describe('GPULocalPermutationReplay', () => {
  it('shares the permutation kernel text with the test and writes the null distribution', () => {
    const {graph, testNodes, replay} = createReplay();
    const replayNodes = replay.getCommandNodes(graph);
    expect(replayNodes.map(node => node.id)).toEqual([
      'golden-replay-setup',
      'golden-replay-permute'
    ]);
    const [testSource] = getGPUCommandNodeKernelSources(
      testNodes.filter(node => node.id === 'golden-permute')
    );
    const [, replaySource] = getGPUCommandNodeKernelSources(replayNodes);
    for (const shared of [
      'createPhiloxStream(key, index, permutation, RANDOM_TAG)',
      'nextPhiloxBelow(&stream, others - drawn)',
      'fn readSwap(',
      'let simulated = getLocalTerm(fx, fy, lagA, lagB);'
    ]) {
      expect(testSource.source).toContain(shared);
      expect(replaySource.source).toContain(shared);
    }
    expect(replaySource.source).toContain(
      'nullDistribution[nullDistributionOffset + permutation] = simulated;'
    );
    expect(replaySource.source).not.toContain('exceedances');
    expect(testSource.source).not.toContain('nullDistribution');
    expect(getStorageBindingCount(replaySource.source)).toBeLessThanOrEqual(8);
    graph.device.destroy();
  });

  it('emits WGSL that parses for every statistic', () => {
    for (const statistic of [
      'localMoran',
      'localGeary',
      'localG',
      'localGStar',
      'localBivariateMoran',
      'localLeeL',
      'localJoinCount'
    ] as const) {
      const {graph, replay} = createReplay({}, {statistic});
      const sources = getParsedKernelSources(replay.getCommandNodes(graph));
      expect(sources.size).toBe(2);
      graph.device.destroy();
    }
  });

  it('requires a recording graph and a lowered test', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const test = createReplayTest(graph);
    test.getCommandNodes(graph);
    const replay = new GPULocalPermutationReplay({
      test,
      row: createTransientView(graph, 'row', 'uint32', 1),
      output: {
        nullDistribution: createTransientView(graph, 'null-distribution', 'float32', 99),
        replayInfo: createTransientView(graph, 'replay-info', 'float32', 3)
      }
    });
    expect(() => replay.getCommandNodes(graph)).toThrow();
    graph.device.destroy();
  });

  it('rejects a row view that is not one element', () => {
    expect(() => createReplay({rowLength: 2})).toThrow(/row length/);
  });

  it('rejects a null distribution shorter than the maximum permutation count', () => {
    expect(() => createReplay({nullDistributionLength: 98})).toThrow(/nullDistribution length/);
  });
});
