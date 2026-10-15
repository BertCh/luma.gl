// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createReplayTest} from './local-permutation-test-fixture';

it('preserves the complete pre-refactor permutation WGSL byte for byte across 162 configurations', () => {
  const kernels: Record<string, string> = {};
  for (const statistic of [
    'localMoran',
    'localGeary',
    'localG',
    'localGStar',
    'localBivariateMoran',
    'localLeeL',
    'localJoinCount',
    'localBivariateJoinCount'
  ] as const) {
    for (const joinCase of statistic === 'localBivariateJoinCount'
      ? (['BJC', 'CLC'] as const)
      : [undefined]) {
      for (const alternative of ['two-sided', 'greater', 'lesser', 'directed', 'folded'] as const) {
        for (const islands of ['exclude', 'keep'] as const) {
          for (const stopAfterExceedances of alternative === 'folded'
            ? [undefined]
            : [undefined, 5]) {
            const graph = new GPUCommandGraph(createNullWebGPUDevice());
            const test = createReplayTest(graph, {
              statistic,
              joinCase,
              alternative,
              islands,
              stopAfterExceedances
            });
            const nodes = test.getCommandNodes(graph).filter(node => node.id === 'golden-permute');
            const key = [
              statistic,
              joinCase ?? '-',
              alternative,
              islands,
              stopAfterExceedances ?? '-'
            ].join('/');
            kernels[key] = getGPUCommandNodeKernelSources(nodes)[0].source;
            graph.device.destroy();
          }
        }
      }
    }
  }
  const fixture = new URL('./fixtures/local-permutation-kernels.json', import.meta.url);
  expect(Object.keys(kernels)).toHaveLength(162);
  expect(kernels).toEqual(JSON.parse(readFileSync(fixture, 'utf8')));
});
