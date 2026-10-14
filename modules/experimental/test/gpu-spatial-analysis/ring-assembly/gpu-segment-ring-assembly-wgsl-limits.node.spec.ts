// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUSegmentRingAssembly} from '../../../src/gpu-spatial-analysis/ring-assembly';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getStorageBindingCount} from '../space-time-tests/wgsl-memory';

it('GPUSegmentRingAssembly kernels stay within 8 storage bindings', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-limits'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `ring-limit-${serial++}`, format, length);
  const nodes = new GPUSegmentRingAssembly({
    endpoints: view('float32x4', 16),
    count: view('uint32', 1),
    groups: view('uint32', 16),
    cancelOpposingSegments: true,
    output: {
      ringOffsets: view('uint32', 5),
      positions: view('float32x2', 32),
      ringShells: view('uint32', 4),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    }
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(10);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
  }
  expect(sources.some(({id}) => id.includes('quantize-extent'))).toBe(true);
});
