// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPULineMerge} from '../../../src/gpu-spatial-analysis/line-merge';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('GPULineMerge kernels parse and test endpoint finiteness on the exponent bits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'line-merge-parse'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `merge-view-${serial++}`, format, length);
  const lineCount = 3;
  const nodes = new GPULineMerge({
    id: 'merge',
    positions: view('float32x2', 8),
    lineOffsets: view('uint32', lineCount + 1),
    output: {
      chainOffsets: view('uint32', lineCount + 1),
      positions: view('float32x2', 8),
      status: {
        count: view('uint32', 1)
      },
      lineChains: view('uint32', lineCount),
      lineOrders: view('uint32', lineCount),
      lineReversed: view('uint32', lineCount)
    }
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(source, id).not.toContain('3.0e38');
    expect(source.split('fn isFiniteBits(').length - 1, id).toBeLessThanOrEqual(1);
  }
  const keys = sources.filter(({id}) => id.endsWith('-keys'));
  expect(keys).toHaveLength(1);
  expect(keys[0].source).toContain('isFiniteBits(x) && isFiniteBits(y)');
  expect(keys[0].source.split('fn isFiniteBits(').length - 1).toBe(1);
});
