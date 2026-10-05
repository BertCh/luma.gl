// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {expect, test} from 'vitest';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

test('createWGSLKernelNode accepts an omitted workload variant', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'kernel-variant'});
  const output = createTransientView(graph, 'output', 'uint32', 4);

  const node = createWGSLKernelNode(graph, {
    id: 'no-variant',
    operation: 'WGSLKernelTest',
    bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
    invocationCount: 4,
    body: 'output[outputOffset + index] = index;'
  });

  expect(node.workload && 'variant' in node.workload, 'no undefined variant key').toBe(false);
  expect(() => graph.add(node)).not.toThrow();
  device.destroy();
});
