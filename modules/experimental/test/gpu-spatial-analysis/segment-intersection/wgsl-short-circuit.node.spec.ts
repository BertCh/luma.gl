// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createCPUGraphMemory, executeCommandNodesOnCPU} from './wgsl-graph-executor';

it('CPU WGSL logical operators short-circuit RHS reads and side effects', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const results = createTransientView(graph, 'results', 'uint32', 4);
  const counter = createTransientView(graph, 'counter', 'uint32', 1);
  const node = createWGSLKernelNode(graph, {
    id: 'short-circuit',
    operation: 'test',
    invocationCount: 1,
    bindings: [
      {name: 'results', view: results, type: 'u32', access: 'read_write'},
      {name: 'counter', view: counter, type: 'atomic<u32>', access: 'read_write'}
    ],
    declarations: `fn record() -> bool { atomicAdd(&counter[counterOffset], 1u); return true; }`,
    body: `let never = index == 1u;
  results[resultsOffset] = select(0u, 1u, never && record());
  results[resultsOffset + 1u] = select(0u, 1u, !never || record());
  results[resultsOffset + 2u] = select(0u, 1u, !never && record());
  results[resultsOffset + 3u] = select(0u, 1u, never || record());
  if (never && results[resultsOffset + 0xffffffffu] == 0u) { results[resultsOffset] = 2u; }`
  });
  const memory = createCPUGraphMemory();
  executeCommandNodesOnCPU([node], memory);
  expect(Array.from(memory.readUint32(results))).toEqual([0, 1, 1, 1]);
  expect(memory.readUint32(counter)[0]).toBe(2);
});
