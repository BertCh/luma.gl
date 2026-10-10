// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {createStableLexicographicIndexSortNodes} from '../../src/utils/stable-lexicographic-index-sort';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';

it('stable lexicographic index sort preserves lower-order tuple components', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const graph = new GPUCommandGraph(device);
  const primaryValues = new Uint32Array([1, 0, 1, 0, 1, 0, 2, 2]);
  const secondaryValues = new Uint32Array([0, 1, 0, 1, 1, 0, 0, 0]);
  const tertiaryValues = new Uint32Array([2, 3, 1, 1, 0, 9, 0, 0]);
  const identityValues = new Uint32Array(primaryValues.length).map((_, index) => index);
  const buffers = [
    createInputBuffer(device, primaryValues),
    createInputBuffer(device, secondaryValues),
    createInputBuffer(device, tertiaryValues),
    createInputBuffer(device, identityValues),
    createOutputBuffer(device, primaryValues.length)
  ];
  const view = (name: string, buffer: (typeof buffers)[number]) =>
    importGraphBuffer(graph, name, buffer, 'uint32', primaryValues.length);
  const primary = view('primary', buffers[0]);
  const secondary = view('secondary', buffers[1]);
  const tertiary = view('tertiary', buffers[2]);
  const identity = view('identity', buffers[3]);
  const output = view('output', buffers[4]);
  const sorted = createStableLexicographicIndexSortNodes(graph, {
    id: 'tuple',
    operation: 'TupleTest',
    indices: identity,
    keys: [{view: primary}, {view: secondary}, {view: tertiary}]
  });
  const nodeIds = sorted.nodes.map(node => node.id);
  expect(nodeIds.filter(nodeId => nodeId.endsWith('-gather'))).toEqual([
    'tuple-pass-0-gather',
    'tuple-pass-1-gather',
    'tuple-pass-2-gather'
  ]);
  expect(nodeIds.some(nodeId => nodeId.startsWith('tuple-pass-0-sort'))).toBe(true);
  expect(nodeIds.some(nodeId => nodeId.startsWith('tuple-pass-1-sort'))).toBe(true);
  expect(nodeIds.some(nodeId => nodeId.startsWith('tuple-pass-2-sort'))).toBe(true);
  expect(sorted.sortedIndices.length).toBe(primaryValues.length);
  expect(sorted.sortedPrimaryKeys.length).toBe(primaryValues.length);
  graph.add([
    ...sorted.nodes,
    createWGSLKernelNode(graph, {
      id: 'copy-result',
      operation: 'TupleTest',
      variant: 'copy',
      bindings: [
        {name: 'input', view: sorted.sortedIndices, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      invocationCount: primaryValues.length,
      body: 'output[outputOffset + index] = input[inputOffset + index];'
    })
  ]);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(buffers[4], primaryValues.length)).toEqual([5, 3, 1, 2, 0, 4, 6, 7]);
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
});
