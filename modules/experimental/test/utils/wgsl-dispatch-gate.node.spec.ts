// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  createSegmentSumNode,
  getSegmentSumDispatchShape
} from '../../src/utils/sorted-segment-sums';
import {createWGSLDispatchGate, createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

it('createWGSLDispatchGate allocates one indirect triple per shape and gates by slot', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const gate = createWGSLDispatchGate(graph, {
    id: 'demo-gate',
    operation: 'GPUDemo',
    shapes: [[1000], [3, 256], [0]]
  });
  expect(gate.view.length).toBe(9);
  expect(gate.binding).toMatchObject({name: 'dispatch', view: gate.view, access: 'read_write'});

  const slotOne = gate.getGate(1);
  expect(slotOne.condition).toMatchObject({
    source: 'gpu',
    mode: 'indirect',
    buffer: gate.view.buffer,
    byteOffset: gate.view.byteOffset + 12
  });
  expect(slotOne.extraResources).toEqual([{buffer: gate.view, usage: 'indirect'}]);
  expect(gate.getGate(0).condition).toMatchObject({byteOffset: gate.view.byteOffset});

  // Slot 0 holds the 1000-invocation layout (workgroup size 256, four workgroups); an empty shape
  // still dispatches one workgroup when armed.
  expect(gate.armWGSL).toContain('dispatch[dispatchOffset + 0u] = 4u;');
  expect(gate.armWGSL).toContain('dispatch[dispatchOffset + 6u] = 1u;');
  expect(gate.disarmWGSL.match(/= 0u;/g)).toHaveLength(3);
  expect(gate.disarmWGSL).toContain('dispatch[dispatchOffset + 3u] = 0u;');
});

it('a gated kernel node carries the gate condition and its indirect resource use', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const gate = createWGSLDispatchGate(graph, {id: 'g', operation: 'GPUDemo', shapes: [[64]]});
  const values = createTransientView(graph, 'values', 'float32', 64);
  const node = createWGSLKernelNode(graph, {
    id: 'gated',
    operation: 'GPUDemo',
    bindings: [{name: 'values', view: values, type: 'f32', access: 'read_write'}],
    invocationCount: 64,
    ...gate.getGate(0),
    body: 'values[valuesOffset + index] = 1.0;'
  });
  expect(node.condition).toMatchObject({source: 'gpu', mode: 'indirect'});
  expect(node.resources).toContainEqual({buffer: gate.view, usage: 'indirect'});
});

it('createSegmentSumNode with a condition uses the kernel path and forwards the gate', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const gate = createWGSLDispatchGate(graph, {id: 'g', operation: 'GPUDemo', shapes: [[8]]});
  const input = createTransientView(graph, 'input', 'float32', 8);
  const segmentOffsets = createTransientView(graph, 'offsets', 'uint32', 4);
  const output = createTransientView(graph, 'output', 'float32', 3);
  const props = {
    operation: 'GPUDemo',
    segmentCount: 3,
    input,
    segmentOffsets,
    output,
    offsetsWithinInput: true
  };

  const plain = createSegmentSumNode(graph, {...props, id: 'plain'});
  expect(plain.condition).toBeUndefined();

  const gated = createSegmentSumNode(graph, {...props, id: 'gated', ...gate.getGate(0)});
  expect(gated.condition).toMatchObject({source: 'gpu', mode: 'indirect'});
  expect(gated.resources).toContainEqual({buffer: gate.view, usage: 'indirect'});
  expect(gated.id).toBe('gated');
});

it('getSegmentSumDispatchShape gives one workgroup per segment', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const gate = createWGSLDispatchGate(graph, {
    id: 'g',
    operation: 'GPUDemo',
    shapes: [getSegmentSumDispatchShape(5)]
  });
  expect(gate.armWGSL).toContain('dispatch[dispatchOffset + 0u] = 5u;');
});

it('createSegmentSumNode rejects views too short for the segment count', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const input = createTransientView(graph, 'input', 'float32', 8);
  const output = createTransientView(graph, 'output', 'float32', 3);
  const props = {id: 'sum', operation: 'GPUDemo', segmentCount: 3, input, output};

  expect(() =>
    createSegmentSumNode(graph, {
      ...props,
      segmentOffsets: createTransientView(graph, 'short-offsets', 'uint32', 3)
    })
  ).toThrow();
  expect(() =>
    createSegmentSumNode(graph, {
      ...props,
      segmentOffsets: createTransientView(graph, 'offsets', 'uint32', 4),
      output: createTransientView(graph, 'short-output', 'float32', 2)
    })
  ).toThrow();
  expect(() =>
    createSegmentSumNode(graph, {
      ...props,
      segmentOffsets: createTransientView(graph, 'ok-offsets', 'uint32', 4)
    })
  ).not.toThrow();
});
