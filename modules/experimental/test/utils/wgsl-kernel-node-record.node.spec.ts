// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {hashGPUComputationText} from '@luma.gl/gpgpu';
import {GPUCommandGraph, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {expect, test} from 'vitest';
import {
  createFillNode,
  createPublishNode,
  createWGSLActiveCountDispatch,
  createWGSLKernelNode
} from '../../src/utils/wgsl-kernel-nodes';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from './gpu-contributor-test-utils';

test('WGSL recording retains the exact source, bindings, dispatch and contributor metadata', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'record', recording: {}});
  const input = createTransientView(graph, 'input', 'uint32', 8);
  const output = createTransientView(graph, 'output', 'uint32', 8);
  const texture = graph.createTransientTexture({
    id: 'texture',
    format: 'rgba8unorm',
    width: 2,
    height: 2,
    usage: 0x4
  });
  const textureView = graph.createTextureView(texture);
  const body = 'output[outputOffset + index] = input[inputOffset + index];';
  const record = {
    label: 'Copy flags',
    namespace: 'selection',
    derivedFrom: ['intent:copy'],
    datum: {keySpace: 'row'},
    loop: {id: 'repeat', iteration: 2},
    formula: 'identity',
    attrs: {purpose: 'test'}
  };
  const props = {
    id: 'copy',
    operation: 'CopyFlags',
    variant: 'linear',
    record,
    bindings: [
      {name: 'input', view: input, type: 'u32' as const, access: 'read' as const},
      {name: 'output', view: output, type: 'u32' as const, access: 'read_write' as const}
    ],
    textures: [{name: 'image', texture: textureView, sampleType: 'float' as const}],
    invocationCount: 8,
    workgroupSize: 4,
    constants: {EXAMPLE: 3},
    body
  };
  try {
    const node = createWGSLKernelNode(graph, props);
    const annotation = node.annotation!;
    const source = annotation.source!;
    expect(annotation).toMatchObject({...record, op: 'CopyFlags'});
    expect(source).toMatchObject({
      id: `wgsl:${hashGPUComputationText(source.text!)}`,
      hash: hashGPUComputationText(source.text!),
      bodyHash: hashGPUComputationText(body),
      language: 'wgsl',
      moduleId: 'CopyFlags/linear',
      entryPoint: 'main',
      constants: {
        EXAMPLE: 3,
        INVOCATION_COUNT: 8,
        WGSL_KERNEL_DISPATCH_X: 2,
        WGSL_KERNEL_DISPATCH_Y: 1
      }
    });
    expect(source.text).toContain(body);
    expect(source.text).toContain('@workgroup_size(4)');
    expect(annotation.workgroupSize).toEqual([4, 1, 1]);
    expect(annotation.dispatch).toEqual({
      workgroups: [2, 1, 1],
      invocationCount: 8,
      indirect: false
    });
    expect(annotation.bindings).toEqual([
      {
        name: 'input',
        resource: input.buffer.id,
        access: 'read',
        elementType: 'u32',
        elementOffset: 0,
        length: 8
      },
      {
        name: 'output',
        resource: output.buffer.id,
        access: 'read_write',
        elementType: 'u32',
        elementOffset: 0,
        length: 8
      },
      {name: 'image', resource: texture.id, access: 'sampled'}
    ]);
    const second = createWGSLKernelNode(graph, {...props, id: 'second', invocationCount: 4});
    expect(second.annotation?.source?.bodyHash).toBe(source.bodyHash);
    expect(getGPUCommandNodeKernelSources([node])[0].source).toBe(source.text);

    const production = new GPUCommandGraph(device, {id: 'production'});
    const productionOutput = createTransientView(production, 'output', 'uint32', 8);
    const productionInput = createTransientView(production, 'input', 'uint32', 8);
    const productionNode = createWGSLKernelNode(production, {
      ...props,
      bindings: [
        {...props.bindings[0], view: productionInput},
        {...props.bindings[1], view: productionOutput}
      ],
      get record(): never {
        throw new Error('Production must not inspect record metadata');
      }
    });
    expect('annotation' in productionNode).toBe(false);
    expect(getGPUCommandNodeKernelSources([productionNode])[0].source).toBe(source.text);
  } finally {
    device.destroy();
  }
});

test('WGSL helper kernels forward record metadata and indirect dispatch bounds', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'helpers', recording: {}});
  const count = createTransientView(graph, 'count', 'uint32', 1);
  const output = createTransientView(graph, 'output', 'uint32', 8);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const total = createTransientView(graph, 'total', 'uint32', 1);
  const record = {namespace: 'helpers'};
  try {
    const dispatch = createWGSLActiveCountDispatch(graph, {
      id: 'dispatch',
      operation: 'Test',
      count,
      maximumItemCount: 8,
      workgroupSize: 4,
      record
    });
    expect(dispatch.updateNode.annotation?.namespace).toBe('helpers');
    const indirect = createWGSLKernelNode(graph, {
      id: 'indirect',
      operation: 'Test',
      bindings: [],
      invocationCount: 8,
      workgroupSize: 4,
      condition: dispatch.condition,
      body: '',
      record: {label: undefined}
    });
    expect(indirect.annotation?.dispatch).toEqual({
      workgroups: [2, 1, 1],
      invocationCount: 8,
      indirect: true
    });
    expect(indirect.annotation).not.toHaveProperty('label');
    expect(indirect.annotation?.source?.moduleId).toBe('Test');
    expect(dispatch.condition.buffer.usage & Buffer.INDIRECT).toBeTruthy();
    const fill = createFillNode(graph, {
      id: 'fill',
      operation: 'Test',
      view: output,
      type: 'u32',
      value: '0u',
      record
    });
    const publish = createPublishNode(graph, {
      id: 'publish',
      operation: 'Test',
      requiredCount: total,
      output: {ids: output, status: {count, overflow}},
      record
    });
    expect(fill.annotation?.namespace).toBe('helpers');
    expect(publish.annotation?.namespace).toBe('helpers');
  } finally {
    device.destroy();
  }
});
