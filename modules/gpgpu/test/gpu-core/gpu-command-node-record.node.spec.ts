// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {hashGPUComputationText} from '@luma.gl/gpgpu';
import {
  GPUCommandGraph,
  getGPUComputeKernelAnnotation,
  createTransientView
} from '@luma.gl/gpgpu/gpu-core';
import {NullDevice} from '@luma.gl/test-utils';
import {expect, test, vi} from 'vitest';
import {createWGSLKernelNode} from '../../../experimental/src/utils/wgsl-kernel-nodes';

test('does not evaluate the descriptor or create an annotation without recording', () => {
  const describe = vi.fn((): never => {
    throw new Error('Must remain lazy');
  });
  expect(getGPUComputeKernelAnnotation({}, describe)).toBeUndefined();
  expect(getGPUComputeKernelAnnotation({recording: undefined}, describe)).toBeUndefined();
  expect(getGPUComputeKernelAnnotation({recording: false}, describe)).toBeUndefined();
  expect(describe).not.toHaveBeenCalled();
});

test('mirrors WGSL helper hashes and records exact bindings and dispatch facts', () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 134217728,
      maxStorageBuffersPerShaderStage: 8
    }
  });
  try {
    const graph = new GPUCommandGraph(device, {recording: {sourceText: true}});
    const storage = createTransientView(graph, 'rainfall', 'uint32', 128);
    const rainfall = graph.createDataView(storage.buffer, {
      format: 'uint32',
      byteOffset: 268,
      length: 16
    });
    const uniform = graph.createTransientBuffer({
      id: 'parameters',
      byteLength: 16,
      usage: Buffer.UNIFORM
    });
    const peer = createWGSLKernelNode(graph, {
      id: 'peer',
      operation: 'GPUScan',
      invocationCount: 16,
      workgroupSize: 8,
      bindings: [{name: 'rainfall', view: rainfall, type: 'u32', access: 'read'}],
      body: 'let value = rainfall[rainfallOffset + index];'
    });
    const source = peer.annotation!.source!;
    const annotation = getGPUComputeKernelAnnotation(graph, () => ({
      op: 'GPUScan',
      moduleId: 'GPUScan/level-scan',
      source: source.text!,
      constants: {COUNT: 16},
      workgroupSize: [8, 1, 1],
      workgroups: [2, 1, 1],
      invocationCount: 16,
      datum: {keySpace: 'stations'},
      loop: {id: 'scan/levels', iteration: 0, phase: 'unconditional', maximumIterations: 2},
      formula: 'gpu-core.scan',
      dedupName: 'level-scan',
      label: 'Rainfall prefix',
      bindings: [
        {name: 'rainfall', view: rainfall, access: 'read'},
        {name: 'parameters', view: uniform, access: 'uniform'}
      ]
    }))!;
    expect(annotation.source).toEqual({
      id: source.id,
      hash: source.hash,
      language: 'wgsl',
      text: source.text,
      moduleId: 'GPUScan/level-scan',
      entryPoint: 'main',
      constants: {COUNT: 16}
    });
    expect(annotation.source?.hash).toBe(hashGPUComputationText(source.text!));
    expect(annotation).toMatchObject({
      op: 'GPUScan',
      label: 'Rainfall prefix',
      datum: {keySpace: 'stations'},
      loop: {id: 'scan/levels', iteration: 0, phase: 'unconditional', maximumIterations: 2},
      formula: 'gpu-core.scan',
      dedupName: 'level-scan',
      workgroupSize: [8, 1, 1],
      dispatch: {workgroups: [2, 1, 1], invocationCount: 16, indirect: false}
    });
    expect(annotation.bindings).toEqual([
      {
        name: 'rainfall',
        resource: 'rainfall',
        access: 'read',
        elementType: 'u32',
        elementOffset: 3,
        length: 16
      },
      {name: 'parameters', resource: 'parameters', access: 'uniform'}
    ]);
    expect(annotation).toEqual(JSON.parse(JSON.stringify(annotation)));
  } finally {
    device.destroy();
  }
});
