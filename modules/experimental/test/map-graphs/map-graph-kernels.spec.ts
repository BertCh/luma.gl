// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  createMapGraphKernelNode,
  type MapGraphKernelBinding
} from '../../src/map-graphs/map-graph-kernels';
import {importGraphBuffer, submitGraph} from '../../src/map-graphs/map-graph-utils';
import {createInputBuffer, createOutputBuffer, readUint32} from './map-graph-test-utils';

/** Runs `callback` inside a WebGPU validation error scope and returns the captured messages. */
async function captureValidationErrors(
  device: Device,
  callback: () => Promise<void>
): Promise<string[]> {
  const gpuDevice = device.handle as GPUDevice;
  gpuDevice.pushErrorScope('validation');
  await callback();
  await device.handle.queue.onSubmittedWorkDone();
  const error = await gpuDevice.popErrorScope();
  return error ? [error.message] : [];
}

it('createMapGraphKernelNode keeps storage bindings the body never reads', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }

  const graph = new GPUCommandGraph(device, {id: 'kernel-unused-bindings'});
  const unusedReadBuffer = createInputBuffer(device, Uint32Array.from([9, 9, 9, 9]));
  const unusedWriteBuffer = createOutputBuffer(device, 4);
  const outputBuffer = createOutputBuffer(device, 4);
  const bindings: MapGraphKernelBinding[] = [
    {
      name: 'unusedRead',
      view: importGraphBuffer(graph, 'unused-read', unusedReadBuffer, 'uint32', 4),
      type: 'u32',
      access: 'read'
    },
    {
      name: 'output',
      view: importGraphBuffer(graph, 'output', outputBuffer, 'uint32', 4),
      type: 'u32',
      access: 'read_write'
    },
    {
      name: 'unusedWrite',
      view: importGraphBuffer(graph, 'unused-write', unusedWriteBuffer, 'uint32', 4),
      type: 'atomic<u32>',
      access: 'read_write'
    }
  ];
  graph.add(
    createMapGraphKernelNode(graph, {
      id: 'write-index',
      operation: 'MapGraphKernelTest',
      variant: 'unused-bindings',
      bindings,
      invocationCount: 4,
      // Reads only `output`: `unusedRead` and `unusedWrite` are declared but never referenced.
      body: 'output[outputOffset + index] = index + 7u;'
    })
  );

  const errors = await captureValidationErrors(device, async () => {
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    compiled.destroy();
  });

  expect(errors, 'no WebGPU validation errors').toEqual([]);
  expect(await readUint32(outputBuffer, 4)).toEqual([7, 8, 9, 10]);

  unusedReadBuffer.destroy();
  unusedWriteBuffer.destroy();
  outputBuffer.destroy();
});
