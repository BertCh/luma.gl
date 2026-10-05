// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, GPUMask} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../utils/gpu-contributor-test-utils';
import {createOutputBuffer, readUint32} from './gpu-contributor-test-utils';

it('GPUParameterBuffer updates contributor inputs between encodings without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }

  const graph = new GPUCommandGraph(device, {id: 'parameter-buffer'});
  const selection = new GPUParameterBuffer(device, {
    id: 'selection',
    format: 'uint32',
    length: 4,
    values: Uint32Array.from([1, 0, 1, 0])
  });
  const outputBuffer = createOutputBuffer(device, 4);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32', 4);
  graph.add(
    new GPUMask({id: 'invert', inputs: [selection.importToGraph(graph)], output, operation: 'not'})
  );
  const compiled = graph.compile();

  submitGraph(device, compiled, undefined);
  expect(await readUint32(outputBuffer, 4)).toEqual([0, 1, 0, 1]);

  selection.write(Uint32Array.from([1, 1]), 2);
  submitGraph(device, compiled, undefined);
  expect(await readUint32(outputBuffer, 4), 'partial write at an element offset').toEqual([
    0, 1, 0, 0
  ]);

  compiled.destroy();
  selection.destroy();
  outputBuffer.destroy();
});
