// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';

export function createRound4Harness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputs = new Map<string, {buffer: Buffer; length: number; integer: boolean}>();
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  return {
    graph,
    input<Format extends 'float32' | 'uint32' | 'float32x2'>(
      name: string,
      values: number[],
      format: Format
    ) {
      const buffer = createInputBuffer(
        device,
        format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
      );
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      );
    },
    output<Format extends 'float32' | 'uint32' | 'float32x2'>(
      name: string,
      length: number,
      format: Format
    ) {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      outputs.set(name, {buffer, length, integer: format === 'uint32'});
      return importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        length / (format === 'float32x2' ? 2 : 1)
      );
    },
    run() {
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
    },
    async read(name: string): Promise<number[]> {
      const output = outputs.get(name)!;
      return output.integer
        ? readUint32(output.buffer, output.length)
        : readFloat32(output.buffer, output.length);
    },
    destroy() {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
