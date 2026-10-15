// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {readFloat32, readUint32} from '../utils/gpu-contributor-test-utils';

/** WP3-owned allocation/readback helper; no numerical reference logic. */
export function createParityResources(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const vectors: GPUVector[] = [];
  const outputs: Record<string, {buffer: Buffer; length: number}> = {};
  function allocate(values: Float32Array | Uint32Array): Buffer {
    const buffer = device.createBuffer({
      data: values.length ? values : new Uint32Array(1),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return buffer;
  }
  return {
    graph,
    input<Format extends 'float32' | 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      values: readonly number[]
    ) {
      const buffer = allocate(
        format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
      );
      return importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      );
    },
    output<Format extends 'float32' | 'uint32'>(name: string, format: Format, length: number) {
      const buffer = allocate(new Uint32Array(length));
      outputs[name] = {buffer, length};
      return importGraphBuffer(graph, name, buffer, format, length);
    },
    vector<Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      values: readonly number[]
    ) {
      const buffer = allocate(
        format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
      );
      outputs[name] = {buffer, length: values.length};
      const vector = new GPUVector<Format>({
        type: 'buffer',
        name,
        format,
        buffer,
        length: values.length,
        ownsBuffer: false
      });
      vectors.push(vector);
      return vector;
    },
    readFloat(name: string) {
      return readFloat32(outputs[name].buffer, outputs[name].length);
    },
    readUint(name: string) {
      return readUint32(outputs[name].buffer, outputs[name].length);
    },
    destroy() {
      for (const vector of vectors) vector.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
