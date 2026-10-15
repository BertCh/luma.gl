// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../utils/gpu-contributor-test-utils';
import earthquake from './fixtures/parity-wp6-earthquake-input.json';

type Format = 'float32' | 'uint32' | 'uint32x2' | 'sint32';

/** Owns only test buffers; every expected result comes from the independent JSON fixtures. */
export class ParityReductionHarness {
  readonly graph: GPUCommandGraph;
  readonly buffers: Buffer[] = [];
  readonly outputs = new Map<string, {buffer: Buffer; format: Format; length: number}>();

  constructor(readonly device: Device) {
    this.graph = new GPUCommandGraph(device);
  }

  input<StorageFormat extends Format>(
    name: string,
    values: Float32Array | Uint32Array | Int32Array,
    format: StorageFormat
  ): GraphDataView<StorageFormat> {
    const buffer = createInputBuffer(this.device, values.length ? values : new Uint32Array(1));
    this.buffers.push(buffer);
    return importGraphBuffer(
      this.graph,
      name,
      buffer,
      format,
      values.length / (format === 'uint32x2' ? 2 : 1)
    );
  }

  output<StorageFormat extends Format>(
    name: string,
    length: number,
    format: StorageFormat
  ): GraphDataView<StorageFormat> {
    const wordCount = length * (format === 'uint32x2' ? 2 : 1);
    const buffer = createOutputBuffer(this.device, wordCount);
    this.buffers.push(buffer);
    this.outputs.set(name, {buffer, format, length: wordCount});
    return importGraphBuffer(this.graph, `out-${name}`, buffer, format, length);
  }

  async read(): Promise<Record<string, number[]>> {
    return Object.fromEntries(
      await Promise.all(
        [...this.outputs].map(async ([name, {buffer, format, length}]) => [
          name,
          await (format === 'float32' ? readFloat32(buffer, length) : readUint32(buffer, length))
        ])
      )
    );
  }

  destroy(): void {
    for (const buffer of this.buffers) buffer.destroy();
  }
}

/** Explicit little-endian decoding also works on non-little-endian JS hosts. */
export function decodeParityPayload(payload: {
  base64: string;
  dtype: string;
  length: number;
}): number[] {
  const bytes = Uint8Array.from(atob(payload.base64), character => character.charCodeAt(0));
  if (bytes.byteLength !== payload.length * 4) throw new Error('Invalid parity payload length');
  const view = new DataView(bytes.buffer);
  return Array.from({length: payload.length}, (_, index) =>
    payload.dtype === '<f4' ? view.getFloat32(index * 4, true) : view.getUint32(index * 4, true)
  );
}

export function getParityEarthquakes(): {
  values: number[];
  seconds: number[];
  cells: number[];
  timeOriginMs: number;
} {
  return {
    values: decodeParityPayload(earthquake.magnitude),
    seconds: decodeParityPayload(earthquake.seconds),
    cells: decodeParityPayload(earthquake.cells),
    timeOriginMs: earthquake.timeOriginMs
  };
}
