// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

type ViewFormat =
  | 'float32'
  | 'uint32'
  | 'sint32'
  | 'float32x2'
  | 'uint32x2'
  | 'uint32x4'
  | 'float32x4';

/** Owns the buffers and the graph of one recipe test and reads results back. */
export class RecipeTestFixture {
  readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly parameterBuffers: GPUParameterBuffer[] = [];

  constructor(
    readonly device: Device,
    id: string
  ) {
    this.graph = new GPUCommandGraph(device, {id});
  }

  /** Uploads `values` and imports them into the graph as a read-only view. */
  input<Format extends ViewFormat>(
    name: string,
    values: Float32Array | Uint32Array | Int32Array,
    format: Format,
    length: number
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return importGraphBuffer(this.graph, name, buffer, format, length) as GraphDataView<Format>;
  }

  /** Creates a zero-filled caller-owned output and imports it. `length` is in rows. */
  output<Format extends ViewFormat>(
    name: string,
    format: Format,
    length: number
  ): {view: GraphDataView<Format>; buffer: Buffer} {
    const components = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    const buffer = createOutputBuffer(this.device, length * components);
    this.buffers.push(buffer);
    return {
      view: importGraphBuffer(this.graph, name, buffer, format, length) as GraphDataView<Format>,
      buffer
    };
  }

  /** Creates a per-frame parameter buffer and returns its graph view. */
  parameters<Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array
  ): GraphDataView<Format> {
    const parameterBuffer = new GPUParameterBuffer(this.device, {
      id: name,
      format,
      length: values.length,
      values
    } as never);
    this.parameterBuffers.push(parameterBuffer);
    return parameterBuffer.importToGraph(this.graph) as GraphDataView<Format>;
  }

  /** Compiles the graph, encodes it once and submits it. Results stay readable until `destroy()`. */
  run(): void {
    this.compiled = this.graph.compile();
    submitGraph(this.device, this.compiled, undefined);
  }

  private compiled?: ReturnType<GPUCommandGraph['compile']>;

  /** Reads `length` float32 values. */
  readFloat32(output: {buffer: Buffer}, length: number): Promise<number[]> {
    return readFloat32(output.buffer, length);
  }

  /** Reads `length` uint32 values. */
  readUint32(output: {buffer: Buffer}, length: number): Promise<number[]> {
    return readUint32(output.buffer, length);
  }

  /** Reads `length` int32 values. */
  async readInt32(output: {buffer: Buffer}, length: number): Promise<number[]> {
    const bits = await readUint32(output.buffer, length);
    return Array.from(new Int32Array(Uint32Array.from(bits).buffer));
  }

  destroy(): void {
    this.compiled?.destroy();
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
    for (const parameterBuffer of this.parameterBuffers) {
      parameterBuffer.destroy();
    }
  }
}

/** Absolute-plus-relative closeness. NaN equals NaN. */
export function isClose(actual: number, expected: number, absolute: number, relative: number) {
  if (Number.isNaN(expected)) {
    return Number.isNaN(actual);
  }
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected);
}
