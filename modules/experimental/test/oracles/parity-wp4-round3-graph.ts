// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {createInputBuffer, createOutputBuffer} from '../utils/gpu-contributor-test-utils';

/** Own only test buffers; contributors and compiled graphs are destroyed by each spec. */
export class ParityRasterGraph {
  readonly graph: GPUCommandGraph;
  readonly buffers: Buffer[] = [];
  constructor(
    readonly device: Device,
    name: string
  ) {
    this.graph = new GPUCommandGraph(device, {id: `parity-wp4-${name}`});
  }
  input<Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array
  ) {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return importGraphBuffer(
      this.graph,
      name,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }
  output<Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) {
    const buffer = createOutputBuffer(this.device, length * (format === 'float32x2' ? 2 : 1));
    this.buffers.push(buffer);
    return {buffer, view: importGraphBuffer(this.graph, name, buffer, format, length)};
  }
  destroy() {
    for (const buffer of this.buffers) buffer.destroy();
  }
}
