// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

const GARBAGE = 0x7f7f7f7f;

/** Small test rig: a graph plus the buffers imported into it. */
export class GraphRig {
  readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private serial = 0;

  constructor(
    readonly device: Device,
    id = 'areal-test'
  ) {
    this.graph = new GPUCommandGraph(device, {id});
  }

  /** Imports an input array. */
  input<Format extends GPUVectorFormat>(
    data: Float32Array | Uint32Array,
    format: Format,
    length: number = data.length
  ): GraphDataView<Format> {
    const buffer = this.device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    this.buffers.push(buffer);
    return importGraphBuffer(this.graph, `input-${this.serial++}`, buffer, format, length);
  }

  /** Imports a garbage-filled output. */
  output<Format extends GPUVectorFormat>(
    format: Format,
    length: number
  ): {
    view: GraphDataView<Format>;
    buffer: Buffer;
    readUint32: () => Promise<number[]>;
    readFloat32: () => Promise<number[]>;
  } {
    const buffer = createOutputBuffer(this.device, length);
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
    this.buffers.push(buffer);
    return {
      view: importGraphBuffer(this.graph, `output-${this.serial++}`, buffer, format, length),
      buffer,
      readUint32: () => readUint32(buffer, length),
      readFloat32: () => readFloat32(buffer, length)
    };
  }

  /** Adds producers, compiles and submits once. */
  run(...producers: GPUCommandNodeProducer[]): void {
    for (const producer of producers) {
      this.graph.add(producer);
    }
    const compiled = this.graph.compile();
    submitGraph(this.device, compiled, undefined);
    this.buffers.push({destroy: () => compiled.destroy()} as Buffer);
  }

  /** Releases every buffer and the compiled graph. */
  destroy(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

/** Seeded generator in [0, 1). */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Asserts `actual` matches `expected` within an absolute plus relative tolerance. */
export function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  label: string,
  relative = 2e-5,
  absolute = 1e-6
): void {
  if (actual.length !== expected.length) {
    throw new Error(`${label}: length ${actual.length} != ${expected.length}`);
  }
  for (let index = 0; index < expected.length; index++) {
    if (
      !(
        Math.abs(actual[index] - expected[index]) <=
        absolute + relative * Math.abs(expected[index])
      )
    ) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}
