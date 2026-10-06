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
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** A CPU CSR weights matrix used by the oracles. */
export type CPUWeights = {
  offsets: number[];
  neighbors: number[];
  weights: number[];
};

/** Small graph test rig shared by the S5b specs: imports inputs and outputs, runs, reads back. */
export class AnalysisRig {
  readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly outputBuffers = new Map<GraphDataView, Buffer>();
  private serial = 0;
  private compiled?: {destroy(): void};

  constructor(readonly device: Device) {
    this.graph = new GPUCommandGraph(device, {id: 's5b-test'});
  }

  /** Imports an input array. */
  input<Format extends GPUVectorFormat>(
    data: Float32Array | Uint32Array,
    format: Format,
    length = data.length
  ): GraphDataView<Format> {
    const buffer = this.device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    this.buffers.push(buffer);
    return importGraphBuffer(this.graph, `input-${this.serial++}`, buffer, format, length);
  }

  /** Imports a zero-filled output. */
  output<Format extends GPUVectorFormat>(format: Format, length: number): GraphDataView<Format> {
    const buffer = createOutputBuffer(this.device, length);
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(0x7f7f7f7f));
    this.buffers.push(buffer);
    const view = importGraphBuffer(this.graph, `output-${this.serial++}`, buffer, format, length);
    this.outputBuffers.set(view, buffer);
    return view;
  }

  /** Uploads CPU weights with `slack` extra unspecified slots. */
  weights(csr: CPUWeights, slack = 3): GPUSpatialWeights {
    const capacity = csr.neighbors.length + slack;
    const neighbors = new Uint32Array(capacity).fill(0xdeadbeef);
    neighbors.set(csr.neighbors);
    const weights = new Float32Array(capacity).fill(99);
    weights.set(csr.weights);
    return {
      offsets: this.input(new Uint32Array(csr.offsets), 'uint32'),
      neighbors: this.input(neighbors, 'uint32'),
      weights: this.input(weights, 'float32')
    };
  }

  /** Adds producers, compiles and submits once. */
  run(...producers: GPUCommandNodeProducer[]): void {
    for (const producer of producers) {
      this.graph.add(producer);
    }
    const compiled = this.graph.compile();
    this.compiled = compiled;
    submitGraph(this.device, compiled, undefined);
  }

  /** Reads an output view as float32. */
  readFloat(view: GraphDataView<'float32'>): Promise<number[]> {
    return readFloat32(this.outputBuffers.get(view)!, view.length);
  }

  /** Reads an output view as uint32. */
  readUint(view: GraphDataView<'uint32'>): Promise<number[]> {
    return readUint32(this.outputBuffers.get(view)!, view.length);
  }

  /** Releases every buffer and the compiled graph. */
  destroy(): void {
    this.compiled?.destroy();
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

/** mulberry32 generator with values in [0, 1). */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Builds a Gaussian-kernel distance-band CSR between two point sets (self join excludes i == j). */
export function buildKernelWeights(
  rows: Float64Array | number[],
  columns: Float64Array | number[],
  bandwidth: number,
  options: {selfJoin: boolean}
): CPUWeights {
  const rowCount = rows.length / 2;
  const columnCount = columns.length / 2;
  const csr: CPUWeights = {offsets: [0], neighbors: [], weights: []};
  for (let i = 0; i < rowCount; i++) {
    for (let j = 0; j < columnCount; j++) {
      if (options.selfJoin && i === j) continue;
      const distance = Math.hypot(
        rows[2 * i] - columns[2 * j],
        rows[2 * i + 1] - columns[2 * j + 1]
      );
      if (distance <= bandwidth) {
        csr.neighbors.push(j);
        csr.weights.push(Math.fround(Math.exp(-((distance / bandwidth) ** 2))));
      }
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

/** Transposes a CSR (rows become columns), keeping ascending neighbor IDs. */
export function transposeWeights(csr: CPUWeights, columnCount: number): CPUWeights {
  const rows: {index: number; weight: number}[][] = Array.from({length: columnCount}, () => []);
  for (let i = 0; i + 1 < csr.offsets.length; i++) {
    for (let slot = csr.offsets[i]; slot < csr.offsets[i + 1]; slot++) {
      rows[csr.neighbors[slot]].push({index: i, weight: csr.weights[slot]});
    }
  }
  const result: CPUWeights = {offsets: [0], neighbors: [], weights: []};
  for (const row of rows) {
    for (const entry of row) {
      result.neighbors.push(entry.index);
      result.weights.push(entry.weight);
    }
    result.offsets.push(result.neighbors.length);
  }
  return result;
}
