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
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {OracleCSR} from './spatial-weights-oracle';

const GARBAGE = 0x7f7f7f7f;

/** Small test rig: a graph plus the buffers imported into it. */
export class WeightsRig {
  readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private serial = 0;
  private readonly viewBuffers = new Map<GraphDataView, Buffer>();

  constructor(readonly device: Device) {
    this.graph = new GPUCommandGraph(device, {id: 'spatial-weights-test'});
  }

  /** Imports an input array. */
  input<Format extends GPUVectorFormat>(
    data: Float32Array | Uint32Array,
    format: Format,
    length: number
  ): GraphDataView<Format> {
    // COPY_SRC lets specs read in-place transform results back.
    const buffer = this.device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    this.buffers.push(buffer);
    const view = importGraphBuffer(this.graph, `input-${this.serial++}`, buffer, format, length);
    this.viewBuffers.set(view, buffer);
    return view;
  }

  /** Returns the buffer behind an input view created by this rig. */
  bufferOf(view: GraphDataView): Buffer {
    return this.viewBuffers.get(view)!;
  }

  /** Imports a garbage-filled output; read it back with the returned buffer. */
  output<Format extends GPUVectorFormat>(
    format: Format,
    length: number
  ): {view: GraphDataView<Format>; buffer: Buffer} {
    const buffer = createOutputBuffer(this.device, length * 4);
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
    this.buffers.push(buffer);
    return {
      view: importGraphBuffer(this.graph, `output-${this.serial++}`, buffer, format, length),
      buffer
    };
  }

  /** Creates output weights for `rows` rows and `capacity` slots. */
  weightsOutput(rows: number, capacity: number, distances = false) {
    const offsets = this.output('uint32', rows + 1);
    const neighbors = this.output('uint32', capacity);
    const weights = this.output('float32', capacity);
    const distanceOutput = distances ? this.output('float32', capacity) : undefined;
    const spatialWeights: GPUSpatialWeights = {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: weights.view,
      distances: distanceOutput?.view
    };
    return {
      spatialWeights,
      async read(): Promise<OracleCSR> {
        const offsetValues = await readUint32(offsets.buffer, rows + 1);
        const used = offsetValues[rows];
        return {
          offsets: offsetValues,
          neighbors: await readUint32(neighbors.buffer, used),
          weights: await readFloat32(weights.buffer, used),
          distances: distanceOutput ? await readFloat32(distanceOutput.buffer, used) : []
        };
      }
    };
  }

  /** Uploads a CPU CSR as input weights (distances included when present). */
  uploadWeights(csr: OracleCSR, slack = 0): GPUSpatialWeights {
    const capacity = csr.neighbors.length + slack;
    const pad = (values: number[]) => {
      const padded = new Float32Array(capacity);
      padded.set(values);
      return padded;
    };
    const neighbors = new Uint32Array(capacity);
    neighbors.set(csr.neighbors);
    return {
      offsets: this.input(new Uint32Array(csr.offsets), 'uint32', csr.offsets.length),
      neighbors: this.input(neighbors, 'uint32', capacity),
      weights: this.input(pad(csr.weights), 'float32', capacity),
      distances: csr.distances.length
        ? this.input(pad(csr.distances), 'float32', capacity)
        : undefined
    };
  }

  /** Adds producers, compiles, submits once. */
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
