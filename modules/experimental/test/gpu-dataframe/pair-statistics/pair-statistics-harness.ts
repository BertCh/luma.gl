// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Output column requested by a harness: format and row count. */
export type HarnessOutput = {format: 'float32' | 'uint32'; length: number};

/** Views handed to a harness contributor factory. */
export type HarnessViews<Names extends string = string> = {
  positions: GraphDataView<'float32x2'>;
  values?: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  outputs: Record<Names, GraphDataView<'float32'> | GraphDataView<'uint32'>>;
};

/** A compiled graph holding one contributor, re-encoded with new parameters on every `run`. */
export type PairStatisticsHarness<Names extends string = string> = {
  run(parameters: Float32Array): Promise<Record<Names, number[]>>;
  /** Number of graph compilations after the first; stays 0 across parameter changes. */
  readonly rebuildCount: number;
  destroy(): void;
};

/**
 * Builds and compiles one graph around `createContributor`, importing the scene and one caller-owned
 * buffer per requested output.
 */
export function createPairStatisticsHarness<Names extends string>(
  device: Device,
  props: {
    positions: Float32Array;
    values?: Float32Array;
    mask?: Uint32Array;
    parameterLength: number;
    outputs: Record<Names, HarnessOutput>;
    createContributor: (views: HarnessViews<Names>) => GPUCommandNodeProducer;
  }
): PairStatisticsHarness<Names> {
  const rows = props.positions.length / 2;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'pair-statistics-harness'});
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'pair-statistics-parameters',
    format: 'float32',
    length: props.parameterLength
  });
  const outputBuffers = {} as Record<Names, Buffer>;
  const outputViews = {} as HarnessViews<Names>['outputs'];
  for (const [name, output] of Object.entries(props.outputs) as [Names, HarnessOutput][]) {
    outputBuffers[name] = track(createOutputBuffer(device, output.length));
    outputViews[name] = importGraphBuffer(
      graph,
      `output-${name}`,
      outputBuffers[name],
      output.format,
      output.length
    );
  }
  const views: HarnessViews<Names> = {
    positions: importGraphBuffer(
      graph,
      'positions',
      track(createInputBuffer(device, props.positions)),
      'float32x2',
      rows
    ),
    values: props.values
      ? importGraphBuffer(
          graph,
          'values',
          track(createInputBuffer(device, props.values)),
          'float32',
          rows
        )
      : undefined,
    mask: props.mask
      ? importGraphBuffer(
          graph,
          'mask',
          track(createInputBuffer(device, props.mask)),
          'uint32',
          rows
        )
      : undefined,
    parameters: parameterBuffer.importToGraph(graph),
    outputs: outputViews
  };
  graph.add(props.createContributor(views));
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(parameters) {
      parameterBuffer.write(parameters);
      submitGraph(device, compiled, undefined);
      const result = {} as Record<Names, number[]>;
      for (const [name, output] of Object.entries(props.outputs) as [Names, HarnessOutput][]) {
        result[name] =
          output.format === 'uint32'
            ? await readUint32(outputBuffers[name], output.length)
            : await readFloat32(outputBuffers[name], output.length);
      }
      return result;
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Returns the f32 bit patterns of `values`, with every NaN mapped to one key. */
export function getFloatBits(values: number[]): (number | 'nan')[] {
  const bits = new Uint32Array(Float32Array.from(values).buffer);
  return Array.from(bits, (bit, index) => (Number.isNaN(values[index]) ? 'nan' : bit));
}
