// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {expect, vi} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Packed formats the fixture can import and read back. */
export type FixtureFormat = 'float32' | 'float32x2' | 'float32x4' | 'uint32';

const COMPONENTS: Record<FixtureFormat, number> = {
  float32: 1,
  float32x2: 2,
  float32x4: 4,
  uint32: 1
};

/** One input column of a {@link GeometryFixture}. */
export type FixtureInput = {values: Float32Array | Uint32Array; format: FixtureFormat};
/** One output column of a {@link GeometryFixture}. */
export type FixtureOutput = {format: FixtureFormat; length: number};

/** A compiled single-contributor graph with named inputs, outputs and a parameter buffer. */
export type GeometryFixture = {
  /** Writes parameters (when given), encodes once, and reads every output as flat numbers. */
  run(parameters?: Float32Array): Promise<Record<string, number[]>>;
  /** Overwrites an input buffer in place (same length) for the next run. */
  writeInput(name: string, values: Float32Array | Uint32Array): void;
  /** Number of `compile()` calls after setup. */
  getCompileCount(): number;
  destroy(): void;
};

/**
 * Builds a fixture around one contributor.
 *
 * `create` receives imported input, output and parameter views and returns the producer to add.
 */
export function createGeometryFixture(
  device: Device,
  options: {
    inputs: Record<string, FixtureInput>;
    outputs: Record<string, FixtureOutput>;
    parameterLength?: number;
    create: (views: {
      inputs: Record<string, GraphDataView>;
      outputs: Record<string, GraphDataView>;
      parameters: GraphDataView<'float32'>;
    }) => GPUCommandNodeProducer;
  }
): GeometryFixture {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'geometry-fixture'});
  const inputBuffers: Record<string, Buffer> = {};
  const inputViews: Record<string, GraphDataView> = {};
  for (const [name, input] of Object.entries(options.inputs)) {
    const buffer = createInputBuffer(device, input.values);
    buffers.push(buffer);
    inputBuffers[name] = buffer;
    inputViews[name] = importGraphBuffer(
      graph,
      `in-${name}`,
      buffer,
      input.format,
      input.values.length / COMPONENTS[input.format]
    );
  }
  const outputBuffers: Record<string, Buffer> = {};
  const outputViews: Record<string, GraphDataView> = {};
  for (const [name, output] of Object.entries(options.outputs)) {
    const buffer = createOutputBuffer(device, output.length * COMPONENTS[output.format]);
    buffers.push(buffer);
    outputBuffers[name] = buffer;
    outputViews[name] = importGraphBuffer(
      graph,
      `out-${name}`,
      buffer,
      output.format,
      output.length
    );
  }
  const parameterLength = options.parameterLength ?? 4;
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: parameterLength
  });
  graph.add(
    options.create({
      inputs: inputViews,
      outputs: outputViews,
      parameters: parameterBuffer.importToGraph(graph)
    })
  );
  const compiled = graph.compile();
  const compileSpy = vi.spyOn(graph, 'compile');
  return {
    async run(parameters) {
      if (parameters) {
        parameterBuffer.write(parameters);
      }
      submitGraph(device, compiled, undefined);
      const result: Record<string, number[]> = {};
      for (const [name, output] of Object.entries(options.outputs)) {
        const length = output.length * COMPONENTS[output.format];
        result[name] =
          output.format === 'uint32'
            ? await readUint32(outputBuffers[name], length)
            : await readFloat32(outputBuffers[name], length);
      }
      return result;
    },
    writeInput(name, values) {
      inputBuffers[name].write(values);
    },
    getCompileCount() {
      return compileSpy.mock.calls.length;
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

/** Relative-or-absolute closeness: `|a - b| <= absolute + relative * |b|`. */
export function expectClose(
  actual: number,
  expected: number,
  relative: number,
  absolute: number,
  label = ''
): void {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
    return;
  }
  expect(
    Math.abs(actual - expected),
    `${label} actual ${actual} expected ${expected}`
  ).toBeLessThanOrEqual(absolute + relative * Math.abs(expected));
}

/** Deterministic 32-bit PRNG returning values in `[0, 1)`. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Flat polygon or path layout used by the specs: vertices, ring offsets and feature ring offsets. */
export type FlatRings = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  featureRingOffsets: Uint32Array;
};

/** Flattens `features[feature][ring][vertex] = [x, y]`. */
export function flattenFeatures(features: number[][][][]): FlatRings {
  const positions: number[] = [];
  const ringOffsets = [0];
  const featureRingOffsets = [0];
  for (const rings of features) {
    for (const ring of rings) {
      for (const [x, y] of ring) {
        positions.push(x, y);
      }
      ringOffsets.push(positions.length / 2);
    }
    featureRingOffsets.push(ringOffsets.length - 1);
  }
  return {
    positions: new Float32Array(positions),
    ringOffsets: new Uint32Array(ringOffsets),
    featureRingOffsets: new Uint32Array(featureRingOffsets)
  };
}
