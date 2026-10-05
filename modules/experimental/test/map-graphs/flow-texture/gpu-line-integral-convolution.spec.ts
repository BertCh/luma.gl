// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  convolveLineIntegralOnCPU,
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionWordParameterValues,
  GPULineIntegralConvolution,
  type GPULineIntegralConvolutionSettings
} from '../../../src/map-graphs/flow-texture';
import {createInputBuffer, createOutputBuffer, readFloat32} from '../map-graph-test-utils';
import {createVortexField, type TestField} from './flow-texture-scenes';

type Result = {
  values: Float32Array;
  speeds: Float32Array;
  texture: Float32Array;
};

type Fixture = {
  run(settings: GPULineIntegralConvolutionSettings): Promise<Result>;
  destroy(): void;
};

function createFixture(
  device: Device,
  field: TestField,
  width: number,
  height: number,
  stepCount: number
): Fixture {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'lic-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'lic-parameters',
    format: 'float32',
    length: 12
  });
  const wordBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'lic-words',
    format: 'uint32',
    length: 4
  });
  const valuesBuffer = track(createOutputBuffer(device, pixelCount));
  const speedsBuffer = track(createOutputBuffer(device, pixelCount));
  const readbackBuffer = track(createOutputBuffer(device, pixelCount));
  const validityBuffer = track(createOutputBuffer(device, pixelCount));
  const texture = device.createTexture({
    format: 'r32float',
    width,
    height,
    usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
  });
  const textureView = graph.createTextureView(
    graph.importTexture(
      {
        id: 'lic-texture',
        format: 'r32float',
        width,
        height,
        usage: texture.props.usage
      },
      texture
    ),
    {mipLevelCount: 1}
  );
  graph.add(
    new GPULineIntegralConvolution({
      id: 'lic',
      velocities: importGraphBuffer(
        graph,
        'field',
        track(createInputBuffer(device, field.velocities)),
        'float32x2',
        field.width * field.height
      ),
      fieldWidth: field.width,
      fieldHeight: field.height,
      width,
      height,
      stepCount,
      parameters: parameterBuffer.importToGraph(graph),
      wordParameters: wordBuffer.importToGraph(graph),
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pixelCount),
        speeds: importGraphBuffer(graph, 'speeds', speedsBuffer, 'float32', pixelCount),
        texture: textureView as never
      }
    })
  );
  new GPURasterTextureToBuffer({
    id: 'texture-readback',
    input: {
      id: 'texture-band',
      format: 'float32',
      storage: {kind: 'texture', view: textureView}
    },
    output: importGraphBuffer(graph, 'texture-readback', readbackBuffer, 'float32', pixelCount),
    outputValidity: importGraphBuffer(
      graph,
      'texture-validity',
      validityBuffer,
      'uint32',
      pixelCount
    )
  }).addToGraph(graph);
  const compiled = graph.compile();
  return {
    async run(settings) {
      parameterBuffer.write(getGPULineIntegralConvolutionParameterValues(settings));
      wordBuffer.write(getGPULineIntegralConvolutionWordParameterValues(settings));
      submitGraph(device, compiled, undefined);
      return {
        values: Float32Array.from(await readFloat32(valuesBuffer, pixelCount)),
        speeds: Float32Array.from(await readFloat32(speedsBuffer, pixelCount)),
        texture: Float32Array.from(await readFloat32(readbackBuffer, pixelCount))
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      wordBuffer.destroy();
      texture.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Fraction of pixels whose GPU value differs from the oracle by more than `tolerance`. */
function getMismatchFraction(actual: Float32Array, expected: Float32Array, tolerance: number) {
  let mismatches = 0;
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) {
      expect(Number.isNaN(actual[index])).toBe(true);
    } else if (!(Math.abs(actual[index] - expected[index]) <= tolerance)) {
      mismatches++;
    }
  }
  return mismatches / expected.length;
}

it('GPULineIntegralConvolution matches the CPU oracle and writes the texture', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 32 x 32 vortex over [0, 32]^2 with a NaN cell, rendered at 64 x 64.
  const field = createVortexField(32, 0.1);
  field.velocities[2 * (5 * 32 + 5)] = Number.NaN;
  const fixture = createFixture(device, field, 64, 64, 12);
  try {
    for (const settings of [
      {fieldExtent: [0, 0, 1, 1], outputExtent: [0, 0, 0.5, 0.5], seed: 3},
      {
        fieldExtent: [0, 0, 1, 1],
        outputExtent: [0, 0, 0.5, 0.5],
        seed: 3,
        period: 6,
        phase: 0.25,
        minimumSpeed: 0.2
      }
    ] satisfies GPULineIntegralConvolutionSettings[]) {
      const actual = await fixture.run(settings);
      const expected = convolveLineIntegralOnCPU(
        field,
        64,
        64,
        12,
        getGPULineIntegralConvolutionParameterValues(settings),
        settings.seed
      );
      // cos/normalize rounding may move a rare sample across a pixel edge; bound how often.
      expect(getMismatchFraction(actual.values, expected.values, 1e-4)).toBeLessThan(0.02);
      expect(getMismatchFraction(actual.speeds, expected.speeds, 1e-4)).toBe(0);
      expect(Array.from(actual.texture)).toEqual(Array.from(actual.values));
      // The NaN cell blanks the pixels whose bilinear footprint touches it.
      expect(Number.isNaN(actual.values[10 * 64 + 10])).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});

it('GPULineIntegralConvolution animates by phase without rebuilding and replays bitwise', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const field = createVortexField(16, 0.2);
  const fixture = createFixture(device, field, 32, 32, 8);
  try {
    const base: GPULineIntegralConvolutionSettings = {
      fieldExtent: [0, 0, 1, 1],
      outputExtent: [0, 0, 0.5, 0.5],
      seed: 1,
      period: 4
    };
    const first = await fixture.run({...base, phase: 0});
    const moved = await fixture.run({...base, phase: 0.5});
    const again = await fixture.run({...base, phase: 0});
    expect(new Uint32Array(again.values.buffer)).toEqual(new Uint32Array(first.values.buffer));
    let changed = 0;
    for (let index = 0; index < first.values.length; index++) {
      changed += Math.abs(first.values[index] - moved.values[index]) > 1e-3 ? 1 : 0;
    }
    expect(changed).toBeGreaterThan(first.values.length / 2);
    // A new seed changes the noise on the same graph.
    const reseeded = await fixture.run({...base, phase: 0, seed: 2});
    expect(reseeded.values).not.toEqual(first.values);
  } finally {
    fixture.destroy();
  }
});
