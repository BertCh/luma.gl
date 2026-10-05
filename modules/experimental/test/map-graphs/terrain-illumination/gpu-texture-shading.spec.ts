// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUTextureShadingParameterValues,
  GPUTextureShading,
  type GPUTextureShadingSettings
} from '../../../src/map-graphs/terrain-illumination/gpu-texture-shading';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {createSmoothTerrain} from './terrain-horizon-oracle';
import {computeTextureShading} from './texture-shading-oracle';

const WIDTH = 64;
const HEIGHT = 48;
const PIXEL_COUNT = WIDTH * HEIGHT;

function expectClose(actual: number[], expected: number[], tolerance: number): void {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `index ${index}`).toBe(true);
      continue;
    }
    expect(Number.isNaN(actual[index]), `index ${index} expected ${value}`).toBe(false);
    worst = Math.max(worst, Math.abs(actual[index] - value));
  }
  expect(worst).toBeLessThan(tolerance);
}

function getWeights(settings: GPUTextureShadingSettings): number[] {
  return Array.from(getGPUTextureShadingParameterValues(settings).subarray(1, 9));
}

async function runTextureShading(
  device: Device,
  elevation: Float32Array,
  levelCount: number,
  baseSigma: number,
  settingsList: GPUTextureShadingSettings[]
): Promise<void> {
  const elevationBuffer = createInputBuffer(device, elevation);
  const outputBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const validityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const settings = new GPUMapGraphParameterBuffer(device, {
    id: 'texture-shading-settings',
    format: 'float32',
    length: 12,
    values: getGPUTextureShadingParameterValues(settingsList[0])
  });
  const graph = new GPUCommandGraph(device, {id: 'texture-shading-test'});
  graph.add(
    new GPUTextureShading({
      width: WIDTH,
      height: HEIGHT,
      levelCount,
      baseSigma,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      settings: settings.importToGraph(graph),
      textureShade: importGraphBuffer(graph, 'texture-shade', outputBuffer, 'float32', PIXEL_COUNT),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', PIXEL_COUNT)
    })
  );
  const compiled = graph.compile();
  const compileCount = 1;
  for (const values of settingsList) {
    settings.write(getGPUTextureShadingParameterValues(values));
    submitGraph(device, compiled, undefined);
    const expected = computeTextureShading({
      width: WIDTH,
      height: HEIGHT,
      elevation,
      levelCount,
      baseSigma,
      gain: values.gain ?? 1,
      weights: getWeights(values)
    });
    // Elevations span ~200 m; float32 cascades keep ~1e-4 m agreement.
    expectClose(await readFloat32(outputBuffer, PIXEL_COUNT), expected, 2e-3);
    expect(await readUint32(validityBuffer, PIXEL_COUNT)).toEqual(
      expected.map(value => (Number.isNaN(value) ? 0 : 1))
    );
  }
  expect(compileCount).toBe(1);
  compiled.destroy();
  settings.destroy();
  for (const buffer of [elevationBuffer, outputBuffer, validityBuffer]) buffer.destroy();
}

it('GPUTextureShading matches the normalized-convolution oracle with per-frame weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSmoothTerrain(WIDTH, HEIGHT, 13);
  for (const hole of [10 * WIDTH + 10, 10 * WIDTH + 11, 11 * WIDTH + 10, 30 * WIDTH + 50]) {
    elevation[hole] = NaN;
  }
  await runTextureShading(device, elevation, 4, 1, [
    {detail: 0.5},
    {detail: 1, gain: 3},
    {levelWeights: [0, 1, 0, -0.5], gain: 0.25}
  ]);
  await runTextureShading(device, elevation, 1, 1.5, [{detail: 0.5}]);
  await runTextureShading(device, elevation, 6, 1, [{detail: 0.7}]);
});

it('GPUTextureShading returns zero on flat terrain and writes a texture', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const flat = new Float32Array(PIXEL_COUNT).fill(321);
  await runTextureShading(device, flat, 3, 1, [{detail: 0.5}]);
  if (!device.getTextureFormatCapabilities('r32float').store) {
    return;
  }
  const elevation = createSmoothTerrain(WIDTH, HEIGHT, 2);
  const elevationBuffer = createInputBuffer(device, elevation);
  const readbackBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const readbackValidity = createOutputBuffer(device, PIXEL_COUNT);
  const texture = device.createTexture({
    format: 'r32float',
    width: WIDTH,
    height: HEIGHT,
    usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_DST
  });
  const settings = new GPUMapGraphParameterBuffer(device, {
    id: 'texture-settings',
    format: 'float32',
    length: 12,
    values: getGPUTextureShadingParameterValues({detail: 0.5, gain: 2})
  });
  const graph = new GPUCommandGraph(device, {id: 'texture-shading-texture'});
  const textureView = graph.createTextureView(
    graph.importTexture(
      {
        id: 'shade-texture',
        format: 'r32float',
        width: WIDTH,
        height: HEIGHT,
        usage: texture.props.usage
      },
      texture
    ),
    {mipLevelCount: 1}
  ) as never;
  graph.add(
    new GPUTextureShading({
      width: WIDTH,
      height: HEIGHT,
      levelCount: 3,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      settings: settings.importToGraph(graph),
      textureShadeTexture: textureView
    })
  );
  new GPURasterTextureToBuffer({
    id: 'shade-readback',
    input: {
      id: 'shade-band',
      format: 'float32',
      storage: {kind: 'texture', view: textureView}
    },
    output: importGraphBuffer(graph, 'readback', readbackBuffer, 'float32', PIXEL_COUNT),
    outputValidity: importGraphBuffer(
      graph,
      'readback-validity',
      readbackValidity,
      'uint32',
      PIXEL_COUNT
    )
  }).addToGraph(graph);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expectClose(
    await readFloat32(readbackBuffer, PIXEL_COUNT),
    computeTextureShading({
      width: WIDTH,
      height: HEIGHT,
      elevation,
      levelCount: 3,
      baseSigma: 1,
      gain: 2,
      weights: getWeights({detail: 0.5})
    }),
    4e-3
  );
  compiled.destroy();
  settings.destroy();
  for (const resource of [elevationBuffer, readbackBuffer, readbackValidity, texture]) {
    resource.destroy();
  }
});
