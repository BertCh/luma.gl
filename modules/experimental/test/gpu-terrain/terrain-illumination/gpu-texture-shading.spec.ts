// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUTextureShadingParameterValues,
  GPUTextureShading,
  type GPUTextureShadingSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-texture-shading';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSmoothTerrain} from './terrain-horizon-oracle';
import {computeTextureShading} from './texture-shading-oracle';
import {expectClose} from '../terrain-test-utils';

const WIDTH = 64;
const HEIGHT = 48;
const PIXEL_COUNT = WIDTH * HEIGHT;

function getWeights(settings: GPUTextureShadingSettings): number[] {
  return Array.from(getGPUTextureShadingParameterValues(settings).subarray(1, 9));
}

async function runTextureShading(
  device: Device,
  elevation: Float32Array,
  levelCount: number,
  baseSigma: number,
  settingsList: GPUTextureShadingSettings[],
  options: {downsampleLevels?: boolean; hasNodata?: boolean; rangeTolerance?: number} = {}
): Promise<void> {
  const elevationBuffer = createInputBuffer(device, elevation);
  const outputBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const validityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const settings = new GPUParameterBuffer(device, {
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
      downsampleLevels: options.downsampleLevels ?? false,
      hasNodata: options.hasNodata,
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
    const actual = await readFloat32(outputBuffer, PIXEL_COUNT);
    if (options.rangeTolerance === undefined) {
      // Elevations span ~200 m; float32 cascades keep ~1e-4 m agreement.
      expectClose(actual, expected, 2e-3);
    } else {
      const finite = expected.filter(Number.isFinite);
      const range = Math.max(...finite) - Math.min(...finite);
      let maximumDifference = 0;
      for (let pixel = 0; pixel < PIXEL_COUNT; pixel++) {
        expect(Number.isNaN(actual[pixel])).toBe(Number.isNaN(expected[pixel]));
        if (Number.isFinite(expected[pixel])) {
          maximumDifference = Math.max(
            maximumDifference,
            Math.abs(actual[pixel] - expected[pixel])
          );
        }
      }
      expect(maximumDifference).toBeLessThan(options.rangeTolerance * range);
    }
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

it('GPUTextureShading downsampled levels stay within 1% of the exact output, with a nodata hole', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const solid = createSmoothTerrain(WIDTH, HEIGHT, 13);
  const holed = solid.slice();
  for (let row = 18; row < 30; row++) {
    for (let column = 20; column < 34; column++) {
      holed[row * WIDTH + column] = NaN;
    }
  }
  for (const elevation of [solid, holed]) {
    for (const levelCount of [4, 6]) {
      await runTextureShading(device, elevation, levelCount, 1, [{detail: 0.5}, {detail: 1}], {
        downsampleLevels: true,
        rangeTolerance: 0.01
      });
    }
  }
});

it('GPUTextureShading hasNodata false is exact on all-valid terrain', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSmoothTerrain(WIDTH, HEIGHT, 5);
  await runTextureShading(device, elevation, 6, 1, [{detail: 0.5}], {hasNodata: false});
  await runTextureShading(device, elevation, 6, 1, [{detail: 0.5}], {
    hasNodata: false,
    downsampleLevels: true,
    rangeTolerance: 0.01
  });
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
  const settings = new GPUParameterBuffer(device, {
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
