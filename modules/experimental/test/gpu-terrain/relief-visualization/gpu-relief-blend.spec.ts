// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUReliefBlendParameterValues,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
  GPU_RELIEF_BLEND_VAT_FLAT,
  GPUReliefBlend,
  type GPUReliefBlendLayerSettings,
  type GPUReliefBlendMode
} from '../../../src/gpu-terrain/relief-visualization/gpu-relief-blend';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeBlendRVT, type RVTBlendLayer} from './relief-visualization-oracle';
import {expectClose} from './relief-test-utils';

const WIDTH = 32;
const HEIGHT = 24;
const PIXEL_COUNT = WIDTH * HEIGHT;

/** Deterministic layer images covering and exceeding the stretch ranges. */
function createLayer(seed: number, minimum: number, maximum: number): Float32Array {
  const span = maximum - minimum;
  return Float32Array.from({length: PIXEL_COUNT}, (_, index) => {
    const x = index % WIDTH;
    const y = Math.floor(index / WIDTH);
    const wave = Math.sin(0.31 * x * seed + 0.17 * y) * Math.cos(0.11 * y * seed - 0.23 * x);
    return minimum + span * (0.5 + 0.7 * wave);
  });
}

function toRVTLayers(
  bottomToTop: readonly GPUReliefBlendLayerSettings[],
  images: Float32Array[]
): RVTBlendLayer[] {
  const modeMap: Record<GPUReliefBlendMode, RVTBlendLayer['blendMode']> = {
    normal: 'normal',
    multiply: 'multiply',
    screen: 'screen',
    overlay: 'overlay',
    'soft-light': 'soft_light',
    luminosity: 'luminosity'
  };
  // RVT lists the top layer first.
  return bottomToTop
    .map((layer, index) => ({
      image: images[index],
      minimum: layer.minimum,
      maximum: layer.maximum,
      invert: layer.invert ?? false,
      blendMode: modeMap[layer.blendMode ?? 'normal'],
      opacity: layer.opacity ?? 1
    }))
    .reverse();
}

async function runBlend(
  device: Device,
  images: Float32Array[],
  settingsList: (readonly GPUReliefBlendLayerSettings[])[],
  check: (
    blend: number[],
    color: number[],
    validity: number[],
    layers: readonly GPUReliefBlendLayerSettings[]
  ) => void
): Promise<void> {
  const buffers: Buffer[] = [];
  const blendBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const colorBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const validityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  buffers.push(blendBuffer, colorBuffer, validityBuffer);
  const settings = new GPUParameterBuffer(device, {
    id: 'blend-settings',
    format: 'float32',
    length: 25,
    values: getGPUReliefBlendParameterValues(settingsList[0])
  });
  const graph = new GPUCommandGraph(device, {id: 'blend-test'});
  graph.add(
    new GPUReliefBlend({
      width: WIDTH,
      height: HEIGHT,
      layers: images.map((image, index) => {
        const buffer = createInputBuffer(device, image);
        buffers.push(buffer);
        return importGraphBuffer(graph, `layer-${index}`, buffer, 'float32', PIXEL_COUNT);
      }),
      settings: settings.importToGraph(graph),
      blend: importGraphBuffer(graph, 'blend', blendBuffer, 'float32', PIXEL_COUNT),
      color: importGraphBuffer(graph, 'color', colorBuffer, 'uint32', PIXEL_COUNT),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', PIXEL_COUNT)
    })
  );
  const compiled = graph.compile();
  for (const layers of settingsList) {
    settings.write(getGPUReliefBlendParameterValues(layers, new Float32Array(25)));
    submitGraph(device, compiled, undefined);
    check(
      await readFloat32(blendBuffer, PIXEL_COUNT),
      await readUint32(colorBuffer, PIXEL_COUNT),
      await readUint32(validityBuffer, PIXEL_COUNT),
      layers
    );
  }
  compiled.destroy();
  settings.destroy();
  for (const buffer of buffers) buffer.destroy();
}

it('GPUReliefBlend matches the RVT render loop for the VAT presets and every mode', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const images = [
    createLayer(1, 0, 1),
    createLayer(2, 0, 50),
    createLayer(3, 68, 93),
    createLayer(4, 0.7, 1)
  ];
  images[1][5] = NaN;
  const modeSweep = (mode: GPUReliefBlendMode): GPUReliefBlendLayerSettings[] => [
    {minimum: 0, maximum: 1},
    {minimum: 0, maximum: 50, blendMode: mode, opacity: 0.7},
    {minimum: 68, maximum: 93, blendMode: mode, opacity: 0.35, invert: true},
    {minimum: 0.7, maximum: 1, blendMode: mode, opacity: 1}
  ];
  const settingsList = [
    GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
    GPU_RELIEF_BLEND_VAT_FLAT,
    ...(['normal', 'multiply', 'screen', 'overlay', 'soft-light', 'luminosity'] as const).map(
      modeSweep
    )
  ];
  let checked = 0;
  await runBlend(device, images, settingsList, (blend, color, validity, layers) => {
    const expected = computeBlendRVT(toRVTLayers(layers, images));
    expectClose(blend, expected, 2e-5, `case ${checked++}`);
    expect(validity).toEqual(expected.map(value => (Number.isNaN(value) ? 0 : 1)));
    expect(validity[5]).toBe(0);
    expect(color[5]).toBe(0);
    const finite = blend.filter(Number.isFinite);
    expect(Math.max(...finite) - Math.min(...finite)).toBeGreaterThan(0.1);
    for (const value of finite) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    // pack4x8unorm of grey: red byte is round(255 * grey), alpha is 255.
    const sample = 100;
    const byte = Math.round(255 * blend[sample]);
    expect(color[sample]).toBe(((255 << 24) | (byte << 16) | (byte << 8) | byte) >>> 0);
  });
  expect(checked).toBe(settingsList.length);
});

it('GPUReliefBlend uses the bottom layer as is and analytic overlay and soft light values', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const bottom = new Float32Array(PIXEL_COUNT).fill(0.8);
  const top = new Float32Array(PIXEL_COUNT).fill(0.25);
  await runBlend(
    device,
    [bottom, top],
    [
      [
        {minimum: 0, maximum: 1, blendMode: 'multiply', opacity: 0.1},
        {minimum: 0, maximum: 1}
      ],
      [
        {minimum: 0, maximum: 1},
        {minimum: 0, maximum: 1, blendMode: 'overlay', opacity: 1}
      ],
      [
        {minimum: 0, maximum: 1},
        {minimum: 0, maximum: 1, blendMode: 'soft-light', opacity: 1}
      ],
      [
        {minimum: 0, maximum: 1},
        {minimum: 0, maximum: 1, blendMode: 'multiply', opacity: 0.5}
      ]
    ],
    (blend, _color, _validity, layers) => {
      const mode = layers[1].blendMode;
      const expected =
        layers[0].blendMode === 'multiply'
          ? 0.25 // bottom layer: mode and opacity ignored; the top layer is normal at opacity 1
          : mode === 'overlay'
            ? 1 - (1 - 2 * (0.8 - 0.5)) * (1 - 0.25) // background above 0.5
            : mode === 'soft-light'
              ? 2 * 0.8 * 0.25 + 0.8 ** 2 * (1 - 0.5) // active below 0.5
              : 0.5 * (0.25 * 0.8) + 0.5 * 0.8;
      for (const value of blend) {
        expect(value).toBeCloseTo(expected, 5);
      }
    }
  );
});
