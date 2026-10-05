// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../../src/gpu-terrain/terrain-analysis';
import {
  getGPUReliefShadingParameterValues,
  GPUReliefShading,
  type GPUReliefShadingSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom, createSmoothTerrain} from './terrain-horizon-oracle';
import {computeReliefShading, unpackColors} from './relief-shading-oracle';

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

const STYLES: GPUReliefShadingSettings[] = [
  {cellSize: [10, 10]},
  {
    cellSize: [10, 10],
    zFactor: 2,
    lights: [
      {azimuthDegrees: 315, altitudeDegrees: 45, weight: 3},
      {azimuthDegrees: 45, altitudeDegrees: 60},
      {azimuthDegrees: 180, altitudeDegrees: 20, weight: 0.5}
    ],
    hillshadeStrength: 0.8,
    skyViewStrength: 0.6,
    textureShadeStrength: 0.02,
    exposure: 1.2,
    tintStrength: 0.5,
    elevationStops: [
      {elevation: 60, color: [0.3, 0.5, 0.3]},
      {elevation: 110, color: [0.8, 0.75, 0.6]},
      {elevation: 160, color: [1, 1, 1]}
    ]
  },
  {
    cellSize: [12, 8],
    lights: 'mdow',
    lightWeighting: 'fixed',
    warmColor: [1, 0.8, 0.6],
    coolColor: [0.6, 0.7, 1],
    elevationStops: [{elevation: 100, color: [0.9, 0.85, 0.7]}]
  }
];

it('GPUReliefShading matches the oracle across per-frame styles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 40;
  const height = 32;
  const pixelCount = width * height;
  const elevation = createSmoothTerrain(width, height, 9);
  elevation[10 * width + 12] = NaN;
  const random = createRandom(3);
  const skyViewFactor = Float32Array.from({length: pixelCount}, () => 0.6 + 0.4 * random());
  skyViewFactor[5] = NaN;
  const textureShade = Float32Array.from({length: pixelCount}, () => -20 + 40 * random());
  textureShade[7 * width + 30] = NaN;
  const inputs = [elevation, skyViewFactor, textureShade].map(values =>
    createInputBuffer(device, values)
  );
  const outputs = [
    createOutputBuffer(device, pixelCount),
    createOutputBuffer(device, pixelCount),
    createOutputBuffer(device, pixelCount),
    createOutputBuffer(device, pixelCount)
  ];
  const settings = new GPUParameterBuffer(device, {
    id: 'relief-settings',
    format: 'float32',
    length: 80,
    values: getGPUReliefShadingParameterValues(STYLES[0])
  });
  const graph = new GPUCommandGraph(device, {id: 'relief-shading-test'});
  const float = (id: string, buffer: (typeof inputs)[number]) =>
    importGraphBuffer(graph, id, buffer, 'float32', pixelCount);
  graph.add(
    new GPUReliefShading({
      width,
      height,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: float('elevation', inputs[0])}
      },
      settings: settings.importToGraph(graph),
      skyViewFactor: float('svf', inputs[1]),
      textureShade: float('texture-shade', inputs[2]),
      hillshade: float('hillshade', outputs[0]),
      relief: float('relief', outputs[1]),
      color: importGraphBuffer(graph, 'color', outputs[2], 'uint32', pixelCount),
      validity: importGraphBuffer(graph, 'validity', outputs[3], 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  const compileCount = 1;
  for (const style of STYLES) {
    settings.write(getGPUReliefShadingParameterValues(style));
    submitGraph(device, compiled, undefined);
    const expected = computeReliefShading({
      width,
      height,
      elevation,
      settings: style,
      skyViewFactor,
      textureShade
    });
    expectClose(await readFloat32(outputs[0], pixelCount), expected.hillshade, 1e-5);
    expectClose(await readFloat32(outputs[1], pixelCount), expected.relief, 1e-5);
    expect(await readUint32(outputs[3], pixelCount)).toEqual(expected.validity);
    const colors = unpackColors(await readUint32(outputs[2], pixelCount));
    let worstChannel = 0;
    for (const [pixel, color] of expected.color.entries()) {
      for (let channel = 0; channel < 4; channel++) {
        worstChannel = Math.max(worstChannel, Math.abs(colors[pixel][channel] - color[channel]));
      }
    }
    expect(worstChannel).toBeLessThanOrEqual(1);
  }
  expect(compileCount).toBe(1);
  compiled.destroy();
  settings.destroy();
  for (const buffer of [...inputs, ...outputs]) buffer.destroy();
});

it('GPUReliefShading with one light reproduces GPUTerrainDerivatives', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 24;
  const height = 20;
  const pixelCount = width * height;
  const elevation = createSmoothTerrain(width, height, 4);
  const elevationBuffer = createInputBuffer(device, elevation);
  const [reliefHillshade, derivativeHillshade] = [
    createOutputBuffer(device, pixelCount),
    createOutputBuffer(device, pixelCount)
  ];
  const parameterBuffers = [
    new GPUParameterBuffer(device, {
      id: 'relief-settings',
      format: 'float32',
      length: 80,
      values: getGPUReliefShadingParameterValues({
        cellSize: [10, 10],
        zFactor: 1.5,
        lights: [{azimuthDegrees: 300, altitudeDegrees: 35}],
        northEdge: 40,
        southEdge: 39.9
      })
    }),
    new GPUParameterBuffer(device, {
      id: 'derivative-settings',
      format: 'float32',
      length: 8,
      values: getGPUTerrainDerivativesParameterValues({
        cellSize: [10, 10],
        zFactor: 1.5,
        azimuthDegrees: 300,
        altitudeDegrees: 35,
        northEdge: 40,
        southEdge: 39.9
      })
    })
  ];
  const graph = new GPUCommandGraph(device, {id: 'relief-vs-derivatives'});
  const elevationBand = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
    }
  };
  graph.add(
    new GPUReliefShading({
      width,
      height,
      elevation: elevationBand,
      settings: parameterBuffers[0].importToGraph(graph),
      cellSizeMode: 'geographic',
      rowDirection: 'north',
      hillshade: importGraphBuffer(
        graph,
        'relief-hillshade',
        reliefHillshade,
        'float32',
        pixelCount
      )
    })
  );
  graph.add(
    new GPUTerrainDerivatives({
      width,
      height,
      elevation: elevationBand,
      settings: parameterBuffers[1].importToGraph(graph),
      cellSizeMode: 'geographic',
      rowDirection: 'north',
      hillshade: importGraphBuffer(
        graph,
        'derivative-hillshade',
        derivativeHillshade,
        'float32',
        pixelCount
      )
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expectClose(
    await readFloat32(reliefHillshade, pixelCount),
    await readFloat32(derivativeHillshade, pixelCount),
    1e-6
  );
  compiled.destroy();
  for (const buffer of parameterBuffers) buffer.destroy();
  for (const buffer of [elevationBuffer, reliefHillshade, derivativeHillshade]) buffer.destroy();
});
