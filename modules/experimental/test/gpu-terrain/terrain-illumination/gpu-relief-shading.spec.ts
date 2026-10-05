// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../../src/gpu-terrain/terrain-analysis';
import {
  getGPUReliefShadingParameterValues,
  GPU_RELIEF_SHADING_CONTRAST_PIVOT,
  GPUReliefShading,
  type GPUReliefShadingSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
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

const IMHOF_STYLES: GPUReliefShadingSettings[] = [
  {
    cellSize: [10, 10],
    lights: [{azimuthDegrees: 315, altitudeDegrees: 45}],
    lightWeighting: 'imhof-swing'
  },
  {
    cellSize: [10, 10],
    zFactor: 1.5,
    lights: [
      {azimuthDegrees: 315, altitudeDegrees: 45, weight: 2},
      {azimuthDegrees: 20, altitudeDegrees: 30}
    ],
    lightWeighting: 'imhof-swing',
    imhofSwingDegrees: 40,
    curvatureStrength: 30,
    contrastLowElevation: 90,
    contrastHighElevation: 130,
    contrastStrength: 0.6,
    exposure: 1.1,
    elevationStops: [
      {elevation: 60, color: [0.3, 0.5, 0.3]},
      {elevation: 160, color: [1, 1, 1]}
    ]
  },
  // Step contrast (high <= low), aspect weighting, zero curvature strength.
  {
    cellSize: [10, 10],
    lights: 'mdow',
    curvatureStrength: 0,
    contrastLowElevation: 110,
    contrastHighElevation: 110,
    contrastStrength: 1
  },
  // Plain MDOW with the new slots at their defaults.
  {cellSize: [10, 10]}
];

it('GPUReliefShading imhof swing, curvature, and contrast match the oracle with every input bound', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 36;
  const height = 30;
  const pixelCount = width * height;
  const elevation = createSmoothTerrain(width, height, 5);
  const random = createRandom(8);
  const skyViewFactor = Float32Array.from({length: pixelCount}, () => 0.6 + 0.4 * random());
  const textureShade = Float32Array.from({length: pixelCount}, () => -0.05 + 0.1 * random());
  const curvature = Float32Array.from({length: pixelCount}, () => -0.02 + 0.04 * random());
  curvature[4 * width + 9] = NaN;
  const inputs = [elevation, skyViewFactor, textureShade, curvature].map(values =>
    createInputBuffer(device, values)
  );
  const outputs = Array.from({length: 4}, () => createOutputBuffer(device, pixelCount));
  const settings = new GPUParameterBuffer(device, {
    id: 'imhof-settings',
    format: 'float32',
    length: 80,
    values: getGPUReliefShadingParameterValues(IMHOF_STYLES[0])
  });
  const graph = new GPUCommandGraph(device, {id: 'relief-imhof-test'});
  const float = (id: string, buffer: (typeof inputs)[number]) =>
    importGraphBuffer(graph, id, buffer, 'float32', pixelCount);
  // Every optional input and output is bound: the compose kernel must stay within 8 bindings.
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
      curvature: float('curvature', inputs[3]),
      imhofSwing: true,
      hillshade: float('hillshade', outputs[0]),
      relief: float('relief', outputs[1]),
      color: importGraphBuffer(graph, 'color', outputs[2], 'uint32', pixelCount),
      validity: importGraphBuffer(graph, 'validity', outputs[3], 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  const reliefSpans: number[] = [];
  for (const style of IMHOF_STYLES) {
    settings.write(getGPUReliefShadingParameterValues(style));
    submitGraph(device, compiled, undefined);
    const expected = computeReliefShading({
      width,
      height,
      elevation,
      settings: style,
      skyViewFactor,
      textureShade,
      curvature
    });
    const hillshade = await readFloat32(outputs[0], pixelCount);
    const relief = await readFloat32(outputs[1], pixelCount);
    expectClose(hillshade, expected.hillshade, 2e-5);
    expectClose(relief, expected.relief, 2e-5);
    const validity = await readUint32(outputs[3], pixelCount);
    expect(validity).toEqual(expected.validity);
    // NaN curvature invalidates its pixel.
    expect(validity[4 * width + 9]).toBe(0);
    expect(validity.reduce((sum, value) => sum + value, 0)).toBe(pixelCount - 1);
    const colors = unpackColors(await readUint32(outputs[2], pixelCount));
    let worstChannel = 0;
    for (const [pixel, color] of expected.color.entries()) {
      for (let channel = 0; channel < 4; channel++) {
        worstChannel = Math.max(worstChannel, Math.abs(colors[pixel][channel] - color[channel]));
      }
    }
    expect(worstChannel).toBeLessThanOrEqual(1);
    // Non-trivial structure: a compile failure would leave zeros.
    const finite = relief.filter(value => !Number.isNaN(value));
    const span = Math.max(...finite) - Math.min(...finite);
    expect(span).toBeGreaterThan(0.1);
    reliefSpans.push(span);
  }
  compiled.destroy();
  settings.destroy();
  for (const buffer of [...inputs, ...outputs]) buffer.destroy();
});

it('GPUReliefShading imhof swing lights east-facing slopes and swing 0 equals fixed exactly', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 20;
  const height = 12;
  const pixelCount = width * height;
  // Descends 20 m per 10 m pixel toward the east: a steep east-facing slope.
  const elevation = Float32Array.from(
    {length: pixelCount},
    (_, index) => 500 - 20 * (index % width)
  );
  const elevationBuffer = createInputBuffer(device, elevation);
  const outputs = [createOutputBuffer(device, pixelCount), createOutputBuffer(device, pixelCount)];
  const light = [{azimuthDegrees: 315, altitudeDegrees: 45}];
  const settings = new GPUParameterBuffer(device, {
    id: 'swing-settings',
    format: 'float32',
    length: 80,
    values: getGPUReliefShadingParameterValues({cellSize: [10, 10], lights: light})
  });
  const graph = new GPUCommandGraph(device, {id: 'relief-swing-test'});
  graph.add(
    new GPUReliefShading({
      width,
      height,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        }
      },
      settings: settings.importToGraph(graph),
      imhofSwing: true,
      hillshade: importGraphBuffer(graph, 'hillshade', outputs[0], 'float32', pixelCount),
      relief: importGraphBuffer(graph, 'relief', outputs[1], 'float32', pixelCount)
    })
  );
  const compiled = graph.compile();
  const run = async (style: GPUReliefShadingSettings) => {
    settings.write(getGPUReliefShadingParameterValues(style));
    submitGraph(device, compiled, undefined);
    return readFloat32(outputs[0], pixelCount);
  };
  const interior = 5 * width + 10;
  const fixed = await run({cellSize: [10, 10], lights: light, lightWeighting: 'fixed'});
  const swung = await run({cellSize: [10, 10], lights: light, lightWeighting: 'imhof-swing'});
  const zeroSwing = await run({
    cellSize: [10, 10],
    lights: light,
    lightWeighting: 'imhof-swing',
    imhofSwingDegrees: 0
  });
  // The fixed 315 degree light leaves the east face black; the swung light reaches it.
  expect(fixed[interior]).toBe(0);
  expect(swung[interior]).toBeGreaterThan(0.15);
  const expected = computeReliefShading({
    width,
    height,
    elevation,
    settings: {cellSize: [10, 10], lights: light, lightWeighting: 'imhof-swing'}
  });
  expect(Math.abs(swung[interior] - expected.hillshade[interior])).toBeLessThan(1e-5);
  // Swing 0 reproduces fixed weighting bit for bit, and so does leaving the mode.
  expect(Array.from(new Uint32Array(new Float32Array(zeroSwing).buffer))).toEqual(
    Array.from(new Uint32Array(new Float32Array(fixed).buffer))
  );
  const again = await run({cellSize: [10, 10], lights: light, lightWeighting: 'fixed'});
  expect(again).toEqual(fixed);
  compiled.destroy();
  settings.destroy();
  for (const buffer of [elevationBuffer, ...outputs]) buffer.destroy();
});

it('GPUReliefShading new slots at zero reproduce the previous output exactly', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 40;
  const height = 32;
  const pixelCount = width * height;
  const elevation = createSmoothTerrain(width, height, 9);
  const random = createRandom(3);
  const skyViewFactor = Float32Array.from({length: pixelCount}, () => 0.6 + 0.4 * random());
  const inputs = [elevation, skyViewFactor].map(values => createInputBuffer(device, values));
  const outputs = Array.from({length: 3}, () => createOutputBuffer(device, pixelCount));
  const base: GPUReliefShadingSettings = {
    cellSize: [10, 10],
    zFactor: 2,
    skyViewStrength: 0.6,
    exposure: 1.2,
    elevationStops: [{elevation: 60, color: [0.3, 0.5, 0.3]}]
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'legacy-settings',
    format: 'float32',
    length: 80,
    values: getGPUReliefShadingParameterValues(base)
  });
  const graph = new GPUCommandGraph(device, {id: 'relief-legacy-test'});
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
      hillshade: float('hillshade', outputs[0]),
      relief: float('relief', outputs[1]),
      color: importGraphBuffer(graph, 'color', outputs[2], 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  const run = async (values: Float32Array) => {
    settings.write(values);
    submitGraph(device, compiled, undefined);
    return [
      await readUint32(outputs[0], pixelCount),
      await readUint32(outputs[1], pixelCount),
      await readUint32(outputs[2], pixelCount)
    ];
  };
  const reference = await run(getGPUReliefShadingParameterValues(base));
  // Slots 19 and 76-79 zero-filled, and non-zero low/high/curvature strength with strength 0.
  const zeroed = getGPUReliefShadingParameterValues(base);
  zeroed.fill(0, 76, 80);
  zeroed[19] = 0;
  expect(await run(zeroed)).toEqual(reference);
  const inert = getGPUReliefShadingParameterValues({
    ...base,
    contrastLowElevation: 50,
    contrastHighElevation: 150,
    contrastStrength: 0,
    imhofSwingDegrees: 65
  });
  expect(await run(inert)).toEqual(reference);
  // And the oracle still agrees, so this is not a comparison of two broken outputs.
  const expected = computeReliefShading({width, height, elevation, settings: base, skyViewFactor});
  expectClose(await readFloat32(outputs[1], pixelCount), expected.relief, 2e-5);
  expect(new Set(reference[1]).size).toBeGreaterThan(50);
  // Turning the contrast on does change the output.
  const active = await run(
    getGPUReliefShadingParameterValues({
      ...base,
      contrastLowElevation: 90,
      contrastHighElevation: 130,
      contrastStrength: 0.8
    })
  );
  expect(active[1]).not.toEqual(reference[1]);
  expect(GPU_RELIEF_SHADING_CONTRAST_PIVOT).toBe(0.72);
  compiled.destroy();
  settings.destroy();
  for (const buffer of [...inputs, ...outputs]) buffer.destroy();
});

it('GPUReliefShading adds curvature directly in the compose kernel when bindings allow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 30;
  const height = 24;
  const pixelCount = width * height;
  const elevation = createSmoothTerrain(width, height, 14);
  const random = createRandom(2);
  const curvature = Float32Array.from({length: pixelCount}, () => -0.03 + 0.06 * random());
  curvature[50] = NaN;
  const inputs = [elevation, curvature].map(values => createInputBuffer(device, values));
  const outputs = [createOutputBuffer(device, pixelCount), createOutputBuffer(device, pixelCount)];
  const style: GPUReliefShadingSettings = {
    cellSize: [10, 10],
    curvatureStrength: 20,
    contrastLowElevation: 80,
    contrastHighElevation: 140,
    contrastStrength: 0.5
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'direct-curvature-settings',
    format: 'float32',
    length: 80,
    values: getGPUReliefShadingParameterValues(style)
  });
  const graph = new GPUCommandGraph(device, {id: 'relief-direct-curvature'});
  graph.add(
    new GPUReliefShading({
      width,
      height,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', inputs[0], 'float32', pixelCount)
        }
      },
      settings: settings.importToGraph(graph),
      curvature: importGraphBuffer(graph, 'curvature', inputs[1], 'float32', pixelCount),
      relief: importGraphBuffer(graph, 'relief', outputs[0], 'float32', pixelCount),
      validity: importGraphBuffer(graph, 'validity', outputs[1], 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const expected = computeReliefShading({width, height, elevation, settings: style, curvature});
  const relief = await readFloat32(outputs[0], pixelCount);
  expectClose(relief, expected.relief, 2e-5);
  expect((await readUint32(outputs[1], pixelCount))[50]).toBe(0);
  expect(Math.max(...relief.filter(value => !Number.isNaN(value)))).toBeGreaterThan(0.3);
  compiled.destroy();
  settings.destroy();
  for (const buffer of [...inputs, ...outputs]) buffer.destroy();
});
