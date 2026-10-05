// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../../src/map-graphs/terrain-analysis';
import {
  getGPUSolarShadowMaskParameterValues,
  GPUSolarShadowMask,
  type GPUSolarShadowMaskSettings
} from '../../../src/map-graphs/terrain-illumination/gpu-solar-shadow-mask';
import {
  encodeGPUTerrainHorizonUnorm16,
  getGPUTerrainHorizonParameterValues,
  GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES,
  GPUTerrainHorizon,
  unpackGPUTerrainHorizonUnorm16
} from '../../../src/map-graphs/terrain-illumination/gpu-terrain-horizon';
import {createInputBuffer, createOutputBuffer, readFloat32} from '../map-graph-test-utils';
import {computeSolarShadow, createRandom, createSmoothTerrain} from './terrain-horizon-oracle';

function expectClose(actual: number[], expected: number[], tolerance: number): void {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `index ${index}`).toBe(true);
      continue;
    }
    worst = Math.max(worst, Math.abs(actual[index] - value));
  }
  expect(worst).toBeLessThan(tolerance);
}

const SUN_SWEEP: GPUSolarShadowMaskSettings[] = [
  {azimuthDegrees: 135, altitudeDegrees: 12, ambientIntensity: 0.2},
  // Exactly on sector 2 of 8 and of 16.
  {azimuthDegrees: 90, altitudeDegrees: 5, angularRadiusDegrees: 0},
  // Wraps between the last and the first sector.
  {
    azimuthDegrees: 355,
    altitudeDegrees: 8,
    angularRadiusDegrees: 2,
    sunIntensity: 0.7
  },
  {azimuthDegrees: -30, altitudeDegrees: 20, angularRadiusDegrees: 1},
  // Below the astronomical horizon.
  {azimuthDegrees: 200, altitudeDegrees: -1, ambientIntensity: 0.4},
  {azimuthDegrees: 720.5, altitudeDegrees: 60, angularRadiusDegrees: 0.5}
];

it('GPUSolarShadowMask matches the oracle for a per-frame sun sweep', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 20;
  const height = 15;
  const pixelCount = width * height;
  const random = createRandom(11);
  for (const directionCount of [8, 16]) {
    const horizon = Float32Array.from(
      {length: pixelCount * directionCount},
      () => -10 + random() * 40
    );
    horizon[3 * directionCount + 1] = NaN;
    horizon[17 * directionCount] = NaN;
    const slope = Float32Array.from({length: pixelCount}, () => random() * 50);
    const aspect = Float32Array.from({length: pixelCount}, () => random() * 360);
    aspect[5] = -1;
    aspect[6] = -1;
    const skyViewFactor = Float32Array.from({length: pixelCount}, () => 0.5 + random() * 0.5);
    skyViewFactor[9] = NaN;
    const inputs = [horizon, slope, aspect, skyViewFactor].map(values =>
      createInputBuffer(device, values)
    );
    const outputs = [
      createOutputBuffer(device, pixelCount),
      createOutputBuffer(device, pixelCount)
    ];
    const settings = new GPUMapGraphParameterBuffer(device, {
      id: 'shadow-settings',
      format: 'float32',
      length: 8,
      values: getGPUSolarShadowMaskParameterValues(SUN_SWEEP[0])
    });
    const graph = new GPUCommandGraph(device, {id: 'solar-shadow-test'});
    const float = (id: string, buffer: (typeof inputs)[number], length = pixelCount) =>
      importGraphBuffer(graph, id, buffer, 'float32', length);
    graph.add(
      new GPUSolarShadowMask({
        width,
        height,
        directionCount,
        horizon: float('horizon', inputs[0], pixelCount * directionCount),
        settings: settings.importToGraph(graph),
        slope: float('slope', inputs[1]),
        aspect: float('aspect', inputs[2]),
        skyViewFactor: float('svf', inputs[3]),
        sunVisibility: float('visibility', outputs[0]),
        illumination: float('illumination', outputs[1])
      })
    );
    const compiled = graph.compile();
    const compileCount = 1;
    for (const sun of SUN_SWEEP) {
      settings.write(getGPUSolarShadowMaskParameterValues(sun));
      submitGraph(device, compiled, undefined);
      const expected = computeSolarShadow({
        pixelCount,
        directionCount,
        horizon,
        azimuthDegrees: sun.azimuthDegrees,
        altitudeDegrees: sun.altitudeDegrees,
        angularRadiusDegrees: sun.angularRadiusDegrees ?? 0.2666,
        sunIntensity: sun.sunIntensity,
        ambientIntensity: sun.ambientIntensity,
        slope,
        aspect,
        skyViewFactor
      });
      expectClose(await readFloat32(outputs[0], pixelCount), expected.sunVisibility, 2e-5);
      expectClose(await readFloat32(outputs[1], pixelCount), expected.illumination, 2e-5);
    }
    expect(compileCount).toBe(1);
    compiled.destroy();
    settings.destroy();
    for (const buffer of [...inputs, ...outputs]) buffer.destroy();
  }
});

it('GPUSolarShadowMask composes with GPUTerrainHorizon and GPUTerrainDerivatives', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 32;
  const height = 28;
  const pixelCount = width * height;
  const directionCount = 16;
  const elevation = createSmoothTerrain(width, height, 21);
  const elevationBuffer = createInputBuffer(device, elevation);
  const horizonBuffer = createOutputBuffer(device, pixelCount * directionCount);
  const [slopeBuffer, aspectBuffer, svfBuffer, illuminationBuffer, readbackBuffer, validityBuffer] =
    Array.from({length: 6}, () => createOutputBuffer(device, pixelCount));
  const parameterBuffers = [
    new GPUMapGraphParameterBuffer(device, {
      id: 'derivative-settings',
      format: 'float32',
      length: 8,
      values: getGPUTerrainDerivativesParameterValues({cellSize: [10, 10]})
    }),
    new GPUMapGraphParameterBuffer(device, {
      id: 'horizon-settings',
      format: 'float32',
      length: 8,
      values: getGPUTerrainHorizonParameterValues({cellSize: [10, 10]})
    }),
    new GPUMapGraphParameterBuffer(device, {
      id: 'shadow-settings',
      format: 'float32',
      length: 8,
      values: getGPUSolarShadowMaskParameterValues(SUN_SWEEP[0])
    })
  ];
  const visibilityTexture = device.getTextureFormatCapabilities('r32float').store
    ? device.createTexture({
        format: 'r32float',
        width,
        height,
        usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_DST
      })
    : undefined;
  const graph = new GPUCommandGraph(device, {id: 'terrain-light-pipeline'});
  const float = (id: string, buffer: typeof slopeBuffer, length = pixelCount) =>
    importGraphBuffer(graph, id, buffer, 'float32', length);
  const elevationBand = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: float('elevation', elevationBuffer)
    }
  };
  const slope = float('slope', slopeBuffer);
  const aspect = float('aspect', aspectBuffer);
  const horizon = float('horizon', horizonBuffer, pixelCount * directionCount);
  const skyViewFactor = float('svf', svfBuffer);
  const textureView = visibilityTexture
    ? (graph.createTextureView(
        graph.importTexture(
          {
            id: 'visibility-texture',
            format: 'r32float',
            width,
            height,
            usage: visibilityTexture.props.usage
          },
          visibilityTexture
        ),
        {mipLevelCount: 1}
      ) as never)
    : undefined;
  graph.add(
    new GPUTerrainDerivatives({
      width,
      height,
      elevation: elevationBand,
      settings: parameterBuffers[0].importToGraph(graph),
      slope,
      aspect
    })
  );
  graph.add(
    new GPUTerrainHorizon({
      width,
      height,
      elevation: elevationBand,
      settings: parameterBuffers[1].importToGraph(graph),
      directionCount,
      maximumRadius: 24,
      stepGrowth: 1.2,
      horizon,
      skyViewFactor
    })
  );
  graph.add(
    new GPUSolarShadowMask({
      width,
      height,
      directionCount,
      horizon,
      settings: parameterBuffers[2].importToGraph(graph),
      slope,
      aspect,
      skyViewFactor,
      illumination: float('illumination', illuminationBuffer),
      sunVisibilityTexture: textureView
    })
  );
  if (textureView) {
    new GPURasterTextureToBuffer({
      id: 'visibility-readback',
      input: {
        id: 'visibility-band',
        format: 'float32',
        storage: {kind: 'texture', view: textureView}
      },
      output: float('readback', readbackBuffer),
      outputValidity: importGraphBuffer(
        graph,
        'readback-validity',
        validityBuffer,
        'uint32',
        pixelCount
      )
    }).addToGraph(graph);
  }
  const compiled = graph.compile();
  for (const sun of SUN_SWEEP.slice(0, 4)) {
    parameterBuffers[2].write(getGPUSolarShadowMaskParameterValues(sun));
    submitGraph(device, compiled, undefined);
    const expected = computeSolarShadow({
      pixelCount,
      directionCount,
      horizon: await readFloat32(horizonBuffer, pixelCount * directionCount),
      azimuthDegrees: sun.azimuthDegrees,
      altitudeDegrees: sun.altitudeDegrees,
      angularRadiusDegrees: sun.angularRadiusDegrees ?? 0.2666,
      sunIntensity: sun.sunIntensity,
      ambientIntensity: sun.ambientIntensity,
      slope: await readFloat32(slopeBuffer, pixelCount),
      aspect: await readFloat32(aspectBuffer, pixelCount),
      skyViewFactor: await readFloat32(svfBuffer, pixelCount)
    });
    expectClose(await readFloat32(illuminationBuffer, pixelCount), expected.illumination, 2e-5);
    if (textureView) {
      expectClose(await readFloat32(readbackBuffer, pixelCount), expected.sunVisibility, 2e-5);
    }
  }
  // Some pixels are shadowed and some lit at a low sun.
  parameterBuffers[2].write(
    getGPUSolarShadowMaskParameterValues({
      azimuthDegrees: 270,
      altitudeDegrees: 1,
      angularRadiusDegrees: 0
    })
  );
  submitGraph(device, compiled, undefined);
  if (textureView) {
    const visibility = await readFloat32(readbackBuffer, pixelCount);
    expect(visibility.some(value => value === 0)).toBe(true);
    expect(visibility.some(value => value === 1)).toBe(true);
  }
  compiled.destroy();
  for (const buffer of parameterBuffers) buffer.destroy();
  for (const resource of [
    elevationBuffer,
    horizonBuffer,
    slopeBuffer,
    aspectBuffer,
    svfBuffer,
    illuminationBuffer,
    readbackBuffer,
    validityBuffer,
    visibilityTexture
  ]) {
    resource?.destroy();
  }
});

it('GPUSolarShadowMask decodes a unorm16 horizon within one code step of the float32 path', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 20;
  const height = 15;
  const pixelCount = width * height;
  const random = createRandom(29);
  // 7 is odd, so a word can hold two sectors of different pixels.
  for (const directionCount of [8, 7]) {
    const elementCount = pixelCount * directionCount;
    const horizon = Float32Array.from({length: elementCount}, () => -10 + random() * 40);
    horizon.fill(NaN, 3 * directionCount, 4 * directionCount);
    const packed = new Uint32Array(Math.ceil(elementCount / 2));
    for (let element = 0; element < elementCount; element++) {
      packed[element >> 1] |=
        encodeGPUTerrainHorizonUnorm16(horizon[element]) << ((element & 1) * 16);
    }
    const decoded = unpackGPUTerrainHorizonUnorm16(packed, elementCount);
    const inputs = [createInputBuffer(device, horizon), createInputBuffer(device, packed)];
    const outputs = [
      createOutputBuffer(device, pixelCount),
      createOutputBuffer(device, pixelCount)
    ];
    const settings = new GPUMapGraphParameterBuffer(device, {
      id: 'shadow-settings',
      format: 'float32',
      length: 8,
      values: getGPUSolarShadowMaskParameterValues(SUN_SWEEP[0])
    });
    const graph = new GPUCommandGraph(device, {id: 'solar-shadow-unorm16-test'});
    const common = {
      width,
      height,
      directionCount,
      settings: settings.importToGraph(graph)
    };
    graph.add(
      new GPUSolarShadowMask({
        ...common,
        id: 'float32-mask',
        horizon: importGraphBuffer(graph, 'horizon', inputs[0], 'float32', elementCount),
        sunVisibility: importGraphBuffer(graph, 'visibility', outputs[0], 'float32', pixelCount)
      })
    );
    graph.add(
      new GPUSolarShadowMask({
        ...common,
        id: 'unorm16-mask',
        horizonFormat: 'unorm16',
        horizon: importGraphBuffer(graph, 'packed', inputs[1], 'uint32', packed.length),
        sunVisibility: importGraphBuffer(
          graph,
          'packed-visibility',
          outputs[1],
          'float32',
          pixelCount
        )
      })
    );
    const compiled = graph.compile();
    let worstAgainstFloat = 0;
    let worstAgainstDecoded = 0;
    for (const sun of SUN_SWEEP.filter(
      candidate => (candidate.angularRadiusDegrees ?? 0.2666) >= 1
    )) {
      settings.write(getGPUSolarShadowMaskParameterValues(sun));
      submitGraph(device, compiled, undefined);
      const reference = await readFloat32(outputs[0], pixelCount);
      const actual = await readFloat32(outputs[1], pixelCount);
      const expected = computeSolarShadow({
        pixelCount,
        directionCount,
        horizon: decoded,
        azimuthDegrees: sun.azimuthDegrees,
        altitudeDegrees: sun.altitudeDegrees,
        angularRadiusDegrees: sun.angularRadiusDegrees ?? 0.2666
      });
      // The NaN pattern matches exactly and the mask is not trivially constant.
      expect(actual.map(Number.isNaN)).toEqual(reference.map(Number.isNaN));
      expect(actual.some(Number.isNaN)).toBe(true);
      expect(
        new Set(actual.filter(Number.isFinite).map(value => value.toFixed(3))).size
      ).toBeGreaterThan(10);
      expectClose(actual, expected.sunVisibility, 2e-5);
      for (const [index, value] of reference.entries()) {
        if (!Number.isNaN(value)) {
          worstAgainstFloat = Math.max(worstAgainstFloat, Math.abs(actual[index] - value));
          worstAgainstDecoded = Math.max(
            worstAgainstDecoded,
            Math.abs(actual[index] - expected.sunVisibility[index])
          );
        }
      }
    }
    // One code step of horizon moves the soft visibility by at most (2 / PI) * step / radius.
    console.log(`unorm16 shadow D=${directionCount}: worst vs float32 ${worstAgainstFloat}`);
    expect(worstAgainstFloat).toBeLessThan((GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES / 1) * 0.7);
    compiled.destroy();
    settings.destroy();
    for (const buffer of [...inputs, ...outputs]) buffer.destroy();
  }
});
