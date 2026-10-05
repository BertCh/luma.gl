// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUTerrainCastShadowParameterValues,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPUTerrainCastShadow,
  type GPUTerrainCastShadowSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {
  getGPUSolarShadowMaskParameterValues,
  GPUSolarShadowMask
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {createRandom, createSmoothTerrain, getVisibleDiskFraction} from './terrain-horizon-oracle';
import {computeTerrainHorizonSweepBruteForce} from './terrain-horizon-sweep-oracle';

type CastShadowResult = {
  sunVisibility: number[];
  horizonAngle: number[];
  validity: number[];
};

function createCastShadowFixture(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: {maximumRadius?: number; rowDirection?: 'south' | 'north'} = {}
) {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'cast-shadow-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const buffers = {
    sunVisibility: createOutputBuffer(device, pixelCount),
    horizonAngle: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount)
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'cast-shadow-settings',
    format: 'float32',
    length: GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
    values: getGPUTerrainCastShadowParameterValues({
      cellSize: [10, 10],
      azimuthDegrees: 0,
      altitudeDegrees: 0
    })
  });
  const contributor = new GPUTerrainCastShadow({
    width,
    height,
    maximumRadius: options.maximumRadius,
    rowDirection: options.rowDirection,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
      }
    },
    settings: settings.importToGraph(graph),
    sunVisibility: importGraphBuffer(
      graph,
      'sun-visibility',
      buffers.sunVisibility,
      'float32',
      pixelCount
    ),
    horizonAngle: importGraphBuffer(graph, 'horizon', buffers.horizonAngle, 'float32', pixelCount),
    validity: importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', pixelCount)
  });
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    contributor,
    async run(values: GPUTerrainCastShadowSettings): Promise<CastShadowResult> {
      settings.write(
        getGPUTerrainCastShadowParameterValues(values, options.rowDirection ?? 'south')
      );
      submitGraph(device, compiled, undefined);
      return {
        sunVisibility: await readFloat32(buffers.sunVisibility, pixelCount),
        horizonAngle: await readFloat32(buffers.horizonAngle, pixelCount),
        validity: await readUint32(buffers.validity, pixelCount)
      };
    },
    destroy() {
      compiled.destroy();
      settings.destroy();
      elevationBuffer.destroy();
      for (const buffer of Object.values(buffers)) buffer.destroy();
    }
  };
}

/** Pixel-space direction of the sun, float64, as packed by the settings packer. */
function getSunDirection(azimuthDegrees: number, rowDirection: 'south' | 'north' = 'south') {
  const azimuth = (azimuthDegrees * Math.PI) / 180;
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : value);
  return [
    snap(Math.sin(azimuth)),
    snap((rowDirection === 'south' ? -1 : 1) * Math.cos(azimuth))
  ] as [number, number];
}

function expectNaNPattern(actual: number[], expected: number[]): number {
  let worst = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `NaN at ${index}`).toBe(true);
    } else {
      worst = Math.max(worst, Math.abs(actual[index] - value));
    }
  }
  return worst;
}

it('GPUTerrainCastShadow matches the float64 brute force along the sun line', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 52;
  const height = 44;
  const random = createRandom(4);
  const elevation = createSmoothTerrain(width, height, 6);
  for (let index = 0; index < elevation.length; index++) {
    elevation[index] += (random() - 0.5) * 6;
  }
  for (const hole of [8 * width + 9, 8 * width + 10, 30 * width + 40]) {
    elevation[hole] = NaN;
  }
  const fixture = createCastShadowFixture(device, elevation, width, height);
  const altitudeDegrees = 12;
  const angularRadiusDegrees = 0.5;
  const azimuths = [0, 37, 90, 135, 200, 270, 311];
  const horizons: number[][] = [];
  for (const azimuthDegrees of azimuths) {
    const settings = {
      cellSize: [10, 10] as [number, number],
      azimuthDegrees,
      altitudeDegrees,
      angularRadiusDegrees
    };
    const actual = await fixture.run(settings);
    const expected = computeTerrainHorizonSweepBruteForce({
      width,
      height,
      elevation,
      directionCount: 1,
      directions: [getSunDirection(azimuthDegrees)],
      cellSize: [10, 10]
    });
    expect(expectNaNPattern(actual.horizonAngle, expected.horizon)).toBeLessThan(2e-3);
    // Structure: real occlusion and real visibility (not silent zeros).
    expect(actual.horizonAngle.filter(angle => angle > 3).length).toBeGreaterThan(50);
    const visibility = expected.horizon.map(angle =>
      Number.isNaN(angle)
        ? NaN
        : getVisibleDiskFraction(altitudeDegrees, angle, angularRadiusDegrees)
    );
    expect(expectNaNPattern(actual.sunVisibility, visibility)).toBeLessThan(2e-2);
    expect(actual.sunVisibility.filter(value => value > 0.99).length).toBeGreaterThan(100);
    expect(actual.sunVisibility.filter(value => value < 0.01).length).toBeGreaterThan(20);
    expect(actual.validity).toEqual(expected.validity);
    horizons.push(actual.horizonAngle);
  }
  // Per-frame azimuth changes reuse the single compiled graph and really change the output.
  expect(horizons[0]).not.toEqual(horizons[1]);
  expect(horizons[2]).not.toEqual(horizons[5]);
  fixture.destroy();
});

it('GPUTerrainCastShadow casts a wall shadow of length H / tan(altitude)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 61;
  const height = 21;
  const elevation = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    elevation[row * width + 30] = 20;
  }
  const fixture = createCastShadowFixture(device, elevation, width, height);
  const altitudeDegrees = 30;
  // Wall of 20 m on 10 m cells: pixel k columns behind it has horizon atan(2 / k); it is shadowed
  // while that exceeds 30 degrees, i.e. k < 2 / tan(30) = 3.46 -> columns 31..33.
  const east = await fixture.run({
    cellSize: [10, 10],
    azimuthDegrees: 270,
    altitudeDegrees,
    angularRadiusDegrees: 0
  });
  const row = 10;
  for (let column = 0; column < width; column++) {
    const index = row * width + column;
    const behind = column > 30 && column <= 33;
    expect(east.sunVisibility[index], `column ${column}`).toBe(behind ? 0 : 1);
    if (column > 30) {
      const expectedAngle = (Math.atan(2 / (column - 30)) * 180) / Math.PI;
      expect(east.horizonAngle[index]).toBeCloseTo(expectedAngle, 3);
    }
  }
  // Mirrored: sun in the east shadows the pixels west of the wall.
  const west = await fixture.run({
    cellSize: [10, 10],
    azimuthDegrees: 90,
    altitudeDegrees,
    angularRadiusDegrees: 0
  });
  for (let column = 0; column < width; column++) {
    const behind = column < 30 && column >= 27;
    expect(west.sunVisibility[row * width + column], `column ${column}`).toBe(behind ? 0 : 1);
  }
  // A soft disk makes a penumbra at the shadow edge (column 34: horizon 26.6, alt 30, radius 5).
  const soft = await fixture.run({
    cellSize: [10, 10],
    azimuthDegrees: 270,
    altitudeDegrees,
    angularRadiusDegrees: 5
  });
  const penumbra = soft.sunVisibility[row * width + 34];
  const edgeAngle = (Math.atan(2 / 4) * 180) / Math.PI;
  expect(penumbra).toBeCloseTo(getVisibleDiskFraction(altitudeDegrees, edgeAngle, 5), 4);
  expect(penumbra).toBeGreaterThan(0.1);
  expect(penumbra).toBeLessThan(0.95);
  fixture.destroy();
});

it('GPUTerrainCastShadow handles a sun below the horizon and a high sun on a plane', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 20;
  const height = 16;
  const plane = new Float32Array(width * height).fill(1234.5);
  plane[3 * width + 3] = NaN;
  const fixture = createCastShadowFixture(device, plane, width, height);
  const night = await fixture.run({
    cellSize: [10, 10],
    azimuthDegrees: 123,
    altitudeDegrees: -10
  });
  const day = await fixture.run({cellSize: [10, 10], azimuthDegrees: 123, altitudeDegrees: 60});
  for (let index = 0; index < width * height; index++) {
    if (index === 3 * width + 3) {
      expect(Number.isNaN(night.sunVisibility[index])).toBe(true);
      expect(Number.isNaN(day.sunVisibility[index])).toBe(true);
      expect(day.validity[index]).toBe(0);
      continue;
    }
    expect(night.sunVisibility[index]).toBe(0);
    expect(day.sunVisibility[index]).toBe(1);
    expect(day.horizonAngle[index]).toBe(0);
    expect(day.validity[index]).toBe(1);
  }
  fixture.destroy();
});

it('GPUTerrainCastShadow equals GPUSolarShadowMask fed by the sweep horizon at exact azimuths', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 40;
  const height = 36;
  const random = createRandom(9);
  const elevation = createSmoothTerrain(width, height, 2);
  for (let index = 0; index < elevation.length; index++) {
    elevation[index] += (random() - 0.5) * 4;
  }
  const pixelCount = width * height;
  const directionCount = 8;
  const horizonResult = computeTerrainHorizonSweepBruteForce({
    width,
    height,
    elevation,
    directionCount,
    cellSize: [10, 10]
  });
  const graph = new GPUCommandGraph(device, {id: 'mask-test'});
  const horizonBuffer = createInputBuffer(device, Float32Array.from(horizonResult.horizon));
  const visibilityBuffer = createOutputBuffer(device, pixelCount);
  const maskSettings = new GPUParameterBuffer(device, {
    id: 'mask-settings',
    format: 'float32',
    length: 8,
    values: getGPUSolarShadowMaskParameterValues({azimuthDegrees: 0, altitudeDegrees: 0})
  });
  graph.add(
    new GPUSolarShadowMask({
      width,
      height,
      directionCount,
      horizon: importGraphBuffer(
        graph,
        'horizon',
        horizonBuffer,
        'float32',
        pixelCount * directionCount
      ),
      settings: maskSettings.importToGraph(graph),
      sunVisibility: importGraphBuffer(graph, 'visibility', visibilityBuffer, 'float32', pixelCount)
    })
  );
  const compiled = graph.compile();
  const fixture = createCastShadowFixture(device, elevation, width, height);
  for (const azimuthDegrees of [0, 90, 135, 270]) {
    const sun = {azimuthDegrees, altitudeDegrees: 9, angularRadiusDegrees: 0.4};
    maskSettings.write(getGPUSolarShadowMaskParameterValues(sun));
    submitGraph(device, compiled, undefined);
    const mask = await readFloat32(visibilityBuffer, pixelCount);
    const cast = await fixture.run({cellSize: [10, 10], ...sun});
    expect(cast.sunVisibility.filter(value => value > 0.99).length).toBeGreaterThan(100);
    expect(cast.sunVisibility.filter(value => value < 0.01).length).toBeGreaterThan(20);
    const worst = expectNaNPattern(cast.sunVisibility, mask);
    expect(worst, `azimuth ${azimuthDegrees}`).toBeLessThan(1e-3);
  }
  fixture.destroy();
  compiled.destroy();
  maskSettings.destroy();
  horizonBuffer.destroy();
  visibilityBuffer.destroy();
});
