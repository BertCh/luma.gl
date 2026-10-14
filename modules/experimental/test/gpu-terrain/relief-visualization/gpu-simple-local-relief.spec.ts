// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUSimpleLocalReliefParameterValues,
  GPUSimpleLocalRelief,
  type GPUSimpleLocalReliefSettings
} from '../../../src/gpu-terrain/relief-visualization/gpu-simple-local-relief';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSmoothTerrain} from '../terrain-illumination/terrain-horizon-oracle';
import {computeSimpleLocalReliefRVT} from './relief-visualization-oracle';
import {
  createElevationBand,
  expectClose,
  getMaximumMagnitude,
  punchHoles
} from './relief-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const WIDTH = 64;
const HEIGHT = 48;
const PIXEL_COUNT = WIDTH * HEIGHT;

async function runSimpleLocalRelief(
  device: Device,
  elevation: Float32Array,
  maximumRadius: number,
  settingsList: GPUSimpleLocalReliefSettings[],
  check: (relief: number[], validity: number[], settings: GPUSimpleLocalReliefSettings) => void
): Promise<void> {
  const buffers: Buffer[] = [];
  const reliefBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const validityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  buffers.push(reliefBuffer, validityBuffer);
  const settings = new GPUParameterBuffer(device, {
    id: 'slrm-settings',
    format: 'float32',
    length: 4,
    values: getGPUSimpleLocalReliefParameterValues(settingsList[0])
  });
  const graph = new GPUCommandGraph(device, {id: 'slrm-test'});
  graph.add(
    new GPUSimpleLocalRelief({
      grid: makeGPURasterMetadata({width: WIDTH, height: HEIGHT}),
      maximumRadius,
      elevation: createElevationBand(graph, device, elevation, buffers),
      parameters: settings.importToGraph(graph),
      output: {
        relief: importGraphBuffer(graph, 'relief', reliefBuffer, 'float32', PIXEL_COUNT),
        outputValidity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', PIXEL_COUNT)
      }
    })
  );
  const compiled = graph.compile();
  for (const values of settingsList) {
    settings.write(getGPUSimpleLocalReliefParameterValues(values));
    submitGraph(device, compiled, undefined);
    check(
      await readFloat32(reliefBuffer, PIXEL_COUNT),
      await readUint32(validityBuffer, PIXEL_COUNT),
      values
    );
  }
  compiled.destroy();
  settings.destroy();
  for (const buffer of buffers) buffer.destroy();
}

it('GPUSimpleLocalRelief matches the RVT oracle with nodata and per-frame exaggeration', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const elevation = createSmoothTerrain(WIDTH, HEIGHT, 13);
  punchHoles(elevation, WIDTH, [
    [10, 10],
    [11, 10],
    [10, 11],
    [50, 30],
    [0, 0],
    [63, 47]
  ]);
  for (const radius of [1, 7, 20]) {
    await runSimpleLocalRelief(
      device,
      elevation,
      radius,
      [{}, {verticalExaggeration: 2.5}, {verticalExaggeration: -1}],
      (relief, validity, settings) => {
        const expected = computeSimpleLocalReliefRVT({
          width: WIDTH,
          height: HEIGHT,
          elevation,
          radius: settings.radius ?? radius,
          verticalExaggeration: settings.verticalExaggeration
        });
        expectClose(relief, expected, 2e-4, `radius ${radius}`);
        expect(validity).toEqual(expected.map(value => (Number.isNaN(value) ? 0 : 1)));
        // A silent-zero compile failure would pass the closeness check on flat input only.
        expect(getMaximumMagnitude(relief)).toBeGreaterThan(1);
      }
    );
  }
});

it('GPUSimpleLocalRelief keeps centimetre precision on a 4000 m surface', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const elevation = createSmoothTerrain(WIDTH, HEIGHT, 5).map(value => value / 20 + 4000);
  punchHoles(elevation, WIDTH, [[20, 20]]);
  await runSimpleLocalRelief(device, elevation, 20, [{}], relief => {
    const expected = computeSimpleLocalReliefRVT({
      width: WIDTH,
      height: HEIGHT,
      elevation,
      radius: 20
    });
    // The float32 elevation step is 0.5 mm; a float32 running sum would be off by centimetres.
    const worst = expectClose(relief, expected, 5e-4, 'offset surface');
    expect(worst).toBeLessThan(5e-4);
    expect(getMaximumMagnitude(relief)).toBeGreaterThan(0.05);
  });
});

it('GPUSimpleLocalRelief is zero on a plane and analytic on a bump', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const radius = 5;
  const plane = Float32Array.from(
    {length: PIXEL_COUNT},
    (_, index) => 4000 + 0.5 * (index % WIDTH) + 0.25 * Math.floor(index / WIDTH)
  );
  await runSimpleLocalRelief(device, plane, radius, [{}], relief => {
    for (let row = radius; row < HEIGHT - radius; row++) {
      for (let column = radius; column < WIDTH - radius; column++) {
        expect(Math.abs(relief[row * WIDTH + column])).toBeLessThan(2e-4);
      }
    }
    // Edge clamping bends the window at the border, so the border is not zero.
    expect(Math.abs(relief[0])).toBeGreaterThan(0.1);
  });
  const bump = new Float32Array(PIXEL_COUNT);
  bump[24 * WIDTH + 32] = 10;
  await runSimpleLocalRelief(device, bump, radius, [{verticalExaggeration: 2}], relief => {
    const windowArea = (2 * radius + 1) ** 2;
    expect(relief[24 * WIDTH + 32]).toBeCloseTo(2 * 10 * (1 - 1 / windowArea), 4);
    expect(relief[24 * WIDTH + 33]).toBeCloseTo((-2 * 10) / windowArea, 4);
    expect(Math.abs(relief[24 * WIDTH + 32 + radius + 1])).toBe(0);
  });
});

it('GPUSimpleLocalRelief serves every per-frame radius from one compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const elevation = createSmoothTerrain(WIDTH, HEIGHT, 13);
  punchHoles(elevation, WIDTH, [
    [10, 10],
    [11, 10],
    [50, 30],
    [0, 0]
  ]);
  const maximumRadius = 20;
  const relieves: number[][] = [];
  // One graph with maximumRadius 20 is rewritten per frame; each frame must match the oracle at
  // its own radius. A radius of 0 (omitted) means maximumRadius; the oversized word clamps to it.
  await runSimpleLocalRelief(
    device,
    elevation,
    maximumRadius,
    [{radius: 1}, {radius: 3, verticalExaggeration: 2}, {radius: 7}, {radius: 20}, {}, {radius: 1}],
    (relief, validity, settings) => {
      const radius = settings.radius ?? maximumRadius;
      const expected = computeSimpleLocalReliefRVT({
        width: WIDTH,
        height: HEIGHT,
        elevation,
        radius,
        verticalExaggeration: settings.verticalExaggeration
      });
      expectClose(relief, expected, 2e-4, `per-frame radius ${radius}`);
      expect(validity).toEqual(expected.map(value => (Number.isNaN(value) ? 0 : 1)));
      expect(getMaximumMagnitude(relief)).toBeGreaterThan(0.05);
      relieves.push(relief);
    }
  );
  // The radius is really read per frame: radius 1 and radius 20 differ, and the omitted radius is
  // bit-identical to an explicit radius equal to maximumRadius.
  expect(relieves[0].some((value, index) => Math.abs(value - relieves[3][index]) > 1e-3)).toBe(
    true
  );
  expect(relieves[4]).toEqual(relieves[3]);
  expect(relieves[5]).toEqual(relieves[0]);
});

it('GPUSimpleLocalRelief is translation invariant above the former absolute elevation limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const base = Float32Array.from({length: PIXEL_COUNT}, (_, index) => (index % 17) * 0.25);
  base[0] = NaN;
  let baseline: number[] = [];
  for (const offset of [0, 1000000, -1000000]) {
    await runSimpleLocalRelief(
      device,
      base.map(value => value + offset),
      20,
      [{}],
      relief => {
        if (offset === 0) baseline = relief;
        else expect(relief).toEqual(baseline);
      }
    );
  }
});
