// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPULocalDominanceParameterValues,
  getGPULocalDominanceShifts,
  GPULocalDominance,
  type GPULocalDominanceGeometry,
  type GPULocalDominanceSettings
} from '../../../src/gpu-terrain/relief-visualization/gpu-local-dominance';
import {createOutputBuffer, readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createSmoothTerrain} from '../terrain-illumination/terrain-horizon-oracle';
import {computeLocalDominanceRVT, pythonRound} from './relief-visualization-oracle';
import {
  createElevationBand,
  expectClose,
  getMaximumMagnitude,
  punchHoles
} from './relief-test-utils';

const WIDTH = 64;
const HEIGHT = 48;
const PIXEL_COUNT = WIDTH * HEIGHT;

async function runLocalDominance(
  device: Device,
  elevation: Float32Array,
  geometry: GPULocalDominanceGeometry,
  settingsList: GPULocalDominanceSettings[],
  check: (dominance: number[], validity: number[], settings: GPULocalDominanceSettings) => void
): Promise<void> {
  const buffers: Buffer[] = [];
  const dominanceBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const validityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  buffers.push(dominanceBuffer, validityBuffer);
  const settings = new GPUParameterBuffer(device, {
    id: 'dominance-settings',
    format: 'float32',
    length: 4,
    values: getGPULocalDominanceParameterValues(settingsList[0])
  });
  const graph = new GPUCommandGraph(device, {id: 'dominance-test'});
  graph.add(
    new GPULocalDominance({
      ...geometry,
      width: WIDTH,
      height: HEIGHT,
      elevation: createElevationBand(graph, device, elevation, buffers),
      settings: settings.importToGraph(graph),
      dominance: importGraphBuffer(graph, 'dominance', dominanceBuffer, 'float32', PIXEL_COUNT),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', PIXEL_COUNT)
    })
  );
  const compiled = graph.compile();
  for (const values of settingsList) {
    settings.write(getGPULocalDominanceParameterValues(values));
    submitGraph(device, compiled, undefined);
    check(
      await readFloat32(dominanceBuffer, PIXEL_COUNT),
      await readUint32(validityBuffer, PIXEL_COUNT),
      values
    );
  }
  compiled.destroy();
  settings.destroy();
  for (const buffer of buffers) buffer.destroy();
}

it('local dominance offsets use Python half-even rounding and RVT counts', () => {
  expect(pythonRound(0.5)).toBe(0);
  expect(pythonRound(1.5)).toBe(2);
  expect(pythonRound(2.5)).toBe(2);
  expect(pythonRound(-0.5)).toBe(0);
  expect(pythonRound(-1.5)).toBe(-2);
  expect(pythonRound(-2.5)).toBe(-2);
  const defaults = getGPULocalDominanceShifts();
  // 11 distances (10..20), int(359 / 15 + 1) = 24 angles.
  expect(defaults.distanceCount).toBe(11);
  expect(defaults.angleCount).toBe(24);
  expect(defaults.count).toBe(264);
  // Angle 0, distance 10: row 0, column +10; angle 90 deg, distance 10: row +10, column 0.
  expect([defaults.rowShifts[0], defaults.columnShifts[0]]).toEqual([0, 10]);
  const quarter = 6 * defaults.distanceCount;
  expect([defaults.rowShifts[quarter], defaults.columnShifts[quarter]]).toEqual([10, 0]);
  // 45 degrees, distance 10 is 7.07 -> 7; distance 14 is 9.899 -> 10.
  const diagonal = 3 * defaults.distanceCount;
  expect([defaults.rowShifts[diagonal], defaults.columnShifts[diagonal]]).toEqual([7, 7]);
  expect(defaults.rowShifts[diagonal + 4]).toBe(10);
  expect(() => getGPULocalDominanceShifts({minimumRadius: 30, maximumRadius: 20})).toThrow(
    /minimumRadius/
  );
  expect(() => getGPULocalDominanceShifts({angularResolution: 0})).toThrow(/angularResolution/);
  expect(() =>
    getGPULocalDominanceShifts({minimumRadius: 1, maximumRadius: 200, angularResolution: 1})
  ).toThrow(/limit/);
});

it('GPULocalDominance matches the RVT roll loop with nodata and per-frame settings', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSmoothTerrain(WIDTH, HEIGHT, 8).map(value => value / 5);
  punchHoles(elevation, WIDTH, [
    [30, 20],
    [31, 20],
    [30, 21],
    [3, 3],
    [60, 45]
  ]);
  const settingsList = [
    {},
    {observerHeight: 4, verticalExaggeration: 2},
    {observerHeight: 0.4, verticalExaggeration: -1}
  ];
  for (const geometry of [
    {},
    {minimumRadius: 3, maximumRadius: 9, radiusIncrement: 2, angularResolution: 45}
  ]) {
    await runLocalDominance(
      device,
      elevation,
      geometry,
      settingsList,
      (dominance, validity, settings) => {
        const expected = computeLocalDominanceRVT({
          width: WIDTH,
          height: HEIGHT,
          elevation,
          ...geometry,
          ...settings
        });
        expectClose(dominance, expected, 2e-4, JSON.stringify([geometry, settings]));
        expect(validity).toEqual(expected.map(value => (Number.isNaN(value) ? 0 : 1)));
        const finite = dominance.filter(Number.isFinite);
        // A real result varies; a failed compile would give zeros or a constant.
        expect(Math.max(...finite) - Math.min(...finite)).toBeGreaterThan(0.05);
        expect(getMaximumMagnitude(dominance)).toBeGreaterThan(0.1);
      }
    );
  }
});

it('GPULocalDominance is 1 on a flat plane and a bump dominates its surroundings', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const flat = new Float32Array(PIXEL_COUNT).fill(1234.5);
  await runLocalDominance(device, flat, {}, [{}, {observerHeight: 3}], dominance => {
    // Every offset contributes obs / d * (2 d + inc), exactly the normalization.
    for (const value of dominance) {
      expect(Math.abs(value - 1)).toBeLessThan(2e-5);
    }
  });
  const bump = new Float32Array(PIXEL_COUNT);
  bump[24 * WIDTH + 32] = 6;
  await runLocalDominance(device, bump, {minimumRadius: 2, maximumRadius: 6}, [{}], dominance => {
    // The bump sees every neighbour below it and exceeds the flat value 1; neighbours in the
    // ring see a higher cell and fall below it.
    expect(dominance[24 * WIDTH + 32]).toBeGreaterThan(1);
    expect(dominance[24 * WIDTH + 34]).toBeLessThan(1);
    expect(dominance[0]).toBeCloseTo(1, 4);
  });
});
