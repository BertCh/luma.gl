// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUMultiScaleReliefParameterValues,
  getGPUMultiScaleReliefRadii,
  GPUMultiScaleRelief,
  type GPUMultiScaleReliefScales
} from '../../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSmoothTerrain} from '../terrain-illumination/terrain-horizon-oracle';
import {computeMultiScaleReliefRVT} from './relief-visualization-oracle';
import {
  createElevationBand,
  expectClose,
  getMaximumMagnitude,
  punchHoles
} from './relief-test-utils';

const WIDTH = 64;
const HEIGHT = 48;
const PIXEL_COUNT = WIDTH * HEIGHT;

async function runMultiScaleRelief(
  device: Device,
  elevation: Float32Array,
  scales: GPUMultiScaleReliefScales,
  exaggerations: number[]
): Promise<void> {
  const buffers: Buffer[] = [];
  const reliefBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const validityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  buffers.push(reliefBuffer, validityBuffer);
  const settings = new GPUParameterBuffer(device, {
    id: 'msrm-settings',
    format: 'float32',
    length: 4,
    values: getGPUMultiScaleReliefParameterValues({verticalExaggeration: exaggerations[0]})
  });
  const graph = new GPUCommandGraph(device, {id: 'msrm-test'});
  graph.add(
    new GPUMultiScaleRelief({
      ...scales,
      width: WIDTH,
      height: HEIGHT,
      elevation: createElevationBand(graph, device, elevation, buffers),
      settings: settings.importToGraph(graph),
      relief: importGraphBuffer(graph, 'relief', reliefBuffer, 'float32', PIXEL_COUNT),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', PIXEL_COUNT)
    })
  );
  const compiled = graph.compile();
  for (const verticalExaggeration of exaggerations) {
    settings.write(getGPUMultiScaleReliefParameterValues({verticalExaggeration}));
    submitGraph(device, compiled, undefined);
    // RVT's literal loop over every scale, against the telescoped two-filter GPU sum.
    const expected = computeMultiScaleReliefRVT({
      width: WIDTH,
      height: HEIGHT,
      elevation,
      verticalExaggeration,
      ...scales
    });
    const relief = await readFloat32(reliefBuffer, PIXEL_COUNT);
    expectClose(relief, expected, 3e-4, JSON.stringify(scales));
    expect(await readUint32(validityBuffer, PIXEL_COUNT)).toEqual(
      expected.map(value => (Number.isNaN(value) ? 0 : 1))
    );
    expect(getMaximumMagnitude(relief)).toBeGreaterThan(0.05);
  }
  compiled.destroy();
  settings.destroy();
  for (const buffer of buffers) buffer.destroy();
}

it('getGPUMultiScaleReliefRadii follows the RVT index formulas', () => {
  // i = floor(0^1) = 0, n = ceil((19 / 2)^1) = 10.
  const linear = getGPUMultiScaleReliefRadii({
    resolution: 1,
    featureMinimum: 1,
    featureMaximum: 20,
    scalingFactor: 1
  });
  expect(linear.firstIndex).toBe(0);
  expect(linear.lastIndex).toBe(10);
  expect(linear.radii).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  // i = floor(sqrt(1)) = 1, n = ceil(sqrt(19.5)) = 5.
  const squared = getGPUMultiScaleReliefRadii({
    resolution: 1,
    featureMinimum: 3,
    featureMaximum: 40,
    scalingFactor: 2.9
  });
  expect(squared.radii).toEqual([1, 4, 9, 16, 25]);
  expect(squared.firstRadius).toBe(1);
  expect(squared.lastRadius).toBe(25);
  // featureMinimum below the resolution is raised to it.
  expect(
    getGPUMultiScaleReliefRadii({
      resolution: 2,
      featureMinimum: 0.1,
      featureMaximum: 30,
      scalingFactor: 1
    }).firstIndex
  ).toBe(0);
});

it('GPUMultiScaleRelief equals the literal RVT loop for several scale ladders', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSmoothTerrain(WIDTH, HEIGHT, 21);
  punchHoles(elevation, WIDTH, [
    [5, 5],
    [6, 5],
    [40, 20],
    [63, 0]
  ]);
  await runMultiScaleRelief(
    device,
    elevation,
    {resolution: 1, featureMinimum: 1, featureMaximum: 20, scalingFactor: 1},
    [1, 3, -0.5]
  );
  await runMultiScaleRelief(
    device,
    elevation,
    {resolution: 1, featureMinimum: 3, featureMaximum: 40, scalingFactor: 2},
    [1, 2]
  );
  await runMultiScaleRelief(
    device,
    elevation,
    {resolution: 0.5, featureMinimum: 2, featureMaximum: 9, scalingFactor: 1},
    [1]
  );
});
