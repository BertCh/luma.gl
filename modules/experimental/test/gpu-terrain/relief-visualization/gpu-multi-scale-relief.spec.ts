// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNode} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
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
import {
  computeMultiScaleReliefRVT,
  computeGroundMultiScaleReliefRVT,
  computeRectangleMeanFilterRVT,
  getGroundReliefRowRadii
} from './relief-visualization-oracle';
import {
  createElevationBand,
  expectClose,
  getMaximumMagnitude,
  punchHoles
} from './relief-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

import {
  getRasterGridGeoreference,
  getRasterGroundCellSize
} from '../../../src/analysis-core/raster-grid';
import {GPU_RELIEF_DEFAULT_QUANTUM} from '../../../src/gpu-terrain/relief-visualization/relief-visualization-utils';

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
      grid: makeGPURasterMetadata({width: WIDTH, height: HEIGHT}),
      elevation: createElevationBand(graph, device, elevation, buffers),
      parameters: settings.importToGraph(graph),
      output: {
        relief: importGraphBuffer(graph, 'relief', reliefBuffer, 'float32', PIXEL_COUNT),
        outputValidity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', PIXEL_COUNT)
      }
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
  requireTestDevice(device);
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

it('GPUMultiScaleRelief matches RVT above the former absolute elevation cap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const elevation = Float32Array.from(
    {length: PIXEL_COUNT},
    (_, index) => 1000000 + (index % 17) * 0.25
  );
  elevation[0] = NaN;
  await runMultiScaleRelief(
    device,
    elevation,
    {resolution: 1, featureMinimum: 3, featureMaximum: 30, scalingFactor: 2},
    [1, 2]
  );
});

for (const separable of [false, true]) {
  it(`GPUMultiScaleRelief geographic rectangles at 60 N match the oracle (separable=${separable})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = makeGPURasterMetadata({
      width: WIDTH,
      height: HEIGHT,
      cellSize: 0.001,
      origin: [0, 60 + HEIGHT * 0.0005]
    });
    const groundCell = getRasterGroundCellSize(
      'geographic',
      getRasterGridGeoreference(grid),
      HEIGHT / 2,
      HEIGHT
    );
    // The coarsest horizontal radius crosses 9.5 at the latitude between the two center rows.
    // Neighboring rows round to 9 or 10, exercising the separable approximation.
    const resolution = (groundCell[0] * 9.5) / 5;
    const scales = {
      resolution,
      featureMinimum: resolution * 3.1,
      featureMaximum: resolution * 10,
      scalingFactor: 1
    };
    const elevation = Float32Array.from(
      {length: PIXEL_COUNT},
      (_, index) =>
        100 +
        0.1 * Math.sin((index % WIDTH) * 0.3) +
        0.2 * Math.cos(Math.floor(index / WIDTH) * 0.2)
    );
    punchHoles(elevation, WIDTH, [
      [0, 0],
      [31, 23],
      [32, 24],
      [63, 47]
    ]);
    const rowRadii = getGroundReliefRowRadii(grid, 'geographic', 5 * resolution);
    expect(new Set(rowRadii.map(radius => radius[0]))).toEqual(new Set([9, 10]));
    const buffers: Buffer[] = [];
    const reliefBuffer = createOutputBuffer(device, PIXEL_COUNT);
    const validityBuffer = createOutputBuffer(device, PIXEL_COUNT);
    buffers.push(reliefBuffer, validityBuffer);
    const settings = new GPUParameterBuffer(device, {
      id: 'ground-msrm-settings',
      format: 'float32',
      length: 4,
      values: getGPUMultiScaleReliefParameterValues()
    });
    const graph = new GPUCommandGraph(device, {id: 'ground-msrm-test'});
    const contributor = new GPUMultiScaleRelief({
      ...scales,
      grid,
      cellSizeMode: 'geographic',
      elevation: createElevationBand(graph, device, elevation, buffers),
      parameters: settings.importToGraph(graph),
      output: {
        relief: importGraphBuffer(graph, 'relief', reliefBuffer, 'float32', PIXEL_COUNT),
        outputValidity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', PIXEL_COUNT)
      }
    });
    // Lower only the declaration-time limit: each scalar plane fits, the table does not.
    // Restore before compilation and submission, including when declaration throws.
    const originalLimits = device.limits;
    let nodes: readonly GPUCommandNode<void>[];
    try {
      if (separable)
        Object.defineProperty(device, 'limits', {
          // WebGPU limits are prototype getters; spreading them loses the dispatch limit.
          value: new Proxy(originalLimits, {
            get(target, property) {
              return property === 'maxStorageBufferBindingSize'
                ? PIXEL_COUNT * 4
                : Reflect.get(target, property, target);
            }
          }),
          configurable: true
        });
      nodes = contributor.getCommandNodes(graph);
    } finally {
      Object.defineProperty(device, 'limits', {value: originalLimits, configurable: true});
    }
    expect(nodes.some(node => node.id.endsWith('-last-rows'))).toBe(separable);
    graph.add(nodes);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const expected = computeGroundMultiScaleReliefRVT({
        ...scales,
        width: WIDTH,
        height: HEIGHT,
        grid,
        cellSizeMode: 'geographic',
        elevation
      });
      const fine = computeRectangleMeanFilterRVT(
        elevation,
        WIDTH,
        HEIGHT,
        getGroundReliefRowRadii(grid, 'geographic', resolution)
      );
      const coarse = computeRectangleMeanFilterRVT(elevation, WIDTH, HEIGHT, rowRadii);
      const relief = await readFloat32(reliefBuffer, PIXEL_COUNT);
      const quantum = GPU_RELIEF_DEFAULT_QUANTUM;
      for (let index = 0; index < PIXEL_COUNT; index++) {
        if (Number.isNaN(expected[index])) {
          expect(relief[index]).toBeNaN();
          continue;
        }
        // Two quantum/2 mean errors, divided by n-i=4, plus float32 deviation/combine ULPs.
        // Separable uses 0.02 elevation units for this smooth, <0.6-unit-span fixture:
        // rows crossing the horizontal rounding boundary have an extra pixel on each side.
        const deviationMagnitude =
          Math.abs(elevation[index] - fine[index]) + Math.abs(elevation[index] - coarse[index]);
        const bound = separable
          ? 0.02
          : (quantum + 8 * 2 ** -24 * (deviationMagnitude + quantum)) / 4;
        expect(Math.abs(relief[index] - expected[index])).toBeLessThanOrEqual(bound);
      }
      expect(await readUint32(validityBuffer, PIXEL_COUNT)).toEqual(
        expected.map(value => (Number.isNaN(value) ? 0 : 1))
      );
    } finally {
      compiled.destroy();
      settings.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
