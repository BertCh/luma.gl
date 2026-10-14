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
  getGPUTerrainVectorRuggednessParameterValues,
  GPUTerrainVectorRuggedness,
  type GPUTerrainVectorRuggednessProps
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeVectorRuggedness,
  type VectorRuggednessOracleOptions
} from './terrain-ruggedness-oracle';
import {createSeededRandom} from '../terrain-oracle-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const WIDTH = 11;
const HEIGHT = 9;
const PIXEL_COUNT = WIDTH * HEIGHT;

/** Grid of the test raster with square cells of `cellSize` world units. */
function makeGrid(
  cellSize: number,
  options: {origin?: [number, number]; rowDirection?: 'south' | 'north'} = {}
) {
  return makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize, ...options});
}

function createTerrain(): Float32Array {
  const random = createSeededRandom(987);
  return Float32Array.from({length: PIXEL_COUNT}, () => 1000 + 30 * random());
}

type VectorFixture = {
  graph: GPUCommandGraph;
  settings: GPUParameterBuffer<'float32'>;
  vrm: Buffer;
  validity: Buffer;
  owned: Buffer[];
};

function createFixture(
  device: Device,
  elevation: Float32Array,
  settingsValues: Float32Array,
  overrides: Partial<GPUTerrainVectorRuggednessProps> = {},
  mask?: Uint32Array
): VectorFixture {
  const graph = new GPUCommandGraph(device, {id: 'terrain-vrm-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = mask ? createInputBuffer(device, mask) : undefined;
  const vrm = createOutputBuffer(device, PIXEL_COUNT);
  const validity = createOutputBuffer(device, PIXEL_COUNT);
  const settings = new GPUParameterBuffer(device, {
    id: 'vrm-settings',
    format: 'float32',
    length: 8,
    values: settingsValues
  });
  graph.add(
    new GPUTerrainVectorRuggedness({
      grid: makeGrid(10),
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        },
        validity: maskBuffer
          ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', PIXEL_COUNT)
          : undefined
      },
      parameters: settings.importToGraph(graph),
      output: {
        vectorRuggedness: importGraphBuffer(graph, 'vrm', vrm, 'float32', PIXEL_COUNT),
        outputValidity: importGraphBuffer(graph, 'validity', validity, 'uint32', PIXEL_COUNT)
      },
      ...overrides
    })
  );
  return {
    graph,
    settings,
    vrm,
    validity,
    owned: [elevationBuffer, vrm, validity, ...(maskBuffer ? [maskBuffer] : [])]
  };
}

function destroyFixture(fixture: VectorFixture): void {
  fixture.settings.destroy();
  for (const buffer of fixture.owned) buffer.destroy();
}

/** Asserts agreement with the oracle and returns the largest absolute error. */
async function expectOracle(
  fixture: VectorFixture,
  elevation: Float32Array,
  settingsValues: Float32Array,
  options: VectorRuggednessOracleOptions,
  mask?: Uint32Array
): Promise<number> {
  const oracle = computeVectorRuggedness(elevation, mask, WIDTH, HEIGHT, settingsValues, options);
  const actual = await readFloat32(fixture.vrm, PIXEL_COUNT);
  expect(await readUint32(fixture.validity, PIXEL_COUNT)).toEqual(oracle.validity);
  let maximumError = 0;
  for (const [index, value] of oracle.vectorRuggedness.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index])).toBe(true);
      continue;
    }
    const error = Math.abs(actual[index] - value);
    maximumError = Math.max(maximumError, error);
    expect(error).toBeLessThan(2e-5 + Math.abs(value) * 1e-4);
  }
  return maximumError;
}

it('GPUTerrainVectorRuggedness is zero on a plane and positive on rough terrain', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const settings = getGPUTerrainVectorRuggednessParameterValues({}, makeGrid(10));
  const plane = Float32Array.from(
    {length: PIXEL_COUNT},
    (_, index) => 2 * (index % WIDTH) + 3 * Math.floor(index / WIDTH)
  );
  const planeFixture = createFixture(device, plane, settings, {maximumRadius: 2});
  let compiled = planeFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const planeValues = await readFloat32(planeFixture.vrm, PIXEL_COUNT);
  // Clamped border normals see a halved gradient, so only cells whose window avoids the border
  // (radius 2 plus the Horn footprint) are exactly flat.
  let interiorCount = 0;
  for (let row = 3; row < HEIGHT - 3; row++) {
    for (let column = 3; column < WIDTH - 3; column++) {
      expect(Math.abs(planeValues[row * WIDTH + column])).toBeLessThan(1e-6);
      interiorCount++;
    }
  }
  expect(interiorCount).toBeGreaterThan(0);
  expect((await readUint32(planeFixture.validity, PIXEL_COUNT)).every(value => value === 1)).toBe(
    true
  );
  compiled.destroy();
  destroyFixture(planeFixture);

  const sawtooth = Float32Array.from(
    {length: PIXEL_COUNT},
    (_, index) => ((index % WIDTH) % 3) * 12
  );
  const sawFixture = createFixture(device, sawtooth, settings);
  compiled = sawFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const sawValues = await readFloat32(sawFixture.vrm, PIXEL_COUNT);
  expect(sawValues[4 * WIDTH + 5]).toBeGreaterThan(0.05);
  expect(sawValues.every(value => value >= 0 && value <= 1)).toBe(true);
  compiled.destroy();
  destroyFixture(sawFixture);
});

it('GPUTerrainVectorRuggedness matches the oracle for radius, border, and nodata variants', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const terrain = createTerrain();
  const grid = makeGrid(8);
  const mask = new Uint32Array(PIXEL_COUNT).fill(1);
  for (const index of [12, 13, 45, 46, 60, 98]) {
    mask[index] = 0;
  }
  const errors: Record<string, number> = {};
  // Radius 3 and 12 take the separable row/column path (12 clips every window to the grid).
  for (const radius of [1, 3, 12]) {
    const settings = getGPUTerrainVectorRuggednessParameterValues({zFactor: 1.5, radius}, grid);
    for (const borderMode of ['clamp', 'nodata'] as const) {
      for (const useMask of [false, true]) {
        const fixture = createFixture(
          device,
          terrain,
          settings,
          {grid, maximumRadius: radius, borderMode},
          useMask ? mask : undefined
        );
        const compiled = fixture.graph.compile();
        submitGraph(device, compiled, undefined);
        const key = `r${radius}/${borderMode}/${useMask ? 'mask' : 'full'}`;
        errors[key] = await expectOracle(
          fixture,
          terrain,
          settings,
          {radius, borderMode},
          useMask ? mask : undefined
        );
        const values = await readFloat32(fixture.vrm, PIXEL_COUNT);
        expect(values.some(value => value > 0.01)).toBe(true);
        const validity = await readUint32(fixture.validity, PIXEL_COUNT);
        expect(validity.includes(1)).toBe(true);
        expect(validity.includes(0)).toBe(useMask || borderMode === 'nodata');
        compiled.destroy();
        destroyFixture(fixture);
      }
    }
  }
});

it('GPUTerrainVectorRuggedness handles geographic mode, north rows, and settings rewrites', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const terrain = createTerrain();
  const northGrid = (latitude: number) =>
    makeGrid(0.0005, {origin: [0, latitude], rowDirection: 'north'});
  const geographic = getGPUTerrainVectorRuggednessParameterValues({radius: 3}, northGrid(60));
  const fixture = createFixture(device, terrain, geographic, {
    grid: northGrid(60),
    cellSizeMode: 'geographic',
    rowDirection: 'north',
    maximumRadius: 3
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const options: VectorRuggednessOracleOptions = {
    cellSizeMode: 'geographic',
    rowDirection: 'north',
    radius: 3
  };
  await expectOracle(fixture, terrain, geographic, options);
  const before = await readFloat32(fixture.vrm, PIXEL_COUNT);
  expect(before.some(value => value > 0.001)).toBe(true);

  const updated = getGPUTerrainVectorRuggednessParameterValues(
    {zFactor: 4, radius: 3},
    northGrid(30)
  );
  fixture.settings.write(updated);
  submitGraph(device, compiled, undefined);
  await expectOracle(fixture, terrain, updated, options);
  const after = await readFloat32(fixture.vrm, PIXEL_COUNT);
  expect(after.some((value, index) => Math.abs(value - before[index]) > 1e-4)).toBe(true);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUTerrainVectorRuggedness serves every per-frame radius from one pipeline set', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const terrain = createTerrain();
  const grid = makeGrid(8);
  const mask = new Uint32Array(PIXEL_COUNT).fill(1);
  for (const index of [12, 13, 45, 46, 60, 98]) {
    mask[index] = 0;
  }
  // maximumRadius 2 takes the direct window sum, 5 and 12 the separable row and column passes
  // (12 clips every window to the grid). One compiled graph per maximum is rewritten per frame.
  for (const [maximumRadius, radii] of [
    [2, [1, 2]],
    [5, [1, 2, 3, 4, 5]],
    [12, [1, 3, 7, 12]]
  ] as const) {
    const fixture = createFixture(
      device,
      terrain,
      getGPUTerrainVectorRuggednessParameterValues({zFactor: 1.5}, grid),
      {grid, maximumRadius},
      mask
    );
    const compiled = fixture.graph.compile();
    const results = new Map<number, number[]>();
    for (const radius of radii) {
      const settings = getGPUTerrainVectorRuggednessParameterValues({zFactor: 1.5, radius}, grid);
      fixture.settings.write(settings);
      submitGraph(device, compiled, undefined);
      await expectOracle(fixture, terrain, settings, {radius}, mask);
      results.set(radius, await readFloat32(fixture.vrm, PIXEL_COUNT));
    }
    if (radii.length > 1) {
      // Different radii give different images, so the per-frame word is really read.
      const first = results.get(radii[0])!;
      const last = results.get(radii[radii.length - 1])!;
      expect(last.some((value, index) => Math.abs(value - first[index]) > 1e-4)).toBe(true);
    }
    // An omitted radius (word 0) and an explicit radius equal to the maximum are bit-identical.
    const omitted = getGPUTerrainVectorRuggednessParameterValues({zFactor: 1.5}, grid);
    fixture.settings.write(omitted);
    submitGraph(device, compiled, undefined);
    const omittedValues = await readFloat32(fixture.vrm, PIXEL_COUNT);
    await expectOracle(fixture, terrain, omitted, {radius: maximumRadius}, mask);
    const explicit = getGPUTerrainVectorRuggednessParameterValues(
      {zFactor: 1.5, radius: maximumRadius},
      grid
    );
    fixture.settings.write(explicit);
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readFloat32(fixture.vrm, PIXEL_COUNT))).toEqual(
      Array.from(omittedValues)
    );
    // A radius above the maximum written to the buffer directly clamps to the maximum.
    const oversized = Float32Array.from(explicit);
    oversized[5] = maximumRadius + 7;
    fixture.settings.write(oversized);
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readFloat32(fixture.vrm, PIXEL_COUNT))).toEqual(
      Array.from(omittedValues)
    );
    compiled.destroy();
    destroyFixture(fixture);
  }
});
