// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUTerrainVectorRuggednessParameterValues,
  GPUTerrainVectorRuggedness,
  type GPUTerrainVectorRuggednessProps
} from '../../../src/map-graphs/topographic-position/gpu-terrain-vector-ruggedness';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeVectorRuggedness,
  type VectorRuggednessOracleOptions
} from './terrain-ruggedness-oracle';

const WIDTH = 11;
const HEIGHT = 9;
const PIXEL_COUNT = WIDTH * HEIGHT;

function createTerrain(): Float32Array {
  let state = 987;
  return Float32Array.from({length: PIXEL_COUNT}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return 1000 + 30 * (state / 0x100000000);
  });
}

type VectorFixture = {
  graph: GPUCommandGraph;
  settings: GPUMapGraphParameterBuffer<'float32'>;
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
  const settings = new GPUMapGraphParameterBuffer(device, {
    id: 'vrm-settings',
    format: 'float32',
    length: 8,
    values: settingsValues
  });
  graph.add(
    new GPUTerrainVectorRuggedness({
      width: WIDTH,
      height: HEIGHT,
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
      settings: settings.importToGraph(graph),
      vectorRuggedness: importGraphBuffer(graph, 'vrm', vrm, 'float32', PIXEL_COUNT),
      validity: importGraphBuffer(graph, 'validity', validity, 'uint32', PIXEL_COUNT),
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
  if (!device) {
    return;
  }
  const settings = getGPUTerrainVectorRuggednessParameterValues({cellSize: [10, 10]});
  const plane = Float32Array.from(
    {length: PIXEL_COUNT},
    (_, index) => 2 * (index % WIDTH) + 3 * Math.floor(index / WIDTH)
  );
  const planeFixture = createFixture(device, plane, settings, {radius: 2});
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
  if (!device) {
    return;
  }
  const terrain = createTerrain();
  const settings = getGPUTerrainVectorRuggednessParameterValues({cellSize: [8, 8], zFactor: 1.5});
  const mask = new Uint32Array(PIXEL_COUNT).fill(1);
  for (const index of [12, 13, 45, 46, 60, 98]) {
    mask[index] = 0;
  }
  const errors: Record<string, number> = {};
  for (const radius of [1, 3]) {
    for (const borderMode of ['clamp', 'nodata'] as const) {
      for (const useMask of [false, true]) {
        const fixture = createFixture(
          device,
          terrain,
          settings,
          {radius, borderMode},
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
  console.log('vrm max abs errors', JSON.stringify(errors));
});

it('GPUTerrainVectorRuggedness handles geographic mode, north rows, and settings rewrites', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const terrain = createTerrain();
  const geographic = getGPUTerrainVectorRuggednessParameterValues({
    cellSize: [0.0005, 0.0005],
    northEdge: 60,
    southEdge: 59.95
  });
  const fixture = createFixture(device, terrain, geographic, {
    cellSizeMode: 'geographic',
    rowDirection: 'north',
    radius: 3
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const options: VectorRuggednessOracleOptions = {
    cellSizeMode: 'geographic',
    rowDirection: 'north',
    radius: 3
  };
  const error = await expectOracle(fixture, terrain, geographic, options);
  const before = await readFloat32(fixture.vrm, PIXEL_COUNT);
  expect(before.some(value => value > 0.001)).toBe(true);

  const updated = getGPUTerrainVectorRuggednessParameterValues({
    cellSize: [0.0005, 0.0005],
    zFactor: 4,
    northEdge: 30,
    southEdge: 29.95
  });
  fixture.settings.write(updated);
  submitGraph(device, compiled, undefined);
  const updatedError = await expectOracle(fixture, terrain, updated, options);
  const after = await readFloat32(fixture.vrm, PIXEL_COUNT);
  expect(after.some((value, index) => Math.abs(value - before[index]) > 1e-4)).toBe(true);
  console.log('vrm geographic max abs errors', error, updatedError);
  compiled.destroy();
  destroyFixture(fixture);
});
