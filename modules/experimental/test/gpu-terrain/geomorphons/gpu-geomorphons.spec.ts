// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_GEOMORPHON_FORMS,
  GPUGeomorphons,
  getGPUGeomorphonsParameterValues,
  type GPUGeomorphonsProps
} from '../../../src/gpu-terrain/geomorphons';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeGeomorphons, type GeomorphonOracleOptions} from './geomorphons-oracle';

type Outputs = {forms: number[]; ternary: number[]; pattern: number[]; validity: number[]};

type Fixture = {
  graph: GPUCommandGraph;
  settings: GPUParameterBuffer<'float32'>;
  buffers: Record<keyof Outputs, Buffer>;
  owned: Buffer[];
};

function createFixture(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  settingsValues: Float32Array,
  props: Partial<GPUGeomorphonsProps>,
  mask?: Uint32Array
): Fixture {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'geomorphons-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = mask ? createInputBuffer(device, mask) : undefined;
  const buffers = {
    forms: createOutputBuffer(device, pixelCount),
    ternary: createOutputBuffer(device, pixelCount),
    pattern: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount)
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'geomorphon-settings',
    format: 'float32',
    length: 8,
    values: settingsValues
  });
  graph.add(
    new GPUGeomorphons({
      width,
      height,
      searchRadius: 6,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        },
        validity: maskBuffer
          ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', pixelCount)
          : undefined
      },
      settings: settings.importToGraph(graph),
      forms: importGraphBuffer(graph, 'forms', buffers.forms, 'uint32', pixelCount),
      ternary: importGraphBuffer(graph, 'ternary', buffers.ternary, 'uint32', pixelCount),
      pattern: importGraphBuffer(graph, 'pattern', buffers.pattern, 'uint32', pixelCount),
      validity: importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', pixelCount),
      ...props
    })
  );
  return {
    graph,
    settings,
    buffers,
    owned: [elevationBuffer, ...(maskBuffer ? [maskBuffer] : []), ...Object.values(buffers)]
  };
}

function destroyFixture(fixture: Fixture): void {
  fixture.settings.destroy();
  for (const buffer of fixture.owned) buffer.destroy();
}

async function readOutputs(fixture: Fixture, pixelCount: number): Promise<Outputs> {
  return {
    forms: await readUint32(fixture.buffers.forms, pixelCount),
    ternary: await readUint32(fixture.buffers.ternary, pixelCount),
    pattern: await readUint32(fixture.buffers.pattern, pixelCount),
    validity: await readUint32(fixture.buffers.validity, pixelCount)
  };
}

function countMismatches(actual: Outputs, expected: Outputs): Record<keyof Outputs, number> {
  const count = (left: number[], right: number[]) =>
    left.reduce((total, value, index) => total + (value === right[index] ? 0 : 1), 0);
  return {
    forms: count(actual.forms, expected.forms),
    ternary: count(actual.ternary, expected.ternary),
    pattern: count(actual.pattern, expected.pattern),
    validity: count(actual.validity, expected.validity)
  };
}

function createIntegerTerrain(width: number, height: number): Float32Array {
  let state = 12345;
  const noise = new Float32Array(width * height);
  for (let index = 0; index < noise.length; index++) {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    noise[index] = (state >>> 16) % 7;
  }
  return Float32Array.from(noise, (value, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    return Math.round(12 * Math.sin(column / 4) + 10 * Math.cos(row / 3)) + value;
  });
}

function createSmoothTerrain(width: number, height: number): Float32Array {
  return Float32Array.from({length: width * height}, (_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    return (
      40 * Math.sin(column / 5.3) * Math.cos(row / 4.1) + 0.37 * column + 7 * Math.sin(row / 1.7)
    );
  });
}

const WIDTH = 48;
const HEIGHT = 40;

type Case = {
  name: string;
  options: GeomorphonOracleOptions;
  settings: Parameters<typeof getGPUGeomorphonsParameterValues>[0];
  props?: Partial<GPUGeomorphonsProps>;
};

const CASES: Case[] = [];
for (const comparison of ['anglev1', 'anglev2', 'anglev2-distance'] as const) {
  CASES.push({
    name: `${comparison} plain`,
    options: {searchRadius: 6, comparison},
    settings: {cellSize: [10, 10]},
    props: {comparison}
  });
  CASES.push({
    name: `${comparison} flat distance`,
    options: {searchRadius: 6, comparison},
    settings: {cellSize: [10, 10], flatDistance: 25, flatThresholdDegrees: 3, zFactor: 1.5},
    props: {comparison}
  });
  CASES.push({
    name: `${comparison} skip 1 north rows`,
    options: {searchRadius: 6, skipRadius: 1, comparison, rowDirection: 'north'},
    settings: {cellSize: [10, 10]},
    props: {comparison, skipRadius: 1, rowDirection: 'north'}
  });
}
CASES.unshift({
  name: 'uniform anisotropic 7x10',
  options: {searchRadius: 6},
  settings: {cellSize: [7, 10]}
});
CASES.unshift({
  name: 'uniform anisotropic 10x7',
  options: {searchRadius: 6},
  settings: {cellSize: [10, 7]}
});
CASES.push({
  name: 'geographic equator',
  options: {searchRadius: 6, cellSizeMode: 'geographic'},
  settings: {cellSize: [0.0001, 0.0001]},
  props: {cellSizeMode: 'geographic'}
});
CASES.push({
  name: 'geographic',
  options: {searchRadius: 7, comparison: 'anglev2', cellSizeMode: 'geographic'},
  settings: {cellSize: [0.0001, 0.0001], northEdge: 47, southEdge: 46.9},
  props: {searchRadius: 7, comparison: 'anglev2', cellSizeMode: 'geographic'}
});
CASES.push({
  name: 'web-mercator',
  options: {searchRadius: 5, cellSizeMode: 'web-mercator'},
  settings: {cellSize: [12, 9], northEdge: 0.3, southEdge: 0.32},
  props: {searchRadius: 5, cellSizeMode: 'web-mercator'}
});

for (const [terrainName, createTerrain] of [
  ['integer', createIntegerTerrain],
  ['smooth', createSmoothTerrain]
] as const) {
  it(`GPUGeomorphons matches the GRASS oracle on ${terrainName} terrain`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const elevation = createTerrain(WIDTH, HEIGHT);
    for (const testCase of CASES) {
      const settingsValues = getGPUGeomorphonsParameterValues(testCase.settings);
      const fixture = createFixture(
        device,
        elevation,
        WIDTH,
        HEIGHT,
        settingsValues,
        testCase.props ?? {}
      );
      const compiled = fixture.graph.compile();
      submitGraph(device, compiled, undefined);
      const actual = await readOutputs(fixture, WIDTH * HEIGHT);
      const expected = computeGeomorphons(
        elevation,
        undefined,
        WIDTH,
        HEIGHT,
        settingsValues,
        testCase.options
      );
      const mismatches = countMismatches(actual, expected);
      console.info(`geomorphons ${terrainName} ${testCase.name}: mismatches`, mismatches);
      expect(mismatches, `${terrainName} ${testCase.name}`).toEqual({
        forms: 0,
        ternary: 0,
        pattern: 0,
        validity: 0
      });
      // A failed WGSL compile yields zeros, so require structure.
      expect(new Set(actual.forms).size).toBeGreaterThan(2);
      expect(actual.validity.filter(value => value === 1).length).toBeGreaterThan(100);
      compiled.destroy();
      destroyFixture(fixture);
    }
  });
}

it('GPUGeomorphons honors nodata and rewrites settings without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createIntegerTerrain(WIDTH, HEIGHT);
  const mask = new Uint32Array(WIDTH * HEIGHT).fill(1);
  for (let index = 0; index < mask.length; index += 11) mask[index] = 0;
  for (let row = 14; row < 20; row++) {
    for (let column = 20; column < 26; column++) mask[row * WIDTH + column] = 0;
  }
  const first = getGPUGeomorphonsParameterValues({cellSize: [10, 10]});
  const fixture = createFixture(device, elevation, WIDTH, HEIGHT, first, {}, mask);
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  let actual = await readOutputs(fixture, WIDTH * HEIGHT);
  let expected = computeGeomorphons(elevation, mask, WIDTH, HEIGHT, first, {searchRadius: 6});
  console.info('geomorphons nodata mismatches', countMismatches(actual, expected));
  expect(countMismatches(actual, expected)).toEqual({
    forms: 0,
    ternary: 0,
    pattern: 0,
    validity: 0
  });
  expect(new Set(actual.validity)).toEqual(new Set([0, 1]));
  expect(actual.forms[16 * WIDTH + 22]).toBe(0);

  const second = getGPUGeomorphonsParameterValues({
    cellSize: [10, 10],
    flatThresholdDegrees: 8,
    zFactor: 0.5
  });
  fixture.settings.write(second);
  submitGraph(device, compiled, undefined);
  actual = await readOutputs(fixture, WIDTH * HEIGHT);
  expected = computeGeomorphons(elevation, mask, WIDTH, HEIGHT, second, {searchRadius: 6});
  expect(countMismatches(actual, expected)).toEqual({
    forms: 0,
    ternary: 0,
    pattern: 0,
    validity: 0
  });
  expect(actual.forms.filter(value => value === GPU_GEOMORPHON_FORMS.flat).length).toBeGreaterThan(
    0
  );

  // Invalid settings invalidate every cell.
  fixture.settings.write(getGPUGeomorphonsParameterValues({cellSize: [10, 10], zFactor: NaN}));
  submitGraph(device, compiled, undefined);
  actual = await readOutputs(fixture, WIDTH * HEIGHT);
  expect(actual.validity.every(value => value === 0)).toBe(true);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUGeomorphons classifies synthetic landforms at the center', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const size = 21;
  const center = 10;
  const shapes: [string, (column: number, row: number) => number, number][] = [
    [
      'peak',
      (column, row) => 100 - 0.5 * ((column - center) ** 2 + (row - center) ** 2),
      GPU_GEOMORPHON_FORMS.peak
    ],
    [
      'pit',
      (column, row) => 0.5 * ((column - center) ** 2 + (row - center) ** 2),
      GPU_GEOMORPHON_FORMS.pit
    ],
    // A tilted parabola: an exactly linear plane ties zenith and nadir, which GRASS reads as flat.
    ['slope', column => 3 * column + 0.05 * column ** 2, GPU_GEOMORPHON_FORMS.slope],
    ['flat', column => 0.1 * column, GPU_GEOMORPHON_FORMS.flat],
    ['ridge', column => -0.5 * (column - center) ** 2, GPU_GEOMORPHON_FORMS.ridge],
    ['valley', column => 0.5 * (column - center) ** 2, GPU_GEOMORPHON_FORMS.valley]
  ];
  const settingsValues = getGPUGeomorphonsParameterValues({cellSize: [10, 10]});
  for (const [name, shape, expectedForm] of shapes) {
    const elevation = Float32Array.from({length: size * size}, (_, index) =>
      shape(index % size, Math.floor(index / size))
    );
    const fixture = createFixture(device, elevation, size, size, settingsValues, {});
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const actual = await readOutputs(fixture, size * size);
    expect(actual.forms[center * size + center], name).toBe(expectedForm);
    expect(actual.validity[center * size + center], name).toBe(1);
    compiled.destroy();
    destroyFixture(fixture);
  }
});

it('GPUGeomorphons resolves integer ties toward the nearest sample and the comparison mode', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Heights 4, 8 along east give equal angles; nearest sample must win, so 3 cells out is not
  // a new zenith. Staircase rises 4 per cell east, and the surface is a plateau elsewhere.
  const size = 11;
  const elevation = Float32Array.from({length: size * size}, (_, index) => {
    const column = index % size;
    return column > 5 ? 4 * (column - 5) : 0;
  });
  const settingsValues = getGPUGeomorphonsParameterValues({cellSize: [10, 10]});
  for (const comparison of ['anglev1', 'anglev2', 'anglev2-distance'] as const) {
    const fixture = createFixture(device, elevation, size, size, settingsValues, {
      comparison,
      searchRadius: 4
    });
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const actual = await readOutputs(fixture, size * size);
    const expected = computeGeomorphons(elevation, undefined, size, size, settingsValues, {
      searchRadius: 4,
      comparison
    });
    expect(countMismatches(actual, expected)).toEqual({
      forms: 0,
      ternary: 0,
      pattern: 0,
      validity: 0
    });
    // East digit: equal angles at every step, so the nearest sample is both zenith and nadir.
    // anglev1 reads the tie as flat (digit 1); both anglev2 modes resolve it to +1 (digit 2).
    const eastDigit = Math.floor(actual.pattern[5 * size + 5] / 3 ** 7) % 3;
    expect(eastDigit, comparison).toBe(comparison === 'anglev1' ? 1 : 2);
    compiled.destroy();
    destroyFixture(fixture);
  }
});
