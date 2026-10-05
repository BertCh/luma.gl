// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import type {GPURasterBand} from '../../../src/gpu-raster';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPURasterZonalStatistics,
  type GPURasterZonalStatisticsProps
} from '../../../src/gpu-raster/raster-zonal-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeZonalStatistics,
  type ZonalStatisticsOracleOptions,
  type ZonalStatisticsOracleResult
} from './raster-zonal-statistics-oracle';

const COLUMNS = ['cellCounts', 'valueCounts', 'sums', 'means', 'minimums', 'maximums'] as const;
const NO_DATA = -9999;

/** Deterministic pseudo-random generator so failures reproduce. */
function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

type TestData = {
  zones: Uint32Array;
  values: Float32Array;
  validity: Uint32Array;
};

/** Zones 0..9 plus out-of-range IDs, quarter-step values, NaNs, nodata, and invalid cells. */
function createRandomData(width: number, height: number, seed: number): TestData {
  const random = createRandom(seed);
  const count = width * height;
  const zones = new Uint32Array(count);
  const values = new Float32Array(count);
  const validity = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    const roll = random();
    zones[index] = roll < 0.08 ? 10 + Math.floor(random() * 5) : Math.floor(random() * 9);
    const valueRoll = random();
    values[index] =
      valueRoll < 0.07
        ? Number.NaN
        : valueRoll < 0.14
          ? NO_DATA
          : Math.round((random() * 200 - 100) * 4) / 4;
    validity[index] = random() < 0.1 ? 0 : 1;
  }
  return {zones, values, validity};
}

function expectCloseArray(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index])).toBe(true);
    } else {
      expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
        Math.max(1e-4, Math.abs(value) * 1e-4)
      );
    }
  }
}

type Fixture = {
  graph: GPUCommandGraph;
  zonesBuffer: Buffer;
  valuesBuffer?: Buffer;
  columnBuffers: Record<(typeof COLUMNS)[number], Buffer>;
  overflowBuffer: Buffer;
  owned: Buffer[];
  zoneCapacity: number;
};

type FixtureOptions = {
  width: number;
  height: number;
  data: TestData;
  zoneCapacity: number;
  ignoredZone?: number;
  sumOrder?: 'atomic' | 'sorted';
  withValidity?: boolean;
  band?: Partial<Pick<GPURasterBand, 'noDataValue' | 'scale' | 'offset'>>;
  makeBand?: (graph: GPUCommandGraph, fixture: {device: Device}) => GPURasterBand;
};

function createFixture(device: Device, options: FixtureOptions): Fixture {
  const {width, height, data, zoneCapacity} = options;
  const cellCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'zonal-statistics-test'});
  const owned: Buffer[] = [];
  const zonesBuffer = createInputBuffer(device, data.zones);
  owned.push(zonesBuffer);
  let band: GPURasterBand;
  let valuesBuffer: Buffer | undefined;
  if (options.makeBand) {
    band = options.makeBand(graph, {device});
  } else {
    valuesBuffer = createInputBuffer(device, data.values);
    const validityBuffer = options.withValidity
      ? createInputBuffer(device, data.validity)
      : undefined;
    owned.push(valuesBuffer, ...(validityBuffer ? [validityBuffer] : []));
    band = {
      id: 'values',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount)
      },
      validity: validityBuffer
        ? importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', cellCount)
        : undefined,
      ...options.band
    } as GPURasterBand;
  }
  const columnBuffers = {} as Fixture['columnBuffers'];
  const output: GPURasterZonalStatisticsProps['output'] = {};
  for (const name of COLUMNS) {
    const buffer = createOutputBuffer(device, zoneCapacity);
    owned.push(buffer);
    columnBuffers[name] = buffer;
    (output as Record<string, unknown>)[name] = importGraphBuffer(
      graph,
      name,
      buffer,
      name === 'cellCounts' || name === 'valueCounts' ? 'uint32' : 'float32',
      zoneCapacity
    );
  }
  const overflowBuffer = createOutputBuffer(device, 1);
  owned.push(overflowBuffer);
  graph.add(
    new GPURasterZonalStatistics({
      width,
      height,
      zones: importGraphBuffer(graph, 'zones', zonesBuffer, 'uint32', cellCount),
      values: band,
      zoneCapacity,
      ignoredZone: options.ignoredZone,
      sumOrder: options.sumOrder,
      output,
      overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
    })
  );
  return {
    graph,
    zonesBuffer,
    valuesBuffer,
    columnBuffers,
    overflowBuffer,
    owned,
    zoneCapacity
  };
}

async function expectMatchesOracle(
  fixture: Fixture,
  expected: ZonalStatisticsOracleResult
): Promise<void> {
  const {zoneCapacity, columnBuffers} = fixture;
  expect(await readUint32(columnBuffers.cellCounts, zoneCapacity)).toEqual(expected.cellCounts);
  expect(await readUint32(columnBuffers.valueCounts, zoneCapacity)).toEqual(expected.valueCounts);
  for (const name of ['sums', 'means', 'minimums', 'maximums'] as const) {
    expectCloseArray(await readFloat32(columnBuffers[name], zoneCapacity), expected[name]);
  }
  expect(await readUint32(fixture.overflowBuffer, 1)).toEqual([expected.overflow]);
}

function destroyFixture(fixture: Fixture): void {
  for (const buffer of fixture.owned) buffer.destroy();
}

const WIDTH = 37;
const HEIGHT = 23;

it('GPURasterZonalStatistics matches the oracle on a random grid with calibration', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const data = createRandomData(WIDTH, HEIGHT, 7);
  // Zone 9 is guaranteed empty: the generator only emits zones 0..8 and 10..14.
  const oracleOptions: ZonalStatisticsOracleOptions = {
    zoneCapacity: 10,
    validity: data.validity,
    noDataValue: NO_DATA,
    scale: 0.5,
    offset: 1
  };
  const fixture = createFixture(device, {
    width: WIDTH,
    height: HEIGHT,
    data,
    zoneCapacity: 10,
    withValidity: true,
    band: {noDataValue: NO_DATA, scale: 0.5, offset: 1}
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const expected = computeZonalStatistics(data.zones, data.values, oracleOptions);
  expect(expected.overflow).toBe(1);
  expect(expected.cellCounts[9]).toBe(0);
  expect(expected.valueCounts.some(count => count > 0 && count < expected.cellCounts[0])).toBe(
    true
  );
  await expectMatchesOracle(fixture, expected);
  const means = await readFloat32(fixture.columnBuffers.means, 10);
  const minimums = await readFloat32(fixture.columnBuffers.minimums, 10);
  const maximums = await readFloat32(fixture.columnBuffers.maximums, 10);
  expect([means[9], minimums[9], maximums[9]].every(Number.isNaN)).toBe(true);
  expect((await readFloat32(fixture.columnBuffers.sums, 10))[9]).toBe(0);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPURasterZonalStatistics skips ignoredZone silently and reports overflow only for others', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const data = createRandomData(WIDTH, HEIGHT, 11);
  data.zones.forEach((zone, index) => {
    if (zone >= 10) data.zones[index] = 3;
  });
  // Background zone 0 is ignored: no overflow even though zone 0 is in range.
  let fixture = createFixture(device, {
    width: WIDTH,
    height: HEIGHT,
    data,
    zoneCapacity: 4,
    ignoredZone: 0,
    band: {noDataValue: NO_DATA}
  });
  let compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  let expected = computeZonalStatistics(data.zones, data.values, {
    zoneCapacity: 4,
    ignoredZone: 0,
    noDataValue: NO_DATA
  });
  expect(expected.cellCounts[0]).toBe(0);
  // Zones 4..8 exceed capacity 4 and are not ignored.
  expect(expected.overflow).toBe(1);
  await expectMatchesOracle(fixture, expected);
  compiled.destroy();
  destroyFixture(fixture);

  // Only ignored and in-range zones: overflow is 0.
  data.zones.forEach((zone, index) => {
    data.zones[index] = zone % 4;
  });
  fixture = createFixture(device, {
    width: WIDTH,
    height: HEIGHT,
    data,
    zoneCapacity: 4,
    ignoredZone: 0,
    band: {noDataValue: NO_DATA}
  });
  compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expected = computeZonalStatistics(data.zones, data.values, {
    zoneCapacity: 4,
    ignoredZone: 0,
    noDataValue: NO_DATA
  });
  expect(expected.overflow).toBe(0);
  await expectMatchesOracle(fixture, expected);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPURasterZonalStatistics handles a single cell and zoneCapacity 1', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const single = {
    zones: Uint32Array.of(0),
    values: Float32Array.of(2.5),
    validity: Uint32Array.of(1)
  };
  let fixture = createFixture(device, {
    width: 1,
    height: 1,
    data: single,
    zoneCapacity: 1
  });
  let compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(
    fixture,
    computeZonalStatistics(single.zones, single.values, {zoneCapacity: 1})
  );
  expect(await readFloat32(fixture.columnBuffers.means, 1)).toEqual([2.5]);
  compiled.destroy();
  destroyFixture(fixture);

  const data = createRandomData(5, 4, 3);
  data.zones.forEach((zone, index) => {
    data.zones[index] = zone % 3 === 0 ? 0 : zone;
  });
  fixture = createFixture(device, {
    width: 5,
    height: 4,
    data,
    zoneCapacity: 1,
    band: {noDataValue: NO_DATA}
  });
  compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(
    fixture,
    computeZonalStatistics(data.zones, data.values, {
      zoneCapacity: 1,
      noDataValue: NO_DATA
    })
  );
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPURasterZonalStatistics recomputes after per-frame rewrites without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const first = createRandomData(WIDTH, HEIGHT, 21);
  const fixture = createFixture(device, {
    width: WIDTH,
    height: HEIGHT,
    data: first,
    zoneCapacity: 10,
    withValidity: true,
    band: {noDataValue: NO_DATA}
  });
  const options: ZonalStatisticsOracleOptions = {
    zoneCapacity: 10,
    validity: first.validity,
    noDataValue: NO_DATA
  };
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(fixture, computeZonalStatistics(first.zones, first.values, options));

  // Second frame: no out-of-range zones, so overflow must drop back to 0.
  const second = createRandomData(WIDTH, HEIGHT, 22);
  second.zones.forEach((zone, index) => {
    second.zones[index] = zone % 10;
  });
  fixture.zonesBuffer.write(second.zones);
  fixture.valuesBuffer!.write(second.values);
  submitGraph(device, compiled, undefined);
  const expected = computeZonalStatistics(second.zones, second.values, {
    ...options,
    validity: first.validity
  });
  expect(expected.overflow).toBe(0);
  await expectMatchesOracle(fixture, expected);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPURasterZonalStatistics aggregates a texture-backed value band', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || !device.getTextureFormatCapabilities('r32float').store) {
    return;
  }
  const data = createRandomData(WIDTH, HEIGHT, 5);
  const texture = device.createTexture({
    format: 'r32float',
    width: WIDTH,
    height: HEIGHT,
    usage: Texture.SAMPLE | Texture.COPY_DST
  });
  texture.writeData(data.values);
  const fixture = createFixture(device, {
    width: WIDTH,
    height: HEIGHT,
    data,
    zoneCapacity: 10,
    makeBand: graph => ({
      id: 'texture-values',
      format: 'float32',
      noDataValue: NO_DATA,
      scale: 0.5,
      offset: 1,
      storage: {
        kind: 'texture',
        view: graph.createTextureView(
          graph.importTexture(
            {
              id: 'values-texture',
              format: 'r32float',
              width: WIDTH,
              height: HEIGHT,
              usage: texture.props.usage
            },
            texture
          ),
          {mipLevelCount: 1}
        ) as never
      }
    })
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(
    fixture,
    computeZonalStatistics(data.zones, data.values, {
      zoneCapacity: 10,
      noDataValue: NO_DATA,
      scale: 0.5,
      offset: 1
    })
  );
  compiled.destroy();
  texture.destroy();
  destroyFixture(fixture);
});

/** Large grid, few zones, widely varying magnitudes: float sums are order sensitive. */
function createLargeData(width: number, height: number, seed: number): TestData {
  const random = createRandom(seed);
  const count = width * height;
  const zones = new Uint32Array(count);
  const values = new Float32Array(count);
  const validity = new Uint32Array(count).fill(1);
  for (let index = 0; index < count; index++) {
    const roll = random();
    zones[index] = roll < 0.03 ? 6 + Math.floor(random() * 3) : Math.floor(random() * 4);
    const magnitude = 10 ** (Math.floor(random() * 7) - 3);
    values[index] = (random() - 0.4) * magnitude;
    if (random() < 0.05) values[index] = Number.NaN;
  }
  return {zones, values, validity};
}

async function readBits(buffer: Buffer, count: number): Promise<number[]> {
  return readUint32(buffer, count);
}

it('GPURasterZonalStatistics sumOrder sorted is bitwise reproducible and matches the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 400;
  const height = 300;
  const data = createLargeData(width, height, 99);
  const fixture = createFixture(device, {
    width,
    height,
    data,
    zoneCapacity: 5,
    ignoredZone: 2,
    sumOrder: 'sorted'
  });
  const compiled = fixture.graph.compile();
  let firstSums: number[] | undefined;
  let firstMeans: number[] | undefined;
  for (let run = 0; run < 4; run++) {
    submitGraph(device, compiled, undefined);
    const sums = await readBits(fixture.columnBuffers.sums, 5);
    const means = await readBits(fixture.columnBuffers.means, 5);
    firstSums ??= sums;
    firstMeans ??= means;
    expect(sums).toEqual(firstSums);
    expect(means).toEqual(firstMeans);
  }
  // ignoredZone 2 is inside capacity and excluded; zones 6..8 overflow.
  const expected = computeZonalStatistics(data.zones, data.values, {
    zoneCapacity: 5,
    ignoredZone: 2
  });
  expect(expected.cellCounts[2]).toBe(0);
  expect(expected.overflow).toBe(1);
  expect(await readUint32(fixture.columnBuffers.cellCounts, 5)).toEqual(expected.cellCounts);
  expect(await readUint32(fixture.columnBuffers.valueCounts, 5)).toEqual(expected.valueCounts);
  const sums = await readFloat32(fixture.columnBuffers.sums, 5);
  const means = await readFloat32(fixture.columnBuffers.means, 5);
  for (const zone of [0, 1, 3]) {
    // Float32 sequential CPU sums drift from a tree sum; allow a loose relative tolerance.
    const scale = Math.max(1, Math.abs(expected.sums[zone]), expected.valueCounts[zone]);
    expect(Math.abs(sums[zone] - expected.sums[zone])).toBeLessThanOrEqual(scale * 1e-3 + 1);
    expect(Math.abs(means[zone] - expected.means[zone])).toBeLessThanOrEqual(
      Math.max(1e-3, Math.abs(expected.means[zone]) * 1e-2)
    );
  }
  expect(sums[0]).not.toBe(0);
  // Zone 2 is ignored and zone 4 never occurs: empty.
  for (const zone of [2, 4]) {
    expect(sums[zone]).toBe(0);
    expect(Number.isNaN(means[zone])).toBe(true);
  }
  expect(await readUint32(fixture.overflowBuffer, 1)).toEqual([1]);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPURasterZonalStatistics sumOrder sorted matches the oracle with calibration and validity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const data = createRandomData(WIDTH, HEIGHT, 7);
  const fixture = createFixture(device, {
    width: WIDTH,
    height: HEIGHT,
    data,
    zoneCapacity: 10,
    withValidity: true,
    band: {noDataValue: NO_DATA, scale: 0.5, offset: 1},
    sumOrder: 'sorted'
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(
    fixture,
    computeZonalStatistics(data.zones, data.values, {
      zoneCapacity: 10,
      validity: data.validity,
      noDataValue: NO_DATA,
      scale: 0.5,
      offset: 1
    })
  );
  compiled.destroy();
  destroyFixture(fixture);
});
