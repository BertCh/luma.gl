// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainRuggedness,
  type GPUTerrainRuggednessProps
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-ruggedness';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeTerrainRuggedness} from './terrain-ruggedness-oracle';

function createTerrain(width: number, height: number): Float32Array {
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  return Float32Array.from({length: width * height}, () => 1000 + 40 * random());
}

type RuggednessRun = {
  topographicPositionIndex: number[];
  terrainRuggednessIndex: number[];
  roughness: number[];
  validity: number[];
};

async function runRuggedness(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: Partial<GPUTerrainRuggednessProps> = {},
  mask?: Uint32Array
): Promise<RuggednessRun> {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'terrain-ruggedness-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = mask ? createInputBuffer(device, mask) : undefined;
  const outputs: Record<'tpi' | 'rug' | 'roughness' | 'validity', Buffer> = {
    tpi: createOutputBuffer(device, pixelCount),
    rug: createOutputBuffer(device, pixelCount),
    roughness: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount)
  };
  graph.add(
    new GPUTerrainRuggedness({
      width,
      height,
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
      topographicPositionIndex: importGraphBuffer(graph, 'tpi', outputs.tpi, 'float32', pixelCount),
      terrainRuggednessIndex: importGraphBuffer(graph, 'rug', outputs.rug, 'float32', pixelCount),
      roughness: importGraphBuffer(graph, 'roughness', outputs.roughness, 'float32', pixelCount),
      validity: importGraphBuffer(graph, 'validity', outputs.validity, 'uint32', pixelCount),
      ...options
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    topographicPositionIndex: await readFloat32(outputs.tpi, pixelCount),
    terrainRuggednessIndex: await readFloat32(outputs.rug, pixelCount),
    roughness: await readFloat32(outputs.roughness, pixelCount),
    validity: await readUint32(outputs.validity, pixelCount)
  };
  compiled.destroy();
  elevationBuffer.destroy();
  maskBuffer?.destroy();
  for (const buffer of Object.values(outputs)) buffer.destroy();
  return result;
}

/** Compares against the oracle and returns the largest absolute error. */
function expectMatchesOracle(actual: number[], expected: number[]): number {
  expect(actual.length).toBe(expected.length);
  let maximumError = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index])).toBe(true);
      continue;
    }
    const error = Math.abs(actual[index] - value);
    maximumError = Math.max(maximumError, error);
    expect(error).toBeLessThan(Math.max(1e-4, Math.abs(value) * 1e-5));
  }
  return maximumError;
}

const measuredErrors: Record<string, number> = {};

it('GPUTerrainRuggedness reproduces a hand-computed 3x3 window', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const window = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 10]);
  const riley = await runRuggedness(device, window, 3, 3);
  expect(riley.validity).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 0]);
  expect(riley.topographicPositionIndex[4]).toBeCloseTo(-0.125, 6);
  expect(riley.terrainRuggednessIndex[4]).toBeCloseTo(Math.sqrt(69), 5);
  expect(riley.roughness[4]).toBeCloseTo(9, 6);
  const wilson = await runRuggedness(device, window, 3, 3, {
    terrainRuggednessAlgorithm: 'wilson'
  });
  expect(wilson.terrainRuggednessIndex[4]).toBeCloseTo(2.625, 6);
});

it('GPUTerrainRuggedness gives zero TPI and max-min roughness on a plane', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 6;
  const height = 5;
  const plane = Float32Array.from(
    {length: width * height},
    (_, index) => 2 * (index % width) + 3 * Math.floor(index / width)
  );
  const result = await runRuggedness(device, plane, width, height);
  const centre = 2 * width + 2;
  expect(result.validity[centre]).toBe(1);
  expect(result.topographicPositionIndex[centre]).toBeCloseTo(0, 6);
  expect(result.roughness[centre]).toBe(10);
  expect(result.terrainRuggednessIndex[centre]).toBeCloseTo(Math.sqrt(78), 4);
});

it('GPUTerrainRuggedness matches the gdaldem oracle for both edge modes and algorithms', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 9;
  const height = 7;
  const terrain = createTerrain(width, height);
  const mask = new Uint32Array(width * height).fill(1);
  for (const index of [0, 13, 14, 31, 40, 62, 8, 54]) {
    mask[index] = 0;
  }
  for (const edgeMode of ['nodata', 'extrapolate'] as const) {
    for (const algorithm of ['riley', 'wilson'] as const) {
      for (const useMask of [false, true]) {
        const run = await runRuggedness(
          device,
          terrain,
          width,
          height,
          {edgeMode, terrainRuggednessAlgorithm: algorithm},
          useMask ? mask : undefined
        );
        const oracle = computeTerrainRuggedness(
          terrain,
          useMask ? mask : undefined,
          width,
          height,
          {
            edgeMode
          }
        );
        const key = `${edgeMode}/${algorithm}/${useMask ? 'mask' : 'full'}`;
        expect(run.validity).toEqual(oracle.validity);
        expect(run.validity.includes(0) || !useMask).toBe(true);
        expect(run.validity.includes(1)).toBe(true);
        measuredErrors[`${key}/tpi`] = expectMatchesOracle(
          run.topographicPositionIndex,
          oracle.topographicPositionIndex
        );
        measuredErrors[`${key}/tri`] = expectMatchesOracle(
          run.terrainRuggednessIndex,
          algorithm === 'riley' ? oracle.rileyRuggedness : oracle.wilsonRuggedness
        );
        measuredErrors[`${key}/roughness`] = expectMatchesOracle(run.roughness, oracle.roughness);
        expect(run.roughness.some(value => value > 1)).toBe(true);
      }
    }
  }
  // Report measured maximum absolute errors in the test log.
  console.log('ruggedness max abs errors', JSON.stringify(measuredErrors));
});

it('GPUTerrainRuggedness extrapolates corners and tiny grids like gdaldem', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [width, height] of [
    [2, 2],
    [2, 5],
    [5, 2],
    [3, 3]
  ]) {
    const terrain = createTerrain(width, height);
    const run = await runRuggedness(device, terrain, width, height, {edgeMode: 'extrapolate'});
    const oracle = computeTerrainRuggedness(terrain, undefined, width, height, {
      edgeMode: 'extrapolate'
    });
    expect(run.validity.every(value => value === 1)).toBe(true);
    expectMatchesOracle(run.topographicPositionIndex, oracle.topographicPositionIndex);
    expectMatchesOracle(run.terrainRuggednessIndex, oracle.rileyRuggedness);
    expectMatchesOracle(run.roughness, oracle.roughness);
    expect(run.roughness.some(value => value > 1)).toBe(true);
  }
  // Hand check of the 2x2 top-left corner: window rows are [2a-c, 2a-c.., ] with clamped columns.
  const grid = Float32Array.from([10, 20, 40, 80]);
  const corner = await runRuggedness(device, grid, 2, 2, {edgeMode: 'extrapolate'});
  // centre 10; north: 2*10-40=-20 (columns clamped to 0,0,1: -20,-20,2*20-80=-40);
  // middle: 10,10,20; south: 40,40,80.
  const window = [-20, -20, -40, 10, 10, 20, 40, 40, 80];
  const neighbours = [0, 1, 2, 3, 5, 6, 7, 8].map(k => window[k] - 10);
  expect(corner.topographicPositionIndex[0]).toBeCloseTo(
    -neighbours.reduce((sum, value) => sum + value, 0) / 8,
    4
  );
  expect(corner.roughness[0]).toBeCloseTo(120, 4);
});

it('GPUTerrainRuggedness supports single outputs and nodata edge mode on thin grids', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const terrain = createTerrain(2, 2);
  const run = await runRuggedness(device, terrain, 2, 2);
  expect(run.validity).toEqual([0, 0, 0, 0]);
  expect(run.roughness.every(value => Number.isNaN(value))).toBe(true);
});
