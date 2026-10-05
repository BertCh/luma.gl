// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  GPUTerrainWeissLandforms,
  getGPUTerrainWeissLandformsParameterValues,
  type GPUTerrainWeissStandardization
} from '../../../src/map-graphs/topographic-position';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {createTopographicTerrain} from './topographic-position-oracle';
import {computeWeissLandforms} from './weiss-landforms-oracle';

const WIDTH = 48;
const HEIGHT = 40;
const PIXEL_COUNT = WIDTH * HEIGHT;
const SCALES = [{radius: 2}, {radius: 8, innerRadius: 3}] as const;

type WeissRun = {landforms: number[]; standardized: number[]; validity: number[]};

async function runWeiss(
  device: Device,
  elevation: Float32Array,
  standardization: GPUTerrainWeissStandardization,
  settingsSequence: Float32Array[],
  mask?: Uint32Array
): Promise<WeissRun[]> {
  const graph = new GPUCommandGraph(device, {id: 'weiss-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = mask ? createInputBuffer(device, mask) : undefined;
  const buffers = {
    landforms: createOutputBuffer(device, PIXEL_COUNT),
    standardized: createOutputBuffer(device, 2 * PIXEL_COUNT),
    validity: createOutputBuffer(device, PIXEL_COUNT)
  };
  const settings = new GPUMapGraphParameterBuffer(device, {
    id: 'weiss-settings',
    format: 'float32',
    length: 8,
    values: settingsSequence[0]
  });
  graph.add(
    new GPUTerrainWeissLandforms({
      width: WIDTH,
      height: HEIGHT,
      smallScale: SCALES[0],
      largeScale: SCALES[1],
      standardization,
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
      landforms: importGraphBuffer(graph, 'landforms', buffers.landforms, 'uint32', PIXEL_COUNT),
      standardizedPosition: importGraphBuffer(
        graph,
        'standardized',
        buffers.standardized,
        'float32',
        2 * PIXEL_COUNT
      ),
      validity: importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', PIXEL_COUNT)
    })
  );
  const compiled = graph.compile();
  const runs: WeissRun[] = [];
  for (const values of settingsSequence) {
    settings.write(values);
    submitGraph(device, compiled, undefined);
    runs.push({
      landforms: await readUint32(buffers.landforms, PIXEL_COUNT),
      standardized: await readFloat32(buffers.standardized, 2 * PIXEL_COUNT),
      validity: await readUint32(buffers.validity, PIXEL_COUNT)
    });
  }
  compiled.destroy();
  settings.destroy();
  const owned: Buffer[] = [elevationBuffer, ...(maskBuffer ? [maskBuffer] : [])];
  for (const buffer of [...owned, ...Object.values(buffers)]) buffer.destroy();
  return runs;
}

function expectMatchesOracle(
  run: WeissRun,
  elevation: Float32Array,
  standardization: GPUTerrainWeissStandardization,
  settings: {cellSize: number; standardThreshold?: number; slopeThresholdDegrees?: number},
  mask?: Uint32Array
): void {
  const oracle = computeWeissLandforms(elevation, mask, WIDTH, HEIGHT, {
    scales: SCALES,
    standardization,
    ...settings
  });
  expect(run.validity).toEqual(oracle.validity);
  let maximumError = 0;
  for (const [index, value] of oracle.standardizedPosition.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(run.standardized[index])).toBe(true);
      continue;
    }
    maximumError = Math.max(maximumError, Math.abs(run.standardized[index] - value));
  }
  expect(maximumError).toBeLessThan(1e-4);
  // Classes may only differ where a standardized value or the slope sits within float32 noise of
  // a threshold.
  const threshold = settings.standardThreshold ?? 1;
  const slopeThreshold = settings.slopeThresholdDegrees ?? 5;
  for (const [index, landform] of oracle.landforms.entries()) {
    if (run.landforms[index] === landform) continue;
    const nearThreshold =
      [oracle.standardizedPosition[index], oracle.standardizedPosition[PIXEL_COUNT + index]].some(
        value => Math.abs(Math.abs(value) - threshold) < 1e-4
      ) || Math.abs(oracle.slope[index] - slopeThreshold) < 1e-3;
    expect(nearThreshold, `landform ${index}: ${run.landforms[index]} vs ${landform}`).toBe(true);
  }
}

it('GPUTerrainWeissLandforms classifies with global standardization like the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createTopographicTerrain(WIDTH, HEIGHT, 900, 60);
  const mask = new Uint32Array(PIXEL_COUNT).fill(1);
  for (const index of [3, 10 * WIDTH + 17, 25 * WIDTH + 40]) mask[index] = 0;
  const first = {cellSize: 10};
  const second = {cellSize: 30, standardThreshold: 0.5, slopeThresholdDegrees: 2};
  const runs = await runWeiss(
    device,
    elevation,
    'global',
    [
      getGPUTerrainWeissLandformsParameterValues({cellSize: [10, 10]}),
      getGPUTerrainWeissLandformsParameterValues({
        cellSize: [30, 30],
        standardThreshold: 0.5,
        slopeThresholdDegrees: 2
      })
    ],
    mask
  );
  // Several classes must appear, guarding against silent all-zero output.
  expect(new Set(runs[0].landforms).size).toBeGreaterThan(5);
  expect(runs[0].validity.filter(value => value === 0).length).toBeGreaterThan(0);
  expectMatchesOracle(runs[0], elevation, 'global', first, mask);
  expectMatchesOracle(runs[1], elevation, 'global', second, mask);
});

it('GPUTerrainWeissLandforms supports local (DEV) standardization', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createTopographicTerrain(WIDTH, HEIGHT, 2400, 200);
  const [run] = await runWeiss(device, elevation, 'local', [
    getGPUTerrainWeissLandformsParameterValues({cellSize: [5, 5]})
  ]);
  expect(new Set(run.landforms).size).toBeGreaterThan(4);
  expectMatchesOracle(run, elevation, 'local', {cellSize: 5});
});

it('GPUTerrainWeissLandforms finds a mountain top and a canyon', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A cone at one quarter and a pit at three quarters of a gently tilted plane.
  const elevation = Float32Array.from({length: PIXEL_COUNT}, (_, index) => {
    const column = index % WIDTH;
    const row = Math.floor(index / WIDTH);
    const peak = Math.max(0, 80 - 6 * Math.hypot(column - 12, row - 20));
    const pit = Math.max(0, 80 - 6 * Math.hypot(column - 36, row - 20));
    return 500 + 0.01 * column + peak - pit;
  });
  const [run] = await runWeiss(device, elevation, 'global', [
    getGPUTerrainWeissLandformsParameterValues({cellSize: [10, 10]})
  ]);
  expect(run.landforms[20 * WIDTH + 12]).toBe(10);
  expect(run.landforms[20 * WIDTH + 36]).toBe(1);
  expect(run.landforms[5 * WIDTH + 24]).toBe(5);
});
