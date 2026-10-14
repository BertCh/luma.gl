// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata, type GPURasterMetadata} from '../../../src/gpu-raster';
import {
  getRasterGridGeoreference,
  type GPURasterCellSizeMode
} from '../../../src/analysis-core/raster-grid';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  GPU_TERRAIN_VISIBILITY as V,
  GPUTerrainLineOfSight,
  GPUTerrainViewshed,
  getGPUTerrainLineOfSightParameterValues,
  getGPUTerrainViewshedParameterValues,
  type GPUTerrainLineOfSightSettings,
  type GPUTerrainSightLineTraversal
} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {computeTerrainSightLine, createFractalTerrain} from './terrain-analysis-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type LineOfSightRun = {codes: number[]; clearance: number[]};

async function runLineOfSight(
  device: Device,
  elevation: Float32Array,
  grid: GPURasterMetadata,
  pairs: Float32Array,
  settings: GPUTerrainLineOfSightSettings,
  options: {
    traversal?: GPUTerrainSightLineTraversal;
    cellSizeMode?: GPURasterCellSizeMode;
    pairHeights?: Float32Array;
    clearance?: boolean;
  } = {}
): Promise<LineOfSightRun> {
  const {width, height} = grid;
  const pairCount = pairs.length / 4;
  const elevationBuffer = createInputBuffer(device, elevation);
  const pairBuffer = createInputBuffer(device, pairs);
  const heightBuffer = options.pairHeights
    ? createInputBuffer(device, options.pairHeights)
    : undefined;
  const visibilityBuffer = createOutputBuffer(device, pairCount);
  const clearanceBuffer = createOutputBuffer(device, pairCount);
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'los-settings',
    format: 'float32',
    length: 12,
    values: getGPUTerrainLineOfSightParameterValues(settings, grid)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-line-of-sight-test'});
  graph.add(
    new GPUTerrainLineOfSight({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', width * height)
        }
      },
      pairs: importGraphBuffer(graph, 'pairs', pairBuffer, 'float32x4', pairCount),
      pairHeights: heightBuffer
        ? importGraphBuffer(graph, 'heights', heightBuffer, 'float32x2', pairCount)
        : undefined,
      parameters: settingsBuffer.importToGraph(graph),
      traversal: options.traversal,
      cellSizeMode: options.cellSizeMode,
      output: {
        visibility: importGraphBuffer(graph, 'visibility', visibilityBuffer, 'uint32', pairCount),
        clearance:
          options.clearance === false
            ? undefined
            : importGraphBuffer(graph, 'clearance', clearanceBuffer, 'float32', pairCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    codes: await readUint32(visibilityBuffer, pairCount),
    clearance: await readFloat32(clearanceBuffer, pairCount)
  };
  compiled.destroy();
  settingsBuffer.destroy();
  for (const buffer of [
    elevationBuffer,
    pairBuffer,
    heightBuffer,
    visibilityBuffer,
    clearanceBuffer
  ]) {
    buffer?.destroy();
  }
  return result;
}

async function runViewshedCodes(
  device: Device,
  elevation: Float32Array,
  grid: GPURasterMetadata,
  observer: [number, number],
  settings: GPUTerrainLineOfSightSettings
): Promise<number[]> {
  const pixelCount = grid.width * grid.height;
  const elevationBuffer = createInputBuffer(device, elevation);
  const visibilityBuffer = createOutputBuffer(device, pixelCount);
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'viewshed-settings',
    format: 'float32',
    length: 10,
    values: getGPUTerrainViewshedParameterValues(
      {
        observer,
        observerHeight: settings.observerHeight,
        targetHeight: settings.targetHeight,
        maximumDistance: settings.maximumDistance,
        curvatureCoefficient: settings.curvatureCoefficient
      },
      grid
    )
  });
  const graph = new GPUCommandGraph(device, {id: 'los-reference-viewshed'});
  graph.add(
    new GPUTerrainViewshed({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        }
      },
      parameters: settingsBuffer.importToGraph(graph),
      output: {
        visibility: importGraphBuffer(graph, 'visibility', visibilityBuffer, 'uint32', pixelCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const codes = await readUint32(visibilityBuffer, pixelCount);
  compiled.destroy();
  settingsBuffer.destroy();
  elevationBuffer.destroy();
  visibilityBuffer.destroy();
  return codes;
}

const WIDTH = 30;
const HEIGHT = 22;
const GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 30});
const STRIP_GRID = makeGPURasterMetadata({width: 12, height: 1, cellSize: 1});

it('GPUTerrainLineOfSight reproduces the viewshed codes for every pixel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const terrain = createFractalTerrain(WIDTH, HEIGHT, 5, 40);
  const settings: GPUTerrainLineOfSightSettings = {
    observerHeight: 1.7,
    targetHeight: 0.5,
    maximumDistance: 600,
    curvatureCoefficient: 2e-5
  };
  const observer: [number, number] = [12.25, 9.5];
  const pairs = new Float32Array(WIDTH * HEIGHT * 4);
  for (let index = 0; index < WIDTH * HEIGHT; index++) {
    pairs.set([observer[0], observer[1], index % WIDTH, Math.floor(index / WIDTH)], index * 4);
  }
  const expected = await runViewshedCodes(device, terrain, GRID, observer, settings);
  const march = await runLineOfSight(device, terrain, GRID, pairs, settings);
  const pyramid = await runLineOfSight(device, terrain, GRID, pairs, settings, {
    traversal: 'pyramid'
  });
  expect(march.codes).toEqual(expected);
  expect(pyramid.codes).toEqual(march.codes);
  // Clearance is exact for both traversals (bit-identical).
  expect(Array.from(new Uint32Array(new Float32Array(pyramid.clearance).buffer))).toEqual(
    Array.from(new Uint32Array(new Float32Array(march.clearance).buffer))
  );
  expect(expected.some(code => code === V.hidden)).toBe(true);
  expect(expected.some(code => code === V.visible)).toBe(true);
  expect(expected.some(code => code === V.outOfRange)).toBe(true);
  for (const [index, code] of march.codes.entries()) {
    if (code === V.hidden) {
      expect(march.clearance[index]).toBeLessThan(0);
    } else if (code === V.visible) {
      expect(march.clearance[index]).toBeGreaterThanOrEqual(0);
    } else {
      expect(march.clearance[index]).toBeNaN();
    }
  }
  // Without the clearance output the early exit gives the same codes.
  const early = await runLineOfSight(device, terrain, GRID, pairs, settings, {
    clearance: false
  });
  expect(early.codes).toEqual(march.codes);
});

it('GPUTerrainLineOfSight applies pair heights, noData and the float64 oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 12 x 1 strip with a wall of 3 at column 6 and an eye of 1 at column 0.
  const strip = new Float32Array(12);
  strip[6] = 3;
  const settings: GPUTerrainLineOfSightSettings = {observerHeight: 1};
  const pairs = Float32Array.from([
    0,
    0,
    10,
    0, // behind the wall
    0,
    0,
    4,
    0, // in front of the wall
    0,
    0,
    0,
    0, // zero distance
    -1,
    0,
    4,
    0, // observer outside
    0,
    0,
    12,
    0, // target outside
    0,
    0,
    10,
    0 // behind the wall, with a tall observer
  ]);
  const pairHeights = Float32Array.from([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 50, 0]);
  const run = await runLineOfSight(device, strip, STRIP_GRID, pairs, settings, {pairHeights});
  expect(run.codes).toEqual([V.hidden, V.visible, V.visible, V.noData, V.noData, V.visible]);
  expect(run.clearance[0]).toBeLessThan(0);
  expect(run.clearance[1]).toBeGreaterThan(0);
  expect(run.clearance[3]).toBeNaN();
  const limited = await runLineOfSight(device, strip, STRIP_GRID, pairs, {
    ...settings,
    maximumDistance: 5
  });
  expect(limited.codes.slice(0, 2)).toEqual([V.outOfRange, V.visible]);

  if (isSoftwareDevice(device)) {
    return;
  }
  const terrain = createFractalTerrain(WIDTH, HEIGHT, 21, 40);
  const toleranceSettings: GPUTerrainLineOfSightSettings = {
    observerHeight: 2,
    curvatureCoefficient: 2e-5,
    toleranceMeters: 2,
    tolerancePerKilometer: 1,
    targetIgnoreDistance: 60
  };
  const pairList: number[] = [];
  for (let index = 0; index < 200; index++) {
    pairList.push(
      (index * 7.3) % (WIDTH - 1),
      (index * 3.1) % (HEIGHT - 1),
      (index * 5.7 + 3) % (WIDTH - 1),
      (index * 11.9 + 1) % (HEIGHT - 1)
    );
  }
  const list = Float32Array.from(pairList);
  const gpu = await runLineOfSight(device, terrain, GRID, list, toleranceSettings, {
    traversal: 'pyramid'
  });
  const settingValues = getGPUTerrainLineOfSightParameterValues(toleranceSettings, GRID);
  let mismatches = 0;
  for (let index = 0; index < 200; index++) {
    const expected = computeTerrainSightLine(
      terrain,
      undefined,
      WIDTH,
      HEIGHT,
      [list[index * 4], list[index * 4 + 1]],
      [list[index * 4 + 2], list[index * 4 + 3]],
      settingValues
    );
    if (gpu.codes[index] !== expected.code) {
      mismatches++;
      const edge = Math.min(
        Math.abs(expected.maxSlope - (expected.targetSlope + expected.band)),
        Math.abs(expected.maxSlope - (expected.targetSlope - expected.band))
      );
      expect(edge).toBeLessThan(1e-4);
    }
  }
  expect(mismatches).toBeLessThan(2);
  expect(new Set(gpu.codes).size).toBeGreaterThan(1);
});

it('GPUTerrainLineOfSight geographic pairs at 60 degrees match the ground-meter oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 64;
  const height = 64;
  const grid = makeGPURasterMetadata({width, height});
  grid.affine = [1 / 3600, 0, 0, 0, -1 / 3600, 60 + 32 / 3600];
  const terrain = createFractalTerrain(width, height, 21, 40);
  const settings: GPUTerrainLineOfSightSettings = {
    observerHeight: 2,
    maximumDistance: 700,
    curvatureCoefficient: 2e-5,
    toleranceMeters: 2,
    tolerancePerKilometer: 1,
    targetIgnoreDistance: 60
  };
  const pairs = new Float32Array(200 * 4);
  for (let index = 0; index < 200; index++) {
    pairs.set(
      [(index * 7.3) % 63, (index * 3.1) % 63, (index * 5.7 + 3) % 63, (index * 11.9 + 1) % 63],
      index * 4
    );
  }
  const march = await runLineOfSight(device, terrain, grid, pairs, settings, {
    cellSizeMode: 'geographic'
  });
  const pyramid = await runLineOfSight(device, terrain, grid, pairs, settings, {
    cellSizeMode: 'geographic',
    traversal: 'pyramid'
  });
  expect(pyramid.codes).toEqual(march.codes);
  expect(new Uint32Array(new Float32Array(pyramid.clearance).buffer)).toEqual(
    new Uint32Array(new Float32Array(march.clearance).buffer)
  );
  const parameters = getGPUTerrainLineOfSightParameterValues(settings, grid);
  let mismatches = 0;
  for (let index = 0; index < 200; index++) {
    const expected = computeTerrainSightLine(
      terrain,
      undefined,
      width,
      height,
      [pairs[index * 4], pairs[index * 4 + 1]],
      [pairs[index * 4 + 2], pairs[index * 4 + 3]],
      parameters,
      undefined,
      {cellSizeMode: 'geographic', georeference: getRasterGridGeoreference(grid)}
    );
    if (march.codes[index] !== expected.code) {
      mismatches++;
      const edge = Math.min(
        Math.abs(expected.maxSlope - (expected.targetSlope + expected.band)),
        Math.abs(expected.maxSlope - (expected.targetSlope - expected.band))
      );
      expect(edge).toBeLessThan(1e-4);
    }
    if (Number.isNaN(expected.clearance)) {
      expect(march.clearance[index]).toBeNaN();
    } else {
      expect(Math.abs(march.clearance[index] - expected.clearance)).toBeLessThan(
        Math.max(0.01, Math.abs(expected.clearance) * 1e-5)
      );
    }
  }
  expect(mismatches).toBeLessThan(2);
  expect(march.codes).toContain(V.outOfRange);
  expect(march.codes).toContain(V.visible);
  expect(march.codes).toContain(V.hidden);
});
