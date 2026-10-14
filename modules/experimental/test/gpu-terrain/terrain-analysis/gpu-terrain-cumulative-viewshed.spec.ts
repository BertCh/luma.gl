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
import {
  GPU_TERRAIN_VISIBILITY as V,
  GPUTerrainCumulativeViewshed,
  GPUTerrainViewshed,
  getGPUTerrainLineOfSightParameterValues,
  getGPUTerrainViewshedParameterValues,
  type GPUTerrainLineOfSightSettings,
  type GPUTerrainSightLineTraversal
} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeTerrainSightLine, createFractalTerrain} from './terrain-analysis-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import type {GPUTerrainHeightReference} from '../../../src/analysis-core/contracts/vertical-datum';

const WIDTH = 40;
const HEIGHT = 30;
const PIXEL_COUNT = WIDTH * HEIGHT;
const OBSERVERS: [number, number][] = [
  [20, 15],
  [5.5, 4.25],
  [33, 25],
  [-3, 10], // outside: contributes nothing
  [12.75, 22]
];
const GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 30});
const SETTINGS: GPUTerrainLineOfSightSettings = {
  observerHeight: 2,
  targetHeight: 1,
  maximumDistance: 900,
  curvatureCoefficient: 2e-5
};

async function runCumulative(
  device: Device,
  terrain: Float32Array,
  traversal: GPUTerrainSightLineTraversal,
  observersPerDispatch: number,
  options: {
    /** Compile-time observer height interpretation. */
    heightReference?: GPUTerrainHeightReference;
    /** Optional per-observer heights in the selected reference. */
    observerHeights?: Float32Array;
    grid?: GPURasterMetadata;
    observers?: [number, number][];
    settings?: GPUTerrainLineOfSightSettings;
    cellSizeMode?: GPURasterCellSizeMode;
  } = {}
) {
  const grid = options.grid ?? GRID;
  const observers = options.observers ?? OBSERVERS;
  const pixelCount = grid.width * grid.height;
  const settings = options.settings ?? {...SETTINGS, toleranceMeters: 2};
  const elevationBuffer = createInputBuffer(device, terrain);
  const observerBuffer = createInputBuffer(device, Float32Array.from(observers.flat()));
  const heightBuffer = options.observerHeights
    ? createInputBuffer(device, options.observerHeights)
    : undefined;
  const visibleBuffer = createOutputBuffer(device, pixelCount);
  const marginalBuffer = createOutputBuffer(device, pixelCount);
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'cumulative-settings',
    format: 'float32',
    length: 12,
    values: getGPUTerrainLineOfSightParameterValues(settings, grid)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-cumulative-test'});
  const contributor = new GPUTerrainCumulativeViewshed({
    grid,
    cellSizeMode: options.cellSizeMode,
    heightReference: options.heightReference,
    observerHeights: heightBuffer
      ? importGraphBuffer(graph, 'heights', heightBuffer, 'float32', observers.length)
      : undefined,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
      }
    },
    observers: importGraphBuffer(graph, 'observers', observerBuffer, 'float32x2', observers.length),
    parameters: settingsBuffer.importToGraph(graph),
    traversal,
    observersPerDispatch,
    output: {
      visibleCount: importGraphBuffer(graph, 'visible', visibleBuffer, 'uint32', pixelCount),
      marginalCount: importGraphBuffer(graph, 'marginal', marginalBuffer, 'uint32', pixelCount)
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    visible: await readUint32(visibleBuffer, pixelCount),
    marginal: await readUint32(marginalBuffer, pixelCount)
  };
  compiled.destroy();
  settingsBuffer.destroy();
  heightBuffer?.destroy();
  for (const buffer of [elevationBuffer, observerBuffer, visibleBuffer, marginalBuffer]) {
    buffer.destroy();
  }
  return result;
}

it('GPUTerrainCumulativeViewshed absolute observer heights match ground mode over no-data ground', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 12, height: 1});
  const terrain = Float32Array.from({length: 12}, (_, column) => (column === 6 ? 20 : 10));
  const observers: [number, number][] = [
    [0, 0],
    [1, 0],
    [-1, 0]
  ];
  const observerHeights = Float32Array.from([12, 14, 12]);
  const settings = {observerHeight: 50, targetHeight: 1, toleranceMeters: 1};
  const parameters = getGPUTerrainLineOfSightParameterValues(settings, grid);
  for (const traversal of ['march', 'pyramid'] as const) {
    const options = {
      grid,
      observers,
      settings,
      observerHeights,
      heightReference: 'absolute' as const
    };
    const ground = await runCumulative(device, terrain, traversal, 2, {
      ...options,
      heightReference: 'ground',
      observerHeights: Float32Array.from([2, 4, 2])
    });
    const absolute = await runCumulative(device, terrain, traversal, 2, options);
    const oracle = (values: Float32Array) => {
      const visible = new Array<number>(12).fill(0);
      const marginal = new Array<number>(12).fill(0);
      for (let column = 0; column < 12; column++) {
        for (const [index, observer] of observers.entries()) {
          const result = computeTerrainSightLine(
            values,
            undefined,
            12,
            1,
            observer,
            [column, 0],
            parameters,
            [observerHeights[index], settings.targetHeight],
            options
          );
          visible[column] += Number(result.code === V.visible);
          marginal[column] += Number(result.code === V.marginal);
        }
      }
      return {visible, marginal};
    };
    expect(absolute).toEqual(ground);
    expect(absolute).toEqual(oracle(terrain));
    const missing = terrain.slice();
    missing[0] = Number.NaN;
    const result = await runCumulative(device, missing, traversal, 2, options);
    expect(result).toEqual(oracle(missing));
    expect(result.visible[4]).toBe(2);
    expect(result.visible[0]).toBe(0);
  }
});

async function runViewshed(
  device: Device,
  terrain: Float32Array,
  observer: [number, number],
  toleranceMeters: number
): Promise<number[]> {
  const elevationBuffer = createInputBuffer(device, terrain);
  const visibilityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'viewshed-settings',
    format: 'float32',
    length: 10,
    values: getGPUTerrainViewshedParameterValues(
      {
        observer,
        observerHeight: SETTINGS.observerHeight,
        targetHeight: SETTINGS.targetHeight,
        maximumDistance: SETTINGS.maximumDistance,
        curvatureCoefficient: SETTINGS.curvatureCoefficient
      },
      GRID
    )
  });
  const toleranceBuffer = new GPUParameterBuffer(device, {
    id: 'viewshed-tolerance',
    format: 'float32',
    length: 4,
    values: Float32Array.from([toleranceMeters, 0, 0, 0])
  });
  const graph = new GPUCommandGraph(device, {id: 'cumulative-reference-viewshed'});
  graph.add(
    new GPUTerrainViewshed({
      grid: GRID,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      parameters: settingsBuffer.importToGraph(graph),
      toleranceParameters: toleranceBuffer.importToGraph(graph),
      output: {
        visibility: importGraphBuffer(graph, 'visibility', visibilityBuffer, 'uint32', PIXEL_COUNT)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const codes = await readUint32(visibilityBuffer, PIXEL_COUNT);
  compiled.destroy();
  settingsBuffer.destroy();
  toleranceBuffer.destroy();
  elevationBuffer.destroy();
  visibilityBuffer.destroy();
  return codes;
}

it('GPUTerrainCumulativeViewshed sums single-observer viewsheds over batches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const terrain = createFractalTerrain(WIDTH, HEIGHT, 3, 45);
  const expectedVisible = new Array<number>(PIXEL_COUNT).fill(0);
  const expectedMarginal = new Array<number>(PIXEL_COUNT).fill(0);
  for (const observer of OBSERVERS) {
    const codes = await runViewshed(device, terrain, observer, 2);
    // An outside observer gives all noData in the viewshed, which counts as neither.
    for (const [index, code] of codes.entries()) {
      expectedVisible[index] += code === V.visible ? 1 : 0;
      expectedMarginal[index] += code === V.marginal ? 1 : 0;
    }
  }
  const march = await runCumulative(device, terrain, 'march', 3);
  expect(march.visible).toEqual(expectedVisible);
  expect(march.marginal).toEqual(expectedMarginal);
  expect(march.visible.some(count => count > 1)).toBe(true);
  expect(march.visible.some(count => count === 0)).toBe(true);
  expect(march.marginal.some(count => count > 0)).toBe(true);
  const pyramid = await runCumulative(device, terrain, 'pyramid', 2);
  expect(pyramid.visible).toEqual(march.visible);
  expect(pyramid.marginal).toEqual(march.marginal);
});

it('GPUTerrainCumulativeViewshed geographic counts at 60 degrees match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 64;
  const height = 64;
  const grid = makeGPURasterMetadata({width, height});
  grid.affine = [1 / 3600, 0, 0, 0, -1 / 3600, 60 + 32 / 3600];
  const terrain = createFractalTerrain(width, height, 3, 45);
  const observers: [number, number][] = [
    [20, 15],
    [5.5, 4.25],
    [53, 45]
  ];
  const settings: GPUTerrainLineOfSightSettings = {
    observerHeight: 2,
    targetHeight: 1,
    maximumDistance: 700,
    curvatureCoefficient: 2e-5,
    toleranceMeters: 2,
    tolerancePerKilometer: 1,
    targetIgnoreDistance: 60
  };
  const parameters = getGPUTerrainLineOfSightParameterValues(settings, grid);
  const oracleOptions = {
    cellSizeMode: 'geographic' as const,
    georeference: getRasterGridGeoreference(grid)
  };
  const expectedVisible = new Array<number>(width * height).fill(0);
  const expectedMarginal = new Array<number>(width * height).fill(0);
  const nearTie = new Array<boolean>(width * height).fill(false);
  for (const observer of observers) {
    for (let index = 0; index < width * height; index++) {
      const expected = computeTerrainSightLine(
        terrain,
        undefined,
        width,
        height,
        observer,
        [index % width, Math.floor(index / width)],
        parameters,
        undefined,
        oracleOptions
      );
      expectedVisible[index] += expected.code === V.visible ? 1 : 0;
      expectedMarginal[index] += expected.code === V.marginal ? 1 : 0;
      const edge = Math.min(
        Math.abs(expected.maxSlope - (expected.targetSlope + expected.band)),
        Math.abs(expected.maxSlope - (expected.targetSlope - expected.band))
      );
      nearTie[index] ||= edge < 1e-4;
    }
  }
  const options = {grid, observers, settings, cellSizeMode: 'geographic' as const};
  const march = await runCumulative(device, terrain, 'march', 2, options);
  const pyramid = await runCumulative(device, terrain, 'pyramid', 1, options);
  expect(pyramid).toEqual(march);
  let mismatches = 0;
  for (let index = 0; index < width * height; index++) {
    if (
      march.visible[index] !== expectedVisible[index] ||
      march.marginal[index] !== expectedMarginal[index]
    ) {
      mismatches++;
      // Use the viewshed oracle's float32 tie allowance for the summed codes too.
      expect(nearTie[index]).toBe(true);
    }
  }
  expect(mismatches / (width * height)).toBeLessThan(0.005);
  expect(march.visible.some(count => count > 1)).toBe(true);
  expect(march.visible).toContain(0);
  expect(march.marginal.some(count => count > 0)).toBe(true);
});
