// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_TERRAIN_VISIBILITY as V,
  GPUTerrainCumulativeViewshed,
  GPUTerrainViewshed,
  getGPUTerrainSightLineParameterValues,
  getGPUTerrainViewshedParameterValues,
  type GPUTerrainSightLineSettings,
  type GPUTerrainSightLineTraversal
} from '../../../src/gpu-terrain/terrain-analysis';
import {createInputBuffer, createOutputBuffer, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createFractalTerrain} from './terrain-analysis-oracle';

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
const SETTINGS: GPUTerrainSightLineSettings = {
  observerHeight: 2,
  targetHeight: 1,
  maxDistance: 900,
  cellSize: [30, 30],
  curvatureCoefficient: 2e-5
};

async function runCumulative(
  device: Device,
  terrain: Float32Array,
  traversal: GPUTerrainSightLineTraversal,
  observersPerDispatch: number
) {
  const elevationBuffer = createInputBuffer(device, terrain);
  const observerBuffer = createInputBuffer(device, Float32Array.from(OBSERVERS.flat()));
  const visibleBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const marginalBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'cumulative-settings',
    format: 'float32',
    length: 12,
    values: getGPUTerrainSightLineParameterValues({...SETTINGS, toleranceMeters: 2})
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-cumulative-test'});
  const recipe = new GPUTerrainCumulativeViewshed({
    width: WIDTH,
    height: HEIGHT,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
      }
    },
    observers: importGraphBuffer(graph, 'observers', observerBuffer, 'float32x2', OBSERVERS.length),
    settings: settingsBuffer.importToGraph(graph),
    traversal,
    observersPerDispatch,
    visibleCount: importGraphBuffer(graph, 'visible', visibleBuffer, 'uint32', PIXEL_COUNT),
    marginalCount: importGraphBuffer(graph, 'marginal', marginalBuffer, 'uint32', PIXEL_COUNT)
  });
  graph.add(recipe);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    visible: await readUint32(visibleBuffer, PIXEL_COUNT),
    marginal: await readUint32(marginalBuffer, PIXEL_COUNT)
  };
  compiled.destroy();
  settingsBuffer.destroy();
  for (const buffer of [elevationBuffer, observerBuffer, visibleBuffer, marginalBuffer]) {
    buffer.destroy();
  }
  return result;
}

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
    length: 8,
    values: getGPUTerrainViewshedParameterValues({
      observer,
      observerHeight: SETTINGS.observerHeight,
      targetHeight: SETTINGS.targetHeight,
      maxDistance: SETTINGS.maxDistance,
      cellSize: SETTINGS.cellSize,
      curvatureCoefficient: SETTINGS.curvatureCoefficient
    })
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
      width: WIDTH,
      height: HEIGHT,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      settings: settingsBuffer.importToGraph(graph),
      tolerance: toleranceBuffer.importToGraph(graph),
      visibility: importGraphBuffer(graph, 'visibility', visibilityBuffer, 'uint32', PIXEL_COUNT)
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
  if (!device) {
    return;
  }
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
