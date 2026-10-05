// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUPointHorizonVisibilityParameterValues,
  GPUPointHorizonVisibility,
  type GPUPointHorizonModelOptions,
  type GPUPointHorizonVisibilitySettings
} from '../../../src/gpu-terrain/point-horizon';
import {GPU_TERRAIN_VISIBILITY as V} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {createFractalTerrain} from './point-horizon-oracle';

type VisibilityConfig = {
  width: number;
  height: number;
  values: Float32Array;
  observers: readonly (readonly [number, number, number])[];
  /** `[column, row, height, observerIndex]` */
  targets: readonly (readonly [number, number, number, number])[];
  settings: GPUPointHorizonVisibilitySettings;
  model: GPUPointHorizonModelOptions;
};

async function runVisibility(device: Device, config: VisibilityConfig) {
  const {width, height, values, observers, targets, model} = config;
  const terrain = createInputBuffer(device, values);
  const observerBuffer = createInputBuffer(
    device,
    Float32Array.from(observers.flatMap(([column, row, eyeHeight]) => [column, row, eyeHeight, 0]))
  );
  const targetBuffer = createInputBuffer(device, Float32Array.from(targets.flat()));
  const settings = new GPUParameterBuffer(device, {
    id: 'visibility-settings',
    format: 'float32',
    length: 12,
    values: getGPUPointHorizonVisibilityParameterValues(config.settings)
  });
  const visibility = createOutputBuffer(device, targets.length);
  const details = createOutputBuffer(device, targets.length * 4);
  const graph = new GPUCommandGraph(device, {id: 'point-horizon-visibility-test'});
  graph.add(
    new GPUPointHorizonVisibility({
      ...model,
      width,
      height,
      elevation: {
        id: 'terrain',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'terrain', terrain, 'float32', width * height)
        }
      },
      observers: importGraphBuffer(
        graph,
        'observers',
        observerBuffer,
        'float32x4',
        observers.length
      ),
      targets: importGraphBuffer(graph, 'targets', targetBuffer, 'float32x4', targets.length),
      settings: settings.importToGraph(graph),
      visibility: importGraphBuffer(graph, 'visibility', visibility, 'uint32', targets.length),
      details: importGraphBuffer(graph, 'details', details, 'float32x4', targets.length)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const codes = await readUint32(visibility, targets.length);
  const rows = await readFloat32(details, targets.length * 4);
  compiled.destroy();
  settings.destroy();
  for (const buffer of [terrain, observerBuffer, targetBuffer, visibility, details]) {
    buffer.destroy();
  }
  return {codes, rows};
}

const SIZE = 128;
const MODEL: GPUPointHorizonModelOptions = {
  projection: 'planar',
  maximumDistance: 1000,
  cellSize: 10
};
const SETTINGS: GPUPointHorizonVisibilitySettings = {cellSize: [10, 10]};
const RIDGE_COLUMN = 40; // 200 m east of the observer at column 20
const RIDGE_HEIGHT = 30;
const OBSERVER: readonly [number, number, number] = [20, 64, 2];

function createRidgeTerrain(withRidge: boolean, peakGround = 0): Float32Array {
  const values = new Float32Array(SIZE * SIZE);
  if (withRidge) {
    for (let row = 0; row < SIZE; row++) {
      values[row * SIZE + RIDGE_COLUMN] = RIDGE_HEIGHT;
    }
  }
  values[64 * SIZE + 80] = peakGround;
  return values;
}

it('GPUPointHorizonVisibility classifies targets behind a ridge', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const run = (values: Float32Array, targets: VisibilityConfig['targets'], model = MODEL) =>
    runVisibility(device, {
      width: SIZE,
      height: SIZE,
      values,
      observers: [OBSERVER],
      targets,
      settings: SETTINGS,
      model
    });
  // 600 m east of the observer. The ridge top subtends atan(28 / 200) from the eye.
  const behind = await run(createRidgeTerrain(true), [
    [80, 64, 0, 0], // ground level behind the ridge: hidden
    [80, 64, 86, 0] // tangent (86 - 2) / 600 equals the ridge tangent 28 / 200: marginal
  ]);
  expect(behind.codes[0]).toBe(V.hidden);
  expect(behind.codes[1]).toBe(V.marginal);
  const open = await run(createRidgeTerrain(false), [[80, 64, 100, 0]]);
  expect(open.codes[0]).toBe(V.visible);
  // Details: target angle, occluder angle (ridge), skyline angle.
  const ridgeAngle = (Math.atan(28 / 200) * 180) / Math.PI;
  expect(Math.abs(behind.rows[1] - ridgeAngle)).toBeLessThan(0.05);
  expect(Math.abs(behind.rows[5] - ridgeAngle)).toBeLessThan(0.05);
  expect(behind.rows[0]).toBeLessThan(0);
  // Clearly above the ridge: visible and on the skyline (it is the highest point of its ray).
  const peak = await run(createRidgeTerrain(true, 150), [[80, 64, 50, 0]]);
  expect(peak.codes[0]).toBe(V.visible);
  expect(peak.rows[3]).toBe(1);
  expect(peak.rows[0]).toBeGreaterThan(peak.rows[1]);
  // The ridge itself is a visible target but below the skyline of a higher peak behind it.
  const ridgeTarget = await run(createRidgeTerrain(true, 150), [[RIDGE_COLUMN, 64, 0, 0]]);
  expect(ridgeTarget.codes[0]).toBe(V.visible);
  expect(ridgeTarget.rows[3]).toBe(0);
});

it('GPUPointHorizonVisibility reports noData and outOfRange', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createRidgeTerrain(false);
  const result = await runVisibility(device, {
    width: SIZE,
    height: SIZE,
    values,
    observers: [OBSERVER, [-5, 10, 2]],
    targets: [
      [80, 64, 100, 0], // visible
      [125, 64, 0, 0], // 1050 m > maximumDistance 1000: outOfRange
      [80, 64, 0, 1], // observer outside the grid: noData
      [80, 64, 0, 7], // no such observer: noData
      [200, 64, 0, 0], // target outside the grid: noData
      [20, 64, 0, 0] // coincident with the observer: visible
    ],
    settings: SETTINGS,
    model: MODEL
  });
  expect(result.codes).toEqual([V.visible, V.outOfRange, V.noData, V.noData, V.noData, V.visible]);
  expect(result.rows[4 * 4]).toBeNaN();
  expect(result.rows[1 * 4]).toBeNaN();
});

it('GPUPointHorizonVisibility web mercator agrees with a planar equivalent', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 192;
  const height = 192;
  const values = createFractalTerrain(width, height, 5, 500, 36);
  const worldPixelSize = 512 * 2 ** 12;
  const latitude = (50 * Math.PI) / 180;
  const originY =
    worldPixelSize * (0.5 - Math.asinh(Math.tan(latitude)) / (2 * Math.PI)) - 96 - 0.5;
  const targets: [number, number, number, number][] = [];
  for (const [column, row] of [
    [140, 96],
    [96, 40],
    [60, 150],
    [150, 150],
    [30, 30]
  ]) {
    targets.push([column, row, 5, 0], [column, row, 900, 0]);
  }
  const base = {
    width,
    height,
    values,
    observers: [[96, 96, 10] as const],
    targets
  };
  const mercator = await runVisibility(device, {
    ...base,
    settings: {worldPixelSize, originY, curvatureCoefficient: 2e-5},
    model: {projection: 'web-mercator', maximumDistance: 2500, cellSize: 13, maximumLatitude: 50}
  });
  // At 50 degrees north a Web Mercator pixel is 19.1 m * cos(50) = 12.3 m in both directions.
  const pixelMeters = (40075016.68557849 / worldPixelSize) * Math.cos(latitude);
  const planar = await runVisibility(device, {
    ...base,
    settings: {cellSize: [pixelMeters, pixelMeters], curvatureCoefficient: 2e-5},
    model: {projection: 'planar', maximumDistance: 2500, cellSize: 13}
  });
  const nonTrivial = new Set([...mercator.codes, ...planar.codes]);
  expect(nonTrivial.has(V.hidden)).toBe(true);
  expect(nonTrivial.has(V.visible)).toBe(true);
  // Target angles agree to the Mercator-versus-plane scale error at 2 km (about 1e-3 relative).
  for (let index = 0; index < targets.length; index++) {
    if (mercator.codes[index] === V.noData || planar.codes[index] === V.noData) {
      expect(mercator.codes[index]).toBe(planar.codes[index]);
      continue;
    }
    const tolerance = isSoftwareDevice(device) ? 2 : 0.1;
    expect(Math.abs(mercator.rows[index * 4] - planar.rows[index * 4])).toBeLessThan(tolerance);
    if (!isSoftwareDevice(device)) {
      expect(mercator.codes[index]).toBe(planar.codes[index]);
    }
  }
});
