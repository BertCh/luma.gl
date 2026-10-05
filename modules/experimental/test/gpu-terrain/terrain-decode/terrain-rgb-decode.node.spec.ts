// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUTerrainRGBDecode} from '../../../src/gpu-terrain/terrain-decode/gpu-terrain-rgb-decode';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  decodeMapboxFloat64,
  decodeTerrariumFloat64,
  emulateMapboxFloat32,
  emulateTerrariumFloat32,
  getExpectedFloat32,
  measureMapboxRounding
} from './terrain-rgb-decode-oracle';

it('Terrarium float32 emulation equals the exact float64 decode for all 2^24 triples', () => {
  let mismatchCount = 0;
  let firstMismatch = '';
  for (let code = 0; code < 1 << 24; code++) {
    const red = code >> 16;
    const green = (code >> 8) & 255;
    const blue = code & 255;
    const exact = decodeTerrariumFloat64(red, green, blue);
    const emulated = emulateTerrariumFloat32(red, green, blue);
    // Bit-exact: the float64 value is exactly representable, and Object.is distinguishes -0.
    if (!Object.is(emulated, Math.fround(exact)) || emulated !== exact) {
      mismatchCount++;
      firstMismatch ||= `${red},${green},${blue}`;
    }
  }
  expect(mismatchCount, firstMismatch).toBe(0);
}, 120000);

it('Mapbox float32 emulation is one rounded multiply within the stated bound of M / 10', () => {
  let mismatchCount = 0;
  for (let code = 0; code < 1 << 24; code++) {
    const red = code >> 16;
    const green = (code >> 8) & 255;
    const blue = code & 255;
    const emulated = emulateMapboxFloat32(red, green, blue);
    const product = Math.fround((code - 100000) * Math.fround(0.1));
    if (!Object.is(emulated, product)) {
      mismatchCount++;
    }
  }
  expect(mismatchCount).toBe(0);
  const measurement = measureMapboxRounding();
  expect(measurement.maximumBoundRatio).toBeLessThanOrEqual(1);
  expect(measurement.maximumUlpDifference).toBe(1);
  expect(measurement.differingCodeCount).toBe(3355441);
  expect(measurement.maximumErrorMetresWithin9000).toBeLessThan(0.001);
  // The float64 formula -10000 + 0.1 * N agrees with M / 10 to rounding of float64 arithmetic.
  expect(Math.abs(decodeMapboxFloat64(255, 255, 255) - (16777215 - 100000) / 10)).toBeLessThan(
    1e-8
  );
}, 120000);

it('decodes the documented edge codes', () => {
  expect(getExpectedFloat32('terrarium', 0, 0, 0)).toBe(-32768);
  expect(getExpectedFloat32('terrarium', 255, 255, 255)).toBe(32767 + 255 / 256);
  expect(Object.is(getExpectedFloat32('terrarium', 128, 0, 0), 0)).toBe(true);
  expect(getExpectedFloat32('terrarium', 127, 255, 255)).toBe(-1 / 256);
  expect(getExpectedFloat32('terrarium', 128, 0, 1)).toBe(1 / 256);
  expect(getExpectedFloat32('terrarium', 128, 1, 0)).toBe(1);
  // Mapbox: code 100000 = (1, 134, 160) is 0 m; code 0 is -10000 m.
  expect(Object.is(getExpectedFloat32('mapbox', 1, 134, 160), 0)).toBe(true);
  expect(getExpectedFloat32('mapbox', 0, 0, 0)).toBe(-10000);
  expect(getExpectedFloat32('mapbox', 1, 134, 161)).toBe(Math.fround(0.1));
  expect(getExpectedFloat32('mapbox', 1, 134, 159)).toBe(-Math.fround(0.1));
  expect(getExpectedFloat32('mapbox', 255, 255, 255)).toBe(
    Math.fround((16777215 - 100000) * Math.fround(0.1))
  );
});

it('GPUTerrainRGBDecode validates props and schedules one node', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Record<string, unknown> = {}, extent: [number, number] = [4, 3]) => {
    instance++;
    const pixelCount = extent[0] * extent[1];
    return new GPUTerrainRGBDecode({
      width: extent[0],
      height: extent[1],
      encoding: 'terrarium',
      input: {buffer: createTransientView(graph, `words-${instance}`, 'uint32', pixelCount)},
      values: createTransientView(graph, `values-${instance}`, 'float32', pixelCount),
      validity: createTransientView(graph, `validity-${instance}`, 'uint32', pixelCount),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.recipe).toBe('terrain-rgb-decode');
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toEqual(['terrain-rgb-decode-decode']);
  expect(
    create({id: 'dem', encoding: 'mapbox', clampBathymetry: true, noDataRGB: [1, 2, 3]})
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['dem-decode']);
  expect(() => create({validRange: [-Infinity, Infinity]})).not.toThrow();

  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() => create({encoding: 'png'})).toThrow(/encoding/);
  expect(() => create({input: {}})).toThrow(/exactly one/);
  expect(() =>
    create({input: {buffer: createTransientView(graph, 'short-words', 'uint32', 11)}})
  ).toThrow(/one packed RGBA8 word per pixel/);
  expect(() => create({values: createTransientView(graph, 'short-values', 'float32', 11)})).toThrow(
    /one value per pixel/
  );
  expect(() =>
    create({validity: createTransientView(graph, 'short-validity', 'uint32', 13)})
  ).toThrow(/one value per pixel/);
  expect(() =>
    create({inputValidity: createTransientView(graph, 'short-input-validity', 'uint32', 5)})
  ).toThrow(/inputValidity/);
  expect(() => create({noDataRGB: [0, 256, 0]})).toThrow(/noDataRGB/);
  expect(() => create({validRange: [10, -10]})).toThrow(/validRange/);
  expect(() => create({validRange: [Number.NaN, 1]})).toThrow(/validRange/);
  const words = createTransientView(graph, 'aliased-words', 'uint32', 12);
  expect(() => create({input: {buffer: words}, validity: words})).toThrow(/share buffers/);
  const sharedOutput = createTransientView(graph, 'shared-output', 'uint32', 12);
  expect(() => create({validity: sharedOutput, inputValidity: sharedOutput})).toThrow(
    /share buffers/
  );
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainRGBDecode({
      width: 2,
      height: 2,
      encoding: 'mapbox',
      input: {buffer: createTransientView(otherGraph, 'foreign-words', 'uint32', 4)},
      values: createTransientView(otherGraph, 'foreign-values', 'float32', 4),
      validity: createTransientView(otherGraph, 'foreign-validity', 'uint32', 4)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
