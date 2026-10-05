// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPURasterStretchParameterValues,
  GPU_RASTER_STRETCH_PARAMETER_LENGTH,
  GPU_RASTER_STRETCH_STATISTICS_INDEX,
  GPURasterStretch
} from '../../../src/gpu-raster/raster-stretch';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeRasterStretchStatistics,
  getPaletteColorOnCPU,
  packColor,
  stretchValueOnCPU
} from './raster-stretch-oracle';

let serial = 0;

function view<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

function withGraph(callback: (graph: GPUCommandGraph) => void): void {
  const device = createNullWebGPUDevice();
  callback(new GPUCommandGraph(device));
  device.destroy();
}

it('raster-stretch parameter packer lays out settings and validates ranges', () => {
  const defaults = getGPURasterStretchParameterValues();
  expect(defaults.length).toBe(GPU_RASTER_STRETCH_PARAMETER_LENGTH);
  expect(Array.from(defaults.subarray(0, 4))).toEqual([0, 0, 1 << 24, 1 << 24]);
  expect(Array.from(defaults.subarray(4, 8))).toEqual([0, 0, 0, 0]);
  expect(defaults[8]).toBeCloseTo(0.02, 6);
  expect(defaults[9]).toBeCloseTo(0.98, 6);
  expect(defaults[10]).toBe(1);
  expect(defaults[12]).toBe(0.5);
  const packed = getGPURasterStretchParameterValues({
    window: [2.7, -3, 9, 11],
    domain: [1, 5],
    mode: 'equalize',
    percentiles: [1, 99],
    gamma: 2,
    sigmoidContrast: 5,
    sigmoidMidpoint: 0.25,
    paletteInterpolation: 'linear'
  });
  expect(Array.from(packed)).toEqual([
    2,
    0,
    9,
    11,
    1,
    1,
    5,
    2,
    Math.fround(0.01),
    Math.fround(0.99),
    2,
    5,
    0.25,
    1,
    0,
    0
  ]);
  expect(GPU_RASTER_STRETCH_STATISTICS_INDEX.validCount).toBe(4);
  expect(() => getGPURasterStretchParameterValues({domain: [3, 2]})).toThrow(/domain/);
  expect(() => getGPURasterStretchParameterValues({domain: [0, NaN]})).toThrow(/domain/);
  expect(() => getGPURasterStretchParameterValues({percentiles: [90, 10]})).toThrow(/percentiles/);
  expect(() => getGPURasterStretchParameterValues({percentiles: [-1, 10]})).toThrow(/percentiles/);
  expect(() => getGPURasterStretchParameterValues({gamma: 0})).toThrow(/gamma/);
  expect(() => getGPURasterStretchParameterValues({sigmoidContrast: -1})).toThrow(
    /sigmoidContrast/
  );
  expect(() => getGPURasterStretchParameterValues({sigmoidMidpoint: 1})).toThrow(/sigmoidMidpoint/);
  expect(() => getGPURasterStretchParameterValues({mode: 'jenks' as 'linear'})).toThrow(
    /not supported/
  );
  expect(() => getGPURasterStretchParameterValues({}, new Float32Array(4))).toThrow(/target/);
  expect(() => getGPURasterStretchParameterValues({window: [NaN, 0, 1, 1]})).toThrow(/window/);
});

it('raster-stretch oracle replicates the documented definitions', () => {
  // Histogram, CDF, percentile interpolation and degenerate mapping on a tiny raster.
  const scene = {
    values: Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, NaN, Infinity]),
    width: 13,
    height: 1,
    binCount: 16,
    lutSize: 4
  };
  const parameters = getGPURasterStretchParameterValues({
    mode: 'percentile',
    percentiles: [0, 100]
  });
  const stats = computeRasterStretchStatistics(scene, parameters);
  expect(stats.validCount).toBe(11);
  expect([stats.domainMin, stats.domainMax]).toEqual([0, 10]);
  expect(stats.total).toBe(11);
  expect(stats.cdf[15]).toBe(11);
  // 0th percentile is the lower edge of the first occupied bin, 100th the upper edge of the last.
  expect(stats.lo).toBe(0);
  expect(stats.hi).toBeCloseTo(10, 5);
  expect(stretchValueOnCPU(5, parameters, stats, 16)).toBeCloseTo(0.5, 5);
  expect(stretchValueOnCPU(Infinity, parameters, stats, 16)).toBe(1);
  expect(stretchValueOnCPU(-Infinity, parameters, stats, 16)).toBe(0);
  // Degenerate range.
  const constant = computeRasterStretchStatistics(
    {...scene, values: new Float32Array(13).fill(4)},
    getGPURasterStretchParameterValues()
  );
  expect(constant.lo).toBe(constant.hi);
  expect(stretchValueOnCPU(4, getGPURasterStretchParameterValues(), constant, 16)).toBe(0.5);
  // Sigmoid keeps the end points.
  const sigmoid = getGPURasterStretchParameterValues({sigmoidContrast: 8});
  const linear = computeRasterStretchStatistics(scene, sigmoid);
  expect(stretchValueOnCPU(0, sigmoid, linear, 16)).toBeCloseTo(0, 12);
  expect(stretchValueOnCPU(10, sigmoid, linear, 16)).toBeCloseTo(1, 12);
  expect(stretchValueOnCPU(5, sigmoid, linear, 16)).toBeCloseTo(0.5, 6);
  // Palette: nearest and linear lookup, nodata is transparent.
  const palette = Uint32Array.from([packColor(0, 0, 0, 255), packColor(100, 200, 50, 255)]);
  expect(getPaletteColorOnCPU(0.49, palette, false)).toBe(palette[0]);
  expect(getPaletteColorOnCPU(0.5, palette, false)).toBe(palette[1]);
  expect(getPaletteColorOnCPU(0.5, palette, true)).toBe(packColor(50, 100, 25, 255));
  expect(getPaletteColorOnCPU(NaN, palette, true)).toBe(0);
});

it('GPURasterStretch validates props', () => {
  withGraph(graph => {
    const values = view(graph, 'float32', 12);
    const parameters = view(graph, 'float32', GPU_RASTER_STRETCH_PARAMETER_LENGTH);
    const base = {values, width: 4, height: 3, parameters};
    expect(() => new GPURasterStretch({...base, output: {}})).toThrow(/at least one output/);
    expect(
      () => new GPURasterStretch({...base, output: {colors: view(graph, 'uint32', 12)}})
    ).toThrow(/require palette/);
    expect(
      () =>
        new GPURasterStretch({
          ...base,
          output: {lutColors: view(graph, 'uint32', 256)}
        })
    ).toThrow(/require palette/);
    expect(
      () =>
        new GPURasterStretch({
          ...base,
          binCount: 8,
          output: {stretched: view(graph, 'float32', 12)}
        })
    ).toThrow(/binCount/);
    expect(
      () =>
        new GPURasterStretch({
          ...base,
          binCount: 100000,
          output: {statistics: view(graph, 'float32', 8)}
        })
    ).toThrow(/binCount/);
    expect(
      () => new GPURasterStretch({...base, lutSize: 1, output: {lut: view(graph, 'float32', 1)}})
    ).toThrow(/lutSize/);
    expect(
      () =>
        new GPURasterStretch({...base, width: 0, output: {stretched: view(graph, 'float32', 12)}})
    ).toThrow(/width/);
    expect(
      () =>
        new GPURasterStretch({
          ...base,
          values: view(graph, 'float32', 11),
          output: {stretched: view(graph, 'float32', 12)}
        })
    ).toThrow(/values/);
    expect(
      () =>
        new GPURasterStretch({
          ...base,
          values: view(graph, 'uint32', 12) as never,
          output: {stretched: view(graph, 'float32', 12)}
        })
    ).toThrow(/float32/);
    expect(
      () => new GPURasterStretch({...base, output: {stretched: view(graph, 'float32', 11)}})
    ).toThrow(/output.stretched/);
    expect(
      () =>
        new GPURasterStretch({
          ...base,
          parameters: view(graph, 'float32', 8),
          output: {stretched: view(graph, 'float32', 12)}
        })
    ).toThrow(/parameters/);
    expect(
      () => new GPURasterStretch({...base, output: {histogram: view(graph, 'uint32', 2000)}})
    ).toThrow(/exactly binCount/);
    expect(
      () =>
        new GPURasterStretch({
          ...base,
          noDataValue: NaN,
          output: {stretched: view(graph, 'float32', 12)}
        })
    ).toThrow(/noDataValue/);
    expect(() => new GPURasterStretch({...base, output: {stretched: values as never}})).toThrow(
      /share buffers/
    );
    expect(
      () =>
        new GPURasterStretch({
          ...base,
          palette: view(graph, 'float32', 4) as never,
          output: {stretched: view(graph, 'float32', 12)}
        })
    ).toThrow(/uint32/);
  });
});

it('GPURasterStretch wires deterministic nodes within the binding limit', () => {
  withGraph(graph => {
    const stretch = new GPURasterStretch({
      id: 'stretch',
      values: view(graph, 'float32', 12),
      width: 4,
      height: 3,
      validity: view(graph, 'uint32', 12),
      regionMask: view(graph, 'uint32', 12),
      parameters: view(graph, 'float32', GPU_RASTER_STRETCH_PARAMETER_LENGTH),
      palette: view(graph, 'uint32', 5),
      binCount: 64,
      lutSize: 32,
      output: {
        stretched: view(graph, 'float32', 12),
        colors: view(graph, 'uint32', 12),
        lut: view(graph, 'float32', 32),
        lutColors: view(graph, 'uint32', 32),
        histogram: view(graph, 'uint32', 64),
        statistics: view(graph, 'float32', 8)
      }
    });
    const ids = stretch.getCommandNodes(graph).map(node => node.id);
    expect(ids).toEqual(
      expect.arrayContaining([
        'stretch-clear-keys',
        'stretch-reduce',
        'stretch-domain',
        'stretch-clear-histogram',
        'stretch-histogram',
        'stretch-finalize',
        'stretch-lut',
        'stretch-apply'
      ])
    );
    expect(ids.indexOf('stretch-reduce')).toBeLessThan(ids.indexOf('stretch-histogram'));
    expect(ids.indexOf('stretch-histogram')).toBeLessThan(ids.indexOf('stretch-finalize'));
    expect(ids.indexOf('stretch-finalize')).toBeLessThan(ids.indexOf('stretch-apply'));
    expect(new Set(ids).size).toBe(ids.length);
  });
  // A histogram-only contributor stops after the finalize kernel.
  withGraph(graph => {
    const stretch = new GPURasterStretch({
      values: view(graph, 'float32', 12),
      width: 4,
      height: 3,
      parameters: view(graph, 'float32', GPU_RASTER_STRETCH_PARAMETER_LENGTH),
      output: {histogram: view(graph, 'uint32', 1024)}
    });
    const ids = stretch.getCommandNodes(graph).map(node => node.id);
    expect(ids.some(id => id.endsWith('-apply') || id.endsWith('-lut'))).toBe(false);
    expect(ids.at(-1)).toBe('raster-stretch-finalize');
  });
});
