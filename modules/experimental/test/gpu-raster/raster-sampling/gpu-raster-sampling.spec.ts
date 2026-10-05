// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  type GPURasterSamplingSettings
} from '../../../src/gpu-raster/raster-sampling';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {createRandom, expectFloatArraysClose} from '../raster-algebra/raster-algebra-test-utils';
import {sampleRasterPointsOnCPU, type OracleRaster} from './raster-sampling-oracle';

type Scene = {
  raster: OracleRaster;
  positions: Float32Array;
};

/** Raster plus points; `dyadic` keeps every coordinate and weight exactly representable in f32. */
function createScene(
  width: number,
  height: number,
  pointCount: number,
  seed: number,
  options: {withFlags: boolean}
): Scene {
  const random = createRandom(seed);
  const values = new Float32Array(width * height);
  const validity = new Uint32Array(width * height).fill(1);
  for (let cell = 0; cell < values.length; cell++) {
    const roll = random();
    values[cell] = Math.floor(random() * 100);
    if (options.withFlags) {
      if (roll < 0.08) {
        values[cell] = NaN;
      } else if (roll < 0.11) {
        values[cell] = -9999;
      } else if (roll < 0.14) {
        validity[cell] = 0;
      }
    }
  }
  const positions = new Float32Array(pointCount * 2);
  for (let point = 0; point < pointCount; point++) {
    // Multiples of 1/8 covering a border outside the [0, width] x [0, height] extent.
    positions[2 * point] = Math.round(random() * (width + 4) * 8) / 8 - 2;
    positions[2 * point + 1] = Math.round(random() * (height + 4) * 8) / 8 - 2;
  }
  return {
    raster: {
      width,
      height,
      values,
      validity: options.withFlags ? validity : undefined,
      noDataValue: options.withFlags ? -9999 : undefined
    },
    positions
  };
}

async function createHarness(scene: Scene, withPointCount: boolean) {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return null;
  }
  const {raster, positions} = scene;
  const pointCapacity = positions.length / 2;
  const buffers = {
    raster: createInputBuffer(device, raster.values),
    validity: raster.validity ? createInputBuffer(device, raster.validity) : undefined,
    positions: createInputBuffer(device, positions),
    pointCount: createInputBuffer(device, new Uint32Array([pointCapacity])),
    values: createOutputBuffer(device, pointCapacity),
    validityOut: createOutputBuffer(device, pointCapacity)
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'sampling-parameters',
    format: 'float32',
    length: 12
  });
  const graph = new GPUCommandGraph(device, {id: 'sampling-graph'});
  graph.add(
    new GPURasterSampling({
      id: 'sampling',
      width: raster.width,
      height: raster.height,
      values: importGraphBuffer(graph, 'raster', buffers.raster, 'float32', raster.values.length),
      validity: buffers.validity
        ? importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', raster.values.length)
        : undefined,
      noDataValue: raster.noDataValue,
      positions: importGraphBuffer(
        graph,
        'positions',
        buffers.positions,
        'float32x2',
        pointCapacity
      ),
      pointCount: withPointCount
        ? importGraphBuffer(graph, 'point-count', buffers.pointCount, 'uint32', 1)
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        values: importGraphBuffer(graph, 'values', buffers.values, 'float32', pointCapacity),
        validity: importGraphBuffer(
          graph,
          'validity-out',
          buffers.validityOut,
          'uint32',
          pointCapacity
        )
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings: GPURasterSamplingSettings, activeCount?: number) {
      parameterBuffer.write(getGPURasterSamplingParameterValues(settings));
      if (activeCount !== undefined) {
        buffers.pointCount.write(new Uint32Array([activeCount]));
      }
      submitGraph(device, compiled, undefined);
      return {
        values: await readFloat32(buffers.values, pointCapacity),
        validity: await readUint32(buffers.validityOut, pointCapacity)
      };
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of Object.values(buffers)) {
        buffer?.destroy();
      }
    }
  };
}

it('GPURasterSampling matches the CPU oracle exactly on dyadic scenes as method, policy and count change', async () => {
  const scene = createScene(32, 16, 600, 11, {withFlags: true});
  const harness = await createHarness(scene, true);
  if (!harness) {
    return;
  }
  const extent = [0, 0, 32, 16] as const;
  const frames: {
    method: 'nearest' | 'bilinear' | 'bicubic';
    noDataPolicy: 'strict' | 'renormalize';
    activeCount: number;
  }[] = [
    {method: 'nearest', noDataPolicy: 'strict', activeCount: 600},
    {method: 'bilinear', noDataPolicy: 'strict', activeCount: 600},
    {method: 'bilinear', noDataPolicy: 'renormalize', activeCount: 600},
    {method: 'bicubic', noDataPolicy: 'strict', activeCount: 600},
    {method: 'bicubic', noDataPolicy: 'renormalize', activeCount: 600},
    {method: 'bicubic', noDataPolicy: 'renormalize', activeCount: 250},
    {method: 'nearest', noDataPolicy: 'renormalize', activeCount: 0},
    {method: 'bilinear', noDataPolicy: 'strict', activeCount: 1000}
  ];
  for (const frame of frames) {
    const settings: GPURasterSamplingSettings = {
      width: 32,
      height: 16,
      extent,
      method: frame.method,
      noDataPolicy: frame.noDataPolicy
    };
    const result = await harness.run(settings, frame.activeCount);
    const expected = sampleRasterPointsOnCPU(
      scene.raster,
      getGPURasterSamplingParameterValues(settings),
      scene.positions,
      frame.activeCount
    );
    const label = `${frame.method}/${frame.noDataPolicy}/${frame.activeCount}`;
    // Strict and nearest arithmetic is exact; renormalize divides, which WebGPU only bounds.
    expectFloatArraysClose(
      result.values,
      expected,
      frame.noDataPolicy === 'renormalize' && frame.method !== 'nearest' ? 4 : 0,
      label
    );
    expect(result.validity, label).toEqual(
      Array.from(expected, value => (Number.isNaN(value) ? 0 : 1))
    );
    if (frame.activeCount === 600) {
      // The scene exercises outside, hole and ordinary samples.
      expect(expected.some(Number.isNaN)).toBe(true);
      expect(expected.some(value => !Number.isNaN(value))).toBe(true);
    }
  }
  expect(harness.rebuildCount).toBe(0);
  harness.destroy();
});

it('GPURasterSampling agrees with the oracle on a non-dyadic georeference within a few ULP', async () => {
  const width = 40;
  const height = 25;
  const extent = [-10.3, 5.1, 20.7, 18.2] as const;
  const random = createRandom(5);
  const values = new Float32Array(width * height);
  for (let cell = 0; cell < values.length; cell++) {
    values[cell] = cell % 37 === 0 ? NaN : 50 + 40 * Math.sin(cell * 0.13) + random();
  }
  const positions = new Float32Array(1200);
  for (let point = 0; point < 600; point++) {
    positions[2 * point] = extent[0] - 1 + random() * (extent[2] - extent[0] + 2);
    positions[2 * point + 1] = extent[1] - 1 + random() * (extent[3] - extent[1] + 2);
  }
  const scene: Scene = {raster: {width, height, values}, positions};
  const harness = await createHarness(scene, false);
  if (!harness) {
    return;
  }
  for (const method of ['nearest', 'bilinear', 'bicubic'] as const) {
    for (const noDataPolicy of ['strict', 'renormalize'] as const) {
      const settings: GPURasterSamplingSettings = {width, height, extent, method, noDataPolicy};
      const result = await harness.run(settings);
      const expected = sampleRasterPointsOnCPU(
        scene.raster,
        getGPURasterSamplingParameterValues(settings),
        positions
      );
      for (let point = 0; point < 600; point++) {
        const gpu = result.values[point];
        const cpu = expected[point];
        const label = `${method}/${noDataPolicy} point ${point} (${gpu} vs ${cpu})`;
        expect(Number.isNaN(gpu), label).toBe(Number.isNaN(cpu));
        if (!Number.isNaN(cpu)) {
          // FMA contraction and f32 position rounding: a few ULP of the 100-scale data per cell.
          expect(Math.abs(gpu - cpu), label).toBeLessThanOrEqual(method === 'nearest' ? 0 : 2e-4);
        }
      }
    }
  }
  expect(harness.rebuildCount).toBe(0);
  harness.destroy();
});

it('GPURasterSampling reproduces an analytic plane under bilinear and bicubic sampling', async () => {
  const width = 64;
  const height = 48;
  const extent = [100, -20, 164, 4] as const;
  const cellWidth = (extent[2] - extent[0]) / width;
  const cellHeight = (extent[3] - extent[1]) / height;
  const plane = (x: number, y: number) => 3 * x - 2 * y + 1;
  const values = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      values[row * width + column] = plane(
        extent[0] + (column + 0.5) * cellWidth,
        extent[1] + (row + 0.5) * cellHeight
      );
    }
  }
  const random = createRandom(21);
  const positions = new Float32Array(400);
  for (let point = 0; point < 200; point++) {
    // Keep the whole 4x4 support away from the clamped border.
    positions[2 * point] = extent[0] + 2 * cellWidth + random() * (width - 4) * cellWidth;
    positions[2 * point + 1] = extent[1] + 2 * cellHeight + random() * (height - 4) * cellHeight;
  }
  const harness = await createHarness({raster: {width, height, values}, positions}, false);
  if (!harness) {
    return;
  }
  for (const method of ['bilinear', 'bicubic'] as const) {
    const result = await harness.run({width, height, extent, method});
    for (let point = 0; point < 200; point++) {
      const expected = plane(positions[2 * point], positions[2 * point + 1]);
      // f32 data of magnitude ~500: bound is a few ULP.
      expect(Math.abs(result.values[point] - expected), `${method} ${point}`).toBeLessThan(5e-3);
    }
  }
  // Points on the extent boundary stay valid (clamped); just outside they are NaN.
  expect(harness.rebuildCount).toBe(0);
  harness.destroy();
});

it('GPURasterSampling handles centres, edges and outside points and clamps to the edge', async () => {
  const width = 4;
  const height = 4;
  const values = new Float32Array(16);
  for (let cell = 0; cell < 16; cell++) {
    values[cell] = cell * 10;
  }
  const points = [
    [0.5, 0.5], // centre of cell 0
    [3.5, 3.5], // centre of cell 15
    [0, 0], // corner: clamped to cell 0
    [4, 4], // opposite corner: cell 15
    [1, 1], // boundary shared by four cells
    [-0.001, 2], // outside
    [2, 4.001], // outside
    [4, 2.5] // on the max edge, centre row 2
  ];
  const positions = new Float32Array(points.flat());
  const harness = await createHarness({raster: {width, height, values}, positions}, false);
  if (!harness) {
    return;
  }
  const extent = [0, 0, 4, 4] as const;
  const nearest = await harness.run({width, height, extent, method: 'nearest'});
  expect(nearest.values.map(value => (Number.isNaN(value) ? -1 : value))).toEqual([
    0, 150, 0, 150, 50, -1, -1, 110
  ]);
  const bilinear = await harness.run({width, height, extent, method: 'bilinear'});
  expect(bilinear.values.map(value => (Number.isNaN(value) ? -1 : value))).toEqual([
    0, 150, 0, 150, 25, -1, -1, 110
  ]);
  const bicubic = await harness.run({width, height, extent, method: 'bicubic'});
  expect(bicubic.values[0]).toBe(0);
  expect(bicubic.values[1]).toBe(150);
  // Next to the border the clamped 4x4 support is no longer planar: it must match the oracle
  // exactly (dyadic weights) and differ from the bilinear plane value 25.
  const expected = sampleRasterPointsOnCPU(
    {width, height, values},
    getGPURasterSamplingParameterValues({width, height, extent, method: 'bicubic'}),
    positions
  );
  expectFloatArraysClose(bicubic.values, expected, 0, 'bicubic clamped');
  expect(bicubic.values[4]).not.toBe(25);
  expect(bicubic.validity).toEqual([1, 1, 1, 1, 1, 0, 0, 1]);
  harness.destroy();
});
