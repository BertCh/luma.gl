// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  createGPUPointDensityGaussianKernel,
  getGPUPointDensityHexagonGridSize,
  GPUPointDensity
} from '../../../src/geospatial/point-density';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  computeExtent,
  computeGridDensity,
  computeHexagonDensity,
  computeHistogram,
  convolveZero,
  createSeededPoints
} from './point-density-oracle';

const BOUNDS: [number, number, number, number] = [0, 0, 8, 8];
const GRID: [number, number] = [8, 8];
const CELL_COUNT = 64;
const BIN_COUNT = 6;
const EXTREME_WEIGHT = 1e6;

type Fixture = {positions: number[]; weights: number[]; rowCount: number};

/** Seeded points where every 4th row gets an in-bounds position and an extreme weight. */
function createFixture(rowCount: number): Fixture {
  const positions = createSeededPoints(11, rowCount, BOUNDS);
  const weights = Array.from({length: rowCount}, (_, index) => 1 + (index % 5));
  return {positions, weights, rowCount};
}

/** Mask pattern `frame`: rows with `(index + frame) % 4 === 0` are masked. */
function createMask(rowCount: number, frame: number): number[] {
  return Array.from({length: rowCount}, (_, index) => ((index + frame) % 4 === 0 ? 0 : 7));
}

/** Gives masked rows an extreme weight so any leak is visible in sums, extent and histogram. */
function poisonMaskedWeights(weights: number[], mask: number[]): number[] {
  return weights.map((weight, index) => (mask[index] ? weight : EXTREME_WEIGHT));
}

/** CPU oracle input: masked rows become NaN positions. */
function applyMask(positions: number[], mask: number[]): number[] {
  const result = positions.slice();
  for (const [index, value] of mask.entries()) {
    if (!value) {
      result[index * 2] = Number.NaN;
      result[index * 2 + 1] = Number.NaN;
    }
  }
  return result;
}

function expectClose(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    expect(actual[index]).toBeCloseTo(value, 4);
  }
}

it('GPUPointDensity mask: grid statistics, extent, and histogram ignore masked rows per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, weights, rowCount} = createFixture(300);
  const frame0Mask = createMask(rowCount, 0);
  const positionsBuffer = createInputBuffer(device, Float32Array.from(positions));
  const weightsBuffer = createInputBuffer(device, Float32Array.from(weights));
  const maskBuffer = createInputBuffer(device, Uint32Array.from(frame0Mask));
  const valuesBuffer = createOutputBuffer(device, CELL_COUNT);
  const countsBuffer = createOutputBuffer(device, CELL_COUNT);
  const sumsBuffer = createOutputBuffer(device, CELL_COUNT);
  const extentBuffer = createOutputBuffer(device, 2);
  const histogramBuffer = createOutputBuffer(device, BIN_COUNT);
  const graph = new GPUCommandGraph(device, {id: 'density-mask-grid'});
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', rowCount),
      mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rowCount),
      bounds: BOUNDS,
      gridSize: GRID,
      statistic: 'sum',
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', CELL_COUNT),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', CELL_COUNT),
        sums: importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', CELL_COUNT),
        extent: importGraphBuffer(graph, 'extent', extentBuffer, 'float32', 2),
        histogram: importGraphBuffer(graph, 'histogram', histogramBuffer, 'uint32', BIN_COUNT)
      }
    })
  );
  const compiled = graph.compile();

  // Frame 0 and 1 use different masks and extreme masked weights, on one compiled graph. Frame 2
  // is all ones: it must equal the unmasked result.
  for (const frame of [0, 1, 2]) {
    const mask = frame === 2 ? new Array(rowCount).fill(1) : createMask(rowCount, frame);
    const poisoned = poisonMaskedWeights(weights, mask);
    maskBuffer.write(Uint32Array.from(mask));
    weightsBuffer.write(Float32Array.from(poisoned));
    submitGraph(device, compiled, undefined);

    const oracle = computeGridDensity(applyMask(positions, mask), poisoned, BOUNDS, GRID);
    expect(await readUint32(countsBuffer, CELL_COUNT)).toEqual(oracle.counts);
    expectClose(await readFloat32(sumsBuffer, CELL_COUNT), oracle.sums);
    expectClose(await readFloat32(valuesBuffer, CELL_COUNT), oracle.sums);
    const extent = computeExtent(oracle.sums, oracle.counts);
    const actualExtent = await readFloat32(extentBuffer, 2);
    expectClose(actualExtent, extent);
    if (frame < 2) {
      // Control: a masked row's extreme weight would otherwise dominate the extent.
      expect(actualExtent[1]).toBeLessThan(EXTREME_WEIGHT);
    }
    expect(await readUint32(histogramBuffer, BIN_COUNT)).toEqual(
      computeHistogram(oracle.sums, oracle.counts, extent, BIN_COUNT)
    );
  }

  compiled.destroy();
  for (const buffer of [
    positionsBuffer,
    weightsBuffer,
    maskBuffer,
    valuesBuffer,
    countsBuffer,
    sumsBuffer,
    extentBuffer,
    histogramBuffer
  ]) {
    buffer.destroy();
  }
});

it('GPUPointDensity mask: chunked positions and smoothing input exclude masked rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, rowCount} = createFixture(120);
  const split = 50;
  const mask = createMask(rowCount, 1);
  const kernel = createGPUPointDensityGaussianKernel(1);
  const firstBuffer = createInputBuffer(device, Float32Array.from(positions.slice(0, split * 2)));
  const secondBuffer = createInputBuffer(device, Float32Array.from(positions.slice(split * 2)));
  const maskBuffer = createInputBuffer(device, Uint32Array.from(mask));
  const kernelBuffer = createInputBuffer(device, kernel);
  const valuesBuffer = createOutputBuffer(device, CELL_COUNT);
  const graph = new GPUCommandGraph(device, {id: 'density-mask-chunked'});
  graph.add(
    new GPUPointDensity({
      positions: createVectorView('positions', 'float32x2', [
        importGraphBuffer(graph, 'p0', firstBuffer, 'float32x2', split),
        importGraphBuffer(graph, 'p1', secondBuffer, 'float32x2', rowCount - split)
      ]),
      mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rowCount),
      bounds: BOUNDS,
      gridSize: GRID,
      smoothing: {
        kernel: importGraphBuffer(graph, 'kernel', kernelBuffer, 'float32', 9),
        kernelWidth: 3,
        kernelHeight: 3
      },
      output: {values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', CELL_COUNT)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const oracle = computeGridDensity(applyMask(positions, mask), undefined, BOUNDS, GRID);
  expectClose(
    await readFloat32(valuesBuffer, CELL_COUNT),
    convolveZero(oracle.counts, 8, 8, kernel, 3, 3)
  );

  compiled.destroy();
  for (const buffer of [firstBuffer, secondBuffer, maskBuffer, kernelBuffer, valuesBuffer]) {
    buffer.destroy();
  }
});

it('GPUPointDensity mask: hexagon binning ignores masked rows and extreme masked weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const radius = 1;
  const {positions, weights, rowCount} = createFixture(300);
  const gridSize = getGPUPointDensityHexagonGridSize(BOUNDS, radius);
  const cellCount = gridSize[0] * gridSize[1];
  const positionsBuffer = createInputBuffer(device, Float32Array.from(positions));
  const weightsBuffer = createInputBuffer(device, Float32Array.from(weights));
  const maskBuffer = createInputBuffer(device, Uint32Array.from(createMask(rowCount, 0)));
  const valuesBuffer = createOutputBuffer(device, cellCount);
  const countsBuffer = createOutputBuffer(device, cellCount);
  const sumsBuffer = createOutputBuffer(device, cellCount);
  const extentBuffer = createOutputBuffer(device, 2);
  const histogramBuffer = createOutputBuffer(device, BIN_COUNT);
  const graph = new GPUCommandGraph(device, {id: 'density-mask-hexagon'});
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', rowCount),
      mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rowCount),
      bounds: BOUNDS,
      gridSize,
      binning: 'hexagon',
      hexagonRadius: radius,
      statistic: 'sum',
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', cellCount),
        sums: importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', cellCount),
        extent: importGraphBuffer(graph, 'extent', extentBuffer, 'float32', 2),
        histogram: importGraphBuffer(graph, 'histogram', histogramBuffer, 'uint32', BIN_COUNT)
      }
    })
  );
  const compiled = graph.compile();

  for (const frame of [0, 3]) {
    const mask = createMask(rowCount, frame);
    const poisoned = poisonMaskedWeights(weights, mask);
    maskBuffer.write(Uint32Array.from(mask));
    weightsBuffer.write(Float32Array.from(poisoned));
    submitGraph(device, compiled, undefined);

    const oracle = computeHexagonDensity(
      applyMask(positions, mask),
      poisoned,
      BOUNDS,
      gridSize,
      radius
    );
    expect(await readUint32(countsBuffer, cellCount)).toEqual(oracle.counts);
    expectClose(await readFloat32(sumsBuffer, cellCount), oracle.sums);
    const extent = computeExtent(oracle.sums, oracle.counts);
    const actualExtent = await readFloat32(extentBuffer, 2);
    expectClose(actualExtent, extent);
    expect(actualExtent[1]).toBeLessThan(EXTREME_WEIGHT);
    expect(await readUint32(histogramBuffer, BIN_COUNT)).toEqual(
      computeHistogram(oracle.sums, oracle.counts, extent, BIN_COUNT)
    );
  }

  compiled.destroy();
  for (const buffer of [
    positionsBuffer,
    weightsBuffer,
    maskBuffer,
    valuesBuffer,
    countsBuffer,
    sumsBuffer,
    extentBuffer,
    histogramBuffer
  ]) {
    buffer.destroy();
  }
});
