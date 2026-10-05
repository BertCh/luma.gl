// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, getGPUConvolutionSupport} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  createGPUPointDensityGaussianKernel,
  getGPUPointDensityHexagonGridSize,
  GPUPointDensity
} from '../../../src/map-graphs/point-density';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeExtent,
  computeGridDensity,
  computeHexagonDensity,
  computeHistogram,
  convolveZero,
  createSeededPoints,
  findNearestHexagon
} from './point-density-oracle';

const FIXTURE_POSITIONS = [
  0.5,
  0.5,
  0.5,
  0.5,
  1.5,
  0.5,
  3.5,
  1.5,
  4,
  2,
  5,
  5,
  Number.NaN,
  0,
  2.5,
  1.5
];
const FIXTURE_WEIGHTS = [1, 2, 3, 4, 5, 6, 7, 8];

function expectClose(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    expect(actual[index]).toBeCloseTo(value, 5);
  }
}

function destroyAll(resources: {destroy(): void}[]): void {
  for (const resource of resources) {
    resource.destroy();
  }
}

it('GPUPointDensity grid statistics, texture, extent, and histogram follow per-frame bounds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const hasTexture = device.getTextureFormatCapabilities('r32float').store;
  const positionsBuffer = createInputBuffer(device, Float32Array.from(FIXTURE_POSITIONS));
  const weightsBuffer = createInputBuffer(device, Float32Array.from(FIXTURE_WEIGHTS));
  const valuesBuffer = createOutputBuffer(device, 8);
  const countsBuffer = createOutputBuffer(device, 8);
  const sumsBuffer = createOutputBuffer(device, 8);
  const meansBuffer = createOutputBuffer(device, 8);
  const extentBuffer = createOutputBuffer(device, 2);
  const histogramBuffer = createOutputBuffer(device, 4);
  const overflowBuffer = createOutputBuffer(device, 1);
  const textureReadbackBuffer = createOutputBuffer(device, 8);
  const textureValidityBuffer = createOutputBuffer(device, 8);
  const bounds = new GPUMapGraphParameterBuffer(device, {
    id: 'bounds',
    format: 'float32',
    length: 4,
    values: Float32Array.from([0, 0, 4, 2])
  });
  const texture = hasTexture
    ? device.createTexture({
        format: 'r32float',
        width: 4,
        height: 2,
        usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
      })
    : undefined;

  const graph = new GPUCommandGraph(device, {id: 'density-grid'});
  const textureView = texture
    ? graph.createTextureView(
        graph.importTexture(
          {id: 'texture', format: 'r32float', width: 4, height: 2, usage: texture.props.usage},
          texture
        ),
        {mipLevelCount: 1}
      )
    : undefined;
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 8),
      weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 8),
      bounds: bounds.importToGraph(graph),
      gridSize: [4, 2],
      statistic: 'mean',
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 8),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 8),
        sums: importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', 8),
        means: importGraphBuffer(graph, 'means', meansBuffer, 'float32', 8),
        extent: importGraphBuffer(graph, 'extent', extentBuffer, 'float32', 2),
        histogram: importGraphBuffer(graph, 'histogram', histogramBuffer, 'uint32', 4),
        overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1),
        texture: textureView as never
      }
    })
  );
  if (textureView) {
    new GPURasterTextureToBuffer({
      id: 'texture-readback',
      input: {id: 'texture-band', format: 'float32', storage: {kind: 'texture', view: textureView}},
      output: importGraphBuffer(graph, 'texture-readback', textureReadbackBuffer, 'float32', 8),
      outputValidity: importGraphBuffer(
        graph,
        'texture-validity',
        textureValidityBuffer,
        'uint32',
        8
      )
    }).addToGraph(graph);
  }
  const compiled = graph.compile();

  const verify = async (boundsValues: [number, number, number, number]) => {
    const oracle = computeGridDensity(FIXTURE_POSITIONS, FIXTURE_WEIGHTS, boundsValues, [4, 2]);
    const values = await readFloat32(valuesBuffer, 8);
    expect(await readUint32(countsBuffer, 8)).toEqual(oracle.counts);
    expectClose(await readFloat32(sumsBuffer, 8), oracle.sums);
    expectClose(await readFloat32(meansBuffer, 8), oracle.means);
    expectClose(values, oracle.means);
    const extent = computeExtent(oracle.means, oracle.counts);
    expectClose(await readFloat32(extentBuffer, 2), extent);
    expect(await readUint32(histogramBuffer, 4)).toEqual(
      computeHistogram(oracle.means, oracle.counts, extent, 4)
    );
    expect(await readUint32(overflowBuffer, 1)).toEqual([0]);
    if (textureView) {
      expectClose(await readFloat32(textureReadbackBuffer, 8), values);
    }
    return oracle;
  };

  submitGraph(device, compiled, undefined);
  const oracle = await verify([0, 0, 4, 2]);
  expect(oracle.counts).toEqual([2, 1, 0, 0, 0, 0, 1, 2]);
  expect(oracle.sums).toEqual([3, 3, 0, 0, 0, 0, 8, 9]);
  expect(await readUint32(histogramBuffer, 4)).toEqual([2, 1, 0, 1]);
  expectClose(await readFloat32(extentBuffer, 2), [1.5, 8]);

  bounds.write(Float32Array.from([0, 0, 2, 2]));
  submitGraph(device, compiled, undefined);
  await verify([0, 0, 2, 2]);

  compiled.destroy();
  bounds.destroy();
  destroyAll([
    positionsBuffer,
    weightsBuffer,
    valuesBuffer,
    countsBuffer,
    sumsBuffer,
    meansBuffer,
    extentBuffer,
    histogramBuffer,
    overflowBuffer,
    textureReadbackBuffer,
    textureValidityBuffer,
    ...(texture ? [texture] : [])
  ]);
});

it('GPUPointDensity writes an unweighted count field with internal counts', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positionsBuffer = createInputBuffer(device, Float32Array.from(FIXTURE_POSITIONS));
  const valuesBuffer = createOutputBuffer(device, 8);
  const extentBuffer = createOutputBuffer(device, 2);
  const graph = new GPUCommandGraph(device, {id: 'density-count'});
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 8),
      bounds: [0, 0, 4, 2],
      gridSize: [4, 2],
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 8),
        extent: importGraphBuffer(graph, 'extent', extentBuffer, 'float32', 2)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readFloat32(valuesBuffer, 8)).toEqual([2, 1, 0, 0, 0, 0, 1, 2]);
  expect(await readFloat32(extentBuffer, 2)).toEqual([1, 2]);
  compiled.destroy();
  destroyAll([positionsBuffer, valuesBuffer, extentBuffer]);
});

/** Seeded points that are not near a hexagon tie for any of the given radii. */
function createHexagonFixture(radii: number[]): number[] {
  const candidates = createSeededPoints(7, 200, [0, 0, 6, 6]);
  const points: number[] = [];
  for (let index = 0; index < candidates.length / 2 && points.length < 128; index++) {
    const [x, y] = [candidates[index * 2], candidates[index * 2 + 1]];
    const isClear = radii.every(radius => {
      const nearest = findNearestHexagon(x, y, 0, 0, radius, 10, 10);
      return nearest.secondNearest - nearest.nearest >= 1e-3 * radius;
    });
    if (isClear) {
      points.push(x, y);
    }
  }
  return [...points, Number.NaN, Number.NaN, 7, 7];
}

it('GPUPointDensity bins hexagons with a per-frame radius and reports lattice overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createHexagonFixture([1, 1.4, 0.6]);
  const rowCount = positions.length / 2;
  const weights = Array.from({length: rowCount}, (_, index) => index + 1);
  const gridSize = getGPUPointDensityHexagonGridSize([0, 0, 6, 6], 1);
  const cellCount = gridSize[0] * gridSize[1];
  const positionsBuffer = createInputBuffer(device, Float32Array.from(positions));
  const weightsBuffer = createInputBuffer(device, Float32Array.from(weights));
  const valuesBuffer = createOutputBuffer(device, cellCount);
  const countsBuffer = createOutputBuffer(device, cellCount);
  const sumsBuffer = createOutputBuffer(device, cellCount);
  const overflowBuffer = createOutputBuffer(device, 1);
  const radius = new GPUMapGraphParameterBuffer(device, {
    id: 'radius',
    format: 'float32',
    length: 1,
    values: Float32Array.from([1])
  });
  const graph = new GPUCommandGraph(device, {id: 'density-hexagon'});
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', rowCount),
      bounds: [0, 0, 6, 6],
      gridSize,
      binning: 'hexagon',
      hexagonRadius: radius.importToGraph(graph),
      statistic: 'sum',
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', cellCount),
        sums: importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', cellCount),
        overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();

  for (const [radiusValue, expectedOverflow] of [
    [1, 0],
    [1.4, 0],
    [0.6, 1]
  ] as const) {
    radius.write(Float32Array.from([radiusValue]));
    submitGraph(device, compiled, undefined);
    const oracle = computeHexagonDensity(positions, weights, [0, 0, 6, 6], gridSize, radiusValue);
    expect(oracle.overflow).toBe(expectedOverflow);
    expect(await readUint32(countsBuffer, cellCount)).toEqual(oracle.counts);
    expectClose(await readFloat32(sumsBuffer, cellCount), oracle.sums);
    expectClose(await readFloat32(valuesBuffer, cellCount), oracle.sums);
    expect(await readUint32(overflowBuffer, 1)).toEqual([expectedOverflow]);
  }

  compiled.destroy();
  radius.destroy();
  destroyAll([
    positionsBuffer,
    weightsBuffer,
    valuesBuffer,
    countsBuffer,
    sumsBuffer,
    overflowBuffer
  ]);
});

it('GPUPointDensity gives identical results for chunked positions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const firstBuffer = createInputBuffer(device, Float32Array.from(FIXTURE_POSITIONS.slice(0, 10)));
  const secondBuffer = createInputBuffer(device, Float32Array.from(FIXTURE_POSITIONS.slice(10)));
  const weightsBuffer = createInputBuffer(device, Float32Array.from(FIXTURE_WEIGHTS));
  const valuesBuffer = createOutputBuffer(device, 8);
  const countsBuffer = createOutputBuffer(device, 8);
  const graph = new GPUCommandGraph(device, {id: 'density-chunked'});
  graph.add(
    new GPUPointDensity({
      positions: createVectorView('positions', 'float32x2', [
        importGraphBuffer(graph, 'p0', firstBuffer, 'float32x2', 5),
        importGraphBuffer(graph, 'p1', secondBuffer, 'float32x2', 3)
      ]),
      weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 8),
      bounds: [0, 0, 4, 2],
      gridSize: [4, 2],
      statistic: 'sum',
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 8),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 8)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const oracle = computeGridDensity(FIXTURE_POSITIONS, FIXTURE_WEIGHTS, [0, 0, 4, 2], [4, 2]);
  expect(await readUint32(countsBuffer, 8)).toEqual(oracle.counts);
  expectClose(await readFloat32(valuesBuffer, 8), oracle.sums);
  compiled.destroy();
  destroyAll([firstBuffer, secondBuffer, weightsBuffer, valuesBuffer, countsBuffer]);
});

it('GPUPointDensity smooths the field with a per-frame kernel', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const support = getGPUConvolutionSupport(device, {
    width: 5,
    height: 5,
    kernelWidth: 3,
    kernelHeight: 3,
    boundary: 'zero'
  });
  if (!support.supported) {
    return;
  }
  const positionsBuffer = createInputBuffer(device, Float32Array.from([2.5, 2.5]));
  const valuesBuffer = createOutputBuffer(device, 25);
  const extentBuffer = createOutputBuffer(device, 2);
  const gaussian = createGPUPointDensityGaussianKernel(1);
  const kernel = new GPUMapGraphParameterBuffer(device, {
    id: 'kernel',
    format: 'float32',
    length: 9,
    values: gaussian
  });
  const graph = new GPUCommandGraph(device, {id: 'density-smooth'});
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 1),
      bounds: [0, 0, 5, 5],
      gridSize: [5, 5],
      smoothing: {kernel: kernel.importToGraph(graph), kernelWidth: 3, kernelHeight: 3},
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 25),
        extent: importGraphBuffer(graph, 'extent', extentBuffer, 'float32', 2)
      }
    })
  );
  const compiled = graph.compile();
  const counts = computeGridDensity([2.5, 2.5], undefined, [0, 0, 5, 5], [5, 5]).counts;

  submitGraph(device, compiled, undefined);
  expectClose(await readFloat32(valuesBuffer, 25), convolveZero(counts, 5, 5, gaussian, 3, 3));
  expectClose(await readFloat32(extentBuffer, 2), [0, gaussian[4]]);

  const box = new Float32Array(9).fill(1 / 9);
  kernel.write(box);
  submitGraph(device, compiled, undefined);
  expectClose(await readFloat32(valuesBuffer, 25), convolveZero(counts, 5, 5, box, 3, 3));
  expectClose(await readFloat32(extentBuffer, 2), [0, 1 / 9]);

  compiled.destroy();
  kernel.destroy();
  destroyAll([positionsBuffer, valuesBuffer, extentBuffer]);
});

it('GPUPointDensity handles empty input', async () => {
  const device = (await getWebGPUTestDevice()) as Device | null;
  if (!device) {
    return;
  }
  const positionsBuffer = createInputBuffer(device, new Float32Array(2));
  const buffers: Buffer[] = [
    createOutputBuffer(device, 4),
    createOutputBuffer(device, 4),
    createOutputBuffer(device, 2),
    createOutputBuffer(device, 1)
  ];
  const [valuesBuffer, countsBuffer, extentBuffer, overflowBuffer] = buffers;
  const graph = new GPUCommandGraph(device, {id: 'density-empty'});
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 0),
      bounds: [0, 0, 1, 1],
      gridSize: [2, 2],
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 4),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 4),
        extent: importGraphBuffer(graph, 'extent', extentBuffer, 'float32', 2),
        overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readFloat32(valuesBuffer, 4)).toEqual([0, 0, 0, 0]);
  expect(await readUint32(countsBuffer, 4)).toEqual([0, 0, 0, 0]);
  expect(await readFloat32(extentBuffer, 2)).toEqual([0, 0]);
  expect(await readUint32(overflowBuffer, 1)).toEqual([0]);
  compiled.destroy();
  destroyAll([positionsBuffer, ...buffers]);
});
