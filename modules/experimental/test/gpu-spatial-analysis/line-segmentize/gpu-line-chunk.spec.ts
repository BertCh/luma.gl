// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPULineChunkParameterValues,
  GPULineChunk,
  type GPULineChunkParameters
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {createFlatPaths, createRandom, type FlatPaths} from './line-segmentize-oracle';
import {chunkPaths} from './line-chunk-oracle';
import {
  createLinePathFixture,
  expectLinePathParity,
  getSphericalErrorMeters
} from './line-path-test-utils';

function createChunkFixture(
  device: Device,
  paths: FlatPaths,
  options: {
    mode: 'chunk' | 'substring';
    capacity: number;
    pathCapacity: number;
    spherical?: boolean;
  }
) {
  return createLinePathFixture(device, {
    inputs: {
      positions: {values: paths.positions, format: 'float32x2'},
      pathOffsets: {values: paths.pathOffsets, format: 'uint32'}
    },
    capacity: options.capacity,
    pathCapacity: options.pathCapacity,
    createContributor: (inputs, parameters, output) =>
      new GPULineChunk({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        pathOffsets: inputs['pathOffsets'] as GraphDataView<'uint32'>,
        mode: options.mode,
        coordinateSystem: options.spherical ? 'spherical' : 'planar',
        parameters,
        output
      })
  });
}

function createRandomPaths(seed: number): FlatPaths {
  const random = createRandom(seed);
  return createFlatPaths(
    Array.from({length: 80}, (_, pathIndex) => {
      if (pathIndex % 19 === 3) {
        return [];
      }
      if (pathIndex % 17 === 5) {
        return [[random() * 50, random() * 50]];
      }
      if (pathIndex % 23 === 7) {
        // Zero-length path with repeated vertices.
        return [
          [5, 5],
          [5, 5]
        ];
      }
      let x = random() * 50;
      let y = random() * 50;
      return Array.from({length: 2 + Math.floor(random() * 9)}, (__, vertex) => {
        if (vertex > 0 && random() > 0.1) {
          x += (random() - 0.5) * 8;
          y += (random() - 0.5) * 8;
        }
        return [x, y];
      });
    })
  );
}

it('GPULineChunk splits paths into chunks like the f64 oracle, per-frame chunk length', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createRandomPaths(3);
  const capacity = 6000;
  const pathCapacity = 1500;
  const fixture = createChunkFixture(device, paths, {mode: 'chunk', capacity, pathCapacity});
  for (const chunkLength of [3.3, 7.77, 50, 0, 1.01]) {
    const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkPaths(paths, {mode: 'chunk', chunkLength, pathCapacity});
    expect(actual.pathCount).toBe(Math.min(expected.pieceTotal, pathCapacity));
    expect(actual.sourcePaths).toEqual(expected.sourcePaths);
    expectLinePathParity(actual, expected, capacity, 2e-5 * 60, 2e-5);
    expect(actual.overflow).toBe(0);
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineChunk reports path and vertex capacity overflow with valid offsets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createRandomPaths(5);
  const pathCapacity = 40;
  const capacity = 150;
  const fixture = createChunkFixture(device, paths, {mode: 'chunk', capacity, pathCapacity});
  const chunkLength = 2.5;
  const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
  const expected = chunkPaths(paths, {mode: 'chunk', chunkLength, pathCapacity});
  expect(expected.pieceTotal).toBeGreaterThan(pathCapacity);
  expect(actual.overflow).toBe(1);
  expect(actual.pathCount).toBe(pathCapacity);
  expectLinePathParity(actual, expected, capacity, 2e-3, 2e-5, undefined, 1);
  // A tiny chunk length is capped instead of wrapping counts.
  const tiny = await fixture.run(getGPULineChunkParameterValues({chunkLength: 1e-30}));
  expect(tiny.overflow).toBe(1);
  expect(tiny.pathCount).toBe(pathCapacity);
  fixture.destroy();
});

it('GPULineChunk extracts substrings by measure range', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createRandomPaths(7);
  const pathCount = paths.pathOffsets.length - 1;
  const capacity = 2000;
  const fixture = createChunkFixture(device, paths, {
    mode: 'substring',
    capacity,
    pathCapacity: pathCount
  });
  const ranges: GPULineChunkParameters[] = [
    {startMeasure: 1.25, endMeasure: 9.5},
    {startMeasure: -5, endMeasure: 1e9},
    {startMeasure: 4, endMeasure: 4},
    {startMeasure: 6, endMeasure: 2},
    {startMeasure: 0, endMeasure: 0.5}
  ];
  for (const range of ranges) {
    const actual = await fixture.run(getGPULineChunkParameterValues(range));
    const expected = chunkPaths(paths, {mode: 'substring', ...range, pathCapacity: pathCount});
    expectLinePathParity(actual, expected, capacity, 2e-5 * 60, 2e-5);
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineChunk chunks spherical paths across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createFlatPaths([
    [
      [170, 10],
      [-175, 20],
      [-160, 15]
    ],
    [
      [2.35, 48.85],
      [2.36, 48.86],
      [2.38, 48.855]
    ]
  ]);
  const capacity = 3000;
  const pathCapacity = 400;
  const fixture = createChunkFixture(device, paths, {
    mode: 'chunk',
    capacity,
    pathCapacity,
    spherical: true
  });
  for (const chunkLength of [250000, 25000]) {
    const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkPaths(paths, {
      mode: 'chunk',
      chunkLength,
      pathCapacity,
      spherical: true
    });
    expect(actual.overflow).toBe(0);
    expectLinePathParity(actual, expected, capacity, 5, 2e-5, getSphericalErrorMeters);
  }
  fixture.destroy();
});

/**
 * Axis-aligned staircase paths with integer step lengths: every row measure is an exact integer in
 * f32 whatever the summation order, so chunk boundaries cannot flip on rounding.
 */
function createWalkPaths(seed: number, lengths: number[]): FlatPaths {
  const random = createRandom(seed);
  return createFlatPaths(
    lengths.map(length => {
      let x = Math.floor(random() * 50);
      let y = Math.floor(random() * 50);
      return Array.from({length}, (_, row) => {
        if (row > 0) {
          const step = (random() < 0.5 ? -1 : 1) * (1 + Math.floor(random() * 4));
          if (random() < 0.5) {
            x += step;
          } else {
            y += step;
          }
        }
        return [x, y];
      });
    })
  );
}

it('GPULineChunk handles long paths whose prefix spans many scan tiles and pieces spanning many rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Path lengths around the 64-row tile size (63, 64, 65, 129; every path is longer than the substring start)
  // and one 2600-row path, so the
  // cooperative prefix, the tile carry and the per-vertex emit all see partial and full tiles.
  const paths = createWalkPaths(13, [30, 63, 2600, 64, 65, 129, 40]);
  const pathCount = paths.pathOffsets.length - 1;
  const capacity = 12000;
  const chunked = createChunkFixture(device, paths, {mode: 'chunk', capacity, pathCapacity: 800});
  for (const chunkLength of [900, 13, 13.5, 1e6]) {
    const actual = await chunked.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkPaths(paths, {mode: 'chunk', chunkLength, pathCapacity: 800});
    expectLinePathParity(actual, expected, capacity, 2e-5 * 600, 2e-5);
  }
  chunked.destroy();
  const substring = createChunkFixture(device, paths, {
    mode: 'substring',
    capacity,
    pathCapacity: pathCount
  });
  // One piece per path that copies almost the whole 2600-row path.
  for (const range of [
    {startMeasure: 7.5, endMeasure: 1e9},
    {startMeasure: 12.5, endMeasure: 4000.25}
  ]) {
    const actual = await substring.run(getGPULineChunkParameterValues(range));
    const expected = chunkPaths(paths, {mode: 'substring', ...range, pathCapacity: pathCount});
    expectLinePathParity(actual, expected, capacity, 2e-5 * 600, 2e-5);
  }
  substring.destroy();
});
