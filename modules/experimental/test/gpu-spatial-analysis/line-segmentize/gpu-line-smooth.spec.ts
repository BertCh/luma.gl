// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPULineSmoothParameterValues,
  GPULineSmooth
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-smooth';
import {createFlatPaths, createRandom, type FlatPaths} from './line-segmentize-oracle';
import {smoothPaths} from './line-smooth-oracle';
import {createLinePathFixture, expectLinePathParity} from './line-path-test-utils';

function createSmoothFixture(
  device: Device,
  paths: FlatPaths,
  options: {capacity: number; iterations: number; closed: boolean}
) {
  return createLinePathFixture(device, {
    inputs: {
      positions: {values: paths.positions, format: 'float32x2'},
      pathOffsets: {values: paths.pathOffsets, format: 'uint32'}
    },
    capacity: options.capacity,
    pathCapacity: paths.pathOffsets.length - 1,
    withVertexColumns: false,
    createContributor: (inputs, parameters, output) =>
      new GPULineSmooth({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        pathOffsets: inputs['pathOffsets'] as GraphDataView<'uint32'>,
        iterations: options.iterations,
        closed: options.closed,
        parameters,
        output
      })
  });
}

function createRandomPaths(seed: number, closed: boolean): FlatPaths {
  const random = createRandom(seed);
  const paths = Array.from({length: 150}, (_, pathIndex) => {
    const length =
      pathIndex % 23 === 2
        ? 0
        : pathIndex % 19 === 3
          ? 1
          : pathIndex % 17 === 4
            ? 2
            : 3 + Math.floor(random() * 7);
    const vertices = Array.from({length}, () => [random() * 100, random() * 100]);
    // Some closed rings repeat their first vertex.
    if (closed && length >= 3 && pathIndex % 2 === 0) {
      vertices.push(vertices[0]);
    }
    return vertices;
  });
  return createFlatPaths(paths);
}

for (const closed of [false, true]) {
  it(`GPULineSmooth matches the Chaikin oracle for ${closed ? 'closed rings' : 'open paths'}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const paths = createRandomPaths(closed ? 2 : 1, closed);
    for (const iterations of [1, 2, 3]) {
      const capacity = (paths.positions.length / 2) * 2 ** iterations + 200;
      const fixture = createSmoothFixture(device, paths, {capacity, iterations, closed});
      for (const ratio of [0.25, 0.1, 0.5]) {
        const actual = await fixture.run(getGPULineSmoothParameterValues({ratio}));
        const expected = smoothPaths(paths, {iterations, ratio, closed});
        expectLinePathParity(actual, expected, capacity, 1e-4, 0);
        expect(actual.overflow).toBe(0);
      }
      expect(fixture.getCompileCount()).toBe(0);
      fixture.destroy();
    }
  });
}

it('GPULineSmooth truncates at capacity with valid offsets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createRandomPaths(4, false);
  const capacity = 500;
  const iterations = 3;
  const fixture = createSmoothFixture(device, paths, {capacity, iterations, closed: false});
  const actual = await fixture.run(getGPULineSmoothParameterValues());
  const expected = smoothPaths(paths, {iterations, ratio: 0.25, closed: false});
  expect(expected.positions.length).toBeGreaterThan(capacity);
  expectLinePathParity(actual, expected, capacity, 1e-4, 0);
  expect(actual.overflow).toBe(1);
  fixture.destroy();
});

it('GPULineSmooth keeps endpoints and doubles open paths', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const paths = createFlatPaths([
    [
      [0, 0],
      [4, 0],
      [4, 4]
    ],
    [[1, 1]]
  ]);
  const fixture = createSmoothFixture(device, paths, {capacity: 16, iterations: 1, closed: false});
  const actual = await fixture.run(getGPULineSmoothParameterValues());
  expect(actual.positions).toEqual([
    [0, 0],
    [1, 0],
    [3, 0],
    [4, 1],
    [4, 3],
    [4, 4],
    [1, 1]
  ]);
  expect(actual.pathOffsets).toEqual([0, 6, 7]);
  fixture.destroy();
});
