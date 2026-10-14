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
import {createFlatLines, createRandom, type FlatLines} from './line-segmentize-oracle';
import {smoothLines} from './line-smooth-oracle';
import {createLineLineFixture, expectLineLineParity} from './line-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function createSmoothFixture(
  device: Device,
  lines: FlatLines,
  options: {capacity: number; iterationCount: number; closed: boolean}
) {
  return createLineLineFixture(device, {
    inputs: {
      positions: {values: lines.positions, format: 'float32x2'},
      lineOffsets: {values: lines.lineOffsets, format: 'uint32'}
    },
    capacity: options.capacity,
    lineCapacity: lines.lineOffsets.length - 1,
    withVertexColumns: false,
    createContributor: (inputs, parameters, output) =>
      new GPULineSmooth({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
        iterationCount: options.iterationCount,
        closed: options.closed,
        parameters,
        output
      })
  });
}

function createRandomLines(seed: number, closed: boolean): FlatLines {
  const random = createRandom(seed);
  const lines = Array.from({length: 150}, (_, lineIndex) => {
    const length =
      lineIndex % 23 === 2
        ? 0
        : lineIndex % 19 === 3
          ? 1
          : lineIndex % 17 === 4
            ? 2
            : 3 + Math.floor(random() * 7);
    const vertices = Array.from({length}, () => [random() * 100, random() * 100]);
    // Some closed rings repeat their first vertex.
    if (closed && length >= 3 && lineIndex % 2 === 0) {
      vertices.push(vertices[0]);
    }
    return vertices;
  });
  return createFlatLines(lines);
}

for (const closed of [false, true]) {
  it(`GPULineSmooth matches the Chaikin oracle for ${closed ? 'closed rings' : 'open lines'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lines = createRandomLines(closed ? 2 : 1, closed);
    for (const iterationCount of [1, 2, 3]) {
      const capacity = (lines.positions.length / 2) * 2 ** iterationCount + 200;
      const fixture = createSmoothFixture(device, lines, {capacity, iterationCount, closed});
      for (const ratio of [0.25, 0.1, 0.5]) {
        const actual = await fixture.run(getGPULineSmoothParameterValues({ratio}));
        const expected = smoothLines(lines, {iterationCount, ratio, closed});
        expectLineLineParity(actual, expected, capacity, 1e-4, 0);
        expect(actual.overflow).toBe(0);
      }
      expect(fixture.getCompileCount()).toBe(0);
      fixture.destroy();
    }
  });
}

it('GPULineSmooth truncates at capacity with valid offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomLines(4, false);
  const capacity = 500;
  const iterationCount = 3;
  const fixture = createSmoothFixture(device, lines, {capacity, iterationCount, closed: false});
  const actual = await fixture.run(getGPULineSmoothParameterValues());
  const expected = smoothLines(lines, {iterationCount, ratio: 0.25, closed: false});
  expect(expected.positions.length).toBeGreaterThan(capacity);
  expectLineLineParity(actual, expected, capacity, 1e-4, 0);
  expect(actual.overflow).toBe(1);
  fixture.destroy();
});

it('GPULineSmooth keeps endpoints and doubles open lines', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [0, 0],
      [4, 0],
      [4, 4]
    ],
    [[1, 1]]
  ]);
  const fixture = createSmoothFixture(device, lines, {
    capacity: 16,
    iterationCount: 1,
    closed: false
  });
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
  expect(actual.lineOffsets).toEqual([0, 6, 7]);
  fixture.destroy();
});
