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
  type GPULineChunkSettings
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {createFlatLines, createRandom, type FlatLines} from './line-segmentize-oracle';
import {chunkLines} from './line-chunk-oracle';
import {
  createLineLineFixture,
  expectLineLineParity,
  getSphericalErrorMeters
} from './line-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function createChunkFixture(
  device: Device,
  lines: FlatLines,
  options: {
    mode: 'chunk' | 'substring';
    capacity: number;
    lineCapacity: number;
    spherical?: boolean;
  }
) {
  return createLineLineFixture(device, {
    inputs: {
      positions: {values: lines.positions, format: 'float32x2'},
      lineOffsets: {values: lines.lineOffsets, format: 'uint32'}
    },
    capacity: options.capacity,
    lineCapacity: options.lineCapacity,
    createContributor: (inputs, parameters, output) =>
      new GPULineChunk({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
        mode: options.mode,
        spatialContext: options.spherical
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          : undefined,
        parameters,
        output
      })
  });
}

function createRandomLines(seed: number): FlatLines {
  const random = createRandom(seed);
  return createFlatLines(
    Array.from({length: 80}, (_, lineIndex) => {
      if (lineIndex % 19 === 3) {
        return [];
      }
      if (lineIndex % 17 === 5) {
        return [[random() * 50, random() * 50]];
      }
      if (lineIndex % 23 === 7) {
        // Zero-length line with repeated vertices.
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

it('GPULineChunk splits lines into chunks like the f64 oracle, per-frame chunk length', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomLines(3);
  const capacity = 6000;
  const lineCapacity = 1500;
  const fixture = createChunkFixture(device, lines, {mode: 'chunk', capacity, lineCapacity});
  for (const chunkLength of [3.3, 7.77, 50, 0, 1.01]) {
    const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkLines(lines, {mode: 'chunk', chunkLength, lineCapacity});
    expect(actual.lineCount).toBe(Math.min(expected.pieceTotal, lineCapacity));
    expect(actual.sourceLineIndices).toEqual(expected.sourceLineIndices);
    expectLineLineParity(actual, expected, capacity, 2e-5 * 60, 2e-5);
    expect(actual.overflow).toBe(0);
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineChunk reports line and vertex capacity overflow with valid offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomLines(5);
  const lineCapacity = 40;
  const capacity = 150;
  const fixture = createChunkFixture(device, lines, {mode: 'chunk', capacity, lineCapacity});
  const chunkLength = 2.5;
  const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
  const expected = chunkLines(lines, {mode: 'chunk', chunkLength, lineCapacity});
  expect(expected.pieceTotal).toBeGreaterThan(lineCapacity);
  expect(actual.overflow).toBe(1);
  expect(actual.lineCount).toBe(lineCapacity);
  expectLineLineParity(actual, expected, capacity, 2e-3, 2e-5, undefined, 1);
  // A tiny chunk length is capped instead of wrapping counts.
  const tiny = await fixture.run(getGPULineChunkParameterValues({chunkLength: 1e-30}));
  expect(tiny.overflow).toBe(1);
  expect(tiny.lineCount).toBe(lineCapacity);
  fixture.destroy();
});

it('GPULineChunk extracts substrings by measure range', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomLines(7);
  const lineCount = lines.lineOffsets.length - 1;
  const capacity = 2000;
  const fixture = createChunkFixture(device, lines, {
    mode: 'substring',
    capacity,
    lineCapacity: lineCount
  });
  const ranges: GPULineChunkSettings[] = [
    {startMeasure: 1.25, endMeasure: 9.5},
    {startMeasure: -5, endMeasure: 1e9},
    {startMeasure: 4, endMeasure: 4},
    {startMeasure: 6, endMeasure: 2},
    {startMeasure: 0, endMeasure: 0.5}
  ];
  for (const range of ranges) {
    const actual = await fixture.run(getGPULineChunkParameterValues(range));
    const expected = chunkLines(lines, {mode: 'substring', ...range, lineCapacity: lineCount});
    expectLineLineParity(actual, expected, capacity, 2e-5 * 60, 2e-5);
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineChunk chunks spherical lines across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
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
  const lineCapacity = 400;
  const fixture = createChunkFixture(device, lines, {
    mode: 'chunk',
    capacity,
    lineCapacity,
    spherical: true
  });
  for (const chunkLength of [250000, 25000]) {
    const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkLines(lines, {
      mode: 'chunk',
      chunkLength,
      lineCapacity,
      spherical: true
    });
    expect(actual.overflow).toBe(0);
    expectLineLineParity(actual, expected, capacity, 5, 2e-5, getSphericalErrorMeters);
  }
  fixture.destroy();
});

/**
 * Axis-aligned staircase lines with integer step lengths: every row measure is an exact integer in
 * f32 whatever the summation order, so chunk boundaries cannot flip on rounding.
 */
function createWalkLines(seed: number, lengths: number[]): FlatLines {
  const random = createRandom(seed);
  return createFlatLines(
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

it('GPULineChunk handles long lines whose prefix spans many scan tiles and pieces spanning many rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Line lengths around the 64-row tile size (63, 64, 65, 129; every line is longer than the substring start)
  // and one 2600-row line, so the
  // cooperative prefix, the tile carry and the per-vertex emit all see partial and full tiles.
  const lines = createWalkLines(13, [30, 63, 2600, 64, 65, 129, 40]);
  const lineCount = lines.lineOffsets.length - 1;
  const capacity = 12000;
  const chunked = createChunkFixture(device, lines, {mode: 'chunk', capacity, lineCapacity: 800});
  for (const chunkLength of [900, 13, 13.5, 1e6]) {
    const actual = await chunked.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkLines(lines, {mode: 'chunk', chunkLength, lineCapacity: 800});
    expectLineLineParity(actual, expected, capacity, 2e-5 * 600, 2e-5);
  }
  chunked.destroy();
  const substring = createChunkFixture(device, lines, {
    mode: 'substring',
    capacity,
    lineCapacity: lineCount
  });
  // One piece per line that copies almost the whole 2600-row line.
  for (const range of [
    {startMeasure: 7.5, endMeasure: 1e9},
    {startMeasure: 12.5, endMeasure: 4000.25}
  ]) {
    const actual = await substring.run(getGPULineChunkParameterValues(range));
    const expected = chunkLines(lines, {mode: 'substring', ...range, lineCapacity: lineCount});
    expectLineLineParity(actual, expected, capacity, 2e-5 * 600, 2e-5);
  }
  substring.destroy();
});
