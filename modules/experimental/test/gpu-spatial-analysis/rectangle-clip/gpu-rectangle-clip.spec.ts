// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPURectangleClipParameterValues,
  GPURectangleClip
} from '../../../src/gpu-spatial-analysis/rectangle-clip';
import {
  createGeometryFixture,
  createRandom,
  expectClose
} from '../outline-geometry/geometry-fixture';
import {
  clipLines,
  clipRing,
  getRingArea,
  type Point,
  type Rectangle
} from './rectangle-clip-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

function flatten(lines: Point[][]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const line of lines) {
    for (const point of line) {
      positions.push(point[0], point[1]);
    }
    offsets.push(positions.length / 2);
  }
  return {
    positions: new Float32Array(positions),
    offsets: new Uint32Array(offsets),
    // One polygon per feature with a single ring each.
    identity: Uint32Array.from({length: offsets.length}, (_, index) => index)
  };
}

function round(lines: Point[][]): Point[][] {
  return lines.map(line => line.map(([x, y]) => [Math.fround(x), Math.fround(y)] as Point));
}

function createFixture(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  lines: Point[][],
  geometryType: 'lines' | 'polygons',
  capacity: number,
  lineCapacity: number
) {
  const flat = flatten(lines);
  return createGeometryFixture(device, {
    inputs: {
      positions: {values: flat.positions, format: 'float32x2'},
      lineOffsets: {values: flat.offsets, format: 'uint32'},
      ...(geometryType === 'polygons'
        ? {
            featureOffsets: {values: flat.identity, format: 'uint32' as const},
            polygonOffsets: {values: flat.identity, format: 'uint32' as const}
          }
        : {})
    },
    outputs: {
      positions: {format: 'float32x2', length: capacity},
      lineOffsets: {format: 'uint32', length: lineCapacity + 1},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      requiredCount: {format: 'uint32', length: 1},
      lineCount: {format: 'uint32', length: 1},
      ...(geometryType === 'lines'
        ? {sourceLineIndices: {format: 'uint32' as const, length: lineCapacity}}
        : {})
    },
    create: ({inputs, outputs, parameters}) =>
      new GPURectangleClip({
        positions: inputs['positions'] as never,
        ...(geometryType === 'polygons'
          ? {
              geometryType,
              featureOffsets: inputs['featureOffsets'] as never,
              polygonOffsets: inputs['polygonOffsets'] as never,
              ringOffsets: inputs['lineOffsets'] as never
            }
          : {geometryType, lineOffsets: inputs['lineOffsets'] as never}),
        parameters,
        output: {
          positions: outputs['positions'] as never,
          lineOffsets: outputs['lineOffsets'] as never,
          status: {
            count: outputs['count'] as never,
            overflow: outputs['overflow'] as never,
            requiredCount: outputs['requiredCount'] as never,
            lineCount: outputs['lineCount'] as never
          },
          sourceLineIndices: outputs['sourceLineIndices'] as never
        }
      })
  });
}

function readLines(result: Record<string, number[]>, lineCount: number): Point[][] {
  return Array.from({length: lineCount}, (_, line) =>
    Array.from(
      {length: result['lineOffsets'][line + 1] - result['lineOffsets'][line]},
      (__, vertex) => {
        const row = result['lineOffsets'][line] + vertex;
        return [result['positions'][2 * row], result['positions'][2 * row + 1]] as Point;
      }
    )
  );
}

function expectLinesClose(actual: Point[][], expected: Point[][], absolute: number) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((line, lineIndex) => {
    expect(line.length, `line ${lineIndex}`).toBe(expected[lineIndex].length);
    line.forEach((point, vertex) => {
      expectClose(point[0], expected[lineIndex][vertex][0], 1e-6, absolute);
      expectClose(point[1], expected[lineIndex][vertex][1], 1e-6, absolute);
    });
  });
}

it('GPURectangleClip clips lines like the f64 oracle and matches Shapely clip_by_rect', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Shapely 2.1.2 clip_by_rect(0, 0, 10, 10): lengths 16.2777, 12.1655 and 0 (total 28.4432).
  const fixed: Point[][] = [
    [
      [-5, 5],
      [5, 5],
      [15, 5],
      [15, 15],
      [5, 12],
      [-2, 2]
    ],
    [
      [2, 2],
      [8, 3],
      [9, 9]
    ],
    [
      [-3, -3],
      [-1, -1]
    ]
  ];
  const random = createRandom(9);
  const randomLines: Point[][] = Array.from({length: 150}, () => {
    const count = 2 + Math.floor(random() * 6);
    return Array.from({length: count}, () => [random() * 24 - 7, random() * 24 - 7] as Point);
  });
  for (const [lines, label] of [
    [fixed, 'fixed'],
    [randomLines, 'random']
  ] as const) {
    const capacity = 2048;
    const lineCapacity = 512;
    const fixture = createFixture(device, [...lines], 'lines', capacity, lineCapacity);
    for (const rect of [
      {minX: 0, minY: 0, maxX: 10, maxY: 10},
      {minX: 3.5, minY: -2, maxX: 8, maxY: 6}
    ] as Rectangle[]) {
      const result = await fixture.run(getGPURectangleClipParameterValues(rect));
      const expected = clipLines(round([...lines]), rect);
      expect(result['overflow'][0]).toBe(0);
      expect(result['lineCount'][0]).toBe(expected.lines.length);
      expect(result['count'][0]).toBe(expected.lines.flat().length);
      expect(result['requiredCount'][0]).toBe(expected.lines.flat().length);
      expect(result['count'][0]).toBeGreaterThan(0);
      expectLinesClose(readLines(result, result['lineCount'][0]), expected.lines, 1e-4);
      expect(result['sourceLineIndices'].slice(0, expected.lines.length)).toEqual(
        expected.sourceLineIndices
      );
      expect(result['lineOffsets'][result['lineCount'][0]]).toBe(result['count'][0]);
      if (label === 'fixed' && rect.maxX === 10) {
        const length = readLines(result, result['lineCount'][0]).reduce(
          (sum, line) =>
            sum +
            line
              .slice(1)
              .reduce(
                (lineSum, point, index) =>
                  lineSum + Math.hypot(point[0] - line[index][0], point[1] - line[index][1]),
                0
              ),
          0
        );
        expectClose(length, 28.443182234402343, 1e-5, 1e-4, 'Shapely length');
      }
    }
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  }
});

it('GPURectangleClip reports overflow when line or vertex capacity is exceeded', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const zigzag: Point[] = Array.from({length: 40}, (_, index) => [
    index,
    index % 2 === 0 ? -1 : 11
  ]);
  const fixture = createFixture(device, [zigzag], 'lines', 16, 4);
  const result = await fixture.run(
    getGPURectangleClipParameterValues({minX: 0, minY: 0, maxX: 40, maxY: 10})
  );
  expect(result['overflow'][0]).toBe(1);
  expect(result['count'][0]).toBeLessThanOrEqual(16);
  expect(result['requiredCount'][0]).toBeGreaterThan(16);
  expect(result['lineOffsets'][4]).toBeLessThanOrEqual(16);
  fixture.destroy();
});

it('GPURectangleClip clips polygon rings like f64 Sutherland-Hodgman and matches Shapely areas', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Shapely 2.1.2 clip_by_rect(0, 0, 10, 10) areas: 100, 40 (C shape), 100, 7.5.
  const rings: Point[][] = [
    [
      [-5, -5],
      [15, -5],
      [15, 15],
      [-5, 15]
    ],
    [
      [2, 2],
      [20, 2],
      [20, 4],
      [4, 4],
      [4, 8],
      [20, 8],
      [20, 10],
      [2, 10]
    ],
    [
      [-10, 5],
      [5, -10],
      [20, 5],
      [5, 20]
    ],
    [
      [1, 1],
      [4, 1],
      [2.5, 6]
    ],
    // Fully outside: becomes an empty ring.
    [
      [30, 30],
      [31, 30],
      [30, 31]
    ]
  ];
  const shapelyAreas = [100, 40, 100, 7.5];
  const capacity = 256;
  const fixture = createFixture(device, rings, 'polygons', capacity, rings.length);
  const rect = {minX: 0, minY: 0, maxX: 10, maxY: 10};
  const result = await fixture.run(getGPURectangleClipParameterValues(rect));
  expect(result['overflow'][0]).toBe(0);
  expect(result['lineCount'][0]).toBe(rings.length);
  const actual = readLines(result, rings.length);
  const expected = round(rings).map(ring => clipRing(ring, rect));
  expectLinesClose(actual, expected, 1e-4);
  expect(actual[4].length).toBe(0);
  shapelyAreas.forEach((area, index) => {
    expectClose(getRingArea(actual[index]), area, 1e-5, 1e-3, `ring ${index} area`);
  });
  // Moving the rectangle per frame changes the result without recompiling.
  const moved = await fixture.run(
    getGPURectangleClipParameterValues({minX: 6, minY: 6, maxX: 12, maxY: 30})
  );
  const movedExpected = round(rings).map(ring =>
    clipRing(ring, {minX: 6, minY: 6, maxX: 12, maxY: 30})
  );
  expectLinesClose(readLines(moved, rings.length), movedExpected, 1e-4);
  expect(moved['count'][0]).toBe(movedExpected.flat().length);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
  const small = createFixture(device, rings, 'polygons', 6, rings.length);
  const clipped = await small.run(getGPURectangleClipParameterValues(rect));
  expect(clipped['overflow'][0]).toBe(1);
  expect(clipped['count'][0]).toBeLessThanOrEqual(6);
  small.destroy();
});
