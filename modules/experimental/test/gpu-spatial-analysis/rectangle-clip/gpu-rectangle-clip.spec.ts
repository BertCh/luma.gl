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

function flatten(paths: Point[][]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const path of paths) {
    for (const point of path) {
      positions.push(point[0], point[1]);
    }
    offsets.push(positions.length / 2);
  }
  return {positions: new Float32Array(positions), offsets: new Uint32Array(offsets)};
}

function round(paths: Point[][]): Point[][] {
  return paths.map(path => path.map(([x, y]) => [Math.fround(x), Math.fround(y)] as Point));
}

function createFixture(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  paths: Point[][],
  geometryType: 'lines' | 'polygons',
  capacity: number,
  pathCapacity: number
) {
  const flat = flatten(paths);
  return createGeometryFixture(device, {
    inputs: {
      positions: {values: flat.positions, format: 'float32x2'},
      pathOffsets: {values: flat.offsets, format: 'uint32'}
    },
    outputs: {
      positions: {format: 'float32x2', length: capacity},
      pathOffsets: {format: 'uint32', length: pathCapacity + 1},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      totalCount: {format: 'uint32', length: 1},
      pathCount: {format: 'uint32', length: 1},
      ...(geometryType === 'lines'
        ? {sourcePaths: {format: 'uint32' as const, length: pathCapacity}}
        : {})
    },
    create: ({inputs, outputs, parameters}) =>
      new GPURectangleClip({
        positions: inputs['positions'] as never,
        pathOffsets: inputs['pathOffsets'] as never,
        geometryType,
        parameters,
        output: {
          positions: outputs['positions'] as never,
          pathOffsets: outputs['pathOffsets'] as never,
          count: outputs['count'] as never,
          overflow: outputs['overflow'] as never,
          totalCount: outputs['totalCount'] as never,
          pathCount: outputs['pathCount'] as never,
          sourcePaths: outputs['sourcePaths'] as never
        }
      })
  });
}

function readPaths(result: Record<string, number[]>, pathCount: number): Point[][] {
  return Array.from({length: pathCount}, (_, path) =>
    Array.from(
      {length: result['pathOffsets'][path + 1] - result['pathOffsets'][path]},
      (__, vertex) => {
        const row = result['pathOffsets'][path] + vertex;
        return [result['positions'][2 * row], result['positions'][2 * row + 1]] as Point;
      }
    )
  );
}

function expectPathsClose(actual: Point[][], expected: Point[][], absolute: number) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((path, pathIndex) => {
    expect(path.length, `path ${pathIndex}`).toBe(expected[pathIndex].length);
    path.forEach((point, vertex) => {
      expectClose(point[0], expected[pathIndex][vertex][0], 1e-6, absolute);
      expectClose(point[1], expected[pathIndex][vertex][1], 1e-6, absolute);
    });
  });
}

it('GPURectangleClip clips lines like the f64 oracle and matches Shapely clip_by_rect', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
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
  const randomPaths: Point[][] = Array.from({length: 150}, () => {
    const count = 2 + Math.floor(random() * 6);
    return Array.from({length: count}, () => [random() * 24 - 7, random() * 24 - 7] as Point);
  });
  for (const [paths, label] of [
    [fixed, 'fixed'],
    [randomPaths, 'random']
  ] as const) {
    const capacity = 2048;
    const pathCapacity = 512;
    const fixture = createFixture(device, [...paths], 'lines', capacity, pathCapacity);
    for (const rect of [
      {minX: 0, minY: 0, maxX: 10, maxY: 10},
      {minX: 3.5, minY: -2, maxX: 8, maxY: 6}
    ] as Rectangle[]) {
      const result = await fixture.run(getGPURectangleClipParameterValues(rect));
      const expected = clipLines(round([...paths]), rect);
      expect(result['overflow'][0]).toBe(0);
      expect(result['pathCount'][0]).toBe(expected.paths.length);
      expect(result['count'][0]).toBe(expected.paths.flat().length);
      expect(result['totalCount'][0]).toBe(expected.paths.flat().length);
      expect(result['count'][0]).toBeGreaterThan(0);
      expectPathsClose(readPaths(result, result['pathCount'][0]), expected.paths, 1e-4);
      expect(result['sourcePaths'].slice(0, expected.paths.length)).toEqual(expected.sourcePaths);
      expect(result['pathOffsets'][result['pathCount'][0]]).toBe(result['count'][0]);
      if (label === 'fixed' && rect.maxX === 10) {
        const length = readPaths(result, result['pathCount'][0]).reduce(
          (sum, path) =>
            sum +
            path
              .slice(1)
              .reduce(
                (pathSum, point, index) =>
                  pathSum + Math.hypot(point[0] - path[index][0], point[1] - path[index][1]),
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

it('GPURectangleClip reports overflow when path or vertex capacity is exceeded', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
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
  expect(result['totalCount'][0]).toBeGreaterThan(16);
  expect(result['pathOffsets'][4]).toBeLessThanOrEqual(16);
  fixture.destroy();
});

it('GPURectangleClip clips polygon rings like f64 Sutherland-Hodgman and matches Shapely areas', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
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
  expect(result['pathCount'][0]).toBe(rings.length);
  const actual = readPaths(result, rings.length);
  const expected = round(rings).map(ring => clipRing(ring, rect));
  expectPathsClose(actual, expected, 1e-4);
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
  expectPathsClose(readPaths(moved, rings.length), movedExpected, 1e-4);
  expect(moved['count'][0]).toBe(movedExpected.flat().length);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
  const small = createFixture(device, rings, 'polygons', 6, rings.length);
  const clipped = await small.run(getGPURectangleClipParameterValues(rect));
  expect(clipped['overflow'][0]).toBe(1);
  expect(clipped['count'][0]).toBeLessThanOrEqual(6);
  small.destroy();
});
