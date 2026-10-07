// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {runLineSplit, type Point} from './line-split-harness';
import {SHAPELY_LINE_SPLIT_FIXTURE} from './shapely-line-split-fixture';

it('GPULineSplit#splits an X crossing at the crossing point', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runLineSplit(device, [
    [
      [0, 0],
      [2, 2]
    ],
    [
      [0, 2],
      [2, 0]
    ]
  ]);
  expect(result.lineIds).toEqual([0, 0, 1, 1]);
  expect(result.pieces).toEqual([
    [
      [0, 0],
      [1, 1]
    ],
    [
      [1, 1],
      [2, 2]
    ],
    [
      [0, 2],
      [1, 1]
    ],
    [
      [1, 1],
      [2, 0]
    ]
  ]);
  expect(result.overflow).toBe(0);
  expect(result.totalCount).toBe(4);
  expect(result.totalVertexCount).toBe(8);
});

it('GPULineSplit#splits at a shared vertex without duplicating it and ignores line ends', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runLineSplit(device, [
    [
      [0, 0],
      [1, 0],
      [2, 0]
    ],
    [
      [1, -1],
      [1, 1]
    ],
    // Ends on line 0 (T junction): splits line 0 (already split at that vertex), not itself.
    [
      [1, 3],
      [1, 1.5]
    ]
  ]);
  expect(result.lineIds).toEqual([0, 0, 1, 1, 2]);
  expect(result.pieces[0]).toEqual([
    [0, 0],
    [1, 0]
  ]);
  expect(result.pieces[1]).toEqual([
    [1, 0],
    [2, 0]
  ]);
  expect(result.pieces[4]).toEqual([
    [1, 3],
    [1, 1.5]
  ]);
});

it('GPULineSplit#splits a line at its own crossing and keeps unsplit lines whole', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runLineSplit(device, [
    [
      [0, 0],
      [2, 2],
      [2, 0],
      [0, 2]
    ],
    [
      [10, 10],
      [11, 10],
      [12, 11]
    ],
    [[5, 5]]
  ]);
  expect(result.lineIds).toEqual([0, 0, 0, 1]);
  expect(result.pieces).toEqual([
    [
      [0, 0],
      [1, 1]
    ],
    [
      [1, 1],
      [2, 2],
      [2, 0],
      [1, 1]
    ],
    [
      [1, 1],
      [0, 2]
    ],
    [
      [10, 10],
      [11, 10],
      [12, 11]
    ]
  ]);
});

it('GPULineSplit#orders several splits on one segment and splits at both ends of an overlap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runLineSplit(device, [
    [
      [0, 0],
      [10, 0]
    ],
    // Crossed out of order, so the sort has work to do.
    [
      [7, -1],
      [7, 1]
    ],
    [
      [3, -1],
      [3, 1]
    ],
    [
      [5, -1],
      [5, 1]
    ],
    // Overlaps line 0 on [2, 4] (two shared-span ends, one of them coinciding with line 2's cross).
    [
      [2, 0],
      [4, 0]
    ]
  ]);
  const first = result.pieces.slice(0, result.lineIds.lastIndexOf(0) + 1);
  expect(first.map(piece => piece.map(point => point[0]))).toEqual([
    [0, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 7],
    [7, 10]
  ]);
});

it('GPULineSplit#reports overflow and keeps a consistent prefix', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const lines: Point[][] = [
    [
      [0, 0],
      [2, 2]
    ],
    [
      [0, 2],
      [2, 0]
    ]
  ];
  const limitedPieces = await runLineSplit(device, lines, {pieceCapacity: 3});
  expect(limitedPieces.count).toBe(3);
  expect(limitedPieces.overflow).toBe(1);
  expect(limitedPieces.totalCount).toBe(4);
  const limitedVertices = await runLineSplit(device, lines, {vertexCapacity: 5});
  expect(limitedVertices.count).toBe(2);
  expect(limitedVertices.vertexCount).toBe(4);
  expect(limitedVertices.overflow).toBe(1);
  const limitedPairs = await runLineSplit(device, lines, {intersectionCapacity: 1});
  expect(limitedPairs.overflow).toBe(0);
  expect(limitedPairs.count).toBe(4);
  const manyLines: Point[][] = [];
  for (let line = 0; line < 6; line++) {
    manyLines.push([
      [line, -10],
      [line, 10]
    ]);
    manyLines.push([
      [-10, line + 0.5],
      [10, line + 0.5]
    ]);
  }
  expect((await runLineSplit(device, manyLines, {intersectionCapacity: 4})).overflow).toBe(1);
});

it('GPULineSplit#matches Shapely unary_union noding on random lines', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runLineSplit(device, SHAPELY_LINE_SPLIT_FIXTURE.lines as never);
  expect(result.overflow).toBe(0);
  const expected = SHAPELY_LINE_SPLIT_FIXTURE.pieces;
  expect(result.count).toBe(expected.length);
  const length = (piece: Point[]) => {
    let total = 0;
    for (let index = 1; index < piece.length; index++) {
      total += Math.hypot(
        piece[index][0] - piece[index - 1][0],
        piece[index][1] - piece[index - 1][1]
      );
    }
    return total;
  };
  const unused = new Set(expected.map((_, index) => index));
  for (const piece of result.pieces) {
    const start = piece[0];
    const end = piece[piece.length - 1];
    const match = [...unused].find(index => {
      const reference = expected[index];
      const near = (a: Point, b: readonly number[]) =>
        Math.abs(a[0] - b[0]) < 1e-3 && Math.abs(a[1] - b[1]) < 1e-3;
      return (
        reference.vertices === piece.length &&
        Math.abs(reference.length - length(piece)) < 1e-3 &&
        ((near(start, reference.start) && near(end, reference.end)) ||
          (near(start, reference.end) && near(end, reference.start)))
      );
    });
    expect(match).toBeDefined();
    unused.delete(match as number);
  }
  expect(unused.size).toBe(0);
});

it('GPULineSplit#orders many crossings of one segment (heap-sorted slot) and keeps duplicates once', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // 90 vertical lines cross one 2-vertex segment: far more events on a single slot than the
  // insertion-sort limit, listed in a scrambled order. Two of them coincide at x = 45.
  const crossingCount = 90;
  const xs: number[] = [];
  for (let index = 0; index < crossingCount; index++) {
    xs.push(1 + ((index * 37) % crossingCount));
  }
  xs.push(45);
  const lines: [number, number][][] = [
    [
      [0, 0],
      [100, 0]
    ]
  ];
  for (const x of xs) {
    lines.push([
      [x, -1],
      [x, 1]
    ]);
  }
  const result = await runLineSplit(device, lines, {
    intersectionCapacity: 1024,
    pieceCapacity: 1024,
    vertexCapacity: 8192
  });
  const first = result.pieces.slice(0, result.lineIds.lastIndexOf(0) + 1);
  const unique = [...new Set(xs)].sort((a, b) => a - b);
  const expectedBreaks = [0, ...unique, 100];
  expect(first.map(piece => [piece[0][0], piece[piece.length - 1][0]])).toEqual(
    expectedBreaks.slice(0, -1).map((x, index) => [x, expectedBreaks[index + 1]])
  );
  expect(result.overflow).toBe(0);
});
