// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {runLineClip, type Point} from './line-clip-harness';
import {SHAPELY_LINE_CLIP_FIXTURE} from './shapely-line-clip-fixture';

const SQUARE = [
  [
    [
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
      ]
    ]
  ]
];

function getLength(piece: readonly (readonly number[])[]): number {
  let total = 0;
  for (let index = 1; index < piece.length; index++) {
    total += Math.hypot(
      piece[index][0] - piece[index - 1][0],
      piece[index][1] - piece[index - 1][1]
    );
  }
  return total;
}

it('GPULineClipByPolygon#clips a crossing line and keeps the outside parts for difference', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const line = [
    [-5, 5],
    [15, 5]
  ];
  const inside = await runLineClip(device, [line], SQUARE);
  expect(inside.pieces).toEqual([
    [
      [0, 5],
      [10, 5]
    ]
  ]);
  expect(inside.lineIds).toEqual([0]);
  expect(inside.overflow).toBe(0);
  expect(inside.uncertainCount).toBe(0);
  const outside = await runLineClip(device, [line], SQUARE, {mode: 'outside'});
  expect(outside.pieces).toEqual([
    [
      [-5, 5],
      [0, 5]
    ],
    [
      [10, 5],
      [15, 5]
    ]
  ]);
  expect(outside.lineIds).toEqual([0, 0]);
});

it('GPULineClipByPolygon#counts boundary pieces as inside and merges kept pieces', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runLineClip(
    device,
    [
      // Along the bottom edge and out the far side: the boundary run joins the inside run.
      [
        [-3, 0],
        [4, 0],
        [8, 4]
      ],
      // Never leaves the polygon: stays one piece with its own vertices.
      [
        [1, 1],
        [5, 2],
        [9, 9]
      ],
      // Touches the polygon at a corner only.
      [
        [-2, -2],
        [0, 0],
        [-2, 2]
      ]
    ],
    SQUARE
  );
  expect(result.lineIds).toEqual([0, 1]);
  expect(result.pieces).toEqual([
    [
      [0, 0],
      [4, 0],
      [8, 4]
    ],
    [
      [1, 1],
      [5, 2],
      [9, 9]
    ]
  ]);
  const outside = await runLineClip(
    device,
    [
      [
        [-3, 0],
        [4, 0],
        [8, 4]
      ]
    ],
    SQUARE,
    {mode: 'outside'}
  );
  // The boundary run belongs to the polygon, so only the part left of the corner remains.
  expect(outside.pieces).toEqual([
    [
      [-3, 0],
      [0, 0]
    ]
  ]);
});

it('GPULineClipByPolygon#matches Shapely intersection and difference on random and edge lines', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {lines, features, expected, simple, selfOverlap} = SHAPELY_LINE_CLIP_FIXTURE;
  for (const mode of ['inside', 'outside'] as const) {
    const start = performance.now();
    const result = await runLineClip(device, lines as never, features as never, {
      mode,
      pieceCapacity: 512,
      vertexCapacity: 4096
    });
    const elapsed = performance.now() - start;
    expect(result.overflow).toBe(0);
    expect(result.uncertainCount).toBe(0);
    let totalLength = 0;
    let pieceTotal = 0;
    for (let line = 0; line < lines.length; line++) {
      const reference = expected[mode][line];
      const mine: Point[][] = result.pieces.filter((_, piece) => result.lineIds[piece] === line);
      const length = mine.reduce((sum, piece) => sum + getLength(piece), 0);
      if (!selfOverlap[line]) {
        expect(Math.abs(length - reference.length), `${mode} line ${line} length`).toBeLessThan(
          2e-3
        );
      }
      if (!simple[line] || selfOverlap[line]) {
        // Shapely nodes the result where the line crosses itself; the contributor keeps the line whole.
        totalLength += length;
        pieceTotal += mine.length;
        continue;
      }
      expect(mine.length, `${mode} line ${line} piece count`).toBe(reference.pieces.length);
      const unused = new Set(reference.pieces.map((_, index) => index));
      for (const piece of mine) {
        const ends = [piece[0], piece[piece.length - 1]];
        const near = (a: readonly number[], b: readonly number[]) =>
          Math.abs(a[0] - b[0]) < 1e-3 && Math.abs(a[1] - b[1]) < 1e-3;
        const match = [...unused].find(index => {
          const other = reference.pieces[index];
          const first = other[0];
          const last = other[other.length - 1];
          return (
            (near(ends[0], first) && near(ends[1], last)) ||
            (near(ends[0], last) && near(ends[1], first))
          );
        });
        expect(match, `${mode} line ${line} piece ends`).toBeDefined();
        unused.delete(match as number);
      }
      totalLength += length;
      pieceTotal += mine.length;
    }
    console.log(
      `GPULineClipByPolygon ${mode}: ${lines.length} lines -> ${pieceTotal} pieces, ` +
        `length ${totalLength.toFixed(3)}, ${elapsed.toFixed(0)} ms`
    );
  }
});

it('GPULineClipByPolygon#reports overflow and keeps a consistent prefix', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const lines = [
    [
      [-5, 2],
      [15, 2]
    ],
    [
      [-5, 5],
      [15, 5]
    ],
    [
      [-5, 8],
      [15, 8]
    ]
  ];
  const pieces = await runLineClip(device, lines, SQUARE, {pieceCapacity: 2});
  expect(pieces.count).toBe(2);
  expect(pieces.requiredCount).toBe(3);
  expect(pieces.overflow).toBe(1);
  expect(pieces.lineIds).toEqual([0, 1]);
  const vertices = await runLineClip(device, lines, SQUARE, {vertexCapacity: 5});
  expect(vertices.count).toBe(2);
  expect(vertices.vertexCount).toBe(4);
  expect(vertices.overflow).toBe(1);
  const pairs = await runLineClip(device, lines, SQUARE, {intersectionCapacity: 2});
  expect(pairs.overflow).toBe(1);
});

it('GPULineClipByPolygon#orders a long skewed crossing stream with cooperative radix passes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const toothCount = 257;
  const ring: Point[] = [
    [0, 1],
    [toothCount * 2, 1],
    [toothCount * 2, -1]
  ];
  for (let tooth = toothCount * 2 - 1; tooth >= 0; tooth--) {
    ring.push([tooth, tooth % 2 === 0 ? 0.5 : -1]);
  }
  ring.push([0, -1]);
  const result = await runLineClip(
    device,
    [
      [
        [-1, 0],
        [toothCount * 2 + 1, 0]
      ]
    ],
    [[[ring]]],
    {
      intersectionCapacity: 2048,
      candidateCapacity: 2048,
      pieceCapacity: toothCount + 4,
      vertexCapacity: toothCount * 2 + 8
    }
  );
  expect(result.overflow).toBe(0);
  expect(result.pieces.length).toBeGreaterThan(256);
  for (let piece = 1; piece < result.pieces.length; piece++) {
    expect(result.pieces[piece][0][0]).toBeGreaterThanOrEqual(
      result.pieces[piece - 1][result.pieces[piece - 1].length - 1][0]
    );
  }
});

it('GPULineClipByPolygon#compiles within the default storage buffer limit', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) return;
  const result = await runLineClip(
    device,
    [
      [
        [-5, 5],
        [15, 5]
      ]
    ],
    SQUARE
  );
  expect(result.pieces).toEqual([
    [
      [0, 5],
      [10, 5]
    ]
  ]);
});
