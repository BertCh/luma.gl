// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {runSharedPaths, type Point} from './line-clip-harness';
import {SHAPELY_SHARED_PATHS_FIXTURE} from './shapely-shared-paths-fixture';

function getLength(run: readonly (readonly number[])[]): number {
  let total = 0;
  for (let index = 1; index < run.length; index++) {
    total += Math.hypot(run[index][0] - run[index - 1][0], run[index][1] - run[index - 1][1]);
  }
  return total;
}

it('GPUSharedPaths#reports forward and backward runs along the left line', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runSharedPaths(
    device,
    [
      [
        [0, 0],
        [2, 0],
        [4, 0]
      ],
      [
        [0, 3],
        [4, 3]
      ]
    ],
    [
      // Runs backward over [1, 3] of line 0, across its middle vertex.
      [
        [3, 0],
        [1, 0]
      ],
      // Forward over [1, 4] of line 1; the end beyond the left line is not shared.
      [
        [1, 3],
        [5, 3]
      ],
      // Crosses line 0 at a single point: no path.
      [
        [2, -1],
        [2, 1]
      ]
    ]
  );
  expect(result.count).toBe(2);
  expect(result.leftLineIds).toEqual([0, 1]);
  expect(result.rightLineIds).toEqual([0, 1]);
  expect(result.forward).toEqual([0, 1]);
  expect(result.runs).toEqual([
    [
      [1, 0],
      [2, 0],
      [3, 0]
    ],
    [
      [1, 3],
      [4, 3]
    ]
  ]);
  expect(result.overflow).toBe(0);
});

it('GPUSharedPaths#matches Shapely shared_paths on lattice walks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {left, right, shared} = SHAPELY_SHARED_PATHS_FIXTURE;
  const start = performance.now();
  const result = await runSharedPaths(device, left as never, right as never);
  const elapsed = performance.now() - start;
  expect(result.overflow).toBe(0);
  const used = new Set<number>();
  let forwardLength = 0;
  let backwardLength = 0;
  for (const reference of shared) {
    for (const [direction, expected] of [
      [1, reference.forward],
      [0, reference.backward]
    ] as const) {
      const mine: Point[][] = [];
      result.runs.forEach((run, index) => {
        if (
          result.leftLineIds[index] === reference.left &&
          result.rightLineIds[index] === reference.right &&
          result.forward[index] === direction
        ) {
          mine.push(run);
          used.add(index);
        }
      });
      const length = mine.reduce((sum, run) => sum + getLength(run), 0);
      expect(
        Math.abs(length - expected.length),
        `pair ${reference.left},${reference.right} direction ${direction} length`
      ).toBeLessThan(1e-4);
      expect(mine.length, `pair ${reference.left},${reference.right} run count`).toBe(
        expected.pieces.length
      );
      const unused = new Set(expected.pieces.map((_, index) => index));
      for (const run of mine) {
        const ends = [run[0], run[run.length - 1]];
        const match = [...unused].find(index => {
          const other = expected.pieces[index];
          const first = other[0];
          const last = other[other.length - 1];
          const same = (a: readonly number[], b: readonly number[]) =>
            a[0] === b[0] && a[1] === b[1];
          return (
            (same(ends[0], first) && same(ends[1], last)) ||
            (same(ends[0], last) && same(ends[1], first))
          );
        });
        expect(match).toBeDefined();
        unused.delete(match as number);
      }
      if (direction === 1) {
        forwardLength += length;
      } else {
        backwardLength += length;
      }
    }
  }
  // No run for a pair Shapely reports as empty.
  expect(used.size).toBe(result.count);
  console.log(
    `GPUSharedPaths: ${shared.length} pairs, ${result.count} runs, forward ${forwardLength.toFixed(3)}, ` +
      `backward ${backwardLength.toFixed(3)}, ${elapsed.toFixed(0)} ms`
  );
});

it('GPUSharedPaths#reports overflow and keeps a consistent prefix', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const left = [
    [
      [0, 0],
      [10, 0]
    ],
    [
      [0, 1],
      [10, 1]
    ]
  ];
  const right = [
    [
      [2, 0],
      [4, 0]
    ],
    [
      [2, 1],
      [4, 1]
    ]
  ];
  const limited = await runSharedPaths(device, left, right, {runCapacity: 1});
  expect(limited.count).toBe(1);
  expect(limited.requiredCount).toBe(2);
  expect(limited.overflow).toBe(1);
  const pairs = await runSharedPaths(device, left, right, {intersectionCapacity: 1});
  expect(pairs.overflow).toBe(1);
  const vertices = await runSharedPaths(device, left, right, {vertexCapacity: 3});
  expect(vertices.count).toBe(1);
  expect(vertices.overflow).toBe(1);
});

it('GPUSharedPaths#compiles within the default storage buffer limit', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) return;
  const result = await runSharedPaths(
    device,
    [
      [
        [0, 0],
        [4, 0]
      ]
    ],
    [
      [
        [1, 0],
        [3, 0]
      ]
    ]
  );
  expect(result.runs).toEqual([
    [
      [1, 0],
      [3, 0]
    ]
  ]);
});
