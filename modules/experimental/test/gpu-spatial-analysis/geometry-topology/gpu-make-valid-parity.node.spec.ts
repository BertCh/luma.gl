// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';

type Point = readonly [number, number];

const sign = (value: number) => (value > 0 ? 1 : value < 0 ? -1 : 0);
const compare = (value: number, reference: number) => sign(value - reference);
// Integer coordinates keep the double-precision determinant exact, like orientSign on the GPU.
const orient = (a: Point, b: Point, c: Point) =>
  sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));

/**
 * CPU replay of the `face-parity` kernel of GPUMakeValid: parity of the number of crossings of a
 * +x ray from the symbolic point a + delta * (b - a) + epsilon * leftNormal(b - a), which lies
 * strictly inside the face on the left of the directed edge a -> b.
 */
function hasOddSourceParity(rings: readonly (readonly Point[])[], a: Point, b: Point): boolean {
  let probeSign = compare(b[1], a[1]);
  if (probeSign === 0) probeSign = compare(b[0], a[0]);
  const isAbove = (value: Point) => (value[1] !== a[1] ? value[1] > a[1] : probeSign < 0);
  let parity = 0;
  for (const ring of rings) {
    for (let i = 0; i + 1 < ring.length; i++) {
      const first = ring[i];
      const second = ring[i + 1];
      const firstAbove = isAbove(first);
      if (firstAbove === isAbove(second)) continue;
      const low = firstAbove ? second : first;
      const high = firstAbove ? first : second;
      let side = orient(low, high, a);
      if (side === 0) {
        side = orient(low, high, b);
        if (side === 0) {
          side = compare(high[0], low[0]) * compare(b[0], a[0]);
          if (side === 0) side = compare(high[1], low[1]) * compare(b[1], a[1]);
        }
      }
      if (side > 0) parity ^= 1;
    }
  }
  return parity === 1;
}

// Faces and expectations follow shapely 2.2.0 `make_valid` on the same rings
// (scratchpad/fix/work/W2A/makevalid/oracle.py): a face survives when its source parity is odd.
const SQUARE: Point[] = [
  [0, 0],
  [4, 0],
  [4, 4],
  [0, 4],
  [0, 0]
];
const HOLE: Point[] = [
  [1, 1],
  [1, 3],
  [3, 3],
  [3, 1],
  [1, 1]
];

it('source parity keeps the annulus and drops the face filling the hole', () => {
  const rings = [SQUARE, HOLE];
  expect(hasOddSourceParity(rings, [0, 0], [4, 0])).toBe(true);
  // Island face: counter-clockwise copy of the hole ring, covered by both rings (even).
  expect(hasOddSourceParity(rings, [1, 1], [3, 1])).toBe(false);
});

it('source parity keeps both triangles of a bow tie', () => {
  const rings = [
    [
      [0, 0],
      [2, 2],
      [0, 2],
      [2, 0],
      [0, 0]
    ] as Point[]
  ];
  // Bottom and top triangles meet at (1, 1); each is covered once.
  expect(hasOddSourceParity(rings, [0, 0], [2, 0])).toBe(true);
  expect(hasOddSourceParity(rings, [2, 2], [0, 2])).toBe(true);
  // The reversed bottom edge has the exterior on its left.
  expect(hasOddSourceParity(rings, [2, 0], [0, 0])).toBe(false);
});

it('source parity drops the lens of a hole ring crossing the shell and keeps the lobe', () => {
  const crossing: Point[] = [
    [3, 1],
    [5, 1],
    [5, 3],
    [3, 3],
    [3, 1]
  ];
  const rings = [SQUARE, crossing];
  // Lens [3,4] x [1,3]: probe starts at (3, 1), a vertex of the hole ring, collinear with it.
  expect(hasOddSourceParity(rings, [3, 1], [4, 1])).toBe(false);
  // Shell remainder.
  expect(hasOddSourceParity(rings, [0, 0], [4, 0])).toBe(true);
  // Lobe outside the shell: probe starts on the shell's right side, a T junction.
  expect(hasOddSourceParity(rings, [4, 1], [5, 1])).toBe(true);
});

it('source parity keeps an island inside a hole', () => {
  const rings: Point[][] = [
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
      [0, 0]
    ],
    [
      [2, 2],
      [2, 8],
      [8, 8],
      [8, 2],
      [2, 2]
    ],
    [
      [4, 4],
      [6, 4],
      [6, 6],
      [4, 6],
      [4, 4]
    ]
  ];
  expect(hasOddSourceParity(rings, [0, 0], [10, 0])).toBe(true);
  // Water between the lake shell and the island: covered twice (even).
  expect(hasOddSourceParity(rings, [2, 2], [8, 2])).toBe(false);
  expect(hasOddSourceParity(rings, [4, 4], [6, 4])).toBe(true);
});
