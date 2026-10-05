// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createXorshiftRandom} from '../terrain-oracle-utils';

/** CPU oracle for terrain critical-point classification (pure comparisons). Test-only. */

/** Class codes, mirrored here so the oracle does not import the contributor. */
export const CRITICAL_POINT = {
  regular: 0,
  peak: 1,
  pit: 2,
  saddle: 3,
  boundary: 4,
  noData: 5
} as const;

/** Ring offsets `[columnOffset, rowOffset]` in cyclic order. Rows grow downward, so north is -1. */
export const RING_OFFSETS = {
  8: [
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1]
  ],
  6: [
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 0],
    [-1, -1],
    [0, -1]
  ]
} as const;

/** Classification result: one class and one sign-change count per pixel. */
export type CriticalPointResult = {
  classes: Uint32Array;
  signChanges: Uint32Array;
  counts: Uint32Array;
};

/** Classifies every pixel of a `width * height` float32 grid. Invalid pixels may hold any value. */
export function classifyCriticalPoints(
  elevation: Float32Array,
  validity: Uint32Array | undefined,
  width: number,
  height: number,
  connectivity: 8 | 6 = 8
): CriticalPointResult {
  const ring = RING_OFFSETS[connectivity];
  const classes = new Uint32Array(width * height);
  const signChanges = new Uint32Array(width * height);
  const counts = new Uint32Array(6);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      let code: number = CRITICAL_POINT.regular;
      let changes = 0;
      if (validity && validity[index] === 0) {
        code = CRITICAL_POINT.noData;
      } else if (column === 0 || row === 0 || column === width - 1 || row === height - 1) {
        code = CRITICAL_POINT.boundary;
      } else {
        const higher: boolean[] = [];
        let allValid = true;
        for (const [columnOffset, rowOffset] of ring) {
          const neighborIndex = (row + rowOffset) * width + column + columnOffset;
          if (validity && validity[neighborIndex] === 0) {
            allValid = false;
            break;
          }
          const neighborHeight = elevation[neighborIndex];
          const centerHeight = elevation[index];
          higher.push(
            neighborHeight > centerHeight ||
              (neighborHeight === centerHeight && neighborIndex > index)
          );
        }
        if (!allValid) {
          code = CRITICAL_POINT.boundary;
        } else {
          for (let k = 0; k < higher.length; k++) {
            if (higher[k] !== higher[(k + 1) % higher.length]) {
              changes++;
            }
          }
          if (changes === 0) {
            code = higher[0] ? CRITICAL_POINT.pit : CRITICAL_POINT.peak;
          } else if (changes === 2) {
            code = CRITICAL_POINT.regular;
          } else {
            code = CRITICAL_POINT.saddle;
          }
        }
      }
      classes[index] = code;
      signChanges[index] = changes;
      counts[code]++;
    }
  }
  return {classes, signChanges, counts};
}

/**
 * Euler (Morse) index sum `peaks - saddles + pits` with saddle multiplicity `changes / 2 - 1`,
 * over the pixels `[columnStart, columnEnd) x [rowStart, rowEnd)`.
 */
export function getEulerSum(
  result: CriticalPointResult,
  width: number,
  window: {columnStart: number; columnEnd: number; rowStart: number; rowEnd: number}
): {peaks: number; pits: number; saddleMultiplicity: number; sum: number} {
  let peaks = 0;
  let pits = 0;
  let saddleMultiplicity = 0;
  for (let row = window.rowStart; row < window.rowEnd; row++) {
    for (let column = window.columnStart; column < window.columnEnd; column++) {
      const index = row * width + column;
      const code = result.classes[index];
      if (code === CRITICAL_POINT.peak) {
        peaks++;
      } else if (code === CRITICAL_POINT.pit) {
        pits++;
      } else if (code === CRITICAL_POINT.saddle) {
        saddleMultiplicity += result.signChanges[index] / 2 - 1;
      }
    }
  }
  return {peaks, pits, saddleMultiplicity, sum: peaks - saddleMultiplicity + pits};
}

/** Seeded float32 random DEM; `levels > 0` quantises to create plateaus and ties. */
export function createRandomElevation(
  width: number,
  height: number,
  seed: number,
  levels = 0
): Float32Array {
  const next = createXorshiftRandom(seed);
  return Float32Array.from({length: width * height}, () =>
    levels > 0 ? Math.floor(next() * levels) : next() * 100
  );
}

/** Egg-crate `sin(2 pi (x + phase) / period) * sin(2 pi (y + phase) / period)` with phase off the grid. */
export function createEggCrate(
  width: number,
  height: number,
  period: number,
  phase: number
): Float32Array {
  return Float32Array.from({length: width * height}, (_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    return (
      Math.sin((2 * Math.PI * (column + phase)) / period) *
      Math.sin((2 * Math.PI * (row + phase)) / period)
    );
  });
}

/** Sum of two Gaussian bumps; the second is lower and off-center so no height is symmetric. */
export function createTwoBumps(size: number): Float32Array {
  const bump = (x: number, y: number, centerX: number, centerY: number, sigma: number) =>
    Math.exp(-((x - centerX) ** 2 + (y - centerY) ** 2) / (2 * sigma * sigma));
  return Float32Array.from({length: size * size}, (_, index) => {
    const column = index % size;
    const row = Math.floor(index / size);
    return (
      100 * bump(column, row, size * 0.3 + 0.17, size * 0.35 + 0.23, size / 9) +
      80 * bump(column, row, size * 0.7 + 0.11, size * 0.62 + 0.29, size / 9)
    );
  });
}
