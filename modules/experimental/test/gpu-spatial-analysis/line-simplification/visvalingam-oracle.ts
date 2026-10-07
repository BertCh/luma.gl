// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getFloatBits} from './line-simplification-oracle';

const fround = Math.fround;
const ENDPOINT_BITS = 0x7f800000;

/** Result of {@link computeVisvalingamImportance}. */
export type VisvalingamImportanceResult = {
  /** f32 bits of the effective area of each row; `0x7f800000` (+Infinity) for endpoints. */
  importanceBits: Uint32Array;
  /** Rounds that ran, including the final round that removes the last rows. */
  roundCount: number;
  /** Whether every interior row was decided within `maximumRounds`. */
  converged: boolean;
};

/** Mirrors the `area` kernel: `0.5 * |cross|` from correctly rounded f32 operations. */
export function getTriangleAreaBits(
  positions: Float32Array,
  previousRow: number,
  row: number,
  nextRow: number
): number {
  const originX = positions[2 * previousRow];
  const originY = positions[2 * previousRow + 1];
  const toMiddleX = fround(positions[2 * row] - originX);
  const toMiddleY = fround(positions[2 * row + 1] - originY);
  const toNextX = fround(positions[2 * nextRow] - originX);
  const toNextY = fround(positions[2 * nextRow + 1] - originY);
  const cross = fround(fround(toMiddleX * toNextY) - fround(toMiddleY * toNextX));
  return getFloatBits(fround(Math.abs(cross) * 0.5));
}

/**
 * CPU mirror of the GPU Visvalingam-Whyatt rounds: bit-identical importance and round count.
 * Each round removes every surviving interior row whose `(area, row)` orders before all surviving
 * interior rows within `radius` steps on both sides; the effective area is
 * `max(area, floor)`, and removal raises the floor of both neighbors.
 */
export function computeVisvalingamImportance(
  positions: Float32Array,
  trackOffsets: Uint32Array,
  radius: number,
  maximumRounds: number
): VisvalingamImportanceResult {
  const rowCount = positions.length / 2;
  const previous = new Uint32Array(rowCount);
  const next = new Uint32Array(rowCount);
  const floor = new Uint32Array(rowCount);
  const key = new Uint32Array(rowCount);
  const importanceBits = new Uint32Array(rowCount);
  const lineCount = trackOffsets.length - 1;
  for (let row = 0; row < rowCount; row++) {
    previous[row] = row;
    next[row] = row;
  }
  for (let line = 0; line < lineCount; line++) {
    const first = trackOffsets[line];
    const last = trackOffsets[line + 1] - 1;
    for (let row = first; row <= last; row++) {
      if (row === first || row === last) {
        importanceBits[row] = ENDPOINT_BITS;
      } else {
        previous[row] = row - 1;
        next[row] = row + 1;
      }
    }
  }
  const isUndecided = (row: number) => previous[row] < row && row < next[row];
  const hasSmallerNeighbor = (row: number, forward: boolean) => {
    let cursor = row;
    for (let step = 0; step < radius; step++) {
      cursor = forward ? next[cursor] : previous[cursor];
      if (!isUndecided(cursor)) {
        return false;
      }
      if (key[cursor] < key[row] || (key[cursor] === key[row] && cursor < row)) {
        return true;
      }
    }
    return false;
  };
  let roundCount = 0;
  let converged = false;
  for (let round = 0; round < maximumRounds; round++) {
    roundCount++;
    for (let row = 0; row < rowCount; row++) {
      if (isUndecided(row)) {
        key[row] = getTriangleAreaBits(positions, previous[row], row, next[row]);
      }
    }
    const removed: number[] = [];
    let remaining = false;
    for (let row = 0; row < rowCount; row++) {
      if (!isUndecided(row)) {
        continue;
      }
      if (hasSmallerNeighbor(row, false) || hasSmallerNeighbor(row, true)) {
        remaining = true;
      } else {
        importanceBits[row] = Math.max(key[row], floor[row]);
        removed.push(row);
      }
    }
    for (const row of removed) {
      const before = previous[row];
      const after = next[row];
      next[before] = after;
      previous[after] = before;
      floor[before] = Math.max(floor[before], importanceBits[row]);
      floor[after] = Math.max(floor[after], importanceBits[row]);
      previous[row] = row;
      next[row] = row;
    }
    if (!remaining) {
      converged = true;
      break;
    }
  }
  for (let row = 0; row < rowCount; row++) {
    if (isUndecided(row)) {
      importanceBits[row] = ENDPOINT_BITS;
    }
  }
  return {importanceBits, roundCount, converged};
}

/** Rows whose importance is strictly greater than `tolerance`, ascending. */
export function getKeptRowsAboveTolerance(
  importanceBits: Uint32Array,
  tolerance: number
): number[] {
  const kept: number[] = [];
  const toleranceBits = getFloatBits(tolerance);
  for (let row = 0; row < importanceBits.length; row++) {
    if (importanceBits[row] > toleranceBits) {
      kept.push(row);
    }
  }
  return kept;
}
