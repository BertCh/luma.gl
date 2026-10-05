// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// BigInt CPU port of H3 `h3NeighborRotations` (Apache-2.0), mirroring `H3_NEIGHBOR_WGSL`.

import {
  H3_BASE_CELL_HOME_FACES,
  H3_BASE_CELL_NEIGHBOR_ROTATIONS,
  H3_BASE_CELL_NEIGHBORS,
  H3_INVALID_BASE_CELL,
  H3_NEW_ADJUSTMENT_CLASS_II,
  H3_NEW_ADJUSTMENT_CLASS_III,
  H3_NEW_DIGIT_CLASS_II,
  H3_NEW_DIGIT_CLASS_III,
  H3_PENTAGON_CLOCKWISE_OFFSET_FACES
} from '../../../src/map-graphs/cell-topology/h3-neighbor-tables';

const PENTAGON_BASE_CELLS = new Set([4, 14, 24, 38, 49, 58, 63, 72, 83, 97, 107, 117]);
const ROTATE_60_COUNTER_CLOCKWISE = [0, 5, 3, 1, 6, 4, 2, 7];
const ROTATE_60_CLOCKWISE = [0, 3, 6, 2, 5, 1, 4, 7];

/** Resolution of an H3 index. */
export function getH3Resolution(cell: bigint): number {
  return Number((cell >> 52n) & 15n);
}

/** Base cell of an H3 index. */
export function getH3BaseCell(cell: bigint): number {
  return Number((cell >> 45n) & 127n);
}

/** Digit at `resolution` (1-15) of an H3 index. */
export function getH3Digit(cell: bigint, resolution: number): number {
  return Number((cell >> BigInt(3 * (15 - resolution))) & 7n);
}

function setH3Digit(cell: bigint, resolution: number, digit: number): bigint {
  const shift = BigInt(3 * (15 - resolution));
  return (cell & ~(7n << shift)) | (BigInt(digit) << shift);
}

function setH3BaseCell(cell: bigint, baseCell: number): bigint {
  return (cell & ~(127n << 45n)) | (BigInt(baseCell) << 45n);
}

/** First nonzero digit of an H3 index, or 0 for a center-child chain. */
export function getH3LeadingNonZeroDigit(cell: bigint): number {
  const resolution = getH3Resolution(cell);
  for (let level = 1; level <= resolution; level++) {
    const digit = getH3Digit(cell, level);
    if (digit !== 0) {
      return digit;
    }
  }
  return 0;
}

/** True for one of the 12 pentagons at any resolution. */
export function isH3Pentagon(cell: bigint): boolean {
  return PENTAGON_BASE_CELLS.has(getH3BaseCell(cell)) && getH3LeadingNonZeroDigit(cell) === 0;
}

function rotateDigits(cell: bigint, rotation: readonly number[]): bigint {
  let result = cell;
  const resolution = getH3Resolution(cell);
  for (let level = 1; level <= resolution; level++) {
    result = setH3Digit(result, level, rotation[getH3Digit(result, level)]);
  }
  return result;
}

function rotatePentagon60CounterClockwise(cell: bigint): bigint {
  let result = cell;
  let foundFirstNonZeroDigit = false;
  const resolution = getH3Resolution(cell);
  for (let level = 1; level <= resolution; level++) {
    result = setH3Digit(result, level, ROTATE_60_COUNTER_CLOCKWISE[getH3Digit(result, level)]);
    if (!foundFirstNonZeroDigit && getH3Digit(result, level) !== 0) {
      foundFirstNonZeroDigit = true;
      if (getH3LeadingNonZeroDigit(result) === 1) {
        result = rotateDigits(result, ROTATE_60_COUNTER_CLOCKWISE);
      }
    }
  }
  return result;
}

/**
 * Neighbor of a valid H3 cell in direction 1-6 (K, J, JK, I, IK, IJ), or `0n` for the deleted K
 * direction at a pentagon center.
 */
export function getH3Neighbor(cell: bigint, direction: number): bigint {
  if (direction === 1 && isH3Pentagon(cell)) {
    // Also at resolution 0, where H3 itself would return the IK neighbor a second time.
    return 0n;
  }
  let current = cell;
  let stepDirection = direction;
  let newRotations = 0;
  const oldBaseCell = getH3BaseCell(cell);
  const oldLeadingDigit = getH3LeadingNonZeroDigit(cell);
  for (let level = getH3Resolution(cell); ; level--) {
    if (level === 0) {
      let tableIndex = oldBaseCell * 7 + stepDirection;
      if (H3_BASE_CELL_NEIGHBORS[tableIndex] === H3_INVALID_BASE_CELL) {
        tableIndex = oldBaseCell * 7 + 5;
        current = rotateDigits(current, ROTATE_60_COUNTER_CLOCKWISE);
      }
      current = setH3BaseCell(current, H3_BASE_CELL_NEIGHBORS[tableIndex]);
      newRotations = H3_BASE_CELL_NEIGHBOR_ROTATIONS[tableIndex];
      break;
    }
    const oldDigit = getH3Digit(current, level);
    const isClassIII = (level & 1) === 1;
    const digits = isClassIII ? H3_NEW_DIGIT_CLASS_III : H3_NEW_DIGIT_CLASS_II;
    const adjustments = isClassIII ? H3_NEW_ADJUSTMENT_CLASS_III : H3_NEW_ADJUSTMENT_CLASS_II;
    current = setH3Digit(current, level, digits[oldDigit * 7 + stepDirection]);
    const nextDirection = adjustments[oldDigit * 7 + stepDirection];
    if (nextDirection === 0) {
      break;
    }
    stepDirection = nextDirection;
  }

  const newBaseCell = getH3BaseCell(current);
  if (!PENTAGON_BASE_CELLS.has(newBaseCell)) {
    for (let rotation = 0; rotation < newRotations; rotation++) {
      current = rotateDigits(current, ROTATE_60_COUNTER_CLOCKWISE);
    }
    return current;
  }
  if (getH3LeadingNonZeroDigit(current) === 1) {
    if (oldBaseCell !== newBaseCell) {
      const clockwiseFaces = H3_PENTAGON_CLOCKWISE_OFFSET_FACES[newBaseCell];
      const isClockwise = clockwiseFaces?.includes(H3_BASE_CELL_HOME_FACES[oldBaseCell]) ?? false;
      current = rotateDigits(
        current,
        isClockwise ? ROTATE_60_CLOCKWISE : ROTATE_60_COUNTER_CLOCKWISE
      );
    } else if (oldLeadingDigit === 0) {
      return 0n;
    } else if (oldLeadingDigit === 3) {
      current = rotateDigits(current, ROTATE_60_COUNTER_CLOCKWISE);
    } else if (oldLeadingDigit === 5) {
      current = rotateDigits(current, ROTATE_60_CLOCKWISE);
    } else {
      return 0n;
    }
  }
  for (let rotation = 0; rotation < newRotations; rotation++) {
    current = rotatePentagon60CounterClockwise(current);
  }
  return current;
}

/** Cells within `k` steps of `cell`, with distances, by breadth-first neighbor stepping. */
export function getH3DiskByBreadthFirstSearch(cell: bigint, k: number): Map<bigint, number> {
  const distances = new Map<bigint, number>([[cell, 0]]);
  let frontier = [cell];
  for (let distance = 1; distance <= k; distance++) {
    const next: bigint[] = [];
    for (const origin of frontier) {
      for (let direction = 1; direction <= 6; direction++) {
        const neighbor = getH3Neighbor(origin, direction);
        if (neighbor !== 0n && !distances.has(neighbor)) {
          distances.set(neighbor, distance);
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  return distances;
}
