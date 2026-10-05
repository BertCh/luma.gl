// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPULineSimplificationMetric} from '../../../src/geospatial/line-simplification';

const fround = Math.fround;
const FLOAT_VIEW = new Float32Array(1);
const BITS_VIEW = new Uint32Array(FLOAT_VIEW.buffer);
const MAXIMUM_FINITE = fround(3.40282347e38);
const ENDPOINT_BITS = 0x7f800000;
const NO_ROW = 0xffffffff;

/** Returns the f32 bit pattern of `value`. */
export function getFloatBits(value: number): number {
  FLOAT_VIEW[0] = value;
  return BITS_VIEW[0];
}

function getFloatFromBits(bits: number): number {
  BITS_VIEW[0] = bits >>> 0;
  return FLOAT_VIEW[0];
}

function getNextUp(value: number): number {
  return getFloatFromBits(getFloatBits(value) + 1);
}

function getNextDown(value: number): number {
  return getFloatFromBits(getFloatBits(value) - 1);
}

/** Largest f32 `q >= 0` with `fround(q * denominator) <= numerator` (mirrors `lineDivideFloor`). */
export function divideFloor(numerator: number, denominator: number): number {
  if (!(numerator > 0)) {
    return 0;
  }
  let quotient = fround(numerator / denominator);
  if (!(quotient >= 0)) {
    quotient = 0;
  }
  quotient = Math.min(quotient, MAXIMUM_FINITE);
  while (quotient > 0 && fround(quotient * denominator) > numerator) {
    quotient = getNextDown(quotient);
  }
  while (fround(getNextUp(quotient) * denominator) <= numerator) {
    quotient = getNextUp(quotient);
  }
  return quotient;
}

/** Largest f32 `s >= 0` with `fround(s * s) <= value` (mirrors `lineSqrtFloor`). */
export function sqrtFloor(value: number): number {
  if (!(value > 0)) {
    return 0;
  }
  let root = fround(Math.sqrt(value));
  if (!(root >= 0)) {
    root = 0;
  }
  root = Math.min(root, MAXIMUM_FINITE);
  while (root > 0 && fround(root * root) > value) {
    root = getNextDown(root);
  }
  while (fround(getNextUp(root) * getNextUp(root)) <= value) {
    root = getNextUp(root);
  }
  return root;
}

function getPointDistance(px: number, py: number, ax: number, ay: number): number {
  const deltaX = fround(px - ax);
  const deltaY = fround(py - ay);
  return sqrtFloor(fround(fround(deltaX * deltaX) + fround(deltaY * deltaY)));
}

/** Mirrors `lineSegmentDistance`. */
export function getSegmentDistance(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number
): number {
  const segmentX = fround(bx - ax);
  const segmentY = fround(by - ay);
  const offsetX = fround(px - ax);
  const offsetY = fround(py - ay);
  const lengthSquared = fround(fround(segmentX * segmentX) + fround(segmentY * segmentY));
  const projection = fround(fround(offsetX * segmentX) + fround(offsetY * segmentY));
  if (lengthSquared <= 0 || projection <= 0) {
    return getPointDistance(px, py, ax, ay);
  }
  if (projection >= lengthSquared) {
    return getPointDistance(px, py, bx, by);
  }
  const segmentLength = sqrtFloor(lengthSquared);
  if (segmentLength <= 0) {
    return getPointDistance(px, py, ax, ay);
  }
  const cross = Math.abs(fround(fround(segmentX * offsetY) - fround(segmentY * offsetX)));
  return divideFloor(cross, segmentLength);
}

/** Mirrors `lineTimeRatioDistance` (TD-TR synchronized Euclidean distance). */
export function getTimeRatioDistance(
  [px, py, pt]: readonly number[],
  [ax, ay, at]: readonly number[],
  [bx, by, bt]: readonly number[]
): number {
  const duration = fround(bt - at);
  if (!(duration > 0)) {
    return getPointDistance(px, py, ax, ay);
  }
  const elapsed = fround(pt - at);
  let ratio = divideFloor(Math.abs(elapsed), duration);
  if (elapsed < 0) {
    ratio = -ratio;
  }
  const expectedX = fround(ax + fround(ratio * fround(bx - ax)));
  const expectedY = fround(ay + fround(ratio * fround(by - ay)));
  return getPointDistance(px, py, expectedX, expectedY);
}

/** Input of the CPU oracles. */
export type LineSimplificationScene = {
  /** Interleaved f32 positions, two per row. */
  positions: Float32Array;
  /** `lineCount + 1` row offsets. */
  trackOffsets: Uint32Array;
  /** Optional f32 times, one per row. */
  timestamps?: Float32Array;
  /** Distance metric. Default `'segment'`. */
  metric?: GPULineSimplificationMetric;
};

/** Returns the canonical distance key of `row` against the chord `left..right`. */
function getDistanceKey(
  scene: LineSimplificationScene,
  row: number,
  left: number,
  right: number
): number {
  const {positions, timestamps} = scene;
  let distance: number;
  if (scene.metric === 'time-ratio') {
    const sample = (index: number) => [
      positions[2 * index],
      positions[2 * index + 1],
      timestamps ? timestamps[index] : 0
    ];
    distance = getTimeRatioDistance(sample(row), sample(left), sample(right));
  } else {
    distance = getSegmentDistance(
      positions[2 * row],
      positions[2 * row + 1],
      positions[2 * left],
      positions[2 * left + 1],
      positions[2 * right],
      positions[2 * right + 1]
    );
  }
  return distance <= 0 ? 0 : getFloatBits(distance);
}

/** Result of {@link computeParallelImportance}. */
export type ParallelImportanceResult = {
  /** f32 bit patterns of the importance column. */
  importanceBits: Uint32Array;
  /** Rounds that ran. */
  roundCount: number;
  /** Whether every row was decided within the cap. */
  converged: boolean;
};

/**
 * CPU oracle of the level-synchronous formulation run by `GPULineSimplification`: same init,
 * rounds, ties, zero-interval shortcut, gate, and unresolved fallback.
 */
export function computeParallelImportance(
  scene: LineSimplificationScene,
  maximumRounds: number
): ParallelImportanceResult {
  const rowCount = scene.positions.length / 2;
  const {trackOffsets} = scene;
  const lineCount = trackOffsets.length - 1;
  const left = new Uint32Array(rowCount);
  const right = new Uint32Array(rowCount);
  const importanceBits = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    left[row] = row;
    right[row] = row;
  }
  for (let line = 0; line < lineCount; line++) {
    const start = trackOffsets[line];
    const end = trackOffsets[line + 1];
    if (end <= start) {
      continue;
    }
    for (let row = start; row < end; row++) {
      if (row === start || row === end - 1) {
        importanceBits[row] = ENDPOINT_BITS;
      } else {
        left[row] = start;
        right[row] = end - 1;
      }
    }
  }
  const isUndecided = (row: number) => left[row] < row && row < right[row];

  let roundCount = 0;
  let converged = false;
  while (roundCount < maximumRounds) {
    const keys = new Uint32Array(rowCount);
    const bestKeys = new Map<number, number>();
    const bestRows = new Map<number, number>();
    for (let row = 0; row < rowCount; row++) {
      if (isUndecided(row)) {
        keys[row] = getDistanceKey(scene, row, left[row], right[row]);
        bestKeys.set(left[row], Math.max(bestKeys.get(left[row]) ?? 0, keys[row]));
      }
    }
    for (let row = 0; row < rowCount; row++) {
      if (isUndecided(row) && keys[row] === bestKeys.get(left[row])) {
        bestRows.set(left[row], Math.min(bestRows.get(left[row]) ?? NO_ROW, row));
      }
    }
    let remaining = false;
    for (let row = 0; row < rowCount; row++) {
      if (!isUndecided(row)) {
        continue;
      }
      const leftAnchor = left[row];
      const rightAnchor = right[row];
      if (bestKeys.get(leftAnchor) === 0) {
        importanceBits[row] = 0;
        left[row] = row;
        right[row] = row;
        continue;
      }
      const splitRow = bestRows.get(leftAnchor) as number;
      if (row === splitRow) {
        const parent = Math.min(importanceBits[leftAnchor], importanceBits[rightAnchor]);
        importanceBits[row] = Math.min(keys[row], parent);
        left[row] = row;
        right[row] = row;
        continue;
      }
      if (row < splitRow) {
        right[row] = splitRow;
      } else {
        left[row] = splitRow;
      }
      remaining = true;
    }
    roundCount++;
    if (!remaining) {
      converged = true;
      break;
    }
  }
  for (let row = 0; row < rowCount; row++) {
    if (isUndecided(row)) {
      importanceBits[row] = Math.min(importanceBits[left[row]], importanceBits[right[row]]);
    }
  }
  return {importanceBits, roundCount, converged};
}

/** Rows kept at `tolerance` by an importance column: rows inside a line with importance > tolerance. */
export function getKeptRowsFromImportance(
  scene: LineSimplificationScene,
  importanceBits: Uint32Array,
  tolerance: number
): number[] {
  const {trackOffsets} = scene;
  const first = trackOffsets[0];
  const end = trackOffsets[trackOffsets.length - 1];
  const tolerance32 = fround(tolerance);
  const kept: number[] = [];
  for (let row = first; row < end; row++) {
    if (getFloatFromBits(importanceBits[row]) > tolerance32) {
      kept.push(row);
    }
  }
  return kept;
}

/**
 * Classic recursive Douglas-Peucker at `tolerance` over every line: keep both endpoints, split at
 * the first row with the largest distance when that distance is `> tolerance`, recurse.
 */
export function simplifyDouglasPeucker(
  scene: LineSimplificationScene,
  tolerance: number
): number[] {
  const {trackOffsets} = scene;
  const tolerance32 = fround(tolerance);
  const kept: number[] = [];
  const recurse = (start: number, last: number): void => {
    let bestRow = -1;
    let bestDistance = -1;
    for (let row = start + 1; row < last; row++) {
      const distance = getFloatFromBits(getDistanceKey(scene, row, start, last));
      if (distance > bestDistance) {
        bestDistance = distance;
        bestRow = row;
      }
    }
    if (bestRow >= 0 && bestDistance > tolerance32) {
      recurse(start, bestRow);
      kept.push(bestRow);
      recurse(bestRow, last);
    }
  };
  for (let line = 0; line + 1 < trackOffsets.length; line++) {
    const start = trackOffsets[line];
    const end = trackOffsets[line + 1];
    if (end <= start) {
      continue;
    }
    kept.push(start);
    if (end - 1 > start) {
      recurse(start, end - 1);
      kept.push(end - 1);
    }
  }
  return kept;
}

/** Per-line kept counts and first positions in the ascending kept list. */
export function getKeptLineRanges(
  scene: LineSimplificationScene,
  keptRows: readonly number[]
): {lineCounts: number[]; lineStarts: number[]} {
  const {trackOffsets} = scene;
  const lineCounts: number[] = [];
  const lineStarts: number[] = [];
  const lowerBound = (row: number) => {
    let index = 0;
    while (index < keptRows.length && keptRows[index] < row) {
      index++;
    }
    return index;
  };
  for (let line = 0; line + 1 < trackOffsets.length; line++) {
    const start = trackOffsets[line];
    const end = Math.max(trackOffsets[line + 1], start);
    const first = lowerBound(start);
    lineStarts.push(first);
    lineCounts.push(lowerBound(end) - first);
  }
  return {lineCounts, lineStarts};
}
