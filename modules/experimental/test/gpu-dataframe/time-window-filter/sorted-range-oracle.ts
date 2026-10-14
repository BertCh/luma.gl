// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU oracles of the sorted range mode of `GPUTimeWindowFilter`.
 *
 * `scanRange` applies the mask-path acceptance to every row; `searchRange` mirrors the two binary
 * searches of the range kernel. On non-decreasing input the two must agree.
 */

/** Float rows: `high` with optional double-single `low` parts. */
export type SortedFloatRows = {high: ArrayLike<number>; low?: ArrayLike<number>};

/** Window of the float path: double-single start and end. */
export type SortedFloatWindow = {
  startHigh: number;
  startLow: number;
  endHigh: number;
  endLow: number;
};

/** Window of the word path: integer parts with f32 fractions in `[0, 1)`. */
export type SortedWordWindow = {
  start: bigint;
  startFraction: number;
  end: bigint;
  endFraction: number;
};

/** Contiguous accepted rows. */
export type SortedRange = {firstRow: number; rowCount: number};

/** Splits a double into the float32 `high` and `low` parts the filter reads. */
export function splitFloatWindow(start: number, end: number): SortedFloatWindow {
  const startHigh = Math.fround(start);
  const endHigh = Math.fround(end);
  return {
    startHigh,
    startLow: Math.fround(start - startHigh),
    endHigh,
    endLow: Math.fround(end - endHigh)
  };
}

/** Exact double-single `a >= b` as in `isTimeAtLeast`. */
export function isFloatTimeAtLeast(
  aHigh: number,
  aLow: number,
  bHigh: number,
  bLow: number
): boolean {
  return aHigh > bHigh || (aHigh === bHigh && aLow >= bLow);
}

/** Exact `a + aFraction >= b + bFraction` for Int64 values as in `isTimeWordsAtLeast`. */
export function isWordTimeAtLeast(
  a: bigint,
  aFraction: number,
  b: bigint,
  bFraction: number
): boolean {
  return a === b ? aFraction >= bFraction : a > b;
}

function isFloatAccepted(rows: SortedFloatRows, row: number, window: SortedFloatWindow): boolean {
  const low = rows.low?.[row] ?? 0;
  return (
    isFloatTimeAtLeast(window.endHigh, window.endLow, window.startHigh, window.startLow) &&
    isFloatTimeAtLeast(rows.high[row], low, window.startHigh, window.startLow) &&
    isFloatTimeAtLeast(window.endHigh, window.endLow, rows.high[row], low)
  );
}

function isWordAccepted(times: ArrayLike<bigint>, row: number, window: SortedWordWindow) {
  return (
    isWordTimeAtLeast(window.end, window.endFraction, window.start, window.startFraction) &&
    isWordTimeAtLeast(times[row], 0, window.start, window.startFraction) &&
    isWordTimeAtLeast(window.end, window.endFraction, times[row], 0)
  );
}

/** Brute force: the accepted rows by the mask-path test, as a range (throws if not contiguous). */
export function scanRange(rowCount: number, isAccepted: (row: number) => boolean): SortedRange {
  let firstRow = -1;
  let count = 0;
  for (let row = 0; row < rowCount; row++) {
    if (!isAccepted(row)) {
      continue;
    }
    if (firstRow < 0) {
      firstRow = row;
    } else if (firstRow + count !== row) {
      throw new Error('accepted rows are not contiguous: the input is not sorted');
    }
    count++;
  }
  return {firstRow: firstRow < 0 ? 0 : firstRow, rowCount: count};
}

/** Brute-force range of float rows. */
export function scanFloatRange(rows: SortedFloatRows, window: SortedFloatWindow): SortedRange {
  return scanRange(rows.high.length, row => isFloatAccepted(rows, row, window));
}

/** Brute-force range of word rows. */
export function scanWordRange(times: ArrayLike<bigint>, window: SortedWordWindow): SortedRange {
  return scanRange(times.length, row => isWordAccepted(times, row, window));
}

/** The two binary searches of the kernel: lower bound of the start, upper bound of the end. */
export function searchRange(
  rowCount: number,
  isWindowValid: boolean,
  isBeforeStart: (row: number) => boolean,
  isAfterEnd: (row: number) => boolean
): SortedRange {
  if (!isWindowValid) {
    return {firstRow: 0, rowCount: 0};
  }
  let low = 0;
  let high = rowCount;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (isBeforeStart(middle)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const firstRow = low;
  high = rowCount;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (isAfterEnd(middle)) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return {firstRow, rowCount: low - firstRow};
}

/** Binary-search range of float rows. */
export function searchFloatRange(rows: SortedFloatRows, window: SortedFloatWindow): SortedRange {
  const low = (row: number) => rows.low?.[row] ?? 0;
  return searchRange(
    rows.high.length,
    isFloatTimeAtLeast(window.endHigh, window.endLow, window.startHigh, window.startLow),
    row => !isFloatTimeAtLeast(rows.high[row], low(row), window.startHigh, window.startLow),
    row => !isFloatTimeAtLeast(window.endHigh, window.endLow, rows.high[row], low(row))
  );
}

/** Binary-search range of word rows. */
export function searchWordRange(times: ArrayLike<bigint>, window: SortedWordWindow): SortedRange {
  return searchRange(
    times.length,
    isWordTimeAtLeast(window.end, window.endFraction, window.start, window.startFraction),
    row => !isWordTimeAtLeast(times[row], 0, window.start, window.startFraction),
    row => !isWordTimeAtLeast(window.end, window.endFraction, times[row], 0)
  );
}

/**
 * Canonical form of a range for comparison: an empty range has no meaningful `firstRow` (the
 * kernel reports the insertion point), so it normalizes to row 0.
 */
export function normalizeRange(range: SortedRange): SortedRange {
  return range.rowCount === 0 ? {firstRow: 0, rowCount: 0} : range;
}
