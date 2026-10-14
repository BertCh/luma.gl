// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU oracle of the range search of `GPUTemporalReduction` with `sorted: true`: the contiguous
 * rows whose bucket lands in `[0, bucketCount)`, found by the same two binary searches as the
 * kernel with the same prefix and suffix tests (see `temporal-reduction-sorted-range.ts`).
 */

/** Contiguous rows to reduce. */
export type TemporalRange = {firstRow: number; rowCount: number};

function searchRange(
  rowCount: number,
  isValid: boolean,
  isBelow: (row: number) => boolean,
  isAbove: (row: number) => boolean
): TemporalRange {
  if (!isValid) {
    return {firstRow: 0, rowCount: 0};
  }
  let low = 0;
  let high = rowCount;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (isBelow(middle)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const firstRow = low;
  high = rowCount;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (isAbove(middle)) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return {firstRow, rowCount: low - firstRow};
}

/** Range of float32 times: below when `fround(t - origin) < 0`, above when `>= fround(B * w)`. */
export function searchFloatTemporalRange(
  times: Float32Array,
  bucketCount: number,
  origin: number,
  width: number
): TemporalRange {
  const originValue = Math.fround(origin);
  const widthValue = Math.fround(width);
  const isValid = !Number.isNaN(originValue) && widthValue > 0 && Number.isFinite(widthValue);
  const limit = Math.fround(bucketCount * widthValue);
  return searchRange(
    times.length,
    isValid,
    row => Math.fround(times[row] - originValue) < 0,
    row => !(Math.fround(times[row] - originValue) < limit)
  );
}

/** Range of Int64 times: below when `t < origin`, above when `(t - origin) / width >= B`. */
export function searchWordTemporalRange(
  times: BigInt64Array,
  bucketCount: number,
  origin: bigint,
  width: bigint
): TemporalRange {
  return searchRange(
    times.length,
    width !== 0n,
    row => times[row] < origin,
    row => (times[row] - origin) / width >= BigInt(bucketCount)
  );
}
