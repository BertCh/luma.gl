// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Float32 parameter elements per column: `[lo, hi]` of the histogram domain. */
export const GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN = 2;

/** Largest compile-time column count. */
export const GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT = 16;

/** Largest compile-time top-category count. */
export const GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT = 64;

/** Code of a missing value in a category column. */
export const GPU_COLUMN_PROFILE_NULL_CATEGORY = 0xffffffff;

/**
 * Field offsets inside one column's block of `output.statistics`: `statistics[column * STATISTIC_COUNT + field]`.
 *
 * - `count`: masked-in rows that are not null. Numeric: not NaN (`+-Infinity` count). Category: code is not `0xffffffff`.
 * - `nullCount`: masked-in NaN rows (numeric) or `0xffffffff` codes (category).
 * - `minimum`, `maximum`: exact over the counted rows, `+-Infinity` included, `-0 < +0`. Category columns report the smallest and largest code. NaN when `count` is 0.
 * - `sum`, `mean`, `variance`, `sampleVariance`, `standardDeviation`: over the finite rows only
 *   (`+-Infinity` is excluded from the moments). `variance` is the population variance `M2 / n`,
 *   `sampleVariance` is `M2 / (n - 1)` (NaN for `n < 2`). NaN when there is no finite row, and for
 *   every category column.
 * - `distinctEstimate`: HyperLogLog estimate of the number of distinct counted values; 0 when `count` is 0.
 * - `overflowCount`: category columns only, counted rows with a code `>= categoryCount`; 0 for numeric columns.
 */
export const GPU_COLUMN_PROFILE_STATISTIC = {
  count: 0,
  nullCount: 1,
  minimum: 2,
  maximum: 3,
  sum: 4,
  mean: 5,
  variance: 6,
  sampleVariance: 7,
  standardDeviation: 8,
  distinctEstimate: 9,
  overflowCount: 10
} as const;

/** Float32 values per column in `output.statistics`. */
export const GPU_COLUMN_PROFILE_STATISTIC_COUNT = 11;

/** Number of float32 elements of a histogram-domain parameter view for `columnCount` columns. */
export function getGPUColumnProfileParameterLength(columnCount: number): number {
  return columnCount * GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN;
}

/** Histogram domain of one column. A NaN or missing bound means "automatic". */
export type GPUColumnProfileDomain = {lo?: number; hi?: number} | readonly [number, number];

/**
 * Packs per-frame histogram domains of `GPUColumnProfile`.
 *
 * Layout (float32): `[lo_0, hi_0, lo_1, hi_1, ...]`, column-major. A NaN (or omitted) `lo` or `hi`
 * means automatic: the minimum (or maximum) of the finite values counted in the same frame. A
 * domain with a non-finite or inverted pair after resolution gives an all-zero histogram, and
 * `lo == hi` counts the values equal to it in bin 0.
 *
 * @param domains One entry per column, `undefined` for fully automatic.
 * @param columnCount Number of columns. Defaults to `domains.length`.
 * @param target Optional destination of at least `columnCount * 2` elements.
 */
export function getGPUColumnProfileParameterValues(
  domains: readonly (GPUColumnProfileDomain | undefined)[],
  columnCount: number = domains.length,
  target?: Float32Array
): Float32Array {
  if (!Number.isInteger(columnCount) || columnCount < 1) {
    throw new Error('Column profile column count must be a positive integer');
  }
  if (domains.length > columnCount) {
    throw new Error('Column profile domains exceed columnCount');
  }
  const length = getGPUColumnProfileParameterLength(columnCount);
  const values = target ?? new Float32Array(length);
  if (values.length < length) {
    throw new Error(`Column profile parameter target must hold ${length} elements`);
  }
  for (let column = 0; column < columnCount; column++) {
    const domain = domains[column];
    let lo = NaN;
    let hi = NaN;
    if (domain) {
      if (Array.isArray(domain)) {
        [lo, hi] = domain as readonly [number, number];
      } else {
        lo = (domain as {lo?: number}).lo ?? NaN;
        hi = (domain as {hi?: number}).hi ?? NaN;
      }
    }
    values[column * GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN] = lo;
    values[column * GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN + 1] = hi;
  }
  return values;
}
