// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Returns the sorted finite values of a column (a copy). */
export function getSortedFinite(values: ArrayLike<number>): Float64Array {
  const finite: number[] = [];
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) finite.push(values[index]);
  }
  return Float64Array.from(finite).sort();
}

/** Linear-interpolated quantile `q` in `[0, 1]` of a sorted array; NaN when it is empty. */
export function getSortedQuantile(sorted: ArrayLike<number>, q: number): number {
  if (sorted.length === 0) return Number.NaN;
  const position = Math.min(Math.max(q, 0), 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(lower + 1, sorted.length - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}
