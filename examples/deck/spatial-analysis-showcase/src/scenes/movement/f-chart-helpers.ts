// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {BarChartData, LineChartData} from '../scene';

/** Counts `values` into `bins` equal bins over `[low, high]`; out-of-range values go to the end bins. */
export function binValues(
  values: ArrayLike<number>,
  low: number,
  high: number,
  bins: number,
  count = values.length
): Float64Array {
  const counts = new Float64Array(bins);
  const scale = bins / (high - low);
  for (let index = 0; index < count; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    counts[Math.min(bins - 1, Math.max(0, Math.floor((value - low) * scale)))]++;
  }
  return counts;
}

/** Histogram chart data over `[low, high]` with sensible defaults. */
export function histogramChart(
  counts: ArrayLike<number>,
  low: number,
  high: number,
  options: Partial<BarChartData> & {xLabel: string; yLabel?: string}
): BarChartData {
  return {
    kind: 'histogram',
    values: counts,
    xDomain: [low, high],
    height: 120,
    yLabel: 'count',
    ...options
  };
}

/** Single-series line chart. */
export function lineChart(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  options: Partial<LineChartData> & {xLabel: string; yLabel?: string; label?: string}
): LineChartData {
  const {label, ...rest} = options;
  return {kind: 'line', series: [{label, y, x, area: true}], height: 120, ...rest};
}

/** Sorted-copy quantile (`fraction` 0 to 1) of the first `count` values. */
export function quantile(values: ArrayLike<number>, fraction: number, count = values.length) {
  const sorted = Float64Array.from({length: count}, (_, index) => values[index]).sort();
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

/**
 * Counts positive integers into power-of-two bins (1, 2, 3-4, 5-8, ...) and returns the counts with
 * their labels. Values below 1 are skipped.
 */
export function binLog2(values: ArrayLike<number>, maxBins = 12, count = values.length) {
  const counts = new Float64Array(maxBins);
  for (let index = 0; index < count; index++) {
    const value = values[index];
    if (!(value >= 1)) continue;
    counts[Math.min(maxBins - 1, Math.ceil(Math.log2(value)))]++;
  }
  const labels = Array.from({length: maxBins}, (_, bin) => {
    if (bin === 0) return '1';
    const low = 2 ** (bin - 1) + 1;
    const high = 2 ** bin;
    const text = low === high ? `${low}` : `${low}-${high}`;
    return bin % 2 === 1 || bin === maxBins - 1 ? text : '';
  });
  return {counts, labels};
}
