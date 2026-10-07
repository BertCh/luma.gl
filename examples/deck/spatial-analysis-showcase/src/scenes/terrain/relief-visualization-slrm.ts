// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The reference scale of the local relief step, computed once on the CPU: the 98th percentile of
 * the absolute simple local relief `|z - mean_r(z)|` at one window radius. The step freezes class
 * breaks from it, so every window radius is drawn against the same limits (honest stretch).
 * Cells whose window leaves the raster are skipped, as the GPU edge clamp would repeat edge cells.
 */

const HISTOGRAM_BINS = 4096;

/**
 * The 98th percentile of `|z - box mean|` over the cells whose `(2r + 1)` square window lies fully
 * inside the raster. Two separable running means, then a histogram: O(cells), no sort.
 *
 * @param values Elevation in metres, row-major, `width * height`; non-finite cells are skipped.
 * @returns The percentile in metres, or `NaN` when no window fits.
 */
export function getLocalReliefPercentile(
  values: ArrayLike<number>,
  width: number,
  height: number,
  radius: number,
  fraction = 0.98
): number {
  const side = 2 * radius + 1;
  if (width < side || height < side) return Number.NaN;
  const columns = width - 2 * radius;
  // Row means over the window, kept at the interior columns only.
  const rowMeans = new Float32Array(columns * height);
  for (let row = 0; row < height; row++) {
    let sum = 0;
    for (let column = 0; column < side; column++) sum += values[row * width + column];
    rowMeans[row * columns] = sum / side;
    for (let column = 1; column < columns; column++) {
      sum += values[row * width + column + side - 1] - values[row * width + column - 1];
      rowMeans[row * columns + column] = sum / side;
    }
  }
  const rows = height - 2 * radius;
  const relief = new Float32Array(columns * rows);
  for (let column = 0; column < columns; column++) {
    let sum = 0;
    for (let row = 0; row < side; row++) sum += rowMeans[row * columns + column];
    for (let row = 0; row < rows; row++) {
      if (row > 0) {
        sum +=
          rowMeans[(row + side - 1) * columns + column] - rowMeans[(row - 1) * columns + column];
      }
      const center = values[(row + radius) * width + column + radius];
      relief[row * columns + column] = Math.abs(center - sum / side);
    }
  }
  let maximum = 0;
  for (const value of relief) if (Number.isFinite(value) && value > maximum) maximum = value;
  if (maximum === 0) return 0;
  const histogram = new Uint32Array(HISTOGRAM_BINS);
  let count = 0;
  for (const value of relief) {
    if (!Number.isFinite(value)) continue;
    histogram[Math.min(HISTOGRAM_BINS - 1, Math.floor((value / maximum) * HISTOGRAM_BINS))]++;
    count++;
  }
  const target = fraction * count;
  let seen = 0;
  for (let bin = 0; bin < HISTOGRAM_BINS; bin++) {
    seen += histogram[bin];
    if (seen >= target) return ((bin + 1) / HISTOGRAM_BINS) * maximum;
  }
  return maximum;
}
