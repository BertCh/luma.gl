// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const BLOCKS = '▁▂▃▄▅▆▇█';

/** One-line bar chart of non-negative or signed values scaled between their minimum and maximum. */
export function formatSparkline(values: ArrayLike<number>, floor?: number): string {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  if (floor !== undefined) minimum = Math.min(minimum, floor);
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return '';
  const span = Math.max(maximum - minimum, 1e-20);
  let text = '';
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      text += '·';
      continue;
    }
    text += BLOCKS[Math.max(0, Math.min(7, Math.round(((value - minimum) / span) * 7)))];
  }
  return text;
}

/** Formats a p-value, using `< 0.0001` instead of zero. */
export function formatPValue(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  if (value < 0.0001) return '< 0.0001';
  return value.toFixed(value < 0.01 ? 4 : 3);
}

/** Formats a statistic with a fixed number of digits, or a dash when it is not finite. */
export function formatNumber(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

/** Appends a circle of `segmentCount` segments (x0, y0, x1, y1) to `target`. */
export function appendCircleSegments(
  target: Float32Array,
  offset: number,
  centerX: number,
  centerY: number,
  radius: number,
  segmentCount: number
): number {
  let cursor = offset;
  for (let index = 0; index < segmentCount; index++) {
    const start = (index / segmentCount) * Math.PI * 2;
    const end = ((index + 1) / segmentCount) * Math.PI * 2;
    target[cursor++] = centerX + Math.cos(start) * radius;
    target[cursor++] = centerY + Math.sin(start) * radius;
    target[cursor++] = centerX + Math.cos(end) * radius;
    target[cursor++] = centerY + Math.sin(end) * radius;
  }
  return cursor;
}

/** Percentile of the finite values of an array (nearest rank). */
export function getPercentile(values: ArrayLike<number>, percentile: number): number {
  const finite: number[] = [];
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) finite.push(values[index]);
  }
  if (!finite.length) return 0;
  finite.sort((left, right) => left - right);
  return finite[Math.min(finite.length - 1, Math.floor((percentile / 100) * finite.length))];
}

/**
 * Runs a readback handler and reports its errors. The shared summary reader swallows exceptions
 * thrown inside its callback, which would leave a readout empty without a trace.
 */
export function runGuarded(label: string, action: () => void): void {
  try {
    action();
  } catch (error) {
    // biome-ignore lint/suspicious/noConsole: a failed readback handler must be visible
    console.warn(`${label} failed`, error);
  }
}
