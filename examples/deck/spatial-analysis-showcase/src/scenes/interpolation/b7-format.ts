// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const BLOCKS = '▁▂▃▄▅▆▇█';

/**
 * Renders values as a one-line bar chart with unicode block characters (NaN shows a dot), scaled
 * to `maximum`. Readouts accept strings, so this is the showcase's tiny inline chart.
 */
export function formatSparkline(values: ArrayLike<number>, maximum: number): string {
  let text = '';
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      text += '·';
      continue;
    }
    const level = Math.round((Math.max(0, value) / Math.max(maximum, 1e-20)) * (BLOCKS.length - 1));
    text += BLOCKS[Math.min(BLOCKS.length - 1, level)];
  }
  return text;
}

/** Formats a number with a sensible number of digits for a readout. */
export function formatSignificant(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return 'n/a';
  const magnitude = Math.abs(value);
  if (magnitude >= 10 ** digits) return Math.round(value).toLocaleString('en-US');
  if (magnitude === 0) return '0';
  return Number(value.toPrecision(digits)).toLocaleString('en-US', {maximumFractionDigits: 6});
}

/** Deterministic 32-bit integer hash. */
export function hashInteger(value: number): number {
  let hash = Math.imul(value ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
}
