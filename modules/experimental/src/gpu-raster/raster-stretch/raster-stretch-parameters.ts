// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateParameterTarget} from '../raster-algebra/raster-algebra-utils';

/** float32 elements in a {@link GPURasterStretch} parameter view. */
export const GPU_RASTER_STRETCH_PARAMETER_LENGTH = 16;

/** Largest raster width or height supported (cell windows are stored as float32 integers). */
export const GPU_RASTER_STRETCH_MAXIMUM_EXTENT = 1 << 24;

/** float32 rows of the optional `statistics` output of {@link GPURasterStretch}. */
export const GPU_RASTER_STRETCH_STATISTICS_LENGTH = 8;

/**
 * Row indices of the `statistics` output: `[domainMin, domainMax, lo, hi, validCount, binWidth, 0, 0]`.
 *
 * With an automatic domain and no included cell, `domainMin`, `domainMax`, `lo`, `hi` and
 * `binWidth` are NaN and `validCount` is 0. `validCount` is float32, so it is exact up to 2^24.
 */
export const GPU_RASTER_STRETCH_STATISTICS_INDEX = {
  domainMin: 0,
  domainMax: 1,
  lo: 2,
  hi: 3,
  validCount: 4,
  binWidth: 5
} as const;

/** Stretch modes of {@link GPURasterStretch}. */
export type GPURasterStretchMode = 'linear' | 'percentile' | 'equalize';

/** Per-frame settings of {@link GPURasterStretch}. Changing them never recompiles. */
export type GPURasterStretchSettings = {
  /**
   * Statistics window `[column0, row0, column1, row1)` in cells; clamped to the raster. Defaults to
   * the whole raster. Only statistics (min/max, histogram) use it: the apply step runs on all cells.
   */
  window?: readonly [number, number, number, number];
  /**
   * `'auto'` (default) takes the exact minimum and maximum of the included valid cells, or an
   * explicit `[min, max]` with `min <= max`. Cells outside an explicit domain are not counted in
   * the histogram and clamp to 0 or 1 when applied.
   */
  domain?: 'auto' | readonly [number, number];
  /**
   * `'linear'` (default) maps the domain to `[0, 1]`, `'percentile'` maps the `percentiles` of the
   * histogram, `'equalize'` maps through the histogram CDF.
   */
  mode?: GPURasterStretchMode;
  /** Percentile mode: `[low, high]` in percent, `0 <= low <= high <= 100`. Defaults to `[2, 98]`. */
  percentiles?: readonly [number, number];
  /** Applies `t' = t ** gamma` after the mode. Positive, defaults to 1. */
  gamma?: number;
  /**
   * Sigmoidal contrast (rio-color / ImageMagick style) applied after gamma, normalised so 0 maps to
   * 0 and 1 to 1. Zero or undefined disables it; otherwise `(0, 80]`.
   */
  sigmoidContrast?: number;
  /** Sigmoid midpoint in `(0, 1)`. Defaults to 0.5. */
  sigmoidMidpoint?: number;
  /** Palette lookup: `'nearest'` (default) or `'linear'` channel interpolation. */
  paletteInterpolation?: 'nearest' | 'linear';
};

function getWindowValue(value: number, name: string): number {
  if (!Number.isFinite(value)) {
    throw new Error(`Raster stretch window ${name} must be finite`);
  }
  return Math.min(Math.max(Math.floor(value), 0), GPU_RASTER_STRETCH_MAXIMUM_EXTENT);
}

/**
 * Packs per-frame {@link GPURasterStretch} parameters.
 *
 * Layout (float32): `[column0, row0, column1, row1, explicitDomain, domainMin, domainMax, mode,
 * percentileLow, percentileHigh, gamma, sigmoidContrast, sigmoidMidpoint, linearPalette, 0, 0]`
 * with `mode` 0 (linear), 1 (percentile), 2 (equalize) and percentiles as fractions. Window values
 * are floored and clamped to `[0, 2^24]` (the kernel clamps them to the raster).
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least {@link GPU_RASTER_STRETCH_PARAMETER_LENGTH}.
 * @throws If the target is short or any setting is out of range.
 */
export function getGPURasterStretchParameterValues(
  settings: GPURasterStretchSettings = {},
  target: Float32Array = new Float32Array(GPU_RASTER_STRETCH_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget('Raster stretch', target, GPU_RASTER_STRETCH_PARAMETER_LENGTH);
  target.fill(0, 0, GPU_RASTER_STRETCH_PARAMETER_LENGTH);
  const window = settings.window ?? [
    0,
    0,
    GPU_RASTER_STRETCH_MAXIMUM_EXTENT,
    GPU_RASTER_STRETCH_MAXIMUM_EXTENT
  ];
  target[0] = getWindowValue(window[0], 'column0');
  target[1] = getWindowValue(window[1], 'row0');
  target[2] = getWindowValue(window[2], 'column1');
  target[3] = getWindowValue(window[3], 'row1');
  const domain = settings.domain ?? 'auto';
  if (domain !== 'auto') {
    const [minimum, maximum] = domain;
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum) {
      throw new Error('Raster stretch domain must be finite with min <= max');
    }
    if (!Number.isFinite(Math.fround(maximum - minimum)) && Number.isFinite(maximum - minimum)) {
      throw new Error('Raster stretch domain range must fit in float32');
    }
    target[4] = 1;
    target[5] = minimum;
    target[6] = maximum;
  }
  const mode = settings.mode ?? 'linear';
  if (mode !== 'linear' && mode !== 'percentile' && mode !== 'equalize') {
    throw new Error(`Raster stretch mode ${String(mode)} is not supported`);
  }
  target[7] = mode === 'linear' ? 0 : mode === 'percentile' ? 1 : 2;
  const [percentileLow, percentileHigh] = settings.percentiles ?? [2, 98];
  if (
    !Number.isFinite(percentileLow) ||
    !Number.isFinite(percentileHigh) ||
    percentileLow < 0 ||
    percentileHigh > 100 ||
    percentileLow > percentileHigh
  ) {
    throw new Error('Raster stretch percentiles must satisfy 0 <= low <= high <= 100');
  }
  target[8] = percentileLow / 100;
  target[9] = percentileHigh / 100;
  const gamma = settings.gamma ?? 1;
  if (!Number.isFinite(gamma) || gamma <= 0) {
    throw new Error('Raster stretch gamma must be positive and finite');
  }
  target[10] = gamma;
  const contrast = settings.sigmoidContrast ?? 0;
  if (!Number.isFinite(contrast) || contrast < 0 || contrast > 80) {
    throw new Error('Raster stretch sigmoidContrast must be in [0, 80]');
  }
  target[11] = contrast;
  const midpoint = settings.sigmoidMidpoint ?? 0.5;
  if (!Number.isFinite(midpoint) || midpoint <= 0 || midpoint >= 1) {
    throw new Error('Raster stretch sigmoidMidpoint must be in (0, 1)');
  }
  target[12] = midpoint;
  const interpolation = settings.paletteInterpolation ?? 'nearest';
  if (interpolation !== 'nearest' && interpolation !== 'linear') {
    throw new Error(
      `Raster stretch paletteInterpolation ${String(interpolation)} is not supported`
    );
  }
  target[13] = interpolation === 'linear' ? 1 : 0;
  return target;
}
