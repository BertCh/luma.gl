// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Proportional (graduated) symbol sizing. Pure TypeScript; mirrors the shared GPU point layer.
 */

/** `sqrt` makes disc area proportional to value (default); `linear` makes the radius proportional. */
export type ProportionalScale = 'sqrt' | 'linear';

/** Rounds a positive value onto the 1-2-5 series (`floor`, `round` to nearest in log space, or `ceil`). Returns 0 for values <= 0. */
export function getNiceValue(value: number, mode: 'floor' | 'round' | 'ceil'): number {
  if (!(value > 0) || !Number.isFinite(value)) return 0;
  const exponent = Math.floor(Math.log10(value));
  const magnitude = 10 ** exponent;
  const fraction = value / magnitude;
  let nice: number;
  if (mode === 'floor') nice = fraction >= 5 ? 5 : fraction >= 2 ? 2 : 1;
  else if (mode === 'ceil') nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  else
    nice =
      fraction < Math.SQRT2 ? 1 : fraction < Math.sqrt(10) ? 2 : fraction < Math.sqrt(50) ? 5 : 10;
  return Number((nice * magnitude).toPrecision(12));
}

/**
 * Disc radius in pixels: `max(minRadiusPixels, maxRadiusPixels * sqrt(clamp(value / maxValue, 0, 1)))`
 * for `sqrt` scale, or without the square root for `linear`. Identical to the GPU layer's sizing.
 */
export function getProportionalRadius(
  value: number,
  maxValue: number,
  maxRadiusPixels: number,
  options: {minRadiusPixels?: number; scale?: ProportionalScale} = {}
): number {
  const {minRadiusPixels = 0, scale = 'sqrt'} = options;
  const ratio = maxValue > 0 ? Math.min(1, Math.max(0, value / maxValue)) : 0;
  const shaped = scale === 'linear' ? ratio : Math.sqrt(ratio);
  return Math.max(minRadiusPixels, maxRadiusPixels * shaped);
}

/**
 * Entries for a nested size legend, largest first: the maximum rounded down to 1/2/5 x 10^k, then
 * roughly halves and quarters rounded to nice numbers (duplicates removed, so fewer than `count`
 * entries are possible for small ranges).
 */
export function getSizeLegendEntries(
  maxValue: number,
  maxRadiusPixels: number,
  options: {
    count?: number;
    minRadiusPixels?: number;
    scale?: ProportionalScale;
    format?: (value: number) => string;
  } = {}
): {radiusPixels: number; label: string}[] {
  const {
    count = 3,
    minRadiusPixels,
    scale,
    format = (value: number) => value.toLocaleString('en-US', {maximumFractionDigits: 2})
  } = options;
  const top = getNiceValue(maxValue, 'floor');
  const entries: {radiusPixels: number; label: string}[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < count; i++) {
    const value = i === 0 ? top : getNiceValue(top * 0.5 ** i, 'round');
    if (!(value > 0) || seen.has(value)) continue;
    seen.add(value);
    entries.push({
      radiusPixels: getProportionalRadius(value, maxValue, maxRadiusPixels, {
        minRadiusPixels,
        scale
      }),
      label: format(value)
    });
  }
  return entries;
}
