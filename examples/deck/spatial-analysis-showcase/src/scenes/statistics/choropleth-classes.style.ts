// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassBreaks, getExtent} from '../../cartography/breaks';
import {type ClassSchemeName, getClassPalette, makeClassTable} from '../../cartography/class-table';
import {NO_DATA_COLOR} from '../../cartography/hue-registry';
import type {ClassColor, ClassTable} from '../../cartography/types';
import {BIVARIATE_PALETTES, type PaletteColor} from '../../engine/ramps';
import {packColor} from './b5-palettes';

/**
 * Symbolisation of the choropleth-classes scene that is not a shared table: which published
 * ColorBrewer scheme each variable and method uses, the frozen baseline classification of the
 * swipe compare, number formats and the pure statistics the readouts need. Pure data: no luma.gl.
 */

/** The ground a table or line is drawn on. */
export type Ground = 'light' | 'dark';

/** Display names of the classification methods. */
export const METHOD_LABELS = {
  quantile: 'Quantile',
  'equal-interval': 'Equal interval',
  'standard-deviation': 'Standard deviation',
  'head-tail': 'Head/tail breaks',
  'box-plot': 'Box plot',
  'maximum-breaks': 'Maximum breaks',
  'natural-breaks': 'Natural breaks (Jenks)',
  custom: 'Round-number edges'
} as const;

/** Variables whose distribution is shown on a log axis: heavy right tails of positive values. */
export const LOG_VARIABLES: ReadonlySet<string> = new Set(['popDensity', 'population']);

/**
 * The ColorBrewer scheme of a variable, one hue per question (SYNTHESIS 3.5): people (density
 * and counts) YlOrBr, income PuBu, health burden RdPu, every other indicator Blues.
 */
export function getVariableScheme(variableId: string): ClassSchemeName {
  switch (variableId) {
    case 'popDensity':
    case 'population':
      return 'YlOrBr';
    case 'medianHouseholdIncome':
      return 'PuBu';
    case 'places_diabetes_ageAdj':
    case 'places_obesity_ageAdj':
      return 'RdPu';
    default:
      return 'Blues';
  }
}

/**
 * The scheme of a class table: the variable's sequential hue, or a diverging scheme when the
 * method has a real midpoint (standard-deviation classes about the mean: RdBu, red above; box
 * plot about the median: PuOr, orange above).
 */
export function getMethodScheme(variableId: string, method: string): ClassSchemeName {
  if (method === 'standard-deviation') return 'RdBu';
  if (method === 'box-plot') return 'PuOr';
  return getVariableScheme(variableId);
}

/** True when the scheme is diverging and its classes carry a midpoint. */
export function isDivergingMethod(method: string): boolean {
  return method === 'standard-deviation' || method === 'box-plot';
}

/** The published `classCount`-class table of a scheme on a ground. */
export function getSchemeColors(
  scheme: ClassSchemeName,
  classCount: number,
  ground: Ground
): PaletteColor[] {
  return classCount < 1 ? [] : getClassPalette(scheme, classCount, {ground});
}

/** Packs RGBA colours into the rgba8 layout of the classification contributors. */
export function packColors(colors: readonly ClassColor[]): Uint32Array {
  return Uint32Array.from(colors, color =>
    packColor(color[0], color[1], color[2], color[3] ?? 255)
  );
}

/** The packed no-data colour: flat grey on every ground, never transparent. */
export function getPackedNoDataColor(ground: Ground): number {
  const color = NO_DATA_COLOR[ground];
  return packColor(color[0], color[1], color[2], color[3]);
}

/** The 3 x 3 bivariate palette of a ground (Stevens teal-pink on paper, a proposal on dark). */
export function getBivariatePalette(ground: Ground): readonly PaletteColor[] {
  return ground === 'dark' ? BIVARIATE_PALETTES.darkGround : BIVARIATE_PALETTES.tealPink;
}

/** Breaks and unit text of a value: thousands separators, precision by magnitude. */
export function formatClassValue(value: number): string {
  if (!Number.isFinite(value)) return '–';
  const magnitude = Math.abs(value);
  const digits = magnitude >= 100 ? 0 : magnitude >= 10 ? 1 : magnitude >= 1 ? 2 : 3;
  return value.toLocaleString('en-US', {maximumFractionDigits: digits});
}

/** Whether a legend entry names the percentile cut or plain missing data. */
export function getNoDataLabel(lowerPercentile: number, upperPercentile: number): string {
  return lowerPercentile > 0 || upperPercentile < 100
    ? 'Excluded by the percentile cut'
    : 'No data';
}

/** Methods the frozen swipe baseline can use. */
export type BaselineMethod = 'equal-interval' | 'quantile';

/**
 * The frozen side-a classification of a swipe compare: breaks computed once from the finite
 * values with the CPU reference of `cartography/breaks`, in the same ColorBrewer table as the
 * live side (or in `scheme` when the comparison is between two hues).
 */
export function makeBaselineTable(options: {
  values: ArrayLike<number>;
  method: BaselineMethod;
  classCount: number;
  scheme: ClassSchemeName;
  ground: Ground;
  unit: string;
  noDataLabel: string;
}): ClassTable {
  const breaks = getClassBreaks(options.values, options.classCount, options.method);
  return makeClassTable({
    breaks,
    scheme: options.scheme,
    ground: options.ground,
    unit: options.unit,
    extent: getExtent(options.values),
    format: formatClassValue,
    method: METHOD_LABELS[options.method],
    noData: {label: options.noDataLabel}
  });
}

/** Median of an ascending array. */
function getSortedMedian(sorted: ArrayLike<number>, start = 0, end = sorted.length): number {
  const count = end - start;
  if (count <= 0) return 0;
  const middle = start + (count >> 1);
  return count % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Goodness of absolute deviation fit of a classification, `1 - ADCM / ADAM` (mapclassify
 * `gadf`): absolute deviations from the class medians over those from the overall median.
 * `sorted` is the ascending finite values; `breaks` the interior class breaks.
 */
export function getGoodnessOfAbsoluteDeviationFit(
  sorted: ArrayLike<number>,
  breaks: readonly number[]
): number {
  const total = sorted.length;
  if (total === 0) return Number.NaN;
  const overallMedian = getSortedMedian(sorted);
  let overall = 0;
  for (let index = 0; index < total; index++) overall += Math.abs(sorted[index] - overallMedian);
  if (overall === 0) return 1;
  // Class members are contiguous in an ascending array: walk the breaks once.
  let within = 0;
  let start = 0;
  for (let classIndex = 0; classIndex <= breaks.length; classIndex++) {
    let end = start;
    while (end < total && (classIndex === breaks.length || sorted[end] < breaks[classIndex])) end++;
    const median = getSortedMedian(sorted, start, end);
    for (let index = start; index < end; index++) within += Math.abs(sorted[index] - median);
    start = end;
  }
  return 1 - within / overall;
}

/** Share (0-1) of the ascending `sorted` values at or below `value`, by binary search. */
export function getRankShare(sorted: ArrayLike<number>, value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return sorted.length ? low / sorted.length : Number.NaN;
}

/**
 * How the equal-width bins of natural breaks are used: the number of bins that hold at least one
 * value and the share of values in the first bin, over `[low, high]` (the filtered range).
 */
export function getBinUsage(
  sorted: ArrayLike<number>,
  low: number,
  high: number,
  binCount: number
): {used: number; firstShare: number} {
  const span = high - low;
  if (!(span > 0) || sorted.length === 0) return {used: sorted.length ? 1 : 0, firstShare: 1};
  const occupied = new Uint8Array(binCount);
  let inRange = 0;
  let first = 0;
  for (let index = 0; index < sorted.length; index++) {
    const value = sorted[index];
    if (value < low || value > high) continue;
    const bin = Math.min(binCount - 1, Math.floor(((value - low) / span) * binCount));
    occupied[bin] = 1;
    if (bin === 0) first++;
    inRange++;
  }
  let used = 0;
  for (let bin = 0; bin < binCount; bin++) used += occupied[bin];
  return {used, firstShare: inRange ? first / inRange : 0};
}
