// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeClassTable} from '../../cartography/class-table';
import {hexToRgba, MAP_INK} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';

/**
 * Symbolisation of the rate-smoothing scene that is not a shared table: the septile table of the
 * birth rate, the standardised-rate (z) table, the extreme-county rings and the selection ink.
 * Pure data: no luma.gl, so the scene file can import it.
 */

/** The ground a table or line is drawn on. */
export type Ground = 'light' | 'dark';

/** Rates are stored per woman-year in GPU buffers; the maps and legends show them per 1,000. */
export const RATE_SCALE = 1000;

/** Normalisation basis of every rate legend and tooltip row. */
export const RATE_BASIS = 'per 1,000 women aged 15-44 per year';

/** Class count of the rate maps (ColorBrewer YlOrBr-7). */
export const RATE_CLASS_COUNT = 7;

/** Breaks of the standardised-rate (z) table: the 1, 1.96 and 2.58 standard-deviation steps. */
export const Z_BREAKS: readonly number[] = [-2.58, -1.96, -1, 1, 1.96, 2.58];

/** Labels of the seven z classes, low first. The middle class is "as expected". */
export const Z_LABELS: readonly string[] = [
  'below -2.58',
  '-2.58 to -1.96',
  '-1.96 to -1',
  '-1 to 1 (as expected)',
  '1 to 1.96',
  '1.96 to 2.58',
  'above 2.58'
];

/** Two-sided normal limits of the funnel plot: 95 % and 99.8 %. */
export const FUNNEL_LIMITS: readonly [number, number] = [1.96, 3.09];

/** Share of counties ringed at each end of the raw rates. */
export const EXTREME_TAIL = 0.05;

/** Rural-small counties of the spread readout: fewer residents than this. */
export const SMALL_COUNTY_RESIDENTS = 5000;

/** Counties with fewer births than this (over three years) are "few births" counties. */
export const FEW_BIRTHS = 30;

/** Alpha multipliers of value-by-alpha: a county the data hardly speaks for keeps a quarter. */
export const FADE_ALPHA_OUTPUT: readonly [number, number] = [0.25, 1];

/** Alpha multipliers with value-by-alpha off: every county fully opaque. */
export const NO_FADE_ALPHA_OUTPUT: readonly [number, number] = [1, 1];

/**
 * Septile breaks of the empirical-Bayes rate, rounded to 0.1 so the layer and the legend labels
 * agree. Computed once and then frozen across the raw, empirical-Bayes and spatial maps and the
 * swipe, so a change of colour is a change of estimate.
 */
export function getRateBreaks(smoothedPer1000: ArrayLike<number>): number[] {
  const sorted = Float64Array.from(smoothedPer1000).sort();
  const breaks: number[] = [];
  for (let index = 1; index < RATE_CLASS_COUNT; index++) {
    const position = (index / RATE_CLASS_COUNT) * (sorted.length - 1);
    const lower = Math.floor(position);
    const upper = Math.min(lower + 1, sorted.length - 1);
    const value = sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    breaks.push(Math.round(value * 10) / 10);
  }
  return breaks;
}

/** The one class table of every rate map: YlOrBr-7 over the shared septile breaks. */
export function getRateTable(
  breaks: readonly number[],
  extent: readonly [number, number],
  ground: Ground
): ClassTable {
  return makeClassTable({
    breaks,
    scheme: 'YlOrBr',
    ground,
    unit: 'births',
    extent,
    method: 'Septiles of the empirical-Bayes rate, shared by every rate map',
    noData: {label: 'No data'},
    format: value => value.toFixed(1)
  });
}

/** The standardised-rate table: RdBu-7, red above the pooled rate, breaks at the z steps. */
export function getZTable(ground: Ground): ClassTable {
  return makeClassTable({
    breaks: Z_BREAKS,
    scheme: 'RdBu',
    ground,
    labels: Z_LABELS,
    unit: 'standard deviations',
    method: 'Assuncao-Reis z: how many standard deviations from the pooled rate',
    noData: {label: 'No data'}
  });
}

/** Ink of the extreme rings and the selection: achromatic, from the ground. */
export function getInkColor(ground: Ground, alpha = 255): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].ink, alpha);
  return [color[0], color[1], color[2], color[3]];
}

/** Ground-coloured casing under rings and selections. */
export function getCasingColor(ground: Ground, alpha = 235): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].halo, alpha);
  return [color[0], color[1], color[2], color[3]];
}

/** Ghost colour of the raw dots behind the smoothed dots of the funnel plot. */
export function getGhostColor(ground: Ground): PaletteColor {
  return ground === 'dark' ? [138, 148, 163, 90] : [138, 148, 158, 110];
}

/** Funnel limit curves: the ink, a touch lighter than the selection. */
export function getFunnelCurveColor(ground: Ground): PaletteColor {
  return getInkColor(ground, 255);
}

/** Compact number for chart axes: 1.2k, 340k, 3.4M. */
export function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return '';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(magnitude >= 1e7 ? 0 : 1)}M`;
  if (magnitude >= 1e3) return `${(value / 1e3).toFixed(magnitude >= 1e4 ? 0 : 1)}k`;
  return value.toFixed(0);
}
