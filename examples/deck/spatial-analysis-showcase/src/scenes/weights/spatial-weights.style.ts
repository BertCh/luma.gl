// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassBreaks} from '../../cartography/breaks';
import {getClassPalette, getDivergingBreaks, makeClassTable} from '../../cartography/class-table';
import {hexToRgba, MAP_INK} from '../../cartography/hue-registry';
import type {ClassSchemeName} from '../../cartography/class-table';
import type {ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';
import type {SpatialWeightsDisplay, SpatialWeightsOptions} from './spatial-weights.compute';

/**
 * Symbolisation of the spatial-weights scene. Hue answers the question the map asks: BuGn is
 * "membership in W" (neighbour, weight, cardinality), YlOrBr is the variable and its lag, PuOr is
 * the deviation of the neighbourhood from the place (orange = the neighbourhood is higher). Every
 * class table here is the one object the layer, the legend, the tooltip and the histogram read.
 * Pure data: no luma.gl.
 */

/** The ground a table or line is drawn on. */
export type Ground = 'light' | 'dark';

const TRANSPARENT: PaletteColor = [0, 0, 0, 0];

/** Fill opacity of the neighbour classes (a little under 1 so the hairlines read through). */
const MEMBER_ALPHA = 230;

/**
 * The display a state really draws. The lattice has no focus-row weights and no lag difference
 * (its fills read the neighbour classes, the variable and the lag only).
 */
export function getEffectiveDisplay(
  state: Pick<SpatialWeightsOptions, 'lattice' | 'display'>
): SpatialWeightsDisplay {
  if (state.lattice) {
    if (state.display === 'weights') return 'focus';
    if (state.display === 'difference') return 'lag';
  }
  return state.display;
}

/** Frame of the contiguous United States (the shared national camera, west, south, east, north). */
export const CONUS_BOUNDS = [-124.8, 24.4, -66.9, 49.4] as const;

/** Frame of the Mountain West, where the counties are few and large (west, south, east, north). */
export const MOUNTAIN_WEST_BOUNDS = [-116.5, 31.5, -101.5, 49] as const;

/** Interior breaks of the neighbour-count classes: `0 | 1-3 | 4-5 | 6 | 7-8 | 9+`. */
export const CARDINALITY_BREAKS: readonly number[] = [1, 4, 6, 7, 9];

/** Largest neighbour count the cardinality chart lists. */
export const CARDINALITY_CHART_LIMIT = 14;

/** Interior breaks of the one-way classes: `0 | 1 | 2 | 3+`. */
export const ONE_WAY_BREAKS: readonly number[] = [1, 2, 3];

/** Interior breaks of the kernel-weight classes, as shares of the largest weight of the row. */
export const WEIGHT_BREAKS: readonly number[] = [0.2, 0.4, 0.6, 0.8];

/**
 * `count` classes of the published BuGn table, low class first. On light grounds the palest class
 * (`#edf8fb`, nearly the paper) is skipped, so the lowest class that is drawn still reads; dark
 * grounds use the authored dark table as it is.
 */
export function getMembershipColors(count: number, ground: Ground): PaletteColor[] {
  return ground === 'dark'
    ? getClassPalette('BuGn', count, {ground})
    : getClassPalette('BuGn', count + 1).slice(1);
}

/** Ink (selection, rings, focus outline) of a ground as RGBA. */
export function getInkColor(ground: Ground, alpha = 255): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].ink, alpha);
  return [color[0], color[1], color[2], color[3]];
}

/** The ground-coloured casing under an ink line (the halo token of the ground) as RGBA. */
export function getCasingColor(ground: Ground, alpha = 235): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].halo, alpha);
  return [color[0], color[1], color[2], color[3]];
}

/** Colour of the "not a neighbour" swatch: the context ink, faint. */
export function getContextSwatch(ground: Ground): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].context, 90);
  return [color[0], color[1], color[2], color[3]];
}

/**
 * Category palette of the focus display (`space.focusClass`): 0 other places (nothing drawn),
 * 1 neighbours of the focus (a mid BuGn class), 2 the focus place (the darkest BuGn class).
 */
export function getFocusPalette(ground: Ground): PaletteColor[] {
  const classes = getMembershipColors(5, ground);
  const neighbour = classes[3];
  const focus = classes[4];
  return [TRANSPARENT, [neighbour[0], neighbour[1], neighbour[2], MEMBER_ALPHA], focus];
}

/** The class table of the neighbour count: islands are their own, unfilled class. */
export function getCardinalityTable(ground: Ground, unit = 'neighbours per place'): ClassTable {
  return makeClassTable({
    breaks: CARDINALITY_BREAKS,
    colors: [TRANSPARENT, ...getMembershipColors(5, ground)],
    transparent: [0],
    labels: ['0 (island)', '1-3', '4-5', '6', '7-8', '9+'],
    unit,
    method: 'Fixed classes, the same for every rule',
    noData: {label: 'No data'}
  });
}

/** The class table of one-way links: the first class is "every link returned", unfilled. */
export function getOneWayTable(ground: Ground): ClassTable {
  return makeClassTable({
    breaks: ONE_WAY_BREAKS,
    colors: [TRANSPARENT, ...getMembershipColors(3, ground)],
    transparent: [0],
    labels: ['0 (every link returned)', '1', '2', '3+'],
    unit: 'one-way links per place',
    method: 'Links a place lists that are not listed back',
    noData: {label: 'No data'}
  });
}

/** The class table of the weights of the focus row, as shares of the row's largest weight. */
export function getWeightTable(ground: Ground): ClassTable {
  return makeClassTable({
    breaks: WEIGHT_BREAKS,
    colors: getMembershipColors(5, ground),
    labels: ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1'],
    unit: 'share of the largest weight',
    method: 'Equal classes of the weight divided by the largest weight of the row',
    noData: {label: 'Not a neighbour'}
  });
}

/**
 * Quantile classes (5, fewer when values tie) of `values` in a sequential scheme. The breaks are
 * computed from the variable once and reused for its lag, so a smoothed lag shows as a class
 * shift on one legend.
 */
export function getQuantileTable(
  values: ArrayLike<number>,
  scheme: ClassSchemeName,
  ground: Ground,
  options: {unit: string; format: (value: number) => string; method: string; noDataLabel: string}
): ClassTable {
  const breaks = getClassBreaks(values, 5, 'quantile');
  return makeClassTable({
    breaks,
    scheme,
    ground,
    unit: options.unit,
    format: options.format,
    method: options.method,
    noData: {label: options.noDataLabel}
  });
}

/** A "nice" step (1, 2, 2.5, 5 times a power of ten) at or above a quarter of `value`. */
export function getNiceStep(value: number): number {
  if (!(value > 0)) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  for (const factor of [1, 2, 2.5, 5, 10]) {
    if (factor * magnitude >= value) return factor * magnitude;
  }
  return 10 * magnitude;
}

/**
 * The seven PuOr classes of "lag minus value": a neutral middle class (|difference| below one
 * step) and three steps either way. Orange means the neighbourhood is higher than the place.
 * The step is a quarter of the spread of the variable, so the breaks do not move when the rule
 * changes.
 */
export function getDifferenceTable(
  spread: number,
  ground: Ground,
  options: {unit: string; format: (value: number) => string}
): ClassTable {
  const step = getNiceStep(spread / 4);
  return makeClassTable({
    breaks: getDivergingBreaks(0, [step, 2 * step, 3 * step]),
    scheme: 'PuOr',
    ground,
    unit: options.unit,
    format: options.format,
    method: `Steps of ${options.format(step)} either side of zero`,
    noData: {label: 'No data or island'}
  });
}

/** Quantile classes of a neighbourhood statistic on the hue that fits what it measures. */
export function getSummaryTable(
  values: ArrayLike<number>,
  scheme: ClassSchemeName,
  ground: Ground,
  options: {unit: string; format: (value: number) => string; title: string}
): ClassTable {
  const breaks = getClassBreaks(values, 5, 'quantile');
  return makeClassTable({
    breaks,
    scheme,
    ground,
    unit: options.unit,
    format: options.format,
    method: `Quantiles of ${options.title.toLowerCase()}`,
    noData: {label: 'No data'}
  });
}

/**
 * Community-area boundary over the tracts (the zone-boundary tier of the city steps): `#3A3F4B`
 * 0.8 px at alpha 140 on light grounds, the light ink on dark.
 */
export function getAreaLineStyle(ground: Ground): {color: PaletteColor; widthPixels: number} {
  return ground === 'dark'
    ? {color: [170, 182, 195, 150], widthPixels: 0.8}
    : {color: [58, 63, 75, 140], widthPixels: 0.8};
}
