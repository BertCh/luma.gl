// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassPalette, makeClassTable} from '../../cartography/class-table';
import {getRegistryColors, NO_DATA_COLOR} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import {BIVARIATE_PALETTES, type PaletteColor} from '../../engine/ramps';
import type {GeographyId, VariableId, VariableInfo} from './b4-geography';
import {getGiTable, getMoranQuadrantColors} from './hot-spots.style';

/**
 * Symbolisation of the global-autocorrelation scene: the frozen quantile classes of a variable,
 * the untested Moran quadrant colours, the bivariate key and the join-count pair. The class tables
 * are the single source of the map, the legend, the tooltip and the scatterplot colours. Pure
 * data: no luma.gl.
 */

type Ground = 'light' | 'dark';

/** Variables of the health-burden family (CDC PLACES prevalence). */
const HEALTH_VARIABLES: ReadonlySet<VariableId> = new Set([
  'diabetes',
  'obesity',
  'depression',
  'smoking',
  'inactivity'
]);

/** Number of classes of the value maps (quantiles). */
export const CLASS_COUNT = 5;

/** Alpha of the untested quadrant fills, 0.55 of 255, weaker than the tested clusters of the next story. */
export const QUADRANT_ALPHA = 140;

/** Join-count colours, literal black and white as the names say (a near-black and a near-white on paper). */
export const BLACK_JOIN: PaletteColor = [43, 43, 43, 255];
export const WHITE_JOIN: PaletteColor = [250, 250, 250, 255];

/**
 * Unit text of a variable's legend: counties report age-adjusted PLACES prevalences, tracts the
 * crude ones.
 */
export function getUnitText(info: VariableInfo, geography: GeographyId): string {
  const ageAdjusted = geography === 'us-counties' && HEALTH_VARIABLES.has(info.id);
  return ageAdjusted ? `${info.unit}, age-adjusted` : info.unit;
}

/** Formats a value of a variable with its digits (thousands separators for income). */
export function formatVariableValue(info: VariableInfo, value: number): string {
  if (!Number.isFinite(value)) return 'no data';
  return value.toLocaleString('en-US', {
    minimumFractionDigits: info.digits,
    maximumFractionDigits: info.digits
  });
}

/**
 * The five-class table of a variable on a ground. The breaks are computed once per variable and
 * passed in, so every toggle, swipe and lag map shares them. Health burdens use the people hue
 * (YlOrBr), income PuBu and the other socio-economic indicators Blues.
 */
export function getVariableClassTable(
  info: VariableInfo,
  geography: GeographyId,
  breaks: readonly number[],
  extent: readonly [number, number],
  ground: Ground
): ClassTable {
  const classCount = breaks.length + 1;
  const colors = HEALTH_VARIABLES.has(info.id)
    ? getRegistryColors('people', ground, classCount)
    : info.id === 'income'
      ? getClassPalette('PuBu', classCount, {ground})
      : getRegistryColors('socioEconomic', ground, classCount);
  return makeClassTable({
    breaks,
    colors,
    unit: getUnitText(info, geography),
    extent,
    method: 'Quantile classes: equal numbers of places',
    format: value => formatVariableValue(info, value),
    noData: {label: 'No data'}
  });
}

/** Colours of the four Moran scatterplot quadrants at the weak "untested" alpha. */
export function getQuadrantColors(ground: Ground) {
  const colors = getMoranQuadrantColors(getGiTable(ground, 'No data'));
  const weak = (color: PaletteColor): PaletteColor => [
    color[0],
    color[1],
    color[2],
    QUADRANT_ALPHA
  ];
  return {
    highHigh: weak(colors.highHigh),
    lowHigh: weak(colors.lowHigh),
    lowLow: weak(colors.lowLow),
    highLow: weak(colors.highLow)
  };
}

/**
 * The 5-entry category palette of the quadrant buffer: code 0 (no neighbours, an island) takes
 * the no-data colour, codes 1-4 the quadrants (1 HH, 2 LH, 3 LL, 4 HL).
 */
export function getQuadrantPalette(ground: Ground): PaletteColor[] {
  const colors = getQuadrantColors(ground);
  return [NO_DATA_COLOR[ground], colors.highHigh, colors.lowHigh, colors.lowLow, colors.highLow];
}

/** Quadrant colours in the scatterplot's chart order I-IV (HH, LH, LL, HL), opaque enough for dots. */
export function getScatterPalette(ground: Ground): PaletteColor[] {
  const colors = getMoranQuadrantColors(getGiTable(ground, 'No data'));
  return [colors.highHigh, colors.lowHigh, colors.lowLow, colors.highLow].map(
    color => [color[0], color[1], color[2], 215] as PaletteColor
  );
}

/** Legend entry order of the quadrant legend (HH, LL, HL, LH) and the quadrant code of each entry. */
export const QUADRANT_LEGEND_TO_CODE: readonly number[] = [1, 3, 4, 2];

/** The bivariate key (3 x 3, `row * 3 + column`) on a ground. */
export function getBivariateColors(ground: Ground): PaletteColor[] {
  return (ground === 'dark' ? BIVARIATE_PALETTES.darkGround : BIVARIATE_PALETTES.tealPink).map(
    color => [color[0], color[1], color[2], 255] as PaletteColor
  );
}
