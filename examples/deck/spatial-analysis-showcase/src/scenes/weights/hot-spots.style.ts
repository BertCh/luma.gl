// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GI_STAR_BREAKS, makeGiStarClassTable} from '../../cartography/class-table';
import {MAP_INK, hexToRgba} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';

/**
 * Symbolisation of the hot-spots scene that is not a shared table: the Gi* classes themselves
 * come from `makeGiStarClassTable` (the layer, the legend and the tooltip read that one table).
 * What remains here is the glue between that table and the GPU: the category palettes of the
 * `bins` and quadrant buffers, the legend-to-category maps of the interactive legends and the
 * line tiers. Pure data: no luma.gl.
 */

/** The ground a table or line is drawn on. */
type Ground = 'light' | 'dark';

const HIDDEN: PaletteColor = [0, 0, 0, 0];

/** The z-score range of the z display and its histogram (no cell reaches 5 in practice). */
export const Z_DISPLAY_RANGE = [-4, 4] as const;

/** Two-sided normal critical values of the 90, 95 and 99 percent bins (analytic Gi*). */
export const GI_CRITICAL_Z = [1.645, 1.96, 2.576] as const;

/**
 * The Gi* class table of a ground with a "no observations" swatch. Class order is the table's:
 * cold 99, cold 95, cold 90, not significant, hot 90, hot 95, hot 99. The GPU histogram of the
 * bins has the same order, so its counts feed the legend directly.
 */
export function getGiTable(ground: Ground, noDataLabel: string): ClassTable {
  const table = makeGiStarClassTable({ground});
  return {...table, noData: {label: noDataLabel}};
}

/**
 * Gi* bins are `sint32` in `-3..3`. Read as uint32 modulo 8 they select category 0 (not
 * significant), 1..3 (hot 90/95/99) and 7, 6, 5 (cold 90/95/99); entry 4 is never used. Legend
 * class `i` (table order) maps to category `GI_LEGEND_TO_CATEGORY[i]`.
 */
export const GI_LEGEND_TO_CATEGORY: readonly number[] = [5, 6, 7, 0, 1, 2, 3];

/** The 8-entry category palette of the Gi* `bins` buffer, from the table's colours. */
export function getGiPalette(table: ClassTable, showNotSignificant: boolean): PaletteColor[] {
  const [cold99, cold95, cold90, neutral, hot90, hot95, hot99] = table.colors.map(
    color => [color[0], color[1], color[2], color[3] ?? 255] as PaletteColor
  );
  return [
    showNotSignificant ? neutral : HIDDEN,
    hot90,
    hot95,
    hot99,
    HIDDEN,
    cold99,
    cold95,
    cold90
  ];
}

/** Colours of the local Moran quadrants, from the same table: clusters saturated, outliers light. */
export function getMoranQuadrantColors(table: ClassTable) {
  const color = (index: number) => {
    const entry = table.colors[index];
    return [entry[0], entry[1], entry[2], 255] as PaletteColor;
  };
  const neutral = table.colors[3];
  return {
    highHigh: color(6),
    lowHigh: color(1),
    lowLow: color(0),
    highLow: color(5),
    notSignificant: [neutral[0], neutral[1], neutral[2], neutral[3] ?? 255] as PaletteColor
  };
}

/**
 * Local Moran quadrant codes: 0 not significant, 1 HH, 2 LH, 3 LL, 4 HL. The legend lists
 * `[HH, LL, HL, LH, not significant]`, so legend entry `i` is category `MORAN_LEGEND_TO_CATEGORY[i]`.
 */
export const MORAN_LEGEND_TO_CATEGORY: readonly number[] = [1, 3, 4, 2, 0];

/** The 8-entry category palette of local Moran quadrant codes. */
export function getMoranPalette(table: ClassTable, showNotSignificant: boolean): PaletteColor[] {
  const {highHigh, lowHigh, lowLow, highLow, notSignificant} = getMoranQuadrantColors(table);
  return [
    showNotSignificant ? notSignificant : HIDDEN,
    highHigh,
    lowHigh,
    lowLow,
    highLow,
    HIDDEN,
    HIDDEN,
    HIDDEN
  ];
}

/** Break values of the Gi* table, for chart guides (`+-1.65`, `+-1.96`, `+-2.58`). */
export const GI_BREAKS: readonly number[] = GI_STAR_BREAKS;

/** Polygon hairline (tier 3): white at 0.63 on light grounds, ground ink at 0.5 on dark. */
export function getHairlineColor(ground: Ground): PaletteColor {
  return ground === 'dark' ? [14, 17, 22, 128] : [255, 255, 255, 161];
}

/** Context outlines under data (tracts behind the cells): grey, thin, translucent. */
export function getContextLineColor(ground: Ground): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].context, ground === 'dark' ? 90 : 72);
  return [color[0], color[1], color[2], color[3]];
}

/**
 * Zone boundary tier (state lines): 0.9 px `#3A3F4B` at 0.8 on light over a 1.6 px white casing;
 * on dark the light ink over a ground-coloured casing.
 */
export function getStateLineStyle(ground: Ground): {
  color: PaletteColor;
  casing: PaletteColor;
  widthPixels: number;
  casingPixels: number;
} {
  return ground === 'dark'
    ? {color: [170, 182, 195, 200], casing: [14, 17, 22, 130], widthPixels: 0.9, casingPixels: 1.6}
    : {color: [58, 63, 75, 204], casing: [255, 255, 255, 190], widthPixels: 0.9, casingPixels: 1.6};
}
