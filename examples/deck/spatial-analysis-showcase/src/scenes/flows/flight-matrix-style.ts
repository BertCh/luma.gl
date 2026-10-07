// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The look of the flight-matrix story (design sheet `FID/design/flight-matrix.md`): the matrix
 * card geometry, the sheet colours of the paper ground, the classed integer count table and the
 * observed-versus-expected table. Pure TypeScript with no GPU imports, so the scene file may import
 * it.
 *
 * - **A matrix is not geography.** The matrix steps sit on the `paperSheet` ground: no tiles, and
 *   the card is a plot area one shade lighter than the sheet with a one pixel rule.
 * - **Counts are small integers**, so they are classed `1 | 2 | 3-5 | 6-10 | 11-20 | 21+` on
 *   ColorBrewer YlGnBu with the pale end dropped, and an empty cell is the plot colour, distinct
 *   from the lowest class.
 * - **Observed over expected** is the registry's PuOr deviation table on `log2(observed/expected)`
 *   with the midpoint "as expected by chance"; orange is above.
 */

import {getClassPalette, hexToRgba, makeClassTable} from '../../cartography/class-table';
import {getRegistryColors, type MapGround} from '../../cartography/hue-registry';
import type {ClassTable} from '../scene';
import type {PaletteColor} from '../../engine/ramps';
import {getGroupPalette} from './flows-style';

// ---------------------------------------------------------------------------------------------
// Card geometry
// ---------------------------------------------------------------------------------------------

/** Side of the matrix card in metres around longitude 0, latitude 0 (planar `METER_OFFSETS`). */
export const MATRIX_SIDE = 10_000_000;
/** Half the card side. */
export const MATRIX_HALF = MATRIX_SIDE / 2;
/** Distance of the continent strips from the card edge, as a share of the side. */
export const STRIP_GAP_SHARE = 0.022;
/** Distance of the axis labels from the card edge, as a share of the side. */
export const LABEL_GAP_SHARE = 0.04;
/** Half the side of the sheet and mask quads: far larger than any view of the card. */
export const SHEET_HALF = 15_000_000;

const EARTH_RADIUS = 6_378_137;

/** Web Mercator metres at the equator origin to `[longitude, latitude]` (for camera bounds only). */
export function metersToLngLat(x: number, y: number): [number, number] {
  return [
    (x / EARTH_RADIUS) * (180 / Math.PI),
    (2 * Math.atan(Math.exp(y / EARTH_RADIUS)) - Math.PI / 2) * (180 / Math.PI)
  ];
}

/**
 * Camera bounds that frame the card with room for the axis labels on the left and the strips and
 * caption above: `[west, south, east, north]`.
 */
export const MATRIX_CAMERA_BOUNDS: readonly [number, number, number, number] = (() => {
  const [west, south] = metersToLngLat(
    -MATRIX_HALF - MATRIX_SIDE * 0.27,
    -MATRIX_HALF - MATRIX_SIDE * 0.05
  );
  const [east, north] = metersToLngLat(
    MATRIX_HALF + MATRIX_SIDE * 0.05,
    MATRIX_HALF + MATRIX_SIDE * 0.1
  );
  return [west, south, east, north];
})();

// ---------------------------------------------------------------------------------------------
// The paper sheet
// ---------------------------------------------------------------------------------------------

/** Colours of the matrix sheet on one ground. */
export type SheetColors = {
  /** The page under everything (the `paperSheet` ground's background). */
  sheet: PaletteColor;
  /** The plot area of the card: one shade lighter (light) or brighter (dark) than the sheet. */
  plot: PaletteColor;
  /** The card frame and the continent block lines. */
  rule: PaletteColor;
  /** Ink of captions drawn on the card. */
  ink: PaletteColor;
};

const SHEET_COLORS: Record<MapGround, SheetColors> = {
  light: {
    sheet: hexToRgba('#F4F1EA'),
    plot: hexToRgba('#FBF9F4'),
    rule: hexToRgba('#1F2933'),
    ink: hexToRgba('#1F2933')
  },
  dark: {
    sheet: hexToRgba('#14171C'),
    plot: hexToRgba('#1B1F26'),
    rule: hexToRgba('#E8EDF2'),
    ink: hexToRgba('#E8EDF2')
  }
};

/** The sheet colours that match the `paperSheet` ground on `ground`. */
export function getSheetColors(ground: MapGround): SheetColors {
  return SHEET_COLORS[ground];
}

/** Hues of the six continents on a ground, in `CONTINENT_NAMES` order (identity, never rank). */
export function getContinentPalette(ground: MapGround, alpha = 255): PaletteColor[] {
  return getGroupPalette(ground, alpha).slice(0, 6);
}

// ---------------------------------------------------------------------------------------------
// Classed integer counts
// ---------------------------------------------------------------------------------------------

/** Interior breaks of the cell classes `1 | 2 | 3-5 | 6-10 | 11-20 | 21+`. */
export const COUNT_BREAKS: readonly number[] = [2, 3, 6, 11, 21];
/** Class labels of the count table. */
export const COUNT_LABELS: readonly string[] = ['1', '2', '3-5', '6-10', '11-20', '21+'];

/**
 * The class table of the matrix cells: six classes of the published nine-class YlGnBu table with
 * the three palest dropped, so the lowest class (a single airport pair, the commonest cell) stays
 * visible on paper; on the dark sheet the authored dark table with the same classes. Layer, legend,
 * tooltip and bar chart all read this one table.
 */
export function getCountTable(ground: MapGround, emptyCells?: number): ClassTable {
  return makeClassTable({
    breaks: COUNT_BREAKS,
    colors: getClassPalette('YlGnBu', 9, {ground}).slice(3),
    labels: COUNT_LABELS,
    unit: 'airport pairs per cell',
    method: 'Integer counts in fixed classes; an empty cell is the plot colour',
    noData: {
      label: 'No route',
      color: getSheetColors(ground).plot,
      ...(emptyCells === undefined ? {} : {count: emptyCells})
    }
  });
}

// ---------------------------------------------------------------------------------------------
// Observed versus expected
// ---------------------------------------------------------------------------------------------

/** Breaks of `log2(observed / expected)`: 1/4, 1/2, 0.8, 1.25, 2 and 4 times the chance value. */
export const CHANCE_BREAKS: readonly number[] = [-2, -1, Math.log2(0.8), Math.log2(1.25), 1, 2];
/** Words of the seven deviation classes, low class first. */
export const CHANCE_LABELS: readonly string[] = [
  'At most a quarter',
  'A quarter to a half',
  'Half to 0.8 times',
  'As expected by chance',
  '1.25 to 2 times',
  '2 to 4 times',
  'At least 4 times'
];

/**
 * The seven-class deviation table (PuOr, orange = more than chance). The neutral class is the
 * midpoint "as expected by chance" (a ratio between 0.8 and 1.25).
 */
export function getChanceTable(ground: MapGround): ClassTable {
  return makeClassTable({
    breaks: CHANCE_BREAKS,
    colors: getRegistryColors('deviation', ground, 7),
    labels: CHANCE_LABELS,
    unit: 'observed over expected',
    method: 'Expected: degree-preserving chance, D(a) x D(b) / 2m',
    noData: {label: 'No routes between the groups'}
  });
}
