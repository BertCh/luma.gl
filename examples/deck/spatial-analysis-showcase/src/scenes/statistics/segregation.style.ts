// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassPalette, makeClassTable} from '../../cartography/class-table';
import {hexToRgba, MAP_INK, RACE_GROUP_COLORS} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';

/**
 * Symbolisation of the segregation scene that is not a shared table: the group registry, the
 * hue-by-dominance palette of the largest-group map, the two class tables of the environment
 * maps and the line tiers. Pure data: no luma.gl, so the scene file can import it.
 */

/** The ground a table or line is drawn on. */
export type Ground = 'light' | 'dark';

/**
 * Population groups in the column order of the `groupCounts` buffer. `registryIndex` is the
 * position of the group in `RACE_GROUP_COLORS` and the legend (White, Black, Hispanic, Asian,
 * Other), so a group keeps one hue family in every step.
 */
export const SEGREGATION_GROUPS = [
  {id: 'nhBlack', label: 'Black', registryIndex: 1},
  {id: 'nhWhite', label: 'White', registryIndex: 0},
  {id: 'hispanic', label: 'Hispanic', registryIndex: 2},
  {id: 'nhAsian', label: 'Asian', registryIndex: 3},
  {id: 'nhOther', label: 'Other or multiracial', registryIndex: 4}
] as const;

/** Number of groups. */
export const GROUP_COUNT = SEGREGATION_GROUPS.length;

/** Class value of a tract without residents (the layer's no-data sentinel). */
export const NO_GROUP_CLASS = 0xffffffff;

/** Share of the largest group at which a tract becomes a majority and a supermajority tract. */
export const DOMINANCE_EDGES = [0.5, 0.8] as const;

/** Column labels of the dominance key, one per tier. */
export const DOMINANCE_TIER_LABELS = ['Under 50%', '50 to 80%', '80% or more'] as const;

/**
 * The three tints of each group hue, plurality then majority then supermajority. Light ground:
 * published ColorBrewer steps of the group's single-hue family (Blues, Oranges, Greens, Purples,
 * Greys), the same hues as `RACE_GROUP_COLORS`. Dark ground: the dark table of the same scheme
 * (strongest class brightest), same positions.
 */
const LIGHT_TIER_HEXES: readonly (readonly [string, string, string])[] = [
  ['#BDD7E7', '#6BAED6', '#08519C'],
  ['#FDBE85', '#FD8D3C', '#D94701'],
  ['#BAE4B3', '#74C476', '#006D2C'],
  ['#CBC9E2', '#9E9AC8', '#54278F'],
  ['#D9D9D9', '#969696', '#525252']
];

const TIER_SCHEMES = ['Blues', 'Oranges', 'Greens', 'Purples', 'Greys'] as const;

/**
 * The 15 colours of the largest-group map, class `registryGroup * 3 + tier`. The layer palette,
 * the matrix legend and the tooltip swatches all read this one list.
 */
export function getDominanceColors(ground: Ground): PaletteColor[] {
  const colors: PaletteColor[] = [];
  TIER_SCHEMES.forEach((scheme, group) => {
    if (ground === 'light') {
      for (const hex of LIGHT_TIER_HEXES[group]) colors.push(hexToRgba(hex));
    } else {
      // Positions 1, 2 and 4 of the 5-class table, as on the light ground.
      const table = getClassPalette(scheme, 5, {ground});
      for (const position of [1, 2, 4]) colors.push(table[position]);
    }
  });
  return colors;
}

/** Colour of one group at full strength (the stacked bars and tooltip rows), from the registry. */
export function getGroupColor(groupIndex: number, ground: Ground): PaletteColor {
  return RACE_GROUP_COLORS[ground][SEGREGATION_GROUPS[groupIndex].registryIndex];
}

/** Tier (0 plurality, 1 majority, 2 supermajority) of a largest-group share. */
export function getDominanceTier(share: number): 0 | 1 | 2 {
  return share < DOMINANCE_EDGES[0] ? 0 : share < DOMINANCE_EDGES[1] ? 1 : 2;
}

/** Steps of the over- and under-representation classes, as log2 of the share over the city share. */
export const RELATIVE_BREAKS: readonly number[] = [-1.5, -0.75, -0.25, 0.25, 0.75, 1.5];

/** Clip of the relative value, in log2 units (4 times under to 4 times over). */
export const RELATIVE_CLIP = 2;

/** Labels of the seven relative classes: multiples of the citywide share, low first. */
export const RELATIVE_LABELS: readonly string[] = [
  'under 0.35x',
  '0.35 to 0.6x',
  '0.6 to 0.85x',
  '0.85 to 1.2x (about the city share)',
  '1.2 to 1.7x',
  '1.7 to 2.8x',
  'over 2.8x'
];

/** Breaks of the local-diversity classes, as a share of the maximum entropy. */
export const DIVERSITY_BREAKS: readonly number[] = [0.4, 0.55, 0.7, 0.85];

/**
 * Over- and under-representation of a group in the environment of each tract: PuOr-7 (deviation
 * from a reference, orange above), steps at 0.35, 0.6, 0.85, 1.2, 1.7 and 2.8 times the citywide
 * share. Frozen across scales, groups and the swipe.
 */
export function getRelativeTable(ground: Ground): ClassTable {
  return makeClassTable({
    breaks: RELATIVE_BREAKS,
    scheme: 'PuOr',
    ground,
    labels: RELATIVE_LABELS,
    unit: 'x the citywide share',
    method: 'Environment share over the citywide share; steps at log2 of -1.5 to +1.5',
    noData: {label: 'No residents', hatched: true}
  });
}

/**
 * Local diversity (entropy over its maximum): Greys-5 from the published 6-class table without
 * its near-white first step, so the least mixed class still reads on the paper. Grey means
 * "between groups" in this scene; the hues keep meaning groups.
 */
export function getDiversityTable(ground: Ground): ClassTable {
  const colors = getClassPalette('Greys', 6, {ground}).slice(1);
  return makeClassTable({
    breaks: DIVERSITY_BREAKS,
    colors,
    ground,
    extent: [0, 1],
    unit: 'of the maximum diversity',
    method: 'Entropy of the environment over ln 5; 1 is an equal mix of five groups',
    format: value => value.toFixed(2),
    noData: {label: 'No residents', hatched: true}
  });
}

/** Ink of selections and ring-adjacent outlines: achromatic, from the ground. */
export function getInkColor(ground: Ground, alpha = 255): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].ink, alpha);
  return [color[0], color[1], color[2], color[3]];
}

/** Ground-coloured casing under selections. */
export function getCasingColor(ground: Ground, alpha = 235): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].halo, alpha);
  return [color[0], color[1], color[2], color[3]];
}

/**
 * Community-area boundary tier. As context over the tracts: `#3A3F4B` 0.8 px at alpha 140. As the
 * subject (the unit of the map): 1.8 px at alpha 230 over a 3 px ground casing.
 */
export function getAreaLineStyle(
  ground: Ground,
  subject: boolean
): {color: PaletteColor; widthPixels: number; casing: PaletteColor; casingPixels: number} {
  const ink: PaletteColor = ground === 'dark' ? [170, 182, 195, 255] : [58, 63, 75, 255];
  const casing: PaletteColor = ground === 'dark' ? [14, 17, 22, 150] : [255, 255, 255, 190];
  return subject
    ? {
        color: [ink[0], ink[1], ink[2], 230],
        widthPixels: 1.8,
        casing,
        casingPixels: 3
      }
    : {
        color: [ink[0], ink[1], ink[2], 140],
        widthPixels: 0.8,
        casing: [casing[0], casing[1], casing[2], 0],
        casingPixels: 0.8
      };
}

/** Index value with `digits` decimals (two in prose, three where sums are compared). */
export function formatIndex(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}
