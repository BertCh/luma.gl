// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassPalette, makeClassTable} from '../../cartography/class-table';
import {formatCount} from '../../cartography/live-text';
import type {ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';

/**
 * Symbolisation of the similar-places scene that is not a shared table: the rank classes (the
 * layer, the legend and the tooltip read one table), the labels of counties and attribute values,
 * and the Census regions the far-end story counts. Pure data: no luma.gl.
 */

/** The ground a table or line is drawn on. */
export type Ground = 'light' | 'dark';

/** Which end of the ranking is listed first. */
export type SimilarityDirection = 'most' | 'least';

/** Upper rank bounds (exclusive) of the three classes after the match set. */
export const RANK_CLASS_LIMITS = [50, 200, 600] as const;

/** Rank classes: the match set, three bands and the rest. */
export const RANK_CLASS_COUNT = 5;

/** Alpha (0-255) of the "rest" class: 0.55, so the ground stays visible where nothing is said. */
const REST_CLASS_ALPHA = 140;

/** Interior rank breaks of the table for `matchCount` matches: `[N, 50, 200, 600]`. */
export function getRankBreaks(matchCount: number): number[] {
  return [Math.max(1, matchCount), ...RANK_CLASS_LIMITS];
}

/** Options of {@link getRankTable}. */
export type RankTableOptions = {
  direction: SimilarityDirection;
  /** Counties in the match set (the first class). */
  matchCount: number;
  /** Counties that received a rank, for the unit. */
  rankedCount: number;
  ground: Ground;
};

/**
 * The rank class table: five classes, the match set darkest. Most similar uses ColorBrewer BuGn,
 * least similar RdPu, so a reader never confuses "far" with "near". The last class is drawn at
 * alpha 0.55. Class `k` holds ranks (0-based) below break `k`, so the table's class index equals
 * the class index the GPU kernel writes.
 */
export function getRankTable(options: RankTableOptions): ClassTable {
  const {direction, matchCount, rankedCount, ground} = options;
  const limit = Math.max(1, matchCount);
  const scheme = direction === 'most' ? 'BuGn' : 'RdPu';
  // getClassPalette runs low class first: reverse so that the match set takes the darkest colour.
  const palette = getClassPalette(scheme, RANK_CLASS_COUNT, {ground}).reverse();
  const colors: PaletteColor[] = palette.map((color, index) => [
    color[0],
    color[1],
    color[2],
    index === RANK_CLASS_COUNT - 1 ? REST_CLASS_ALPHA : 255
  ]);
  const [first, second, third] = RANK_CLASS_LIMITS;
  const labels = [
    direction === 'most' ? `Top ${limit}` : `Least similar ${limit}`,
    `${limit + 1} to ${first}`,
    `${first + 1} to ${second}`,
    `${second + 1} to ${third}`,
    `Beyond ${third}`
  ];
  return makeClassTable({
    breaks: getRankBreaks(limit),
    colors,
    labels,
    unit: `rank of ${formatCount(rankedCount)} counties${direction === 'least' ? ', least similar first' : ''}`,
    method: 'Rank by weighted distance in standardised attribute space',
    noData: {label: 'Missing attributes'}
  });
}

/** Class index (0 is the match set) of a 0-based rank; the CPU twin of the GPU kernel. */
export function getRankClass(rank: number, matchCount: number): number {
  const breaks = getRankBreaks(matchCount);
  let index = 0;
  while (index < breaks.length && rank >= breaks[index]) index++;
  return index;
}

/** Attribute values of tooltips and notes: thousands separators above 1,000, else one decimal. */
export function formatAttributeValue(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return Math.round(value).toLocaleString('en-US');
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 1) return value.toFixed(1);
  return value.toFixed(2);
}

/** A county as the card and the map name it: `Loudoun, VA`; independent cities say so. */
export function getCountyName(properties: Readonly<Record<string, unknown>> | null): string {
  if (!properties) return 'Unknown county';
  const name = String(properties['name'] ?? '');
  const state = String(properties['state'] ?? '');
  const fips = Number(properties['fips']);
  // Independent cities share names with counties (Fairfax, Richmond, Baltimore): their FIPS
  // county codes start at 500 in the four states that have them.
  const isCity =
    ['VA', 'MD', 'MO', 'NV'].includes(state) && Number.isFinite(fips) && fips % 1000 >= 500;
  return `${name}${isCity ? ' city' : ''}, ${state}`;
}

/** Census Bureau South region (including DC): the far-end story counts these. */
export const SOUTH_STATES: ReadonlySet<string> = new Set([
  'DE',
  'MD',
  'DC',
  'VA',
  'WV',
  'NC',
  'SC',
  'GA',
  'FL',
  'KY',
  'TN',
  'AL',
  'MS',
  'AR',
  'LA',
  'OK',
  'TX'
]);

/** The two most useful presets of the attribute weights; every other weight is 0. */
export const WEIGHT_SETS = {
  /** Income, poverty, schooling and internet access. */
  economy: [0, 1, 6, 7],
  /** Diabetes and obesity prevalence. */
  health: [10, 11]
} as const;

/** Weights with `1` on the listed attribute indices and `0` elsewhere. */
export function getWeights(attributeCount: number, indices?: readonly number[]): number[] {
  return Array.from({length: attributeCount}, (_, index) =>
    !indices || indices.includes(index) ? 1 : 0
  );
}
