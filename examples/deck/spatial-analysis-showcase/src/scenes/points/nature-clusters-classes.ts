// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Colour of the nature-clusters story. Cluster SIZE is a magnitude, so hot spots are coloured by
 * four fixed size classes (YlGnBu, the registry `nature` hue); cluster IDENTITY is arbitrary, so it
 * is never coloured, except for a k-means partition, which is nominal and gets Okabe-Ito hues
 * assigned by map colouring.
 */

import {getClassPalette, makeClassTable} from '../../cartography/class-table';
import {
  hexToRgba,
  MAP_INK,
  OKABE_ITO_DARK,
  OKABE_ITO_LIGHT,
  OTHER_GREY
} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import {SIZE_BREAKS} from './nature-clusters-analysis';

/** Ground of the map under the data. */
export type ClusterGround = 'light' | 'dark';

/** How many distinct hues a partition may use (the qualitative limit of rule 2.3). */
export const PARTITION_HUES = 7;

/** Iteration caps the k-means step walks through; each cap is its own compiled graph. */
export const ITERATION_LADDER: readonly number[] = [1, 2, 4, 8, 16, 48];

/** The epsilon values (m) of the sweep behind the scale chart; one parameter write each. */
export const SWEEP_EPSILONS: readonly number[] = [100, 150, 200, 300, 400, 600, 800, 1200];

/** Legend and tooltip labels of the size classes. */
export const SIZE_CLASS_LABELS: readonly string[] = [
  'Under 50',
  '50 to 199',
  '200 to 999',
  '1,000 or more'
];

/** Colour of a record that is in no hot spot (noise), at the alpha it is drawn with. */
export function getNoiseColor(ground: ClusterGround, alpha = 77): [number, number, number, number] {
  const [r, g, b] = hexToRgba(OTHER_GREY[ground]);
  return [r, g, b, alpha];
}

/**
 * The one class table of hot-spot size: fixed breaks 50 / 200 / 1,000 (never re-derived when a
 * toggle changes, rule 4), the four darker classes of the five-class YlGnBu table so the lowest
 * class is never near-white on paper.
 *
 * @param noiseCount Records in no hot spot, for the legend's no-data entry.
 */
export function makeSizeTable(ground: ClusterGround, noiseCount?: number): ClassTable {
  return makeClassTable({
    breaks: SIZE_BREAKS,
    colors: getClassPalette('YlGnBu', 5, {ground}).slice(1),
    labels: SIZE_CLASS_LABELS,
    unit: 'records per hot spot',
    method: 'Fixed breaks: 50, 200 and 1,000 records',
    noData: {
      label: 'Not in a hot spot (noise)',
      color: getNoiseColor(ground, 160),
      ...(noiseCount !== undefined ? {count: noiseCount} : {})
    }
  });
}

/** The seven hues of the k-means partition (Okabe-Ito, lifted on a dark ground). */
export function getPartitionPalette(ground: ClusterGround): [number, number, number, number][] {
  const palette = ground === 'dark' ? OKABE_ITO_DARK : OKABE_ITO_LIGHT;
  return palette.slice(0, PARTITION_HUES).map(color => [color[0], color[1], color[2], 255]);
}

/** Ink of the map furniture on a ground (rings, dots, ellipses), RGBA 0-255. */
export function getInk(ground: ClusterGround, alpha = 235): [number, number, number, number] {
  const [r, g, b] = hexToRgba(MAP_INK[ground].ink);
  return [r, g, b, alpha];
}

/** `--map-signal` on a ground: the thing the reader controls (a weight, a median). */
export function getSignal(ground: ClusterGround): [number, number, number, number] {
  const [r, g, b] = hexToRgba(MAP_INK[ground].signal);
  return [r, g, b, 255];
}
