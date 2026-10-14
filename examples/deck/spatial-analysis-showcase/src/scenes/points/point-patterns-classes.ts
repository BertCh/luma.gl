// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The two classified fills of the point-patterns story, each one class table that the layer, the
 * legend and the tooltip all read:
 *
 * - nearest-neighbour distance, in multiples of the distance E that a random pattern would give
 *   (fixed, comparable break points; a class of its own for exactly 0 m);
 * - quadrat counts divided by their area-weighted CSR expectations.
 */

import {getClassPalette, makeClassTable} from '../../cartography/class-table';
import {formatDistance} from '../../cartography/live-text';
import type {ClassTable} from '../../cartography/types';
import {getParameterInk, type PointsGround} from './b1-points-look';

/** Class edges of the nearest-neighbour distance, as multiples of the CSR expectation E. */
export const NEAREST_MULTIPLES: readonly number[] = [0.25, 0.5, 1, 2];

/** Distances below this many metres count as "the same coordinate" (float32 positions are exact there). */
export const SAME_COORDINATE_METERS = 0.01;

/** Labels of the six nearest-neighbour classes (class 0 is the exact duplicates). */
export const NEAREST_LABELS: readonly string[] = [
  'Same coordinate (0 m)',
  '< 0.25 × expected',
  '0.25 to 0.5 × expected',
  '0.5 to 1 × expected',
  '1 to 2 × expected',
  '≥ 2 × expected'
];

/**
 * The nearest-neighbour table: breaks at 0 m, 0.25 E, 0.5 E, 1 E and 2 E. Short distances are the
 * emphasised end on both grounds (bright on night, dark on paper), the lowest of the five measured
 * classes never fades into the ground. Class 0 (duplicates) is ink-white and drawn as a ring.
 *
 * @param expected The expected nearest-neighbour distance E under CSR, in metres.
 */
export function makeNearestTable(expected: number, ground: PointsGround): ClassTable {
  const edges = NEAREST_MULTIPLES.map(multiple => multiple * expected);
  // Six classes, the faintest dropped: on night the authored low class sits at the ground's own
  // lightness and on paper the palest is near white, and a far neighbour must still be visible.
  const measured = getClassPalette('YlGnBu', 6, {ground, reverse: true}).slice(0, 5);
  const ink = getParameterInk(ground, 255);
  return makeClassTable({
    breaks: [SAME_COORDINATE_METERS, ...edges],
    colors: [ink, ...measured],
    labels: NEAREST_LABELS,
    unit: 'm',
    method: `Multiples of the distance a random pattern gives, E = ${formatDistance(expected)}`,
    noData: {label: 'Not in the analysis', color: [128, 128, 128, 60]}
  });
}

/** A quadrat classification and what it was computed from. */
export type QuadratClasses = {
  table: ClassTable;
  /** Quadrats per class (class 0 is the empty ones). */
  counts: number[];
};

/**
 * Classes of GPU counts divided by expected counts, where expectation is proportional to the
 * GPU quadrat area. NaN denotes zero-area quadrats outside the window; zero denotes no records.
 */
export function makeQuadratTable(ratios: ArrayLike<number>, ground: PointsGround): QuadratClasses {
  const breaks = [1e-20, 0.5, 1, 2, 4];
  const table = makeClassTable({
    breaks,
    colors: getClassPalette('YlGnBu', 6, {ground}),
    transparent: [0],
    labels: [
      'No records',
      '< 0.5 × expected',
      '0.5 to 1 × expected',
      '1 to 2 × expected',
      '2 to 4 × expected',
      '≥ 4 × expected'
    ],
    unit: 'observed / expected',
    method:
      'Expected count = included records × quadrat area / window area; areas measured on the GPU',
    noData: {label: 'Outside the window', color: [0, 0, 0, 0]}
  });
  const classCounts = new Array<number>(6).fill(0);
  for (let index = 0; index < ratios.length; index++) {
    if (!Number.isFinite(ratios[index])) continue;
    let classIndex = 0;
    while (classIndex < breaks.length && ratios[index] >= breaks[classIndex]) classIndex++;
    classCounts[classIndex]++;
  }
  return {table, counts: classCounts};
}
