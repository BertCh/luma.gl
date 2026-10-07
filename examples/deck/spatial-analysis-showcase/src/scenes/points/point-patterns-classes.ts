// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The two classified fills of the point-patterns story, each one class table that the layer, the
 * legend and the tooltip all read:
 *
 * - nearest-neighbour distance, in multiples of the distance E that a random pattern would give
 *   (fixed, comparable break points; a class of its own for exactly 0 m);
 * - quadrat counts, natural breaks of the non-empty quadrats, frozen per selection.
 */

import {getClassBreaks, getGoodnessOfVarianceFit} from '../../cartography/breaks';
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
 * The quadrat table: class 0 is "no records" (transparent, the lake and the empty quadrats), the
 * other classes are natural breaks (Jenks) of the non-empty counts with the goodness of variance fit
 * in the method line. Compute it once per quadrat grid and selection and keep it fixed.
 */
export function makeQuadratTable(counts: ArrayLike<number>, ground: PointsGround): QuadratClasses {
  const nonEmpty: number[] = [];
  let maximum = 0;
  for (let index = 0; index < counts.length; index++) {
    if (counts[index] > 0) nonEmpty.push(counts[index]);
    maximum = Math.max(maximum, counts[index]);
  }
  const interior = nonEmpty.length ? getClassBreaks(nonEmpty, 4, 'natural-breaks') : [];
  const breaks = [1, ...interior.filter(value => value > 1)];
  const classCount = breaks.length + 1;
  const labels = ['No records'];
  for (let index = 1; index < classCount; index++) {
    const low = breaks[index - 1];
    const high = index < breaks.length ? breaks[index] - 1 : maximum;
    labels.push(low >= high ? `${low}` : `${low} to ${high}`);
  }
  const gvf = nonEmpty.length ? getGoodnessOfVarianceFit(nonEmpty, interior) : 1;
  const table = makeClassTable({
    breaks,
    colors: getClassPalette('YlGnBu', classCount, {ground}),
    transparent: [0],
    labels,
    unit: 'records per quadrat',
    extent: [0, Math.max(maximum, 1)],
    method: `Natural breaks (Jenks) of the non-empty quadrats, GVF ${gvf.toFixed(2)}`,
    noData: {label: 'Outside the window', color: [128, 128, 128, 60]}
  });
  const classCounts = new Array<number>(classCount).fill(0);
  for (let index = 0; index < counts.length; index++) {
    let classIndex = 0;
    while (classIndex < breaks.length && counts[index] >= breaks[classIndex]) classIndex++;
    classCounts[classIndex]++;
  }
  return {table, counts: classCounts};
}
