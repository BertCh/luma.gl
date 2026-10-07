// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Place labels and finding notes of terrain-basics. The place names of each step come from the
 * verified `ALPS` gazetteer (`terrain-places.ts`); notes and outlines that carry a number or a
 * data-derived position are built at run time from what the GPU found.
 */

import {formatCount, liveText} from '../../cartography/live-text';
import type {LngLat, MapAnnotation} from '../../cartography/types';
import {terrainLabels} from './terrain-places';

/** Gazetteer ids of the place labels of each step (the Gornergletscher label is added at run time). */
export const STEP_PLACES = {
  /** Step 1 (up to eight labels with the glacier). */
  form: [
    'matterhorn',
    'zermatt',
    'gornergrat',
    'breithorn',
    'klein-matterhorn',
    'hornli-hut',
    'schwarzsee'
  ],
  decode: ['matterhorn', 'riffelberg', 'gornergrat'],
  slopeClasses: ['matterhorn', 'hornli-hut', 'zermatt', 'gornergrat', 'breithorn'],
  groundUnits: ['matterhorn', 'hornli-hut'],
  aspect: ['matterhorn', 'zermatt', 'gornergrat', 'breithorn'],
  rugged: ['gornergrat', 'breithorn', 'klein-matterhorn']
} as const;

/** Zoom overrides so every label of a step shows at that step's camera. */
const LABEL_OVERRIDES = {
  'hornli-hut': {minZoom: 12},
  schwarzsee: {minZoom: 12},
  riffelberg: {minZoom: 11.5},
  'klein-matterhorn': {minZoom: 11}
} as const;

/** The place labels of a step: gazetteer names with elevations, in priority order. */
export function getStepPlaces(ids: readonly string[]): MapAnnotation[] {
  return terrainLabels(ids, LABEL_OVERRIDES);
}

/**
 * The note at the DEM's highest cell near the Matterhorn: the decoded top cell reads lower than
 * the published summit because the cell is 6.6 m wide and heights are quantised.
 */
export function getSummitNote(
  summit: LngLat,
  topCellMeters: number,
  cellMeters: number,
  quantumMeters: number
): MapAnnotation {
  return {
    kind: 'note',
    id: 'basics:summit-note',
    coordinate: summit,
    title: liveText('DEM top cell {height:integer} m', {height: topCellMeters}),
    text: liveText(
      'The published summit is a little higher: {cell:fixed:1} m cells, {step:fixed:2} m height steps',
      {
        cell: cellMeters,
        step: quantumMeters
      }
    ),
    anchor: 'se',
    distance: 70,
    priority: 5
  };
}

/** The note at the steepest cell of the tile for the model on screen. */
export function getSteepestNote(
  position: LngLat,
  steepestDegrees: number,
  model: 'ground' | 'mercator'
): MapAnnotation {
  return {
    kind: 'note',
    id: 'basics:steepest-note',
    coordinate: position,
    title: liveText('Steepest cell: {slope:fixed:0}°', {slope: steepestDegrees}),
    text: model === 'ground' ? 'on ground cells' : 'with Mercator pixels as metres',
    anchor: 'ne',
    distance: 60,
    priority: 5
  };
}

/** The dashed outline and label of the blanked alpha-0 patch. */
export function getPatchOutline(bounds: readonly [number, number, number, number]): MapAnnotation {
  const [west, south, east, north] = bounds;
  return {
    kind: 'outline',
    id: 'basics:patch',
    rings: [
      [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south]
      ]
    ],
    text: 'Alpha-0 patch over Riffelberg and Rotenboden',
    dashed: true
  };
}

/** The dashed frame where the DEM ends, so its edge is a deliberate boundary and not a mistake. */
export function getDataEdgeFrame(bounds: readonly [number, number, number, number]): MapAnnotation {
  return {kind: 'frame', id: 'basics:frame', bounds, text: 'Data ends here'};
}

/** Counts rendered as a readout string, for example `4.2 M cells`. */
export function formatMillions(count: number, unit: string): string {
  return count >= 1e6 ? `${(count / 1e6).toFixed(1)} M ${unit}` : `${formatCount(count)} ${unit}`;
}
