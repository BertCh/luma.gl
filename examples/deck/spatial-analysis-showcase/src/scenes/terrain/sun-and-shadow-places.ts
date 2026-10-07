// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Place labels and finding notes of sun-and-shadow. Names, coordinates and elevations of each step
 * come from the verified `ALPS` gazetteer through `terrain-places.ts`; the notes, the light arrows
 * and the search ring carry numbers or directions read from the sun and the map, so they are built
 * at run time.
 */

import {getCompassName} from './b14b-terrain';
import {liveText} from '../../cartography/live-text';
import {geodesicDestination} from '../../cartography/reference-geometry';
import type {LngLat, MapAnnotation} from '../../cartography/types';
import {formatDayMonth} from './sun-and-shadow.style';
import {terrainLabels} from './terrain-places';

/** Gazetteer ids of the place labels of each step (at most six; the first step may carry more). */
export const STEP_PLACES = {
  question: [
    'zermatt',
    'matterhorn',
    'gornergrat',
    'riffelberg',
    'dufourspitze',
    'gornergletscher'
  ],
  light: ['zermatt', 'matterhorn', 'gornergrat'],
  horizon: ['zermatt', 'matterhorn', 'gornergrat', 'riffelberg'],
  cast: ['zermatt', 'matterhorn', 'gornergrat'],
  day: ['zermatt', 'matterhorn', 'gornergrat', 'riffelberg'],
  hours: ['zermatt', 'matterhorn', 'gornergrat', 'dufourspitze', 'breithorn', 'gornergletscher']
} as const;

/** Zoom overrides so every label of a step shows at that step's camera. */
const LABEL_OVERRIDES = {
  riffelberg: {minZoom: 11},
  gornergrat: {minZoom: 10.5},
  gornergletscher: {minZoom: 10},
  breithorn: {minZoom: 10}
} as const;

/** The place labels of a step: gazetteer names with elevations, in priority order. */
export function getStepPlaces(ids: readonly string[]): MapAnnotation[] {
  return terrainLabels(ids, LABEL_OVERRIDES);
}

/** The note at the village: its hours of direct sun on the shown date, from the CPU timeline. */
export function getVillageNote(
  coordinate: LngLat,
  sunHours: number,
  dayOfYear: number
): MapAnnotation {
  return {
    kind: 'note',
    id: 'sun:village-note',
    coordinate,
    title: liveText('{hours:fixed:1} h of direct sun on {date}', {
      hours: sunHours,
      date: formatDayMonth(dayOfYear)
    }),
    text: 'In the village, clear sky',
    anchor: 'sw',
    distance: 54,
    priority: 6
  };
}

/** Where the arrows of the two lights start and stop, in metres from the village. */
const ARROW_START_METERS = 3600;
const ARROW_END_METERS = 1100;

/**
 * Arrows for the two lights of the lighting story: the map's light from the north-west (always),
 * and the real sun from its azimuth (when the real sun is shown and above the horizon). Both
 * point at the village from the side the light comes from.
 */
export function getLightArrows(
  village: LngLat,
  showSun: boolean,
  sun: {azimuth: number; altitude: number} | null,
  mapLightAzimuth: number
): MapAnnotation[] {
  const arrows: MapAnnotation[] = [
    {
      kind: 'arrow',
      id: 'sun:map-light',
      from: geodesicDestination(village, mapLightAzimuth, ARROW_START_METERS),
      to: geodesicDestination(village, mapLightAzimuth, ARROW_END_METERS),
      text: 'Map light, north-west',
      tone: showSun ? 'muted' : 'ink'
    }
  ];
  if (showSun && sun && sun.altitude > 0) {
    arrows.push({
      kind: 'arrow',
      id: 'sun:real-sun',
      from: geodesicDestination(village, sun.azimuth, ARROW_START_METERS),
      to: geodesicDestination(village, sun.azimuth, ARROW_END_METERS),
      text: `The sun, ${getCompassName(sun.azimuth)}`,
      tone: 'accent'
    });
  }
  return arrows;
}

/** The dashed search ring of the horizon map around the pinned cell, with its radius on it. */
export function getSearchRing(coordinate: LngLat, radiusMeters: number): MapAnnotation {
  return {
    kind: 'ring',
    id: 'sun:search-ring',
    coordinate,
    radiusMeters,
    text: liveText('Search radius {radius:km}', {radius: radiusMeters}),
    dashed: true
  };
}
