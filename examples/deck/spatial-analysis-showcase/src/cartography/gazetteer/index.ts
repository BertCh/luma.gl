// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Gazetteers: one verified coordinate per named place, per geography, and the helpers that turn
 * places into map annotations. Scenes name places by id (`labelsFor(CHICAGO, ['loop'])`) instead
 * of typing coordinates into step text; where the data holds the feature, resolve the anchor with
 * `../anchors` first.
 */

import type {ViewState} from '../../engine/types';
import {haversineMeters} from '../anchors';
import type {MapAnnotation} from '../types';
import type {Gazetteer, Place, PlaceKind} from './types';

export {ALPS} from './alps';
export {CHICAGO} from './chicago';
export {DIXIE} from './dixie';
export {GRAND_CANYON} from './grand-canyon';
export {MONTREAL} from './montreal';
export {NYC} from './nyc';
export {RANDSTAD} from './randstad';
export {US} from './us';
export {WORLD} from './world';
export type {Gazetteer, Place, PlaceKind} from './types';

/**
 * The standard camera of each geography, so sibling stories open on the same frame (pitch 0).
 * Chicago and New York are the synthesis frames; the others are centred on the gazetteer's places
 * and sized to hold them at a 1000 x 700 px map.
 * Example: `initialView: CITY_FRAMES.chicago`.
 */
export const CITY_FRAMES = {
  chicago: {longitude: -87.69, latitude: 41.84, zoom: 10.25, pitch: 0, bearing: 0},
  nyc: {longitude: -73.97, latitude: 40.74, zoom: 10.9, pitch: 0, bearing: 0},
  montreal: {longitude: -73.6, latitude: 45.52, zoom: 11, pitch: 0, bearing: 0},
  randstad: {longitude: 4.75, latitude: 52.1, zoom: 8.8, pitch: 0, bearing: 0},
  grandCanyon: {longitude: -112.08, latitude: 36.12, zoom: 11.3, pitch: 0, bearing: 0},
  alps: {longitude: 7.76, latitude: 45.97, zoom: 11.7, pitch: 0, bearing: 0},
  dixie: {longitude: -120.87, latitude: 40.15, zoom: 9, pitch: 0, bearing: 0}
} as const satisfies Record<string, ViewState>;

/** Normalizes a name for matching: lower case, no diacritics, only letters and digits. */
function normalizeName(name: string): string {
  return name
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

/** Marker for point places: parks are named without a dot, everything else gets one. */
function getPointMarker(kind: PlaceKind): 'dot' | 'none' {
  return kind === 'park' ? 'none' : 'dot';
}

/**
 * Converts a place to a map annotation, choosing the annotation kind and style from `place.kind`:
 * water, river, ocean and current become `water` (oceans `large` unless the place sets `size`);
 * district, neighborhood and region become `area`; peak, landform, glacier and pass become
 * `landform` (peaks get the triangle marker and `elevationMeters`); everything else becomes a
 * `point` whose `rank` is `'subject'` at priority 2 or more and `'context'` below. `minZoom`,
 * `priority` and a stable `id` (`place:<id>`) are copied; `overrides` win over everything.
 * Example: `placeToAnnotation(ALPS.places.matterhorn, {tone: 'accent'})`.
 */
export function placeToAnnotation(
  place: Place,
  overrides: Partial<MapAnnotation> = {}
): MapAnnotation {
  const base = {
    id: `place:${place.id}`,
    coordinate: place.lngLat,
    text: place.name,
    ...(place.minZoom !== undefined ? {minZoom: place.minZoom} : {}),
    ...(place.priority !== undefined ? {priority: place.priority} : {})
  };
  let annotation: MapAnnotation;
  switch (place.kind) {
    case 'water':
    case 'river':
    case 'ocean':
    case 'current':
      annotation = {
        ...base,
        kind: 'water',
        size: place.size ?? (place.kind === 'ocean' ? 'large' : 'medium')
      };
      break;
    case 'district':
    case 'region':
      annotation = {...base, kind: 'area', size: place.size ?? 'medium'};
      break;
    case 'neighborhood':
      annotation = {...base, kind: 'area', size: place.size ?? 'small'};
      break;
    case 'peak':
    case 'landform':
    case 'glacier':
    case 'pass':
      annotation = {
        ...base,
        kind: 'landform',
        marker: place.kind === 'peak' ? 'peak' : 'none',
        ...(place.elevationM !== undefined ? {elevationMeters: place.elevationM} : {}),
        ...(place.size ? {size: place.size} : {})
      };
      break;
    default:
      annotation = {
        ...base,
        kind: 'point',
        marker: getPointMarker(place.kind),
        rank: (place.priority ?? 0) >= 2 ? 'subject' : 'context'
      };
  }
  return {...annotation, ...overrides} as MapAnnotation;
}

/**
 * Annotations for the listed place ids, in order, with optional per-id overrides. In development
 * an unknown id throws an error that lists close matches; in production it is skipped.
 * Example: `labelsFor(CHICAGO, ['loop', 'lake-michigan'], {loop: {priority: 3}})`.
 */
export function labelsFor(
  gazetteer: Gazetteer,
  ids: readonly string[],
  overrides: Readonly<Record<string, Partial<MapAnnotation>>> = {}
): MapAnnotation[] {
  const annotations: MapAnnotation[] = [];
  for (const id of ids) {
    const place = gazetteer.places[id];
    if (!place) {
      if (import.meta.env.DEV) {
        const needle = id.split('-')[0];
        const close = Object.keys(gazetteer.places)
          .filter(key => key.includes(needle))
          .slice(0, 6);
        throw new Error(
          `Unknown place "${id}" in gazetteer "${gazetteer.id}"` +
            (close.length ? `. Did you mean: ${close.join(', ')}?` : '')
        );
      }
      continue;
    }
    annotations.push(placeToAnnotation(place, overrides[id]));
  }
  return annotations;
}

/** Options of `nearestPlace` and `nearestPlaceLabel`. */
export type NearestPlaceOptions = {
  /** Only consider these kinds. Default: all. */
  kinds?: readonly PlaceKind[];
  /** Ignore places farther than this. Default: no limit. */
  maxDistanceMeters?: number;
};

/**
 * The nearest place of a gazetteer to `lngLat` (great-circle distance), or `null` when none
 * qualifies.
 * Example: `nearestPlace(CHICAGO, [-87.63, 41.96], {kinds: ['park', 'district']})?.name`.
 */
export function nearestPlace(
  gazetteer: Gazetteer,
  lngLat: readonly [number, number],
  options: NearestPlaceOptions = {}
): Place | null {
  return findNearest(gazetteer, lngLat, options)?.place ?? null;
}

/** Nearest place and its distance. */
function findNearest(
  gazetteer: Gazetteer,
  lngLat: readonly [number, number],
  options: NearestPlaceOptions
): {place: Place; distanceMeters: number} | null {
  const maxDistance = options.maxDistanceMeters ?? Infinity;
  let best: {place: Place; distanceMeters: number} | null = null;
  for (const place of Object.values(gazetteer.places)) {
    if (options.kinds && !options.kinds.includes(place.kind)) continue;
    const distanceMeters = haversineMeters(lngLat, place.lngLat);
    if (distanceMeters > maxDistance) continue;
    if (!best || distanceMeters < best.distanceMeters) best = {place, distanceMeters};
  }
  return best;
}

/** Compass point (8 winds) of the bearing from `a` to `b`. */
function getCompassPoint(a: readonly [number, number], b: readonly [number, number]): string {
  const radians = Math.PI / 180;
  const east = (b[0] - a[0]) * Math.cos(((a[1] + b[1]) / 2) * radians);
  const north = b[1] - a[1];
  const winds = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
  return winds[(Math.round(Math.atan2(north, east) / (Math.PI / 4)) + 8) % 8];
}

/**
 * A short phrase locating `lngLat` for a finding note: `"near The Loop"` within 1.5 km of the
 * nearest place, otherwise `"4 km NE of The Loop"`. Returns `null` when no place qualifies.
 * Example: `nearestPlaceLabel(CHICAGO, [-87.64, 41.78], {kinds: ['district']})` is `"near Englewood"`.
 */
export function nearestPlaceLabel(
  gazetteer: Gazetteer,
  lngLat: readonly [number, number],
  options: NearestPlaceOptions = {}
): string | null {
  const found = findNearest(gazetteer, lngLat, options);
  if (!found) return null;
  const {place, distanceMeters} = found;
  if (distanceMeters <= 1500) return `near ${place.name}`;
  const kilometers = distanceMeters / 1000;
  const rounded =
    kilometers < 10 ? kilometers.toFixed(1).replace(/\.0$/, '') : String(Math.round(kilometers));
  return `${rounded} km ${getCompassPoint(place.lngLat, lngLat)} of ${place.name}`;
}

/**
 * Finds a place by id, name or alias, ignoring case and diacritics (`"ohare"` finds "O'Hare
 * Airport" through its alias). Returns `null` when nothing matches.
 * Example: `findPlace(US, 'ATL')?.lngLat`.
 */
export function findPlace(gazetteer: Gazetteer, nameOrAlias: string): Place | null {
  const direct = gazetteer.places[nameOrAlias];
  if (direct) return direct;
  const wanted = normalizeName(nameOrAlias);
  if (!wanted) return null;
  const squashed = wanted.replace(/ /g, '');
  for (const place of Object.values(gazetteer.places)) {
    const names = [place.id, place.name, ...(place.aliases ?? [])];
    for (const name of names) {
      const normalized = normalizeName(name);
      if (normalized === wanted || normalized.replace(/ /g, '') === squashed) return place;
    }
  }
  return null;
}
