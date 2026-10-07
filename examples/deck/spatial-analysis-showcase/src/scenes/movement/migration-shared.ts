// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {createAzimuthalEquidistant, createTrackSet, type TrackSet} from './b12-tracks';

/**
 * Shared helpers of the `migration-*` scenes: the poopdeck animals dataset as a track set (azimuthal
 * equidistant meters for the analysis, lon/lat degrees for drawing), species colors, calendar labels and
 * a small gazetteer for naming sites.
 */

export const MIGRATION_DATASET_ID = 'poopdeck-animals';
export const SECONDS_PER_DAY = 86400;
export const DAYS_IN_YEAR = 366;
export const YEAR_START_MILLISECONDS = Date.UTC(2024, 0, 1);

/** Species in the dataset's category order. */
export const MIGRATION_SPECIES = [
  'Circus aeruginosus',
  'Circus pygargus',
  'Platalea leucorodia'
] as const;
export const MIGRATION_SPECIES_LABELS = [
  'Western marsh harrier',
  "Montagu's harrier",
  'Eurasian spoonbill'
] as const;
/** Okabe-Ito based species colors, readable on light and dark basemaps. */
export const MIGRATION_SPECIES_COLORS: readonly (readonly [number, number, number, number])[] = [
  [86, 180, 233, 255],
  [240, 150, 30, 255],
  [204, 121, 167, 255]
];

/** Where each source dataset's birds were tagged. */
const SOURCE_PLACES: Record<string, string> = {
  MH_WATERLAND: 'Waterland (Belgium-Netherlands border)',
  H_GRONINGEN: 'Groningen (Netherlands)',
  MH_ANTWERPEN: 'near Antwerp (Belgium)',
  BOP_RODENT: 'Flanders (Belgium)',
  SPOONBILL_VLAANDEREN: 'Flanders (Belgium)'
};

/** The migration tracks with the per-track attributes of the dataset. */
export type MigrationTrackSet = TrackSet & {
  /** Species index per track (see {@link MIGRATION_SPECIES}). */
  species: Uint8Array;
  /** Individual (tag) index per track; one animal owns several tracks, one per tagged year. */
  individual: Uint16Array;
  individualIds: readonly string[];
  /** Tagged-year number per track (1 = first year). */
  year: Uint8Array;
  /** Source dataset index per track. */
  source: Uint8Array;
  sourceNames: readonly string[];
  /** Days between the first and last fix of each track. */
  coverageDays: Float32Array;
  /** Path length in meters of each track in the analysis projection. */
  pathMeters: Float32Array;
};

/** Loads `poopdeck-animals` as a {@link MigrationTrackSet}. */
export function loadMigrationTracks(dataset: LoadedDataset): MigrationTrackSet {
  const lngLat = dataset.column<Float32Array>('vertices');
  const bbox = dataset.manifest.bbox;
  const center: [number, number] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
  const projection = createAzimuthalEquidistant(center);
  const positions = new Float32Array(lngLat.length);
  for (let index = 0; index < lngLat.length; index += 2) {
    const [x, y] = projection.project(lngLat[index], lngLat[index + 1]);
    positions[index] = x;
    positions[index + 1] = y;
  }
  // Seconds of the year as float32: exact to 2 s above 2^24 s, ample for fixes about 2 hours apart.
  const timestamps = Float32Array.from(dataset.column<Uint32Array>('timestamp'));
  const offsets = dataset.column<Uint32Array>('pathOffsets');
  const set = createTrackSet({
    offsets,
    positions,
    lngLat,
    timestamps,
    origin: center,
    project: projection.project,
    unproject: projection.unproject,
    drawInDegrees: true
  });
  const coverageDays = new Float32Array(set.trackCount);
  const pathMeters = new Float32Array(set.trackCount);
  for (let track = 0; track < set.trackCount; track++) {
    const first = offsets[track];
    const last = offsets[track + 1] - 1;
    coverageDays[track] = (timestamps[last] - timestamps[first]) / SECONDS_PER_DAY;
    let length = 0;
    for (let vertex = first; vertex < last; vertex++) {
      length += Math.hypot(
        positions[vertex * 2 + 2] - positions[vertex * 2],
        positions[vertex * 2 + 3] - positions[vertex * 2 + 1]
      );
    }
    pathMeters[track] = length;
  }
  return {
    ...set,
    species: dataset.column<Uint8Array>('species'),
    individual: dataset.column<Uint16Array>('individual'),
    individualIds: dataset.categories('individual'),
    year: dataset.column<Uint8Array>('year'),
    source: dataset.column<Uint8Array>('source'),
    sourceNames: dataset.categories('source'),
    coverageDays,
    pathMeters
  };
}

/** `Tag H197169, year 2, Western marsh harrier tagged near Antwerp (Belgium)`. */
export function describeMigrationTrack(set: MigrationTrackSet, track: number): string {
  const species = MIGRATION_SPECIES_LABELS[set.species[track]];
  const place = SOURCE_PLACES[set.sourceNames[set.source[track]]] ?? '';
  return `${species} ${set.individualIds[set.individual[track]]}, tagged year ${set.year[track]} (${place})`;
}

/** Tracks of `species` (index) or all when `species` is `null`, restricted to fixes in `[from, to]` seconds of the year. */
export type TrackSubset = {
  trackCount: number;
  vertexCount: number;
  /** Longitude/latitude degrees, one row per kept vertex. */
  positions: Float32Array;
  /** `trackCount + 1` offsets. */
  offsets: Uint32Array;
  /** Source track of each subset path. */
  tracks: Uint32Array;
};

/** Cuts each track to its fixes inside the time window and keeps the tracks of the species. */
export function buildTrackSubset(
  set: MigrationTrackSet,
  species: number | null,
  fromSeconds: number,
  toSeconds: number
): TrackSubset {
  const paths: {track: number; first: number; last: number}[] = [];
  let vertexCount = 0;
  for (let track = 0; track < set.trackCount; track++) {
    if (species !== null && set.species[track] !== species) continue;
    let first = -1;
    let last = -1;
    for (let vertex = set.offsets[track]; vertex < set.offsets[track + 1]; vertex++) {
      const time = set.timestamps[vertex];
      if (time < fromSeconds || time > toSeconds) continue;
      if (first < 0) first = vertex;
      last = vertex;
    }
    if (first >= 0 && last > first) {
      paths.push({track, first, last});
      vertexCount += last - first + 1;
    }
  }
  const positions = new Float32Array(vertexCount * 2);
  const offsets = new Uint32Array(paths.length + 1);
  const tracks = new Uint32Array(paths.length);
  let row = 0;
  paths.forEach((path, index) => {
    offsets[index] = row;
    tracks[index] = path.track;
    positions.set(set.lngLat.subarray(path.first * 2, (path.last + 1) * 2), row * 2);
    row += path.last - path.first + 1;
  });
  offsets[paths.length] = row;
  return {trackCount: paths.length, vertexCount, positions, offsets, tracks};
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** `14 Apr` for a day number since 1 January (0-based). */
export function formatYearDay(day: number): string {
  const date = new Date(YEAR_START_MILLISECONDS + Math.floor(day) * SECONDS_PER_DAY * 1000);
  return `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]}`;
}

/** `14 Apr` for seconds since 1 January. */
export function formatYearSeconds(seconds: number): string {
  return formatYearDay(seconds / SECONDS_PER_DAY);
}

/** Day of the year (0-based) of the first of a month, 0-based month. */
export function monthStartDay(month: number): number {
  return Math.round(
    (Date.UTC(2024, month, 1) - YEAR_START_MILLISECONDS) / (SECONDS_PER_DAY * 1000)
  );
}

/** Day of the folded year (0-based) on which each month starts. */
export const MONTH_START_DAYS = Array.from({length: 12}, (_, month) => monthStartDay(month));

/** Seasons used by the flyway and timing scenes, as day-of-year ranges. */
export const MIGRATION_SEASONS = {
  year: {label: 'Whole year', days: [0, 366]},
  spring: {label: 'Spring (1 Mar to 15 Jun)', days: [60, 167]},
  breeding: {label: 'Breeding (15 Jun to 31 Jul)', days: [167, 213]},
  autumn: {label: 'Autumn (1 Aug to 30 Nov)', days: [213, 335]},
  winter: {label: 'Winter (1 Dec to 28 Feb)', days: [335, 366]}
} as const;
export type MigrationSeason = keyof typeof MIGRATION_SEASONS;

/**
 * Well-known stopover and wintering areas, used to name a site by its nearest landmark within 150 km. The
 * coordinates are approximate area centers, good to a few tens of kilometers.
 */
const PLACES: readonly {name: string; lng: number; lat: number}[] = [
  {name: 'Wadden Sea (Texel)', lng: 4.9, lat: 53.1},
  {name: 'Rhine-Meuse-Scheldt delta', lng: 4.2, lat: 51.7},
  {name: 'Flanders and the Scheldt basin', lng: 3.6, lat: 51.0},
  {name: 'Vendee coast (France)', lng: -1.8, lat: 46.7},
  {name: 'Atlantic plains of Morocco', lng: -7.0, lat: 32.5},
  {name: 'Sahel of southern Mauritania and Mali', lng: -9.0, lat: 15.5},
  {name: 'Bay of Somme', lng: 1.6, lat: 50.2},
  {name: 'Camargue', lng: 4.6, lat: 43.5},
  {name: 'Ebro delta', lng: 0.8, lat: 40.7},
  {name: 'Tagus estuary', lng: -9.0, lat: 38.8},
  {name: 'Donana marshes', lng: -6.4, lat: 37.0},
  {name: 'Strait of Gibraltar', lng: -5.5, lat: 36.0},
  {name: 'Merja Zerga (Morocco)', lng: -6.3, lat: 34.8},
  {name: 'Sous-Massa (Morocco)', lng: -9.6, lat: 30.0},
  {name: "Banc d'Arguin (Mauritania)", lng: -16.3, lat: 19.9},
  {name: 'Senegal river delta', lng: -16.3, lat: 16.4},
  {name: 'Saloum and Gambia estuaries', lng: -16.6, lat: 13.5},
  {name: 'Bijagos (Guinea-Bissau)', lng: -15.8, lat: 11.3},
  {name: 'Inner Niger Delta (Mali)', lng: -4.5, lat: 14.5},
  {name: 'Lake Chad', lng: 14.0, lat: 13.0},
  {name: 'Lake Volta (Ghana)', lng: 0.0, lat: 7.5},
  {name: 'Hula Valley (Israel)', lng: 35.6, lat: 33.1},
  {name: 'Burgas lakes (Bulgaria)', lng: 27.4, lat: 42.5},
  {name: 'Po delta (Italy)', lng: 12.3, lat: 44.9},
  {name: 'Gulf of Tunis', lng: 10.3, lat: 36.9}
];

/** Great-circle distance in kilometers (haversine). */
export function distanceKilometers(lng0: number, lat0: number, lng1: number, lat1: number): number {
  const radians = Math.PI / 180;
  const dLat = (lat1 - lat0) * radians;
  const dLng = (lng1 - lng0) * radians;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat0 * radians) * Math.cos(lat1 * radians) * Math.sin(dLng / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(h));
}

/** Name of the nearest landmark within 150 km, else coordinates. */
export function nameSite(lng: number, lat: number): string {
  let best = '';
  let bestDistance = 150;
  for (const place of PLACES) {
    const distance = distanceKilometers(lng, lat, place.lng, place.lat);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = place.name;
    }
  }
  const coordinates = `${Math.abs(lat).toFixed(1)}${lat >= 0 ? 'N' : 'S'} ${Math.abs(lng).toFixed(1)}${lng >= 0 ? 'E' : 'W'}`;
  return best ? `${best} (${coordinates})` : coordinates;
}
