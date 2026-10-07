// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {HURRICANE_CLASS} from '../../cartography/hue-registry';
import {LocalMetricProjection} from '../../engine/projection';
import {createAzimuthalEquidistant, createTrackSet, type TrackSet} from '../movement/b12-tracks';

/**
 * Shared helpers of the hurricane scenes: the IBTrACS storms as a {@link TrackSet} in the planar
 * system and with the time base each scene needs, the Saffir-Simpson palette and storm formatters.
 */

/** Seconds from the Unix epoch to the dataset's `timeOrigin` (1980-01-01T00:00:00Z). */
export const IBTRACS_ORIGIN_UNIX_SECONDS = 315532800;

/** Saffir-Simpson class names in the order of the dataset's `category` column. */
export const HURRICANE_CATEGORIES = ['TD', 'TS', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5'];

/** Long names of the classes with their wind range (knots), for legends. */
export const HURRICANE_CATEGORY_LABELS = [
  'Tropical depression (under 34 kt)',
  'Tropical storm (34-63 kt)',
  'Category 1 (64-82 kt)',
  'Category 2 (83-95 kt)',
  'Category 3 (96-112 kt)',
  'Category 4 (113-136 kt)',
  'Category 5 (137 kt or more)'
];

/** One color per class, ordered by intensity; readable on light and dark basemaps. */
// The light member is suitable for the paper stories. Abyss scenes use their halo to keep these
// ordered class colours legible; the canonical table lives in the cartography registry.
export const HURRICANE_CATEGORY_COLORS = HURRICANE_CLASS.light;

/** Up to eight family colors (Okabe-Ito order without black). */
export const HURRICANE_FAMILY_COLORS: readonly (readonly [number, number, number, number])[] = [
  [86, 180, 233, 255],
  [240, 150, 30, 255],
  [0, 158, 115, 255],
  [204, 121, 167, 255],
  [240, 228, 66, 255],
  [0, 114, 178, 255],
  [213, 94, 0, 255],
  [150, 156, 168, 255]
];

/** Palette index (class) of a wind in knots, from the Saffir-Simpson wind ranges. */
export function getCategoryOfWind(knots: number): number {
  if (knots < 34) return 0;
  if (knots < 64) return 1;
  if (knots < 83) return 2;
  if (knots < 96) return 3;
  if (knots < 113) return 4;
  if (knots < 137) return 5;
  return 6;
}

/** Centre of the azimuthal-equidistant system used for track comparison. */
export const ATLANTIC_CENTER: readonly [number, number] = [-60, 28];

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** `12 Aug 2005` for seconds since 1980-01-01. */
export function formatStormDate(seconds: number): string {
  const date = new Date((IBTRACS_ORIGIN_UNIX_SECONDS + seconds) * 1000);
  return `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}

/** `12 Aug` for a day of the year (0 is 1 January of a non-leap year). */
export function formatDayOfYear(day: number): string {
  const date = new Date(Date.UTC(2001, 0, 1) + Math.floor(day) * 86400000);
  return `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]}`;
}

/** The storms of `ibtracs-north-atlantic` as tracks plus their per-vertex and per-storm columns. */
export type HurricaneTrackSet = TrackSet & {
  /** Sustained wind in knots per vertex. */
  wind: Float32Array;
  /** Minimum central pressure in hPa per vertex, 0 where missing. */
  pressure: Float32Array;
  /** Saffir-Simpson class per vertex (index into {@link HURRICANE_CATEGORIES}). */
  category: Uint32Array;
  /** IBTrACS distance to land per vertex in km. */
  distanceToLand: Float32Array;
  /** Season (year) per storm. */
  season: Uint16Array;
  /** Peak wind per storm in knots. */
  maxWind: Uint8Array;
  /** Peak class per storm. */
  maxCategory: Uint8Array;
  /** Peak class per storm as `uint32` for GPU palettes. */
  maxCategoryWords: Uint32Array;
  names: readonly string[];
  stormIds: readonly string[];
  /** Seconds since 1980-01-01 of each vertex (exact, not the scene's `timestamps`). */
  absoluteSeconds: Uint32Array;
  /** Seconds since 1980-01-01 of each storm's first fix. */
  startSeconds: Float64Array;
  /** Number of seasons from the first to the last storm (46 for 1980-2025). */
  seasonCount: number;
};

export type HurricaneTrackOptions = {
  /** `azimuthal` keeps true distances (families); `mercator` matches deck.gl meters (rasters). */
  projection: 'azimuthal' | 'mercator';
  /** Origin of the Mercator system; ignored for `azimuthal`. */
  origin?: readonly [number, number];
  /**
   * `storm`: seconds since each storm's first fix. `calendar`: hours since 1 January 00:00 UTC of
   * the year of each storm's first fix, so every season lies on one calendar.
   */
  timeBase: 'storm' | 'calendar';
};

/** Loads `ibtracs-north-atlantic` as a {@link HurricaneTrackSet}. */
export function loadHurricaneTracks(
  dataset: LoadedDataset,
  options: HurricaneTrackOptions
): HurricaneTrackSet {
  const lngLat = dataset.column<Float32Array>('vertices');
  const offsets = dataset.column<Uint32Array>('pathOffsets');
  const absoluteSeconds = dataset.column<Uint32Array>('timestamp');
  const trackCount = offsets.length - 1;
  const center = options.origin ?? ATLANTIC_CENTER;
  const projection =
    options.projection === 'azimuthal'
      ? createAzimuthalEquidistant(center)
      : (() => {
          const local = new LocalMetricProjection(center);
          return {
            project: (longitude: number, latitude: number) => local.project(longitude, latitude),
            unproject: (x: number, y: number) => local.unproject(x, y)
          };
        })();
  const positions = new Float32Array(lngLat.length);
  for (let vertex = 0; vertex < lngLat.length / 2; vertex++) {
    const [x, y] = projection.project(lngLat[vertex * 2], lngLat[vertex * 2 + 1]);
    positions[vertex * 2] = x;
    positions[vertex * 2 + 1] = y;
  }
  const startSeconds = new Float64Array(trackCount);
  const timestamps = new Float32Array(absoluteSeconds.length);
  for (let track = 0; track < trackCount; track++) {
    const first = absoluteSeconds[offsets[track]];
    startSeconds[track] = first;
    let base = first;
    let divisor = 1;
    if (options.timeBase === 'calendar') {
      const year = new Date((IBTRACS_ORIGIN_UNIX_SECONDS + first) * 1000).getUTCFullYear();
      base = Date.UTC(year, 0, 1) / 1000 - IBTRACS_ORIGIN_UNIX_SECONDS;
      divisor = 3600;
    }
    for (let vertex = offsets[track]; vertex < offsets[track + 1]; vertex++) {
      timestamps[vertex] = (absoluteSeconds[vertex] - base) / divisor;
    }
  }
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
  const maxCategory = dataset.column<Uint8Array>('maxCategory');
  const season = dataset.column<Uint16Array>('season');
  let firstSeason = Infinity;
  let lastSeason = -Infinity;
  for (const value of season) {
    firstSeason = Math.min(firstSeason, value);
    lastSeason = Math.max(lastSeason, value);
  }
  return {
    ...set,
    wind: Float32Array.from(dataset.column<Uint8Array>('wind')),
    pressure: Float32Array.from(dataset.column<Uint16Array>('pressure')),
    category: Uint32Array.from(dataset.column<Uint8Array>('category')),
    distanceToLand: Float32Array.from(dataset.column<Uint16Array>('dist2land')),
    season,
    maxWind: dataset.column<Uint8Array>('maxWind'),
    maxCategory,
    maxCategoryWords: Uint32Array.from(maxCategory),
    names: (dataset.properties.names as string[] | undefined) ?? [],
    stormIds: (dataset.properties.stormIds as string[] | undefined) ?? [],
    absoluteSeconds,
    startSeconds,
    seasonCount: lastSeason - firstSeason + 1
  };
}

/** `Katrina 2005`, or `Unnamed 1985` for storms that never got a name. */
export function getStormLabel(storms: HurricaneTrackSet, track: number): string {
  return `${storms.names[track] ?? 'Storm'} ${storms.season[track]}`;
}

/** `Katrina 2005: peak 150 kt (Cat 5), 24 Aug to 31 Aug`. */
export function describeStorm(storms: HurricaneTrackSet, track: number): string {
  const first = storms.offsets[track];
  const last = storms.offsets[track + 1] - 1;
  const peak = storms.maxWind[track];
  return `${getStormLabel(storms, track)}: peak ${peak} kt (${HURRICANE_CATEGORIES[storms.maxCategory[track]]}), ${formatStormDate(storms.absoluteSeconds[first])} to ${formatStormDate(storms.absoluteSeconds[last])}`;
}

/** Index of a named storm in a season, or -1. */
export function findStorm(storms: HurricaneTrackSet, name: string, season: number): number {
  for (let track = 0; track < storms.trackCount; track++) {
    if (storms.season[track] === season && storms.names[track].toLowerCase() === name) return track;
  }
  return -1;
}

/** Track with the longest planar path, used as a fallback selection. */
export function findLongestStorm(storms: HurricaneTrackSet): number {
  let best = 0;
  let bestLength = -1;
  for (let track = 0; track < storms.trackCount; track++) {
    let length = 0;
    for (let vertex = storms.offsets[track] + 1; vertex < storms.offsets[track + 1]; vertex++) {
      length += Math.hypot(
        storms.positions[vertex * 2] - storms.positions[vertex * 2 - 2],
        storms.positions[vertex * 2 + 1] - storms.positions[vertex * 2 - 1]
      );
    }
    if (length > bestLength) {
      bestLength = length;
      best = track;
    }
  }
  return best;
}
