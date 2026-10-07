// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {createTrackSet, type TrackSet} from '../movement/b12-tracks';

/**
 * Shared helpers of the satellite scenes: the SGP4 ground tracks of `celestrak-ground-tracks` in
 * contributor layout, with longitude/latitude degrees as the planar frame.
 *
 * Why degrees and not meters: the trajectory contributors are planar and have no antimeridian
 * handling, and a global dataset has no single local projection. The build script cuts every track
 * at the antimeridian, so tracks never wrap; longitude and latitude are then a valid planar frame for
 * interpolation (30 s samples are at most 0.3 degrees apart) and for the spherical modes of
 * `GPULineDensity` and `GPUOutlineGeometry`, and layers draw with `COORDINATE_SYSTEM.LNGLAT`.
 */

export const SATELLITE_DATASET_ID = 'celestrak-ground-tracks';

/** Group names, in the dataset's category order. */
export const SATELLITE_GROUPS = [
  'Stations',
  'Starlink (sample)',
  'GPS',
  'Weather',
  'Earth observation'
] as const;

/** One color per group (Okabe-Ito based), readable on light and dark basemaps. */
export const SATELLITE_GROUP_COLORS: readonly (readonly [number, number, number, number])[] = [
  [240, 150, 30, 255],
  [60, 170, 235, 255],
  [220, 80, 70, 255],
  [0, 170, 125, 255],
  [190, 130, 240, 255]
];

/** Per-satellite metadata from the manifest. */
export type SatelliteInfo = {
  name: string;
  norad: number;
  group: number;
  orbitType: number;
  inclination: number;
  meanAltitudeKm: number;
  periodMinutes: number;
  eccentricity: number;
  tleAgeDays: number;
  swathKm: number;
  sensor: string;
};

export type SatelliteTrackSet = TrackSet & {
  /** Geodetic altitude in meters, one row per vertex. */
  altitudes: Float32Array;
  /** Satellite index per track (a satellite has several tracks: they are cut at the antimeridian). */
  satelliteIndex: Uint32Array;
  /** Group index per track. */
  group: Uint32Array;
  /** Orbit type per track (0 LEO, 1 MEO, 2 HEO). */
  orbitType: Uint32Array;
  inclination: Float32Array;
  /** Mean altitude over the window in meters, per track. */
  meanAltitude: Float32Array;
  /** Nominal imaging swath width in km per track, 0 when none. */
  swathKm: Float32Array;
  satellites: readonly SatelliteInfo[];
  satelliteCount: number;
  /** Unix seconds of time zero. */
  timeOriginUnixSeconds: number;
  /** Seconds covered by the window. */
  durationSeconds: number;
};

/** Loads the satellite dataset as tracks in longitude/latitude degrees. */
export function loadSatelliteTracks(dataset: LoadedDataset): SatelliteTrackSet {
  const lngLat = dataset.column<Float32Array>('vertices');
  const set = createTrackSet({
    offsets: dataset.column<Uint32Array>('pathOffsets'),
    positions: lngLat,
    lngLat,
    timestamps: Float32Array.from(dataset.column<Uint32Array>('timestamp')),
    origin: [0, 0],
    project: (longitude, latitude) => [longitude, latitude],
    unproject: (x, y) => [x, y],
    drawInDegrees: true
  });
  const satellites = (dataset.properties.satellites as SatelliteInfo[] | undefined) ?? [];
  return {
    ...set,
    altitudes: dataset.column<Float32Array>('altitude'),
    satelliteIndex: dataset.column<Uint32Array>('satellite'),
    group: Uint32Array.from(dataset.column<Uint8Array>('group')),
    orbitType: Uint32Array.from(dataset.column<Uint8Array>('orbitType')),
    inclination: dataset.column<Float32Array>('inclination'),
    meanAltitude: dataset.column<Float32Array>('meanAltitude'),
    swathKm: dataset.column<Float32Array>('swathKm'),
    satellites,
    satelliteCount: satellites.length,
    timeOriginUnixSeconds: Number(dataset.properties.timeOriginUnixSeconds ?? 0),
    durationSeconds: Number(dataset.properties.durationSeconds ?? 10800)
  };
}

/** Tracks and their per-vertex columns restricted to a list of tracks, in contributor layout. */
export type TrackSubset = {
  /** Original track index of each subset track. */
  tracks: Uint32Array;
  trackCount: number;
  vertexCount: number;
  offsets: Uint32Array;
  /** Longitude/latitude per vertex. */
  positions: Float32Array;
  timestamps: Float32Array;
  altitudes: Float32Array;
  /** Satellite index per subset track. */
  satelliteIndex: Uint32Array;
  /** Satellite index per vertex (for drawing per-vertex geometry). */
  vertexSatellite: Uint32Array;
  swathKm: Float32Array;
};

/** Copies the given tracks into new packed arrays (contributor inputs are packed views). */
export function createTrackSubset(set: SatelliteTrackSet, tracks: readonly number[]): TrackSubset {
  let vertexCount = 0;
  for (const track of tracks) vertexCount += set.offsets[track + 1] - set.offsets[track];
  const offsets = new Uint32Array(tracks.length + 1);
  const positions = new Float32Array(vertexCount * 2);
  const timestamps = new Float32Array(vertexCount);
  const altitudes = new Float32Array(vertexCount);
  const satelliteIndex = new Uint32Array(tracks.length);
  const vertexSatellite = new Uint32Array(vertexCount);
  const swathKm = new Float32Array(tracks.length);
  let row = 0;
  tracks.forEach((track, index) => {
    const first = set.offsets[track];
    const last = set.offsets[track + 1];
    positions.set(set.positions.subarray(first * 2, last * 2), row * 2);
    timestamps.set(set.timestamps.subarray(first, last), row);
    altitudes.set(set.altitudes.subarray(first, last), row);
    satelliteIndex[index] = set.satelliteIndex[track];
    swathKm[index] = set.swathKm[track];
    vertexSatellite.fill(set.satelliteIndex[track], row, row + last - first);
    row += last - first;
    offsets[index + 1] = row;
  });
  return {
    tracks: Uint32Array.from(tracks),
    trackCount: tracks.length,
    vertexCount,
    offsets,
    positions,
    timestamps,
    altitudes,
    satelliteIndex,
    vertexSatellite,
    swathKm
  };
}

/** Indices of the tracks whose group is in `groups` (all tracks when `groups` is null). */
export function selectTracksByGroup(
  set: SatelliteTrackSet,
  groups: readonly number[] | null
): number[] {
  const tracks: number[] = [];
  for (let track = 0; track < set.trackCount; track++) {
    if (!groups || groups.includes(set.group[track])) tracks.push(track);
  }
  return tracks;
}

/** `HH:MM UTC` of a time in seconds since the dataset origin. */
export function formatSatelliteClock(set: SatelliteTrackSet, seconds: number): string {
  const date = new Date((set.timeOriginUnixSeconds + seconds) * 1000);
  const hours = String(date.getUTCHours()).padStart(2, '0');
  const minutes = String(date.getUTCMinutes()).padStart(2, '0');
  return `${hours}:${minutes} UTC`;
}

/** `+h:mm` elapsed since the window start. */
export function formatElapsed(seconds: number): string {
  const total = Math.round(seconds / 60);
  return `${Math.floor(total / 60)} h ${String(total % 60).padStart(2, '0')} min`;
}

/** Group palette in the 0-255 RGBA form the layers take. */
export function getGroupLegendEntries(): {
  color: readonly [number, number, number, number];
  label: string;
}[] {
  return SATELLITE_GROUPS.map((label, index) => ({color: SATELLITE_GROUP_COLORS[index], label}));
}

/** Mean Earth radius in meters. */
export const EARTH_RADIUS_METERS = 6371008.8;
