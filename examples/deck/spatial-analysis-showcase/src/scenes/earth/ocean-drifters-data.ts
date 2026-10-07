// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {createTrackSet, type TrackSet} from '../movement/b12-tracks';

/**
 * Shared data helpers of the ocean-drifters scenes: the 2017 Global Drifter Program tracks
 * (`poopdeck-drifters`), the annual-mean ECCO current field derived from the modelled particles
 * (`poopdeck-ecco-currents`), regions, great-circle distance and formatters. Everything here is
 * plain typed arrays; the GPU work lives in the scene compute modules.
 */

/** Lead days compared with the model (day 0 is the release). */
export const LEAD_DAYS = 30;
/** Slots per track in the daily tables: days 0 to 30. */
export const LEAD_SLOTS = LEAD_DAYS + 1;
/** Seconds per day. */
export const DAY_SECONDS = 86400;
/** Mean Earth radius in meters. */
export const EARTH_RADIUS_METERS = 6371008.8;
/** Meters per degree of latitude on the mean sphere. */
export const METERS_PER_DEGREE = (Math.PI * EARTH_RADIUS_METERS) / 180;

/** The gridded current field: 0.5 degree cells, row 0 at the SOUTH edge once loaded. */
export const FIELD = {
  width: 720,
  height: 320,
  cellDegrees: 0.5,
  west: -180,
  south: -80,
  east: 180,
  north: 80,
  /** `[originX, originY, cellWidth, cellHeight]` as `getGPUParticleAdvectionParameterValues` wants. */
  extent: [-180, -80, 0.5, 0.5] as readonly [number, number, number, number]
} as const;
/** Cells in the current field. */
export const FIELD_CELLS = FIELD.width * FIELD.height;

/** Drifter tracks and the real daily positions the model is compared with. */
export type DrifterSet = TrackSet & {
  /** Track (release) count; a track may have several path pieces. */
  releaseCount: number;
  /** Piece to track index. */
  pieceTrack: Uint16Array;
  /** Days since the dataset time origin of the first fix of each track (the release). */
  releaseDays: Float32Array;
  /** `1` when the track starts after 5 January (new deployment or record), `0` when already at sea. */
  deployed: Uint8Array;
  /** `lon, lat` of the real drifter at release + 0..30 days, `releaseCount * 31` rows, NaN = no fix. */
  daily: Float32Array;
  /** `lon, lat` of each release. */
  release: Float32Array;
  /** Sea-surface temperature in deg C per vertex, NaN = none. */
  sst: Float32Array;
  /** Unix milliseconds of day 0 of the time axis. */
  timeOriginMs: number;
  /** Real daily path as day-major segments (`row = day * releaseCount + track`, `x0, y0, x1, y1`). */
  dailySegments: Float32Array;
  /** `1` where both ends of a daily segment exist. Same row layout as `dailySegments`. */
  dailySegmentValid: Uint8Array;
  /** Vertices of every piece clipped to the first 30 days after release, for line density. */
  firstMonthPositions: Float32Array;
  /** Piece offsets into `firstMonthPositions`. */
  firstMonthOffsets: Uint32Array;
};

/** Loads `poopdeck-drifters`. */
export function loadDrifters(dataset: LoadedDataset): DrifterSet {
  const pieceOffsets = dataset.column<Uint32Array>('pathOffsets');
  const lngLat = dataset.column<Float32Array>('vertices');
  const rawTimes = dataset.column<Uint32Array>('timestamp');
  const timesDays = new Float32Array(rawTimes.length);
  for (let vertex = 0; vertex < rawTimes.length; vertex++) {
    timesDays[vertex] = rawTimes[vertex] / DAY_SECONDS;
  }
  const rawSst = dataset.column<Uint8Array>('sst');
  const sstOffset = Number(dataset.properties.sstOffset ?? -5);
  const sstScale = Number(dataset.properties.sstScale ?? 0.2);
  const sst = new Float32Array(rawSst.length);
  for (let vertex = 0; vertex < rawSst.length; vertex++) {
    sst[vertex] = rawSst[vertex] === 255 ? Number.NaN : rawSst[vertex] * sstScale + sstOffset;
  }
  const pieceTrack = dataset.column<Uint16Array>('trackId');
  const rawRelease = dataset.column<Uint32Array>('releaseTime');
  const releaseCount = rawRelease.length;
  const releaseDays = new Float32Array(releaseCount);
  for (let track = 0; track < releaseCount; track++) {
    releaseDays[track] = rawRelease[track] / DAY_SECONDS;
  }
  const daily = dataset.column<Float32Array>('daily');
  const release = new Float32Array(releaseCount * 2);
  for (let track = 0; track < releaseCount; track++) {
    release[track * 2] = daily[track * LEAD_SLOTS * 2];
    release[track * 2 + 1] = daily[track * LEAD_SLOTS * 2 + 1];
  }

  const set = createTrackSet({
    offsets: pieceOffsets,
    positions: lngLat,
    lngLat,
    timestamps: timesDays,
    origin: [0, 0],
    project: (longitude, latitude) => [longitude, latitude],
    unproject: (x, y) => [x, y],
    drawInDegrees: true
  });

  // Real daily path, day-major so "the first d days" is one contiguous run of rows.
  const segmentSlots = LEAD_DAYS * releaseCount;
  const dailySegments = new Float32Array(segmentSlots * 4);
  const dailySegmentValid = new Uint8Array(segmentSlots);
  for (let day = 0; day < LEAD_DAYS; day++) {
    for (let track = 0; track < releaseCount; track++) {
      const a = (track * LEAD_SLOTS + day) * 2;
      const b = a + 2;
      const row = day * releaseCount + track;
      const finite =
        Number.isFinite(daily[a]) &&
        Number.isFinite(daily[a + 1]) &&
        Number.isFinite(daily[b]) &&
        Number.isFinite(daily[b + 1]) &&
        Math.abs(daily[b] - daily[a]) < 180;
      if (finite) {
        dailySegments.set([daily[a], daily[a + 1], daily[b], daily[b + 1]], row * 4);
        dailySegmentValid[row] = 1;
      } else {
        dailySegments.fill(Number.NaN, row * 4, row * 4 + 4);
      }
    }
  }

  // First 30 days of every piece, for GPULineDensity on the real tracks.
  const keptPositions: number[] = [];
  const keptOffsets: number[] = [0];
  for (let piece = 0; piece < pieceOffsets.length - 1; piece++) {
    const track = pieceTrack[piece];
    const limit = releaseDays[track] + LEAD_DAYS;
    for (let vertex = pieceOffsets[piece]; vertex < pieceOffsets[piece + 1]; vertex++) {
      if (timesDays[vertex] > limit) break;
      keptPositions.push(lngLat[vertex * 2], lngLat[vertex * 2 + 1]);
    }
    keptOffsets.push(keptPositions.length / 2);
  }

  return {
    ...set,
    releaseCount,
    pieceTrack,
    releaseDays,
    deployed: dataset.column<Uint8Array>('deployed'),
    daily,
    release,
    sst,
    timeOriginMs: Number(dataset.properties.timeOriginMs),
    dailySegments,
    dailySegmentValid,
    firstMonthPositions: Float32Array.from(keptPositions),
    firstMonthOffsets: Uint32Array.from(keptOffsets)
  };
}

/** The annual-mean current field of ECCO on the advection grid. */
export type CurrentField = {
  /** Interleaved `u, v` in degrees per day, row 0 at the south, NaN on cells without data. */
  holes: Float32Array;
  /** Same, with NaN replaced by 0 (land cells hold particles in place). */
  filled: Float32Array;
  /** Speed in m/s per cell, row 0 at the south, NaN without data. */
  speed: Float32Array;
  /** Kernel weight per cell (about the number of 3.5-day samples), south first. */
  weight: Float32Array;
  /** Peak mean speed in m/s. */
  peakSpeed: number;
  /** Number of cells with data. */
  validCells: number;
  /** 3.5-day samples that were binned. */
  samples: number;
};

/** Loads the field raster of `poopdeck-ecco-currents` and flips it to south-first rows. */
export function loadCurrentField(dataset: LoadedDataset): CurrentField {
  const raster = dataset.raster;
  if (!raster || raster.width !== FIELD.width || raster.height !== FIELD.height) {
    throw new Error('poopdeck-ecco-currents has no 720 x 320 current raster');
  }
  const values = raster.values as Float32Array;
  const planeSize = FIELD_CELLS;
  const holes = new Float32Array(planeSize * 2);
  const filled = new Float32Array(planeSize * 2);
  const speed = new Float32Array(planeSize);
  const weight = new Float32Array(planeSize);
  let peakSpeed = 0;
  let validCells = 0;
  for (let row = 0; row < FIELD.height; row++) {
    const sourceRow = FIELD.height - 1 - row;
    const latitude = FIELD.south + (row + 0.5) * FIELD.cellDegrees;
    const cosine = Math.cos((latitude * Math.PI) / 180);
    for (let column = 0; column < FIELD.width; column++) {
      const source = sourceRow * FIELD.width + column;
      const target = row * FIELD.width + column;
      const u = values[source];
      const v = values[planeSize + source];
      weight[target] = values[2 * planeSize + source];
      holes[target * 2] = u;
      holes[target * 2 + 1] = v;
      if (Number.isFinite(u) && Number.isFinite(v)) {
        filled[target * 2] = u;
        filled[target * 2 + 1] = v;
        speed[target] = (Math.hypot(u * cosine, v) * METERS_PER_DEGREE) / DAY_SECONDS;
        peakSpeed = Math.max(peakSpeed, speed[target]);
        validCells++;
      } else {
        speed[target] = Number.NaN;
      }
    }
  }
  const field = (dataset.properties.field ?? {}) as {segmentsBinned?: number};
  return {holes, filled, speed, weight, peakSpeed, validCells, samples: field.segmentsBinned ?? 0};
}

/** Cell index (south-first) of a longitude and latitude, or `-1` outside the grid. */
export function getFieldCell(longitude: number, latitude: number): number {
  const column = Math.floor((longitude - FIELD.west) / FIELD.cellDegrees);
  const row = Math.floor((latitude - FIELD.south) / FIELD.cellDegrees);
  if (column < 0 || column >= FIELD.width || row < 0 || row >= FIELD.height) return -1;
  return row * FIELD.width + column;
}

/** A named ocean region used to filter and group the comparison. */
export type OceanRegion = {
  id: string;
  label: string;
  /** Longitude and latitude test on a release position. */
  contains: (longitude: number, latitude: number) => boolean;
  /** Camera that frames the region. */
  view: {longitude: number; latitude: number; zoom: number};
};

/** Regions of the comparison. `all` is first. */
export const OCEAN_REGIONS: readonly OceanRegion[] = [
  {
    id: 'all',
    label: 'All oceans',
    contains: () => true,
    view: {longitude: -20, latitude: 15, zoom: 1.5}
  },
  {
    id: 'gulfStream',
    label: 'Gulf Stream',
    contains: (lon, lat) => lon >= -82 && lon <= -40 && lat >= 28 && lat <= 48,
    view: {longitude: -62, latitude: 38, zoom: 4.1}
  },
  {
    id: 'kuroshio',
    label: 'Kuroshio',
    contains: (lon, lat) => lon >= 125 && lon <= 180 && lat >= 20 && lat <= 45,
    view: {longitude: 152, latitude: 33, zoom: 3.9}
  },
  {
    id: 'agulhas',
    label: 'Agulhas',
    contains: (lon, lat) => lon >= 10 && lon <= 60 && lat >= -48 && lat <= -20,
    view: {longitude: 32, latitude: -36, zoom: 3.9}
  },
  {
    id: 'southern',
    label: 'Southern Ocean',
    contains: (_lon, lat) => lat < -45,
    view: {longitude: 60, latitude: -56, zoom: 2.4}
  },
  {
    id: 'tropicalPacific',
    label: 'Tropical Pacific',
    contains: (lon, lat) => Math.abs(lat) <= 15 && (lon >= 150 || lon <= -90),
    view: {longitude: -150, latitude: 0, zoom: 2.4}
  }
];

/** Great-circle distance in meters on the mean sphere (haversine). */
export function getGreatCircleMeters(
  longitudeA: number,
  latitudeA: number,
  longitudeB: number,
  latitudeB: number
): number {
  const toRadians = Math.PI / 180;
  const dLatitude = (latitudeB - latitudeA) * toRadians;
  const dLongitude = (longitudeB - longitudeA) * toRadians;
  const a =
    Math.sin(dLatitude / 2) ** 2 +
    Math.cos(latitudeA * toRadians) *
      Math.cos(latitudeB * toRadians) *
      Math.sin(dLongitude / 2) ** 2;
  return 2 * EARTH_RADIUS_METERS * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** Quantile (`fraction` 0 to 1) of a sorted array, linearly interpolated. NaN when empty. */
export function getSortedQuantile(sorted: ArrayLike<number>, count: number, fraction: number) {
  if (count <= 0) return Number.NaN;
  const position = fraction * (count - 1);
  const low = Math.floor(position);
  const high = Math.min(count - 1, low + 1);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

/** `12 Mar 2017` from days since the time origin. */
export function formatDriftDate(timeOriginMs: number, days: number): string {
  return new Date(timeOriginMs + days * DAY_SECONDS * 1000)
    .toUTCString()
    .replace(/^\w+, /, '')
    .replace(/ \d\d:\d\d:\d\d GMT$/, '');
}

/** `1,234 km` from meters. */
export function formatKilometers(meters: number): string {
  if (!Number.isFinite(meters)) return 'n/a';
  return `${Math.round(meters / 1000).toLocaleString('en-US')} km`;
}

/** `42°N 61°W` from a position. */
export function formatPosition(longitude: number, latitude: number): string {
  return `${Math.abs(latitude).toFixed(1)}°${latitude >= 0 ? 'N' : 'S'} ${Math.abs(longitude).toFixed(1)}°${longitude >= 0 ? 'E' : 'W'}`;
}
