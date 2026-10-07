// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {createAzimuthalEquidistant, createTrackSet, type TrackSet} from './b12-tracks';
import type {FlightColor} from './flight-corridors-layers';

/** Dataset id of the flight scenes. */
export const FLIGHT_DATASET_ID = 'poopdeck-adsb-paths';

/** Metres per second in one knot. */
export const METERS_PER_KNOT = 1852 / 3600;
/** Knots in one metre per second. */
export const KNOTS_PER_METER_SECOND = 3600 / 1852;

/** Compass class of a flight or step: the two the story is about, then the rest. */
export const FLIGHT_DIRECTIONS = ['eastbound', 'westbound', 'northbound', 'southbound'] as const;

/** Direction colors, shared by the layer palette and the legend. Colorblind-safe blue and orange. */
export const FLIGHT_DIRECTION_COLORS: readonly FlightColor[] = [
  [64, 160, 255, 255],
  [255, 140, 50, 255],
  [140, 210, 150, 255],
  [200, 150, 230, 255]
];

/** Flights in the layout the analysis contributors expect, plus per-vertex columns. */
export type FlightSet = TrackSet & {
  /** Altitude in metres per vertex (float32 so GPU graphs can read it). */
  altitude: Float32Array;
  /** Ground speed in knots per vertex, as reported by the aircraft. */
  reportedKnots: Float32Array;
  /** Direction class of the step that ends at each vertex (the first vertex repeats the second). */
  vertexDirection: Uint32Array;
  /** Direction class of each flight from its first to its last vertex. */
  trackDirection: Uint32Array;
  /** Great-circle distance in metres between a flight's first and last vertex. */
  trackChord: Float32Array;
  /** Highest altitude of each flight in metres. */
  trackPeak: Float32Array;
  /** `[west, south, east, north]` of the vertices. */
  bbox: readonly [number, number, number, number];
  /** UTC midnight of the day, Unix milliseconds. */
  timeOriginMs: number;
};

const EARTH_RADIUS = 6371008.8;
const RADIANS = Math.PI / 180;

/** Initial great-circle bearing in degrees clockwise from north. */
export function getBearing(lon0: number, lat0: number, lon1: number, lat1: number): number {
  const phi0 = lat0 * RADIANS;
  const phi1 = lat1 * RADIANS;
  const delta = (lon1 - lon0) * RADIANS;
  const y = Math.sin(delta) * Math.cos(phi1);
  const x = Math.cos(phi0) * Math.sin(phi1) - Math.sin(phi0) * Math.cos(phi1) * Math.cos(delta);
  return (((Math.atan2(y, x) / RADIANS) % 360) + 360) % 360;
}

/** Great-circle distance in metres (haversine). */
export function getGreatCircleDistance(
  lon0: number,
  lat0: number,
  lon1: number,
  lat1: number
): number {
  const phi0 = lat0 * RADIANS;
  const phi1 = lat1 * RADIANS;
  const a =
    Math.sin((phi1 - phi0) / 2) ** 2 +
    Math.cos(phi0) * Math.cos(phi1) * Math.sin(((lon1 - lon0) * RADIANS) / 2) ** 2;
  return 2 * EARTH_RADIUS * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** 0 east, 1 west, 2 north, 3 south, from a bearing in degrees. */
export function getDirectionClass(bearing: number): number {
  if (bearing >= 45 && bearing < 135) return 0;
  if (bearing >= 225 && bearing < 315) return 1;
  if (bearing >= 315 || bearing < 45) return 2;
  return 3;
}

/**
 * Loads the flight dataset. Analysis runs in azimuthal-equidistant metres around the middle of
 * the United States (distances from the center are exact; the tangential scale grows to about
 * 1.04 at the coasts), drawing uses longitude and latitude degrees.
 */
export function loadFlights(dataset: LoadedDataset): FlightSet {
  const lngLat = dataset.column<Float32Array>('vertices');
  const offsets = dataset.column<Uint32Array>('pathOffsets');
  const vertexCount = lngLat.length / 2;
  const trackCount = offsets.length - 1;
  const center: [number, number] = [-96, 38.5];
  const projection = createAzimuthalEquidistant(center);
  const positions = new Float32Array(lngLat.length);
  let west = 180;
  let south = 90;
  let east = -180;
  let north = -90;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const longitude = lngLat[vertex * 2];
    const latitude = lngLat[vertex * 2 + 1];
    const [x, y] = projection.project(longitude, latitude);
    positions[vertex * 2] = x;
    positions[vertex * 2 + 1] = y;
    west = Math.min(west, longitude);
    east = Math.max(east, longitude);
    south = Math.min(south, latitude);
    north = Math.max(north, latitude);
  }
  const set = createTrackSet({
    offsets,
    positions,
    lngLat,
    timestamps: Float32Array.from(dataset.column<Uint32Array>('timestamp')),
    origin: center,
    project: projection.project,
    unproject: projection.unproject,
    drawInDegrees: true
  });
  const altitude = Float32Array.from(dataset.column<Uint16Array>('altitude'));
  const reportedKnots = Float32Array.from(dataset.column<Uint8Array>('groundSpeed'), v => v * 3);
  const vertexDirection = new Uint32Array(vertexCount);
  const trackDirection = new Uint32Array(trackCount);
  const trackChord = new Float32Array(trackCount);
  const trackPeak = new Float32Array(trackCount);
  for (let track = 0; track < trackCount; track++) {
    const first = offsets[track];
    const last = offsets[track + 1] - 1;
    let peak = 0;
    for (let vertex = first; vertex <= last; vertex++) {
      peak = Math.max(peak, altitude[vertex]);
      if (vertex > first) {
        vertexDirection[vertex] = getDirectionClass(
          getBearing(
            lngLat[(vertex - 1) * 2],
            lngLat[(vertex - 1) * 2 + 1],
            lngLat[vertex * 2],
            lngLat[vertex * 2 + 1]
          )
        );
      }
    }
    vertexDirection[first] = vertexDirection[Math.min(last, first + 1)];
    trackPeak[track] = peak;
    trackDirection[track] = getDirectionClass(
      getBearing(lngLat[first * 2], lngLat[first * 2 + 1], lngLat[last * 2], lngLat[last * 2 + 1])
    );
    trackChord[track] = getGreatCircleDistance(
      lngLat[first * 2],
      lngLat[first * 2 + 1],
      lngLat[last * 2],
      lngLat[last * 2 + 1]
    );
  }
  return {
    ...set,
    altitude,
    reportedKnots,
    vertexDirection,
    trackDirection,
    trackChord,
    trackPeak,
    bbox: [west, south, east, north],
    timeOriginMs: (dataset.properties.timeOriginMs as number | undefined) ?? 1578268800000
  };
}

/** Median of the first `count` values of an array (sorted copy). 0 when empty. */
export function getMedian(values: ArrayLike<number>, count = values.length): number {
  if (count === 0) return 0;
  const sorted = Float64Array.from({length: count}, (_, index) => values[index]).sort();
  const middle = count >> 1;
  return count % 2 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
}

/** `HH:MM` of a UTC clock in seconds, then the same instant in US Eastern and Pacific standard time. */
export function formatClockUtc(seconds: number): string {
  const clock = (value: number) => {
    const total = Math.floor(((value % 86400) + 86400) % 86400);
    return `${String(Math.floor(total / 3600)).padStart(2, '0')}:${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}`;
  };
  return `${clock(seconds)} UTC (${clock(seconds - 5 * 3600)} EST, ${clock(seconds - 8 * 3600)} PST)`;
}
