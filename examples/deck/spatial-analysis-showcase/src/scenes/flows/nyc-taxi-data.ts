// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

/** Local metric origin shared by every NYC taxi scene (Midtown Manhattan). */
export const NYC_TAXI_ORIGIN: readonly [number, number] = [-73.9857, 40.7484];

/** Hours between the dataset time origin (Thu 1 Jan 2015, 00:00 local) and the last pickup. */
export const NYC_TAXI_HOURS = 38.75;

/** Brush and histogram domains of the crossfilter scene. */
export const HOUR_DOMAIN = [0, 39] as const;
export const DISTANCE_DOMAIN = [0, 15] as const;
export const FARE_DOMAIN = [0, 60] as const;
export const PASSENGER_DOMAIN = [1, 6] as const;

/** Color ranges of the point attributes, shared by layers and legends. */
export const TAXI_COLOR_RANGES = {
  fare: [4, 45],
  distance: [0.4, 12],
  time: [0, NYC_TAXI_HOURS],
  passengers: [1, 6]
} as const;

/** Named map brushes: center `[longitude, latitude]` and half size in meters. */
export const AREA_PRESETS = {
  midtown: {center: [-73.9857, 40.7549], halfWidth: 1800, halfHeight: 2600},
  downtown: {center: [-74.0105, 40.7114], halfWidth: 1800, halfHeight: 2200},
  jfk: {center: [-73.7822, 40.6446], halfWidth: 3600, halfHeight: 3000},
  laguardia: {center: [-73.8726, 40.7743], halfWidth: 1900, halfHeight: 1500}
} as const;

const WEEKDAYS = ['Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed'];

/** Formats hours since the time origin as `Thu 08:30` (day of week plus clock). */
export function formatTaxiTime(hours: number): string {
  const total = Math.max(0, Math.round(hours * 60));
  const day = Math.floor(total / 1440);
  const minuteOfDay = total % 1440;
  const clock = `${String(Math.floor(minuteOfDay / 60)).padStart(2, '0')}:${String(minuteOfDay % 60).padStart(2, '0')}`;
  return `${WEEKDAYS[day % 7]} ${1 + day} Jan ${clock}`;
}

/** Decoded trips: planar meters around {@link NYC_TAXI_ORIGIN} and plain float columns. */
export type TaxiTrips = {
  count: number;
  /** `x, y` meters of the pickup point per trip. */
  pickup: Float32Array;
  /** `x, y` meters of the dropoff point per trip. */
  dropoff: Float32Array;
  /** Pickup time in hours since the time origin. */
  pickupHour: Float32Array;
  /** Dropoff time in hours since the time origin (pickup plus the routed duration). */
  dropoffHour: Float32Array;
  /** Trip distance in miles. */
  distance: Float32Array;
  /** Metered fare in USD, no tips. */
  fare: Float32Array;
  /** Passengers, 1 to 6. */
  passengers: Uint32Array;
  /** Local metric projection. */
  project: (longitude: number, latitude: number) => [number, number];
  unproject: (x: number, y: number) => [number, number];
  /** Bounds `[minX, minY, maxX, maxY]` of every pickup and dropoff, in meters. */
  bounds: [number, number, number, number];
};

/** Decodes the quantized `poopdeck-nyc-taxi` columns (see its README). */
export function loadTaxiTrips(dataset: LoadedDataset): TaxiTrips {
  const projection = dataset.getProjection(NYC_TAXI_ORIGIN);
  const count = dataset.count;
  const quantBbox = (dataset.manifest.properties as {quantBbox: number[]}).quantBbox;
  const [west, south, east, north] = quantBbox;
  const originXY = dataset.column<Uint16Array>('origin');
  const destinationXY = dataset.column<Uint16Array>('destination');
  const pickup = new Float32Array(count * 2);
  const dropoff = new Float32Array(count * 2);
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const decode = (quantized: Uint16Array, out: Float32Array, row: number) => {
    const longitude = west + (quantized[row * 2] / 65535) * (east - west);
    const latitude = south + (quantized[row * 2 + 1] / 65535) * (north - south);
    const [x, y] = projection.project(longitude, latitude);
    out[row * 2] = x;
    out[row * 2 + 1] = y;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  };
  const pickupTimeSteps = dataset.column<Uint16Array>('pickupTime');
  const durationSteps = dataset.column<Uint8Array>('duration');
  const distanceSteps = dataset.column<Uint8Array>('distance');
  const fareSteps = dataset.column<Uint8Array>('fare');
  const passengerCounts = dataset.column<Uint8Array>('passengers');
  const pickupHour = new Float32Array(count);
  const dropoffHour = new Float32Array(count);
  const distance = new Float32Array(count);
  const fare = new Float32Array(count);
  const passengers = new Uint32Array(count);
  for (let row = 0; row < count; row++) {
    decode(originXY, pickup, row);
    decode(destinationXY, dropoff, row);
    pickupHour[row] = (pickupTimeSteps[row] * 4) / 3600;
    dropoffHour[row] = pickupHour[row] + (durationSteps[row] * 15) / 3600;
    distance[row] = distanceSteps[row] / 10;
    fare[row] = fareSteps[row] / 2;
    passengers[row] = passengerCounts[row];
  }
  return {
    count,
    pickup,
    dropoff,
    pickupHour,
    dropoffHour,
    distance,
    fare,
    passengers,
    project: (longitude, latitude) => projection.project(longitude, latitude),
    unproject: (x, y) => projection.unproject(x, y),
    bounds: [minX, minY, maxX, maxY]
  };
}

/** Counts `values` into `bins` equal bins over `[low, high)`, optionally for rows passing `keep`. */
export function countBins(
  values: ArrayLike<number>,
  low: number,
  high: number,
  bins: number
): Float64Array {
  const counts = new Float64Array(bins);
  const scale = bins / (high - low);
  for (let index = 0; index < values.length; index++) {
    const bin = Math.floor((values[index] - low) * scale);
    if (bin >= 0 && bin < bins) counts[bin]++;
  }
  return counts;
}

/** Hour of day (0 to 24, fractional) of every pickup, for cyclic colour: Thursday and Friday overlay. */
export function getTaxiHourOfDay(pickupHour: Float32Array): Float32Array {
  const hourOfDay = new Float32Array(pickupHour.length);
  for (let row = 0; row < pickupHour.length; row++) {
    hourOfDay[row] = pickupHour[row] % 24;
  }
  return hourOfDay;
}

/** Straight pickup-to-dropoff segments `x0, y0, x1, y1` in meters, one per trip (trip links). */
export function getTaxiTripSegments(trips: TaxiTrips): Float32Array {
  const segments = new Float32Array(trips.count * 4);
  for (let row = 0; row < trips.count; row++) {
    segments[row * 4] = trips.pickup[row * 2];
    segments[row * 4 + 1] = trips.pickup[row * 2 + 1];
    segments[row * 4 + 2] = trips.dropoff[row * 2];
    segments[row * 4 + 3] = trips.dropoff[row * 2 + 1];
  }
  return segments;
}
