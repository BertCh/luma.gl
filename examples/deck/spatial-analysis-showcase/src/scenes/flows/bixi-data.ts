// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

/** Planar origin shared by every Montreal BIXI layer. */
export const MONTREAL_ORIGIN: [number, number] = [-73.58, 45.52];

/** Slots of the weekday and weekend profiles: `daytype * 24 + hour`. */
export const SLOT_COUNT = 48;

/** The BIXI August 2024 flows in the layout the scenes upload. */
export type BixiFlows = {
  stationCount: number;
  /** Zone count of the flow aggregation: every station plus one sentinel zone for rare pairs. */
  zoneCount: number;
  names: readonly string[];
  /** `[longitude, latitude]` per station. */
  lngLat: Float32Array;
  /** Planar meters per zone about {@link MONTREAL_ORIGIN}; the sentinel zone is NaN. */
  centers: Float32Array;
  /** Borough index per station. */
  borough: Uint8Array;
  boroughNames: readonly string[];
  /** Rides leaving and arriving per station (to or from another station). */
  departures: Uint32Array;
  arrivals: Uint32Array;
  /** Station pairs of the whole month, most frequent first. */
  pairs: {origin: Uint32Array; destination: Uint32Array; count: Float32Array};
  /**
   * Weekday/weekend x hour rows: the frequent pairs plus residual rows to and from the sentinel
   * zone, so per-station totals over all rows are exact.
   */
  slices: {
    rowCount: number;
    origin: Uint32Array;
    destination: Uint32Array;
    hour: Float32Array;
    dayType: Uint8Array;
    count: Float32Array;
  };
  /** Rides per hour of the month, 744 values from 2024-08-01 00:00 local. */
  monthHourly: Uint32Array;
  totalRides: number;
  sameStationRides: number;
  project: (longitude: number, latitude: number) => [number, number];
};

/** Reads the `bixi-flows` dataset. */
export function readBixiFlows(dataset: LoadedDataset): BixiFlows {
  const properties = dataset.properties as {
    stationNames: string[];
    stationCount: number;
    totalRides: number;
    sameStationRides: number;
  };
  const stationCount = properties.stationCount;
  const projection = dataset.getProjection(MONTREAL_ORIGIN);
  const stationCenters = dataset.projectColumn('locations', MONTREAL_ORIGIN);
  const centers = new Float32Array((stationCount + 1) * 2);
  centers.set(stationCenters.subarray(0, stationCount * 2));
  centers[stationCount * 2] = Number.NaN;
  centers[stationCount * 2 + 1] = Number.NaN;

  const monthOrigin = dataset.column<Uint16Array>('origin');
  const monthCount = dataset.column<Uint16Array>('count');
  const monthDestination = dataset.column<Uint16Array>('destination');
  const pairs = {
    origin: Uint32Array.from(monthOrigin),
    destination: Uint32Array.from(monthDestination),
    count: Float32Array.from(monthCount)
  };

  const hourlyOrigin = dataset.column<Uint16Array>('hourlyOrigin');
  const hourlyDestination = dataset.column<Uint16Array>('hourlyDestination');
  const hourlyHour = dataset.column<Uint8Array>('hourlyHour');
  const hourlyDayType = dataset.column<Uint8Array>('hourlyDaytype');
  const hourlyCount = dataset.column<Uint8Array>('hourlyCount');
  const residualOrigin = dataset.column<Uint16Array>('residualOrigin');
  const residualDestination = dataset.column<Uint16Array>('residualDestination');
  const residualHour = dataset.column<Uint8Array>('residualHour');
  const residualDayType = dataset.column<Uint8Array>('residualDaytype');
  const residualCount = dataset.column<Uint16Array>('residualCount');
  const hourlyRows = hourlyCount.length;
  const rowCount = hourlyRows + residualCount.length;
  const slices = {
    rowCount,
    origin: new Uint32Array(rowCount),
    destination: new Uint32Array(rowCount),
    hour: new Float32Array(rowCount),
    dayType: new Uint8Array(rowCount),
    count: new Float32Array(rowCount)
  };
  for (let row = 0; row < hourlyRows; row++) {
    slices.origin[row] = hourlyOrigin[row];
    slices.destination[row] = hourlyDestination[row];
    slices.hour[row] = hourlyHour[row];
    slices.dayType[row] = hourlyDayType[row];
    slices.count[row] = hourlyCount[row];
  }
  for (let row = 0; row < residualCount.length; row++) {
    const target = hourlyRows + row;
    slices.origin[target] = residualOrigin[row];
    slices.destination[target] = residualDestination[row];
    slices.hour[target] = residualHour[row];
    slices.dayType[target] = residualDayType[row];
    slices.count[target] = residualCount[row];
  }
  return {
    stationCount,
    zoneCount: stationCount + 1,
    names: properties.stationNames,
    lngLat: dataset.column<Float32Array>('locations'),
    centers,
    borough: dataset.column<Uint8Array>('stationBorough'),
    boroughNames: dataset.categories('stationBorough'),
    departures: dataset.column<Uint32Array>('stationDepartures'),
    arrivals: dataset.column<Uint32Array>('stationArrivals'),
    pairs,
    slices,
    monthHourly: dataset.column<Uint32Array>('monthHourly'),
    totalRides: properties.totalRides,
    sameStationRides: properties.sameStationRides,
    project: (longitude, latitude) => projection.project(longitude, latitude)
  };
}

/**
 * Rides leaving and arriving per station and slot (`daytype * 24 + hour`), from the slice rows.
 * Rows are `[station][slot]`; the sentinel row (index `stationCount`) is left out.
 */
export function buildStationProfiles(flows: BixiFlows): {
  out: Float32Array;
  incoming: Float32Array;
} {
  const out = new Float32Array(flows.stationCount * SLOT_COUNT);
  const incoming = new Float32Array(flows.stationCount * SLOT_COUNT);
  const {slices, stationCount} = flows;
  for (let row = 0; row < slices.rowCount; row++) {
    const slot = slices.dayType[row] * 24 + slices.hour[row];
    const origin = slices.origin[row];
    const destination = slices.destination[row];
    if (origin < stationCount) out[origin * SLOT_COUNT + slot] += slices.count[row];
    if (destination < stationCount) incoming[destination * SLOT_COUNT + slot] += slices.count[row];
  }
  return {out, incoming};
}

/** `HH:MM` for a fractional hour of day. */
export function formatHourOfDay(hour: number): string {
  const wrapped = ((hour % 24) + 24) % 24;
  const whole = Math.floor(wrapped);
  const minutes = Math.round((wrapped - whole) * 60);
  return `${String(minutes === 60 ? (whole + 1) % 24 : whole).padStart(2, '0')}:${String(minutes === 60 ? 0 : minutes).padStart(2, '0')}`;
}

/** Compact number: 1.2k, 15k, 1.9M. */
export function formatCompact(value: number): string {
  const magnitude = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  if (magnitude >= 1e6) return `${sign}${(magnitude / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e3) return `${sign}${(magnitude / 1e3).toFixed(magnitude >= 1e4 ? 0 : 1)}k`;
  return `${sign}${magnitude.toFixed(0)}`;
}

/** Shortens a station name for chart labels. */
export function shortStationName(name: string, length = 16): string {
  const trimmed = name.replace(/^Métro /, 'M. ');
  return trimmed.length > length ? `${trimmed.slice(0, length - 1)}…` : trimmed;
}

/** Index of the station nearest to a pixel within `maxPixels`, or -1. */
export function findStationNearPixel(
  viewport: {project: (coordinate: number[]) => number[]} | null,
  lngLat: Float32Array,
  stationCount: number,
  pixel: readonly [number, number],
  maxPixels = 12
): number {
  if (!viewport) return -1;
  let best = -1;
  let bestDistance = maxPixels * maxPixels;
  for (let station = 0; station < stationCount; station++) {
    const [x, y] = viewport.project([lngLat[station * 2], lngLat[station * 2 + 1]]);
    const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
    if (squared < bestDistance) {
      bestDistance = squared;
      best = station;
    }
  }
  return best;
}
