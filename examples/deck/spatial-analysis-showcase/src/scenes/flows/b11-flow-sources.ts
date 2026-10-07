// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {LocalMetricProjection} from '../../engine/projection';
import {createZoneRaster, type ZoneRaster} from './b11-zone-raster';

/** Planar origin shared by every Chicago flow layer. */
export const CHICAGO_ORIGIN: [number, number] = [-87.68, 41.84];

/** Origin-destination rows of one dataset, ready to upload. */
export type FlowSource = {
  id: 'taxi' | 'commute';
  /** Number of source rows. */
  rowCount: number;
  zoneCount: number;
  zoneNames: readonly string[];
  /** Planar meters per zone. */
  centers: Float32Array;
  origin: Uint32Array;
  destination: Uint32Array;
  /** Pickup hour as float32 (taxi only). */
  hour: Float32Array | null;
  /** 0 weekday, 1 weekend (taxi only). */
  dayType: Uint8Array | null;
  /** Float32 weight per row by option value. */
  weights: Record<string, Float32Array>;
  /** Display unit and prefix per weight option. */
  weightUnits: Record<string, {unit: string; prefix: string}>;
  raster: ZoneRaster;
};

/** Reads the 2023 Chicago taxi flows between the 77 community areas, sliced by weekday and hour. */
export function readTaxiSource(
  taxi: LoadedDataset,
  areas: LoadedDataset,
  projection: LocalMetricProjection
): FlowSource {
  const centers = taxi.projectColumn('locations', projection.origin);
  const zoneNames = taxi.properties.areaNames as string[];
  const origin = taxi.column<Uint32Array>('hourlyOrigin');
  const destination = taxi.column<Uint32Array>('hourlyDestination');
  const count = taxi.column<Uint32Array>('hourlyCount');
  const fare = taxi.column<Float32Array>('hourlyMeanFare');
  const seconds = taxi.column<Float32Array>('hourlyMeanSeconds');
  const hourBytes = taxi.column<Uint8Array>('hourlyHour');
  const dayType = taxi.column<Uint8Array>('hourlyDaytype');
  const rowCount = count.length;
  const trips = new Float32Array(rowCount);
  const revenue = new Float32Array(rowCount);
  const rideHours = new Float32Array(rowCount);
  const hour = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    trips[row] = count[row];
    revenue[row] = count[row] * fare[row];
    rideHours[row] = (count[row] * seconds[row]) / 3600;
    hour[row] = hourBytes[row];
  }
  const raster = createZoneRaster(
    areas.geojson!,
    projection,
    zoneNames.length,
    properties => Number(properties.id) - 1,
    180
  );
  return {
    id: 'taxi',
    rowCount,
    zoneCount: zoneNames.length,
    zoneNames,
    centers,
    origin,
    destination,
    hour,
    dayType,
    weights: {trips, fare: revenue, duration: rideHours},
    weightUnits: {
      trips: {unit: 'trips', prefix: ''},
      fare: {unit: 'in fares', prefix: '$'},
      duration: {unit: 'ride hours', prefix: ''}
    },
    raster
  };
}

/** Reads the 2021 LODES home-to-work flows between Chicago census tracts. */
export function readCommuteSource(
  lodes: LoadedDataset,
  tracts: LoadedDataset,
  areaNames: readonly string[],
  projection: LocalMetricProjection
): FlowSource {
  const centers = lodes.projectColumn('locations', projection.origin);
  const features = tracts.geojson!.features;
  const zoneNames = features.map(feature => {
    const properties = feature.properties ?? {};
    const area = areaNames[Number(properties.communityArea) - 1] ?? 'Chicago';
    return `Tract ${String(properties.GEOID).slice(5)} (${area})`;
  });
  const jobs = Float32Array.from(lodes.column<Uint32Array>('count'));
  const low = Float32Array.from(lodes.column<Uint32Array>('earningsLow'));
  const middle = Float32Array.from(lodes.column<Uint32Array>('earningsMid'));
  const high = Float32Array.from(lodes.column<Uint32Array>('earningsHigh'));
  const raster = createZoneRaster(
    tracts.geojson!,
    projection,
    zoneNames.length,
    (properties, featureIndex) =>
      properties.index === undefined ? featureIndex : Number(properties.index),
    150
  );
  return {
    id: 'commute',
    rowCount: jobs.length,
    zoneCount: zoneNames.length,
    zoneNames,
    centers,
    origin: lodes.column<Uint32Array>('origin'),
    destination: lodes.column<Uint32Array>('destination'),
    hour: null,
    dayType: null,
    weights: {all: jobs, low, mid: middle, high},
    weightUnits: {
      all: {unit: 'jobs', prefix: ''},
      low: {unit: 'low-earning jobs', prefix: ''},
      mid: {unit: 'mid-earning jobs', prefix: ''},
      high: {unit: 'high-earning jobs', prefix: ''}
    },
    raster
  };
}
