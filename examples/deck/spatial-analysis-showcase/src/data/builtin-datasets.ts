// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {LocalMetricProjection, createSeededRandom} from '../engine/projection';
import type {DatasetPayload} from './catalog';
import {fetchJson} from './loaders';

/**
 * Datasets that are not shipped under `public/data/` but fetched from the deck.gl-data repository,
 * with a deterministic synthetic fallback for offline use, `?data=synthetic` and headless runs.
 * They produce the same `DatasetPayload` shape as a shipped manifest. They exist so the exemplar
 * scenes work before real datasets arrive; prefer a shipped dataset for new scenes.
 */
type BuiltinDataset = {
  load: (signal?: AbortSignal) => Promise<DatasetPayload>;
  synthetic: () => DatasetPayload;
};

const DECK_DATA_URL = 'https://raw.githubusercontent.com/visgl/deck.gl-data/master';

const NEW_YORK_ORIGIN: readonly [number, number] = [-73.985, 40.745];
const NEW_YORK_BBOX: [number, number, number, number] = [-74.02, 40.7, -73.93, 40.8];
const SAN_FRANCISCO_ORIGIN: readonly [number, number] = [-122.44, 37.76];
const SAN_FRANCISCO_BBOX: [number, number, number, number] = [-122.52, 37.7, -122.36, 37.83];

function lngLatFromMeters(origin: readonly [number, number], meters: ArrayLike<number>) {
  const projection = new LocalMetricProjection(origin);
  const lngLat = new Float32Array(meters.length);
  for (let index = 0; index + 1 < meters.length; index += 2) {
    const [longitude, latitude] = projection.unproject(meters[index], meters[index + 1]);
    lngLat[index] = longitude;
    lngLat[index + 1] = latitude;
  }
  return lngLat;
}

type TripRecord = {vendor: number; path: number[][]; timestamps: number[]};

function makeTripsPayload(
  lngLat: Float32Array,
  timestamps: Float32Array,
  pathOffsets: Uint32Array,
  vendors: Uint8Array,
  properties: Record<string, unknown>
): DatasetPayload {
  return {
    manifest: {
      id: 'ny-taxi-trips',
      version: 1,
      kind: 'trajectories',
      count: vendors.length,
      bbox: NEW_YORK_BBOX,
      crs: 'EPSG:4326',
      properties
    },
    columns: {
      position: {data: lngLat, components: 2},
      timestamp: {data: timestamps, components: 1, unit: 'seconds'},
      pathOffsets: {data: pathOffsets, components: 1},
      vendor: {data: vendors, components: 1}
    },
    geojson: null,
    raster: null
  };
}

const nyTaxiTrips: BuiltinDataset = {
  async load(signal) {
    const trips = await fetchJson<TripRecord[]>(
      `${DECK_DATA_URL}/examples/trips/trips-v7.json`,
      signal
    );
    const vertexCount = trips.reduce((total, trip) => total + trip.path.length, 0);
    const lngLat = new Float32Array(vertexCount * 2);
    const timestamps = new Float32Array(vertexCount);
    const pathOffsets = new Uint32Array(trips.length + 1);
    const vendors = new Uint8Array(trips.length);
    let vertex = 0;
    trips.forEach((trip, tripIndex) => {
      pathOffsets[tripIndex] = vertex;
      vendors[tripIndex] = trip.vendor;
      trip.path.forEach((coordinate, pathIndex) => {
        lngLat[vertex * 2] = coordinate[0];
        lngLat[vertex * 2 + 1] = coordinate[1];
        timestamps[vertex] = trip.timestamps[pathIndex];
        vertex++;
      });
    });
    pathOffsets[trips.length] = vertex;
    return makeTripsPayload(lngLat, timestamps, pathOffsets, vendors, {source: 'deck.gl-data'});
  },
  synthetic() {
    // Random walks on a Manhattan-like grid: 12 avenues x 100 streets, rotated 29 degrees.
    const columns = 12;
    const rows = 100;
    const spacingX = 260;
    const spacingY = 80;
    const rotation = -0.506;
    const getPoint = (column: number, row: number): [number, number] => {
      const x = (column - columns / 2) * spacingX;
      const y = (row - rows / 2) * spacingY;
      return [
        x * Math.cos(rotation) - y * Math.sin(rotation),
        x * Math.sin(rotation) + y * Math.cos(rotation)
      ];
    };
    const random = createSeededRandom(7);
    const meters: number[] = [];
    const times: number[] = [];
    const offsets = [0];
    const vendorList: number[] = [];
    for (let trip = 0; trip < 600; trip++) {
      let column = Math.floor(random() * columns);
      let row = Math.floor(random() * rows);
      let time = random() * 1800;
      const steps = 20 + Math.floor(random() * 40);
      for (let step = 0; step < steps; step++) {
        const [x, y] = getPoint(column, row);
        meters.push(x, y);
        times.push(time);
        if (random() < 0.5) {
          column = Math.max(0, Math.min(columns - 1, column + (random() < 0.5 ? -1 : 1)));
        } else {
          row = Math.max(0, Math.min(rows - 1, row + (random() < 0.5 ? -1 : 1)));
        }
        time += 6 + random() * 18;
      }
      offsets.push(meters.length / 2);
      vendorList.push(trip % 2);
    }
    return makeTripsPayload(
      lngLatFromMeters(NEW_YORK_ORIGIN, meters),
      Float32Array.from(times),
      Uint32Array.from(offsets),
      Uint8Array.from(vendorList),
      {source: 'synthetic'}
    );
  }
};

function makeParkingPayload(
  lngLat: Float32Array,
  spaces: Float32Array,
  properties: Record<string, unknown>
): DatasetPayload {
  return {
    manifest: {
      id: 'sf-bike-parking',
      version: 1,
      kind: 'points',
      count: spaces.length,
      bbox: SAN_FRANCISCO_BBOX,
      crs: 'EPSG:4326',
      properties
    },
    columns: {
      position: {data: lngLat, components: 2},
      spaces: {data: spaces, components: 1, unit: 'bicycle spaces'}
    },
    geojson: null,
    raster: null
  };
}

const sfBikeParking: BuiltinDataset = {
  async load(signal) {
    const rows = await fetchJson<{SPACES: number; COORDINATES: number[]}[]>(
      `${DECK_DATA_URL}/website/sf-bike-parking.json`,
      signal
    );
    const valid = rows.filter(
      row => Array.isArray(row.COORDINATES) && row.COORDINATES.every(Number.isFinite)
    );
    const lngLat = new Float32Array(valid.length * 2);
    const spaces = new Float32Array(valid.length);
    valid.forEach((row, index) => {
      lngLat[index * 2] = row.COORDINATES[0];
      lngLat[index * 2 + 1] = row.COORDINATES[1];
      spaces[index] = Number(row.SPACES) || 0;
    });
    return makeParkingPayload(lngLat, spaces, {source: 'deck.gl-data'});
  },
  synthetic() {
    const random = createSeededRandom(13);
    const count = 2500;
    const meters = new Float32Array(count * 2);
    const spaces = new Float32Array(count);
    const hotspots = [
      [2600, 2400],
      [500, 1200],
      [-1500, 0],
      [1800, -1800]
    ];
    for (let index = 0; index < count; index++) {
      const hotspot = hotspots[index % hotspots.length];
      const radius = 2200 * Math.sqrt(random());
      const angle = random() * Math.PI * 2;
      const stretch = index % 3 === 0 ? 1.6 : 0.6;
      meters[index * 2] = hotspot[0] + Math.cos(angle) * radius * stretch;
      meters[index * 2 + 1] = hotspot[1] + Math.sin(angle) * radius * stretch;
      spaces[index] = 2 * (1 + Math.floor(random() * random() * 12));
    }
    return makeParkingPayload(lngLatFromMeters(SAN_FRANCISCO_ORIGIN, meters), spaces, {
      source: 'synthetic'
    });
  }
};

/** Builtin remote datasets by id. */
export const BUILTIN_DATASETS: Record<string, BuiltinDataset | undefined> = {
  'ny-taxi-trips': nyTaxiTrips,
  'sf-bike-parking': sfBikeParking
};
