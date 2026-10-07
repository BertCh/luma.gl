// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {formatPlaybackTime} from '../../engine/playback';
import {createTrackSet, type TrackSet} from '../movement/b12-tracks';

/**
 * Shared helpers of the three transit scenes: the Randstad morning-peak trips of
 * `poopdeck-gtfs-nl` as a track set, the mode palette, and the Dutch rail graph of
 * `gtfs-nl-rail-graph`. All of it is CPU preparation that runs once while a scene is created.
 */

/** Window length of `poopdeck-gtfs-nl` in seconds (07:00-09:00 local time). */
export const TRANSIT_WINDOW_SECONDS = 7200;
/** Local clock time of the window start, in seconds after midnight (07:00 CEST). */
export const TRANSIT_WINDOW_START_LOCAL = 7 * 3600;

/** Modes in the dataset's `route_type` category order. */
export const TRANSIT_MODES = ['tram', 'bus', 'metro', 'rail', 'ferry'] as const;
export type TransitMode = (typeof TRANSIT_MODES)[number];

export const TRANSIT_MODE_LABELS: Record<TransitMode, string> = {
  tram: 'Tram',
  bus: 'Bus',
  metro: 'Metro',
  rail: 'Train',
  ferry: 'Ferry'
};

/** One color per mode, readable on light and dark basemaps (Okabe-Ito based). */
export const TRANSIT_MODE_COLORS: readonly (readonly [number, number, number, number])[] = [
  [0, 158, 115, 255],
  [86, 180, 233, 255],
  [204, 121, 167, 255],
  [240, 150, 30, 255],
  [170, 120, 220, 255]
];

/** Legend entries of the modes. */
export function getTransitLegendEntries() {
  return TRANSIT_MODES.map((mode, index) => ({
    color: TRANSIT_MODE_COLORS[index],
    label: TRANSIT_MODE_LABELS[mode]
  }));
}

/** `HH:MM` local time (CEST) of seconds since the start of the window. */
export function formatTransitClock(seconds: number): string {
  return formatPlaybackTime.clock(seconds + TRANSIT_WINDOW_START_LOCAL);
}

/** Scheduled trips in contributor layout with the per-trip columns the scenes use. */
export type TransitTrips = TrackSet & {
  /** Mode index per trip (see {@link TRANSIT_MODES}). */
  mode: Uint32Array;
  /** Index into `routeNames` per trip. */
  routeName: Uint16Array;
  routeNames: readonly string[];
  /** Dense id of the (mode, route name) pair per trip: the "line" of a trip. */
  line: Uint32Array;
  /** Mode index and display name of each line id. */
  lines: readonly {mode: number; name: string; trips: number}[];
  /** First and last time of each trip in seconds since the window start. */
  startTime: Uint32Array;
  endTime: Uint32Array;
  modeTripCounts: readonly number[];
};

/** Loads `poopdeck-gtfs-nl` as {@link TransitTrips}. */
export function loadTransitTrips(dataset: LoadedDataset): TransitTrips {
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const set = createTrackSet({
    offsets: dataset.column<Uint32Array>('pathOffsets'),
    positions: dataset.projectColumn('vertices', origin),
    lngLat: dataset.column<Float32Array>('vertices'),
    timestamps: Float32Array.from(dataset.column<Uint32Array>('timestamp')),
    origin,
    project: (longitude, latitude) => projection.project(longitude, latitude),
    unproject: (x, y) => projection.unproject(x, y),
    drawInDegrees: false
  });
  const modeNames = dataset.categories('route_type');
  const modeRemap = Uint32Array.from(modeNames, name => {
    const index = TRANSIT_MODES.indexOf(name as TransitMode);
    return index < 0 ? TRANSIT_MODES.indexOf('bus') : index;
  });
  const mode = Uint32Array.from(dataset.column<Uint8Array>('route_type'), code => modeRemap[code]);
  const routeName = dataset.column<Uint16Array>('route_short_name');
  const routeNames = dataset.categories('route_short_name');
  const lineIds = new Map<number, number>();
  const lines: {mode: number; name: string; trips: number}[] = [];
  const line = new Uint32Array(set.trackCount);
  const modeTripCounts = new Array<number>(TRANSIT_MODES.length).fill(0);
  for (let trip = 0; trip < set.trackCount; trip++) {
    const key = mode[trip] * 65536 + routeName[trip];
    let id = lineIds.get(key);
    if (id === undefined) {
      id = lines.length;
      lineIds.set(key, id);
      lines.push({mode: mode[trip], name: routeNames[routeName[trip]] || '(no name)', trips: 0});
    }
    lines[id].trips++;
    line[trip] = id;
    modeTripCounts[mode[trip]]++;
  }
  return {
    ...set,
    mode,
    routeName,
    routeNames,
    line,
    lines,
    startTime: dataset.column<Uint32Array>('startTime'),
    endTime: dataset.column<Uint32Array>('endTime'),
    modeTripCounts
  };
}

/** Vehicles in service per mode at every `stepSeconds` of the window, from the trip start and end times. */
export function countVehiclesInService(
  trips: TransitTrips,
  stepSeconds = 60
): {times: Float64Array; perMode: Float64Array[]; total: Float64Array} {
  const bins = Math.floor(TRANSIT_WINDOW_SECONDS / stepSeconds) + 1;
  const perMode = TRANSIT_MODES.map(() => new Float64Array(bins));
  for (let trip = 0; trip < trips.trackCount; trip++) {
    const first = Math.floor(trips.startTime[trip] / stepSeconds);
    const last = Math.min(bins - 1, Math.floor(trips.endTime[trip] / stepSeconds));
    const counts = perMode[trips.mode[trip]];
    for (let bin = first; bin <= last; bin++) counts[bin]++;
  }
  const total = new Float64Array(bins);
  const times = new Float64Array(bins);
  for (let bin = 0; bin < bins; bin++) {
    times[bin] = bin * stepSeconds;
    for (const counts of perMode) total[bin] += counts[bin];
  }
  return {times, perMode, total};
}

/** Train classes of `gtfs-nl-rail-graph.edgeClass`. */
export const RAIL_CLASS_LABELS = [
  'Intercity and international',
  'Express (Sneltrein)',
  'Stopping (Sprinter, Stoptrein)'
] as const;

/** The Dutch rail graph in CSR layout with drawing tables, planar meters around `origin`. */
export type RailGraph = {
  origin: readonly [number, number];
  project: (longitude: number, latitude: number) => [number, number];
  unproject: (x: number, y: number) => [number, number];
  nodeCount: number;
  edgeCount: number;
  names: readonly string[];
  /** `x, y` meters per station. */
  nodePositions: Float32Array;
  /** `lng, lat` degrees per station. */
  nodeLngLat: Float32Array;
  /** CSR row offsets (`nodeCount + 1`). */
  offsets: Uint32Array;
  targets: Uint32Array;
  sources: Uint32Array;
  /** Train class per edge. */
  edgeClass: Uint8Array;
  /** Median scheduled in-vehicle seconds per edge. */
  travelTime: Float32Array;
  /** Trains per local hour per edge: `edgeCount * 24`. */
  tripsPerHour: Uint16Array;
  /** `x0, y0, x1, y1` per edge. */
  segments: Float32Array;
  /** Planar `[minX, minY, maxX, maxY]` of the stations. */
  bounds: [number, number, number, number];
};

/** Loads `gtfs-nl-rail-graph` as a {@link RailGraph}. */
export function loadRailGraph(dataset: LoadedDataset): RailGraph {
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const nodePositions = dataset.projectColumn('nodes', origin);
  const nodeCount = nodePositions.length / 2;
  const sources = dataset.column<Uint32Array>('edgeSource');
  const targets = dataset.column<Uint32Array>('edgeTarget');
  const edgeCount = sources.length;
  const segments = new Float32Array(edgeCount * 4);
  for (let edge = 0; edge < edgeCount; edge++) {
    segments.set(nodePositions.subarray(sources[edge] * 2, sources[edge] * 2 + 2), edge * 4);
    segments.set(nodePositions.subarray(targets[edge] * 2, targets[edge] * 2 + 2), edge * 4 + 2);
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let node = 0; node < nodeCount; node++) {
    minX = Math.min(minX, nodePositions[node * 2]);
    maxX = Math.max(maxX, nodePositions[node * 2]);
    minY = Math.min(minY, nodePositions[node * 2 + 1]);
    maxY = Math.max(maxY, nodePositions[node * 2 + 1]);
  }
  const tripsPerHour = dataset.column<Uint16Array>('edgeTripsPerHour');
  return {
    origin,
    project: (longitude, latitude) => projection.project(longitude, latitude),
    unproject: (x, y) => projection.unproject(x, y),
    nodeCount,
    edgeCount,
    names: (dataset.properties.stationNames as string[] | undefined) ?? [],
    nodePositions,
    nodeLngLat: dataset.column<Float32Array>('nodes'),
    offsets: dataset.column<Uint32Array>('nodeEdgeOffsets'),
    targets,
    sources,
    edgeClass: dataset.column<Uint8Array>('edgeClass'),
    travelTime: dataset.column<Float32Array>('edgeTravelTime'),
    tripsPerHour,
    segments,
    bounds: [minX, minY, maxX, maxY]
  };
}

/** Major stations used as the rows of the cost matrix (matched by name, missing ones are skipped). */
export const TRANSIT_HUB_NAMES = [
  'Amsterdam Centraal',
  'Rotterdam Centraal',
  'Den Haag Centraal',
  'Utrecht Centraal',
  'Schiphol Airport',
  'Eindhoven Centraal',
  'Arnhem Centraal',
  'Zwolle',
  'Groningen',
  'Leeuwarden',
  'Maastricht',
  'Breda',
  'Enschede',
  'Vlissingen'
] as const;

/** Index of the station called `name`, or -1. */
export function findStation(graph: RailGraph, name: string): number {
  return graph.names.indexOf(name);
}

/** Nearest station to a planar point and its distance in meters. */
export function findNearestStation(
  graph: RailGraph,
  x: number,
  y: number
): {node: number; distance: number} {
  let best = -1;
  let bestSquared = Infinity;
  for (let node = 0; node < graph.nodeCount; node++) {
    const dx = graph.nodePositions[node * 2] - x;
    const dy = graph.nodePositions[node * 2 + 1] - y;
    const squared = dx * dx + dy * dy;
    if (squared < bestSquared) {
      bestSquared = squared;
      best = node;
    }
  }
  return {node: best, distance: Math.sqrt(bestSquared)};
}
