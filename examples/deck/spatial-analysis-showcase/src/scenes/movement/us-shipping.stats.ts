// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  describePlace,
  SHIPPING_GATES,
  SHIPPING_TYPES,
  type GateEdges,
  type ShippingTracks
} from './us-shipping.tracks';

/**
 * CPU summaries of the small results the GPU graphs read back: speed histograms by vessel type,
 * hourly gate crossings and the anchorage ranking. Everything here works on arrays of a few
 * thousand rows, never on the full tracks except the one-off speed histogram.
 */

/** One-knot bins from 0 to 30 knots; faster steps fall in the last bin. */
export const SPEED_BIN_COUNT = 30;
/** Steps below this speed count as stationary, not as a bin. */
export const MOVING_KNOTS = 0.5;

/** Vessel-hours per speed bin and type, and the stationary share, for one speed source. */
export type SpeedHistograms = {
  /** `[type][bin]` hours of vessel time. Row `SHIPPING_TYPES.length` is every type. */
  hours: Float64Array[];
  /** Hours of vessel time below {@link MOVING_KNOTS}, per type (and all). */
  stationaryHours: Float64Array;
};

/**
 * Time-weighted speed histograms: every step between two fixes adds its duration to the bin of its
 * speed. `speedKnots[vertex]` is the speed of the step that ends at `vertex`.
 */
export function buildSpeedHistograms(
  tracks: ShippingTracks,
  speedKnots: ArrayLike<number>
): SpeedHistograms {
  const typeCount = SHIPPING_TYPES.length;
  const hours = Array.from({length: typeCount + 1}, () => new Float64Array(SPEED_BIN_COUNT));
  const stationaryHours = new Float64Array(typeCount + 1);
  for (let track = 0; track < tracks.trackCount; track++) {
    const type = tracks.category[track];
    for (let vertex = tracks.offsets[track] + 1; vertex < tracks.offsets[track + 1]; vertex++) {
      const knots = speedKnots[vertex];
      if (!Number.isFinite(knots)) continue;
      const duration = (tracks.timestamps[vertex] - tracks.timestamps[vertex - 1]) / 3600;
      if (!(duration > 0)) continue;
      if (knots < MOVING_KNOTS) {
        stationaryHours[type] += duration;
        stationaryHours[typeCount] += duration;
        continue;
      }
      const bin = Math.min(SPEED_BIN_COUNT - 1, Math.floor(knots));
      hours[type][bin] += duration;
      hours[typeCount][bin] += duration;
    }
  }
  return {hours, stationaryHours};
}

/** Time-weighted median speed in knots of a histogram row (bin centers), or `NaN` when empty. */
export function getHistogramMedian(row: Float64Array): number {
  let total = 0;
  for (const value of row) total += value;
  if (total <= 0) return Number.NaN;
  let running = 0;
  for (let bin = 0; bin < row.length; bin++) {
    running += row[bin];
    if (running >= total / 2) {
      const before = running - row[bin];
      return bin + (row[bin] > 0 ? (total / 2 - before) / row[bin] : 0.5);
    }
  }
  return row.length;
}

/** Hourly crossings of every gate, split by direction. */
export type GateCrossings = {
  /** `[gate][direction][hour]`, direction 0 is the gate's first direction. */
  hourly: Uint32Array[][];
  totals: Uint32Array;
  eventCount: number;
};

/**
 * Counts enter events per gate, hour and direction. `eventTimes` are relative to the first
 * timestamp of the event's track; the direction is the side of the gate rectangle the track
 * enters from.
 */
export function countGateCrossings(
  tracks: ShippingTracks,
  edges: GateEdges,
  events: {
    count: number;
    tracks: Uint32Array;
    zones: Uint32Array;
    types: Uint32Array;
    times: Float32Array;
    positions: Float32Array;
  }
): GateCrossings {
  const gateCount = SHIPPING_GATES.length;
  const hourly = Array.from({length: gateCount}, () => [new Uint32Array(24), new Uint32Array(24)]);
  const totals = new Uint32Array(gateCount);
  for (let event = 0; event < events.count; event++) {
    if (events.types[event] !== 0) continue;
    const gate = events.zones[event];
    if (gate >= gateCount) continue;
    const absolute = tracks.trackStartTimes[events.tracks[event]] + events.times[event];
    const hour = Math.min(23, Math.max(0, Math.floor(absolute / 3600)));
    const x = events.positions[event * 2] - edges.centers[gate * 2];
    const y = events.positions[event * 2 + 1] - edges.centers[gate * 2 + 1];
    const side = edges.axes[gate * 2] * y - edges.axes[gate * 2 + 1] * x;
    const first = side * edges.firstSigns[gate] > 0;
    hourly[gate][first ? 0 : 1][hour]++;
    totals[gate]++;
  }
  return {hourly, totals, eventCount: events.count};
}

/** One ranked stopping place. */
export type Anchorage = {
  /** Cluster slot on the GPU. */
  cluster: number;
  longitude: number;
  latitude: number;
  /** Number of stops merged into the cluster. */
  stops: number;
  /** Distinct vessels with a stop in the cluster. */
  vessels: number;
  /** Summed stop duration in seconds. */
  dwellSeconds: number;
  place: string;
};

/**
 * Ranks clusters of stops by dwell or by number of stops. `labels[stop]` is the cluster of each
 * stop (or `noise`), `trackOfStop[stop]` its track.
 */
export function rankAnchorages(
  tracks: ShippingTracks,
  input: {
    stopCount: number;
    clusterCount: number;
    labels: Uint32Array;
    trackOfStop: Uint32Array;
    durations: Float32Array;
    centroids: Float32Array;
    noise: number;
  },
  rankBy: 'dwell' | 'stops'
): {ranked: Anchorage[]; noiseStops: number; clusteredStops: number} {
  const {clusterCount} = input;
  const dwell = new Float64Array(clusterCount);
  const stops = new Uint32Array(clusterCount);
  const vesselSets = Array.from({length: clusterCount}, () => new Set<number>());
  let noiseStops = 0;
  let clusteredStops = 0;
  for (let stop = 0; stop < input.stopCount; stop++) {
    const label = input.labels[stop];
    if (label === input.noise || label >= clusterCount) {
      noiseStops++;
      continue;
    }
    clusteredStops++;
    dwell[label] += input.durations[stop];
    stops[label]++;
    vesselSets[label].add(tracks.vesselIndex[input.trackOfStop[stop]]);
  }
  const order = Array.from({length: clusterCount}, (_, cluster) => cluster).sort((a, b) =>
    rankBy === 'dwell' ? dwell[b] - dwell[a] : stops[b] - stops[a]
  );
  const ranked = order.map(cluster => {
    const [longitude, latitude] = tracks.unproject(
      input.centroids[cluster * 2],
      input.centroids[cluster * 2 + 1]
    );
    return {
      cluster,
      longitude,
      latitude,
      stops: stops[cluster],
      vessels: vesselSets[cluster].size,
      dwellSeconds: dwell[cluster],
      place: describePlace(longitude, latitude)
    };
  });
  return {ranked, noiseStops, clusteredStops};
}

/** Compact hours, `412 h` or `3.2 h`. */
export function formatHours(seconds: number): string {
  const hours = seconds / 3600;
  return hours >= 100 ? `${Math.round(hours).toLocaleString('en-US')} h` : `${hours.toFixed(1)} h`;
}
