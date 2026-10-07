// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  SHIPPING_GATES,
  SHIPPING_TYPE_GROUP,
  type GateEdges,
  type ShippingTracks
} from './us-shipping.tracks';

/**
 * CPU summaries of the small results the GPU graphs read back and of the tracks themselves:
 * hourly gate crossings by direction, the fix cadence, the share of vessels per group in a
 * window of the map, and the day's activity curve. Everything here works on arrays of at most a
 * few hundred thousand rows and runs once per readback, never per frame.
 */

const EARTH_RADIUS_KILOMETERS = 6371.0088;
const DEGREES = Math.PI / 180;

/** Great-circle distance in kilometres between two `[longitude, latitude]` points. */
export function getGreatCircleKilometers(
  from: readonly [number, number],
  to: readonly [number, number]
): number {
  const latitudeFrom = from[1] * DEGREES;
  const latitudeTo = to[1] * DEGREES;
  const sinLatitude = Math.sin((latitudeTo - latitudeFrom) / 2);
  const sinLongitude = Math.sin(((to[0] - from[0]) * DEGREES) / 2);
  const haversine =
    sinLatitude * sinLatitude +
    Math.cos(latitudeFrom) * Math.cos(latitudeTo) * sinLongitude * sinLongitude;
  return 2 * EARTH_RADIUS_KILOMETERS * Math.asin(Math.min(1, Math.sqrt(haversine)));
}

// ---------------------------------------------------------------------------------------------
// Gate crossings
// ---------------------------------------------------------------------------------------------

/** Hourly crossings of every gate, split by direction. */
export type GateCrossings = {
  /** `[gate][direction][hour]`, direction 0 is the gate's first direction. */
  hourly: Uint32Array[][];
  totals: Uint32Array;
  eventCount: number;
  /** Seconds between the two fixes either side of each crossing, one per counted crossing. */
  chordSeconds: Float32Array;
  /** Gate of each entry of {@link chordSeconds}. */
  chordGates: Uint32Array;
};

/**
 * Direction of an enter event: `0` when the track enters the gate rectangle from the side that
 * makes it travel in the gate's first direction, `1` otherwise. `x` and `y` are the event's
 * position in analysis meters.
 */
export function getCrossingDirection(edges: GateEdges, gate: number, x: number, y: number): 0 | 1 {
  const relativeX = x - edges.centers[gate * 2];
  const relativeY = y - edges.centers[gate * 2 + 1];
  const side = edges.axes[gate * 2] * relativeY - edges.axes[gate * 2 + 1] * relativeX;
  return side * edges.firstSigns[gate] > 0 ? 0 : 1;
}

/** Seconds between the fixes either side of `absoluteSeconds` on a track (0 outside its span). */
export function getBracketSeconds(
  tracks: ShippingTracks,
  track: number,
  absoluteSeconds: number
): number {
  let low = tracks.offsets[track];
  let high = tracks.offsets[track + 1] - 1;
  if (high <= low) return 0;
  // First fix at or after the crossing time.
  while (low < high) {
    const middle = (low + high) >> 1;
    if (tracks.timestamps[middle] >= absoluteSeconds) high = middle;
    else low = middle + 1;
  }
  if (low === tracks.offsets[track]) return 0;
  return tracks.timestamps[low] - tracks.timestamps[low - 1];
}

/**
 * Counts enter events per gate, hour and direction. `eventTimes` are relative to the first
 * timestamp of the event's track; the direction is the side of the gate rectangle the track
 * enters from. Also collects the time between the two fixes that bracket each crossing, because
 * the crossing time is interpolated along that chord.
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
  const chords: number[] = [];
  const chordGateList: number[] = [];
  for (let event = 0; event < events.count; event++) {
    if (events.types[event] !== 0) continue;
    const gate = events.zones[event];
    if (gate >= gateCount) continue;
    const absolute = tracks.trackStartTimes[events.tracks[event]] + events.times[event];
    const hour = Math.min(23, Math.max(0, Math.floor(absolute / 3600)));
    const direction = getCrossingDirection(
      edges,
      gate,
      events.positions[event * 2],
      events.positions[event * 2 + 1]
    );
    hourly[gate][direction][hour]++;
    totals[gate]++;
    const chord = getBracketSeconds(tracks, events.tracks[event], absolute);
    if (chord > 0) {
      chords.push(chord);
      chordGateList.push(gate);
    }
  }
  return {
    hourly,
    totals,
    eventCount: events.count,
    chordSeconds: Float32Array.from(chords),
    chordGates: Uint32Array.from(chordGateList)
  };
}

/** Median of the chords of one gate, or of every gate when `gate` is `-1`; `NaN` when none. */
export function getMedianChordSeconds(crossings: GateCrossings, gate: number): number {
  const values: number[] = [];
  for (let index = 0; index < crossings.chordSeconds.length; index++) {
    if (gate < 0 || crossings.chordGates[index] === gate)
      values.push(crossings.chordSeconds[index]);
  }
  return getMedian(values);
}

/** Median of a list of numbers, or `NaN` when it is empty. */
export function getMedian(values: ArrayLike<number>): number {
  if (values.length === 0) return Number.NaN;
  const sorted = Float64Array.from(values).sort();
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

// ---------------------------------------------------------------------------------------------
// Fix cadence and coverage
// ---------------------------------------------------------------------------------------------

/** The time between consecutive fixes of the same track, ascending. */
export type FixIntervals = {
  /** Seconds between consecutive fixes of one track, sorted ascending. */
  seconds: Float32Array;
  medianSeconds: number;
};

/** Collects every fix-to-fix interval of the day. */
export function getFixIntervals(tracks: ShippingTracks): FixIntervals {
  const values = new Float32Array(Math.max(0, tracks.vertexCount - tracks.trackCount));
  let count = 0;
  for (let track = 0; track < tracks.trackCount; track++) {
    for (let vertex = tracks.offsets[track] + 1; vertex < tracks.offsets[track + 1]; vertex++) {
      values[count++] = tracks.timestamps[vertex] - tracks.timestamps[vertex - 1];
    }
  }
  const seconds = values.subarray(0, count).sort();
  return {seconds, medianSeconds: count > 0 ? seconds[count >> 1] : Number.NaN};
}

/** Share of intervals longer than `thresholdSeconds` (the share a gap limit would call a gap). */
export function getShareAbove(intervals: FixIntervals, thresholdSeconds: number): number {
  const values = intervals.seconds;
  if (values.length === 0) return Number.NaN;
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (values[middle] > thresholdSeconds) high = middle;
    else low = middle + 1;
  }
  return (values.length - low) / values.length;
}

/**
 * Vessels with a track alive at each of `binCount` equal bins of the day: the activity curve
 * behind the time bar.
 */
export function getActivityHistogram(tracks: ShippingTracks, binCount: number): number[] {
  const histogram = new Array<number>(binCount).fill(0);
  const binSeconds = 86400 / binCount;
  for (let track = 0; track < tracks.trackCount; track++) {
    if (tracks.offsets[track + 1] <= tracks.offsets[track]) continue;
    const start = tracks.timestamps[tracks.offsets[track]];
    const end = tracks.timestamps[tracks.offsets[track + 1] - 1];
    const first = Math.max(0, Math.floor(start / binSeconds));
    const last = Math.min(binCount - 1, Math.floor(end / binSeconds));
    for (let bin = first; bin <= last; bin++) histogram[bin]++;
  }
  return histogram;
}

/**
 * The longitude west of which `share` of the fixes in a latitude band lie, counting only fixes
 * east of `minimumLongitude`: where terrestrial reception thins out offshore.
 */
export function getReceiverEdge(
  tracks: ShippingTracks,
  band: readonly [number, number],
  minimumLongitude: number,
  share = 0.99
): number {
  const longitudes: number[] = [];
  for (let vertex = 0; vertex < tracks.vertexCount; vertex++) {
    const latitude = tracks.lngLat[vertex * 2 + 1];
    const longitude = tracks.lngLat[vertex * 2];
    if (latitude >= band[0] && latitude <= band[1] && longitude > minimumLongitude) {
      longitudes.push(longitude);
    }
  }
  if (longitudes.length === 0) return Number.NaN;
  longitudes.sort((a, b) => a - b);
  return longitudes[Math.min(longitudes.length - 1, Math.floor(share * longitudes.length))];
}

// ---------------------------------------------------------------------------------------------
// Vessel groups
// ---------------------------------------------------------------------------------------------

/** Great-circle length, midpoint and vessel group of every segment of every track. */
export type SegmentTable = {
  kilometers: Float32Array;
  midLongitude: Float32Array;
  midLatitude: Float32Array;
  /** Vessel group (0 passenger, 1 cargo and tanker, 2 tug and tow, 3 other) per segment. */
  group: Uint8Array;
};

/** Builds the {@link SegmentTable} of the day. */
export function buildSegmentTable(tracks: ShippingTracks): SegmentTable {
  const count = tracks.segmentCount;
  const kilometers = new Float32Array(count);
  const midLongitude = new Float32Array(count);
  const midLatitude = new Float32Array(count);
  const group = new Uint8Array(count);
  for (let segment = 0; segment < count; segment++) {
    const longitudeFrom = tracks.segments[segment * 4];
    const latitudeFrom = tracks.segments[segment * 4 + 1];
    const longitudeTo = tracks.segments[segment * 4 + 2];
    const latitudeTo = tracks.segments[segment * 4 + 3];
    kilometers[segment] = getGreatCircleKilometers(
      [longitudeFrom, latitudeFrom],
      [longitudeTo, latitudeTo]
    );
    midLongitude[segment] = (longitudeFrom + longitudeTo) / 2;
    midLatitude[segment] = (latitudeFrom + latitudeTo) / 2;
    group[segment] = SHIPPING_TYPE_GROUP[tracks.category[tracks.segmentTracks[segment]]];
  }
  return {kilometers, midLongitude, midLatitude, group};
}

/**
 * Kilometres of track per vessel group over the day for the segments whose midpoint lies in
 * `bounds` (`[west, south, east, north]`).
 */
export function getGroupKilometers(
  table: SegmentTable,
  bounds: readonly [number, number, number, number]
): number[] {
  const totals = [0, 0, 0, 0];
  const [west, south, east, north] = bounds;
  for (let segment = 0; segment < table.kilometers.length; segment++) {
    const longitude = table.midLongitude[segment];
    const latitude = table.midLatitude[segment];
    if (longitude < west || longitude > east || latitude < south || latitude > north) continue;
    totals[table.group[segment]] += table.kilometers[segment];
  }
  return totals;
}

/** Tracks that come within `radiusKilometers` of a point, and how many of them are tugs and tows. */
export function summarizeTracksNear(
  tracks: ShippingTracks,
  center: readonly [number, number],
  radiusKilometers: number
): {tracks: number; towShare: number} {
  const seen = new Uint8Array(tracks.trackCount);
  const cosLatitude = Math.cos(center[1] * DEGREES);
  const kilometersPerDegree = EARTH_RADIUS_KILOMETERS * DEGREES;
  for (let track = 0; track < tracks.trackCount; track++) {
    for (let vertex = tracks.offsets[track]; vertex < tracks.offsets[track + 1]; vertex++) {
      const dx = (tracks.lngLat[vertex * 2] - center[0]) * cosLatitude * kilometersPerDegree;
      const dy = (tracks.lngLat[vertex * 2 + 1] - center[1]) * kilometersPerDegree;
      if (dx * dx + dy * dy <= radiusKilometers * radiusKilometers) {
        seen[track] = 1;
        break;
      }
    }
  }
  let count = 0;
  let tows = 0;
  for (let track = 0; track < tracks.trackCount; track++) {
    if (!seen[track]) continue;
    count++;
    if (SHIPPING_TYPE_GROUP[tracks.category[track]] === 2) tows++;
  }
  return {tracks: count, towShare: count > 0 ? tows / count : Number.NaN};
}
