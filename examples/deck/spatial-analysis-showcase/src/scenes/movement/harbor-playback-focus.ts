// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU helpers of harbor-playback. The GPU does the work for every track at once; these functions
 * repeat it for one track so the story can show the arithmetic ("two fixes and a fraction"), pick
 * the vessels the steps look at, and summarise the day. Nothing here feeds the GPU graphs.
 *
 * - {@link findTrackBracket}: the same binary search and linear interpolation `GPUTrajectoryPlayhead`
 *   runs per track.
 * - {@link getFerryTracks}, {@link pickMovingFerry}, {@link pickTransitTrack}: which vessel a step
 *   follows, decided from the data by a rule (never by a typed track number).
 * - {@link getReportingProfile}: tracks reporting per quarter hour, the longest silence of the whole
 *   feed and the fix cadences.
 * - {@link summariseStops}: the stop list read back from the GPU, joined to vessel groups and to
 *   the NOAA anchorages.
 */

import {VESSEL_CATEGORIES, type VesselTrackSet} from './b12-tracks';
import {type ZoneSet, ZONE_KINDS} from './b12-zones';

// ---------------------------------------------------------------------------------------------
// Playhead arithmetic for one track
// ---------------------------------------------------------------------------------------------

/** The two fixes either side of a time in one track, and where the time sits between them. */
export type TrackBracket = {
  /** Row of the fix at or before the time. */
  startRow: number;
  /** Row of the next fix. */
  endRow: number;
  /** 0 at the first fix, 1 at the second. */
  fraction: number;
  startTime: number;
  endTime: number;
  /** Interpolated planar position in metres. */
  x: number;
  y: number;
  /** Speed of the segment in metres per second. */
  speed: number;
  /** Heading of the segment in radians, counterclockwise from +x (the contributor's convention). */
  heading: number;
};

/**
 * Finds the segment `[startRow, endRow]` with `time(startRow) <= time <= time(endRow)` by binary
 * search and interpolates the position linearly, as `GPUTrajectoryPlayhead` does for every track.
 * Returns `null` before the first fix, after the last, or for a track with a single fix.
 */
export function findTrackBracket(
  vessels: VesselTrackSet,
  track: number,
  time: number
): TrackBracket | null {
  const first = vessels.offsets[track];
  const last = vessels.offsets[track + 1] - 1;
  if (last <= first) return null;
  const times = vessels.timestamps;
  if (time < times[first] || time > times[last]) return null;
  // Largest row in [first, last - 1] whose time is at or before the playhead.
  let low = first;
  let high = last - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (times[middle] <= time) low = middle;
    else high = middle - 1;
  }
  const startRow = low;
  const endRow = low + 1;
  const interval = times[endRow] - times[startRow];
  const fraction = interval > 0 ? Math.min(1, Math.max(0, (time - times[startRow]) / interval)) : 0;
  const positions = vessels.positions;
  const x0 = positions[startRow * 2];
  const y0 = positions[startRow * 2 + 1];
  const x1 = positions[endRow * 2];
  const y1 = positions[endRow * 2 + 1];
  const length = Math.hypot(x1 - x0, y1 - y0);
  return {
    startRow,
    endRow,
    fraction,
    startTime: times[startRow],
    endTime: times[endRow],
    x: x0 + (x1 - x0) * fraction,
    y: y0 + (y1 - y0) * fraction,
    speed: interval > 0 ? length / interval : 0,
    heading: Math.atan2(y1 - y0, x1 - x0)
  };
}

/** Row index of the first segment of each track in the drawing tables (`trackCount + 1` entries). */
export function getSegmentRowStarts(vessels: VesselTrackSet): Uint32Array {
  const starts = new Uint32Array(vessels.trackCount + 1);
  for (let track = 0; track < vessels.trackCount; track++) {
    const length = vessels.offsets[track + 1] - vessels.offsets[track];
    starts[track + 1] = starts[track] + Math.max(0, length - 1);
  }
  return starts;
}

/** All tracks of the vessel that owns `track` (a vessel has several after a long silence), by time. */
export function getVesselTracks(vessels: VesselTrackSet, track: number): number[] {
  const owner = vessels.vesselIndex[track];
  const tracks: number[] = [];
  for (let candidate = 0; candidate < vessels.trackCount; candidate++) {
    if (vessels.vesselIndex[candidate] === owner) tracks.push(candidate);
  }
  return tracks.sort(
    (a, b) => vessels.timestamps[vessels.offsets[a]] - vessels.timestamps[vessels.offsets[b]]
  );
}

// ---------------------------------------------------------------------------------------------
// Which vessel does a step follow?
// ---------------------------------------------------------------------------------------------

/**
 * Passenger tracks of vessels at least `minimumLength` metres long that come within `radiusMeters`
 * of both ferry terminals (`terminalA` and `terminalB` are `[longitude, latitude]` from the
 * gazetteer): the ferries of the St. George to Whitehall run, which is the Staten Island Ferry's
 * route. The length floor keeps tour boats and water taxis that pass the slips out of the list.
 */
export function getFerryTracks(
  vessels: VesselTrackSet,
  terminalA: readonly [number, number],
  terminalB: readonly [number, number],
  radiusMeters = 250,
  minimumLength = 50
): number[] {
  const passenger = VESSEL_CATEGORIES.indexOf('passenger');
  const [ax, ay] = vessels.project(terminalA[0], terminalA[1]);
  const [bx, by] = vessels.project(terminalB[0], terminalB[1]);
  const limit = radiusMeters * radiusMeters;
  const tracks: number[] = [];
  for (let track = 0; track < vessels.trackCount; track++) {
    if (vessels.category[track] !== passenger || vessels.length[track] < minimumLength) continue;
    let nearA = false;
    let nearB = false;
    for (let row = vessels.offsets[track]; row < vessels.offsets[track + 1]; row++) {
      const x = vessels.positions[row * 2];
      const y = vessels.positions[row * 2 + 1];
      if ((x - ax) ** 2 + (y - ay) ** 2 <= limit) nearA = true;
      if ((x - bx) ** 2 + (y - by) ** 2 <= limit) nearB = true;
      if (nearA && nearB) break;
    }
    if (nearA && nearB) tracks.push(track);
  }
  return tracks;
}

/**
 * The fastest of `candidates` that has a bracket at `time` and is faster than `minimumSpeed`
 * (m/s): a ferry caught mid-crossing, where the two fixes are far enough apart to see. Returns
 * `null` when none is moving.
 */
export function pickMovingFerry(
  vessels: VesselTrackSet,
  candidates: readonly number[],
  time: number,
  minimumSpeed: number
): {track: number; bracket: TrackBracket} | null {
  let best: {track: number; bracket: TrackBracket} | null = null;
  for (const track of candidates) {
    const bracket = findTrackBracket(vessels, track, time);
    if (!bracket || bracket.speed < minimumSpeed) continue;
    if (!best || bracket.speed > best.bracket.speed) best = {track, bracket};
  }
  return best;
}

/**
 * A track that teaches resampling: a single passage of a passenger boat, tug or ship that moves,
 * waits for a while and keeps going (about half of its time below the stop speed), on a route that
 * does not double back. A ferry would not do: it runs the same crossing all day in one track, so
 * 32 samples would land on the same line. Returns -1 when no track qualifies.
 */
export function pickTransitTrack(vessels: VesselTrackSet, stoppedSpeed: number): number {
  const wanted = new Set(
    (['passenger', 'tug', 'cargo', 'tanker'] as const).map(name => VESSEL_CATEGORIES.indexOf(name))
  );
  let best = -1;
  let bestScore = -Infinity;
  for (let track = 0; track < vessels.trackCount; track++) {
    const first = vessels.offsets[track];
    const last = vessels.offsets[track + 1] - 1;
    const fixCount = last - first + 1;
    if (!wanted.has(vessels.category[track]) || fixCount < 40 || fixCount > 220) continue;
    let pathLength = 0;
    let slowTime = 0;
    for (let row = first; row < last; row++) {
      const length = Math.hypot(
        vessels.positions[row * 2 + 2] - vessels.positions[row * 2],
        vessels.positions[row * 2 + 3] - vessels.positions[row * 2 + 1]
      );
      const interval = vessels.timestamps[row + 1] - vessels.timestamps[row];
      pathLength += length;
      if (interval > 0 && length / interval < stoppedSpeed) slowTime += interval;
    }
    const duration = vessels.timestamps[last] - vessels.timestamps[first];
    if (duration <= 0 || pathLength < 8000 || pathLength > 45000) continue;
    const slowShare = slowTime / duration;
    const displacement = Math.hypot(
      vessels.positions[last * 2] - vessels.positions[first * 2],
      vessels.positions[last * 2 + 1] - vessels.positions[first * 2 + 1]
    );
    const straightness = displacement / pathLength;
    if (slowShare < 0.3 || slowShare > 0.7 || straightness < 0.5) continue;
    const score =
      -Math.abs(slowShare - 0.5) +
      0.5 * straightness +
      (vessels.category[track] === VESSEL_CATEGORIES.indexOf('passenger') ? 0.1 : 0);
    if (score > bestScore) {
      bestScore = score;
      best = track;
    }
  }
  return best;
}

/** `[west, south, east, north]` of a track's fixes in longitude and latitude. */
export function getTrackBounds(
  vessels: VesselTrackSet,
  track: number
): [number, number, number, number] {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (let row = vessels.offsets[track]; row < vessels.offsets[track + 1]; row++) {
    const [longitude, latitude] = vessels.unproject(
      vessels.positions[row * 2],
      vessels.positions[row * 2 + 1]
    );
    west = Math.min(west, longitude);
    east = Math.max(east, longitude);
    south = Math.min(south, latitude);
    north = Math.max(north, latitude);
  }
  return [west, south, east, north];
}

/**
 * Positions of `count` samples spaced equally in time over a track (first and last fix included),
 * mirroring `GPUTrajectoryResample` with `spacing: 'time'` for one track, and the size of the
 * densest pile: the most samples within `radiusMeters` of one sample. Used for the note on the
 * map ("13 of 32 samples at one berth"); the dots themselves come from the GPU.
 */
export function getTimeSamplePile(
  vessels: VesselTrackSet,
  track: number,
  count: number,
  radiusMeters: number
): {pileSize: number; x: number; y: number} {
  const first = vessels.timestamps[vessels.offsets[track]];
  const last = vessels.timestamps[vessels.offsets[track + 1] - 1];
  const points: [number, number][] = [];
  for (let sample = 0; sample < count; sample++) {
    const time = first + ((last - first) * sample) / Math.max(1, count - 1);
    const bracket = findTrackBracket(vessels, track, time);
    if (bracket) points.push([bracket.x, bracket.y]);
  }
  let best = {pileSize: 0, x: 0, y: 0};
  for (const [x, y] of points) {
    const members = points.filter(
      ([otherX, otherY]) => Math.hypot(otherX - x, otherY - y) <= radiusMeters
    );
    if (members.length > best.pileSize) {
      best = {
        pileSize: members.length,
        x: members.reduce((sum, member) => sum + member[0], 0) / members.length,
        y: members.reduce((sum, member) => sum + member[1], 0) / members.length
      };
    }
  }
  return best;
}

/** Planar length of a track's path in metres. */
export function getTrackPathLength(vessels: VesselTrackSet, track: number): number {
  let length = 0;
  for (let row = vessels.offsets[track]; row < vessels.offsets[track + 1] - 1; row++) {
    length += Math.hypot(
      vessels.positions[row * 2 + 2] - vessels.positions[row * 2],
      vessels.positions[row * 2 + 3] - vessels.positions[row * 2 + 1]
    );
  }
  return length;
}

// ---------------------------------------------------------------------------------------------
// The shape of the day
// ---------------------------------------------------------------------------------------------

/** How the day is covered: reporting per quarter hour, the longest silence, the fix cadences. */
export type ReportingProfile = {
  /** Tracks with at least one fix in each quarter hour of the UTC day (96 bins). */
  quarterHourTracks: Float64Array;
  /** Tracks with at least one fix in each UTC hour. */
  hourlyTracks: Float64Array;
  /** Start of the busiest UTC hour in seconds. */
  busiestHourStart: number;
  /** The longest stretch with no fix from any vessel, when it is longer than ten minutes. */
  silence: {start: number; length: number} | null;
  /** Median seconds between consecutive fixes of a track. */
  fixInterval: number;
  /** Median seconds between consecutive fixes while the vessel is below `stoppedSpeed`. */
  mooredInterval: number;
};

/** Reads the day's coverage from the tracks. `stoppedSpeed` is in metres per second. */
export function getReportingProfile(
  vessels: VesselTrackSet,
  stoppedSpeed: number
): ReportingProfile {
  const quarterHourTracks = new Float64Array(96);
  const hourlyTracks = new Float64Array(24);
  const intervals: number[] = [];
  const mooredIntervals: number[] = [];
  for (let track = 0; track < vessels.trackCount; track++) {
    let previousQuarter = -1;
    let previousHour = -1;
    for (let row = vessels.offsets[track]; row < vessels.offsets[track + 1]; row++) {
      const time = vessels.timestamps[row];
      const quarter = Math.min(95, Math.max(0, Math.floor(time / 900)));
      const hour = Math.min(23, Math.max(0, Math.floor(time / 3600)));
      if (quarter !== previousQuarter) quarterHourTracks[quarter]++;
      if (hour !== previousHour) hourlyTracks[hour]++;
      previousQuarter = quarter;
      previousHour = hour;
      if (row > vessels.offsets[track]) {
        const interval = time - vessels.timestamps[row - 1];
        if (interval > 0) {
          intervals.push(interval);
          const length = Math.hypot(
            vessels.positions[row * 2] - vessels.positions[row * 2 - 2],
            vessels.positions[row * 2 + 1] - vessels.positions[row * 2 - 1]
          );
          if (length / interval < stoppedSpeed) mooredIntervals.push(interval);
        }
      }
    }
  }
  let busiest = 0;
  for (let hour = 1; hour < 24; hour++) {
    if (hourlyTracks[hour] > hourlyTracks[busiest]) busiest = hour;
  }
  // The longest stretch in which no vessel reported at all.
  const allTimes = Float32Array.from(vessels.timestamps).sort();
  let silenceLength = 0;
  let silenceStart = 0;
  for (let index = 1; index < allTimes.length; index++) {
    const gap = allTimes[index] - allTimes[index - 1];
    if (gap > silenceLength) {
      silenceLength = gap;
      silenceStart = allTimes[index - 1];
    }
  }
  return {
    quarterHourTracks,
    hourlyTracks,
    busiestHourStart: busiest * 3600,
    silence: silenceLength > 600 ? {start: silenceStart, length: silenceLength} : null,
    fixInterval: getMedian(intervals),
    mooredInterval: getMedian(mooredIntervals)
  };
}

/** Median of a list (0 for an empty one). */
function getMedian(values: readonly number[]): number {
  if (!values.length) return 0;
  const sorted = Float64Array.from(values).sort();
  return sorted[sorted.length >> 1];
}

// ---------------------------------------------------------------------------------------------
// Stops joined to groups and anchorages
// ---------------------------------------------------------------------------------------------

/** A NOAA anchorage, merged by its cleaned name (a few are split into several polygons). */
export type AnchorageGroup = {
  /** `Anchorage 21B`. */
  name: string;
  /** The water it lies in: `Upper Bay`. */
  region: string;
  rings: Float32Array[][];
  /** `[minX, minY, maxX, maxY]` of each polygon, planar metres. */
  bounds: [number, number, number, number][];
  /** Where to put its label, `[longitude, latitude]`. */
  labelPoint: [number, number];
};

/**
 * Turns a NOAA anchorage name into the short form on a chart: `Upper Bay Anchorage Area Number
 * 21B` becomes `Anchorage 21B` in `Upper Bay`. Names without a number (`Sandy Hook Bay Anchorage`)
 * stay as they are.
 */
export function cleanAnchorageName(raw: string): {name: string; region: string} {
  const match =
    /^(.*?)\s*Anchorage(?: Area)?(?: Number| No\.?)?\s+(\d+[A-Z]?)(?:\s+(West|East))?$/i.exec(
      raw.trim()
    );
  if (!match) return {name: raw.trim(), region: ''};
  const region = (match[1] || '').split(',')[0].trim();
  return {name: `Anchorage ${match[2]}${match[3] ? ` ${match[3]}` : ''}`, region};
}

/** The anchorage polygons of `ais-zones`, merged by cleaned name. */
export function buildAnchorages(
  zones: ZoneSet,
  unproject: (x: number, y: number) => [number, number]
): AnchorageGroup[] {
  const anchorageKind = ZONE_KINDS.indexOf('anchorage');
  const groups = new Map<string, AnchorageGroup>();
  const biggest = new Map<string, number>();
  for (let zone = 0; zone < zones.zoneCount; zone++) {
    if (zones.kinds[zone] !== anchorageKind) continue;
    const {name, region} = cleanAnchorageName(zones.names[zone]);
    // Numbers repeat across waters (an Upper Bay and a Port of New York anchorage 9), so the
    // region is part of the identity; the split polygons of one anchorage share both.
    const key = `${name}|${region}`;
    let group = groups.get(key);
    if (!group) {
      group = {name, region, rings: [], bounds: [], labelPoint: [0, 0]};
      groups.set(key, group);
    }
    const rings = zones.zoneRings[zone];
    group.rings.push(rings);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const ring of rings) {
      for (let index = 0; index < ring.length; index += 2) {
        minX = Math.min(minX, ring[index]);
        maxX = Math.max(maxX, ring[index]);
        minY = Math.min(minY, ring[index + 1]);
        maxY = Math.max(maxY, ring[index + 1]);
      }
    }
    group.bounds.push([minX, minY, maxX, maxY]);
    // The label sits in the middle of the biggest polygon of the group.
    if (zones.areas[zone] > (biggest.get(key) ?? -1)) {
      biggest.set(key, zones.areas[zone]);
      group.labelPoint = unproject((minX + maxX) / 2, (minY + maxY) / 2);
    }
  }
  return [...groups.values()];
}

/** Even-odd test of a point against the rings of one polygon. */
function isInsideRings(rings: readonly Float32Array[], x: number, y: number): boolean {
  let inside = false;
  for (const ring of rings) {
    const count = ring.length / 2;
    for (let index = 0, previous = count - 1; index < count; previous = index++) {
      const x0 = ring[index * 2];
      const y0 = ring[index * 2 + 1];
      const x1 = ring[previous * 2];
      const y1 = ring[previous * 2 + 1];
      if (y0 > y !== y1 > y && x < ((x1 - x0) * (y - y0)) / (y1 - y0) + x0) inside = !inside;
    }
  }
  return inside;
}

/** Index of the anchorage group containing a planar point, or -1. */
export function findAnchorage(anchorages: readonly AnchorageGroup[], x: number, y: number): number {
  for (let index = 0; index < anchorages.length; index++) {
    const group = anchorages[index];
    for (let polygon = 0; polygon < group.rings.length; polygon++) {
      const [minX, minY, maxX, maxY] = group.bounds[polygon];
      if (x < minX || x > maxX || y < minY || y > maxY) continue;
      if (isInsideRings(group.rings[polygon], x, y)) return index;
    }
  }
  return -1;
}

/** What the stop list says once it is joined to vessel kinds and anchorages. */
export type StopSummary = {
  /** Stops in the list. */
  count: number;
  /** Stops per dwell class (`classBreaksSeconds.length + 1` entries). */
  classCounts: number[];
  /** Share of stops made by tugs and passenger vessels (ferries, tour boats). */
  tugAndFerryShare: number;
  /** Share of stops whose centroid lies inside an official anchorage. */
  anchoredShare: number;
  /** The anchorage with the most stops, or `null`. */
  busiestAnchorage: {name: string; region: string; count: number; lngLat: [number, number]} | null;
  /** The densest 700 m cell of stops, or `null`. */
  busiestCluster: {count: number; lngLat: [number, number]} | null;
};

/**
 * Joins the stop list to the data around it. `trackIds[i]` is the track of stop `i`,
 * `centroids` holds `x, y` planar metres per stop and `durations` seconds.
 */
export function summariseStops(input: {
  vessels: VesselTrackSet;
  anchorages: readonly AnchorageGroup[];
  count: number;
  trackIds: ArrayLike<number>;
  centroids: ArrayLike<number>;
  durations: ArrayLike<number>;
  classBreaksSeconds: readonly number[];
}): StopSummary {
  const {vessels, anchorages, count, trackIds, centroids, durations, classBreaksSeconds} = input;
  const classCounts = new Array<number>(classBreaksSeconds.length + 1).fill(0);
  const tug = VESSEL_CATEGORIES.indexOf('tug');
  const passenger = VESSEL_CATEGORIES.indexOf('passenger');
  const perAnchorage = new Array<number>(anchorages.length).fill(0);
  const cells = new Map<string, {count: number; sumX: number; sumY: number}>();
  let tugAndFerry = 0;
  let anchored = 0;
  for (let stop = 0; stop < count; stop++) {
    const duration = durations[stop];
    let stopClass = 0;
    while (stopClass < classBreaksSeconds.length && duration >= classBreaksSeconds[stopClass]) {
      stopClass++;
    }
    classCounts[stopClass]++;
    const category = vessels.category[trackIds[stop]];
    if (category === tug || category === passenger) tugAndFerry++;
    const x = centroids[stop * 2];
    const y = centroids[stop * 2 + 1];
    const anchorage = findAnchorage(anchorages, x, y);
    if (anchorage >= 0) {
      anchored++;
      perAnchorage[anchorage]++;
    }
    const key = `${Math.floor(x / 700)},${Math.floor(y / 700)}`;
    const cell = cells.get(key) ?? {count: 0, sumX: 0, sumY: 0};
    cell.count++;
    cell.sumX += x;
    cell.sumY += y;
    cells.set(key, cell);
  }
  let busiestAnchorageIndex = -1;
  for (let index = 0; index < perAnchorage.length; index++) {
    if (
      perAnchorage[index] > 0 &&
      perAnchorage[index] > (perAnchorage[busiestAnchorageIndex] ?? 0)
    ) {
      busiestAnchorageIndex = index;
    }
  }
  let busiestCell: {count: number; sumX: number; sumY: number} | null = null;
  for (const cell of cells.values()) {
    if (!busiestCell || cell.count > busiestCell.count) busiestCell = cell;
  }
  return {
    count,
    classCounts,
    tugAndFerryShare: count ? tugAndFerry / count : 0,
    anchoredShare: count ? anchored / count : 0,
    busiestAnchorage:
      busiestAnchorageIndex >= 0
        ? {
            name: anchorages[busiestAnchorageIndex].name,
            region: anchorages[busiestAnchorageIndex].region,
            count: perAnchorage[busiestAnchorageIndex],
            lngLat: anchorages[busiestAnchorageIndex].labelPoint
          }
        : null,
    busiestCluster: busiestCell
      ? {
          count: busiestCell.count,
          lngLat: vessels.unproject(
            busiestCell.sumX / busiestCell.count,
            busiestCell.sumY / busiestCell.count
          )
        }
      : null
  };
}
