// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU reference for `GPUZoneEvents`, written in float64 from the documented rules: half-open
 * segment-edge crossings, an even-odd +x ray for the initial state, per-(track, zone) toggling,
 * and `(track, time, edge)` ordering.
 */

import type {OracleTracks} from '../trajectory-interpolation/trajectory-interpolation-oracle';

/** Zone boundary edges. */
export type ZoneEdges = {
  starts: Float32Array;
  ends: Float32Array;
  zones: Uint32Array;
};

/** One enter or exit event. */
export type ZoneOracleEvent = {
  track: number;
  zone: number;
  /** 0 enter, 1 exit. */
  type: number;
  time: number;
  row: number;
  edge: number;
  /** Crossing position. */
  x: number;
  y: number;
};

/** Oracle result. */
export type ZoneOracleResult = {
  /** Every event of every track, ordered by track, time, edge. */
  events: ZoneOracleEvent[];
  dwellTimes: number[];
  visitCounts: number[];
  trackEventCounts: number[];
  /** First enter per `(track, zone)`: 0 when starting inside, -1 when never visited. */
  firstEnterTimes: number[];
  /** Last exit per `(track, zone)`: the track duration when still inside at the end, else 0. */
  lastExitTimes: number[];
};

/** Builds closed-ring edges for `rings`, each `[zone, [x0, y0, x1, y1, ...]]`. */
export function createRingEdges(
  rings: readonly (readonly [number, readonly number[]])[]
): ZoneEdges {
  const starts: number[] = [];
  const ends: number[] = [];
  const zones: number[] = [];
  for (const [zone, coordinates] of rings) {
    const vertexCount = coordinates.length / 2;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const next = (vertex + 1) % vertexCount;
      starts.push(coordinates[2 * vertex], coordinates[2 * vertex + 1]);
      ends.push(coordinates[2 * next], coordinates[2 * next + 1]);
      zones.push(zone);
    }
  }
  return {
    starts: Float32Array.from(starts),
    ends: Float32Array.from(ends),
    zones: Uint32Array.from(zones)
  };
}

function getRelativeTimes(tracks: OracleTracks, start: number, end: number): number[] {
  const first = tracks.times.kind === 'words' ? tracks.times.values[start] : 0;
  const result: number[] = [];
  for (let row = start; row < end; row++) {
    result.push(
      tracks.times.kind === 'words'
        ? Number(tracks.times.values[row] - (first as bigint))
        : tracks.times.values[row] - tracks.times.values[start]
    );
  }
  return result;
}

/** CPU reference. `maximumEvents` is only used by callers that truncate the list. */
export function computeZoneEventsOracle(
  tracks: OracleTracks,
  edges: ZoneEdges,
  zoneCount: number
): ZoneOracleResult {
  const trackCount = tracks.trackOffsets.length - 1;
  const edgeCount = edges.zones.length;
  const result: ZoneOracleResult = {
    events: [],
    dwellTimes: new Array(trackCount * zoneCount).fill(0),
    visitCounts: new Array(trackCount * zoneCount).fill(0),
    trackEventCounts: new Array(trackCount).fill(0),
    firstEnterTimes: new Array(trackCount * zoneCount).fill(-1),
    lastExitTimes: new Array(trackCount * zoneCount).fill(0)
  };
  const {positions} = tracks;
  for (let track = 0; track < trackCount; track++) {
    const start = tracks.trackOffsets[track];
    const end = tracks.trackOffsets[track + 1];
    if (end <= start) {
      continue;
    }
    const times = getRelativeTimes(tracks, start, end);
    const duration = times[times.length - 1];
    const state = new Array<number>(zoneCount).fill(0);
    const [px, py] = [positions[2 * start], positions[2 * start + 1]];
    for (let edge = 0; edge < edgeCount; edge++) {
      const [ax, ay] = [edges.starts[2 * edge], edges.starts[2 * edge + 1]];
      const [bx, by] = [edges.ends[2 * edge], edges.ends[2 * edge + 1]];
      if (edges.zones[edge] >= zoneCount || ay > py === by > py) {
        continue;
      }
      if (px < ax + ((py - ay) * (bx - ax)) / (by - ay)) {
        state[edges.zones[edge]] ^= 1;
      }
    }
    const enterTimes = state.map(() => 0);
    for (let zone = 0; zone < zoneCount; zone++) {
      result.visitCounts[track * zoneCount + zone] = state[zone];
      result.firstEnterTimes[track * zoneCount + zone] = state[zone] === 1 ? 0 : -1;
    }
    const trackEvents: ZoneOracleEvent[] = [];
    for (let row = start + 1; row < end; row++) {
      const [x0, y0] = [positions[2 * (row - 1)], positions[2 * (row - 1) + 1]];
      const [dx, dy] = [positions[2 * row] - x0, positions[2 * row + 1] - y0];
      for (let edge = 0; edge < edgeCount; edge++) {
        if (edges.zones[edge] >= zoneCount) {
          continue;
        }
        const [ax, ay] = [edges.starts[2 * edge], edges.starts[2 * edge + 1]];
        const [ex, ey] = [edges.ends[2 * edge] - ax, edges.ends[2 * edge + 1] - ay];
        const denominator = dx * ey - dy * ex;
        if (denominator === 0) {
          continue;
        }
        const [wx, wy] = [ax - x0, ay - y0];
        const along = (wx * ey - wy * ex) / denominator;
        const across = (wx * dy - wy * dx) / denominator;
        if (along >= 0 && along < 1 && across >= 0 && across < 1) {
          const t0 = times[row - 1 - start];
          const t1 = times[row - start];
          trackEvents.push({
            track,
            zone: edges.zones[edge],
            type: 0,
            time: Math.max(t0 + along * (t1 - t0), 0),
            row,
            edge,
            x: x0 + along * dx,
            y: y0 + along * dy
          });
        }
      }
    }
    trackEvents.sort((a, b) => a.time - b.time || a.edge - b.edge);
    for (const event of trackEvents) {
      const cell = track * zoneCount + event.zone;
      state[event.zone] ^= 1;
      event.type = state[event.zone] === 1 ? 0 : 1;
      if (event.type === 0) {
        if (result.firstEnterTimes[cell] < 0) {
          result.firstEnterTimes[cell] = event.time;
        }
        enterTimes[event.zone] = event.time;
        result.visitCounts[cell]++;
      } else {
        result.dwellTimes[cell] += event.time - enterTimes[event.zone];
        result.lastExitTimes[cell] = event.time;
      }
    }
    for (let zone = 0; zone < zoneCount; zone++) {
      if (state[zone] === 1) {
        result.dwellTimes[track * zoneCount + zone] += duration - enterTimes[zone];
        result.lastExitTimes[track * zoneCount + zone] = duration;
      }
    }
    result.trackEventCounts[track] = trackEvents.length;
    result.events.push(...trackEvents);
  }
  return result;
}
