// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {matchByOverlap} from '../../cartography/stable-hues';

/**
 * Pure helpers of the bixi-communities scene: identity-stable hues for groups of stations, convex
 * hulls for the group wash, the majority borough of a group (the seams) and a CPU replay of the
 * label-propagation vote. No luma.gl, no DOM.
 */

/** At most this many groups take a hue; every other group is the neutral grey "other". */
export const MAX_HUES = 7;
/** A group smaller than this many stations collapses to grey "other". */
export const MIN_GROUP_STATIONS = 8;
/** Palette slot of the neutral grey "other" (the eighth entry of `getGroupPalette`). */
export const OTHER_SLOT = 7;

/** What a later partition needs to inherit the hues of an earlier one. */
export type HueMemory = {
  /** Dense hued group per station (`-1`: grey). */
  dense: Int32Array;
  /** Palette slot of each dense group. */
  colors: number[];
};

/** One group of stations that has a hue. */
export type HuedGroup = {
  /** The label the partition gave the group. */
  label: number;
  size: number;
  /** Palette slot `0`-`6`. */
  slot: number;
  /** Index into {@link HueMemory.colors}. */
  dense: number;
};

/** Result of {@link assignGroupHues}. */
export type StationGroups = {
  /** Palette slot per station: `0`-`6` hues, {@link OTHER_SLOT} grey. */
  slots: Uint32Array;
  /** Hued groups, largest first. */
  hued: HuedGroup[];
  /** Size of every group (hued or not), by label. */
  sizes: Map<number, number>;
  /** Every group of the partition, hued or grey. */
  groupCount: number;
  /** Stations in grey groups. */
  greyStations: number;
  /** The memory a following partition can inherit from (the caller decides whether to keep it). */
  memory: HueMemory;
};

/**
 * Gives groups of stations hues that keep their identity. Only the seven largest groups of at
 * least {@link MIN_GROUP_STATIONS} stations take a hue. With no `memory` (the first partition) the
 * hues run west to east by the group's mean longitude; with a `memory` each group inherits the hue
 * of the earlier group it overlaps most (`matchByOverlap`, an optimal one-to-one assignment), so a
 * split keeps the hue of its larger part and a merge keeps one of its parents.
 *
 * @param labels One label per station (any integers).
 * @param lngLat Interleaved `[longitude, latitude]` per station.
 */
export function assignGroupHues(
  labels: ArrayLike<number>,
  lngLat: ArrayLike<number>,
  memory: HueMemory | null
): StationGroups {
  const stationCount = labels.length;
  const sizes = new Map<number, number>();
  const longitudeSums = new Map<number, number>();
  for (let station = 0; station < stationCount; station++) {
    const label = labels[station];
    sizes.set(label, (sizes.get(label) ?? 0) + 1);
    longitudeSums.set(label, (longitudeSums.get(label) ?? 0) + lngLat[station * 2]);
  }
  const eligible = Array.from(sizes.entries())
    .filter(([, size]) => size >= MIN_GROUP_STATIONS)
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, MAX_HUES);
  const denseOfLabel = new Map<number, number>(eligible.map(([label], index) => [label, index]));
  const dense = new Int32Array(stationCount).fill(-1);
  for (let station = 0; station < stationCount; station++) {
    dense[station] = denseOfLabel.get(labels[station]) ?? -1;
  }
  let colors: number[];
  if (memory) {
    colors = matchByOverlap(memory.dense, dense, MAX_HUES, memory.colors);
  } else {
    const byLongitude = eligible
      .map(([label, size], index) => ({
        index,
        longitude: (longitudeSums.get(label) ?? 0) / size
      }))
      .sort((a, b) => a.longitude - b.longitude);
    colors = new Array<number>(eligible.length).fill(0);
    byLongitude.forEach((entry, rank) => {
      colors[entry.index] = rank;
    });
  }
  const slots = new Uint32Array(stationCount);
  let greyStations = 0;
  for (let station = 0; station < stationCount; station++) {
    const group = dense[station];
    slots[station] = group >= 0 ? colors[group] : OTHER_SLOT;
    if (group < 0) greyStations++;
  }
  const hued: HuedGroup[] = eligible.map(([label, size], index) => ({
    label,
    size,
    slot: colors[index],
    dense: index
  }));
  return {
    slots,
    hued,
    sizes,
    groupCount: sizes.size,
    greyStations,
    memory: {dense, colors}
  };
}

/** The borough that holds most of a group's stations (ties go to the lower borough index). */
export function getMajorityBoroughs(
  labels: ArrayLike<number>,
  borough: ArrayLike<number>
): Map<number, number> {
  const counts = new Map<number, Map<number, number>>();
  for (let station = 0; station < labels.length; station++) {
    let perBorough = counts.get(labels[station]);
    if (!perBorough) {
      perBorough = new Map();
      counts.set(labels[station], perBorough);
    }
    perBorough.set(borough[station], (perBorough.get(borough[station]) ?? 0) + 1);
  }
  const majority = new Map<number, number>();
  for (const [label, perBorough] of counts) {
    let best = -1;
    let bestCount = -1;
    for (const [index, count] of perBorough) {
      if (count > bestCount || (count === bestCount && index < best)) {
        best = index;
        bestCount = count;
      }
    }
    majority.set(label, best);
  }
  return majority;
}

/** `[x, y]` in planar metres. */
export type Point = readonly [number, number];

function getCross(o: Point, a: Point, b: Point): number {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

/** Convex hull (Andrew's monotone chain), counter-clockwise, without the repeated first point. */
export function getConvexHull(points: readonly Point[]): Point[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length < 3) return sorted;
  const lower: Point[] = [];
  for (const point of sorted) {
    while (
      lower.length >= 2 &&
      getCross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0
    ) {
      lower.pop();
    }
    lower.push(point);
  }
  const upper: Point[] = [];
  for (let index = sorted.length - 1; index >= 0; index--) {
    const point = sorted[index];
    while (
      upper.length >= 2 &&
      getCross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0
    ) {
      upper.pop();
    }
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function getMedian(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Drops the outliers of a group before its hull is drawn: stations farther from the group's
 * median point than `factor` times the median distance (never less than `floorMeters`), so one
 * stray station does not stretch the wash across the island.
 */
export function trimOutliers(points: readonly Point[], floorMeters = 1000, factor = 3): Point[] {
  if (points.length < 4) return [...points];
  const medianX = getMedian(points.map(point => point[0]));
  const medianY = getMedian(points.map(point => point[1]));
  const distances = points.map(point => Math.hypot(point[0] - medianX, point[1] - medianY));
  const limit = Math.max(floorMeters, factor * getMedian(distances));
  return points.filter((_, index) => distances[index] <= limit);
}

/** Active undirected edges, as the CPU replay reads them. */
export type ReplayEdges = {
  /** Endpoints of every edge of the month graph. */
  a: ArrayLike<number>;
  b: ArrayLike<number>;
  /** `1` when the edge is in the graph. */
  mask: ArrayLike<number>;
  count: number;
};

/** Labels after every round of the replay and how many stations changed in each round. */
export type PropagationReplay = {
  /** `rounds + 1` label arrays: round 0 is every station's own index. */
  labels: Uint32Array[];
  /** Stations whose label changed in round `r + 1`, for `r = 0 ..< rounds`. */
  changed: number[];
};

/**
 * CPU replay of `GPUGraphLabelPropagation` on an undirected graph: every round is synchronous
 * (it reads the previous round's whole snapshot), a station casts one vote for its own label and
 * one for every neighbour occurrence, the most frequent label wins and a tie goes to the lowest
 * label. Edge weights are ignored, as on the GPU. The replay exists to show the algorithm round by
 * round; the scene checks its last round against the GPU's labels.
 */
export function replayLabelPropagation(
  stationCount: number,
  edges: ReplayEdges,
  rounds: number
): PropagationReplay {
  const offsets = new Uint32Array(stationCount + 1);
  for (let edge = 0; edge < edges.count; edge++) {
    if (!edges.mask[edge]) continue;
    offsets[edges.a[edge] + 1]++;
    offsets[edges.b[edge] + 1]++;
  }
  for (let station = 0; station < stationCount; station++) offsets[station + 1] += offsets[station];
  const neighbors = new Uint32Array(offsets[stationCount]);
  const cursor = offsets.slice(0, stationCount);
  for (let edge = 0; edge < edges.count; edge++) {
    if (!edges.mask[edge]) continue;
    neighbors[cursor[edges.a[edge]]++] = edges.b[edge];
    neighbors[cursor[edges.b[edge]]++] = edges.a[edge];
  }
  const labels: Uint32Array[] = [Uint32Array.from({length: stationCount}, (_, station) => station)];
  const changed: number[] = [];
  const votes = new Uint32Array(stationCount);
  const touched: number[] = [];
  for (let round = 0; round < rounds; round++) {
    const previous = labels[round];
    const next = new Uint32Array(stationCount);
    let changes = 0;
    for (let station = 0; station < stationCount; station++) {
      touched.length = 0;
      const vote = (label: number) => {
        if (votes[label]++ === 0) touched.push(label);
      };
      vote(previous[station]);
      for (let slot = offsets[station]; slot < offsets[station + 1]; slot++) {
        vote(previous[neighbors[slot]]);
      }
      let best = previous[station];
      let bestVotes = 0;
      for (const label of touched) {
        const count = votes[label];
        if (count > bestVotes || (count === bestVotes && label < best)) {
          best = label;
          bestVotes = count;
        }
        votes[label] = 0;
      }
      next[station] = best;
      if (best !== previous[station]) changes++;
    }
    labels.push(next);
    changed.push(changes);
  }
  return {labels, changed};
}
