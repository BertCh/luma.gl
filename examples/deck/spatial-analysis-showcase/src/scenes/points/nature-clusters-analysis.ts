// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The CPU side of the nature-clusters story: what the story reads back from the GPU labels and hulls
 * and turns into drawing arrays and numbers. Pure TypeScript, no GPU and no DOM.
 *
 * - cluster sizes, the per-point size buffer (class colour), and the draw-id lists for noise,
 *   members, core and border points;
 * - convex hulls as rings in local metres, their areas, a triangle fan and outline segments for the
 *   polygon and segment layers;
 * - the greedy map colouring of a k-means partition, so neighbouring groups never share a hue;
 * - the three sample points whose epsilon discs the core / border / noise step draws;
 * - the share of a hull that lies outside the city (a Monte Carlo sample).
 */

import {matchStableColors, type StableHuePrevious} from '../../cartography/stable-hues';

/** Fixed class breaks of the hot-spot size classes (records per hot spot), shared by every toggle. */
export const SIZE_BREAKS: readonly number[] = [50, 200, 1000];

/** The uint32 label `GPUSpatialClustering` gives to a record that is in no hot spot. */
export const NOISE_LABEL = 0xffffffff;

/** Class index (0-3) of a cluster of `size` records: the number of breaks at or below it. */
export function getSizeClass(size: number): number {
  let index = 0;
  for (const boundary of SIZE_BREAKS) if (size >= boundary) index++;
  return index;
}

/** One convex hull in local metres. */
export type HullShape = {
  /** Cluster id (the row of every per-group buffer). */
  group: number;
  /** Records in the cluster. */
  size: number;
  /** Size class of the cluster. */
  sizeClass: number;
  /** Counter-clockwise ring, flat `x, y` pairs, not closed. */
  ring: Float32Array;
  /** Planar area in m². */
  areaSquareMeters: number;
  /** `[minX, minY, maxX, maxY]` of the ring. */
  bounds: [number, number, number, number];
};

/** What one readback of labels and hulls becomes. */
export type ClusterAnalysis = {
  pointCount: number;
  groupCount: number;
  /** Records per cluster id. */
  sizes: Uint32Array;
  /** Clusters with at least one record. */
  clusterCount: number;
  memberCount: number;
  noiseCount: number;
  coreCount: number;
  borderCount: number;
  /** Per record: the size of its cluster, NaN for noise (colours the class table). */
  pointSizes: Float32Array;
  /** Per cluster id: its size (0 for unused ids), for hull colours. */
  groupSizes: Float32Array;
  /** Rows of noise records. */
  noiseIds: Uint32Array;
  /** Rows of clustered records, smallest cluster first so the biggest draw on top. */
  memberIds: Uint32Array;
  /** Rows of core records, smallest cluster first. */
  coreIds: Uint32Array;
  /** Rows of border records (in a cluster, not core), smallest cluster first. */
  borderIds: Uint32Array;
  /** Cluster ids by size, largest first. */
  ranked: number[];
  /** Hot spots per size class. */
  classCounts: number[];
  /** Mean position of each cluster's records, flat `x, y` in metres (NaN when empty). */
  groupCentres: Float64Array;
  hulls: HullShape[];
};

/**
 * Counts the clusters, classifies them and builds every per-record array the layers draw from.
 *
 * @param coreFlags Per record 1 when it is a core point, or `null` (k-means, the recipe).
 * @param hasNoise True for DBSCAN, where `NOISE_LABEL` marks records in no cluster.
 */
export function analyzeLabels(input: {
  positions: Float32Array;
  labels: Uint32Array;
  coreFlags: Uint32Array | null;
  groupCount: number;
  hasNoise: boolean;
}): ClusterAnalysis {
  const {positions, labels, coreFlags, groupCount, hasNoise} = input;
  const pointCount = labels.length;
  const sizes = new Uint32Array(groupCount);
  const sumX = new Float64Array(groupCount);
  const sumY = new Float64Array(groupCount);
  for (let row = 0; row < pointCount; row++) {
    const label = labels[row];
    if (label >= groupCount) continue;
    sizes[label]++;
    sumX[label] += positions[row * 2];
    sumY[label] += positions[row * 2 + 1];
  }
  const groupCentres = new Float64Array(groupCount * 2).fill(Number.NaN);
  const groupSizes = new Float32Array(groupCount);
  const ranked: number[] = [];
  const classCounts = Array.from({length: SIZE_BREAKS.length + 1}, () => 0);
  for (let group = 0; group < groupCount; group++) {
    const size = sizes[group];
    if (size === 0) continue;
    groupSizes[group] = size;
    groupCentres[group * 2] = sumX[group] / size;
    groupCentres[group * 2 + 1] = sumY[group] / size;
    ranked.push(group);
    classCounts[getSizeClass(size)]++;
  }
  ranked.sort((a, b) => sizes[b] - sizes[a] || a - b);

  const pointSizes = new Float32Array(pointCount);
  const noise: number[] = [];
  const members: number[] = [];
  let coreCount = 0;
  let borderCount = 0;
  for (let row = 0; row < pointCount; row++) {
    const label = labels[row];
    const clustered = label < groupCount && sizes[label] > 0;
    if (!clustered || (hasNoise && label === NOISE_LABEL)) {
      pointSizes[row] = Number.NaN;
      noise.push(row);
      continue;
    }
    pointSizes[row] = sizes[label];
    members.push(row);
  }
  // Smallest cluster first: the biggest, darkest hot spot is drawn last, on top.
  members.sort((a, b) => pointSizes[a] - pointSizes[b] || a - b);
  const core: number[] = [];
  const border: number[] = [];
  for (const row of members) {
    if (coreFlags?.[row]) {
      core.push(row);
      coreCount++;
    } else {
      border.push(row);
      borderCount++;
    }
  }
  return {
    pointCount,
    groupCount,
    sizes,
    clusterCount: ranked.length,
    memberCount: members.length,
    noiseCount: noise.length,
    coreCount,
    borderCount,
    pointSizes,
    groupSizes,
    noiseIds: Uint32Array.from(noise),
    memberIds: Uint32Array.from(members),
    coreIds: Uint32Array.from(core),
    borderIds: Uint32Array.from(border),
    ranked,
    classCounts,
    groupCentres,
    hulls: []
  };
}

/** Signed shoelace area of a flat ring (positive counter-clockwise). */
export function getRingSignedArea(ring: ArrayLike<number>): number {
  const count = ring.length / 2;
  let sum = 0;
  for (let i = 0, j = count - 1; i < count; j = i++) {
    sum += ring[j * 2] * ring[i * 2 + 1] - ring[i * 2] * ring[j * 2 + 1];
  }
  return sum / 2;
}

/** Even-odd point-in-ring test in planar metres. */
export function isPointInRing(x: number, y: number, ring: ArrayLike<number>): boolean {
  const count = ring.length / 2;
  let inside = false;
  for (let i = 0, j = count - 1; i < count; j = i++) {
    const xi = ring[i * 2];
    const yi = ring[i * 2 + 1];
    const xj = ring[j * 2];
    const yj = ring[j * 2 + 1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * Reads the hulls `GPUGroupConvexHull` wrote (one ring per cluster, over-cap hulls absent) and
 * attaches them to the analysis.
 */
export function attachHulls(
  analysis: ClusterAnalysis,
  hullOffsets: Uint32Array,
  hullCounts: Uint32Array,
  hullPositions: Float32Array
): void {
  const hulls: HullShape[] = [];
  const total = hullOffsets[analysis.groupCount] ?? 0;
  for (const group of analysis.ranked) {
    const count = hullCounts[group];
    const first = hullOffsets[group];
    if (count < 3 || first + count > total || (first + count) * 2 > hullPositions.length) continue;
    const ring = hullPositions.slice(first * 2, (first + count) * 2);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < count; i++) {
      minX = Math.min(minX, ring[i * 2]);
      maxX = Math.max(maxX, ring[i * 2]);
      minY = Math.min(minY, ring[i * 2 + 1]);
      maxY = Math.max(maxY, ring[i * 2 + 1]);
    }
    const size = analysis.sizes[group];
    hulls.push({
      group,
      size,
      sizeClass: getSizeClass(size),
      ring,
      areaSquareMeters: Math.abs(getRingSignedArea(ring)),
      bounds: [minX, minY, maxX, maxY]
    });
  }
  analysis.hulls = hulls;
}

/** Triangle fans and outline segments of the hulls, ready to upload to the polygon and segment layers. */
export type HullMesh = {
  /** Triangle-list vertices, flat `x, y`. */
  triangles: Float32Array;
  /** Cluster id of every triangle vertex. */
  triangleFeatures: Uint32Array;
  /** Outline segments, flat `x0, y0, x1, y1`. */
  outline: Float32Array;
  /** Cluster id of every outline segment. */
  outlineFeatures: Uint32Array;
};

/** Fans every (convex) hull into triangles and closes its outline. */
export function buildHullMesh(hulls: readonly HullShape[]): HullMesh {
  let triangleVertices = 0;
  let segments = 0;
  for (const hull of hulls) {
    const count = hull.ring.length / 2;
    triangleVertices += (count - 2) * 3;
    segments += count;
  }
  const triangles = new Float32Array(triangleVertices * 2);
  const triangleFeatures = new Uint32Array(triangleVertices);
  const outline = new Float32Array(segments * 4);
  const outlineFeatures = new Uint32Array(segments);
  let vertex = 0;
  let segment = 0;
  for (const hull of hulls) {
    const {ring, group} = hull;
    const count = ring.length / 2;
    for (let i = 1; i < count - 1; i++) {
      for (const index of [0, i, i + 1]) {
        triangles[vertex * 2] = ring[index * 2];
        triangles[vertex * 2 + 1] = ring[index * 2 + 1];
        triangleFeatures[vertex++] = group;
      }
    }
    for (let i = 0; i < count; i++) {
      const next = (i + 1) % count;
      outline.set([ring[i * 2], ring[i * 2 + 1], ring[next * 2], ring[next * 2 + 1]], segment * 4);
      outlineFeatures[segment++] = group;
    }
  }
  return {triangles, triangleFeatures, outline, outlineFeatures};
}

/**
 * The hull that contains `(x, y)` (the smallest one when hulls overlap), or `null`. Linear in the
 * number of hulls with a bounding-box prefilter.
 */
export function findHullAt(hulls: readonly HullShape[], x: number, y: number): HullShape | null {
  let best: HullShape | null = null;
  for (const hull of hulls) {
    const [minX, minY, maxX, maxY] = hull.bounds;
    if (x < minX || x > maxX || y < minY || y > maxY) continue;
    if (!isPointInRing(x, y, hull.ring)) continue;
    if (!best || hull.areaSquareMeters < best.areaSquareMeters) best = hull;
  }
  return best;
}

/**
 * Share of a hull that is not city land, from a Monte Carlo sample: `sampleCount` uniform points
 * in the hull are tested with `isInCity`.
 */
export function getShareOutsideCity(
  hull: HullShape,
  isInCity: (x: number, y: number) => boolean,
  random: () => number,
  sampleCount = 2000
): number {
  const [minX, minY, maxX, maxY] = hull.bounds;
  let sampled = 0;
  let outside = 0;
  let attempts = 0;
  while (sampled < sampleCount && attempts < sampleCount * 50) {
    attempts++;
    const x = minX + random() * (maxX - minX);
    const y = minY + random() * (maxY - minY);
    if (!isPointInRing(x, y, hull.ring)) continue;
    sampled++;
    if (!isInCity(x, y)) outside++;
  }
  return sampled > 0 ? outside / sampled : 0;
}

// ---------------------------------------------------------------------------------------------
// K-means partition colours
// ---------------------------------------------------------------------------------------------

/**
 * Colour slots for a k-means partition. Neighbouring groups never share a slot when the palette
 * allows it: two centres are neighbours when either is among the other's four nearest centres.
 * Slots start from the previous solution's (the Hungarian match of `matchStableColors`, so a group
 * keeps its hue when k or the seed changes) and a clash with a neighbour is repaired greedily,
 * most-connected group first.
 *
 * @param centres Flat `x, y` in metres, one per group (NaN centres get slot 0).
 * @param previous Centres and slots of the previous solution.
 */
export function colourPartition(
  centres: ArrayLike<number>,
  groupCount: number,
  previous: readonly StableHuePrevious[],
  paletteSize: number
): number[] {
  const live: number[] = [];
  for (let group = 0; group < groupCount; group++) {
    if (Number.isFinite(centres[group * 2]) && Number.isFinite(centres[group * 2 + 1])) {
      live.push(group);
    }
  }
  const slots = new Array<number>(groupCount).fill(0);
  if (live.length === 0) return slots;
  const stable = matchStableColors(
    previous,
    live.map(group => ({center: [centres[group * 2], centres[group * 2 + 1]]})),
    paletteSize
  );
  live.forEach((group, index) => {
    slots[group] = stable[index];
  });
  // Nearest-four graph, symmetrised.
  const neighbours = new Map<number, Set<number>>(live.map(group => [group, new Set<number>()]));
  for (const group of live) {
    const nearest = live
      .filter(other => other !== group)
      .map(other => ({
        other,
        distance: Math.hypot(
          centres[group * 2] - centres[other * 2],
          centres[group * 2 + 1] - centres[other * 2 + 1]
        )
      }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 4);
    for (const {other} of nearest) {
      neighbours.get(group)?.add(other);
      neighbours.get(other)?.add(group);
    }
  }
  const order = [...live].sort(
    (a, b) => (neighbours.get(b)?.size ?? 0) - (neighbours.get(a)?.size ?? 0) || a - b
  );
  const settled = new Set<number>();
  for (const group of order) {
    const taken = new Map<number, number>();
    for (const other of neighbours.get(group) ?? []) {
      if (settled.has(other)) taken.set(slots[other], (taken.get(slots[other]) ?? 0) + 1);
    }
    if (!taken.has(slots[group])) {
      settled.add(group);
      continue;
    }
    let bestSlot = slots[group];
    let bestClash = Infinity;
    for (let slot = 0; slot < paletteSize; slot++) {
      const clash = taken.get(slot) ?? 0;
      if (clash < bestClash) {
        bestClash = clash;
        bestSlot = slot;
      }
    }
    slots[group] = bestSlot;
    settled.add(group);
  }
  return slots;
}

// ---------------------------------------------------------------------------------------------
// Epsilon discs of the core / border / noise step
// ---------------------------------------------------------------------------------------------

/** What an epsilon disc says about its centre record. */
export type DiscRole = 'core' | 'border' | 'noise';

/** One record chosen to wear an epsilon disc, with its live neighbour count. */
export type EpsilonSample = {
  role: DiscRole;
  row: number;
  /** Local metres. */
  x: number;
  y: number;
  /** Records within epsilon, the record itself included (the DBSCAN convention). */
  neighbours: number;
};

/**
 * Picks one core, one border and one noise record near `centre` (within `windowMeters`) and counts
 * the records within epsilon of each on the CPU. The core is the one whose count is nearest 1.6
 * times the threshold, the border the one nearest 0.6 times, the noise record the one nearest 0.3
 * times, so each disc holds a countable handful of dots; discs are kept apart where the data allow.
 */
export function pickEpsilonSamples(input: {
  positions: Float32Array;
  labels: Uint32Array;
  coreFlags: Uint32Array;
  epsilon: number;
  minimumPoints: number;
  centre: readonly [number, number];
  windowMeters: number;
}): EpsilonSample[] {
  const {positions, labels, coreFlags, epsilon, minimumPoints, centre, windowMeters} = input;
  const count = labels.length;
  const roleOf = (row: number): DiscRole =>
    labels[row] === NOISE_LABEL ? 'noise' : coreFlags[row] ? 'core' : 'border';
  const candidates: Record<DiscRole, number[]> = {core: [], border: [], noise: []};
  for (let row = 0; row < count; row++) {
    const dx = positions[row * 2] - centre[0];
    const dy = positions[row * 2 + 1] - centre[1];
    if (dx * dx + dy * dy > windowMeters * windowMeters) continue;
    candidates[roleOf(row)].push(row);
  }
  const countNeighbours = (row: number) => {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    const limit = epsilon * epsilon;
    let total = 0;
    for (let other = 0; other < count; other++) {
      const dx = positions[other * 2] - x;
      const dy = positions[other * 2 + 1] - y;
      if (dx * dx + dy * dy <= limit) total++;
    }
    return total;
  };
  const samples: EpsilonSample[] = [];
  const targets: [DiscRole, number][] = [
    ['core', 1.6 * minimumPoints],
    ['border', 0.6 * minimumPoints],
    ['noise', 0.3 * minimumPoints]
  ];
  for (const [role, target] of targets) {
    // A deterministic thin sample of the candidates keeps the counting cheap.
    const pool = candidates[role];
    const stride = Math.max(1, Math.floor(pool.length / 160));
    let chosen: EpsilonSample | null = null;
    // First keep the discs apart; if the data leave no room, accept an overlap.
    for (const separation of [2.2 * epsilon, 0]) {
      let bestScore = Infinity;
      for (let index = 0; index < pool.length; index += stride) {
        const row = pool[index];
        const x = positions[row * 2];
        const y = positions[row * 2 + 1];
        if (!samples.every(sample => Math.hypot(sample.x - x, sample.y - y) >= separation)) {
          continue;
        }
        const neighbours = countNeighbours(row);
        const distance = Math.hypot(x - centre[0], y - centre[1]) / windowMeters;
        const score = Math.abs(neighbours - target) / Math.max(target, 1) + 0.35 * distance;
        if (score < bestScore) {
          bestScore = score;
          chosen = {role, row, x, y, neighbours};
        }
      }
      if (chosen) break;
    }
    if (chosen) samples.push(chosen);
  }
  return samples;
}
