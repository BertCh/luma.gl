// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure CPU helpers of the flight-matrix story: group labels, the matrix window arithmetic, the
 * degree-preserving chance model and the small tables the labels and readouts are built from.
 * None of it touches the GPU; the scene uses it for block boundaries, hover, the observed-vs-expected
 * matrix and as the CPU twin the GPU results are checked against.
 */

import {createSeededRandom} from '../../engine/projection';
import type {FlightNetwork} from './b11-flight-data';

/** The ways the airports can be put in order (the matrix rows and columns). */
export type MatrixOrder = 'input' | 'shuffle' | 'degree' | 'continent' | 'country';

/** Group labels of every airport and the keys the orderings sort by. */
export type MatrixGroups = {
  /** Continent index (`CONTINENT_NAMES`) per airport. */
  continent: Uint32Array;
  /** Continent labels randomly reassigned among the airports: same group sizes, no structure. */
  shuffled: Uint32Array;
  /** Country id per airport, continent-major, big countries first within a continent. */
  country: Uint32Array;
  countryNames: string[];
  /** Continent index of every country id. */
  countryContinent: Uint32Array;
  /** `maxDegree - degree`: ascending sort puts the busiest airport first. */
  degreeKey: Uint32Array;
};

/**
 * Builds the group labels. Countries are numbered continent by continent and, inside a continent,
 * by their number of airports (largest first), so a country order keeps continents contiguous.
 */
export function buildGroups(network: FlightNetwork): MatrixGroups {
  const countryCounts = new Map<string, number>();
  for (const airport of network.airports) {
    countryCounts.set(airport.country, (countryCounts.get(airport.country) ?? 0) + 1);
  }
  const continentOfCountry = new Map<string, number>();
  network.airports.forEach((airport, index) => {
    continentOfCountry.set(airport.country, network.continent[index]);
  });
  const countryList = [...countryCounts.keys()].sort(
    (a, b) =>
      continentOfCountry.get(a)! - continentOfCountry.get(b)! ||
      countryCounts.get(b)! - countryCounts.get(a)! ||
      a.localeCompare(b)
  );
  const countryIds = new Map(countryList.map((country, index) => [country, index]));
  const country = new Uint32Array(network.nodeCount);
  const continent = new Uint32Array(network.nodeCount);
  const degreeKey = new Uint32Array(network.nodeCount);
  const maximumDegree = Math.max(...network.degree);
  network.airports.forEach((airport, index) => {
    country[index] = countryIds.get(airport.country)!;
    continent[index] = network.continent[index];
    degreeKey[index] = maximumDegree - network.degree[index];
  });
  // Fisher-Yates over a copy of the continent labels: the null model keeps every group size.
  const shuffled = Uint32Array.from(continent);
  const random = createSeededRandom(20140601);
  for (let index = shuffled.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    const held = shuffled[index];
    shuffled[index] = shuffled[other];
    shuffled[other] = held;
  }
  return {
    continent,
    shuffled,
    country,
    countryNames: countryList,
    countryContinent: Uint32Array.from(countryList.map(name => continentOfCountry.get(name)!)),
    degreeKey
  };
}

/** The group key and tie key `GPUAdjacencyMatrixOrder` sorts by for one ordering. */
export function getOrderKeys(
  groups: MatrixGroups,
  order: MatrixOrder
): {groupKeys: Uint32Array; tieKeys: Uint32Array} {
  const zeros = new Uint32Array(groups.continent.length);
  switch (order) {
    case 'shuffle':
      return {groupKeys: groups.shuffled, tieKeys: groups.degreeKey};
    case 'degree':
      return {groupKeys: zeros, tieKeys: groups.degreeKey};
    case 'continent':
      return {groupKeys: groups.continent, tieKeys: groups.degreeKey};
    case 'country':
      return {groupKeys: groups.country, tieKeys: groups.degreeKey};
    default:
      return {groupKeys: zeros, tieKeys: zeros};
  }
}

/** The group labels an ordering is made of, or `null` when it has no blocks. */
export function getOrderBlocks(groups: MatrixGroups, order: MatrixOrder): Uint32Array | null {
  if (order === 'continent') return groups.continent;
  if (order === 'shuffle') return groups.shuffled;
  if (order === 'country') return groups.country;
  return null;
}

/** Cumulative block starts (`groupCount + 1` entries) of labels sorted by group id. */
export function getBlockStarts(labels: Uint32Array, groupCount: number): number[] {
  const sizes = new Array<number>(groupCount).fill(0);
  for (const label of labels) sizes[label]++;
  const starts = [0];
  for (const size of sizes) starts.push(starts[starts.length - 1] + size);
  return starts;
}

// ---------------------------------------------------------------------------------------------
// The matrix window
// ---------------------------------------------------------------------------------------------

/** A stretch of matrix positions on one axis. */
export type AxisWindow = {start: number; extent: number};

/** The window of the matrix image: what the GPU bins and what the card shows. */
export type MatrixWindow = {
  /** First position of the rows and of the columns shown. */
  rowStart: number;
  colStart: number;
  /** Airports per side of the window the reader asked for. */
  extent: number;
  /** Bins per side the graph was compiled with. */
  resolution: number;
  /**
   * Airports per cell side, a whole number so that every cell holds exactly `k x k` airports and
   * no bin is skipped: `ceil(extent / resolution)`.
   */
  positionsPerCell: number;
  /** Cells per side that are drawn: `ceil(extent / positionsPerCell)`, at most `resolution`. */
  cells: number;
  /** Positions the card spans: `cells * positionsPerCell` (at least `extent`). */
  span: number;
  /** Exclusive ends of the window words written to the GPU: `start + k * resolution`. */
  rowEnd: number;
  colEnd: number;
};

/**
 * The window arithmetic that removes the bin striping. `GPUAdjacencyMatrix` bins position `p` into
 * `floor((p - start) * R / (end - start))`; when the window holds fewer positions than `R`, bins
 * alternate empty. The fix is to make `end - start` an exact multiple `k * R` of the resolution
 * (`k = ceil(extent / R)`, 1 when the window holds fewer airports than bins). Every drawn cell then
 * covers exactly `k` positions per side; positions past the shown extent land in bins the scene
 * hides, and a window smaller than `R` is clamped to one cell per airport.
 */
export function getMatrixWindow(
  rows: AxisWindow,
  columns: AxisWindow,
  resolution: number
): MatrixWindow {
  const extent = Math.max(1, Math.max(rows.extent, columns.extent));
  const positionsPerCell = Math.max(1, Math.ceil(extent / resolution));
  const cells = Math.ceil(extent / positionsPerCell);
  return {
    rowStart: rows.start,
    colStart: columns.start,
    extent,
    resolution,
    positionsPerCell,
    cells,
    span: cells * positionsPerCell,
    rowEnd: rows.start + positionsPerCell * resolution,
    colEnd: columns.start + positionsPerCell * resolution
  };
}

// ---------------------------------------------------------------------------------------------
// Groups: shares, chance, anchors
// ---------------------------------------------------------------------------------------------

/** Share of the route pairs whose two airports carry the same label. */
export function getIntraShare(
  labels: ArrayLike<number>,
  source: ArrayLike<number>,
  target: ArrayLike<number>
): number {
  let intra = 0;
  for (let edge = 0; edge < source.length; edge++) {
    if (labels[source[edge]] === labels[target[edge]]) intra++;
  }
  return source.length > 0 ? intra / source.length : 0;
}

/** The observed-versus-expected matrix of a grouping under the degree-preserving null model. */
export type ChanceModel = {
  groupCount: number;
  /** Route ends from group `a` into group `b` (a diagonal pair counts twice), `groupCount^2`. */
  observed: Float64Array;
  /** `D_a * D_b / (2m)`: the same count if every route end were paired at random. */
  expected: Float64Array;
  /** `log2(observed / expected)`; `-Infinity` where nothing is observed. */
  logRatio: Float64Array;
  /** Route-end degree sum `D` of every group. */
  groupDegree: Float64Array;
  /** Routes inside groups divided by all routes. */
  observedShare: number;
  /** The same share expected under the null model: sum over groups of `(D_a / 2m)^2`. */
  expectedShare: number;
  /** Newman modularity of the grouping: observed minus expected share. */
  modularity: number;
};

/**
 * Observed route ends between groups against the configuration-model expectation: with every
 * airport keeping its degree and all route ends paired at random, group `a` sends `D_a * D_b / 2m`
 * of the `2m` route ends to group `b`. The diagonal counts each inside route twice, like the
 * symmetric matrix does, so observed and expected sum to the same `2m`.
 */
export function computeChance(
  labels: ArrayLike<number>,
  groupCount: number,
  source: ArrayLike<number>,
  target: ArrayLike<number>
): ChanceModel {
  const routeCount = source.length;
  const observed = new Float64Array(groupCount * groupCount);
  const groupDegree = new Float64Array(groupCount);
  let inside = 0;
  for (let edge = 0; edge < routeCount; edge++) {
    const a = labels[source[edge]];
    const b = labels[target[edge]];
    observed[a * groupCount + b]++;
    observed[b * groupCount + a]++;
    groupDegree[a]++;
    groupDegree[b]++;
    if (a === b) inside++;
  }
  const totalEnds = 2 * routeCount;
  const expected = new Float64Array(groupCount * groupCount);
  const logRatio = new Float64Array(groupCount * groupCount);
  let expectedShare = 0;
  for (let a = 0; a < groupCount; a++) {
    expectedShare += (groupDegree[a] / totalEnds) ** 2;
    for (let b = 0; b < groupCount; b++) {
      const index = a * groupCount + b;
      expected[index] = (groupDegree[a] * groupDegree[b]) / totalEnds;
      logRatio[index] =
        observed[index] > 0 && expected[index] > 0
          ? Math.log2(observed[index] / expected[index])
          : Number.NEGATIVE_INFINITY;
    }
  }
  const observedShare = routeCount > 0 ? inside / routeCount : 0;
  return {
    groupCount,
    observed,
    expected,
    logRatio,
    groupDegree,
    observedShare,
    expectedShare,
    modularity: observedShare - expectedShare
  };
}

/**
 * The airports of every group, busiest first (ties by lower index). The first member is the
 * group's anchor: supernodes sit on it, not on the mean longitude and latitude, which lands Asia in
 * the Himalayas and splits groups that cross the antimeridian.
 */
export function getMembersByDegree(
  labels: ArrayLike<number>,
  groupCount: number,
  degree: ArrayLike<number>
): number[][] {
  const members: number[][] = Array.from({length: groupCount}, () => []);
  for (let vertex = 0; vertex < labels.length; vertex++) members[labels[vertex]].push(vertex);
  for (const list of members) list.sort((a, b) => degree[b] - degree[a] || a - b);
  return members;
}

/** Largest number of routes between two distinct groups (the superedge scale of a grouping). */
export function getLargestSuperedge(
  labels: ArrayLike<number>,
  source: ArrayLike<number>,
  target: ArrayLike<number>
): number {
  const counts = new Map<number, number>();
  let largest = 0;
  for (let edge = 0; edge < source.length; edge++) {
    const a = labels[source[edge]];
    const b = labels[target[edge]];
    if (a === b) continue;
    const key = Math.min(a, b) * 65536 + Math.max(a, b);
    const next = (counts.get(key) ?? 0) + 1;
    counts.set(key, next);
    if (next > largest) largest = next;
  }
  return largest;
}

/**
 * Power-of-two histogram of positive integers: `1`, `2-3`, `4-7`, ... with the last bin open
 * (`512+` for ten bins).
 */
export function binLog2(
  values: ArrayLike<number>,
  binCount: number
): {counts: number[]; labels: string[]} {
  const counts = new Array<number>(binCount).fill(0);
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!(value >= 1)) continue;
    counts[Math.min(binCount - 1, Math.floor(Math.log2(value)))]++;
  }
  const labels = counts.map((_, bin) =>
    bin === binCount - 1 ? `${2 ** bin}+` : bin === 0 ? '1' : `${2 ** bin}-${2 ** (bin + 1) - 1}`
  );
  return {counts, labels};
}

/**
 * Which airports to name on the world map: the busiest airport of every continent first, then the
 * busiest remaining ones that are not within `minimumSeparation` degrees of a chosen one.
 */
export function pickHubs(
  network: FlightNetwork,
  members: readonly (readonly number[])[],
  count: number,
  minimumSeparation = 7
): number[] {
  const chosen: number[] = [];
  const isClose = (candidate: number) =>
    chosen.some(
      other =>
        Math.abs(network.lonLat[other * 2] - network.lonLat[candidate * 2]) < minimumSeparation &&
        Math.abs(network.lonLat[other * 2 + 1] - network.lonLat[candidate * 2 + 1]) <
          minimumSeparation
    );
  for (const list of members) if (list.length) chosen.push(list[0]);
  const byDegree = Array.from({length: network.nodeCount}, (_, vertex) => vertex).sort(
    (a, b) => network.degree[b] - network.degree[a] || a - b
  );
  for (const vertex of byDegree) {
    if (chosen.length >= count) break;
    if (!chosen.includes(vertex) && !isClose(vertex)) chosen.push(vertex);
  }
  return chosen.slice(0, count);
}
