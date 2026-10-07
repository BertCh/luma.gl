// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The CPU side of `points/street-density`: what is read from the street vertices (bearings, the
 * east-west arterial profile), from a density readback (city cells, class counts, histogram,
 * gaps) and from polygon totals (the rank flip). Pure TypeScript with no GPU or engine imports.
 * The GPU does the length-per-cell work; these functions only describe and rank its results.
 */

import {getClassCounts, getHistogram} from '../../cartography/breaks';
import {
  GRID_TOLERANCE_DEGREES,
  LATTICE_TOLERANCE_METERS,
  MILE_METERS,
  MILE_PROFILE_BIN_METERS,
  ROSE_BIN_COUNT
} from './street-density-look';

/** The unique streets in local metres, as the density graph reads them. */
export type StreetVertices = {
  /** `x, y` pairs in metres, polylines back to back. */
  positions: Float32Array;
  /** `pathCount + 1` offsets into `positions` (in vertices). */
  pathOffsets: Uint32Array;
  /** OSM class per path (0 motorway, 1 trunk, 2 primary, 3 secondary, 4 tertiary, 5 residential, 6 other). */
  pathClass: Uint8Array;
};

// ---------------------------------------------------------------------------------------------
// Bearings
// ---------------------------------------------------------------------------------------------

/** Length-weighted street bearings. */
export type BearingStatistics = {
  /** Street length in km per 10-degree bin, folded (bin `i` and bin `i + 18` are equal). */
  rose: number[];
  /** Share of the length within the grid tolerance of north-south or east-west (0 to 1). */
  gridShare: number;
  /** Total street length in metres. */
  totalMeters: number;
};

/**
 * Bearings of every street segment, weighted by its length and folded to 0-180 degrees so a street
 * and its opposite direction count together. A segment is "on the grid" when its bearing is within
 * {@link GRID_TOLERANCE_DEGREES} of an axis; the rose bins are 10 degrees wide and centred on the
 * axes, so two of the 18 folded bins are exactly the grid.
 */
export function getBearingStatistics(streets: StreetVertices): BearingStatistics {
  const half = ROSE_BIN_COUNT / 2;
  const binDegrees = 360 / ROSE_BIN_COUNT;
  const folded = new Float64Array(half);
  let onGrid = 0;
  let total = 0;
  forEachSegment(streets, (x0, y0, x1, y1) => {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const length = Math.hypot(dx, dy);
    if (length === 0) return;
    const bearing = getFoldedBearing(dx, dy);
    folded[Math.round(bearing / binDegrees) % half] += length;
    const offAxis = bearing % 90;
    if (Math.min(offAxis, 90 - offAxis) <= GRID_TOLERANCE_DEGREES) onGrid += length;
    total += length;
  });
  const rose = Array.from({length: ROSE_BIN_COUNT}, (_, index) => folded[index % half] / 1000);
  return {rose, gridShare: total > 0 ? onGrid / total : 0, totalMeters: total};
}

/** Bearing of a direction in degrees clockwise from north, folded into `[0, 180)`. */
function getFoldedBearing(dx: number, dy: number): number {
  const degrees = (Math.atan2(dx, dy) * 180) / Math.PI;
  return ((degrees % 180) + 180) % 180;
}

function forEachSegment(
  streets: StreetVertices,
  visit: (x0: number, y0: number, x1: number, y1: number, pathClass: number) => void
): void {
  const {positions, pathOffsets, pathClass} = streets;
  for (let path = 0; path < pathClass.length; path++) {
    for (let vertex = pathOffsets[path]; vertex + 1 < pathOffsets[path + 1]; vertex++) {
      visit(
        positions[vertex * 2],
        positions[vertex * 2 + 1],
        positions[vertex * 2 + 2],
        positions[vertex * 2 + 3],
        pathClass[path]
      );
    }
  }
}

// ---------------------------------------------------------------------------------------------
// The mile grid
// ---------------------------------------------------------------------------------------------

/** East-west arterial length by north-south position, and how well a survey lattice fits it. */
export type ArterialProfile = {
  /** East-west arterial length in km per bin, south to north. */
  bins: number[];
  /** Bin height in metres. */
  binMeters: number;
  /** Position of the profile's south edge in local metres (y). */
  southMeters: number;
  /** Offset in metres from the south edge of the best-fitting mile lattice. */
  milePhaseMeters: number;
  /** Share of the length within the lattice tolerance of the best mile lattice (0 to 1). */
  mileShare: number;
  /** The same for the best half-mile lattice. */
  halfMileShare: number;
  /** Share expected from a lattice placed at random (tolerance times two over the spacing). */
  mileChance: number;
  halfMileChance: number;
  /** East-west arterial length in the corridor, metres. */
  totalMeters: number;
};

/**
 * East-west primary and secondary roads (bearing within the grid tolerance of east-west) inside
 * `corridor` `[minX, minY, maxX, maxY]` metres, as a profile by north-south position. The mile and
 * half-mile lattices are fitted by trying every phase in 20 m steps: the share is what the best
 * phase catches, against what an arbitrary one catches by chance.
 */
export function getArterialProfile(
  streets: StreetVertices,
  corridor: readonly [number, number, number, number]
): ArterialProfile {
  const [minX, minY, maxX, maxY] = corridor;
  const binMeters = MILE_PROFILE_BIN_METERS;
  const bins = new Float64Array(Math.max(1, Math.ceil((maxY - minY) / binMeters)));
  const positions: number[] = [];
  const lengths: number[] = [];
  let total = 0;
  forEachSegment(streets, (x0, y0, x1, y1, pathClass) => {
    if (pathClass < 2 || pathClass > 3) return;
    const dx = x1 - x0;
    const dy = y1 - y0;
    const length = Math.hypot(dx, dy);
    if (length < 1) return;
    if (Math.abs(getFoldedBearing(dx, dy) - 90) > GRID_TOLERANCE_DEGREES) return;
    const x = (x0 + x1) / 2;
    const y = (y0 + y1) / 2;
    if (x < minX || x > maxX || y < minY || y > maxY) return;
    bins[Math.min(bins.length - 1, Math.floor((y - minY) / binMeters))] += length;
    positions.push(y - minY);
    lengths.push(length);
    total += length;
  });
  const fit = (spacing: number) => {
    let bestShare = 0;
    let bestPhase = 0;
    for (let phase = 0; phase < spacing; phase += 20) {
      let caught = 0;
      for (let index = 0; index < positions.length; index++) {
        let offset = (((positions[index] - phase) % spacing) + spacing) % spacing;
        if (offset > spacing / 2) offset -= spacing;
        if (Math.abs(offset) <= LATTICE_TOLERANCE_METERS) caught += lengths[index];
      }
      if (caught > bestShare) {
        bestShare = caught;
        bestPhase = phase;
      }
    }
    return {share: total > 0 ? bestShare / total : 0, phase: bestPhase};
  };
  const mile = fit(MILE_METERS);
  const half = fit(MILE_METERS / 2);
  return {
    bins: Array.from(bins, value => value / 1000),
    binMeters,
    southMeters: minY,
    milePhaseMeters: mile.phase,
    mileShare: mile.share,
    halfMileShare: half.share,
    mileChance: (2 * LATTICE_TOLERANCE_METERS) / MILE_METERS,
    halfMileChance: (2 * LATTICE_TOLERANCE_METERS) / (MILE_METERS / 2),
    totalMeters: total
  };
}

// ---------------------------------------------------------------------------------------------
// The density field
// ---------------------------------------------------------------------------------------------

/**
 * Marks the cells of a `columns x rows` grid whose centre lies inside the rings (even-odd over all
 * rings, so holes and separate parts both work). The grid spans `bounds` `[minX, minY, maxX,
 * maxY]` in metres and row 0 is the south edge. A scanline fill: linear in ring edges plus cells.
 */
export function rasterizeRingsMask(
  rings: readonly Float64Array[],
  bounds: readonly [number, number, number, number],
  columns: number,
  rows: number
): Uint8Array {
  const mask = new Uint8Array(columns * rows);
  const [minX, minY, maxX, maxY] = bounds;
  const cellWidth = (maxX - minX) / columns;
  const cellHeight = (maxY - minY) / rows;
  const crossings: number[][] = Array.from({length: rows}, () => []);
  for (const ring of rings) {
    const count = ring.length / 2;
    for (let index = 0, previous = count - 1; index < count; previous = index++) {
      const x0 = (ring[previous * 2] - minX) / cellWidth;
      const y0 = (ring[previous * 2 + 1] - minY) / cellHeight;
      const x1 = (ring[index * 2] - minX) / cellWidth;
      const y1 = (ring[index * 2 + 1] - minY) / cellHeight;
      if (y0 === y1) continue;
      const rowStart = Math.max(0, Math.ceil(Math.min(y0, y1) - 0.5));
      const rowEnd = Math.min(rows - 1, Math.ceil(Math.max(y0, y1) - 0.5) - 1);
      for (let row = rowStart; row <= rowEnd; row++) {
        crossings[row].push(x0 + ((row + 0.5 - y0) / (y1 - y0)) * (x1 - x0));
      }
    }
  }
  crossings.forEach((xs, row) => {
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const first = Math.max(0, Math.ceil(xs[k] - 0.5));
      const last = Math.min(columns - 1, Math.ceil(xs[k + 1] - 0.5) - 1);
      if (last >= first) mask.fill(1, row * columns + first, row * columns + last + 1);
    }
  });
  return mask;
}

/** What a density readback says about the cells inside the city. */
export type FieldStatistics = {
  /** Densities in km of street per km² of the non-empty city cells, ascending. */
  sorted: Float32Array;
  /** Cells inside the city (centre inside the limit), empty or not. */
  cityCells: number;
  /** Non-empty city cells per class of `breaks` (km per km²). */
  classCounts: number[];
  /** Non-empty city cells per histogram bin over `[0, histogramTop]`. */
  histogram: number[];
  /** Index (row * columns + column) and density of the busiest city cell, or -1. */
  peakIndex: number;
  peak: number;
  /** Median density of the non-empty city cells. */
  median: number;
};

/** Bins of the density histogram. */
export const HISTOGRAM_BINS = 30;

/**
 * Describes the density cells inside the city. `densities` are metres of street per m² (x 1000 is
 * km per km²); cells outside the city and cells with no street are left out, because "no data" and
 * "none" are different statements.
 */
export function getFieldStatistics(
  densities: Float32Array,
  cityMask: Uint8Array,
  breaks: readonly number[],
  histogramTop: number
): FieldStatistics {
  const values: number[] = [];
  let cityCells = 0;
  let peakIndex = -1;
  let peak = 0;
  for (let index = 0; index < densities.length; index++) {
    if (!cityMask[index]) continue;
    cityCells++;
    const density = densities[index] * 1000;
    if (!(density > 0)) continue;
    values.push(density);
    if (density > peak) {
      peak = density;
      peakIndex = index;
    }
  }
  const sorted = Float32Array.from(values).sort();
  return {
    sorted,
    cityCells,
    classCounts: getClassCounts(sorted, breaks),
    histogram: getHistogram(
      sorted.map(value => Math.min(value, histogramTop)),
      [0, histogramTop],
      HISTOGRAM_BINS
    ),
    peakIndex,
    peak,
    median: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0
  };
}

/** Number of values at or below `value` in an ascending array (binary search). */
export function countAtOrBelow(sorted: ArrayLike<number>, value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (sorted[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return low;
}

/** One connected gap in the street grid. */
export type VoidRegion = {
  /** Cells in the region. */
  cells: number;
  /** Centre of the region in local metres. */
  x: number;
  y: number;
  /** The region's cell index nearest its centre (a cell that is itself part of the gap). */
  centerIndex: number;
};

/**
 * Finds the connected gaps of the street grid: city cells under `threshold` km per km² (empty
 * cells included), joined through their eight neighbours. Gaps that touch the edge of the grid are
 * dropped (the grid cuts them off) and so are gaps under `minimumCells`. Largest first.
 */
export function findVoids(
  densities: Float32Array,
  cityMask: Uint8Array,
  columns: number,
  rows: number,
  bounds: readonly [number, number, number, number],
  threshold: number,
  minimumCells: number
): VoidRegion[] {
  const cellWidth = (bounds[2] - bounds[0]) / columns;
  const cellHeight = (bounds[3] - bounds[1]) / rows;
  const isGap = (index: number) => cityMask[index] === 1 && densities[index] * 1000 < threshold;
  const visited = new Uint8Array(columns * rows);
  const regions: VoidRegion[] = [];
  const stack: number[] = [];
  for (let start = 0; start < visited.length; start++) {
    if (visited[start] || !isGap(start)) continue;
    visited[start] = 1;
    stack.push(start);
    const members: number[] = [];
    let touchesEdge = false;
    while (stack.length) {
      const index = stack.pop() as number;
      members.push(index);
      const column = index % columns;
      const row = (index - column) / columns;
      if (column === 0 || row === 0 || column === columns - 1 || row === rows - 1) {
        touchesEdge = true;
      }
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const x = column + dx;
          const y = row + dy;
          if (x < 0 || y < 0 || x >= columns || y >= rows) continue;
          const neighbour = y * columns + x;
          if (visited[neighbour] || !isGap(neighbour)) continue;
          visited[neighbour] = 1;
          stack.push(neighbour);
        }
      }
    }
    if (touchesEdge || members.length < minimumCells) continue;
    let sumColumn = 0;
    let sumRow = 0;
    for (const index of members) {
      sumColumn += index % columns;
      sumRow += Math.floor(index / columns);
    }
    const centerColumn = sumColumn / members.length;
    const centerRow = sumRow / members.length;
    let centerIndex = members[0];
    let nearest = Infinity;
    for (const index of members) {
      const distance = (index % columns) - centerColumn;
      const rowDistance = Math.floor(index / columns) - centerRow;
      const squared = distance * distance + rowDistance * rowDistance;
      if (squared < nearest) {
        nearest = squared;
        centerIndex = index;
      }
    }
    regions.push({
      cells: members.length,
      x: bounds[0] + (centerColumn + 0.5) * cellWidth,
      y: bounds[1] + (centerRow + 0.5) * cellHeight,
      centerIndex
    });
  }
  return regions.sort((a, b) => b.cells - a.cells);
}

// ---------------------------------------------------------------------------------------------
// Polygons: ranks
// ---------------------------------------------------------------------------------------------

/** Rank (1 = largest) of every value; ties share the lower rank number. */
export function getRanks(values: ArrayLike<number>): number[] {
  const order = Array.from({length: values.length}, (_, index) => index).sort(
    (a, b) => values[b] - values[a]
  );
  const ranks = new Array<number>(values.length).fill(0);
  order.forEach((feature, position) => {
    const tied = position > 0 && values[order[position - 1]] === values[feature];
    ranks[feature] = tied ? ranks[order[position - 1]] : position + 1;
  });
  return ranks;
}

/** One row of the rank-flip slope chart. */
export type RankFlipRow = {
  feature: number;
  rankByLength: number;
  rankByDensity: number;
};

/**
 * The rows of the rank-flip chart: the features in the top `topCount` by street length or by
 * street length per area, with both ranks, and the `moverCount` biggest movers among them.
 */
export function getRankFlipRows(
  lengths: ArrayLike<number>,
  densities: ArrayLike<number>,
  topCount: number,
  moverCount: number
): {rows: RankFlipRow[]; movers: Set<number>} {
  const lengthRanks = getRanks(lengths);
  const densityRanks = getRanks(densities);
  const rows: RankFlipRow[] = [];
  for (let feature = 0; feature < lengths.length; feature++) {
    if (lengthRanks[feature] <= topCount || densityRanks[feature] <= topCount) {
      rows.push({
        feature,
        rankByLength: lengthRanks[feature],
        rankByDensity: densityRanks[feature]
      });
    }
  }
  const movers = new Set(
    [...rows]
      .sort(
        (a, b) =>
          Math.abs(b.rankByLength - b.rankByDensity) - Math.abs(a.rankByLength - a.rankByDensity)
      )
      .slice(0, moverCount)
      .map(row => row.feature)
  );
  return {rows, movers};
}
