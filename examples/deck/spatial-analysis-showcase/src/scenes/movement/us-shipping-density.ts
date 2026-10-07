// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGreatCircleKilometers} from './us-shipping.stats';
import {
  CLIP_WALK_REGION,
  SHIPPING_TYPES,
  SHIPPING_TYPE_LABELS,
  type ShippingTracks
} from './us-shipping.tracks';

/**
 * The line-density grid of the `us-shipping-day` scene on the CPU side: the geometry of its
 * cells, the readback summary the colour classes are made from, the four stretches of the
 * classification lesson, and a CPU replica of the clip-and-walk that `GPULineDensity` runs so
 * the story can draw one real segment cell by cell.
 */

const EARTH_RADIUS_KILOMETERS = 6371.0088;
const DEGREES = Math.PI / 180;

/** The four ways to class a heavy-tailed value, in the order of the control. */
export const STRETCHES = ['linear', 'sqrt', 'log', 'quantile'] as const;

/** One of {@link STRETCHES}. */
export type Stretch = (typeof STRETCHES)[number];

/** Option labels of {@link STRETCHES}. */
export const STRETCH_LABELS: Record<Stretch, string> = {
  linear: 'Linear',
  sqrt: 'Square root',
  log: 'Log',
  quantile: 'Quantile'
};

/** The method line of the legend for each stretch. */
export const STRETCH_METHODS: Record<Stretch, string> = {
  linear: 'Equal-width classes up to the busiest cell',
  sqrt: 'Equal-width classes on the square root',
  log: 'Equal-width classes on the logarithm',
  quantile: 'Quantiles: a fifth of the occupied cells in each class'
};

/** What a cell holds: the track length inside it, or that length per km² of cell. */
export type DensityBasis = 'length' | 'density';

/** The grid the density graph runs on: `[west, south, east, north]` degrees and cell counts. */
export type DensityGrid = {
  columns: number;
  rows: number;
  bounds: readonly [number, number, number, number];
};

/** The density graph's readback, with the sorted occupied values every class table needs. */
export type DensityField = {
  grid: DensityGrid;
  /** Track length per cell in meters (row 0 is the south row). */
  lengths: Float32Array;
  /** Meters of track per square meter of cell. */
  densities: Float32Array;
  occupiedCount: number;
  totalKilometers: number;
  /** Segment-cell pieces the clip-and-walk emitted. */
  pieces: number;
  overflow: boolean;
  /** Occupied cells' track length in km, ascending. */
  sortedKilometers: Float32Array;
  /** Occupied cells' track length in km per km², ascending. */
  sortedDensity: Float32Array;
};

/** Meters-to-display factor: km for a length, km per km² for a density. */
export function getBasisScale(basis: DensityBasis): number {
  return basis === 'length' ? 0.001 : 1000;
}

/** The sorted occupied values of a basis, in display units. */
export function getSortedValues(field: DensityField, basis: DensityBasis): Float32Array {
  return basis === 'length' ? field.sortedKilometers : field.sortedDensity;
}

/** Summarises a readback of the two output buffers. */
export function summarizeDensityField(
  grid: DensityGrid,
  lengths: Float32Array,
  densities: Float32Array,
  pieces: number,
  overflow: boolean
): DensityField {
  let occupiedCount = 0;
  let totalMeters = 0;
  for (let cell = 0; cell < lengths.length; cell++) {
    if (lengths[cell] > 0) {
      occupiedCount++;
      totalMeters += lengths[cell];
    }
  }
  const sortedKilometers = new Float32Array(occupiedCount);
  const sortedDensity = new Float32Array(occupiedCount);
  let row = 0;
  for (let cell = 0; cell < lengths.length; cell++) {
    if (lengths[cell] > 0) {
      sortedKilometers[row] = lengths[cell] * 0.001;
      sortedDensity[row] = densities[cell] * 1000;
      row++;
    }
  }
  sortedKilometers.sort();
  sortedDensity.sort();
  return {
    grid,
    lengths,
    densities,
    occupiedCount,
    totalKilometers: totalMeters / 1000,
    pieces,
    overflow,
    sortedKilometers,
    sortedDensity
  };
}

/** Value of one cell in display units. */
export function getCellValue(field: DensityField, cell: number, basis: DensityBasis): number {
  return (basis === 'length' ? field.lengths[cell] : field.densities[cell]) * getBasisScale(basis);
}

/** Index of the occupied cell with the largest value of `basis`, or `-1` when none is occupied. */
export function getBusiestCell(field: DensityField, basis: DensityBasis): number {
  const values = basis === 'length' ? field.lengths : field.densities;
  let best = -1;
  let bestValue = 0;
  for (let cell = 0; cell < values.length; cell++) {
    if (values[cell] > bestValue) {
      bestValue = values[cell];
      best = cell;
    }
  }
  return best;
}

/** Cell index of a longitude and latitude, or `-1` outside the grid. */
export function getCellIndexAt(grid: DensityGrid, longitude: number, latitude: number): number {
  const [west, south, east, north] = grid.bounds;
  if (longitude < west || longitude >= east || latitude < south || latitude >= north) return -1;
  const column = Math.floor(((longitude - west) / (east - west)) * grid.columns);
  const row = Math.floor(((latitude - south) / (north - south)) * grid.rows);
  return row * grid.columns + column;
}

/** `[west, south, east, north]` of a cell. */
export function getCellBounds(grid: DensityGrid, cell: number): [number, number, number, number] {
  const [west, south, east, north] = grid.bounds;
  const width = (east - west) / grid.columns;
  const height = (north - south) / grid.rows;
  const column = cell % grid.columns;
  const row = Math.floor(cell / grid.columns);
  return [
    west + column * width,
    south + row * height,
    west + (column + 1) * width,
    south + (row + 1) * height
  ];
}

/** Centre `[longitude, latitude]` of a cell. */
export function getCellCenter(grid: DensityGrid, cell: number): [number, number] {
  const [west, south, east, north] = getCellBounds(grid, cell);
  return [(west + east) / 2, (south + north) / 2];
}

/** Ground size of a cell at `latitude`: width shrinks with the cosine of the latitude. */
export function getCellSizeKilometers(
  grid: DensityGrid,
  latitude: number
): {width: number; height: number} {
  const [west, south, east, north] = grid.bounds;
  return {
    width:
      ((east - west) / grid.columns) *
      DEGREES *
      EARTH_RADIUS_KILOMETERS *
      Math.cos(latitude * DEGREES),
    height: ((north - south) / grid.rows) * DEGREES * EARTH_RADIUS_KILOMETERS
  };
}

/** Exact spherical area in km² of a cell in row `row`. */
export function getCellAreaSquareKilometers(grid: DensityGrid, row: number): number {
  const [west, south, east, north] = grid.bounds;
  const width = ((east - west) / grid.columns) * DEGREES;
  const latitudeSouth = (south + (row * (north - south)) / grid.rows) * DEGREES;
  const latitudeNorth = (south + ((row + 1) * (north - south)) / grid.rows) * DEGREES;
  return (
    EARTH_RADIUS_KILOMETERS *
    EARTH_RADIUS_KILOMETERS *
    width *
    (Math.sin(latitudeNorth) - Math.sin(latitudeSouth))
  );
}

/** Makes class breaks strictly ascending (a class never has zero width). */
function makeStrictlyAscending(breaks: number[]): number[] {
  const result: number[] = [];
  for (const value of breaks) {
    const previous = result.length > 0 ? result[result.length - 1] : -Infinity;
    result.push(value > previous ? value : previous + Math.abs(previous) * 1e-6 + 1e-12);
  }
  return result;
}

/**
 * The interior breaks of `classCount` classes over the occupied cells' `sorted` values (ascending,
 * all positive) for one stretch. Linear, square-root and log classes are equal in the transformed
 * value between zero (the minimum, for the log) and the maximum; quantile classes hold an equal
 * share of the occupied cells.
 */
export function getStretchBreaks(
  sorted: ArrayLike<number>,
  stretch: Stretch,
  classCount = 5
): number[] {
  const count = sorted.length;
  if (count === 0) return [];
  const minimum = sorted[0];
  const maximum = sorted[count - 1];
  const breaks: number[] = [];
  for (let index = 1; index < classCount; index++) {
    const fraction = index / classCount;
    switch (stretch) {
      case 'linear':
        breaks.push(maximum * fraction);
        break;
      case 'sqrt':
        breaks.push(maximum * fraction * fraction);
        break;
      case 'log':
        breaks.push(minimum * (maximum / minimum) ** fraction);
        break;
      default:
        breaks.push(sorted[Math.min(count - 1, Math.floor(fraction * count))]);
    }
  }
  return makeStrictlyAscending(breaks);
}

/** Share (0-1) of the occupied cells whose value is at or below `value`. */
export function getShareAtOrBelow(sorted: ArrayLike<number>, value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return sorted.length > 0 ? low / sorted.length : Number.NaN;
}

/** Median of the occupied cells' values (the array is ascending). */
export function getSortedMedian(sorted: ArrayLike<number>): number {
  const count = sorted.length;
  if (count === 0) return Number.NaN;
  return count % 2 === 1 ? sorted[count >> 1] : (sorted[count / 2 - 1] + sorted[count / 2]) / 2;
}

/** A density value with three significant figures: `0.0042`, `1.84`, `18.4`, `1,842`. */
export function formatDensityValue(value: number): string {
  if (!Number.isFinite(value)) return '–';
  if (value >= 1000) return Math.round(value).toLocaleString('en-US');
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(2);
  if (value >= 0.01) return value.toFixed(3);
  return value.toPrecision(2);
}

/**
 * Counts per bin of `log10(value)` between the smallest and the largest occupied value, so the
 * heavy tail and the bulk of the lanes show in one histogram.
 */
export function getLogHistogram(
  sorted: ArrayLike<number>,
  binCount: number
): {counts: number[]; domain: [number, number]} {
  const count = sorted.length;
  if (count === 0) return {counts: new Array<number>(binCount).fill(0), domain: [0, 1]};
  const low = Math.log10(sorted[0]);
  const high = Math.log10(sorted[count - 1]);
  const span = Math.max(high - low, 1e-9);
  const counts = new Array<number>(binCount).fill(0);
  for (let index = 0; index < count; index++) {
    const bin = Math.min(
      binCount - 1,
      Math.floor(((Math.log10(sorted[index]) - low) / span) * binCount)
    );
    counts[bin]++;
  }
  return {counts, domain: [low, high]};
}

// ---------------------------------------------------------------------------------------------
// Clip and walk
// ---------------------------------------------------------------------------------------------

/** One piece of a segment: the part of it inside one cell, and its great-circle length. */
export type ClipPiece = {
  column: number;
  row: number;
  from: [number, number];
  to: [number, number];
  kilometers: number;
};

/**
 * What `GPULineDensity` does to one segment: clip it to the grid (the straight line in longitude
 * and latitude), cut it where it crosses a cell edge, and measure each piece as a great-circle
 * distance. A piece on a cell edge belongs to the upper or right cell, as on the GPU.
 */
export function clipAndWalk(
  grid: DensityGrid,
  from: readonly [number, number],
  to: readonly [number, number]
): ClipPiece[] {
  const [west, south, east, north] = grid.bounds;
  const width = (east - west) / grid.columns;
  const height = (north - south) / grid.rows;
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const parameters = [0, 1];
  if (dx !== 0) {
    const first = Math.ceil((Math.min(from[0], to[0]) - west) / width);
    const last = Math.floor((Math.max(from[0], to[0]) - west) / width);
    for (let line = first; line <= last; line++) {
      const t = (west + line * width - from[0]) / dx;
      if (t > 0 && t < 1) parameters.push(t);
    }
  }
  if (dy !== 0) {
    const first = Math.ceil((Math.min(from[1], to[1]) - south) / height);
    const last = Math.floor((Math.max(from[1], to[1]) - south) / height);
    for (let line = first; line <= last; line++) {
      const t = (south + line * height - from[1]) / dy;
      if (t > 0 && t < 1) parameters.push(t);
    }
  }
  parameters.sort((a, b) => a - b);
  const pieces: ClipPiece[] = [];
  for (let index = 1; index < parameters.length; index++) {
    const start = parameters[index - 1];
    const end = parameters[index];
    if (end - start < 1e-9) continue;
    const middle = (start + end) / 2;
    const column = Math.floor((from[0] + dx * middle - west) / width);
    const row = Math.floor((from[1] + dy * middle - south) / height);
    if (column < 0 || column >= grid.columns || row < 0 || row >= grid.rows) continue;
    const pieceFrom: [number, number] = [from[0] + dx * start, from[1] + dy * start];
    const pieceTo: [number, number] = [from[0] + dx * end, from[1] + dy * end];
    pieces.push({
      column,
      row,
      from: pieceFrom,
      to: pieceTo,
      kilometers: getGreatCircleKilometers(pieceFrom, pieceTo)
    });
  }
  return pieces;
}

/** The real segment the clip-and-walk diagram draws. */
export type ClipWalkSegment = {
  from: [number, number];
  to: [number, number];
  /** Vessel type label of the track. */
  typeLabel: string;
  kilometers: number;
  /** Seconds between its two fixes. */
  seconds: number;
};

/**
 * The longest segment of a tanker (else a cargo ship, else any vessel) inside
 * {@link CLIP_WALK_REGION}: a long straight chord across open water that crosses several cells.
 */
export function pickClipWalkSegment(tracks: ShippingTracks): ClipWalkSegment | null {
  const [west, south, east, north] = CLIP_WALK_REGION;
  const tanker = SHIPPING_TYPES.indexOf('tanker');
  const cargo = SHIPPING_TYPES.indexOf('cargo');
  const best: Record<'tanker' | 'cargo' | 'any', {segment: number; kilometers: number}> = {
    tanker: {segment: -1, kilometers: 0},
    cargo: {segment: -1, kilometers: 0},
    any: {segment: -1, kilometers: 0}
  };
  for (let segment = 0; segment < tracks.segmentCount; segment++) {
    const fromLongitude = tracks.segments[segment * 4];
    const fromLatitude = tracks.segments[segment * 4 + 1];
    const toLongitude = tracks.segments[segment * 4 + 2];
    const toLatitude = tracks.segments[segment * 4 + 3];
    if (
      Math.min(fromLongitude, toLongitude) < west ||
      Math.max(fromLongitude, toLongitude) > east ||
      Math.min(fromLatitude, toLatitude) < south ||
      Math.max(fromLatitude, toLatitude) > north
    ) {
      continue;
    }
    const kilometers = getGreatCircleKilometers(
      [fromLongitude, fromLatitude],
      [toLongitude, toLatitude]
    );
    const type = tracks.category[tracks.segmentTracks[segment]];
    if (type === tanker && kilometers > best.tanker.kilometers) best.tanker = {segment, kilometers};
    if (type === cargo && kilometers > best.cargo.kilometers) best.cargo = {segment, kilometers};
    if (kilometers > best.any.kilometers) best.any = {segment, kilometers};
  }
  const chosen = [best.tanker, best.cargo, best.any].find(candidate => candidate.segment >= 0);
  if (!chosen) return null;
  const segment = chosen.segment;
  return {
    from: [tracks.segments[segment * 4], tracks.segments[segment * 4 + 1]],
    to: [tracks.segments[segment * 4 + 2], tracks.segments[segment * 4 + 3]],
    typeLabel: SHIPPING_TYPE_LABELS[SHIPPING_TYPES[tracks.category[tracks.segmentTracks[segment]]]],
    kilometers: chosen.kilometers,
    seconds: tracks.segmentEndTimes[segment] - tracks.segmentStartTimes[segment]
  };
}
