// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU-side facts of the taxi-flows story that never change with the window: the lattice geometry
 * that mirrors what the GPU assigns, the ONE width scale of the whole dataset, the zone-area
 * denominators and the fixed class breaks of the zone backdrop. Everything here is computed from
 * the source rows once per source, so a colour or width change on the map is a data change.
 */

import {
  getGPUPointDensityHexagonCell,
  getGPUPointDensityHexagonCenter
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {getQuantileBreaks} from '../../cartography/breaks';
import type {FlowSource} from './b11-flow-sources';

/** `[minX, minY, maxX, maxY]` in planar metres. */
export type Bounds = [number, number, number, number];

/** The zone kinds that are lattices rather than the dataset's own areas. */
export type LatticeKind = 'hexagon' | 'grid';

/** A lattice laid over the zone centres: its bounds, size and active dimensions. */
export type Lattice = {
  kind: LatticeKind;
  bounds: Bounds;
  columns: number;
  rows: number;
  /** Hexagon radius or square cell size in metres. */
  size: number;
};

const SQRT3 = Math.sqrt(3);

/** Smallest lattice size on the slider: it sets the compile-time lattice capacity. */
export const MINIMUM_ZONE_SIZE = 400;

/** Radii of the preset chips of the zoning step, in metres. */
export const LATTICE_PRESET_SIZES: readonly number[] = [1500, 3000, 6000];

/**
 * Days of each type in 2023 (it began on a Sunday: 52 weeks and one more Sunday). The zone
 * backdrop is a rate per hour of an average day of the chosen type.
 */
export const DAYS_IN_2023 = {all: 365, weekday: 260, weekend: 105} as const;

/** Names the city data spells differently from the way people write them. */
const DISPLAY_NAMES: Readonly<Record<string, string>> = {
  Ohare: "O'Hare",
  'Mckinley Park': 'McKinley Park'
};

/** The way a community area is written on the map (`Ohare` is O'Hare). */
export function getAreaDisplayName(name: string): string {
  return DISPLAY_NAMES[name] ?? name;
}

// ---------------------------------------------------------------------------------------------
// Lattice geometry (mirrors the zone ids the GPU assigns)
// ---------------------------------------------------------------------------------------------

/** `[minX, minY, maxX, maxY]` of the zone centres. */
export function getCenterBounds(centers: Float32Array, zoneCount: number): Bounds {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let zone = 0; zone < zoneCount; zone++) {
    minX = Math.min(minX, centers[zone * 2]);
    maxX = Math.max(maxX, centers[zone * 2]);
    minY = Math.min(minY, centers[zone * 2 + 1]);
    maxY = Math.max(maxY, centers[zone * 2 + 1]);
  }
  return [minX, minY, maxX, maxY];
}

/** Lattice bounds for a zone size: the zone centres padded so every centre is inside a cell. */
export function getLatticeBounds(centerBounds: Bounds, kind: LatticeKind, size: number): Bounds {
  const [minX, minY, maxX, maxY] = centerBounds;
  const pad = kind === 'hexagon' ? 1.2 * size : 100;
  return [minX - pad, minY - pad, maxX + pad, maxY + pad];
}

/** `[columns, rows]` of a lattice over `bounds`. */
export function getLatticeGridSize(
  bounds: Bounds,
  kind: LatticeKind,
  size: number
): [number, number] {
  const width = bounds[2] - bounds[0];
  const height = bounds[3] - bounds[1];
  return kind === 'hexagon'
    ? [Math.ceil(width / (SQRT3 * size)) + 1, Math.ceil(height / (1.5 * size)) + 1]
    : [Math.ceil(width / size), Math.ceil(height / size)];
}

/** The active lattice of a zone size. */
export function getLattice(centerBounds: Bounds, kind: LatticeKind, size: number): Lattice {
  const bounds = getLatticeBounds(centerBounds, kind, size);
  const [columns, rows] = getLatticeGridSize(bounds, kind, size);
  return {kind, bounds, columns, rows, size};
}

/** Zone id of the lattice cell under a planar position, or `-1` outside the lattice. */
export function getLatticeZone(lattice: Lattice, x: number, y: number): number {
  const {bounds, columns, rows, size} = lattice;
  let column: number;
  let row: number;
  if (lattice.kind === 'hexagon') {
    [column, row] = getGPUPointDensityHexagonCell(x, y, bounds[0], bounds[1], size);
  } else {
    column = Math.floor(((x - bounds[0]) / (bounds[2] - bounds[0])) * columns);
    row = Math.floor(((y - bounds[1]) / (bounds[3] - bounds[1])) * rows);
  }
  return column >= 0 && row >= 0 && column < columns && row < rows ? row * columns + column : -1;
}

/** Planar centre of a lattice zone. */
export function getLatticeZoneCenter(lattice: Lattice, zone: number): [number, number] {
  const column = zone % lattice.columns;
  const row = Math.floor(zone / lattice.columns);
  const {bounds, size} = lattice;
  if (lattice.kind === 'hexagon') {
    return getGPUPointDensityHexagonCenter(column, row, bounds[0], bounds[1], size);
  }
  const cellWidth = (bounds[2] - bounds[0]) / lattice.columns;
  const cellHeight = (bounds[3] - bounds[1]) / lattice.rows;
  return [bounds[0] + (column + 0.5) * cellWidth, bounds[1] + (row + 0.5) * cellHeight];
}

/** Area of one lattice cell in square kilometres. */
export function getLatticeCellAreaKm2(lattice: Lattice): number {
  if (lattice.kind === 'hexagon') return (1.5 * SQRT3 * lattice.size * lattice.size) / 1e6;
  const cellWidth = (lattice.bounds[2] - lattice.bounds[0]) / lattice.columns;
  const cellHeight = (lattice.bounds[3] - lattice.bounds[1]) / lattice.rows;
  return (cellWidth * cellHeight) / 1e6;
}

// ---------------------------------------------------------------------------------------------
// The one width scale
// ---------------------------------------------------------------------------------------------

/** The largest flow and the largest interior flow of the whole dataset (one scale for all views). */
export type FlowScale = {
  /** Largest between-zone pair under any zoning of the story. Width 10 px. */
  maxFlow: number;
  /** Largest same-zone pair under any zoning: the interior circle at its full radius. */
  maxInterior: number;
};

/**
 * The widest flow and the largest interior flow of the dataset: every hour and day type summed,
 * the dataset's own areas and the preset lattices considered together, so a window, a day type
 * or a zoning toggle never rescales the map. Commutes take the maximum over every earnings
 * weight, so low, middle and high earnings compare on one scale.
 */
export function getFlowScale(
  source: FlowSource,
  weightKey: string,
  centerBounds: Bounds
): FlowScale {
  if (source.id === 'commute') {
    let maxFlow = 0;
    for (const weights of Object.values(source.weights)) {
      for (let row = 0; row < weights.length; row++) maxFlow = Math.max(maxFlow, weights[row]);
    }
    return {maxFlow, maxInterior: 0};
  }
  const weights = source.weights[weightKey];
  const zoneCount = source.zoneCount;
  // Distinct native pairs (about five thousand of the 77 x 77 possible).
  const nativePairs = new Map<number, number>();
  for (let row = 0; row < source.rowCount; row++) {
    const key = source.origin[row] * zoneCount + source.destination[row];
    nativePairs.set(key, (nativePairs.get(key) ?? 0) + weights[row]);
  }
  let maxFlow = 0;
  let maxInterior = 0;
  const consider = (zoneOf: (zone: number) => number, cellCount: number) => {
    const pairs = new Map<number, number>();
    for (const [key, weight] of nativePairs) {
      const origin = zoneOf(Math.floor(key / zoneCount));
      const destination = zoneOf(key % zoneCount);
      if (origin < 0 || destination < 0) continue;
      const cellKey = origin * cellCount + destination;
      pairs.set(cellKey, (pairs.get(cellKey) ?? 0) + weight);
    }
    for (const [key, weight] of pairs) {
      if (Math.floor(key / cellCount) === key % cellCount)
        maxInterior = Math.max(maxInterior, weight);
      else maxFlow = Math.max(maxFlow, weight);
    }
  };
  consider(zone => zone, zoneCount);
  for (const size of LATTICE_PRESET_SIZES) {
    const lattice = getLattice(centerBounds, 'hexagon', size);
    const cellOf = new Int32Array(zoneCount);
    for (let zone = 0; zone < zoneCount; zone++) {
      cellOf[zone] = getLatticeZone(
        lattice,
        source.centers[zone * 2],
        source.centers[zone * 2 + 1]
      );
    }
    consider(zone => cellOf[zone], lattice.columns * lattice.rows);
  }
  return {maxFlow, maxInterior};
}

// ---------------------------------------------------------------------------------------------
// Zone backdrop: fixed class breaks
// ---------------------------------------------------------------------------------------------

/** Divisor turning a window total into a rate: days of the type times hours of the window. */
export function getRateDivisor(
  source: FlowSource['id'],
  dayType: keyof typeof DAYS_IN_2023,
  windowHours: number
): number {
  return source === 'taxi' ? DAYS_IN_2023[dayType] * Math.max(1, windowHours) : 1;
}

/** Rounds a break to two significant digits so legend labels are honest about the precision. */
function roundBreak(value: number): number {
  return Number(value.toPrecision(2));
}

/**
 * Five quantile classes of the zone backdrop, computed once per source from every row (all hours,
 * all days): departures or arrivals between zones per km2 per hour of an average day (taxi), or
 * jobs arriving per km2 (commute, always the all-jobs weight so earnings compare). Fixed across
 * windows and weights, so a colour change is a data change.
 */
export function getZoneDensityBreaks(
  source: FlowSource,
  weightKey: string,
  totals: 'departures' | 'arrivals'
): {breaks: number[]; extent: [number, number]} {
  const weights = source.weights[source.id === 'commute' ? 'all' : weightKey];
  const totalsByZone = new Float64Array(source.zoneCount);
  const useDestination = source.id === 'commute' || totals === 'arrivals';
  for (let row = 0; row < source.rowCount; row++) {
    const origin = source.origin[row];
    const destination = source.destination[row];
    if (origin === destination) continue;
    totalsByZone[useDestination ? destination : origin] += weights[row];
  }
  const divisor = getRateDivisor(source.id, 'all', 24);
  const densities: number[] = [];
  for (let zone = 0; zone < source.zoneCount; zone++) {
    const density = totalsByZone[zone] / Math.max(source.zoneAreaKm2[zone], 1e-6) / divisor;
    if (density > 0) densities.push(density);
  }
  const breaks = getQuantileBreaks(densities, 5)
    .map(roundBreak)
    .filter((value, index, all) => value > 0 && (index === 0 || value > all[index - 1]));
  return {breaks, extent: [Math.min(...densities), Math.max(...densities)]};
}

/** A rate or count with precision that suits its size (`0.42`, `3.1`, `47`, `1,200`). */
export function formatDensity(value: number): string {
  if (!Number.isFinite(value)) return '–';
  if (value >= 1000) return Math.round(value).toLocaleString('en-US');
  if (value >= 10) return String(Math.round(value));
  if (value >= 1) return value.toFixed(1).replace(/\.0$/, '');
  return String(Number(value.toPrecision(2)));
}
