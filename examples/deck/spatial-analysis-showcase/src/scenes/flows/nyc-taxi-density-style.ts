// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Scales, class tables and the load-time reference field of the nyc-taxi-density story (design
 * sheet `FID/design/nyc-taxi-density.md`).
 *
 * The map's grid follows the camera, so a cell's ride count changes with the zoom. Every colour
 * scale here is therefore written in **trips per km2**, and its breaks are computed ONCE at load
 * from a fixed reference grid ({@link buildDensityReference}) and never move with the camera,
 * the brush or the step.
 */

import {getClassBreaks, getQuantileBreaks} from '../../cartography/breaks';
import {makeClassTable} from '../../cartography/class-table';
import {NYC} from '../../cartography/gazetteer';
import type {ClassTable} from '../../cartography/types';
import {getClassColors} from '../../engine/ramps';
import {formatTaxiTime, type TaxiTrips} from './nyc-taxi-data';

/** Dark-ground ramps of the count fields: pickups warm, drop-offs cool, both trimmed (rule 3). */
export const DENSITY_RAMP = 'inferno' as const;
export const DROPOFF_DENSITY_RAMP = 'mako' as const;
export const DENSITY_RAMP_RANGE = [0.15, 1] as const;
/** Classes of the quantile stretch. */
export const QUANTILE_CLASS_COUNT = 7;
/** Cell of the reference grid, between the city-wide cell (about 900 m) and the street cell. */
export const REFERENCE_CELL_METERS = 500;
/** Rides a cell needs before its mean fare is drawn (the default of the slider). */
export const DEFAULT_MINIMUM_RIDES = 20;
/** Radius of the "core" around the gazetteer Midtown for the pickup and drop-off shares. */
export const CORE_RADIUS_METERS = 4000;
/** The one-window length (hours) of the time presets, so windows are comparable. */
export const WINDOW_HOURS = 3;
/** Presets of the time step: start hours since midnight on Thursday 1 January. */
export const WINDOW_STARTS = [0, 31, 18] as const;

/** Output of {@link buildDensityReference}: everything the scales need, read from the data. */
export type DensityReference = {
  /** Brightest colour of the whole-period field: the 98th percentile of non-empty cells, per km2. */
  fullClip: number;
  /** Interior breaks of the quantile stretch, per km2 (seven classes). */
  quantileBreaks: number[];
  /**
   * Brightest colour per hour of window: the largest 98th percentile of the preset windows over
   * their length, per km2 per hour. A window of `h` hours uses `h` times this value, so windows
   * of different lengths share one scale.
   */
  windowClipPerHour: number;
  /** Interior breaks of the mean-fare classes in whole dollars (natural breaks of cell means). */
  meanFareBreaks: number[];
  /** Share of pickups and of drop-offs within {@link CORE_RADIUS_METERS} of Midtown. */
  coreShare: {pickups: number; dropoffs: number};
};

function getPercentile(sorted: ArrayLike<number>, share: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
}

/**
 * Counts pickups on a fixed square grid over the data bounds. Optional `window` keeps the pickups
 * in `[from, to)` hours; `fareSums` accumulates the fare per cell.
 */
function binPickups(
  trips: TaxiTrips,
  window: readonly [number, number] | null
): {counts: Float64Array; fareSums: Float64Array} {
  const [minX, minY, maxX, maxY] = trips.bounds;
  const columns = Math.floor((maxX - minX) / REFERENCE_CELL_METERS) + 1;
  const rows = Math.floor((maxY - minY) / REFERENCE_CELL_METERS) + 1;
  const counts = new Float64Array(columns * rows);
  const fareSums = new Float64Array(columns * rows);
  for (let row = 0; row < trips.count; row++) {
    const hour = trips.pickupHour[row];
    if (window && (hour < window[0] || hour >= window[1])) continue;
    const column = Math.floor((trips.pickup[row * 2] - minX) / REFERENCE_CELL_METERS);
    const cellRow = Math.floor((trips.pickup[row * 2 + 1] - minY) / REFERENCE_CELL_METERS);
    const cell = cellRow * columns + column;
    counts[cell]++;
    fareSums[cell] += trips.fare[row];
  }
  return {counts, fareSums};
}

/** Per-km2 densities of the non-empty cells, ascending. */
function getDensities(counts: Float64Array): Float64Array {
  const area = (REFERENCE_CELL_METERS * REFERENCE_CELL_METERS) / 1e6;
  const densities: number[] = [];
  for (const count of counts) if (count > 0) densities.push(count / area);
  return Float64Array.from(densities).sort();
}

/**
 * Reads the scales of the story from the loaded trips, once: the clip and the quantile breaks of
 * the whole-period density, the clip of a window, the mean-fare breaks and the core shares.
 */
export function buildDensityReference(trips: TaxiTrips): DensityReference {
  const whole = binPickups(trips, null);
  const densities = getDensities(whole.counts);
  const fullClip = getPercentile(densities, 0.98);
  const quantileBreaks = getQuantileBreaks(densities, QUANTILE_CLASS_COUNT);

  let windowClipPerHour = 0;
  for (const start of WINDOW_STARTS) {
    const window = binPickups(trips, [start, start + WINDOW_HOURS]);
    windowClipPerHour = Math.max(
      windowClipPerHour,
      getPercentile(getDensities(window.counts), 0.98) / WINDOW_HOURS
    );
  }

  const means: number[] = [];
  for (let cell = 0; cell < whole.counts.length; cell++) {
    if (whole.counts[cell] >= DEFAULT_MINIMUM_RIDES) {
      means.push(whole.fareSums[cell] / whole.counts[cell]);
    }
  }
  // Whole dollars read the legend easily; a rounding that merges two breaks drops one class.
  const meanFareBreaks = [
    ...new Set(getClassBreaks(means, 6, 'natural-breaks').map(value => Math.round(value)))
  ].sort((a, b) => a - b);

  const [longitude, latitude] = NYC.places.midtown.lngLat;
  const [centerX, centerY] = trips.project(longitude, latitude);
  let corePickups = 0;
  let coreDropoffs = 0;
  for (let row = 0; row < trips.count; row++) {
    if (
      Math.hypot(trips.pickup[row * 2] - centerX, trips.pickup[row * 2 + 1] - centerY) <=
      CORE_RADIUS_METERS
    ) {
      corePickups++;
    }
    if (
      Math.hypot(trips.dropoff[row * 2] - centerX, trips.dropoff[row * 2 + 1] - centerY) <=
      CORE_RADIUS_METERS
    ) {
      coreDropoffs++;
    }
  }
  return {
    fullClip,
    quantileBreaks,
    windowClipPerHour,
    meanFareBreaks,
    coreShare: {pickups: corePickups / trips.count, dropoffs: coreDropoffs / trips.count}
  };
}

/** Formats a density (trips per km2) with thousands separators. */
export function formatDensity(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

/**
 * The quantile stretch as one class table: seven classes of the chosen dark ramp, `breaks` in
 * trips per km2 (already scaled to the window), shared by the layer, the legend and the tooltip.
 */
export function makeQuantileDensityTable(
  breaks: readonly number[],
  ramp: typeof DENSITY_RAMP | typeof DROPOFF_DENSITY_RAMP,
  extentMaximum: number
): ClassTable {
  return makeClassTable({
    breaks,
    colors: getClassColors(ramp, breaks.length + 1, false, 255, DENSITY_RAMP_RANGE),
    unit: 'trips',
    format: formatDensity,
    extent: [0, extentMaximum],
    method: 'Quantiles of the cells at load, fixed',
    noData: {label: 'No rides in the cell'}
  });
}

/**
 * Mean fare as paper classes: `YlGnBu` (cool, so a price is not read as the warm count glow of
 * the other steps; a fare is neither harm nor people), breaks fixed at load, cells with too few
 * rides hatched.
 */
export function makeMeanFareTable(
  breaks: readonly number[],
  ground: 'light' | 'dark',
  minimumRides: number
): ClassTable {
  return makeClassTable({
    breaks,
    scheme: 'YlGnBu',
    ground,
    unit: 'USD per ride',
    format: value => `$${value}`,
    method: 'Natural breaks of the cell means, fixed at load',
    noData: {label: `Fewer than ${minimumRides} rides`, hatched: true}
  });
}

/** `Thu 1 Jan 14:00` to `Thu 14:00` for window labels. */
export function formatWindowTime(hours: number): string {
  return formatTaxiTime(hours).replace(/ \d+ Jan/, '');
}
