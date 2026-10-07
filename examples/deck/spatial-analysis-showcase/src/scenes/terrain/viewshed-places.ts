// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The named places of the viewshed story, resolved against the DEM the analysis runs on: the summits
 * whose visibility decides which peaks get a name (PeakFinder policy: name what you can see, list
 * what you cannot), the lookout stations of the cumulative viewshed, and the largest gap in the
 * lookout coverage. Names and coordinates come from the `ALPS` gazetteer and the `alps-context`
 * dataset; nothing is typed here.
 */

import {GPU_TERRAIN_VISIBILITY} from '@luma.gl/experimental/gpu-terrain';
import type {MapAnnotation} from '../../cartography/types';
import type {TerrainDem} from './cpu-dem';
import {
  type AlpsContext,
  findContextPlace,
  type ContextPlaceKind,
  peakLabels,
  snapLngLatToHighestCell
} from './terrain-places';

/** A named summit snapped to the highest DEM cell near it. */
export type Summit = {
  /** Name as the labels spell it. */
  name: string;
  /** Published elevation in metres, or null. */
  elevationMeters: number | null;
  /** `[longitude, latitude]` of the summit cell of this DEM. */
  lngLat: [number, number];
  column: number;
  row: number;
  /** Raster index of the summit cell. */
  cellIndex: number;
  /** DEM elevation of the summit cell, metres (lower than the published value on a coarse grid). */
  demElevation: number;
};

/**
 * The summits whose cells the story reads, in label priority order: the two the story always
 * names first, then the rest by fame. Gazetteer ids and OSM names; unknown ones are skipped.
 */
export const SUMMIT_REQUESTS: readonly string[] = [
  'matterhorn',
  'dufourspitze',
  'liskamm',
  'breithorn',
  'castor',
  'pollux',
  'Zumsteinspitze',
  'Nordendspitze',
  'dom',
  'weisshorn',
  'dent-blanche',
  'taschhorn',
  'Rimpfischhorn',
  'Strahlhorn',
  'Obergabelhorn'
];

/** The summits that are always named, hidden or not. */
export const ALWAYS_NAMED = ['Matterhorn', 'Dufourspitze'] as const;

/** Resolves {@link SUMMIT_REQUESTS} on a DEM: the label coordinate snapped to the summit cell. */
export function resolveSummits(context: AlpsContext, dem: TerrainDem): Summit[] {
  const labels = peakLabels(context, {names: SUMMIT_REQUESTS, window: dem.lngLatBounds});
  const summits: Summit[] = [];
  const taken = new Set<number>();
  for (const label of labels) {
    if (label.kind !== 'landform') continue;
    const snapped = snapLngLatToHighestCell(dem, label.coordinate, 160);
    if (!snapped) continue;
    const cellIndex = snapped.row * dem.width + snapped.column;
    // Two requests that snap to one cell (a summit and its shoulder) name it once.
    if (taken.has(cellIndex)) continue;
    taken.add(cellIndex);
    summits.push({
      name: label.text,
      elevationMeters: label.elevationMeters ?? null,
      lngLat: snapped.lngLat,
      column: snapped.column,
      row: snapped.row,
      cellIndex,
      demElevation: snapped.elevation
    });
  }
  return summits;
}

/** The names the story reads and the cells they gather, as one record per summit. */
export type PeakVisibility = {
  summit: Summit;
  /** `GPU_TERRAIN_VISIBILITY` code at the summit cell. */
  code: number;
};

/** What {@link getPeakVisibility} found. */
export type PeakSelection = {
  /** Place labels to draw: the always-named summits and the visible ones, up to `maxLabels`. */
  labels: MapAnnotation[];
  /** Names of the summits in range whose cell is hidden. */
  hidden: string[];
  /** Names of the summits in range whose cell is visible or marginal. */
  visible: string[];
};

/**
 * Applies the PeakFinder policy to the codes read at the summit cells: the always-named summits
 * keep their label, further labels go to summits that can be seen (in label priority order) until
 * `maxLabels` are used, and the hidden ones in range are listed so a readout can name them.
 *
 * @param codes One `GPU_TERRAIN_VISIBILITY` code per summit, in the order of `summits`.
 */
export function getPeakVisibility(
  summits: readonly Summit[],
  codes: ArrayLike<number>,
  options: {maxLabels: number; extraNamed?: readonly string[]}
): PeakSelection {
  const named = new Set<string>([...ALWAYS_NAMED, ...(options.extraNamed ?? [])]);
  const labels: MapAnnotation[] = [];
  const hidden: string[] = [];
  const visible: string[] = [];
  summits.forEach((summit, index) => {
    const code = codes[index];
    const isVisible =
      code === GPU_TERRAIN_VISIBILITY.visible || code === GPU_TERRAIN_VISIBILITY.marginal;
    if (code === GPU_TERRAIN_VISIBILITY.hidden) hidden.push(summit.name);
    if (isVisible) visible.push(summit.name);
  });
  const toLabel = (summit: Summit, priority: number): MapAnnotation => ({
    kind: 'landform',
    id: `summit:${summit.name}`,
    coordinate: summit.lngLat,
    text: summit.name,
    marker: 'peak',
    ...(summit.elevationMeters !== null ? {elevationMeters: summit.elevationMeters} : {}),
    priority
  });
  summits.forEach((summit, index) => {
    if (named.has(summit.name)) labels.push(toLabel(summit, 20 - index));
  });
  summits.forEach((summit, index) => {
    if (labels.length >= options.maxLabels) return;
    if (named.has(summit.name)) return;
    const code = codes[index];
    if (code === GPU_TERRAIN_VISIBILITY.visible) labels.push(toLabel(summit, 10 - index));
  });
  return {labels, hidden, visible};
}

/** A lookout of the cumulative viewshed. */
export type Lookout = {
  name: string;
  /** Station elevation in metres from the dataset, or null. */
  elevationMeters: number | null;
  lngLat: readonly [number, number];
};

/**
 * The six lookouts of the cumulative viewshed: Gornergrat station first (the draggable observer)
 * and five stations of the railway and the lifts that stand at different heights and on different
 * sides of the valley.
 */
const LOOKOUT_REQUESTS: readonly {name: string; kinds: readonly ContextPlaceKind[]}[] = [
  {name: 'Gornergrat', kinds: ['rail-station']},
  {name: 'Rotenboden', kinds: ['rail-station']},
  {name: 'Riffelalp', kinds: ['rail-station']},
  {name: 'Schwarzsee', kinds: ['lift-station']},
  {name: 'Trockener Steg', kinds: ['lift-station']},
  {name: 'Matterhorn Glacier Paradise', kinds: ['lift-station']}
];

/**
 * Resolves the lookouts in the context dataset. A station that the dataset lacks is dropped (the
 * observer count slider then stops earlier); Gornergrat is required and falls back to `fallback`.
 */
export function resolveLookouts(
  context: AlpsContext,
  fallback: {name: string; lngLat: readonly [number, number]}
): Lookout[] {
  const lookouts: Lookout[] = [];
  for (const request of LOOKOUT_REQUESTS) {
    const place = findContextPlace(context, request.name, request.kinds);
    if (place) {
      lookouts.push({
        name: request.name,
        elevationMeters: place.elevationMeters,
        lngLat: place.lngLat
      });
    } else if (lookouts.length === 0) {
      lookouts.push({name: fallback.name, elevationMeters: null, lngLat: fallback.lngLat});
    }
  }
  return lookouts;
}

/** The deepest point of the unseen ground: the centre of the largest circle with no seen cell. */
export type UnseenGap = {
  /** Centre cell on the coarse mask, `[column, row]`. */
  coarseColumn: number;
  coarseRow: number;
  /** Radius of the largest circle containing only unseen cells, in coarse cells. */
  radiusCells: number;
};

/**
 * The largest unseen gap of a coarse mask (1 = unseen, 0 = seen or beyond every lookout's reach):
 * a two-pass chamfer distance (3-4 weights) from every cell to the nearest cell that is not
 * unseen, with the mask edge counting as not unseen (the data ends there). The cell with the
 * largest distance is the centre of the largest circle that holds only unseen cells.
 *
 * Why not "the largest connected patch": at this coverage the unseen cells form one connected
 * region, whose centroid is not a place; the centre of the largest empty circle is.
 */
export function getLargestUnseenGap(
  mask: ArrayLike<number>,
  width: number,
  height: number
): UnseenGap | null {
  const big = 1e9;
  const distance = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      distance[index] = mask[index] === 1 ? big : 0;
    }
  }
  const at = (column: number, row: number) =>
    column < 0 || row < 0 || column >= width || row >= height ? 0 : distance[row * width + column];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      if (distance[index] === 0) continue;
      distance[index] = Math.min(
        distance[index],
        at(column - 1, row) + 3,
        at(column, row - 1) + 3,
        at(column - 1, row - 1) + 4,
        at(column + 1, row - 1) + 4
      );
    }
  }
  let best = 0;
  let bestIndex = -1;
  for (let row = height - 1; row >= 0; row--) {
    for (let column = width - 1; column >= 0; column--) {
      const index = row * width + column;
      if (distance[index] === 0) continue;
      distance[index] = Math.min(
        distance[index],
        at(column + 1, row) + 3,
        at(column, row + 1) + 3,
        at(column + 1, row + 1) + 4,
        at(column - 1, row + 1) + 4
      );
      if (distance[index] > best) {
        best = distance[index];
        bestIndex = index;
      }
    }
  }
  if (bestIndex < 0) return null;
  // Chamfer 3-4 units are three per cell step.
  return {
    coarseColumn: bestIndex % width,
    coarseRow: Math.floor(bestIndex / width),
    radiusCells: best / 3
  };
}
